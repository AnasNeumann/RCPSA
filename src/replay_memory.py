from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch._prims_common import DeviceLikeType

from conf import MEMORY_CAPACITY, W_LB, W_FINAL

# ====================================================
# =*= Model file for GNN tree-shaped replay memory =*=
# ====================================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

@dataclass
class PossibleAction:
    id: int
    lb: int

    def __repr__(self):
        return f'PA(id={self.id}, lb={self.lb})'

class Transition:
    """
        One transition in the DRL MEMORY TREE
    """
    def __init__(self, action: Tensor, previous_graph: HeteroData, graph: HeteroData, lb: int, delta_lb: int, possible_actions: list[PossibleAction], parent=None):
        self.action: Tensor               = action
        self.graph: HeteroData            = graph.clone().to('cpu')
        self.delta_lb: int                = delta_lb
        self.lb: int                      = lb
        self.previous_graph: HeteroData   = previous_graph.clone().to('cpu')
        self.parent: Transition           = parent
        self.in_memory: bool              = False
        self.reward: Tensor               = None
        self.makespan: int                = 0
        self.nb_visits: int               = 1
        self.possible_actions: list[PossibleAction] = possible_actions
        self.makespans: list[int]         = []
        self.next: list[Transition]       = []
        if parent is not None and self not in parent.next:
            self.parent.next.append(self)

    def depth(self, current=None):
        t: Transition = current if current is not None else self
        depth_children: list[int] = []
        for child in t.next:
            depth_children.append(self.depth(current=child))
        if depth_children:
            return 1 + max(depth_children)
        return 1
    
    def same(self, t: 'Transition') -> bool:
        t: Transition
        return self.parent == t.parent and torch.equal(self.action, t.action)
    
    def compute_reward(self, makespan: int, device: DeviceLikeType):
        r: float       = (-1.0) * ((makespan * W_FINAL) + (self.delta_lb * W_LB))
        self.reward    = torch.tensor([r], device=device)
        self.makespan  = makespan
        self.lb        = min(self.lb, makespan)
        self.nb_visits = 1
        self.makespans.append(makespan)

    def revisit(self, t: 'Transition'):
        self.nb_visits += 1
        self.graph      = t.graph
        self.makespans.append(t.makespan)
        if self.makespan >= t.makespan:
            self.reward   = t.reward
            self.makespan = t.makespan

    def refine_from_possible_children(self, memory: 'Memory', init_lb: int, cut_bad_branches: bool, best_Cmax: int):
        if self.possible_actions:
            self.lb = min(a.lb for a in self.possible_actions)
        if self.parent is not None:
            self.delta_lb = self.lb - self.parent.lb
            for a in self.parent.possible_actions:
                if a.id == self.action.item():
                    a.lb = self.lb
                    break
        else:
            self.delta_lb = self.lb - init_lb 
        if self.next and cut_bad_branches:
            self.possible_actions = [a for a in self.possible_actions if a.lb <= best_Cmax]
            to_remove             = [t for t in self.next if t.lb > best_Cmax and t.in_memory == True]
            self.next             = [t for t in self.next if t.lb <= best_Cmax]
            for t in to_remove:
                t.in_memory = False
                memory.flat_transitions.remove(t)
            if self.parent is not None and not self.possible_actions:
                self.parent.refine_from_possible_children(memory, init_lb, True, best_Cmax)

class ITree:
    """
        The tree memory of one specific instance
    """
    def __init__(self, global_memory, instance_name: str, device: DeviceLikeType):
        self.instance_name: int                 = instance_name
        self.tree_transitions: list[Transition] = []
        self.device: DeviceLikeType             = device
        self.global_memory: Memory              = global_memory

    def search_transition(self, action: int, current_transition: Transition = None) -> Transition:
        to_test: list[Transition] = self.tree_transitions if current_transition is None else current_transition.next
        for t in to_test:
            if t.action.item() == action:
                return t
        return None

    def compute_rewards(self, transition: Transition, final_makespan: int) -> Tensor:
        transition.compute_reward(makespan=final_makespan, device=self.device)
        for _next in transition.next:
            self.compute_rewards(transition=_next, final_makespan=final_makespan)

    def add_or_update_transition(self, transition: Transition, final_makespan: int, need_rewards: bool=True) -> Transition:
        if need_rewards:
            self.compute_rewards(transition=transition, final_makespan=final_makespan)
        if transition.parent is None:
            _found: bool = False
            for _other_first in self.tree_transitions:
                if _other_first.same(transition):
                    _found = True
                    _other_first.revisit(transition)
                    for _next in transition.next:
                        _next.parent = _other_first
                        self.add_or_update_transition(transition=_next, final_makespan=final_makespan, need_rewards=False)
                    return _other_first
            if not _found:
                self.tree_transitions.append(transition)
                _t: Transition = transition
                while True:
                    self.global_memory.add_into_flat_memory(_t)
                    if not _t.next:
                        break
                    _t = _t.next[0]
                return transition
        else:
            _found: bool = False
            for _existing in transition.parent.next:
                if _existing.same(transition):
                    _found = True
                    _existing.revisit(transition)
                    for _next in transition.next:
                        _next.parent = _existing
                        self.add_or_update_transition(transition=_next, final_makespan=final_makespan, need_rewards=False)
                    return _existing
            if not _found:
                transition.parent.next.append(transition)
                _t: Transition = transition.parent
                while True:
                    if not _t.parent:
                        break
                    _t = _t.parent
                while True:
                    self.global_memory.add_into_flat_memory(_t)
                    if not _t.next:
                        break
                    _t = _t.next[0]
                return transition

class Memory:
    """
        The DRL memory for all instances with: 
            1. A global flat memory (for sampling)
            2. A tree-shaped memory by instance (for mainting up-to-date rewards)
    """
    def __init__(self, device: DeviceLikeType):
        self.device = device
        self.instance_trees: list[ITree] = []
        self.flat_transitions: list[Transition] = []

    def add_into_flat_memory(self, transition: Transition):
        if not transition.in_memory:
            transition.in_memory = True
            if len(self.flat_transitions) == MEMORY_CAPACITY:
                _old: Transition = self.flat_transitions.pop(0)
                _old.in_memory = False
            self.flat_transitions.append(transition)
    
    def add_instance_if_new(self, instance_name: str) -> ITree:
        for tree in self.instance_trees:
            if tree.instance_name == instance_name:
                return tree
        new_tree: ITree = ITree(global_memory=self, instance_name=instance_name, device=self.device)
        self.instance_trees.append(new_tree)
        return new_tree
    
    def get_instance_by_name(self, instance_name: str) -> ITree:
        for tree in self.instance_trees:
            if tree.instance_name == instance_name:
                return tree
        return None
