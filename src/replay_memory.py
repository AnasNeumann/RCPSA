import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch._prims_common import DeviceLikeType
from conf import MEMORY_CAPACITY, W_FINAL, W_NON_FINAL, W_DURATION

# ====================================================
# =*= Model file for GNN tree-shaped replay memory =*=
# ====================================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

class Transition:
    """
        One transition in the DRL MEMORY TREE
    """
    def __init__(self, action: Tensor, previous_graph: HeteroData, graph: HeteroData, delta_duration: int, lb: int, delta_lb: int, parent=None):
        self.action: Tensor             = action
        self.graph: HeteroData          = graph
        self.delta_lb: int              = delta_lb
        self.lb: int                    = lb
        self.previous_graph: HeteroData = previous_graph
        self.delta_duration: int        = delta_duration
        self.parent: Transition         = parent
        self.in_memory: bool            = False
        self.reward: Tensor             = None
        self.makespan: int              = 0
        self.next: list[Transition]     = []
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
    
    def same(self, t) -> bool:
        t: Transition
        return self.parent == t.parent and torch.equal(self.action, t.action)
    
    def compute_reward(self, makespan: int, device: DeviceLikeType, is_last: bool = False):
        w: float = W_FINAL if is_last else W_NON_FINAL
        r: float = (-1.0) * (makespan * w + self.delta_duration * W_DURATION + self.delta_lb)
        self.reward   = torch.tensor([r], device=device)
        self.makespan = makespan
        self.lb       = min(self.lb, makespan)

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
        transition.compute_reward(makespan=final_makespan, device=self.device, is_last=(len(transition.next) == 0))
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
                    _other_first.reward = torch.max(_other_first.reward, transition.reward)
                    _other_first.makespan = min(_other_first.makespan, transition.makespan)
                    for _next in transition.next:
                        _next.parent = _other_first
                        self.add_or_update_transition(transition=_next, final_makespan=final_makespan, need_rewards=False)
                    return _other_first
            if not _found:
                self.tree_transitions.append(transition)
                self.global_memory.add_into_flat_memory(transition)
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
                    _existing.reward = torch.max(_existing.reward, transition.reward)
                    _existing.makespan = min(_existing.makespan, transition.makespan)
                    for _next in transition.next:
                        _next.parent = _existing
                        self.add_or_update_transition(transition=_next, final_makespan=final_makespan, need_rewards=False)
                    return _existing
            if not _found:
                transition.parent.next.append(transition)
                self.global_memory.add_into_flat_memory(transition)
                _t: Transition = transition
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
        transition.in_memory = True
        self.flat_transitions.append(transition)
        if len(self.flat_transitions) > MEMORY_CAPACITY:
            _old: Transition = self.flat_transitions.pop(0)
            _old.in_memory = False
    
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
