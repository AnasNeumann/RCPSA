import argparse
import os 
import time
import gc
from itertools import count
import math
import random
import numpy as np

import torch
from torch import Tensor
from torch.optim import AdamW

from conf import INTERACTIVE, LR, NB_EPISODES, EPS_DECAY, EPS_END, EPS_START, GREEDY_RATE, TOP_K, INFINITY
from src.common import display_final_computing_time
from src.state import State
from src.neural_nets import HyperGraphGNN
from src.replay_memory import Transition, Memory, ITree, PossibleAction
from src.tracker import Tracker
from src.instance_reader import build_instance
from src.dqn_functions import optimize_policy_net, optimize_target_net, select_action, take_step
from src.scheduling_functions import find_feasible_tasks

# ================================
# =*= MAIN CODE OF THE PROJECT =*=
# ================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

def diversify(memory: ITree, current_transition: Transition, possible_actions: list[PossibleAction], device: str, best_known_Cmax: int) -> Tensor:
    """
        Select a random feasible action that has not been tried yet in the current branch of the tree
    """
    unseen: list[PossibleAction] = []
    for a in possible_actions:
        _next_t: Transition = memory.search_transition(action=a.id, current_transition=current_transition)
        if _next_t is None and a.lb <= best_known_Cmax:
            unseen.append(a)
    if unseen:
        print(len(unseen), "... unseen actions (with good lower bounds) found for diversification")
        unseen.sort(key=lambda a: a.lb)
        idx: int               = np.random.choice(min(TOP_K, len(unseen)))
        action: PossibleAction = unseen[idx]
        return torch.tensor([[action.id]], device=device, dtype=torch.long), False
    return None, True

def search_possible_actions(state: State, current_transition: Transition, best_known_Cmax: int, remove_bad_LB_banches: bool = False) -> list[PossibleAction]:
    if current_transition is not None:
        if remove_bad_LB_banches: 
            current_transition.possible_actions = [a for a in current_transition.possible_actions if a.lb <= best_known_Cmax]
        return current_transition.possible_actions
    else:
        possible_actions: list[PossibleAction] = []
        feasible_tasks: list[dict] = find_feasible_tasks(state.tasks, state.scheduled_tasks)
        for task in feasible_tasks: 
            _next_state, _ = take_step(state=state, action=task["Id"])
            if _next_state.make_span < INFINITY:
                lb: int    = _next_state.compute_lower_bound()
                if lb <= best_known_Cmax or not remove_bad_LB_banches:
                    possible_actions.append(PossibleAction(id=task["Id"], lb=lb))
        return possible_actions

def solve(path: str, instance_type: str, instance_name: str, interactive: bool):
    """
        Main function to fine-tune (witn e-greedy DQN) and solve an RCPSP instance using the pre-trained dual-encoder transformer
    """
    _device                                = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing device: {_device}...")
    _saving_path: str                      = f"{path}data/best_sols/{instance_type}/{instance_name}/{instance_name}"
    _instance_path: str                    = f"{path}data/instances/{instance_type}/{instance_name}.RCP"
    start_time                             = time.time()
    _steps: int                            = 0
    _best_episode: int                     = -1
    _tasks, _resources                     = build_instance(_instance_path)
    _best_state: State                     = State.from_problem(tasks=_tasks, resources=_resources, device=_device)
    Cmax: int                              = INFINITY
    _POLICY_NET: HyperGraphGNN             = HyperGraphGNN(task_features=State.TASK_FEATURES, resource_features=State.RESOURCE_FEATURES, demand_features=State.DEMAND_FEATURES, num_tasks=len(_tasks), num_resources=len(_resources)).to(_device)
    _TARGET_NET: HyperGraphGNN             = HyperGraphGNN(task_features=State.TASK_FEATURES, resource_features=State.RESOURCE_FEATURES, demand_features=State.DEMAND_FEATURES, num_tasks=len(_tasks), num_resources=len(_resources)).to(_device)
    _LOSS_TRACKER: Tracker                 = Tracker(xlabel="Episode", ylabel="Loss", title="Huber Loss (policy network)", color="blue", show=interactive)
    _Cmax_TRACKER: Tracker                 = Tracker(xlabel="Episode", ylabel="Makespan", title="Final Makespan by DQN episode", color="red", show=interactive)
    _EPSILON_TRACKER: Tracker              = Tracker(xlabel="Episode", ylabel="epsilon", title="Diversity rate", color="black", show=interactive)
    _REPLAY_MEMORY: Memory                 = Memory(device=_device)
    _TREE: ITree                           = _REPLAY_MEMORY.add_instance_if_new(instance_name=instance_name)
    possible_actions: list[PossibleAction] = search_possible_actions(state=_best_state, current_transition=None, best_known_Cmax=Cmax)
    _best_state.graph                      = _best_state.to_hyper_graph(possible_actions=possible_actions, transitions=_TREE.tree_transitions)
    for param in _POLICY_NET.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    _TARGET_NET.load_state_dict(_POLICY_NET.state_dict())
    _POLICY_NET.train()
    _TARGET_NET.eval()
    _OPTIMIZER: AdamW = AdamW(_POLICY_NET.parameters(), lr=LR, amsgrad=True)
    for _episode in range(1, NB_EPISODES+1):
        _e: float                                 = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * _episode / EPS_DECAY)
        transition: Transition                    = None
        _search_transition: bool                  = True
        _state: State                             = State.from_empty_solution(_best_state, _tasks, _resources)
        possible_actions                          = search_possible_actions(state=_state, current_transition=None, best_known_Cmax=Cmax)
        _state.graph                              = _state.to_hyper_graph(possible_actions=possible_actions, transitions=_TREE.tree_transitions)
        _prev_lb: int                             = _best_state.init_lb
        _prev_ub: int                             = _best_state.init_ub
        _transitions_in_episode: list[Transition] = []
        diversified_step: int = random.randint(0, len(_tasks)-1)
        for step in count():
            should_diversify: bool = _search_transition and (step == diversified_step)
            if should_diversify:
                _action_idx, failed = diversify(memory=_TREE, current_transition=transition, possible_actions=possible_actions, device=_device, best_known_Cmax=Cmax)
                if failed:
                    should_diversify  = False
                    diversified_step += 1
            if not should_diversify:
                _action_idx: Tensor = select_action(state=_state, policy_net=_POLICY_NET, e=_e, greedy=random.random() < GREEDY_RATE, device=_device, memory=_REPLAY_MEMORY, possible_actions=possible_actions)
            if _search_transition:
                transition        = _TREE.search_transition(action=_action_idx.item(), current_transition=transition)
                _search_transition = transition is not None
            _steps                += 1
            _next_state, task      = take_step(state=_state, action=_action_idx.item())
            possible_actions       = search_possible_actions(state=_next_state, current_transition=transition, best_known_Cmax=Cmax, remove_bad_LB_banches=False)
            print(possible_actions)
            if not possible_actions and not _next_state.done:
                print(f"-> ERROR: No possible actions with lower bound < current best Cmax ({Cmax})!")
                break
            _next_state.graph      = _next_state.to_hyper_graph(possible_actions=possible_actions, transitions=transition.next if transition is not None else [])
            _next_lb: int          = _next_state.compute_lower_bound()
            _next_ub: int          = _next_state.compute_upper_bound()
            _delta_LB: int         = _next_lb - _prev_lb
            _delta_UB: int         = _next_ub - _prev_ub
            _delta_duration: int   = max(0, _next_state.make_span - _state.make_span + task["Duration"])
            _transitions_in_episode.append(Transition(action=_action_idx, 
                                                      previous_graph=_state.graph, 
                                                      graph=_next_state.graph, 
                                                      lb=_next_lb, 
                                                      delta_lb=_delta_LB, 
                                                      ub=_next_ub, 
                                                      delta_ub=_delta_UB, 
                                                      delta_duration=_delta_duration, 
                                                      possible_actions=possible_actions,
                                                      parent=_transitions_in_episode[-1] if _transitions_in_episode else None))
            _state                 = _next_state
            _prev_lb               = _next_lb
            _prev_ub               = _next_ub

            # END OF EPISODE
            if _state.done:
                _EPSILON_TRACKER.update(_e)
                _Cmax_TRACKER.update(_state.make_span)
                _TREE.add_or_update_transition(transition=_transitions_in_episode[0], final_makespan=_state.make_span)
                huber_loss: float = optimize_policy_net(memory=_REPLAY_MEMORY, policy_net=_POLICY_NET, target_net=_TARGET_NET, optimizer=_OPTIMIZER, tracker=_LOSS_TRACKER, nb_tasks=len(_tasks), device=_device)
                optimize_target_net(policy_net=_POLICY_NET, target_net=_TARGET_NET)
                if _state.make_span < Cmax:
                    _best_state   = _state
                    _best_episode = _episode
                    Cmax          = _state.make_span
                print(f"DQN Episode: {_episode} -- random_step: {diversified_step} -- Makespan: {_state.make_span} (best: {Cmax}) -- Є: {_e:.3f} -- Huber Loss: {huber_loss:.2f}")

                # TIME TO FREE SOME MEMORY                
                if _episode % 50 == 0 and _episode > 0:
                    gc.collect()
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                # LAST EPISODE
                if _episode == NB_EPISODES:
                    print(f"Saving files...")
                    os.makedirs(os.path.dirname(_saving_path), exist_ok=True)
                    with open(_saving_path + "_results.txt", "w", encoding="utf-8") as file:
                        file.write(f"Best episode: {_best_episode}\nHuber Loss:{huber_loss:.2f}\nMakespan: {_best_state.make_span}\nSequence: {_best_state.id}")
                    _Cmax_TRACKER.save(_saving_path + "_DRL_makespan")
                    _EPSILON_TRACKER.save(_saving_path + "_diversity")
                    _LOSS_TRACKER.save(_saving_path + "_loss")
                    torch.save(_TARGET_NET.state_dict(), _saving_path + "_target_gnn.pth")
                    torch.save(_POLICY_NET.state_dict(), _saving_path + "_policy_gnn.pth")
                    torch.save(_OPTIMIZER.state_dict(), _saving_path + "_adamw_gnn_optimizer.pth")
                break
    print(f"END, BEST SOLUTION: State = {_best_state.id} -- Makespan = {_best_state.make_span}")
    display_final_computing_time(start_time, instance, path, use_end_stream=False)

# python main.py --type=J60 --instance=J601_1 --path=.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCPSP-RL production job")
    parser.add_argument("--instance", help="The instance to train on", required=False)
    parser.add_argument("--path", help="Local path on DRAC server", required=True)
    parser.add_argument("--type", help="The type of instances to train on", required=True)
    args         = parser.parse_args()
    _global_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
    type         = args.type
    instance     = args.instance
    print(f"Solving instance: {instance} from {type}...")
    print(f"Running path: {_global_path}...")
    solve(path=_global_path, instance_type=type, instance_name=instance, interactive=INTERACTIVE)