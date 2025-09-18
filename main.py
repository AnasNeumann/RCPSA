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

from conf import INTERACTIVE, LR, NB_EPISODES, EPS_DECAY, EPS_END, EPS_START, INT_AND_DIV_RATE, GREEDY_RATE, TOP_K
from src.common import display_final_computing_time
from src.state import State
from src.neural_nets import HyperGraphGNN
from src.replay_memory import Transition, Memory, ITree
from src.tracker import Tracker
from src.instance_reader import build_instance
from src.dqn_functions import take_step, optimize_policy_net, optimize_target_net, select_action
from src.scheduling_functions import find_feasible_tasks
from src.local_search import local_search
from src.crossover import X

# ================================
# =*= MAIN CODE OF THE PROJECT =*=
# ================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

def intensify(state: State, memory: ITree, t: Transition, device: str) -> Tensor:
    """
        Select the best feasible action already tried in the current branch of the tree
    """
    possible_actions: list[int] = [a["Id"] for a in find_feasible_tasks(state.tasks, state.scheduled_tasks)]
    random.shuffle(possible_actions)
    best_cmax: int      = math.inf
    best_LB: int        = math.inf
    best_action: Tensor = torch.tensor([[random.choice(possible_actions)]], device=device, dtype=torch.long)
    for a in possible_actions:
        _next_t: Transition = memory.search_transition(action=a, current_transition=t)
        if _next_t is not None:
            if (_next_t.makespan < best_cmax and _next_t.lb <= (best_LB * 1.15)) or (_next_t.makespan <= (best_cmax* 1.15) and _next_t.lb <= best_LB):
                best_cmax   = _next_t.makespan
                best_action = _next_t.action
                best_LB     = _next_t.lb
    return best_action

def diversify(state: State, memory: ITree, t: Transition, device: str, alpha: float=0.8) -> Tensor:
    """
        Select a random feasible action that has not been tried yet in the current branch of the tree
    """
    unseen: list                = []
    possible_actions: list[int] = [a["Id"] for a in find_feasible_tasks(state.tasks, state.scheduled_tasks)]
    random.shuffle(possible_actions)
    for a in possible_actions:
        _next_t: Transition = memory.search_transition(action=a, current_transition=t)
        if _next_t is None:
            _next_state, _ = take_step(state=state, action=a)
            lb: int        = _next_state.compute_lower_bound()
            unseen.append((a, lb))
    if unseen:
        unseen   = unseen[:min(TOP_K, len(unseen))]
        lbs      = np.array([lb for _, lb in unseen], dtype=float)
        norm_lbs = (lbs - lbs.min()) / (lbs.max() - lbs.min() + 1e-9)
        w        = np.exp(-alpha * norm_lbs)
        p        = w / w.sum()
        idx      = np.random.choice(len(unseen), p=p)
        action   = unseen[idx][0]
        return torch.tensor([[action]], device=device, dtype=torch.long)
    return torch.tensor([[random.choice(possible_actions)]], device=device, dtype=torch.long)

def solve(path: str, instance_type: str, instance_name: str, interactive: bool):
    """
        Main function to fine-tune (witn e-greedy DQN) and solve an RCPSP instance using the pre-trained dual-encoder transformer
    """
    _device                    = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing device: {_device}...")
    _saving_path: str          = f"{path}data/best_sols/{instance_type}/{instance_name}/{instance_name}"
    _instance_path: str        = f"{path}data/instances/{instance_type}/{instance_name}.RCP"
    start_time                 = time.time()
    _steps: int                = 0
    _best_episode: int         = -1
    _tasks, _resources         = build_instance(_instance_path)
    _best_state: State         = State.from_problem(tasks=_tasks, resources=_resources, device=_device)
    _POLICY_NET: HyperGraphGNN = HyperGraphGNN(task_features=State.TASK_FEATURES, resource_features=State.RESOURCE_FEATURES, demand_features=State.DEMAND_FEATURES).to(_device)
    _TARGET_NET: HyperGraphGNN = HyperGraphGNN(task_features=State.TASK_FEATURES, resource_features=State.RESOURCE_FEATURES, demand_features=State.DEMAND_FEATURES).to(_device)
    _LOSS_TRACKER: Tracker     = Tracker(xlabel="Episode", ylabel="Loss", title="Huber Loss (policy network)", color="blue", show=interactive)
    _Cmax_TRACKER: Tracker     = Tracker(xlabel="Episode", ylabel="Makespan", title="Final Makespan by episode", color="red", show=interactive)
    _EPSILON_TRACKER: Tracker  = Tracker(xlabel="Episode", ylabel="epsilon", title="Diversity rate", color="green", show=interactive)
    _REPLAY_MEMORY: Memory     = Memory(device=_device)
    _TREE: ITree               = _REPLAY_MEMORY.add_instance_if_new(instance_name=instance_name)
    _X: X                      = X(tasks=_tasks, resources=_resources, device=_device)
    for param in _POLICY_NET.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    _TARGET_NET.load_state_dict(_POLICY_NET.state_dict())
    torch.compile(_POLICY_NET)
    torch.compile(_TARGET_NET)
    _OPTIMIZER: AdamW          = AdamW(_POLICY_NET.parameters(), lr=LR, amsgrad=True)
    _lb: int                   = _best_state.lower_bound
    for _episode in range(1, NB_EPISODES+1):
        _e: float                                 = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * _episode / EPS_DECAY)
        _state: State                             = State.from_empty_solution(_best_state, _tasks, _resources)
        _prev_lb: int                             = _lb
        _transitions_in_episode: list[Transition] = []
        _transition: Transition    = None
        _search_transition: bool   = True
        for _ in count():
            if not _search_transition or random.random() >= INT_AND_DIV_RATE:
                _greedy: bool        = random.random() >= GREEDY_RATE
                _action_idx: Tensor  = select_action(state=_state, policy_net=_POLICY_NET, e=_e, greedy=_greedy, device=_device, memory=_REPLAY_MEMORY)
            else:
                _action_idx: Tensor  = diversify(state=_state, memory=_TREE, t=_transition, device=_device) if random.random() <= _e else intensify(state=_state, memory=_TREE, t=_transition, device=_device)
            if _search_transition:
                _transition        = _TREE.search_transition(action=_action_idx.item(), current_transition=_transition)
                _search_transition = _transition is not None
            _steps            += 1
            _next_state, task  = take_step(state=_state, action=_action_idx.item())
            _next_state.graph  = _next_state.to_hyper_graph()
            _next_lb: int      = _next_state.compute_lower_bound()
            _delta_LB: int     = _next_lb - _prev_lb
            _delta_duration: int = max(0, _next_state.make_span - _state.make_span + task["Duration"])
            _transitions_in_episode.append(Transition(action=_action_idx, previous_graph=_state.graph, graph=_next_state.graph, lb=_next_lb, delta_lb=_delta_LB, delta_duration=_delta_duration, parent=_transitions_in_episode[-1] if _transitions_in_episode else None))
            _state             = _next_state
            _prev_lb           = _next_lb

            # END OF EPISODE
            if _state.done:
                _X.add(_state)
                _EPSILON_TRACKER.update(_e)
                _Cmax_TRACKER.update(_state.make_span)
                _TREE.add_or_update_transition(transition=_transitions_in_episode[0], final_makespan=_state.make_span)
                huber_loss: float = optimize_policy_net(memory=_REPLAY_MEMORY, policy_net=_POLICY_NET, target_net=_TARGET_NET, optimizer=_OPTIMIZER, tracker=_LOSS_TRACKER, nb_tasks=len(_tasks), device=_device)
                optimize_target_net(policy_net=_POLICY_NET, target_net=_TARGET_NET)
                if _state.make_span < _best_state.make_span:
                    _best_state   = _state
                    _best_episode = _episode
                print(f"DQN Episode: {_episode} -- Makespan: {_state.make_span} (best: {_best_state.make_span}) -- Є: {_e:.3f} -- Huber Loss: {huber_loss:.2f}")
                
                # TIME TO RUN LOCAL SEARCH
                ls_found, ls_solution, ls_transitions = local_search(_state, _tasks, _resources, _device)
                if ls_found:
                    _X.add(ls_solution)
                    _TREE.add_or_update_transition(transition=ls_transitions[0], final_makespan=ls_solution.make_span)
                    if ls_solution.make_span < _best_state.make_span:
                        _best_state   = ls_solution
                        _best_episode = _episode
                    print(f"LS successfull episode: -- Makespan: {_state.make_span}->{ls_solution.make_span}")
                
                # TIME TO RUN CROSSOVER OPERATOR (X)
                if _X.available():
                    X_solution, feasible, X_transitions = _X.run()
                    if feasible:
                        lsX_found, X_solution, lsX_transitions = local_search(X_solution, _tasks, _resources, _device)
                        X_transitions                          = lsX_transitions if lsX_found else X_transitions
                        _TREE.add_or_update_transition(transition=X_transitions[0], final_makespan=X_solution.make_span)
                        _X.add(X_solution)
                        if X_solution.make_span < _best_state.make_span:
                            _best_state   = X_solution
                            _best_episode = _episode
                        print(f"X feasible episode: -- Makespan: {X_solution.make_span}")

                # TIME TO FREE SOME MEMORY                
                if _episode > 0 and _device.type == "mps" and _episode % 50 == 0:
                        gc.collect()
                        torch.mps.empty_cache()

                # LAST EPISODE
                if _episode == NB_EPISODES:
                    print(f"Saving files...")
                    os.makedirs(os.path.dirname(_saving_path), exist_ok=True)
                    with open(_saving_path + "_results.txt", "w", encoding="utf-8") as file:
                        file.write(f"Best episode: {_best_episode}\nHuber Loss:{huber_loss:.2f}\nMakespan: {_best_state.make_span}\nSequence: {_best_state.id}")
                    _Cmax_TRACKER.save(_saving_path + "_makespan")
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