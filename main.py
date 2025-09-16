import argparse
import os 
import time
from itertools import count
import math
import random

import torch
from torch import Tensor
from torch.optim import AdamW

from conf import INTERACTIVE, LR, NB_EPISODES, EPS_DECAY, EPS_END, EPS_START, CANDIDATES, INTENSIFY_RATE, GREEDY_RATE, INTENSIFY_INC
from src.common import display_final_computing_time
from src.state import State
from src.neural_nets import HyperGraphGNN
from src.replay_memory import Transition, Memory, ITree
from src.tracker import Tracker
from src.instance_reader import build_instance
from src.dqn_functions import take_step, optimize_policy_net, optimize_target_net, select_actions, select_action

# ================================
# =*= MAIN CODE OF THE PROJECT =*=
# ================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

def intensify(state: State, policy_net: HyperGraphGNN, device: torch.device, candidates: int=CANDIDATES) -> tuple[float, Tensor]:
        """
            Explore the search space using e-greedy DQN with limited candidates at each step
        """
        best_val: float     = -math.inf
        best_action: Tensor = None
        if not state.done:
            if candidates > 0:
                actions, Qvalues = select_actions(state=state, policy_net=policy_net, device=device, C=candidates)
                for i, candidate in enumerate(actions[0]):
                    _next_state: State = take_step(state=state, action=candidate.item())
                    _next_state.graph  = _next_state.to_hyper_graph()
                    val, _             = intensify(state=_next_state, policy_net=policy_net, device=device, candidates=candidates-1) # recursive call with decreased candidates!
                    Qvalue: float      = Qvalues[0, i].item() + (INTENSIFY_INC * val)
                    if best_val < Qvalue:
                        best_val    = Qvalue
                        best_action = candidate
                return best_val, torch.tensor([[best_action.item()]], device=device, dtype=torch.long)
            return 0, None
        return (INTENSIFY_INC * -state.make_span), None

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
        for _ in count():
            if random.random() >= INTENSIFY_RATE: # explore solution with three e-greedy strategies: random, soft greedy (one step), hard greedy (one step)
                _greedy: bool     = random.random() >= GREEDY_RATE
                _action_idx: int  = select_action(state=_state, policy_net=_POLICY_NET, e=_e, greedy=_greedy, device=_device, memory=_REPLAY_MEMORY)
            else: # intensify the search with multiple candidates and recursive exploration (only greedy, multiple steps)
                _, _action_idx = intensify(state=_state, policy_net=_POLICY_NET, device=_device, candidates=CANDIDATES)                
            _steps            += 1
            _next_state: State = take_step(state=_state, action=_action_idx.item())
            _next_state.graph  = _next_state.to_hyper_graph()
            _next_lb: int      = _next_state.compute_lower_bound()
            _delta: int        = _next_lb - _prev_lb
            _transitions_in_episode.append(Transition(action=_action_idx, previous_graph=_state.graph, graph=_next_state.graph, delta_duration=_delta, parent=_transitions_in_episode[-1] if _transitions_in_episode else None))
            _state             = _next_state
            _prev_lb           = _next_lb
            if _state.done:
                _EPSILON_TRACKER.update(_e)
                _Cmax_TRACKER.update(_state.make_span)
                _TREE.add_or_update_transition(transition=_transitions_in_episode[0], lb=_lb, final_makespan=_state.make_span)
                huber_loss: float = optimize_policy_net(memory=_REPLAY_MEMORY, policy_net=_POLICY_NET, target_net=_TARGET_NET, optimizer=_OPTIMIZER, tracker=_LOSS_TRACKER, nb_tasks=len(_tasks), device=_device)
                optimize_target_net(policy_net=_POLICY_NET, target_net=_TARGET_NET)
                if _state.make_span < _best_state.make_span:
                    _best_state   = _state
                    _best_episode = _episode
                print(f"Episode: {_episode} -- Makespan: {_state.make_span} (best: {_best_state.make_span}) -- Huber Loss: {huber_loss:.2f}")
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