import math
import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch import device as Device
from torch import Tensor
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_max_pool

from conf import TAU, EPS_END, BATCH_SIZE, EPS_START, GAMMA, O

from src.neural_nets import HyperGraphGNN 
from src.state import State
from src.replay_memory import Transition, Memory, ITree
from src.tracker import Tracker
from src.scheduling_functions import find_feasible_tasks, find_possible_start_day_for_task

# ==========================================================================
# =*= Reinforcement Learning (DQN) related functions only for GNN solver =*=
# ==========================================================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

def build_impossible_state(impossible_state: State, task: dict):
    """
        Build an impossible state with high penalties
        Works for both type of state (simple matrix and transformer)
    """
    impossible_state.scheduled_tasks.append(task["Id"])
    impossible_state.id = f'{impossible_state.id}_{task["Id"]}'
    impossible_state.make_span = 100000
    impossible_state.reward = -100000
    impossible_state.done = True
    print("IMPOSSIBLE")
    return impossible_state

def check_precedence_feasibility(state: State, task)->bool:
    """
        Check if a task can be executed (if no predecessor not executed yet)
        Works for both type of state (simple matrix and transformer)
    """
    if task["Id"] in state.scheduled_tasks:
        return False
    preds = task["Predecessors"]
    predecessors = [t['Id'] for t in state.tasks if t['Id'] in preds]
    for t in predecessors:
        if t not in state.scheduled_tasks:
            return False
    return True

def ssgs(tasks: list[dict], resources: list[tuple[int, int]], task: dict, ub: int) -> dict:
    """
        Find the earliest feasible start day for a task using the Serial Schedule Generation Scheme (SSGS)
        Return the task with updated "Start" and "Finish" fields (or -1 if not possible within the horizon)
    """
    min_start_day = 1
    predecessor_ids = task["Predecessors"]
    predecessors = [t for t in tasks if t['Id'] in predecessor_ids]
    for predecessor in predecessors:
        if predecessor["Finish"] >= min_start_day:
            min_start_day = predecessor["Finish"] + (not (not (predecessor["Duration"] * task["Duration"])))
    start_day = find_possible_start_day_for_task(tasks, resources, task, min_start_day, ub)
    if start_day > 0:
        task["Start"] = start_day
        task["Finish"] = start_day + task["Duration"] - (not (not (task["Duration"])))
    return task

def take_step(state: State, action: int):
    """
        Take a step in the environment by selecting an action (task) to schedule
    """
    new_state: State = State.from_partial_solution(state, build_graph=False)
    try:
        task = [t for t in new_state.tasks if t["Id"] == action][0]
    except:
        print(f"{action} not found")
        exit()
    feasible: bool = check_precedence_feasibility(state, task)
    task = ssgs(new_state.tasks, new_state.resources, task, 10000)
    if not feasible or task["Start"] <= 0:
        new_state = build_impossible_state(new_state, task)
    else:
        new_state.scheduled_tasks.append(task["Id"])
        new_state.id = f'{new_state.id}_{task["Id"]}' if task["Id"] > 0 else f'{task["Id"]}'
        new_state.make_span = max(new_state.make_span, task["Finish"])
        if len(new_state.scheduled_tasks) == len(new_state.tasks):
            new_state.done = True
    return new_state

def multi_strategy_selection(state: State, intensify: bool, policy_net: HyperGraphGNN, memory: ITree, device: Device, current_transition: Transition=None, next_transition_possibly_known: bool=False):
    possible_actions = find_feasible_tasks(state.tasks, state.scheduled_tasks)
    if intensify: # First family of strategies: Q-value intensification (try to redo best solution but modify them a bit)
        if next_transition_possibly_known and random.random() < DQN_ELITISM_RATE: # Selection stategy 1/4: redo a previous action that yielded the highest reward
            _trs: list[(Transition, any)] = [(memory.search_transition([task["Id"]], current_transition), task) for task in possible_actions]
            _filtered_trs: list[(Transition, any)] = [(transition, task) for (transition, task) in _trs if transition is not None]
            if _filtered_trs:
                _action: list[(Transition, any)] = max(_filtered_trs, key=lambda t: t[0].reward.item())
                return torch.tensor([[_action[1]["Id"]]], device=device, dtype=torch.long), _action[0], True
        with torch.no_grad(): # Selection stategy 2/4: trust the DQN and get the highest predicted Q value
            Q_values = policy_net(Batch.from_data_list([state.graph]).to(device))
            possible_idx = torch.tensor([action['Id'] for action in possible_actions], device=device)
            selected_values = Q_values[possible_idx].squeeze(-1)
            _, index = selected_values.max(0)
            action = possible_idx[index].item()
            if next_transition_possibly_known:
                _trs: list[Transition] = [memory.search_transition([task["Id"]], current_transition) for task in possible_actions]
                return torch.tensor([[action]], device=device, dtype=torch.long), _trs[index], _trs[index] is not None
            return torch.tensor([[action]], device=device, dtype=torch.long), None, False
    else: # Second family of stategies: diversification (try to get unseen branches)
        if next_transition_possibly_known: # Selection stategy 3/4: try to get a new decision never search before
            _trs: list[(Transition, any)] = [(memory.search_transition([task["Id"]], current_transition), task) for task in possible_actions]
            if random.random() < DQN_DIVERSITY_RATE:
                random.shuffle(_trs)
                for _t, task in _trs:
                    if _t is None:
                        return torch.tensor([[task["Id"]]], device=device, dtype=torch.long), None, False
            index = random.randrange(len(possible_actions))
            action = possible_actions[index]["Id"]
            return torch.tensor([[action]], device=device, dtype=torch.long), _trs[index][0], _trs[index][0] is not None
        action = random.choice(possible_actions)["Id"] # Section stategy 4/4: random selection of an action
        return torch.tensor([[action]], device=device, dtype=torch.long), None, False

def select_action(state: State, policy_net: HyperGraphGNN, memory_size: int, steps: int, device: Device, decay: int, memory: ITree=None, current_transition: Transition=None, next_transition_possibly_known: bool=False):
    """
        Select a feasible-only action using the current policy network OR random (when replay memory is still relatively empty)
    """
    eps_threshold: float = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / decay)
    possible_actions = find_feasible_tasks(state.tasks, state.scheduled_tasks)
    
    # First family of strategies: Q-value intensification (try to redo best solution but modify them a bit)
    if random.random() > eps_threshold and memory_size >= BATCH_SIZE:
        
        # Selection stategy 1/4: redo a previous action that yielded the highest reward
        if next_transition_possibly_known and random.random() < DQN_ELITISM_RATE:
            _trs: list[(Transition, any)] = [(memory.search_transition([task["Id"]], current_transition), task) for task in possible_actions]
            _filtered_trs: list[(Transition, any)] = [(transition, task) for (transition, task) in _trs if transition is not None]
            if _filtered_trs:
                _action: list[(Transition, any)] = max(_filtered_trs, key=lambda t: t[0].reward.item())
                return torch.tensor([[_action[1]["Id"]]], device=device, dtype=torch.long), _action[0], True, eps_threshold
        
        # Selection stategy 2/4: trust the DQN and get the highest predicted Q value
        with torch.no_grad():
            Q_values = policy_net(Batch.from_data_list([state.graph]).to(device))
            possible_idx = torch.tensor([action['Id'] for action in possible_actions], device=device)
            selected_values = Q_values[possible_idx].squeeze(-1)
            _, index = selected_values.max(0)
            action = possible_idx[index].item()
            if next_transition_possibly_known:
                _trs: list[Transition] = [memory.search_transition([task["Id"]], current_transition) for task in possible_actions]
                return torch.tensor([[action]], device=device, dtype=torch.long), _trs[index], _trs[index] is not None, eps_threshold
            return torch.tensor([[action]], device=device, dtype=torch.long), None, False, eps_threshold
   
    # Second family of stategies: diversification (try to get unseen branches)
    else:
        # Selection stategy 3/4: try to get a new decision never search before
        if next_transition_possibly_known:
            _trs: list[(Transition, any)] = [(memory.search_transition([task["Id"]], current_transition), task) for task in possible_actions]
            if random.random() < DQN_DIVERSITY_RATE:
                random.shuffle(_trs)
                for _t, task in _trs:
                    if _t is None:
                        return torch.tensor([[task["Id"]]], device=device, dtype=torch.long), None, False, eps_threshold
            index = random.randrange(len(possible_actions))
            action = possible_actions[index]["Id"]
            return torch.tensor([[action]], device=device, dtype=torch.long), _trs[index][0], _trs[index][0] is not None, eps_threshold
        
        # Section stategy 4/4: random selection of an action
        action = random.choice(possible_actions)["Id"]
        return torch.tensor([[action]], device=device, dtype=torch.long), None, False, eps_threshold

def _build_batch_indices(actions_local_indices: Tensor, nb_tasks :int, batch_size: int):
    graph_offsets: Tensor = torch.arange(batch_size, device=actions_local_indices.device) * nb_tasks
    actions_global_indices: Tensor = graph_offsets.view(-1, 1) + actions_local_indices
    return actions_global_indices.long()

def optimize_policy_net(memory: Memory, policy_net: HyperGraphGNN, target_net: HyperGraphGNN, optimizer: AdamW, tracker: Tracker, nb_tasks: int, device: Device):
    """
        Optimize the polict network using the Huber loss between selected action and expected best action (based on approx Q-value)
            y = reward r + discounted factor γ x MAX_Q_VALUES(state s+1) predicted with Q_target
            x = predicted quality of (s, a) using the policy network
            L(x, y) = 1/2 (x-y)^2 for small errors (|x-y| ≤ δ) else δ|x-y| - 1/2 x δ^2
    """
    _samples_size = min(len(memory.flat_transitions), BATCH_SIZE)
    sampled_idx: list[int] = random.sample(range(len(memory.flat_transitions)), _samples_size)
    sampled_transitions: list = [memory.flat_transitions[id] for id in sampled_idx]
    b_actions, b_previous_graphs, b_graphs, b_rewards = zip(*[(t.action, t.previous_graph, t.graph, t.reward) for t in sampled_transitions])
    graph_batch: HeteroData = Batch.from_data_list(b_previous_graphs).to(device)
    next_graph_batch: HeteroData = Batch.from_data_list(b_graphs).to(device)
    action_batch = _build_batch_indices(actions_local_indices=torch.cat(b_actions), nb_tasks=nb_tasks, batch_size=_samples_size) # Shape: [batch_size, 1]
    reward_batch: Tensor = torch.cat(b_rewards).squeeze(-1) # Shape: [batch_size]
    state_all_q_values: Tensor = policy_net(graph_batch).squeeze(-1)  # Shape: [num_tasks for all graphs in the batch]
    state_action_q_values: Tensor = state_all_q_values[action_batch.squeeze(-1)]  # Shape: [batch_size]
    with torch.no_grad():
        next_all_q_values: Tensor = target_net(next_graph_batch).squeeze(-1)  # Shape: [num_total_next_tasks]
    next_state_max_q_values: Tensor = global_max_pool(next_all_q_values, next_graph_batch[O].batch)  # Shape: [num_non_final_states < reward_batch]
    expected_state_action_values: Tensor = (next_state_max_q_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_q_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    printed_loss = loss.detach().cpu().item()
    tracker.update(loss_value=printed_loss)
    return printed_loss

def optimize_target_net(policy_net: HyperGraphGNN, target_net: HyperGraphGNN):
    """
        Optimize the target network based on the policy one
    """
    _target_weights = target_net.state_dict()
    _policy_weights = policy_net.state_dict()
    for param in _policy_weights:
        _target_weights[param] = _policy_weights[param] * TAU + _target_weights[param] * (1 - TAU)
    target_net.load_state_dict(_target_weights)