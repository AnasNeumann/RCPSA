import math
import random
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch import device as Device
from torch import Tensor
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_max_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau

from conf import TAU, BATCH_SIZE, TOP_K, GAMMA, O, TEMPERATURE, INFINITY

from src.neural_nets import HyperGraphGNN 
from src.state import State
from src.replay_memory import Memory, PossibleAction
from src.tracker import Tracker
from src.scheduling_functions import find_possible_start_day_for_task

# ==========================================================================
# =*= Reinforcement Learning (DQN) related functions only for GNN solver =*=
# ==========================================================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

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
    min_start_day   = 1
    predecessor_ids = task["Predecessors"]
    predecessors    = [t for t in tasks if t['Id'] in predecessor_ids]
    for predecessor in predecessors:
        if predecessor["Finish"] >= min_start_day:
            min_start_day = predecessor["Finish"] + (not (not (predecessor["Duration"] * task["Duration"])))
    start_day = find_possible_start_day_for_task(tasks, resources, task, min_start_day, ub)
    if start_day > 0:
        task["Start"]  = start_day
        task["Finish"] = start_day + task["Duration"] - (not (not (task["Duration"])))
    return task

def build_impossible_state(impossible_state: State, task: dict):
    """
        Build an impossible state with high penalties
        Works for both type of state (simple matrix and transformer)
    """
    impossible_state.scheduled_tasks.append(task["Id"])
    impossible_state.id = f'{impossible_state.id}_{task["Id"]}'
    impossible_state.make_span = INFINITY
    impossible_state.reward = -INFINITY
    impossible_state.done = True

class HypotheticalStep:
    """
        Context manager to temporarily apply a task to the state, 
        compute metrics, and revert the state exactly as it was.
    """
    def __init__(self, state: State, task_id: int):
        self.state: State        = state
        self.task_id: int        = task_id
        self.task: dict          = None
        self.old_start: int      = 0
        self.old_finish: int     = 0
        self.old_makespan: int   = state.make_span
        self.was_scheduled: bool = False
        self.success: bool       = False

    def __enter__(self):
        self.task       = next(t for t in self.state.tasks if t["Id"] == self.task_id)
        self.old_start  = self.task.get("Start", 0)
        self.old_finish = self.task.get("Finish", 0)
        feasible = True
        for p_id in self.task["Predecessors"]:
            if p_id not in self.state.scheduled_tasks:
                feasible = False
                break
        if feasible:
            updated_task = ssgs(self.state.tasks, self.state.resources, self.task, 10000)
            if updated_task["Start"] > 0:
                self.state.scheduled_tasks.append(self.task_id)
                self.state.make_span = max(self.state.make_span, updated_task["Finish"])
                self.was_scheduled   = True
                self.success         = True     
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.was_scheduled:
            self.state.scheduled_tasks.pop()
            self.state.make_span = self.old_makespan
            self.task["Start"]   = self.old_start
            self.task["Finish"]  = self.old_finish

def take_step(state: State, action: int):
    """
        Take a step in the environment by selecting an action (task) to schedule
    """
    try:
        task = [t for t in state.tasks if t["Id"] == action][0]
    except:
        print(f"{action} not found")
        exit()
    feasible: bool = check_precedence_feasibility(state, task)
    task: dict     = ssgs(state.tasks, state.resources, task, 10000)
    if not feasible or task["Start"] <= 0:
        build_impossible_state(state, task)
    else:
        state.scheduled_tasks.append(task["Id"])
        state.id = f'{state.id}_{task["Id"]}' if task["Id"] > 0 else f'{task["Id"]}'
        state.make_span = max(state.make_span, task["Finish"])
        if len(state.scheduled_tasks) == len(state.tasks):
            state.done = True

mps_amp = (torch.autocast(device_type="mps", dtype=torch.float16) if torch.backends.mps.is_available() else nullcontext())

def select_action(state: State, policy_net: HyperGraphGNN, e: float, greedy: bool, possible_actions: list[PossibleAction], device: Device, memory: Memory=None) -> Tensor:
    """
        Select a feasible-only action using the current policy network OR random (when replay memory is still relatively empty)
    """
    action: int      = -1
    if random.random() > e and len(memory.flat_transitions) >= BATCH_SIZE: 
        with torch.inference_mode(), mps_amp:                                
            Q_values: Tensor = policy_net(Batch.from_data_list([state.graph]).to(device))
            possible_idx     = torch.tensor([action.id for action in possible_actions], device=device)
            selected_values  = Q_values[possible_idx].squeeze(-1)
            if greedy:
                _, index     = selected_values.max(0)
            else:
                topk      = min(TOP_K, len(selected_values))                          # robust value     
                vals, idx = torch.topk(selected_values.view(-1), k=topk)              # largest-Q actions
                vals      = torch.nan_to_num(vals, nan=-1e9, posinf=1e9, neginf=-1e9) # finite
                vals      = vals - vals.max()                                         # improves soft‑max stability
                p         = torch.softmax(vals / TEMPERATURE, dim=0)                  # Boltzmann exploration
                index     = idx[torch.multinomial(p, 1)].item()
            action        = possible_idx[index].item()
    else:
        action = random.choice(possible_actions).id
    return torch.tensor([[action]], device=device, dtype=torch.long)

def _build_batch_indices(actions_local_indices: Tensor, nb_tasks :int, batch_size: int):
    graph_offsets: Tensor = torch.arange(batch_size, device=actions_local_indices.device) * nb_tasks
    actions_global_indices: Tensor = graph_offsets.view(-1, 1) + actions_local_indices
    return actions_global_indices.long()

def optimize_policy_net(memory: Memory, policy_net: HyperGraphGNN, target_net: HyperGraphGNN, optimizer: AdamW, scheduler: ReduceLROnPlateau, tracker: Tracker, nb_tasks: int, device: Device):
    """
        Optimize the polict network using the Huber loss between selected action and expected best action (based on approx Q-value)
            y = reward r + discounted factor γ x MAX_Q_VALUES(state s+1) predicted with Q_target
            x = predicted quality of (s, a) using the policy network
            L(x, y) = 1/2 (x-y)^2 for small errors (|x-y| ≤ δ) else δ|x-y| - 1/2 x δ^2
    """
    _samples_size: int                                = min(len(memory.flat_transitions), BATCH_SIZE)
    sampled_transitions: list                         = random.sample(list(memory.flat_transitions), _samples_size)
    b_actions, b_previous_graphs, b_graphs, b_rewards = zip(*[(t.action, t.previous_graph, t.graph, t.reward) for t in sampled_transitions])
    b_dones: Tensor                                   = torch.tensor([len(t.next) == 0 for t in sampled_transitions], device=device, dtype=torch.float32)
    graph_batch: HeteroData                           = Batch.from_data_list(b_previous_graphs).to(device)
    next_graph_batch: HeteroData                      = Batch.from_data_list(b_graphs).to(device)
    action_batch                                      = _build_batch_indices(actions_local_indices=torch.cat(b_actions), nb_tasks=nb_tasks, batch_size=_samples_size) # Shape: [batch_size, 1]
    reward_batch: Tensor                              = torch.cat(b_rewards).squeeze(-1)                      # Shape: [batch_size]
    state_all_q_values: Tensor                        = policy_net(graph_batch).squeeze(-1)                   # Shape: [num_tasks for all graphs in the batch]
    state_action_q_values: Tensor                     = state_all_q_values[action_batch.squeeze(-1)]          # Shape: [batch_size]
    with torch.no_grad():
        feasible_mask                        = next_graph_batch[O].x[:, 10].bool()
        next_all_q_values: Tensor            = target_net(next_graph_batch).squeeze(-1)                       # Shape: [num_total_next_tasks]
        next_all_q_values[~feasible_mask]    = -float('inf')
        next_state_max_q_values: Tensor      = global_max_pool(next_all_q_values, next_graph_batch[O].batch)  # Shape: [num_non_final_states < reward_batch]
        expected_state_action_values: Tensor = reward_batch + (1.0 - b_dones) * next_state_max_q_values * GAMMA    
    criterion = nn.SmoothL1Loss(beta=1.0).to(device)
    loss = criterion(state_action_q_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 20)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 2.0)
    optimizer.step()
    scheduler.step(loss)
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