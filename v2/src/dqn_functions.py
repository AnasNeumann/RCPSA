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

from v2.conf import TAU, BATCH_SIZE, TOP_K, GAMMA, O, TEMPERATURE

from v2.src.neural_nets import HyperGraphGNN 
from v2.src.state import State
from v2.src.replay_memory import Memory
from v2.src.tracker import Tracker

# ==========================================================================
# =*= Reinforcement Learning (DQN) related functions only for GNN solver =*=
# ==========================================================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

mps_amp = (torch.autocast(device_type="mps", dtype=torch.float16) if torch.backends.mps.is_available() else nullcontext())

def select_action(state: State, policy_net: HyperGraphGNN, e: float, greedy: bool, possible_actions: list[dict], device: Device, memory: Memory=None) -> Tensor:
    """
        Select a feasible-only action using the current policy network OR random (when replay memory is still relatively empty)
    """
    action: int      = -1
    if random.random() > e and len(memory.flat_transitions) >= BATCH_SIZE: 
        with torch.inference_mode(), mps_amp:                                
            Q_values: Tensor = policy_net(Batch.from_data_list([state.graph]).to(device))
            possible_idx     = torch.tensor([action['Id'] for action in possible_actions], device=device)
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
        action = random.choice(possible_actions)["Id"]
    return torch.tensor([[action]], device=device, dtype=torch.long)

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
    sampled_idx: list[int]                            = random.sample(range(len(memory.flat_transitions)), _samples_size)
    sampled_transitions: list                         = [memory.flat_transitions[id] for id in sampled_idx]
    b_actions, b_previous_graphs, b_graphs, b_rewards = zip(*[(t.action, t.previous_graph, t.graph, t.reward) for t in sampled_transitions])
    b_dones: Tensor                                   = torch.tensor([len(t.next) == 0 for t in sampled_transitions], device=device, dtype=torch.float32)
    graph_batch: HeteroData                           = Batch.from_data_list(b_previous_graphs).to(device)
    next_graph_batch: HeteroData                      = Batch.from_data_list(b_graphs).to(device)
    action_batch                                      = _build_batch_indices(actions_local_indices=torch.cat(b_actions), nb_tasks=nb_tasks, batch_size=_samples_size) # Shape: [batch_size, 1]
    reward_batch: Tensor                              = torch.cat(b_rewards).squeeze(-1)                      # Shape: [batch_size]
    state_all_q_values: Tensor                        = policy_net(graph_batch).squeeze(-1)                   # Shape: [num_tasks for all graphs in the batch]
    state_action_q_values: Tensor                     = state_all_q_values[action_batch.squeeze(-1)]          # Shape: [batch_size]
    with torch.no_grad():
        next_all_q_values: Tensor            = target_net(next_graph_batch).squeeze(-1)                       # Shape: [num_total_next_tasks]
        next_state_max_q_values: Tensor      = global_max_pool(next_all_q_values, next_graph_batch[O].batch)  # Shape: [num_non_final_states < reward_batch]
        expected_state_action_values: Tensor = reward_batch + (1.0 - b_dones) * next_state_max_q_values * GAMMA
    criterion = nn.SmoothL1Loss(beta=1.0).to(device)
    loss = criterion(state_action_q_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 20)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 2.0)
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