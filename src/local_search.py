import random
import copy

import torch
from torch import Tensor
from torch import device as Device

from src.state import State
from src.dqn_functions import check_precedence_feasibility, ssgs
from src.replay_memory import Transition
from conf import MAX_ITTRS

# ===================================================
# =*= Local Search operator adapted to GNN solver =*=
# ====================================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

def local_search(current_solution: State, tasks: list, resources: list, device: Device, max_iterations: int = MAX_ITTRS, force: bool = True) -> tuple[bool, State, list[Transition]]:
    """
        Performs a local search on an RCPSP solution using two neighborhood moves
    """
    best_solution: State          = current_solution
    transitions: list[Transition] = []
    n: int                        = len(current_solution.scheduled_tasks)
    found_new_solution: bool      = False
    for _ in range(max_iterations):
        neighborhood_type: list[str] = random.choices([["S", "R", "M"], ["sR", "rS", "sM", "mS", "rM", "mR"]], weights=[0.65, 0.35], k=1)[0]
        c: str = random.choice(neighborhood_type)
        match c:
            case "S":
                neighbor: list[int] = _swap_two_tasks(best_solution.scheduled_tasks, n)
            case "R":
                neighbor: list[int] = _reinsert_task(best_solution.scheduled_tasks, n)
            case "M":
                neighbor: list[int] = _block_move(best_solution.scheduled_tasks, n)
            case "sR":
                neighbor: list[int] = _reinsert_task(_swap_two_tasks(best_solution.scheduled_tasks, n), n)
            case "rS":
                neighbor: list[int] = _swap_two_tasks(_reinsert_task(best_solution.scheduled_tasks, n), n)
            case "sM":
                neighbor: list[int] = _block_move(_swap_two_tasks(best_solution.scheduled_tasks, n), n)
            case "mS":
                neighbor: list[int] = _swap_two_tasks(_block_move(best_solution.scheduled_tasks, n), n)
            case "mR":
                neighbor: list[int] = _reinsert_task(_block_move(best_solution.scheduled_tasks, n), n)
            case "rM":
                neighbor: list[int] = _block_move(_reinsert_task(best_solution.scheduled_tasks, n), n)
            case _:
                print("Error: Unknown mutation operator")
        current_solution, _feasible, _transitions = _execute_solution(tasks, resources, best_solution, neighbor, device)
        if _feasible and ((force and current_solution.make_span < best_solution.make_span) or (not force and current_solution.make_span <= best_solution.make_span)):
            found_new_solution = True
            transitions        = _transitions
            best_solution      = current_solution
    return found_new_solution, best_solution, transitions

def _swap_two_tasks(decisions: list, n: int) -> list[int]:
    """
        Mutation operator 1: swap the position of two randomly selected tasks (smallest neighborhood)
    """
    new_solution = copy.deepcopy(decisions)
    i, j = random.sample(range(1, n-1), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def _reinsert_task(decisions: list, n: int) -> list[int]:
    """
        Mutation operator 2: insert a task at a new random place
    """
    new_solution = copy.deepcopy(decisions)
    i = random.randrange(1, n-1)
    task_to_move = new_solution.pop(i)
    j = random.randrange(1, n)
    new_solution.insert(j, task_to_move)
    return new_solution

def _block_move(decisions: list, n: int) -> list[int]:
    """
        Mutation operator 3: move a whole bloc of tasks (largest neighborhood)
    """
    new_solution = copy.deepcopy(decisions)
    i, j = random.sample(range(1, n-1), 2)
    if i > j:
        i, j = j, i
    block = new_solution[i:j+1]
    del new_solution[i:j+1]
    k = random.randrange(n)
    for idx, task_id in enumerate(block):
        new_solution.insert(k + idx, task_id)
    return new_solution

def _execute_solution(tasks: list, resources: list, base_solution: State, decisions: list, _device: Device):
    """
        Compute makespan and check feasibility (also returns the complete list of transitions)
    """
    _current_solution: State      = State.from_empty_solution(base_solution, tasks, resources)
    transitions: list[Transition] = []
    feasible: bool                = True
    _init_lb: int                 = _current_solution.compute_lower_bound()
    _prev_lb: int                 = _init_lb
    for _, action_done in enumerate(decisions):
        _next_state: State = State.from_partial_solution(_current_solution, build_graph=False)
        task               = [t for t in _next_state.tasks if t["Id"] == action_done][0]
        feasible: bool     = check_precedence_feasibility(_current_solution, task)
        task               = ssgs(_next_state.tasks, _next_state.resources, task, 10000)
        if not feasible or task["Start"] <= 0:
            feasible = False
            break
        else:
            _next_state.scheduled_tasks.append(task["Id"])
            _next_state.id        = f'{_next_state.id}_{task["Id"]}' if task["Id"] > 0 else f'{task["Id"]}'
            _next_state.make_span = max(_next_state.make_span, task["Finish"])
            _next_state.graph     = _next_state.to_hyper_graph()
            action_idx: Tensor    = torch.tensor([[action_done]], device=_device, dtype=torch.long)
            _next_lb              = _next_state.compute_lower_bound()
            _delta_LB: int        = _next_lb - _prev_lb
            _delta_duration: int  = max(0, _next_state.make_span - _current_solution.make_span + task["Duration"])
            transitions.append(Transition(action=action_idx, previous_graph=_current_solution.graph, graph=_next_state.graph, delta_duration=_delta_duration, delta_lb=_delta_LB, lb=_next_lb, parent=transitions[-1] if transitions else None))
            _current_solution     = _next_state
            _prev_lb              = _next_lb
    _current_solution.done = True
    return _current_solution, feasible, transitions