import random
from sortedcontainers import SortedKeyList

import torch
from torch import Tensor
from torch import device as Device

from src.state import State
from src.dqn_functions import check_precedence_feasibility, ssgs
from src.replay_memory import Transition
from conf import ELITS

# ==========================================================
# =*= Genetic Operator (crossover) adapted to GNN solver =*=
# ==========================================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

def key_func(solution: State) -> float:
    return - solution.make_span

class X:
    def __init__(self, tasks: list, resources: list, device: Device):
        self.tasks: list                      = tasks
        self.resources: list                  = resources
        self.precedence: dict[int, list[int]] = {task['Id']: task['Predecessors'] for task in tasks}
        self.device: Device                   = device
        self.n: int                           = len(tasks)
        self.population                       = SortedKeyList(key=key_func)
        self.population_ids                   = set()

    def available(self) -> bool:
        return len(self.population) > 10

    def add(self, solution: State) -> None:
        if solution.id not in self.population_ids:
            if len(self.population) < ELITS:
                self.population.add(solution)
                self.population_ids.add(solution.id)
            elif solution.make_span < self.population[0].make_span:
                worst_obj = self.population[0]
                self.population.pop(0)
                self.population_ids.remove(worst_obj.id)
                self.population.add(solution)
                self.population_ids.add(solution.id)
                print(f"\t New elite solution with makespan={solution.make_span} added in {self.population[-1].make_span} --> {self.population[0].make_span} ({len(self.population)})")

    def sample_good_solution(self) -> State:
        return random.choice(self.population)
    
    def run(self) -> tuple[State, bool, list[Transition]]:
        parent1: list[int] = self.sample_good_solution().scheduled_tasks
        parent2: list[int] = self.sample_good_solution().scheduled_tasks
        decisions: list[int] = self._PPX(parent1=parent1, parent2=parent2) if random.random() <= 0.5 else self._OX(parent1=parent1, parent2=parent2)
        offspring, feasible, transitions = self._execute_solution(decisions)
        return offspring, feasible, transitions

    def _PPX(self, parent1: list[int], parent2: list[int]) -> list[int]:
        """
            Precedence-Preserving Crossover (PPX) operator for genetic algorithms.
        """
        offspring = []
        remaining = set(parent1)  # All tasks to be scheduled
        while remaining:
            eligible = [task for task in remaining 
                        if all(pred in offspring for pred in self.precedence.get(task, []))]
            if not eligible:
                raise ValueError("No eligible tasks found. Check the precedence constraints...")
            chosen_parent = parent1 if random.random() < 0.5 else parent2
            for task in chosen_parent:
                if task in eligible:
                    offspring.append(task)
                    remaining.remove(task)
                    break
        return offspring

    def _OX(self, parent1: list[int], parent2: list[int]) -> list[int]:
        """
            Order Crossover (OX) operator for genetic algorithms.
        """
        start, end = sorted(random.sample(range(self.n), 2))
        offspring = [None] * self.n
        offspring[start:end+1] = parent1[start:end+1]
        p2_filtered = [gene for gene in parent2 if gene not in offspring]
        idx = 0
        for i in range(self.n):
            if offspring[i] is None:
                offspring[i] = p2_filtered[idx]
                idx += 1
        return offspring

    def _execute_solution(self, decisions: list[int]) -> tuple[State, bool, list[Transition]]:
        """
            Compute makespan and check feasibility (also returns the complete list of transitions)
        """
        _current_solution: State      = State.from_empty_solution(self.population[0], self.tasks, self.resources)
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
                _next_lb              = _next_state.compute_lower_bound()
                _delta_LB: int        = _next_lb - _prev_lb
                _delta_duration: int  = max(0, _next_state.make_span - _current_solution.make_span + task["Duration"])
                action_idx: Tensor = torch.tensor([[action_done]], device=self.device, dtype=torch.long)
                transitions.append(Transition(action=action_idx, previous_graph=_current_solution.graph, graph=_next_state.graph, delta_duration=_delta_duration, delta_lb=_delta_LB, lb=_next_lb, parent=transitions[-1] if transitions else None))
                _current_solution = _next_state
                _prev_lb = _next_lb
        _current_solution.done = True
        return _current_solution, feasible, transitions