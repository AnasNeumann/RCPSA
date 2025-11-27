

import copy
import math
from collections import deque
import networkx as nx
import numpy as np
import copy

import torch
from torch import device as Device
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from v2.conf import O, P, D, R, S
from v2.src.scheduling_functions import find_possible_start_day_for_task
from v2.src.instance_reader import khan_topological_sort
from v2.src.replay_memory import Transition, PossibleAction

# ===========================================
# =*= Model file for an Hyper-Graph State =*=
# ===========================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

class State():
    TASK_FEATURES: int     = 19
    RESOURCE_FEATURES: int = 2
    DEMAND_FEATURES: int   = 1

    def __init__(self, device: Device, p_id: str="", p_make_span: int=0, p_tasks: list=[], p_resources: list=[], p_scheduled_tasks: list=[], std_durations: list = [], lower_bound: int = 0, init_lb: int = 0, upper_bound: int = 0, init_ub: int = 0, indirect_successors: list = [], critical_path: list = [], max_duration: int = 0):
        self.id: str                   = p_id
        self.device                    = device
        self.done: bool                = False
        self.make_span: int            = p_make_span
        self.reward: int               = -100000
        self.tasks: list               = copy.deepcopy(p_tasks)
        self.resources: list           = copy.deepcopy(p_resources)
        self.n_tasks: int              = len(self.tasks)
        self.n_resources: int          = len(self.resources)
        self.scheduled_tasks: list     = copy.deepcopy(p_scheduled_tasks)
        self.indirect_successors: list = indirect_successors if indirect_successors else self._compute_num_indirect_successors()
        self.critical_path             = critical_path if critical_path else self.extract_critical_path()
        if std_durations:
            self.std_durations = std_durations
            self.max_duration  = max_duration
        else:
            self.std_durations, self.max_duration = self._compute_standard_durations()
        self.lower_bound       = lower_bound if lower_bound > 0 else self.compute_lower_bound()
        self.upper_bound       = upper_bound if lower_bound > 0 else self.compute_upper_bound()
        self.init_lb           = init_lb if init_lb > 0 else self.lower_bound
        self.init_ub           = init_ub if init_ub > 0 else self.upper_bound
        self.graph: HeteroData = None

    @classmethod
    def from_partial_solution(cls, s):
        return State(device=s.device, p_id=s.id, p_make_span=s.make_span, p_tasks=s.tasks, p_resources=s.resources, p_scheduled_tasks=s.scheduled_tasks, std_durations=s.std_durations, lower_bound=s.lower_bound, init_lb=s.init_lb, upper_bound=s.upper_bound, init_ub=s.init_ub, indirect_successors=s.indirect_successors, critical_path=s.critical_path, max_duration=s.max_duration)

    @classmethod
    def from_problem(cls, tasks: list, resources: list, device: Device, makespan: int = math.inf):
        return State(device=device, p_id="", p_make_span=makespan, p_tasks=tasks, p_resources=resources, p_scheduled_tasks=[], std_durations=[], lower_bound=0, init_lb=0, upper_bound=0, init_ub=0, indirect_successors=[], critical_path=[], max_duration=0)

    @classmethod
    def from_empty_solution(cls, s, tasks: list, resources: list):
        return State(device=s.device, p_id=s.id, p_make_span=0, p_tasks=tasks, p_resources=resources, p_scheduled_tasks=[], std_durations=s.std_durations, lower_bound=s.lower_bound, init_lb=s.init_lb, upper_bound=s.upper_bound, init_ub=s.init_ub, indirect_successors=s.indirect_successors, critical_path=s.critical_path, max_duration=s.max_duration)

    def compute_lower_bound(self) -> int:
        """
            Combines LB_CPM - Critical Path Method (Longest path based on precedence) and LB_RES - Resource Capacity Bound (Volume of work / Capacity)
        """
        es             = {}
        scheduled_set  = set(self.scheduled_tasks)
        remaining_work = {} 
        in_degree      = {t["Id"]: 0 for t in self.tasks}
        graph          = {t["Id"]: [] for t in self.tasks}
        for t in self.tasks:
            tid = t["Id"]
            for pid in t["Predecessors"]: 
                graph[pid].append(tid)
                in_degree[tid] += 1
            if tid in scheduled_set:
                es[tid] = t.get("Finish", 0) 
            else:
                es[tid] = 0
                reqs = []
                if "Resource" in t:
                    for r_id, _ in self.resources:
                        reqs.append(t["Resource"].get(str(r_id), 0))
                elif "GlobalResources" in t:
                    reqs = t["GlobalResources"]
                for r_idx, amount in enumerate(reqs):
                    if amount > 0:
                        term = t["Duration"] * amount
                        remaining_work[r_idx] = remaining_work.get(r_idx, 0) + term
        queue = deque([t["Id"] for t in self.tasks if in_degree[t["Id"]] == 0])
        max_cpm = 0
        min_unscheduled_es = float('inf') 
        while queue:
            u_id             = queue.popleft()
            current_finish_u = es[u_id]
            if u_id not in scheduled_set:
                task_u             = self.get_task(u_id)
                current_finish_u  += task_u["Duration"]
                min_unscheduled_es = min(min_unscheduled_es, es[u_id])
            max_cpm = max(max_cpm, current_finish_u)
            for v_id in graph[u_id]: 
                if es[v_id] < current_finish_u:
                    es[v_id] = current_finish_u
                in_degree[v_id] -= 1
                if in_degree[v_id] == 0:
                    queue.append(v_id)
        lb_res = 0
        if min_unscheduled_es != float('inf'):
            max_load_duration = 0
            capacities = [r[1] for r in self.resources] 
            for r_idx, work in remaining_work.items():
                cap = capacities[r_idx]
                if cap > 0:
                    load_duration = math.ceil(work / cap)
                    if load_duration > max_load_duration:
                        max_load_duration = load_duration
            lb_res = min_unscheduled_es + max_load_duration
        return max(max_cpm, lb_res)

    def compute_upper_bound(self, priority: str = "slack") -> int:
        """
            Compute a feasible upper bound on the makespan using a serial schedule
            generation scheme from the current partial schedule.
        """
        task_finish = {t["Id"]: t.get("Finish", 0) for t in self.tasks if t["Id"] in self.scheduled_tasks}
        in_degree   = {t["Id"]: len(t["Predecessors"]) for t in self.tasks}
        for t in self.tasks:
            if t["Id"] not in self.scheduled_tasks:
                current_preds = [p for p in t["Predecessors"] if p in task_finish]
                in_degree[t["Id"]] -= len(current_preds)
        usage_profile = {} 
        for t_id in self.scheduled_tasks:
            t        = self.get_task(t_id)
            start    = t.get("Start", 0)
            duration = t["Duration"]
            reqs     = []
            if "Resource" in t:
                for r_id, _ in self.resources:
                    reqs.append(t["Resource"].get(str(r_id), 0))
            elif "GlobalResources" in t:
                reqs = t["GlobalResources"]
            for time in range(start, start + duration):
                if time not in usage_profile: usage_profile[time] = {}
                for r_idx, amount in enumerate(reqs):
                    usage_profile[time][r_idx] = usage_profile[time].get(r_idx, 0) + amount
        eligible = []
        for t in self.tasks:
            if t["Id"] not in self.scheduled_tasks and in_degree[t["Id"]] == 0:
                eligible.append(t)
        current_makespan = max(task_finish.values(), default=0)
        scheduled_count  = len(self.scheduled_tasks)
        capacities       = [r[1] for r in self.resources] 
        while scheduled_count < self.n_tasks:
            if not eligible: break
            if priority   == "duration": eligible.sort(key=lambda t: (-t["Duration"], t["ES"]))
            elif priority == "successors": eligible.sort(key=lambda t: (-len(t["Successors"]), -t["Duration"]))
            else: eligible.sort(key=lambda t: (t["LS"] - t["ES"], -t["Duration"]))
            task     = eligible.pop(0) 
            duration = task["Duration"]
            reqs     = []
            if "Resource" in task:
                for r_id, _ in self.resources:
                    reqs.append(task["Resource"].get(str(r_id), 0))
            elif "GlobalResources" in task:
                reqs = task["GlobalResources"]
            pred_finish_time = 0
            for p_id in task["Predecessors"]:
                f = task_finish.get(p_id, 0)
                if f > pred_finish_time: pred_finish_time = f
            start_time = pred_finish_time
            if duration > 0:
                while True:
                    feasible = True
                    for t in range(start_time, start_time + duration):
                        current_usage = usage_profile.get(t, {})
                        for r_idx, amount in enumerate(reqs):
                            if amount > 0:
                                used = current_usage.get(r_idx, 0)
                                cap  = capacities[r_idx]
                                if used + amount > cap:
                                    feasible = False
                                    break
                        if not feasible: break
                    if feasible: break
                    start_time += 1
            finish_time             = start_time + duration
            task_finish[task["Id"]] = finish_time
            current_makespan        = max(current_makespan, finish_time)
            if duration > 0:
                for t in range(start_time, finish_time):
                    if t not in usage_profile: usage_profile[t] = {}
                    for r_idx, amount in enumerate(reqs):
                        if amount > 0:
                            usage_profile[t][r_idx] = usage_profile[t].get(r_idx, 0) + amount
            for succ_id in task["Successors"]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    eligible.append(self.get_task(succ_id))
            scheduled_count += 1
        return current_makespan

    def _compute_standard_durations(self):
        """
            Durations of tasks measured as a percentage between the min and max duration
        """
        durations    = [t['Duration'] for t in self.tasks]
        min_d, max_d = min(durations), max(durations)
        denom        = max(max_d - min_d, 1)
        return [max(d - min_d, 1e-4) / denom for d in durations], max_d
    
    def _compute_num_indirect_successors(self) -> int:
        """
            Returns how many tasks (direct or indirect) eventually succeed task i in the DAG.
        """
        result: list = []
        for i in range(self.n_tasks):
            visited = set()
            queue = deque(self.tasks[i]["Successors"])
            while queue:
                s = queue.popleft()
                if s not in visited:
                    visited.add(s)
                    for nxt in self.tasks[s]["Successors"]:
                        if nxt not in visited:
                            queue.append(nxt)
            result.append(len(visited))
        return result

    def extract_critical_path(self) -> list[int]:
        """
            Extract a critical path based on tasks with zero (LS - ES) slack.
        """
        task_map = {t["Id"]: t for t in self.tasks}
        critical_ids = {t["Id"] for t in self.tasks if (t["LS"] - t["ES"]) == 0}
        def dfs(current: int, path: list[int]) -> list[int] | None:
            path = path + [current]
            current_task = task_map[current]
            if not current_task["Successors"]:
                return path
            for succ in current_task["Successors"]:
                if succ in critical_ids:
                    result = dfs(succ, path)
                    if result is not None:
                        return result
            return None
        for task in self.tasks:
            if not task["Predecessors"] and task["Id"] in critical_ids:
                cp = dfs(task["Id"], [])
                if cp is not None:
                    return cp
        return []
    
    def to_hyper_graph(self, possible_actions: list[PossibleAction], transitions: list[Transition] = []) -> HeteroData:
        """
            Convert the state to a hypergraph representation
        """
        graph: HeteroData = HeteroData()

        # 1. Operation nodes
        op_features: list = []
        current_progress: float = (len(self.scheduled_tasks) + 1) / len(self.tasks)
        for i, task in enumerate(self.tasks):
            
            if task["Duration"] > 0:
                past_vector: list = [0.0 for _ in range(6)]
                if transitions:
                    for transition in transitions:
                        if transition.action.item() == task["Id"]:
                            cmaxs = np.array(transition.makespans)
                            past_vector = [
                                float(np.min(cmaxs)/ self.init_ub),
                                float(np.percentile(cmaxs, 25)/ self.init_ub),
                                float(np.median(cmaxs)/ self.init_ub),
                                float(np.percentile(cmaxs, 75)/ self.init_ub),
                                float(np.max(cmaxs)/ self.init_ub),
                                math.log1p(transition.nb_visits)
                            ]
                            break
                possible: bool            = task["Id"] in [a.id for a in possible_actions]
                std_Lb: float             = self.lower_bound / self.init_ub if possible else 1.0
                scheduled_flag: float     = 1.0 if task["Id"] in self.scheduled_tasks else 0.0
                feasible_flag: float      = 1.0 if possible else 0.0
                remaining_duration: float = max(task["Finish"] - self.make_span, 0.0) / task["Duration"] if task["Id"] in self.scheduled_tasks else 1.0
                feature_vector = [float(self.std_durations[i]),                               # 1. duration as non-zero percentage of max duration
                                    float(task["ES"] / self.init_ub),                         # 2. earliest start time as percentage of upper bound
                                    float(task["LS"] / self.init_ub),                         # 3. latest start time as percentage of upper bound
                                    float(task["EF"] / self.init_ub),                         # 4. earliest finish time as percentage of upper bound
                                    1.0 if task["Id"] in self.critical_path else 0.0,         # 5. is the task part of the critical path or not?
                                    float(remaining_duration),                                # 6. remaining duration as percentage of task duration
                                    float(task.get("Start", 0.0) / self.init_ub),             # 7. start time as percentage of upper bound
                                    float(task.get("Finish", 0.0) / self.init_ub),            # 8. end time as percentage of upper bound
                                    float(self.indirect_successors[i] / self.n_tasks),        # 9. number of indirect successors as percentage of total tasks
                                    scheduled_flag,                                           # 10. scheduled tast
                                    feasible_flag,                                            # 11. feasibility flag
                                    current_progress,                                         # 12. progress ratio
                                    std_Lb]                                                   # 13. standardized lower bound
                feature_vector.extend([float(v) for v in past_vector])                        # 14->19. past visit vector (min, Q1, median, Q3, max of Cmax + nb visits)
                op_features.append(feature_vector)
            else:
                op_features.append([0.0 for _ in range(self.TASK_FEATURES)])
        graph[O].x = torch.tensor(op_features, dtype=torch.float)

        # 2. Resource nodes
        remaining_tasks_by_resource: list = [0] * self.n_resources
        nb_tasks_by_resource: list = [0] * self.n_resources
        used_capacity = {}
        for r_id, _ in self.resources:
            used_capacity[r_id] = 0
            for task in self.tasks:
                capacity_required: int = task["Resource"].get(str(r_id), 0)
                if capacity_required > 0:
                    nb_tasks_by_resource[r_id - 1] += 1
                    executed: bool = task["Id"] in self.scheduled_tasks
                    if executed:
                        st = task.get("Start", 0)
                        ft = task.get("Finish", 0)
                        if st <= self.make_span < ft:
                            used_capacity[r_id] += capacity_required
                    else:
                        remaining_tasks_by_resource[r_id - 1] += 1
        res_features: list = []
        for (r_id, capacity) in self.resources:
            res_features.append([float((capacity-used_capacity[r_id]) / capacity),                          # 1. current available capacity as percentage of max capacity
                                float(remaining_tasks_by_resource[r_id-1] / nb_tasks_by_resource[r_id-1])]) # 2. remaining task as percentage of total tasks to execute
        graph[R].x = torch.tensor(res_features, dtype=torch.float)

        # 3. Precedence edges
        prec_src: list = []
        prec_dst: list = []
        for i, task in enumerate(self.tasks):
            for succ in task["Successors"]:
                prec_src.append(i)
                prec_dst.append(succ)
        graph[O, P, O].edge_index = torch.tensor([prec_src, prec_dst], dtype=torch.long)
        graph[O, S, O].edge_index = graph[O, P, O].edge_index.flip(0)

        # 4. Requirement edges
        req_src: list = []
        req_dst: list = []
        req_attr: list = []
        for i, task in enumerate(self.tasks):
            for r_idx, (r_id, capacity) in enumerate(self.resources):
                capacity_used: int = task["Resource"].get(str(r_id), 0)
                if capacity_used > 0:
                    capacity_pct: float = float(capacity_used / capacity)
                    req_src.append(i)
                    req_dst.append(r_idx)
                    req_attr.append([capacity_pct]) # 1. resource usage as percentage of max capacity
        graph[O, D, R].edge_index = torch.tensor([req_src, req_dst], dtype=torch.long)
        graph[O, D, R].edge_attr = torch.tensor(req_attr, dtype=torch.float)
        graph[R, D, O].edge_index = graph[O, D, R].edge_index.flip(0)
        graph = graph.to(self.device)
        return graph

    def display_graph(self):
        G = to_networkx(self.graph, node_attrs=['x'], edge_attrs=[])
        nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=10)
