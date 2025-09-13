

import copy
import math
from collections import deque
import networkx as nx

import torch
from torch import device as Device
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from conf import O, P, D, R, S
from src.scheduling_functions import find_possible_start_day_for_task
from src.instance_reader import khan_topological_sort

# ===========================================
# =*= Model file for an Hyper-Graph State =*=
# ===========================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

class State():
    TASK_FEATURES: int     = 9
    RESOURCE_FEATURES: int = 2
    DEMAND_FEATURES: int   = 1

    def __init__(self, device: Device, p_id: str="", p_make_span: int=0, p_tasks: list=[], p_resources: list=[], p_scheduled_tasks: list=[], std_durations: list = [], lower_bound: int = 0, init_lb: int = 0, indirect_successors: list = [], critical_path: list = [], max_duration: int = 0, build_graph: bool = False):
        self.id: str                   = p_id
        self.device                    = device
        self.source: str               = "DQN"
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
        self.init_lb  = init_lb if init_lb > 0 else self._precedence_only_critical_path()
        self.graph: HeteroData = self.to_hyper_graph() if build_graph else None

    @classmethod
    def from_partial_solution(cls, s, build_graph: bool = False):
        return State(device=s.device, p_id=s.id, p_make_span=s.make_span, p_tasks=s.tasks, p_resources=s.resources, p_scheduled_tasks=s.scheduled_tasks, std_durations=s.std_durations, lower_bound=s.lower_bound, init_lb=s.init_lb, indirect_successors=s.indirect_successors, critical_path=s.critical_path, max_duration=s.max_duration, build_graph=build_graph)

    @classmethod
    def from_problem(cls, tasks: list, resources: list, device: Device, makespan: int = math.inf):
        return State(device=device, p_id="", p_make_span=makespan, p_tasks=tasks, p_resources=resources, p_scheduled_tasks=[], std_durations=[], lower_bound=0, init_lb=0, indirect_successors=[], critical_path=[], max_duration=0, build_graph=True)

    @classmethod
    def from_empty_solution(cls, s, tasks: list, resources: list):
        return State(device=s.device, p_id=s.id, p_make_span=0, p_tasks=tasks, p_resources=resources, p_scheduled_tasks=[], std_durations=s.std_durations, lower_bound=s.lower_bound, init_lb=s.init_lb, indirect_successors=s.indirect_successors, critical_path=s.critical_path, max_duration=s.max_duration, build_graph=True)
    
    def get_possible_dates(self, tasks: list[dict], resources: list[tuple[int, int]], task: dict, ub: int) -> dict:
        """
            Get possible dates for a task
        """
        min_start_day   = 0
        predecessor_ids = task["Predecessors"]
        predecessors    = [t for t in tasks if t['Id'] in predecessor_ids]
        for predecessor in predecessors:
            if predecessor["Finish"] >= min_start_day:
                min_start_day = predecessor["Finish"] + (not (not (predecessor["Duration"] * task["Duration"])))
        start_day = find_possible_start_day_for_task(tasks, resources, task, min_start_day, ub)
        return start_day, start_day + task["Duration"] - (not (not (task["Duration"])))
    
    def _precedence_only_critical_path(self) -> int:
        # Earliest times ignoring resources, no fixed Start/Finish enforced.
        task_map = {t["Id"]: t for t in self.tasks}
        order = khan_topological_sort(self.tasks)
        EF = {tid: 0 for tid in task_map}
        for tid in order:
            t = task_map[tid]
            es = 0 if not t["Predecessors"] else max(EF[p] for p in t["Predecessors"])
            EF[tid] = es + t["Duration"]
        return max(EF.values()) if EF else 0

    def compute_lower_bound(self):
        lb_cp_cond = self._conditional_precedence_lb()
        lb_res_rem = self._resource_load_lb_remaining()
        mk:int     = self.make_span if self.make_span >= 0 and self.make_span != math.inf else 0
        return max(mk, lb_cp_cond, mk + lb_res_rem)
    
    def _conditional_precedence_lb(self) -> int:
        """
            Precedence-only earliest completion while respecting any fixed Start/Finish
            for already-scheduled tasks. Ignores resource capacities → valid LB.
        """
        tasks = self.tasks
        task_map = {t["Id"]: t for t in tasks}
        order = khan_topological_sort(tasks)
        EF = {}
        for tid in order:
            t = task_map[tid]
            if t.get("Start", 0) > 0 or t.get("Finish", 0) > 0:
                st = t.get("Start", 0)
                ft = t.get("Finish", st + t["Duration"])
                if t["Predecessors"]:
                    min_es = max(EF[p] for p in t["Predecessors"])
                    st = max(st, min_es)
                    ft = st + t["Duration"]
                EF[tid] = ft
            else:
                es = 0 if not t["Predecessors"] else max(EF[p] for p in t["Predecessors"])
                EF[tid] = es + t["Duration"]
        return max(EF.values()) if EF else self.make_span

    def _resource_load_lb_remaining(self) -> int:
        """
            Resource-load LB for remaining (unscheduled) work.
            For each renewable resource k: ceil( sum_i a[i,k] * p[i] / C[k] ).
            This lower-bounds the remaining *added* time due to capacity limits.
        """
        capacity       = {rid: c for (rid, c) in self.resources}
        remaining_work = {rid: 0 for rid, _ in self.resources}
        scheduled      = set(self.scheduled_tasks)
        for t in self.tasks:
            if t["Id"] in scheduled:
                continue
            p = t["Duration"]
            for rid, _ in self.resources:
                demand = t["Resource"].get(str(rid), 0)
                if demand > 0:
                    remaining_work[rid] += demand * p
        lb_list = []
        for rid, w in remaining_work.items():
            Ck = max(capacity[rid], 1)
            lb_list.append(math.ceil(w / Ck))
        return max(lb_list) if lb_list else 0

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
    
    def to_hyper_graph(self) -> HeteroData:
        """
            Convert the state to a hypergraph representation
        """
        graph: HeteroData = HeteroData()

        # 1. Operation nodes
        op_features: list = []
        for i, task in enumerate(self.tasks):
            if task["Duration"] > 0:
                start_step: int = self.scheduled_tasks.index(i) if i in self.scheduled_tasks else -1
                remaining_duration: float = max(0, task["Duration"] - self.make_span + start_step) / task["Duration"] if start_step >= 0 else 1.0
                op_features.append([float(self.std_durations[i]),                            # 1. duration as non-zero percentage of max duration
                                    float(task["ES"] / self.init_lb),                        # 2. earliest start time as percentage of lower bound
                                    float(task["LS"] / self.init_lb),                        # 3. latest start time as percentage of lower bound
                                    float(task["EF"] / self.init_lb),                        # 4. earliest finish time as percentage of lower bound
                                    float(1.0 if task["Id"] in self.critical_path else 0.0), # 5. is the task part of the critical path or not?
                                    float(remaining_duration),                               # 6. remaining duration as percentage of task duration
                                    float(task.get("Start", 0.0) / self.init_lb),            # 7. start time as percentage of lower bound
                                    float(task.get("Finish", 0.0) / self.init_lb),           # 8. end time as percentage of lower bound
                                    float(self.indirect_successors[i] / self.n_tasks)])      # 9. number of indirect successors as percentage of total tasks
            else:
                op_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
                    executed: bool = i in self.scheduled_tasks
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
        graph.to(self.device)
        return graph

    def display_graph(self):
        G = to_networkx(self.graph, node_attrs=['x'], edge_attrs=[])
        nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=10)