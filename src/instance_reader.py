# ==========================================
# =*= FUNCTIONS TO READ AN INSTANCE FILE =*=
# ==========================================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

def build_instance(instance_path: str, print_file: bool = True) -> tuple[list[dict], list[dict]]:
    """
        Build a RCPSP instance to solve (from a local file)
    """
    _tasks, _resources, _ = read_j(instance_path, print_file)
    for task in _tasks:
        successors = [t for t in _tasks if t['Id'] in task["Successors"]]
        for s in successors:
            s["Predecessors"].append(task["Id"])
    _tasks = compute_earliest_times(_tasks)
    _tasks = compute_latest_times(_tasks)
    return _tasks, _resources

def compute_earliest_times(tasks: list[int]) -> None:
    """
        Computes the Earliest Start (ES) and Earliest Finish (EF) for each task.
    """
    task_map = {t["Id"]: t for t in tasks}
    order = khan_topological_sort(tasks)
    for tid in order:
        task = task_map[tid]
        if task["Predecessors"]:
            task["ES"] = max(task_map[pid]["EF"] for pid in task["Predecessors"])
        else:
            task["ES"] = 0
        task["EF"] = task["ES"] + task["Duration"]
    return tasks

def compute_latest_times(tasks: list[int]) -> None:
    """
        Computes the Latest Finish (LF) and Latest Start (LS) times for each task, working backwards from the project finish time.
    """
    task_map = {t["Id"]: t for t in tasks}
    order = khan_topological_sort(tasks)
    project_finish = max(t["EF"] for t in tasks if not t["Successors"])
    for tid in reversed(order):
        task = task_map[tid]
        if task["Successors"]:
            task["LF"] = min(task_map[s]["LS"] for s in task["Successors"])
        else:
            task["LF"] = project_finish
        task["LS"] = task["LF"] - task["Duration"]
    return tasks

def khan_topological_sort(tasks: list[int]) -> list[int]:
    """
        returns a list of task IDs in a valid execution order according to precedence
    """
    in_degree = {t["Id"]: len(t["Predecessors"]) for t in tasks}
    sorted_ids = []
    zero_in_degree = [t["Id"] for t in tasks if in_degree[t["Id"]] == 0]
    while zero_in_degree:
        current = zero_in_degree.pop(0)
        sorted_ids.append(current)
        for t in tasks:
            if current in t["Predecessors"]:
                in_degree[t["Id"]] -= 1
                if in_degree[t["Id"]] == 0:
                    zero_in_degree.append(t["Id"])
    return sorted_ids

def read_j(filename, print_file: bool = True):
    if print_file:
        print(f"Start > Reading input file: {filename}")
    file = open(filename, "r")
    nb_tasks = 0
    nb_resources = 0
    resources_capacities = []
    tasks_list = []
    succession_list = []
    for i, line in enumerate(file, 1):
        line_list = [int(num) for num in line.split() if num.strip().isdigit()]
        if i == 1:
            nb_tasks, nb_resources = line_list[0], line_list[1]
            if print_file:
                print("\t\tTask size:", nb_tasks, "Resource size:", nb_resources)
        elif i == 2:
            for j in range(0, nb_resources):
                resources_capacities.append((j + 1, line_list[j]))
        else:
            resource_demand = {}
            for j in range(0, nb_resources):
                resource_demand[f"{j + 1}"] = line_list[j + 1]
            task = dict(Task=f"Task {i - 3}",
                        Start=0, Finish=0,
                        Duration=line_list[0],
                        Id=i - 3,
                        ES=0, EF=0,
                        LS=0, LF=0,
                        Predecessors=[],
                        Successors=[],
                        Resource=resource_demand)
            tasks_list.append(task)
            for j in range(1 + nb_resources + 1, len(line_list)):
                succession_list.append((i - 3, line_list[j] - 1))
                task["Successors"].append(line_list[j] - 1)
    file.close()
    if print_file:
        print(f"End > Reading input file : {filename}")
    return tasks_list, resources_capacities, succession_list