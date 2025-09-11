# ===========================
# =*= Scheduling function =*=
# ===========================
__author__  = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "MIT License"

def find_active_tasks_on_day(tasks: list[dict], day: int) -> list[dict]:
    active_tasks = [task for task in tasks if task["Start"] <= day <= task["Finish"]]
    return active_tasks

def check_resource_feasibility_on_day(resources: list[tuple], tasks: list[dict], task: dict, day: int) -> bool:
    active_tasks = find_active_tasks_on_day(tasks, day)
    consumption = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0
    }
    for t in active_tasks: # sum consumption by resource for active tasks
        for i in range(len(t["Resource"])):
            consumption[f"{i + 1}"] += t["Resource"][f"{i + 1}"]
    for i in range(len(task["Resource"])): # add resource of the selected task
        consumption[f"{i + 1}"] += task["Resource"][f"{i + 1}"]
    for r in resources:
        if r[1] < consumption[f"{r[0]}"]:
            return False
    return True

def find_possible_start_day_for_task(tasks: list[dict], resources: list[tuple], task: dict, possible_day: int, horizon: int) -> int:
    day = possible_day
    while day <= horizon:
        feasible = True
        day      = possible_day
        duration = 1
        while duration <= task["Duration"]:
            feasible = check_resource_feasibility_on_day(resources, tasks, task, day)
            if not feasible:
                possible_day = day + 1
                break
            duration += 1
            day      += 1
        if feasible:
            return possible_day
    return -1

def find_feasible_tasks(tasks: list[dict], processed_tasks: list[int]) -> list[dict]:
    feasible_tasks = []
    for t in tasks:
        if t["Id"] not in processed_tasks:
            feasible = True
            for p in t["Predecessors"]:
                if p not in processed_tasks:
                    feasible = False
                    break
            if feasible:
                feasible_tasks.append(t)
    return feasible_tasks