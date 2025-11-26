from v2.src.state import State

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

def take_step(state: State, action: int) -> tuple[State, dict]:
    """
        Take a step in the environment by selecting an action (task) to schedule
    """
    new_state: State = State.from_partial_solution(state)
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
    return new_state, task