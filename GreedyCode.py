import re
import ast
from collections import defaultdict
from queue import Queue

def parse_gsf_file(filename):
    tasks = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                match = re.match(
                    r"^(.*?)\s+\('(.+?)',\s*(\d+)\)\s+([0-9.]+)\s+([0-9.]+)\s+(.*)$",
                    line
                )
                if not match:
                    raise ValueError("Line does not match expected format")

                name = match.group(1).strip()
                processor_type = match.group(2).strip()
                processor_id = int(match.group(3))
                start_time = float(match.group(4))
                end_time = float(match.group(5))
                dependencies_raw = match.group(6).strip()

                try:
                    dependencies = ast.literal_eval(dependencies_raw)
                except:
                    dependencies = []

                tasks.append({
                    'task_id': len(tasks),
                    'task_name': name,
                    'processor_type': processor_type,
                    'start_time': start_time,
                    'end_time': end_time,
                    'processing_time': end_time - start_time,
                    'dependencies': dependencies
                })

            except Exception as e:
                print(f"Reason: {e}\nError parsing line:\n{line}\n")
    return tasks

def build_dependency_map(tasks):
    task_dependencies = defaultdict(list)
    for task in tasks:
        task_dependencies[task["task_name"]] = []
    for task in tasks:
        current_task = task["task_name"]
        for dependent in task["dependencies"]:
            task_dependencies[dependent].append(current_task)
    return task_dependencies


def schedule_tasks(tasks):
    total_tasks = len(tasks)
    print(f"Total Tasks : {total_tasks}")
    dependencies = build_dependency_map(tasks)
    Tasks_Time = {t["task_name"]: t["processing_time"] for t in tasks}
    Task_Processor_Type = {t["task_name"]: t["processor_type"] for t in tasks}  

    processors = {
        "P1":  {"busy": False, "type": 'MIPS'},
        "P2":  {"busy": False, "type": 'MIPS'},
        "P3":  {"busy": False, "type": 'MIPS'},
        "P4":  {"busy": False, "type": 'MIPS'},
        "P5":  {"busy": False, "type": 'MIPS'},
        "P6":  {"busy": False, "type": 'VMP'},
        "P7":  {"busy": False, "type": 'VMP'},
        "P8":  {"busy": False, "type": 'VMP'},
        "P9":  {"busy": False, "type": 'VMP'},
        "P10": {"busy": False, "type": 'PMA'},
        "P11": {"busy": False, "type": 'PMA'},
        "P12": {"busy": False, "type": 'PMAC'},
        "P13": {"busy": False, "type": 'PMAC'},
        "P14": {"busy": False, "type": 'MPC'},
        "P15": {"busy": False, "type": 'MPC'},
        "P16": {"busy": False, "type": 'MPC'},
        "P17": {"busy": False, "type": 'MPC'},
        "P18": {"busy": False, "type": 'MPC'},
    }

    readyQueue = Queue()
    added_to_ready = set()
    current_list = []
    tasks_per_processor = {key: [] for key in processors}
    unprocessed_tasks = set()
    total_time = 0.0
    task_start_time = {}
    task_end_time = {}

    finished_task_set = set()

    for task in tasks:
        task_name = task["task_name"]
        deps = dependencies[task_name]
        if not deps:
            readyQueue.put(task_name)
            added_to_ready.add(task_name)

    while total_tasks > 0:
        tasks_assigned_this_round = 0
        queue_size = readyQueue.qsize()

        for _ in range(queue_size):
            t_id = readyQueue.get() #function get_best
            required_type = Task_Processor_Type[t_id]
            processor_found = None

            for proc_key, proc_info in processors.items():
                if (not proc_info["busy"]) and (proc_info["type"] == required_type):
                    proc_info["busy"] = True
                    current_list.append([proc_key, t_id, Tasks_Time[t_id]])
                    task_start_time[t_id] = total_time
                    processor_found = proc_key
                    tasks_assigned_this_round += 1
                    break

            if not processor_found:
                processor_exists = any(p["type"] == required_type for p in processors.values())
                if processor_exists:
                    readyQueue.put(t_id)
                else:
                    unprocessed_tasks.add(t_id)

        if tasks_assigned_this_round == 0 and not current_list and not readyQueue.empty():
            break

        if not current_list and readyQueue.empty():
            print(total_tasks)
            break

        if current_list:
            min_time = min(item[2] for item in current_list)
            finished_items = [it for it in current_list if (it[2] - min_time) == 0]

            total_time += min_time
            for it in current_list:
                it[2] -= min_time

            total_tasks -= len(finished_items)

            for done_item in finished_items:
                proc_key, done_task_id, _ = done_item
                tasks_per_processor[proc_key].append(done_task_id)
                processors[proc_key]["busy"] = False
                current_list.remove(done_item)
                task_end_time[done_task_id] = total_time
                finished_task_set.add(done_task_id)

            for task_name, deps in dependencies.items():
                new_deps = [d for d in deps if d not in finished_task_set]
                if len(new_deps) < len(deps):
                    dependencies[task_name] = new_deps

                if not new_deps and task_name not in added_to_ready:
                    readyQueue.put(task_name)
                    added_to_ready.add(task_name)

    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(f"Total Execution Time: {total_time:.2f}\n\n")
        for proc, task_ids in tasks_per_processor.items():
            if not task_ids:
                continue
            f.write(f"{proc} ({processors[proc]['type']}):\n")
            for t_id in task_ids:
                start = task_start_time.get(t_id, 0)
                end = task_end_time.get(t_id, 0)
                f.write(f"  Task  - {t_id} | Start: {start:.2f}, End: {end:.2f}\n")
            f.write("\n")


        

def compute_lb1(tasks):
    processor_count = {
        "MIPS": 5,
        "VMP": 3,
        "PMA": 2,
        "PMAC": 2,
        "MPC": 5
    }
    times_by_type = defaultdict(float)
    for t in tasks:
        ptype = t["processor_type"]
        if ptype in processor_count:
            times_by_type[ptype] += t["processing_time"]

    lb1 = 0.0
    for ptype, total_time in times_by_type.items():
        part = total_time / processor_count[ptype]
        lb1 = max(lb1, part)
    return lb1


def compute_lb2(tasks):
    # Get the forward dependency map: task -> list of tasks that depend on it
    forward_graph = build_dependency_map(tasks)
    
    # Build a map: task_name -> processing_time
    task_time = {t["task_name"]: t["processing_time"] for t in tasks}

    # Build a reverse dependency map: task -> tasks it depends on
    reverse_deps = {t["task_name"]: set(t["dependencies"]) for t in tasks}

    # Identify root tasks: tasks that no one depends on
    all_tasks = set(task_time.keys())
    dependent_tasks = set()
    for dep_list in forward_graph.values():
        dependent_tasks.update(dep_list)
    roots = list(all_tasks - dependent_tasks)

    # DFS with memoization to find the longest path
    memo = {}

    def dfs(task):
        if task in memo:
            return memo[task]
        
        max_len = 0
        for child in forward_graph[task]:
            max_len = max(max_len, dfs(child))
        
        memo[task] = task_time[task] + max_len
        return memo[task]

    # Start DFS from all root tasks
    lb2 = 0
    for root in roots:
        lb2 = max(lb2, dfs(root))
    
    return lb2


if __name__ == "__main__":
    filename = "gsf.000001.prof"
    tasks = parse_gsf_file(filename)
    dependencies = build_dependency_map(tasks)
    ready=schedule_tasks(tasks)

    # Write the results of the dependency_map to file
    # with open("deps.txt", "w", encoding="utf-8") as f:
    #  for task in tasks:  # iterate in original order
    #     task_name = task["task_name"]
    #     deps = dependencies[task_name]
    #     if deps:
    #         f.write(f"Task '{task_name}' is dependent on: {deps}\n")
    #     else:
    #         f.write(f"Task '{task_name}' has no dependencies and can be scheduled immediately.\n")

    lb1 = compute_lb1(tasks)
    print(f"LB1 = {lb1:.2f}")

    lb2 = compute_lb2(tasks)
    print(f"LB2 (Critical Path) = {lb2:.2f}")
