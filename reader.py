import re
import ast
from collections import defaultdict

import re
from collections import defaultdict

def parse_gsf_file(filename):
    tasks = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                match = re.match(
                    r"^(.*?)\s+\('.*?',\s*\d+\)\s+([0-9.]+)\s+([0-9.]+)\s+(.*)$",
                    line
                )
                if not match:
                    raise ValueError("Line does not match expected format")

                name = match.group(1).strip()
                start_time = float(match.group(2))
                end_time = float(match.group(3))
                dependencies_raw = match.group(4).strip()

                try:
                    dependencies = ast.literal_eval(dependencies_raw)
                except:
                    dependencies = []

                tasks.append({
                    'task_name': name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'dependencies': dependencies
                })

            except Exception as e:
                print(f"Reason: {e}\nError parsing line:\n{line}\n")
    return tasks



def build_dependency_map(tasks):
    """
    Builds a mapping for each task to the list of tasks it depends on.
    (key = dependent task, value = list of prerequisite task names)
    """
    task_dependencies = defaultdict(list)

    # Ensure all tasks exist in the map
    for task in tasks:
        task_dependencies[task["task_name"]] = []

    # Fill dependencies: dependent -> prerequisite
    for task in tasks:
        current_task = task["task_name"]
        for dependent in task["dependencies"]:
            task_dependencies[dependent].append(current_task)

    return task_dependencies


if __name__ == "__main__":
    filename = "gsf.000000.prof"
    tasks = parse_gsf_file(filename)
    dependency_map = build_dependency_map(tasks)

    # Write results to file
    with open("output.txt", "w", encoding="utf-8") as f:
     for task in tasks:  # iterate in original order
        task_name = task["task_name"]
        deps = dependency_map[task_name]
        if deps:
            f.write(f"Task '{task_name}' is dependent on: {deps}\n")
        else:
            f.write(f"Task '{task_name}' has no dependencies and can be scheduled immediately.\n")

