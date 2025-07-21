                                                                                                                                                                                                                                                                                               
import re
import ast
from collections import defaultdict, deque
from queue import Queue
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

# -------------------- Q-Network and Replay Buffer --------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action_index, reward, next_state):
        self.buffer.append((state, action_index, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
        )

    def __len__(self):
        return len(self.buffer)

# -------------------- Utilities --------------------
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
                print(f"Error parsing line: {e}\n{line}")
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

def count_tasks_depending_on_each(tasks):
    dependencies = build_dependency_map(tasks)
    dependents_count = defaultdict(int)
    for task in tasks:
        dependents_count[task["task_name"]] = 0
    for task, deps in dependencies.items():
        for dep in deps:
            dependents_count[dep] += 1
    return dependents_count

def encode_state(global_data, ready_tasks, processors, dependencies, dependents_count, max_ready_tasks=300):
    state = []
    types = ['MIPS', 'VMP', 'PMA', 'PMAC', 'MPC']

    state.append(global_data["current_time"] / global_data["max_time"])
    state.append(global_data["remaining_tasks"] / global_data["total_tasks"])
    state.append(global_data["completed_tasks"] / global_data["total_tasks"])
    state.append(len(ready_tasks) / global_data["total_tasks"])

    for t in types:
        free = sum(1 for p in processors.values() if p['type'] == t and not p['busy'])
        state.append(free / global_data["processor_counts"][t])
    for t in types:
        load = sum(1 for p in processors.values() if p['type'] == t and p['busy'])
        state.append(load / global_data["processor_counts"][t])

    for i in range(max_ready_tasks):
        if i < len(ready_tasks):
            task = ready_tasks[i]
            one_hot = [1 if t == task['processor_type'] else 0 for t in types]
            time = task['processing_time'] / global_data["max_task_time"]
            num_deps = dependents_count.get(task["task_name"], 0) / global_data["max_deps"]
            state.extend([time] + one_hot + [num_deps])
        else:
            state.extend([0.0] + [0]*len(types) + [0.0])
    return torch.tensor([state], dtype=torch.float32)

# -------------------- RL Training --------------------
def train_q_network(q_network, optimizer, replay_buffer, gamma=0.99, batch_size=64):
    if len(replay_buffer) < batch_size:
        return None
    states, actions, rewards, next_states = replay_buffer.sample(batch_size)
    q_values = q_network(states).squeeze(2)
    next_q_values = q_network(next_states).squeeze(2)

    selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
    max_next_q_values = next_q_values.max(1)[0]
    targets = rewards + gamma * max_next_q_values

    loss = F.mse_loss(selected_q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def select_action(q_network, state, epsilon, ready_count):
    if random.random() < epsilon:
        return random.randint(0, ready_count - 1)
    with torch.no_grad():
        q_values = q_network(state).squeeze(0)
        return torch.argmax(q_values[:ready_count]).item()

# -------------------- Main Training Loop --------------------
def run_training(tasks, episodes=100):
    dependents_count = count_tasks_depending_on_each(tasks)
    max_task_time = max(task["processing_time"] for task in tasks)
    max_deps = max(dependents_count.values()) if dependents_count else 1

    state_dim = 4 + 10 + 300 * 7
    q_network = QNetwork(state_dim)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)
    buffer = ReplayBuffer()
    loss_log = []
    reward_log = []

    for ep in range(episodes):
        print(f"[Episode {ep}]")
        tasks_copy = [t.copy() for t in tasks]
        deps = build_dependency_map(tasks_copy)
        processors = {f"P{i+1}": {"busy": False, "type": tp} for i, tp in enumerate(
            ['MIPS']*5 + ['VMP']*4 + ['PMA']*2 + ['PMAC']*2 + ['MPC']*5)}
        processor_counts = {tp: sum(1 for p in processors.values() if p['type'] == tp) for tp in ['MIPS','VMP','PMA','PMAC','MPC']}

        time = 0.0
        finished_set = set()
        ready = [t for t in tasks_copy if not t["dependencies"]]
        added = set(t["task_name"] for t in ready)
        running = []
        rewards = 0

        while len(finished_set) < len(tasks):
            global_data = {
                "current_time": time,
                "total_tasks": len(tasks),
                "remaining_tasks": len(tasks) - len(finished_set),
                "completed_tasks": len(finished_set),
                "processor_counts": processor_counts,
                "max_time": 10000,
                "max_task_time": max_task_time,
                "max_deps": max_deps
            }

            state = encode_state(global_data, ready, processors, deps, dependents_count)
            if not ready:
                time += 1
                continue

            action_index = select_action(q_network, state, epsilon=0.1, ready_count=len(ready))
            task = ready[action_index]

            proc = None
            for k, v in processors.items():
                if not v["busy"] and v["type"] == task["processor_type"]:
                    proc = k
                    break
            if not proc:
                time += 1
                print('hi')
                continue

            duration = task["processing_time"]
            processors[proc]["busy"] = True
            time += duration
            processors[proc]["busy"] = False
            finished_set.add(task["task_name"])

            for t in tasks_copy:
                t["dependencies"] = [d for d in t["dependencies"] if d != task["task_name"]]
            new_ready = [t for t in tasks_copy if not t["dependencies"] and t["task_name"] not in added and t["task_name"] not in finished_set]
            for t in new_ready:
                added.add(t["task_name"])
                ready.append(t)

            next_state = encode_state(global_data, ready, processors, deps, dependents_count)
            reward = -duration
            rewards += reward
            buffer.add(state, action_index, reward, next_state)
            train_loss = train_q_network(q_network, optimizer, buffer)
            if train_loss is not None:
                loss_log.append(train_loss)

            ready.pop(action_index)

        print(f"Episode {ep}] Total Reward: {rewards}")
        reward_log.append(rewards)

    torch.save(q_network.state_dict(), "q_scheduler_model.pt")
    with open("reward_log.txt", "w") as f:
        f.writelines([f"{r}\n" for r in reward_log])
    with open("loss_log.txt", "w") as f:
        f.writelines([f"{l}\n" for l in loss_log])
    print("Model saved to q_scheduler_model.pt")

# -------------------- Entry --------------------
if __name__ == "__main__":
    filename = "gsf.000007.prof"
    tasks = parse_gsf_file(filename)
    run_training(tasks, episodes=50)
