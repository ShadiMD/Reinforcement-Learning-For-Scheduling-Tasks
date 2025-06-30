


import re
import ast
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, defaultdict
from tqdm import trange

# ------------------ Data Parsing ------------------
def parse_gsf_file(filename):
    tasks = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                match = re.match(
                    r"^(.*?)\s+\('(.+?)',\s*(\d+)\)\s+([0-9.]+)\s+([0-9.]+)\s+(.*)$", line)
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

# ------------------ Q-network ------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long).view(-1),
            torch.tensor(rewards, dtype=torch.float32).view(-1),
            torch.stack(next_states)
        )

    def __len__(self):
        return len(self.buffer)

# ------------------ State encoding ------------------
def encode_state(global_data, ready_tasks, processors, dependencies, max_ready_tasks=10):
    state = []
    types = ['MIPS', 'VMP', 'PMA', 'PMAC', 'MPC']

    state.append(global_data["current_time"] / global_data["max_time"])
    state.append(global_data["remaining_tasks"] / global_data["total_tasks"])
    state.append(global_data["completed_tasks"] / global_data["total_tasks"])
    state.append(len(ready_tasks) / global_data["total_tasks"])

    for t in types:
        count = sum(1 for p in processors.values() if p['type'] == t and not p['busy'])
        state.append(count / global_data["processor_counts"][t])

    for t in types:
        load = sum(1 for p in processors.values() if p['type'] == t and p['busy'])
        state.append(load / global_data["processor_counts"][t])

    for i in range(max_ready_tasks):
        if i < len(ready_tasks):
            task = ready_tasks[i]
            one_hot = [1 if t == task['processor_type'] else 0 for t in types]
            time = task['processing_time'] / global_data["max_task_time"]
            num_deps = len(dependencies[task["task_name"]]) / global_data["max_deps"]
            state.extend([time] + one_hot + [num_deps])
        else:
            state.extend([0.0] + [0]*len(types) + [0.0])
    return np.array(state, dtype=np.float32)

def select_task_epsilon_greedy(state, q_network, epsilon, ready_tasks_count):
    if random.random() < epsilon:
        return random.randint(0, ready_tasks_count - 1)
    with torch.no_grad():
        q_values = q_network(state)
        return torch.argmax(q_values[0][:ready_tasks_count]).item()

# ------------------ Training step ------------------
def train_step(q_net, target_net, optimizer, batch, gamma=0.99):
    states, actions, rewards, next_states = batch
    q_values = q_net(states).squeeze(1)
    actions = actions.view(-1).long()
    action_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
    with torch.no_grad():
        next_q = target_net(next_states).squeeze(1).max(1)[0]
        target = rewards + gamma * next_q
    loss = F.mse_loss(action_q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ------------------ Task Scheduling Environment ------------------
class TaskSchedulingEnv:
    def __init__(self, raw_tasks, processor_template, max_ready_tasks=10):
        self.original_tasks = copy.deepcopy(raw_tasks)
        self.processor_template = processor_template
        self.max_ready_tasks = max_ready_tasks
        self.types = ['MIPS', 'VMP', 'PMA', 'PMAC', 'MPC']
        self.processor_counts = {t: sum(1 for p in processor_template.values() if p['type'] == t) for t in self.types}
        self.max_task_time = max(t["processing_time"] for t in self.original_tasks)
        self.max_deps = max(len(t["dependencies"]) for t in self.original_tasks)
        self.reset()

    def reset(self):
        random.shuffle(self.original_tasks)  
        self.tasks = copy.deepcopy(self.original_tasks)
        self.total_tasks = len(self.tasks)
        self.time = 0.0
        self.done = False
        self.dependencies = build_dependency_map(self.tasks)
        self.Tasks_Time = {t["task_name"]: t["processing_time"] for t in self.tasks}
        self.Task_Processor_Type = {t["task_name"]: t["processor_type"] for t in self.tasks}
        self.ready = []
        self.in_progress = []
        self.finished = set()
        self.task_start_time = {}
        self.task_end_time = {}
        self.total_time = 0.0
        self.processors = copy.deepcopy(self.processor_template)
        for t in self.tasks:
            if not t["dependencies"]:
                self.ready.append(t)
        return self.encode_state()

    def step(self, action_index):
        if self.done or len(self.ready) == 0:
            return self.encode_state(), 0, self.done
        selected_task = self.ready[action_index]
        task_name = selected_task["task_name"]
        task_type = selected_task["processor_type"]
        task_time = selected_task["processing_time"]
        proc_assigned = None
        for proc_id, proc in self.processors.items():
            if not proc["busy"] and proc["type"] == task_type:
                self.processors[proc_id]["busy"] = True
                self.in_progress.append([proc_id, task_name, task_time])
                self.task_start_time[task_name] = self.time
                proc_assigned = proc_id
                break
        if not proc_assigned:
            return self.encode_state(), -1, self.done
        min_time = min(item[2] for item in self.in_progress)
        self.time += min_time
        for item in self.in_progress:
            item[2] -= min_time
        finished_now = [it for it in self.in_progress if it[2] == 0]
        for proc_id, t_id, _ in finished_now:
            self.finished.add(t_id)
            self.task_end_time[t_id] = self.time
            self.processors[proc_id]["busy"] = False
        self.in_progress = [it for it in self.in_progress if it[2] > 0]
        self.ready = []
        for task in self.tasks:
            tname = task["task_name"]
            if tname not in self.finished and tname not in [it[1] for it in self.in_progress]:
                deps = self.dependencies[tname]
                if all(dep in self.finished for dep in deps):
                    if task not in self.ready:
                        self.ready.append(task)
        if len(self.finished) == self.total_tasks:
            self.done = True
        reward = max(-100.0, -min_time)
        return self.encode_state(), reward, self.done

    def encode_state(self):
        global_info = {
            "current_time": self.time,
            "remaining_tasks": self.total_tasks - len(self.finished),
            "completed_tasks": len(self.finished),
            "total_tasks": self.total_tasks,
            "processor_counts": self.processor_counts,
            "max_time": 1000,
            "max_task_time": self.max_task_time,
            "max_deps": self.max_deps
        }
        return torch.tensor(encode_state(global_info, self.ready, self.processors, self.dependencies, self.max_ready_tasks)).unsqueeze(0)

# ------------------ Main Training Loop ------------------
if __name__ == "__main__":
    filename = "gsf.000007.prof"
    tasks = parse_gsf_file(filename)
    processors = {f"P{i}": {"busy": False, "type": t}
        for i, t in enumerate(
            ['MIPS']*5 + ['VMP']*4 + ['PMA']*2 + ['PMAC']*2 + ['MPC']*5, 1)}

    MAX_EPISODES = 400
    MAX_READY_TASKS = 10
    GAMMA = 0.99
    EPSILON_START = 0.9
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.995
    BATCH_SIZE = 64
    LR = 1e-3
    TARGET_UPDATE_INTERVAL = 10

    env = TaskSchedulingEnv(tasks, processors, MAX_READY_TASKS)
    input_dim = len(env.encode_state()[0])
    output_dim = MAX_READY_TASKS

    q_net = QNetwork(input_dim, output_dim)
    target_net = QNetwork(input_dim, output_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(10000)
    epsilon = EPSILON_START

    for episode in trange(MAX_EPISODES):
        state = env.reset()
        total_reward = 0
        while not env.done:
            valid_actions = min(len(env.ready), MAX_READY_TASKS)
            if valid_actions == 0:
                break
            action = select_task_epsilon_greedy(state, q_net, epsilon, valid_actions)
            next_state, reward, done = env.step(action)
            replay_buffer.add(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                train_step(q_net, target_net, optimizer, batch, gamma=GAMMA)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if episode % TARGET_UPDATE_INTERVAL == 0:
            target_net.load_state_dict(q_net.state_dict())
        print(f"[Episode {episode}] Final Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f} | Steps: {len(env.finished)}")

    torch.save(q_net.state_dict(), "q_scheduler_model.pt")
    print("Model saved to q_scheduler_model.pt")
