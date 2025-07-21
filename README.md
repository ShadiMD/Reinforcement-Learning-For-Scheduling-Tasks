Reinforcement Learning for Task Scheduling
Project Overview
This project explores Task Scheduling in Multi-Processor Environments using two approaches:

Greedy Scheduling – a simple heuristic baseline.

Reinforcement Learning (RL) – a Deep Q-Learning model that tries to learn a better scheduling policy.

The goal is to minimize the total execution time (Makespan) of a set of tasks with:

Precedence constraints (tasks depend on others finishing first).

Processor-type constraints (each task can only run on specific types of processors).

Task scheduling is a NP-Hard problem and appears widely in:

Embedded systems (e.g., automotive or chip design)

Data centers and high-performance computing clusters

Industrial workflows

Data Understanding
The dataset used is a GSF profile file (example: gsf.000007.prof), containing:

Task name

Processor type & ID

Start time

End time

Dependencies (list of other tasks that must finish first)

We also build:

Dependency Graph → to know which tasks can start now.

Processor Map → to know how many processors of each type are available.

Greedy Scheduling Baseline
The Greedy Scheduler works as follows:

Parse all tasks and build the ready queue (tasks with no remaining dependencies).

At each time step:

Pick the next ready task (FIFO style).

Assign it to the first available processor of the correct type.

If no processor available, put it back in the queue.

Advance time by the shortest running task and repeat.

This gives us a valid schedule, but not necessarily optimal.

We can also calculate lower bounds for comparison:

LB1 – workload divided by processor capacity.

LB2 – length of the critical path in the dependency graph.

Observed issue:
Greedy scheduling doesn’t prioritize important tasks:

It may delay “critical” tasks that block many others.

It may waste resources by scheduling “easy” tasks that don’t unlock anything.

Result → the makespan is significantly larger than the critical path lower bound.

Reinforcement Learning Scheduler
To improve on greedy, we tried Deep Q-Learning (DQN):

State Encoding

Current ready tasks (features: processing time, dependencies, criticality)

Processor status (free/busy)

Global info (time elapsed, number of completed tasks)

Actions

Selecting one task from the ready queue to schedule next.

Reward Function

Negative processing time per step → encourages faster makespan.

Small penalty for leaving processors idle.

Learning Loop

The agent schedules tasks → observes resulting state → stores (state, action, reward, next_state) in replay buffer.

Periodically trains the Q-network to predict better task selection policies.

What Happened with RL?
After implementing and running the RL agent:

It did learn some patterns, like preferring tasks with many dependents.

But the overall makespan improvement over greedy was small.
