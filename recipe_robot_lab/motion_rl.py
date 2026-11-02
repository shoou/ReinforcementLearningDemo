"""Masked DQN for Panda station-to-station Cartesian waypoint choices."""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
from torch import nn


X = (0.43, 0.48, 0.53)
Y = (-0.21, -0.14, -0.08, 0.0, 0.08, 0.14, 0.21)
STATION_NODES = {"rack": (0, 0, 0), "doser": (2, 2, 0),
                 "mixer": (2, 4, 0), "ph_meter": (0, 6, 0)}
BLOCKS = ((1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1),
          (1, 5, 1), (0, 3, 1), (2, 3, 1))
DELTAS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
          (0, 0, 1), (0, 0, -1))


class MotionQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(7, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(),
                                    nn.Linear(64, len(DELTAS)))

    def forward(self, state):
        return self.layers(state)


def observe(node: tuple[int, int, int], goal: tuple[int, int, int],
            blocked: tuple[int, int, int] | None = None) -> np.ndarray:
    bx, by = (-1.0, -1.0) if blocked is None else (blocked[0] / 2, blocked[1] / 6)
    return np.array([node[0] / 2, node[1] / 6, node[2],
                     goal[0] / 2, goal[1] / 6, bx, by], dtype=np.float32)


def legal(node: tuple[int, int, int], goal: tuple[int, int, int],
          blocked: tuple[int, int, int] | None = None) -> dict[int, tuple]:
    if node[2] == 0:
        lifted = (node[0], node[1], 1)
        return {4: lifted} if lifted != blocked else {}
    choices = {}
    for action, (dx, dy, dz) in enumerate(DELTAS[:4]):
        nxt = (node[0] + dx, node[1] + dy, node[2] + dz)
        if 0 <= nxt[0] < len(X) and 0 <= nxt[1] < len(Y) and nxt != blocked:
            choices[action] = nxt
    if node[:2] == goal[:2]:
        choices[5] = goal
    return choices


def distance(node: tuple, goal: tuple) -> int:
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1]) + (node[2] == 1)


def shortest_action(node: tuple, goal: tuple,
                    blocked: tuple | None = None) -> int:
    choices = legal(node, goal, blocked)
    def remaining(next_node):
        queue = deque([(next_node, 0)])
        seen = {next_node}
        while queue:
            current, steps = queue.popleft()
            if current == goal:
                return steps
            for successor in legal(current, goal, blocked).values():
                if successor not in seen:
                    seen.add(successor)
                    queue.append((successor, steps + 1))
        return 1000
    return min(choices, key=lambda action: (remaining(choices[action]), action))


def greedy(model: MotionQNet, node: tuple, goal: tuple,
           blocked: tuple | None = None) -> int:
    choices = legal(node, goal, blocked)
    with torch.no_grad():
        values = model(torch.from_numpy(observe(node, goal, blocked)).unsqueeze(0))[0].numpy()
    return max(choices, key=lambda action: float(values[action]))


def plan(model: MotionQNet, source: str, destination: str,
         max_steps: int = 24,
         blocked: tuple[int, int, int] | None = None) -> tuple[list[tuple], bool]:
    node, goal = STATION_NODES[source], STATION_NODES[destination]
    if node == goal:
        return [], False
    path = []
    seen = {node}
    for _ in range(max_steps):
        action = greedy(model, node, goal, blocked)
        node = legal(node, goal, blocked)[action]
        if node in seen and node != goal:
            break
        path.append(node)
        seen.add(node)
        if node == goal:
            return path, False
    # A shortest-path fallback prevents an undertrained policy from stalling
    # the protocol. Every fallback is logged and counted separately.
    return _shortest_path(source, destination, blocked), True


def train(episodes: int = 600, seed: int = 42) -> tuple[MotionQNet, dict]:
    random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model, target = MotionQNet(), MotionQNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    # Supervised warm start covers all safe states and goals before RL updates.
    demonstrations = []
    for goal in STATION_NODES.values():
        for x in range(len(X)):
            for y in range(len(Y)):
                for z in (0, 1):
                    node = (x, y, z)
                    if node != goal:
                        demonstrations.append((observe(node, goal), shortest_action(node, goal)))
    states = torch.tensor(np.stack([d[0] for d in demonstrations]))
    actions = torch.tensor([d[1] for d in demonstrations], dtype=torch.long)
    for _ in range(80):
        loss = nn.functional.cross_entropy(model(states), actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    target.load_state_dict(model.state_dict())

    names = tuple(STATION_NODES)
    obstacle_cases = [(a, b, block) for a in names for b in names if a != b
                      for block in BLOCKS]
    warm_success = 1 - float(np.mean([
        plan(model, a, b, blocked=block)[1] for a, b, block in obstacle_cases]))

    memory = deque(maxlen=12000)
    successes = []
    curve = []
    updates = 0
    for episode in range(episodes):
        source, destination = rng.choice(names, size=2, replace=False)
        node, goal = STATION_NODES[str(source)], STATION_NODES[str(destination)]
        blocked = BLOCKS[int(rng.integers(len(BLOCKS)))] if rng.random() < 0.8 else None
        for step in range(24):
            choices = legal(node, goal, blocked)
            epsilon = max(0.05, 0.4 * (1 - episode / max(episodes, 1)))
            action = int(rng.choice(list(choices))) if rng.random() < epsilon else greedy(model, node, goal, blocked)
            nxt = choices[action]
            done = nxt == goal or step == 23
            reward = -0.02 + 0.12 * (distance(node, goal) - distance(nxt, goal))
            if nxt == goal:
                reward += 2.0
            mask = np.zeros(len(DELTAS), dtype=bool)
            mask[list(legal(nxt, goal, blocked))] = True
            memory.append((observe(node, goal, blocked), action, reward,
                           observe(nxt, goal, blocked), mask, done))
            node = nxt
            if len(memory) >= 64:
                batch = random.sample(memory, 64)
                obs = torch.tensor(np.stack([x[0] for x in batch]))
                act = torch.tensor([x[1] for x in batch], dtype=torch.long)
                rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
                next_obs = torch.tensor(np.stack([x[3] for x in batch]))
                masks = torch.tensor(np.stack([x[4] for x in batch]))
                dones = torch.tensor([x[5] for x in batch], dtype=torch.bool)
                with torch.no_grad():
                    future = target(next_obs).masked_fill(~masks, -1e9).max(1).values
                    expected = rewards + 0.95 * future * (~dones)
                current = model(obs).gather(1, act[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(current, expected)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                updates += 1
                if updates % 100 == 0:
                    target.load_state_dict(model.state_dict())
            if done:
                successes.append(node == goal)
                break
        if (episode + 1) % 50 == 0 or episode + 1 == episodes:
            curve.append({"episode": episode + 1,
                          "success_rate": round(float(np.mean(successes[-50:])), 3)})
    independent = [plan(model, a, b)[1] for a in names for b in names if a != b]
    obstacle_fallbacks = [plan(model, a, b, blocked=block)[1]
                          for a, b, block in obstacle_cases]
    route_lengths = [len(plan(model, a, b)[0]) for a in names for b in names if a != b]
    shortest_lengths = [len(_shortest_path(a, b)) for a in names for b in names if a != b]
    obstacle_lengths = [len(plan(model, a, b, blocked=block)[0])
                        for a, b, block in obstacle_cases]
    obstacle_shortest = [len(_shortest_path(a, b, block))
                         for a, b, block in obstacle_cases]
    example_route, example_fallback = plan(model, "doser", "ph_meter",
                                           blocked=BLOCKS[2])
    return model, {"episodes": episodes,
                   "training_success_last_100": round(float(np.mean(successes[-100:])), 3),
                   "station_pair_policy_success": round(1 - float(np.mean(independent)), 3),
                   "station_pair_count": len(independent),
                   "mean_route_steps": round(float(np.mean(route_lengths)), 2),
                   "mean_shortest_steps": round(float(np.mean(shortest_lengths)), 2),
                   "obstacle_cases": len(obstacle_cases),
                   "warm_start_direct_success": round(warm_success, 3),
                   "after_rl_direct_success": round(1 - float(np.mean(obstacle_fallbacks)), 3),
                   "obstacle_mean_route_steps": round(float(np.mean(obstacle_lengths)), 2),
                   "obstacle_mean_shortest_steps": round(float(np.mean(obstacle_shortest)), 2),
                   "example_obstacle_route": {
                       "source": "doser", "destination": "ph_meter",
                       "blocked_node": list(BLOCKS[2]),
                       "route": [list(node) for node in example_route],
                       "fallback": example_fallback},
                   "curve": curve}


def _shortest_path(source: str, destination: str,
                   blocked: tuple | None = None) -> list[tuple]:
    node, goal = STATION_NODES[source], STATION_NODES[destination]
    path = []
    while node != goal:
        node = legal(node, goal, blocked)[shortest_action(node, goal, blocked)]
        path.append(node)
    return path


def waypoint(node: tuple, destination: str) -> tuple[float, float, float]:
    if node[2] == 1:
        return X[node[0]], Y[node[1]], 0.34
    from arm_sim import STATIONS
    return STATIONS[destination]
