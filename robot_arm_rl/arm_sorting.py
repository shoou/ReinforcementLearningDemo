"""Tabular Q-learning for a guarded, indexed pick-and-place robot arm.

Standard-library only. Run this file to train and generate a standalone HTML replay.
All dimensions and cycle times are synthetic teaching assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from pathlib import Path


L1, L2 = 0.27, 0.23  # metres
A1 = tuple(range(25, 156, 10))
A2 = tuple(range(-130, -9, 10))
HOME = (7, 10)
PICKS = ((4, 5), (4, 6))  # indexed conveyor stops
BINS = ((10, 10), (11, 10))
OBSTACLE = (0.13, 0.18, 0.285, 0.335)  # fixed machine fixture
MAX_STEPS = 90
DT = 0.25  # seconds per joint command
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def joints(q):
    t1, t2 = math.radians(A1[q[0]]), math.radians(A2[q[1]])
    elbow = (L1 * math.cos(t1), L1 * math.sin(t1))
    tip = (elbow[0] + L2 * math.cos(t1 + t2),
           elbow[1] + L2 * math.sin(t1 + t2))
    return elbow, tip


def fixture_at_angles(theta1, theta2):
    """Sample both links against the fixture with 10 mm clearance."""
    t1, t2 = math.radians(theta1), math.radians(theta2)
    elbow = (L1 * math.cos(t1), L1 * math.sin(t1))
    tip = (elbow[0] + L2 * math.cos(t1 + t2),
           elbow[1] + L2 * math.sin(t1 + t2))
    x0, x1, y0, y1 = OBSTACLE
    x0 -= 0.01
    x1 += 0.01
    y0 -= 0.01
    y1 += 0.01
    for start, end in (((0.0, 0.0), elbow), (elbow, tip)):
        for k in range(41):
            f = k / 40
            x = start[0] + f * (end[0] - start[0])
            y = start[1] + f * (end[1] - start[1])
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
    return False


def near_fixture(q):
    return fixture_at_angles(A1[q[0]], A2[q[1]])


SAFE = {(i, j) for i in range(len(A1)) for j in range(len(A2))
        if not near_fixture((i, j))}


def motion_safe(a, b):
    """Check intermediate poses within a 10° command, not only endpoints."""
    if b not in SAFE:
        return False
    for k in range(1, 10):
        f = k / 10
        t1 = A1[a[0]] * (1 - f) + A1[b[0]] * f
        t2 = A2[a[1]] * (1 - f) + A2[b[1]] * f
        if fixture_at_angles(t1, t2):
            return False
    return True


MOTIONS = {}
for _q in SAFE:
    MOTIONS[_q] = []
    for _action, (_di, _dj) in enumerate(ACTIONS):
        _new = (_q[0] + _di, _q[1] + _dj)
        if motion_safe(_q, _new):
            MOTIONS[_q].append((_action, _new))


def neighbors(q):
    return iter(MOTIONS[q])


def distance(q, goal):
    a, b = joints(q)[1], joints(goal)[1]
    return math.hypot(a[0] - b[0], a[1] - b[1])


class ArmEnv:
    """One indexed parcel moves from a conveyor stop to its assigned bin."""

    def __init__(self, pickup, bin_id):
        self.pickup = pickup
        self.bin_id = bin_id
        self.reset()

    def reset(self):
        self.q = HOME
        self.phase = 0  # 0: approach and grasp; 1: carry and release
        self.steps = 0
        self.done = False
        self.success = False
        self.interlocks = 0
        return self.state()

    def state(self):
        return (self.pickup, self.bin_id, self.phase, *self.q)

    def goal(self):
        return PICKS[self.pickup] if self.phase == 0 else BINS[self.bin_id]

    def step(self, action):
        if self.done:
            raise RuntimeError("episode ended")
        old_dist = distance(self.q, self.goal())
        options = dict(neighbors(self.q))
        valid = action in options
        if valid:
            self.q = options[action]
        else:
            self.interlocks += 1
        self.steps += 1
        reward = -0.025 + (8.0 * (old_dist - distance(self.q, self.goal())) if valid else -0.18)
        event = "move" if valid else "interlock"
        # Grasp/release is a separate tool command, issued automatically only
        # when the end effector is at the indexed station and nearly stationary.
        if self.q == self.goal():
            if self.phase == 0:
                self.phase = 1
                reward += 2.0
                event = "grasp"
            else:
                self.done = self.success = True
                reward += 4.0
                event = "release"
        if self.steps >= MAX_STEPS and not self.done:
            self.done = True
            reward -= 2.0
            event = "timeout"
        return self.state(), reward, self.done, event


def shortest_length(start, goal):
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        q, n = queue.popleft()
        if q == goal:
            return n
        for _, nxt in neighbors(q):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, n + 1))
    return None


def choose(qtable, state, rng, epsilon):
    if rng.random() < epsilon:
        return rng.randrange(4)
    values = qtable.get(state, [0.0] * 4)
    best = max(values)
    candidates = [a for a, v in enumerate(values) if v >= best - 1e-10]
    return rng.choice(candidates)


def train(episodes, seed):
    rng = random.Random(seed)
    table = {}
    history = []
    for episode in range(episodes):
        env = ArmEnv(rng.randrange(2), rng.randrange(2))
        state = env.reset()
        epsilon = max(0.025, 0.55 * (1 - episode / max(1, episodes)) ** 1.5)
        for _ in range(MAX_STEPS):
            action = choose(table, state, rng, epsilon)
            next_state, reward, done, _ = env.step(action)
            row = table.setdefault(state, [0.0] * 4)
            future = 0 if done else max(table.get(next_state, [0.0] * 4))
            row[action] += 0.25 * (reward + 0.98 * future - row[action])
            state = next_state
            if done:
                break
        history.append(int(env.success))
        if (episode + 1) % max(1, episodes // 20) == 0:
            print(f"episode {episode + 1}/{episodes}: last-100 success {sum(history[-100:]) / len(history[-100:]):.0%}")
    return table, history


def replay(table, pickup, bin_id, seed):
    env = ArmEnv(pickup, bin_id)
    rng = random.Random(seed)
    frames = []
    def record(event):
        elbow, tip = joints(env.q)
        frames.append({"q": env.q, "elbow": elbow, "tip": tip,
                       "phase": env.phase, "event": event, "step": env.steps})
    record("start")
    for _ in range(MAX_STEPS):
        action = choose(table, env.state(), rng, 0)
        _, _, done, event = env.step(action)
        record(event)
        if done:
            break
    return env, frames


def render_html(payload, path):
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    path.write_text(template.replace("/*DATA*/", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    for target in (*PICKS, *BINS, HOME):
        if target not in SAFE:
            raise RuntimeError(f"station or home in fixture: {target}")
    for pickup in PICKS:
        for bin_q in BINS:
            if shortest_length(HOME, pickup) is None or shortest_length(pickup, bin_q) is None:
                raise RuntimeError("a station is unreachable")
    table, history = train(args.episodes, args.seed)
    cases = []
    for p in range(2):
        for b in range(2):
            env, frames = replay(table, p, b, args.seed + p * 2 + b)
            ideal = shortest_length(HOME, PICKS[p]) + shortest_length(PICKS[p], BINS[b])
            cases.append({"pickup": p, "bin": b, "success": env.success,
                          "steps": env.steps, "ideal_steps": ideal,
                          "interlocks": env.interlocks, "frames": frames})
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"cases": cases, "history": history, "episodes": args.episodes,
               "seed": args.seed, "picks": [joints(q)[1] for q in PICKS],
               "bins": [joints(q)[1] for q in BINS], "fixture": OBSTACLE,
               "dt": DT, "safe_count": len(SAFE)}
    render_html(payload, args.output / "demo.html")
    summary = {"episodes": args.episodes, "seed": args.seed,
               "last_100_train_success": sum(history[-100:]) / len(history[-100:]),
               "cases": [{k: v for k, v in c.items() if k != "frames"} for c in cases]}
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output / 'demo.html'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
