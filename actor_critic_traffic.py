"""Actor-Critic 教学案例：让智能信号灯应对不断变化的城市车流。"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


HORIZON = 60  # 一个回合模拟 60 分钟，每一步为 1 分钟
CAPACITY = 5  # 绿灯一分钟最多放行的车辆数
PHASES = ("南北", "东西")


class TrafficLight:
    """单路口模拟器；到达量由随机种子决定，便于公平比较策略。"""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.arrivals = np.zeros((HORIZON, 2), dtype=np.int64)
        for t in range(HORIZON):
            # 前半小时东西向有活动散场车流，后半小时南北向通勤车流增加。
            rates = (0.8, 3.0) if t < 30 else (3.0, 0.8)
            self.arrivals[t] = rng.poisson(rates)
        self.reset()

    def reset(self) -> np.ndarray:
        self.queues = np.zeros(2, dtype=np.int64)
        self.phase = 0
        self.t = 0
        self.total_wait = 0
        self.total_passed = 0
        self.switches = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        angle = 2 * math.pi * self.t / HORIZON
        return np.array([
            min(self.queues[0], 40) / 20,
            min(self.queues[1], 40) / 20,
            float(self.phase == 0),
            float(self.phase == 1),
            math.sin(angle), math.cos(angle),
        ], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if action not in (0, 1) or self.t >= HORIZON:
            raise ValueError("动作必须是 0/1，且回合尚未结束")
        switched = action != self.phase
        self.phase = action
        self.switches += int(switched)
        self.queues += self.arrivals[self.t]
        # 切换方向需要黄灯和全红过渡，本分钟只能放行一半车辆。
        capacity = CAPACITY // 2 if switched else CAPACITY
        passed = min(int(self.queues[action]), capacity)
        self.queues[action] -= passed
        self.total_passed += passed
        waiting = int(self.queues.sum())
        self.total_wait += waiting
        # 排队车辆数对应本分钟的车辆等待分钟；额外惩罚频繁切换。
        reward = (3 * passed - waiting - 2 * int(switched)) / 10.0
        self.t += 1
        return self.observe(), reward, self.t == HORIZON, {
            "arrivals": self.arrivals[self.t - 1].copy(),
            "queues": self.queues.copy(),
            "passed": passed,
            "switched": switched,
            "waiting": waiting,
        }


class ActorCritic(torch.nn.Module):
    """独立特征层；Actor 输出动作概率，Critic 输出状态价值。"""

    def __init__(self):
        super().__init__()
        self.actor_body = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
        )
        self.critic_body = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
        )
        self.actor = torch.nn.Linear(64, 2)
        self.critic = torch.nn.Linear(64, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(self.actor_body(states)), self.critic(self.critic_body(states)).squeeze(-1)


def rollout(model: ActorCritic, seed: int, explore: bool) -> tuple[list, dict]:
    env = TrafficLight(seed)
    states, actions, rewards, records = [], [], [], []
    state = env.reset()
    for t in range(HORIZON):
        if explore:
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(state).unsqueeze(0))
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
        else:
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(state).unsqueeze(0))
                action = int(logits.argmax(dim=-1).item())
        states.append(state)
        actions.append(action)
        state, reward, _, info = env.step(action)
        rewards.append(reward)
        records.append({"t": t + 1, "action": action, **info})
    return [states, actions, rewards], {
        "records": records,
        "wait": env.total_wait,
        "passed": env.total_passed,
        "switches": env.switches,
        "reward": float(sum(rewards)),
    }


def fixed_timing(seed: int) -> dict:
    """对照组：每 5 分钟切换一次放行方向。"""
    env = TrafficLight(seed)
    records = []
    for t in range(HORIZON):
        action = (t // 5) % 2
        _, _, _, info = env.step(action)
        records.append({"t": t + 1, "action": action, **info})
    return {"records": records, "wait": env.total_wait,
            "passed": env.total_passed, "switches": env.switches}


def train(episodes: int, seed: int) -> tuple[ActorCritic, list[float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    history = []
    gamma, gae_lambda = 0.98, 0.95
    for episode in range(episodes):
        (states, actions, rewards), result = rollout(model, seed + episode, explore=True)
        state_tensor = torch.from_numpy(np.stack(states))
        logits, values = model(state_tensor)
        # 单回合结束后的价值是 0；用 TD 误差反向累积 GAE 优势。
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_values = torch.cat((values[1:].detach(), values.new_zeros(1)))
        deltas = rewards_tensor + gamma * next_values - values.detach()
        advantages = torch.empty_like(deltas)
        gae = torch.zeros((), dtype=torch.float32)
        for t in reversed(range(HORIZON)):
            gae = deltas[t] + gamma * gae_lambda * gae
            advantages[t] = gae
        targets = advantages + values.detach()
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probs = distribution.log_prob(torch.tensor(actions))
        actor_loss = -(log_probs * normalized.detach()).mean()
        critic_loss = torch.nn.functional.smooth_l1_loss(values, targets)
        loss = actor_loss + 0.5 * critic_loss - 0.05 * distribution.entropy().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(result["wait"])
        if (episode + 1) % 100 == 0 or episode + 1 == episodes:
            recent = np.mean(history[-min(100, len(history)):])
            print(f"回合 {episode + 1:4d}/{episodes} | 最近平均等待 {recent:.0f} 车辆·分钟")
    return model, history


def compare(model: ActorCritic, seeds: list[int]) -> dict[str, float]:
    learned = [rollout(model, seed, explore=False)[1] for seed in seeds]
    fixed = [fixed_timing(seed) for seed in seeds]
    return {
        "learned_wait": float(np.mean([x["wait"] for x in learned])),
        "fixed_wait": float(np.mean([x["wait"] for x in fixed])),
        "learned_passed": float(np.mean([x["passed"] for x in learned])),
        "fixed_passed": float(np.mean([x["passed"] for x in fixed])),
        "learned_switches": float(np.mean([x["switches"] for x in learned])),
        "fixed_switches": float(np.mean([x["switches"] for x in fixed])),
    }


def save_chart(history: list[float], learned: dict, fixed: dict, stats: dict, output: Path) -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    axes[0].plot(history, color="#93c5fd", alpha=0.55, linewidth=1)
    width = min(40, len(history))
    smooth = np.convolve(history, np.ones(width) / width, mode="valid")
    axes[0].plot(np.arange(width - 1, len(history)), smooth, color="#1d4ed8", linewidth=2)
    axes[0].set(title="训练：每小时累计等待", xlabel="训练回合", ylabel="车辆·分钟")
    axes[1].bar(["固定配时", "Actor-Critic"], [stats["fixed_wait"], stats["learned_wait"]],
                color=["#f59e0b", "#2563eb"])
    axes[1].set(title="100 个相同车流场景的平均等待", ylabel="车辆·分钟")
    for label, result, color in [("固定配时", fixed, "#f59e0b"), ("Actor-Critic", learned, "#2563eb")]:
        records = result["records"]
        queues = np.array([r["queues"] for r in records])
        axes[2].plot(np.arange(1, HORIZON + 1), queues.sum(axis=1), label=label, color=color)
    axes[2].axvline(30, color="#64748b", linestyle="--", linewidth=1)
    axes[2].set(title="同一场景：路口排队车辆", xlabel="模拟分钟", ylabel="车辆数")
    axes[2].legend()
    fig.savefig(output / "training_and_comparison.png", dpi=160)
    plt.close(fig)


def save_animation(result: dict, output: Path) -> None:
    """把测试车流中的相位切换和排队变化做成轻量 GIF。"""
    frames = []
    font = ImageFont.load_default()
    for record in result["records"][::2]:
        image = Image.new("RGB", (640, 340), "#f8fafc")
        draw = ImageDraw.Draw(image)
        draw.text((25, 20), f"SMART INTERSECTION  |  minute {record['t']:02d}/60", fill="#0f172a", font=font)
        draw.text((25, 52), "Green: " + ("NORTH-SOUTH" if record["action"] == 0 else "EAST-WEST"),
                  fill="#15803d", font=font)
        for i, label in enumerate(("NORTH-SOUTH", "EAST-WEST")):
            y = 105 + i * 95
            count = int(record["queues"][i])
            draw.text((25, y), f"{label:14s}  {count:2d} cars waiting", fill="#334155", font=font)
            draw.rounded_rectangle((25, y + 28, 600, y + 56), radius=7, fill="#e2e8f0")
            draw.rounded_rectangle((25, y + 28, 25 + min(count, 40) / 40 * 575, y + 56),
                                   radius=7, fill="#f59e0b" if i != record["action"] else "#22c55e")
        draw.text((25, 306), f"Passed this minute: {record['passed']}  |  Switched: {record['switched']}",
                  fill="#475569", font=font)
        frames.append(image)
    frames[0].save(output / "intersection.gif", save_all=True, append_images=frames[1:],
                   duration=180, loop=0, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("actor_critic_output"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes 必须至少为 1")
    args.output.mkdir(parents=True, exist_ok=True)
    model, history = train(args.episodes, args.seed)
    torch.save(model.state_dict(), args.output / "actor_critic_model.pt")
    test_seeds = list(range(100_000, 100_100))  # 与训练随机种子分开
    stats = compare(model, test_seeds)
    learned = rollout(model, test_seeds[0], explore=False)[1]
    fixed = fixed_timing(test_seeds[0])
    save_chart(history, learned, fixed, stats, args.output)
    save_animation(learned, args.output)
    reduction = 100 * (stats["fixed_wait"] - stats["learned_wait"]) / stats["fixed_wait"]
    print(f"测试集平均等待：固定配时 {stats['fixed_wait']:.1f}，Actor-Critic {stats['learned_wait']:.1f} 车辆·分钟")
    print(f"累计等待降幅：{reduction:.1f}%（正值表示改善）")
    print(f"结果已保存到 {args.output.resolve()}")


if __name__ == "__main__":
    main()
