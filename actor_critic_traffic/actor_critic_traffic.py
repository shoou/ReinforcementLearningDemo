"""Actor-Critic 教学案例：让智能信号灯应对不断变化的城市车流。

运行流程：生成车流 -> 与环境交互 -> 用 Actor-Critic 更新策略 -> 与固定配时比较。
一个回合对应一个路口的 60 分钟；这里的车辆数和时间均为简化模拟值。
"""

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
PHASES = ("南北", "东西")  # 动作 0、1 对应的放行方向


class TrafficLight:
    """单路口模拟器；到达量由随机种子决定，便于公平比较策略。"""

    def __init__(self, seed: int):
        # 先生成整回合的到达车辆；相同种子对应相同车流，方便比较策略。
        rng = np.random.default_rng(seed)
        self.arrivals = np.zeros((HORIZON, 2), dtype=np.int64)
        # arrivals[t, 0/1] 分别是第 t 分钟南北/东西方向的新到车辆数。
        for t in range(HORIZON):
            # 前半小时东西向有活动散场车流，后半小时南北向通勤车流增加。
            # 泊松分布按给定平均到达率生成非负整数，每分钟独立采样。
            rates = (0.8, 3.0) if t < 30 else (3.0, 0.8)
            self.arrivals[t] = rng.poisson(rates)
        self.reset()

    def reset(self) -> np.ndarray:
        """清空累计指标，并返回回合开始时的观测。"""
        self.queues = np.zeros(2, dtype=np.int64)
        self.phase = 0  # 初始绿灯为南北向；第一次选东西向也计一次切换。
        self.t = 0
        self.total_wait = 0  # 每分钟末的排队数之和，单位为车辆·分钟。
        self.total_passed = 0  # 整回合已通过路口的车辆数。
        self.switches = 0  # 绿灯方向改变的次数。
        return self.observe()

    def observe(self) -> np.ndarray:
        """构造 6 维状态：两向队列、当前相位和周期时间。"""
        # 将分钟数映射到完整圆周，使网络能区分车流变化前后的时段。
        angle = 2 * math.pi * self.t / HORIZON
        return np.array([
            # 队列截断到 40 辆后缩放，避免过大的数值主导网络输入。
            min(self.queues[0], 40) / 20,
            min(self.queues[1], 40) / 20,
            # 用两个独热位表示当前绿灯方向。
            float(self.phase == 0),
            float(self.phase == 1),
            # 正弦和余弦让回合首尾的时间编码连续。
            math.sin(angle), math.cos(angle),
        ], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """执行一分钟，返回 (下一状态, 奖励, 是否结束, 本分钟明细)。

        action=0 放行南北向，action=1 放行东西向；两向不能同时放行。
        """
        if action not in (0, 1) or self.t >= HORIZON:
            raise ValueError("动作必须是 0/1，且回合尚未结束")
        # 先决定信号灯，再加入本分钟到达的车辆，最后从绿灯方向放行。
        switched = action != self.phase
        self.phase = action
        self.switches += int(switched)
        self.queues += self.arrivals[self.t]
        # 切换方向需要黄灯和全红过渡，本分钟只能放行一半车辆。
        capacity = CAPACITY // 2 if switched else CAPACITY
        # 实际放行量既不能超过绿灯容量，也不能超过该方向现有车辆数。
        passed = min(int(self.queues[action]), capacity)
        self.queues[action] -= passed
        self.total_passed += passed
        # 本分钟末仍在两向队列中的车辆，各贡献一分钟等待时间。
        waiting = int(self.queues.sum())
        self.total_wait += waiting
        # 排队车辆数对应本分钟的车辆等待分钟；额外惩罚频繁切换。
        # 放行车辆给正奖励，使策略同时考虑通行量和排队时间。
        # 单步奖励 = 通行收益 - 当前排队代价 - 切灯代价；除以 10 缩小数值范围。
        reward = (3 * passed - waiting - 2 * int(switched)) / 10.0
        self.t += 1
        # copy() 固定本分钟的快照，避免后续 step 修改已保存的记录。
        return self.observe(), reward, self.t == HORIZON, {
            "arrivals": self.arrivals[self.t - 1].copy(),
            "queues": self.queues.copy(),
            "passed": passed,
            "switched": switched,
            "waiting": waiting,
        }


class ActorCritic(torch.nn.Module):
    """
    Actor 和 Critic 各有一套独立的特征层，输入都是 6 维观测。

    Actor 输出两个动作的 logits，经 Categorical 转成动作概率；
    Critic 输出 V(s)，估计从当前状态开始的折扣累计奖励。
    """

    def __init__(self):
        super().__init__()
        # 两个网络都使用 6 -> 64 -> 64 的 Tanh 多层感知机。
        self.actor_body = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
        )
        self.critic_body = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
        )
        self.actor = torch.nn.Linear(64, 2)  # 两个未归一化的动作分数。
        self.critic = torch.nn.Linear(64, 1)  # 每个状态对应一个标量价值。

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """批量输入形状为 [N, 6]；输出形状分别为 [N, 2] 和 [N]。"""
        return self.actor(self.actor_body(states)), self.critic(self.critic_body(states)).squeeze(-1)


def rollout(model: ActorCritic, seed: int, explore: bool) -> tuple[list, dict]:
    """运行一个回合，返回训练轨迹及用于统计和可视化的结果。

    explore=True 按策略概率采样，允许尝试其他动作；False 选最大 logit。
    """
    env = TrafficLight(seed)
    states, actions, rewards, records = [], [], [], []
    state = env.reset()
    for t in range(HORIZON):
        # 采样过程只收集轨迹；训练时会重新前向计算以建立梯度图。
        # unsqueeze(0) 将单条 6 维观测变成网络需要的 [1, 6] 批量输入。
        if explore:
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(state).unsqueeze(0))
                # Categorical 接受 logits，内部会按 softmax 得到采样概率。
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
        else:
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(state).unsqueeze(0))
                action = int(logits.argmax(dim=-1).item())
        # 保存的是行动前的状态及其动作，奖励则由该动作的 step 返回。
        states.append(state)
        actions.append(action)
        state, reward, _, info = env.step(action)
        rewards.append(reward)
        # t 从 0 开始计数，展示给用户的分钟数从 1 开始。
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
        # 第 0～4 分钟放行南北向，第 5～9 分钟放行东西向，依此交替。
        action = (t // 5) % 2
        _, _, _, info = env.step(action)
        records.append({"t": t + 1, "action": action, **info})
    return {"records": records, "wait": env.total_wait,
            "passed": env.total_passed, "switches": env.switches}


def train(episodes: int, seed: int) -> tuple[ActorCritic, list[float]]:
    """每采样一个完整回合就更新一次网络，返回模型和等待时间历史。"""
    # 固定三个随机数生成器的种子，使初始化和探索采样尽量可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    history = []  # 每个训练回合的累计等待，供学习曲线使用。
    # gamma 是未来奖励折扣；gae_lambda 控制 GAE 对较远 TD 误差的权重。
    gamma, gae_lambda = 0.98, 0.95
    for episode in range(episodes):
        # 各回合使用不同车流；采样动作带有随机性，供策略学习。
        (states, actions, rewards), result = rollout(model, seed + episode, explore=True)
        # [60, 6]：每一行是做出当分钟动作之前的状态。
        state_tensor = torch.from_numpy(np.stack(states))
        # logits: [60, 2]；values: [60]。这次前向计算参与梯度更新。
        logits, values = model(state_tensor)
        # 回合终点没有后续收益，因此末状态的下一步价值设为 0。
        # TD 误差 δ_t = r_t + gamma * V(s_{t+1}) - V(s_t)。
        # 构造优势与目标时 detach，避免梯度穿过这些用于监督的量。
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_values = torch.cat((values[1:].detach(), values.new_zeros(1)))
        deltas = rewards_tensor + gamma * next_values - values.detach()
        advantages = torch.empty_like(deltas)
        gae = torch.zeros((), dtype=torch.float32)
        # GAE 递推：A_t = δ_t + gamma * lambda * A_{t+1}；末尾 A_{60}=0。
        # 从后往前计算，使每个时刻都包含其后续的折扣 TD 误差。
        for t in reversed(range(HORIZON)):
            gae = deltas[t] + gamma * gae_lambda * gae
            advantages[t] = gae
        # V(s_t) + A_t 作为 Critic 的拟合目标；目标本身不参与反向传播。
        targets = advantages + values.detach()
        # 标准化优势，降低不同回合之间策略梯度的尺度波动。
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        distribution = torch.distributions.Categorical(logits=logits)
        # 取轨迹中实际采样动作的对数概率；提高正优势动作的概率，
        # 降低负优势动作的概率。detach 防止优势目标反向影响 Critic。
        log_probs = distribution.log_prob(torch.tensor(actions))
        actor_loss = -(log_probs * normalized.detach()).mean()
        # 平滑 L1 损失训练 Critic 预测状态价值，对较大误差更稳健。
        critic_loss = torch.nn.functional.smooth_l1_loss(values, targets)
        # 熵越大，动作分布越分散；负的熵项鼓励训练期间继续探索。
        # 0.5 与 0.05 分别是价值损失和熵项的权重。
        loss = actor_loss + 0.5 * critic_loss - 0.05 * distribution.entropy().mean()
        # 清除上一回合的梯度，计算本回合梯度并更新参数。
        optimizer.zero_grad()
        loss.backward()
        # 限制梯度范数，减轻偶发的大更新对训练稳定性的影响。
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(result["wait"])
        # 每 100 回合或最后一回合，报告最近最多 100 回合的平均等待。
        if (episode + 1) % 100 == 0 or episode + 1 == episodes:
            recent = np.mean(history[-min(100, len(history)):])
            print(f"回合 {episode + 1:4d}/{episodes} | 最近平均等待 {recent:.0f} 车辆·分钟")
    return model, history


def compare(model: ActorCritic, seeds: list[int]) -> dict[str, float]:
    """在同一批车流种子上比较两种策略，返回各指标的场景平均值。"""
    # 相同种子保证两种策略面对相同的到达车辆序列。
    learned = [rollout(model, seed, explore=False)[1] for seed in seeds]
    fixed = [fixed_timing(seed) for seed in seeds]
    return {
        # wait 是累计车辆·分钟，越低越好；passed 是总通行量，越高越好。
        "learned_wait": float(np.mean([x["wait"] for x in learned])),
        "fixed_wait": float(np.mean([x["wait"] for x in fixed])),
        "learned_passed": float(np.mean([x["passed"] for x in learned])),
        "fixed_passed": float(np.mean([x["passed"] for x in fixed])),
        # switches 可用于观察策略是否通过频繁切灯换取短期收益。
        "learned_switches": float(np.mean([x["switches"] for x in learned])),
        "fixed_switches": float(np.mean([x["switches"] for x in fixed])),
    }


def save_chart(history: list[float], learned: dict, fixed: dict, stats: dict, output: Path) -> None:
    """保存训练曲线、测试集平均等待和单场景排队曲线。"""
    import os
    # 把 Matplotlib 配置缓存写入结果目录，避免依赖用户主目录的写权限。
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    # 无图形界面时也能直接把图保存到文件。
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    # 淡线显示逐回合结果，深线显示至多 40 回合的移动平均。
    axes[0].plot(history, color="#93c5fd", alpha=0.55, linewidth=1)
    width = min(40, len(history))
    smooth = np.convolve(history, np.ones(width) / width, mode="valid")
    axes[0].plot(np.arange(width - 1, len(history)), smooth, color="#1d4ed8", linewidth=2)
    axes[0].set(title="训练：每小时累计等待", xlabel="训练回合", ylabel="车辆·分钟")
    # 中图比较整批测试场景的平均累计等待。
    axes[1].bar(["固定配时", "Actor-Critic"], [stats["fixed_wait"], stats["learned_wait"]],
                color=["#f59e0b", "#2563eb"])
    axes[1].set(title="100 个相同车流场景的平均等待", ylabel="车辆·分钟")
    # 右图使用同一个测试种子；虚线标出车流主方向变化的时刻。
    for label, result, color in [("固定配时", fixed, "#f59e0b"), ("Actor-Critic", learned, "#2563eb")]:
        records = result["records"]
        # 每分钟两向队列相加，得到该时刻路口仍在等待的车辆数。
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
    # 每隔两分钟取一帧，控制 GIF 的文件大小。
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
            # 条形长度最多按 40 辆显示；绿灯方向为绿色，另一方向为橙色。
            draw.rounded_rectangle((25, y + 28, 25 + min(count, 40) / 40 * 575, y + 56),
                                   radius=7, fill="#f59e0b" if i != record["action"] else "#22c55e")
        draw.text((25, 306), f"Passed this minute: {record['passed']}  |  Switched: {record['switched']}",
                  fill="#475569", font=font)
        frames.append(image)
    # 180 毫秒切换一帧并无限循环播放。
    frames[0].save(output / "intersection.gif", save_all=True, append_images=frames[1:],
                   duration=180, loop=0, optimize=True)


def main() -> None:
    """解析参数，训练模型，并保存评估结果与可视化文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("actor_critic_output"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes 必须至少为 1")
    args.output.mkdir(parents=True, exist_ok=True)
    model, history = train(args.episodes, args.seed)
    # 保存模型权重，之后可按同样网络结构加载并使用。
    torch.save(model.state_dict(), args.output / "actor_critic_model.pt")
    test_seeds = list(range(100_000, 100_100))  # 与训练随机种子分开
    # 整批种子用于平均指标；第一组种子用于画两策略的具体轨迹。
    stats = compare(model, test_seeds)
    learned = rollout(model, test_seeds[0], explore=False)[1]
    fixed = fixed_timing(test_seeds[0])
    save_chart(history, learned, fixed, stats, args.output)
    save_animation(learned, args.output)
    # 正值表示学习策略比固定配时减少了等待，负值表示表现更差。
    reduction = 100 * (stats["fixed_wait"] - stats["learned_wait"]) / stats["fixed_wait"]
    print(f"测试集平均等待：固定配时 {stats['fixed_wait']:.1f}，Actor-Critic {stats['learned_wait']:.1f} 车辆·分钟")
    print(f"累计等待降幅：{reduction:.1f}%（正值表示改善）")
    print(f"结果已保存到 {args.output.resolve()}")


if __name__ == "__main__":
    main()
