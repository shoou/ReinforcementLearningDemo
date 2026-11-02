"""可运行的 DQN 网格寻路案例；输出训练曲线、路线图和动画。"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# 地图坐标采用（行，列）：左上角是 (0, 0)，向下行号增大，向右列号增大。
SIZE = 32
START = (0, 0)
GOAL = (SIZE - 1, SIZE - 1)
# 五道横墙的缺口左右交替，形成必须绕行的路线；起点和终点保持畅通。
WALLS = {
    (row, col)
    for index, row in enumerate(range(5, SIZE - 1, 6))
    for col in range(SIZE)
    if col not in (range(2, 8) if index % 2 == 0 else range(SIZE - 8, SIZE - 2))
}
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # 上、下、左、右


def goal_distances() -> dict[tuple[int, int], int]:
    """计算每个可达格到终点的最短距离，供奖励塑形和课程训练使用。"""
    distances = {GOAL: 0}
    queue = deque([GOAL])
    while queue:
        row, col = queue.popleft()
        for dr, dc in ACTIONS:
            point = (row + dr, col + dc)
            if (0 <= point[0] < SIZE and 0 <= point[1] < SIZE
                    and point not in WALLS and point not in distances):
                distances[point] = distances[(row, col)] + 1
                queue.append(point)
    return distances


DISTANCES = goal_distances()


def training_route() -> list[tuple[int, int]]:
    """得到一条可行的最短路线，用于从近到远安排训练起点。"""
    if START not in DISTANCES:
        raise ValueError("起点无法到达终点")
    route = [START]
    while route[-1] != GOAL:
        row, col = route[-1]
        route.append(next(
            (row + dr, col + dc) for dr, dc in ACTIONS
            if DISTANCES.get((row + dr, col + dc)) == DISTANCES[(row, col)] - 1
        ))
    return route


TRAINING_ROUTE = training_route()


class GridWorld:
    """固定地图的寻路环境，接口仿照强化学习常见的 reset/step。"""

    def __init__(self, max_steps: int = SIZE * 8):
        self.max_steps = max_steps
        self.reset()

    def reset(self, start: tuple[int, int] = START) -> np.ndarray:
        """开始新回合；训练可从路线中途开始，评估始终从起点开始。"""
        if start not in DISTANCES:
            raise ValueError("训练起点必须是可到达终点的空格")
        self.position = start
        self.steps = 0
        return self.state()

    def state(self) -> np.ndarray:
        # 坐标和四个方向的可通行性、最短距离变化，让网络能在不同位置复用经验。
        row, col = self.position
        features = [row / (SIZE - 1), col / (SIZE - 1)]
        for dr, dc in ACTIONS:
            neighbor = (row + dr, col + dc)
            features.append(float(DISTANCES[self.position] - DISTANCES[neighbor])
                            if neighbor in DISTANCES else 0.0)
        for dr, dc in ACTIONS:
            features.append(float((row + dr, col + dc) in DISTANCES))
        return np.asarray(features, dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        """执行一个动作，返回（新状态、奖励、是否结束、是否到达目标）。"""
        row = self.position[0] + ACTIONS[action][0]
        col = self.position[1] + ACTIONS[action][1]
        candidate = (row, col)
        # 撞墙或越界时保持原地，但仍消耗一步，并给予较大的负奖励。
        valid = candidate in DISTANCES
        previous_distance = DISTANCES[self.position]
        if valid:
            self.position = candidate
        self.steps += 1
        success = self.position == GOAL
        done = success or self.steps >= self.max_steps
        # 实际绕墙距离缩短时给予反馈，避免长路线中始终得不到目标信号。
        reward = 10.0 if success else (-1.2 if not valid else
                                        -0.01 + (previous_distance - DISTANCES[self.position]))
        return self.state(), reward, done, success


class QNetwork(torch.nn.Module):
    """把位置状态映射为 4 个动作的 Q 值。"""

    def __init__(self):
        super().__init__()
        # 两层隐藏层；最后一层不做激活，因为 Q 值可以为负数。
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(ACTIONS)),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


def choose_action(network: QNetwork, state: np.ndarray, epsilon: float) -> int:
    """ε-greedy：以 ε 的概率随机探索，否则选择 Q 值最大的动作。"""
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    # 选动作时只做推理，不需要构建梯度计算图。
    with torch.no_grad():
        return int(network(torch.from_numpy(state).unsqueeze(0)).argmax(dim=1).item())


def learn(
    online: QNetwork,
    target: QNetwork,
    replay: deque,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 64,
    gamma: float = 0.99,
) -> float:
    """从经验回放池采样一批转移，执行一次 DQN 参数更新。"""
    # 每条经验是 (s, a, r, s', done)；随机采样可打散相邻步的相关性。
    batch = random.sample(replay, batch_size)
    states = torch.from_numpy(np.stack([item[0] for item in batch]))
    actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
    rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    next_states = torch.from_numpy(np.stack([item[3] for item in batch]))
    dones = torch.tensor([item[4] for item in batch], dtype=torch.float32)

    # 在线网络会输出每个动作的 Q 值，这里只取实际执行动作 a 对应的值。
    predicted = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        # 使用更新较慢的目标网络计算 y = r + γ(1-done)max Q_target(s', a')。
        best_next = target(next_states).max(dim=1).values
        # 终止状态没有未来回报。
        target_q = rewards + gamma * (1.0 - dones) * best_next
    # L1平滑损失比平方误差对偶发的大误差更稳健。
    loss = torch.nn.functional.smooth_l1_loss(predicted, target_q)
    optimizer.zero_grad()
    loss.backward()
    # 限制梯度大小，降低训练过程突然波动的风险。
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def train(episodes: int, seed: int) -> tuple[QNetwork, dict[str, list[float]]]:
    """训练智能体，同时记录用于绘制曲线的奖励、成功率等指标。"""
    # 固定随机种子，便于复现相近的训练结果。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    environment = GridWorld()
    # online 负责学习，target 负责计算相对稳定的训练目标。
    online, target = QNetwork(), QNetwork()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=0.0003)
    # 回放池只保留最近 20,000 条经验，满后自动丢弃最早的经验。
    replay: deque = deque(maxlen=20_000)
    history: dict[str, list[float]] = {"reward": [], "success": [], "epsilon": [], "loss": []}
    updates = 0

    for episode in range(episodes):
        progress = episode / max(1, episodes - 1)
        # 先从目标附近学习短路线，再逐渐延长；最后阶段固定从地图起点训练。
        if progress < 0.65:
            max_distance = min(len(TRAINING_ROUTE) - 1,
                               8 + int(progress / 0.65 * (len(TRAINING_ROUTE) - 1)))
            start_index = random.randint(len(TRAINING_ROUTE) - 1 - max_distance,
                                         len(TRAINING_ROUTE) - 2)
            state = environment.reset(TRAINING_ROUTE[start_index])
        else:
            state = environment.reset()
        # 探索率随总训练回合数线性下降，保留 0.1 的随机探索。
        epsilon = max(0.1, 1.0 - 0.9 * progress / 0.8)
        total_reward = 0.0
        episode_losses = []
        for _ in range(environment.max_steps):
            action = choose_action(online, state, epsilon)
            next_state, reward, done, success = environment.step(action)
            # 即使动作撞墙，也将这一步存入回放池，让网络学会避开它。
            replay.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            # 每四步更新一次，降低大地图训练的计算量。
            if len(replay) >= 64 and environment.steps % 4 == 0:
                episode_losses.append(learn(online, target, replay, optimizer))
                updates += 1
                # 目标网络每 500 次梯度更新才同步一次在线网络的参数。
                if updates % 500 == 0:
                    target.load_state_dict(online.state_dict())
            if done:
                break

        # 按回合保存指标；画图时成功率会再计算最近 30 回合的平均值。
        history["reward"].append(total_reward)
        history["success"].append(float(success))
        history["epsilon"].append(epsilon)
        history["loss"].append(float(np.mean(episode_losses)) if episode_losses else 0.0)
        if (episode + 1) % 50 == 0 or episode + 1 == episodes:
            recent = history["success"][-50:]
            print(f"Episode {episode + 1:4d}/{episodes} | "
                  f"recent success {np.mean(recent):.0%} | epsilon {epsilon:.3f}")
    return online, history


def evaluate(network: QNetwork) -> tuple[list[tuple[int, int]], float, bool]:
    """关闭随机探索，用训练后的策略走一回合并记录整条路线。"""
    environment = GridWorld()
    state = environment.reset()
    route = [environment.position]
    total_reward = 0.0
    success = False
    for _ in range(environment.max_steps):
        action = choose_action(network, state, epsilon=0.0)
        state, reward, done, success = environment.step(action)
        route.append(environment.position)
        total_reward += reward
        if done:
            break
    return route, total_reward, success


def draw_frame(route: list[tuple[int, int]], step: int) -> Image.Image:
    """把路线的第 step 步画成一张图片，供静态图和 GIF 复用。"""
    cell, margin, heading = max(20, 400 // SIZE), 18, 48
    image = Image.new("RGB", (SIZE * cell + 2 * margin, SIZE * cell + 2 * margin + heading), "#f7fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((margin, 19), f"DQN route - step {step}/{len(route) - 1}", fill="#183153", font=font)
    # 绿色为起点、黄色为目标、深色为障碍，蓝色圆点为智能体。
    for row in range(SIZE):
        for col in range(SIZE):
            x0, y0 = margin + col * cell, margin + heading + row * cell
            color = "#334155" if (row, col) in WALLS else "#ffffff"
            if (row, col) == START:
                color = "#bbf7d0"
            if (row, col) == GOAL:
                color = "#fde68a"
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline="#cbd5e1", width=2)
    # 先画走过的连线，再用圆点标出历史位置和当前位置。
    points = [(margin + col * cell + cell // 2, margin + heading + row * cell + cell // 2)
              for row, col in route[:step + 1]]
    if len(points) >= 2:
        draw.line(points, fill="#60a5fa", width=5)
    for index, (row, col) in enumerate(route[:step]):
        x = margin + col * cell + cell // 2
        y = margin + heading + row * cell + cell // 2
        radius = max(2, cell // 8)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#94a3b8")
    row, col = route[step]
    x = margin + col * cell + cell // 2
    y = margin + heading + row * cell + cell // 2
    radius = max(6, cell // 3)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#2563eb", outline="white", width=2)
    draw.text((x - 4, y - 5), "A", fill="white", font=font)
    return image


def save_visuals(history: dict[str, list[float]], route: list[tuple[int, int]], output: Path) -> None:
    """保存训练曲线、最终路线图，以及逐帧播放的寻路 GIF。"""
    # 设置可写的缓存目录，避免在只读 HOME 中创建 Matplotlib 文件。
    (output / ".cache").mkdir(exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(output / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def moving_average(values: list[float], width: int = 30) -> np.ndarray:
        """计算截至当前回合的最近 width 个值的平均值，用于平滑曲线。"""
        data = np.asarray(values)
        return np.array([data[max(0, i - width + 1):i + 1].mean() for i in range(len(data))])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    episodes = np.arange(1, len(history["reward"]) + 1)
    # 左：单回合奖励与平滑趋势；中：最近 30 回合成功率；右：探索概率。
    axes[0].plot(episodes, history["reward"], color="#bfdbfe", linewidth=1)
    axes[0].plot(episodes, moving_average(history["reward"]), color="#2563eb", linewidth=2)
    axes[0].set(title="Episode reward", xlabel="Episode", ylabel="Reward")
    axes[1].plot(episodes, moving_average(history["success"]), color="#16a34a", linewidth=2)
    axes[1].set(title="Success rate (last 30)", xlabel="Episode", ylabel="Rate", ylim=(-0.03, 1.03))
    axes[2].plot(episodes, history["epsilon"], color="#d97706", linewidth=2)
    axes[2].set(title="Exploration probability", xlabel="Episode", ylabel="Epsilon", ylim=(0, 1.03))
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training.png", dpi=160)
    plt.close(fig)

    # 同一组帧既生成最后一步的静态图，也合成为循环播放的 GIF。
    frames = [draw_frame(route, step) for step in range(len(route))]
    frames[-1].save(output / "route.png")
    frames[0].save(output / "demo.gif", save_all=True, append_images=frames[1:], duration=450, loop=0)


def main() -> None:
    """解析命令行参数，依次完成训练、评估、模型保存和可视化。"""
    parser = argparse.ArgumentParser(description="DQN 网格寻路教学案例")
    parser.add_argument("--episodes", type=int, default=500, help="训练回合数，默认 500")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--output", type=Path, default=Path("dqn_output"), help="输出目录")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes 必须大于 0")
    args.output.mkdir(parents=True, exist_ok=True)
    network, history = train(args.episodes, args.seed)
    route, reward, success = evaluate(network)
    torch.save(network.state_dict(), args.output / "dqn_model.pt")
    save_visuals(history, route, args.output)
    print(f"Greedy evaluation: success={success}, steps={len(route) - 1}, reward={reward:.2f}")
    print(f"Files saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
