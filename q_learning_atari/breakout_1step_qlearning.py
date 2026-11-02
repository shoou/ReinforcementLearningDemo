"""Breakout 的表格型一步 Q-learning 教学案例。

安装：python -m pip install "gymnasium[atari]" ale-py numpy
训练：python breakout_1step_qlearning.py --episodes 500 --steps 1000
评估：python breakout_1step_qlearning.py --mode eval
实时观看：python breakout_1step_qlearning.py --mode eval --render --eval-episodes 1

从 Atari RAM 提取球和挡板位置、球运动方向，并用一步 Q-learning 更新 Q 表。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ALE 环境 ID、动作顺序和默认模型路径。Q 表的动作维按 ACTIONS 的编号存储，
# 因此运行时必须确认环境实际提供的最小动作集合与这里完全一致。
GAME = "ALE/Breakout-v5"
ACTIONS = ("NOOP", "FIRE", "RIGHT", "LEFT")  # 动作编号须与 ALE 一致
DEFAULT_MODEL = Path("qlearning_output/breakout_qtable.npz")

# Breakout RAM 布局，参考：
# https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/breakout.py
# Atari 的 RAM 观测是长度 128 的字节数组；下列索引指向特定游戏变量。
RAM_SIZE = 128
PADDLE_X_RAM = 72
BALL_X_RAM = 99
BALL_Y_RAM = 101
LIVES_RAM = 57
# 球的纵坐标为 0 或大于此值时，视作尚未发球或已掉球。
BALL_LOST_Y = 187
# 球相对挡板的水平距离最多保留 ±96，再按每 16 像素分箱。
OFFSET_WIDTH = 16
OFFSET_LIMIT = 96
# 4 个阶段 × 13 个偏移档 × 3 个水平运动方向 × 3 个竖直运动方向
# × 4 个动作。一个格子保存该状态下执行一个动作的估计回报。
Q_SHAPE = (4, 13, 3, 3, len(ACTIONS))  # 阶段、水平偏移、水平/竖直方向、动作

# ε 从 1.0 线性降到 0.05；训练时另给少量追球反馈和每步惩罚。
DEFAULT_EPSILON_START = 1.0
DEFAULT_EPSILON_END = 0.05
DEFAULT_PROGRESS_REWARD = 0.015
DEFAULT_STEP_PENALTY = 0.001
LOG_INTERVAL = 25
EVAL_SEED_OFFSET = 10_000  # 评估回合使用不同于训练回合的随机种子


def make_env(render_mode: str | None = None, difficulty: int = 0):
    """创建输出 RAM 的 Breakout 环境，并核对动作编号。

    render_mode 为 "human" 时由 ALE 显示游戏窗口；默认不显示。
    """
    try:
        import gymnasium as gym
        from ale_py.registration import register_v5_envs
    except ImportError as exc:
        raise SystemExit(
            '缺少 Atari 依赖：python -m pip install "gymnasium[atari]" ale-py numpy'
        ) from exc
    # 某些 gymnasium/ale-py 组合不会自动注册 v5 环境。
    if GAME not in gym.envs.registry:
        register_v5_envs()
    env = gym.make(GAME, obs_type="ram", render_mode=render_mode,
                   difficulty=difficulty)
    # 如果动作顺序变化，Q 表的最后一维会指向错误的实际操作。
    if tuple(env.unwrapped.get_action_meanings()) != ACTIONS:
        env.close()
        raise RuntimeError(f"{GAME} 动作集合与预期不符")
    return env


def encode_state(ram: np.ndarray,
                 previous_ram: np.ndarray | None = None) -> tuple[int, int, int, int]:
    """返回球的阶段、相对挡板偏移及水平/竖直方向，供 Q 表索引。

    返回的四元组依次对应 Q_SHAPE 的前四维。方向由相邻两帧 RAM 的
    位置差估计；回合第一帧方向设为未知档位 1。
    """
    if ram.shape != (RAM_SIZE,):
        raise ValueError(f"预期 {RAM_SIZE} 字节 RAM，实际为 {ram.shape}")
    ball_x = int(ram[BALL_X_RAM])
    ball_y = int(ram[BALL_Y_RAM])
    paddle_x = int(ram[PADDLE_X_RAM])
    # 未开球或掉球时单独编码；训练/评估会自动按 FIRE。
    if ball_y == 0 or ball_y > BALL_LOST_Y:
        return 0, 6, 1, 1
    # 运动中的球按纵向位置分为上部(1)、中部(2)、下部(3)。
    phase = 1 if ball_y < 100 else (2 if ball_y < 150 else 3)
    # 偏移截断到 [-96, 96]，再以 16 像素为宽度分成 13 档。
    offset = max(-OFFSET_LIMIT, min(ball_x - paddle_x, OFFSET_LIMIT))
    # 加上 96 后得到非负索引 0～12；6 表示水平位置大致对齐。
    offset_bin = (offset + OFFSET_LIMIT) // OFFSET_WIDTH
    if previous_ram is None:
        vx = vy = 1
    else:
        # 当前帧减上一帧，得到球在两个环境交互步之间的位移。
        dx = ball_x - int(previous_ram[BALL_X_RAM])
        dy = ball_y - int(previous_ram[BALL_Y_RAM])
        # 0/1/2 分别表示负向、静止或未知、正向。
        vx = 0 if dx < 0 else (2 if dx > 0 else 1)
        vy = 0 if dy < 0 else (2 if dy > 0 else 1)
    return phase, offset_bin, vx, vy


def training_reward(game_reward: float, old_ram: np.ndarray,
                    new_ram: np.ndarray, progress_reward: float,
                    step_penalty: float) -> float:
    """计算训练奖励：原始得分 + 追球反馈 - 步惩罚 - 掉球惩罚。

    追球反馈只影响学习更新；训练日志和评估仍只累计环境原始得分。
    """
    old_y = int(old_ram[BALL_Y_RAM])
    new_y = int(new_ram[BALL_Y_RAM])
    alignment = 0.0
    # 球位于画面下部且尚未消失时，才评价挡板是否追近球。
    if 140 <= old_y <= BALL_LOST_Y and new_y != 0:
        old_gap = abs(int(old_ram[BALL_X_RAM]) - int(old_ram[PADDLE_X_RAM]))
        new_gap = abs(int(new_ram[BALL_X_RAM]) - int(new_ram[PADDLE_X_RAM]))
        # 旧偏差减新偏差：变小为正，变大为负。除以 16 将位移换算成
        # 一个偏移档的比例，clip 把单步反馈限制在 ±progress_reward。
        alignment = progress_reward * np.clip(
            (old_gap - new_gap) / OFFSET_WIDTH, -1, 1
        )
    # RAM 中的生命数减少意味着掉球；额外扣 0.1，强于普通步惩罚。
    life_loss = 0.1 if new_ram[LIVES_RAM] < old_ram[LIVES_RAM] else 0
    return float(game_reward) + alignment - step_penalty - life_loss


def epsilon_for_episode(episode: int, start: float, end: float,
                        decay_episodes: int) -> float:
    """线性探索率：第 0 回合为 start，第 decay_episodes 回合达到 end。

    此后保持 end；默认衰减期覆盖从第一回合到最后一回合。
    """
    fraction = min(1.0, episode / decay_episodes)
    return start + (end - start) * fraction


def choose_action(q: np.ndarray, state: tuple[int, int, int, int],
                  epsilon: float, rng: np.random.Generator) -> int:
    """以 ε 概率随机探索，否则从最高 Q 值动作中随机选择。

    多个动作的 Q 值并列最高时随机打破平局，避免初始全零表固定选动作 0。
    """
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[-1]))
    values = q[state]
    return int(rng.choice(np.flatnonzero(values == values.max())))


def forced_action(ram: np.ndarray) -> int | None:
    """未开球或掉球时返回 FIRE（动作 1）；否则返回 None。

    训练和三种评估策略都先调用此函数，保证它们能正常开始游戏。
    """
    y = int(ram[BALL_Y_RAM])
    if y == 0 or y > BALL_LOST_Y:
        return 1
    return None


def baseline_action(ram: np.ndarray) -> int:
    """球在挡板右侧就向右、左侧就向左，5 像素内保持不动。"""
    gap = int(ram[BALL_X_RAM]) - int(ram[PADDLE_X_RAM])
    return 2 if gap > 5 else (3 if gap < -5 else 0)


def train(episodes: int, steps: int, seed: int, alpha: float, gamma: float,
          model_path: Path, difficulty: int = 0,
          epsilon_start: float = DEFAULT_EPSILON_START,
          epsilon_end: float = DEFAULT_EPSILON_END,
          epsilon_decay_episodes: int | None = None,
          progress_reward: float = DEFAULT_PROGRESS_REWARD,
          step_penalty: float = DEFAULT_STEP_PENALTY) -> np.ndarray:
    """从零初始化 Q 表，用一步 Q-learning 训练并保存结果。

    alpha 是当前样本的学习率；gamma 是下一状态最佳 Q 值的折扣率。
    episodes 是回合数，steps 是单回合最多执行的环境交互步数。
    """
    rng = np.random.default_rng(seed)
    # 起始 Q 值全为 0；状态四元组和动作编号合起来定位一个标量。
    q = np.zeros(Q_SHAPE, dtype=np.float32)
    scores = []
    # 多回合时默认在最后一回合达到最终 ε；max(1, ...) 避免单回合时除以零。
    decay_episodes = (max(1, episodes - 1) if epsilon_decay_episodes is None
                      else epsilon_decay_episodes)
    env = make_env(difficulty=difficulty)
    try:
        for episode in range(episodes):
            # 每回合重置环境并改变种子；相同命令参数可复现这一序列。
            ram, _ = env.reset(seed=seed + episode)
            previous_ram = None  # 首帧无法判断球的运动方向
            score = 0.0
            epsilon = epsilon_for_episode(
                episode, epsilon_start, epsilon_end, decay_episodes
            )
            for step in range(steps):
                state = encode_state(ram, previous_ram)
                # 未开球或掉球时按 FIRE；否则按当前回合的 ε-greedy 策略。
                required = forced_action(ram)
                action = (required if required is not None
                          else choose_action(q, state, epsilon, rng))
                next_ram, game_reward, terminated, truncated, _ = env.step(action)
                # 比较当前 RAM 和下一帧 RAM，估计下一状态的球运动方向。
                next_state = encode_state(next_ram, ram)
                # 环境结束、环境截断或达到用户设置的步数上限，都结束本回合。
                done = terminated or truncated or step + 1 == steps
                reward = training_reward(game_reward, ram, next_ram,
                                         progress_reward, step_penalty)
                # 一步 Q-learning 的目标：r + γ max_a' Q(s', a')。
                # 使用下一状态的最大 Q 值是离策略更新；终止状态不再有未来收益。
                target = reward + (0 if done else gamma * float(q[next_state].max()))
                # 只修改当前 (状态, 动作) 对应的表格项：
                # Q(s,a) <- Q(s,a) + α [target - Q(s,a)]。
                q[state + (action,)] += alpha * (target - q[state + (action,)])
                score += float(game_reward)  # 日志只记录原始得分
                # 保存本帧，让下一轮能从相邻帧位置差估计球的方向。
                previous_ram = ram
                ram = next_ram
                if done:
                    break
            scores.append(score)
            # 最近 LOG_INTERVAL 回合的平均值用于观察训练变化，非训练奖励。
            if (episode + 1) % LOG_INTERVAL == 0 or episode + 1 == episodes:
                print(f"训练 {episode + 1:4d}/{episodes} | 最近平均真实得分 "
                      f"{np.mean(scores[-LOG_INTERVAL:]):.2f} | ε={epsilon:.2f}")
    finally:
        # 即使训练途中抛错，也释放 Atari 环境与可能创建的窗口资源。
        env.close()
    # 同一压缩文件保存 Q 表、逐回合真实得分和训练参数，便于后续评估。
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, q=q, scores=np.asarray(scores),
                        seed=seed, episodes=episodes, steps=steps,
                        difficulty=difficulty, alpha=alpha, gamma=gamma,
                        epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                        epsilon_decay_episodes=decay_episodes,
                        progress_reward=progress_reward,
                        step_penalty=step_penalty, game=GAME)
    print(f"Q 表已保存：{model_path}")
    return q


def evaluate(q: np.ndarray, episodes: int, steps: int, seed: int,
             render: bool = False, difficulty: int = 0) -> None:
    """用同一组回合种子比较 Q-learning、追球基线和随机策略。

    Q-learning 以 ε=0 选动作；并列最高 Q 值仍会随机打破平局。
    评估不使用训练辅助奖励，只比较游戏原始得分。
    """
    rng = np.random.default_rng(seed)
    for name in ("Q-learning", "追球基线", "随机"):
        # human 模式只用于学习策略；其余策略不显示画面。
        env = make_env(render_mode="human" if render and name == "Q-learning"
                       else None, difficulty=difficulty)
        try:
            scores = []
            for episode in range(episodes):
                ram, _ = env.reset(seed=seed + episode)
                previous_ram = None
                total = 0.0
                for _ in range(steps):
                    # 三种策略都先完成必要的 FIRE 开球动作。
                    required = forced_action(ram)
                    if required is not None:
                        action = required
                    elif name == "Q-learning":
                        action = choose_action(q, encode_state(ram, previous_ram),
                                               0.0, rng)
                    elif name == "追球基线":
                        # 对照策略只依赖球与挡板的水平位置，不读取 Q 表。
                        action = baseline_action(ram)
                    else:
                        action = int(rng.integers(q.shape[-1]))
                    next_ram, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)  # 评估只统计游戏原始奖励
                    previous_ram, ram = ram, next_ram
                    if terminated or truncated:
                        break
                scores.append(total)
            print(f"{name:12s} | 平均真实得分 {np.mean(scores):.2f} "
                  f"| 各回合 {scores}")
        finally:
            env.close()


def main() -> None:
    """解析参数，训练或加载 Breakout Q 表，最后执行评估。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=1000, help="每回合最多交互步数")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--difficulty", type=int, choices=(0, 1), default=0,
                        help="Breakout 游戏难度：0（默认）或 1")
    parser.add_argument("--render", action="store_true",
                        help="评估时实时显示 Q-learning 的 Atari 画面")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.15, help="Q 值学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="未来奖励折扣率")
    parser.add_argument("--epsilon-start", type=float,
                        default=DEFAULT_EPSILON_START, help="初始随机探索概率")
    parser.add_argument("--epsilon-end", type=float,
                        default=DEFAULT_EPSILON_END, help="最终随机探索概率")
    parser.add_argument("--epsilon-decay-episodes", type=int, default=None,
                        help="探索率线性衰减所用回合数；默认覆盖全部训练回合")
    parser.add_argument("--progress-reward", type=float,
                        default=DEFAULT_PROGRESS_REWARD, help="挡板追球的辅助奖励强度")
    parser.add_argument("--step-penalty", type=float,
                        default=DEFAULT_STEP_PENALTY, help="每步固定惩罚")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Q 表文件路径")
    args = parser.parse_args()
    # 提前拒绝无效超参数，避免运行很久后才因数值或索引错误终止。
    if min(args.episodes, args.steps, args.eval_episodes) <= 0:
        parser.error("episodes、steps 和 eval-episodes 必须为正整数")
    if not 0 < args.alpha <= 1 or not 0 <= args.gamma <= 1:
        parser.error("alpha 必须在 (0,1]，gamma 必须在 [0,1]")
    if not 0 <= args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("探索率需满足 0 <= epsilon-end <= epsilon-start <= 1")
    if args.epsilon_decay_episodes is not None and args.epsilon_decay_episodes <= 0:
        parser.error("epsilon-decay-episodes 必须为正整数")
    if args.progress_reward < 0 or args.step_penalty < 0:
        parser.error("progress-reward 和 step-penalty 不得为负数")
    if args.mode == "train":
        # 训练从全零 Q 表开始；完成后立即用独立种子执行评估。
        q = train(args.episodes, args.steps, args.seed, args.alpha, args.gamma,
                  args.model, difficulty=args.difficulty,
                  epsilon_start=args.epsilon_start,
                  epsilon_end=args.epsilon_end,
                  epsilon_decay_episodes=args.epsilon_decay_episodes,
                  progress_reward=args.progress_reward,
                  step_penalty=args.step_penalty)
    else:
        # 加载现有模型时核对游戏 ID 与表格维度，避免误用其他游戏的 Q 表。
        if not args.model.exists():
            parser.error(f"找不到 Q 表：{args.model}；请先运行训练")
        with np.load(args.model, allow_pickle=False) as data:
            q = data["q"]
            saved_game = str(data["game"]) if "game" in data else "ALE/Freeway-v5"
            if saved_game != GAME:
                parser.error(f"模型属于 {saved_game}，当前游戏是 {GAME}")
            if "difficulty" in data and int(data["difficulty"]) != args.difficulty:
                # 游戏难度变化不改变表格形状，但成绩可能不可直接比较。
                print(f"提示：模型训练难度为 {int(data['difficulty'])}，"
                      f"当前评估难度为 {args.difficulty}。")
        if q.shape != Q_SHAPE:
            parser.error(f"Q 表形状不匹配：{q.shape}，预期 {Q_SHAPE}")
    # 评估种子加上偏移量，通常可避免重放训练时见过的回合。
    evaluate(q, args.eval_episodes, args.steps, args.seed + EVAL_SEED_OFFSET,
             render=args.render, difficulty=args.difficulty)


if __name__ == "__main__":
    main()
