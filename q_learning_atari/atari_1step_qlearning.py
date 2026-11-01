"""Atari Freeway / Breakout 的表格型一步 Q-learning 教学案例。

安装：python -m pip install "gymnasium[atari]" ale-py numpy
训练：python atari_1step_qlearning.py --episodes 300
Breakout：python atari_1step_qlearning.py --game breakout --episodes 500
评估：python atari_1step_qlearning.py --mode eval
实时观看：python atari_1step_qlearning.py --mode eval --render --eval-episodes 1
难度 1：python atari_1step_qlearning.py --mode eval --difficulty 1
探索率：python atari_1step_qlearning.py --epsilon-start 1 --epsilon-end 0.1 --epsilon-decay-episodes 150

Freeway 用小鸡高度和附近车辆距离作为状态；Breakout 用球、挡板位置及
球的运动方向作为状态。两者都从 Atari RAM 提取特征，并使用相同更新规则：
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ALE 还提供 Pong、SpaceInvaders 等游戏；新增游戏需实现专属状态与动作逻辑。
GAME = "ALE/Freeway-v5"  # 默认游戏；也可用 --game breakout 选择第二个示例
ACTIONS = ("NOOP", "UP", "DOWN")
GAME_IDS = {"freeway": GAME, "breakout": "ALE/Breakout-v5"}
GAME_ACTIONS = {
    GAME: ACTIONS,
    GAME_IDS["breakout"]: ("NOOP", "FIRE", "RIGHT", "LEFT"),
}
DEFAULT_MODEL = Path("qlearning_output/freeway_qtable.npz")
DEFAULT_MODELS = {
    GAME: DEFAULT_MODEL,
    GAME_IDS["breakout"]: Path("qlearning_output/breakout_qtable.npz"),
}

# 以下是 Freeway 游戏本身的 RAM 布局和画面坐标，不是学习超参数。
# 位置映射参考 OCAtari: https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/freeway.py
RAM_SIZE = 128
CHICKEN_Y_RAM = 14
FIRST_CAR_X_RAM = 117
CAR_LANES = 10
CHICKEN_X = 44
CHICKEN_Y_ORIGIN = 193
CAR_X_OFFSET = 3
FIRST_LANE_Y = 27
LANE_GAP = 16
LANE_LOOKAHEAD = 18

# 状态抽象：将连续位置压缩成有限个 Q 表格子。
HEIGHT_BIN_WIDTH = 8
HEIGHT_BINS = 26
NEAR_CAR_DISTANCE = 16
MID_CAR_DISTANCE = 40
NO_NEARBY_CAR_DISTANCE = 160
Q_SHAPE = (HEIGHT_BINS, 3, len(ACTIONS))  # 高度、危险程度、动作
BREAKOUT_Q_SHAPE = (4, 13, 3, 3, 4)  # 阶段、球与挡板距离、水平/竖直方向、动作
Q_SHAPES = {GAME: Q_SHAPE, GAME_IDS["breakout"]: BREAKOUT_Q_SHAPE}

# Breakout RAM: https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/breakout.py
BREAKOUT_PADDLE_X_RAM = 72
BREAKOUT_BALL_X_RAM = 99
BREAKOUT_BALL_Y_RAM = 101
BREAKOUT_LIVES_RAM = 57
BREAKOUT_BALL_LOST_Y = 187
BREAKOUT_OFFSET_WIDTH = 16
BREAKOUT_OFFSET_LIMIT = 96

# 这些默认值可通过命令行调整；衰减期默认覆盖整个训练过程。
DEFAULT_EPSILON_START = 1.0
DEFAULT_EPSILON_END = 0.05
DEFAULT_PROGRESS_REWARD = 0.015
DEFAULT_STEP_PENALTY = 0.001
PROGRESS_LIMIT = 8  # 单步前进奖励截断，防止碰撞/重置造成异常值
LOG_INTERVAL = 25
EVAL_SEED_OFFSET = 10_000  # 评估与训练使用不同随机种子


def make_env(render_mode: str | None = None, difficulty: int = 0,
             game: str = GAME):
    try:
        import gymnasium as gym
        from ale_py.registration import register_v5_envs
    except ImportError as exc:
        raise SystemExit(
            '缺少 Atari 依赖：python -m pip install "gymnasium[atari]" ale-py numpy'
        ) from exc
    # Gymnasium 1.0 的 register_envs 是空操作；部分 ale-py 版本导入时
    # 也会因可选向量环境不兼容而跳过自动注册，因此显式注册 v5 环境。
    if game not in gym.envs.registry:
        register_v5_envs()
    env = gym.make(game, obs_type="ram", render_mode=render_mode,
                   difficulty=difficulty)
    if tuple(env.unwrapped.get_action_meanings()) != GAME_ACTIONS[game]:
        env.close()
        raise RuntimeError(f"{game} 动作集合与预期不符")
    return env


def encode_state(ram: np.ndarray, game: str = GAME,
                 previous_ram: np.ndarray | None = None) -> tuple[int, ...]:
    """将游戏 RAM 转为小型离散状态，具体特征随游戏而变。"""
    if ram.shape != (RAM_SIZE,):
        raise ValueError(f"预期 {RAM_SIZE} 字节 RAM，实际为 {ram.shape}")
    if game == GAME_IDS["breakout"]:
        ball_x = int(ram[BREAKOUT_BALL_X_RAM])
        ball_y = int(ram[BREAKOUT_BALL_Y_RAM])
        paddle_x = int(ram[BREAKOUT_PADDLE_X_RAM])
        # 未开球或掉球时进入单独阶段，训练/评估会自动按 FIRE。
        if ball_y == 0 or ball_y > BREAKOUT_BALL_LOST_Y:
            return 0, 6, 1, 1
        phase = 1 if ball_y < 100 else (2 if ball_y < 150 else 3)
        offset = max(-BREAKOUT_OFFSET_LIMIT,
                     min(ball_x - paddle_x, BREAKOUT_OFFSET_LIMIT))
        offset_bin = (offset + BREAKOUT_OFFSET_LIMIT) // BREAKOUT_OFFSET_WIDTH
        if previous_ram is None:
            vx = vy = 1  # 首帧未知方向
        else:
            dx = ball_x - int(previous_ram[BREAKOUT_BALL_X_RAM])
            dy = ball_y - int(previous_ram[BREAKOUT_BALL_Y_RAM])
            vx = 0 if dx < 0 else (2 if dx > 0 else 1)
            vy = 0 if dy < 0 else (2 if dy > 0 else 1)
        return phase, offset_bin, vx, vy
    if game != GAME:
        raise ValueError(f"未实现状态提取：{game}")

    # Freeway: RAM[14] 是小鸡高度，RAM[117-i] 是第 i 条车道的车辆 x。
    height = int(ram[CHICKEN_Y_RAM])
    chicken_y = CHICKEN_Y_ORIGIN - height
    distances = [
        abs(int(ram[FIRST_CAR_X_RAM - lane]) - CAR_X_OFFSET - CHICKEN_X)
        for lane in range(CAR_LANES)
        if abs(FIRST_LANE_Y + LANE_GAP * lane - chicken_y) <= LANE_LOOKAHEAD
    ]
    closest = min(distances, default=NO_NEARBY_CAR_DISTANCE)
    danger = (0 if closest < NEAR_CAR_DISTANCE
              else 1 if closest < MID_CAR_DISTANCE else 2)
    return min(height // HEIGHT_BIN_WIDTH, HEIGHT_BINS - 1), danger


def training_reward(game_reward: float, old_ram: np.ndarray,
                    new_ram: np.ndarray, progress_reward: float,
                    step_penalty: float, game: str = GAME) -> float:
    """真实得分加少量引导奖励，评估时仍只统计游戏原始得分。"""
    if game == GAME_IDS["breakout"]:
        # 球接近挡板时，奖励挡板追近球；掉球则用 RAM 生命数给惩罚。
        old_y = int(old_ram[BREAKOUT_BALL_Y_RAM])
        new_y = int(new_ram[BREAKOUT_BALL_Y_RAM])
        alignment = 0.0
        if 140 <= old_y <= BREAKOUT_BALL_LOST_Y and new_y != 0:
            old_gap = abs(int(old_ram[BREAKOUT_BALL_X_RAM])
                          - int(old_ram[BREAKOUT_PADDLE_X_RAM]))
            new_gap = abs(int(new_ram[BREAKOUT_BALL_X_RAM])
                          - int(new_ram[BREAKOUT_PADDLE_X_RAM]))
            alignment = progress_reward * np.clip(
                (old_gap - new_gap) / BREAKOUT_OFFSET_WIDTH, -1, 1
            )
        life_loss = 0.1 if new_ram[BREAKOUT_LIVES_RAM] < old_ram[BREAKOUT_LIVES_RAM] else 0
        return float(game_reward) + alignment - step_penalty - life_loss
    # Freeway 过关后小鸡回到底部，此时只给真实得分，不误判为后退。
    if game_reward > 0:
        return float(game_reward)
    delta = int(new_ram[CHICKEN_Y_RAM]) - int(old_ram[CHICKEN_Y_RAM])
    return (-step_penalty
            + progress_reward * np.clip(delta, -PROGRESS_LIMIT, PROGRESS_LIMIT)
            / PROGRESS_LIMIT)


def epsilon_for_episode(episode: int, start: float, end: float,
                        decay_episodes: int) -> float:
    """线性探索率：第 0 回合为 start，第 decay_episodes 回合达到 end。

    例如 start=1、end=0.05、decay_episodes=299 时，第 299 回合为 0.05；
    它与原先训练 300 回合时的公式等价。
    """
    fraction = min(1.0, episode / decay_episodes)
    return start + (end - start) * fraction


def choose_action(q: np.ndarray, state: tuple[int, ...], epsilon: float,
                  rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[-1]))
    values = q[state]
    return int(rng.choice(np.flatnonzero(values == values.max())))


def forced_action(game: str, ram: np.ndarray) -> int | None:
    """Breakout 必须按 FIRE 开球；其余时刻由 Q-learning 选择动作。"""
    if game == GAME_IDS["breakout"]:
        y = int(ram[BREAKOUT_BALL_Y_RAM])
        if y == 0 or y > BREAKOUT_BALL_LOST_Y:
            return 1
    return None


def baseline_action(game: str, ram: np.ndarray) -> int:
    if game == GAME:
        return 1  # Freeway：始终向上
    gap = int(ram[BREAKOUT_BALL_X_RAM]) - int(ram[BREAKOUT_PADDLE_X_RAM])
    return 2 if gap > 5 else (3 if gap < -5 else 0)  # Breakout：追球


def train(episodes: int, steps: int, seed: int, alpha: float, gamma: float,
          model_path: Path, difficulty: int = 0,
          epsilon_start: float = DEFAULT_EPSILON_START,
          epsilon_end: float = DEFAULT_EPSILON_END,
          epsilon_decay_episodes: int | None = None,
          progress_reward: float = DEFAULT_PROGRESS_REWARD,
          step_penalty: float = DEFAULT_STEP_PENALTY,
          game: str = GAME) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = np.zeros(Q_SHAPES[game], dtype=np.float32)
    scores = []
    decay_episodes = (max(1, episodes - 1) if epsilon_decay_episodes is None
                      else epsilon_decay_episodes)
    env = make_env(difficulty=difficulty, game=game)
    try:
        for episode in range(episodes):
            ram, _ = env.reset(seed=seed + episode)
            previous_ram = None
            score = 0.0
            epsilon = epsilon_for_episode(
                episode, epsilon_start, epsilon_end, decay_episodes
            )
            for step in range(steps):
                state = encode_state(ram, game, previous_ram)
                required = forced_action(game, ram)
                action = (required if required is not None
                          else choose_action(q, state, epsilon, rng))
                next_ram, game_reward, terminated, truncated, _ = env.step(action)
                next_state = encode_state(next_ram, game, ram)
                done = terminated or truncated or step + 1 == steps
                reward = training_reward(game_reward, ram, next_ram,
                                         progress_reward, step_penalty, game)
                # 一步 Q-learning；终止状态的未来价值设为 0。
                target = reward + (0 if done else gamma * float(q[next_state].max()))
                q[state + (action,)] += alpha * (target - q[state + (action,)])
                score += float(game_reward)
                previous_ram = ram
                ram = next_ram
                if done:
                    break
            scores.append(score)
            if (episode + 1) % LOG_INTERVAL == 0 or episode + 1 == episodes:
                print(f"训练 {episode + 1:4d}/{episodes} | 最近平均真实得分 "
                      f"{np.mean(scores[-LOG_INTERVAL:]):.2f} | ε={epsilon:.2f}")
    finally:
        env.close()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, q=q, scores=np.asarray(scores),
                        seed=seed, episodes=episodes, steps=steps,
                        difficulty=difficulty, alpha=alpha, gamma=gamma,
                        epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                        epsilon_decay_episodes=decay_episodes,
                        progress_reward=progress_reward,
                        step_penalty=step_penalty, game=game)
    print(f"Q 表已保存：{model_path}")
    return q


def evaluate(q: np.ndarray, episodes: int, steps: int, seed: int,
             render: bool = False, difficulty: int = 0,
             game: str = GAME) -> None:
    """只用游戏原始奖励比较策略；可实时显示 Q-learning 的游戏画面。"""
    rng = np.random.default_rng(seed)
    baseline_name = "始终向上" if game == GAME else "追球基线"
    for name in ("Q-learning", baseline_name, "随机"):
        # ALE 的 human 模式自行显示画面，并按游戏帧率运行。
        env = make_env(render_mode="human" if render and name == "Q-learning"
                       else None, difficulty=difficulty, game=game)
        try:
            scores = []
            for episode in range(episodes):
                ram, _ = env.reset(seed=seed + episode)
                previous_ram = None
                total = 0.0
                for _ in range(steps):
                    required = forced_action(game, ram)
                    if required is not None:
                        action = required
                    elif name == "Q-learning":
                        action = choose_action(
                            q, encode_state(ram, game, previous_ram), 0.0, rng
                        )
                    elif name == baseline_name:
                        action = baseline_action(game, ram)
                    else:
                        action = int(rng.integers(q.shape[-1]))
                    next_ram, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    previous_ram, ram = ram, next_ram
                    if terminated or truncated:
                        break
                scores.append(total)
            print(f"{name:12s} | 平均真实得分 {np.mean(scores):.2f} "
                  f"| 各回合 {scores}")
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--game", choices=tuple(GAME_IDS), default="freeway",
                        help="Atari 案例：freeway（默认）或 breakout")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=1000, help="每回合最多交互步数")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--difficulty", type=int, choices=(0, 1), default=0,
                        help="Freeway 游戏难度：0（默认）或 1")
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
                        default=DEFAULT_PROGRESS_REWARD,
                        help="辅助奖励强度：Freeway 前进 / Breakout 挡板追球")
    parser.add_argument("--step-penalty", type=float,
                        default=DEFAULT_STEP_PENALTY, help="每步固定惩罚")
    parser.add_argument("--model", type=Path, default=None,
                        help="模型路径；默认每个游戏使用各自的 Q 表文件")
    args = parser.parse_args()
    game = GAME_IDS[args.game]
    model_path = args.model or DEFAULT_MODELS[game]
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
        q = train(args.episodes, args.steps, args.seed, args.alpha, args.gamma,
                  model_path, difficulty=args.difficulty,
                  epsilon_start=args.epsilon_start,
                  epsilon_end=args.epsilon_end,
                  epsilon_decay_episodes=args.epsilon_decay_episodes,
                  progress_reward=args.progress_reward,
                  step_penalty=args.step_penalty, game=game)
    else:
        if not model_path.exists():
            parser.error(f"找不到 Q 表：{model_path}；请先运行训练")
        with np.load(model_path, allow_pickle=False) as data:
            q = data["q"]
            saved_game = str(data["game"]) if "game" in data else GAME
            if saved_game != game:
                parser.error(f"模型属于 {saved_game}，当前游戏是 {game}")
            if "difficulty" in data and int(data["difficulty"]) != args.difficulty:
                print(f"提示：模型训练难度为 {int(data['difficulty'])}，"
                      f"当前评估难度为 {args.difficulty}。")
        if q.shape != Q_SHAPES[game]:
            parser.error(f"Q 表形状不匹配：{q.shape}，预期 {Q_SHAPES[game]}")
    evaluate(q, args.eval_episodes, args.steps, args.seed + EVAL_SEED_OFFSET,
             render=args.render, difficulty=args.difficulty, game=game)


if __name__ == "__main__":
    main()
