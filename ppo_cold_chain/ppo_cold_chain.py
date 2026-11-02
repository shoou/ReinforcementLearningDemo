"""PPO 案例：冷链仓库的制冷、电池与配送车充电联合调度。

一天分为 48 个半小时决策时段。环境根据合成电价、光伏、基础负荷和环境温度
计算费用；策略每步从 27 个联合动作中选择一个，并通过动作掩码满足硬约束。
训练、验证和测试使用不同随机种子生成的日场景，费用只用于教学比较。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


# 动作中的电量均指一个半小时内的 kWh，不是瞬时功率 kW。
STEPS = 48                 # 每步半小时，模拟一天
BATTERY_KWH = 18.0
BATTERY_LEVELS = (-3.0, 0.0, 3.0)  # 电池输入/输出电量，kWh；负数为放电
EV_LEVELS = (0.0, 2.0, 4.0)       # 半小时充入车辆的电量，kWh
COOL_KWH = (0.0, 1.8, 3.6)  # 0.0：关闭制冷；低档制冷（半小时耗电 1.8 kWh）；高档制冷（半小时耗电 3.6 kWh）。

# 3 档电池 × 3 档车辆充电 × 3 档制冷，共 27 个离散联合动作。
ACTIONS = [(b, e, c) for b in BATTERY_LEVELS for e in EV_LEVELS for c in range(3)]


class ColdChainEnv:
    """一个有硬约束的合成冷链微电网；同一随机种子对应同一天场景。"""

    def __init__(self, seed: int):
        # 逐日生成 48 个时段的外生序列；价格、光伏、负荷和温度带有随机扰动。
        # arrive/depart 也是半小时时段索引，ev_required 是离站前所需电量。
        rng = np.random.default_rng(seed)
        hour = np.arange(STEPS) / 2
        # 根据时段划分峰谷电价（17:00–21:00 为高峰 1.25，23:00 至次日 7:00 为低谷 0.42，其余为平段 0.78）
        # 加入正态分布随机扰动并截断至不低于 0.25。
        self.price = np.clip(
            np.where((hour >= 17) & (hour < 21), 1.25, np.where((hour < 7) | (hour >= 23), 0.42, 0.78))
            + rng.normal(0, 0.055, STEPS), 0.25, None)
        # 光伏出力 (solar)：基于正弦函数模拟白天的太阳光照曲线，叠加均匀分布与正态分布扰动
        # 通过 np.maximum 确保出力不为负。
        self.solar = np.maximum(0, 7.5 * np.sin(np.pi * (hour - 6) / 12))
        self.solar *= rng.uniform(0.55, 1.05)
        self.solar = np.maximum(0, self.solar + rng.normal(0, 0.55, STEPS))
        # 基础负荷 (base)：使用带有周期性正弦波动和正态噪声的基准负荷曲线，限制最低负荷不低于 1.5。             
        self.base = np.maximum(1.5, 4.0 + 0.7 * np.sin(2 * np.pi * hour / 24)
                               + rng.normal(0, 0.45, STEPS))
        # 环境温度 (ambient)：通过正弦函数模拟全天温度变化趋势，并叠加正态噪声。
        self.ambient = 26 + 5 * np.sin(2 * np.pi * (hour - 8) / 24)
        self.ambient += rng.normal(0, 0.7, STEPS)
        # 到达时间 (arrive)：随机生成在索引 13 到 17 之间的时段，代表车辆到达微电网的时间。
        self.arrive = int(rng.integers(13, 18))
        # 离开时间 (depart)：随机生成在索引 34 到 40 之间的时段，代表车辆离站时间。
        self.depart = int(rng.integers(34, 40))
        # 电量需求 (ev_required)：从 24、28、32 中随机选择一个数值，作为车辆离开前必须满足的充电硬约束目标。
        self.ev_required = float(rng.choice((24, 28, 32)))
        self.reset()

    def reset(self) -> np.ndarray:
        """恢复一天的初始状态；费用分项和轨迹记录也从零开始。"""
        self.t = 0
        self.soc = 9.0
        self.temp = 5.0
        self.ev_remaining = self.ev_required
        self.peak = 0.0
        self.cost = 0.0
        self.energy_cost = 0.0
        self.demand_cost = 0.0
        self.wear_cost = 0.0
        self.violations = 0
        self.history = []
        return self.observe()

    def _ev_present(self) -> bool:
        """车辆只在到站（含）至离站（不含）的时段内可充电。"""
        return self.arrive <= self.t < self.depart

    def _next_temp(self, cool: int) -> float:
        """用一阶温度模型：环境传热升温，制冷档位使库温下降。"""
        return self.temp + 0.07 * (self.ambient[self.t] - self.temp) - 1.3 * cool

    def observe(self) -> np.ndarray:
        """13 维归一化观测：当前状态、当前外生量和未来最多四档的均值。"""
        t = min(self.t, STEPS - 1)
        # 包含当前时段；末尾不足四档时只对仍存在的时段取均值。
        future = slice(t, min(t + 4, STEPS))
        return np.array([
            self.t / STEPS, self.soc / BATTERY_KWH, self.temp / 10,
            self.ev_remaining / 32, float(self._ev_present()),
            max(0, self.depart - self.t) / STEPS,
            self.peak / 20, self.price[t] / 1.5, self.solar[t] / 10,
            self.base[t] / 10, self.ambient[t] / 35,
            np.mean(self.price[future]) / 1.5,
            np.mean(self.solar[future]) / 10,
        ], dtype=np.float32)

    def action_mask(self) -> np.ndarray:
        """过滤违反 SOC、日终储能、车辆离站或库温限制的联合动作。"""
        if self.t >= STEPS:
            raise RuntimeError("回合已结束")
        mask = np.zeros(len(ACTIONS), dtype=bool)
        present = self._ev_present()
        slots_left = self.depart - self.t if present else 0
        for i, (battery, ev, cool) in enumerate(ACTIONS):
            # 充电效率 90%；放电档位直接从电池 SOC 中扣除对应电量。
            new_soc = self.soc + (0.9 * battery if battery > 0 else battery)
            if not 0 <= new_soc <= BATTERY_KWH:
                continue
            # 日终至少恢复至初始电量，避免靠耗尽电池虚增收益。
            if new_soc + 2.7 * (STEPS - self.t - 1) < 9.0 - 1e-6:
                continue
            if present:
                # 当前动作不能超过待充电量，也不能把必需充电留到不足的未来时段。
                if ev > self.ev_remaining + 1e-6:
                    continue
                # 留下的充电窗口必须足以完成车辆任务。
                if self.ev_remaining - ev > 4 * (slots_left - 1) + 1e-6:
                    continue
            elif ev != 0:
                continue
            if not 2.0 <= self._next_temp(cool) <= 8.0:
                continue
            mask[i] = True
        if not mask.any():
            raise RuntimeError(f"无可行动作：t={self.t}, 温度={self.temp:.2f}")
        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """执行一个半小时动作；奖励为负费用除以 10，便于网络优化。"""
        if not 0 <= action < len(ACTIONS) or not self.action_mask()[action]:
            raise ValueError("动作违反电池、温度或配送车出发约束")
        battery, ev, cool = ACTIONS[action]
        t = self.t
        self.soc += 0.9 * battery if battery > 0 else battery
        self.temp = self._next_temp(cool)
        if self._ev_present():
            self.ev_remaining -= ev
        # 放电有 90% 的能量到达交流母线；不允许向电网售电。
        net = self.base[t] + COOL_KWH[cool] + ev + (battery if battery > 0 else 0.9 * battery) - self.solar[t]
        import_kwh = max(0.0, net)
        energy = import_kwh * self.price[t]
        # 只对本时段刷新了历史峰值的部分计费；全天累加等价于 2×最终峰值。
        demand = 2.0 * max(0.0, import_kwh - self.peak)  # 单日峰值电量代理费用
        wear = 0.035 * abs(battery)
        self.peak = max(self.peak, import_kwh)
        self.energy_cost += energy
        self.demand_cost += demand
        self.wear_cost += wear
        step_cost = energy + demand + wear
        self.cost += step_cost
        self.t += 1
        if self.t == self.depart and self.ev_remaining > 1e-5:
            self.violations += 1
        record = {
            "step": self.t, "price": float(self.price[t]), "solar": float(self.solar[t]),
            "base": float(self.base[t]), "import_kwh": float(import_kwh),
            "battery_kwh": battery, "ev_kwh": ev, "cool": cool,
            "soc": float(self.soc), "temp": float(self.temp),
            "ev_remaining": float(self.ev_remaining), "cost": float(step_cost),
        }
        self.history.append(record)
        done = self.t == STEPS
        return self.observe(), -step_cost / 10.0, done, record

    def summary(self) -> dict:
        """汇总一天的成本、约束结果和每时段记录，供比较与绘图。"""
        return {
            "total_cost": self.cost, "energy_cost": self.energy_cost,
            "demand_cost": self.demand_cost, "wear_cost": self.wear_cost,
            "peak_kwh": self.peak, "ev_unfilled_kwh": max(0, self.ev_remaining),
            "violations": self.violations, "history": self.history,
        }


class Policy(nn.Module):
    """共享特征网络加两个输出头: Actor 给动作 logits, Critic 估计状态价值。"""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(13, 128), nn.Tanh(),
                                  nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, len(ACTIONS))
        self.critic = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


def distribution(logits: torch.Tensor, masks: torch.Tensor) -> torch.distributions.Categorical:
    """把非法动作的概率压到近零，构造仅在可行动作上采样的分类分布。"""
    return torch.distributions.Categorical(logits=logits.masked_fill(~masks, -1e9))


def collect(model: Policy, seeds: list[int]) -> tuple[dict[str, torch.Tensor], list[float]]:
    """用当前策略完成若干整天轨迹，保存 PPO 更新所需的旧 logp 和价值。"""
    rows = {k: [] for k in ("obs", "mask", "action", "logp", "value", "reward", "done")}
    costs = []
    for seed in seeds:
        env = ColdChainEnv(seed)
        obs = env.reset()
        for _ in range(STEPS):
            mask = env.action_mask()
            with torch.no_grad():
                # 采样策略而非选最大概率动作，以便收集探索数据。
                logits, value = model(torch.from_numpy(obs).unsqueeze(0))
                dist = distribution(logits, torch.from_numpy(mask).unsqueeze(0))
                action = dist.sample()
                logp = dist.log_prob(action)
            next_obs, reward, done, _ = env.step(int(action.item()))
            for key, val in zip(rows, (obs, mask, int(action), float(logp),
                                       float(value), reward, done)):
                rows[key].append(val)
            obs = next_obs
        costs.append(env.cost)
    # 按“天”顺序拼接转移；done 标志使 GAE 在不同天之间停止传播。
    batch = {
        "obs": torch.from_numpy(np.stack(rows["obs"])),
        "mask": torch.from_numpy(np.stack(rows["mask"])),
        "action": torch.tensor(rows["action"], dtype=torch.long),
        "logp": torch.tensor(rows["logp"], dtype=torch.float32),
        "value": torch.tensor(rows["value"], dtype=torch.float32),
        "reward": torch.tensor(rows["reward"], dtype=torch.float32),
        "done": torch.tensor(rows["done"], dtype=torch.float32),
    }
    return batch, costs


def gae_returns(batch: dict[str, torch.Tensor], gamma=1.0, lam=0.97) -> tuple[torch.Tensor, torch.Tensor]:
    """逆序计算 GAE 优势和 critic 回归目标; gamma=1 对日内费用不折扣。"""
    rewards, values, dones = batch["reward"], batch["value"], batch["done"]
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for i in reversed(range(len(rewards))):
        # 最后一时段或任一日终的后续价值都乘 nonterminal=0，不跨日自举。
        next_value = 0.0 if i == len(rewards) - 1 else float(values[i + 1])
        nonterminal = 1.0 - float(dones[i])
        delta = float(rewards[i]) + gamma * next_value * nonterminal - float(values[i])
        gae = delta + gamma * lam * nonterminal * gae
        advantages[i] = gae
    return advantages, advantages + values


def train(updates: int, seed: int, envs_per_update: int) -> tuple[Policy, list[float], dict]:
    """迭代采样与 PPO 更新，并按固定验证日成本选择最终检查点。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    model = Policy()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    curve = []
    rng = np.random.default_rng(seed)
    # 固定验证日只用于挑选检查点，最终的 100 个测试日不参与训练和选模。
    validation_seeds = list(range(1_000_000, 1_032_000, 1000))
    best_cost = float("inf")
    best_state = None
    best_update = 0
    for update in range(updates):
        # 训练后段逐渐减小学习率和熵奖励，让策略从探索转向稳定调度。
        progress = update / max(1, updates - 1)
        for group in optimizer.param_groups:
            group["lr"] = 3e-4 * (1 - 0.9 * progress)
        entropy_coef = 0.01 * (1 - progress) + 0.001 * progress
        seeds = [int(x) for x in rng.integers(0, 1_000_000_000, envs_per_update)]
        batch, costs = collect(model, seeds)
        # 目标是一天的总费用；不对晚间费用额外折扣。
        advantages, returns = gae_returns(batch)
        # 优势标准化改善数值尺度，不改变同一批样本的优劣次序。
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = len(batch["obs"])
        # 同一批轨迹做四轮小批量更新；ratio 使用采样时保存的旧策略 logp。
        for _ in range(4):
            for idx in torch.randperm(n).split(128):
                logits, values = model(batch["obs"][idx])
                dist = distribution(logits, batch["mask"][idx])
                ratio = torch.exp(dist.log_prob(batch["action"][idx]) - batch["logp"][idx])
                a = advantages[idx]
                # PPO 裁剪目标把策略概率比限制在约 [0.8, 1.2] 的有效改进区间。
                policy_loss = -torch.minimum(ratio * a, torch.clamp(ratio, 0.8, 1.2) * a).mean()
                # critic 预测 GAE 构造的回报；熵项鼓励分布保留一定探索。
                value_loss = 0.5 * (values - returns[idx]).square().mean()
                loss = policy_loss + 0.5 * value_loss - entropy_coef * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        curve.append(float(np.mean(costs)))
        if (update + 1) % 20 == 0 or update + 1 == updates:
            # 验证日不参与梯度更新；只用于在已训练的多个检查点中选一个。
            validation_cost = float(np.mean([
                rollout(model, s)["total_cost"] for s in validation_seeds
            ]))
            if validation_cost < best_cost:
                best_cost = validation_cost
                best_state = copy.deepcopy(model.state_dict())
                best_update = update + 1
            print(f"PPO 更新 {update + 1:3d}/{updates} | 本批平均日成本 {curve[-1]:.1f}"
                  f" | 验证成本 {validation_cost:.1f} | 最佳第 {best_update} 轮")
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, curve, {"best_update": best_update, "validation_cost": best_cost,
                          "validation_days": len(validation_seeds)}


def rollout(model: Policy | None, seed: int, baseline: str = "rule") -> dict:
    """在一个完整测试日确定性执行 PPO 或规则/班表基线。

    规则基线先按当前库温、光伏、电价和车辆剩余充电时间设定目标档位，
    再从满足电池、库温和离站约束的动作中，选加权偏差最小的联合动作。
    """
    env = ColdChainEnv(seed)
    obs = env.reset()
    for _ in range(STEPS):
        mask = env.action_mask()
        if model is not None:
            with torch.no_grad():
                # 测试使用最大概率动作，避免评估噪声来自策略采样。
                logits, _ = model(torch.from_numpy(obs).unsqueeze(0))
                action = int(distribution(logits, torch.from_numpy(mask).unsqueeze(0)).probs.argmax())
        else:
            # 所有对照方案都经过相同的硬约束掩码。
            def score(i: int) -> float:
                battery, ev, cool = ACTIONS[i]
                if baseline == "fixed":
                    # 班表基线偏好中档制冷、车辆不充电、电池不动作；当剩余充电
                    # 时段不足时，动作掩码会迫使它为车辆充电。
                    return abs(cool - 1) * 2 + ev * 0.5 + abs(battery) * 0.3
                # 制冷：库温高于 6.3°C 选高档，高于 4.2°C 选中档，
                # 否则倾向关闭；最终档位仍须保证下一时段库温在 2–8°C。
                desired_cool = 2 if env.temp > 6.3 else (1 if env.temp > 4.2 else 0)
                present = env._ev_present()
                slots = max(1, env.depart - env.t)
                # 车辆：用剩余需求除以可充电时段，估计每时段至少应充多少。
                # 光伏大于 3 kWh 或电价低于 0.6 时优先充 4 kWh，
                # 其他时段倾向充 2 kWh；若平均需求超过 2 kWh，也充 4 kWh。
                # 车辆不在场时，动作掩码只允许 ev=0。
                needed = env.ev_remaining / slots if present else 0
                desired_ev = 4 if present and (env.solar[env.t] > 3 or env.price[env.t] < 0.6) else 2
                if needed > 2:
                    desired_ev = 4
                # 电池：电价高于 1.0 时放电 3 kWh；低于 0.55 且 SOC
                # 小于 14 kWh 时充电 3 kWh；其余时段不动作。
                desired_battery = -3 if env.price[env.t] > 1.0 else (
                    3 if env.price[env.t] < 0.55 and env.soc < 14 else 0)
                # 偏差权重依次为制冷 2、车辆 0.4、电池 0.2；仅比较合法动作。
                return (abs(cool - desired_cool) * 2 + abs(ev - desired_ev) * 0.4
                        + abs(battery - desired_battery) * 0.2)
            action = min(np.flatnonzero(mask), key=score)
        obs, _, _, _ = env.step(action)
    return env.summary()


def evaluate(model: Policy, seeds: list[int]) -> tuple[dict, dict]:
    """三个方案在同一组日场景比较，返回平均指标及首日详细轨迹。"""
    learned = [rollout(model, s) for s in seeds]
    rule = [rollout(None, s, "rule") for s in seeds]
    fixed = [rollout(None, s, "fixed") for s in seeds]
    keys = ("total_cost", "energy_cost", "demand_cost", "wear_cost", "peak_kwh",
            "ev_unfilled_kwh", "violations")
    def aggregate(results: list[dict]) -> dict:
        return {key: float(np.mean([r[key] for r in results])) for key in keys}
    return ({"ppo": aggregate(learned), "rule": aggregate(rule),
             "fixed": aggregate(fixed)},
            {"ppo": learned[0], "rule": rule[0], "fixed": fixed[0]})


def save_chart(curve: list[float], examples: dict, stats: dict, output: Path) -> None:
    """输出训练成本、方案对比及一个测试日的取电和库温图。"""
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    # 训练曲线用最多 15 次更新的简单移动平均，保留原曲线作背景。
    smooth = np.convolve(curve, np.ones(min(15, len(curve))) / min(15, len(curve)), mode="valid")
    axes[0, 0].plot(curve, alpha=0.3)
    axes[0, 0].plot(np.arange(len(smooth)) + len(curve) - len(smooth), smooth)
    axes[0, 0].set(title="训练日成本", xlabel="PPO 更新次数", ylabel="模拟费用")
    axes[0, 1].bar(["班表", "规则", "PPO"],
                   [stats[k]["total_cost"] for k in ("fixed", "rule", "ppo")])
    axes[0, 1].set(title="相同测试日的平均成本", ylabel="模拟费用")
    for name, result in (("班表", examples["fixed"]), ("规则", examples["rule"]),
                         ("PPO", examples["ppo"])):
        h = result["history"]
        x = np.arange(STEPS) / 2
        axes[1, 0].plot(x, [v["import_kwh"] for v in h], label=name)
        axes[1, 1].plot(x, [v["temp"] for v in h], label=name)
    axes[1, 0].set(title="电网取电", xlabel="小时", ylabel="每半小时 kWh")
    axes[1, 1].set(title="库温", xlabel="小时", ylabel="摄氏度", ylim=(1.5, 8.5))
    for ax in axes[1]:
        ax.legend()
    fig.savefig(output / "training_and_dispatch.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """训练、选择检查点，在独立测试日比较基线并保存模型与图表。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("ppo_output"))
    args = parser.parse_args()
    if args.updates < 1 or args.envs < 1:
        parser.error("--updates 和 --envs 必须大于零")
    args.output.mkdir(parents=True, exist_ok=True)
    model, curve, selection = train(args.updates, args.seed, args.envs)
    # 这 100 个种子既不用于训练，也不用于选择最佳更新轮次。
    test_seeds = list(range(2_000_000, 2_100_000, 1000))
    stats, examples = evaluate(model, test_seeds)
    stats.update({"test_days": len(test_seeds), "updates": args.updates, "seed": args.seed,
                  **selection})
    torch.save(model.state_dict(), args.output / "ppo_model.pt")
    (args.output / "metrics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    save_chart(curve, examples, stats, args.output)
    print("100 个测试日平均成本：" + "，".join(
        f"{name} {stats[key]['total_cost']:.2f}"
        for name, key in (("班表", "fixed"), ("规则", "rule"), ("PPO", "ppo"))))
    print("配送违约：" + "，".join(
        f"{name} {stats[key]['violations']:.0f}"
        for name, key in (("班表", "fixed"), ("规则", "rule"), ("PPO", "ppo"))))
    print(f"结果保存至 {args.output}")


if __name__ == "__main__":
    main()
