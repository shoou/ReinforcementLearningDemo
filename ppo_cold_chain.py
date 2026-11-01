"""PPO案例：冷链仓库的制冷、电池与配送车充电联合调度。

所有负荷、天气和价格均为合成数据，仅供课堂实验。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


STEPS = 48                 # 每步半小时，模拟一天
BATTERY_KWH = 18.0
BATTERY_LEVELS = (-3.0, 0.0, 3.0)  # 电池输入/输出电量，kWh；负数为放电
EV_LEVELS = (0.0, 2.0, 4.0)       # 半小时充入车辆的电量，kWh
COOL_KWH = (0.0, 1.8, 3.6)
ACTIONS = [(b, e, c) for b in BATTERY_LEVELS for e in EV_LEVELS for c in range(3)]


class ColdChainEnv:
    """一个有硬约束的合成冷链微电网。动作掩码只依赖当前可观测状态。"""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        hour = np.arange(STEPS) / 2
        self.price = np.clip(
            np.where((hour >= 17) & (hour < 21), 1.25,
                     np.where((hour < 7) | (hour >= 23), 0.42, 0.78))
            + rng.normal(0, 0.055, STEPS), 0.25, None)
        self.solar = np.maximum(0, 7.5 * np.sin(np.pi * (hour - 6) / 12))
        self.solar *= rng.uniform(0.55, 1.05)
        self.solar = np.maximum(0, self.solar + rng.normal(0, 0.55, STEPS))
        self.base = np.maximum(1.5, 4.0 + 0.7 * np.sin(2 * np.pi * hour / 24)
                               + rng.normal(0, 0.45, STEPS))
        self.ambient = 26 + 5 * np.sin(2 * np.pi * (hour - 8) / 24)
        self.ambient += rng.normal(0, 0.7, STEPS)
        self.arrive = int(rng.integers(13, 18))
        self.depart = int(rng.integers(34, 40))
        self.ev_required = float(rng.choice((24, 28, 32)))
        self.reset()

    def reset(self) -> np.ndarray:
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
        return self.arrive <= self.t < self.depart

    def _next_temp(self, cool: int) -> float:
        return self.temp + 0.07 * (self.ambient[self.t] - self.temp) - 1.3 * cool

    def observe(self) -> np.ndarray:
        t = min(self.t, STEPS - 1)
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
        if self.t >= STEPS:
            raise RuntimeError("回合已结束")
        mask = np.zeros(len(ACTIONS), dtype=bool)
        present = self._ev_present()
        slots_left = self.depart - self.t if present else 0
        for i, (battery, ev, cool) in enumerate(ACTIONS):
            new_soc = self.soc + (0.9 * battery if battery > 0 else battery)
            if not 0 <= new_soc <= BATTERY_KWH:
                continue
            # 日终至少恢复至初始电量，避免靠耗尽电池虚增收益。
            if new_soc + 2.7 * (STEPS - self.t - 1) < 9.0 - 1e-6:
                continue
            if present:
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
        return {
            "total_cost": self.cost, "energy_cost": self.energy_cost,
            "demand_cost": self.demand_cost, "wear_cost": self.wear_cost,
            "peak_kwh": self.peak, "ev_unfilled_kwh": max(0, self.ev_remaining),
            "violations": self.violations, "history": self.history,
        }


class Policy(nn.Module):
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
    return torch.distributions.Categorical(logits=logits.masked_fill(~masks, -1e9))


def collect(model: Policy, seeds: list[int]) -> tuple[dict[str, torch.Tensor], list[float]]:
    rows = {k: [] for k in ("obs", "mask", "action", "logp", "value", "reward", "done")}
    costs = []
    for seed in seeds:
        env = ColdChainEnv(seed)
        obs = env.reset()
        for _ in range(STEPS):
            mask = env.action_mask()
            with torch.no_grad():
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


def gae_returns(batch: dict[str, torch.Tensor], gamma=0.99, lam=0.95) -> tuple[torch.Tensor, torch.Tensor]:
    rewards, values, dones = batch["reward"], batch["value"], batch["done"]
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for i in reversed(range(len(rewards))):
        next_value = 0.0 if i == len(rewards) - 1 else float(values[i + 1])
        nonterminal = 1.0 - float(dones[i])
        delta = float(rewards[i]) + gamma * next_value * nonterminal - float(values[i])
        gae = delta + gamma * lam * nonterminal * gae
        advantages[i] = gae
    return advantages, advantages + values


def train(updates: int, seed: int, envs_per_update: int) -> tuple[Policy, list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    model = Policy()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    curve = []
    rng = np.random.default_rng(seed)
    for update in range(updates):
        seeds = [int(x) for x in rng.integers(0, 1_000_000_000, envs_per_update)]
        batch, costs = collect(model, seeds)
        advantages, returns = gae_returns(batch)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = len(batch["obs"])
        for _ in range(4):
            for idx in torch.randperm(n).split(128):
                logits, values = model(batch["obs"][idx])
                dist = distribution(logits, batch["mask"][idx])
                ratio = torch.exp(dist.log_prob(batch["action"][idx]) - batch["logp"][idx])
                a = advantages[idx]
                policy_loss = -torch.minimum(ratio * a, torch.clamp(ratio, 0.8, 1.2) * a).mean()
                value_loss = 0.5 * (values - returns[idx]).square().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
        curve.append(float(np.mean(costs)))
        if (update + 1) % 20 == 0 or update + 1 == updates:
            print(f"PPO 更新 {update + 1:3d}/{updates} | 本批平均日成本 {curve[-1]:.1f}")
    return model, curve


def rollout(model: Policy | None, seed: int, baseline: str = "rule") -> dict:
    env = ColdChainEnv(seed)
    obs = env.reset()
    for _ in range(STEPS):
        mask = env.action_mask()
        if model is not None:
            with torch.no_grad():
                logits, _ = model(torch.from_numpy(obs).unsqueeze(0))
                action = int(distribution(logits, torch.from_numpy(mask).unsqueeze(0)).probs.argmax())
        else:
            # 所有对照方案都经过相同的硬约束掩码。
            def score(i: int) -> float:
                battery, ev, cool = ACTIONS[i]
                if baseline == "fixed":
                    # 班表策略：尽量保持中档制冷，临近出车才充电，不主动用电池。
                    return abs(cool - 1) * 2 + ev * 0.5 + abs(battery) * 0.3
                # 手写规则：温度接近上限时加强制冷；有太阳或价格低时优先充车；
                # 高价时放电，低价时充电。
                desired_cool = 2 if env.temp > 6.3 else (1 if env.temp > 4.2 else 0)
                present = env._ev_present()
                slots = max(1, env.depart - env.t)
                needed = env.ev_remaining / slots if present else 0
                desired_ev = 4 if present and (env.solar[env.t] > 3 or env.price[env.t] < 0.6) else 2
                if needed > 2:
                    desired_ev = 4
                desired_battery = -3 if env.price[env.t] > 1.0 else (
                    3 if env.price[env.t] < 0.55 and env.soc < 14 else 0)
                return (abs(cool - desired_cool) * 2 + abs(ev - desired_ev) * 0.4
                        + abs(battery - desired_battery) * 0.2)
            action = min(np.flatnonzero(mask), key=score)
        obs, _, _, _ = env.step(action)
    return env.summary()


def evaluate(model: Policy, seeds: list[int]) -> tuple[dict, dict]:
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
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("ppo_output"))
    args = parser.parse_args()
    if args.updates < 1 or args.envs < 1:
        parser.error("--updates 和 --envs 必须大于零")
    args.output.mkdir(parents=True, exist_ok=True)
    model, curve = train(args.updates, args.seed, args.envs)
    test_seeds = list(range(2_000_000, 2_100_000, 1000))
    stats, examples = evaluate(model, test_seeds)
    stats.update({"test_days": len(test_seeds), "updates": args.updates, "seed": args.seed})
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
