"""DQN for measured-pH correction; actions are bounded dispense requests."""
from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
from torch import nn

from compiler import Protocol
from twin import BufferModel, make_sample


ACTIONS = (("stop", 0.0), ("citric_acid", 0.1), ("citric_acid", 0.25),
           ("citric_acid", 0.5), ("trisodium_citrate", 0.1),
           ("trisodium_citrate", 0.25), ("trisodium_citrate", 0.5))


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(),
                                    nn.Linear(64, len(ACTIONS)))

    def forward(self, x):
        return self.layers(x)


def observe(ph: float, protocol: Protocol, reserve_ml: float, ratio: float) -> np.ndarray:
    return np.array([(ph - protocol.target_ph) * 2,
                     reserve_ml / protocol.reserve_ml,
                     np.log(max(ratio, 1e-6)) / 2], dtype=np.float32)


def action_mask(reserve_ml: float) -> np.ndarray:
    return np.array([name == "stop" or volume <= reserve_ml + 1e-7
                     for name, volume in ACTIONS], dtype=bool)


def choose(model: QNet, obs: np.ndarray, mask: np.ndarray) -> int:
    return int(np.argmax(np.where(mask, q_values(model, obs), -1e9)))


def q_values(model: QNet, obs: np.ndarray) -> list[float]:
    with torch.no_grad():
        values = model(torch.from_numpy(obs).unsqueeze(0))[0].numpy()
    return [round(float(value), 4) for value in values]


def train(protocol: Protocol, episodes: int = 900, seed: int = 42,
          model: BufferModel | None = None) -> tuple[QNet, dict]:
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(1)
    policy, target = QNet(), QNet()
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=8e-4)
    memory = deque(maxlen=16000)
    losses = []
    successes = []
    episode_returns = []
    curve = []
    updates = 0

    for episode in range(episodes):
        acid = float(rng.uniform(4.5, 8.0)) * protocol.final_volume_ml / 20
        base = float(rng.uniform(6.0, 11.0)) * protocol.final_volume_ml / 20
        if acid + base > protocol.final_volume_ml - protocol.reserve_ml:
            base = protocol.final_volume_ml - protocol.reserve_ml - acid
        sample = make_sample(protocol, acid, base, protocol.mix_seconds,
                             rng, float(rng.uniform(-0.10, 0.10)), model=model)
        ph = sample.measure_ph()
        total_reward = 0.0
        for step in range(5):
            obs = observe(ph, protocol, sample.reserve_ml, sample.base_ml / sample.acid_ml)
            mask = action_mask(sample.reserve_ml)
            epsilon = max(0.06, 0.7 * (1 - episode / episodes))
            valid = np.flatnonzero(mask)
            act = int(rng.choice(valid)) if rng.random() < epsilon else choose(policy, obs, mask)
            reagent, amount = ACTIONS[act]
            old_error = abs(ph - protocol.target_ph)
            if reagent != "stop":
                sample.add(reagent, amount)
                sample.mix(protocol.mix_seconds)
                ph = sample.measure_ph()
            new_error = abs(ph - protocol.target_ph)
            done = reagent == "stop" or step == 4 or sample.reserve_ml < 0.099
            reward = 8 * (old_error - new_error) - 0.04 * amount
            if reagent != "stop":
                reward -= 0.005 * protocol.mix_seconds
            if done:
                reward += 1.2 if new_error <= protocol.ph_tolerance else -3 * new_error
            total_reward += reward
            next_obs = observe(ph, protocol, sample.reserve_ml, sample.base_ml / sample.acid_ml)
            next_mask = action_mask(sample.reserve_ml)
            memory.append((obs, act, reward, next_obs, next_mask, done))
            if len(memory) >= 64:
                batch = random.sample(memory, 64)
                o = torch.tensor(np.stack([x[0] for x in batch]))
                a = torch.tensor([x[1] for x in batch], dtype=torch.long)
                r = torch.tensor([x[2] for x in batch], dtype=torch.float32)
                no = torch.tensor(np.stack([x[3] for x in batch]))
                nm = torch.tensor(np.stack([x[4] for x in batch]))
                d = torch.tensor([x[5] for x in batch], dtype=torch.bool)
                with torch.no_grad():
                    future = target(no).masked_fill(~nm, -1e9).max(1).values
                    expected = r + 0.94 * future * (~d)
                current = policy(o).gather(1, a[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(current, expected)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.item()))
                updates += 1
                if updates % 100 == 0:
                    target.load_state_dict(policy.state_dict())
            if done:
                successes.append(new_error <= protocol.ph_tolerance)
                break
        episode_returns.append(total_reward)
        if (episode + 1) % 50 == 0 or episode + 1 == episodes:
            curve.append({"episode": episode + 1,
                          "success_rate": round(float(np.mean(successes[-50:])), 3),
                          "mean_return": round(float(np.mean(episode_returns[-50:])), 3)})
    target.load_state_dict(policy.state_dict())
    return policy, {"episodes": episodes, "training_success_last_100":
                    round(float(np.mean(successes[-100:])), 3),
                    "mean_td_loss_last_100": round(float(np.mean(losses[-100:])), 4),
                    "curve": curve}

