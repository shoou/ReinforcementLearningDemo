"""Paired synthetic stress tests for dosing policies and stronger baselines."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from compiler import Protocol
from dose_rl import ACTIONS, action_mask, choose, observe
from twin import BufferModel, Sample, make_sample


@dataclass(frozen=True)
class Scenario:
    name: str
    acid_range: tuple[float, float]
    base_range: tuple[float, float]
    model_shift: bool = False
    transfer_cv: float = 0.004
    sensor_sd_ph: float = 0.0025
    bias_sd_ph: float = 0.008


SCENARIOS = (
    Scenario("in_distribution", (4.5, 8.0), (6.0, 11.0)),
    Scenario("recipe_edge", (4.0, 4.5), (6.0, 11.0)),
    Scenario("model_shift", (4.5, 8.0), (6.0, 11.0), True,
             transfer_cv=0.012, sensor_sd_ph=0.004, bias_sd_ph=0.025),
    Scenario("combined_shift", (4.0, 8.5), (5.5, 11.0), True,
             transfer_cv=0.012, sensor_sd_ph=0.004, bias_sd_ph=0.025),
)


@dataclass(frozen=True)
class ShiftedBufferModel(BufferModel):
    """Hidden curvature unavailable to the learned or model-based policy."""
    nonlinear_coefficient: float = 0.0

    def predict(self, acid_mmol: float, base_mmol: float, mix_seconds: int) -> float:
        ratio = max(base_mmol, 1e-6) / max(acid_mmol, 1e-6)
        return (super().predict(acid_mmol, base_mmol, mix_seconds)
                + self.nonlinear_coefficient * math.log10(ratio) ** 2)


def _fixed_step_action(ph: float, protocol: Protocol, reserve_ml: float) -> int:
    if abs(ph - protocol.target_ph) <= protocol.ph_tolerance or reserve_ml < 0.1:
        return 0
    reagent = "citric_acid" if ph > protocol.target_ph else "trisodium_citrate"
    return next(index for index, (name, volume) in enumerate(ACTIONS)
                if name == reagent and volume == 0.1)


def _model_based_action(ph: float, protocol: Protocol, reserve_ml: float,
                        sample: Sample, model: BufferModel) -> int:
    """One-step lookahead using measured pH and an imperfect nominal model."""
    if abs(ph - protocol.target_ph) <= protocol.ph_tolerance:
        return 0
    acid_mmol = sample.acid_ml * protocol.acid_stock_m
    base_mmol = sample.base_ml * protocol.base_stock_m
    current = model.predict(acid_mmol, base_mmol, sample.mix_seconds)
    choices = [(abs(ph - protocol.target_ph), 0)]
    for index, (name, amount) in enumerate(ACTIONS[1:], start=1):
        if amount > reserve_ml + 1e-7:
            continue
        next_acid = acid_mmol + (amount * protocol.acid_stock_m if name == "citric_acid" else 0)
        next_base = base_mmol + (amount * protocol.base_stock_m if name == "trisodium_citrate" else 0)
        projected = ph + model.predict(next_acid, next_base, sample.mix_seconds) - current
        choices.append((abs(projected - protocol.target_ph) + 0.003 * amount, index))
    return min(choices)[1]


def _trial_setup(protocol: Protocol, model: BufferModel, trial_seed: int,
                 scenario: Scenario) -> tuple[float, float, float, BufferModel]:
    design = np.random.default_rng(trial_seed)
    scale = protocol.final_volume_ml / 20
    acid = float(design.uniform(*scenario.acid_range)) * scale
    base = float(design.uniform(*scenario.base_range)) * scale
    if acid + base > protocol.final_volume_ml - protocol.reserve_ml:
        base = protocol.final_volume_ml - protocol.reserve_ml - acid
    bias = float(design.normal(0.0, scenario.bias_sd_ph))
    if scenario.model_shift:
        hidden = ShiftedBufferModel(
            intercept=model.intercept + float(design.uniform(-0.08, 0.08)),
            log_ratio_coefficient=model.log_ratio_coefficient * float(design.uniform(0.85, 1.15)),
            mixing_coefficient=model.mixing_coefficient * float(design.uniform(0.5, 1.5)),
            nonlinear_coefficient=float(design.uniform(-0.12, 0.12)),
        )
    else:
        hidden = model
    return acid, base, bias, hidden


def evaluate_dose_policies(protocol: Protocol, policy, model: BufferModel,
                           seed: int, trials: int = 120,
                           scenario: Scenario = SCENARIOS[0]) -> dict:
    """Same initial state and noise seed for every policy on each trial."""
    if trials < 2:
        raise ValueError("至少需要两次独立评估")
    master = np.random.default_rng(seed)
    seeds = master.integers(0, 2**31 - 1, size=trials)
    names = ("dqn", "fixed_step", "model_based", "no_adjustment")
    records = {name: [] for name in names}
    for trial_seed in seeds:
        acid, base, bias, hidden = _trial_setup(protocol, model, int(trial_seed), scenario)
        for name in names:
            rng = np.random.default_rng(int(trial_seed) + 1)
            sample = make_sample(protocol, acid, base, protocol.mix_seconds, rng,
                                 bias, model=hidden, transfer_cv=scenario.transfer_cv,
                                 sensor_sd_ph=scenario.sensor_sd_ph)
            initial_ph = None
            ph = None
            added_ml = 0.0
            steps = 0
            failure = None
            try:
                ph = sample.measure_ph()
                initial_ph = ph
                for _ in range(4):
                    reserve = max(0.0, min(sample.reserve_ml,
                                           protocol.reserve_ml - added_ml))
                    if name == "no_adjustment":
                        action_id = 0
                    elif name == "fixed_step":
                        action_id = _fixed_step_action(ph, protocol, reserve)
                    elif name == "model_based":
                        action_id = _model_based_action(ph, protocol, reserve, sample, model)
                    else:
                        obs = observe(ph, protocol, reserve,
                                      sample.base_ml / sample.acid_ml)
                        action_id = choose(policy, obs, action_mask(reserve))
                    reagent, amount = ACTIONS[action_id]
                    if reagent == "stop":
                        break
                    sample.add(reagent, amount)
                    sample.mix(protocol.mix_seconds)
                    ph = sample.measure_ph()
                    added_ml += amount
                    steps += 1
            except (RuntimeError, ValueError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
            records[name].append({
                "initial_error": abs(initial_ph - protocol.target_ph)
                if initial_ph is not None else None,
                "final_error": abs(ph - protocol.target_ph)
                if ph is not None and failure is None else None,
                "hit": bool(ph is not None and failure is None and
                            abs(ph - protocol.target_ph) <= protocol.ph_tolerance),
                "added_ml": added_ml, "steps": steps, "failure": failure,
            })

    summary = {}
    for name, rows in records.items():
        final_errors = [r["final_error"] for r in rows if r["final_error"] is not None]
        initial_errors = [r["initial_error"] for r in rows if r["initial_error"] is not None]
        hits = np.array([r["hit"] for r in rows], dtype=float)
        rate = float(np.mean(hits))
        half_width = 1.96 * float(np.sqrt(rate * (1 - rate) / trials))
        summary[name] = {
            "trials": trials,
            "mean_initial_abs_error_ph": round(float(np.mean(initial_errors)), 4)
            if initial_errors else None,
            "mean_final_abs_error_ph": round(float(np.mean(final_errors)), 4)
            if final_errors else None,
            "tolerance_hit_rate": round(rate, 3),
            "hit_rate_ci95": [round(max(0, rate - half_width), 3),
                              round(min(1, rate + half_width), 3)],
            "failed_trials": sum(r["failure"] is not None for r in rows),
            "mean_added_ml": round(float(np.mean([r["added_ml"] for r in rows])), 4),
            "mean_adjustment_steps": round(float(np.mean([r["steps"] for r in rows])), 3),
        }
    paired = {}
    for baseline in ("fixed_step", "model_based", "no_adjustment"):
        delta = np.array([int(d["hit"]) - int(b["hit"])
                          for d, b in zip(records["dqn"], records[baseline])])
        rng = np.random.default_rng(seed + len(baseline))
        sampled = rng.integers(0, trials, size=(1000, trials))
        bounds = np.percentile(delta[sampled].mean(axis=1), [2.5, 97.5])
        paired[baseline] = {
            "dqn_hit_rate_difference": round(float(np.mean(delta)), 3),
            "bootstrap_ci95": [round(float(bounds[0]), 3),
                               round(float(bounds[1]), 3)],
        }
    return {"source": "synthetic_held_out", "scenario": scenario.name,
            "seed": seed, "policies": summary, "paired_comparisons": paired}


def evaluate_suite(protocol: Protocol, policies: list, model: BufferModel,
                   seed: int, trials: int = 120) -> dict:
    """Report each scenario across independent policy training seeds."""
    if not policies:
        raise ValueError("至少需要一组已训练策略")
    result = {}
    for scenario in SCENARIOS:
        runs = [evaluate_dose_policies(protocol, policy, model, seed, trials,
                                       scenario=scenario) for policy in policies]
        aggregate = {}
        for name in runs[0]["policies"]:
            rates = [run["policies"][name]["tolerance_hit_rate"] for run in runs]
            errors = [run["policies"][name]["mean_final_abs_error_ph"] for run in runs]
            valid_errors = [error for error in errors if error is not None]
            aggregate[name] = {
                "mean_hit_rate": round(float(np.mean(rates)), 3),
                "min_hit_rate": round(float(np.min(rates)), 3),
                "max_hit_rate": round(float(np.max(rates)), 3),
                "mean_abs_error_ph": round(float(np.mean(valid_errors)), 4)
                if valid_errors else None,
                "mean_added_ml": round(float(np.mean(
                    [run["policies"][name]["mean_added_ml"] for run in runs])), 4),
            }
        result[scenario.name] = {"aggregate": aggregate, "per_training_seed": runs}
    return {"source": "synthetic_stress_test", "trials_per_scenario": trials,
            "training_seeds": len(policies), "scenarios": result}
