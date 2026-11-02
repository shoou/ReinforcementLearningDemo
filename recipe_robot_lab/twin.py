"""Calibratable teaching surrogate for a citrate buffer, not a wet-lab predictor."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from compiler import Protocol


@dataclass(frozen=True)
class BufferModel:
    intercept: float = 4.76
    log_ratio_coefficient: float = 1.0
    mixing_coefficient: float = 0.06

    def predict(self, acid_mmol: float, base_mmol: float, mix_seconds: int) -> float:
        ratio = max(base_mmol, 1e-6) / max(acid_mmol, 1e-6)
        mixing_feature = math.exp(-mix_seconds / 9.0) * math.sin(ratio * 5)
        return (self.intercept + self.log_ratio_coefficient * math.log10(ratio)
                + self.mixing_coefficient * mixing_feature)


@dataclass
class Sample:
    protocol: Protocol
    acid_ml: float
    base_ml: float
    water_ml: float
    mix_seconds: int
    instrument_bias: float
    rng: np.random.Generator
    model: BufferModel = field(default_factory=BufferModel)
    is_mixed: bool = False
    age_minutes: float = 0.0
    drift_per_minute: float = 0.0
    transfer_cv: float = 0.004
    sensor_sd_ph: float = 0.0025

    @property
    def volume_ml(self) -> float:
        return self.acid_ml + self.base_ml + self.water_ml

    @property
    def reserve_ml(self) -> float:
        return max(0.0, self.protocol.final_volume_ml - self.volume_ml)

    def true_ph(self) -> float:
        # Henderson-Hasselbalch-like empirical surrogate. Citrate is triprotic;
        # this is intentionally NOT a validated chemical equilibrium model.
        acid_mmol = self.acid_ml * self.protocol.acid_stock_m
        base_mmol = self.base_ml * self.protocol.base_stock_m
        return (self.model.predict(acid_mmol, base_mmol, self.mix_seconds)
                + self.instrument_bias + self.age_minutes * self.drift_per_minute)

    def measure_stable_ph(self) -> dict:
        if not self.is_mixed:
            raise RuntimeError("样品加液后尚未混合，不能测量 pH")
        target = self.true_ph()
        transient = float(self.rng.uniform(-0.035, 0.035))
        raw = []
        for index in range(self.protocol.max_measurement_reads):
            value = target + transient * math.exp(-index / 1.2)
            value += float(self.rng.normal(0.0, self.sensor_sd_ph))
            raw.append(round(value, 4))
            if len(raw) >= 3 and max(raw[-3:]) - min(raw[-3:]) <= self.protocol.measurement_stability_ph:
                return {"ph": round(float(np.median(raw[-3:])), 3),
                        "raw_ph": raw, "stable": True,
                        "elapsed_s": len(raw) * self.protocol.measurement_interval_s}
        raise RuntimeError(f"pH 读数在 {len(raw)} 次采样后仍不稳定：{raw}")

    def measure_ph(self) -> float:
        return self.measure_stable_ph()["ph"]

    def mix(self, seconds: int) -> None:
        if self.acid_ml <= 0 or self.base_ml <= 0:
            raise RuntimeError("两种母液尚未加入，不能混合")
        if not 5 <= seconds <= 90:
            raise ValueError("混合时间越界")
        self.mix_seconds = seconds
        self.is_mixed = True

    def hold(self, minutes: float) -> None:
        if minutes < 0 or not self.is_mixed:
            raise RuntimeError("只能在混合后等待非负时间")
        self.age_minutes += minutes

    def add(self, reagent: str, requested_ml: float) -> float:
        if requested_ml <= 0 or requested_ml > self.reserve_ml + 1e-9:
            raise ValueError("加液量超出样品剩余容量")
        actual = requested_ml * float(self.rng.normal(1.0, self.transfer_cv))
        actual = min(actual, self.reserve_ml)
        if reagent == "citric_acid":
            self.acid_ml += actual
        elif reagent == "trisodium_citrate":
            self.base_ml += actual
        elif reagent == "water":
            self.water_ml += actual
        else:
            raise ValueError(f"未知试剂: {reagent}")
        self.is_mixed = False
        return round(actual, 5)

    def top_up(self) -> float:
        remaining = self.reserve_ml
        self.water_ml += remaining
        self.is_mixed = False
        return round(remaining, 5)


def make_sample(protocol: Protocol, acid_ml: float, base_ml: float, mix_seconds: int,
                rng: np.random.Generator, bias: float,
                model: BufferModel | None = None,
                transfer_cv: float = 0.004,
                sensor_sd_ph: float = 0.0025) -> Sample:
    water_ml = protocol.final_volume_ml - protocol.reserve_ml - acid_ml - base_ml
    if water_ml < 0:
        raise ValueError("母液体积超出容器容量")
    # Initial transfers each have a small, independent simulated volume error.
    actual_acid = acid_ml * float(rng.normal(1.0, transfer_cv))
    actual_base = base_ml * float(rng.normal(1.0, transfer_cv))
    actual_water = water_ml * float(rng.normal(1.0, transfer_cv))
    actual_water = max(0.0, min(actual_water, protocol.final_volume_ml - actual_acid - actual_base))
    sample = Sample(protocol, actual_acid, actual_base, actual_water,
                    mix_seconds, bias, rng, model=model or BufferModel(),
                    transfer_cv=transfer_cv, sensor_sd_ph=sensor_sd_ph)
    sample.mix(mix_seconds)
    return sample

