"""Fit a buffer surrogate from held-out, clearly labeled synthetic measurements."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from compiler import Protocol
from twin import BufferModel


def features(acid_ml: float, base_ml: float, mix_seconds: int,
             protocol: Protocol) -> tuple[float, float, float]:
    acid_mmol = acid_ml * protocol.acid_stock_m
    base_mmol = base_ml * protocol.base_stock_m
    ratio = base_mmol / acid_mmol
    return (1.0, math.log10(ratio), math.exp(-mix_seconds / 9.0) * math.sin(ratio * 5))


def generate_synthetic_rows(protocol: Protocol, count: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    hidden = BufferModel(intercept=4.795, log_ratio_coefficient=1.015,
                         mixing_coefficient=0.055)
    rows = []
    for index in range(count):
        acid = round(float(rng.uniform(4.0, 8.0)) * protocol.final_volume_ml / 20, 4)
        base = round(float(rng.uniform(6.0, 11.0)) * protocol.final_volume_ml / 20, 4)
        mix = int(rng.choice([8, 12, 20, 30, 45, 60]))
        ph = hidden.predict(acid * protocol.acid_stock_m,
                            base * protocol.base_stock_m, mix)
        ph += float(rng.normal(0.0, 0.006))
        rows.append({"sample_id": f"CAL-{index + 1:03d}", "source": "synthetic",
                     "acid_ml": acid, "base_ml": base, "mix_seconds": mix,
                     "measured_ph": round(ph, 4)})
    return rows


def fit_model(rows: list[dict], protocol: Protocol, seed: int) -> tuple[BufferModel, dict]:
    if len(rows) < 20:
        raise ValueError("至少需要 20 条标定记录以保留独立验证集")
    x = np.asarray([features(r["acid_ml"], r["base_ml"], r["mix_seconds"], protocol)
                    for r in rows], dtype=float)
    y = np.asarray([r["measured_ph"] for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(rows))
    split = max(1, int(0.8 * len(rows)))
    train_ids, validation_ids = indices[:split], indices[split:]
    coefficients, *_ = np.linalg.lstsq(x[train_ids], y[train_ids], rcond=None)
    model = BufferModel(*map(float, coefficients))
    validation_pred = x[validation_ids] @ coefficients
    errors = validation_pred - y[validation_ids]
    metrics = {"source": "synthetic", "train_samples": len(train_ids),
               "validation_samples": len(validation_ids),
               "validation_rmse_ph": round(float(np.sqrt(np.mean(errors ** 2))), 5),
               "validation_max_abs_error_ph": round(float(np.max(np.abs(errors))), 5),
               "seed": seed}
    return model, metrics


def calibrate(protocol: Protocol, output: Path, seed: int,
              count: int = 80) -> tuple[BufferModel, dict]:
    rows = generate_synthetic_rows(protocol, count, seed)
    model, metrics = fit_model(rows, protocol, seed + 1)
    with (output / "synthetic_calibration.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "calibration_model.json").write_text(
        json.dumps({"model": asdict(model), "metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return model, metrics
