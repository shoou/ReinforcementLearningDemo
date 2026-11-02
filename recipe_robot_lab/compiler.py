"""Compile a deliberately narrow Chinese buffer protocol into checked operations.

This is a local, deterministic compiler. It never asks an LLM to emit motor commands.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass


DEFAULT_TEXT = (
    "配制20 mL目标pH 5.0的柠檬酸-柠檬酸钠缓冲液。"
    "使用0.1 mol/L柠檬酸和0.1 mol/L柠檬酸钠母液，混合20秒后测量pH，"
    "根据结果调整配方，记录用量、制备时间和测量结果。"
)


@dataclass(frozen=True)
class Protocol:
    final_volume_ml: float
    target_ph: float
    acid_stock_m: float
    base_stock_m: float
    mix_seconds: int
    ph_tolerance: float = 0.10
    reserve_ml: float = 1.0
    measurement_stability_ph: float = 0.015
    max_measurement_reads: int = 8
    measurement_interval_s: float = 2.0
    chemical_system: str = "citric_acid_trisodium_citrate"

    def as_dict(self) -> dict:
        return asdict(self)


def _unique_match(pattern: str, text: str, label: str) -> re.Match:
    found = list(re.finditer(pattern, text, flags=re.I))
    if len(found) != 1:
        raise ValueError(f"缺少或无法识别{label}；请明确写出数值和单位")
    return found[0]


def compile_text(text: str) -> Protocol:
    """Only accept the supported citrate system and explicit concentrations.

    Reject unsupported chemicals and ambiguous units rather than guessing.
    """
    if not text.strip():
        raise ValueError("实验说明不能为空")
    if not all(x in text for x in ("柠檬酸", "柠檬酸钠")):
        raise ValueError("此案例只支持柠檬酸和柠檬酸钠母液")
    volume_match = _unique_match(r"(?:配制|制备|总体积)\s*(\d+(?:\.\d+)?)\s*mL", text, "终体积 mL")
    target_match = _unique_match(r"(?:目标\s*)?pH\s*(\d+(?:\.\d+)?)", text, "目标 pH")
    mix_match = _unique_match(r"混(?:合|匀)\s*(\d+(?:\.\d+)?)\s*秒", text, "混合时间（秒）")
    volume, target, mix = (float(m.group(1)) for m in
                           (volume_match, target_match, mix_match))
    concentration_matches = list(re.finditer(
        r"(\d+(?:\.\d+)?)\s*(?:mol\s*/\s*L|M)\s*柠檬酸(钠)?", text, re.I))
    stocks = {"base" if m.group(2) else "acid": float(m.group(1))
              for m in concentration_matches}
    if len(concentration_matches) != 2:
        raise ValueError("请各写一次柠檬酸和柠檬酸钠母液浓度")
    if set(stocks) != {"acid", "base"}:
        raise ValueError("请分别写明两种母液的浓度，例如 0.1 mol/L柠檬酸和0.1 mol/L柠檬酸钠")
    # Every non-punctuation span must belong to the supported grammar. This
    # prevents an extra chemical or operation from being silently discarded.
    spans = [volume_match, target_match, mix_match, *concentration_matches]
    remainder = list(text)
    for match in spans:
        remainder[match.start():match.end()] = " " * (match.end() - match.start())
    remainder_text = "".join(remainder)
    for phrase in ("的柠檬酸-柠檬酸钠缓冲液", "柠檬酸-柠檬酸钠缓冲液",
                   "的柠檬酸盐缓冲液", "柠檬酸盐缓冲液",
                   "记录用量、制备时间和测量结果", "根据结果调整配方",
                   "后测量pH", "测量pH", "使用", "母液", "和"):
        remainder_text = remainder_text.replace(phrase, "")
    remainder_text = re.sub(r"[\s，。；;、,:：]+", "", remainder_text)
    if remainder_text:
        raise ValueError(f"存在未识别的试剂或操作：{remainder_text[:40]}")
    if not 5 <= volume <= 100:
        raise ValueError("此教学工作站只支持 5–100 mL 终体积")
    if not 4.6 <= target <= 5.2:
        raise ValueError("此案例候选配方只覆盖目标 pH 4.6–5.2")
    if not all(0.01 <= x <= 0.2 for x in stocks.values()):
        raise ValueError("母液浓度须在 0.01–0.2 mol/L 范围内")
    if not 5 <= mix <= 90:
        raise ValueError("混合时间须在 5–90 秒范围内")
    reserve = min(1.0, volume * 0.05)
    return Protocol(volume, target, stocks["acid"], stocks["base"], int(mix), reserve_ml=reserve)


def compile_actions(protocol: Protocol, acid_ml: float, base_ml: float, mix_seconds: int) -> list[dict]:
    """Generate an executable intermediate representation, not raw joint motion."""
    if not 5 <= mix_seconds <= 90:
        raise ValueError("混合时间越界")
    if acid_ml <= 0 or base_ml <= 0:
        raise ValueError("两种母液体积必须大于零")
    water_ml = protocol.final_volume_ml - protocol.reserve_ml - acid_ml - base_ml
    if water_ml < 0:
        raise ValueError("试剂用量超过预留调整体积后的容量")
    for value in (acid_ml, base_ml, water_ml):
        if value and value < 0.05:
            raise ValueError("非零加液量不能小于 0.05 mL")
    return [
        {"op": "identify", "object": "sample_vial"},
        {"op": "grasp", "object": "sample_vial"},
        {"op": "move", "station": "doser"},
        {"op": "dispense", "source": "citric_acid", "volume_ml": round(acid_ml, 4)},
        {"op": "dispense", "source": "trisodium_citrate", "volume_ml": round(base_ml, 4)},
        {"op": "dispense", "source": "water", "volume_ml": round(water_ml, 4)},
        {"op": "move", "station": "mixer"},
        {"op": "mix", "seconds": mix_seconds},
        {"op": "move", "station": "ph_meter"},
        {"op": "measure", "property": "ph"},
        {"op": "adjust", "max_additional_ml": protocol.reserve_ml},
        {"op": "move", "station": "doser"},
        {"op": "top_up", "final_volume_ml": protocol.final_volume_ml},
        {"op": "move", "station": "mixer"},
        {"op": "mix", "seconds": mix_seconds},
        {"op": "move", "station": "ph_meter"},
        {"op": "measure", "property": "final_ph"},
        {"op": "hold", "minutes": 10},
        {"op": "measure", "property": "ph_after_10min"},
        {"op": "move", "station": "rack"},
        {"op": "release", "object": "sample_vial"},
    ]
