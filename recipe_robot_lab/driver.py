"""The execution boundary between a compiled protocol and laboratory devices."""
from __future__ import annotations

from typing import Protocol as TypingProtocol

import numpy as np

from arm_sim import ArmCell
from compiler import Protocol
from twin import BufferModel, Sample


class LabDriver(TypingProtocol):
    """A real adapter must return instrument acknowledgements and measured data."""

    mode: str
    trace: list[dict]
    randomization: dict

    def identify(self, object_name: str) -> bool: ...
    def grasp(self) -> None: ...
    def move(self, station: str) -> None: ...
    def dispense(self, source: str, volume_ml: float) -> float: ...
    def mix(self, seconds: int) -> None: ...
    def measure_ph(self, kind: str) -> dict: ...
    def hold(self, minutes: float) -> None: ...
    def top_up(self, final_volume_ml: float) -> float: ...
    def release(self) -> None: ...
    def state(self) -> dict: ...
    def capture(self, path) -> None: ...
    def abort(self, reason: str) -> None: ...
    def close(self) -> None: ...


class DeviceFault(RuntimeError):
    def __init__(self, operation: str, occurrence: int):
        self.operation = operation
        self.occurrence = occurrence
        super().__init__(f"模拟设备故障：{operation} 第 {occurrence} 次调用")


class SimLabDriver:
    mode = "simulation"

    def __init__(self, protocol: Protocol, acid_ml: float, base_ml: float,
                 mix_seconds: int, rng: np.random.Generator, bias: float,
                 gui: bool = False, fault_at: tuple[str, int] | None = None,
                 model: BufferModel | None = None, motion_policy=None):
        self.cell = ArmCell(gui=gui, motion_policy=motion_policy)
        self.cell.reset_sample(rng)
        self.sample = Sample(protocol, 0.0, 0.0, 0.0, 0, bias, rng,
                             model=model or BufferModel())
        self.sample.drift_per_minute = float(rng.normal(0.0, 0.0008))
        self.trace = self.cell.trace
        self.randomization = self.cell.sample_randomization | {
            "ph_instrument_bias": bias, "buffer_model_intercept": self.sample.model.intercept}
        self._initial = {"citric_acid": acid_ml, "trisodium_citrate": base_ml,
                         "water": protocol.final_volume_ml - protocol.reserve_ml - acid_ml - base_ml}
        self._initial_done = set()
        self.instrument_time_s = 0.0
        self.fault_at = fault_at
        self.call_counts = {}

    def _check_fault(self, operation: str):
        count = self.call_counts.get(operation, 0) + 1
        self.call_counts[operation] = count
        if self.fault_at == (operation, count):
            raise DeviceFault(operation, count)

    def identify(self, object_name: str) -> bool:
        return object_name == "sample_vial"

    def grasp(self) -> None:
        self.cell.grasp()

    def move(self, station: str) -> None:
        self.cell.move(station)

    def dispense(self, source: str, volume_ml: float) -> float:
        self._check_fault("dispense")
        self.cell.instrument({"op": "dispense", "source": source, "volume_ml": volume_ml})
        self.instrument_time_s += 1.0 + volume_ml / 2.0
        if source in self._initial and source not in self._initial_done:
            if abs(volume_ml - self._initial[source]) > 1e-4:
                raise ValueError("初始加液命令与编译协议不一致")
            self._initial_done.add(source)
            if volume_ml == 0:
                return 0.0
            return self.sample.add(source, volume_ml)
        return self.sample.add(source, volume_ml)

    def mix(self, seconds: int) -> None:
        self._check_fault("mix")
        self.cell.instrument({"op": "mix", "seconds": seconds})
        self.sample.mix(seconds)
        self.instrument_time_s += seconds

    def measure_ph(self, kind: str) -> dict:
        self._check_fault("measure")
        self.cell.instrument({"op": "measure", "property": kind})
        result = self.sample.measure_stable_ph()
        self.instrument_time_s += result["elapsed_s"]
        return result

    def top_up(self, final_volume_ml: float) -> float:
        self._check_fault("top_up")
        if abs(final_volume_ml - self.sample.protocol.final_volume_ml) > 1e-6:
            raise ValueError("终体积与协议不一致")
        self.cell.instrument({"op": "top_up", "final_volume_ml": final_volume_ml})
        amount = self.sample.top_up()
        self.instrument_time_s += 1.0 + amount / 2.0
        return amount

    def hold(self, minutes: float) -> None:
        self._check_fault("hold")
        self.cell.instrument({"op": "hold", "minutes": minutes})
        self.sample.hold(minutes)
        self.instrument_time_s += minutes * 60

    def release(self) -> None:
        self.cell.release()

    def state(self) -> dict:
        return {"reserve_ml": self.sample.reserve_ml,
                "acid_base_ratio": self.sample.base_ml / max(self.sample.acid_ml, 1e-9),
                "is_mixed": self.sample.is_mixed,
                "time_s": round(self.cell.sim_time_s + self.instrument_time_s, 3),
                "unsafe_station_contacts": self.cell.unsafe_contacts,
                "motion_policy_moves": self.cell.motion_policy_moves,
                "motion_fallbacks": self.cell.motion_fallbacks,
                "sample_age_minutes": self.sample.age_minutes,
                "actual_ml": {"citric_acid": round(self.sample.acid_ml, 5),
                              "trisodium_citrate": round(self.sample.base_ml, 5),
                              "water": round(self.sample.water_ml, 5)},
                "total_volume_ml": round(self.sample.volume_ml, 5)}

    def capture(self, path) -> None:
        self.cell.capture(path)

    def abort(self, reason: str) -> None:
        self.cell.safe_stop(reason)

    def close(self) -> None:
        self.cell.close()
