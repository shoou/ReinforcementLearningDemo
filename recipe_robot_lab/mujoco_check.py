"""Cross-engine Panda kinematics and actuator replay with an attached payload.

This replays recorded waypoints in a separate MuJoCo workcell. The payload is
fixed to the hand; gripper force and chemistry are outside this audit.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _tool_marker(data, hand_id: int) -> np.ndarray:
    return data.xpos[hand_id] + data.xmat[hand_id].reshape(3, 3) @ np.array([0, 0, 0.105])


def _dynamic_replay(model_path: Path, experiments_path: Path, mujoco,
                    command_slew_limit: float = 0.6) -> dict:
    """Run position actuators against Panda inertia and a fixed 80 g vial."""
    from arm_sim import STATIONS

    root = ET.parse(model_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Panda MJCF 缺少 compiler 节点")
    compiler.set("meshdir", str((model_path.parent / "assets").resolve()))
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Panda MJCF 缺少 worldbody 节点")
    for name, (x, y, _) in STATIONS.items():
        ET.SubElement(world, "geom", name=f"station_{name}", type="box",
                      pos=f"{x} {y} 0.012", size="0.045 0.04 0.012",
                      rgba="0.35 0.55 0.65 1")
    hand = root.find(".//body[@name='hand']")
    if hand is None:
        raise ValueError("Panda MJCF 缺少 hand 刚体")
    vial = ET.SubElement(hand, "body", name="carried_vial", pos="0 0 0.21")
    ET.SubElement(vial, "geom", type="cylinder", size="0.014 0.035",
                  mass="0.08", rgba="0.7 0.8 1 0.8", contype="0",
                  conaffinity="0")
    with tempfile.TemporaryDirectory(prefix="recipe_mujoco_") as directory:
        scene_path = Path(directory) / "workcell.xml"
        ET.ElementTree(root).write(scene_path, encoding="unicode")
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    hand_id = model.body("hand").id
    joint_positions = [int(model.jnt_qposadr[model.joint(f"joint{i + 1}").id])
                       for i in range(7)]
    station_geoms = {model.geom(f"station_{name}").id for name in STATIONS}
    effort_limits = np.max(np.abs(model.actuator_forcerange[:7]), axis=1)
    errors = []
    max_effort = 0.0
    max_velocity = 0.0
    effort_limit_samples = 0
    effort_samples = 0
    joint_saturation_counts = np.zeros(7, dtype=int)
    station_penetrations = 0
    simulated_s = 0.0
    completed_rows = 0
    with experiments_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("status") != "completed":
                continue
            trace = record.get("arm_trace", [])
            grasp = next((i for i, event in enumerate(trace)
                          if event["event"] == "grasp_vial"), None)
            release = next((i for i, event in enumerate(trace)
                            if event["event"] == "release_vial"), None)
            if grasp is None or release is None:
                continue
            poses = [event for event in trace[grasp - 1:release]
                     if "joints_rad" in event and "actual_m" in event]
            if len(poses) < 2:
                continue
            data = mujoco.MjData(model)
            data.qpos[joint_positions] = poses[0]["joints_rad"]
            data.ctrl[:7] = poses[0]["joints_rad"]
            mujoco.mj_forward(model, data)
            completed_rows += 1
            for pose in poses[1:]:
                target = np.asarray(pose["joints_rad"], dtype=float)
                remaining = max(1, round((300 / 240) / model.opt.timestep))
                while remaining:
                    chunk = min(10, remaining)
                    chunk_s = chunk * model.opt.timestep
                    data.ctrl[:7] += np.clip(target - data.ctrl[:7],
                                             -command_slew_limit * chunk_s,
                                             command_slew_limit * chunk_s)
                    mujoco.mj_step(model, data, nstep=chunk)
                    max_effort = max(max_effort, float(np.max(np.abs(data.actuator_force[:7]))))
                    max_velocity = max(max_velocity, float(np.max(np.abs(data.qvel[:7]))))
                    saturated = np.abs(data.actuator_force[:7]) >= effort_limits - 1e-3
                    effort_limit_samples += int(np.any(saturated))
                    joint_saturation_counts += saturated.astype(int)
                    effort_samples += 1
                    station_penetrations += sum(
                        contact.dist < -0.002 and
                        (contact.geom1 in station_geoms or contact.geom2 in station_geoms)
                        for contact in data.contact)
                    remaining -= chunk
                errors.append(float(np.linalg.norm(
                    _tool_marker(data, hand_id) - pose["actual_m"])))
                simulated_s += 300 / 240
    if not errors:
        raise ValueError("没有可回放的完整抓取轨迹")
    return {
        "engine": "MuJoCo", "model": "Menagerie Panda + fixed 80 g vial",
        "replayed_experiments": completed_rows,
        "replayed_waypoints": len(errors),
        "simulated_seconds": round(simulated_s, 2),
        "max_waypoint_error_m": round(float(np.max(errors)), 5),
        "p95_waypoint_error_m": round(float(np.percentile(errors, 95)), 5),
        "max_abs_actuator_torque_nm": round(max_effort, 3),
        "max_abs_joint_speed_rad_s": round(max_velocity, 3),
        "effort_saturation_fraction": round(effort_limit_samples / effort_samples, 4),
        "joint_saturation_fractions": [round(float(value / effort_samples), 4)
                                       for value in joint_saturation_counts],
        "command_slew_limit_rad_s": command_slew_limit,
        "station_penetration_samples": station_penetrations,
        "within_25mm_and_no_station_penetration":
            bool(max(errors) <= 0.025 and station_penetrations == 0),
        "scope": "actuator_and_fixed_payload_replay_payload_collision_disabled_no_gripper_force_or_fluid_model",
    }


def audit(model_path: Path, experiments_path: Path) -> dict:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("需要安装 mujoco Python 包") from exc
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    hand_id = model.body("hand").id
    joint_ids = [model.joint(f"joint{index + 1}").id for index in range(7)]
    entries = 0
    errors = []
    violations = 0
    with experiments_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            for event in record.get("arm_trace", []):
                if "joints_rad" not in event or "actual_m" not in event:
                    continue
                joints = event["joints_rad"]
                if len(joints) != 7:
                    raise ValueError("轨迹中的 Panda 关节数不是 7")
                for joint_id, angle in zip(joint_ids, joints):
                    qpos_id = model.jnt_qposadr[joint_id]
                    data.qpos[qpos_id] = angle
                    lower, upper = model.jnt_range[joint_id]
                    violations += int(angle < lower - 1e-3 or angle > upper + 1e-3)
                mujoco.mj_forward(model, data)
                # PyBullet link 11 is 105 mm along the Panda hand's local z.
                marker = _tool_marker(data, hand_id)
                errors.append(float(np.linalg.norm(marker - event["actual_m"])))
                entries += 1
    if not errors:
        raise ValueError("实验记录没有可核验的关节轨迹")
    result = {
        "source": "pybullet_joint_trace_replayed_as_mujoco_forward_kinematics",
        "model": str(model_path), "model_name": model_path.stem,
        "checked_poses": entries,
        "max_marker_disagreement_m": round(float(np.max(errors)), 6),
        "p95_marker_disagreement_m": round(float(np.percentile(errors, 95)), 6),
        "joint_limit_violations": violations,
        "within_5mm": bool(max(errors) <= 0.005 and violations == 0),
        "scope": "static_kinematics_plus_separate_actuator_replay_no_real_robot_validation",
    }
    result["dynamic_replay"] = _dynamic_replay(model_path, experiments_path, mujoco)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="官方 MuJoCo Menagerie franka_emika_panda/panda.xml")
    parser.add_argument("--experiments", type=Path,
                        default=Path(__file__).parent / "output" / "experiments.jsonl")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "output" / "mujoco_audit.json")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("未找到 Panda MJCF 模型文件")
    result = audit(args.model, args.experiments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["within_5mm"] or not result["dynamic_replay"]["within_25mm_and_no_station_penetration"]:
        raise SystemExit("MuJoCo 运动学或动力学回放超出阈值")


if __name__ == "__main__":
    main()
