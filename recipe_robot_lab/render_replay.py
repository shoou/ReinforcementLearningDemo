"""Render a compact GIF from recorded Panda poses for classroom inspection."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from arm_sim import ArmCell, STATIONS, TOOL_LINK


def render_replay(row: dict, output: Path, frame_count: int = 28) -> dict:
    """Visual trace replay; physics and instruments are not rerun here."""
    timeline = []
    held = False
    station = "rack"
    for event in row["arm_trace"]:
        if event["event"] == "grasp_vial":
            held = True
        elif event["event"] == "release_vial":
            held = False
        elif event["event"] == "motion_policy_plan":
            station = event["destination"]
        if "joints_rad" in event:
            timeline.append((event, held, station))
    if len(timeline) < 2:
        raise ValueError("机械臂轨迹太短，无法生成回放")
    indices = np.unique(np.linspace(0, len(timeline) - 1,
                                    min(frame_count, len(timeline)), dtype=int))
    frames = []
    cell = ArmCell(gui=False)
    try:
        with tempfile.TemporaryDirectory(prefix="recipe_frames_") as directory:
            frame_path = Path(directory) / "frame.png"
            for frame_no, index in enumerate(indices):
                event, vial_held, destination = timeline[int(index)]
                for joint, angle in enumerate(event["joints_rad"]):
                    p.resetJointState(cell.robot, joint, angle)
                if vial_held:
                    tool = p.getLinkState(cell.robot, TOOL_LINK,
                                          computeForwardKinematics=True)[4:6]
                    vial_pos, vial_quat = p.multiplyTransforms(
                        tool[0], tool[1], (0, 0, 0.11), (0, 0, 0, 1))
                else:
                    x, y, _ = STATIONS["rack"]
                    vial_pos, vial_quat = (x, y, 0.058), (0, 0, 0, 1)
                p.resetBasePositionAndOrientation(cell.vial, vial_pos, vial_quat)
                cell.capture(frame_path)
                with Image.open(frame_path) as screenshot:
                    frame = screenshot.convert("RGB").resize((640, 400))
                draw = ImageDraw.Draw(frame)
                draw.rounded_rectangle((12, 12, 425, 54), radius=8,
                                       fill=(16, 35, 43))
                caption = (f"{row['experiment_id']}  |  {destination.upper()}  |  "
                           f"{frame_no + 1}/{len(indices)}")
                draw.text((24, 25), caption, fill=(240, 250, 248))
                frames.append(frame)
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=170, loop=0, optimize=True)
    finally:
        cell.close()
    return {"kind": "recorded_pose_visualization", "frames": len(frames),
            "duration_s": round(len(frames) * 0.17, 2), "file": output.name}
