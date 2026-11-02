"""PyBullet Panda cell: kinematic station transfers and abstract instrument events.

No fluid particles, pipette mechanism, pH electrode physics, or real controller API.
"""
from __future__ import annotations

import math

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:
    raise RuntimeError("需要 PyBullet；参见 README.md 的安装说明") from exc


STATIONS = {
    "rack": (0.43, -0.21, 0.17),
    "doser": (0.53, -0.08, 0.20),
    "mixer": (0.53, 0.08, 0.20),
    "ph_meter": (0.43, 0.21, 0.20),
}
HOVER_Z = 0.34
TOOL_LINK = 11
REST = (0.0, -0.4, 0.0, -2.3, 0.0, 2.0, 0.8)


class ArmCell:
    def __init__(self, gui: bool = False, motion_policy=None):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("无法启动 PyBullet")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.loadURDF("plane.urdf", [0, 0, -0.005])
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        self.lower = [p.getJointInfo(self.robot, i)[8] for i in range(7)]
        self.upper = [p.getJointInfo(self.robot, i)[9] for i in range(7)]
        for joint, value in enumerate(REST):
            p.resetJointState(self.robot, joint, value)
        self.trace = []
        self.constraint = None
        self.vial = None
        self.station = "rack"
        self.sample_randomization = {}
        self.sim_time_s = 0.0
        self.station_bodies = []
        self.unsafe_contacts = 0
        self.motion_policy = motion_policy
        self.motion_policy_moves = 0
        self.motion_fallbacks = 0
        for name, xyz in STATIONS.items():
            colour = {"rack": (0.25, 0.45, 0.75, 1),
                      "doser": (0.35, 0.7, 0.45, 1),
                      "mixer": (0.85, 0.57, 0.21, 1),
                      "ph_meter": (0.6, 0.44, 0.73, 1)}[name]
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=(0.045, 0.04, 0.012),
                                         rgbaColor=colour)
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=(0.045, 0.04, 0.012))
            body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=collision,
                                     baseVisualShapeIndex=visual,
                                     basePosition=(xyz[0], xyz[1], 0.012))
            self.station_bodies.append(body)
        self._decorate_workcell()
        self._new_vial()

    @staticmethod
    def _visual_box(position, half_extents, colour):
        shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                    rgbaColor=colour)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape,
                          basePosition=position)

    @staticmethod
    def _visual_cylinder(position, radius, height, colour):
        shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius,
                                    length=height, rgbaColor=colour)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape,
                          basePosition=position)

    def _decorate_workcell(self):
        """Instrument housings are visual; station pads retain collision checks."""
        steel = (0.67, 0.73, 0.76, 1)
        dark = (0.13, 0.20, 0.25, 1)
        glass = (0.72, 0.86, 0.94, 0.65)
        self._visual_box((0.43, 0, -0.027), (0.42, 0.36, 0.022), steel)
        self._visual_box((0.43, -0.21, 0.035), (0.07, 0.055, 0.009), dark)
        for y in (-0.23, -0.19):
            self._visual_cylinder((0.43, y, 0.054), 0.017, 0.022, glass)
        # Liquid-handling head and two reagent reservoirs.
        self._visual_box((0.67, -0.08, 0.12), (0.045, 0.055, 0.12), steel)
        self._visual_box((0.61, -0.08, 0.225), (0.055, 0.022, 0.014), dark)
        self._visual_cylinder((0.56, -0.08, 0.172), 0.004, 0.085, dark)
        self._visual_cylinder((0.70, -0.17, 0.065), 0.025, 0.10, (0.58, 0.80, 0.88, 0.8))
        self._visual_cylinder((0.70, -0.23, 0.065), 0.025, 0.10, (0.90, 0.78, 0.58, 0.8))
        # Mixer housing with a separate status indicator.
        self._visual_box((0.65, 0.08, 0.075), (0.06, 0.055, 0.07), dark)
        self._visual_box((0.65, 0.08, 0.14), (0.042, 0.012, 0.017), (0.12, 0.76, 0.58, 1))
        # pH meter display and electrode arm.
        self._visual_box((0.62, 0.21, 0.105), (0.065, 0.045, 0.085), steel)
        self._visual_box((0.57, 0.21, 0.13), (0.008, 0.034, 0.022), (0.10, 0.42, 0.54, 1))
        self._visual_box((0.52, 0.21, 0.205), (0.075, 0.006, 0.006), dark)
        self._visual_cylinder((0.45, 0.21, 0.145), 0.003, 0.075, dark)

    def _new_vial(self, offset=(0.0, 0.0), mass=0.08, friction=0.7):
        if self.vial is not None:
            p.removeBody(self.vial)
        shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.014, height=0.07)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.014,
                                     length=0.07, rgbaColor=(0.78, 0.89, 0.98, 0.8))
        x, y, _ = STATIONS["rack"]
        self.vial = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape,
                                      baseVisualShapeIndex=visual,
                                      basePosition=(x + offset[0], y + offset[1], 0.058))
        p.changeDynamics(self.vial, -1, lateralFriction=friction)
        self.sample_randomization = {"rack_offset_m": list(offset),
                                     "vial_mass_kg": mass, "vial_friction": friction}
        self.station = "rack"

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def _tool_pos(self):
        return p.getLinkState(self.robot, TOOL_LINK, computeForwardKinematics=True)[4]

    def _move_to(self, target, label):
        q = p.calculateInverseKinematics(
            self.robot, TOOL_LINK, target,
            p.getQuaternionFromEuler((math.pi, 0, 0)),
            lowerLimits=self.lower, upperLimits=self.upper,
            jointRanges=[hi - lo for lo, hi in zip(self.lower, self.upper)],
            restPoses=REST,
            maxNumIterations=180, residualThreshold=1e-4)[:7]
        for joint, angle in enumerate(q):
            info = p.getJointInfo(self.robot, joint)
            if not info[8] - 1e-3 <= angle <= info[9] + 1e-3:
                raise RuntimeError(f"关节 {joint} 超出限位")
        for tick in range(300):
            for joint, angle in enumerate(q):
                p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL,
                                        targetPosition=angle, force=87, maxVelocity=0.8)
            p.stepSimulation()
            if tick % 4 == 0:
                for station_body in self.station_bodies:
                    if any(contact[8] < -0.002 for contact in
                           p.getContactPoints(self.robot, station_body)):
                        self.unsafe_contacts += 1
                        raise RuntimeError(f"{label} 中机械臂与工位发生穿透接触")
        self.sim_time_s += 300 / 240
        actual = self._tool_pos()
        error = math.dist(actual, target)
        if error > 0.025:
            raise RuntimeError(f"{label} 末端定位误差 {error:.3f} m")
        self.trace.append({"event": label, "target_m": list(target),
                           "actual_m": [round(x, 5) for x in actual],
                           "position_error_m": round(error, 5),
                           "joints_rad": [round(p.getJointState(self.robot, i)[0], 5)
                                          for i in range(7)]})

    def grasp(self):
        x, y, z = STATIONS["rack"]
        self._move_to((x, y, HOVER_Z), "approach_rack")
        self._move_to((x, y, z), "grasp_pose")
        centre = p.getBasePositionAndOrientation(self.vial)[0]
        if math.dist(self._tool_pos(), centre) > 0.16:
            raise RuntimeError("样品瓶不在可抓取范围")
        tool_pose = p.getLinkState(self.robot, TOOL_LINK, computeForwardKinematics=True)[4:6]
        vial_pose = p.getBasePositionAndOrientation(self.vial)
        inv = p.invertTransform(*tool_pose)
        relative = p.multiplyTransforms(*inv, *vial_pose)
        self.constraint = p.createConstraint(
            self.robot, TOOL_LINK, self.vial, -1, p.JOINT_FIXED,
            [0, 0, 0], relative[0], [0, 0, 0], parentFrameOrientation=relative[1])
        p.changeConstraint(self.constraint, maxForce=80)
        self.trace.append({"event": "grasp_vial", "tool": "parallel_gripper_model"})

    def move(self, name: str):
        if self.constraint is None:
            raise RuntimeError("移动样品前须完成抓取")
        if name not in STATIONS:
            raise ValueError(f"未知工位: {name}")
        new = STATIONS[name]
        if self.motion_policy is None:
            old = STATIONS[self.station]
            self._move_to((old[0], old[1], HOVER_Z), "lift_from_" + self.station)
            self._move_to((new[0], new[1], HOVER_Z), "travel_to_" + name)
            self._move_to(new, "dock_at_" + name)
        else:
            from motion_rl import plan, waypoint
            nodes, fallback = plan(self.motion_policy, self.station, name)
            self.motion_policy_moves += 1
            self.motion_fallbacks += int(fallback)
            self.trace.append({"event": "motion_policy_plan", "source": self.station,
                               "destination": name, "waypoint_count": len(nodes),
                               "fallback": fallback})
            for node in nodes:
                self._move_to(waypoint(node, name),
                              "fallback_waypoint" if fallback else "rl_waypoint")
        vial_xyz = p.getBasePositionAndOrientation(self.vial)[0]
        if math.dist(vial_xyz[:2], new[:2]) > 0.025:
            raise RuntimeError(f"样品瓶未随机械臂到达 {name} 工位")
        self.trace.append({"event": "vial_at_station", "station": name,
                           "vial_position_m": [round(v, 5) for v in vial_xyz]})
        self.station = name

    def instrument(self, event: dict):
        op = event["op"]
        required = {"dispense": "doser", "mix": "mixer", "measure": "ph_meter",
                    "top_up": "doser", "hold": "ph_meter"}
        if op in required and self.station != required[op]:
            raise RuntimeError(f"{op} 只能在 {required[op]} 工位执行")
        # The instrument is a separate device; only an abstract command/ack is modeled.
        self.trace.append({"event": op, "station": self.station,
                           "command": event, "ack": "simulated"})

    def release(self):
        if self.station != "rack":
            raise RuntimeError("样品瓶只能在样品架释放")
        p.removeConstraint(self.constraint)
        self.constraint = None
        self.trace.append({"event": "release_vial", "station": "rack"})
        self._move_to((STATIONS["rack"][0], STATIONS["rack"][1], HOVER_Z),
                      "retreat_from_rack")

    def safe_stop(self, reason: str):
        for joint in range(7):
            position = p.getJointState(self.robot, joint)[0]
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL,
                                    targetPosition=position, force=87, maxVelocity=0)
        self.trace.append({"event": "safe_stop", "reason": reason,
                           "vial_held": self.constraint is not None})

    def reset_sample(self, rng=None):
        if self.constraint is not None:
            raise RuntimeError("无法在抓取状态重置样品")
        if rng is None:
            self._new_vial()
        else:
            self._new_vial(offset=tuple(float(v) for v in rng.uniform(-0.002, 0.002, 2)),
                           mass=float(rng.uniform(0.06, 0.10)),
                           friction=float(rng.uniform(0.5, 0.9)))
        self.trace = []

    def capture(self, path):
        """Save a CPU-rendered inspection image of the simulated workcell."""
        import numpy as np
        from PIL import Image

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.32, 0.0, 0.25), distance=1.25,
            yaw=55, pitch=-28, roll=0, upAxisIndex=2)
        projection = p.computeProjectionMatrixFOV(55, 800 / 500, 0.05, 3.0)
        _, _, rgba, _, _ = p.getCameraImage(
            800, 500, viewMatrix=view, projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER)
        Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(500, 800, 4)).convert("RGB").save(path)

