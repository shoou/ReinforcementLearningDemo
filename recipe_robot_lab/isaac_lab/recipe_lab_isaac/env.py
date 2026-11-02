"""Direct-workflow Franka docking task; this trains motion, not chemistry."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import DirectRLEnv

if TYPE_CHECKING:
    from .env_cfg import LabDockEnvCfg


class LabDockEnv(DirectRLEnv):
    cfg: LabDockEnvCfg

    def __init__(self, cfg: LabDockEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        joint_ids, _ = self.robot.find_joints("panda_joint[1-7]")
        hand_ids, _ = self.robot.find_bodies("panda_hand")
        if len(joint_ids) != 7 or len(hand_ids) != 1:
            raise RuntimeError("Franka Panda 资产关节或手部名称与任务配置不匹配")
        self.joint_ids = torch.as_tensor(joint_ids, device=self.device)
        self.hand_id = hand_ids[0]
        self.station_xyz = torch.tensor(
            [(0.43, -0.21, 0.17), (0.53, -0.08, 0.20),
             (0.53, 0.08, 0.20), (0.43, 0.21, 0.20)],
            dtype=torch.float32, device=self.device,
        )
        self.goal_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.joint_target = self.robot.data.default_joint_pos.torch[:, self.joint_ids].clone()
        self.last_actions = torch.zeros((self.num_envs, 7), device=self.device)

    def _hand_position(self) -> torch.Tensor:
        return self.robot.data.body_pos_w.torch[:, self.hand_id, :]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.last_actions = torch.clamp(actions, -1.0, 1.0)
        current = self.robot.data.joint_pos.torch[:, self.joint_ids]
        requested = current + self.cfg.joint_action_scale * self.last_actions
        limits = self.robot.data.soft_joint_pos_limits.torch[:, self.joint_ids]
        self.joint_target = torch.clamp(requested, limits[..., 0], limits[..., 1])

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target_index(target=self.joint_target,
                                                    joint_ids=self.joint_ids)

    def _get_observations(self) -> dict:
        q = self.robot.data.joint_pos.torch[:, self.joint_ids]
        dq = self.robot.data.joint_vel.torch[:, self.joint_ids]
        hand_local = self._hand_position() - self.scene.env_origins
        delta = self.goal_w - self._hand_position()
        obs = torch.cat((q - self.robot.data.default_joint_pos.torch[:, self.joint_ids],
                         dq, hand_local, delta), dim=-1)
        return {"policy": obs}

    def _distance(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.goal_w - self._hand_position(), dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        distance = self._distance()
        success = distance < self.cfg.success_distance_m
        return (2.0 * torch.exp(-10.0 * distance)
                - 0.02 * torch.sum(self.last_actions ** 2, dim=-1)
                + 5.0 * success.float())

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        success = self._distance() < self.cfg.success_distance_m
        timeout = self.episode_length_buf >= self.max_episode_length
        return success, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        count = len(env_ids)
        q = self.robot.data.default_joint_pos.torch[env_ids].clone()
        q[:, self.joint_ids] += (2 * torch.rand((count, 7), device=self.device) - 1) * self.cfg.initial_joint_noise_rad
        limits = self.robot.data.soft_joint_pos_limits.torch[env_ids]
        q = q.clamp(limits[..., 0], limits[..., 1])
        dq = torch.zeros_like(q)
        root = self.robot.data.default_root_pose.torch[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        root_velocity = self.robot.data.default_root_vel.torch[env_ids].clone()
        self.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=dq, env_ids=env_ids)
        self.joint_target[env_ids] = q[:, self.joint_ids]

        stations = torch.randint(0, len(self.station_xyz), (count,), device=self.device)
        self.goal_w[env_ids] = self.station_xyz[stations] + self.scene.env_origins[env_ids]
        self.goal_w[env_ids, :2] += (
            2 * torch.rand((count, 2), device=self.device) - 1
        ) * self.cfg.target_xy_noise_m
