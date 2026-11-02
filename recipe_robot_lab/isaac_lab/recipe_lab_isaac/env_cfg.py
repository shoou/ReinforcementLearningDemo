"""Isaac Lab 3 scene and curriculum for the Panda station-docking skill."""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab_assets.robots import FRANKA_PANDA_CFG


def station_cfg(name: str, xyz: tuple[float, float, float], color: tuple[float, float, float]):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.09, 0.08, 0.024),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=xyz),
    )


@configclass
class LabSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    rack = station_cfg("Rack", (0.43, -0.21, 0.012), (0.25, 0.45, 0.75))
    doser = station_cfg("Doser", (0.53, -0.08, 0.012), (0.35, 0.70, 0.45))
    mixer = station_cfg("Mixer", (0.53, 0.08, 0.012), (0.85, 0.57, 0.21))
    ph_meter = station_cfg("PhMeter", (0.43, 0.21, 0.012), (0.60, 0.44, 0.73))
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0)
    )


@configclass
class LabDockEnvCfg(DirectRLEnvCfg):
    """Learn joint-space docking at the rack, doser, mixer, or pH station."""

    decimation = 4
    episode_length_s = 8.0
    action_space = 7
    observation_space = 20
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120)
    scene: LabSceneCfg = LabSceneCfg(num_envs=256, env_spacing=2.0)
    joint_action_scale = 0.035
    success_distance_m = 0.025
    initial_joint_noise_rad = 0.06
    target_xy_noise_m = 0.002
    action_noise_model: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0, std=0.01, operation="add"
    )
    observation_noise_model: GaussianNoiseCfg = GaussianNoiseCfg(
        mean=0.0, std=0.001, operation="add"
    )
