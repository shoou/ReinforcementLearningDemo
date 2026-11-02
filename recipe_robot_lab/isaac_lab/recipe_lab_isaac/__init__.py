"""Register the laboratory docking skill with Isaac Lab's task discovery."""

import gymnasium as gym

gym.register(
    id="RecipeLab-Dock-Franka-Direct-v0",
    entry_point="recipe_lab_isaac.env:LabDockEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "recipe_lab_isaac.env_cfg:LabDockEnvCfg",
        "rsl_rl_cfg_entry_point": "recipe_lab_isaac.agents:LabDockPpoCfg",
        "default_agent": "rsl_rl",
    },
)
