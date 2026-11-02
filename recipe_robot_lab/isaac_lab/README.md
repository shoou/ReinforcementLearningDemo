# Isaac Lab 工位靠近任务（可选）

该包注册 `RecipeLab-Dock-Franka-Direct-v0`，用于训练 Franka Panda 末端靠近样品架、加液器、混合器或 pH 计工位。场景使用碰撞方块表示工位；7 维关节增量动作、20 维本体与目标观测，以及随机初始关节角、目标位置和观测/动作噪声。奖励主要来自末端到目标的距离。任务目前没有瓶体抓取和液体动力学，也不与主案例的 `SimLabDriver` 联动。

这是按 Isaac Lab 3 的 DirectRLEnv 与 RSL-RL 配置接口编写的任务源码。当前开发机为 macOS，未安装 Isaac Lab，故**未通过 Isaac Lab 运行时验证**。请在 [官方支持的平台和环境](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)安装 Isaac Lab 和 RSL-RL 后，再按该版本的项目安装方式将本目录安装到同一 Python 环境：

```bash
python -m pip install -e recipe_robot_lab/isaac_lab
isaaclab train --rl_library rsl_rl --task RecipeLab-Dock-Franka-Direct-v0
```

官方源码工作区使用 `uv run isaaclab train ...` 时，请先把本包安装到该 `uv` 环境，再运行相同任务 ID。首次执行应先检查任务注册与仿真后端兼容性，再开始训练；本仓库没有可宣称收敛的 Isaac Lab checkpoint。主案例请运行仓库根目录的 `bash recipe_robot_lab/run.sh`。
