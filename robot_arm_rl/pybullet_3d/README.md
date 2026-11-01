# PyBullet 三维机械臂强化学习分拣

这是上一版二维案例的三维升级版，采用开源 [PyBullet](https://github.com/bulletphysics/bullet3) 物理仿真平台和其自带的 [Franka Panda 七关节 URDF](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf)。默认在 `DIRECT` 模式运行，动画由 CPU 的 TinyRenderer 离线生成，不要求独立显卡或显示器。PyBullet 的 URDF、逆运动学、碰撞检测和 CPU 渲染能力见[官方快速入门](https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html)。

打开预生成的 [三维回放与训练报告](output/report.html)，可以在 **4 个取件位 × 4 个料箱**的订单矩阵中切换动画，查看训练曲线、避障路径长度、接触与跟踪误差。单个动画也可直接打开，如[取件位 4 → 料箱 1](output/order_4_to_1.gif)。报告一次只加载选中的 GIF，方便普通笔记本浏览。

## 在本机运行

本工作区已建立独立的 `.venv-physics`，可直接运行：

```bash
robot_arm_rl/.venv-physics/bin/python robot_arm_rl/pybullet_3d/panda_sorting.py
```

在其他机器上，建议 Python 3.10–3.13，并建立独立虚拟环境。macOS Apple Silicon 使用 `pybullet-arm64==3.2.8`；Linux x86-64 可使用 `pybullet==3.2.7`。这两个包都提供 `import pybullet` 和 `pybullet_data`。动画另需 `pillow`。当前版本与平台轮子以 [PyPI 的 pybullet-arm64](https://pypi.org/project/pybullet-arm64/) 和 [pybullet](https://pypi.org/project/pybullet/) 页面为准。

```bash
python3.13 -m venv .venv-physics
# macOS Apple Silicon
.venv-physics/bin/python -m pip install pybullet-arm64==3.2.8 pillow
# Linux x86-64：改为安装 pybullet==3.2.7 pillow
.venv-physics/bin/python robot_arm_rl/pybullet_3d/panda_sorting.py
```

`--no-gif` 可略过渲染；`--gui` 另外打开 PyBullet 原生三维窗口（需要桌面显示环境）；`--episodes`、`--seed`、`--stress-trials` 和 `--output` 可调整实验。

## 与真实分拣工位的对应

三维场景有一条索引输送带上的四个取件位、固定设备障碍、四块独立料箱接收区域、0.08 kg 标准包裹和 Panda 七关节臂。取件位与料箱沿工作台横向分布在 `x = 0.40、0.48、0.64、0.72 m`，每个订单只生成一个包裹；因此 16 种订单覆盖了近端、远端、同列和跨列搬运。末端工具被建模为真空吸附器：只有靠近取件位才建立固定约束；到料箱位置后解除约束，让包裹在重力下落到对应接收区域。这个工具模型对应吸盘抓取流程，但没有建模真空压力、漏气或物体表面质量。

策略的状态是订单编号、取件/放置阶段及离散三维末端格点；动作是朝六个相邻格点移动。PyBullet 先对每个格点求逆运动学，并沿每条候选边采样关节插值；机械臂与设备之间要求 12 mm 的最近距离，携带包裹时还检查其保守包围盒。这个**由物理仿真器验证的安全动作图**供 Q-learning 训练。训练阶段在图上快速执行，不逐回合进行完整动力学步进；训练后则由 PyBullet 电机以 240 Hz 物理步进执行策略，再检测实际接触、末端误差和落箱位置。因此，训练速度较快，同时保留了独立的动力学回放验证。

奖励包含每步时间惩罚、接近目标的距离变化、抓取/放置奖励，以及越界和超时惩罚。默认训练 12,000 回合，16 种订单由相同策略学习；初始位姿在障碍上方，低位横穿设备会被阻止，携带包裹时需要更高的越障路径。

默认种子 42 的一次运行结果：16 种订单的物理回放全部完成，路径为 20–27 步，对应离散安全图的最短路径为 19–24 步。16 次回放的设备接触次数均为 0，最大末端跟踪误差为 4.5 mm。另以包裹平面位置 ±5 mm、质量 0.04–0.12 kg、摩擦系数 0.4–1.0 做每单三次扰动回放，48/48 完成。指标记录在 [metrics.json](output/metrics.json)，16 个订单的模拟关节目标和工具事件记录在 [joint_trajectories.json](output/joint_trajectories.json)。这些是模拟结果，**不是实体产线的可靠性指标**。

## 实体迁移需要补齐的工作

轨迹文件提供 SI 单位的末端坐标和关节目标，便于接入实际工作站的软件流程；其中的关节值属于此仿真 URDF，不可直接作为实体设备命令。应用到实体工位前，至少需要：

1. 用真实机器人和工具 URDF 替换模型，标定基座、相机、输送带、料箱和工具坐标系；用现场测量更新质量、摩擦、障碍与工件分布。
2. 用实际驱动器限制速度、加速度、力矩和制动距离，验证连续轨迹、自碰撞、工作空间、急停及人员进入后的安全动作；由独立安全控制系统执行联锁。
3. 用视觉或传感器反馈确认包裹位姿、吸附成功和落箱结果，加入抓取失败、掉件、遮挡和工位偏移后的恢复流程。
4. 在真实数据上做离线回放、扰动与极端情况测试，再在隔离的实体工位以低速逐步验证；按具体设备的安全规范完成验收。

此脚本只输出仿真轨迹与动画，不连接实体机器人控制器。
