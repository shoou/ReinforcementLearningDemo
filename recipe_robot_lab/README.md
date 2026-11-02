# 配方编译 × 机械臂实验 × 深度强化学习

这是一个可运行的**纯仿真**柠檬酸—柠檬酸钠缓冲液案例。受限中文编译器把实验说明转成带物料、容量和工位约束的动作协议；PyBullet 中的 Franka Panda 搬运样品瓶；模拟加液器、混合器和 pH 计逐步修改样品状态并返回记录。加液 DQN 依据 pH 读数补液，运动 DQN 选择工位间安全路点，高斯过程依据整轮实验评分提出下一组配方和混合时长。

所有化学数据、仪器数据与标定数据均为合成数据。此案例没有接入真实机械臂或湿实验，也没有把 Isaac Lab 策略用于主闭环。

## 低配置仿真平台选择

| 平台 | 本机状态 | 本案例用途 | 边界 |
| --- | --- | --- | --- |
| PyBullet | macOS arm64 已运行 | 默认机械臂、抓取、工位接触和相机画面 | 仪器与液体仍由数字孪生模拟 |
| MuJoCo | macOS arm64 已运行可选核验 | 官方 Panda MJCF 的运动学复核，以及带固定 80 g 负载的执行器与机械臂工位接触回放 | 样品瓶碰撞关闭；未模拟夹爪握力和液体动力学 |
| Webots | 未在本机安装 | 若以后需要完整图形化工作站，可再评估 | 完整 GUI 对低配置设备通常更重 |
| Isaac Lab | 当前 Mac 不支持 | 保留可选任务源码供支持的 GPU 工作站使用 | 当前没有运行时验证 |

选择依据：[MuJoCo 官方平台支持](https://mujoco.readthedocs.io/en/latest/programming/)、[官方 Panda 模型](https://mujoco.readthedocs.io/en/stable/models.html)、[Webots 平台概览](https://cyberbotics.com/)、[Isaac Lab 安装要求](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)。这些平台都不能替代真实设备标定和湿实验数据。

## 运行主闭环

需要 Python 3.10+、NumPy、SciPy、PyTorch、PyBullet 和 Pillow。依赖范围见 [`requirements.txt`](requirements.txt)。在本项目的 macOS Apple Silicon 环境，`run.sh` 可自动选用已安装 PyBullet 的 Python 3.13，无需直接调用可能不存在的 `python3.13` 命令：

```bash
bash recipe_robot_lab/run.sh
```

其他环境可安装 `numpy scipy torch pillow`，macOS Apple Silicon 安装 `pybullet-arm64`，Linux x86-64 安装 `pybullet`。两种物理包都提供 `import pybullet`。示例：

```bash
bash recipe_robot_lab/run.sh --rounds 8 --seed 7 --output /tmp/recipe_run
bash recipe_robot_lab/run.sh --rounds 4 --fault-op measure:1 --output /tmp/recipe_fault
```

`--gui` 显示 PyBullet 窗口；`--rl-episodes`、`--motion-episodes` 和 `--validation-repeats` 控制训练与重复次数。`--eval-training-seeds` 与 `--eval-trials` 控制压力评估，默认用 3 个独立补液 DQN 训练种子，每场景 120 组配对试验。默认做 6 次优化实验和最佳候选的 3 次独立模拟重复。输出目录包含 `report.html`、`motion_replay.gif`、`environment.json`、`protocol.json`、`experiments.jsonl`、`validation.jsonl`、`summary.json`、`policy_evaluation.json`、`scene.png`、两个 DQN 权重，以及 `synthetic_calibration.csv` 和 `calibration_model.json`。协议执行异常会尝试安全停机并记录；安全停机本身失败时终止后续实验。失败轮不进入高斯过程拟合。

`--text` 接受限定格式的中文说明，需各写一次终体积、目标 pH、两种母液浓度和混合秒数。编译器仅支持该缓冲体系；遇到额外试剂、加热等未识别操作会拒绝整段说明，而不会默默跳过。示例输入见 `compiler.py` 的 `DEFAULT_TEXT`。它是确定性受限编译器，不是通用语言模型。

## 执行与反馈

1. `compiler.py` 编译 `identify → grasp → move → dispense → mix → measure → adjust → top_up → remix → measure → hold → measure → release`，并检查体积、浓度和工位顺序。
2. `driver.py` 实现 `LabDriver` 接口。`SimLabDriver` 从空瓶开始，只有收到加液命令才改变样品；加液后必须重新混合才能测量。注入仪器故障时，机械臂停止并保留故障轨迹。
3. `arm_sim.py` 在 PyBullet 中执行 Panda 逆运动学和关节位置控制，使用固定约束模拟握持。工位有碰撞几何，记录路点误差、工位接触、动作耗时与运动策略回退次数。运动 DQN 选工作区路点；异常路线回退到图搜索。另在 84 组禁行路点规划挑战中测示范预热与 RL 更新的差异；这些禁行路点尚未作为实体障碍加入主工作站。
4. `twin.py` 给出可标定的教学 pH 代理模型、移液误差、传感器噪声、混合状态和随时间漂移。pH 计采集至多 8 次，最近 3 次范围不超过 0.015 才接受读数；终测后等待 10 分钟再次测量。
5. `calibration.py` 生成 80 条合成标定记录，按 64/16 划分训练集和留出集，拟合代理模型并报告留出集 RMSE。该指标只衡量合成数据拟合。
6. `dose_rl.py` 训练补液 DQN，输出停止或追加母液；`motion_rl.py` 先用最短路示范预热，再用 DQN 更新工位运动策略；`run_case.py` 用高斯过程期望改进量选下一组酸/盐体积和混合时长。每轮原始读数、动作 Q 值、机械臂轨迹及评分均写入 JSONL。
7. `evaluation.py` 比较 DQN、固定步长、模型一步预测和不调整；压力评估覆盖训练分布、配方边界、代理模型参数及未建模曲率与仪器偏移、组合偏移。每场景使用 3 个独立训练种子，并报告命中率区间、pH 误差与用液量。
8. `mujoco_check.py` 可选地做 Panda 跨引擎运动学与带负载的关节动力学回放；`render_replay.py` 根据记录姿态制作展示动画；`environment.json` 保存运行环境、参数与源码哈希。

pH 代理式只适合展示闭环软件结构。柠檬酸为三元酸，代码中的近似式不预测真实缓冲液 pH；合成标定、参数偏移压力试验和重复实验也不构成真实化学验证。模型一步预测基线与 DQN 均知道同一代理模型的形式，因此比较只涉及该受控仿真任务。

## 可选的 MuJoCo 交叉核验

在与运行脚本相同的 Python 环境安装 `mujoco`，并获取 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda) 官方 Panda 模型；模型文件及 `assets/` 须保持在同一目录结构。可用 Git 的稀疏检出减少下载：

```bash
git clone --filter=blob:none --sparse --depth=1 https://github.com/google-deepmind/mujoco_menagerie.git /tmp/mujoco_menagerie
git -C /tmp/mujoco_menagerie sparse-checkout set franka_emika_panda
python -m pip install mujoco
bash recipe_robot_lab/run.sh --mujoco-model /tmp/mujoco_menagerie/franka_emika_panda/panda.xml
```

若 `run.sh` 因当前环境缺少 PyBullet 而切换 Python，请把 MuJoCo 安装到它实际选用的 Python，或按环境配置 `PYTHONPATH`。可单独运行 `python recipe_robot_lab/mujoco_check.py --model /path/to/panda.xml` 来核验已有实验日志。通过时生成 `mujoco_audit.json`：静态复核所有记录姿态；动力学部分以 0.6 rad/s 的控制目标斜率重放带固定 80 g 负载的 Panda 关节控制，检查末端误差、机械臂与工位穿透、速度与力矩到限比例。样品瓶自身的碰撞在该回放中关闭。当前回放的力矩到限比例偏高，不能据此宣称可直接在真机执行。可选依赖见 [`requirements-mujoco.txt`](requirements-mujoco.txt)。

## 与最初方案逐项对照

| 环节 | 当前实现 | 仍需真实条件才能完成 |
| --- | --- | --- |
| 自然语言配方编译 | 受限中文语法、显式单位和未知操作拒绝 | 开放文本解析与实验人员审核 |
| 具身执行 | Panda 三维仿真、抓瓶约束、工位碰撞、DQN 路点规划和协议异常停机 | 真实抓力、设备几何、传感器与控制接口 |
| 物理平台 | 低配置 PyBullet 主闭环、MuJoCo 官方 Panda 运动学与带负载关节动力学回放 | 实体夹持、设备碰撞与跨平台力控校准 |
| 化学测量 | 合成标定、稳定读数、10 分钟复测及原始数据记录 | 真实 pH 电极和多批次实验数据 |
| 强化学习 | 补液 DQN、运动 DQN、四场景多训练种子压力评估、禁行路点消融和动作 Q 值日志 | 真实数据与实体障碍上的策略验证 |
| 闭环优化 | 高斯过程逐轮提议、失败轮排除、最佳配方独立重复 | 真实实验预算、重复性与不确定度评估 |

## Isaac Lab 可选任务

`isaac_lab/` 提供独立的 `RecipeLab-Dock-Franka-Direct-v0` 任务：Panda 关节动作、4 个带碰撞几何的实验工位、目标位置与关节初态扰动、动作及观测噪声、RSL-RL PPO 配置。这一任务训练的是**机械臂末端工位靠近技能**；不包含抓瓶、液体、仪器通讯或主闭环的配方优化。它与 PyBullet 主案例并列，尚未实现策略权重跨物理引擎迁移。

当前 macOS 机器没有可用的 Isaac Lab 运行环境，因此只做了源码静态检查，未做 Isaac Lab 启动、训练或收敛验证。支持的平台与运行方式见 [Isaac Lab 安装文档](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html)；可选任务的安装和尝试命令见 [`isaac_lab/README.md`](isaac_lab/README.md)。

## 验证

在本项目已配置好的环境运行：

```bash
PYTHONPATH=recipe_robot_lab:robot_arm_rl/.venv-physics/lib/python3.13/site-packages \
  /Users/albert/miniconda3/bin/python3.13 -m unittest discover -s recipe_robot_lab -p 'test_*.py' -v
```

测试覆盖受限文本编译、工位顺序、空瓶加液及混合门槛、合成标定留出集、运动策略覆盖、异常安全停机和完整仿真闭环。实体接入需要另外完成设备 SDK 适配、坐标与体积标定、碰撞和急停联锁、以及真实 pH 数据验证。
