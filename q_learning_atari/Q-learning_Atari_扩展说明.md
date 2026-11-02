# Atari 表格型 Q-learning：Freeway 与 Breakout

脚本：[Freeway](freeway_1step_qlearning.py) · [Breakout](breakout_1step_qlearning.py)

## 运行

以下命令在 `q_learning_atari/` 目录执行。

```bash
python -m pip install "gymnasium[atari]" ale-py numpy

# Freeway 沿用原来的模型文件
python freeway_1step_qlearning.py --episodes 300
python freeway_1step_qlearning.py --mode eval --render --eval-episodes 1

# Breakout 使用独立的模型文件
python breakout_1step_qlearning.py --episodes 500 --steps 1000
python breakout_1step_qlearning.py --mode eval --render --eval-episodes 1
```

默认模型分别保存到 `qlearning_output/freeway_qtable.npz` 和
`qlearning_output/breakout_qtable.npz`。运行 `--mode eval` 前需要先训练对应游戏。
`--difficulty 0|1`、`--steps` 和探索率等参数对两个游戏都适用。
`--eval-episodes` 是每种策略评估的回合数；每回合最多执行 `--steps` 步。
`--render` 只实时显示学习策略，另外两个基线在后台运行。打印的评估得分始终使用游戏原始奖励。

## 两个游戏分别做了什么

| 部分 | Freeway | Breakout |
| --- | --- | --- |
| 环境 | `ALE/Freeway-v5` | `ALE/Breakout-v5` |
| 动作 | NOOP、UP、DOWN | NOOP、FIRE、RIGHT、LEFT |
| RAM 状态 | 小鸡高度；附近车距离 | 球与挡板的横向距离；球的高度区域和运动方向 |
| 辅助训练奖励 | 向上移动 | 球靠近挡板时，挡板追近球 |
| 必要开局动作 | 无 | 未开球或掉球后自动 FIRE |
| 对照策略 | 始终向上 | 根据球横坐标追球 |

两个独立脚本采用同一条 Q 值更新公式、ε-greedy 探索和训练/评估流程。
状态抽象有意压缩信息，让 Q 表可以用于课堂演示；它们不是像素级 Atari
深度强化学习的性能基准。Breakout 的球路和砖块布局较复杂，即使能运行，
也不保证表格型 Q-learning 超过追球基线。评估时应比较原始得分，
不要把辅助训练奖励当成最终成绩。

## 扩展到第三个 Atari 游戏

1. 在 [ALE 官方游戏列表](https://ale.farama.org/environments/) 中确认环境 ID、
   最小动作集合、是否需要 FIRE 开局，以及难度选项。
2. 以较接近的新游戏脚本为起点，另建一个脚本并设置 `GAME`、`ACTIONS`、
   `DEFAULT_MODEL` 和 `Q_SHAPE`。每个游戏使用自己的模型文件，避免加载错 Q 表。
3. 读取游戏 RAM 文档或可靠的对象提取实现，在 `encode_state()` 中选取
   少量与决策相关的特征并离散化。可参考
   [OCAtari 的 RAM 提取代码](https://github.com/k4ntz/OC_Atari/tree/master/ocatari/ram)
   和 [AtariARI 的 RAM 标注](https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py)。
   需要速度等动态特征时，使用 `previous_ram`；检查每个状态维度的取值
   都落在 `Q_SHAPE` 内。
4. 在 `training_reward()` 中决定是否添加小幅辅助奖励。真实环境奖励仍
   保留在训练目标里；`evaluate()` 只累计真实得分。
5. 如需自动开局，参考 Breakout 脚本的 `forced_action()`；
   给新游戏设置简单对照策略。
6. 先用很短的训练验证环境能启动、状态索引合法、模型能保存和加载，
   然后增加回合数；与随机及简单规则基线比较真实得分。若离散状态过多、
   长训练仍无明显改善，可以考虑函数近似或 DQN，而不是无限增大 Q 表。

不能只修改 `GAME` 常量：不同游戏的 RAM 含义、动作索引、开局方式和
奖励结构都不同。例如 Pong 的最小动作数是 6，直接套用 Freeway 的状态
和 Q 表会得到错误的训练含义。

参考：[Freeway 官方文档](https://ale.farama.org/environments/freeway/)、
[Breakout 官方文档](https://ale.farama.org/environments/breakout/)。
