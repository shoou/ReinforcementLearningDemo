# DQN 强化学习案例：网格寻路

这个案例让智能体从左上角出发，避开深色障碍，走到右下角目标。它是一个**独立的 PyTorch 教学案例**，不依赖本目录原有的 Atari 训练脚本和 checkpoint。网格任务训练较快，方便观察 DQN 的核心机制。

## 运行

需要 Python 3.9+，以及 `torch`、`numpy`、`matplotlib`、`Pillow`：

```bash
python3 -m pip install torch numpy matplotlib pillow
python3 dqn_gridworld.py
```

训练默认 450 回合，图片与模型保存在 `dqn_output/`。运行后打开：

- `dqn_output/training.png`：每回合奖励、最近 30 回合成功率、探索概率。
- `dqn_output/demo.gif`：训练后的智能体逐步寻路动画。
- `dqn_output/route.png`：最终路线静态图。
- `dqn_output/dqn_model.pt`：训练得到的网络参数。

![DQN 训练曲线](dqn_output/training.png)

![DQN 寻路动画](dqn_output/demo.gif)

可用 `python3 dqn_gridworld.py --episodes 600 --seed 7 --output my_run` 调整回合数、随机种子和输出位置。DQN 训练有随机性；如果最后一回合贪心演示未到达目标，可增加回合数或更换种子。

## 任务与算法

环境是 5 × 5 固定地图。状态是智能体位置的 25 维 one-hot 向量；动作有上、下、左、右四种。到达目标奖励 `+2`，合法移动 `-0.04`，撞墙或越界 `-0.12`；到达目标或走满 30 步时回合结束。目标固定，因此状态中不需要另行编码目标位置。

```mermaid
flowchart LR
    E[网格环境<br/>当前状态 s] --> A[ε-greedy<br/>选择动作 a]
    A --> T[执行动作<br/>得到 r, s', done]
    T --> R[经验回放池<br/>保存转移]
    R --> B[随机抽取 64 条]
    B --> Q[在线网络<br/>预测 Q(s,a)]
    B --> Y[目标网络<br/>计算训练目标]
    Q --> L[Huber 损失<br/>更新在线网络]
    Y --> L
    L --> C[每 150 次更新<br/>复制参数到目标网络]
    C --> E
```

核心更新目标是

`y = r + γ(1 - done) max_a' Q_target(s', a')`

其中 `γ = 0.95`。在线网络输出四个动作的 Q 值；目标网络定期同步，减少训练目标的快速波动。经验回放随机抽样，减弱连续轨迹的相关性。`ε` 从约 1 逐渐降到 0.05，让智能体先探索，再更多地利用已学策略。演示阶段使用 `ε = 0`，即始终选择预测 Q 值最大的动作。

图中浅蓝线是单回合奖励，深蓝线是 30 回合移动平均；绿色曲线接近 1，表示近期大部分回合能到达目标。这个小任务适合讲解 DQN；若要处理 Atari 图像，需要把状态输入和网络改为图像预处理与卷积网络，并使用更多训练步数。
