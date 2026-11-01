# 强化学习教学案例

每个案例的脚本、说明和相关数据放在同一个目录。进入对应目录后，按其中说明运行；脚本的默认输出也会留在该案例目录。

| 案例 | 内容 | 入口 |
| --- | --- | --- |
| [Atari Q-learning](q_learning_atari/Q-learning_Atari_扩展说明.md) | Freeway、Breakout 的表格型 Q-learning | `q_learning_atari/atari_1step_qlearning.py` |
| [DQN 网格寻路](dqn_gridworld/DQN_案例说明.md) | 经验回放与目标网络 | `dqn_gridworld/dqn_gridworld.py` |
| [Actor-Critic 智能信号灯](actor_critic_traffic/Actor-Critic_案例说明.md) | 动态车流控制及交互式展示 | `actor_critic_traffic/actor_critic_traffic.py` |
| [PPO 冷链调度](ppo_cold_chain/PPO_案例说明.md) | 电池、制冷与车辆充电联合决策 | `ppo_cold_chain/ppo_cold_chain.py` |
| [大模型售后决策](llm_refund/强化学习_大模型售后决策案例.md) | 单步强化学习微调及网页对比 | `llm_refund/rl_llm_refund.py` |
| [机械臂分拣](robot_arm_rl/README.md) | 机械臂动作控制与可视化 | `robot_arm_rl/arm_sorting.py` |

例如，运行信号灯案例：

```bash
cd actor_critic_traffic
python3 actor_critic_traffic.py
python3 actor_critic_traffic_web.py
```

通用参考资料和旧环境依赖备份位于 [`references/`](references/)；`AGENTS.md` 和 `.gitignore` 保留在项目根目录。
