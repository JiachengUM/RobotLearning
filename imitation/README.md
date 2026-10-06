# Reach imitation learning：采集专家示范

```bash
.venv/bin/python imitation/collect_demo.py --episodes 500 --output imitation/reach_dataset.npz
```

依赖沿用项目虚拟环境中的 MuJoCo、NumPy、Matplotlib。不训练神经网络。输出文件已存在时拒绝覆盖，重新采集请换 `--output`。可用 `--episodes 2000` 增大数据量。

## 控制与数据

复用 `mujoco/reach.xml`。每回合从随机关节角、零速度开始；目标在 XY 平面半径 0.4–1.8m 圆环内按面积均匀采样。专家逐步计算 `xdot=2*(target-ee)`，用 DLS `qdot=J.T @ solve(J@J.T + 0.05**2 I, xdot)` 近似伪逆，再将最大关节速度限制为 1.5rad/s。

动作频率 50Hz，每个动作保持 10 个 0.002s 物理步。每物理步执行 `ctrl += action*timestep`，由 XML 的 position actuator 驱动物理机械臂。因此 action 是关节目标速度，不是力矩，也不是直接设置实际关节速度。

每条样本保存动作执行前 `obs`、实际使用的 `action`、执行后 `next_obs`，及回合结束标记。每个控制步都记录；不是每个底层物理步都记录。

| 数组 | 形状 / 含义 |
|---|---|
| obs、next_obs | (N,10)，依次为 qpos[2]、qvel[2]、ee_xy[2]、target_xy[2]、ctrl[2] |
| action | (N,2)，两关节目标速度 rad/s |
| done | (N,)，terminated 或 truncated |
| terminated | 成功：位置误差 < 1cm 且速度范数 < 0.03rad/s，连续 10 个控制步 |
| truncated | 达到 15s 超时且未成功 |
| episode_id | 每条 transition 所属回合 |
| episode_offsets | 回合边界索引，长度 E+1 |
| metadata | JSON 字符串，包含种子、版本、模型哈希、时步、观察顺序和专家参数 |

状态/动作保存 float32；NPZ 无 pickle。所有回合（包括超时）保留，避免隐藏专家失败。DLS 专家在奇异区域可能停滞，部分回合不能在时限内收敛；不要把超时视作成功示范。

## 读取与训练前分割

```python
from imitation.dataset import ReachDataset

ds = ReachDataset('imitation/reach_dataset.npz')
transition = ds[0]
episode = ds.episode(0)
train_ids, val_ids = ds.split_episodes(validation_fraction=0.2, seed=0)
# 只训练成功回合时，分别在每个 split 内筛选：
train_ids = [i for i in train_ids if ds.episode(i)['terminated'][-1]]
```

按回合分割，不要把同一轨迹的相邻帧随机分到训练与验证集。归一化参数也只从训练集估计。

## 验收概念

- **state**：环境用于推进仿真的内部状态。本例包括关节角、速度、目标、累计位置命令等；一般还可能有执行器内部状态。
- **observation**：提供给策略的状态信息。本例是上面的 10 维向量；包含 ctrl 是因为动作会累积到位置命令，下一步动力学依赖它。
- **action**：策略发给环境的控制输入。本例为 2 维关节目标速度，通过适配器转换成 ctrl。
- **trajectory**：按时间排列的 `(obs, action, next_obs, done)` 转移序列，可以是一个回合或其片段。
- **episode**：从 reset 开始，到成功或超时结束的一次完整尝试；本数据每回合对应一条完整轨迹。

`reach_dataset.png` 展示每回合的目标分布、控制步数和所有转移的动作直方图；长回合对动作直方图贡献更多。`reach_dataset.json` 给出成功/超时和样本统计。
