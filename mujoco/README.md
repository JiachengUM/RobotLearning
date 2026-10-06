# MuJoCo reaching 入门

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r mujoco/requirements.txt
# macOS 窗口模式（必须用 mjpython）
.venv/bin/mjpython mujoco/reach_env.py --episodes 10
# Linux / Windows：使用虚拟环境中的 python 运行同一脚本
# 无窗口验收
.venv/bin/python mujoco/reach_env.py --headless --episodes 20
```

蓝色、青色为两根连杆，黄色为末端，红色球为目标。每回合随机初始关节角和可达目标；到达并稳定 0.2 秒后切换下一回合。终端打印初态、目标、最终误差和状态。默认运行 5 回合后关闭，可关闭窗口提前退出。

## 控制链

`target → site_xpos（当前 EE 位置）→ mj_jacSite / DLS IK → joint target → 限速 data.ctrl → position actuator → mj_step`

`reach.xml` 定义两根 1m 连杆、两个绕 Z 轴旋转的关节、末端 site、目标球和位置 actuator。机械臂在 z=0.2m 水平面内运动，只控制 XY 位置，不控制末端朝向。没有避障、自碰撞或神经网络。

`controller.py` 在独立 MjData 上迭代 DLS IK，避免修改真实仿真状态。IK 从当前关节角开始，必要时使用弯曲初值重试。求解完成后，以每关节 1.5rad/s 的目标变化速度传给位置伺服器。真实机械臂依靠执行器力矩运动，只有 reset 会直接赋值真实 qpos。

`reach_env.py` 加载模型、随机初始化、按 timestep 更新控制和物理状态。目标按面积均匀采样自半径 0.4–1.8m 的圆环，避开两端奇异点；不宣称任意三维目标可达。

## 四个核心量

| 量 | 本模型中的含义 | 单位 |
|---|---|---|
| `data.qpos` | 实际肩关节、肘关节角度 | rad |
| `data.qvel` | 实际关节角速度 | rad/s |
| `data.ctrl` | 两个 position actuator 的目标关节角 | rad |
| `model.opt.timestep` | 每次 `mj_step` 推进的仿真时间，默认 0.002 | s |

`ctrl` 的含义取决于 actuator 类型：本例是位置目标；若换成 motor 则不再是目标角度。位置伺服器的力矩包含 `kp*(ctrl-qpos)-kv*qvel`，并受执行器力矩上限约束。`qpos` 不会在写入 `ctrl` 后瞬间变成目标值。

每回合验收要求 EE 误差 < 1cm，关节速度向量范数 < 0.03rad/s，持续 100 个物理步（0.2s）。超过 `--timeout`（默认 12 个仿真秒）仍不达标，会打印 TIMEOUT 并最终返回非零退出码。无窗口模式无需按真实时间等待；窗口模式用 sleep 尽量按实时速度播放。

本例两个关节不设角度限位，控制器选择相对当前角度最近的等价目标。关节位置/速度向量长度恰为 2；具有 free/ball joint 的其他模型中，qpos 和 qvel 长度可能不同。

官方参考：https://mujoco.readthedocs.io/en/stable/python.html （macOS passive viewer 需要 mjpython）。
