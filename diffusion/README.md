# 2D Diffusion Policy，从零实现

```bash
python diffusion/2d.py
```

依赖：`torch numpy matplotlib pillow`。默认在 CPU 上训练 5000 次更新，输出到 `diffusion/outputs/`。

## 数据

- `o=+1`：等概率抽取 `(-2, 2)` 或 `(2, 2)`，再添加协方差为 `0.1 I` 的高斯噪声。
- `o=-1`：中心变成 `(-2, -2)` 和 `(2, -2)`。
- `a` 是二维动作向量；这里用一个动作作为最小 policy 示例，没有动作序列或环境执行。

## 网络与训练

`NoiseMLP` 是恰好三层 Linear：`35 → 128 → 128 → 2`，前两层使用 SiLU。
输入包括二维 noisy action、32 维固定 sin/cos 时间特征，以及一维 observation。

1. 从 dataset 取 `(a⁰, o)`，随机选择扩散时刻 `k`，采样 `ε ~ N(0,I)`。
2. 手写前向加噪：`aᵏ = sqrt(ᾱk) a⁰ + sqrt(1-ᾱk) ε`。
3. 最小化 `mean((εθ(aᵏ,k,o) - ε)²)`。

`Diffusion` 手写 cosine noise schedule 和 DDPM 反向采样：

```text
μθ = (aᵏ - βk / sqrt(1-ᾱk) * εθ(aᵏ,k,o)) / sqrt(αk)
aᵏ⁻¹ = μθ + sqrt(posterior_variance[k]) * z
```

最后一步不加噪声。代码数组索引 `0...T-1` 对应数学时刻 `1...T`。

## 亲眼看去噪

- `outputs/denoising.gif`：同一批 `N(0,I)` 点逐步变成两个 mode。上下条件使用相同初始点和相同采样随机数；短线跟踪前 12 个点最近 18 步的移动。颜色按 `o=+1` 的最终左右归属固定，用于追踪点的身份。
- `outputs/summary.png`：反向过程的 0%、50%、80%、100% 快照；最后一列橙色点是训练数据。
- `outputs/loss.png`：噪声预测 MSE 的 100 步移动平均。
- `outputs/trajectories.npz`：两个条件的完整去噪数组，shape 为 `(T+1, samples, 2)`，第一帧是初始噪声，最后一帧是生成动作。
- `outputs/policy.pt`：网络权重和运行配置。

这不是一个将每个点平滑移动到最近中心的确定性映射：DDPM 中间步骤会添加随机噪声。去噪路径也不是绕障碍的机器人运动轨迹；这个例子学的是具有两个可选动作 mode 的条件分布。

可调参数示例：

```bash
python diffusion/2d.py --train-steps 8000 --samples 1000 --seed 42
```
