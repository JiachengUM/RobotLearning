同一个 observation 下，机器人可能有不止一种合理动作。Diffusion Policy 要学习的是这些动作的条件分布，再根据当前观测从中生成动作。这里不要把问题理解成“给定 observation，回归出一个 action”，我们想建模的是一个动作的条件概率分布。

我们从一个双峰二维动作分布出发，依次理解前向加噪、反向生成、噪声预测训练，再对照 [2d.py](/Users/jiachexu/Code/robot/diffusion/2d.py) 看完整实现，最后扩展到真实机器人的动作序列与闭环控制。

## 记号约定

全文与当前代码统一使用 DDPM 的 notation：


| 记号          | 含义                         | 代码对应                                |
| ----------- | -------------------------- | ----------------------------------- |
| x₀          | 真实数据；本例中是一个干净二维动作          | `x_0`                               |
| xₜ          | 加噪到扩散时刻 t 的变量              | `x_t`                               |
| xₜ（t=T）     | 最高噪声水平下的变量，近似标准 Gaussian   | 采样入口 `initial`                      |
| t ∈ {1,…,T} | diffusion timestep，即噪声等级   | 训练时的 `t`、采样时的 `timesteps`           |
| T           | 扩散总步数                      | `self.steps`，默认 100                 |
| o           | observation / condition    | 本例中是 `-1` 或 `+1`                    |
| ε           | 前向闭式加噪时采样的标准高斯噪声           | `epsilon`                           |
| εθ(xₜ,t,o)  | 网络预测的噪声                    | `NoiseMLP`，采样中输出为 `epsilon_hat`     |
| βₜ,αₜ,ᾱₜ   | 单步方差、单步信号系数、累计信号系数         | `beta[t]`、`alpha[t]`、`alpha_bar[t]` |
| β̃ₜ         | 已知 x₀ 时的前向后验方差；本例也用作反向模型方差 | `posterior_variance[t]`             |


代码中的 schedule 数组长度为 **T+1**，直接用数学时刻 `t` 索引。`beta[0]=0` 是占位，`alpha[0]=alpha_bar[0]=1`。训练抽取 `t=1,...,T`，采样循环按 `t=T,...,1` 执行。

扩散时刻 t **不是机器人控制时间**。机器人真实执行动作的时刻另记为 τ。x₀ 的下标 0 表示“未加扩散噪声”，不表示机器人执行的第一个动作。

---



# 一、为什么要学习动作分布？



## 同一个 observation，可以有两个正确动作

假设相同 observation 下，示教动作集中在 (-2,2) 和 (2,2) 两侧。用普通网络回归一个动作：

$$
\hat x_0=f_\theta(o),\qquad
\mathcal L_{\mathrm{MSE}}=\mathbb E\|x_0-f_\theta(o)\|^2.
$$

对固定 o，理想 MSE 解为条件均值：

$$
f^*(o)=\mathbb E[x_0\mid o].
$$

两个 mode 对称且权重各半时，f*(+1)=(0,2)。这个中间点可能处于示教分布的低密度区域；在具有左右绕行策略的任务中，平均动作也可能不可行。

```text
        左侧 mode        MSE 均值        右侧 mode
          (-2,2)           (0,2)           (2,2)
             ●               ×               ●
```

这里画的是动作空间。二维 toy 本身没有建立障碍物、动力学或碰撞约束。

单点回归在单峰、确定性任务中可以非常有效。这里的问题是：**当行为天然多峰时，一个条件均值不足以描述所有合理动作。**

## 从预测一个点，到学习条件分布

目标从

$$
o\mapsto\hat x_0
$$

变成

$$
o\mapsto p_\theta(x_0\mid o).
$$

本例的数据分布为：

$$
\boxed{
p_{\mathrm{data}}(x_0\mid o)
=\frac12\mathcal N\!\left(x_0;(-2,2o)^\top,0.1I\right)
+\frac12\mathcal N\!\left(x_0;(2,2o)^\top,0.1I\right).
}
$$

- o=+1：上方两个 mode，中心为 (-2,2)、(2,2)。
- o=-1：下方两个 mode，中心为 (-2,-2)、(2,-2)。
- 模型看到 o，没有看到决定左右的隐藏变量 `side`。

我们希望每次从 pθ(x₀ ∣ o) 中采样一个动作，多次采样后覆盖两个 mode，而不是都输出中间点。

**最终目标是分布建模；噪声回归是训练这个生成模型的方法。**

## Diffusion：把复杂分布连到简单 Gaussian

```text
双峰数据分布                                 标准 Gaussian
    ●    ●       ── forward：逐步加噪 ──→        ·····
    ●    ●       ←── reverse：学习生成 ──        ·····
```

标准 Gaussian 很容易采样。我们先规定一个前向过程，让数据逐步失去结构、接近 Gaussian；再学习反向生成过程，从随机噪声出发生成数据。

[DDPM](https://arxiv.org/abs/2006.11239) 是一般生成模型。将生成对象设为受 observation 条件约束的动作，就能得到本例的条件动作生成模型。

---



# 二、Forward：数据如何变成噪声？



## 单步加噪是一个线性 Gaussian 过程

定义：

$$
\boxed{
q(x_t\mid x_{t-1})
=\mathcal N\bigl(x_t;\sqrt{\alpha_t}x_{t-1},\beta_tI\bigr),
\qquad \alpha_t=1-\beta_t.
}
$$

等价的采样形式为：

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I).
$$

√αₜ 略微缩小原信号，√βₜεₜ 注入噪声。这两个系数共同控制分布演化。

q 是人为规定的前向过程，不需要神经网络学习。

## 为什么可以直接从 x₀ 得到任意 xₜ？

展开两步：

$$
\begin{aligned}
x_1&=\sqrt{\alpha_1}x_0+\sqrt{\beta_1}\epsilon_1,\\
x_2&=\sqrt{\alpha_2}x_1+\sqrt{\beta_2}\epsilon_2\\
&=\sqrt{\alpha_1\alpha_2}x_0
+\sqrt{\alpha_2\beta_1}\epsilon_1+\sqrt{\beta_2}\epsilon_2.
\end{aligned}
$$

独立 Gaussian 的线性组合仍是 Gaussian。后两项的协方差为：

$$
(\alpha_2\beta_1+\beta_2)I=(1-\alpha_1\alpha_2)I.
$$

定义累计信号系数：

$$
\bar\alpha_t=\prod_{s=1}^t\alpha_s,\qquad \bar\alpha_0=1.
$$

推广得到闭式加噪公式：

$$
\boxed{
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
}
$$

这里的 ε 表示合并后的标准 Gaussian，不是逐步加噪过程中某一特定步的 εₜ。

训练时可以随机抽一个 t，直接构造对应的 xₜ，不用真的模拟 t 次加噪。

## β、α、ᾱ 和 noise schedule

$$
\beta_t\in(0,1),\qquad \alpha_t=1-\beta_t,\qquad
q(x_t\mid x_0)=\mathcal N\bigl(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I\bigr).
$$


| 量                        | 直觉              |
| ------------------------ | --------------- |
| βₜ                       | 本步注入多少噪声        |
| αₜ                       | 本步保留多少信号系数      |
| ᾱₜ                      | 累计保留多少原始信号系数    |
| SNR(t) = ᾱₜ / (1 − ᾱₜ) | 当前噪声水平下的信噪比，t>0 |


原始 DDPM 使用过 linear beta schedule。当前代码使用 cosine schedule，这一设计来自后续的 [Improved DDPM](https://proceedings.mlr.press/v139/nichol21a.html)。

代码先构造归一化 cosine 曲线，再用相邻比值计算 βₜ，并限制最大值为 `0.999`，避免 αₜ=0 导致反向公式除以零。实际的 `alpha_bar` 由裁剪后的 `alpha` 累乘得到。

## 为什么终点近似 N(0,I)？

$$
x_T=\sqrt{\bar\alpha_T}x_0+\sqrt{1-\bar\alpha_T}\epsilon.
$$

当 终点的 ᾱₜ（t=T） 足够小时，数据项很弱，因此：

$$
x_T\approx\epsilon\sim\mathcal N(0,I).
$$

有限 T 下通常是近似相等。接近程度取决于 schedule、终点信噪比以及数据尺度。

前向过程回答了“如何把数据变成噪声”。生成时只有一个新采样的 xₜ（t=T），还需要知道如何反着走。

---



# 三、Reverse：如何从噪声生成数据？



## 反向条件分布不能直接把前向公式倒写

已知的是 q(xₜ ∣ xₜ₋₁)，生成需要的是 q(xₜ₋₁ ∣ xₜ)。

同一个 noisy point 可能来自多个不同的前一步状态。它们各自有多大可能，取决于原始数据分布。例如 y=x+ε，只知道 y=2，无法唯一恢复 x。

反向过程需要学习数据分布提供的统计信息，而不是对一次加噪做普通代数求逆。

## 已知 x₀ 时，后验可以解析求出

训练时，我们知道真实样本 x₀。加入这个条件：

$$
q(x_{t-1}\mid x_t,x_0)
=\frac{q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}.
$$

这里用到了前向过程的 Markov 性质：

$$
q(x_t\mid x_{t-1},x_0)=q(x_t\mid x_{t-1}).
$$

由于右边相关分布均为 Gaussian，t>1 时可以得到 Gaussian 后验；t=1 是退化到已知 x₀ 的零方差端点。

## Exact posterior 的 mean 和 variance

$$
q(x_{t-1}\mid x_t,x_0)
=\mathcal N\bigl(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_tI\bigr),
$$

其中：

$$
\boxed{\tilde\beta_t=\beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}
$$

$$
\boxed{
\tilde\mu_t(x_t,x_0)
=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
}
$$

若 x₀ 已知，就知道往回一步应使用哪个 Gaussian。但在生成时，x₀ 恰好是未知的目标，需要网络提供去噪信息。

`posterior_variance[t]` 来自这里的解析公式，不是网络训练出来的参数。

## 为什么预测噪声可以得到 reverse mean？

由闭式加噪公式，用网络预测噪声后，可以构造：

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t,o)}{\sqrt{\bar\alpha_t}}.
$$

将它代入后验均值并整理：

$$
\boxed{
\mu_\theta(x_t,t,o)
=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t,o)\right).
}
$$

所以预测噪声的用途，是计算反向生成分布的均值。

不能简单写成 xₜ₋₁=xₜ-εθ：网络预测的量是前向闭式公式中的标准 Gaussian noise，不是相邻扩散时刻之间的位移。

## 反向过程是一系列条件采样

先采样：

$$
x_T\sim\mathcal N(0,I).
$$

然后按 t=T,T-1,…,1 更新：

$$
\boxed{x_{t-1}=\mu_\theta(x_t,t,o)+\sigma_tz.}
$$

本例选择 σₜ²=β̃ₜ。当 t>1 时采样 z ∼ N(0,I)；当 t=1 时不再添加噪声：

$$
\tilde\beta_1=\beta_1\frac{1-\bar\alpha_0}{1-\bar\alpha_1}=0.
$$

```text
x_T → 预测 ε → 计算 mean → 采样 x_{T-1} → … → x_0
```

这是当前 DDPM 实现的方差选择。其他 diffusion sampler 可以采用不同的采样规则。

---



# 四、Training：为什么训练看起来只是 MSE？



## 从数据似然到变分目标

生成模型希望提高真实数据的对数似然：

$$
\max_\theta\ \mathbb E_{x_0\sim p_{\mathrm{data}}}\log p_\theta(x_0).
$$

对于包含隐变量 (x₁,…,xₜ)，末端 t=T 的模型：

$$
p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T}.
$$

直接计算通常很难。DDPM 引入已知的前向分布 q((x₁,…,xₜ) ∣ x₀)，其中 t=T 来构造对数似然的变分下界 ELBO。

因此，噪声预测目标与生成模型的概率训练目标有明确联系。

## ELBO 中的核心：匹配反向后验

负 ELBO 中的中间项包含：

$$
D_{\mathrm{KL}}\left(
q(x_{t-1}\mid x_t,x_0)
\;\|\;
p_\theta(x_{t-1}\mid x_t)
\right),\qquad t>1.
$$

左边是训练时可解析求出的后验，右边是学习的反向转移。使用 Gaussian 参数化、固定方差后，这些项中关于均值的优化可以写成带 timestep 权重的噪声预测 MSE。

条件生成时，将反向模型换为 pθ(xₜ₋₁ ∣ xₜ,o)。

完整负 ELBO 还包含端点项。当前代码使用常见的 **simplified objective**，没有实现完整 ELBO，也没有学习反向方差。

## Simplified objective：自己制造监督标签

$$
\boxed{
L_{\mathrm{simple}}
=\mathbb E_{x_0,o,t,\epsilon}
\left\|
\epsilon-\epsilon_\theta\left(
\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t,o
\right)
\right\|_2^2.
}
$$

训练样本由三部分构成：

1. dataset 提供真实的 (x₀,o)。
2. 均匀采样扩散时刻 t ∈ {1,…,T}。
3. 自己采样 ε ∼ N(0,I)，生成 xₜ。

网络输入是 (xₜ,t,o)，监督标签是刚刚采样的 ε。不需要额外人工标注“正确去噪方向”。

这个未加 timestep 权重的 MSE 不等于任意设定下的完整负 ELBO。

---



# 五、对照二维代码：从 dataset 到生成动画

以下代码片段均对应当前 [2d.py](/Users/jiachexu/Code/robot/diffusion/2d.py) 的实际实现；片段中的 `self`、`args` 等由所在函数提供。

## Toy world：二维动作与隐藏 mode

数据生成规则为：

$$
o,s\in\{-1,+1\},\qquad
\boxed{x_0=(2s,2o)^\top+\sqrt{0.1}\xi,\quad\xi\sim\mathcal N(0,I_2).}
$$

`side` 对应 s，与 o 独立采样。对于给定 o，对隐藏变量 s 求和后，得到左右各一半的 Gaussian mixture。

x₀ ∈ ℝ² 是一个二维动作向量。可以将它类比为 (Δx,Δy)，但 toy 没有定义真实物理控制语义，也没有动作序列。

用一维数据也能演示多模态；二维数据让上下条件与左右 mode 更容易同时观察。

## `make_dataset()`：为什么不把 `side` 给模型？

```python
o = (torch.randint(2, (n, 1), generator=generator) * 2 - 1).float()
side = (torch.randint(2, (n, 1), generator=generator) * 2 - 1).float()
means = torch.cat([2 * side, 2 * o], dim=1)
x_0 = means + math.sqrt(0.1) * torch.randn(n, 2, generator=generator)
return o, x_0
```

- `o.shape=(n,1)`，`side.shape=(n,1)`，`means.shape=(n,2)`。
- 协方差是 0.1I，因此噪声应乘标准差 √0.1。
- 函数返回顺序是 `o, x_0`，没有返回 `side`。

如果将 `side` 也作为 condition，给定 (o,s) 后只剩单个 Gaussian。隐藏它，才需要模型在同一个 o 下表示两个 mode。

## `NoiseMLP`：三个 Linear 层预测二维噪声

$$
\epsilon_\theta(x_t,t,o):\mathbb R^2\times\{1,\ldots,T\}\times\{-1,+1\}\to\mathbb R^2.
$$

```text
x_t            (B, 2) ─┐
time_embedding (B,32) ─┼─ cat → (B,35) → Linear(35,128) → SiLU
 o             (B, 1) ─┘               → Linear(128,128) → SiLU
                                        → Linear(128,2)
```

时间编码与输入拼接：

```python
phase = (t.float() / self.steps)[:, None] * self.frequencies[None, :]
time_embedding = torch.cat([phase.sin(), phase.cos()], dim=1)
return self.net(torch.cat([x_t, time_embedding, o], dim=1))
```

`[:, None]` 将 `(B,)` 变为 `(B,1)`；16 个频率扩展为 `(1,16)`，广播后得到 `(B,16)`。sin/cos 拼接成 32 维时间特征。

这里的 `*` 是逐元素乘法与 broadcasting。频率通过 `register_buffer` 保存，会随模型迁移设备，但不参与训练。

## `Diffusion`：schedule 与闭式加噪

当前实现为 schedule 增加第 0 项，直接对齐论文时刻：

```python
self.beta = torch.cat([torch.zeros(1, device=device), (1 - curve[1:] / curve[:-1]).clamp(max=0.999)])
self.alpha = 1 - self.beta
self.alpha_bar = self.alpha.cumprod(0)
previous = self.alpha_bar[:-1]
self.posterior_variance = torch.zeros_like(self.beta)
self.posterior_variance[1:] = self.beta[1:] * (1 - previous) / (1 - self.alpha_bar[1:])
```

`curve` 是已归一化的 cosine 曲线。由于 `beta[0]=0`，有 `alpha[0]=alpha_bar[0]=1`。后验方差只计算 `[1:]`，避免第 0 项出现 `0/0`。

`add_noise()` 直接实现闭式公式：

```python
alpha_bar = self.alpha_bar[t, None]
return alpha_bar.sqrt() * x_0 + (1 - alpha_bar).sqrt() * epsilon
```

`alpha_bar.shape=(B,1)`，可与 `(B,2)` 的动作广播。batch 内不同样本可以选择不同的 `t`，同一样本的两个坐标共享当前噪声强度。

## `train()`：输入、标签与参数更新

```python
x_0, o = actions[ids], observations[ids]
t = torch.randint(1, diffusion.steps + 1, (len(ids),), device=actions.device, generator=generator)
epsilon = torch.randn(x_0.shape, device=actions.device, generator=generator)
x_t = diffusion.add_noise(x_0, t, epsilon)
loss = (model(x_t, t, o) - epsilon).square().mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`torch.randint` 的上界不包含在取值范围中，所以 `steps + 1` 对应抽取 `1,...,T`。

训练时，干净动作 x₀ 和真实噪声 ε 都已知。例如：

$$
x_0=(-2.1,1.9),\quad \bar\alpha_t=0.64,\quad\epsilon=(0.5,-1),
$$

则：

$$
x_t=0.8(-2.1,1.9)+0.6(0.5,-1)=(-1.38,0.92).
$$

网络看到 (xₜ,t,o)，学习预测这次采样的噪声。对所有训练样本做 MSE 优化，理想预测器逼近条件期望 E[ε ∣ xₜ,t,o]。

默认配置：20000 个数据样本、5000 次参数更新、batch size 256、学习率 `1e-3`、100 个扩散时刻。**5000 次训练更新与 100 步反向采样是两个不同概念。**

## `sample()`：从终点噪声一步步生成 x₀

`initial` 是新采样的 Gaussian，不对应一个预存、等待恢复的真实动作。

```python
x_t = initial.clone()
o = torch.full((len(x_t), 1), float(observation), device=x_t.device)
for t in range(self.steps, 0, -1):
    timesteps = torch.full((len(x_t),), t, device=x_t.device, dtype=torch.long)
    epsilon_hat = model(x_t, timesteps, o)
    mean = (x_t - self.beta[t] / (1 - self.alpha_bar[t]).sqrt() * epsilon_hat) / self.alpha[t].sqrt()
    if t > 1:
        z = torch.randn(x_t.shape, device=x_t.device, generator=generator)
        x_t = mean + self.posterior_variance[t].sqrt() * z
    else:
        x_t = mean
```

每轮开头，`x_t` 保存当前 xₜ；更新后，它保存 xₜ₋₁。循环结束时，这个变量里就是生成的 x₀。

`sample()` 返回整个反向过程，shape 为 `(T+1, num_samples, 2)`：

- `history[0]` 对应 xₜ（t=T）。
- `history[j]` 对应 x₍T−j₎。
- `history[-1]` 对应生成的 x₀。

因此动画中 reverse step 从 `0` 增长到 `T`，同时 diffusion timestep 从 `T` 减少到 `0`。

## 看动画：一团 Gaussian 如何变成两簇？

从项目根目录运行：

```bash
.venv/bin/python diffusion/2d.py
```

输出位于 `diffusion/outputs/`：


| 文件                 | 内容                                             |
| ------------------ | ---------------------------------------------- |
| `denoising.gif`    | 两个 observation 下完整的反向去噪过程                      |
| `summary.png`      | 反向过程 0%、50%、80%、100% 的点云快照                     |
| `loss.png`         | 噪声预测 MSE 的移动平均曲线                               |
| `trajectories.npz` | `positive`、`negative` 两个条件下的完整去噪数组，以及 `losses` |
| `policy.pt`        | 模型权重、运行配置和训练 loss                              |


![从标准 Gaussian 到两个条件动作 mode](/Users/jiachexu/Code/robot/diffusion/outputs/denoising.gif)

初始点围绕原点分布。反向过程先逐渐显露条件结构，再收敛到左右两个 mode 附近。

代码为两个 observation 复用同一批初始噪声，并重置相同的采样随机种子，因此可以比较只改变 o 时生成过程的变化。蓝红颜色按 **o=+1 的最终左右归属** 固定，并用于两幅图追踪粒子身份；下方图的颜色不重新表示它自己的左右归属。

短线只显示前 12 个粒子最近 18 步的去噪路径。这些线是 **动作空间中的生成路径**，不是机器人执行轨迹。

## 对比条件均值与生成样本

单点 MSE 的理想预测为：

$$
\hat x_{0,\mathrm{MSE}}(+1)=\mathbb E[x_0\mid o=+1]=(0,2).
$$

Diffusion 的输出则是：

$$
x_{0,\mathrm{generated}}\sim p_\theta(x_0\mid o=+1).
$$

多次采样应在左右 mode 附近形成两簇。代码打印三个检查量：


| 检查量                  | 理想目标                        |
| -------------------- | --------------------------- |
| `left fraction`      | 约 0.5                       |
| `mode residual mean` | 相对最近 mode 中心的残差均值约为 `[0,0]` |
| `variance`           | 残差各坐标方差约为 `[0.1,0.1]`       |


这些量是有限样本下的诊断，不能替代完整的分布评估。loss 降低也不单独证明生成分布正确，需要结合点云与 mode 覆盖情况观察。

这里的 MSE baseline 是条件均值的解析对照，当前脚本没有另行训练一个动作回归器。Gaussian mixture、VAE、flow 等其他生成模型也可以表达多峰分布。

---



# 六、网络学到了什么？



## MSE 的最优解是条件期望

$$
\boxed{\epsilon^*(x_t,t,o)=\mathbb E[\epsilon\mid x_t,t,o].}
$$

一个 noisy location 可能对应多个不同的 clean origin。模型不可能对所有碰巧落在相近位置的样本，都还原其各自唯一的历史噪声。

它学习的是在给定 noisy point、噪声等级和 observation 后，统计意义上最合适的去噪信息。

## 与 score 的关系

令 pₜ(xₜ ∣ o) 表示真实数据经过前向加噪后的条件边际分布，其 score 为：

$$
s_t(x_t,o)=\nabla_{x_t}\log p_t(x_t\mid o).
$$

Gaussian perturbation identity 给出：

$$
\boxed{
\nabla_{x_t}\log p_t(x_t\mid o)
=-\frac{\mathbb E[\epsilon\mid x_t,t,o]}{\sqrt{1-\bar\alpha_t}}
\approx-\frac{\epsilon_\theta(x_t,t,o)}{\sqrt{1-\bar\alpha_t}}.
}
$$

对某一训练对 (x₀,ε)，-ε/√(1 − ᾱₜ) 是扰动核 q(xₜ ∣ x₀) 的 score。边际分布的 score 需要再对 x₀ ∣ xₜ,o 做条件期望。

所以，噪声预测器与当前噪声水平下的分布几何有关：不同位置的点会受到不同的去噪引导。

## 哪些是原理，哪些是实现选择？


| 问题                     | 答案                                                 |
| ---------------------- | -------------------------------------------------- |
| 每一步都必须添加新噪声吗？          | 当前 DDPM 采样中间步骤有随机性；其他采样器可采用确定性过程。初始随机性也能带来不同 mode。 |
| 一定要预测 ε 吗？             | 也可以选择 x₀-prediction、v-prediction 等参数化。             |
| 一定要 cosine schedule 吗？ | schedule 是设计选择，当前 toy 使用 cosine。                   |
| 最后一步一定零方差吗？            | 当前选择 σₜ²=β̃ₜ，因此 t=1 时零方差；不能推广成所有 sampler 的规定。      |
| 为什么生成可能慢？              | 一次生成需要反复调用网络；当前默认每个条件调用 100 次。                     |
| toy 已经是可执行机器人策略吗？      | 它只学习二维人工动作分布，还没有视觉、动力学、接触约束和控制闭环。                  |


---



# 七、从二维 toy 到真正的 Diffusion Policy



## 从一个动作，到一个 action chunk

当前模型生成：

$$
x_0\in\mathbb R^2.
$$

真实动作扩散可以生成一段动作。为继续保持 DDPM 的 notation，把整个动作块记为：

$$
\boxed{x_0=[a_\tau,\ldots,a_{\tau+H-1}]\in\mathbb R^{H\times d_a}.}
$$

这里 aτ 是控制时刻 τ 的物理动作，x₀ 是整个未加扩散噪声的动作块。例如 H=16、dₐ=7 时，batch shape 为 `(B,16,7)`。

前向公式仍然是：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

变化的是样本形状和网络结构。t 表示整段动作的扩散等级；τ,τ+1,… 表示动作执行的控制时间。

## Observation conditioning 与 receding horizon

[Diffusion Policy](https://arxiv.org/abs/2303.04137) 将动作生成与视觉条件、动作序列建模及 receding horizon 控制结合起来。

```text
图像 / 机器人状态 / 观测历史
            ↓
       observation encoder
            ↓ condition
随机动作块 → 迭代去噪 → 预测未来 H 步动作
                              ↓
                        执行前 h 步，h ≤ H
                              ↓
                        获取新观测，再次生成
```

条件分布从 toy 的 pθ(x₀ ∣ o)，扩展为：

$$
p_\theta\left(a_{\tau:\tau+H-1}\mid O_{\tau-L+1:\tau}\right).
$$

只执行动作块的一部分，再根据新观测重新预测，可以形成反馈闭环。生成一个动作序列不意味着把整段序列无限期地开环执行。

从 toy 走向真实机器人，还需要 observation encoder、时序动作网络、数据归一化，以及训练和控制执行的配合。

## 与 π₀、flow matching 的联系

三者都可以放在条件动作生成的框架下理解，但训练目标与生成动力学不同：


| 方法               | 主要角色                                |
| ---------------- | ----------------------------------- |
| DDPM             | 学习离散反向扩散；经典参数化通过预测噪声构造生成转移          |
| Diffusion Policy | 将受观测条件约束的动作序列作为扩散生成对象，并结合闭环执行       |
| π₀               | 在视觉、语言等条件下，以 flow matching 训练动作生成模型 |


[π₀](https://arxiv.org/abs/2410.24164) 使用 flow matching，并非直接复用这里的 DDPM 噪声 MSE 和随机反向采样循环。

理解一个机器人生成策略时，可以依次问：它生成什么对象、条件是什么、预测什么量、如何采样，以及生成结果如何进入控制闭环。

---



# 公式速查


| 概念       | 公式                                               |
| -------- | ------------------------------------------------ |
| 前向一步     | q(xₜ ∣ xₜ₋₁)=N(√αₜxₜ₋₁,βₜI)                      |
| 累计信号     | ᾱₜ = α₁α₂⋯αₜ，ᾱ₀ = 1                            |
| 前向闭式     | xₜ=√ᾱₜx₀+√(1 − ᾱₜ)ε                            |
| 后验方差     | β̃ₜ = βₜ(1 − ᾱₜ₋₁) / (1 − ᾱₜ)                  |
| 预测干净样本   | x̂₀=(xₜ-√(1 − ᾱₜ)εθ)/√ᾱₜ                       |
| 反向均值     | μθ = [xₜ − βₜεθ / √(1 − ᾱₜ)] / √αₜ              |
| 反向采样     | xₜ₋₁=μθ+σₜz；本例 σₜ²=β̃ₜ                           |
| 简化损失     | Lsimple = E[‖ε − εθ(xₜ,t,o)‖²]                   |
| Score 关系 | ∇ₓₜ log pₜ(xₜ ∣ o) = −E[ε ∣ xₜ,t,o] / √(1 − ᾱₜ) |


公式中的 εθ 在本例中都以 (xₜ,t,o) 为输入。反向采样中 t>1 时 z ∼ N(0,I)，t=1 时省略随机项。

# 讨论

1. 为什么动作回归的条件均值可能位于两个示教 mode 之间，而噪声 MSE 可以用于训练一个双峰生成模型？
2. 为什么训练时知道 x₀ 和真实 ε，生成时却没有一个预存的真实 x₀ 等待恢复？
3. 从 `(B,2)` 改成 `(B,H,7)`，哪些扩散公式不变，哪些网络和控制组件需要调整？



# 参考资料

- [Denoising Diffusion Probabilistic Models — Ho, Jain, Abbeel, 2020](https://arxiv.org/abs/2006.11239)：前向过程、后验推导、变分目标与噪声参数化。
- [Improved Denoising Diffusion Probabilistic Models — Nichol, Dhariwal, 2021](https://proceedings.mlr.press/v139/nichol21a.html)：cosine schedule 等改进。
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion — Chi et al., 2023](https://arxiv.org/abs/2303.04137)：条件动作扩散与机器人控制。
- [π₀: A Vision-Language-Action Flow Model for General Robot Control — Black et al., 2024](https://arxiv.org/abs/2410.24164)：flow matching 动作生成。
- [二维条件扩散实现](/Users/jiachexu/Code/robot/diffusion/2d.py)：本文的数据、变量名、索引、网络结构和采样逻辑的对应代码。

