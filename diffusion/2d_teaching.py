"""
条件 2D Diffusion Policy —— 教学注释版

目标：学习条件动作分布 p(a | o)

    o ∈ {-1,+1}

    p(a | o)
      = 0.5 N([-2, 2o]^T, 0.1 I)
      + 0.5 N([+2, 2o]^T, 0.1 I)

这里：
- o 是 observation / condition；
- a 是二维 action，不是机器人运动轨迹；
- side ∈ {-1,+1} 只用于生成数据，代表左/右两个隐藏 mode；
- side 不会给模型，因此同一个 o 下模型必须表示两个合理 action mode。

整个程序分成两部分：

训练：
    clean action a0
      -> 随机选 timestep k
      -> 自己采真实 noise epsilon
      -> 构造 noisy action ak
      -> 网络预测 epsilon_hat
      -> MSE(epsilon_hat, epsilon)

推理：
    aT ~ N(0,I)
      -> 用 epsilon_theta(ak,k,o) 构造 reverse transition
      -> a(T-1) -> ... -> a0
      -> 得到服从 p(a|o) 的 action sample
"""

import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "diffusion-mpl"),
)

import matplotlib
matplotlib.use("Agg")  # 无窗口环境也可以保存图像/GIF

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from torch import nn


# ============================================================
# 1. 构造一个人为设计的双峰 conditional action distribution
# ============================================================

def make_dataset(n, generator):
    """
    生成 n 个 (observation, action) 样本。

    observation:
        o ∈ {-1,+1}, shape=(n,1)

    hidden mode:
        side ∈ {-1,+1}, shape=(n,1)
        side=-1 -> 左 mode
        side=+1 -> 右 mode

    action mean:
        mu = [2*side, 2*o]

    因此：
        o=+1 -> mode centers = (-2,+2), (+2,+2)
        o=-1 -> mode centers = (-2,-2), (+2,-2)

    最后再加 ξ ~ N(0, 0.1I)。

    注意：side 最后不会返回给模型。
    所以模型只看到 o，却必须学会 p(a|o) 的双峰结构。
    """

    # randint(2) 产生 0/1；*2-1 后得到 -1/+1。
    # o.shape = (n,1)
    o = (torch.randint(2, (n, 1), generator=generator) * 2 - 1).float()

    # side 与 o 独立采样；它只决定动作属于左 mode 还是右 mode。
    side = (torch.randint(2, (n, 1), generator=generator) * 2 - 1).float()

    # 2*side: (n,1), 2*o: (n,1)
    # 沿 dim=1 拼接后 means.shape=(n,2)。
    #
    # 数学上单个 action mean 可以写成列向量：
    #
    #       [ 2*side ]
    #   μ = [        ]
    #       [ 2*o    ]
    #
    # PyTorch batch 通常把每个样本放在一行，所以整体 shape 是 (n,2)。
    means = torch.cat([2 * side, 2 * o], dim=1)

    # randn(n,2) ~ N(0,I)。
    # 乘 sqrt(0.1) 后 covariance 变成 0.1I。
    #
    # 所以：action | (o,side) ~ N(means, 0.1I)
    actions = means + math.sqrt(0.1) * torch.randn(
        n, 2, generator=generator
    )

    # 故意不返回 side：它是隐藏 mode variable。
    return o, actions


# ============================================================
# 2. Noise predictor: epsilon_theta(a^k, k, o)
# ============================================================

class NoiseMLP(nn.Module):
    """
    输入：
        a : noisy action, shape=(B,2)
        k : diffusion timestep, shape=(B,)
        o : condition, shape=(B,1)

    输出：
        epsilon_hat, shape=(B,2)

    网络不是直接预测 clean action，而是预测 forward diffusion 中加入的
    标准 Gaussian noise epsilon。
    """

    def __init__(self, steps, hidden=128, time_dim=32):
        super().__init__()
        self.steps = steps

        # time_dim=32，所以用 16 个 frequency，再分别取 sin/cos：16*2=32。
        #
        # register_buffer：
        # - 不是可训练参数；
        # - 会跟随 model.to(device)；
        # - 会保存在 state_dict 中。
        self.register_buffer(
            "frequencies",
            torch.exp(
                torch.linspace(0, math.log(1000), time_dim // 2)
            ),
        )

        # 输入维度：
        #   noisy action 2D + time embedding 32D + observation 1D = 35D
        # 输出 2D，因为 epsilon 和 action shape 相同。
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, a, k, o):
        # k.shape=(B,)
        # (k/steps)[:,None] -> (B,1)
        # frequencies[None,:] -> (1,16)
        # broadcasting 后 phase.shape=(B,16)。
        #
        # 注意这里是逐元素乘法 + broadcasting，不是一般矩阵乘法。
        phase = (
            (k.float() / self.steps)[:, None]
            * self.frequencies[None, :]
        )

        # sin 与 cos 各 (B,16)，拼起来得到 (B,32) 的 time embedding。
        time = torch.cat([phase.sin(), phase.cos()], dim=1)

        # a:(B,2), time:(B,32), o:(B,1)
        # 拼起来 -> (B,35)，送入 MLP -> (B,2)。
        x = torch.cat([a, time, o], dim=1)
        return self.net(x)


# ============================================================
# 3. Diffusion schedule / forward diffusion / reverse sampling
# ============================================================

class Diffusion:
    def __init__(self, steps, device):
        self.steps = steps

        # 核心 notation：
        #
        #   beta_t      : 第 t 步加入多少 noise
        #   alpha_t     : 1 - beta_t
        #   alpha_bar_t : product_{s<=t} alpha_s
        #
        # alpha_bar_t 可以理解为“累计还剩多少原始 signal”。
        # 希望 alpha_bar_0≈1，alpha_bar_T≈0，最终 a_T≈N(0,I)。

        # steps+1 个 curve 点，用相邻比值构造 steps 个 alpha/beta。
        t = torch.linspace(0, 1, steps + 1, device=device)

        # Cosine schedule：从接近 1 平滑下降到接近 0。
        curve = torch.cos(
            (t + 0.008) / 1.008 * math.pi / 2
        ).square()

        # 归一化，让起点精确等于 1。
        curve = curve / curve[0]

        # 若 curve 近似 alpha_bar：
        #
        #   alpha_t = alpha_bar_t / alpha_bar_{t-1}
        #   beta_t  = 1 - alpha_t
        #
        # clamp 避免 beta=1 -> alpha=0 -> reverse 时除以 0。
        self.beta = (
            1 - curve[1:] / curve[:-1]
        ).clamp(max=0.999)

        self.alpha = 1 - self.beta

        # alpha_bar_t = product_{s<=t} alpha_s
        self.alpha_bar = self.alpha.cumprod(0)

        # posterior variance 需要 alpha_bar_{t-1}。
        # 数学上约定 alpha_bar_0=1。
        previous = torch.cat(
            [torch.ones(1, device=device), self.alpha_bar[:-1]]
        )

        # DDPM exact posterior：
        #
        # q(a_{t-1} | a_t, a_0)
        #   = N(mu_tilde_t, beta_tilde_t I)
        #
        # beta_tilde_t
        #   = beta_t * (1-alpha_bar_{t-1})/(1-alpha_bar_t)
        #
        # 这个 variance 来自 Gaussian posterior 的解析推导，不是网络学出来的。
        self.posterior_variance = (
            self.beta * (1 - previous) / (1 - self.alpha_bar)
        )

    def add_noise(self, a0, k, noise):
        """
        Forward diffusion closed form：

            a^k = sqrt(alpha_bar_k) * a^0
                + sqrt(1-alpha_bar_k) * epsilon

        训练时 a0、k、epsilon 都已知，因此可以直接 O(1) 构造任意 ak，
        不需要真的一步一步模拟 0->1->...->k。
        """

        # self.alpha_bar[k] 原本 shape=(B,)
        # 加 None 后变成 (B,1)，可与 a0:(B,2) broadcast。
        alpha_bar = self.alpha_bar[k, None]

        return (
            alpha_bar.sqrt() * a0
            + (1 - alpha_bar).sqrt() * noise
        )

    @torch.no_grad()
    def sample(self, model, observation, initial, generator):
        """
        Reverse diffusion / inference：

            a^T ~ N(0,I)
                -> a^{T-1}
                -> ...
                -> a^0

        inference 时没有真实 a0，也没有训练时那个真实 epsilon，
        因此必须使用学到的 epsilon_theta(a^k,k,o) 提供 denoising 信息。
        """

        model.eval()

        # initial 就是 a^T ~ N(0,I)。
        a = initial.clone()

        # 同一次 conditional sampling 中，所有粒子使用相同 observation。
        # 若 B=num_samples，则 o.shape=(B,1)。
        o = torch.full(
            (len(a), 1),
            float(observation),
            device=a.device,
        )

        # 保存每一步，用于动画。
        history = [a.cpu().numpy().copy()]

        # 代码 k=T-1,...,0 对应数学 t=T,...,1。
        for k in reversed(range(self.steps)):
            # 当前所有样本都处于同一 reverse timestep。
            time = torch.full(
                (len(a),),
                k,
                device=a.device,
                dtype=torch.long,
            )

            # Step 1：网络预测当前 noisy action 的 noise information。
            #
            # training 时 epsilon 是自己采样的真值；
            # inference 时它未知，所以只能靠网络估计。
            epsilon = model(a, time, o)

            # Step 2：计算 reverse Gaussian 的 mean。
            #
            #   mu_theta(a_t,t,o)
            #     = 1/sqrt(alpha_t)
            #       [ a_t
            #         - beta_t/sqrt(1-alpha_bar_t)
            #           * epsilon_theta(a_t,t,o) ]
            #
            # 不能简单写成 a_{t-1}=a_t-epsilon，
            # 因为 epsilon 是 forward closed form 里的标准 Gaussian noise，
            # 不是从 a_t 到 a_{t-1} 的位移。
            mean = (
                a
                - self.beta[k]
                / (1 - self.alpha_bar[k]).sqrt()
                * epsilon
            ) / self.alpha[k].sqrt()

            if k > 0:
                # 中间 step 的 reverse conditional 本身是一个 Gaussian：
                #
                #   p_theta(a_{t-1}|a_t,o)
                #       = N(mean, posterior_variance_t * I)
                #
                # 因此要真正“从这个分布中采样”：
                #
                #   a_{t-1} = mean + sqrt(var_t) * z,
                #   z ~ N(0,I)
                #
                # 注意：中间加 noise 不是专门为了制造 multimodality，
                # 而是标准 DDPM reverse transition 本身就是 stochastic Gaussian。
                z = torch.randn(
                    a.shape,
                    device=a.device,
                    generator=generator,
                )

                a = mean + self.posterior_variance[k].sqrt() * z

            else:
                # 最后一步不加 noise。
                #
                # 因为数学上：
                #
                #   beta_tilde_1
                #     = beta_1*(1-alpha_bar_0)/(1-alpha_bar_1)
                #
                # 而 alpha_bar_0=1，所以 beta_tilde_1=0。
                #
                # 因此最后一步 posterior variance 理论上就是 0：
                #   a_0 = mean
                a = mean

            history.append(a.cpu().numpy().copy())

        # shape = (steps+1, num_samples, 2)
        # history[0]  = a^T
        # history[-1] = a^0
        return np.stack(history)


# ============================================================
# 4. Training loop
# ============================================================

def train(model, diffusion, observations, actions, args, generator):
    """
    DDPM noise-prediction training。

    最重要的逻辑：

        1. dataset 给我们 clean action a0；
        2. 自己随机采 timestep k；
        3. 自己随机采真实 epsilon ~ N(0,I)；
        4. 用 closed form 构造 ak；
        5. 训练 epsilon_theta(ak,k,o) 去预测刚才那个真实 epsilon。

    因为训练时 epsilon 是我们自己加进去的，所以它天然就是监督 target。
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    model.train()

    for step in range(args.train_steps):
        # 随机抽 mini-batch。
        ids = torch.randint(
            len(actions),
            (args.batch_size,),
            device=actions.device,
            generator=generator,
        )

        # a0 是 dataset 中已知的 clean action。
        # a0:(B,2), o:(B,1)
        a0, o = actions[ids], observations[ids]

        # 每条样本随机选一个 noise level。
        # k.shape=(B,)
        k = torch.randint(
            diffusion.steps,
            (len(ids),),
            device=actions.device,
            generator=generator,
        )

        # 训练阶段的“真实 epsilon”：我们自己采的，所以完全已知。
        noise = torch.randn(
            a0.shape,
            device=actions.device,
            generator=generator,
        )

        # 人为把 clean action 弄脏，得到 a^k。
        noisy_actions = diffusion.add_noise(a0, k, noise)

        # 训练 epsilon_theta(a^k,k,o) ≈ epsilon。
        #
        # 在 MSE 下，理想 predictor 会逼近 E[epsilon | a^k,k,o]。
        loss = (
            model(noisy_actions, k, o) - noise
        ).square().mean()

        # 标准 PyTorch 参数更新：
        # zero_grad -> backward -> optimizer.step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step == 0 or (step + 1) % 500 == 0:
            print(
                f"train {step + 1:5d}/{args.train_steps} "
                f"| noise MSE {np.mean(losses[-100:]):.4f}",
                flush=True,
            )

    return np.array(losses)


# ============================================================
# 5. Visualization
# ============================================================

def decorate(ax, o):
    ax.set(
        xlim=(-4.5, 4.5),
        ylim=(-4.5, 4.5),
        xlabel="action x",
        ylabel="action y",
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)

    # 对给定 o，真实 action distribution 有两个 mode center：
    # (-2,2o) 与 (+2,2o)。
    for x in (-2, 2):
        ax.add_patch(
            plt.Circle(
                (x, 2 * o),
                2 * math.sqrt(0.1),
                color="#e09132",
                fill=False,
                ls="--",
            )
        )
        ax.scatter(
            x,
            2 * o,
            marker="x",
            color="#b96910",
            s=65,
            zorder=4,
        )


def visualize(histories, losses, observations, actions, output):
    # 用最终 x 坐标判断粒子落入左/右 mode，仅用于着色。
    colors = np.where(
        histories[1][-1, :, 0] < 0,
        "#3875c6",
        "#e46755",
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 5), constrained_layout=True
    )
    scatters, trails = [], []

    for ax, o in zip(axes, (1, -1)):
        decorate(ax, o)
        points = histories[o][0]  # 初始 a^T Gaussian noise
        scatters.append(
            ax.scatter(
                points[:, 0], points[:, 1],
                c=colors, s=9, alpha=0.6,
            )
        )
        trails.append([
            ax.plot([], [], color=colors[j], lw=0.9, alpha=0.5)[0]
            for j in range(12)
        ])

    total = len(histories[1]) - 1
    frames = [0] * 8 + list(range(total + 1)) + [total] * 20

    def update(index):
        for ax, scatter, lines, o in zip(
            axes, scatters, trails, (1, -1)
        ):
            points = histories[o][index]
            scatter.set_offsets(points)
            ax.set_title(
                f"o = {o:+d} | reverse step {index}/{total} "
                f"| k = {total-index}"
            )

            for j, line in enumerate(lines):
                trace = histories[o][
                    max(0, index - 18):index + 1, j
                ]
                line.set_data(trace[:, 0], trace[:, 1])
        return scatters

    fig.suptitle("Conditional 2D diffusion: noise -> two Gaussian modes")
    FuncAnimation(
        fig, update, frames=frames, interval=70, blit=False
    ).save(
        output / "denoising.gif",
        writer=PillowWriter(fps=14),
        dpi=100,
    )
    plt.close(fig)

    # 若干 reverse diffusion snapshot。
    fig, axes = plt.subplots(
        2, 4, figsize=(15, 8), constrained_layout=True
    )

    for row, o in enumerate((1, -1)):
        for col, fraction in enumerate((0, 0.5, 0.8, 1)):
            ax = axes[row, col]
            decorate(ax, o)
            index = round(total * fraction)
            points = histories[o][index]
            ax.scatter(
                points[:, 0], points[:, 1],
                s=6, alpha=0.5, color="#3875c6",
            )

            if fraction == 1:
                # 最后一帧叠加真实训练数据，比较生成分布和 target distribution。
                data = actions[observations[:, 0] == o][:500]
                ax.scatter(
                    data[:, 0], data[:, 1],
                    s=5, color="#e09132", alpha=0.15,
                    label="dataset",
                )
                ax.legend(loc="lower right")

            ax.set_title(
                f"o={o:+d} | {index}/{total} reverse steps"
            )

    fig.suptitle(
        "Same particles through reverse diffusion "
        "(orange: training data)"
    )
    fig.savefig(output / "summary.png", dpi=140)
    plt.close(fig)

    # Training loss moving average。
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    window = min(100, len(losses))
    ax.plot(
        np.arange(window, len(losses) + 1),
        np.convolve(losses, np.ones(window) / window, "valid"),
    )
    ax.set(
        xlabel="Training step",
        ylabel="Noise prediction MSE",
        title="Training loss (moving average)",
    )
    fig.savefig(output / "loss.png", dpi=140)
    plt.close(fig)


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--dataset-size", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = parser.parse_args()

    if (
        min(
            args.train_steps,
            args.dataset_size,
            args.batch_size,
            args.samples,
        ) < 1
        or args.diffusion_steps < 2
    ):
        parser.error(
            "sizes must be positive and diffusion-steps must be at least 2"
        )

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # 数据 RNG 与训练/采样 RNG 分开固定，方便 reproducibility。
    o, a = make_dataset(
        args.dataset_size,
        torch.Generator().manual_seed(args.seed),
    )

    generator = torch.Generator(device=device).manual_seed(args.seed + 1)

    model = NoiseMLP(args.diffusion_steps).to(device)
    diffusion = Diffusion(args.diffusion_steps, device)

    # ---------------------- Training -------------------------
    losses = train(
        model,
        diffusion,
        o.to(device),
        a.to(device),
        args,
        generator,
    )

    # ---------------------- Inference ------------------------
    # 从简单 Gaussian latent 开始：a^T ~ N(0,I)。
    initial = torch.randn(
        args.samples,
        2,
        device=device,
        generator=generator,
    )

    histories = {}

    for observation in (1, -1):
        # 两个 observation 使用同一份 initial noise，
        # reverse stochasticity 也使用同一 seed。
        # 因而这是一个 controlled experiment：唯一改变的是 condition o。
        sampling_rng = torch.Generator(device=device).manual_seed(args.seed + 2)

        histories[observation] = diffusion.sample(
            model,
            observation,
            initial,
            sampling_rng,
        )

        final = histories[observation][-1]

        # 根据 x 正负，把 sample 匹配到最近的左/右 mode center。
        centers = np.column_stack([
            np.where(final[:, 0] < 0, -2, 2),
            np.full(len(final), 2 * observation),
        ])

        residual = final - centers

        # 理想结果：
        #   left fraction ≈ 0.5
        #   residual mean ≈ [0,0]
        #   residual variance ≈ [0.1,0.1]
        #
        # 说明模型不仅学到了两个 mode center，
        # 还大致学到了 mixture weight 与 Gaussian spread。
        print(
            f"o={observation:+d}: "
            f"left fraction={(final[:, 0] < 0).mean():.3f}, "
            f"mode residual mean={residual.mean(0).round(3)}, "
            f"variance={residual.var(0).round(3)}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "config": {**vars(args), "output": str(args.output)},
            "losses": losses.tolist(),
        },
        args.output / "policy.pt",
    )

    np.savez_compressed(
        args.output / "trajectories.npz",
        positive=histories[1],
        negative=histories[-1],
        losses=losses,
    )

    visualize(
        histories,
        losses,
        o.numpy(),
        a.numpy(),
        args.output,
    )

    print(
        f"Saved animation, figures and model to {args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
