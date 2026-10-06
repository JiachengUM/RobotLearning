"""从零实现一个条件 2D Diffusion Policy（仅依赖 torch/numpy/matplotlib/pillow）。

运行：python diffusion/2d.py
输出：diffusion/outputs/{denoising.gif,summary.png,policy.pt,trajectories.npz}

o ∈ {-1,+1}; p(a|o) = 0.5 N((-2,2o),0.1I) + 0.5 N((2,2o),0.1I).
这里 a 是一个二维动作，不是机器人运动轨迹；动画展示的是动作的去噪轨迹。
"""

import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "diffusion-mpl"))
import matplotlib

matplotlib.use("Agg")  # 无窗口环境也可训练和导出动画
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
from torch import nn


def make_dataset(n, generator):
    """先抽 observation 和左右 mode，再添加方差 0.1（标准差 sqrt(0.1)）的噪声。"""
    o = (torch.randint(2, (n, 1), generator=generator) * 2 - 1).float()
    side = (torch.randint(2, (n, 1), generator=generator) * 2 - 1).float()
    means = torch.cat([2 * side, 2 * o], dim=1)
    actions = means + math.sqrt(0.1) * torch.randn(n, 2, generator=generator)
    return o, actions


class NoiseMLP(nn.Module):
    """εθ(aᵏ,k,o)：恰好三个 Linear 层，时间用固定的 sin/cos 特征表示。"""

    def __init__(self, steps, hidden=128, time_dim=32):
        super().__init__()
        self.steps = steps
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), time_dim // 2)))
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, a, k, o):
        phase = (k.float() / self.steps)[:, None] * self.frequencies[None, :]
        time = torch.cat([phase.sin(), phase.cos()], dim=1)
        return self.net(torch.cat([a, time, o], dim=1))


class Diffusion:
    def __init__(self, steps, device):
        self.steps = steps
        # cosine schedule 使最后的 ᾱ 接近 0，从而 aᵀ ≈ N(0,I)。
        t = torch.linspace(0, 1, steps + 1, device=device)
        curve = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        curve = curve / curve[0]
        self.beta = (1 - curve[1:] / curve[:-1]).clamp(max=0.999)
        self.alpha = 1 - self.beta
        self.alpha_bar = self.alpha.cumprod(0)
        previous = torch.cat([torch.ones(1, device=device), self.alpha_bar[:-1]])
        self.posterior_variance = self.beta * (1 - previous) / (1 - self.alpha_bar)

    def add_noise(self, a0, k, noise):
        # 索引 k=0,...,T-1 对应数学上的扩散时刻 1,...,T。
        alpha_bar = self.alpha_bar[k, None]
        return alpha_bar.sqrt() * a0 + (1 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, observation, initial, generator):
        model.eval()
        a = initial.clone()
        o = torch.full((len(a), 1), float(observation), device=a.device)
        history = [a.cpu().numpy().copy()]
        for k in reversed(range(self.steps)):
            time = torch.full((len(a),), k, device=a.device, dtype=torch.long)
            epsilon = model(a, time, o)
            # DDPM: μθ = (aᵏ - βk / sqrt(1-ᾱk) εθ) / sqrt(αk)
            mean = (a - self.beta[k] / (1 - self.alpha_bar[k]).sqrt() * epsilon) / self.alpha[k].sqrt()
            if k > 0:
                z = torch.randn(a.shape, device=a.device, generator=generator)
                a = mean + self.posterior_variance[k].sqrt() * z
            else:
                a = mean  # 最后一步不再添加噪声
            history.append(a.cpu().numpy().copy())
        return np.stack(history)


def train(model, diffusion, observations, actions, args, generator):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses = []
    model.train()
    for step in range(args.train_steps):
        ids = torch.randint(len(actions), (args.batch_size,), device=actions.device, generator=generator)
        a0, o = actions[ids], observations[ids]
        k = torch.randint(diffusion.steps, (len(ids),), device=actions.device, generator=generator)
        noise = torch.randn(a0.shape, device=actions.device, generator=generator)
        noisy_actions = diffusion.add_noise(a0, k, noise)
        loss = (model(noisy_actions, k, o) - noise).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step == 0 or (step + 1) % 500 == 0:
            print(f"train {step + 1:5d}/{args.train_steps} | noise MSE {np.mean(losses[-100:]):.4f}", flush=True)
    return np.array(losses)


def decorate(ax, o):
    ax.set(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), xlabel="action x", ylabel="action y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    for x in (-2, 2):
        ax.add_patch(plt.Circle((x, 2 * o), 2 * math.sqrt(0.1), color="#e09132", fill=False, ls="--"))
        ax.scatter(x, 2 * o, marker="x", color="#b96910", s=65, zorder=4)


def visualize(histories, losses, observations, actions, output):
    colors = np.where(histories[1][-1, :, 0] < 0, "#3875c6", "#e46755")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    scatters, trails = [], []
    for ax, o in zip(axes, (1, -1)):
        decorate(ax, o)
        points = histories[o][0]
        scatters.append(ax.scatter(points[:, 0], points[:, 1], c=colors, s=9, alpha=0.6))
        trails.append([ax.plot([], [], color=colors[j], lw=0.9, alpha=0.5)[0] for j in range(12)])
    total = len(histories[1]) - 1
    frames = [0] * 8 + list(range(total + 1)) + [total] * 20

    def update(index):
        for ax, scatter, lines, o in zip(axes, scatters, trails, (1, -1)):
            points = histories[o][index]
            scatter.set_offsets(points)
            ax.set_title(f"o = {o:+d} | reverse step {index}/{total} | k = {total-index}")
            for j, line in enumerate(lines):
                trace = histories[o][max(0, index - 18):index + 1, j]
                line.set_data(trace[:, 0], trace[:, 1])
        return scatters

    fig.suptitle("Conditional 2D diffusion: noise → two Gaussian modes")
    FuncAnimation(fig, update, frames=frames, interval=70, blit=False).save(
        output / "denoising.gif", writer=PillowWriter(fps=14), dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    for row, o in enumerate((1, -1)):
        for col, fraction in enumerate((0, 0.5, 0.8, 1)):
            ax = axes[row, col]
            decorate(ax, o)
            index = round(total * fraction)
            points = histories[o][index]
            ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.5, color="#3875c6")
            if fraction == 1:
                data = actions[observations[:, 0] == o][:500]
                ax.scatter(data[:, 0], data[:, 1], s=5, color="#e09132", alpha=0.15, label="dataset")
                ax.legend(loc="lower right")
            ax.set_title(f"o={o:+d} | {index}/{total} reverse steps")
    fig.suptitle("Same particles through reverse diffusion (orange: training data)")
    fig.savefig(output / "summary.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    window = min(100, len(losses))
    ax.plot(np.arange(window, len(losses) + 1), np.convolve(losses, np.ones(window) / window, "valid"))
    ax.set(xlabel="Training step", ylabel="Noise prediction MSE", title="Training loss (moving average)")
    fig.savefig(output / "loss.png", dpi=140)
    plt.close(fig)


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
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    if min(args.train_steps, args.dataset_size, args.batch_size, args.samples) < 1 or args.diffusion_steps < 2:
        parser.error("sizes must be positive and diffusion-steps must be at least 2")
    torch.set_num_threads(4)  # 小 MLP 使用过多 CPU 线程反而会变慢
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # 数据与训练/采样各使用固定 RNG；同一初始噪声用于两个 observation。
    o, a = make_dataset(args.dataset_size, torch.Generator().manual_seed(args.seed))
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    model = NoiseMLP(args.diffusion_steps).to(device)
    diffusion = Diffusion(args.diffusion_steps, device)
    losses = train(model, diffusion, o.to(device), a.to(device), args, generator)
    initial = torch.randn(args.samples, 2, device=device, generator=generator)
    histories = {}
    for observation in (1, -1):
        sampling_rng = torch.Generator(device=device).manual_seed(args.seed + 2)
        histories[observation] = diffusion.sample(model, observation, initial, sampling_rng)
        final = histories[observation][-1]
        centers = np.column_stack([np.where(final[:, 0] < 0, -2, 2), np.full(len(final), 2 * observation)])
        residual = final - centers
        print(f"o={observation:+d}: left fraction={(final[:, 0] < 0).mean():.3f}, "
              f"mode residual mean={residual.mean(0).round(3)}, variance={residual.var(0).round(3)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": {**vars(args), "output": str(args.output)},
                "losses": losses.tolist()}, args.output / "policy.pt")
    np.savez_compressed(args.output / "trajectories.npz", positive=histories[1], negative=histories[-1], losses=losses)
    visualize(histories, losses, o.numpy(), a.numpy(), args.output)
    print(f"Saved animation, figures and model to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
