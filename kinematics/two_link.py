"""平面二连杆逆运动学（IK）与交互式奇异性对比演示。

运行：python3 kinematics/two_link.py
鼠标移到任一机械臂上即可设定共享目标。P 恢复预设轨迹，S/R 将两条臂
重置到接近完全伸直的位形，SPACE 暂停/继续。
使用 --save demo.gif 可导出确定性动画（需要 Pillow）。
角度单位为弧度，长度单位为米，显示的速度单位为 rad/s。
"""

import argparse
from collections import deque
from dataclasses import dataclass

import numpy as np

LENGTHS = np.array([1.0, 1.0])


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite vector of shape (2,)")
    return value


def fk(q):
    """根据关节角 q=[q1, q2] 计算末端执行器位置（正运动学）。"""
    q = _vector(q, "q")
    angles = np.array([q[0], q.sum()])
    return np.array([LENGTHS @ np.cos(angles), LENGTHS @ np.sin(angles)])


def jacobian(q):
    """解析形式的 2x2 位置雅可比矩阵：dx/dq。"""
    q = _vector(q, "q")
    a, b = q[0], q.sum()
    l1, l2 = LENGTHS
    return np.array([[-l1 * np.sin(a) - l2 * np.sin(b), -l2 * np.sin(b)],
                     [l1 * np.cos(a) + l2 * np.cos(b), l2 * np.cos(b)]])


def ik_step(q, target, method="dls", damping=0.1):
    """计算单步无截断的 Δq：J†e 或 J.T (J J.T + damping² I)^-1 e。

    注意返回的是位移量而非速度。动画中统一使用 qdot=gain*Δq 并按
    q += dt*qdot 积分，两种方法处理方式完全相同，保证对比公平。
    """
    error = _vector(target, "target") - fk(q)
    j = jacobian(q)
    if method == "pinv":
        return np.linalg.pinv(j, rcond=1e-12) @ error
    if method != "dls":
        raise ValueError("method must be 'pinv' or 'dls'")
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be finite and positive")
    return j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(2), error)


@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    iterations: int
    error: float


def ik(target, q0=None, method="dls", damping=0.1, tol=1e-5,
       max_iter=2000, gain=0.5, return_info=False):
    """迭代求解逆运动学，返回 q（若 return_info=True 则返回 IKResult）。

    默认种子采用弯曲位形，以避开"完全伸直 + 径向误差"的驻点。
    目标不可达或迭代停滞时抛出 RuntimeError；若指定 return_info=True，
    则返回最后一次迭代结果和收敛标志。
    不做可达性投影，也不截断关节步长，以完整暴露奇异行为。
    """
    target = _vector(target, "target")
    q = _vector([0.5, -1.0] if q0 is None else q0, "q0").copy()
    if not np.isfinite(tol) or tol <= 0 or not np.isfinite(gain) or not 0 < gain <= 1:
        raise ValueError("tol must be positive and gain must be in (0, 1]")
    if not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter must be a nonnegative integer")
    # Validate the method even if the seed already reaches the target.
    ik_step(q, target, method, damping)
    for iteration in range(max_iter + 1):
        error = float(np.linalg.norm(target - fk(q)))
        if error <= tol or iteration == max_iter:
            break
        q += gain * ik_step(q, target, method, damping)
    result = IKResult(q, error <= tol, iteration, error)
    if return_info:
        return result
    if not result.converged:
        raise RuntimeError(f"IK did not converge: error={error:.6g}; use return_info=True")
    return q


def preset_target(t):
    """先在接近完全伸直处施加径向扰动，随后进入平滑的椭圆轨道。"""
    if t < 4.0:
        return np.array([1.95, 0.0])
    phase = 0.6 * (t - 4.0)
    return np.array([1.45 + 0.5 * np.cos(phase), 0.7 * np.sin(phase)])


def animate(damping=0.1, save=None, seconds=16):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    dt, gain = 1 / 60, 3.0  # 时间步长（秒）与速度增益（两种方法共用）
    seed = np.array([-0.0005, 0.001])  # 接近完全伸直的种子位形，用于暴露奇异性
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    speed_ax = fig.add_subplot(grid[1, :])
    methods = ("pinv", "dls")
    colors = ("tab:orange", "tab:blue")
    arms, targets, labels, curves = [], [], [], []
    for ax, method, color in zip(axes, methods, colors):
        ax.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), aspect="equal",
               xlabel="x (m)", ylabel="y (m)",
               title="Pseudoinverse" if method == "pinv" else f"DLS (lambda={damping:g})")
        ax.add_patch(plt.Circle((0, 0), 2, fill=False, ls="--", color="0.7"))
        ax.grid(alpha=0.2)
        arms.append(ax.plot([], [], "o-", lw=4, color=color)[0])
        targets.append(ax.plot([], [], "rx", ms=10, mew=2)[0])
        labels.append(ax.text(0.02, 0.03, "", transform=ax.transAxes, fontsize=9))
        curves.append(speed_ax.plot([], [], color=color, label=method)[0])
    speed_ax.set(xlabel="Simulation time (s)", ylabel="||qdot|| (rad/s)", yscale="symlog")
    speed_ax.legend()
    speed_ax.grid(alpha=0.3)
    title = fig.suptitle("")
    fig.text(0.5, 0.005, "Mouse: target | P: preset | S/R: near-straight reset | SPACE: pause",
             ha="center", fontsize=9)
    state = {}

    def reset():
        state.update(q=[seed.copy(), seed.copy()], t=0.0, target=None, paused=False,
                     history=deque(maxlen=900), speeds=[deque(maxlen=900), deque(maxlen=900)])

    def mouse(event):
        if event.inaxes in axes and event.xdata is not None and event.ydata is not None:
            state["target"] = np.array([event.xdata, event.ydata])

    def key(event):
        """键盘快捷键：S/R 重置，P 恢复预设轨迹，SPACE 暂停/继续。"""
        if event.key in ("s", "r"):
            reset()
        elif event.key == "p":
            state["target"] = None
        elif event.key == " ":
            state["paused"] = not state["paused"]

    def update(_):
        """每帧回调：计算目标、求解两种方法的速度并刷新所有图形元素。"""
        t = state["t"]
        # 无鼠标目标时使用预设轨迹
        target = preset_target(t) if state["target"] is None else state["target"]
        # 两种方法用同一增益计算关节速度 qdot = gain * Δq
        velocities = [gain * ik_step(q, target, method, damping)
                      for q, method in zip(state["q"], methods)]
        if not state["paused"]:
            state["history"].append(t)
            for history, velocity in zip(state["speeds"], velocities):
                history.append(float(np.linalg.norm(velocity)))
        for i, q in enumerate(state["q"]):
            elbow = LENGTHS[0] * np.array([np.cos(q[0]), np.sin(q[0])])
            points = np.vstack(([0, 0], elbow, fk(q)))
            arms[i].set_data(points[:, 0], points[:, 1])
            targets[i].set_data([target[0]], [target[1]])
            labels[i].set_text(f"error={np.linalg.norm(target-fk(q)):.4f} m\n"
                               f"speed={np.linalg.norm(velocities[i]):.3g} rad/s\n"
                               f"sigma_min={np.linalg.svd(jacobian(q), compute_uv=False)[-1]:.2e}")
            curves[i].set_data(list(state["history"]), list(state["speeds"][i]))
        speed_ax.relim()
        speed_ax.autoscale_view()
        mode = "preset" if state["target"] is None else "mouse"
        title.set_text(f"Two-link IK | {mode} | t={t:.2f}s | same gain, no speed clipping")
        if not state["paused"]:
            for q, velocity in zip(state["q"], velocities):
                q += dt * velocity
            state["t"] += dt
        return arms + targets + labels + curves + [title]

    reset()
    fig.canvas.mpl_connect("motion_notify_event", mouse)
    fig.canvas.mpl_connect("key_press_event", key)
    animation = FuncAnimation(fig, update, init_func=lambda: [],
                              frames=round(seconds / dt) if save else None,
                              interval=1000 * dt, blit=False, cache_frame_data=False)
    if save:
        animation.save(save, writer=PillowWriter(fps=round(1 / dt)))
        plt.close(fig)
    else:
        plt.show()
    return animation


if __name__ == "__main__":
    # 命令行入口：校验参数后启动动画
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--save", metavar="FILE.gif")
    parser.add_argument("--seconds", type=float, default=16)
    args = parser.parse_args()
    if not np.isfinite(args.damping) or args.damping <= 0:
        parser.error("--damping must be finite and positive")
    if not np.isfinite(args.seconds) or args.seconds < 1 / 60:
        parser.error("--seconds must be finite and at least 1/60")
    animate(args.damping, args.save, args.seconds)
