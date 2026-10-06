"""二连杆关节空间 RRT：左侧 C-space，右侧播放避障轨迹。

python3 planning/arm_rrt.py
python3 planning/arm_rrt.py --save planning/arm_rrt.gif --snapshot planning/arm_rrt.png --no-show

q1、q2 单位为 rad；使用 [-pi, pi] 硬关节限位，不跨边界绕回。
连杆为零厚度线段；不模拟动力学、连杆厚度或自碰撞。
边检测采用细分和保守的障碍物膨胀，覆盖插值采样之间的扫掠运动。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# 同时支持直接运行脚本和 python -m planning.arm_rrt。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinematics.two_link import LENGTHS, fk
from planning.rrt_2d import (RRTResult, _point, _bounds, nearest, steer, backtrack,
                             sample as sample_2d, collision_free as segment_free)

LIMITS = (-np.pi, np.pi, -np.pi, np.pi)
OBSTACLES = ((1.1, 1.5, -0.3, 0.3),)
START = (-0.8, 0.0)
GOAL = (0.8, 0.0)
RESOLUTION = 0.04


def link_points(q):
    """FK 返回基座、肘部、末端三个点；必须检查两段连杆。"""
    q = _point(q)
    elbow = LENGTHS[0] * np.array([np.cos(q[0]), np.sin(q[0])])
    return np.vstack(([0., 0.], elbow, fk(q)))


def sample(goal=GOAL, goal_bias=0.1, rng=None):
    return sample_2d(LIMITS, goal, goal_bias, rng)


def configuration_free(q, obstacles=OBSTACLES, padding=0.0):
    """给定位形下，对两根连杆做精确线段—闭矩形相交检测。"""
    q = _point(q)
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be nonnegative and finite")
    if np.any(q < -np.pi) or np.any(q > np.pi):
        return False
    rectangles = []
    for rectangle in obstacles:
        xmin, xmax, ymin, ymax = _bounds(rectangle)
        rectangles.append((xmin-padding, xmax+padding, ymin-padding, ymax+padding))
    points = link_points(q)
    reach = float(sum(LENGTHS)) + 1.0
    return all(segment_free(a, b, rectangles, (-reach, reach, -reach, reach))
               for a, b in zip(points, points[1:]))


def collision_free(source, target, obstacles=OBSTACLES, resolution=RESOLUTION):
    """检查关节空间线性插值边，保守覆盖每个子区间的连续运动。

    每个子区间检查中点，膨胀量为所有连杆点相对中点的位移上界：
    L1*|dq1|/2 + L2*|dq1+dq2|/2。正弦运动的弦长不超过弧长。
    因此不会仅因采样间隔而漏检，但可能拒绝非常贴近障碍的安全边。
    """
    source, target = _point(source), _point(target)
    obstacles = tuple(obstacles)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    if not configuration_free(source, obstacles) or not configuration_free(target, obstacles):
        return False
    count = max(1, int(np.ceil(np.max(np.abs(target-source)) / resolution)))
    delta = (target-source) / count
    padding = (LENGTHS[0]*abs(delta[0]) + LENGTHS[1]*abs(delta.sum())) / 2
    return all(configuration_free(source+(i+0.5)*delta, obstacles, padding)
               for i in range(count))


def rrt(start=START, goal=GOAL, obstacles=OBSTACLES, step_size=0.25,
        goal_bias=0.1, max_iter=15000, resolution=RESOLUTION, rng=None):
    """在有硬限位的关节空间内建立 RRT；所有插入的边均做运动碰撞检测。"""
    start, goal = _point(start), _point(goal)
    obstacles = tuple(obstacles)
    steer(start, goal, step_size)
    if not 0 <= goal_bias <= 1 or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("Invalid goal_bias or max_iter")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    if not configuration_free(start, obstacles) or not configuration_free(goal, obstacles):
        raise ValueError("Start/goal is in collision or outside joint limits")
    nodes, parents = [start.copy()], [-1]
    if np.array_equal(start, goal):
        return RRTResult(np.array(nodes), parents, np.array(nodes), 0)
    rng = np.random.default_rng() if rng is None else rng
    for iteration in range(1, max_iter+1):
        target = sample(goal, goal_bias, rng)
        parent = nearest(nodes, target)
        new = steer(nodes[parent], target, step_size)
        if np.linalg.norm(new-nodes[parent]) < 1e-12:
            continue
        if not collision_free(nodes[parent], new, obstacles, resolution):
            continue
        nodes.append(new)
        parents.append(parent)
        if np.linalg.norm(new-goal) <= step_size and collision_free(new, goal, obstacles, resolution):
            if not np.array_equal(new, goal):
                parents.append(len(nodes)-1)
                nodes.append(goal.copy())
            return RRTResult(np.array(nodes), parents, backtrack(nodes, parents), iteration)
    return RRTResult(np.array(nodes), parents, None, max_iter)


def shortcut(path, obstacles=OBSTACLES, attempts=200, resolution=RESOLUTION, rng=None):
    """删去中间节点前，重新检查整个关节插值运动。"""
    points = [_point(q).copy() for q in path]
    obstacles = tuple(obstacles)
    if not points or not isinstance(attempts, int) or attempts < 0:
        raise ValueError("Expected a nonempty path and nonnegative attempts")
    if any(not collision_free(a, b, obstacles, resolution)
           for a, b in zip(points, points[1:] + points[-1:])):
        raise ValueError("Input path is not collision-free")
    rng = np.random.default_rng() if rng is None else rng
    for _ in range(attempts):
        if len(points) < 3:
            break
        i, j = sorted(rng.choice(len(points), 2, replace=False))
        if j > i+1 and collision_free(points[i], points[j], obstacles, resolution):
            del points[i+1:j]
    return np.array(points)


def trajectory(path, speed=0.8, fps=24):
    """以不超过 speed rad/s 的关节向量速度插值，保留所有路径拐点。"""
    if not np.isfinite(speed) or speed <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("speed and fps must be finite and positive")
    frames = [_point(path[0]).copy()]
    for a, b in zip(path, path[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b-a)*fps/speed)))
        frames.extend(np.linspace(a, b, count+1)[1:])
    return np.array(frames)


def animate(result, path, obstacles=OBSTACLES, save=None, snapshot=None,
            show=True, fps=24, speed=0.8, grid_size=100):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle, Patch

    obstacles = tuple(obstacles)
    frames = trajectory(path, speed, fps)
    ee = np.array([fk(q) for q in frames])
    fig, (cax, wax) = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    angles = np.linspace(-np.pi, np.pi, grid_size)
    forbidden = np.array([[not configuration_free((a, b), obstacles)
                           for a in angles] for b in angles])
    cax.pcolormesh(angles, angles, forbidden, shading="nearest",
                  cmap=ListedColormap(["#f7fafc", "#f3b7b7"]), vmin=0, vmax=1)
    edges = [(result.nodes[p], result.nodes[i]) for i, p in enumerate(result.parents) if p != -1]
    cax.add_collection(LineCollection(edges, colors="steelblue", linewidths=0.6, alpha=0.65))
    cax.plot(*result.path.T, "--", color="darkorange", lw=1, label="Raw RRT path")
    cax.plot(*path.T, "o-", color="purple", ms=3, lw=2, label="Shortcut path")
    cax.plot(*np.array([path[0], path[-1]]).T, ":", color="red", label="Direct q interpolation (blocked)")
    cax.scatter(*path[0], color="green", s=45, zorder=5, label="Start")
    cax.scatter(*path[-1], color="red", marker="*", s=90, zorder=5, label="Goal")
    cursor, = cax.plot([], [], "ko", ms=6)
    handles, labels = cax.get_legend_handles_labels()
    cax.legend(handles+[Patch(color="#f3b7b7")], labels+["Forbidden configurations"], fontsize=7, loc="lower left")
    cax.set(xlim=LIMITS[:2], ylim=LIMITS[2:], xlabel="q1 (rad)", ylabel="q2 (rad)",
            title=f"C-space: {len(result.nodes)} RRT nodes", aspect="equal")
    for xmin, xmax, ymin, ymax in obstacles:
        wax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color="0.25"))
    for q, color, label in ((path[0], "green", "Start arm"), (path[-1], "red", "Goal arm")):
        wax.plot(*link_points(q).T, "o--", color=color, alpha=0.4, lw=2, label=label)
    wax.plot(*np.array([fk(path[0]), fk(path[-1])]).T, ":", color="red", label="Straight EE line")
    wax.plot(*ee.T, color="purple", alpha=0.4, lw=1, label="Planned EE trace")
    arm, = wax.plot([], [], "o-", lw=4, color="royalblue", ms=7)
    reach = sum(LENGTHS)+0.2
    wax.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal",
            xlabel="x (m)", ylabel="y (m)", title="Workspace: both links avoid obstacles")
    wax.legend(fontsize=7, loc="lower left")
    wax.grid(alpha=0.2)
    caption = fig.suptitle("")

    def update(index):
        q = frames[index]
        cursor.set_data([q[0]], [q[1]])
        arm.set_data(*link_points(q).T)
        caption.set_text(f"Joint-space RRT | frame {index+1}/{len(frames)} | SPACE: pause/resume")
        return cursor, arm, caption

    animation = FuncAnimation(fig, update, frames=len(frames), init_func=lambda: update(0),
                              interval=1000/fps, blit=False, repeat=True, cache_frame_data=False)
    paused = False

    def key(event):
        nonlocal paused
        if event.key == " ":
            paused = not paused
            animation.pause() if paused else animation.resume()

    fig.canvas.mpl_connect("key_press_event", key)
    if snapshot:
        update(len(frames)//2)
        fig.savefig(snapshot, dpi=140)
    if save:
        animation.save(save, writer=PillowWriter(fps=fps))
    if show:
        plt.show()
    else:
        # 静态导出时也初始化动画，避免未渲染动画的析构警告。
        if not save:
            fig.canvas.draw()
        plt.close(fig)
    return animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=15000)
    parser.add_argument("--save", metavar="FILE.gif")
    parser.add_argument("--snapshot", metavar="FILE.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    result = rrt(max_iter=args.max_iter, rng=rng)
    if result.path is None:
        parser.exit(1, f"No path after {result.iterations} iterations; increase --max-iter.\n")
    path = shortcut(result.path, rng=rng)
    print(f"RRT: {result.iterations} iterations, {len(result.nodes)} nodes; path {len(result.path)} -> {len(path)} nodes")
    print(f"Direct joint interpolation collision-free: {collision_free(START, GOAL)}")
    animate(result, path, save=args.save, snapshot=args.snapshot, show=not args.no_show)


if __name__ == "__main__":
    main()
