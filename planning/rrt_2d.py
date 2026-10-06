from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

BOUNDS = (0.0, 10.0, 0.0, 8.0)  # xmin, xmax, ymin, ymax
OBSTACLES = ((3.0, 6.0, 2.0, 6.0),)
START = (1.0, 6.5)
GOAL = (9.0, 1.5)


def _point(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError("Expected a finite 2-D point")
    return value


def _bounds(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("Expected (xmin, xmax, ymin, ymax)")
    if value[0] >= value[1] or value[2] >= value[3]:
        raise ValueError("Rectangle bounds must have positive width and height")
    return value


def sample(bounds=BOUNDS, goal=GOAL, goal_bias=0.1, rng=None):
    """在工作空间内均匀采样，并以一定概率直接采样目标点（goal bias）。"""
    bounds, goal = _bounds(bounds), _point(goal)
    if not 0 <= goal_bias <= 1:
        raise ValueError("goal_bias must be in [0, 1]")
    rng = np.random.default_rng() if rng is None else rng
    if rng.random() < goal_bias:
        return goal.copy()
    return rng.uniform(bounds[[0, 2]], bounds[[1, 3]])


def nearest(nodes, point):
    """Return the index of the Euclidean nearest tree node."""
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) == 0 or not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must be a nonempty finite (N, 2) array")
    return int(np.argmin(np.sum((nodes - _point(point)) ** 2, axis=1)))


def steer(source, target, step_size=0.35):
    """从 source 朝 target 方向移动，步长至多为 step_size。"""
    source, target = _point(source), _point(target)
    if not np.isfinite(step_size) or step_size <= 0:
        raise ValueError("step_size must be finite and positive")
    delta = target - source
    distance = np.linalg.norm(delta)
    return target.copy() if distance <= step_size else source + step_size * delta / distance


def collision_free(source, target, obstacles=OBSTACLES, bounds=BOUNDS):
    """用 slab 相交测试检查整条线段，不做离散化采样。

    允许触到工作空间边界；禁止触到障碍物边界（闭矩形）。
    正确处理零长度线段以及与矩形边平行的线段。
    """
    source, target = _point(source), _point(target)
    bounds = _bounds(bounds)
    lower, upper = bounds[[0, 2]], bounds[[1, 3]]
    if np.any(source < lower) or np.any(source > upper) or np.any(target < lower) or np.any(target > upper):
        return False
    direction = target - source
    for obstacle in obstacles:
        rectangle = _bounds(obstacle)
        lo, hi = rectangle[[0, 2]], rectangle[[1, 3]]
        enter, leave = 0.0, 1.0
        for axis in range(2):
            if direction[axis] == 0:
                if source[axis] < lo[axis] or source[axis] > hi[axis]:
                    break
            else:
                t1 = (lo[axis] - source[axis]) / direction[axis]
                t2 = (hi[axis] - source[axis]) / direction[axis]
                enter, leave = max(enter, min(t1, t2)), min(leave, max(t1, t2))
                if enter > leave:
                    break
        else:
            return False
    return True


def backtrack(nodes, parents, goal_index=None):
    """沿父节点索引回溯到根（-1），返回从起点到目标的路径数组。"""
    if len(nodes) == 0 or len(nodes) != len(parents):
        raise ValueError("nodes and parents must have the same nonzero length")
    index = len(nodes) - 1 if goal_index is None else goal_index
    path, seen = [], set()
    while index != -1:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(nodes) or index in seen:
            raise ValueError("Invalid parent chain")
        seen.add(index)
        path.append(_point(nodes[index]).copy())
        index = parents[index]
    return np.asarray(path[::-1])


@dataclass
class RRTResult:
    nodes: np.ndarray
    parents: list
    path: np.ndarray | None
    iterations: int


def rrt(start=START, goal=GOAL, bounds=BOUNDS, obstacles=OBSTACLES,
        step_size=0.35, goal_bias=0.1, max_iter=10000, rng=None):
    """生长一棵 RRT；即使未找到路径也返回已探索的树。"""
    start, goal = _point(start), _point(goal)
    bounds = _bounds(bounds)
    obstacles = tuple(_bounds(rectangle) for rectangle in obstacles)
    steer(start, goal, step_size)  # 即使 start==goal 也要先校验参数合法性。
    if not 0 <= goal_bias <= 1 or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("Invalid goal_bias or max_iter")
    if not collision_free(start, start, obstacles, bounds) or not collision_free(goal, goal, obstacles, bounds):
        raise ValueError("Start and goal must be inside the workspace and outside obstacles")
    rng = np.random.default_rng() if rng is None else rng
    nodes, parents = [start.copy()], [-1]
    if np.array_equal(start, goal):
        return RRTResult(np.asarray(nodes), parents, np.asarray(nodes), 0)
    for iteration in range(1, max_iter + 1):
        point = sample(bounds, goal, goal_bias, rng)
        parent = nearest(nodes, point)
        new = steer(nodes[parent], point, step_size)
        if np.linalg.norm(new - nodes[parent]) < 1e-12:
            continue
        if not collision_free(nodes[parent], new, obstacles, bounds):
            continue
        nodes.append(new)
        parents.append(parent)
        if np.linalg.norm(new - goal) <= step_size and collision_free(new, goal, obstacles, bounds):
            if not np.array_equal(new, goal):
                parents.append(len(nodes) - 1)
                nodes.append(goal.copy())
            return RRTResult(np.asarray(nodes), parents, backtrack(nodes, parents), iteration)
    return RRTResult(np.asarray(nodes), parents, None, max_iter)


def shortcut(path, obstacles=OBSTACLES, bounds=BOUNDS, attempts=300, rng=None):
    """随机选取路径上两个节点；若两节点之间的弦线无碰撞，则删去中间节点。

    起点和终点始终保持不变。不修改输入路径。这只是局部简化，
    不保证得到全局最短路径。
    """
    points = [_point(point).copy() for point in path]
    obstacles = tuple(obstacles)
    if not points or not isinstance(attempts, int) or attempts < 0:
        raise ValueError("Expected a nonempty path and nonnegative integer attempts")
    if any(not collision_free(a, b, obstacles, bounds)
           for a, b in zip(points, points[1:] + points[-1:])):
        raise ValueError("Input path is not collision-free")
    rng = np.random.default_rng() if rng is None else rng
    for _ in range(attempts):
        if len(points) <= 2:
            break
        i, j = sorted(rng.choice(len(points), size=2, replace=False))
        if j > i + 1 and collision_free(points[i], points[j], obstacles, bounds):
            del points[i + 1:j]
    return np.asarray(points)


def path_length(path):
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def plot_result(result, shortened, bounds=BOUNDS, obstacles=OBSTACLES,
                start=START, goal=GOAL):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax in axes:
        for xmin, xmax, ymin, ymax in obstacles:
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color="0.25"))
        edges = [(result.nodes[parent], result.nodes[i])
                 for i, parent in enumerate(result.parents) if parent != -1]
        ax.add_collection(LineCollection(edges, colors="steelblue", linewidths=0.6,
                                         alpha=0.4, label="RRT tree"))
        ax.scatter(*start, color="green", s=65, label="Start", zorder=5)
        ax.scatter(*goal, color="crimson", marker="*", s=120, label="Goal", zorder=5)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect="equal", xlabel="x", ylabel="y")
        ax.grid(alpha=0.15)
    if result.path is not None:
        for ax in axes:
            ax.plot(*result.path.T, "--", color="darkorange", lw=1.8, label="Original path")
        axes[0].set_title(f"RRT: {len(result.path)} nodes, length {path_length(result.path):.2f}")
        axes[1].plot(*shortened.T, "o-", color="purple", lw=2.2, ms=4, label="Shortcut path")
        axes[1].set_title(f"Shortcut: {len(shortened)} nodes, length {path_length(shortened):.2f}")
    else:
        for ax in axes:
            ax.set_title("No path found within iteration budget")
    for ax in axes:
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Point-robot RRT | {len(result.nodes)} tree nodes | {result.iterations} iterations")
    return fig


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--step-size", type=float, default=0.35)
    parser.add_argument("--shortcut-attempts", type=int, default=300)
    parser.add_argument("--save", metavar="IMAGE.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    result = rrt(step_size=args.step_size, max_iter=args.max_iter, rng=rng)
    shortened = None
    if result.path is not None:
        shortened = shortcut(result.path, attempts=args.shortcut_attempts, rng=rng)
        print(f"Original: {len(result.path)} nodes, length {path_length(result.path):.3f}")
        print(f"Shortcut: {len(shortened)} nodes, length {path_length(shortened):.3f}")
    else:
        print(f"No path found after {result.iterations} iterations; try a larger budget.")
    fig = plot_result(result, shortened)
    if args.save:
        fig.savefig(args.save, dpi=160)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
