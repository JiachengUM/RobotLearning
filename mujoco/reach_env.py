"""随机初态到随机目标。macOS: .venv/bin/mjpython mujoco/reach_env.py
无窗口验收: .venv/bin/python mujoco/reach_env.py --headless --episodes 20
"""
import argparse
from pathlib import Path
import time

import numpy as np
import mujoco
from controller import ReachController


class ReachEnv:
    def __init__(self, seed=7):
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name('reach.xml')))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.controller = ReachController(self.model)
        self.ee_id = self.model.site('ee').id
        self.mocap_id = self.model.body('target').mocapid[0]
        self.target = None

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.rng.uniform(-np.pi, np.pi, 2)
        self.data.qvel[:] = 0
        # 从可达圆环按面积均匀采样，避开原点与完全伸直的奇异边界。
        radius = np.sqrt(self.rng.uniform(0.4**2, 1.8**2))
        angle = self.rng.uniform(-np.pi, np.pi)
        self.target = np.array([radius*np.cos(angle), radius*np.sin(angle), 0.2])
        self.data.mocap_pos[self.mocap_id] = self.target
        mujoco.mj_forward(self.model, self.data)
        self.controller.set_target(self.target, self.data)
        self.data.ctrl[:] = self.data.qpos
        return self.observe()

    def observe(self):
        return dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(),
                    ctrl=self.data.ctrl.copy(), ee=self.data.site_xpos[self.ee_id].copy(),
                    target=self.target.copy(), time=float(self.data.time))

    def step(self):
        self.controller.update(self.data)
        mujoco.mj_step(self.model, self.data)
        # mj_step 后刷新派生位姿，让观察与当前 qpos 对齐。
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def reached(self):
        return (np.linalg.norm(self.data.site_xpos[self.ee_id]-self.target) < 0.01
                and np.linalg.norm(self.data.qvel) < 0.03)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=12., help='每回合最大仿真秒数')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if args.episodes <= 0 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('episodes and timeout must be positive')
    env = ReachEnv(args.seed)
    failures = 0

    def episode(viewer=None):
        nonlocal failures
        print('reset:', env.observe(), flush=True)
        stable = 0
        for _ in range(int(np.ceil(args.timeout/env.model.opt.timestep))):
            started = time.monotonic()
            if viewer is not None:
                if not viewer.is_running():
                    return False
                with viewer.lock():
                    env.step()
                viewer.sync()
            else:
                env.step()
            stable = stable+1 if env.reached() else 0
            if viewer is not None:
                time.sleep(max(0, env.model.opt.timestep-(time.monotonic()-started)))
            if stable >= 100:  # 连续 0.2 秒位置与速度均达标。
                break
        success = stable >= 100
        failures += not success
        error = np.linalg.norm(env.observe()['ee']-env.target)
        print(f'{"PASS" if success else "TIMEOUT"}: t={env.data.time:.3f}s error={error:.6f}m '
              f'qpos={env.data.qpos} qvel={env.data.qvel} ctrl={env.data.ctrl}', flush=True)
        return True

    env.reset()
    if args.headless:
        for index in range(args.episodes):
            if index:
                env.reset()
            episode()
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0, 0, 0.2]
                viewer.cam.distance = 5
                viewer.cam.elevation = -80
            for index in range(args.episodes):
                if index:
                    with viewer.lock():
                        env.reset()
                if not episode(viewer):
                    break
    if failures:
        raise SystemExit(f'{failures} episodes failed to settle within timeout')


if __name__ == '__main__':
    main()
