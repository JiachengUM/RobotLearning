"""MuJoCo site Jacobian → DLS IK → 限速关节目标 → XML 位置伺服器。"""
import numpy as np
import mujoco


class ReachController:
    def __init__(self, model, damping=0.05, max_speed=1.5):
        if damping <= 0 or max_speed <= 0:
            raise ValueError('damping and max_speed must be positive')
        self.model = model
        self.damping = damping
        self.max_speed = max_speed
        self.site = model.site('ee').id
        self.scratch = mujoco.MjData(model)
        self.goal = None
        self.command = None

    def solve_ik(self, target, q0):
        """用独立 MjData 做 IK，不修改真实仿真 qpos；只控制 XY 位置。"""
        target = np.asarray(target, dtype=float)
        q0 = np.asarray(q0, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('target must be a finite XYZ vector')
        if q0.shape != (2,) or not np.isfinite(q0).all():
            raise ValueError('q0 must be a finite two-joint vector')
        if abs(target[2]-0.2) > 1e-8 or np.linalg.norm(target[:2]) > 2:
            raise ValueError('Target must be in the reachable z=0.2 plane')
        jac = np.zeros((3, self.model.nv))
        # 弯曲重启避开完全伸直时径向误差导致的 Jacobian 驻点。
        for seed in (q0, q0 + [0.2, 0.6], q0 + [-0.2, -0.6]):
            self.scratch.qpos[:] = seed
            for _ in range(600):
                mujoco.mj_forward(self.model, self.scratch)
                error = target[:2]-self.scratch.site_xpos[self.site, :2]
                if np.linalg.norm(error) < 1e-6:
                    q = self.scratch.qpos.copy()
                    return q0 + (q-q0+np.pi) % (2*np.pi)-np.pi
                mujoco.mj_jacSite(self.model, self.scratch, jac, None, self.site)
                j = jac[:2]
                delta = j.T @ np.linalg.solve(j @ j.T+self.damping**2*np.eye(2), error)
                delta *= min(1., 0.2/max(np.linalg.norm(delta), 1e-12))
                self.scratch.qpos[:] += delta
        raise RuntimeError('DLS IK did not converge')

    def set_target(self, target, data):
        self.goal = self.solve_ik(target, data.qpos.copy())
        self.command = data.qpos.copy()

    def update(self, data):
        if self.goal is None:
            raise RuntimeError('Call set_target before update')
        step = self.max_speed*self.model.opt.timestep
        self.command += np.clip(self.goal-self.command, -step, step)
        # position actuator 的 ctrl 是目标角度，不是力矩或直接设置 qpos。
        data.ctrl[:] = self.command
