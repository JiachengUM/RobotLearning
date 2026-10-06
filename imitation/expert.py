"""无学习的 resolved-rate 专家，动作是关节目标速度（rad/s）。"""
from pathlib import Path
import numpy as np
import mujoco

MODEL_PATH = Path(__file__).resolve().parents[1] / 'mujoco' / 'reach.xml'
OBS_FIELDS = ['q1', 'q2', 'dq1', 'dq2', 'ee_x', 'ee_y',
              'target_x', 'target_y', 'ctrl_q1', 'ctrl_q2']


class Expert:
    def __init__(self, model, gain=2.0, damping=0.05, max_speed=1.5):
        if not all(np.isfinite(x) and x > 0 for x in (gain, damping, max_speed)):
            raise ValueError('Expert parameters must be finite and positive')
        self.model, self.gain, self.damping, self.max_speed = model, gain, damping, max_speed
        self.site = model.site('ee').id
        self.jac = np.zeros((3, model.nv))

    def act(self, data, target):
        mujoco.mj_jacSite(self.model, data, self.jac, None, self.site)
        j = self.jac[:2]
        xdot = self.gain * (target[:2] - data.site_xpos[self.site, :2])
        # DLS 是稳定的伪逆近似，避免接近伸直时速度发散。
        qdot = j.T @ np.linalg.solve(j @ j.T + self.damping**2*np.eye(2), xdot)
        return qdot * min(1., self.max_speed / max(np.max(np.abs(qdot)), 1e-12))


class DemoEnv:
    """复用 reach.xml；50Hz 动作，500Hz 物理，积分动作到 position ctrl。"""
    def __init__(self, seed=7):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.site = self.model.site('ee').id
        self.substeps = 10
        self.dt = self.substeps * self.model.opt.timestep
        self.target = np.zeros(3)

    def observe(self):
        return np.concatenate((self.data.qpos, self.data.qvel,
                               self.data.site_xpos[self.site, :2], self.target[:2], self.data.ctrl)).copy()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.rng.uniform(-np.pi, np.pi, 2)
        radius = np.sqrt(self.rng.uniform(0.4**2, 1.8**2))
        theta = self.rng.uniform(-np.pi, np.pi)
        self.target[:] = [radius*np.cos(theta), radius*np.sin(theta), 0.2]
        self.data.mocap_pos[0] = self.target
        self.data.ctrl[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.5+1e-10:
            raise ValueError('Action must be a finite two-joint velocity in [-1.5, 1.5]')
        for _ in range(self.substeps):
            self.data.ctrl[:] += action * self.model.opt.timestep
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def reached(self):
        return (np.linalg.norm(self.data.site_xpos[self.site]-self.target) < .01
                and np.linalg.norm(self.data.qvel) < .03)
