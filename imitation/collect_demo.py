"""采集 Reach 专家示范：python imitation/collect_demo.py --episodes 500。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
if __package__:
    from .expert import DemoEnv, Expert, OBS_FIELDS, MODEL_PATH
    from .dataset import ReachDataset
else:
    from expert import DemoEnv, Expert, OBS_FIELDS, MODEL_PATH
    from dataset import ReachDataset


def plot_dataset(dataset, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = dataset.arrays
    offsets = a['episode_offsets']
    targets = a['obs'][offsets[:-1], 6:8]
    lengths = np.diff(offsets)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    axes[0].scatter(*targets.T, s=9, alpha=.65)
    axes[0].set(title='Target distribution', xlabel='x (m)', ylabel='y (m)', aspect='equal')
    axes[1].hist(lengths, bins=25, color='steelblue')
    axes[1].set(title='Trajectory length', xlabel='Control steps (0.02 s each)', ylabel='Episodes')
    axes[2].hist(a['action'][:, 0], bins=60, alpha=.6, label='Shoulder')
    axes[2].hist(a['action'][:, 1], bins=60, alpha=.6, label='Elbow')
    axes[2].set(title='Action histogram', xlabel='Joint target velocity (rad/s)', ylabel='Transitions')
    axes[2].legend()
    successes = int(a['terminated'].sum())
    fig.suptitle(f'Reach expert | {len(lengths)} episodes | {len(dataset):,} transitions | {successes/len(lengths):.1%} success')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def collect(episodes=500, seed=7, timeout=15., output=Path(__file__).with_name('reach_dataset.npz')):
    if episodes <= 0 or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('episodes and timeout must be positive')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output exists: {output}; choose another --output')
    env = DemoEnv(seed)
    expert = Expert(env.model)
    rows = {key: [] for key in ('obs', 'action', 'next_obs', 'done', 'terminated', 'truncated', 'episode_id')}
    offsets = [0]
    max_steps = int(np.ceil(timeout/env.dt))
    for ep in range(episodes):
        obs = env.reset()
        stable = 0
        for step in range(max_steps):
            action = expert.act(env.data, env.target)
            next_obs = env.step(action)
            stable = stable+1 if env.reached() else 0
            terminated = stable >= 10
            truncated = step == max_steps-1 and not terminated
            for key, value in dict(obs=obs, action=action, next_obs=next_obs,
                                   done=terminated or truncated, terminated=terminated,
                                   truncated=truncated, episode_id=ep).items():
                rows[key].append(value)
            obs = next_obs
            if terminated or truncated:
                break
        offsets.append(len(rows['action']))
        if (ep+1) % 25 == 0 or ep+1 == episodes:
            print(f'{ep+1}/{episodes} episodes; successes={sum(rows["terminated"])}; transitions={offsets[-1]}', flush=True)
    arrays = {key: np.asarray(value, dtype=np.float32 if key in ('obs', 'action', 'next_obs')
                              else np.int64 if key == 'episode_id' else bool) for key, value in rows.items()}
    arrays['episode_offsets'] = np.asarray(offsets, dtype=np.int64)
    metadata = dict(schema_version=1, seed=seed, episodes=episodes, control_dt=env.dt,
                    physics_dt=env.model.opt.timestep, substeps=env.substeps,
                    obs_fields=OBS_FIELDS, action='joint target velocity rad/s; integrate into position ctrl',
                    max_speed=expert.max_speed, gain=expert.gain, damping=expert.damping,
                    timeout=timeout, success_position_tolerance=.01, success_speed_tolerance=.03,
                    success_hold_steps=10, mujoco_version=mujoco.__version__, numpy_version=np.__version__,
                    model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest())
    arrays['metadata'] = np.array(json.dumps(metadata))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    dataset = ReachDataset(output)
    plot_dataset(dataset, output.with_suffix('.png'))
    summary = dict(episodes=episodes, transitions=len(dataset), successes=int(arrays['terminated'].sum()),
                   timeouts=int(arrays['truncated'].sum()), mean_steps=float(np.mean(np.diff(offsets))),
                   max_steps=int(np.max(np.diff(offsets))))
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--timeout', type=float, default=15.)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('reach_dataset.npz'))
    args = parser.parse_args()
    collect(args.episodes, args.seed, args.timeout, args.output)
