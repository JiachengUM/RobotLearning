"""NPZ 数据集读取、完整性检查；不依赖 PyTorch，不允许 pickle。"""
import json
import numpy as np


class ReachDataset:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as file:
            self.arrays = {key: file[key] for key in file.files}
        self.metadata = json.loads(str(self.arrays['metadata']))
        self.validate()

    def __len__(self):
        return len(self.arrays['action'])

    def __getitem__(self, index):
        return {key: self.arrays[key][index].copy() for key in
                ('obs', 'action', 'next_obs', 'done', 'terminated', 'truncated', 'episode_id')}

    def episode(self, index):
        offsets = self.arrays['episode_offsets']
        if not 0 <= index < len(offsets)-1:
            raise IndexError(index)
        return self[slice(offsets[index], offsets[index+1])]

    def validate(self):
        a = self.arrays
        n = len(a['action'])
        for key, width in [('obs', 10), ('action', 2), ('next_obs', 10)]:
            if a[key].shape != (n, width) or not np.isfinite(a[key]).all():
                raise ValueError(f'Invalid {key}')
        for key in ('done', 'terminated', 'truncated', 'episode_id'):
            if a[key].shape != (n,):
                raise ValueError(f'Invalid {key} shape')
        offsets = a['episode_offsets']
        if offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) <= 0):
            raise ValueError('Invalid episode boundaries')
        if not np.array_equal(a['done'], a['terminated'] | a['truncated']):
            raise ValueError('Inconsistent done flags')
        for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
            if not a['done'][end-1] or a['done'][start:end-1].any():
                raise ValueError('Invalid terminal step')
            if not np.all(a['episode_id'][start:end] == i):
                raise ValueError('Invalid episode IDs')
            if not np.array_equal(a['next_obs'][start:end-1], a['obs'][start+1:end]):
                raise ValueError('Transition discontinuity')
        if np.max(np.abs(a['action'])) > 1.50001:
            raise ValueError('Action exceeds velocity bound')

    def split_episodes(self, validation_fraction=.2, seed=0):
        """按回合分割，避免同一轨迹的相邻帧泄漏到验证集。返回回合 ID。"""
        if not 0 < validation_fraction < 1:
            raise ValueError('validation_fraction must be in (0,1)')
        count = len(self.arrays['episode_offsets'])-1
        if count < 2:
            raise ValueError('At least two episodes are required')
        ids = np.random.default_rng(seed).permutation(count)
        nval = min(count-1, max(1, round(count*validation_fraction)))
        return ids[nval:], ids[:nval]
