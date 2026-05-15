"""Dataset generation and loading for JEPA cartpole experiments."""
from __future__ import annotations
import os, warnings
from pathlib import Path
from typing import Dict, Optional
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _compute_lqr_gain():
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    import scipy.linalg
    gt = CartpoleGroundTruth()
    A, B = gt.A_star, gt.B_star
    Q = np.diag([1.0, 1.0, 10.0, 1.0])
    R = np.array([[0.01]])
    try:
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    except Exception:
        warnings.warn('DARE failed; falling back to pole-placement gain.')
        K = np.array([[0.0, 0.0, 10.0, 0.0]])
    return K


def _collect_transitions(env, n_transitions, mode, lqr_gain,
                         action_low, action_high, init_range,
                         lqr_noise_std, rng):
    obs_list, state_list, action_list = [], [], []
    next_obs_list, next_state_list, ep_id_list = [], [], []
    obs, state, _ = env.reset(init_range=init_range)
    steps_since_reset = 0
    collected = 0
    episode_id = 0
    while collected < n_transitions:
        if mode == 'random':
            u = float(rng.uniform(action_low, action_high))
        elif mode == 'lqr':
            u_lqr = float(np.clip((lqr_gain @ state).item(), action_low, action_high))
            u = float(np.clip(u_lqr + float(rng.normal(0.0, lqr_noise_std)),
                              action_low, action_high))
        else:
            raise ValueError(f'Unknown mode: {mode}')
        next_obs, next_state, _, done, _ = env.step(u)
        obs_list.append(obs.copy())
        state_list.append(state.copy())
        action_list.append(np.array([u], dtype=np.float32))
        next_obs_list.append(next_obs.copy())
        next_state_list.append(next_state.copy())
        ep_id_list.append(episode_id)
        collected += 1
        steps_since_reset += 1
        if done or steps_since_reset > 200:
            obs, state, _ = env.reset(init_range=init_range)
            steps_since_reset = 0
            episode_id += 1
        else:
            obs, state = next_obs, next_state
    return {
        'obs':         np.stack(obs_list).astype(np.uint8),
        'states':      np.stack(state_list).astype(np.float32),
        'actions':     np.stack(action_list).astype(np.float32),
        'next_obs':    np.stack(next_obs_list).astype(np.uint8),
        'next_states': np.stack(next_state_list).astype(np.float32),
        'episode_ids': np.array(ep_id_list, dtype=np.int32),
    }


def generate_dataset(dataset_type='random', n_transitions=50000, frame_skip=1,
                     save_path=None, seed=42, train_frac=0.8, val_frac=0.1,
                     action_range=(-5.0, 5.0), init_range=0.1,
                     lqr_init_range=None, lqr_noise_std=0.1, image_size=64,
                     n_equilibrium=0, eq_init_range=0.002, eq_noise_std=0.001):
    from envs.cartpole_visual import ContinuousCartpoleVisual
    rng = np.random.RandomState(seed)
    lqr_gain = _compute_lqr_gain() if dataset_type in ('lqr', 'mixed') else None
    env = ContinuousCartpoleVisual(frame_skip=frame_skip, image_size=image_size,
                                   action_range=(-10.0, 10.0), seed=seed)
    action_low, action_high = action_range
    if lqr_init_range is None:
        lqr_init_range = max(0.05, init_range * 0.5)
    if dataset_type == 'mixed':
        n_random = n_transitions // 2
        n_lqr    = n_transitions - n_random
        data_rand = _collect_transitions(env, n_random, 'random', lqr_gain,
                                         action_low, action_high,
                                         init_range=init_range,
                                         lqr_noise_std=lqr_noise_std, rng=rng)
        data_lqr  = _collect_transitions(env, n_lqr, 'lqr', lqr_gain,
                                          action_low, action_high,
                                          init_range=lqr_init_range,
                                          lqr_noise_std=lqr_noise_std, rng=rng)
        # Offset episode IDs in the second segment so they are globally unique
        data_lqr['episode_ids'] += data_rand['episode_ids'].max() + 1
        data = {key: np.concatenate([data_rand[key], data_lqr[key]], axis=0)
                for key in data_rand}
        # Optional near-equilibrium sequences: teach predictor f(z*,0)≈z*.
        # The predictor never sees "stay at rest" otherwise (random init_range > 0).
        if n_equilibrium > 0:
            data_eq = _collect_transitions(env, n_equilibrium, 'lqr', lqr_gain,
                                           action_low, action_high,
                                           init_range=eq_init_range,
                                           lqr_noise_std=eq_noise_std, rng=rng)
            data_eq['episode_ids'] += data['episode_ids'].max() + 1
            data = {key: np.concatenate([data[key], data_eq[key]], axis=0)
                    for key in data}
            print(f'[data] Added {n_equilibrium} equilibrium transitions '
                  f'(init_range={eq_init_range}, noise_std={eq_noise_std})')
    else:
        init_r = 0.05 if dataset_type == 'lqr' else init_range
        data = _collect_transitions(env, n_transitions, dataset_type, lqr_gain,
                                    action_low, action_high,
                                    init_range=init_r,
                                    lqr_noise_std=lqr_noise_std, rng=rng)
    env.close()
    N = data['obs'].shape[0]
    # Shuffle by episode to avoid leaking future info across splits
    ep_ids   = data['episode_ids']
    unique_ep = np.unique(ep_ids)
    rng.shuffle(unique_ep)
    ep_order = {ep: i for i, ep in enumerate(unique_ep)}
    sort_idx  = np.argsort([ep_order[e] for e in ep_ids], kind='stable')
    for key in data:
        data[key] = data[key][sort_idx]
    N     = len(data['obs'])
    n_tr  = int(N * train_frac)
    n_val = int(N * val_frac)
    splits = {
        'train': np.arange(0,    n_tr),
        'val':   np.arange(n_tr, n_tr + n_val),
        'test':  np.arange(n_tr + n_val, N),
    }
    actions = data['actions']
    action_cov = np.cov(actions.T)
    if action_cov.ndim == 0:
        action_cov = float(action_cov); kappa = 1.0
    else:
        ev = np.maximum(np.linalg.eigvalsh(action_cov), 1e-12)
        kappa = float(ev.max() / ev.min())
    data['splits'] = splits
    data['action_cov'] = np.atleast_2d(action_cov)
    data['action_cov_condition_number'] = kappa
    if save_path is not None:
        _save_hdf5(data, save_path, splits)
    return data


def _save_hdf5(data, path, splits):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with h5py.File(path, 'w') as f:
        for key in ['obs', 'states', 'actions', 'next_obs', 'next_states', 'episode_ids']:
            f.create_dataset(key, data=data[key], compression='gzip', compression_opts=4)
        f.create_dataset('action_cov', data=data['action_cov'])
        f.attrs['action_cov_condition_number'] = float(data['action_cov_condition_number'])
        grp = f.create_group('splits')
        for split_name, idx in splits.items():
            grp.create_dataset(split_name, data=idx)


def load_dataset(path):
    data = {}
    with h5py.File(path, 'r') as f:
        for key in ['obs', 'states', 'actions', 'next_obs', 'next_states',
                    'action_cov', 'episode_ids']:
            if key in f:
                data[key] = f[key][:]
        data['action_cov_condition_number'] = float(
            f.attrs.get('action_cov_condition_number', 1.0))
        splits = {}
        if 'splits' in f:
            for split_name in f['splits']:
                splits[split_name] = f['splits'][split_name][:]
        data['splits'] = splits
    # Backward compat: datasets without episode_ids get a dummy column
    if 'episode_ids' not in data:
        N = len(data['obs'])
        # Treat each transition as its own episode (no multi-step windows)
        data['episode_ids'] = np.arange(N, dtype=np.int32)
    return data


# ── Single-step dataset (used for probes / backward compat) ──────────────────

class TransitionDataset(Dataset):
    def __init__(self, data, split='train'):
        splits = data.get('splits', {})
        idx = splits[split] if splits and split in splits else np.arange(len(data['obs']))
        self.obs        = data['obs'][idx]
        self.states     = data['states'][idx]
        self.actions    = data['actions'][idx]
        self.next_obs   = data['next_obs'][idx]
        self.next_states= data['next_states'][idx]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs      = torch.from_numpy(self.obs[idx]).float().permute(2, 0, 1) / 255.0
        next_obs = torch.from_numpy(self.next_obs[idx]).float().permute(2, 0, 1) / 255.0
        return {
            'obs':        obs,
            'state':      torch.from_numpy(self.states[idx]),
            'action':     torch.from_numpy(self.actions[idx]),
            'next_obs':   next_obs,
            'next_state': torch.from_numpy(self.next_states[idx]),
        }


# ── Multi-step trajectory dataset ────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """Returns windows of H+1 consecutive frames from the same episode.

    Each item:
      obs_seq  : (H+1, 3*frame_stack, img_h, img_w) float32 in [0,1]
      actions  : (H, 1)  float32
      states   : (H+1, 4) float32

    When frame_stack > 1, each observation is [obs_{t-1}, obs_t] stacked
    channel-wise. At the start of a window (k=0), prev = curr (duplicate).
    """

    def __init__(self, data, split='train', horizon: int = 20, frame_stack: int = 1):
        splits = data.get('splits', {})
        idx    = (splits[split] if splits and split in splits
                  else np.arange(len(data['obs'])))
        self.obs        = data['obs'][idx]
        self.states     = data['states'][idx]
        self.actions    = data['actions'][idx]
        self.next_obs   = data['next_obs'][idx]
        self.next_states= data['next_states'][idx]
        self.ep_ids     = (data['episode_ids'][idx]
                           if 'episode_ids' in data
                           else np.arange(len(idx), dtype=np.int32))
        self.horizon     = horizon
        self.frame_stack = frame_stack
        self.valid_starts = self._find_valid_starts()

    def _find_valid_starts(self):
        H  = self.horizon
        ep = self.ep_ids
        N  = len(ep)
        # A window [i, i+H) is valid iff all steps share the same episode id.
        # Vectorised: compare ep[i] with ep[i+1..i+H-1].
        valid = []
        for i in range(N - H):
            if np.all(ep[i:i + H] == ep[i]):
                valid.append(i)
        return np.array(valid, dtype=np.int64)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = int(self.valid_starts[idx])
        H     = self.horizon
        FS    = self.frame_stack

        def _to_tensor(arr):
            return torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0  # (3, h, w)

        def _stack_frames(prev_t, curr_t):
            # prev_t, curr_t: (3, h, w) tensors; returns (3*FS, h, w)
            return torch.cat([prev_t, curr_t], dim=0) if FS > 1 else curr_t

        # Frames: obs[start], obs[start+1], ..., obs[start+H-1], next_obs[start+H-1]
        # With frame stacking, each position gets [obs_{t-1}, obs_t] channel-wise.
        # At k=0, prev = curr (duplicate first frame since no prior observation).
        frames = []
        for k in range(H):
            curr_t = _to_tensor(self.obs[start + k])
            prev_t = _to_tensor(self.obs[start + k - 1]) if k > 0 else curr_t
            frames.append(_stack_frames(prev_t, curr_t))
        # Last frame: prev = obs[start+H-1], curr = next_obs[start+H-1]
        prev_last = _to_tensor(self.obs[start + H - 1])
        curr_last = _to_tensor(self.next_obs[start + H - 1])
        frames.append(_stack_frames(prev_last, curr_last))
        obs_seq = torch.stack(frames)                                  # (H+1, 3*FS, h, w)

        actions = torch.from_numpy(self.actions[start:start + H])     # (H, 1)

        # States: state[start..start+H-1] + next_state[start+H-1]
        states_list = [self.states[start + k] for k in range(H)]
        states_list.append(self.next_states[start + H - 1])
        states = torch.from_numpy(np.stack(states_list))               # (H+1, 4)

        return {'obs_seq': obs_seq, 'actions': actions, 'states': states}


def make_dataloaders(data, batch_size=256, num_workers=0, horizon=1, frame_stack=1):
    """Return dataloaders. horizon=1 -> TransitionDataset; horizon>1 -> TrajectoryDataset."""
    loaders = {}
    for split in ('train', 'val', 'test'):
        if split not in data.get('splits', {}):
            continue
        if horizon > 1:
            ds = TrajectoryDataset(data, split=split, horizon=horizon,
                                   frame_stack=frame_stack)
        else:
            ds = TransitionDataset(data, split=split)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=num_workers, pin_memory=True,
            drop_last=(split == 'train'),
        )
    return loaders
