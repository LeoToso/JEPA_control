"""Dataset generation and loading for JEPA cartpole experiments."""
from __future__ import annotations
import math
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


def _collect_episode(env, episode_length, mode, lqr_gain,
                     action_low, action_high, init_range,
                     lqr_noise_std, rng,
                     pe_action_amplitude=3.0, pe_flip_prob=0.15):
    """Collect exactly one episode of episode_length consecutive steps.

    No mid-episode resets — the done signal from gym is ignored so trajectories
    show the pole falling freely past the gym threshold (12° / 0.21 rad).

    mode:
        'random'  — uniform random actions over full action range
        'lqr'     — LQR + Gaussian noise
        'prbs'    — Pseudo-Random Binary Sequence near equilibrium:
                    hold ±pe_action_amplitude, flip sign with probability
                    pe_flip_prob each step.  Provides persistent excitation
                    with temporal structure for the windowed predictor.
        'passive' — u=0 near equilibrium: captures natural unstable divergence
                    so predictor/DMD can recover rho(A)>1 from data.
    """
    obs, state, _ = env.reset(init_range=init_range)
    obs_list, state_list, action_list = [], [], []
    next_obs_list, next_state_list = [], []

    current_prbs = float(rng.choice([-1, 1])) * pe_action_amplitude

    for step in range(episode_length):
        if mode == 'random':
            u = float(rng.uniform(action_low, action_high))
        elif mode == 'lqr':
            u_lqr = float(np.clip((lqr_gain @ state).item(), action_low, action_high))
            u = float(np.clip(u_lqr + float(rng.normal(0.0, lqr_noise_std)),
                              action_low, action_high))
        elif mode == 'prbs':
            if step == 0:
                current_prbs = float(rng.choice([-1, 1])) * pe_action_amplitude
            elif rng.random() < pe_flip_prob:
                current_prbs = -current_prbs
            u = current_prbs
        elif mode == 'passive':
            u = 0.0
        else:
            raise ValueError(f'Unknown mode: {mode}')

        next_obs, next_state, _, _, _ = env.step(u)
        obs_list.append(obs.copy())
        state_list.append(state.copy())
        action_list.append(np.array([u], dtype=np.float32))
        next_obs_list.append(next_obs.copy())
        next_state_list.append(next_state.copy())
        obs, state = next_obs, next_state

    return {
        'obs':         np.stack(obs_list).astype(np.uint8),
        'states':      np.stack(state_list).astype(np.float32),
        'actions':     np.stack(action_list).astype(np.float32),
        'next_obs':    np.stack(next_obs_list).astype(np.uint8),
        'next_states': np.stack(next_state_list).astype(np.float32),
    }


def _collect_episodes(env, n_episodes, episode_length, mode, lqr_gain,
                      action_low, action_high, init_range,
                      lqr_noise_std, rng, ep_id_offset=0,
                      pe_action_amplitude=3.0, pe_flip_prob=0.15):
    """Collect n_episodes episodes of episode_length steps each.

    Returns flat arrays with episode_ids so the windowed DataLoader can
    extract horizon+1 windows without crossing episode boundaries.
    """
    obs_l, states_l, actions_l = [], [], []
    next_obs_l, next_states_l, ep_ids_l = [], [], []

    for ep_idx in range(n_episodes):
        ep = _collect_episode(
            env, episode_length, mode, lqr_gain,
            action_low, action_high, init_range,
            lqr_noise_std, rng,
            pe_action_amplitude=pe_action_amplitude,
            pe_flip_prob=pe_flip_prob,
        )
        obs_l.append(ep['obs'])
        states_l.append(ep['states'])
        actions_l.append(ep['actions'])
        next_obs_l.append(ep['next_obs'])
        next_states_l.append(ep['next_states'])
        ep_ids_l.append(np.full(episode_length, ep_id_offset + ep_idx, dtype=np.int32))

    return {
        'obs':         np.concatenate(obs_l),
        'states':      np.concatenate(states_l),
        'actions':     np.concatenate(actions_l),
        'next_obs':    np.concatenate(next_obs_l),
        'next_states': np.concatenate(next_states_l),
        'episode_ids': np.concatenate(ep_ids_l),
    }


def generate_dataset(
    # Random action episodes
    n_random_episodes=50, random_ep_len=200,
    random_init_range=1.0,
    # LQR episodes
    n_lqr_episodes=50, lqr_ep_len=200,
    lqr_init_range=0.30, lqr_noise_std=0.25,
    # Near-equilibrium LQR (teaches fp: f(z*,0)≈z*)
    n_equilibrium=500, eq_ep_len=50,
    n_eq_selfloop=200,
    eq_init_range=0.002, eq_noise_std=0.001,
    # PRBS persistent excitation near equilibrium
    n_pe_episodes=0, pe_ep_len=40,
    pe_init_range=0.05, pe_action_amplitude=3.0, pe_flip_prob=0.15,
    # Passive divergence: u=0 near eq → pole falls, teaches rho(A)>1 from data
    n_passive_episodes=0, passive_ep_len=50,
    passive_init_range=0.05,
    # Train/val/test split
    train_frac=0.8, val_frac=0.1,
    # Environment
    frame_skip=1, image_size=64, action_range=(-10.0, 10.0),
    save_path=None, seed=42,
):
    """Generate an episode-centric dataset of cartpole trajectories.

    Each data type is specified as (n_episodes × episode_length) rather than
    a flat transition count.  All episodes run for their full length with no
    mid-episode resets — the gym done signal is ignored so trajectories show
    the pole falling freely past the 12° (0.21 rad) threshold.

    Train/val/test splits are done by shuffling whole episodes before slicing,
    so no episode ever spans two splits.
    """
    from envs.cartpole_visual import ContinuousCartpoleVisual
    rng = np.random.RandomState(seed)
    needs_lqr = (n_lqr_episodes > 0 or n_equilibrium > 0 or n_pe_episodes > 0)
    lqr_gain  = _compute_lqr_gain() if needs_lqr else None
    env = ContinuousCartpoleVisual(frame_skip=frame_skip, image_size=image_size,
                                   action_range=(-10.0, 10.0), seed=seed)
    action_low, action_high = action_range

    all_segments = []
    next_ep_id   = 0

    # ── Random ────────────────────────────────────────────────────────────
    if n_random_episodes > 0:
        n_trans = n_random_episodes * random_ep_len
        print(f'[data] Random:  {n_random_episodes} ep × {random_ep_len} steps'
              f' = {n_trans:,} transitions  (init_range={random_init_range})')
        seg = _collect_episodes(
            env, n_random_episodes, random_ep_len, 'random', lqr_gain,
            action_low, action_high, init_range=random_init_range,
            lqr_noise_std=lqr_noise_std, rng=rng, ep_id_offset=next_ep_id)
        all_segments.append(seg)
        next_ep_id += n_random_episodes

    # ── LQR ───────────────────────────────────────────────────────────────
    if n_lqr_episodes > 0:
        n_trans = n_lqr_episodes * lqr_ep_len
        print(f'[data] LQR:     {n_lqr_episodes} ep × {lqr_ep_len} steps'
              f' = {n_trans:,} transitions  '
              f'(init_range={lqr_init_range}, noise_std={lqr_noise_std})')
        seg = _collect_episodes(
            env, n_lqr_episodes, lqr_ep_len, 'lqr', lqr_gain,
            action_low, action_high, init_range=lqr_init_range,
            lqr_noise_std=lqr_noise_std, rng=rng, ep_id_offset=next_ep_id)
        all_segments.append(seg)
        next_ep_id += n_lqr_episodes

    # ── Near-equilibrium LQR ─────────────────────────────────────────────
    if n_equilibrium > 0:
        n_eq_episodes = max(1, n_equilibrium // eq_ep_len)
        n_trans = n_eq_episodes * eq_ep_len
        print(f'[data] Eq-LQR:  {n_eq_episodes} ep × {eq_ep_len} steps'
              f' = {n_trans:,} transitions  '
              f'(init_range={eq_init_range}, noise_std={eq_noise_std})')
        seg = _collect_episodes(
            env, n_eq_episodes, eq_ep_len, 'lqr', lqr_gain,
            action_low, action_high, init_range=eq_init_range,
            lqr_noise_std=eq_noise_std, rng=rng, ep_id_offset=next_ep_id)
        all_segments.append(seg)
        next_ep_id += n_eq_episodes

    # ── PRBS ─────────────────────────────────────────────────────────────
    if n_pe_episodes > 0:
        n_trans = n_pe_episodes * pe_ep_len
        print(f'[data] PRBS:    {n_pe_episodes} ep × {pe_ep_len} steps'
              f' = {n_trans:,} transitions  '
              f'(init_range={pe_init_range}, amp={pe_action_amplitude},'
              f' p_flip={pe_flip_prob})')
        seg = _collect_episodes(
            env, n_pe_episodes, pe_ep_len, 'prbs', lqr_gain,
            action_low, action_high, init_range=pe_init_range,
            lqr_noise_std=lqr_noise_std, rng=rng, ep_id_offset=next_ep_id,
            pe_action_amplitude=pe_action_amplitude, pe_flip_prob=pe_flip_prob)
        all_segments.append(seg)
        next_ep_id += n_pe_episodes

    # ── Passive ──────────────────────────────────────────────────────────
    if n_passive_episodes > 0:
        n_trans = n_passive_episodes * passive_ep_len
        print(f'[data] Passive: {n_passive_episodes} ep × {passive_ep_len} steps'
              f' = {n_trans:,} transitions  '
              f'(init_range={passive_init_range}, u=0 divergence)')
        seg = _collect_episodes(
            env, n_passive_episodes, passive_ep_len, 'passive', lqr_gain,
            action_low, action_high, init_range=passive_init_range,
            lqr_noise_std=lqr_noise_std, rng=rng, ep_id_offset=next_ep_id)
        all_segments.append(seg)
        next_ep_id += n_passive_episodes

    env.close()

    if not all_segments:
        raise ValueError('No data segments generated — check n_*_episodes parameters.')

    data = {key: np.concatenate([seg[key] for seg in all_segments], axis=0)
            for key in all_segments[0]}

    N = data['obs'].shape[0]
    # Shuffle whole episodes before splitting so no episode straddles two splits
    ep_ids    = data['episode_ids']
    unique_ep = np.unique(ep_ids)
    rng.shuffle(unique_ep)
    ep_order  = {ep: i for i, ep in enumerate(unique_ep)}
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
    actions    = data['actions']
    action_cov = np.cov(actions.T)
    if action_cov.ndim == 0:
        action_cov = float(action_cov); kappa = 1.0
    else:
        ev    = np.maximum(np.linalg.eigvalsh(action_cov), 1e-12)
        kappa = float(ev.max() / ev.min())
    data['splits']                      = splits
    data['action_cov']                  = np.atleast_2d(action_cov)
    data['action_cov_condition_number'] = kappa

    total_eps = next_ep_id
    print(f'[data] Total: {N:,} transitions  '
          f'(train={n_tr:,}, val={n_val:,}, test={N-n_tr-n_val:,})  '
          f'{total_eps} episodes  avg_ep_len={N/total_eps:.0f}')
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
    if 'episode_ids' not in data:
        N = len(data['obs'])
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

    def __init__(self, data, split='train', horizon: int = 20, frame_stack: int = 1,
                 obs_eq: np.ndarray = None, n_eq_selfloop: int = 0):
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
        if obs_eq is not None and n_eq_selfloop > 0:
            self._inject_eq_selfloops(obs_eq, n_eq_selfloop)
        self.valid_starts = self._find_valid_starts()

    def _inject_eq_selfloops(self, obs_eq_np: np.ndarray, n: int) -> None:
        """Append n identical equilibrium self-loop transitions.

        Each transition has obs = next_obs = obs_eq_np, action = 0, state = 0.
        All share one new episode_id so _find_valid_starts sees n-H valid windows.
        pred_loss on these windows forces f(z_star, 0) ≈ z_star directly.
        """
        eq_img = np.tile(obs_eq_np[None], (n, 1, 1, 1)).astype(np.uint8)
        eq_act = np.zeros((n, 1), dtype=np.float32)
        eq_st  = np.zeros((n, 4), dtype=np.float32)
        new_ep = int(self.ep_ids.max()) + 1
        eq_ep  = np.full(n, new_ep, dtype=np.int32)
        self.obs         = np.concatenate([self.obs,         eq_img], axis=0)
        self.next_obs    = np.concatenate([self.next_obs,    eq_img], axis=0)
        self.actions     = np.concatenate([self.actions,     eq_act], axis=0)
        self.states      = np.concatenate([self.states,      eq_st],  axis=0)
        self.next_states = np.concatenate([self.next_states, eq_st],  axis=0)
        self.ep_ids      = np.concatenate([self.ep_ids,      eq_ep],  axis=0)

    def _find_valid_starts(self):
        H  = self.horizon
        ep = self.ep_ids
        N  = len(ep)
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
            return torch.cat([prev_t, curr_t], dim=0) if FS > 1 else curr_t

        frames = []
        for k in range(H):
            curr_t = _to_tensor(self.obs[start + k])
            prev_t = _to_tensor(self.obs[start + k - 1]) if k > 0 else curr_t
            frames.append(_stack_frames(prev_t, curr_t))
        prev_last = _to_tensor(self.obs[start + H - 1])
        curr_last = _to_tensor(self.next_obs[start + H - 1])
        frames.append(_stack_frames(prev_last, curr_last))
        obs_seq = torch.stack(frames)                                  # (H+1, 3*FS, h, w)

        actions = torch.from_numpy(self.actions[start:start + H])     # (H, 1)

        states_list = [self.states[start + k] for k in range(H)]
        states_list.append(self.next_states[start + H - 1])
        states = torch.from_numpy(np.stack(states_list))               # (H+1, 4)

        return {'obs_seq': obs_seq, 'actions': actions, 'states': states}


def make_dataloaders(data, batch_size=256, num_workers=0, horizon=1, frame_stack=1,
                     obs_eq: np.ndarray = None, n_eq_selfloop: int = 0):
    """Return dataloaders. horizon=1 -> TransitionDataset; horizon>1 -> TrajectoryDataset."""
    loaders = {}
    for split in ('train', 'val', 'test'):
        if split not in data.get('splits', {}):
            continue
        if horizon > 1:
            ds = TrajectoryDataset(data, split=split, horizon=horizon,
                                   frame_stack=frame_stack,
                                   obs_eq=(obs_eq if split == 'train' else None),
                                   n_eq_selfloop=(n_eq_selfloop if split == 'train' else 0))
        else:
            ds = TransitionDataset(data, split=split)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=num_workers, pin_memory=True,
            drop_last=(split == 'train'),
        )
    return loaders
