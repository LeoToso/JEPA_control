"""Dataset generation and loading for JEPA cartpole experiments."""
from __future__ import annotations
import math
import os, warnings
from pathlib import Path
from typing import Dict, Optional
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


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
    # Standard LQR: u* = -K @ x.  Return -K so callers can write u = gain @ x.
    return -K


def _collect_episode(env, episode_length, mode, lqr_gain,
                     action_low, action_high, init_range,
                     lqr_noise_std, rng,
                     pe_action_amplitude=3.0, pe_flip_prob=0.15,
                     max_abs_x=2.2, max_abs_theta=1.2,
                     angle_min=0.1, angle_max=1.0,
                     angle_x_range=0.1, angle_velocity_range=0.2,
                     angle_action_noise_std=0.5):
    """Collect up to ``episode_length`` fully observed macro-transitions.

    A terminal transition is discarded. This guarantees that every stored
    transition contains exactly ``frame_skip`` physics steps and prevents
    assigning off-screen or otherwise ambiguous images to extreme states.

    mode:
        'random'  — uniform random actions over full action range
        'lqr'     — LQR + Gaussian noise
        'prbs'    — Pseudo-Random Binary Sequence near equilibrium:
                    hold ±pe_action_amplitude, flip sign with probability
                    pe_flip_prob each step.  Provides persistent excitation
                    with temporal structure for the windowed predictor.
        'passive' — u=0 near equilibrium: captures natural unstable divergence.
        'angle_sweep' — broad signed pole angles with small cart/velocity state
                        and stabilizing LQR plus action noise.
    """
    if mode == 'angle_sweep':
        sign = float(rng.choice([-1.0, 1.0]))
        theta = sign * float(rng.uniform(angle_min, angle_max))
        state0 = np.array([
            rng.uniform(-angle_x_range, angle_x_range),
            rng.uniform(-angle_velocity_range, angle_velocity_range),
            theta,
            rng.uniform(-angle_velocity_range, angle_velocity_range),
        ], dtype=np.float32)
        obs, state, _ = env.reset_to_state(state0)
    else:
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
        elif mode == 'angle_sweep':
            u_lqr = float(np.clip((lqr_gain @ state).item(), action_low, action_high))
            u = float(np.clip(
                u_lqr + rng.normal(0.0, angle_action_noise_std),
                action_low, action_high))
        else:
            raise ValueError(f'Unknown mode: {mode}')

        next_obs, next_state, _, done, _ = env.step(u)
        observable = (abs(float(next_state[0])) <= max_abs_x
                      and abs(float(next_state[2])) <= max_abs_theta)
        if done or not observable:
            break
        obs_list.append(obs.copy())
        state_list.append(state.copy())
        action_list.append(np.array([u], dtype=np.float32))
        next_obs_list.append(next_obs.copy())
        next_state_list.append(next_state.copy())
        obs, state = next_obs, next_state

    if not obs_list:
        return None
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
                      pe_action_amplitude=3.0, pe_action_amplitudes=None,
                      pe_flip_prob=0.15,
                      max_abs_x=2.2, max_abs_theta=1.2,
                      trajectory_type=None,
                      angle_min=0.1, angle_max=1.0,
                      angle_x_range=0.1, angle_velocity_range=0.2,
                      angle_action_noise_std=0.5):
    """Collect n_episodes episodes of episode_length steps each.

    Returns flat arrays with episode_ids so the windowed DataLoader can
    extract horizon+1 windows without crossing episode boundaries.
    """
    obs_l, states_l, actions_l = [], [], []
    next_obs_l, next_states_l, ep_ids_l, type_l = [], [], [], []

    for ep_idx in range(n_episodes):
        episode_pe_amplitude = pe_action_amplitude
        if pe_action_amplitudes:
            episode_pe_amplitude = float(
                pe_action_amplitudes[ep_idx % len(pe_action_amplitudes)])
        ep = _collect_episode(
            env, episode_length, mode, lqr_gain,
            action_low, action_high, init_range,
            lqr_noise_std, rng,
            pe_action_amplitude=episode_pe_amplitude,
            pe_flip_prob=pe_flip_prob,
            max_abs_x=max_abs_x, max_abs_theta=max_abs_theta,
            angle_min=angle_min, angle_max=angle_max,
            angle_x_range=angle_x_range,
            angle_velocity_range=angle_velocity_range,
            angle_action_noise_std=angle_action_noise_std,
        )
        if ep is None:
            continue
        n_ep = len(ep['actions'])
        if n_ep == 0:
            continue
        obs_l.append(ep['obs'])
        states_l.append(ep['states'])
        actions_l.append(ep['actions'])
        next_obs_l.append(ep['next_obs'])
        next_states_l.append(ep['next_states'])
        ep_ids_l.append(np.full(n_ep, ep_id_offset + ep_idx, dtype=np.int32))
        type_l.append(np.full(n_ep, trajectory_type or mode, dtype='S16'))

    if not obs_l:
        raise RuntimeError(
            f'No observable transitions collected for trajectory type {trajectory_type or mode!r}.')
    return {
        'obs':         np.concatenate(obs_l),
        'states':      np.concatenate(states_l),
        'actions':     np.concatenate(actions_l),
        'next_obs':    np.concatenate(next_obs_l),
        'next_states': np.concatenate(next_states_l),
        'episode_ids': np.concatenate(ep_ids_l),
        'trajectory_types': np.concatenate(type_l),
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
    pe_init_range=0.05, pe_action_amplitude=3.0,
    pe_action_amplitudes=None, pe_flip_prob=0.15,
    # Passive divergence: u=0 near eq → pole falls, teaches rho(A)>1 from data
    n_passive_episodes=0, passive_ep_len=50,
    passive_init_range=0.05,
    # Broad-angle episodes: explicitly make pole orientation observable.
    n_angle_episodes=0, angle_ep_len=12,
    angle_min=0.1, angle_max=1.0,
    angle_x_range=0.1, angle_velocity_range=0.2,
    angle_action_noise_std=0.5,
    # Train/val/test split
    train_frac=0.8, val_frac=0.1,
    # Environment
    frame_skip=1, image_size=64, action_range=(-10.0, 10.0),
    max_abs_x=2.2, max_abs_theta=1.2,
    save_path=None, seed=42,
):
    """Generate an episode-centric dataset of cartpole trajectories.

    Episode lengths are upper bounds. Collection stops before a terminal or
    visually ambiguous transition, so stored macro-transitions always contain
    exactly ``frame_skip`` physics steps and remain inside the rendered region.

    Train/val/test splits are done by shuffling whole episodes before slicing,
    so no episode ever spans two splits.
    """
    from envs.cartpole_visual import ContinuousCartpoleVisual
    rng = np.random.RandomState(seed)
    needs_lqr = (n_lqr_episodes > 0 or n_equilibrium > 0
                 or n_pe_episodes > 0 or n_angle_episodes > 0)
    lqr_gain  = _compute_lqr_gain() if needs_lqr else None
    env = ContinuousCartpoleVisual(frame_skip=frame_skip, image_size=image_size,
                                   action_range=(-10.0, 10.0),
                                   theta_threshold=max_abs_theta, seed=seed)
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
            lqr_noise_std=lqr_noise_std, rng=rng, ep_id_offset=next_ep_id,
            max_abs_x=max_abs_x, max_abs_theta=max_abs_theta,
            trajectory_type='random')
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
            lqr_noise_std=lqr_noise_std, rng=rng, ep_id_offset=next_ep_id,
            max_abs_x=max_abs_x, max_abs_theta=max_abs_theta,
            trajectory_type='lqr')
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
            lqr_noise_std=eq_noise_std, rng=rng, ep_id_offset=next_ep_id,
            max_abs_x=max_abs_x, max_abs_theta=max_abs_theta,
            trajectory_type='eq_lqr')
        all_segments.append(seg)
        next_ep_id += n_eq_episodes

    # ── PRBS ─────────────────────────────────────────────────────────────
    if n_pe_episodes > 0:
        n_trans = n_pe_episodes * pe_ep_len
        print(f'[data] PRBS:    {n_pe_episodes} ep × {pe_ep_len} steps'
              f' = {n_trans:,} transitions  '
              f'(init_range={pe_init_range}, '
              f'amp={pe_action_amplitudes or pe_action_amplitude},'
              f' p_flip={pe_flip_prob})')
        seg = _collect_episodes(
            env, n_pe_episodes, pe_ep_len, 'prbs', lqr_gain,
            action_low, action_high, init_range=pe_init_range,
            lqr_noise_std=lqr_noise_std, rng=rng, ep_id_offset=next_ep_id,
            pe_action_amplitude=pe_action_amplitude,
            pe_action_amplitudes=pe_action_amplitudes,
            pe_flip_prob=pe_flip_prob,
            max_abs_x=max_abs_x, max_abs_theta=max_abs_theta,
            trajectory_type='prbs')
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
            lqr_noise_std=lqr_noise_std, rng=rng, ep_id_offset=next_ep_id,
            max_abs_x=max_abs_x, max_abs_theta=max_abs_theta,
            trajectory_type='passive')
        all_segments.append(seg)
        next_ep_id += n_passive_episodes

    # ── Broad-angle supervised trajectories ──────────────────────────────
    if n_angle_episodes > 0:
        n_trans = n_angle_episodes * angle_ep_len
        print(f'[data] Angle:   {n_angle_episodes} ep × {angle_ep_len} steps'
              f' = up to {n_trans:,} transitions  '
              f'(|theta0|=[{angle_min},{angle_max}])')
        seg = _collect_episodes(
            env, n_angle_episodes, angle_ep_len, 'angle_sweep', lqr_gain,
            action_low, action_high, init_range=0.0,
            lqr_noise_std=lqr_noise_std, rng=rng, ep_id_offset=next_ep_id,
            max_abs_x=max_abs_x, max_abs_theta=max_abs_theta,
            trajectory_type='angle_sweep',
            angle_min=angle_min, angle_max=angle_max,
            angle_x_range=angle_x_range,
            angle_velocity_range=angle_velocity_range,
            angle_action_noise_std=angle_action_noise_std)
        all_segments.append(seg)
        next_ep_id += n_angle_episodes

    env.close()

    if not all_segments:
        raise ValueError('No data segments generated — check n_*_episodes parameters.')

    data = {key: np.concatenate([seg[key] for seg in all_segments], axis=0)
            for key in all_segments[0]}

    N = data['obs'].shape[0]
    # Split whole episodes while balancing the signed angle distribution
    # inside every trajectory family. Sorting by mean theta and assigning each
    # block across train/val/test prevents the sign shifts seen in random splits.
    ep_ids    = data['episode_ids']
    ep_types  = data['trajectory_types']
    split_ep_ids = {'train': [], 'val': [], 'test': []}
    unique_types = np.unique(ep_types)
    val_stride = max(2, int(round(1.0 / max(val_frac, 1e-6))))
    test_frac = max(0.0, 1.0 - train_frac - val_frac)
    test_stride = max(2, int(round(1.0 / max(test_frac, 1e-6))))
    for typ in unique_types:
        type_eps = np.unique(ep_ids[ep_types == typ])
        rng.shuffle(type_eps)  # random tie-breaking before stable angle sort
        angle_score = {
            int(ep): float(data['states'][ep_ids == ep, 2].mean())
            for ep in type_eps
        }
        type_eps = np.array(sorted(type_eps, key=lambda ep: angle_score[int(ep)]))
        for rank, ep in enumerate(type_eps):
            # Offset test from validation so both span negative-to-positive theta.
            if len(type_eps) >= 3 and rank % val_stride == 0:
                split_ep_ids['val'].append(int(ep))
            elif len(type_eps) >= 3 and rank % test_stride == test_stride // 2:
                split_ep_ids['test'].append(int(ep))
            else:
                split_ep_ids['train'].append(int(ep))
        # Ensure every sufficiently large family appears in all splits.
        if len(type_eps) >= 3:
            for split_name in ('val', 'test'):
                if not any(ep in set(type_eps.tolist())
                           for ep in split_ep_ids[split_name]):
                    split_ep_ids[split_name].append(
                        split_ep_ids['train'].pop())
    ordered_eps = (split_ep_ids['train'] + split_ep_ids['val'] + split_ep_ids['test'])
    ep_order  = {ep: i for i, ep in enumerate(ordered_eps)}
    sort_idx  = np.argsort([ep_order[e] for e in ep_ids], kind='stable')
    for key in data:
        data[key] = data[key][sort_idx]

    N     = len(data['obs'])
    n_tr  = int(np.isin(data['episode_ids'], split_ep_ids['train']).sum())
    n_val = int(np.isin(data['episode_ids'], split_ep_ids['val']).sum())
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
    data['action_scale'] = np.float32(max(abs(action_low), abs(action_high)))
    train_states = data['states'][splits['train']]
    data['state_mean'] = train_states.mean(axis=0).astype(np.float32)
    data['state_std']  = np.maximum(train_states.std(axis=0), 1e-8).astype(np.float32)

    total_eps = len(np.unique(data['episode_ids']))
    print(f'[data] Total: {N:,} transitions  '
          f'(train={n_tr:,}, val={n_val:,}, test={N-n_tr-n_val:,})  '
          f'{total_eps} episodes  avg_ep_len={N/total_eps:.0f}')
    if save_path is not None:
        _save_hdf5(data, save_path, splits)
    return data


def _save_hdf5(data, path, splits):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with h5py.File(path, 'w') as f:
        for key in ['obs', 'states', 'actions', 'next_obs', 'next_states',
                    'episode_ids', 'trajectory_types']:
            f.create_dataset(key, data=data[key], compression='gzip', compression_opts=4)
        f.create_dataset('action_cov', data=data['action_cov'])
        f.attrs['action_cov_condition_number'] = float(data['action_cov_condition_number'])
        if 'action_scale' in data:
            f.attrs['action_scale'] = float(data['action_scale'])
        if 'state_mean' in data:
            f.create_dataset('state_mean', data=data['state_mean'])
        if 'state_std' in data:
            f.create_dataset('state_std', data=data['state_std'])
        grp = f.create_group('splits')
        for split_name, idx in splits.items():
            grp.create_dataset(split_name, data=idx)


def load_dataset(path):
    data = {}
    with h5py.File(path, 'r') as f:
        for key in ['obs', 'states', 'actions', 'next_obs', 'next_states',
                    'action_cov', 'episode_ids', 'trajectory_types',
                    'state_mean', 'state_std']:
            if key in f:
                data[key] = f[key][:]
        data['action_cov_condition_number'] = float(
            f.attrs.get('action_cov_condition_number', 1.0))
        data['action_scale'] = float(f.attrs.get('action_scale', 1.0))
        splits = {}
        if 'splits' in f:
            for split_name in f['splits']:
                splits[split_name] = f['splits'][split_name][:]
        data['splits'] = splits
    if 'episode_ids' not in data:
        N = len(data['obs'])
        data['episode_ids'] = np.arange(N, dtype=np.int32)
    return data


def load_discrete_dataset_meta(dataset_dir: str) -> dict:
    """Load lightweight metadata from a discrete CartPole dataset directory.

    Reads only states/actions/episode_ids into RAM (fast).  Observations are
    NOT loaded — use DiscreteHDF5TrajectoryDataset for lazy per-batch loading.

    Returns a dict with:
        state_mean / state_std  — computed from train split for normalisation
        dataset_dir             — path string, passed to DiscreteHDF5TrajectoryDataset
        splits                  — {split: n_transitions}  (counts, not index arrays)
        action_scale            — max |action| inferred from dataset metadata
    """
    import json
    dataset_dir = Path(dataset_dir)
    train_states = []
    train_actions = []

    # Read only states and actions from train split to compute normalisation stats
    train_path = dataset_dir / 'train.hdf5'
    if train_path.exists():
        with h5py.File(train_path, 'r') as f:
            for ep_key in sorted(f['episodes'].keys(), key=int):
                s = f['episodes'][ep_key]['states'][:]   # (T+1, 4)
                a = f['episodes'][ep_key]['actions'][:]  # (T,)
                train_states.append(s[:-1].astype(np.float32))
                train_actions.append(a.astype(np.float32))

    if train_states:
        all_s = np.concatenate(train_states, axis=0)
        # Wrap theta (col 2) to [-pi, pi] so passive spinning episodes
        # don't inflate the std and corrupt normalization.
        all_s[:, 2] = np.arctan2(np.sin(all_s[:, 2]), np.cos(all_s[:, 2]))
        state_mean = all_s.mean(axis=0)
        state_std  = all_s.std(axis=0).clip(min=1e-6)
    else:
        state_mean = np.zeros(4, dtype=np.float32)
        state_std  = np.ones(4, dtype=np.float32)

    # Infer action_scale: prefer metadata.json, fall back to max |action| in data
    action_scale = 1.0
    meta_path = dataset_dir / 'metadata.json'
    if meta_path.exists():
        with open(meta_path) as _f:
            _meta = json.load(_f)
        # force_mag is the environment's max force (= action range limit)
        action_scale = float(_meta.get('environment', {}).get('force_mag', 1.0))
    if action_scale == 1.0 and train_actions:
        all_a = np.concatenate(train_actions)
        inferred = float(np.abs(all_a).max())
        if inferred > 1.5:   # binary {0,1} datasets → keep scale=1
            action_scale = inferred

    split_counts = {}
    for split in ('train', 'val', 'test'):
        p = dataset_dir / f'{split}.hdf5'
        if p.exists():
            with h5py.File(p, 'r') as f:
                split_counts[split] = int(f.attrs.get('n_transitions', 0))

    return {
        'dataset_dir': str(dataset_dir),
        'state_mean':  state_mean,
        'state_std':   state_std,
        'splits':      split_counts,
        'action_scale': action_scale,
    }


class DiscreteHDF5TrajectoryDataset(Dataset):
    """Trajectory dataset for the per-episode HDF5 format.

    With preload_obs=True (default): all observations are loaded into a single
    contiguous RAM array at init — __getitem__ is a pure numpy slice with zero
    I/O overhead.  Recommended for small/medium datasets (64x64: ~2.5 GB).

    With preload_obs=False: observations are read lazily per batch via HDF5
    handles opened per worker (multiprocessing safe), with optional on-the-fly
    resize via target_image_size.
    """

    def __init__(self, dataset_dir: str, split: str = 'train',
                 horizon: int = 20, frame_stack: int = 1,
                 state_mean: np.ndarray = None, state_std: np.ndarray = None,
                 action_scale: float = 1.0,
                 target_image_size: int = None, preload_obs: bool = True,
                 data_fraction: float = 1.0,
                 action_context_dim: int = 1):
        self.hdf5_path        = str(Path(dataset_dir) / f'{split}.hdf5')
        self.horizon          = horizon
        self.frame_stack      = frame_stack
        self.state_mean       = state_mean
        self.state_std        = state_std
        self.action_scale     = float(action_scale)
        self.target_image_size = target_image_size
        self.action_context_dim = int(action_context_dim)
        self._handles: dict   = {}   # pid → h5py.File (lazy, only used when not preloaded)

        all_states, all_nstates = [], []
        all_actions, all_ep_ids = [], []
        all_terminated, all_trajectory_types = [], []
        ep_keys_list, local_offsets_list = [], []
        obs_chunks       = [] if preload_obs else None
        ep_obs_starts    = [] if preload_obs else None
        _obs_cursor      = 0
        global_ep_id     = 0

        with h5py.File(self.hdf5_path, 'r') as f:
            ep_grp = f['episodes']
            all_ep_keys = sorted(ep_grp.keys(), key=int)
            n_keep = max(1, int(len(all_ep_keys) * data_fraction))
            for ep_key in all_ep_keys[:n_keep]:
                ep        = ep_grp[ep_key]
                acts_ep   = ep['actions'][:]    # (T,) or (T, frame_skip)
                states_ep = ep['states'][:]     # (T+1, 4)
                T = len(acts_ep)
                # Normalise to 2-D: (T, action_dim)
                if acts_ep.ndim == 1:
                    acts_ep = acts_ep[:, np.newaxis]
                all_states.append(states_ep[:-1].astype(np.float32))
                all_nstates.append(states_ep[1:].astype(np.float32))
                all_actions.append(acts_ep.astype(np.float32))
                all_ep_ids.append(np.full(T, global_ep_id, dtype=np.int32))
                trajectory_type = ep.attrs.get('trajectory_type', 'unknown')
                if isinstance(trajectory_type, bytes):
                    trajectory_type = trajectory_type.decode()
                all_trajectory_types.extend([str(trajectory_type)] * T)
                ep_keys_list.extend([ep_key] * T)
                local_offsets_list.extend(range(T))

                # Load terminated flags (present in no_done datasets to mark
                # soft-reset boundaries; absent in standard datasets → all False).
                if 'terminated' in ep:
                    all_terminated.append(ep['terminated'][:T].astype(bool))
                else:
                    all_terminated.append(np.zeros(T, dtype=bool))

                if preload_obs:
                    obs_ep = ep['observations'][:]          # (T+1, h, w, C) uint8
                    if target_image_size is not None:
                        s = target_image_size
                        t = torch.from_numpy(obs_ep).permute(0, 3, 1, 2).float()
                        t = torch.nn.functional.interpolate(
                            t, size=(s, s), mode='bilinear', align_corners=False)
                        obs_ep = t.permute(0, 2, 3, 1).to(torch.uint8).numpy()
                    ep_obs_starts.append(_obs_cursor)
                    _obs_cursor += len(obs_ep)              # T+1
                    obs_chunks.append(obs_ep)

                global_ep_id += 1

        if not all_states:
            self.states = self.next_states = np.zeros((0, 4), dtype=np.float32)
            self.actions    = np.zeros((0, 1), dtype=np.float32)
            self.ep_ids     = np.zeros(0, dtype=np.int32)
            self.terminated = np.zeros(0, dtype=bool)
            self.trajectory_types = np.empty(0, dtype=object)
        else:
            self.states      = np.concatenate(all_states,      axis=0)
            self.next_states = np.concatenate(all_nstates,     axis=0)
            self.actions     = np.concatenate(all_actions,     axis=0)  # (N, action_dim)
            self.ep_ids      = np.concatenate(all_ep_ids,      axis=0)
            self.terminated  = np.concatenate(all_terminated,  axis=0)  # (N,) bool
            self.trajectory_types = np.asarray(all_trajectory_types, dtype=object)
        self.ep_keys     = np.array(ep_keys_list)
        self.local_offs  = np.array(local_offsets_list, dtype=np.int32)
        self.valid_starts = self._find_valid_starts()

        if preload_obs and obs_chunks:
            self._obs_flat     = np.concatenate(obs_chunks, axis=0)  # (N_frames, h, w, C)
            self._obs_ep_start = np.array(ep_obs_starts, dtype=np.int64)
            mb = self._obs_flat.nbytes / 1e6
            print(f'[dataset:{split}] preloaded {_obs_cursor} frames ({mb:.0f} MB)')
        else:
            self._obs_flat     = None
            self._obs_ep_start = None

    def _find_valid_starts(self):
        H, ep = self.horizon, self.ep_ids
        N = len(ep) - H
        if N <= 0:
            return np.array([], dtype=np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(ep[:N + H], H)  # (N, H)
        mask = np.all(windows == windows[:, :1], axis=1)
        # Also exclude windows that contain a soft-reset boundary (terminated=True).
        if hasattr(self, 'terminated') and self.terminated.any():
            term_win = np.lib.stride_tricks.sliding_window_view(
                self.terminated[:N + H], H)                              # (N, H)
            mask &= ~term_win.any(axis=1)
        return np.where(mask)[0].astype(np.int64)

    def _file(self) -> 'h5py.File':
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = h5py.File(self.hdf5_path, 'r')
        return self._handles[pid]

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = int(self.valid_starts[idx])
        H, FS = self.horizon, self.frame_stack

        # All H transitions share the same episode (guaranteed by valid_starts)
        ep_key      = self.ep_keys[start]
        local_start = int(self.local_offs[start])

        if self._obs_flat is not None:
            # Fast path: pure numpy slice from preloaded RAM array
            ep_offset = int(self._obs_ep_start[int(self.ep_ids[start])])
            base = ep_offset + local_start
            obs_np = self._obs_flat[base:base + H + 1]
            prev0_np = self._obs_flat[base - 1] if local_start > 0 else obs_np[0]
        else:
            # Lazy path: one HDF5 read per sample (slower, used when preload_obs=False)
            obs_np = self._file()['episodes'][ep_key]['observations'][
                local_start: local_start + H + 1]  # (H+1, h, w, C) uint8
            prev0_np = (self._file()['episodes'][ep_key]['observations'][local_start - 1]
                        if local_start > 0 else obs_np[0])
            if self.target_image_size is not None:
                s = self.target_image_size
                # Resize the actual preceding frame together with the window so
                # frame stacking remains shape-consistent for local_start > 0.
                resize_np = np.concatenate([prev0_np[None], obs_np], axis=0)
                t = torch.from_numpy(resize_np).permute(0, 3, 1, 2).float()
                t = torch.nn.functional.interpolate(
                    t, size=(s, s), mode='bilinear', align_corners=False)
                resize_np = t.permute(0, 2, 3, 1).to(torch.uint8).numpy()
                prev0_np, obs_np = resize_np[0], resize_np[1:]

        def _t(arr):
            return torch.from_numpy(arr).permute(2, 0, 1)  # uint8 (C, h, w)

        frames = []
        for k in range(H + 1):
            if FS <= 1:
                frames.append(_t(obs_np[k]))
            else:
                # Build a stack of FS consecutive frames ending at obs_np[k].
                # Indices relative to current position: k-FS+1 ... k (inclusive).
                stack_frames = []
                for j in range(k - FS + 1, k + 1):
                    if j < 0:
                        stack_frames.append(_t(prev0_np))
                    elif j == 0:
                        stack_frames.append(_t(obs_np[0]))
                    else:
                        stack_frames.append(_t(obs_np[j]))
                frames.append(torch.cat(stack_frames, dim=0))
        obs_seq = torch.stack(frames)   # (H+1, 3*FS, h, w) uint8

        actions = torch.from_numpy(self.actions[start: start + H])   # (H, 1)
        if self.action_scale != 1.0:
            actions = actions / self.action_scale

        states = torch.from_numpy(
            np.concatenate([self.states[start: start + H],
                            self.next_states[start + H - 1: start + H]], axis=0))  # (H+1, 4)
        # Wrap theta to [-pi, pi] before normalisation (passive episodes accumulate
        # theta past 2*pi; wrapping eliminates visual-ambiguity in state supervision).
        states[:, 2] = torch.atan2(states[:, 2].sin(), states[:, 2].cos())
        if self.state_mean is not None and self.state_std is not None:
            states = ((states - torch.from_numpy(self.state_mean))
                      / torch.from_numpy(self.state_std))

        result = {
            'obs_seq': obs_seq,
            # Preserve the real frame immediately preceding obs_seq[0].
            # The temporal-difference encoder needs this at arbitrary sampled
            # window starts; duplicating obs_seq[0] would erase its motion cue.
            'prev_obs': _t(prev0_np),
            'actions': actions,
            'states': states,
            'trajectory_type': str(self.trajectory_types[start]),
        }
        ctx = self.action_context_dim
        if ctx > 1:
            # Provide the ctx-1 actions before this window (same episode, zero at boundary).
            local_start = int(self.local_offs[start])
            look_back = min(ctx - 1, local_start)
            if look_back > 0:
                prev_acts = self.actions[start - look_back: start].astype(np.float32)
            else:
                prev_acts = np.zeros((0, self.actions.shape[1]), dtype=np.float32)
            if look_back < ctx - 1:
                pad = np.zeros((ctx - 1 - look_back, self.actions.shape[1]), dtype=np.float32)
                prev_acts = np.concatenate([pad, prev_acts], axis=0)
            prev_actions = torch.from_numpy(prev_acts)  # (ctx-1, action_dim)
            if self.action_scale != 1.0:
                prev_actions = prev_actions / self.action_scale
            result['prev_actions'] = prev_actions
        return result

    def __del__(self):
        for fh in self._handles.values():
            try:
                fh.close()
            except Exception:
                pass


def _estimate_obs_bytes(hdf5_path: str, target_image_size: int | None,
                        data_fraction: float) -> int:
    """Estimate total observation bytes without loading any pixel data.

    Samples up to 100 episodes to estimate average episode length, then
    scales to the full dataset — fast even for 10k-episode files.
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            ep_grp = f['episodes']
            keys = sorted(ep_grp.keys(), key=int)
            n_keep = max(1, int(len(keys) * data_fraction))
            # Image shape from first episode (shape read = metadata only, no pixel I/O)
            _, h, w, c = ep_grp[keys[0]]['observations'].shape
            if target_image_size is not None:
                h = w = target_image_size
            # Sample up to 100 episodes to estimate average episode length
            sample = keys[:min(100, n_keep)]
            avg_frames = sum(len(ep_grp[k]['actions']) + 1 for k in sample) / len(sample)
        return int(avg_frames * n_keep * h * w * c)
    except Exception:
        return 0


def make_discrete_dataloaders(dataset_dir: str, batch_size: int = 256,
                              num_workers: int = 0, horizon: int = 1,
                              frame_stack: int = 1,
                              state_mean: np.ndarray = None,
                              state_std: np.ndarray = None,
                              action_scale: float = 1.0,
                              target_image_size: int = None,
                              preload_obs: bool = True,
                              data_fraction: float = 1.0,
                              balanced_sampling: bool = False,
                              angle_bin_edges=(0.05, 0.2, 0.6),
                              trajectory_type_sampling_weights=None,
                              sampling_audit: bool = False,
                              local_sampling_fraction: float = 0.0,
                              local_sampling_region=(0.10, 0.25, 0.05, 0.50),
                              action_context_dim: int = 1) -> dict:
    """Build DataLoaders from a discrete CartPole HDF5 dataset directory."""
    _PRELOAD_LIMIT_BYTES = 8 * 1024 ** 3  # 8 GB
    loaders = {}
    for split in ('train', 'val', 'test'):
        path = Path(dataset_dir) / f'{split}.hdf5'
        if not path.exists():
            continue
        # Auto-disable obs preloading when dataset would exceed RAM budget
        _preload = preload_obs
        if _preload:
            _est = _estimate_obs_bytes(str(path), target_image_size, data_fraction)
            if _est > _PRELOAD_LIMIT_BYTES:
                print(f'[dataset:{split}] obs too large to preload '
                      f'({_est / 1e9:.1f} GB > 2 GB) — using lazy loading')
                _preload = False
        ds = DiscreteHDF5TrajectoryDataset(
            dataset_dir, split=split, horizon=horizon,
            frame_stack=frame_stack, state_mean=state_mean, state_std=state_std,
            action_scale=action_scale,
            target_image_size=target_image_size, preload_obs=_preload,
            data_fraction=data_fraction,
            action_context_dim=action_context_dim)
        if len(ds) == 0:
            continue
        sampler = None
        if split == 'train' and (balanced_sampling or local_sampling_fraction > 0):
            starts = ds.valid_starts
            weights = np.ones(len(starts), dtype=np.float64)
            if balanced_sampling:
                theta = ds.states[starts, 2]
                magnitude_bin = np.digitize(np.abs(theta), angle_bin_edges)
                sign_bin = (theta >= 0).astype(np.int64)
                types = ds.trajectory_types[starts]
                groups = [(str(typ), int(sign), int(mag))
                          for typ, sign, mag in zip(types, sign_bin, magnitude_bin)]
                group_counts = {}
                bins_per_type = {}
                for group in groups:
                    group_counts[group] = group_counts.get(group, 0) + 1
                    bins_per_type.setdefault(group[0], set()).add(group[1:])
                weights = np.array([
                    1.0 / (len(bins_per_type[group[0]]) * group_counts[group])
                    for group in groups
                ], dtype=np.float64)
                summary = ', '.join(
                    f'{typ}:{sum(str(g[0]) == str(typ) for g in groups)}'
                    for typ in sorted(set(types.tolist())))
                print(f'[dataset:train] balanced sampler  {summary}')

            if trajectory_type_sampling_weights:
                types = ds.trajectory_types[starts]
                requested = {
                    str(key): max(0.0, float(value))
                    for key, value in trajectory_type_sampling_weights.items()
                }
                missing = sorted(set(requested) - set(map(str, np.unique(types))))
                if missing:
                    warnings.warn(f'Trajectory sampling weights contain absent types: {missing}')
                multipliers = np.array(
                    [requested.get(str(typ), 0.0) for typ in types], dtype=np.float64)
                if not np.any(multipliers > 0):
                    raise ValueError('trajectory_type_sampling_weights assigns zero mass to all samples')
                weights *= multipliers

            if local_sampling_fraction > 0:
                frac = float(np.clip(local_sampling_fraction, 0.0, 0.99))
                region = np.asarray(local_sampling_region, dtype=np.float64)
                local = (np.abs(ds.states[starts]) <= region[None]).all(axis=1)
                local_mass = weights[local].sum()
                other_mass = weights[~local].sum()
                if local_mass > 0 and other_mass > 0:
                    boost = frac * other_mass / ((1.0 - frac) * local_mass)
                    weights[local] *= boost
                    expected = weights[local].sum() / weights.sum()
                    print(f'[dataset:train] local sampler  raw={local.mean():.1%}  '
                          f'target={frac:.1%}  expected={expected:.1%}  '
                          f'region={region.tolist()}')
                else:
                    warnings.warn('Local sampling requested but local/nonlocal split is empty')
            if sampling_audit:
                total = weights.sum()
                probs = weights / total
                types = ds.trajectory_types[starts]
                states = ds.states[starts]
                actions = np.asarray(ds.actions[starts]).reshape(len(starts), -1)[:, 0]
                region = np.asarray(local_sampling_region, dtype=np.float64)
                local = (np.abs(states) <= region[None]).all(axis=1)
                print('[sampler audit] type       raw     mass    local   |u|<0.05  mean|u|')
                for typ in sorted(map(str, np.unique(types))):
                    mask = np.asarray([str(value) == typ for value in types])
                    type_mass = probs[mask].sum()
                    conditional = probs[mask] / max(type_mass, 1e-12)
                    print(f'[sampler audit] {typ:<10} {mask.mean():6.1%}  '
                          f'{type_mass:6.1%}  '
                          f'{conditional[local[mask]].sum():6.1%}  '
                          f'{conditional[np.abs(actions[mask]) < 0.05].sum():8.1%}  '
                          f'{np.sum(conditional * np.abs(actions[mask])):7.3f}')
                action_edges = (0.05, 0.5, 1.5, 3.5)
                action_bins = np.digitize(np.abs(actions), action_edges)
                labels = ('<0.05', '0.05-0.5', '0.5-1.5', '1.5-3.5', '>=3.5')
                action_summary = ', '.join(
                    f'{label}:{probs[action_bins == idx].sum():.1%}'
                    for idx, label in enumerate(labels))
                ess = total ** 2 / np.square(weights).sum()
                print(f'[sampler audit] local={probs[local].sum():.1%}  '
                      f'ESS={ess:.0f}/{len(weights)}  action_mass=({action_summary})')
            sampler = WeightedRandomSampler(
                torch.from_numpy(weights), num_samples=len(ds), replacement=True)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == 'train' and sampler is None), sampler=sampler,
            num_workers=num_workers, pin_memory=(num_workers > 0),
            drop_last=(split == 'train'),
            persistent_workers=(num_workers > 0),
            prefetch_factor=(4 if num_workers > 0 else None),
        )
    return loaders


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
                 obs_eq: np.ndarray = None, n_eq_selfloop: int = 0,
                 action_scale: float = 1.0, state_mean: np.ndarray = None,
                 state_std: np.ndarray = None):
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
        self.action_scale = action_scale
        self.state_mean = state_mean
        self.state_std  = state_std
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

        # Build obs_seq with proper FS-frame stacking.
        # raw[k] = obs at step start+k for k in 0..H-1; raw[H] = next_obs at step start+H-1.
        def _raw(k):
            if k < H:
                return _to_tensor(self.obs[start + k])
            return _to_tensor(self.next_obs[start + H - 1])

        frames = []
        for k in range(H + 1):
            if FS <= 1:
                frames.append(_raw(k))
            else:
                stack_frames = []
                for j in range(k - FS + 1, k + 1):
                    stack_frames.append(_raw(max(j, 0)))
                frames.append(torch.cat(stack_frames, dim=0))
        obs_seq = torch.stack(frames)                                  # (H+1, 3*FS, h, w)

        actions = torch.from_numpy(self.actions[start:start + H])     # (H, 1)
        if self.action_scale != 1.0:
            actions = actions / self.action_scale

        states_list = [self.states[start + k] for k in range(H)]
        states_list.append(self.next_states[start + H - 1])
        states = torch.from_numpy(np.stack(states_list))               # (H+1, 4)
        if self.state_mean is not None and self.state_std is not None:
            states = (states - torch.from_numpy(self.state_mean)) / torch.from_numpy(self.state_std)

        return {'obs_seq': obs_seq, 'actions': actions, 'states': states}


def make_dataloaders(data, batch_size=256, num_workers=0, horizon=1, frame_stack=1,
                     obs_eq: np.ndarray = None, n_eq_selfloop: int = 0,
                     action_scale: float = 1.0,
                     state_mean: np.ndarray = None, state_std: np.ndarray = None):
    """Return dataloaders. horizon=1 -> TransitionDataset; horizon>1 -> TrajectoryDataset."""
    loaders = {}
    for split in ('train', 'val', 'test'):
        if split not in data.get('splits', {}):
            continue
        if horizon > 1:
            ds = TrajectoryDataset(data, split=split, horizon=horizon,
                                   frame_stack=frame_stack,
                                   obs_eq=(obs_eq if split == 'train' else None),
                                   n_eq_selfloop=(n_eq_selfloop if split == 'train' else 0),
                                   action_scale=action_scale,
                                   state_mean=state_mean, state_std=state_std)
        else:
            ds = TransitionDataset(data, split=split)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == 'train'),
            num_workers=num_workers, pin_memory=True,
            drop_last=(split == 'train'),
        )
    return loaders
