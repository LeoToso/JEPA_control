"""Collect a cartpole dataset using stable-worldmodel's swm/CartPoleControl-v1 env.

Pilot for testing whether stable-worldmodel's rendering (pole occupies far more
of the frame than in our custom horizontal-view env) fixes the AE decoder's
background-dominance bottleneck (per-pixel MSE dominated by the static
cart/track, so the decoder learns a blurry average pole — see jepa_writeup).

Saves data in the exact HDF5 schema produced by data.dataset.generate_dataset
(obs/states/actions/next_obs/next_states/episode_ids/splits/action_cov), so the
existing AEWorldModel training/eval/visualization pipeline works unchanged.

swm/CartPoleControl-v1 has a Discrete(2) action space (push left / push right).
We map {0, 1} -> {-1.0, +1.0} so the existing action_dim=1 linear action
encoder and [-1, 1] action_range convention apply with zero model changes.

Usage:
    python data/collect_swm_cartpole.py \\
        --n-episodes 200 --episode-length 100 \\
        --image-size 64 --save-path data/cartpole_swm_pilot.h5
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings('ignore')

import numpy as np


ACTION_MAP = {0: -1.0, 1: 1.0}  # Discrete(2) -> continuous {-1, +1}


def _collect_episode(env, episode_length, rng):
    obs_list, state_list, action_list = [], [], []
    next_obs_list, next_state_list = [], []

    _, info = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
    obs = np.asarray(info['pixels'])
    state = np.asarray(info['state'], dtype=np.float32)

    for _ in range(episode_length):
        a_discrete = int(rng.randint(0, 2))
        u = ACTION_MAP[a_discrete]
        _, _, terminated, truncated, info = env.step(a_discrete)
        next_obs = np.asarray(info['pixels'])
        next_state = np.asarray(info['state'], dtype=np.float32)

        obs_list.append(obs.copy())
        state_list.append(state.copy())
        action_list.append(np.array([u], dtype=np.float32))
        next_obs_list.append(next_obs.copy())
        next_state_list.append(next_state.copy())

        obs, state = next_obs, next_state
        if terminated or truncated:
            # No mid-episode resets in our convention — but swm auto-resets
            # on terminal; re-fetch the post-reset frame so obs/next_obs stay
            # consistent (matches "ignore done, keep stepping" only loosely;
            # for a short pilot this is fine — mark episode boundary instead).
            break

    n = len(obs_list)
    return {
        'obs':         np.stack(obs_list).astype(np.uint8),
        'states':      np.stack(state_list).astype(np.float32),
        'actions':     np.stack(action_list).astype(np.float32),
        'next_obs':    np.stack(next_obs_list).astype(np.uint8),
        'next_states': np.stack(next_state_list).astype(np.float32),
    }, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-episodes', type=int, default=200)
    p.add_argument('--episode-length', type=int, default=100)
    p.add_argument('--image-size', type=int, default=64)
    p.add_argument('--train-frac', type=float, default=0.8)
    p.add_argument('--val-frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-path', default='data/cartpole_swm_pilot.h5')
    args = p.parse_args()

    import gymnasium as gym
    import stable_worldmodel as swm
    from data.dataset import _save_hdf5

    rng = np.random.RandomState(args.seed)
    env = gym.make('swm/CartPoleControl-v1', render_mode='rgb_array')
    env = swm.wrapper.AddPixelsWrapper(env, pixels_shape=(args.image_size, args.image_size))

    print(f'[data] Collecting {args.n_episodes} episodes x up to {args.episode_length} '
          f'steps from swm/CartPoleControl-v1 (Discrete(2) -> {ACTION_MAP}) ...')

    segs, ep_ids_l = [], []
    next_ep_id = 0
    for ep_idx in range(args.n_episodes):
        seg, n = _collect_episode(env, args.episode_length, rng)
        if n < 2:
            continue
        segs.append(seg)
        ep_ids_l.append(np.full(n, next_ep_id, dtype=np.int32))
        next_ep_id += 1
    env.close()

    if not segs:
        raise RuntimeError('No episodes collected — check env / episode length.')

    data = {key: np.concatenate([s[key] for s in segs], axis=0) for key in segs[0]}
    data['episode_ids'] = np.concatenate(ep_ids_l)

    N = len(data['obs'])
    ep_ids = data['episode_ids']
    unique_ep = np.unique(ep_ids)
    rng.shuffle(unique_ep)
    ep_order = {ep: i for i, ep in enumerate(unique_ep)}
    sort_idx = np.argsort([ep_order[e] for e in ep_ids], kind='stable')
    for key in data:
        data[key] = data[key][sort_idx]

    n_tr  = int(N * args.train_frac)
    n_val = int(N * args.val_frac)
    splits = {
        'train': np.arange(0, n_tr),
        'val':   np.arange(n_tr, n_tr + n_val),
        'test':  np.arange(n_tr + n_val, N),
    }
    actions = data['actions']
    action_cov = np.atleast_2d(np.cov(actions.T))
    ev = np.maximum(np.linalg.eigvalsh(action_cov), 1e-12)
    data['action_cov'] = action_cov
    data['action_cov_condition_number'] = float(ev.max() / ev.min())

    print(f'[data] Total: {N:,} transitions  (train={n_tr:,}, val={n_val:,}, '
          f'test={N - n_tr - n_val:,})  {next_ep_id} episodes  '
          f'avg_ep_len={N / next_ep_id:.1f}')

    _save_hdf5(data, args.save_path, splits)
    print(f'[data] saved -> {args.save_path}')


if __name__ == '__main__':
    main()
