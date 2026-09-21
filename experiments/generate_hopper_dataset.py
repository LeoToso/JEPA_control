#!/usr/bin/env python
"""Collect visual Hopper-v4 rollouts and save as HDF5 dataset.

Produces train/val/test.hdf5 in the same per-episode format used for
PointMaze and CartPole datasets.

Policy options
--------------
random  (default) — uniform random actions in [-1, 1]
forward           — random actions with a small forward-hop bias
                    (adds positive torque to hip and knee actuators)

With --no-done (recommended):
  When the hopper falls (terminated), the env is soft-reset and collection
  continues until episode_steps is reached.  Each episode in the HDF5 may
  span multiple physical resets; the terminated[] flag marks the boundaries.

Usage
-----
  MUJOCO_GL=egl python experiments/generate_hopper_dataset.py \\
      --output-dir data/hopper_rnd_64 \\
      --n-episodes 2000 --episode-steps 200 \\
      --image-size 64 --seed 0 --no-done

  # With forward-biased policy:
  MUJOCO_GL=egl python experiments/generate_hopper_dataset.py \\
      --output-dir data/hopper_fwd_64 \\
      --n-episodes 2000 --episode-steps 200 \\
      --image-size 64 --seed 0 --no-done --policy forward
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from envs.hopper_visual import HopperVisual

# Hopper actuators: hip (0), knee (1), ankle (2).
# Positive hip + knee torque biases toward hopping forward.
_FORWARD_BIAS_IDX = [0, 1]
_FORWARD_BIAS_MAG = 0.4


def _sample_action(rng: np.random.Generator, policy: str) -> np.ndarray:
    a = rng.uniform(-1.0, 1.0, size=HopperVisual.ACTION_DIM).astype(np.float32)
    if policy == 'forward':
        a[_FORWARD_BIAS_IDX] = np.clip(
            a[_FORWARD_BIAS_IDX] + _FORWARD_BIAS_MAG, -1.0, 1.0)
    return a


def collect_episode(env: HopperVisual, n_steps: int,
                    rng: np.random.Generator,
                    frame_skip: int = 1,
                    no_done: bool = False,
                    policy: str = 'random') -> dict:
    """Collect one episode (up to n_steps macro-steps).

    frame_skip > 1 executes frame_skip gymnasium steps with the same action
    and only renders the final frame (temporal abstraction).

    no_done=True: on termination, soft-reset and continue until n_steps is
    reached.  terminated[] marks physics boundaries.
    """
    obs, state, _ = env.reset()
    observations: list = [obs]
    states:       list = [state]
    actions:      list = []
    rewards:      list = []
    terminated_l: list = []
    truncated_l:  list = []

    need_reset = False

    for _ in range(n_steps):
        if need_reset:
            obs, state, _ = env.reset()
            observations[-1] = obs   # replace last obs with fresh reset obs
            states[-1]       = state
            need_reset       = False

        action = _sample_action(rng, policy)

        # Frame-skip: execute (frame_skip-1) physics steps without rendering,
        # then render on the final step.  Break early if the env terminates.
        done = False
        reward = 0.0
        tl = False
        early_term = False
        for _ in range(frame_skip - 1):
            new_state, r, done, info = env.step_no_render(action)
            reward += r
            if done:
                tl = bool(info.get('TimeLimit.truncated', False))
                early_term = True
                break
        if early_term:
            new_obs = env._render()
        else:
            new_obs, new_state, r, done, info = env.step(action)
            reward += r
            tl = bool(info.get('TimeLimit.truncated', False))

        terminated = bool(done and not tl)

        observations.append(new_obs)
        states.append(new_state)
        actions.append(action)
        rewards.append(float(reward))
        terminated_l.append(terminated)
        truncated_l.append(tl)

        obs   = new_obs
        state = new_state

        if done:
            if no_done:
                need_reset = True
            else:
                break

    return {
        'observations': np.array(observations, dtype=np.uint8),   # (T+1, h, w, 3)
        'states':       np.array(states,       dtype=np.float32), # (T+1, 11)
        'actions':      np.array(actions,      dtype=np.float32), # (T, 3)
        'rewards':      np.array(rewards,      dtype=np.float32), # (T,)
        'terminated':   np.array(terminated_l, dtype=bool),       # (T,)
        'truncated':    np.array(truncated_l,  dtype=bool),       # (T,)
    }


def save_split(episodes: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        grp = f.create_group('episodes')
        for i, ep in enumerate(episodes):
            g = grp.create_group(str(i))
            for key, val in ep.items():
                g.create_dataset(key, data=val,
                                 compression='gzip', compression_opts=4)
            g.attrs['trajectory_type'] = 'random'
            g.attrs['length']          = len(ep['actions'])
        f.attrs['n_episodes']    = len(episodes)
        f.attrs['n_transitions'] = sum(len(ep['actions']) for ep in episodes)
    n_tr = sum(len(ep['actions']) for ep in episodes)
    print(f'  {path.name}: {len(episodes)} episodes, {n_tr} transitions')


def main():
    p = argparse.ArgumentParser(description='Generate visual Hopper-v4 dataset.')
    p.add_argument('--output-dir',    required=True)
    p.add_argument('--n-episodes',    type=int,   default=2000)
    p.add_argument('--episode-steps', type=int,   default=200)
    p.add_argument('--image-size',    type=int,   default=64)
    p.add_argument('--frame-skip',    type=int,   default=1,
                   help='Gymnasium steps per stored macro-step (default 1).')
    p.add_argument('--no-done',       action='store_true', default=False,
                   help='Soft-reset on termination and keep collecting. '
                        'Recommended: random policy causes frequent early falls.')
    p.add_argument('--policy',        default='random',
                   choices=['random', 'forward'],
                   help='random: uniform [-1,1];  '
                        'forward: bias hip+knee actuators for longer episodes.')
    p.add_argument('--train-frac',    type=float, default=0.80)
    p.add_argument('--val-frac',      type=float, default=0.10)
    p.add_argument('--seed',          type=int,   default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    env = HopperVisual(image_size=args.image_size, seed=args.seed)

    print(f'Collecting {args.n_episodes} episodes × {args.episode_steps} steps …')
    print(f'  env=Hopper-v4  image={args.image_size}px  '
          f'frame_skip={args.frame_skip}  policy={args.policy}  '
          f'no_done={args.no_done}')

    episodes = []
    for i in range(args.n_episodes):
        ep = collect_episode(env, args.episode_steps, rng,
                             frame_skip=args.frame_skip,
                             no_done=args.no_done,
                             policy=args.policy)
        episodes.append(ep)
        if (i + 1) % 200 == 0:
            avg_len = np.mean([len(e['actions']) for e in episodes])
            print(f'  {i + 1}/{args.n_episodes}  avg_ep_len={avg_len:.0f}')
    env.close()

    # Shuffle before splitting so train/val/test are i.i.d.
    order = rng.permutation(len(episodes))
    episodes = [episodes[i] for i in order]

    n_tr  = int(args.n_episodes * args.train_frac)
    n_val = int(args.n_episodes * args.val_frac)
    splits = {
        'train': episodes[:n_tr],
        'val':   episodes[n_tr:n_tr + n_val],
        'test':  episodes[n_tr + n_val:],
    }

    out = Path(args.output_dir)
    for split, eps in splits.items():
        if eps:
            save_split(eps, out / f'{split}.hdf5')

    meta = {
        'environment': {
            'name':       'Hopper-v4',
            'image_size': args.image_size,
            'frame_skip': args.frame_skip,
            'no_done':    args.no_done,
            'policy':     args.policy,
        },
        'n_episodes':    args.n_episodes,
        'episode_steps': args.episode_steps,
        'seed':          args.seed,
        'action_dim':    HopperVisual.ACTION_DIM,
        'state_dim':     HopperVisual.STATE_DIM,
    }
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    total = sum(len(ep['actions']) for ep in episodes)
    print(f'[done] → {out}')
    print(f'  Stored action shape: (T, {HopperVisual.ACTION_DIM})')
    print(f'  Stored state  shape: (T+1, {HopperVisual.STATE_DIM})')
    print(f'  Total transitions: {total}')


if __name__ == '__main__':
    main()
