#!/usr/bin/env python
"""Collect visual PointMaze rollouts and save as HDF5 dataset.

Produces train/val/test.hdf5 in the same per-episode format used by
DiscreteHDF5TrajectoryDataset (same structure as the CartPole datasets).

With --frame-skip 5 --multi-action:
  Each macro-step does 5 gymnasium steps, sampling 5 independent 2-D
  sub-actions (DINO-WM style).  Actions stored as (T, frame_skip*2) float32.
  Use action_context_dim = frame_skip*2 = 10 and use_action_history=false
  in the model config.

Without --multi-action (default):
  Each macro-step does frame_skip gymnasium steps with the SAME random 2-D
  action.  Actions stored as (T, 2) float32.

Usage:
  # fs=1 baseline (current behaviour)
  python experiments/generate_pointmaze_dataset.py \\
      --output-dir data/pointmaze_u_fs1_64 \\
      --n-episodes 2000 --episode-steps 100 \\
      --image-size 64 --maze-map U --seed 0

  # fs=5 multi-action (DINO-WM / CartPole-branch style)
  python experiments/generate_pointmaze_dataset.py \\
      --output-dir data/pointmaze_u_fs5_ma_64 \\
      --n-episodes 2000 --episode-steps 100 \\
      --image-size 64 --maze-map U --seed 0 \\
      --frame-skip 5 --multi-action
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np

from envs.pointmaze_visual import PointMazeVisual


def collect_episode(env: PointMazeVisual, n_steps: int,
                    rng: np.random.Generator,
                    frame_skip: int = 1,
                    multi_action: bool = False) -> dict:
    """Collect one episode.

    frame_skip > 1: each macro-step executes frame_skip gymnasium steps.
    multi_action:   each macro-step samples frame_skip independent 2-D
                    sub-actions (vs. repeating one action for all sub-steps).
                    Requires frame_skip > 1.  Actions stored as
                    (T, frame_skip*2) when True, (T, 2) otherwise.
    """
    obs, state, _ = env.reset()
    observations  = [obs]
    states        = [state]
    actions: list = []
    rewards: list = []
    terminated_l: list = []
    truncated_l: list  = []

    for _ in range(n_steps):
        if multi_action and frame_skip > 1:
            sub_actions = rng.uniform(-1.0, 1.0,
                                      size=(frame_skip, 2)).astype(np.float32)
            stored_action = sub_actions.flatten()  # (frame_skip*2,)
            # Execute first (frame_skip-1) sub-steps without rendering.
            done = False
            for s in range(frame_skip - 1):
                new_state, _, done, _ = env.step_no_render(sub_actions[s])
                if done:
                    break
            if done:
                # Episode ended mid-frame; use last rendered obs as new_obs.
                new_obs = obs.copy()
                new_state = new_state
                tl = False
                reward = 0.0
            else:
                new_obs, new_state, reward, done, info = env.step(sub_actions[-1])
                tl = bool(info.get('TimeLimit.truncated', False))
        elif frame_skip > 1:
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            stored_action = action  # (2,)
            done = False
            for s in range(frame_skip - 1):
                new_state, _, done, _ = env.step_no_render(action)
                if done:
                    break
            if done:
                new_obs = obs.copy()
                new_state = new_state
                tl = False
                reward = 0.0
            else:
                new_obs, new_state, reward, done, info = env.step(action)
                tl = bool(info.get('TimeLimit.truncated', False))
        else:
            action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            stored_action = action  # (2,)
            new_obs, new_state, reward, done, info = env.step(action)
            tl = bool(info.get('TimeLimit.truncated', False))

        observations.append(new_obs)
        states.append(new_state)
        actions.append(stored_action)
        rewards.append(float(reward))
        terminated_l.append(bool(done and not tl))
        truncated_l.append(tl)
        obs = new_obs
        state = new_state
        if done:
            break

    return {
        'observations': np.array(observations, dtype=np.uint8),    # (T+1, h, w, 3)
        'states':       np.array(states,       dtype=np.float32),  # (T+1, 4)
        'actions':      np.array(actions,      dtype=np.float32),  # (T, 2) or (T, fs*2)
        'rewards':      np.array(rewards,      dtype=np.float32),  # (T,)
        'terminated':   np.array(terminated_l, dtype=bool),        # (T,)
        'truncated':    np.array(truncated_l,  dtype=bool),        # (T,)
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
    p = argparse.ArgumentParser(
        description='Generate visual PointMaze dataset.')
    p.add_argument('--output-dir',    required=True)
    p.add_argument('--n-episodes',    type=int,   default=2000)
    p.add_argument('--episode-steps', type=int,   default=100)
    p.add_argument('--image-size',    type=int,   default=64)
    p.add_argument('--maze-map',      default='U',
                   help='PointMaze variant: U, Medium, Large, …')
    p.add_argument('--frame-skip',    type=int,   default=1,
                   help='Gymnasium steps per macro-step (default 1 = current behaviour). '
                        'Set 5 to match DINO-WM dynamics depth.')
    p.add_argument('--multi-action',  action='store_true', default=False,
                   help='Store frame_skip independent 2-D sub-actions per macro-step '
                        '(DINO-WM / CartPole-branch style). Requires --frame-skip > 1. '
                        'Actions shape: (T, frame_skip*2) instead of (T, 2). '
                        'Use action_context_dim=frame_skip*2 and '
                        'use_action_history=false in the model config.')
    p.add_argument('--train-frac',    type=float, default=0.80)
    p.add_argument('--val-frac',      type=float, default=0.10)
    p.add_argument('--seed',          type=int,   default=0)
    args = p.parse_args()

    if args.multi_action and args.frame_skip <= 1:
        p.error('--multi-action requires --frame-skip > 1')

    rng = np.random.default_rng(args.seed)

    env_cfg = {'environment': {
        'maze_map':    args.maze_map,
        'image_size':  args.image_size,
        'action_scale': 1.0,
    }}
    env = PointMazeVisual(env_cfg, seed=args.seed)

    action_dim = args.frame_skip * 2 if args.multi_action else 2
    print(f'Collecting {args.n_episodes} episodes × {args.episode_steps} steps …')
    print(f'  frame_skip={args.frame_skip}  multi_action={args.multi_action}'
          f'  stored_action_dim={action_dim}')
    episodes = []
    for i in range(args.n_episodes):
        ep = collect_episode(env, args.episode_steps, rng,
                             frame_skip=args.frame_skip,
                             multi_action=args.multi_action)
        episodes.append(ep)
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{args.n_episodes}')
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
            'maze_map':     args.maze_map,
            'image_size':   args.image_size,
            'action_scale': 1.0,
            'force_mag':    1.0,  # read by load_discrete_dataset_meta
            'frame_skip':   args.frame_skip,
            'multi_action': args.multi_action,
        },
        'n_episodes':    args.n_episodes,
        'episode_steps': args.episode_steps,
        'seed':          args.seed,
        'action_dim':    action_dim,
    }
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'[done] → {out}')
    if args.multi_action:
        print(f'  Stored action shape: (T, {action_dim})  '
              f'[frame_skip={args.frame_skip} × 2D sub-actions]')
        print(f'  Model config: action_context_dim={action_dim}  '
              f'use_action_history=false')


if __name__ == '__main__':
    main()
