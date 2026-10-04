#!/usr/bin/env python
"""Open-loop replay of a SAC episode from the HDF5 dataset.

Executes the stored actions directly in Walker2d-v4 without any planning.
This is the ground truth for "does SAC work on this model?"

Usage
-----
python experiments/replay_sac_episode.py \
    --hdf5-dir /path/to/walker2d_mixed_sac_fs5_64 \
    --episode 0 \
    --render-dir results/sac_replay_frames
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hdf5-dir',  required=True)
    p.add_argument('--episode',   type=int, default=0)
    p.add_argument('--n-episodes', type=int, default=5, help='Replay this many episodes')
    p.add_argument('--find-sac',  action='store_true',
                   help='Scan the dataset and pick the N episodes with highest stored x_vel '
                        '(i.e. SAC forward-walking episodes)')
    p.add_argument('--scan-first', type=int, default=500,
                   help='How many episodes to scan when --find-sac is set')
    p.add_argument('--frame-skip', type=int, default=5,
                   help='Must match the frame_skip used during data collection '
                        '(dataset name "fs5" → 5, Walker2d-v4 default is 4)')
    p.add_argument('--render-dir', default='')
    p.add_argument('--gif-fps',   type=int, default=30)
    args = p.parse_args()

    import gymnasium as gym
    train_hdf5 = Path(args.hdf5_dir) / 'train.hdf5'
    if not train_hdf5.exists():
        import glob
        files = sorted(glob.glob(str(Path(args.hdf5_dir) / '*.hdf5')))
        if not files:
            raise FileNotFoundError(f'No HDF5 files in {args.hdf5_dir}')
        train_hdf5 = files[0]

    hf = h5py.File(train_hdf5, 'r')
    ep_group = hf['episodes'] if 'episodes' in hf else hf
    eps = sorted(ep_group.keys(), key=lambda k: int(k))
    print(f'Dataset: {train_hdf5.name}  episodes={len(eps)}')

    from gymnasium.envs.mujoco import Walker2dEnv
    env = Walker2dEnv(frame_skip=args.frame_skip,
                      render_mode='rgb_array' if args.render_dir else None)

    if args.find_sac:
        # Scan stored states to rank episodes by mean x_vel (SAC episodes walk forward)
        print(f'Scanning first {args.scan_first} episodes for highest x_vel …')
        scores = []
        for k in eps[:args.scan_first]:
            if 'states' in ep_group[k]:
                xv = np.array(ep_group[k]['states'])[:, 8].mean()
                scores.append((float(xv), k))
        scores.sort(reverse=True)
        print(f'  Top 10 by stored x_vel: ' +
              ', '.join(f'{k}({v:+.2f})' for v, k in scores[:10]))
        selected_keys = [k for _, k in scores[:args.n_episodes]]
        ep_indices = [eps.index(k) for k in selected_keys]
    else:
        ep_indices = [(args.episode + i) % len(eps) for i in range(args.n_episodes)]

    for ep_offset, ep_idx in enumerate(ep_indices):
        ep_idx = (args.episode + ep_offset) % len(eps)
        ep_key = eps[ep_idx] if not args.find_sac else selected_keys[ep_offset]
        actions = np.array(ep_group[ep_key]['actions'])  # (T, 6)
        T = len(actions)

        # Reset to the exact initial state stored in the HDF5.
        # SAC actions are state-conditioned; replaying from a different
        # initial state produces garbage (backward walking, early falls).
        obs, info = env.reset(seed=0)   # initialise MuJoCo internals
        ep_data   = ep_group[ep_key]
        if 'states' in ep_data:
            # states[0] is the 17-D gym obs at t=0
            s0 = np.array(ep_data['states'][0], dtype=np.float64)
            # Walker2d-v4 qpos[0]=x, qpos[1:9]=joints (8), qvel[0:9]=velocities
            # gym obs = qpos[1:9] + qvel[0:9]  (obs[0]=z, obs[1]=tilt, obs[2:8]=joints)
            # full qpos: [x, z, tilt, joint0..5] = 9-D; qvel: 9-D
            qpos      = env.unwrapped.data.qpos.copy()
            qvel      = env.unwrapped.data.qvel.copy()
            qpos[1:]  = s0[:8]   # z, tilt, 6 joint angles
            qvel[:]   = s0[8:]   # 9 velocities
            env.unwrapped.set_state(qpos, qvel)
            print(f'  ep={ep_key:>4s}  reset to stored state  z={s0[0]:.3f}  xvel={s0[8]:.3f}', end='')
        else:
            print(f'  ep={ep_key:>4s}  [no states key, using random reset]', end='')

        x_vels = []
        frames = []
        terminated = False

        for t, a in enumerate(actions):
            obs, reward, term, trunc, info = env.step(a)
            x_vels.append(float(info.get('x_velocity', 0.0)))
            if args.render_dir:
                frames.append(env.render())
            terminated = term or trunc
            if terminated:
                break

        avg_vel = float(np.mean(x_vels)) if x_vels else 0.0
        steps   = len(x_vels)
        fwd     = float(env.unwrapped.data.qpos[0])
        h       = float(env.unwrapped.data.qpos[1])
        print(f'  steps={steps:4d}/{T}  '
              f'avg_vel={avg_vel:+.3f}  disp={fwd:+.2f}m  h={h:.3f}  '
              f'survived={not terminated}')

        if args.render_dir and frames:
            out_dir = Path(args.render_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            gif_path = out_dir / f'sac_ep{ep_key}.gif'
            try:
                import imageio
                try:
                    imageio.mimsave(str(gif_path), frames, fps=args.gif_fps)
                except TypeError:
                    imageio.mimsave(str(gif_path), frames,
                                    duration=int(1000 / args.gif_fps))
                print(f'    [gif saved] {gif_path}')
            except ImportError:
                pass

    hf.close()
    env.close()


if __name__ == '__main__':
    main()
