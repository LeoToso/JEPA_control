#!/usr/bin/env python
"""Run the SAC policy closed-loop in the REAL Walker2d-v4 environment.

This is the baseline: what velocity does the SAC policy achieve when
given the true environment observations (no world model, no probe)?

Usage
-----
MUJOCO_GL=glfw python experiments/eval_sac_real_env.py \
    --sac-repo sdpkjc/Walker2d-v4-sac_continuous_action-seed4 \
    --trials 5 --n-steps 600
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sac-repo',  default='sdpkjc/Walker2d-v4-sac_continuous_action-seed4')
    p.add_argument('--sac-ckpt',  default=None, help='Local SAC checkpoint (skip download)')
    p.add_argument('--trials',    type=int, default=5)
    p.add_argument('--n-steps',   type=int, default=600)
    p.add_argument('--seed',      type=int, default=42)
    args = p.parse_args()

    from experiments.walker2d_ppo_utils import download_and_load_sac, load_sac_from_local
    print('Loading SAC policy …')
    policy = (load_sac_from_local(args.sac_ckpt)
              if args.sac_ckpt else
              download_and_load_sac(args.sac_repo, device='cpu'))
    print('SAC policy loaded.')

    import gymnasium as gym
    env = gym.make('Walker2d-v4')

    all_vels = []
    for i in range(args.trials):
        obs, _ = env.reset(seed=args.seed + i)
        x_vels = []
        terminated = False
        step = 0
        while step < args.n_steps and not terminated:
            action = policy.act(obs)
            obs, reward, term, trunc, info = env.step(action)
            x_vels.append(float(info.get('x_velocity', 0.0)))
            terminated = term or trunc
            step += 1

        avg_vel = float(np.mean(x_vels)) if x_vels else 0.0
        fwd     = float(env.unwrapped.data.qpos[0])
        h       = float(env.unwrapped.data.qpos[1])
        survived = (step >= args.n_steps) and not terminated
        all_vels.append(avg_vel)
        status = 'SUCCESS' if survived else 'FAIL'
        print(f'  trial {i:02d}  {status}  steps={step}  '
              f'avg_vel={avg_vel:+.3f}  disp={fwd:+.2f}m  h={h:.3f}')

    print(f'\nMean avg_x_velocity: {np.mean(all_vels):.3f} m/s  '
          f'(range {min(all_vels):.3f} – {max(all_vels):.3f})')
    env.close()


if __name__ == '__main__':
    main()

