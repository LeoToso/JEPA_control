#!/usr/bin/env python
"""Ground-truth planning cost landscape.

For each (theta, theta_dot) grid point runs the GT CartPole environment
H steps with zero action and plots log10||s_H - s_goal||^2.

This is the model-free baseline for Panel 3 of
plot_checkpoint_summary_physical_smwm.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import yaml

from experiments.probe_utils import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg',   required=True, help='Path to env config yaml')
    p.add_argument('--out',   required=True, help='Output file path')
    p.add_argument('--horizon',        type=int,   default=3)
    p.add_argument('--pred-theta-max', type=float, default=25.)
    p.add_argument('--pred-rate-max',  type=float, default=100.)
    p.add_argument('--pred-theta-pts', type=int,   default=13)
    p.add_argument('--pred-rate-pts',  type=int,   default=11)
    args = p.parse_args()

    with open(args.cfg) as f:
        env_cfg = yaml.safe_load(f)
    if 'environment' not in env_cfg:
        env_cfg['environment'] = {
            'frame_skip': 5, 'image_size': 128,
            'action_range': [-10, 10],
            'mass_cart': 1.0, 'mass_pole': 0.1,
            'pole_length': 0.5, 'gravity': 9.8, 'dt': 0.02,
        }

    thetas   = np.linspace(-args.pred_theta_max, args.pred_theta_max,
                           args.pred_theta_pts)
    thetadots = np.linspace(-args.pred_rate_max, args.pred_rate_max,
                            args.pred_rate_pts)
    s_goal = np.zeros(4, dtype=np.float64)

    costs = np.full((len(thetadots), len(thetas)), np.nan)
    env = make_env(env_cfg, seed=0)

    for j, th in enumerate(thetas):
        for i, thd in enumerate(thetadots):
            state = np.array([0., 0., np.deg2rad(th), np.deg2rad(thd)],
                             dtype=np.float64)
            _, s, _ = env.reset_to_state(state)
            for _ in range(args.horizon):
                _, s, _, done, _ = env.step(0.)
                if done:
                    break
            err2 = float(np.sum((s - s_goal) ** 2))
            costs[i, j] = np.log10(err2 + 1e-8)

    env.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.pcolormesh(thetas, thetadots, costs, cmap='RdYlBu_r',
                       shading='auto')
    ax.contour(thetas, thetadots, costs, levels=8, colors='k',
               linewidths=0.5, alpha=0.4)
    ax.scatter([0], [0], marker='*', s=120, c='lime', zorder=5,
               label='Equilibrium')
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'$\dot\theta$ [deg/s]')
    ax.set_title(fr'GT planning cost $H={args.horizon}$, zero action')
    ax.legend(fontsize=7, loc='upper right')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$\log_{10}\|s_H - s_{\rm goal}\|^2$')
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
