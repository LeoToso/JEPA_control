#!/usr/bin/env python
"""Visualize the GT Walker2d limit cycle under the SAC policy.

Three-panel figure (all GT MuJoCo dynamics, no learned model):

  Panel 1 – PCA  : PC1 vs PC2 of the centred 17-D gait state
  Panel 2 – Coord: right-hip angle vs left-hip angle (anti-phase)
  Panel 3 – Phase : right-hip angle vs right-hip angular velocity

Styling matches the rest of the project (periwinkle bg, white grid lines,
bold titles, TICK_SIZE=13, TITLE_SIZE=16).  Each episode is one
time-coloured trajectory (early = light, late = saturated) so multiple
overlapping orbits reveal the attractor shape.

Walker2d-v4 observation layout (17-D)
--------------------------------------
  obs[ 0] = qpos[1]  torso z-height
  obs[ 1] = qpos[2]  torso tilt angle
  obs[ 2] = qpos[3]  right-thigh (hip flexion)   ← right_hip_angle
  obs[ 3] = qpos[4]  right-leg   (knee)
  obs[ 4] = qpos[5]  right-foot  (ankle)
  obs[ 5] = qpos[6]  left-thigh  (hip flexion)   ← left_hip_angle
  obs[ 6] = qpos[7]  left-leg    (knee)
  obs[ 7] = qpos[8]  left-foot   (ankle)
  obs[ 8] = qvel[0]  torso x-velocity (forward)
  obs[ 9] = qvel[1]  torso z-velocity
  obs[10] = qvel[2]  torso angular velocity
  obs[11] = qvel[3]  right-thigh angular velocity ← right_hip_vel
  obs[12] = qvel[4]  right-leg angular velocity
  obs[13] = qvel[5]  right-foot angular velocity
  obs[14] = qvel[6]  left-thigh angular velocity  ← left_hip_vel
  obs[15] = qvel[7]  left-leg angular velocity
  obs[16] = qvel[8]  left-foot angular velocity

Usage
-----
MUJOCO_GL=egl python experiments/plot_gt_limit_cycle_sac.py \\
    --sac-repo sdpkjc/Walker2d-v4-sac_continuous_action-seed4 \\
    --n-episodes 8 --n-steps 600 --warmup 150 \\
    --out results/gt_limit_cycle_sac.pdf
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# ── style ──────────────────────────────────────────────────────────────────────
PANEL_BG   = '#e8eaf2'
GRID_KW    = dict(color='white', linewidth=0.8, alpha=0.9)
TICK_SIZE  = 13
TITLE_SIZE = 16
CMAP       = 'Blues'        # time colormap per episode (light → dark)
LINEWIDTH  = 1.4

# Walker2d obs indices
IDX_RIGHT_HIP_ANG = 2
IDX_LEFT_HIP_ANG  = 5
IDX_RIGHT_HIP_VEL = 11


# ── rollout ────────────────────────────────────────────────────────────────────

def collect_trajectories(
    policy,
    n_episodes: int = 8,
    n_steps:    int = 600,
    warmup:     int = 150,
    seed:       int = 0,
) -> list[np.ndarray]:
    """Roll out the policy; return steady-state state arrays (T, 17)."""
    import gymnasium as gym

    trajs: list[np.ndarray] = []
    for ep in range(n_episodes):
        env = gym.make('Walker2d-v4')
        obs, _ = env.reset(seed=seed + ep)
        states: list[np.ndarray] = []
        for t in range(n_steps + warmup):
            action = policy.act(obs.astype(np.float32))
            obs, _, term, trunc, _ = env.step(action)
            if t >= warmup:
                states.append(obs.astype(np.float32))
            if term or trunc:
                break
        env.close()
        if len(states) >= 10:
            trajs.append(np.stack(states))   # (T, 17)
            print(f'  episode {ep}: {len(states)} steady-state steps')
        else:
            print(f'  episode {ep}: fell before steady state — skipped')
    return trajs


# ── PCA helper ─────────────────────────────────────────────────────────────────

def centred_pca(trajs: list[np.ndarray]) -> list[np.ndarray]:
    """Per-episode centering → global PCA; return list of (T, 2) projections."""
    centred = [t - t.mean(axis=0, keepdims=True) for t in trajs]
    all_c   = np.concatenate(centred, axis=0)
    _, _, Vt = np.linalg.svd(all_c, full_matrices=False)
    V2 = Vt[:2].T                              # (17, 2)
    return [c @ V2 for c in centred]


# ── drawing helpers ────────────────────────────────────────────────────────────

def _add_orbit(ax, xy: np.ndarray, cmap=CMAP, lw=LINEWIDTH) -> None:
    """Draw one time-coloured trajectory on ax."""
    if len(xy) < 2:
        return
    pts  = np.column_stack([xy[:, 0], xy[:, 1]])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    t_n  = np.linspace(0, 1, len(segs))
    lc   = LineCollection(segs, cmap=cmap, norm=Normalize(0, 1),
                          linewidths=lw, alpha=0.85)
    lc.set_array(t_n)
    ax.add_collection(lc)


def _style_ax(ax, xlabel, ylabel, title=None) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_axisbelow(True)
    ax.grid(True, **GRID_KW)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=TITLE_SIZE)
    ax.set_ylabel(ylabel, fontsize=TITLE_SIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=6)


def _autolim(ax, *xys, pad=0.08) -> None:
    xs = np.concatenate([xy[:, 0] for xy in xys if len(xy)])
    ys = np.concatenate([xy[:, 1] for xy in xys if len(xy)])
    rx, ry = xs.ptp(), ys.ptp()
    ax.set_xlim(xs.min() - pad * rx, xs.max() + pad * rx)
    ax.set_ylim(ys.min() - pad * ry, ys.max() + pad * ry)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--sac-repo',   default='sdpkjc/Walker2d-v4-sac_continuous_action-seed4')
    p.add_argument('--sac-ckpt',   default=None, help='Local SAC .cleanrl_model (skip HF)')
    p.add_argument('--n-episodes', type=int, default=8)
    p.add_argument('--n-steps',    type=int, default=600,
                   help='Steady-state steps per episode (after warmup)')
    p.add_argument('--warmup',     type=int, default=150,
                   help='Steps discarded at episode start (transient)')
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--device',     default='cpu')
    p.add_argument('--out',        default='results/gt_limit_cycle_sac.pdf')
    args = p.parse_args()

    # ── load policy ───────────────────────────────────────────────────────────
    from experiments.walker2d_ppo_utils import download_and_load_sac, load_sac_from_local
    if args.sac_ckpt:
        policy = load_sac_from_local(args.sac_ckpt, device=args.device)
    else:
        policy = download_and_load_sac(args.sac_repo, device=args.device)

    # ── collect GT trajectories ───────────────────────────────────────────────
    print(f'\n[collect] {args.n_episodes} episodes × '
          f'({args.warmup} warmup + {args.n_steps} steady) steps …')
    trajs = collect_trajectories(
        policy,
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        warmup=args.warmup,
        seed=args.seed,
    )
    if not trajs:
        print('[error] no valid episodes — policy may not be walking')
        return
    print(f'[collect] {len(trajs)} valid episodes')

    # ── right-hip phase portrait only ─────────────────────────────────────────
    phase_xys = [np.column_stack([t[:, IDX_RIGHT_HIP_ANG],
                                   t[:, IDX_RIGHT_HIP_VEL]]) for t in trajs]

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    for xy in phase_xys:
        _add_orbit(ax, xy)
    _autolim(ax, *phase_xys)
    _style_ax(ax,
              xlabel=r'Right hip $\theta$ (rad)',
              ylabel=r'Right hip $\dot{\theta}$ (rad/s)',
              title=None)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'\n[saved] {out}')


if __name__ == '__main__':
    main()

