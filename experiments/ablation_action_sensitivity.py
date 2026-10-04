#!/usr/bin/env python
"""Ablation: multi-step prediction sensitivity to action perturbations.

Starting from the upright equilibrium we apply a reference action sequence
u_ref (default: 0) and a perturbed version  ũ_t = u_ref + δ_t, δ_t ~ N(0,σ²).

For each model we compare how much the *model predictor* moves vs how much
the *GT encoder* (real environment) moves due to the same perturbation:

  model_disp(k, σ) = E[||ẑ_k(ũ) - ẑ_k(u_ref)||]       predictor displacement
  gt_disp(k, σ)    = E[||z_k^gt(ũ) - z_k^gt(u_ref)||]  GT encoder displacement

  sensitivity_ratio(k, σ) = model_disp(k, σ) / gt_disp(k, σ)

Interpretation
--------------
  ratio ≈ 1  the predictor correctly tracks the real action sensitivity
  ratio ≈ 0  the predictor ignores actions (B_z → 0, MSP collapse)
  ratio > 1  the predictor overshoots (B_z too large / wrong direction)

This normalises out each model's latent scale so models with different
||B_z|| are directly comparable.

Two panels:
  Left  — sensitivity_ratio vs step k  (fixed σ = --sigma-mid)
  Right — sensitivity_ratio at step K = --K-fixed vs σ

Usage
-----
python experiments/ablation_action_sensitivity.py \\
    --model "1SP+EP-IDM:ckpt.pt:cfg.yaml" \\
    --model "MSP+EP-IDM:ckpt.pt:cfg.yaml" \\
    --model "MSP+EP-IDM+SIG:ckpt.pt:cfg.yaml" \\
    --out results/action_sensitivity.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from probe_utils import (
    encode_obs, equilibrium_latent, load_bundle, local_jacobians,
    make_env, predict_one,
)

COLORS   = ['#2166ac', '#d6604d', '#4dac26', '#8856a7', '#f4a582']
PANEL_BG = '#dce3ea'


def _env_patch(bundle):
    if 'environment' not in bundle['env_cfg']:
        bundle['env_cfg']['environment'] = {
            'frame_skip': int(bundle['model_cfg'].get('frame_skip', 5)),
            'image_size': int(bundle['model_cfg'].get('image_size', 128)),
            'action_range': [-10, 10],
            'mass_cart': 1.0, 'mass_pole': 0.1,
            'pole_length': 0.5, 'gravity': 9.8, 'dt': 0.02,
        }


@torch.no_grad()
def _model_rollout(bundle, z0, actions):
    """Predictor rollout; returns list of K latent vectors (numpy)."""
    z = z0.clone()
    zs = []
    for u in actions:
        z = predict_one(bundle, z, float(u))
        zs.append(z.cpu().numpy().flatten())
    return zs


@torch.no_grad()
def _gt_rollout(bundle, s0, actions):
    """GT environment rollout; returns list of K latent vectors (numpy)."""
    env   = make_env(bundle['env_cfg'], seed=0)
    obs, state, _ = env.reset_to_state(s0.copy())
    prev_obs = obs.copy()
    zs = []
    done = False
    for u in actions:
        if not done:
            obs_next, state_next, _, done, _ = env.step(float(u))
            z_gt = encode_obs(bundle, obs_next, prev_obs, state)
            prev_obs  = obs_next.copy()
            state     = state_next.copy()
        else:
            z_gt = zs[-1] if zs else encode_obs(bundle, obs, obs, state)
        zs.append(z_gt.cpu().numpy().flatten()
                  if torch.is_tensor(z_gt) else z_gt)
    env.close()
    return zs


def sensitivity_curves(bundle, K, sigma, n_samples, rng, u_ref=0.0):
    """Compute sensitivity_ratio(k) for k=1..K at a given sigma.

    Returns
    -------
    ratio    : (K,)   mean model_disp / mean gt_disp at each step
    m_disp   : (K,)   mean model displacement (numerator)
    g_disp   : (K,)   mean GT displacement   (denominator)
    """
    action_scale = float(bundle['action_scale'])
    s_eq = np.zeros(4, dtype=np.float64)
    z_eq_t = equilibrium_latent(bundle)

    # Reference rollout (no perturbation)
    ref_actions = np.full(K, u_ref)
    z_ref_model = _model_rollout(bundle, z_eq_t, ref_actions)
    z_ref_gt    = _gt_rollout(bundle, s_eq, ref_actions)

    model_disps = np.zeros((n_samples, K))
    gt_disps    = np.zeros((n_samples, K))

    for s in range(n_samples):
        delta   = rng.normal(0.0, sigma, size=K)
        actions = np.clip(ref_actions + delta, -action_scale, action_scale)

        z_pert_model = _model_rollout(bundle, z_eq_t, actions)
        z_pert_gt    = _gt_rollout(bundle, s_eq, actions)

        for k in range(K):
            model_disps[s, k] = np.linalg.norm(
                z_pert_model[k] - z_ref_model[k])
            gt_disps[s, k] = np.linalg.norm(
                z_pert_gt[k] - z_ref_gt[k])

    m_disp = model_disps.mean(axis=0)
    g_disp = gt_disps.mean(axis=0)
    ratio  = np.where(g_disp > 1e-8, m_disp / g_disp, np.nan)

    return ratio, m_disp, g_disp


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', action='append', dest='models',
                   metavar='LABEL:CKPT:CFG',
                   help='Repeat per model: "label:ckpt:cfg"')
    p.add_argument('--K', type=int, default=20,
                   help='Rollout horizon (steps)')
    p.add_argument('--K-fixed', type=int, default=10,
                   help='Fixed horizon for the σ-sweep panel')
    p.add_argument('--sigma-values', type=float, nargs='+',
                   default=[0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
                   help='Noise levels σ for the σ-sweep panel')
    p.add_argument('--sigma-mid', type=float, default=2.0,
                   help='σ used for the step-by-step panel')
    p.add_argument('--n-samples', type=int, default=50,
                   help='Perturbation draws per (σ, model)')
    p.add_argument('--u-ref', type=float, default=0.0,
                   help='Reference action applied at every step')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    if not args.models:
        p.error('Provide at least one --model "label:ckpt:cfg"')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=13)

    # ideal ratio = 1 reference line
    ax1.axhline(1.0, color='#333333', linewidth=1.2, linestyle=':',
                label='ratio = 1  (ideal)')
    ax2.axhline(1.0, color='#333333', linewidth=1.2, linestyle=':',
                label='ratio = 1  (ideal)')

    steps  = np.arange(1, args.K + 1)
    k_idx  = min(args.K_fixed, args.K) - 1

    for i, spec in enumerate(args.models):
        label, ckpt, cfg = spec.split(':', 2)
        print(f'\n[{label}] loading …')
        bundle = load_bundle(ckpt, cfg, args.device)
        _env_patch(bundle)

        # Jacobian summary
        z_eq_t = equilibrium_latent(bundle)
        A_z, B_z, fp_err = local_jacobians(bundle, z=z_eq_t)
        rho    = float(np.max(np.abs(np.linalg.eigvals(A_z))))
        norm_B = float(np.linalg.norm(B_z))
        info   = f'fp={fp_err:.3f}  ρ={rho:.3f}  ‖B_z‖={norm_B:.3f}'
        print(f'  {info}')

        color = COLORS[i % len(COLORS)]
        lbl   = f'{label}  [{info}]'

        # ── Left panel: ratio vs step k (fixed sigma_mid) ────────────────
        print(f'  K={args.K}, σ={args.sigma_mid}, n={args.n_samples} …')
        rng_mid = np.random.default_rng(args.seed + 100 * i)
        ratio_k, m_d, g_d = sensitivity_curves(
            bundle, args.K, args.sigma_mid, args.n_samples, rng_mid,
            args.u_ref)
        print(f'    ratio @ k=1: {ratio_k[0]:.3f}  '
              f'@ k={args.K}: {ratio_k[-1]:.3f}')

        ax1.plot(steps, ratio_k, color=color, linewidth=2.2, label=lbl)

        # ── Right panel: ratio at K_fixed vs sigma sweep ─────────────────
        ratios_at_K = []
        for j, sigma in enumerate(args.sigma_values):
            print(f'    σ={sigma:.1f} …', end='  ', flush=True)
            rng_s = np.random.default_rng(args.seed + 100 * i + j + 1)
            ratio_s, _, _ = sensitivity_curves(
                bundle, args.K, sigma, args.n_samples, rng_s, args.u_ref)
            ratios_at_K.append(float(ratio_s[k_idx]))
        print()

        ax2.plot(args.sigma_values, ratios_at_K, 'o-',
                 color=color, linewidth=2.2, markersize=6, label=label)

    ax1.set_xlabel('Step $k$', fontsize=14)
    ax1.set_ylabel(
        r'$\frac{\mathrm{E}[\|\hat{z}_k^{\mathrm{pert}}-\hat{z}_k^{\mathrm{ref}}\|]}'
        r'{\mathrm{E}[\|z_k^{\mathrm{gt,pert}}-z_k^{\mathrm{gt,ref}}\|]}$',
        fontsize=13)
    ax1.set_title(
        f'Sensitivity ratio vs horizon  ($\\sigma={args.sigma_mid}$)',
        fontsize=14)
    ax1.legend(fontsize=10, loc='upper center',
               bbox_to_anchor=(0.5, -0.18), ncol=1)

    ax2.set_xlabel(r'Action noise $\sigma$', fontsize=14)
    ax2.set_ylabel(
        f'Sensitivity ratio at $K={args.K_fixed}$', fontsize=13)
    ax2.set_title(
        f'Sensitivity ratio vs noise level  ($K={args.K_fixed}$)',
        fontsize=14)
    ax2.legend(fontsize=11, loc='upper left')

    fig.suptitle(
        'Predictor action-sensitivity relative to real dynamics',
        fontsize=14, y=1.01)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()
