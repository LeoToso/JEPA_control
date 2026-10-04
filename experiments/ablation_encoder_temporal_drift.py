#!/usr/bin/env python
"""Ablation 1 — encoder temporal consistency at equilibrium.

Reset CartPole to the upright equilibrium (or a small perturbation of it) and
step with zero action for T steps.  At equilibrium the physical state barely
changes, so consecutive frame-pairs are virtually identical.  We measure:

    Δz_t = ||z_t - z_{t-1}||

where  z_t = encode_obs(obs_t, obs_{t-1}, s_{t-1}).

If DINOv2 produces large Δz_t even when the scene is static, the predictor
cannot learn f(z_eq, 0) = z_eq because z_eq itself is an unstable
representation — the encoder "jitters" at the operating point that LQR needs.

Usage
-----
python experiments/ablation_encoder_temporal_drift.py \\
    --model "1SP+iBOT:/mnt/.../ibot_model.pt:configs/ibot.yaml" \\
    --model "1SP+DINOv2:/mnt/.../dinov2_model.pt:configs/dinov2.yaml" \\
    --n-steps 60 --eps 0.0 0.02 0.05 --n-seeds 8 \\
    --out results/encoder_temporal_drift.pdf
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

from experiments.probe_utils import encode_obs, load_bundle, make_env

COLORS = ['#2166ac', '#d6604d', '#4dac26', '#8856a7',
          '#f4a582', '#a6cee3', '#fb9a99', '#b2df8a']


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
def temporal_drift(bundle, initial_state, n_steps):
    """Step with u=0 from initial_state; return per-step latent deltas.

    Returns
    -------
    dz : (T,) array of ||z_t - z_{t-1}||, where T <= n_steps
    ds : (T,) array of ||s_t - s_{t-1}|| (physical state drift reference)
    """
    env = make_env(bundle['env_cfg'], seed=0)
    obs, state, _ = env.reset_to_state(np.asarray(initial_state, dtype=np.float64))
    prev_obs = obs.copy()

    # z_0 = encode(obs_0, obs_0, s_0)  (no previous frame available)
    z_prev = encode_obs(bundle, obs, prev_obs, state).cpu().numpy().flatten()
    s_prev = state.copy()

    dz_list, ds_list = [], []

    for _ in range(n_steps):
        prev_obs  = obs.copy()
        prev_state = state.copy()
        obs, state, _, done, _ = env.step(0.0)

        # convention: z_{t+1} = encode(obs_{t+1}, obs_t, s_t)
        z_cur = encode_obs(bundle, obs, prev_obs, prev_state).cpu().numpy().flatten()

        dz_list.append(float(np.linalg.norm(z_cur - z_prev)))
        ds_list.append(float(np.linalg.norm(state - s_prev)))

        z_prev = z_cur
        s_prev = state.copy()
        if done:
            break

    env.close()
    return np.array(dz_list), np.array(ds_list)


def run_model(bundle, initial_states, n_steps):
    """Average Δz and Δs over a list of initial states."""
    dz_all, ds_all = [], []
    for s0 in initial_states:
        dz, ds = temporal_drift(bundle, s0, n_steps)
        if len(dz) < n_steps:
            dz = np.pad(dz, (0, n_steps - len(dz)), constant_values=np.nan)
            ds = np.pad(ds, (0, n_steps - len(ds)), constant_values=np.nan)
        dz_all.append(dz)
        ds_all.append(ds)
    return np.array(dz_all), np.array(ds_all)   # (n_seeds, n_steps)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', action='append', dest='models',
                   metavar='LABEL:CKPT:CFG',
                   help='Repeat per model: "label:ckpt_path:cfg_path"')
    p.add_argument('--n-steps', type=int, default=60,
                   help='Steps with zero action per trial')
    p.add_argument('--eps', type=float, nargs='+', default=[0.0, 0.02, 0.05],
                   help='Initial-state perturbation magnitudes')
    p.add_argument('--n-seeds', type=int, default=8,
                   help='Random perturbation directions per eps > 0')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    if not args.models:
        p.error('Provide at least one --model "label:ckpt:cfg"')

    rng = np.random.default_rng(42)

    # Build initial-state sets for each eps
    init_states = {}
    for eps in args.eps:
        if eps == 0.0:
            init_states[eps] = [np.zeros(4)]
        else:
            seeds = []
            for _ in range(args.n_seeds):
                v = rng.standard_normal(4)
                v /= np.linalg.norm(v)
                seeds.append(v * eps)
            init_states[eps] = seeds

    # ── collect results ──────────────────────────────────────────────────────
    all_results = {}   # label -> {eps -> (dz_arr, ds_arr)}
    for spec in args.models:
        label, ckpt, cfg = spec.split(':', 2)
        print(f'\n[{label}] loading …')
        bundle = load_bundle(ckpt, cfg, args.device)
        _env_patch(bundle)

        all_results[label] = {}
        for eps in args.eps:
            dz_arr, ds_arr = run_model(bundle, init_states[eps], args.n_steps)
            all_results[label][eps] = (dz_arr, ds_arr)
            mean_dz = float(np.nanmean(dz_arr))
            mean_ds = float(np.nanmean(ds_arr))
            print(f'  eps={eps:.3f}  mean Δz={mean_dz:.5f}  mean Δs={mean_ds:.6f}')

    # ── summary table: mean Δz and Δz/Δs ratio ──────────────────────────────
    # Reference Δs (same for all models — same env & initial states)
    ref_label = args.models[0].split(':', 1)[0]
    ds_ref = {e: float(np.nanmean(all_results[ref_label][e][1])) for e in args.eps}

    eps_nonzero = [e for e in args.eps if e > 0]
    max_lbl = max(len(s.split(':')[0]) for s in args.models)

    print('\n── Summary: mean Δz ──')
    header = ['Model'] + [f'eps={e:.3f}' for e in args.eps]
    col_w  = max(12, max_lbl)
    fmt    = f'{{:<{col_w}}}' + '  {:>10}' * len(args.eps)
    print(fmt.format(*header))
    for spec in args.models:
        label = spec.split(':', 1)[0]
        vals = [f'{np.nanmean(all_results[label][e][0]):.5f}' for e in args.eps]
        print(fmt.format(label, *vals))

    if eps_nonzero:
        print('\n── Summary: mean Δz / mean Δs (encoder gain) ──')
        header2 = ['Model'] + [f'eps={e:.3f}' for e in eps_nonzero]
        fmt2 = f'{{:<{col_w}}}' + '  {:>10}' * len(eps_nonzero)
        print(fmt2.format(*header2))
        for spec in args.models:
            label = spec.split(':', 1)[0]
            ratios = []
            for e in eps_nonzero:
                dz_mean = float(np.nanmean(all_results[label][e][0]))
                ds_mean = ds_ref[e]
                ratios.append(f'{dz_mean / ds_mean:.2f}x' if ds_mean > 0 else 'n/a')
            print(fmt2.format(label, *ratios))

    # ── plot: mean Δz vs eps (left) and Δz/Δs gain vs eps (right) ───────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, spec in enumerate(args.models):
        label = spec.split(':', 1)[0]
        means = [float(np.nanmean(all_results[label][e][0])) for e in args.eps]
        ax1.plot(args.eps, means, 'o-', label=label,
                 color=COLORS[i % len(COLORS)], linewidth=2, markersize=6)

    ax1.set_xlabel(r'$\|s_0\|$ (perturbation magnitude)', fontsize=14)
    ax1.set_ylabel(r'mean $\|z_t - z_{t-1}\|$', fontsize=14)
    ax1.tick_params(labelsize=13)
    ax1.grid(alpha=0.25)
    ax1.spines[['top', 'right']].set_visible(False)

    # Gain plot (only non-zero eps)
    if eps_nonzero:
        for i, spec in enumerate(args.models):
            label = spec.split(':', 1)[0]
            gains = []
            for e in eps_nonzero:
                dz_mean = float(np.nanmean(all_results[label][e][0]))
                ds_mean = ds_ref[e]
                gains.append(dz_mean / ds_mean if ds_mean > 0 else np.nan)
            ax2.plot(eps_nonzero, gains, 'o-', label=label,
                     color=COLORS[i % len(COLORS)], linewidth=2, markersize=6)

        ax2.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.4,
                    label='gain = 1 (latent tracks state)')
        ax2.set_xlabel(r'$\|s_0\|$ (perturbation magnitude)', fontsize=14)
        ax2.set_ylabel(r'mean $\|\Delta z\|$ / mean $\|\Delta s\|$  (encoder gain)',
                       fontsize=14)
        ax2.tick_params(labelsize=13)
        ax2.grid(alpha=0.25)
        ax2.spines[['top', 'right']].set_visible(False)

    handles, labels_ = ax1.get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=12,
               loc='upper center', ncol=len(args.models),
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()
