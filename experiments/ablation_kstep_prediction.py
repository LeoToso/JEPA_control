#!/usr/bin/env python
"""Ablation: K-step closed-loop prediction accuracy under GT-LQR.

Both the real environment and the model predictor run the SAME GT-LQR gain
closed-loop, each acting on their own current state estimate:

  Real:    u_t = clip(-K_gt @ x_t,          ±action_scale)
  Model:   u_t = clip(-K_gt @ decode(ẑ_t),  ±action_scale)

Neither side sees the other's state — they evolve independently from x0.
Physics continues even after done=True (no early termination).

Key question: if you run GT-LQR on the model's own predictions, does it
produce the same closed-loop trajectory as running GT-LQR on reality?

Expected finding
----------------
• 1SP+EP-IDM  (‖B_z‖ ≈ 0.54): model responds to actions → closed-loop model
               trajectory tracks the real stabilising trajectory → low error
• MSP+EP-IDM  (‖B_z‖ ≈ 0.38, ρ(A_z) > 1): B_z ≈ 0 → actions computed from
               model decoded state don't stabilise the latent → model diverges
               while reality stabilises → large error

Usage
-----
python experiments/ablation_kstep_prediction.py \\
    --model "1SP+EP-IDM:/path/ckpt.pt:cfg.yaml:results/decoders/decoder_1SP_EP_IDM.pt" \\
    --model "MSP+EP-IDM:/path/ckpt.pt:cfg.yaml:results/decoders/decoder_MSP_EP_IDM.pt" \\
    --K 25 --n-trials 50 --eps 0.10 \\
    --out results/kstep_prediction.pdf
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn

from experiments.probe_utils import (
    encode_obs, equilibrium_latent, load_bundle, make_env, predict_one,
    local_jacobians,
)

PANEL_BG = 'white'
GRID_KW  = dict(color='#cccccc', linewidth=0.8, alpha=0.9)

COLOR_IDM   = '#e07b00'   # orange — IDM / EP-IDM / PR-IDM
COLOR_SIG   = '#2166ac'   # blue   — SIG / PR-SIG
COLOR_OTHER = '#2ca02c'   # green  — everything else

def _label_color(label: str) -> str:
    u = label.upper()
    if 'SIG' in u:
        return COLOR_SIG
    if 'IDM' in u:
        return COLOR_IDM
    return COLOR_OTHER


def solve_dare(A, B, Q, R):
    """Discrete LQR gain via DARE."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


def gt_linearize(env_cfg, eps=1e-4):
    """Numerically compute (A_d, B_d) for the real environment at upright equilibrium.

    Returns the full-step Jacobians (including frame_skip).
    """
    x_eq = np.zeros(4, dtype=np.float64)

    def step_from(state, u):
        env = make_env(env_cfg, seed=0)
        env.reset_to_state(state.copy())
        _, next_state, _, _, _ = env.step(float(u))
        env.close()
        return next_state.astype(np.float64)

    n = 4
    A = np.zeros((n, n))
    for i in range(n):
        xp = x_eq.copy(); xp[i] += eps
        xm = x_eq.copy(); xm[i] -= eps
        A[:, i] = (step_from(xp, 0.0) - step_from(xm, 0.0)) / (2 * eps)

    B = np.zeros((n, 1))
    B[:, 0] = (step_from(x_eq, eps) - step_from(x_eq, -eps)) / (2 * eps)

    return A, B


class StateDecoder(nn.Module):
    def __init__(self, z_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, z):
        return self.net(z)


def load_decoder(decoder_path, device):
    ck = torch.load(decoder_path, map_location=device, weights_only=False)
    dec = StateDecoder(ck['z_dim'], ck['hidden']).to(device)
    dec.load_state_dict(ck['state_dict'])
    dec.eval()
    print(f'    decoder R²: {[f"{v:.3f}" for v in ck["r2_test"]]}')
    return dec


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
def run_trial(bundle, decoder, K_gt, x0, K_steps, action_scale):
    """One trial: BOTH real env and model predictor run GT-LQR closed-loop.

    Real env:        u_t^real  = clip(-K_gt @ x_t,           ±action_scale)
    Model predictor: u_t^model = clip(-K_gt @ decode(ẑ_t),   ±action_scale)

    Both start from the same initial condition; they each compute their own
    action from their own current state estimate.  No episode termination —
    physics continues even after done=True so divergence stays visible.

    Key question: when GT-LQR is applied to the model's own predictions, does
    it produce the same trajectory as GT-LQR applied to reality?

    Returns
    -------
    errors   : (K_steps,)    ||decode(ẑ_k) - x_k^real||  at each step
    real_arr : (K_steps, 4)  real physical states
    pred_arr : (K_steps, 4)  decoded model predictions
    """
    device = bundle['device']

    # Initialise both sides from the same x0
    env = make_env(bundle['env_cfg'], seed=0)
    obs, state, _ = env.reset_to_state(x0.copy())
    prev_obs = obs.copy()

    env2 = make_env(bundle['env_cfg'], seed=0)
    obs0, _, _ = env2.reset_to_state(x0.copy())
    env2.close()
    z = encode_obs(bundle, obs0, obs0, x0)

    real_states = []
    pred_states = []

    for _ in range(K_steps):
        # Real: action from real physical state
        u_real = float(np.clip(-(K_gt @ state.astype(np.float64)).item(),
                               -action_scale, action_scale))

        # Model: action from decoded model latent state
        x_model = decoder(z).cpu().numpy().flatten()
        u_model = float(np.clip(-(K_gt @ x_model.astype(np.float64)).item(),
                                -action_scale, action_scale))

        # Step real env with real action (no break on done)
        prev_obs = obs.copy()
        obs, state, _, _, _ = env.step(u_real)
        real_states.append(state.copy())

        # Step model predictor with model action
        z = predict_one(bundle, z, u_model)
        pred_states.append(decoder(z).cpu().numpy().flatten())

    env.close()

    real_arr = np.array(real_states)   # (K, 4)
    pred_arr = np.array(pred_states)   # (K, 4)
    errors   = np.linalg.norm(real_arr - pred_arr, axis=1)
    return errors, real_arr, pred_arr


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', action='append', dest='models',
                   metavar='LABEL:CKPT:CFG:DECODER',
                   help='"label:ckpt:cfg:decoder_path"')
    p.add_argument('--K', type=int, default=20,
                   help='Prediction horizon (steps)')
    p.add_argument('--n-trials', type=int, default=50,
                   help='Number of random initial conditions')
    p.add_argument('--eps', type=float, default=0.15,
                   help='Initial state perturbation magnitude')
    p.add_argument('--q-pos', type=float, default=1.0,
                   help='LQR cost weight for cart position')
    p.add_argument('--q-vel', type=float, default=0.1,
                   help='LQR cost weight for cart velocity')
    p.add_argument('--q-ang', type=float, default=10.0,
                   help='LQR cost weight for pole angle')
    p.add_argument('--q-angv', type=float, default=0.1,
                   help='LQR cost weight for pole angular velocity')
    p.add_argument('--r-scale', type=float, default=0.01,
                   help='LQR control cost')
    p.add_argument('--no-bar', action='store_true',
                   help='Only plot the left (line) panel, skip the bar chart')
    p.add_argument('--from-data', metavar='PKL',
                   help='Skip simulation; load saved .pkl and regenerate figure')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    if not args.from_data and not args.models:
        p.error('Provide at least one --model "label:ckpt:cfg:decoder"')

    # ── Load saved data and skip simulation ──────────────────────────────────
    if args.from_data:
        print(f'[loading] {args.from_data}')
        with open(args.from_data, 'rb') as f:
            saved = pickle.load(f)
        steps        = np.array(saved['steps'])
        saved_models = saved['results']          # list of dicts
        # allow CLI overrides of K / no_bar
        K_for_bar = saved['K']

        if args.no_bar:
            fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
            ax2 = None
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for ax in ([ax1] if ax2 is None else [ax1, ax2]):
            ax.set_facecolor(PANEL_BG); ax.grid(True, **GRID_KW)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=13)

        for i, res in enumerate(saved_models):
            mean_err = np.array(res['mean_err'])
            sem_err  = np.array(res['sem_err'])
            label    = res['label']
            color    = _label_color(label)
            ls       = '--' if label.upper().startswith('1SP') else '-'
            ax1.plot(steps, mean_err, color=color, linewidth=2.2,
                     linestyle=ls, label=label)
            ax1.fill_between(steps, mean_err - sem_err, mean_err + sem_err,
                             color=color, alpha=0.18)
            if ax2 is not None:
                final_err = float(mean_err[-1])
                ax2.bar(i, final_err, color=color, alpha=0.85, label=label, width=0.55)
                ax2.errorbar(i, final_err, yerr=sem_err[-1],
                             fmt='none', color='#333333', linewidth=1.5, capsize=5)

        ax1.set_xlabel('k-step', fontsize=14)
        ax1.set_ylabel('k-step prediction error', fontsize=14)
        ax1.legend(fontsize=11, loc='upper left')
        if ax2 is not None:
            ax2.set_ylabel(r'$\|\hat{x}_K - x_K^{\mathrm{gt}}\|$  at $K=' +
                           str(K_for_bar) + r'$', fontsize=13)
            ax2.set_title(f'Final-step error per model  ($K={K_for_bar}$)', fontsize=14)
            ax2.set_xticks(range(len(saved_models)))
            ax2.set_xticklabels([r['label'] for r in saved_models],
                                fontsize=11, rotation=15, ha='right')

        fig.tight_layout()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches='tight')
        print(f'[done] {out}')
        return

    # ── Full simulation ───────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    # Initial conditions
    x0s = [rng.uniform(-args.eps, args.eps, size=4).astype(np.float64)
           for _ in range(args.n_trials)]

    # GT-LQR computed lazily from the first model loaded (physics is shared)
    K_gt = None
    action_scale = None

    if args.no_bar:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
        ax2 = None
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax in ([ax1] if ax2 is None else [ax1, ax2]):
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, **GRID_KW)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=13)

    steps = np.arange(1, args.K + 1)
    all_results = []   # accumulate per-model data for saving

    for i, spec in enumerate(args.models):
        parts = spec.split(':', 3)
        label, ckpt, cfg, decoder_path = parts
        print(f'\n[{label}] loading …')
        bundle = load_bundle(ckpt, cfg, args.device)
        _env_patch(bundle)

        decoder = load_decoder(decoder_path, bundle['device'])

        # Compute GT-LQR once from the first successfully loaded bundle
        if K_gt is None:
            print('[GT-LQR] computing physical linearisation …')
            A_gt, B_gt = gt_linearize(bundle['env_cfg'])
            Q_gt = np.diag([args.q_pos, args.q_vel, args.q_ang, args.q_angv])
            R_gt = np.array([[args.r_scale]])
            K_gt = solve_dare(A_gt, B_gt, Q_gt, R_gt)
            rho_cl_gt = float(np.max(np.abs(np.linalg.eigvals(A_gt - B_gt @ K_gt))))
            action_scale = float(bundle['action_scale'])
            print(f'  ‖K_gt‖={np.linalg.norm(K_gt):.2f}  ρ(A-BK)={rho_cl_gt:.4f}')

        # Latent Jacobian summary (diagnostic only)
        z_eq_t = equilibrium_latent(bundle)
        A_z, B_z, fp_err = local_jacobians(bundle, z=z_eq_t)
        rho = float(np.max(np.abs(np.linalg.eigvals(A_z))))
        print(f'  ‖B_z‖={np.linalg.norm(B_z):.3f}  ρ(A_z)={rho:.4f}  fp_err={fp_err:.4f}')

        all_errors = []
        all_real   = []
        all_pred   = []

        for j, x0 in enumerate(x0s):
            err, real_s, pred_s = run_trial(
                bundle, decoder, K_gt, x0, args.K, action_scale)
            all_errors.append(err)
            all_real.append(real_s)
            all_pred.append(pred_s)

        all_errors = np.array(all_errors)   # (n_trials, K)
        mean_err = all_errors.mean(axis=0)
        sem_err  = all_errors.std(axis=0) / np.sqrt(args.n_trials)

        final_err = float(mean_err[-1])
        print(f'  mean error @ k={args.K}: {final_err:.4f}')

        all_results.append({
            'label':      label,
            'all_errors': all_errors,
            'mean_err':   mean_err,
            'sem_err':    sem_err,
            'B_z_norm':   float(np.linalg.norm(B_z)),
            'rho_Az':     rho,
            'fp_err':     fp_err,
        })

        color = _label_color(label)
        ls    = '--' if label.upper().startswith('1SP') else '-'

        # Left: mean ± SEM of ||x̂_k - x_k^gt|| vs step
        ax1.plot(steps, mean_err, color=color, linewidth=2.2,
                 linestyle=ls, label=label)
        ax1.fill_between(steps, mean_err - sem_err, mean_err + sem_err,
                         color=color, alpha=0.18)

        # Right: final-step error distribution (skipped if --no-bar)
        if ax2 is not None:
            ax2.bar(i, final_err, color=color, alpha=0.85,
                    label=label, width=0.55)
            ax2.errorbar(i, final_err, yerr=sem_err[-1],
                         fmt='none', color='#333333', linewidth=1.5, capsize=5)

    ax1.set_xlabel('k-step', fontsize=14)
    ax1.set_ylabel('k-step prediction error', fontsize=14)
    ax1.legend(fontsize=11, loc='upper left')

    if ax2 is not None:
        ax2.set_ylabel(r'$\|\hat{x}_K - x_K^{\mathrm{gt}}\|$  at $K=' +
                       str(args.K) + r'$', fontsize=13)
        ax2.set_title(f'Final-step error per model  ($K={args.K}$)', fontsize=14)
        ax2.set_xticks(range(len(args.models)))
        ax2.set_xticklabels([s.split(':')[0] for s in args.models],
                            fontsize=11, rotation=15, ha='right')

    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')

    # Save raw data so figures can be regenerated without re-running simulation
    data_path = out.with_suffix('.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump({
            'K':       args.K,
            'n_trials': args.n_trials,
            'eps':     args.eps,
            'seed':    args.seed,
            'steps':   steps.tolist(),
            'K_gt':    K_gt,
            'results': all_results,
        }, f)
    print(f'[data]  {data_path}')
    print(f'[done]  {out}')


if __name__ == '__main__':
    main()
