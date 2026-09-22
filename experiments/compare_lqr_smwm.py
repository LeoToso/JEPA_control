#!/usr/bin/env python
"""
Compare LQR control in latent space: AR vs SIGReg SMWM.

For each model:
  1. Linearise the latent predictor at the upright equilibrium: A_z, B_z
  2. Solve the discrete LQR (DARE) with Q = q_scale * I, R = r_scale
  3. Run closed-loop trials: u_t = -K @ (z_t - z*)
  4. Report success rate, fraction stable, settling time

Key prediction: AR should stabilise (B_z has range covering the unstable
latent mode by identifiability); SIGReg may fail (ker(E) ∩ U ≠ {0} →
unstable mode absent from latent dynamics → DARE gain cannot stabilise it).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.linalg
import torch

from sensorimotor_probe_utils import (
    encode_obs, encode_rendered_state, local_jacobians, load_bundle, make_env,
    make_frame_buffer, push_frame)


def checkpoint_label(checkpoint):
    path = Path(checkpoint)
    run_name = (path.parent.parent.name
                if path.parent.name == 'checkpoints' else path.parent.name)
    r = run_name.lower()
    if 'sigreg_ms' in r or 'sigreg-ms' in r:
        return 'SIGReg MS'
    if 'sigreg' in r:
        return 'SIGReg'
    if '_sr_' in r or r.endswith('_sr'):
        return 'State recon.'
    if 'ar_ms' in r or 'ar-ms' in r:
        return 'AR MS'
    if 'dinov2' in r:
        return 'DINOv2'
    if 'proprio' in r:
        return 'Action recon.'
    return run_name


def solve_dare(A, B, Q, R):
    """Discrete LQR gain via DARE.  Returns K (1 × d) or raises."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K  # (1, d)


def pbh_check(A, B, tol=1e-3):
    """PBH rank test for each unstable eigenvalue of A."""
    eigs = np.linalg.eigvals(A)
    unstable = eigs[np.abs(eigs) > 1.0 + tol]
    results = []
    d = A.shape[0]
    for lam in unstable:
        M = np.hstack([A - lam * np.eye(d), B])
        rank = np.linalg.matrix_rank(M, tol=1e-6)
        results.append({'eigenvalue': complex(lam), 'rank': int(rank),
                        'controllable': bool(rank == d)})
    return results


def lqr_trial(bundle, K, z_goal, initial_state, n_steps,
              success_threshold, success_hold_steps, action_scale):
    """One closed-loop LQR episode; returns result dict."""
    env = make_env(bundle['env_cfg'], seed=0)
    obs, state, _ = env.reset_to_state(
        np.asarray(initial_state, dtype=np.float64))
    frame_buf = make_frame_buffer(bundle, obs)

    goal_state = np.zeros(4)
    states = [state.copy()]
    actions = []
    terminated = False

    for _ in range(n_steps):
        with torch.no_grad():
            z = encode_obs(bundle, frame_buf, obs, state)
            z_np = z.cpu().numpy().flatten()

        u = float(-(K @ (z_np - z_goal)).item())
        u = float(np.clip(u, -action_scale, action_scale))

        prev_obs = obs
        obs, state, _, done, _ = env.step(u)
        push_frame(frame_buf, obs)
        states.append(state.copy())
        actions.append(u)
        if done:
            terminated = True
            # Do not break — keep stepping so the trajectory always spans
            # n_steps and we can see how the state diverges after failure.

    env.close()
    states = np.array(states)
    errors = np.linalg.norm(states - goal_state[None], axis=1)

    stable_mask = errors < success_threshold
    # Held: last success_hold_steps all below threshold
    tail = stable_mask[-success_hold_steps:]
    held = bool(len(tail) == success_hold_steps and tail.all())
    success = bool(stable_mask[-1])

    # Settling time: first step of the final stable suffix
    settling = None
    run = 0
    for i in range(len(stable_mask) - 1, -1, -1):
        if stable_mask[i]:
            run += 1
        else:
            break
    if run >= success_hold_steps:
        settling = len(stable_mask) - run

    return {
        'success': success,
        'held': held,
        'terminated': terminated,
        'final_error': float(errors[-1]),
        'max_error': float(np.max(errors)),
        'fraction_stable': float(np.mean(stable_mask)),
        'settling_step': settling,
        'states': states.tolist(),
        'actions': actions,
    }


def main():
    p = argparse.ArgumentParser(
        description='Compare latent-space LQR for AR vs SIGReg SMWM.')
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--cfgs', nargs='+', required=True)
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--n-steps', type=int, default=100,
                   help='Max macro-steps per trial')
    p.add_argument('--success-threshold', type=float, default=0.3,
                   help='Physical state norm threshold for success')
    p.add_argument('--success-hold-steps', type=int, default=10,
                   help='Steps below threshold to count as stabilised')
    p.add_argument('--eps-range', type=float, nargs=2, default=[0.01, 0.15],
                   help='Uniform perturbation range for initial states')
    p.add_argument('--q-scale', type=float, default=1.0,
                   help='Q = q_scale * I_d')
    p.add_argument('--r-scale', type=float, default=1.0,
                   help='R = r_scale (scalar cost on action)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    if len(args.cfgs) == 1:
        args.cfgs = args.cfgs * len(args.ckpts)
    if len(args.cfgs) != len(args.ckpts):
        raise ValueError('--cfgs must have length 1 or match --ckpts')

    rng = np.random.default_rng(args.seed)
    lo, hi = args.eps_range
    initial_states = [
        rng.uniform(-hi, hi, size=4).clip(-hi, hi).astype(np.float64)
        for _ in range(args.trials)
    ]
    # Ensure angle perturbation is meaningful
    for s in initial_states:
        s[2] = rng.uniform(-hi, hi)   # theta
        s[3] = rng.uniform(-hi, hi)   # theta_dot

    results = {
        'protocol': {
            'trials': args.trials,
            'n_steps': args.n_steps,
            'success_threshold': args.success_threshold,
            'success_hold_steps': args.success_hold_steps,
            'q_scale': args.q_scale,
            'r_scale': args.r_scale,
            'eps_range': args.eps_range,
            'seed': args.seed,
        },
        'models': [],
    }

    for ckpt, cfg in zip(args.ckpts, args.cfgs):
        label = checkpoint_label(ckpt)
        print(f'\n[{label}] loading {ckpt}')
        bundle = load_bundle(ckpt, cfg, args.device)
        action_scale = float(bundle['action_scale'])

        # ── Patch env_cfg if results config lacks 'environment' section ──────
        if 'environment' not in bundle['env_cfg']:
            bundle['env_cfg']['environment'] = {
                'frame_skip': int(bundle['model_cfg'].get('frame_skip', 5)),
                'image_size': int(bundle['model_cfg'].get('image_size', 128)),
                'action_range': [-10, 10],
                'mass_cart': 1.0, 'mass_pole': 0.1,
                'pole_length': 0.5, 'gravity': 9.8, 'dt': 0.02,
            }

        # ── Linearise at equilibrium ──────────────────────────────────────────
        print(f'[{label}] computing local Jacobians at z* …')
        z_eq = encode_rendered_state(bundle, np.zeros(4, dtype=np.float32))
        A_z, B_z, fp_err = local_jacobians(bundle, z=z_eq)
        d = A_z.shape[0]
        rho_Az = float(np.max(np.abs(np.linalg.eigvals(A_z))))
        n_unstable = int(np.sum(np.abs(np.linalg.eigvals(A_z)) > 1.0))
        print(f'[{label}] d={d}  ρ(A_z)={rho_Az:.4f}  '
              f'unstable modes={n_unstable}  fp_err={fp_err:.4e}')

        # ── PBH controllability on unstable modes ─────────────────────────────
        pbh = pbh_check(A_z, B_z)
        for item in pbh:
            ctrl = 'controllable' if item['controllable'] else 'NOT controllable'
            print(f'[{label}]   λ={item["eigenvalue"]:.4f}  {ctrl}')

        # ── Solve DARE ────────────────────────────────────────────────────────
        Q = args.q_scale * np.eye(d)
        R = np.array([[args.r_scale]])
        print(f'[{label}] solving DARE …')
        try:
            K = solve_dare(A_z, B_z, Q, R)
        except Exception as e:
            print(f'[{label}] DARE failed: {e}')
            results['models'].append({
                'label': label, 'dare_failed': True, 'dare_error': str(e),
                'rho_Az': rho_Az, 'n_unstable': n_unstable, 'fp_err': fp_err,
            })
            continue
        print(f'[{label}] K computed  ‖K‖={np.linalg.norm(K):.4f}')

        # Closed-loop spectral radius check
        rho_cl = float(np.max(np.abs(np.linalg.eigvals(A_z - B_z @ K))))
        cl_status = 'stable' if rho_cl < 1 else 'UNSTABLE'
        print(f'[{label}] rho(A_z - B_z K) = {rho_cl:.6f}  ({cl_status})')

        # ── Encode goal ───────────────────────────────────────────────────────
        z_goal_t = encode_rendered_state(bundle, np.zeros(4, dtype=np.float32))
        z_goal = z_goal_t.detach().cpu().numpy().flatten()

        # ── Run trials ────────────────────────────────────────────────────────
        trials = []
        n_success = 0
        for i, x0 in enumerate(initial_states):
            row = lqr_trial(
                bundle, K, z_goal, x0, args.n_steps,
                args.success_threshold, args.success_hold_steps,
                action_scale)
            n_success += int(row['success'])
            print(f'[{label}] trial {i:03d}  '
                  f'success={row["success"]}  held={row["held"]}  '
                  f'term={row["terminated"]}  '
                  f'final={row["final_error"]:.5f}')
            trials.append(row)

        sr = n_success / max(len(trials), 1)
        frac_stable = float(np.mean([r['fraction_stable'] for r in trials]))
        print(f'[{label}] success rate: {sr:.1%}  '
              f'mean fraction stable: {frac_stable:.3f}')

        results['models'].append({
            'label': label,
            'success_rate': sr,
            'mean_fraction_stable': frac_stable,
            'rho_Az': rho_Az,
            'rho_closed_loop': rho_cl,
            'n_unstable': n_unstable,
            'fp_err': fp_err,
            'pbh': [{'eigenvalue': str(x['eigenvalue']),
                     'controllable': x['controllable']} for x in pbh],
            'dare_failed': False,
            'trials': trials,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()
