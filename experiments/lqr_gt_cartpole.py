#!/usr/bin/env python
"""LQR with GROUND-TRUTH CartPole dynamics (oracle baseline).

Numerically linearises the true CartPole physics at the upright equilibrium
via central finite differences, solves the discrete LQR (DARE), and runs
closed-loop control.

Trial protocol matches compare_lqr_smwm.py exactly (same seed, same
eps-range) for a fair oracle comparison.

Usage
-----
  python experiments/lqr_gt_cartpole.py \\
      --trials 10 --n-steps 300 \\
      --success-threshold 0.7 --success-hold-steps 10 \\
      --q-scale 1.0 --r-scale 1.0 \\
      --output results/lqr_gt_cartpole.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.linalg

# CartPole physical constants (must match ContinuousCartpoleVisual defaults)
MASS_CART       = 1.0
MASS_POLE       = 0.1
POLE_LENGTH     = 0.5
GRAVITY         = 9.8
DT              = 0.02
FRAME_SKIP      = 5
ACTION_LOW      = -10.0
ACTION_HIGH     =  10.0
THETA_THRESHOLD = 12 * 2 * math.pi / 360   # ~0.2094 rad


# ── pure-numpy CartPole physics (no rendering) ────────────────────────────────

def cartpole_step(state, force, frame_skip=FRAME_SKIP):
    """Simulate frame_skip physics steps; return (next_state, done)."""
    x, x_dot, theta, theta_dot = np.asarray(state, dtype=np.float64)
    force = float(np.clip(force, ACTION_LOW, ACTION_HIGH))
    done  = False
    for _ in range(frame_skip):
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        M, m, l, g   = MASS_CART, MASS_POLE, POLE_LENGTH, GRAVITY
        total_mass    = M + m
        ml            = m * l
        temp          = (force + ml * theta_dot ** 2 * sin_t) / total_mass
        theta_acc     = (g * sin_t - cos_t * temp) / (
                            l * (4.0 / 3.0 - m * cos_t ** 2 / total_mass))
        x_acc         = temp - ml * theta_acc * cos_t / total_mass
        x         += DT * x_dot
        x_dot     += DT * x_acc
        theta     += DT * theta_dot
        theta_dot += DT * theta_acc
        if abs(x) > 2.4 or abs(theta) > THETA_THRESHOLD:
            done = True
            break
    return np.array([x, x_dot, theta, theta_dot], dtype=np.float64), done


# ── finite-difference Jacobians at equilibrium ────────────────────────────────

def jacobians_fd(eps_s=1e-4, eps_u=0.1):
    """Central FD Jacobians A (4×4) and B (4×1) at equilibrium."""
    x0 = np.zeros(4, dtype=np.float64)
    u0 = 0.0

    A = np.zeros((4, 4))
    for j in range(4):
        sp = x0.copy(); sp[j] += eps_s
        sm = x0.copy(); sm[j] -= eps_s
        fp, _ = cartpole_step(sp, u0)
        fm, _ = cartpole_step(sm, u0)
        A[:, j] = (fp - fm) / (2.0 * eps_s)

    B = np.zeros((4, 1))
    fp, _ = cartpole_step(x0, u0 + eps_u)
    fm, _ = cartpole_step(x0, u0 - eps_u)
    B[:, 0] = (fp - fm) / (2.0 * eps_u)

    return A, B


def solve_dare(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K   # (1, 4)


# ── trial evaluation ──────────────────────────────────────────────────────────

def lqr_trial(K, initial_state, n_steps, success_threshold, success_hold_steps):
    state  = np.asarray(initial_state, dtype=np.float64)
    states = [state.copy()]
    consec = 0

    for _ in range(n_steps):
        u     = float(np.clip(-(K @ state).item(), ACTION_LOW, ACTION_HIGH))
        state, done = cartpole_step(state, u)
        states.append(state.copy())

        err = float(np.linalg.norm(state))
        if err < success_threshold:
            consec += 1
            if consec >= success_hold_steps:
                break
        else:
            consec = 0
        if done:
            break

    states  = np.array(states)
    errors  = np.linalg.norm(states, axis=1)
    stable  = errors < success_threshold
    tail    = stable[-success_hold_steps:]

    return {
        'success':         bool(stable[-1]),
        'held':            bool(len(tail) == success_hold_steps and tail.all()),
        'final_error':     float(errors[-1]),
        'max_error':       float(np.max(errors)),
        'fraction_stable': float(np.mean(stable[1:])) if len(stable) > 1 else 0.,
        'start_state':     initial_state.tolist(),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='GT oracle LQR for CartPole — matches compare_lqr_smwm.py protocol.')
    p.add_argument('--trials',             type=int,   default=10)
    p.add_argument('--n-steps',            type=int,   default=300)
    p.add_argument('--success-threshold',  type=float, default=0.7)
    p.add_argument('--success-hold-steps', type=int,   default=10)
    p.add_argument('--q-scale',            type=float, default=1.0)
    p.add_argument('--r-scale',            type=float, default=1.0)
    p.add_argument('--eps-state',          type=float, default=1e-4)
    p.add_argument('--eps-action',         type=float, default=0.1)
    p.add_argument('--eps-range',          type=float, nargs=2, default=[0.01, 0.15],
                   help='Uniform perturbation range for initial states')
    p.add_argument('--seed',               type=int,   default=42)
    p.add_argument('--output',             required=True)
    args = p.parse_args()

    # ── Linearise at equilibrium and solve LQR ────────────────────────────────
    A, B = jacobians_fd(args.eps_state, args.eps_action)
    Q    = args.q_scale * np.eye(4)
    R    = args.r_scale * np.eye(1)
    K    = solve_dare(A, B, Q, R)
    eigs_ol = np.linalg.eigvals(A)
    eigs_cl = np.linalg.eigvals(A - B @ K)
    print(f'[GT-LQR CartPole] A eigs: {np.abs(eigs_ol).round(4)}')
    print(f'[GT-LQR CartPole] closed-loop eigs: {np.abs(eigs_cl).round(4)}')
    print(f'[GT-LQR CartPole] K = {K.round(4)}')

    # ── Initial states — identical to compare_lqr_smwm.py ────────────────────
    rng = np.random.default_rng(args.seed)
    lo, hi = args.eps_range
    initial_states = [
        rng.uniform(-hi, hi, size=4).clip(-hi, hi).astype(np.float64)
        for _ in range(args.trials)
    ]
    for s in initial_states:
        s[2] = rng.uniform(-hi, hi)   # theta
        s[3] = rng.uniform(-hi, hi)   # theta_dot

    trials, n_success = [], 0
    for i, x0 in enumerate(initial_states):
        row = lqr_trial(K, x0, args.n_steps, args.success_threshold,
                        args.success_hold_steps)
        n_success += int(row['success'])
        trials.append(row)
        print(f'[trial {i:03d}] success={row["success"]}  held={row["held"]}  '
              f'final={row["final_error"]:.4f}')

    sr = n_success / max(len(trials), 1)
    print(f'\n[GT-LQR CartPole] success rate: {sr:.1%}  ({n_success}/{len(trials)})')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planner':           'gt_lqr',
            'n_steps':           args.n_steps,
            'success_threshold': args.success_threshold,
            'success_hold_steps': args.success_hold_steps,
            'q_scale':           args.q_scale,
            'r_scale':           args.r_scale,
            'seed':              args.seed,
            'eps_range':         args.eps_range,
            'n_trials':          args.trials,
            'frame_skip':        FRAME_SKIP,
        },
        'K':            K.tolist(),
        'success_rate': sr,
        'trials':       trials,
    }, indent=2))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
