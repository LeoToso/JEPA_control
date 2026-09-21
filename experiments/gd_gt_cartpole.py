#!/usr/bin/env python
"""Gradient-based planning with GROUND-TRUTH CartPole dynamics (oracle baseline).

The CartPole physics are reimplemented in PyTorch so that exact gradients
flow from the cost back through the simulated trajectory — no SPSA or
finite-difference approximation needed.

Optimisation loop (per replan):
  1. Roll out action sequence u through the differentiable physics
  2. Compute cost = ||state_H||²  (objective='last')
           or    = mean_t ||state_t||²  (objective='all')
  3. Adam step on u
  4. Inject Gaussian noise: u ← u + N(0, σ²)
  5. Execute first executed_steps actions, reset, replan.

Trial protocol matches compare_lqr_smwm.py exactly (same seed, same
eps-range initial states) for fair oracle comparison.

Usage
-----
  python experiments/gd_gt_cartpole.py \\
      --trials 10 --n-steps 300 \\
      --planning-horizon 10 --executed-steps 1 \\
      --gd-steps 50 --lr 0.1 --action-noise 0.05 \\
      --success-threshold 0.7 --success-hold-steps 10 \\
      --output results/gd_gt_cartpole.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

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


# ── differentiable CartPole physics (PyTorch) ─────────────────────────────────

def cartpole_step_torch(state, force):
    """One macro-step (FRAME_SKIP physics sub-steps) in differentiable PyTorch.

    state : (..., 4) tensor  [x, x_dot, theta, theta_dot]
    force : (...,)   tensor  raw action (clipped internally)
    Returns next state (..., 4).  No done check — gradients flow through.
    """
    M, m, l, g = MASS_CART, MASS_POLE, POLE_LENGTH, GRAVITY
    f = force.clamp(ACTION_LOW, ACTION_HIGH)
    for _ in range(FRAME_SKIP):
        x         = state[..., 0]
        x_dot     = state[..., 1]
        theta     = state[..., 2]
        theta_dot = state[..., 3]
        sin_t     = torch.sin(theta)
        cos_t     = torch.cos(theta)
        total_mass = M + m
        ml         = m * l
        temp       = (f + ml * theta_dot ** 2 * sin_t) / total_mass
        theta_acc  = (g * sin_t - cos_t * temp) / (
                         l * (4.0 / 3.0 - m * cos_t ** 2 / total_mass))
        x_acc      = temp - ml * theta_acc * cos_t / total_mass
        new_x         = x         + DT * x_dot
        new_x_dot     = x_dot     + DT * x_acc
        new_theta     = theta     + DT * theta_dot
        new_theta_dot = theta_dot + DT * theta_acc
        state = torch.stack([new_x, new_x_dot, new_theta, new_theta_dot], dim=-1)
    return state


# ── numpy CartPole step (for actual env execution, no grad needed) ────────────

def cartpole_step_np(state, force):
    """Pure-numpy macro-step for the execution loop (no rendering needed)."""
    x, x_dot, theta, theta_dot = np.asarray(state, dtype=np.float64)
    force = float(np.clip(force, ACTION_LOW, ACTION_HIGH))
    done  = False
    M, m, l, g = MASS_CART, MASS_POLE, POLE_LENGTH, GRAVITY
    for _ in range(FRAME_SKIP):
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        total_mass = M + m
        ml = m * l
        temp      = (force + ml * theta_dot ** 2 * sin_t) / total_mass
        theta_acc = (g * sin_t - cos_t * temp) / (
                        l * (4.0 / 3.0 - m * cos_t ** 2 / total_mass))
        x_acc     = temp - ml * theta_acc * cos_t / total_mass
        x         += DT * x_dot
        x_dot     += DT * x_acc
        theta     += DT * theta_dot
        theta_dot += DT * theta_acc
        if abs(x) > 2.4 or abs(theta) > THETA_THRESHOLD:
            done = True
            break
    return np.array([x, x_dot, theta, theta_dot], dtype=np.float64), done


# ── GD planner ─────────────────────────────────────────────────────────────────

class CartPoleGTGDPlanner:
    """Exact-gradient GD planner through differentiable CartPole physics."""

    def __init__(self, horizon, executed_steps, gd_steps, lr, action_noise,
                 objective, warm_start, n_restarts, device):
        self.horizon        = int(horizon)
        self.executed_steps = min(int(executed_steps), self.horizon)
        self.gd_steps       = int(gd_steps)
        self.lr             = float(lr)
        self.action_noise   = float(action_noise)
        self.objective      = objective
        self.warm_start     = warm_start
        self.n_restarts     = max(1, int(n_restarts))
        self.device         = device
        self._prev_u        = None

    def _run_one(self, start_state_t, init_u):
        """GD from one initialisation; return (u_clamped, terminal_cost)."""
        u         = init_u.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([u], lr=self.lr)

        for _ in range(self.gd_steps):
            optimizer.zero_grad()
            state = start_state_t.clone()
            if self.objective == 'all':
                costs = []
            for t in range(self.horizon):
                state = cartpole_step_torch(state, u[t])
                if self.objective == 'all':
                    costs.append(state.pow(2).sum())
            if self.objective == 'last':
                loss = state.pow(2).sum()
            else:
                loss = torch.stack(costs).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                u += torch.randn_like(u) * self.action_noise

        with torch.no_grad():
            u_clamped = u.clamp(ACTION_LOW, ACTION_HIGH)
            # Evaluate terminal cost for restart selection
            state = start_state_t.clone()
            for t in range(self.horizon):
                state = cartpole_step_torch(state, u_clamped[t])
            cost = float(state.pow(2).sum().cpu())
        return u_clamped.detach(), cost

    def plan(self, start_state):
        """Return (executed_steps,) numpy action array."""
        s_t = torch.as_tensor(start_state, dtype=torch.float64,
                              device=self.device)

        inits = []
        if self.warm_start and self._prev_u is not None:
            shifted = torch.cat([
                self._prev_u[self.executed_steps:],
                torch.zeros(self.executed_steps, dtype=torch.float64,
                            device=self.device)
            ])
            inits.append(shifted)
        for _ in range(self.n_restarts - len(inits)):
            inits.append(torch.randn(self.horizon, dtype=torch.float64,
                                     device=self.device) * 2.0)

        best_u, best_cost = None, float('inf')
        for init_u in inits:
            u_clamped, cost = self._run_one(s_t, init_u)
            if cost < best_cost:
                best_cost = cost
                best_u    = u_clamped

        self._prev_u = best_u
        return best_u[:self.executed_steps].cpu().numpy()


# ── trial evaluation ──────────────────────────────────────────────────────────

def gd_gt_trial(planner, initial_state, n_steps,
                success_threshold, success_hold_steps):
    state  = np.asarray(initial_state, dtype=np.float64)
    states = [state.copy()]
    step   = 0
    consec = 0
    planner._prev_u = None

    while step < n_steps:
        sequence = planner.plan(state)

        for a in sequence:
            if step >= n_steps:
                break
            state, done = cartpole_step_np(state, float(a))
            states.append(state.copy())
            step += 1

            err = float(np.linalg.norm(state))
            if err < success_threshold:
                consec += 1
                if consec >= success_hold_steps:
                    break
            else:
                consec = 0
            if done:
                break

        if consec >= success_hold_steps or done:
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
        description='GT oracle GBP for CartPole — exact gradients via differentiable physics.')
    p.add_argument('--trials',             type=int,   default=10)
    p.add_argument('--n-steps',            type=int,   default=300)
    p.add_argument('--planning-horizon',   type=int,   default=10)
    p.add_argument('--executed-steps',     type=int,   default=1)
    p.add_argument('--gd-steps',           type=int,   default=50)
    p.add_argument('--lr',                 type=float, default=0.1)
    p.add_argument('--action-noise',       type=float, default=0.05)
    p.add_argument('--objective',          choices=['last', 'all'], default='last')
    p.add_argument('--n-restarts',         type=int,   default=1)
    p.add_argument('--no-warm-start',      action='store_true')
    p.add_argument('--success-threshold',  type=float, default=0.7)
    p.add_argument('--success-hold-steps', type=int,   default=10)
    p.add_argument('--eps-range',          type=float, nargs=2, default=[0.01, 0.15],
                   help='Uniform perturbation range for initial states')
    p.add_argument('--seed',               type=int,   default=42)
    p.add_argument('--device',             default='cpu')
    p.add_argument('--output',             required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)

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

    planner = CartPoleGTGDPlanner(
        horizon        = args.planning_horizon,
        executed_steps = args.executed_steps,
        gd_steps       = args.gd_steps,
        lr             = args.lr,
        action_noise   = args.action_noise,
        objective      = args.objective,
        warm_start     = not args.no_warm_start,
        n_restarts     = args.n_restarts,
        device         = device,
    )
    print(f'[GT-GBP CartPole] exact backprop through differentiable physics')
    print(f'[GT-GBP CartPole] H={planner.horizon} K={planner.executed_steps} '
          f'gd_steps={planner.gd_steps} lr={planner.lr} '
          f'noise={planner.action_noise} objective={planner.objective} '
          f'restarts={planner.n_restarts} frame_skip={FRAME_SKIP}')

    trials, n_success = [], 0
    for i, x0 in enumerate(initial_states):
        torch.manual_seed(args.seed + i)
        print(f'[trial {i:03d}] start=[{x0[0]:.3f},{x0[1]:.3f},{x0[2]:.3f},{x0[3]:.3f}]  ',
              end='', flush=True)
        t0  = time.time()
        row = gd_gt_trial(planner, x0, args.n_steps,
                          args.success_threshold, args.success_hold_steps)
        elapsed = time.time() - t0
        n_success += int(row['success'])
        trials.append(row)
        print(f'success={row["success"]}  held={row["held"]}  '
              f'final={row["final_error"]:.4f}  t={elapsed:.1f}s')

    sr = n_success / max(len(trials), 1)
    print(f'\n[GT-GBP CartPole] success rate: {sr:.1%}  ({n_success}/{len(trials)})')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planner':            'gt_exact_grad',
            'gradient_method':    'autograd_through_pytorch_physics',
            'n_steps':            args.n_steps,
            'planning_horizon':   args.planning_horizon,
            'executed_steps':     args.executed_steps,
            'gd_steps':           args.gd_steps,
            'lr':                 args.lr,
            'action_noise':       args.action_noise,
            'objective':          args.objective,
            'n_restarts':         args.n_restarts,
            'warm_start':         not args.no_warm_start,
            'success_threshold':  args.success_threshold,
            'success_hold_steps': args.success_hold_steps,
            'seed':               args.seed,
            'eps_range':          args.eps_range,
            'n_trials':           args.trials,
            'frame_skip':         FRAME_SKIP,
        },
        'success_rate': sr,
        'trials':       trials,
    }, indent=2))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
