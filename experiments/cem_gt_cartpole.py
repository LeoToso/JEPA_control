#!/usr/bin/env python
"""CEM with GROUND-TRUTH CartPole dynamics (oracle baseline).

Rolls out action sequences through the true CartPole physics instead of
a learned world model. Same CEM hyperparameters and trial protocol as
compare_paper_cem_smwm.py.

Usage
-----
  python experiments/cem_gt_cartpole.py \\
      --trials 10 --n-steps 300 \\
      --planning-horizon 10 --executed-steps 1 \\
      --cem-population 300 --cem-elites 30 --cem-iters 10 \\
      --success-threshold 0.7 --success-hold-steps 10 \\
      --output results/cem_gt_cartpole.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# CartPole physical constants (match ContinuousCartpoleVisual defaults)
MASS_CART       = 1.0
MASS_POLE       = 0.1
POLE_LENGTH     = 0.5
GRAVITY         = 9.8
DT              = 0.02
FRAME_SKIP      = 5
ACTION_LOW      = -10.0
ACTION_HIGH     =  10.0
THETA_THRESHOLD = 12 * 2 * math.pi / 360


# ── vectorised pure-numpy CartPole physics ────────────────────────────────────

def cartpole_step_batch(states, forces, frame_skip=FRAME_SKIP):
    """Simulate frame_skip physics steps for a batch of (state, force) pairs.

    states : (N, 4)  float64
    forces : (N,)    float64
    Returns (next_states, dones) both (N, ...).
    """
    s    = np.asarray(states, dtype=np.float64).copy()    # (N, 4)
    f    = np.clip(np.asarray(forces, dtype=np.float64),
                   ACTION_LOW, ACTION_HIGH)                # (N,)
    done = np.zeros(len(s), dtype=bool)

    M, m, l, g = MASS_CART, MASS_POLE, POLE_LENGTH, GRAVITY
    for _ in range(frame_skip):
        active = ~done
        if not active.any():
            break
        x, x_dot, theta, theta_dot = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        total_mass = M + m
        ml         = m * l
        temp       = (f + ml * theta_dot ** 2 * sin_t) / total_mass
        theta_acc  = (g * sin_t - cos_t * temp) / (
                         l * (4.0 / 3.0 - m * cos_t ** 2 / total_mass))
        x_acc      = temp - ml * theta_acc * cos_t / total_mass
        s[active, 0] += DT * x_dot[active]
        s[active, 1] += DT * x_acc[active]
        s[active, 2] += DT * theta_dot[active]
        s[active, 3] += DT * theta_acc[active]
        done |= (np.abs(s[:, 0]) > 2.4) | (np.abs(s[:, 2]) > THETA_THRESHOLD)
    return s, done


def cartpole_step(state, force, frame_skip=FRAME_SKIP):
    """Single-sample step (convenience wrapper)."""
    s, d = cartpole_step_batch(
        np.asarray(state)[None], np.array([force]), frame_skip)
    return s[0], bool(d[0])


# ── GT CEM planner ────────────────────────────────────────────────────────────

class CartPoleGTCEM:
    """CEM planner using true CartPole physics for rollouts."""

    def __init__(self, horizon, executed_steps,
                 population, elites, iterations, initial_variance):
        self.horizon          = int(horizon)
        self.executed_steps   = min(int(executed_steps), self.horizon)
        self.population       = int(population)
        self.elites           = min(int(elites), self.population)
        self.iterations       = int(iterations)
        self.initial_variance = float(initial_variance)

    def _rollout_costs(self, start_state, actions):
        """Roll out population of action sequences; return terminal ||state||² costs.

        actions : (pop, H)
        Returns costs (pop,).
        """
        pop = actions.shape[0]
        states = np.tile(start_state, (pop, 1)).astype(np.float64)  # (pop, 4)
        done   = np.zeros(pop, dtype=bool)
        for t in range(self.horizon):
            active_states  = states.copy()
            active_forces  = actions[:, t]
            next_s, new_d  = cartpole_step_batch(active_states, active_forces)
            states[~done]  = next_s[~done]
            done           |= new_d
        return np.sum(states ** 2, axis=1)   # ||state_H||²

    def plan(self, start_state):
        """Return (executed_steps,) action array."""
        mean     = np.zeros(self.horizon)
        variance = np.full(self.horizon, self.initial_variance)

        for _ in range(self.iterations):
            noise   = np.random.randn(self.population, self.horizon)
            actions = mean[None] + np.sqrt(variance)[None] * noise
            actions = np.clip(actions, ACTION_LOW, ACTION_HIGH)
            costs   = self._rollout_costs(start_state, actions)
            elite_idx = np.argsort(costs)[:self.elites]
            elite     = actions[elite_idx]
            mean      = elite.mean(0)
            variance  = elite.var(0).clip(min=1e-6)

        return np.clip(mean[:self.executed_steps], ACTION_LOW, ACTION_HIGH)


# ── trial evaluation ──────────────────────────────────────────────────────────

def cem_gt_trial(planner, initial_state, n_steps,
                 success_threshold, success_hold_steps):
    state  = np.asarray(initial_state, dtype=np.float64)
    states = [state.copy()]
    step   = 0
    consec = 0

    while step < n_steps:
        sequence = planner.plan(state)

        for a in sequence:
            if step >= n_steps:
                break
            state, done = cartpole_step(state, float(a))
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
        description='GT oracle CEM for CartPole — matches compare_paper_cem_smwm.py protocol.')
    p.add_argument('--trials',             type=int,   default=10)
    p.add_argument('--n-steps',            type=int,   default=300)
    p.add_argument('--planning-horizon',   type=int,   default=10)
    p.add_argument('--executed-steps',     type=int,   default=1)
    p.add_argument('--cem-population',     type=int,   default=300)
    p.add_argument('--cem-elites',         type=int,   default=30)
    p.add_argument('--cem-iters',          type=int,   default=10)
    p.add_argument('--cem-initial-variance', type=float, default=1.0)
    p.add_argument('--success-threshold',  type=float, default=0.7)
    p.add_argument('--success-hold-steps', type=int,   default=10)
    p.add_argument('--eps-range',          type=float, nargs=2, default=[0.01, 0.15],
                   help='Uniform perturbation range for initial states')
    p.add_argument('--seed',               type=int,   default=42)
    p.add_argument('--output',             required=True)
    args = p.parse_args()

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

    planner = CartPoleGTCEM(
        horizon          = args.planning_horizon,
        executed_steps   = args.executed_steps,
        population       = args.cem_population,
        elites           = args.cem_elites,
        iterations       = args.cem_iters,
        initial_variance = args.cem_initial_variance,
    )
    print(f'[GT-CEM CartPole] H={planner.horizon} K={planner.executed_steps} '
          f'pop={planner.population} elites={planner.elites} '
          f'iters={planner.iterations}')

    trials, n_success = [], 0
    for i, x0 in enumerate(initial_states):
        np.random.seed(args.seed + i)
        row = cem_gt_trial(planner, x0, args.n_steps,
                           args.success_threshold, args.success_hold_steps)
        n_success += int(row['success'])
        trials.append(row)
        print(f'[trial {i:03d}] success={row["success"]}  held={row["held"]}  '
              f'final={row["final_error"]:.4f}')

    sr = n_success / max(len(trials), 1)
    print(f'\n[GT-CEM CartPole] success rate: {sr:.1%}  ({n_success}/{len(trials)})')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planner':           'gt_cem',
            'n_steps':           args.n_steps,
            'planning_horizon':  args.planning_horizon,
            'executed_steps':    args.executed_steps,
            'cem_population':    args.cem_population,
            'cem_elites':        args.cem_elites,
            'cem_iters':         args.cem_iters,
            'cem_initial_variance': args.cem_initial_variance,
            'success_threshold': args.success_threshold,
            'success_hold_steps': args.success_hold_steps,
            'seed':              args.seed,
            'eps_range':         args.eps_range,
            'n_trials':          args.trials,
            'frame_skip':        FRAME_SKIP,
        },
        'success_rate': sr,
        'trials':       trials,
    }, indent=2))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()

