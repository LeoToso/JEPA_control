"""CEM stabilization using ground-truth physical dynamics.

Runs the CEM planner directly in the physical state space:
  - Linear mode:    uses A_star, B_star from the analytical linearization
  - Nonlinear mode: rolls out exact CartPole physics inside CEM

This is the definitive sanity check: if CEM stabilizes with GT dynamics,
the JEPA model (not the planner) is the bottleneck.

Usage:
    python experiments/cem_gt_stabilize.py
    python experiments/cem_gt_stabilize.py --mode nonlinear --horizon 10
    python experiments/cem_gt_stabilize.py --mode linear --horizon 5 --n-trials 50
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn


# ── Thin wrapper: GT physics as a PyTorch "predictor" ────────────────────────

class GTPhysicsPredictor(nn.Module):
    """Wraps exact CartPole physics so CEM nonlinear mode can call it.

    CEM calls predictor(z_win, u_norm) where:
      z_win : (N, W, d) — last W states; we use only the last one
      u_norm: (N, W, 1) — last W actions; we use only the last one (u_norm * action_scale = raw N)
    Returns z_next: (N, d)
    """
    def __init__(self, mass_cart, mass_pole, pole_length, gravity, dt, frame_skip,
                 action_scale=10.0):
        super().__init__()
        self.M = float(mass_cart)
        self.m = float(mass_pole)
        self.l = float(pole_length)
        self.g = float(gravity)
        self.dt = float(dt)
        self.frame_skip = int(frame_skip)
        self.action_scale = float(action_scale)
        self.latent_dim = 4   # physical state dim

    def _physics_step(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """One dt-step of exact CartPole dynamics.  state: (N,4)  u: (N,)"""
        M, m, l, g = self.M, self.m, self.l, self.g
        x, xdot, theta, thetadot = state.T  # each (N,)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        total_mass = M + m
        ml = m * l

        temp      = (u + ml * thetadot**2 * sin_t) / total_mass
        theta_acc = (g * sin_t - cos_t * temp) / (l * (4.0/3.0 - m * cos_t**2 / total_mass))
        x_acc     = temp - ml * theta_acc * cos_t / total_mass

        x_new        = x        + self.dt * xdot
        xdot_new     = xdot     + self.dt * x_acc
        theta_new    = theta    + self.dt * thetadot
        thetadot_new = thetadot + self.dt * theta_acc
        return torch.stack([x_new, xdot_new, theta_new, thetadot_new], dim=1)

    def predict(self, z_win: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        """CEM interface.  z_win: (N, W, 4)  u_norm: (N, W, 1) in [-1, 1] units."""
        state = z_win[:, -1, :]          # (N, 4) — use last window entry
        u_raw = u_norm[:, -1, 0] * self.action_scale   # (N,) — denormalize
        for _ in range(self.frame_skip):
            state = self._physics_step(state, u_raw)
        return state                      # (N, 4)

    def forward(self, z_win, u_norm):
        return self.predict(z_win, u_norm)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode',        choices=['linear', 'nonlinear'], default='nonlinear',
                   help='linear=GT linearization; nonlinear=exact physics in CEM')
    p.add_argument('--horizon',     type=int,   default=10)
    p.add_argument('--n-samples',   type=int,   default=500)
    p.add_argument('--n-elites',    type=int,   default=50)
    p.add_argument('--n-iter',      type=int,   default=5)
    p.add_argument('--init-std',    type=float, default=3.0)
    p.add_argument('--n-trials',    type=int,   default=30,
                   help='Number of stabilization episodes')
    p.add_argument('--T-rollout',   type=int,   default=200,
                   help='Max steps per episode (in frame-skip steps)')
    p.add_argument('--init-scale',  type=float, default=0.15,
                   help='Max initial state perturbation')
    p.add_argument('--stable-thr',  type=float, default=0.1,
                   help='|theta| threshold to count as stabilized')
    p.add_argument('--settle-steps', type=int,  default=50,
                   help='Consecutive steps within stable-thr to declare success')
    p.add_argument('--seed',        type=int,   default=0)
    p.add_argument('--device',      default=None)
    # Physical params
    p.add_argument('--mass-cart',   type=float, default=1.0)
    p.add_argument('--mass-pole',   type=float, default=0.1)
    p.add_argument('--pole-length', type=float, default=0.5)
    p.add_argument('--gravity',     type=float, default=9.8)
    p.add_argument('--dt',          type=float, default=0.02)
    p.add_argument('--frame-skip',  type=int,   default=5)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    rng = np.random.RandomState(args.seed)
    dt_eff = args.dt * args.frame_skip   # effective dt per step

    from ground_truth.cartpole_gt import CartpoleGroundTruth
    from control.cem import CEMLatentPlanner

    gt = CartpoleGroundTruth(
        mass_cart=args.mass_cart, mass_pole=args.mass_pole,
        pole_length=args.pole_length, gravity=args.gravity,
        dt=dt_eff,
    )
    print(f'[GT] A_star shape={gt.A_star.shape}  rho(A*)={gt.spectral_radius:.4f}')
    print(f'     unstable eigs: {np.round(gt.unstable_eigenvalues.real, 4)}')

    # Cost: penalize theta strongly, thetadot moderately, x/xdot lightly
    Q   = np.diag([1.0, 1.0, 100.0, 10.0])
    Q_f = 10.0 * Q
    R   = 0.01

    if args.mode == 'linear':
        print(f'\n[planner] Linear CEM on GT linearization  H={args.horizon}')
        planner = CEMLatentPlanner(
            A=gt.A_star, B=gt.B_star, c_offset=None,
            Q=Q, R=R, Q_f=Q_f,
            horizon=args.horizon, chunk_size=1,
            n_samples=args.n_samples, n_elites=args.n_elites,
            n_iter=args.n_iter, init_std=args.init_std,
            action_lb=-10.0, action_ub=10.0, action_scale=1.0,
            device=device,
        )
    else:
        print(f'\n[planner] Nonlinear CEM on exact GT physics  H={args.horizon}')
        physics = GTPhysicsPredictor(
            mass_cart=args.mass_cart, mass_pole=args.mass_pole,
            pole_length=args.pole_length, gravity=args.gravity,
            dt=args.dt, frame_skip=args.frame_skip, action_scale=10.0,
        ).to(device)
        planner = CEMLatentPlanner(
            predictor=physics, action_encoder=None,
            predictor_window=2,         # W=2 uses windowed path (predict()); W=1 path needs action_encoder
            Q=Q, R=R, Q_f=Q_f,
            horizon=args.horizon, chunk_size=1,
            n_samples=args.n_samples, n_elites=args.n_elites,
            n_iter=args.n_iter, init_std=args.init_std,
            action_lb=-10.0, action_ub=10.0, action_scale=10.0,
            device=device,
        )

    z_star = np.zeros(4, dtype=np.float32)   # upright equilibrium

    # ── Episode loop ──────────────────────────────────────────────────────────
    successes    = 0
    ep_lengths   = []
    frac_stables = []
    t0 = time.time()

    print(f'\n[eval] {args.n_trials} trials  T={args.T_rollout}  '
          f'init_scale={args.init_scale}  stable_thr={args.stable_thr}  '
          f'settle_steps={args.settle_steps}')

    for trial in range(args.n_trials):
        x0 = rng.uniform(-args.init_scale, args.init_scale, 4).astype(np.float32)
        state = x0.copy()
        planner.reset()

        stable_count = 0
        success      = False
        n_stable     = 0

        for t in range(args.T_rollout):
            actions, _ = planner.plan(state, z_star)
            u = float(np.clip(actions[0][0], -10.0, 10.0))

            # Step exact physics (frame_skip steps)
            for _ in range(args.frame_skip):
                x, xdot, theta, thetadot = state
                cos_t = np.cos(theta); sin_t = np.sin(theta)
                total_mass = args.mass_cart + args.mass_pole
                ml = args.mass_pole * args.pole_length
                temp      = (u + ml * thetadot**2 * sin_t) / total_mass
                theta_acc = (args.gravity * sin_t - cos_t * temp) / \
                            (args.pole_length * (4/3 - args.mass_pole * cos_t**2 / total_mass))
                x_acc     = temp - ml * theta_acc * cos_t / total_mass
                state = np.array([
                    x    + args.dt * xdot,
                    xdot + args.dt * x_acc,
                    theta    + args.dt * thetadot,
                    thetadot + args.dt * theta_acc,
                ], dtype=np.float32)

            if abs(state[2]) < args.stable_thr:
                stable_count += 1
                n_stable += 1
                if stable_count >= args.settle_steps:
                    success = True
                    ep_lengths.append(t + 1)
                    break
            else:
                stable_count = 0

        if not success:
            ep_lengths.append(args.T_rollout)
        frac_stables.append(n_stable / args.T_rollout)
        successes += int(success)

        if (trial + 1) % 10 == 0:
            print(f'  trial {trial+1:3d}/{args.n_trials}  '
                  f'success so far: {successes}/{trial+1}  '
                  f'({time.time()-t0:.1f}s)')

    # ── Report ────────────────────────────────────────────────────────────────
    success_rate  = successes / args.n_trials
    mean_ep_len   = np.mean(ep_lengths)
    mean_stable   = np.mean(frac_stables)

    print(f'\n{"="*60}')
    print(f'CEM-GT  mode={args.mode}  H={args.horizon}')
    print(f'{"="*60}')
    print(f'  success_rate    = {success_rate:.3f}  ({successes}/{args.n_trials})')
    print(f'  mean_ep_length  = {mean_ep_len:.1f}')
    print(f'  mean_frac_stable= {mean_stable:.3f}')
    print(f'  total_time      = {time.time()-t0:.1f}s')
    print(f'{"="*60}')

    if success_rate == 0.0:
        print('\n[!] CEM failed even with GT dynamics — check planner settings.')
    elif success_rate < 0.5:
        print('\n[~] Partial success — tune H, Q weights, or init_std.')
    else:
        print('\n[✓] CEM works with GT dynamics — JEPA model is the bottleneck.')


if __name__ == '__main__':
    main()
