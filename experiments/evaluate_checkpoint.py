"""Load a saved (encoder, predictor, decoder) checkpoint, design the LQR
controller its learned latent dynamics imply, and evaluate it in closed loop
on the TRUE system -- with full control over the evaluation protocol.

    python experiments/evaluate_checkpoint.py \\
        --checkpoint results/example2_cartpole_paper_naive/checkpoint_L_pred_L_act.pt \\
        --success-threshold 0.5 --n-steps 100 --n-trials 30 --hold-steps 20

The checkpoint's `extra` metadata records which observation model (sensing
matrix + noise) was used at training time; older checkpoints saved before
that was added fall back to this example's known defaults (see
checkpoint_io._OBS_DEFAULTS), with a printed warning so it's never silent.

By default every trial starts from a random initial state ~ N(0, x0_std^2 I).
Pass --x0 (or, for cartpole_linear, the shorthand --pole-angle0) to pin every
trial to the same, exact starting state instead -- trials will still differ
because the observation model's measurement noise is resampled per trial.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from checkpoint_io import load_checkpoint_with_env
from jepa_lds.control import design_latent_controller, evaluate_controller


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--success-threshold", type=float, default=0.5, help="||x_t|| must stay below this to count as 'stable'")
    p.add_argument("--n-steps", type=int, default=100, help="length of each closed-loop rollout")
    p.add_argument("--n-trials", type=int, default=30, help="number of random initial conditions to test")
    p.add_argument("--hold-steps", type=int, default=20, help="state must stay below threshold for the final N steps to count as a success")
    p.add_argument("--x0-std", type=float, default=0.03, help="std of the random initial state (ignored if --x0/--pole-angle0 is given)")
    p.add_argument("--x0", type=float, nargs="+", default=None, help="exact initial state to use for every trial, e.g. --x0 0 0 0.1 0 for cartpole_linear (length must equal the system's state dimension)")
    p.add_argument("--pole-angle0", type=float, default=None, help="shorthand for cartpole_linear: pins the initial state to [0, 0, angle, 0] (cart centered/at rest, pole at this angle in radians, no angular velocity)")
    p.add_argument("--q-scale", type=float, default=10.0, help="LQR Q = q_scale * I in latent space")
    p.add_argument("--r-scale", type=float, default=1.0, help="LQR R = r_scale * I")
    p.add_argument("--seed", type=int, default=0, help="seed for the closed-loop evaluation trials")
    p.add_argument("--threads", type=int, default=4, help="torch.set_num_threads")
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    system, obs_model, encoder, predictor, _decoder, cfg, extra = load_checkpoint_with_env(args.checkpoint)

    print(f"checkpoint: {args.checkpoint}")
    print(f"  system={system.name}  config={extra.get('config_name', '?')}  trainer={extra.get('trainer', '?')}")
    print(
        f"  cfg: latent_dim={cfg.latent_dim} horizon={cfg.horizon} "
        f"lambda_pred_1step={cfg.lambda_pred_1step} lambda_pred_ms={cfg.lambda_pred_ms} "
        f"lambda_sigreg={cfg.lambda_sigreg} lambda_actrecon_1step={cfg.lambda_actrecon_1step} "
        f"lambda_actrecon_ms={cfg.lambda_actrecon_ms}"
    )

    if args.x0 is not None and args.pole_angle0 is not None:
        raise SystemExit("pass only one of --x0 or --pole-angle0")
    x0 = None
    if args.pole_angle0 is not None:
        if system.name != "cartpole_linear":
            raise SystemExit(
                f"--pole-angle0 assumes the cartpole_linear state layout "
                f"[cart_pos, cart_vel, pole_angle, pole_angular_vel]; system is {system.name!r} -- use --x0 instead"
            )
        x0 = np.array([0.0, 0.0, args.pole_angle0, 0.0])
    elif args.x0 is not None:
        if len(args.x0) != system.n:
            raise SystemExit(f"--x0 must have {system.n} values for system {system.name!r} (got {len(args.x0)})")
        x0 = np.array(args.x0, dtype=np.float64)
    if x0 is not None:
        print(f"\nx0 (fixed for every trial) = {x0}")
    else:
        print(f"\nx0 ~ N(0, {args.x0_std}^2 I), resampled per trial")

    ctrl = design_latent_controller(predictor, q_scale=args.q_scale, r_scale=args.r_scale)
    print(f"\nstabilizable (latent) = {ctrl['stabilizable']}")
    print(f"true eigvals(A)      = {system.eigvals()}")
    print(f"learned eigvals(A_z) = {np.linalg.eigvals(ctrl['A_z'])}")
    if not ctrl["stabilizable"]:
        print("\nNo stabilizing latent controller could be designed: an unstable latent "
              "eigenvalue is uncontrollable (PBH test failed). Nothing to evaluate.")
        return
    print(
        f"latent closed-loop spectral radius = {ctrl['latent_closed_loop_spectral_radius']:.4f}  "
        "(how close eig(A_z - B_z K_z) is to 1 -- closer to 1 means slower, more lightly damped "
        "settling, even when the trajectory is technically converging)"
    )

    ev = evaluate_controller(
        system, obs_model, encoder, ctrl["K_z"],
        n_trials=args.n_trials, n_steps=args.n_steps, x0_std=args.x0_std, x0=x0,
        success_threshold=args.success_threshold, hold_steps=args.hold_steps, seed=args.seed,
    )
    print(
        f"\nsuccess_rate={ev['success_rate'] * 100:.1f}%  "
        f"mean_fraction_stable={ev['mean_fraction_stable']:.3f}  "
        f"final_state_distance_avg={ev['final_state_distance_avg']:.4f}"
    )
    print(
        f"(evaluated with n_trials={args.n_trials}, n_steps={args.n_steps}, "
        f"success_threshold={args.success_threshold}, hold_steps={args.hold_steps})"
    )


if __name__ == "__main__":
    main()
