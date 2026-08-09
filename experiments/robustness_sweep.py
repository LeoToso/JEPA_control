"""Sweep the initial-state perturbation magnitude along a chosen direction
and compare closed-loop success_rate / mean_fraction_stable / final-state
distance across one or more checkpoints -- automates the "try a few angles
and grep success_rate" loop into one repeatable comparison plot, and
(optionally) also saves a trajectory-norm-vs-step plot at specific
magnitudes so you can see BY EYE whether a failure at large perturbation is
divergence, slow convergence, or oscillation (the summary metrics alone
can't tell those apart).

    python experiments/robustness_sweep.py \\
        --checkpoints \\
            results/example2_cartpole_paper_naive/checkpoint_L_pred_L_SIGReg.pt \\
            results/example2_cartpole_paper_naive/checkpoint_L_pred_L_act.pt \\
        --labels SIGReg act \\
        --magnitudes 0.03 0.1 0.2 0.3 0.5 0.8 1.0 \\
        --trajectory-magnitudes 0.2 0.3 0.5 \\
        --n-trials 20 --n-steps 150 --hold-steps 30 --threads 4

By default the perturbation direction is the pole-angle axis [0, 0, 1, 0]
for cartpole_linear (x0 = magnitude * direction); pass --direction to sweep
along any other state-space direction (e.g. the eigenvector of P_z with the
smallest eigenvalue, if you want to target a specific "blind spot").
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from checkpoint_io import load_checkpoint_with_env
from jepa_lds.control import design_latent_controller, evaluate_controller, true_closed_loop_eigenvalues
from jepa_lds.plotting import plot_closed_loop_trajectories, plot_robustness_sweep

_CARTPOLE_DEFAULT_DIRECTION = np.array([0.0, 0.0, 1.0, 0.0])  # pole angle axis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=str, nargs="+", required=True)
    p.add_argument("--labels", type=str, nargs="+", default=None, help="one label per checkpoint (default: each checkpoint's saved config_name)")
    p.add_argument("--direction", type=float, nargs="+", default=None, help="state-space direction to perturb along, un-normalized (default: pole-angle axis [0,0,1,0], cartpole_linear only)")
    p.add_argument("--magnitudes", type=float, nargs="+", default=[0.03, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
    p.add_argument("--trajectory-magnitudes", type=float, nargs="+", default=None, help="also save a per-magnitude closed-loop-trajectory plot at these magnitudes (added to the sweep if not already present)")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-steps", type=int, default=150)
    p.add_argument("--hold-steps", type=int, default=30)
    p.add_argument("--success-threshold", type=float, default=0.5)
    p.add_argument("--q-scale", type=float, default=10.0)
    p.add_argument("--r-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out-dir", type=str, default=None, help="default: alongside the first checkpoint")
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    if args.labels is not None and len(args.labels) != len(args.checkpoints):
        raise SystemExit(f"--labels must have one entry per checkpoint ({len(args.checkpoints)} checkpoints, {len(args.labels)} labels)")

    traj_magnitudes = set(args.trajectory_magnitudes or [])
    magnitudes = sorted(set(args.magnitudes) | traj_magnitudes)

    out_dir = args.out_dir or os.path.dirname(args.checkpoints[0])
    os.makedirs(out_dir, exist_ok=True)

    sweep_results = {}
    traj_by_magnitude = {m: {} for m in traj_magnitudes}
    direction_desc = None

    for i, ckpt_path in enumerate(args.checkpoints):
        system, obs_model, encoder, predictor, _decoder, cfg, extra = load_checkpoint_with_env(ckpt_path)
        label = args.labels[i] if args.labels is not None else extra.get("config_name", os.path.basename(ckpt_path))

        if args.direction is not None:
            direction = np.array(args.direction, dtype=np.float64)
        elif system.name == "cartpole_linear":
            direction = _CARTPOLE_DEFAULT_DIRECTION.copy()
        else:
            raise SystemExit(f"--direction is required for system {system.name!r} (no default direction known)")
        if direction.shape != (system.n,):
            raise SystemExit(f"--direction must have {system.n} values for system {system.name!r} (got {direction.shape[0]})")
        direction = direction / np.linalg.norm(direction)
        direction_desc = direction_desc or f"direction={np.round(direction, 3).tolist()}"

        ctrl = design_latent_controller(predictor, q_scale=args.q_scale, r_scale=args.r_scale)
        latent_radius_str = (
            f"{ctrl['latent_closed_loop_spectral_radius']:.4f}" if ctrl["stabilizable"] else "n/a"
        )
        true_radius_str = "n/a"
        if ctrl["stabilizable"]:
            true_eigs = true_closed_loop_eigenvalues(system, obs_model, encoder, ctrl["K_z"])
            true_radius_str = f"{np.max(np.abs(true_eigs)):.4f}"
        print(
            f"[{label}] stabilizable(latent)={ctrl['stabilizable']}  "
            f"latent_closed_loop_spectral_radius={latent_radius_str} (idealized, latent-only)  "
            f"true_closed_loop_spectral_radius={true_radius_str} (actually simulated, on true state)"
        )

        res = {"success_rate": [], "mean_fraction_stable": [], "final_state_distance_avg": []}
        for mag in magnitudes:
            x0 = mag * direction
            ev = evaluate_controller(
                system, obs_model, encoder, ctrl["K_z"],
                n_trials=args.n_trials, n_steps=args.n_steps, hold_steps=args.hold_steps,
                success_threshold=args.success_threshold, seed=args.seed, x0=x0,
            )
            res["success_rate"].append(ev["success_rate"])
            res["mean_fraction_stable"].append(ev["mean_fraction_stable"])
            res["final_state_distance_avg"].append(ev["final_state_distance_avg"])
            print(
                f"  mag={mag:<6g} success={ev['success_rate']*100:5.1f}%  "
                f"frac_stable={ev['mean_fraction_stable']:.3f}  final_dist={ev['final_state_distance_avg']:.4f}"
            )
            if mag in traj_magnitudes:
                traj_by_magnitude[mag][label] = ev
        sweep_results[label] = res

    sweep_path = os.path.join(out_dir, "robustness_sweep.pdf")
    plot_robustness_sweep(
        magnitudes, sweep_results, sweep_path,
        xlabel=f"perturbation magnitude ({direction_desc})",
        title="Closed-loop robustness vs. initial perturbation size",
    )
    print(f"\nwrote {sweep_path}")

    for mag, eval_results in traj_by_magnitude.items():
        traj_path = os.path.join(out_dir, f"trajectories_mag_{mag:g}.pdf")
        plot_closed_loop_trajectories(eval_results, traj_path)
        print(f"wrote {traj_path}")


if __name__ == "__main__":
    main()
