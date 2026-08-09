"""Reproduce the "5-panel" checkpoint-diagnostic figure from the real
pixel-based project's `plot_checkpoint_summary_physical_smwm.py` for a saved
linear toy-system checkpoint: phase portrait, H-step prediction error,
planning cost, latent norm divergence, and cosine alignment along the
unstable eigenvector.

Panels 2 and 3 are computed by decoding the predictor's own H-step
zero-action latent rollout back to the ORIGINAL PHYSICAL STATE SPACE with a
ridge probe fit on fresh data (D(f_H(z, 0))), matching the "physical"
variant of the reference script rather than measuring error/cost in raw
latent coordinates.

    python experiments/plot_five_panels.py \\
        --checkpoint results/example2_cartpole_paper_naive/checkpoint_L_pred_L_act.pt \\
        --out results/five_panels.pdf

Panels 1-3 are evaluated on a 2D grid sliced out of the full state space
(all other state coordinates held at zero); which two coordinates to grid
over defaults to the system's most physically meaningful pair (`--dims`).
For cartpole_linear's default angle dims (2, 3), axes are displayed in
DEGREES (and deg/s) instead of radians by default -- pass `--degrees off`
to keep radians. The grid's half-width (`--grid-half-width`) is interpreted
in whatever unit is being displayed (degrees or radians/state-units).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from checkpoint_io import load_checkpoint_with_env
from jepa_lds.data import generate_dataset
from jepa_lds.diagnostics import (
    cosine_alignment_unstable_direction,
    fit_state_probe,
    h_step_prediction_error_panel,
    latent_norm_divergence,
    phase_portrait_panel,
    planning_cost_panel,
)
from jepa_lds.plotting import plot_five_panels

_DEFAULT_DIMS = {
    "cartpole_linear": (2, 3),  # pole angle, pole angular velocity
    "double_mode": (0, 1),
    "double_mode_stable": (0, 1),
}
_ANGLE_DIMS = {
    "cartpole_linear": {2, 3},  # dims whose natural unit is radians (or rad/s)
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default=None, help="output PDF path (default: alongside the checkpoint)")
    p.add_argument("--dims", type=int, nargs=2, default=None, help="which 2 state dims to grid over (default: system-specific)")
    p.add_argument(
        "--degrees", choices=["auto", "on", "off"], default="auto",
        help="display angle-like grid axes (and phase-portrait drift) in degrees instead of radians. "
             "'auto' (default) turns this on only when gridding cartpole_linear's default angle dims (2,3). "
             "Does NOT affect the prediction-error/planning-cost VALUES, which stay in the state's natural units.",
    )
    p.add_argument(
        "--grid-half-width", type=float, default=None,
        help="grid spans [-w, w] x [-w, w] in the chosen state dims, in the unit currently being displayed "
             "(degrees if degree-display is active, else radians/state-units). Default: 90 (degrees) if "
             "degree-display is active, else 1.0.",
    )
    p.add_argument("--n-points", type=int, default=25, help="grid resolution per axis for panels 1-3")
    p.add_argument("--h-step", type=int, default=10, help="H for the H-step prediction-error and planning-cost panels (zero-action latent rollout length)")
    p.add_argument("--n-probe-episodes", type=int, default=300, help="episodes generated to fit the state ridge probe")
    p.add_argument("--probe-horizon", type=int, default=30, help="episode length used to fit the state ridge probe")
    p.add_argument("--n-steps", type=int, default=30, help="rollout length for the norm-divergence / cosine-alignment panels")
    p.add_argument("--perturbation-scales", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    system, obs_model, encoder, predictor, _decoder, cfg, extra = load_checkpoint_with_env(args.checkpoint)
    dims = tuple(args.dims) if args.dims is not None else _DEFAULT_DIMS.get(system.name, (0, 1))

    if args.degrees == "auto":
        use_degrees = set(dims) <= _ANGLE_DIMS.get(system.name, set())
    else:
        use_degrees = args.degrees == "on"

    grid_half_width = args.grid_half_width if args.grid_half_width is not None else (90.0 if use_degrees else 1.0)
    unit_label = "degrees" if use_degrees else "state units (e.g. radians for cartpole_linear angle dims)"

    print(f"checkpoint: {args.checkpoint}")
    print(f"  system={system.name}  config={extra.get('config_name', '?')}  trainer={extra.get('trainer', '?')}")
    print(f"  gridding dims={dims} over [-{grid_half_width}, {grid_half_width}]^2 ({unit_label})")

    if use_degrees:
        half_width_rad = np.deg2rad(grid_half_width)
    else:
        half_width_rad = grid_half_width
    lo = (-half_width_rad, -half_width_rad)
    hi = (half_width_rad, half_width_rad)

    rng = np.random.default_rng(args.seed)
    probe_batch = generate_dataset(
        system, obs_model, args.n_probe_episodes, args.probe_horizon, seed=args.seed,
        x0_std=0.03, action_std=0.3, state_clip=8.0,
    )
    beta = fit_state_probe(encoder, probe_batch)

    phase_panel = phase_portrait_panel(system, obs_model, encoder, predictor, beta, dims=dims, lo=lo, hi=hi, n_points=args.n_points)
    pred_err_panel = h_step_prediction_error_panel(system, obs_model, encoder, predictor, beta, dims=dims, lo=lo, hi=hi, n_points=args.n_points, H=args.h_step)
    planning_panel = planning_cost_panel(system, obs_model, encoder, predictor, beta, dims=dims, lo=lo, hi=hi, n_points=args.n_points, H=args.h_step)

    if use_degrees:
        rad2deg = 180.0 / np.pi
        for panel in (phase_panel, pred_err_panel, planning_panel):
            panel["XX"] = panel["XX"] * rad2deg
            panel["YY"] = panel["YY"] * rad2deg
        phase_panel["U_true"] = phase_panel["U_true"] * rad2deg
        phase_panel["V_true"] = phase_panel["V_true"] * rad2deg
        phase_panel["U_learned"] = phase_panel["U_learned"] * rad2deg
        phase_panel["V_learned"] = phase_panel["V_learned"] * rad2deg

    x0_panel = 0.03 * rng.standard_normal(system.n)
    true_norms, rec_norms = latent_norm_divergence(system, obs_model, encoder, predictor, x0_panel, n_steps=args.n_steps, rng=rng)
    cos_over_time = cosine_alignment_unstable_direction(
        system, obs_model, encoder, predictor,
        n_steps=args.n_steps, perturbation_scales=args.perturbation_scales, rng=rng,
    )

    out_path = args.out or os.path.splitext(args.checkpoint)[0] + "_five_panels.pdf"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    title = f"{system.name} -- {extra.get('config_name', '?')} ({extra.get('trainer', '?')})"
    plot_five_panels(
        phase_panel, pred_err_panel, planning_panel, true_norms, rec_norms, cos_over_time, out_path,
        title=title, unit_suffix="deg" if use_degrees else "",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
