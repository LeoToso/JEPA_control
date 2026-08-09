"""Local-stability probe for a saved checkpoint: a 3-panel figure (vector
field, empirical region of attraction, Lyapunov certificate), adapted from
the real pixel-based project's `probe_local_stability_smwm.py`.

  Panel 1 (vector field): true vs learned one-step drift near the origin --
    reuses the same computation as plot_five_panels.py's phase-portrait
    panel, on a smaller/tighter grid.
  Panel 2 (region of attraction): for each grid state, ONE deterministic
    (noiseless) closed-loop rollout under the LEARNED controller
    u_t = -K_z * encoder(y_t) on the TRUE system; green = stays below
    --success-threshold for the final --hold-steps of --n-steps, red =
    fails. The dashed contour overlays the same check for a full-state-
    feedback ORACLE LQR gain designed directly on the true (A, B) -- the
    best any linear controller with perfect state access could do.
  Panel 3 (Lyapunov certificate): pointwise check of whether ONE step under
    the learned controller decreases the oracle's own quadratic Lyapunov
    function V(x) = x^T P_gt x -- an honest, model-free check of whether the
    learned controller's action is a valid descent direction for the TRUE
    system's own cost, independent of what the learned model believes.

IMPORTANT CAVEAT for this exactly-linear, unconstrained toy system (unlike
the original nonlinear/pixel-based cartpole this was adapted from): a
genuinely stabilizing LINEAR controller's true region of attraction is the
WHOLE state space -- nothing here can shrink it structurally. Panel 2
actually measures "how large an initial deviation decays below a FIXED
threshold within a FIXED step budget" -- a convergence-speed budget, not a
structural stability boundary. Still a useful diagnostic (larger initial
deviations genuinely take longer to decay below a fixed absolute
threshold), just don't over-read the boundary shape as a literal basin of
attraction the way you would for the original nonlinear system.

    python experiments/probe_local_stability.py \\
        --checkpoint results/example2_cartpole_paper_naive/checkpoint_L_pred_L_SIGReg.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from checkpoint_io import load_checkpoint_with_env
from jepa_lds.control import design_latent_controller
from jepa_lds.data import generate_dataset
from jepa_lds.diagnostics import (
    fit_state_probe,
    lyapunov_decrease_panel,
    phase_portrait_panel,
    region_of_attraction_panel,
)
from jepa_lds.plotting import plot_local_stability_probe

_DEFAULT_DIMS = {
    "cartpole_linear": (2, 3),  # pole angle, pole angular velocity
    "double_mode": (0, 1),
}
_ANGLE_DIMS = {
    "cartpole_linear": {2, 3},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default=None, help="output PDF path (default: alongside the checkpoint)")
    p.add_argument("--dims", type=int, nargs=2, default=None, help="which 2 state dims to probe (default: system-specific)")
    p.add_argument(
        "--degrees", choices=["auto", "on", "off"], default="auto",
        help="display/grid angle-like dims in degrees instead of radians (auto: on for cartpole_linear's default angle dims)",
    )

    p.add_argument("--vf-dim0-max", type=float, default=8.0, help="vector-field grid half-width along dims[0]")
    p.add_argument("--vf-dim1-max", type=float, default=30.0, help="vector-field grid half-width along dims[1]")
    p.add_argument("--vf-n-dim0", type=int, default=21)
    p.add_argument("--vf-n-dim1", type=int, default=17)

    p.add_argument("--roa-dim0-max", type=float, default=25.0, help="region-of-attraction grid half-width along dims[0]")
    p.add_argument("--roa-dim1-max", type=float, default=80.0, help="region-of-attraction grid half-width along dims[1]")
    p.add_argument("--roa-n-dim0", type=int, default=31)
    p.add_argument("--roa-n-dim1", type=int, default=25)
    p.add_argument("--n-steps", type=int, default=60, help="closed-loop rollout length for the region-of-attraction check")
    p.add_argument("--success-threshold", type=float, default=0.3, help="||x_t|| must stay below this to count as 'stable'")
    p.add_argument("--hold-steps", type=int, default=10, help="state must stay below threshold for the final N steps to count as a success")

    p.add_argument("--lyap-dim0-max", type=float, default=15.0, help="Lyapunov grid half-width along dims[0]")
    p.add_argument("--lyap-dim1-max", type=float, default=50.0, help="Lyapunov grid half-width along dims[1]")
    p.add_argument("--lyap-n-dim0", type=int, default=31)
    p.add_argument("--lyap-n-dim1", type=int, default=25)

    p.add_argument("--q-scale", type=float, default=1.0, help="LQR Q = q_scale * I (both learned-latent and oracle-physical designs)")
    p.add_argument("--r-scale", type=float, default=1.0, help="LQR R = r_scale * I")
    p.add_argument("--n-probe-episodes", type=int, default=300, help="episodes generated to fit the state ridge probe (panel 1 only)")
    p.add_argument("--probe-horizon", type=int, default=30)
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

    def _lo_hi(max0: float, max1: float):
        if use_degrees:
            return (-np.deg2rad(max0), -np.deg2rad(max1)), (np.deg2rad(max0), np.deg2rad(max1))
        return (-max0, -max1), (max0, max1)

    print(f"checkpoint: {args.checkpoint}")
    print(f"  system={system.name}  config={extra.get('config_name', '?')}  trainer={extra.get('trainer', '?')}")
    print(f"  dims={dims}  degrees={use_degrees}")

    ctrl = design_latent_controller(predictor, q_scale=args.q_scale, r_scale=args.r_scale)
    print(f"  stabilizable(latent) = {ctrl['stabilizable']}")
    if not ctrl["stabilizable"]:
        raise ValueError(
            "no stabilizing latent controller could be designed for this checkpoint (unstable latent "
            "mode uncontrollable) -- nothing to probe"
        )
    K_z = ctrl["K_z"]
    K_gt, P_gt = system.dlqr(args.q_scale * np.eye(system.n), args.r_scale * np.eye(system.m))
    print(f"  rho(A_z - B_z K_z)         = {ctrl['latent_closed_loop_spectral_radius']:.4f}")
    print(f"  rho(A - B K_gt) (oracle)   = {system.closed_loop_spectral_radius(K_gt):.4f}")

    rng = np.random.default_rng(args.seed)
    probe_batch = generate_dataset(
        system, obs_model, args.n_probe_episodes, args.probe_horizon, seed=args.seed,
        x0_std=0.03, action_std=0.3, state_clip=8.0,
    )
    beta = fit_state_probe(encoder, probe_batch)

    lo_vf, hi_vf = _lo_hi(args.vf_dim0_max, args.vf_dim1_max)
    lo_roa, hi_roa = _lo_hi(args.roa_dim0_max, args.roa_dim1_max)
    lo_lyap, hi_lyap = _lo_hi(args.lyap_dim0_max, args.lyap_dim1_max)

    print("[panel 1] vector field...")
    vf_panel = phase_portrait_panel(
        system, obs_model, encoder, predictor, beta, dims=dims,
        lo=lo_vf, hi=hi_vf, n_points=(args.vf_n_dim0, args.vf_n_dim1),
    )
    _err = np.hypot(vf_panel["U_true"] - vf_panel["U_learned"], vf_panel["V_true"] - vf_panel["V_learned"])
    _mean_gt_mag = np.hypot(vf_panel["U_true"], vf_panel["V_true"]).mean()
    _norm_err = float(_err.mean() / _mean_gt_mag) if _mean_gt_mag > 1e-12 else float("nan")
    print(f"  normalized mean vector-field error = {_norm_err:.3f}")

    print("[panel 2] empirical region of attraction...")
    roa_panel = region_of_attraction_panel(
        system, obs_model, encoder, K_z, dims=dims,
        lo=lo_roa, hi=hi_roa, n_points=(args.roa_n_dim0, args.roa_n_dim1),
        n_steps=args.n_steps, success_threshold=args.success_threshold, hold_steps=args.hold_steps,
        K_gt=K_gt,
    )
    print(
        f"  learned-LQR success rate = {roa_panel['success_rate']*100:.1f}%   "
        f"oracle success rate = {roa_panel['success_rate_gt']*100:.1f}%"
    )

    print("[panel 3] Lyapunov certificate...")
    lyap_panel = lyapunov_decrease_panel(
        system, obs_model, encoder, K_z, P_gt, dims=dims,
        lo=lo_lyap, hi=hi_lyap, n_points=(args.lyap_n_dim0, args.lyap_n_dim1),
    )
    print(f"  fraction of grid satisfying decrease = {lyap_panel['frac_decrease']*100:.1f}%")

    if use_degrees:
        rad2deg = 180.0 / np.pi
        for panel in (vf_panel, roa_panel, lyap_panel):
            panel["XX"] = panel["XX"] * rad2deg
            panel["YY"] = panel["YY"] * rad2deg
        vf_panel["U_true"] = vf_panel["U_true"] * rad2deg
        vf_panel["V_true"] = vf_panel["V_true"] * rad2deg
        vf_panel["U_learned"] = vf_panel["U_learned"] * rad2deg
        vf_panel["V_learned"] = vf_panel["V_learned"] * rad2deg

    out_path = args.out or os.path.splitext(args.checkpoint)[0] + "_local_stability.pdf"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plot_local_stability_probe(
        vf_panel, roa_panel, lyap_panel, out_path, unit_suffix="deg" if use_degrees else "",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
