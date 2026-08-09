"""Local-stability probe for a saved checkpoint: a 3-panel figure
(unstable-eigenvector alignment, closed-loop trajectories, Lyapunov
certificate), adapted from the real pixel-based project's
`probe_local_stability_smwm.py`.

  Panel 1 (eigenvector alignment): the TRUE system's unstable eigenvector
    vs. the LEARNED predictor's own dominant (largest-|eigenvalue|)
    eigenvector, mapped back to state space via the pseudoinverse of the
    encoder's own EXACT linear map (no data-fit probe involved) and
    normalized to unit length -- a direct check of whether the learned
    dynamics' fastest-growing direction actually points along the true
    unstable mode, rather than inferring it indirectly from a local
    vector field. The true stable eigenvector is shown too (grey), so
    it's visually obvious if "learned" has instead aligned with the wrong
    mode.
  Panel 2 (closed-loop trajectories): one deterministic (up to observation
    noise) --n-steps closed-loop rollout under the LEARNED controller
    u_t = -K_z * encoder(y_t) on the TRUE system per starting state (marked
    with an open circle), defaulting to the 4 corners of a
    --traj-dim0-max x --traj-dim1-max box (override with --traj-x0). A
    direct, qualitative "does this large initial deviation actually get
    driven to the equilibrium" view -- converging trajectories curve back to
    the star, diverging ones exit the (fixed) visible frame.
  Panel 3 (Lyapunov certificate): pointwise check of whether ONE step under
    the learned controller decreases the oracle's own quadratic Lyapunov
    function V(x) = x^T P_gt x -- an honest, model-free check of whether the
    learned controller's action is a valid descent direction for the TRUE
    system's own cost, independent of what the learned model believes.

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
from jepa_lds.diagnostics import (
    closed_loop_trajectory_panel,
    lyapunov_decrease_panel,
    unstable_eigenvector_alignment,
)
from jepa_lds.plotting import plot_local_stability_probe

_DEFAULT_DIMS = {
    "cartpole_linear": (2, 3),  # pole angle, pole angular velocity
    "double_mode": (0, 1),
    "double_mode_stable": (0, 1),
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

    p.add_argument("--traj-dim0-max", type=float, default=25.0, help="default trajectory starting-corner magnitude along dims[0]")
    p.add_argument("--traj-dim1-max", type=float, default=80.0, help="default trajectory starting-corner magnitude along dims[1]")
    p.add_argument(
        "--traj-x0", type=float, nargs=2, action="append", default=None,
        help="explicit starting state (dims[0], dims[1] value) for panel 2; repeat for multiple trajectories "
        "(overrides the default 4-corner box from --traj-dim0-max/--traj-dim1-max)",
    )
    p.add_argument("--n-steps", type=int, default=300, help="closed-loop rollout length for panel 2's trajectories")

    p.add_argument("--lyap-dim0-max", type=float, default=15.0, help="Lyapunov grid half-width along dims[0]")
    p.add_argument("--lyap-dim1-max", type=float, default=50.0, help="Lyapunov grid half-width along dims[1]")
    p.add_argument("--lyap-n-dim0", type=int, default=31)
    p.add_argument("--lyap-n-dim1", type=int, default=25)

    p.add_argument("--q-scale", type=float, default=1.0, help="LQR Q = q_scale * I (both learned-latent and oracle-physical designs)")
    p.add_argument("--r-scale", type=float, default=1.0, help="LQR R = r_scale * I")
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

    def _maybe_deg2rad(v: float) -> float:
        return np.deg2rad(v) if use_degrees else v

    def _corner_x0s(max0: float, max1: float) -> list[np.ndarray]:
        x0s = []
        for s0, s1 in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            x0 = np.zeros(system.n)
            x0[dims[0]] = s0 * max0
            x0[dims[1]] = s1 * max1
            x0s.append(x0)
        return x0s

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

    lo_lyap, hi_lyap = _lo_hi(args.lyap_dim0_max, args.lyap_dim1_max)

    if args.traj_x0 is not None:
        x0s = []
        for v0, v1 in args.traj_x0:
            x0 = np.zeros(system.n)
            x0[dims[0]] = _maybe_deg2rad(v0)
            x0[dims[1]] = _maybe_deg2rad(v1)
            x0s.append(x0)
    else:
        x0s = _corner_x0s(_maybe_deg2rad(args.traj_dim0_max), _maybe_deg2rad(args.traj_dim1_max))

    print("[panel 1] unstable eigenvector alignment...")
    eig_panel = unstable_eigenvector_alignment(system, obs_model, encoder, predictor, dims=dims)
    print(f"  learned dominant eigenvalue = {eig_panel['learned_dominant_eigval']}")
    print(f"  cos_sim(learned, true unstable) = {eig_panel['cos_sim_unstable']:.4f}")
    if "cos_sim_stable" in eig_panel:
        print(f"  cos_sim(learned, true stable)   = {eig_panel['cos_sim_stable']:.4f}")

    print("[panel 2] closed-loop trajectories...")
    traj_panel = closed_loop_trajectory_panel(
        system, obs_model, encoder, K_z, x0s, dims=dims, n_steps=args.n_steps,
    )
    for x0, xs in zip(x0s, traj_panel["trajectories"]):
        print(f"  x0={x0[list(dims)]}  ->  final in-plane distance={float(np.linalg.norm(xs[-1])):.3g}")

    print("[panel 3] Lyapunov certificate...")
    lyap_panel = lyapunov_decrease_panel(
        system, obs_model, encoder, K_z, P_gt, dims=dims,
        lo=lo_lyap, hi=hi_lyap, n_points=(args.lyap_n_dim0, args.lyap_n_dim1),
    )
    print(f"  fraction of grid satisfying decrease = {lyap_panel['frac_decrease']*100:.1f}%")

    # eig_panel's vectors are unit-normalized directions in a slice where both plotted
    # dims share the same rad2deg factor (see _ANGLE_DIMS), so degree-vs-radian display
    # doesn't change their normalized direction -- nothing to rescale there.
    if use_degrees:
        rad2deg = 180.0 / np.pi
        lyap_panel["XX"] = lyap_panel["XX"] * rad2deg
        lyap_panel["YY"] = lyap_panel["YY"] * rad2deg
        traj_panel["trajectories"] = [xs * rad2deg for xs in traj_panel["trajectories"]]
        traj_panel["x0s"] = [x0 * rad2deg for x0 in traj_panel["x0s"]]

    if system.is_open_loop_unstable():
        dominant_mode_label, other_mode_label = "ground truth (unstable)", "ground truth (stable)"
    else:
        # Both modes are actually stable (e.g. Example 4's double_mode_stable) --
        # calling the dominant one "unstable" would be wrong, so number them instead.
        dominant_mode_label, other_mode_label = "ground truth (stable 2)", "ground truth (stable 1)"

    out_path = args.out or os.path.splitext(args.checkpoint)[0] + "_local_stability.pdf"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plot_local_stability_probe(
        eig_panel, traj_panel, lyap_panel, out_path, unit_suffix="deg" if use_degrees else "",
        dominant_mode_label=dominant_mode_label, other_mode_label=other_mode_label,
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
