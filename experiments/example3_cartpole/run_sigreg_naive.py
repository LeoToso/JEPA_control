"""Example 3 (Cartpole) -- SIGReg config, naive joint SGD.

A "killer collapse" construction on the linearized cartpole (4-state, 1
unstable pole mode + 1 stable pole mode + a 2-state marginal cart Jordan
block at eigenvalue 1) -- the n-state generalization of Examples 1/2's
2-state `double_mode` construction.

  System: `make_linearized_cartpole_system` (state = [cart pos, cart vel,
    pole angle, pole angular vel]). Eigenvalues {1, 1, ~1.10 (unstable
    pole), ~0.91 (stable pole)}; the repeated eigenvalue 1 is a genuine
    Jordan block (NOT diagonalizable) spanning cart position/velocity, but
    is still exactly A-invariant as a 2D block (and cart position ALONE is
    also exactly A-invariant, since A@e_pos=e_pos and nothing else in the
    dynamics depends on cart position at all -- see
    `make_cartpole_modal_contaminated_x0_sampler`'s docstring).

  latent_dim=3 (< system.n=4): forces the same exact L_pred=0 tie as
    Examples 1/2, now between whichever 3D A-invariant subspace the
    encoder's 1D kernel excludes. IMPORTANT CAVEAT discovered while
    validating this example: cart position is a "free" degenerate option
    too -- since nothing else in the dynamics depends on it, DROPPING IT
    costs ~zero L_pred regardless of its own IC variance, and both SIGReg
    and (to a lesser extent) action-reconstruction have some tendency to
    drop it. This is harmless for the pedagogical point (cart position
    doesn't affect anything else, so losing it doesn't threaten POLE
    stability), but it does mean a naive full-state success criterion is
    dominated by an irrelevant, physically-inert "the cart drifts along an
    infinite track" mode rather than the unstable-pole-collapse story. This
    script's closed-loop success check (see `_evaluate_success`) therefore
    excludes cart position from the state norm by default
    (--success-dims 1,2,3), matching the common "keep the pole up, don't
    worry about absolute cart position" relaxation used in most cartpole
    benchmarks.

  Validated (3 seeds, default settings): SIGReg reliably collapses the
    unstable pole's modal-coordinate retention (R^2 ~ 0.001-0.012) and
    reliably fails closed-loop control (0% success across all 3 seeds).
    See run_actrecon_naive.py in this folder for the counterpart that
    retains it.

    python experiments/example3_cartpole/run_sigreg_naive.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch

from jepa_lds.checkpoint import save_checkpoint
from jepa_lds.control import closed_loop_rollout, design_latent_controller
from jepa_lds.data import generate_dataset, make_cartpole_modal_contaminated_x0_sampler, make_observation_model
from jepa_lds.diagnostics import unstable_mode_retention
from jepa_lds.systems import make_linearized_cartpole_system
from jepa_lds.train import TrainConfig, train_jepa_naive


def _evaluate_success(system, obs_model, encoder, K_z, dims, n_trials, n_steps, x0_std, threshold, hold_steps, seed):
    """Same success criterion as `control.evaluate_controller` (state norm
    below `threshold` for the final `hold_steps` steps), but computed only
    over `dims` -- see this module's docstring for why cart position
    (dim 0) is excluded by default."""
    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(n_trials):
        x0 = x0_std * rng.standard_normal(system.n)
        xs = closed_loop_rollout(system, obs_model, encoder, K_z, n_steps, x0, 0.0, rng)
        norms = np.linalg.norm(xs[:, dims], axis=1)
        stable = np.isfinite(norms) & (norms < threshold)
        if len(stable) >= hold_steps and np.all(stable[-hold_steps:]):
            successes += 1
    return successes / n_trials


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latent-dim", type=int, default=3, help="deliberately < system.n=4 -- forces the L_pred=0 tie between candidate 3D invariant subspaces")
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--lambda-sigreg", type=float, default=25.0)
    p.add_argument("--collapse-threshold", type=float, default=0.5, help="unstable_mode_retention R^2 below this counts as 'collapsed'")

    p.add_argument("--sigma-cart-pos", type=float, default=0.6, help="std of cart position's initial condition")
    p.add_argument("--sigma-cart-vel", type=float, default=0.6, help="std of cart velocity's initial condition")
    p.add_argument("--sigma-stable", type=float, default=0.3, help="std of the stable pole modal coordinate's initial condition")
    p.add_argument("--sigma-unstable", type=float, default=0.01, help="std of the unstable pole modal coordinate's initial condition")

    p.add_argument("--T", type=int, default=15, help="episode length (must be > horizon)")
    p.add_argument("--n-train-ep", type=int, default=400)
    p.add_argument("--n-val-ep", type=int, default=100)
    p.add_argument("--action-std", type=float, default=0.05)
    p.add_argument("--obs-dim-signal", type=int, default=8)
    p.add_argument("--n-distractor", type=int, default=0)
    p.add_argument("--measurement-noise-std", type=float, default=0.0, help="noiseless by design -- keeps L_pred=0 exactly achievable")

    p.add_argument("--outer-rounds", type=int, default=400, help="also sets the naive trainer's total epoch budget (outer_rounds * inner_epochs)")
    p.add_argument("--inner-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--log-every", type=int, default=50, help="print training progress every N epochs")

    p.add_argument("--q-scale", type=float, default=10.0, help="LQR Q = q_scale * I in latent space")
    p.add_argument("--r-scale", type=float, default=1.0, help="LQR R = r_scale * I")
    p.add_argument("--eval-x0-std", type=float, default=0.05, help="isotropic std of the random initial state for closed-loop eval trials")
    p.add_argument("--eval-n-trials", type=int, default=30)
    p.add_argument("--eval-n-steps", type=int, default=200)
    p.add_argument("--eval-success-threshold", type=float, default=0.5)
    p.add_argument("--eval-hold-steps", type=int, default=15)
    p.add_argument("--eval-seed", type=int, default=555)
    p.add_argument(
        "--success-dims", type=int, nargs="+", default=[1, 2, 3],
        help="state dims included in the closed-loop success norm (default: cart velocity, pole angle, pole angular "
        "velocity -- excludes cart position, dim 0, which is dynamically inert; see module docstring)",
    )

    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--obs-seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--out-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "example3_sigreg_naive"),
    )
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)

    system = make_linearized_cartpole_system(dt=0.02)
    if args.T <= args.horizon:
        raise SystemExit(f"--T={args.T} must be > --horizon={args.horizon}")

    obs_model = make_observation_model(
        system, obs_dim_signal=args.obs_dim_signal, n_distractor=args.n_distractor,
        measurement_noise_std=args.measurement_noise_std, seed=args.obs_seed,
    )
    x0_sampler = make_cartpole_modal_contaminated_x0_sampler(
        system, sigma_cart_pos=args.sigma_cart_pos, sigma_cart_vel=args.sigma_cart_vel,
        sigma_stable=args.sigma_stable, sigma_unstable=args.sigma_unstable,
    )
    mixture = {"passive": 0.3, "random": 0.7} if args.action_std > 0 else {"passive": 1.0}
    train_batch = generate_dataset(
        system, obs_model, args.n_train_ep, args.T, seed=args.data_seed,
        action_std=args.action_std, process_noise_std=0.0, state_clip=50.0,
        x0_sampler=x0_sampler, mixture=mixture,
    )
    val_batch = generate_dataset(
        system, obs_model, args.n_val_ep, args.T, seed=args.data_seed + 10_000,
        action_std=args.action_std, process_noise_std=0.0, state_clip=50.0,
        x0_sampler=x0_sampler, mixture=mixture,
    )

    print(f"eigvals(A) = {system.eigvals()}  spectral_radius = {system.spectral_radius():.4f}")
    print(f"config: L_pred + L_SIGReg (naive joint SGD)  lambda_sigreg={args.lambda_sigreg}  latent_dim={args.latent_dim}  horizon={args.horizon}")

    cfg = TrainConfig(
        latent_dim=args.latent_dim, horizon=args.horizon, outer_rounds=args.outer_rounds,
        inner_epochs=args.inner_epochs, batch_size=args.batch_size, lr=args.lr,
        lambda_pred_1step=0.0, lambda_pred_ms=1.0, lambda_sigreg=args.lambda_sigreg,
        lambda_actrecon_1step=0.0, lambda_actrecon_ms=0.0, seed=args.seed,
    )
    print(f"\n--- training L_pred + L_SIGReg, naive joint SGD (seed={args.seed}) ---")
    enc, pred, dec, hist = train_jepa_naive(system, obs_model, train_batch, cfg, verbose=True, log_every=args.log_every)

    r2 = unstable_mode_retention(system, enc, train_batch, val_batch)
    collapsed = r2 < args.collapse_threshold
    A_z, B_z = pred.matrices()
    print(f"\nunstable_mode_retention R^2 = {r2:.4f}  ({'COLLAPSED' if collapsed else 'retained'}, threshold={args.collapse_threshold})")
    print(f"learned A_z eigvals = {np.linalg.eigvals(A_z)}")

    ctrl = design_latent_controller(pred, q_scale=args.q_scale, r_scale=args.r_scale)
    print(f"stabilizable (latent) = {ctrl['stabilizable']}")
    if ctrl["K_z"] is not None:
        success = _evaluate_success(
            system, obs_model, enc, ctrl["K_z"], args.success_dims,
            args.eval_n_trials, args.eval_n_steps, args.eval_x0_std,
            args.eval_success_threshold, args.eval_hold_steps, args.eval_seed,
        )
        print(f"closed-loop success_rate (dims={args.success_dims}) = {success*100:.1f}%")

    ckpt_path = os.path.join(args.out_dir, f"checkpoint_sigreg_naive_H{args.horizon}_seed{args.seed}.pt")
    save_checkpoint(
        ckpt_path, enc, pred, dec, cfg,
        extra={"obs_dim": obs_model.p, "action_dim": system.m, "system_name": system.name,
               "config_name": f"L_pred + L_SIGReg naive (Example 3: cartpole, H={args.horizon}, seed={args.seed})",
               "trainer": "naive",
               "obs_dim_signal": args.obs_dim_signal, "n_distractor": args.n_distractor,
               "measurement_noise_std": args.measurement_noise_std, "distractor_std": 1.0,
               "obs_seed": args.obs_seed},
    )
    print(f"\nsaved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
