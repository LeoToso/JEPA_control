"""Example 3, SIGReg config: single-seed training run of L_pred + L_SIGReg
on the "killer" construction (see run_example3_killer_collapse.py's module
docstring for the full derivation) that reliably collapses the unstable
mode -- with L_pred driven to (near) exactly zero and the closed-loop LQR
success rate driven to exactly 0%.

Same 2-state `double_mode` system, latent_dim=1 (deliberately smaller than
the true state dimension, forcing an exact L_pred=0 tie between "keep only
the unstable mode" and "keep only the stable mode"), and an initial state
distribution biased toward the stable modal coordinate having much larger
variance than the unstable one. Prints per-outer-round training losses
(pred_1step, pred_ms, sigreg, total) as it goes, then reports the unstable-
mode retention R^2 and closed-loop LQR success rate, and saves a checkpoint
for further analysis with evaluate_checkpoint.py / plot_five_panels.py /
probe_local_stability.py / robustness_sweep.py.

    python experiments/run_example3_sigreg.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from jepa_lds.checkpoint import save_checkpoint
from jepa_lds.control import design_latent_controller, evaluate_controller
from jepa_lds.data import generate_dataset, make_modal_contaminated_x0_sampler, make_observation_model
from jepa_lds.diagnostics import unstable_mode_retention
from jepa_lds.systems import make_double_mode_system
from jepa_lds.train import TrainConfig, train_jepa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latent-dim", type=int, default=1, help="deliberately < system.n=2 -- forces the L_pred=0 tie between the two eigenspaces")
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--lambda-sigreg", type=float, default=25.0)
    p.add_argument("--collapse-threshold", type=float, default=0.5, help="unstable_mode_retention R^2 below this counts as 'collapsed'")

    p.add_argument("--sigma-stable", type=float, default=0.3, help="std of the stable modal coordinate's initial condition")
    p.add_argument("--sigma-unstable", type=float, default=0.01, help="std of the unstable modal coordinate's initial condition")

    p.add_argument("--T", type=int, default=5, help="episode length (must be > horizon)")
    p.add_argument("--n-train-ep", type=int, default=400)
    p.add_argument("--n-val-ep", type=int, default=100)
    p.add_argument("--action-std", type=float, default=0.02, help="validated sweet spot -- see run_example3_killer_collapse.py's module docstring")
    p.add_argument("--obs-dim-signal", type=int, default=6)
    p.add_argument("--n-distractor", type=int, default=0)
    p.add_argument("--measurement-noise-std", type=float, default=0.0, help="noiseless by design -- keeps L_pred=0 exactly achievable")

    p.add_argument("--outer-rounds", type=int, default=60)
    p.add_argument("--inner-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--log-every", type=int, default=5, help="print training progress every N outer rounds")

    p.add_argument("--q-scale", type=float, default=10.0, help="LQR Q = q_scale * I in latent space")
    p.add_argument("--r-scale", type=float, default=1.0, help="LQR R = r_scale * I")
    p.add_argument("--eval-x0-std", type=float, default=0.05, help="isotropic std of the random initial state for closed-loop eval trials")
    p.add_argument("--eval-n-trials", type=int, default=20)
    p.add_argument("--eval-n-steps", type=int, default=100)
    p.add_argument("--eval-success-threshold", type=float, default=0.5)
    p.add_argument("--eval-hold-steps", type=int, default=15)
    p.add_argument("--eval-seed", type=int, default=555)

    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--obs-seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--out-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "example3_sigreg"),
    )
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)

    system = make_double_mode_system()
    if args.T <= args.horizon:
        raise SystemExit(f"--T={args.T} must be > --horizon={args.horizon}")

    obs_model = make_observation_model(
        system, obs_dim_signal=args.obs_dim_signal, n_distractor=args.n_distractor,
        measurement_noise_std=args.measurement_noise_std, seed=args.obs_seed,
    )
    x0_sampler = make_modal_contaminated_x0_sampler(
        system, sigma_stable=args.sigma_stable, sigma_unstable_small=args.sigma_unstable,
        sigma_unstable_large=args.sigma_unstable, contamination_prob=0.0,
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
    _w, _V, Vinv = system.modal_decomposition()
    idx_u = system.unstable_mode_index()
    idx_s = 1 - idx_u
    x_flat = train_batch.x.reshape(-1, system.n)
    xi_u = np.real(x_flat @ Vinv[idx_u].conj())
    xi_s = np.real(x_flat @ Vinv[idx_s].conj())
    print(f"pooled var(unstable modal coord)={xi_u.var():.6f}  var(stable modal coord)={xi_s.var():.6f}")
    print(f"config: L_pred + L_SIGReg  lambda_sigreg={args.lambda_sigreg}  latent_dim={args.latent_dim}  horizon={args.horizon}")

    cfg = TrainConfig(
        latent_dim=args.latent_dim, horizon=args.horizon, outer_rounds=args.outer_rounds,
        inner_epochs=args.inner_epochs, batch_size=args.batch_size, lr=args.lr,
        lambda_pred_1step=0.0, lambda_pred_ms=1.0, lambda_sigreg=args.lambda_sigreg,
        lambda_actrecon_1step=0.0, lambda_actrecon_ms=0.0, seed=args.seed,
    )
    print(f"\n--- training L_pred + L_SIGReg (seed={args.seed}) ---")
    enc, pred, dec, hist = train_jepa(system, obs_model, train_batch, cfg, verbose=True, log_every=args.log_every)

    r2 = unstable_mode_retention(system, enc, train_batch, val_batch)
    collapsed = r2 < args.collapse_threshold
    A_z, B_z = pred.matrices()
    print(f"\nunstable_mode_retention R^2 = {r2:.4f}  ({'COLLAPSED' if collapsed else 'retained'}, threshold={args.collapse_threshold})")
    print(f"learned A_z = {A_z.flatten()}  B_z = {B_z.flatten()}")

    ctrl = design_latent_controller(pred, q_scale=args.q_scale, r_scale=args.r_scale)
    print(f"stabilizable (latent) = {ctrl['stabilizable']}")
    if ctrl["K_z"] is not None:
        ev = evaluate_controller(
            system, obs_model, enc, ctrl["K_z"],
            n_trials=args.eval_n_trials, n_steps=args.eval_n_steps, x0_std=args.eval_x0_std,
            success_threshold=args.eval_success_threshold, hold_steps=args.eval_hold_steps, seed=args.eval_seed,
        )
        print(
            f"closed-loop success_rate={ev['success_rate']*100:.1f}%  "
            f"mean_fraction_stable={ev['mean_fraction_stable']:.3f}  "
            f"final_state_distance_avg={ev['final_state_distance_avg']:.4f}"
        )

    ckpt_path = os.path.join(args.out_dir, f"checkpoint_sigreg_H{args.horizon}_seed{args.seed}.pt")
    save_checkpoint(
        ckpt_path, enc, pred, dec, cfg,
        extra={"obs_dim": obs_model.p, "action_dim": system.m, "system_name": system.name,
               "config_name": f"L_pred + L_SIGReg (killer collapse, H={args.horizon}, seed={args.seed})",
               "trainer": "alternating",
               "obs_dim_signal": args.obs_dim_signal, "n_distractor": args.n_distractor,
               "measurement_noise_std": args.measurement_noise_std, "distractor_std": 1.0,
               "obs_seed": args.obs_seed},
    )
    print(f"\nsaved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
