"""Direct comparison: "regular" JEPA training (encoder + predictor trained
jointly end-to-end via a single Adam optimizer, as in the real pixel-based
project and JEPA training generally) vs. the alternating scheme used
elsewhere in this codebase (closed-form predictor fit + gradient steps on
the encoder only).

Same data, same seed, same loss, same total gradient-step budget for both --
the only thing that differs is *how* the shared objective is optimized. See
train.py's module docstring for why the alternating scheme is used for the
main experiments: naive joint SGD on unstable recursive dynamics reliably
converges to spuriously contractive latent dynamics.

    python experiments/compare_training_schemes.py [--seed 0] [--threads 4]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from jepa_lds.control import design_latent_controller, evaluate_controller
from jepa_lds.data import generate_dataset, make_observation_model
from jepa_lds.diagnostics import unstable_mode_retention
from jepa_lds.systems import make_linearized_cartpole_system
from jepa_lds.train import TrainConfig, train_jepa, train_jepa_naive


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4, help="torch.set_num_threads -- tiny models, more is often worse on shared/many-core machines")
    p.add_argument("--n-train-ep", type=int, default=500)
    p.add_argument("--outer-rounds", type=int, default=40, help="also sets the naive trainer's total epoch budget (outer_rounds * inner_epochs)")
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--config", choices=["sigreg", "actrecon"], default="sigreg",
                    help="which second loss term to add to L_pred (multistep-only): SIGReg or action reconstruction")
    p.add_argument("--lambda-pred-ms", type=float, default=1.0, help="weight on L_pred (multistep rollout, the only prediction term used)")
    p.add_argument("--lambda-sigreg", type=float, default=1.0, help="weight on L_SIGReg (only used when --config sigreg)")
    p.add_argument("--lambda-actrecon", type=float, default=20.0, help="weight on L_act, multistep decoder only (only used when --config actrecon)")
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    system = make_linearized_cartpole_system(dt=0.02)
    T = 30
    horizon = T - 1

    obs_model = make_observation_model(
        system, obs_dim_signal=10, n_distractor=10, measurement_noise_std=0.002, seed=args.obs_seed,
    )
    train_batch = generate_dataset(
        system, obs_model, args.n_train_ep, T, seed=args.seed,
        x0_std=0.03, action_std=0.3, state_clip=8.0,
    )
    val_batch = generate_dataset(
        system, obs_model, 120, T, seed=args.seed + 10_000,
        x0_std=0.03, action_std=0.3, state_clip=8.0,
    )

    print(f"eigvals(A) = {system.eigvals()}  spectral_radius = {system.spectral_radius():.4f}")

    # L_pred is ALWAYS multistep-only here: lambda_pred_1step=0 (no one-step
    # backbone), matching the LaTeX's L_pred exactly. Exactly one of
    # {SIGReg, action reconstruction} is active; the other's weights are 0.
    if args.config == "sigreg":
        config_label = f"L_pred + L_SIGReg (lambda_sigreg={args.lambda_sigreg})"
        lambda_sigreg = args.lambda_sigreg
        lambda_actrecon_1step = 0.0
        lambda_actrecon_ms = 0.0
    else:
        config_label = f"L_pred + L_act (lambda_act={args.lambda_actrecon})"
        lambda_sigreg = 0.0
        lambda_actrecon_1step = 0.0  # one-step decoder off too -- L_act is the multistep decoder only
        lambda_actrecon_ms = args.lambda_actrecon

    cfg = TrainConfig(
        latent_dim=system.n, horizon=horizon, outer_rounds=args.outer_rounds, inner_epochs=6,
        batch_size=4096, lr=1e-2,
        lambda_pred_1step=0.0, lambda_pred_ms=args.lambda_pred_ms,
        lambda_sigreg=lambda_sigreg,
        lambda_actrecon_1step=lambda_actrecon_1step, lambda_actrecon_ms=lambda_actrecon_ms,
        seed=args.seed,
    )
    print(f"config: {config_label}")
    print(
        f"  lambda_pred_1step={cfg.lambda_pred_1step}  lambda_pred_ms={cfg.lambda_pred_ms}  "
        f"lambda_sigreg={cfg.lambda_sigreg}  lambda_actrecon_1step={cfg.lambda_actrecon_1step}  "
        f"lambda_actrecon_ms={cfg.lambda_actrecon_ms}"
    )

    for name, train_fn in [("alternating", train_jepa), ("naive (regular joint SGD)", train_jepa_naive)]:
        print(f"\n--- {name} ---")
        enc, pred, _dec, _hist = train_fn(system, obs_model, train_batch, cfg, verbose=True)
        ctrl = design_latent_controller(pred)
        ev = evaluate_controller(
            system, obs_model, enc, ctrl["K_z"],
            n_trials=args.n_trials, n_steps=100, x0_std=0.03,
            success_threshold=0.5, hold_steps=20, seed=args.seed + 555,
        )
        r2 = unstable_mode_retention(system, enc, train_batch, val_batch)
        A_z, _B_z = pred.matrices()
        print(
            f"[{name}] R2(unstable mode)={r2:.3f}  success={ev['success_rate']*100:.1f}%  "
            f"frac_stable={ev['mean_fraction_stable']:.3f}"
        )
        print(f"  true eigvals(A)   = {system.eigvals()}")
        print(f"  learned eigvals(A_z) = {np.linalg.eigvals(A_z)}")


if __name__ == "__main__":
    main()
