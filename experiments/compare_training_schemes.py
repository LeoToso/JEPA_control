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

    # Tuned config known to make the alternating scheme succeed cleanly
    # (see run_example2_cartpole_linear.py) -- chosen here specifically so
    # the comparison has a clear "succeeds vs fails" contrast to show,
    # rather than two mediocre-and-inconclusive results.
    cfg = TrainConfig(
        latent_dim=system.n, horizon=horizon, outer_rounds=args.outer_rounds, inner_epochs=6,
        batch_size=4096, lr=1e-2,
        lambda_pred_1step=1.0, lambda_pred_ms=1.0,
        lambda_actrecon_1step=20.0, lambda_actrecon_ms=20.0,
        seed=args.seed,
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
