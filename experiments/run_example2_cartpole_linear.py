"""Example 2: 4-state linearized-cartpole LDS (linearized about the upright
equilibrium: one unstable real mode, one stable real mode, a marginal
repeated-eigenvalue-1 cart-translation pair), single input (cart force).

    python experiments/run_example2_cartpole_linear.py [--seed 0]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from common import run_experiment

from jepa_lds.systems import make_linearized_cartpole_system


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-seed", type=int, default=0)
    p.add_argument("--n-train-ep", type=int, default=500)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--outer-rounds", type=int, default=40)
    p.add_argument("--threads", type=int, default=4, help="torch.set_num_threads -- tiny models, more is often worse on shared/many-core machines")
    p.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "results", "example2_cartpole_linear"))
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    system = make_linearized_cartpole_system(dt=0.02)
    T = 30
    run_experiment(
        system=system,
        out_dir=args.out_dir,
        n_train_ep=args.n_train_ep,
        n_val_ep=120,
        T=T,
        x0_std=0.03,
        action_std=0.3,
        state_clip=8.0,
        obs_dim_signal=10,
        n_distractor=10,
        measurement_noise_std=0.002,
        latent_dim=system.n,
        horizon=T - 1,
        outer_rounds=args.outer_rounds,
        inner_epochs=6,
        lr=1e-2,
        lambda_sigreg=400.0,
        lambda_actrecon=20.0,
        lqr_q_scale=10.0,
        n_trials=args.n_trials,
        n_steps=100,
        hold_steps=20,
        seed=args.seed,
        obs_seed=args.obs_seed,
    )


if __name__ == "__main__":
    main()
