"""Example 1: minimal 2-state, open-loop-unstable LDS (one unstable real
mode, one stable real mode, single input, both modes controllable).

    python experiments/run_example1_double_mode.py [--seed 0]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from common import run_experiment

from jepa_lds.systems import make_double_mode_system


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-seed", type=int, default=0)
    p.add_argument("--n-train-ep", type=int, default=400)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--outer-rounds", type=int, default=40)
    p.add_argument("--threads", type=int, default=4, help="torch.set_num_threads -- tiny models, more is often worse on shared/many-core machines")
    p.add_argument("--log-every", type=int, default=10, help="print training progress every N outer rounds")
    p.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "results", "example1_double_mode"))
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    system = make_double_mode_system()
    T = 10
    run_experiment(
        system=system,
        out_dir=args.out_dir,
        n_train_ep=args.n_train_ep,
        n_val_ep=100,
        T=T,
        x0_std=0.03,
        action_std=0.3,
        state_clip=15.0,
        obs_dim_signal=6,
        n_distractor=8,
        measurement_noise_std=0.01,
        # latent_dim == true state dim: no redundant capacity, so a
        # regularizer that fights the informative directions has nowhere to
        # "hide" the sacrifice -- this is what makes the SIGReg failure mode
        # visible instead of masked by spare capacity.
        latent_dim=system.n,
        horizon=T - 1,
        outer_rounds=args.outer_rounds,
        inner_epochs=6,
        lr=1e-2,
        lambda_sigreg=150.0,
        lambda_actrecon=20.0,
        lqr_q_scale=10.0,
        n_trials=args.n_trials,
        n_steps=80,
        hold_steps=15,
        seed=args.seed,
        obs_seed=args.obs_seed,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
