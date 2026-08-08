"""Cartpole example trained exactly as specified in the paper draft: only
two configurations, each with every loss weight set to 1, and L_pred
consisting solely of the multistep recursive rollout term (no one-step
teacher-forced backbone) -- matching the two boxed optimization problems in
the LaTeX write-up literally, rather than the more heavily-tuned 4-config
sweep used for the README's headline numbers (run_example2_cartpole_linear.py).

    python experiments/run_example2_cartpole_paper.py [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from jepa_lds.checkpoint import save_checkpoint
from jepa_lds.control import design_latent_controller, evaluate_controller
from jepa_lds.data import generate_dataset, make_observation_model
from jepa_lds.diagnostics import eigenvalue_comparison, unstable_mode_retention
from jepa_lds.plotting import (
    plot_closed_loop_trajectories,
    plot_eigenvalues,
    plot_summary_bars,
    plot_training_curves,
)
from jepa_lds.systems import make_linearized_cartpole_system
from jepa_lds.train import TrainConfig, train_jepa


def _slug(name: str) -> str:
    return name.replace(" ", "_").replace("+", "").replace("__", "_")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obs-seed", type=int, default=0)
    p.add_argument("--n-train-ep", type=int, default=500)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--outer-rounds", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lambda-sigreg", type=float, default=5.0, help="weight on L_SIGReg in the L_pred+L_SIGReg config")
    p.add_argument("--lambda-actrecon", type=float, default=5.0, help="weight on L_act (multistep decoder) in the L_pred+L_act config")
    p.add_argument("--config", choices=["sigreg", "actrecon", "both"], default="both",
                    help="train only L_pred+L_SIGReg, only L_pred+L_act, or both (default)")
    p.add_argument("--threads", type=int, default=4, help="torch.set_num_threads -- tiny models, more is often worse on shared/many-core machines")
    p.add_argument("--log-every", type=int, default=10, help="print training progress every N outer rounds")
    p.add_argument(
        "--out-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "example2_cartpole_paper"),
    )
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

    # The two boxed problems from the LaTeX: L_pred is the multistep
    # rollout ONLY (lambda_pred_1step=0 -- no one-step backbone), L_act is
    # the multistep action decoder ONLY (lambda_actrecon_1step=0).
    # lambda_pred_ms is always 1; the second term's weight is CLI-configurable.
    all_configs = {
        "L_pred + L_SIGReg": TrainConfig(
            latent_dim=system.n, horizon=horizon, outer_rounds=args.outer_rounds, inner_epochs=6, lr=1e-2,
            batch_size=args.batch_size,
            lambda_pred_1step=0.0, lambda_pred_ms=1.0,
            lambda_sigreg=args.lambda_sigreg,
            lambda_actrecon_1step=0.0, lambda_actrecon_ms=0.0,
            seed=args.seed,
        ),
        "L_pred + L_act": TrainConfig(
            latent_dim=system.n, horizon=horizon, outer_rounds=args.outer_rounds, inner_epochs=6, lr=1e-2,
            batch_size=args.batch_size,
            lambda_pred_1step=0.0, lambda_pred_ms=1.0,
            lambda_sigreg=0.0,
            lambda_actrecon_1step=0.0, lambda_actrecon_ms=args.lambda_actrecon,
            seed=args.seed,
        ),
    }
    if args.config == "sigreg":
        configs = {"L_pred + L_SIGReg": all_configs["L_pred + L_SIGReg"]}
    elif args.config == "actrecon":
        configs = {"L_pred + L_act": all_configs["L_pred + L_act"]}
    else:
        configs = all_configs
    print(f"config={args.config}  lambda_sigreg={args.lambda_sigreg}  lambda_actrecon={args.lambda_actrecon}")

    os.makedirs(args.out_dir, exist_ok=True)
    encoders, predictors, histories = {}, {}, {}
    summary, eval_results, latent_eigs = {}, {}, {}
    for name, cfg in configs.items():
        print(f"\n--- training {name} ---")
        enc, pred, dec, hist = train_jepa(system, obs_model, train_batch, cfg, verbose=True, log_every=args.log_every)
        encoders[name], predictors[name], histories[name] = enc, pred, hist

        ckpt_path = os.path.join(args.out_dir, f"checkpoint_{_slug(name)}.pt")
        save_checkpoint(
            ckpt_path, enc, pred, dec, cfg,
            extra={"obs_dim": obs_model.p, "action_dim": system.m, "system_name": system.name,
                   "config_name": name, "trainer": "alternating"},
        )
        print(f"  saved checkpoint to {ckpt_path}")

        ctrl = design_latent_controller(pred)
        ev = evaluate_controller(
            system, obs_model, enc, ctrl["K_z"],
            n_trials=args.n_trials, n_steps=100, x0_std=0.03,
            success_threshold=0.5, hold_steps=20, seed=args.seed + 555,
        )
        r2 = unstable_mode_retention(system, enc, train_batch, val_batch)
        eig_cmp = eigenvalue_comparison(system, pred)
        latent_eigs[name] = eig_cmp["latent_eigvals"]
        eval_results[name] = ev
        summary[name] = {
            "stabilizable_in_latent": ctrl["stabilizable"],
            "unstable_mode_retention_R2": r2,
            "success_rate": ev["success_rate"],
            "mean_fraction_stable": ev["mean_fraction_stable"],
            "final_state_distance_avg": ev["final_state_distance_avg"],
        }
        print(
            f"[{name}] stabilizable={ctrl['stabilizable']}  R2={r2:.3f}  "
            f"success={ev['success_rate']*100:.1f}%  frac_stable={ev['mean_fraction_stable']:.3f}"
        )

    plot_training_curves(histories, os.path.join(args.out_dir, "training_curves.pdf"))
    plot_eigenvalues(system.eigvals(), latent_eigs, os.path.join(args.out_dir, "eigenvalues.pdf"))
    plot_closed_loop_trajectories(eval_results, os.path.join(args.out_dir, "closed_loop_trajectories.pdf"))
    plot_summary_bars(
        list(summary.keys()),
        [summary[k]["unstable_mode_retention_R2"] for k in summary],
        [summary[k]["success_rate"] for k in summary],
        os.path.join(args.out_dir, "summary_bars.pdf"),
    )

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# cartpole_linear -- paper-exact (2 configs, L_pred multistep-only)\n\n")
        f.write(f"lambda_sigreg={args.lambda_sigreg}  lambda_actrecon={args.lambda_actrecon}\n\n")
        f.write(f"true eigvals(A): {system.eigvals()}\n\n")
        f.write("| config | stabilizable (latent) | unstable-mode R^2 | success rate | mean frac stable |\n")
        f.write("|---|---|---|---|---|\n")
        for name, s in summary.items():
            f.write(
                f"| {name} | {s['stabilizable_in_latent']} | {s['unstable_mode_retention_R2']:.3f} | "
                f"{s['success_rate']*100:.1f}% | {s['mean_fraction_stable']:.3f} |\n"
            )
    print(f"\nWrote summary to {md_path}")


if __name__ == "__main__":
    main()
