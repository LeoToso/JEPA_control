"""Shared end-to-end pipeline: generate data -> train the four configurations
-> design a latent controller for each -> evaluate closed-loop on the TRUE
system -> diagnostics + plots + a markdown summary table.

Configurations compared, exactly mirroring the structure of the cartpole
JEPA notes:

  * ``pred_only``     multistep prediction, no regularizer      (expect near-total collapse)
  * ``pred_sigreg``   multistep prediction + SIGReg              (the notes' "SIGReg" / "MS SIGReg" case)
  * ``pred_actrecon``  multistep prediction + multistep action reconstruction (the notes' "AR" / "MS AR" case)
  * ``oracle``          fixed, known-good encoder                (upper-bound reference, not a trained JEPA)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from jepa_lds.control import design_latent_controller, evaluate_controller
from jepa_lds.data import EpisodeBatch, generate_dataset, make_observation_model
from jepa_lds.diagnostics import (
    cosine_alignment_unstable_direction,
    eigenvalue_comparison,
    latent_norm_divergence,
    unstable_mode_retention,
)
from jepa_lds.models import FixedLinearEncoder
from jepa_lds.plotting import (
    plot_closed_loop_trajectories,
    plot_cosine_alignment,
    plot_eigenvalues,
    plot_norm_divergence,
    plot_summary_bars,
    plot_training_curves,
)
from jepa_lds.systems import LTISystem
from jepa_lds.train import TrainConfig, train_jepa


def run_experiment(
    system: LTISystem,
    out_dir: str,
    n_train_ep: int = 400,
    n_val_ep: int = 100,
    T: int = 10,
    x0_std: float = 0.03,
    action_std: float = 0.3,
    state_clip: float = 15.0,
    obs_dim_signal: int | None = None,
    n_distractor: int = 8,
    measurement_noise_std: float = 0.01,
    distractor_std: float = 1.0,
    latent_dim: int = 8,
    horizon: int = 4,
    outer_rounds: int = 25,
    inner_epochs: int = 6,
    lr: float = 1e-2,
    batch_size: int = 256,
    lambda_sigreg: float = 1.0,
    lambda_actrecon: float = 1.0,
    sigreg_directions: int = 16,
    success_threshold: float = 0.5,
    n_trials: int = 30,
    n_steps: int = 80,
    hold_steps: int = 15,
    lqr_q_scale: float = 10.0,
    seed: int = 0,
    obs_seed: int = 0,
    verbose: bool = True,
):
    """`obs_seed` fixes the random "sensor" matrix (the observation model)
    independently of `seed`: it represents a fixed physical sensing
    apparatus, not something that should be re-randomized on every training
    run. Varying `seed` alone (data + init) while keeping `obs_seed` fixed is
    what makes the multi-seed robustness comparison in the notebook/tests
    meaningful -- letting the sensor itself vary too adds an unrelated
    (and sometimes poorly-conditioned) extra source of variance."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n===== {system.name} =====")
    print(f"eigvals(A) = {system.eigvals()}  spectral_radius = {system.spectral_radius():.4f}")
    print(f"controllable = {system.is_controllable()}  stabilizable = {system.is_stabilizable()}")

    obs_model = make_observation_model(
        system,
        obs_dim_signal=obs_dim_signal,
        n_distractor=n_distractor,
        measurement_noise_std=measurement_noise_std,
        distractor_std=distractor_std,
        seed=obs_seed,
    )
    train_batch = generate_dataset(
        system, obs_model, n_train_ep, T, seed=seed,
        x0_std=x0_std, action_std=action_std, state_clip=state_clip,
    )
    val_batch = generate_dataset(
        system, obs_model, n_val_ep, T, seed=seed + 10_000,
        x0_std=x0_std, action_std=action_std, state_clip=state_clip,
    )
    print(f"train episodes={n_train_ep} val episodes={n_val_ep} T={T} obs_dim={obs_model.p}")

    # ---- report how non-Gaussian the true unstable mode is vs the stable
    # mode -- this is the mechanistic reason SIGReg fights against retaining it.
    x_flat = train_batch.x.reshape(-1, system.n)
    _w, _V, Vinv = system.modal_decomposition()
    idx_u = system.unstable_mode_index()
    idx_s = 1 - idx_u if system.n == 2 else None
    xi_u = np.real(x_flat @ Vinv[idx_u].conj())
    kurt_u = float(np.mean((xi_u - xi_u.mean()) ** 4) / (np.var(xi_u) ** 2 + 1e-12) - 3.0)
    print(f"excess kurtosis of true unstable modal coordinate (pooled over dataset): {kurt_u:.2f}")
    if idx_s is not None:
        xi_s = np.real(x_flat @ Vinv[idx_s].conj())
        kurt_s = float(np.mean((xi_s - xi_s.mean()) ** 4) / (np.var(xi_s) ** 2 + 1e-12) - 3.0)
        print(f"excess kurtosis of true stable modal coordinate (pooled over dataset):   {kurt_s:.2f}")

    configs = {
        "pred_only": TrainConfig(
            latent_dim=latent_dim, horizon=horizon, outer_rounds=outer_rounds, inner_epochs=inner_epochs, lr=lr, batch_size=batch_size,
            lambda_pred_1step=1.0, lambda_pred_ms=1.0, lambda_sigreg=0.0,
            lambda_actrecon_1step=0.0, lambda_actrecon_ms=0.0, seed=seed,
        ),
        "pred_sigreg": TrainConfig(
            latent_dim=latent_dim, horizon=horizon, outer_rounds=outer_rounds, inner_epochs=inner_epochs, lr=lr, batch_size=batch_size,
            lambda_pred_1step=1.0, lambda_pred_ms=1.0, lambda_sigreg=lambda_sigreg,
            lambda_actrecon_1step=0.0, lambda_actrecon_ms=0.0,
            sigreg_directions=sigreg_directions, seed=seed,
        ),
        "pred_actrecon": TrainConfig(
            latent_dim=latent_dim, horizon=horizon, outer_rounds=outer_rounds, inner_epochs=inner_epochs, lr=lr, batch_size=batch_size,
            lambda_pred_1step=1.0, lambda_pred_ms=1.0, lambda_sigreg=0.0,
            lambda_actrecon_1step=lambda_actrecon, lambda_actrecon_ms=lambda_actrecon, seed=seed,
        ),
    }

    encoders, predictors, histories = {}, {}, {}
    for name, cfg in configs.items():
        print(f"\n--- training {name} ---")
        enc, pred, _dec, hist = train_jepa(system, obs_model, train_batch, cfg, verbose=verbose)
        encoders[name], predictors[name], histories[name] = enc, pred, hist

    # oracle: fixed, known-good encoder recovering x (up to measurement
    # noise) from the observation's signal block; only the predictor is fit.
    print("\n--- fitting oracle (fixed faithful encoder) ---")
    oracle_encoder = FixedLinearEncoder(obs_model.oracle_decoder_weight(system.n))
    oracle_cfg = TrainConfig(
        latent_dim=system.n, horizon=horizon, outer_rounds=outer_rounds, inner_epochs=inner_epochs, lr=lr, batch_size=batch_size,
        lambda_pred_1step=1.0, lambda_pred_ms=1.0, lambda_sigreg=0.0,
        lambda_actrecon_1step=0.0, lambda_actrecon_ms=0.0, seed=seed,
    )
    oracle_encoder, oracle_predictor, _dec, oracle_hist = train_jepa(
        system, obs_model, train_batch, oracle_cfg,
        encoder=oracle_encoder, train_encoder=False, verbose=verbose,
    )
    encoders["oracle"], predictors["oracle"], histories["oracle"] = oracle_encoder, oracle_predictor, oracle_hist

    # ---- control synthesis + closed-loop evaluation + diagnostics ----
    summary = {}
    eval_results = {}
    latent_eigs_for_plot = {}
    for name in encoders:
        enc, pred = encoders[name], predictors[name]
        ctrl = design_latent_controller(pred, q_scale=lqr_q_scale)
        ev = evaluate_controller(
            system, obs_model, enc, ctrl["K_z"],
            n_trials=n_trials, n_steps=n_steps, x0_std=x0_std,
            success_threshold=success_threshold, hold_steps=hold_steps, seed=seed + 555,
        )
        r2 = unstable_mode_retention(system, enc, train_batch, val_batch)
        eig_cmp = eigenvalue_comparison(system, pred)
        latent_eigs_for_plot[name] = eig_cmp["latent_eigvals"]

        summary[name] = {
            "stabilizable_in_latent": ctrl["stabilizable"],
            "uncontrollable_unstable_latent_modes": [str(z) for z in ctrl["uncontrollable_unstable_modes"]],
            "latent_spectral_radius": ctrl["latent_spectral_radius"],
            "unstable_mode_retention_R2": r2,
            "success_rate": ev["success_rate"],
            "mean_fraction_stable": ev["mean_fraction_stable"],
            "final_state_distance_avg": ev["final_state_distance_avg"],
        }
        eval_results[name] = ev
        print(
            f"[{name:>14s}] stabilizable={ctrl['stabilizable']!s:5s} "
            f"R2(unstable mode)={r2:6.3f}  success={ev['success_rate']*100:5.1f}%  "
            f"frac_stable={ev['mean_fraction_stable']:.3f}  final_dist={ev['final_state_distance_avg']:.4f}"
        )

    # ---- plots ----
    plot_training_curves(histories, os.path.join(out_dir, "training_curves.pdf"))
    plot_eigenvalues(system.eigvals(), latent_eigs_for_plot, os.path.join(out_dir, "eigenvalues.pdf"))
    plot_closed_loop_trajectories(eval_results, os.path.join(out_dir, "closed_loop_trajectories.pdf"))
    plot_summary_bars(
        list(summary.keys()),
        [summary[k]["unstable_mode_retention_R2"] for k in summary],
        [summary[k]["success_rate"] for k in summary],
        os.path.join(out_dir, "summary_bars.pdf"),
    )

    rng = np.random.default_rng(seed + 999)
    x0_panel = x0_std * rng.standard_normal(system.n)
    for name in ("pred_sigreg", "pred_actrecon"):
        true_n, rec_n = latent_norm_divergence(
            system, obs_model, encoders[name], predictors[name], x0_panel, n_steps=T, rng=rng
        )
        plot_norm_divergence(
            true_n, rec_n, os.path.join(out_dir, f"norm_divergence_{name}.pdf"), title=f"{system.name}: {name}"
        )
        cos = cosine_alignment_unstable_direction(
            system, obs_model, encoders[name], predictors[name],
            n_steps=T, perturbation_scales=[0.02, 0.05, 0.1, 0.2], rng=rng,
        )
        plot_cosine_alignment(
            cos, os.path.join(out_dir, f"cosine_alignment_{name}.pdf"), title=f"{system.name}: {name}"
        )

    # ---- markdown + json summary ----
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(f"# {system.name}\n\n")
        f.write(f"true eigvals(A): {system.eigvals()}\n\n")
        f.write(f"spectral radius: {system.spectral_radius():.4f}  (open-loop unstable)\n\n")
        f.write("| config | stabilizable (latent) | unstable-mode R^2 | success rate | mean frac stable | final state dist |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, s in summary.items():
            f.write(
                f"| {name} | {s['stabilizable_in_latent']} | {s['unstable_mode_retention_R2']:.3f} | "
                f"{s['success_rate']*100:.1f}% | {s['mean_fraction_stable']:.3f} | {s['final_state_distance_avg']:.4f} |\n"
            )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary to {md_path}")

    return summary
