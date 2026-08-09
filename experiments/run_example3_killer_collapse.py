"""Example 3: a minimal, deliberately UNDER-complete-latent construction
where L_pred + L_SIGReg reliably collapses the unstable mode -- with
L_pred driven to (near) exactly zero regardless of the prediction horizon
H -- while L_pred + L_act (action reconstruction) reliably retains it, on
the exact same data.

THE CONSTRUCTION
-----------------
Same 2-state `double_mode` system used elsewhere in this repo (one real
unstable eigenvalue, one real stable eigenvalue, rotated so the modal
directions aren't axis-aligned with the raw state). Two things make this
"killer":

1. `--latent-dim 1`, deliberately SMALLER than the true state dimension
   (2). Since the system is diagonalizable with distinct real eigenvalues
   and the dynamics are exactly noiseless, the ONLY 1-D linear encodings
   that achieve EXACTLY ZERO multistep prediction loss, for ANY horizon H,
   are the two eigenspaces themselves -- any other direction isn't
   forward-invariant under the dynamics and picks up nonzero error. So
   L_pred is exactly tied at 0 between "keep only the unstable mode" and
   "keep only the stable mode", for every H: L_pred cannot express any
   preference between them at all, by construction.

2. The initial state is drawn with the STABLE modal coordinate given a
   much LARGER standard deviation than the unstable one
   (`--sigma-stable` >> `--sigma-unstable`, via
   `make_modal_contaminated_x0_sampler` with `contamination_prob=0`, i.e.
   a plain per-coordinate Gaussian x0 with asymmetric scales -- no heavy
   tails needed for this construction). This variance asymmetry biases
   the alternating scheme's closed-form predictor fit toward the
   higher-variance (stable) direction from the very first outer round.

With BOTH ingredients, SIGReg reliably (empirically: the large majority of
random seeds, and effectively all of them at a moderate lambda_sigreg)
collapses the unstable mode -- while action reconstruction reliably keeps
it, because reconstructing the action sequence from a latent window
structurally rewards retaining whichever mode's response to actions
AMPLIFIES over the window (the unstable one) rather than decays (the
stable one), independent of any variance imbalance in the data.

This script reruns each configuration across `--n-seeds` random
initializations (holding the DATA fixed across seeds, varying only the
encoder's random init) at each horizon in `--horizons`, and reports the
fraction of seeds whose learned encoder ends up collapsing the unstable
mode -- turning the underlying claim ("this is the reliable, structural
outcome, not a fluke of one lucky/unlucky seed") into a directly
falsifiable, reproducible number.

CAVEAT on episode length vs. horizon: the "stable mode has larger pooled
variance" property only holds within a bounded time window -- the unstable
mode's variance grows like lambda_u^(2t) and eventually overtakes the
stable mode's shrinking lambda_s^(2t) variance regardless of how much
smaller its initial condition was (at the default sigma values, this
crossover is around t~7-8). SIGReg's marginal is computed by pooling z
over the WHOLE episode of length T, not just the H-step training window,
so once T grows past the crossover point the pooled-variance asymmetry
this construction relies on gets muddied. Keep `--T` comfortably above
`max(--horizons)` but not so large that it approaches the crossover
(defaults: T=5, horizons up to 4 -- validated empirically at these
settings across 15 random seeds each).

    python experiments/run_example3_killer_collapse.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from jepa_lds.checkpoint import save_checkpoint
from jepa_lds.control import design_latent_controller, evaluate_controller
from jepa_lds.data import generate_dataset, make_modal_contaminated_x0_sampler, make_observation_model
from jepa_lds.diagnostics import unstable_mode_retention
from jepa_lds.losses import sigreg_loss
from jepa_lds.plotting import plot_collapse_rate_sweep
from jepa_lds.systems import make_double_mode_system
from jepa_lds.train import TrainConfig, train_jepa


def _static_sigreg_preference(system, train_batch) -> None:
    """Prints SIGReg's own raw loss for each modal coordinate ALONE,
    rescaled to unit variance -- a training-free sanity check of which
    direction SIGReg analytically prefers, independent of any optimization
    dynamics."""
    _w, _V, Vinv = system.modal_decomposition()
    idx_u = system.unstable_mode_index()
    idx_s = 1 - idx_u
    x_flat = train_batch.x.reshape(-1, system.n)
    xi_u = np.real(x_flat @ Vinv[idx_u].conj())
    xi_s = np.real(x_flat @ Vinv[idx_s].conj())
    torch.manual_seed(0)
    for name, xi in [("unstable", xi_u), ("stable", xi_s)]:
        z = torch.tensor(xi / xi.std(), dtype=torch.float32).unsqueeze(1)
        loss = sigreg_loss(z, n_directions=16).item()
        print(f"  static sigreg_loss({name} modal coord, rescaled to unit variance) = {loss:.5f}")


def _run_one(system, obs_model, train_batch, val_batch, cfg: TrainConfig):
    enc, pred, dec, hist = train_jepa(system, obs_model, train_batch, cfg, verbose=False)
    r2 = unstable_mode_retention(system, enc, train_batch, val_batch)
    return enc, pred, dec, hist, r2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["sigreg", "actrecon", "both"], default="both")
    p.add_argument("--latent-dim", type=int, default=1, help="deliberately < system.n=2 -- forces the L_pred=0 tie between the two eigenspaces")
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4],
                    help="must all be < --T; the variance-asymmetry construction is validated for T=5, H<=4 -- "
                         "at longer T, the unstable mode's exponential growth eventually overtakes the stable "
                         "mode's pooled variance too (a crossover around t~7-8 at the default sigma values), "
                         "which can muddy the effect. Keep T > max(horizons) with a comfortable margin.")
    p.add_argument("--n-seeds", type=int, default=15, help="random encoder-init reruns per (config, horizon), data held fixed")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--collapse-threshold", type=float, default=0.5, help="unstable_mode_retention R^2 below this counts as 'collapsed'")

    p.add_argument("--lambda-sigreg", type=float, default=25.0)
    p.add_argument("--lambda-actrecon", type=float, default=25.0)
    p.add_argument("--sigma-stable", type=float, default=0.3, help="std of the stable modal coordinate's initial condition")
    p.add_argument("--sigma-unstable", type=float, default=0.01, help="std of the unstable modal coordinate's initial condition")
    p.add_argument("--contamination-prob", type=float, default=0.0, help="optional: also make the unstable coordinate's rare tail heavier (0 = plain Gaussian, matching the validated construction)")
    p.add_argument("--sigma-unstable-large", type=float, default=None, help="only used if --contamination-prob > 0; defaults to --sigma-unstable (no contamination)")

    p.add_argument("--T", type=int, default=5, help="episode length (must be > max(horizons))")
    p.add_argument("--n-train-ep", type=int, default=400)
    p.add_argument("--n-val-ep", type=int, default=100)
    p.add_argument("--action-std", type=float, default=0.05, help="small nonzero actions -- needed for action reconstruction to have signal, negligible enough to preserve the variance asymmetry")
    p.add_argument("--obs-dim-signal", type=int, default=6)
    p.add_argument("--n-distractor", type=int, default=0)
    p.add_argument("--measurement-noise-std", type=float, default=0.0, help="noiseless by design -- keeps L_pred=0 exactly achievable")

    p.add_argument("--outer-rounds", type=int, default=60)
    p.add_argument("--inner-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-2)

    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--obs-seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--out-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results", "example3_killer_collapse"),
    )
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)

    system = make_double_mode_system()
    if args.T <= max(args.horizons):
        raise SystemExit(f"--T={args.T} must be > max(--horizons)={max(args.horizons)}")

    obs_model = make_observation_model(
        system, obs_dim_signal=args.obs_dim_signal, n_distractor=args.n_distractor,
        measurement_noise_std=args.measurement_noise_std, seed=args.obs_seed,
    )
    sigma_unstable_large = args.sigma_unstable_large if args.sigma_unstable_large is not None else args.sigma_unstable
    x0_sampler = make_modal_contaminated_x0_sampler(
        system, sigma_stable=args.sigma_stable, sigma_unstable_small=args.sigma_unstable,
        sigma_unstable_large=sigma_unstable_large, contamination_prob=args.contamination_prob,
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

    def excess_kurtosis(v):
        v = v - v.mean()
        return float(np.mean(v**4) / (np.var(v) ** 2 + 1e-12) - 3.0)

    print(f"pooled var(unstable modal coord)={xi_u.var():.6f}  var(stable modal coord)={xi_s.var():.6f}")
    print(f"pooled excess kurtosis: unstable={excess_kurtosis(xi_u):.3f}  stable={excess_kurtosis(xi_s):.3f}")
    _static_sigreg_preference(system, train_batch)

    all_configs = {
        "sigreg": dict(lambda_sigreg=args.lambda_sigreg, lambda_actrecon_1step=0.0, lambda_actrecon_ms=0.0),
        "actrecon": dict(lambda_sigreg=0.0, lambda_actrecon_1step=0.0, lambda_actrecon_ms=args.lambda_actrecon),
    }
    configs = all_configs if args.config == "both" else {args.config: all_configs[args.config]}

    collapse_rates: dict[str, list[float]] = {name: [] for name in configs}
    summary = {}
    for horizon in args.horizons:
        print(f"\n=== horizon H={horizon} ===")
        for name, extra_cfg in configs.items():
            n_collapsed = 0
            r2s = []
            best = None  # (r2, enc, pred, dec, cfg) for the seed-start run, saved as the representative checkpoint
            for i in range(args.n_seeds):
                seed = args.seed_start + i
                cfg = TrainConfig(
                    latent_dim=args.latent_dim, horizon=horizon, outer_rounds=args.outer_rounds,
                    inner_epochs=args.inner_epochs, batch_size=args.batch_size, lr=args.lr,
                    lambda_pred_1step=0.0, lambda_pred_ms=1.0, seed=seed, **extra_cfg,
                )
                enc, pred, dec, hist, r2 = _run_one(system, obs_model, train_batch, val_batch, cfg)
                r2s.append(r2)
                collapsed = r2 < args.collapse_threshold
                n_collapsed += int(collapsed)
                if seed == args.seed_start:
                    best = (enc, pred, dec, cfg)
            rate = n_collapsed / args.n_seeds
            collapse_rates[name].append(rate)
            summary[f"{name}_H{horizon}"] = {
                "collapse_rate": rate, "mean_R2": float(np.mean(r2s)), "R2_values": r2s,
            }
            print(
                f"  [{name:9s}] collapsed unstable mode in {n_collapsed}/{args.n_seeds} seeds "
                f"(mean R2={np.mean(r2s):.3f}, R2 range=[{min(r2s):.3f}, {max(r2s):.3f}])"
            )

            enc, pred, dec, cfg = best
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_{name}_H{horizon}.pt")
            save_checkpoint(
                ckpt_path, enc, pred, dec, cfg,
                extra={"obs_dim": obs_model.p, "action_dim": system.m, "system_name": system.name,
                       "config_name": f"{name} (killer collapse, H={horizon})", "trainer": "alternating",
                       "obs_dim_signal": args.obs_dim_signal, "n_distractor": args.n_distractor,
                       "measurement_noise_std": args.measurement_noise_std, "distractor_std": 1.0,
                       "obs_seed": args.obs_seed},
            )

    plot_path = os.path.join(args.out_dir, "collapse_rate_sweep.pdf")
    plot_collapse_rate_sweep(
        args.horizons, collapse_rates, plot_path,
        title="Unstable-mode collapse rate vs. prediction horizon (double_mode, latent_dim=1)",
    )
    print(f"\nwrote {plot_path}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Example 3: killer collapse construction (double_mode, latent_dim=1)\n\n")
        f.write(f"true eigvals(A): {system.eigvals()}\n\n")
        f.write(f"sigma_stable={args.sigma_stable}  sigma_unstable={args.sigma_unstable}  n_seeds={args.n_seeds}\n\n")
        f.write("| config | horizon | collapse rate | mean R^2 |\n")
        f.write("|---|---|---|---|\n")
        for horizon in args.horizons:
            for name in configs:
                s = summary[f"{name}_H{horizon}"]
                f.write(f"| {name} | {horizon} | {s['collapse_rate']*100:.0f}% | {s['mean_R2']:.3f} |\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
