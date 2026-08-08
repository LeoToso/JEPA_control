"""Load a saved (encoder, predictor, decoder) checkpoint, design the LQR
controller its learned latent dynamics imply, and evaluate it in closed loop
on the TRUE system -- with full control over the evaluation protocol.

    python experiments/evaluate_checkpoint.py \\
        --checkpoint results/example2_cartpole_paper_naive/checkpoint_L_pred_L_act.pt \\
        --success-threshold 0.5 --n-steps 100 --n-trials 30 --hold-steps 20

The checkpoint's `extra` metadata records which observation model (sensing
matrix + noise) was used at training time; older checkpoints saved before
that was added fall back to this example's known defaults (see
_OBS_DEFAULTS below), with a printed warning so it's never silent.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from jepa_lds.checkpoint import load_checkpoint
from jepa_lds.control import design_latent_controller, evaluate_controller
from jepa_lds.data import make_observation_model
from jepa_lds.systems import make_double_mode_system, make_linearized_cartpole_system

_SYSTEM_MAKERS = {
    "cartpole_linear": lambda: make_linearized_cartpole_system(dt=0.02),
    "double_mode": make_double_mode_system,
}
_OBS_DEFAULTS = {
    "cartpole_linear": dict(obs_dim_signal=10, n_distractor=10, measurement_noise_std=0.002, distractor_std=1.0, seed=0),
    "double_mode": dict(obs_dim_signal=6, n_distractor=8, measurement_noise_std=0.01, distractor_std=1.0, seed=0),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--success-threshold", type=float, default=0.5, help="||x_t|| must stay below this to count as 'stable'")
    p.add_argument("--n-steps", type=int, default=100, help="length of each closed-loop rollout")
    p.add_argument("--n-trials", type=int, default=30, help="number of random initial conditions to test")
    p.add_argument("--hold-steps", type=int, default=20, help="state must stay below threshold for the final N steps to count as a success")
    p.add_argument("--x0-std", type=float, default=0.03, help="std of the random initial state")
    p.add_argument("--q-scale", type=float, default=10.0, help="LQR Q = q_scale * I in latent space")
    p.add_argument("--r-scale", type=float, default=1.0, help="LQR R = r_scale * I")
    p.add_argument("--seed", type=int, default=0, help="seed for the closed-loop evaluation trials")
    p.add_argument("--threads", type=int, default=4, help="torch.set_num_threads")
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    raw = torch.load(args.checkpoint, map_location="cpu")
    extra_raw = raw.get("extra", {})
    system_name = extra_raw.get("system_name")
    if system_name not in _SYSTEM_MAKERS:
        raise ValueError(
            f"checkpoint's extra['system_name']={system_name!r} is missing or unrecognized "
            f"(known: {list(_SYSTEM_MAKERS)}) -- can't reconstruct the ground-truth system"
        )
    system = _SYSTEM_MAKERS[system_name]()

    obs_kwargs = {
        k: extra_raw[k]
        for k in ("obs_dim_signal", "n_distractor", "measurement_noise_std", "distractor_std", "obs_seed")
        if k in extra_raw
    }
    if "obs_seed" in obs_kwargs:
        obs_kwargs["seed"] = obs_kwargs.pop("obs_seed")
    if not obs_kwargs:
        print(f"[!] checkpoint predates saved obs-model params -- falling back to {system_name}'s known defaults")
        obs_kwargs = _OBS_DEFAULTS[system_name]
    obs_model = make_observation_model(system, **obs_kwargs)

    if "obs_dim" in extra_raw and obs_model.p != extra_raw["obs_dim"]:
        raise ValueError(
            f"reconstructed observation model has p={obs_model.p} but checkpoint was trained with "
            f"obs_dim={extra_raw['obs_dim']} -- obs-model reconstruction doesn't match training, "
            f"results would be meaningless. Check obs_kwargs above."
        )

    encoder, predictor, _decoder, cfg, extra = load_checkpoint(args.checkpoint, obs_dim=obs_model.p, action_dim=system.m)

    print(f"checkpoint: {args.checkpoint}")
    print(f"  system={system_name}  config={extra.get('config_name', '?')}  trainer={extra.get('trainer', '?')}")
    print(
        f"  cfg: latent_dim={cfg.latent_dim} horizon={cfg.horizon} "
        f"lambda_pred_1step={cfg.lambda_pred_1step} lambda_pred_ms={cfg.lambda_pred_ms} "
        f"lambda_sigreg={cfg.lambda_sigreg} lambda_actrecon_1step={cfg.lambda_actrecon_1step} "
        f"lambda_actrecon_ms={cfg.lambda_actrecon_ms}"
    )

    ctrl = design_latent_controller(predictor, q_scale=args.q_scale, r_scale=args.r_scale)
    print(f"\nstabilizable (latent) = {ctrl['stabilizable']}")
    print(f"true eigvals(A)      = {system.eigvals()}")
    print(f"learned eigvals(A_z) = {np.linalg.eigvals(ctrl['A_z'])}")
    if not ctrl["stabilizable"]:
        print("\nNo stabilizing latent controller could be designed: an unstable latent "
              "eigenvalue is uncontrollable (PBH test failed). Nothing to evaluate.")
        return

    ev = evaluate_controller(
        system, obs_model, encoder, ctrl["K_z"],
        n_trials=args.n_trials, n_steps=args.n_steps, x0_std=args.x0_std,
        success_threshold=args.success_threshold, hold_steps=args.hold_steps, seed=args.seed,
    )
    print(
        f"\nsuccess_rate={ev['success_rate'] * 100:.1f}%  "
        f"mean_fraction_stable={ev['mean_fraction_stable']:.3f}  "
        f"final_state_distance_avg={ev['final_state_distance_avg']:.4f}"
    )
    print(
        f"(evaluated with n_trials={args.n_trials}, n_steps={args.n_steps}, "
        f"success_threshold={args.success_threshold}, hold_steps={args.hold_steps})"
    )


if __name__ == "__main__":
    main()
