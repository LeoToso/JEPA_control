"""Shared helper for reconstructing the ground-truth system + observation
model that a saved checkpoint was trained with, so analysis scripts
(`probe_local_stability.py`, `probe_eigenvector_alignment_grid.py`) don't
each duplicate the same `_SYSTEM_MAKERS`/`_OBS_DEFAULTS` bookkeeping.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from jepa_lds.checkpoint import load_checkpoint
from jepa_lds.data import make_observation_model
from jepa_lds.systems import make_double_mode_system, make_linearized_cartpole_system

_SYSTEM_MAKERS = {
    "cartpole_linear": lambda: make_linearized_cartpole_system(dt=0.02),
    "double_mode": make_double_mode_system,
    # Example 2 (synthetic, stable): same construction as "double_mode" but
    # BOTH modes stable (unstable_eig=0.25 < 1, stable_eig=0.85 unchanged)
    # -- system is open-loop stable, so a collapsed mode no longer breaks
    # control.
    "double_mode_stable": lambda: make_double_mode_system(unstable_eig=0.25, name="double_mode_stable"),
}
_OBS_DEFAULTS = {
    "cartpole_linear": dict(obs_dim_signal=10, n_distractor=10, measurement_noise_std=0.002, distractor_std=1.0, seed=0),
    "double_mode": dict(obs_dim_signal=6, n_distractor=8, measurement_noise_std=0.01, distractor_std=1.0, seed=0),
    "double_mode_stable": dict(obs_dim_signal=6, n_distractor=0, measurement_noise_std=0.0, distractor_std=1.0, seed=0),
}


def load_checkpoint_with_env(path: str):
    """Loads a checkpoint and reconstructs the (system, obs_model) it was
    trained on from its `extra` metadata, falling back to this example's
    known defaults (with a printed warning) for older checkpoints that
    predate saving the obs-model params.

    Returns (system, obs_model, encoder, predictor, decoder, cfg, extra).
    """
    raw = torch.load(path, map_location="cpu", weights_only=True)
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

    encoder, predictor, decoder, cfg, extra = load_checkpoint(path, obs_dim=obs_model.p, action_dim=system.m)
    return system, obs_model, encoder, predictor, decoder, cfg, extra
