"""Save/load a trained (encoder, predictor, decoder) triple plus the
TrainConfig used to produce it, so a run's models can be reloaded later for
analysis (diagnostics, plotting, control synthesis) without retraining.
"""
from __future__ import annotations

import dataclasses
import os

import torch

from .models import LinearEncoder, LinearLatentPredictor, MultistepActionDecoder, OneStepActionDecoder
from .train import TrainConfig


def save_checkpoint(path: str, encoder, predictor, decoder, cfg: TrainConfig, extra: dict | None = None) -> None:
    """`extra` can hold anything picklable you want alongside the model,
    e.g. {"obs_dim": obs_model.p, "action_dim": system.m, "system_name": system.name}."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "encoder_state_dict": encoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "decoder_state_dict": decoder.state_dict() if decoder is not None else None,
        "decoder_class": type(decoder).__name__ if decoder is not None else None,
        "cfg": dataclasses.asdict(cfg),
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, obs_dim: int, action_dim: int):
    """Returns (encoder, predictor, decoder, cfg, extra). Reconstructs the
    model architecture from `cfg` (saved alongside the weights) plus the
    `obs_dim`/`action_dim` you supply -- these aren't stored in `cfg` since
    they come from the observation model / system, not the training
    hyperparameters, so pass the same values used to train the checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = TrainConfig(**ckpt["cfg"])

    encoder = LinearEncoder(obs_dim, cfg.latent_dim, bias=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    predictor = LinearLatentPredictor(cfg.latent_dim, action_dim)
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    predictor.eval()

    decoder = None
    if ckpt["decoder_state_dict"] is not None:
        if ckpt["decoder_class"] == "MultistepActionDecoder":
            decoder = MultistepActionDecoder(cfg.latent_dim, action_dim, cfg.horizon)
        elif ckpt["decoder_class"] == "OneStepActionDecoder":
            decoder = OneStepActionDecoder(cfg.latent_dim, action_dim)
        else:
            raise ValueError(f"unknown decoder class in checkpoint: {ckpt['decoder_class']!r}")
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        decoder.eval()

    return encoder, predictor, decoder, cfg, ckpt["extra"]
