"""Autoencoder world model: ViT encoder + pixel decoder + MLP predictor.

Extends JEPAModel with a pixel decoder so that L_recon = MSE(decode(z), obs)
can be added to the training objective.  The decoder is not used at inference
time — CEM only calls model.encoder and model.predict(), which are inherited
unchanged from JEPAModel.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from models.jepa import JEPAModel, JEPAConfig
from models.decoder import PixelDecoder


class AEWorldModel(JEPAModel):
    """JEPAModel + PixelDecoder.

    The added decoder forces the encoder to retain visual information about
    the current observation, preventing the encoder collapse that afflicts
    pure-prediction objectives when the collapse prevention signal is weak.

    Only the online encoder's latent is decoded (not the target encoder's),
    so reconstruction gradients flow only through the online branch.
    """

    def __init__(self, config: JEPAConfig):
        super().__init__(config)
        # Decoder always reconstructs a single frame (3 channels), regardless of frame_stack
        _out_chans = config.in_chans // max(config.frame_stack, 1)
        self.decoder = PixelDecoder(
            latent_dim=config.latent_dim,
            image_size=config.image_size,
            in_chans=_out_chans,
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d) → obs_hat: (B, C, H, W) in [0, 1]"""
        return self.decoder(z)

    def reconstruct(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode then decode.  Returns (z, obs_hat)."""
        z = self.encoder(obs)
        return z, self.decoder(z)
