"""JEPA world model: ViT encoder + MLP predictor + action encoder."""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Optional, Dict
import torch
import torch.nn as nn
from models.vit_encoder import ViTEncoder
from models.predictor import MLPPredictor
from models.action_encoder import make_action_encoder, IdentityActionEncoder


@dataclass
class JEPAConfig:
    variant: str = 'E-full'
    latent_dim: int = 32
    action_dim: int = 1
    action_latent_dim: int = 32
    action_encoder: str = 'none'   # 'none' | 'linear' | 'mlp'
    # ViT hyperparameters
    image_size: int = 64
    patch_size: int = 8
    in_chans: int = 3              # 3 * frame_stack (set by JEPAModel.__init__)
    frame_stack: int = 1           # number of consecutive frames stacked channel-wise
    vit_embed_dim: int = 128
    vit_depth: int = 4
    vit_num_heads: int = 4
    vit_mlp_ratio: float = 2.0
    # Predictor
    predictor_hidden_dim: int = 256
    predictor_n_layers: int = 2
    predictor_window: int = 1      # window size W for windowed MLP predictor

    @classmethod
    def from_dict(cls, d: dict) -> 'JEPAConfig':
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @property
    def action_encoder_type(self):
        return self.action_encoder


class JEPAModel(nn.Module):
    def __init__(self, config: JEPAConfig):
        super().__init__()
        self.config = config
        config.in_chans = config.frame_stack * 3
        self.encoder = ViTEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.vit_embed_dim,
            depth=config.vit_depth,
            num_heads=config.vit_num_heads,
            mlp_ratio=config.vit_mlp_ratio,
            latent_dim=config.latent_dim,
        )
        self.action_encoder = make_action_encoder(
            variant=config.action_encoder_type,
            action_dim=config.action_dim,
            latent_action_dim=config.action_latent_dim,
        )
        self.predictor = MLPPredictor(
            latent_dim=config.latent_dim,
            action_dim=self.action_encoder.latent_action_dim,
            hidden_dim=config.predictor_hidden_dim,
            n_layers=config.predictor_n_layers,
            window=config.predictor_window,
        )

        # EMA target encoder: same architecture as online encoder, not in optimizer.
        # Provides slowly-moving prediction targets that stabilise pred_loss training.
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.996) -> None:
        """EMA update: target ← momentum * target + (1 - momentum) * online."""
        for p_online, p_target in zip(self.encoder.parameters(),
                                       self.target_encoder.parameters()):
            p_target.data.mul_(momentum).add_(p_online.data, alpha=1 - momentum)

    @property
    def latent_dim(self):
        return self.config.latent_dim

    def predict(self, z_win: torch.Tensor, u_win: torch.Tensor) -> torch.Tensor:
        """Windowed prediction: z_{t+1} = f([z_{t-W+1},...,z_t], [u_{t-W+1},...,u_t]).

        Parameters
        ----------
        z_win : (B, W, d)  — window of W latent states, most recent last
        u_win : (B, W, 1)  — window of W raw actions (scalars), most recent last

        Returns
        -------
        z_next : (B, d)
        """
        B, W, d = z_win.shape
        # Encode each action in the window separately, then concatenate
        # u_win: (B, W, 1) -> reshape to (B*W, 1), encode, reshape back
        u_flat = u_win.reshape(B * W, 1)                   # (B*W, 1)
        a_flat = self.action_encoder(u_flat)                # (B*W, d_a)
        d_a = a_flat.shape[-1]
        a_win = a_flat.reshape(B, W, d_a)                  # (B, W, d_a)

        # Flatten window dimensions: (B, W, d) -> (B, W*d)
        z_flat = z_win.reshape(B, W * d)                   # (B, W*d)
        a_flat_cat = a_win.reshape(B, W * d_a)             # (B, W*d_a)

        return self.predictor(z_flat, a_flat_cat)          # (B, d)

    def forward(self, obs, action, next_obs):
        z_t   = self.encoder(obs)
        a_t   = self.action_encoder(action)
        z_hat = self.predictor(z_t, a_t)
        with torch.no_grad():
            z_next_sg = self.encoder(next_obs)
        z_next = self.encoder(next_obs)
        return {'z_t': z_t, 'a_t': a_t, 'z_hat': z_hat,
                'z_next': z_next, 'z_next_sg': z_next_sg}

    def get_config_dict(self):
        c = self.config
        return {
            'variant': c.variant, 'latent_dim': c.latent_dim,
            'action_dim': c.action_dim, 'action_latent_dim': c.action_latent_dim,
            'image_size': c.image_size, 'patch_size': c.patch_size,
            'frame_stack': c.frame_stack, 'in_chans': c.in_chans,
            'vit_embed_dim': c.vit_embed_dim, 'vit_depth': c.vit_depth,
            'vit_num_heads': c.vit_num_heads,
            'predictor_window': c.predictor_window,
        }


def make_jepa(variant='E-full', **kwargs) -> JEPAModel:
    cfg = JEPAConfig(variant=variant)
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return JEPAModel(cfg)
