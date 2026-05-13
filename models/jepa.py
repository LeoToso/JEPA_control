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
    vit_embed_dim: int = 128
    vit_depth: int = 4
    vit_num_heads: int = 4
    vit_mlp_ratio: float = 2.0
    # Predictor
    predictor_hidden_dim: int = 256

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
        self.encoder = ViTEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
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
            'vit_embed_dim': c.vit_embed_dim, 'vit_depth': c.vit_depth,
            'vit_num_heads': c.vit_num_heads,
        }


def make_jepa(variant='E-full', **kwargs) -> JEPAModel:
    cfg = JEPAConfig(variant=variant)
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return JEPAModel(cfg)
