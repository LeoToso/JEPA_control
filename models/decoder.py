"""Pixel decoder for autoencoder world-model baseline."""
from __future__ import annotations
import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    """Transposed-conv decoder: z ∈ R^d → (C, img_size, img_size) image.

    Architecture (for img_size=64, base_ch=128):
        Linear(d → base_ch * 8 * 8) → Reshape(base_ch, 8, 8)
        ConvTranspose2d(base_ch → 64, 4, stride=2, padding=1)  → (64, 16, 16)
        ConvTranspose2d(64 → 32,      4, stride=2, padding=1)  → (32, 32, 32)
        ConvTranspose2d(32 → in_chans, 4, stride=2, padding=1) → (C,  64, 64)
        Sigmoid (outputs in [0, 1])
    """

    def __init__(self, latent_dim: int = 32, image_size: int = 64, in_chans: int = 3,
                 base_ch: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.in_chans   = in_chans
        self.base_ch    = base_ch
        # Spatial base is always image_size // 8
        self.base_size  = image_size // 8

        self.fc = nn.Linear(latent_dim, base_ch * self.base_size * self.base_size)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_chans, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d) → img: (B, in_chans, image_size, image_size)"""
        h = self.fc(z).view(z.shape[0], self.base_ch, self.base_size, self.base_size)
        return self.deconv(h)
