"""Bounou et al. (NeurIPS 2021) CNN encoder: 6 conv-blocks → latent z."""
from __future__ import annotations
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """6 blocks of Conv3x3 → MaxPool(2) → BN → ReLU (last block: no ReLU).

    Channel progression: in_chans → 8 → 16 → 32 → 64 → 64 → latent_dim
    Spatial resolution:  64 → 32 → 16 → 8 → 4 → 2 → 1
    Output: (B, latent_dim) after flattening the 1×1 spatial feature map.
    """

    def __init__(self, in_chans: int = 6, latent_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        channels = [in_chans, 8, 16, 32, 64, 64, latent_dim]
        layers: list[nn.Module] = []
        for i in range(6):
            c_in, c_out = channels[i], channels[i + 1]
            is_last = (i == 5)
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.MaxPool2d(2),
            ]
            if not is_last:
                # BN + ReLU on blocks 0-4 (spatial ≥ 2×2, batch-safe)
                layers += [nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            # last block: no BN (spatial = 1×1 → BN fails for batch_size=1), no ReLU
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_chans, H, W) → z: (B, latent_dim)"""
        return self.net(x).flatten(1)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)
