"""Bounou et al. (NeurIPS 2021) symmetric CNN decoder: latent z → pixel frame."""
from __future__ import annotations
import torch
import torch.nn as nn


class CNNDecoder(nn.Module):
    """Symmetric reverse of CNNEncoder: 6 transposed-conv blocks.

    Channel progression: latent_dim → 64 → 64 → 32 → 16 → 8 → out_chans
    Spatial resolution:  1 → 2 → 4 → 8 → 16 → 32 → 64
    Output: (B, out_chans, 64, 64) in [0, 1] via Sigmoid on final block.
    """

    def __init__(self, latent_dim: int = 8, out_chans: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        channels = [latent_dim, 64, 64, 32, 16, 8, out_chans]
        layers: list[nn.Module] = []
        for i in range(6):
            c_in, c_out = channels[i], channels[i + 1]
            is_last = (i == 5)
            layers.append(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2,
                                   padding=1, output_padding=1)
            )
            if not is_last:
                layers += [nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            else:
                layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → img: (B, out_chans, 64, 64)"""
        return self.net(z.unsqueeze(-1).unsqueeze(-1))
