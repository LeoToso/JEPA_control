"""ViT visual encoder: (3, img_size, img_size) -> z in R^d."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, image_size=64, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Small ViT: patch-embed -> transformer blocks -> mean pool -> linear projection."""

    def __init__(self, image_size=64, patch_size=8, in_chans=3,
                 embed_dim=128, depth=4, num_heads=4, mlp_ratio=2.0,
                 latent_dim=32, dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        self.register_buffer('pos_embed', self._sinusoidal_pos(num_patches, embed_dim))
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, latent_dim)
        self._init_weights()

    @staticmethod
    def _sinusoidal_pos(n, d):
        pe = torch.zeros(1, n, d)
        pos = torch.arange(n).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[:d // 2])
        return pe

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):                          # x: (B, C, H, W)
        p = self.patch_embed(x)                    # (B, N, embed_dim)
        p = p + self.pos_embed
        tokens = self.blocks(p)                    # (B, N, embed_dim)
        tokens = self.norm(tokens)
        z = self.proj(tokens.mean(dim=1))          # mean pool → (B, latent_dim)
        return z

    @torch.no_grad()
    def encode(self, x):
        return self(x)

