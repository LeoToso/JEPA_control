"""MLP predictor f_theta: (z_t,a_t)->z_{t+1}_hat."""
from __future__ import annotations
import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    """N-layer MLP predictor.

    window=1: input = [z_t, a_t]  (standard Markov)
    window>1: input = [z_{t-W+1},...,z_t, a_{t-W+1},...,a_t]  (pre-flattened by caller)
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=256, n_layers=2, window=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.window = window
        in_dim = window * (latent_dim + action_dim)
        layers = []
        for i in range(n_layers - 1):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z, a):
        # z: (B, W*d) or (B, d) when window=1
        # a: (B, W*d_a) or (B, d_a) when window=1
        return self.net(torch.cat([z, a], dim=-1))
