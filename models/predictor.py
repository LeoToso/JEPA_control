"""MLP predictor f_theta: (z_t,a_t)->z_{t+1}_hat."""
from __future__ import annotations
import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    """N-layer MLP predictor.

    Input: concatenation of z_t (d) and action embedding a_t (d_u).
    Output: predicted next latent z_{t+1}_hat (d).
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=256, n_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        in_dim = latent_dim + action_dim
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
        return self.net(torch.cat([z, a], dim=-1))
