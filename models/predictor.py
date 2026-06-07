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


class ResidualMLPPredictor(nn.Module):
    """Markovian predictor with the equilibrium fixed point enforced by construction.

        f(z, c) = z + g(z, c) - g(z*, 0)

    so f(z*, 0) = z* exactly, for any g. z* is tracked as a buffer, refreshed
    by the trainer via set_z_star() (e.g. from encoder(obs_eq) each step).
    Window=1 (Markovian) only.
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=256, n_layers=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.window = 1
        self.g = MLPPredictor(latent_dim, action_dim, hidden_dim, n_layers, window=1)
        self.register_buffer('z_star', torch.zeros(latent_dim))

    @torch.no_grad()
    def set_z_star(self, z_star: torch.Tensor) -> None:
        self.z_star.copy_(z_star.detach().to(self.z_star.device))

    def forward(self, z, a):
        # z: (B, d), a: (B, d_a)
        g_zc = self.g(z, a)
        z_star_b = self.z_star.unsqueeze(0).expand(z.shape[0], -1)        # (B, d)
        a_zero   = torch.zeros(z.shape[0], a.shape[-1], device=a.device, dtype=a.dtype)
        g_star   = self.g(z_star_b, a_zero)                               # (B, d)
        return z + g_zc - g_star
