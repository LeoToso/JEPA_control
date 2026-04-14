"""MLP predictor f_theta: (z_t,a_t)->z_{t+1}_hat."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPredictor(nn.Module):
    """2-layer MLP predictor.

    Input: concatenation of z_t (d) and action embedding a_t (d_u).
    Output: predicted next latent z_{t+1}_hat (d).
    """
    def __init__(self,latent_dim=32,action_dim=1,hidden_dim=256):
        super().__init__()
        self.latent_dim=latent_dim
        self.action_dim=action_dim
        self.net=nn.Sequential(
            nn.Linear(latent_dim+action_dim,hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim,latent_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,z,a):
        inp=torch.cat([z,a],dim=-1)
        return self.net(inp)
