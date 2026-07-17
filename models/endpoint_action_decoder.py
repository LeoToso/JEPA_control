"""Endpoint action decoder: reconstructs an action sequence from only z_start and z_end.

No intermediate latent states, predicted states, physical states, or observations
are passed to this decoder.  That restriction is the point: it tests whether the
endpoint latent representations retain sufficient information to recover the
action-induced trajectory between them.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class EndpointActionDecoder(nn.Module):
    """MLP that maps a pair of latent endpoints to an action sequence.

    Architecture (per spec):
        input
            → Linear(input_dim, hidden_dim) → LayerNorm(hidden_dim) → GELU
            → [Linear(hidden_dim, hidden_dim) → GELU] × (n_layers - 1)
            → Linear(hidden_dim, H_act * action_dim)
            → reshape(B, H_act, action_dim)

    Parameters
    ----------
    latent_dim : int
        Dimensionality of each endpoint latent vector (d for latent, 4 for physical state).
    action_dim : int
        Number of action dimensions (1 for cartpole).
    H_act : int
        Number of actions to reconstruct.
    hidden_dim : int
        Width of all hidden layers.
    n_layers : int
        Total number of Linear layers including the first and last.
        Must be ≥ 2 (first + last).
    use_delta : bool
        If True, input = concat(z_start, z_end, z_end - z_start)  → 3 * latent_dim.
        If False, input = concat(z_start, z_end)                    → 2 * latent_dim.
        The delta term is an endpoint-only feature — no intermediate states.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 1,
        H_act: int = 5,
        hidden_dim: int = 256,
        n_layers: int = 3,
        use_delta: bool = True,
    ):
        super().__init__()
        self.H_act = H_act
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_delta = use_delta
        if n_layers < 2:
            raise ValueError(f'n_layers must be ≥ 2, got {n_layers}')

        input_dim = latent_dim * (3 if use_delta else 2)

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, H_act * action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z_start: torch.Tensor, z_end: torch.Tensor) -> torch.Tensor:
        """Decode an action sequence from latent endpoints.

        Parameters
        ----------
        z_start : (B, latent_dim)   — latent at trajectory start
        z_end   : (B, latent_dim)   — latent at trajectory end

        Returns
        -------
        action_hat : (B, H_act, action_dim)
        """
        if self.use_delta:
            x = torch.cat([z_start, z_end, z_end - z_start], dim=-1)
        else:
            x = torch.cat([z_start, z_end], dim=-1)
        B = z_start.shape[0]
        return self.net(x).reshape(B, self.H_act, self.action_dim)
