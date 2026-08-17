"""Encoder / latent predictor / action-decoder for the toy JEPA world model.

Everything is kept strictly linear, matching the user's "linear setting":
the encoder is a single linear layer, the latent predictor is an explicit
linear state-space map z_{t+1} = A_z z_t + B_z a_t (so (A_z, B_z) can be read
off exactly, with no system-identification step needed), and the action
decoder is a single linear layer over a concatenated latent window.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(obs_dim, latent_dim, bias=bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.linear(y)


class FixedLinearEncoder(nn.Module):
    """A non-trainable encoder z = W y. Used to build the "oracle" reference
    baseline (a faithful, known-good encoding of the true state)."""

    def __init__(self, W: np.ndarray):
        super().__init__()
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.W.T


class LinearLatentPredictor(nn.Module):
    """z_{t+1} = A_z z_t + B_z a_t."""

    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(action_dim, latent_dim, bias=False)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.A(z) + self.B(a)

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """actions: (batch, H, action_dim). Returns latents (batch, H+1,
        latent_dim), non-teacher-forced (recursive) with z[:, 0] = z0."""
        zs = [z0]
        z = z0
        for t in range(actions.shape[1]):
            z = self.forward(z, actions[:, t])
            zs.append(z)
        return torch.stack(zs, dim=1)

    @torch.no_grad()
    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        return self.A.weight.detach().cpu().numpy(), self.B.weight.detach().cpu().numpy()


class OneStepActionDecoder(nn.Module):
    """Reconstructs a_t from the concatenated pair (z_t, z_{t+1})."""

    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim * 2, action_dim)

    def forward(self, z_pair: torch.Tensor) -> torch.Tensor:
        return self.linear(z_pair)


class MultistepActionDecoder(nn.Module):
    """Reconstructs the action sequence a_t..a_{t+H-1} from the latent window
    z_t..z_{t+H} (concatenated). This is what forces the encoder to retain
    the actuated (hence, for a stabilizable-but-open-loop-unstable system,
    the unstable-and-controllable) subspace of the true state."""

    def __init__(self, latent_dim: int, action_dim: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.linear = nn.Linear(latent_dim * (horizon + 1), action_dim * horizon)

    def forward(self, z_window: torch.Tensor) -> torch.Tensor:
        b = z_window.shape[0]
        out = self.linear(z_window.reshape(b, -1))
        return out.reshape(b, self.horizon, self.action_dim)


class EndpointActionDecoder(nn.Module):
    """Reconstructs the action sequence a_t..a_{t+H-1} from ONLY the two
    endpoint latents (z_t, z_{t+H}), concatenated -- no intermediate latent
    in the window is ever passed to the decoder. Same single-linear-layer
    depth as `MultistepActionDecoder`, just a narrower input."""

    def __init__(self, latent_dim: int, action_dim: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.linear = nn.Linear(latent_dim * 2, action_dim * horizon)

    def forward(self, z_endpoints: torch.Tensor) -> torch.Tensor:
        b = z_endpoints.shape[0]
        out = self.linear(z_endpoints.reshape(b, -1))
        return out.reshape(b, self.horizon, self.action_dim)
