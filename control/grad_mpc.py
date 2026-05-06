"""Gradient-based latent MPC: optimises through the JEPA predictor directly.

Unlike LatentMPC (which uses a linearised A,B), this planner rolls out the
full nonlinear predictor and back-propagates cost gradients to find the optimal
action sequence. If the encoder captures theta nonlinearly (not captured by the
linear W probe), this can recover control performance that linear MPC misses.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch


class GradientLatentMPC:
    """Shooting MPC via gradient descent through the JEPA predictor.

    Dynamics  : z_{t+1} = predictor(z_t, action_encoder(u_t))
    Stage cost: ||z_k - z*||_Q^2 + ||u_k||_R^2
    Terminal  : ||z_H - z*||_{Q_f}^2

    Optimises u_0...u_{H-1} using Adam for `n_iter` steps.
    Action constraints enforced by clamping after each gradient step.
    """

    def __init__(
        self,
        predictor,          # MLPPredictor
        action_encoder,     # LinearActionEncoder / IdentityActionEncoder
        Q: np.ndarray,
        R: np.ndarray,
        Q_f: np.ndarray,
        horizon: int = 20,
        chunk_size: int = 1,
        action_lb: float = -10.0,
        action_ub: float = 10.0,
        lr: float = 0.05,
        n_iter: int = 40,
        device=None,
    ):
        if device is None:
            device = next(predictor.parameters()).device
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.horizon = horizon
        self.chunk_size = min(chunk_size, horizon)
        self.action_lb = action_lb
        self.action_ub = action_ub
        self.lr = lr
        self.n_iter = n_iter
        self.device = device

        # Pre-cast cost matrices to tensors.
        self.Q_t = torch.tensor(Q, dtype=torch.float32, device=device)
        self.R_t = torch.tensor(R, dtype=torch.float32, device=device)
        self.Qf_t = torch.tensor(Q_f, dtype=torch.float32, device=device)

        self._u_warm: Optional[torch.Tensor] = None  # warm-start from last plan

    # ------------------------------------------------------------------
    def _cost(self, z_init: torch.Tensor, z_star: torch.Tensor,
              u_seq: torch.Tensor) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """Roll out predictor and compute cost. Returns (scalar, pred_z list)."""
        self.predictor.eval()
        self.action_encoder.eval()

        z = z_init  # (1, d)
        cost = torch.zeros(1, device=self.device)
        traj = [z.detach().cpu().numpy()[0]]

        for k in range(self.horizon):
            u_k = torch.clamp(u_seq[k], self.action_lb, self.action_ub).unsqueeze(0)  # (1, m)
            a_k = self.action_encoder(u_k)       # (1, d_a)
            dz = z - z_star                       # (1, d)
            cost = cost + (dz @ self.Q_t @ dz.T).squeeze()
            cost = cost + (u_k @ self.R_t @ u_k.T).squeeze()
            z = self.predictor(z, a_k)            # (1, d)
            traj.append(z.detach().cpu().numpy()[0])

        dz_f = z - z_star
        cost = cost + (dz_f @ self.Qf_t @ dz_f.T).squeeze()
        return cost, traj

    # ------------------------------------------------------------------
    def plan(
        self, z_t: np.ndarray, z_star: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Optimise action sequence from z_t toward z_star.

        Returns
        -------
        actions  : list of chunk_size clipped action arrays, each (d_u,)
        pred_zs  : (H+1, d) predicted latent trajectory
        """
        z_init = torch.tensor(z_t, dtype=torch.float32,
                              device=self.device).unsqueeze(0)   # (1, d)
        z_star_t = torch.tensor(z_star, dtype=torch.float32,
                                device=self.device).unsqueeze(0)  # (1, d)

        d_u = self.R_t.shape[0]

        # Warm start: shift previous plan by chunk_size.
        if self._u_warm is not None and self._u_warm.shape[0] == self.horizon:
            u_seq = torch.cat([
                self._u_warm[self.chunk_size:],
                torch.zeros(self.chunk_size, d_u, device=self.device)
            ]).detach().requires_grad_(True)
        else:
            u_seq = torch.zeros(self.horizon, d_u, device=self.device,
                                requires_grad=True)

        optimizer = torch.optim.Adam([u_seq], lr=self.lr)

        for _ in range(self.n_iter):
            optimizer.zero_grad()
            cost, _ = self._cost(z_init, z_star_t, u_seq)
            cost.backward()
            optimizer.step()
            with torch.no_grad():
                u_seq.clamp_(self.action_lb, self.action_ub)

        # Re-run to get clean trajectory.
        with torch.no_grad():
            _, traj = self._cost(z_init, z_star_t, u_seq)

        self._u_warm = u_seq.detach()
        actions = [u_seq[k].detach().cpu().numpy()
                   for k in range(self.chunk_size)]
        return actions, np.array(traj)

    # ------------------------------------------------------------------
    @property
    def A(self):
        raise AttributeError("GradientLatentMPC has no linear A matrix")

    @property
    def B(self):
        raise AttributeError("GradientLatentMPC has no linear B matrix")

    def summary(self) -> str:
        return (
            f"GradientLatentMPC  H={self.horizon}  chunk={self.chunk_size}"
            f"  lr={self.lr}  n_iter={self.n_iter}"
            f"  lb={self.action_lb}  ub={self.action_ub}"
        )
