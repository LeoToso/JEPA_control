"""Cross-Entropy Method planner in the learned latent space.

Works with either:
  - Linear dynamics:    z_{t+1} = A(z_t - z*) + B u_t + z* + c_offset
  - Nonlinear dynamics: z_{t+1} = predictor(z_t, action_encoder(u_t))

CEM avoids gradient computation entirely: it samples N action sequences,
rolls them all forward in a single batched pass, ranks by cost, and refits a
Gaussian to the top-K elites.  This sidesteps the B/w_u alignment problem that
hurts gradient-based planners: control authority over the unstable mode only
needs to exist in the forward dynamics, not in a gradient path through B.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch


class CEMLatentPlanner:
    """Cross-Entropy Method trajectory optimiser in latent space.

    Parameters
    ----------
    A, B, c_offset : linear-mode matrices (numpy).  If provided, uses the
        affine model z_{t+1} = A(z_t-z*) + B u_t + z* + c_offset.
    predictor, action_encoder : nonlinear-mode PyTorch modules.  Used when
        A is None.
    Q, R, Q_f : cost matrices (numpy).  R may be a scalar or (d_u, d_u) array.
    horizon, chunk_size : planning horizon and action-chunking size.
    n_samples : number of action sequences sampled per CEM iteration.
    n_elites  : top-K sequences used to refit the Gaussian.
    n_iter    : number of CEM refinement iterations per call to plan().
    init_std  : initial standard deviation of the action distribution.
    """

    def __init__(
        self,
        # ── linear mode ──────────────────────────────────────────────────
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        c_offset: Optional[np.ndarray] = None,
        # ── nonlinear mode ───────────────────────────────────────────────
        predictor=None,
        action_encoder=None,
        # ── cost ─────────────────────────────────────────────────────────
        Q: Optional[np.ndarray] = None,
        R=0.01,
        Q_f: Optional[np.ndarray] = None,
        # ── planning ─────────────────────────────────────────────────────
        horizon: int = 10,
        chunk_size: int = 1,
        n_samples: int = 500,
        n_elites: int = 50,
        n_iter: int = 5,
        init_std: float = 3.0,
        action_lb: float = -10.0,
        action_ub: float = 10.0,
        device=None,
    ):
        self.horizon    = horizon
        self.chunk_size = min(chunk_size, horizon)
        self.n_samples  = n_samples
        self.n_elites   = min(n_elites, n_samples)
        self.n_iter     = n_iter
        self.init_std   = init_std
        self.action_lb  = action_lb
        self.action_ub  = action_ub

        self._linear_mode = A is not None

        # Resolve device
        if device is None:
            if predictor is not None:
                device = next(predictor.parameters()).device
            else:
                device = torch.device('cpu')
        self.device = device

        if self._linear_mode:
            d = A.shape[0]
            self._A = torch.tensor(A, dtype=torch.float32, device=device)
            self._B = torch.tensor(
                B.reshape(d, -1), dtype=torch.float32, device=device
            )
            c = c_offset if c_offset is not None else np.zeros(d)
            self._c = torch.tensor(c, dtype=torch.float32, device=device)
            # Expose numpy copies for rollout.py compatibility
            self._A_np = A
            self._B_np = B.reshape(d, -1)
        else:
            self.predictor      = predictor
            self.action_encoder = action_encoder
            d = predictor.latent_dim
            self._A_np = np.eye(d)
            self._B_np = np.zeros((d, 1))

        if Q is None:
            Q = np.eye(d)
        if Q_f is None:
            Q_f = Q

        self._Q  = torch.tensor(Q,   dtype=torch.float32, device=device)
        self._Qf = torch.tensor(Q_f, dtype=torch.float32, device=device)

        # R can be scalar or matrix; store as scalar float for the 1-D action case
        R_arr = np.atleast_2d(R) if not np.isscalar(R) else np.array([[float(R)]])
        self._R_scalar = float(R_arr[0, 0])

    # ── dynamics ─────────────────────────────────────────────────────────────

    def _step(self, z: torch.Tensor, u: torch.Tensor,
              z_star: torch.Tensor) -> torch.Tensor:
        """Single-step dynamics.  z, z_star: (N, d).  u: (N, 1)."""
        if self._linear_mode:
            dz = z - z_star
            return dz @ self._A.T + u @ self._B.T + z_star + self._c
        else:
            with torch.no_grad():
                a = self.action_encoder(u)   # (N, d_a)
                return self.predictor(z, a)  # (N, d)

    # ── cost evaluation ───────────────────────────────────────────────────────

    def _rollout_cost(
        self,
        z0: torch.Tensor,       # (1, d)  current latent state
        z_star: torch.Tensor,   # (1, d)  target latent state
        U: torch.Tensor,        # (N, H)  action sequences
    ) -> torch.Tensor:          # (N,)    total costs
        N = U.shape[0]
        z = z0.expand(N, -1)    # (N, d)
        zs = z_star.expand(N, -1)

        costs = torch.zeros(N, device=self.device)
        for t in range(self.horizon):
            u_t = U[:, t:t+1]               # (N, 1)
            dz  = z - zs
            costs += ((dz @ self._Q) * dz).sum(-1)          # stage state cost
            costs += self._R_scalar * (u_t * u_t).squeeze(-1)  # control cost
            z = self._step(z, u_t, zs)

        dz_f = z - zs
        costs += ((dz_f @ self._Qf) * dz_f).sum(-1)        # terminal cost
        return costs

    # ── planning ─────────────────────────────────────────────────────────────

    def plan(
        self, z_t: np.ndarray, z_star: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Run CEM from z_t toward z_star.

        Returns
        -------
        actions  : list of chunk_size clipped action arrays, each shape (1,)
        pred_zs  : (H+1, d) latent trajectory under the final mean actions
        """
        z0 = torch.tensor(z_t,    dtype=torch.float32, device=self.device).unsqueeze(0)
        zs = torch.tensor(z_star, dtype=torch.float32, device=self.device).unsqueeze(0)

        mu    = torch.zeros(self.horizon, device=self.device)
        sigma = torch.full((self.horizon,), self.init_std, device=self.device)

        for _ in range(self.n_iter):
            eps = torch.randn(self.n_samples, self.horizon, device=self.device)
            U   = (mu + sigma * eps).clamp(self.action_lb, self.action_ub)

            costs     = self._rollout_cost(z0, zs, U)
            elite_idx = torch.argsort(costs)[: self.n_elites]
            U_elite   = U[elite_idx]

            mu    = U_elite.mean(0)
            sigma = U_elite.std(0).clamp(min=0.1)

        u_out = mu.clamp(self.action_lb, self.action_ub)

        # Collect trajectory under mean actions
        with torch.no_grad():
            z = z0.clone()
            traj = [z_t.copy()]
            for t in range(self.horizon):
                u_t = u_out[t:t+1].unsqueeze(-1)   # (1, 1)
                z   = self._step(z, u_t, zs)
                traj.append(z[0].cpu().numpy())

        actions = [np.array([float(u_out[k].cpu())]) for k in range(self.chunk_size)]
        return actions, np.array(traj)

    # ── compatibility shims for rollout.py ───────────────────────────────────

    @property
    def A(self) -> np.ndarray:
        return self._A_np

    @property
    def B(self) -> np.ndarray:
        return self._B_np

    def summary(self) -> str:
        mode = 'linear' if self._linear_mode else 'nonlinear'
        return (
            f"CEM-{mode}  H={self.horizon}  chunk={self.chunk_size}"
            f"  N={self.n_samples}  elites={self.n_elites}"
            f"  iter={self.n_iter}  σ0={self.init_std}"
            f"  lb={self.action_lb}  ub={self.action_ub}"
        )
