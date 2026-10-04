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
        predictor_window: int = 1,
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
        warm_start_sigma: float = 0.5,
        action_lb: float = -10.0,
        action_ub: float = 10.0,
        action_scale: float = 1.0,
        action_dim: int = 1,
        device=None,
    ):
        self._action_dim      = int(action_dim)
        self.horizon          = horizon
        self.chunk_size       = min(chunk_size, horizon)
        self.n_samples        = n_samples
        self.n_elites         = min(n_elites, n_samples)
        self.n_iter           = n_iter
        self.init_std         = init_std
        self.warm_start_sigma = warm_start_sigma
        self.action_lb        = action_lb
        self.action_ub        = action_ub
        self._action_scale    = float(action_scale)
        self._prev_mu: Optional[torch.Tensor] = None

        self._linear_mode = A is not None
        self._predictor_window = predictor_window
        # History for nonlinear windowed mode (reset on reset())
        self._z_hist = None   # list of W tensors (1, d)
        self._u_hist = None   # list of W-1 tensors (1, action_dim) -- past chosen actions

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
            self._B_np = np.zeros((d, self._action_dim))

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
        """Single-step dynamics.  z, z_star: (N, d).  u: (N, 1).
        For nonlinear W=1 (Markov) case only.  Windowed nonlinear uses _step_windowed."""
        if self._linear_mode:
            dz = z - z_star
            return dz @ self._A.T + u @ self._B.T + z_star + self._c
        else:
            # Markov (W=1) nonlinear path. Accept either the complete JEPAModel
            # or the historical raw predictor + action_encoder pair.
            u_norm = u / self._action_scale
            with torch.no_grad():
                if hasattr(self.predictor, 'predict'):
                    return self.predictor.predict(
                        z.unsqueeze(1), u_norm.unsqueeze(1))
                a = self.action_encoder(u_norm)                    # (N, d_a)
                return self.predictor(z, a)                        # (N, d)

    def _step_windowed(self, z_win: torch.Tensor, u_win: torch.Tensor) -> torch.Tensor:
        """Windowed nonlinear step.
        z_win: (N, W, d)  u_win: (N, W, 1)  — u_win contains RAW actions.
        Returns z_next: (N, d)
        Assumes self.predictor is a JEPAModel (has .predict()) or MLPPredictor directly.
        Model was trained on normalized actions; divide by action_scale before passing.
        """
        u_norm = u_win / self._action_scale   # normalize: raw → model units
        with torch.no_grad():
            # Check if predictor has a predict() method (JEPAModel) or is raw MLPPredictor
            if hasattr(self.predictor, 'predict'):
                return self.predictor.predict(z_win, u_norm)
            else:
                # Raw MLPPredictor: encode actions then call predictor
                N, W, d = z_win.shape
                u_flat = u_norm.reshape(N * W, -1)   # (N*W, action_dim)
                a_flat = self.action_encoder(u_flat)
                d_a = a_flat.shape[-1]
                a_win = a_flat.reshape(N, W, d_a)
                z_flat = z_win.reshape(N, W * d)
                a_flat_cat = a_win.reshape(N, W * d_a)
                return self.predictor(z_flat, a_flat_cat)

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

        W = self._predictor_window

        # For windowed nonlinear mode, initialize a window buffer per trajectory
        if not self._linear_mode and W > 1:
            # Build initial z window: W-1 history entries + current z0 as last entry.
            # _z_hist stores W entries ending at the PREVIOUS z, so we must replace the
            # last slot with z0 (actual current state) so the first prediction computes
            # f([z_{t-W+2},...,z_t, z_{t+1}], u_0) = z_{t+2}  (correct next state).
            # Without this, the first prediction gives z_{t+1} (already known), making
            # u_0 effectively unconstrained and the plan one step stale.
            if self._z_hist is not None:
                z_win_list = ([h.expand(N, -1) for h in self._z_hist[-(W-1):]]
                              + [z])          # (W-1) history + current z0
            else:
                z_win_list = [z] * W  # cold start: all z0

            if self._u_hist is not None:
                # _u_hist is a list of W-1 tensors each (1, action_dim)
                u_win_list = [h.expand(N, -1) for h in self._u_hist]
            else:
                u_win_list = [torch.zeros(N, self._action_dim, device=self.device)] * (W - 1)

        costs = torch.zeros(N, device=self.device)
        for t in range(self.horizon):
            u_t = U[:, t]                   # (N, action_dim)
            dz  = z - zs
            costs += ((dz @ self._Q) * dz).sum(-1)          # stage state cost
            costs += self._R_scalar * (u_t * u_t).sum(-1)   # control cost (sum over action_dim)

            if self._linear_mode:
                z = self._step(z, u_t, zs)
            elif W > 1:
                # Build (N, W, d) and (N, W, 1) window tensors
                z_win_tensor = torch.stack(z_win_list[-W:], dim=1)      # (N, W, d)
                u_full_list = u_win_list[-(W-1):] + [u_t]               # W entries
                u_win_tensor = torch.stack(u_full_list, dim=1)          # (N, W, 1)
                z_new = self._step_windowed(z_win_tensor, u_win_tensor)  # (N, d)
                z_win_list.append(z_new)
                u_win_list.append(u_t)
                z = z_new
            else:
                z = self._step(z, u_t, zs)

        dz_f = z - zs
        costs += ((dz_f @ self._Qf) * dz_f).sum(-1)        # terminal cost
        return costs

    # ── planning ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset warm-start state; call between episodes."""
        self._prev_mu = None
        self._z_hist = None
        self._u_hist = None

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

        # Warm-start: shift previous optimal sequence by one step.
        # Without warm-starting, every step starts from mu=0 σ=3 and converges
        # to bang-bang solutions; warm-starting gives smoother, more consistent plans.
        _adim = self._action_dim
        if self._prev_mu is not None:
            mu    = torch.cat([self._prev_mu[1:],
                               torch.zeros(1, _adim, device=self.device)])
            sigma = torch.full((self.horizon, _adim), self.warm_start_sigma,
                               device=self.device)
        else:
            mu    = torch.zeros(self.horizon, _adim, device=self.device)
            sigma = torch.full((self.horizon, _adim), self.init_std, device=self.device)

        for _ in range(self.n_iter):
            eps = torch.randn(self.n_samples, self.horizon, _adim, device=self.device)
            U   = (mu + sigma * eps).clamp(self.action_lb, self.action_ub)

            costs     = self._rollout_cost(z0, zs, U)
            elite_idx = torch.argsort(costs)[: self.n_elites]
            U_elite   = U[elite_idx]

            mu    = U_elite.mean(0)
            sigma = U_elite.std(0).clamp(min=0.05)

        self._prev_mu = mu.detach()
        u_out = mu.clamp(self.action_lb, self.action_ub)

        W = self._predictor_window

        # Collect trajectory under mean actions
        with torch.no_grad():
            z = z0.clone()
            traj = [z_t.copy()]
            if not self._linear_mode and W > 1:
                # Same window fix as _rollout_cost: W-1 history + current z0
                if self._z_hist is not None:
                    z_win_list = ([h.clone() for h in self._z_hist[-(W-1):]]
                                  + [z.clone()])
                else:
                    z_win_list = [z.clone()] * W
                if self._u_hist is not None:
                    u_win_list = [h.clone() for h in self._u_hist]
                else:
                    u_win_list = [torch.zeros(1, _adim, device=self.device)] * (W - 1)
            for t in range(self.horizon):
                u_t = u_out[t:t+1]             # (1, action_dim)
                if self._linear_mode:
                    z = self._step(z, u_t, zs)
                elif W > 1:
                    z_win_tensor = torch.stack(z_win_list[-W:], dim=1)       # (1, W, d)
                    u_full_list = u_win_list[-(W-1):] + [u_t]
                    u_win_tensor = torch.stack(u_full_list, dim=1)           # (1, W, action_dim)
                    z = self._step_windowed(z_win_tensor, u_win_tensor)      # (1, d)
                    z_win_list.append(z)
                    u_win_list.append(u_t)
                else:
                    z = self._step(z, u_t, zs)
                traj.append(z[0].cpu().numpy())

        # Update history with current z_t and chosen first action (nonlinear W>1 only)
        if not self._linear_mode and W > 1:
            u_chosen = u_out[0:1]   # (1, action_dim)
            if self._z_hist is None:
                # Initialize: W copies of z0
                self._z_hist = [z0.clone()] * W
            else:
                self._z_hist = (self._z_hist + [z0.clone()])[-W:]
            if self._u_hist is None:
                self._u_hist = [torch.zeros(1, _adim, device=self.device)] * (W - 1)
            else:
                self._u_hist = (self._u_hist + [u_chosen.clone()])[-(W - 1):]

        # Return chunk_size action arrays, each shape (action_dim,)
        actions = [u_out[k].cpu().numpy() for k in range(self.chunk_size)]
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
            f"  iter={self.n_iter}  σ0={self.init_std}  σ_warm={self.warm_start_sigma}"
            f"  lb={self.action_lb}  ub={self.action_ub}"
        )

