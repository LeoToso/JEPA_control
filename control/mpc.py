"""Finite-horizon latent MPC with backward Riccati and action chunking.

The model is trained on 1-step predictions (obs, action, next_obs transitions
at dt=0.02s × frame_skip). MPC compounds this 1-step model H times to plan
ahead. With H=20 and frame_skip=1 that is a 0.4 s planning horizon.

Action chunking: at each re-planning event the controller solves for the full
H-step optimal sequence and deploys the first `chunk_size` actions before
re-planning. chunk_size=1 is standard receding-horizon MPC; larger values
reduce re-planning frequency (smoother, cheaper) at the cost of reactivity.
"""
from __future__ import annotations
import warnings
from typing import List, Optional, Tuple
import numpy as np


class LatentMPC:
    """Finite-horizon LQR-MPC in the learned latent space.

    Dynamics  : z_{t+1} = A z_t + B u_t   (learned, linear)
    Stage cost: ||z_k - z*||_Q^2 + ||u_k||_R^2
    Terminal  : ||z_H - z*||_{Q_f}^2

    Gains are precomputed once via backward Riccati, making online planning O(d^2).
    Action constraints are enforced by clipping after the unconstrained solve.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int = 20,
        action_lb: float = -10.0,
        action_ub: float = 10.0,
        chunk_size: int = 1,
        Q_f: Optional[np.ndarray] = None,
        u_offset: float = 0.0,
        c_offset: Optional[np.ndarray] = None,
    ):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Q_f = Q_f if Q_f is not None else Q.copy()
        self.horizon = horizon
        self.action_lb = action_lb
        self.action_ub = action_ub
        self.chunk_size = min(chunk_size, horizon)
        self.u_offset = u_offset   # scalar feedforward: cancels (A-I)z* bias
        # Constant affine offset in dynamics: z_{t+1} = A·z_t + B·u + c_offset.
        # Set to f(z*,0)-z* to compensate predictor fixed-point drift so the
        # linear rollout matches reality at and near the equilibrium.
        self.c_offset = c_offset if c_offset is not None else np.zeros(A.shape[0])
        self._precompute_gains()

    def _precompute_gains(self):
        """Compute time-varying MPC gains K[0], ..., K[H-1].

        Strategy:
        1. Solve the infinite-horizon DARE (via scipy Schur method) to get P∞.
        2. Backward-iterate the Riccati from P∞ for H steps to get time-varying
           gains close to the steady-state.  Starting from P∞ guarantees the
           gains at step 0 are correct even when H is too small to converge from
           the terminal Q_f (which fails for poorly-conditioned 2nd-order systems).
        3. If DARE fails (system not stabilisable), fall back to backward Riccati
           starting from Q_f — same as the old behaviour.
        """
        try:
            import scipy.linalg
            P_ss = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        except Exception:
            P_ss = self.Q_f.copy()

        P = P_ss.copy()
        gains = []
        for _ in range(self.horizon):
            M = self.R + self.B.T @ P @ self.B          # (d_u, d_u)
            K = np.linalg.solve(M, self.B.T @ P @ self.A)  # (d_u, d)
            P = self.Q + self.A.T @ P @ (self.A - self.B @ K)
            gains.append(K)
        gains.reverse()   # gains[k] = optimal gain at horizon step k
        self.K_list = gains

    def plan(
        self, z_t: np.ndarray, z_star: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Plan from z_t toward z_star.

        Returns
        -------
        actions   : list of chunk_size clipped action arrays, each (d_u,)
        pred_zs   : (H+1, d) predicted latent trajectory [z_t, z_1, ..., z_H]
        """
        z = z_t.copy()
        actions: List[np.ndarray] = []
        pred_zs = [z.copy()]
        for k in range(self.horizon):
            u = np.clip(
                -self.K_list[k] @ (z - z_star) + self.u_offset,
                self.action_lb,
                self.action_ub,
            )
            if k < self.chunk_size:
                actions.append(u)
            z = self.A @ z + self.B @ u + self.c_offset
            pred_zs.append(z.copy())
        return actions, np.array(pred_zs)   # (chunk_size,), (H+1, d)

    def summary(self) -> str:
        rho = float(np.max(np.abs(np.linalg.eigvals(self.A))))
        return (
            f"LatentMPC  H={self.horizon}  chunk={self.chunk_size}"
            f"  d={self.A.shape[0]}  d_u={self.B.shape[1]}"
            f"  rho(A)={rho:.4f}  lb={self.action_lb}  ub={self.action_ub}"
        )
