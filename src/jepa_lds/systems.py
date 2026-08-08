"""Discrete-time LTI ground-truth plants used as the "original dynamics" in
the JEPA-control toy experiments.

Both example systems are open-loop unstable (spectral radius > 1) and
stabilizable, so a linear controller designed on a *faithful* latent
representation should always be able to stabilize them. Whether a controller
designed on a *learned* latent representation can still do so is exactly the
question the rest of this codebase probes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete


@dataclass
class LTISystem:
    """x_{t+1} = A x_t + B u_t."""

    A: np.ndarray
    B: np.ndarray
    name: str = "system"

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=np.float64)
        self.B = np.asarray(self.B, dtype=np.float64)
        n, n2 = self.A.shape
        if n != n2:
            raise ValueError("A must be square")
        if self.B.shape[0] != n:
            raise ValueError("B rows must match state dimension")
        self.n = n
        self.m = self.B.shape[1]

    # ---- spectral properties -------------------------------------------
    def eigvals(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)

    def spectral_radius(self) -> float:
        return float(np.max(np.abs(self.eigvals())))

    def is_open_loop_unstable(self) -> bool:
        return self.spectral_radius() > 1.0 + 1e-9

    def modal_decomposition(self):
        """A = V diag(w) V^{-1}. Rows of V^{-1} are the modal (left-eigenvector)
        coordinates: xi = V^{-1} x."""
        w, V = np.linalg.eig(self.A)
        Vinv = np.linalg.inv(V)
        return w, V, Vinv

    def unstable_mode_index(self) -> int:
        w, _, _ = self.modal_decomposition()
        return int(np.argmax(np.abs(w)))

    def unstable_modal_coordinate(self, x: np.ndarray) -> np.ndarray:
        """Project state(s) x (..., n) onto the modal coordinate of the most
        unstable eigenvalue. Used as the ground-truth target for the
        "unstable-mode retention" ridge probe."""
        _, _, Vinv = self.modal_decomposition()
        idx = self.unstable_mode_index()
        row = Vinv[idx]
        xi = x @ row.conj()
        return np.real(xi)

    # ---- controllability -------------------------------------------------
    def controllability_matrix(self) -> np.ndarray:
        mats = [self.B]
        Ak = np.eye(self.n)
        for _ in range(1, self.n):
            Ak = Ak @ self.A
            mats.append(Ak @ self.B)
        return np.hstack(mats)

    def is_controllable(self, tol: float = 1e-8) -> bool:
        C = self.controllability_matrix()
        return np.linalg.matrix_rank(C, tol=tol) == self.n

    def pbh_uncontrollable_modes(self, tol: float = 1e-6) -> list[complex]:
        """Popov-Belevitch-Hautus test: eigenvalues lambda of A for which
        rank([A - lambda I, B]) < n, i.e. modes the input cannot influence."""
        bad = []
        for lam in self.eigvals():
            M = np.hstack([self.A - lam * np.eye(self.n), self.B])
            if np.linalg.matrix_rank(M, tol=tol) < self.n:
                bad.append(lam)
        return bad

    def is_stabilizable(self, tol: float = 1e-6) -> bool:
        """Weaker than controllable: only the unstable/marginal modes need to
        be controllable."""
        return all(abs(lam) < 1.0 - 1e-9 for lam in self.pbh_uncontrollable_modes(tol))

    # ---- control synthesis -------------------------------------------------
    def dlqr(self, Q: np.ndarray | None = None, R: np.ndarray | None = None):
        """Solve the discrete algebraic Riccati equation; return gain K with
        u = -K x stabilizing x_{t+1} = (A - BK) x_t."""
        Q = np.eye(self.n) if Q is None else Q
        R = np.eye(self.m) if R is None else R
        P = solve_discrete_are(self.A, self.B, Q, R)
        K = np.linalg.solve(self.B.T @ P @ self.B + R, self.B.T @ P @ self.A)
        return K, P

    def closed_loop_spectral_radius(self, K: np.ndarray) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.A - self.B @ K))))

    # ---- simulation -------------------------------------------------
    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        process_noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        x_next = self.A @ x + self.B @ u
        if process_noise_std > 0:
            rng = rng or np.random.default_rng()
            x_next = x_next + process_noise_std * rng.standard_normal(self.n)
        return x_next

    def simulate(
        self,
        x0: np.ndarray,
        actions: np.ndarray,
        process_noise_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """actions: (T, m). Returns states (T+1, n) including x0."""
        T = actions.shape[0]
        xs = np.zeros((T + 1, self.n))
        xs[0] = x0
        for t in range(T):
            xs[t + 1] = self.step(xs[t], actions[t], process_noise_std, rng)
        return xs


def similarity_transform(Am: np.ndarray, Bm: np.ndarray, T: np.ndarray):
    Tinv = np.linalg.inv(T)
    return T @ Am @ Tinv, T @ Bm


def make_double_mode_system(
    unstable_eig: float = 1.25,
    stable_eig: float = 0.85,
    mix_angle: float = 0.6,
    name: str = "double_mode",
) -> LTISystem:
    """Example 1: minimal 2-state system with one open-loop-unstable real mode
    and one stable real mode, both controllable through the single input.
    A rotation similarity transform decouples the state coordinates from the
    modal (eigen-) coordinates so the unstable direction is not trivially
    axis-aligned with what the encoder observes."""
    Am = np.diag([unstable_eig, stable_eig])
    Bm = np.array([[1.0], [0.6]])
    c, s = np.cos(mix_angle), np.sin(mix_angle)
    T = np.array([[c, -s], [s, c]])
    A, B = similarity_transform(Am, Bm, T)
    return LTISystem(A, B, name=name)


def make_linearized_cartpole_system(
    dt: float = 0.02,
    M: float = 1.0,
    m: float = 0.1,
    l: float = 0.5,
    g: float = 9.81,
    name: str = "cartpole_linear",
) -> LTISystem:
    """Example 2: zero-order-hold discretization of the cart-pole linearized
    about the upright (unstable) equilibrium. State = [cart pos, cart vel,
    pole angle, pole angular vel], input = horizontal force on the cart.

    Continuous-time spectrum is {0, 0, +omega, -omega} with
    omega = sqrt((M+m) g / (M l)) -- a genuine saddle. After ZOH
    discretization the cart's free-integrator pair maps to a repeated
    eigenvalue at 1 (marginal) and the pole saddle maps to one eigenvalue
    > 1 (unstable) and one < 1 (stable), mirroring the pixel-based cartpole
    project this toy codebase is meant to explain.
    """
    Ac = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -m * g / M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, (M + m) * g / (M * l), 0.0],
        ]
    )
    Bc = np.array([[0.0], [1.0 / M], [0.0], [-1.0 / (M * l)]])
    Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, np.eye(4), np.zeros((4, 1))), dt, method="zoh")
    return LTISystem(Ad, Bd, name=name)
