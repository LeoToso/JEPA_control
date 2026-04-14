"""
Analytical linearization of the cartpole around the unstable upright equilibrium.

State: x = (cart_position, cart_velocity, pole_angle, pole_angular_velocity)
Input: u = force on cart (scalar)
Equilibrium: x* = (0, 0, 0, 0), u* = 0
"""

import numpy as np
import scipy.linalg
import warnings


class CartpoleGroundTruth:
    """
    Analytical cartpole linearization and control-theoretic properties.

    Parameters
    ----------
    mass_cart : float  M, cart mass (kg)
    mass_pole : float  m, pole mass (kg)
    pole_length : float  l, half-pole length (m)
    gravity : float  g, gravitational acceleration (m/s^2)
    dt : float  sampling period (s)
    """

    def __init__(
        self,
        mass_cart: float = 1.0,
        mass_pole: float = 0.1,
        pole_length: float = 0.5,
        gravity: float = 9.8,
        dt: float = 0.02,
    ):
        self.M = mass_cart
        self.m = mass_pole
        self.l = pole_length
        self.g = gravity
        self.dt = dt

        self._build_continuous()
        self._discretize()
        self._build_output_map()
        self._compute_spectral_properties()
        self._compute_transmission_zeros()
        self._compute_markov_parameters()

    def _build_continuous(self):
        M, m, g, l = self.M, self.m, self.g, self.l
        M_tot = M + m
        denom_coeff = 4 * M + m
        self.A_c = np.array([
            [0, 1,                             0, 0],
            [0, 0,        -3.0 * m * g / denom_coeff, 0],
            [0, 0,                             0, 1],
            [0, 0,  3.0 * M_tot * g / (l * denom_coeff), 0],
        ], dtype=float)

        self.B_c = np.array([
            [0],
            [4.0 / denom_coeff],
            [0],
            [-3.0 / (l * denom_coeff)],
        ], dtype=float)

        self.A_c_approx = np.array([
            [0, 1, -(m * g) / M, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, (M + m) * g / (M * l), 0],
        ], dtype=float)

    def _discretize(self):
        n = self.A_c.shape[0]
        m_in = self.B_c.shape[1]
        dt = self.dt
        # ZOH discretisation via matrix exponential (scipy-version-agnostic).
        # Augment: expm([[A, B], [0, 0]] * dt) partitioned into [Ad, Bd].
        em_upper = np.hstack([self.A_c, self.B_c])
        em_lower = np.zeros((m_in, n + m_in))
        em = scipy.linalg.expm(np.vstack([em_upper, em_lower]) * dt)
        self.A_star = em[:n, :n]
        self.B_star = em[:n, n:]

    def _build_output_map(self):
        self.C_star = np.array([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ], dtype=float)
        self.D_star = np.zeros((self.C_star.shape[0], self.B_star.shape[1]))
        self.n = self.A_star.shape[0]
        self.m_in = self.B_star.shape[1]
        self.p = self.C_star.shape[0]

    def _compute_spectral_properties(self):
        eigvals, eigvecs = np.linalg.eig(self.A_star)
        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs

        eps_marginal = 1e-6
        self.stable_mask = np.abs(eigvals) <= 1.0 + eps_marginal
        self.unstable_mask = np.abs(eigvals) > 1.0 + eps_marginal
        self.near_unstable_mask = np.abs(eigvals) >= 1.0 - 0.05

        self.stable_eigenvalues = eigvals[self.stable_mask]
        self.unstable_eigenvalues = eigvals[self.unstable_mask]
        self.ell_star = int(np.sum(self.unstable_mask))

        try:
            V_inv = np.linalg.inv(eigvecs)
            self.left_eigenvectors = V_inv.conj()
        except np.linalg.LinAlgError:
            self.left_eigenvectors = None

        if self.ell_star > 0:
            idx = np.argmax(np.abs(eigvals[self.unstable_mask]))
            unstable_idxs = np.where(self.unstable_mask)[0]
            self.dominant_unstable_idx = unstable_idxs[idx]
            self.dominant_unstable_eigenvalue = eigvals[self.dominant_unstable_idx]
            self.dominant_right_eigenvector = eigvecs[:, self.dominant_unstable_idx]
            if self.left_eigenvectors is not None:
                self.dominant_left_eigenvector = self.left_eigenvectors[
                    self.dominant_unstable_idx, :]
        else:
            self.dominant_unstable_eigenvalue = None
            self.dominant_right_eigenvector = None
            self.dominant_left_eigenvector = None

        self.spectral_radius = float(np.max(np.abs(eigvals)))

    def _compute_transmission_zeros(self):
        """Compute transmission zeros via square Rosenbrock pencil per SISO channel."""
        A, B, C, D = self.A_star, self.B_star, self.C_star, self.D_star
        n, m_in = self.n, self.m_in

        all_zeros = []
        self.channel_zeros = {}

        for i in range(self.p):
            Ci = C[i:i + 1, :]
            Di = D[i:i + 1, :]

            size = n + m_in
            M_mat = np.zeros((size, size), dtype=float)
            N_mat = np.zeros((size, size), dtype=float)

            M_mat[:n, :n] = A
            M_mat[:n, n:] = B
            M_mat[n:, :n] = -Ci
            M_mat[n:, n:] = -Di

            N_mat[:n, :n] = np.eye(n)

            try:
                ev = scipy.linalg.eigvals(M_mat, N_mat)
                finite_mask = np.isfinite(ev) & (np.abs(ev) < 1e6)
                zeros_i = ev[finite_mask]
                self.channel_zeros[i] = zeros_i
                all_zeros.append(zeros_i)
            except Exception:
                self.channel_zeros[i] = np.array([], dtype=complex)

        if all_zeros:
            self.transmission_zeros = np.concatenate(all_zeros)
        else:
            self.transmission_zeros = np.array([], dtype=complex)

        self.nmp_zeros = self.transmission_zeros[np.abs(self.transmission_zeros) > 1.0 + 1e-6]
        self.mp_zeros = self.transmission_zeros[np.abs(self.transmission_zeros) <= 1.0 + 1e-6]
        self.is_nmp = len(self.nmp_zeros) > 0

    def _compute_markov_parameters(self, K: int = 16):
        """Compute H_k = C A^{k-1} B for k=1,...,K."""
        A, B, C = self.A_star, self.B_star, self.C_star
        p, m_in = C.shape[0], B.shape[1]
        self.markov_params = np.zeros((K, p, m_in))
        Ak = np.eye(self.n)
        for k in range(K):
            if k > 0:
                Ak = Ak @ A
            self.markov_params[k] = C @ Ak @ B

        epsilon_H = 1e-6
        self.relative_degree = None
        for k in range(K):
            if np.linalg.norm(self.markov_params[k]) > epsilon_H:
                self.relative_degree = k + 1
                break
        if self.relative_degree is None:
            self.relative_degree = K

    def validate_linearization(self, n_steps: int = 50,
                                x0_scale: float = 0.005) -> dict:
        """Validate linearisation via Euler integration comparison."""
        M, m, g, l, dt = self.M, self.m, self.g, self.l, self.dt
        A_c = self.A_c
        B_c = self.B_c

        rng = np.random.RandomState(0)
        x0 = rng.uniform(-x0_scale, x0_scale, size=4)
        u = 0.0

        x_lin = np.zeros((n_steps, 4))
        x_lin[0] = x0
        for t in range(n_steps - 1):
            x_lin[t + 1] = x_lin[t] + dt * (A_c @ x_lin[t] + B_c.flatten() * u)

        x_nonlin = np.zeros((n_steps, 4))
        x_nonlin[0] = x0
        for t in range(n_steps - 1):
            pos, vel, ang, ang_vel = x_nonlin[t]
            sin_a = np.sin(ang)
            cos_a = np.cos(ang)
            total_mass = M + m
            ml = m * l

            temp = (u + ml * ang_vel ** 2 * sin_a) / total_mass
            ang_acc = (g * sin_a - cos_a * temp) / (
                l * (4.0 / 3.0 - m * cos_a ** 2 / total_mass))
            pos_acc = temp - ml * ang_acc * cos_a / total_mass

            x_nonlin[t + 1, 0] = pos + dt * vel
            x_nonlin[t + 1, 1] = vel + dt * pos_acc
            x_nonlin[t + 1, 2] = ang + dt * ang_vel
            x_nonlin[t + 1, 3] = ang_vel + dt * ang_acc

        norms_nonlin = np.maximum(np.linalg.norm(x_nonlin, axis=1), 1e-10)
        errors = np.linalg.norm(x_lin - x_nonlin, axis=1) / norms_nonlin

        return {
            "max_relative_error": float(np.max(errors)),
            "mean_relative_error": float(np.mean(errors)),
            "passes_5pct": bool(np.max(errors) < 0.05),
        }

    def summary(self) -> str:
        lines = [
            "CartpoleGroundTruth Summary",
            "=" * 40,
            f"Parameters: M={self.M}, m={self.m}, l={self.l}, g={self.g}, dt={self.dt}",
            f"State dim n={self.n}, Input dim m={self.m_in}, Output dim p={self.p}",
            f"Eigenvalues of A*: {np.round(self.eigenvalues, 4)}",
            f"Spectral radius rho(A*) = {self.spectral_radius:.4f}",
            f"Unstable modes ell* = {self.ell_star}",
            f"Unstable eigenvalues: {np.round(self.unstable_eigenvalues, 4)}",
            f"Transmission zeros: {np.round(self.transmission_zeros, 4)}",
            f"NMP zeros: {np.round(self.nmp_zeros, 4)}",
            f"Is NMP: {self.is_nmp}",
            f"Relative degree r* = {self.relative_degree}",
            f"Markov params H_1 = {np.round(self.markov_params[0], 6)}",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "A_c": self.A_c.tolist(),
            "B_c": self.B_c.tolist(),
            "A_star": self.A_star.tolist(),
            "B_star": self.B_star.tolist(),
            "C_star": self.C_star.tolist(),
            "D_star": self.D_star.tolist(),
            "eigenvalues_real": np.real(self.eigenvalues).tolist(),
            "eigenvalues_imag": np.imag(self.eigenvalues).tolist(),
            "spectral_radius": self.spectral_radius,
            "ell_star": self.ell_star,
            "unstable_eigenvalues_real": np.real(self.unstable_eigenvalues).tolist(),
            "unstable_eigenvalues_imag": np.imag(self.unstable_eigenvalues).tolist(),
            "transmission_zeros_real": np.real(self.transmission_zeros).tolist(),
            "transmission_zeros_imag": np.imag(self.transmission_zeros).tolist(),
            "nmp_zeros_real": np.real(self.nmp_zeros).tolist(),
            "nmp_zeros_imag": np.imag(self.nmp_zeros).tolist(),
            "is_nmp": self.is_nmp,
            "relative_degree": self.relative_degree,
            "n": self.n,
            "m_in": self.m_in,
            "p": self.p,
            "markov_params": self.markov_params.tolist(),
        }


_default_gt: CartpoleGroundTruth | None = None


def get_cartpole_gt(**kwargs) -> CartpoleGroundTruth:
    """Return a cached CartpoleGroundTruth instance."""
    global _default_gt
    if _default_gt is None or kwargs:
        _default_gt = CartpoleGroundTruth(**kwargs)
    return _default_gt


if __name__ == "__main__":
    gt = CartpoleGroundTruth()
    print(gt.summary())
    val = gt.validate_linearization()
    print(f"\nLinearisation validation: {val}")
