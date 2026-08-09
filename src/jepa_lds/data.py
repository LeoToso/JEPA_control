"""Synthetic "sensing" of the LTI ground-truth state, and trajectory/episode
generation with the same open-loop exploration mixture described in the
cartpole JEPA notes (passive / PRBS / random / LQR-guided).

The observation model maps the true state through a fixed, redundant linear
"rendering" matrix plus extra i.i.d. nuisance channels -- a stand-in for the
pixel encoder in the real project: the JEPA encoder has to *learn* which
directions of a higher-dimensional signal are dynamically relevant, exactly
as a ViT encoder has to learn which pixels matter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from .systems import LTISystem


@dataclass
class ObservationModel:
    C: np.ndarray  # (p_signal, n) fixed linear "rendering" of the state
    n_distractor: int = 0
    measurement_noise_std: float = 0.01
    distractor_std: float = 1.0

    @property
    def signal_dim(self) -> int:
        return self.C.shape[0]

    @property
    def p(self) -> int:
        return self.signal_dim + self.n_distractor

    def observe(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        y_signal = self.C @ x + self.measurement_noise_std * rng.standard_normal(self.signal_dim)
        if self.n_distractor:
            y_noise = self.distractor_std * rng.standard_normal(self.n_distractor)
            return np.concatenate([y_signal, y_noise])
        return y_signal

    def oracle_decoder_weight(self, n: int) -> np.ndarray:
        """Weight matrix W (n, p) such that W @ y approx recovers x from the
        *signal* block only (ignoring nuisance channels). Used to build the
        reference "faithful encoder" baseline."""
        C_pinv = np.linalg.pinv(self.C)  # (n, p_signal)
        return np.hstack([C_pinv, np.zeros((n, self.n_distractor))])


def make_observation_model(
    system: LTISystem,
    obs_dim_signal: int | None = None,
    n_distractor: int = 8,
    measurement_noise_std: float = 0.01,
    distractor_std: float = 1.0,
    seed: int = 0,
) -> ObservationModel:
    rng = np.random.default_rng(seed)
    obs_dim_signal = obs_dim_signal or max(system.n * 2, 6)
    C = rng.standard_normal((obs_dim_signal, system.n))
    C /= np.linalg.norm(C, axis=0, keepdims=True)
    return ObservationModel(
        C=C,
        n_distractor=n_distractor,
        measurement_noise_std=measurement_noise_std,
        distractor_std=distractor_std,
    )


@dataclass
class EpisodeBatch:
    y: np.ndarray  # (n_ep, T+1, p)
    x: np.ndarray  # (n_ep, T+1, n) -- ground truth, kept only for diagnostics
    a: np.ndarray  # (n_ep, T, m)
    policy: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_ep,)


def generate_prbs(T: int, m: int, rng: np.random.Generator, switch_prob: float = 0.1, amplitude: float = 1.0):
    a = np.zeros((T, m))
    cur = amplitude * rng.choice([-1.0, 1.0], size=m)
    for t in range(T):
        flip = rng.random(m) < switch_prob
        cur = np.where(flip, -cur, cur)
        a[t] = cur
    return a


def generate_episode(
    system: LTISystem,
    obs_model: ObservationModel,
    T: int,
    policy: str,
    x0_std: float,
    process_noise_std: float,
    rng: np.random.Generator,
    state_clip: float | None,
    K_lqr: np.ndarray | None,
    action_std: float,
    x0_sampler=None,
):
    x0 = x0_sampler(rng) if x0_sampler is not None else x0_std * rng.standard_normal(system.n)
    xs = np.zeros((T + 1, system.n))
    ys = np.zeros((T + 1, obs_model.p))
    actions = np.zeros((T, system.m))
    xs[0] = x0
    ys[0] = obs_model.observe(x0, rng)

    prbs_actions = generate_prbs(T, system.m, rng, amplitude=action_std) if policy == "prbs" else None

    for t in range(T):
        if policy == "passive":
            u = np.zeros(system.m)
        elif policy == "random":
            u = action_std * rng.standard_normal(system.m)
        elif policy == "prbs":
            u = prbs_actions[t]
        elif policy == "lqr":
            u = -K_lqr @ xs[t] + 0.1 * action_std * rng.standard_normal(system.m)
        else:
            raise ValueError(f"unknown policy {policy!r}")
        actions[t] = u
        x_next = system.step(xs[t], u, process_noise_std, rng)
        if state_clip is not None:
            x_next = np.clip(x_next, -state_clip, state_clip)
        xs[t + 1] = x_next
        ys[t + 1] = obs_model.observe(x_next, rng)
    return xs, ys, actions


def generate_dataset(
    system: LTISystem,
    obs_model: ObservationModel,
    n_episodes: int,
    T: int,
    seed: int,
    mixture: dict[str, float] | None = None,
    x0_std: float = 0.05,
    process_noise_std: float = 0.0,
    state_clip: float | None = 10.0,
    action_std: float = 1.0,
    x0_sampler=None,
) -> EpisodeBatch:
    """Generate `n_episodes` open-loop episodes of length `T` from a mixture
    of exploration policies. Defaults mirror the sampling mass reported in
    the cartpole JEPA notes: 75% open-loop (passive/PRBS/random), 25% LQR.

    `x0_sampler`, if given, is a callable `rng -> x0 (n,)` used instead of
    the default isotropic `x0_std * N(0, I)` draw -- e.g.
    `make_modal_contaminated_x0_sampler` for a heavy-tailed initial
    condition on a specific modal coordinate."""
    mixture = mixture or {"passive": 0.25, "prbs": 0.30, "random": 0.20, "lqr": 0.25}
    rng = np.random.default_rng(seed)
    K_lqr, _ = system.dlqr()
    policies = list(mixture.keys())
    probs = np.array(list(mixture.values()), dtype=np.float64)
    probs /= probs.sum()
    choice = rng.choice(policies, size=n_episodes, p=probs)

    X, Y, Aacts = [], [], []
    for pol in choice:
        xs, ys, a = generate_episode(
            system, obs_model, T, pol, x0_std, process_noise_std, rng, state_clip, K_lqr, action_std,
            x0_sampler=x0_sampler,
        )
        X.append(xs)
        Y.append(ys)
        Aacts.append(a)
    return EpisodeBatch(y=np.stack(Y), x=np.stack(X), a=np.stack(Aacts), policy=choice)


def make_modal_contaminated_x0_sampler(
    system: LTISystem,
    sigma_stable: float = 0.03,
    sigma_unstable_small: float = 0.01,
    sigma_unstable_large: float = 3.0,
    contamination_prob: float = 0.05,
):
    """Returns a callable `rng -> x0` that draws the system's two MODAL
    initial coordinates independently -- the stable one from an ordinary
    Gaussian (a "safe", SIGReg-friendly direction), the unstable one from a
    Gaussian SCALE MIXTURE ("contaminated normal": with probability
    `contamination_prob` draw from N(0, sigma_unstable_large^2), otherwise
    from N(0, sigma_unstable_small^2)) -- then maps back to raw state
    coordinates through the system's eigenvector matrix.

    The scale mixture gives the unstable modal coordinate's pooled marginal
    a large, freely-tunable excess kurtosis (heavy tails) that NO linear
    reparametrization of z can remove without driving the coefficient on
    that direction toward 0 -- since a linear combination of a heavy-tailed
    variable and Gaussian noise stays heavy-tailed for any nonzero
    coefficient on the heavy-tailed part. That is the mechanism behind an
    encoder trained with SIGReg collapsing this direction regardless of the
    prediction horizon H used elsewhere in training: the SIGReg loss is a
    property of z's marginal distribution alone and never depends on H.

    Only defined for a 2-state system with a real unstable/stable eigenvalue
    pair (e.g. `make_double_mode_system`)."""
    if system.n != 2:
        raise ValueError("modal-contaminated x0 sampling is only implemented for 2-state systems")
    w, V, _Vinv = system.modal_decomposition()
    if np.any(np.abs(w.imag) > 1e-9):
        raise ValueError("modal-contaminated x0 sampling requires a real eigenvalue pair")
    idx_u = system.unstable_mode_index()
    idx_s = 1 - idx_u
    V = V.real

    def sample(rng: np.random.Generator) -> np.ndarray:
        contaminated = rng.random() < contamination_prob
        sigma_u = sigma_unstable_large if contaminated else sigma_unstable_small
        xi = np.zeros(2)
        xi[idx_u] = sigma_u * rng.standard_normal()
        xi[idx_s] = sigma_stable * rng.standard_normal()
        return V @ xi

    return sample


class WindowDataset(Dataset):
    """Sliding windows of length H+1 (observations) / H (actions) for
    multistep-prediction and multistep-action-reconstruction training."""

    def __init__(self, batch: EpisodeBatch, horizon: int):
        self.horizon = horizon
        n_ep, Tp1, _p = batch.y.shape
        T = Tp1 - 1
        if T < horizon:
            raise ValueError(f"episode length {T} shorter than horizon {horizon}")
        windows_y, windows_a = [], []
        for e in range(n_ep):
            for t in range(0, T - horizon + 1):
                windows_y.append(batch.y[e, t : t + horizon + 1])
                windows_a.append(batch.a[e, t : t + horizon])
        self.y = torch.tensor(np.stack(windows_y), dtype=torch.float32)
        self.a = torch.tensor(np.stack(windows_a), dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.y[idx], self.a[idx]
