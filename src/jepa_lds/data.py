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
):
    x0 = x0_std * rng.standard_normal(system.n)
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
) -> EpisodeBatch:
    """Generate `n_episodes` open-loop episodes of length `T` from a mixture
    of exploration policies. Defaults mirror the sampling mass reported in
    the cartpole JEPA notes: 75% open-loop (passive/PRBS/random), 25% LQR."""
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
            system, obs_model, T, pol, x0_std, process_noise_std, rng, state_clip, K_lqr, action_std
        )
        X.append(xs)
        Y.append(ys)
        Aacts.append(a)
    return EpisodeBatch(y=np.stack(Y), x=np.stack(X), a=np.stack(Aacts), policy=choice)


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
