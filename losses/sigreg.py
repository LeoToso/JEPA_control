"""
Sketched Isotropic Gaussian Regularisation (SIGreg).

From LeJEPA: Balestriero & LeCun, arXiv:2511.08544 (2025).

Enforces z ~ N(0, I) via the Epps-Pulley goodness-of-fit test applied to
K random 1-D projections of the batch embeddings.  Prevents representation
collapse without heuristics (stop-gradient, teacher networks, negative pairs).

Loss = 0 iff the marginal distribution of every linear projection of z is
standard normal — which is equivalent (by Cramer-Wold) to z ~ N(0, I).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def sigreg_loss(
    z: torch.Tensor,
    num_slices: int = 128,
    num_points: int = 17,
) -> torch.Tensor:
    """
    Args:
        z: Batch of embeddings (B, D).
        num_slices: Number of random 1-D projections (K).  128 suffices for
            D=32; the paper uses 1024 for large D.
        num_points: Frequency evaluation points for the Epps-Pulley statistic.

    Returns:
        Scalar SIGreg loss (≥ 0; equals 0 when z ~ N(0, I)).
    """
    B, D = z.shape

    # Random unit projection vectors W: (K, D)
    W = torch.randn(num_slices, D, device=z.device, dtype=z.dtype)
    W = F.normalize(W, dim=1)

    # Project batch onto each direction: y[k, b] = W[k] · z[b]
    y = W @ z.T  # (K, B)

    # Frequency grid ω for the Epps-Pulley test — covers the informative range
    # for N(0,1): characteristic fn exp(-ω²/2) is non-trivial roughly in [0.5, 3]
    omega = torch.linspace(0.5, 3.0, num_points, device=z.device, dtype=z.dtype)  # (P,)

    # ω · y: (K, P, B)
    wy = omega[None, :, None] * y[:, None, :]

    # Empirical characteristic function: φ_N(ω) = (1/B) Σ_b exp(i ω y_b)
    phi_real = wy.cos().mean(dim=-1)  # (K, P)
    phi_imag = wy.sin().mean(dim=-1)  # (K, P)

    # Target characteristic function of N(0, 1): exp(-ω²/2)  [real-valued]
    target = (-0.5 * omega ** 2).exp()  # (P,)

    # Epps-Pulley statistic: |φ_N(ω) - target|² per (slice, frequency)
    ep = (phi_real - target[None, :]) ** 2 + phi_imag ** 2  # (K, P)

    return ep.mean()
