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
    variance_floor_weight: float = 0.0,
    variance_target: float = 1.0,
    variance_eps: float = 1e-4,
) -> torch.Tensor:
    """
    Args:
        z: Batch of embeddings (B, D).
        num_slices: Number of random 1-D projections (K).  128 suffices for
            D=32; the paper uses 1024 for large D.
        num_points: Frequency evaluation points for the Epps-Pulley statistic.
        variance_floor_weight: Optional VICReg-style standard-deviation floor.
            The characteristic-function statistic has zero gradient at exact
            constant collapse; this term amplifies any residual batch variation.
        variance_target: Minimum standard deviation for every latent dimension.
        variance_eps: Numerical stabilizer inside the standard deviation.

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

    loss = ep.mean()

    if variance_floor_weight > 0:
        # Epps-Pulley alone is stationary at a constant zero embedding:
        # d cos(omega*y)/dy = 0 and the sine residual is zero at y=0.
        # A standard-deviation hinge supplies a strong gradient whenever there
        # is residual (even tiny) sample-to-sample variation.
        std = torch.sqrt(z.var(dim=0, unbiased=False) + variance_eps)
        variance_floor = F.relu(variance_target - std).mean()
        loss = loss + variance_floor_weight * variance_floor

    return loss
