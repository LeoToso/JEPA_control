"""Jacobian of the JEPA predictor at the equilibrium latent state."""
from __future__ import annotations
import numpy as np
import torch


def compute_jacobian_np(predictor, action_encoder, z_star: np.ndarray, device) -> tuple:
    """Return (A_jac, B_jac) as numpy arrays. No gradient tracking."""
    d = len(z_star)
    z = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)  # (1, d)
    u_zero = torch.zeros(1, 1, device=device)

    predictor.eval(); action_encoder.eval()

    # A_jac: d backward passes through predictor w.r.t. z
    z_in = z.detach().requires_grad_(True)
    a_zero = action_encoder(u_zero).detach()
    z_out = predictor(z_in, a_zero)   # (1, d)
    A_rows = []
    for i in range(d):
        g = torch.autograd.grad(z_out[0, i], z_in, retain_graph=(i < d - 1))[0]
        A_rows.append(g[0].detach())
    A_jac = torch.stack(A_rows).cpu().numpy()   # (d, d)

    # B_jac: d backward passes through predictor w.r.t. u
    u_in = torch.zeros(1, 1, device=device, requires_grad=True)
    a_in = action_encoder(u_in)
    z_out_b = predictor(z.detach(), a_in)
    B_cols = []
    for i in range(d):
        g = torch.autograd.grad(z_out_b[0, i], u_in, retain_graph=(i < d - 1))[0]
        B_cols.append(g[0, 0].detach())
    B_jac = torch.stack(B_cols).unsqueeze(-1).cpu().numpy()   # (d, 1)

    return A_jac, B_jac


def compute_jacobian_torch(predictor, action_encoder, z_star_t: torch.Tensor, device) -> tuple:
    """Return (A_jac, B_jac) as torch tensors with gradient support.
    Used for spectral/PBH regularisation during training.
    z_star_t: (d,) tensor, should be detached (we attach grad inside).
    """
    d = z_star_t.shape[0]
    u_zero = torch.zeros(1, 1, device=device)
    a_zero = action_encoder(u_zero).detach()

    # A_jac with create_graph so gradients flow into predictor weights
    z_in = z_star_t.detach().unsqueeze(0).requires_grad_(True)
    z_out = predictor(z_in, a_zero)
    A_rows = []
    for i in range(d):
        g = torch.autograd.grad(
            z_out[0, i], z_in,
            create_graph=True, retain_graph=True
        )[0]
        A_rows.append(g[0])
    A_jac = torch.stack(A_rows)   # (d, d)

    # B_jac
    u_in = torch.zeros(1, 1, device=device, requires_grad=True)
    a_in = action_encoder(u_in)
    z_out_b = predictor(z_star_t.detach().unsqueeze(0), a_in)
    B_cols = []
    for i in range(d):
        g = torch.autograd.grad(
            z_out_b[0, i], u_in,
            create_graph=True, retain_graph=True
        )[0]
        B_cols.append(g[0, 0])
    B_jac = torch.stack(B_cols).unsqueeze(-1)   # (d, 1)

    return A_jac, B_jac
