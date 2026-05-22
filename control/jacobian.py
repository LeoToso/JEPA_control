"""Jacobian of the JEPA predictor at the equilibrium latent state."""
from __future__ import annotations
import numpy as np
import torch


def compute_jacobian_np(model, z_star: np.ndarray, device) -> tuple:
    """Return (A_jac, B_jac) as numpy arrays. No gradient tracking.

    Primary signature: compute_jacobian_np(model, z_star, device)
    where model is a JEPAModel instance.

    For windowed predictors (W>1), the Jacobian is the PARTIAL derivative
    w.r.t. the last (current) entry in the window, with history fixed at z*.
    A and B remain (d×d) and (d×1) respectively — same size as Markov case.
    """
    d = len(z_star)
    W = getattr(model.config, 'predictor_window', 1)

    z_eq = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)  # (1, d)

    model.eval()

    # Build history window filled with z* (all W entries)
    # For A_jac: differentiate w.r.t. last z entry; history entries are detached
    # z_win shape: (1, W, d)
    z_hist_detached = z_eq.detach().unsqueeze(1).expand(1, W - 1, d)  # (1, W-1, d)
    z_last = z_eq.detach().requires_grad_(True)                        # (1, d)
    z_last_unsq = z_last.unsqueeze(1)                                  # (1, 1, d)
    if W > 1:
        z_win = torch.cat([z_hist_detached, z_last_unsq], dim=1)      # (1, W, d)
    else:
        z_win = z_last_unsq                                            # (1, 1, d)

    # u_win: all zeros, shape (1, W, 1)
    u_win = torch.zeros(1, W, 1, device=device)

    z_out = model.predict(z_win, u_win)   # (1, d)

    A_rows = []
    for i in range(d):
        g = torch.autograd.grad(z_out[0, i], z_last, retain_graph=(i < d - 1))[0]
        A_rows.append(g[0].detach())
    A_jac = torch.stack(A_rows).cpu().numpy()   # (d, d)

    # B_jac: differentiate w.r.t. last u entry; history entries are zeros
    # u_win: (1, W, 1) with last entry having requires_grad
    u_hist = torch.zeros(1, W - 1, 1, device=device)
    u_last = torch.zeros(1, 1, 1, device=device, requires_grad=True)
    if W > 1:
        u_win_b = torch.cat([u_hist, u_last], dim=1)   # (1, W, 1)
    else:
        u_win_b = u_last                                # (1, 1, 1)

    z_win_b = z_eq.detach().unsqueeze(1).expand(1, W, d)  # (1, W, d) — all z* detached
    z_out_b = model.predict(z_win_b, u_win_b)              # (1, d)

    B_cols = []
    for i in range(d):
        g = torch.autograd.grad(z_out_b[0, i], u_last, retain_graph=(i < d - 1))[0]
        B_cols.append(g[0, 0, 0].detach())
    B_jac = torch.stack(B_cols).unsqueeze(-1).cpu().numpy()   # (d, 1)

    return A_jac, B_jac


def compute_jacobian_torch(model, z_star_t: torch.Tensor, device) -> tuple:
    """Return (A_jac, B_jac) as torch tensors with gradient support.
    Used for spectral/PBH regularisation during training.

    Primary signature: compute_jacobian_torch(model, z_star_t, device)
    where model is a JEPAModel and z_star_t is a (d,) tensor (detached).

    For windowed predictors (W>1), the Jacobian is the PARTIAL derivative
    w.r.t. the last (current) entry in the window, with history fixed at z*.
    A and B remain (d×d) and (d×1) respectively.
    """
    d = z_star_t.shape[0]
    W = getattr(model.config, 'predictor_window', 1)

    z_eq = z_star_t.detach().unsqueeze(0)  # (1, d)

    # A_jac: create_graph=True so gradients flow into predictor weights
    z_hist_detached = z_eq.unsqueeze(1).expand(1, W - 1, d)  # (1, W-1, d)
    z_last = z_eq.requires_grad_(True)                         # already (1, d)
    # Note: requires_grad_ on a view may not work; use a fresh tensor
    z_last = z_star_t.detach().unsqueeze(0).requires_grad_(True)  # (1, d)
    z_last_unsq = z_last.unsqueeze(1)                              # (1, 1, d)
    if W > 1:
        z_win = torch.cat([z_hist_detached, z_last_unsq], dim=1)  # (1, W, d)
    else:
        z_win = z_last_unsq                                        # (1, 1, d)

    u_win = torch.zeros(1, W, 1, device=device)
    z_out = model.predict(z_win, u_win)   # (1, d)

    A_rows = []
    for i in range(d):
        g = torch.autograd.grad(
            z_out[0, i], z_last,
            create_graph=True, retain_graph=True
        )[0]
        A_rows.append(g[0])
    A_jac = torch.stack(A_rows)   # (d, d)

    # B_jac
    u_hist = torch.zeros(1, W - 1, 1, device=device)
    u_last = torch.zeros(1, 1, 1, device=device, requires_grad=True)
    if W > 1:
        u_win_b = torch.cat([u_hist, u_last], dim=1)   # (1, W, 1)
    else:
        u_win_b = u_last                                # (1, 1, 1)

    z_win_b = z_star_t.detach().unsqueeze(0).unsqueeze(1).expand(1, W, d)  # (1, W, d)
    z_out_b = model.predict(z_win_b, u_win_b)                               # (1, d)

    B_cols = []
    for i in range(d):
        g = torch.autograd.grad(
            z_out_b[0, i], u_last,
            create_graph=True, retain_graph=True
        )[0]
        B_cols.append(g[0, 0, 0])
    B_jac = torch.stack(B_cols).unsqueeze(-1)   # (d, 1)

    return A_jac, B_jac
