"""Jacobian of the JEPA predictor at the equilibrium latent state.

For windowed predictors (W > 1) the correct linearisation is the companion-form
augmented system on the stacked state Z_t = [z_{t-W+1}, ..., z_{t-1}, z_t]:

    Z_{t+1} = A_aug @ Z_t  +  B_aug @ u_t

    A_aug = [ 0   I   0  ...  0  ]   (Wd × Wd)
            [ 0   0   I  ...  0  ]
            [ ...                ]
            [ J₀  J₁  ...  J_{W-1}]

    B_aug = [ 0     ]              (Wd × m)
            [ ...   ]
            [ B_jac ]

where J_k = ∂z_{t+1}/∂z_{window[k]} evaluated at z* for all positions.

The partial derivative w.r.t. only the last window entry (old behaviour) is
J_{W-1} only, which omits J₀..J_{W-2} and gives wrong eigenvalues whenever
the predictor's hidden state genuinely uses history.
"""
from __future__ import annotations
import numpy as np
import torch


def _build_companion(J_blocks: list, d: int, W: int, device) -> torch.Tensor:
    """Build (Wd × Wd) companion matrix from W Jacobian blocks (each d×d)."""
    rows = []
    for k in range(W - 1):
        row = [torch.zeros(d, d, device=device) for _ in range(W)]
        row[k + 1] = torch.eye(d, device=device)
        rows.append(torch.cat(row, dim=1))          # (d, Wd)
    rows.append(torch.cat(J_blocks, dim=1))          # (d, Wd) — last row
    return torch.cat(rows, dim=0)                    # (Wd, Wd)


def compute_jacobian_np(model, z_star: np.ndarray, device) -> tuple:
    """Return (A_aug, B_aug) as numpy arrays.  No gradient tracking.

    A_aug : (Wd, Wd)  companion-form augmented state Jacobian
    B_aug : (Wd, m)   augmented input matrix (non-zero only in last d rows)

    For W=1 this reduces to the standard (d×d) / (d×m) pair.
    """
    d = len(z_star)
    W = getattr(model.config, 'predictor_window', 1)

    z_eq = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)  # (1, d)
    model.eval()

    # ── A_aug: differentiate w.r.t. every window entry ────────────────────────
    z_entries = [z_eq.detach().clone().requires_grad_(True) for _ in range(W)]
    z_win = torch.cat([e.unsqueeze(1) for e in z_entries], dim=1)  # (1, W, d)
    u_win = torch.zeros(1, W, 1, device=device)
    z_out = model.predict(z_win, u_win)  # (1, d)

    J_blocks = []
    for k in range(W):
        rows = []
        for i in range(d):
            g = torch.autograd.grad(z_out[0, i], z_entries[k],
                                    retain_graph=True)[0]
            rows.append(g[0].detach())          # (d,)
        J_blocks.append(torch.stack(rows))      # (d, d)

    A_aug = _build_companion(J_blocks, d, W, device).cpu().numpy()  # (Wd, Wd)

    # ── B_aug: ∂z_{t+1}/∂c_t, placed in the last d rows ──────────────────────
    m = model.action_encoder.latent_action_dim
    c_hist = torch.zeros(1, W - 1, m, device=device)
    c_last = torch.zeros(1, 1, m, device=device, requires_grad=True)
    c_win  = torch.cat([c_hist, c_last], dim=1) if W > 1 else c_last

    z_win_b = z_eq.detach().unsqueeze(1).expand(1, W, d)
    z_out_b = model.predict_from_encoded(z_win_b, c_win)  # (1, d)

    B_last_rows = []
    for i in range(d):
        g = torch.autograd.grad(z_out_b[0, i], c_last,
                                retain_graph=(i < d - 1))[0]
        B_last_rows.append(g[0, 0].detach())    # (m,)
    B_last = torch.stack(B_last_rows).cpu().numpy()  # (d, m)

    B_aug = np.zeros(((W - 1) * d + d, m))
    B_aug[(W - 1) * d:, :] = B_last

    return A_aug, B_aug


def compute_jacobian_torch(model, z_star_t: torch.Tensor, device) -> tuple:
    """Return (A_aug, B_aug) as torch tensors with gradient support.
    Used for spectral/PBH/dynSIG regularisation during training.

    A_aug : (Wd, Wd)  companion-form augmented state Jacobian (create_graph)
    B_aug : (Wd, m)   augmented input matrix (non-zero only in last d rows)
    """
    d = z_star_t.shape[0]
    W = getattr(model.config, 'predictor_window', 1)

    z_eq = z_star_t.detach().unsqueeze(0)  # (1, d)

    # ── A_aug: all W window entries are leaf tensors ───────────────────────────
    z_entries = [z_eq.clone().requires_grad_(True) for _ in range(W)]
    z_win = torch.cat([e.unsqueeze(1) for e in z_entries], dim=1)  # (1, W, d)
    u_win = torch.zeros(1, W, 1, device=device)
    z_out = model.predict(z_win, u_win)  # (1, d)

    J_blocks = []
    for k in range(W):
        rows = []
        for i in range(d):
            g = torch.autograd.grad(
                z_out[0, i], z_entries[k],
                create_graph=True, retain_graph=True
            )[0]
            rows.append(g[0])               # (d,)
        J_blocks.append(torch.stack(rows))  # (d, d)

    A_aug = _build_companion(J_blocks, d, W, device)  # (Wd, Wd)

    # ── B_aug: ∂z_{t+1}/∂c_t (current action only), last d rows ──────────────
    m = model.action_encoder.latent_action_dim
    c_hist = torch.zeros(1, W - 1, m, device=device)
    c_last = torch.zeros(1, 1, m, device=device, requires_grad=True)
    c_win  = torch.cat([c_hist, c_last], dim=1) if W > 1 else c_last

    z_win_b = z_star_t.detach().unsqueeze(0).unsqueeze(1).expand(1, W, d)
    z_out_b = model.predict_from_encoded(z_win_b, c_win)  # (1, d)

    B_last_rows = []
    for i in range(d):
        g = torch.autograd.grad(
            z_out_b[0, i], c_last,
            create_graph=True, retain_graph=True
        )[0]
        B_last_rows.append(g[0, 0])         # (m,)
    B_last = torch.stack(B_last_rows)       # (d, m)

    B_zeros = torch.zeros((W - 1) * d, m, device=device)
    B_aug = torch.cat([B_zeros, B_last], dim=0)  # (Wd, m)

    return A_aug, B_aug
