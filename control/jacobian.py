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

    # u_win: all zeros, shape (1, W, act_dim)
    act_dim = model.action_encoder.action_dim
    u_win = torch.zeros(1, W, act_dim, device=device)

    z_out = model.predict(z_win, u_win)   # (1, d)

    A_rows = []
    for i in range(d):
        g = torch.autograd.grad(z_out[0, i], z_last, retain_graph=(i < d - 1))[0]
        A_rows.append(g[0].detach())
    A_jac = torch.stack(A_rows).cpu().numpy()   # (d, d)

    # B_jac: ∂z_{t+1}/∂c_t where c = action_encoder(u).
    # Differentiating w.r.t. the encoded action (not the raw scalar) gives
    # B ∈ R^{d × d_a} — wide matrix when d_a > 1 (action lifting).
    m = model.action_encoder.latent_action_dim
    c_hist = torch.zeros(1, W - 1, m, device=device)
    c_last = torch.zeros(1, 1, m, device=device, requires_grad=True)
    if W > 1:
        c_win_b = torch.cat([c_hist, c_last], dim=1)   # (1, W, m)
    else:
        c_win_b = c_last                                # (1, 1, m)

    z_win_b = z_eq.detach().unsqueeze(1).expand(1, W, d)  # (1, W, d) — all z* detached
    z_out_b = model.predict_from_encoded(z_win_b, c_win_b)  # (1, d)

    B_cols = []
    for i in range(d):
        g = torch.autograd.grad(z_out_b[0, i], c_last, retain_graph=(i < d - 1))[0]
        B_cols.append(g[0, 0].detach())   # (m,)
    B_jac = torch.stack(B_cols).cpu().numpy()   # (d, m)

    return A_jac, B_jac


def compute_augmented_jacobian_np(model, z_star: np.ndarray, device) -> tuple:
    """Return (A_aug, B_aug) for the full W*d-dimensional augmented system.

    For a windowed predictor with window W and latent dim d the true
    discrete-time system has augmented state

        s_t = [z_t, z_{t-1}, ..., z_{t-W+1}]  ∈  R^{W*d}

    and the linearisation at s* = [z*, z*, ..., z*], u*=0 is:

        A_aug  (W*d × W*d)
          rows 0:d       — [∂f/∂z_t | ∂f/∂z_{t-1} | ... | ∂f/∂z_{t-W+1}]
          rows d:2d      — [I_d | 0 | ... | 0]   (z_t → z_{t-1} shift)
          rows 2d:3d     — [0 | I_d | ... | 0]
          ...

        B_aug  (W*d × m)
          rows 0:d       — B_jac  (∂f/∂c, encoded-action Jacobian)
          rows d:        — zeros  (history not affected by current action)

    rho(A_aug) is the correct spectral radius of the learned dynamics;
    the partial-Jacobian rho from compute_jacobian_np is only exact for W=1.

    For W=1 this is identical to compute_jacobian_np.
    """
    d = len(z_star)
    W = getattr(model.config, 'predictor_window', 1)
    Wd = W * d

    z_eq = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)
    model.eval()

    # One leaf tensor per window position.
    # Convention matches the predictor: index 0 = oldest (z_{t-W+1}),
    # index W-1 = newest (z_t).
    z_entries = [z_eq.detach().clone().requires_grad_(True) for _ in range(W)]

    z_win = torch.cat([e.unsqueeze(1) for e in z_entries], dim=1) if W > 1 \
            else z_entries[0].unsqueeze(1)                          # (1, W, d)
    act_dim = model.action_encoder.action_dim
    u_win = torch.zeros(1, W, act_dim, device=device)
    z_out = model.predict(z_win, u_win)                             # (1, d)

    # Partial Jacobians: partial_blocks[k] = ∂f/∂z_{t-k}  (d×d)
    # Augmented column block k corresponds to z_{t-k} = z_entries[W-1-k].
    partial_blocks = []
    total_calls = d * W
    call_idx = 0
    for k in range(W):
        entry = z_entries[W - 1 - k]
        rows = []
        for i in range(d):
            call_idx += 1
            g = torch.autograd.grad(
                z_out[0, i], entry,
                retain_graph=(call_idx < total_calls)
            )[0]
            rows.append(g[0].detach().cpu().numpy())
        partial_blocks.append(np.stack(rows))   # (d, d)

    # Assemble A_aug
    A_aug = np.zeros((Wd, Wd))
    for k in range(W):
        A_aug[0:d, k*d:(k+1)*d] = partial_blocks[k]
    for k in range(W - 1):                      # shift identity blocks
        A_aug[(k+1)*d:(k+2)*d, k*d:(k+1)*d] = np.eye(d)

    # B_aug: same B_jac computation as compute_jacobian_np, padded with zeros
    m = model.action_encoder.latent_action_dim
    c_hist = torch.zeros(1, W - 1, m, device=device)
    c_last = torch.zeros(1, 1, m, device=device, requires_grad=True)
    c_win_b = torch.cat([c_hist, c_last], dim=1) if W > 1 else c_last
    z_win_b = z_eq.detach().unsqueeze(1).expand(1, W, d)
    z_out_b = model.predict_from_encoded(z_win_b, c_win_b)

    B_cols = []
    for i in range(d):
        g = torch.autograd.grad(z_out_b[0, i], c_last,
                                retain_graph=(i < d - 1))[0]
        B_cols.append(g[0, 0].detach().cpu().numpy())
    B_jac = np.stack(B_cols)           # (d, m)

    B_aug = np.zeros((Wd, m))
    B_aug[0:d] = B_jac

    return A_aug, B_aug


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

    act_dim = model.action_encoder.action_dim
    u_win = torch.zeros(1, W, act_dim, device=device)
    z_out = model.predict(z_win, u_win)   # (1, d)

    A_rows = []
    for i in range(d):
        g = torch.autograd.grad(
            z_out[0, i], z_last,
            create_graph=True, retain_graph=True
        )[0]
        A_rows.append(g[0])
    A_jac = torch.stack(A_rows)   # (d, d)

    # B_jac: ∂z_{t+1}/∂c_t (encoded action), shape (d, m) for action lifting.
    m = model.action_encoder.latent_action_dim
    c_hist = torch.zeros(1, W - 1, m, device=device)
    c_last = torch.zeros(1, 1, m, device=device, requires_grad=True)
    if W > 1:
        c_win_b = torch.cat([c_hist, c_last], dim=1)   # (1, W, m)
    else:
        c_win_b = c_last                                # (1, 1, m)

    z_win_b = z_star_t.detach().unsqueeze(0).unsqueeze(1).expand(1, W, d)  # (1, W, d)
    z_out_b = model.predict_from_encoded(z_win_b, c_win_b)                  # (1, d)

    B_cols = []
    for i in range(d):
        g = torch.autograd.grad(
            z_out_b[0, i], c_last,
            create_graph=True, retain_graph=True
        )[0]
        B_cols.append(g[0, 0])   # (m,)
    B_jac = torch.stack(B_cols)   # (d, m)

    return A_jac, B_jac

