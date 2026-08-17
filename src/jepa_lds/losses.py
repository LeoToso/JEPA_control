"""Losses: multistep latent prediction, SIGReg (sketched isotropic-Gaussian
regularization), and multistep action reconstruction.

SIGReg here follows the LeJEPA recipe: embeddings are projected onto random
1D directions, and each projection is pushed toward matching the
characteristic function of a standard normal (an Epps-Pulley-style
goodness-of-fit statistic, which is differentiable and -- by the
Cramer-Wold theorem -- matching all 1D projections implies the joint
distribution is isotropic Gaussian). Critically, this regularizer only
constrains the *marginal distribution* of z: it has no notion of actions or
controllability, which is exactly the gap the action-reconstruction loss is
designed to close.
"""
from __future__ import annotations

import torch


def encode_window(encoder, y_window: torch.Tensor) -> torch.Tensor:
    B, Hp1, p = y_window.shape
    return encoder(y_window.reshape(B * Hp1, p)).reshape(B, Hp1, -1)


def one_step_prediction_loss(predictor, z: torch.Tensor, a_window: torch.Tensor) -> torch.Tensor:
    """Teacher-forced one-step error at every consecutive pair in the window
    (this is the notes' L_fwd, H=1). Because each pair is fit independently
    of the others, this term is well-conditioned to optimize even when the
    true dynamics are unstable -- unlike a purely recursive multistep
    rollout, whose gradients explode/vanish through repeated multiplication
    by A_z and bias gradient descent toward spuriously *contractive*
    solutions that under-estimate the true instability."""
    B, H, m = a_window.shape
    z_pred_next = predictor(z[:, :-1].reshape(B * H, -1), a_window.reshape(B * H, m))
    err = z_pred_next.reshape(B, H, -1) - z[:, 1:]
    return (err ** 2).sum(dim=-1).mean()


def multistep_prediction_loss(predictor, z: torch.Tensor, a_window: torch.Tensor) -> torch.Tensor:
    """Recursive (non-teacher-forced) H-step rollout from the encoded first
    frame, compared against the encoded true future frames (the notes'
    L_rollout, H=3)."""
    z_pred = predictor.rollout(z[:, 0], a_window)  # (B, H+1, d)
    err = z_pred[:, 1:] - z[:, 1:]
    return (err ** 2).sum(dim=-1).mean()


def action_reconstruction_loss(decoder, z_window: torch.Tensor, a_window: torch.Tensor):
    """Multistep action reconstruction: recover the whole action sequence
    from the whole latent window at once."""
    a_hat = decoder(z_window)
    return ((a_hat - a_window) ** 2).sum(dim=-1).mean()


def action_reconstruction_endpoint_loss(decoder, z_window: torch.Tensor, a_window: torch.Tensor):
    """Multistep action reconstruction from ONLY the window's two endpoints
    (z_t, z_{t+H}) -- the decoder never sees, and is never given access to,
    any intermediate latent z_{t+1}..z_{t+H-1}."""
    z_endpoints = torch.cat([z_window[:, 0], z_window[:, -1]], dim=-1)
    a_hat = decoder(z_endpoints)
    return ((a_hat - a_window) ** 2).sum(dim=-1).mean()


def one_step_action_reconstruction_loss(decoder, z: torch.Tensor, a_window: torch.Tensor) -> torch.Tensor:
    """Recover a_t from (z_t, z_{t+1}) at every consecutive pair in the
    window."""
    B, H, m = a_window.shape
    pairs = torch.cat([z[:, :-1], z[:, 1:]], dim=-1).reshape(B * H, -1)
    a_hat = decoder(pairs).reshape(B, H, m)
    return ((a_hat - a_window) ** 2).sum(dim=-1).mean()


def _char_fn_gaussian_stat(s: torch.Tensor, t_grid: torch.Tensor) -> torch.Tensor:
    """Epps-Pulley-type statistic: weighted squared distance between the
    empirical characteristic function of `s` and that of N(0,1), evaluated
    on a grid `t_grid` and weighted by the standard-normal density (so the
    test concentrates where a Gaussian's characteristic function has most of
    its mass)."""
    st = s.unsqueeze(0) * t_grid.unsqueeze(1)  # (M, N)
    cos_emp = torch.cos(st).mean(dim=1)
    sin_emp = torch.sin(st).mean(dim=1)
    gauss = torch.exp(-0.5 * t_grid ** 2)
    diff2 = (cos_emp - gauss) ** 2 + sin_emp ** 2
    return (diff2 * gauss).sum() / gauss.sum()


def sigreg_loss(z: torch.Tensor, n_directions: int = 16, n_t: int = 17, t_max: float = 4.0) -> torch.Tensor:
    """z: (B, d). Sample `n_directions` random unit directions and average the
    characteristic-function normality statistic across them."""
    B, d = z.shape
    device = z.device
    w = torch.randn(d, n_directions, device=device)
    w = w / w.norm(dim=0, keepdim=True)
    proj = z @ w  # (B, n_directions)
    t_grid = torch.linspace(-t_max, t_max, n_t, device=device)
    stats = [_char_fn_gaussian_stat(proj[:, k], t_grid) for k in range(n_directions)]
    return torch.stack(stats).mean()
