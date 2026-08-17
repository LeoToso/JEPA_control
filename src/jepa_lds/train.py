"""Training procedure for the toy JEPA world model.

A naive *joint* gradient-descent fit of a linear encoder and a linear
recursive predictor from random initialization is a genuinely hard,
non-convex bilinear problem: empirically it reliably converges to spurious,
overly-contractive latent dynamics (A_z with the wrong, much-too-small
spectral radius) even when the achieved prediction loss is numerically
zero, regardless of which regularizer is used. That is a real optimization
pathology of naive joint SGD, not the phenomenon this codebase is trying to
isolate.

To get a fair comparison of what SIGReg vs. action-reconstruction *do* to a
representation (rather than conflating that with "can SGD solve linear
system identification from scratch"), training here uses an alternating
scheme:

  1. Freeze the encoder; solve for the predictor (A_z, B_z) in closed form
     by ordinary least squares (this sub-problem is exactly linear).
  2. Freeze the predictor; take several Adam steps on the encoder (and, if
     enabled, the action decoders) to minimize prediction error plus the
     configured regularizer (SIGReg or action-reconstruction). This
     sub-problem is a (possibly regularized) linear-network least-squares
     fit, which per Baldi & Hornik (1989) has no bad local minima for the
     unregularized/action-reconstruction case.
  3. Repeat.

This reliably recovers the true system's spectrum in the unregularized
case (verified in ``tests/test_train.py``), which is the necessary baseline
for then asking whether SIGReg or action-reconstruction distorts it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import EpisodeBatch, WindowDataset
from .losses import (
    action_reconstruction_endpoint_loss,
    action_reconstruction_loss,
    encode_window,
    multistep_prediction_loss,
    one_step_action_reconstruction_loss,
    one_step_prediction_loss,
    sigreg_loss,
)
from .models import EndpointActionDecoder, LinearEncoder, LinearLatentPredictor, MultistepActionDecoder, OneStepActionDecoder


@dataclass
class TrainConfig:
    latent_dim: int = 8
    horizon: int = 5
    lr: float = 1e-2
    outer_rounds: int = 25
    inner_epochs: int = 6
    batch_size: int = 256
    lambda_pred_1step: float = 1.0
    lambda_pred_ms: float = 1.0
    lambda_sigreg: float = 0.0
    lambda_actrecon_1step: float = 0.0
    lambda_actrecon_ms: float = 0.0
    lambda_actrecon_endpoint: float = 0.0
    sigreg_directions: int = 16
    weight_decay: float = 0.0
    seed: int = 0


def _closed_form_predictor_fit(encoder, predictor, Y1: torch.Tensor, Y2: torch.Tensor, A_pairs: torch.Tensor):
    """Ordinary least squares fit of (A_z, B_z) given the current (fixed)
    encoder, using every consecutive observation pair in the dataset."""
    with torch.no_grad():
        z1 = encoder(Y1).numpy()
        z2 = encoder(Y2).numpy()
    Phi = np.hstack([z1, A_pairs.numpy()])
    Theta, *_ = np.linalg.lstsq(Phi, z2, rcond=None)
    AB = Theta.T
    d = z1.shape[1]
    with torch.no_grad():
        predictor.A.weight.copy_(torch.tensor(AB[:, :d], dtype=torch.float32))
        predictor.B.weight.copy_(torch.tensor(AB[:, d:], dtype=torch.float32))


def train_jepa(
    system,
    obs_model,
    train_batch: EpisodeBatch,
    cfg: TrainConfig,
    encoder=None,
    train_encoder: bool = True,
    verbose: bool = True,
    log_every: int = 10,
):
    """If `encoder` is given and `train_encoder=False`, it is used as-is
    (e.g. the oracle FixedLinearEncoder) and excluded from optimization; the
    predictor is then fit once in closed form and no alternation is needed."""
    torch.manual_seed(cfg.seed)
    if encoder is None:
        encoder = LinearEncoder(obs_model.p, cfg.latent_dim, bias=False)
    predictor = LinearLatentPredictor(cfg.latent_dim, system.m)
    predictor.requires_grad_(False)  # always updated in closed form, never by gradient
    ms_decoder = MultistepActionDecoder(cfg.latent_dim, system.m, cfg.horizon) if cfg.lambda_actrecon_ms > 0 else None
    endpoint_decoder = EndpointActionDecoder(cfg.latent_dim, system.m, cfg.horizon) if cfg.lambda_actrecon_endpoint > 0 else None
    one_step_decoder = OneStepActionDecoder(cfg.latent_dim, system.m) if cfg.lambda_actrecon_1step > 0 else None

    Y1 = torch.tensor(train_batch.y[:, :-1, :].reshape(-1, obs_model.p), dtype=torch.float32)
    Y2 = torch.tensor(train_batch.y[:, 1:, :].reshape(-1, obs_model.p), dtype=torch.float32)
    A_pairs = torch.tensor(train_batch.a.reshape(-1, system.m), dtype=torch.float32)

    train_ds = WindowDataset(train_batch, cfg.horizon)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    params = []
    if train_encoder:
        params += list(encoder.parameters())
    if ms_decoder is not None:
        params += list(ms_decoder.parameters())
    if endpoint_decoder is not None:
        params += list(endpoint_decoder.parameters())
    if one_step_decoder is not None:
        params += list(one_step_decoder.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay) if params else None

    history = []
    n_outer = cfg.outer_rounds if opt is not None else 1
    for outer in range(n_outer):
        _closed_form_predictor_fit(encoder, predictor, Y1, Y2, A_pairs)

        if opt is None:
            break  # nothing trainable (oracle encoder, no action decoder): one exact solve suffices

        for _inner in range(cfg.inner_epochs):
            totals = {"pred_1step": 0.0, "pred_ms": 0.0, "sigreg": 0.0, "actrecon": 0.0, "total": 0.0}
            n_batches = 0
            for y_window, a_window in loader:
                opt.zero_grad()
                z = encode_window(encoder, y_window)

                loss_1step = one_step_prediction_loss(predictor, z, a_window)
                loss_ms = multistep_prediction_loss(predictor, z, a_window)
                loss = cfg.lambda_pred_1step * loss_1step + cfg.lambda_pred_ms * loss_ms

                loss_sigreg = torch.tensor(0.0)
                loss_ar = torch.tensor(0.0)
                if cfg.lambda_sigreg > 0:
                    z_flat = z.reshape(-1, cfg.latent_dim)
                    loss_sigreg = sigreg_loss(z_flat, n_directions=cfg.sigreg_directions)
                    loss = loss + cfg.lambda_sigreg * loss_sigreg
                if ms_decoder is not None:
                    loss_ar_ms = action_reconstruction_loss(ms_decoder, z, a_window)
                    loss = loss + cfg.lambda_actrecon_ms * loss_ar_ms
                    loss_ar = loss_ar + loss_ar_ms.detach()
                if endpoint_decoder is not None:
                    loss_ar_ep = action_reconstruction_endpoint_loss(endpoint_decoder, z, a_window)
                    loss = loss + cfg.lambda_actrecon_endpoint * loss_ar_ep
                    loss_ar = loss_ar + loss_ar_ep.detach()
                if one_step_decoder is not None:
                    loss_ar_1s = one_step_action_reconstruction_loss(one_step_decoder, z, a_window)
                    loss = loss + cfg.lambda_actrecon_1step * loss_ar_1s
                    loss_ar = loss_ar + loss_ar_1s.detach()
                loss.backward()
                opt.step()

                totals["pred_1step"] += (cfg.lambda_pred_1step * loss_1step).item()
                totals["pred_ms"] += (cfg.lambda_pred_ms * loss_ms).item()
                totals["sigreg"] += (cfg.lambda_sigreg * loss_sigreg).item()
                totals["actrecon"] += loss_ar.item() if torch.is_tensor(loss_ar) else loss_ar
                totals["total"] += loss.item()
                n_batches += 1
            for k in totals:
                totals[k] /= n_batches
            totals["pred"] = totals["pred_1step"] + totals["pred_ms"]
            history.append(totals)

        if verbose and (outer % log_every == 0 or outer == n_outer - 1):
            t = history[-1]
            print(
                f"[outer {outer:3d}] total={t['total']:.4f} pred1={t['pred_1step']:.4f} "
                f"predms={t['pred_ms']:.4f} sigreg={t['sigreg']:.4f} ar={t['actrecon']:.4f}"
            )

    # final closed-form refit of the predictor against the fully-trained encoder
    _closed_form_predictor_fit(encoder, predictor, Y1, Y2, A_pairs)

    decoder = ms_decoder or endpoint_decoder or one_step_decoder
    return encoder, predictor, decoder, history


def train_jepa_naive(
    system,
    obs_model,
    train_batch: EpisodeBatch,
    cfg: TrainConfig,
    verbose: bool = True,
    log_every: int = 10,
):
    """"Regular" JEPA training, for direct comparison against `train_jepa`'s
    alternating scheme: encoder AND predictor are trained jointly, via a
    single Adam optimizer over all parameters together, every step -- no
    closed-form solves, nothing ever frozen. This is what the real
    pixel-based project (and JEPA training generally) actually does, since a
    closed-form predictor fit is only possible because everything here is
    linear.

    Uses the exact same loss functions as `train_jepa` and the same total
    gradient-step budget (`outer_rounds * inner_epochs` steps), so the two
    are a fair, like-for-like comparison of *how* the same objective is
    optimized.
    """
    torch.manual_seed(cfg.seed)
    encoder = LinearEncoder(obs_model.p, cfg.latent_dim, bias=False)
    predictor = LinearLatentPredictor(cfg.latent_dim, system.m)
    ms_decoder = MultistepActionDecoder(cfg.latent_dim, system.m, cfg.horizon) if cfg.lambda_actrecon_ms > 0 else None
    endpoint_decoder = EndpointActionDecoder(cfg.latent_dim, system.m, cfg.horizon) if cfg.lambda_actrecon_endpoint > 0 else None
    one_step_decoder = OneStepActionDecoder(cfg.latent_dim, system.m) if cfg.lambda_actrecon_1step > 0 else None

    train_ds = WindowDataset(train_batch, cfg.horizon)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    params = list(encoder.parameters()) + list(predictor.parameters())
    if ms_decoder is not None:
        params += list(ms_decoder.parameters())
    if endpoint_decoder is not None:
        params += list(endpoint_decoder.parameters())
    if one_step_decoder is not None:
        params += list(one_step_decoder.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_epochs = cfg.outer_rounds * cfg.inner_epochs  # same step budget as train_jepa
    history = []
    for epoch in range(total_epochs):
        totals = {"pred_1step": 0.0, "pred_ms": 0.0, "sigreg": 0.0, "actrecon": 0.0, "total": 0.0}
        n_batches = 0
        for y_window, a_window in loader:
            opt.zero_grad()
            z = encode_window(encoder, y_window)

            loss_1step = one_step_prediction_loss(predictor, z, a_window)
            loss_ms = multistep_prediction_loss(predictor, z, a_window)
            loss = cfg.lambda_pred_1step * loss_1step + cfg.lambda_pred_ms * loss_ms

            loss_sigreg = torch.tensor(0.0)
            loss_ar = torch.tensor(0.0)
            if cfg.lambda_sigreg > 0:
                z_flat = z.reshape(-1, cfg.latent_dim)
                loss_sigreg = sigreg_loss(z_flat, n_directions=cfg.sigreg_directions)
                loss = loss + cfg.lambda_sigreg * loss_sigreg
            if ms_decoder is not None:
                loss_ar_ms = action_reconstruction_loss(ms_decoder, z, a_window)
                loss = loss + cfg.lambda_actrecon_ms * loss_ar_ms
                loss_ar = loss_ar + loss_ar_ms.detach()
            if endpoint_decoder is not None:
                loss_ar_ep = action_reconstruction_endpoint_loss(endpoint_decoder, z, a_window)
                loss = loss + cfg.lambda_actrecon_endpoint * loss_ar_ep
                loss_ar = loss_ar + loss_ar_ep.detach()
            if one_step_decoder is not None:
                loss_ar_1s = one_step_action_reconstruction_loss(one_step_decoder, z, a_window)
                loss = loss + cfg.lambda_actrecon_1step * loss_ar_1s
                loss_ar = loss_ar + loss_ar_1s.detach()
            loss.backward()
            opt.step()

            totals["pred_1step"] += (cfg.lambda_pred_1step * loss_1step).item()
            totals["pred_ms"] += (cfg.lambda_pred_ms * loss_ms).item()
            totals["sigreg"] += (cfg.lambda_sigreg * loss_sigreg).item()
            totals["actrecon"] += loss_ar.item() if torch.is_tensor(loss_ar) else loss_ar
            totals["total"] += loss.item()
            n_batches += 1
        for k in totals:
            totals[k] /= n_batches
        totals["pred"] = totals["pred_1step"] + totals["pred_ms"]
        history.append(totals)

        if verbose and (epoch % log_every == 0 or epoch == total_epochs - 1):
            t = history[-1]
            print(
                f"[epoch {epoch:4d}] total={t['total']:.4f} pred1={t['pred_1step']:.4f} "
                f"predms={t['pred_ms']:.4f} sigreg={t['sigreg']:.4f} ar={t['actrecon']:.4f}"
            )

    decoder = ms_decoder or endpoint_decoder or one_step_decoder
    return encoder, predictor, decoder, history
