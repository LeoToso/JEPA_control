"""Training loop for JEPA v2 (ViT encoder + multi-step predictor + Jacobian control)."""
from __future__ import annotations
import csv, os, random, time, warnings
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses.prediction import vicreg_loss


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class SlidingWindowDMDc:
    def __init__(self, window_size=1000, refit_every=50):
        self.window_size = window_size
        self.refit_every = refit_every
        self.Z_buf, self.A_buf, self.Z_next_buf = [], [], []
        self.step_count = 0
        self.A_hat = None
        self.B_hat = None

    def update(self, z, a, z_next):
        self.Z_buf.append(z)
        self.A_buf.append(a)
        self.Z_next_buf.append(z_next)
        if len(self.Z_buf) > self.window_size:
            self.Z_buf.pop(0)
            self.A_buf.pop(0)
            self.Z_next_buf.pop(0)
        self.step_count += 1
        if self.step_count % self.refit_every == 0 and len(self.Z_buf) > 32:
            self._refit()

    def _refit(self):
        from identification.dmdc import fit_dmdc
        Z = np.array(self.Z_buf)
        A = np.array(self.A_buf)
        Z_next = np.array(self.Z_next_buf)
        try:
            self.A_hat, self.B_hat = fit_dmdc(Z, A, Z_next)
        except Exception:
            pass


class Trainer:
    def __init__(self, model, config_dict, gt=None, save_dir='checkpoints', device=None, seed=42):
        self.model   = model
        self.cfg     = config_dict
        self.gt      = gt
        self.save_dir= Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device  = device or (torch.device('cuda') if torch.cuda.is_available()
                                   else torch.device('cpu'))
        self.seed    = seed
        set_all_seeds(seed)
        self.model.to(self.device)

        # Loss weights
        self.lambda_pred  = float(self.cfg.get('lambda_pred',  1.0))
        self.lambda_state = float(self.cfg.get('lambda_state', 1.0))
        self.lambda_spec  = float(self.cfg.get('lambda_spec',  1.0))
        self.lambda_PBH   = float(self.cfg.get('lambda_PBH',   1.0))
        self.use_vicreg   = bool(self.cfg.get('use_vicreg', True))
        self.vicreg_lambda= float(self.cfg.get('vicreg_lambda', 25.0))
        self.vicreg_mu    = float(self.cfg.get('vicreg_mu',     25.0))
        self.vicreg_nu    = float(self.cfg.get('vicreg_nu',      1.0))
        self.jacobian_every = int(self.cfg.get('jacobian_every', 50))

        lr           = float(self.cfg.get('lr', 1e-4))
        weight_decay = float(self.cfg.get('weight_decay', 1e-4))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)

        # Auxiliary state head (theta-weighted supervision)
        if self.lambda_state > 0:
            d_lat = model.config.latent_dim
            self.state_head = nn.Linear(d_lat, 4).to(self.device)
            self.optimizer.add_param_group({'params': self.state_head.parameters()})
        else:
            self.state_head = None

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.cfg.get('epochs', 100)),
            eta_min=lr * 0.1,
        )

        self.true_unstable_eigs = gt.unstable_eigenvalues if gt is not None else None
        # Running estimate of z*: updated each time we see an equilibrium encoding
        self._z_star_ema: Optional[torch.Tensor] = None

        self.log_path = self.save_dir / 'training_log.csv'
        self._init_csv_log()
        self.best_val_loss = float('inf')
        self.global_step   = 0
        self.epoch         = 0

    # ── CSV logging ──────────────────────────────────────────────────────────
    def _init_csv_log(self):
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'split', 'step', 'total_loss', 'pred_loss',
                 'vicreg', 'state_loss', 'spec_loss', 'pbh_loss'])

    def _log_csv(self, epoch, split, step, info):
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, split, step,
                info.get('total_loss', ''), info.get('pred_loss', ''),
                info.get('vicreg_total', ''), info.get('state_loss', ''),
                info.get('spec_loss', ''), info.get('pbh_loss', ''),
            ])

    # ── Loss computation ──────────────────────────────────────────────────────
    def _compute_loss(self, batch, is_train=True):
        # batch keys: obs_seq (B,H+1,3,h,w), actions (B,H,1), states (B,H+1,4)
        obs_seq = batch['obs_seq'].to(self.device)   # (B, H+1, 3, h, w)
        actions = batch['actions'].to(self.device)   # (B, H, 1)
        B, H1, C, h, w = obs_seq.shape
        H = H1 - 1

        # Encode all frames once
        obs_flat = obs_seq.view(B * H1, C, h, w)
        z_flat   = self.model.encoder(obs_flat)       # (B*(H+1), d)
        d        = z_flat.shape[-1]
        z_all    = z_flat.view(B, H1, d)              # (B, H+1, d)

        # Maintain EMA of equilibrium encoding across batches (training only)
        if is_train:
            z0_mean = z_all[:, 0].mean(dim=0).detach()
            if self._z_star_ema is None:
                self._z_star_ema = z0_mean.clone()
            else:
                self._z_star_ema = 0.99 * self._z_star_ema + 0.01 * z0_mean

        # Stop-gradient targets for multi-step prediction
        z_targets = z_all[:, 1:].detach()             # (B, H, d)

        # Multi-step unrolled prediction loss
        z_curr = z_all[:, 0]
        pred_loss = torch.zeros(1, device=self.device)
        for k in range(H):
            u_k   = actions[:, k]                                   # (B, 1)
            a_k   = self.model.action_encoder(u_k)                  # (B, d_a)
            z_hat = self.model.predictor(z_curr, a_k)               # (B, d)
            pred_loss = pred_loss + F.mse_loss(z_hat, z_targets[:, k])
            z_curr = z_hat   # feed predicted z forward
        pred_loss = pred_loss / H

        total_loss = self.lambda_pred * pred_loss
        info = {'pred_loss': pred_loss.item()}

        # VICReg on (z_0, z_1) – prevents representation collapse
        if self.use_vicreg:
            vic_loss, vic_info = vicreg_loss(
                z_all[:, 0], z_all[:, 1].detach(),
                lambda_var=self.vicreg_lambda,
                mu_cov=self.vicreg_mu,
                nu_inv=self.vicreg_nu,
            )
            total_loss = total_loss + vic_loss
            info.update(vic_info)
            info['vicreg_total'] = vic_loss.item()

        # State reconstruction: supervise z_0 on state_0 and z_hat on state_1
        if self.lambda_state > 0 and self.state_head is not None and 'states' in batch:
            states = batch['states'].to(self.device).float()  # (B, H+1, 4)
            w = torch.tensor([10., 1., 10., 1.], device=self.device)
            # z_0 -> state_0
            state_loss = (w * (self.state_head(z_all[:, 0]) - states[:, 0]).pow(2)).mean()
            # z_all[:, 1] (encoder of obs_1) -> state_1
            state_loss = state_loss + (
                w * (self.state_head(z_all[:, 1]) - states[:, 1]).pow(2)
            ).mean()
            total_loss = total_loss + self.lambda_state * state_loss
            info['state_loss'] = state_loss.item()

        # Jacobian regularisation (spectral + PBH) every jacobian_every steps
        if (is_train and self.global_step % self.jacobian_every == 0
                and self.true_unstable_eigs is not None
                and len(self.true_unstable_eigs) > 0
                and (self.lambda_spec > 0 or self.lambda_PBH > 0)):
            try:
                from control.jacobian import compute_jacobian_torch
                # Use EMA-tracked z* (more stable than per-batch mean)
                z_star_t = (self._z_star_ema if self._z_star_ema is not None
                            else z_all[:, 0].mean(dim=0).detach())
                A_jac, B_jac = compute_jacobian_torch(
                    self.model.predictor, self.model.action_encoder,
                    z_star_t, self.device,
                )
                if self.lambda_spec > 0:
                    eigvals = torch.linalg.eigvals(A_jac)
                    true_eigs_t = torch.tensor(
                        self.true_unstable_eigs,
                        dtype=eigvals.dtype, device=eigvals.device,
                    )
                    spec_terms = []
                    for lam_star in true_eigs_t:
                        diff    = eigvals - lam_star
                        dist_sq = diff.real ** 2 + diff.imag ** 2
                        spec_terms.append(dist_sq.min())
                    spec_loss = torch.stack(spec_terms).sum()
                    # Penalise when spectral radius is below the GT unstable value
                    target_rho = float(max(np.abs(self.true_unstable_eigs)))
                    rho_jac_t  = torch.max(torch.abs(eigvals))
                    spec_loss  = spec_loss + 2.0 * torch.relu(
                        torch.tensor(target_rho, dtype=rho_jac_t.dtype,
                                     device=rho_jac_t.device) - rho_jac_t)
                    total_loss = total_loss + self.lambda_spec * spec_loss
                    info['spec_loss'] = spec_loss.item()
                if self.lambda_PBH > 0:
                    pbh_terms = []
                    d_dyn = A_jac.shape[0]
                    for lam_star_val in self.true_unstable_eigs:
                        lam_r = torch.tensor(
                            float(np.real(lam_star_val)),
                            dtype=A_jac.dtype, device=A_jac.device,
                        )
                        M_S   = torch.cat(
                            [lam_r * torch.eye(d_dyn, device=A_jac.device,
                                               dtype=A_jac.dtype) - A_jac,
                             B_jac], dim=-1,
                        )
                        sv    = torch.linalg.svdvals(M_S)
                        pbh_terms.append(-torch.log(sv[-1] + 1e-6))
                    pbh_loss = torch.stack(pbh_terms).mean()
                    total_loss = total_loss + self.lambda_PBH * pbh_loss
                    info['pbh_loss'] = pbh_loss.item()
            except Exception as exc:
                warnings.warn(f'Jacobian regularisation failed: {exc}')

        info['total_loss'] = total_loss.item()
        return total_loss, info

    # ── Train / val epochs ────────────────────────────────────────────────────
    def train_epoch(self, train_loader):
        self.model.train()
        metrics = {}
        for batch in tqdm(train_loader, desc=f'Epoch {self.epoch} [train]',
                          leave=False, dynamic_ncols=True):
            self.optimizer.zero_grad()
            loss, info = self._compute_loss(batch, is_train=True)
            loss.backward()
            params = list(self.model.parameters())
            if self.state_head is not None:
                params += list(self.state_head.parameters())
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            self.optimizer.step()
            for k, v in info.items():
                if isinstance(v, (int, float)):
                    metrics.setdefault(k, []).append(v)
            self.global_step += 1
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    @torch.no_grad()
    def val_epoch(self, val_loader):
        self.model.eval()
        metrics = {}
        for batch in tqdm(val_loader, desc=f'Epoch {self.epoch} [val]',
                          leave=False, dynamic_ncols=True):
            _, info = self._compute_loss(batch, is_train=False)
            for k, v in info.items():
                if isinstance(v, (int, float)):
                    metrics.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def fit(self, train_loader, val_loader, epochs=None, checkpoint_every=10):
        if epochs is None:
            epochs = int(self.cfg.get('epochs', 100))
        history = {'train': [], 'val': []}
        best_state      = None
        _best_val_epoch = 0
        _spec_conv_epoch = None   # first epoch where spec_loss < 0.01
        spec_thresh     = float(self.cfg.get('spec_converge_thresh', 0.01))

        for epoch in range(epochs):
            self.epoch = epoch
            t0 = time.time()
            tr  = self.train_epoch(train_loader)
            val = self.val_epoch(val_loader)
            self.scheduler.step()
            history['train'].append(tr)
            history['val'].append(val)
            val_loss = val.get('total_loss', float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_state      = {k: v.cpu().clone()
                                   for k, v in self.model.state_dict().items()}
                _best_val_epoch = epoch
            # Track first epoch where spectral reg has converged
            if (_spec_conv_epoch is None
                    and tr.get('spec_loss', float('inf')) < spec_thresh):
                _spec_conv_epoch = epoch
            self._log_csv(epoch, 'train', self.global_step, tr)
            self._log_csv(epoch, 'val',   self.global_step, val)
            dt = time.time() - t0
            state_str = f"  state={tr.get('state_loss', 0):.4f}" if 'state_loss' in tr else ''
            spec_str  = f"  spec={tr.get('spec_loss',  0):.4f}"  if 'spec_loss'  in tr else ''
            print(f'[Epoch {epoch+1:3d}/{epochs}]'
                  f'  train={tr.get("total_loss",0):.4f}'
                  f'  val={val_loss:.4f}'
                  f'  pred={tr.get("pred_loss",0):.4f}'
                  f'{state_str}{spec_str}'
                  f'  lr={self.optimizer.param_groups[0]["lr"]:.2e}'
                  f'  dt={dt:.1f}s')

        # Store final-epoch weights before any restoration
        self.final_state = {k: v.cpu().clone()
                            for k, v in self.model.state_dict().items()}
        self._spec_converged_epoch = _spec_conv_epoch
        self._best_val_epoch       = _best_val_epoch

        if best_state is None:
            return history

        # Choose model for Jacobian / control evaluation:
        #   • If spectral reg converged at or before best-val epoch → best-val model
        #     has both low val loss AND correct spectral properties.
        #   • Otherwise spectral convergence lags best-val → keep final-epoch weights.
        if (_spec_conv_epoch is not None
                and _spec_conv_epoch <= _best_val_epoch):
            self.model.load_state_dict({k: v.to(self.device)
                                         for k, v in best_state.items()})
            print(f'[train] Using best-val model  '
                  f'epoch={_best_val_epoch+1}  val={self.best_val_loss:.4f}  '
                  f'(spec converged epoch {_spec_conv_epoch+1})')
        else:
            print(f'[train] Using final-epoch model  '
                  f'(spec converged epoch '
                  f'{_spec_conv_epoch+1 if _spec_conv_epoch is not None else "never"}'
                  f' > best-val epoch {_best_val_epoch+1})')
        return history

    def get_jacobian(self, z_star: np.ndarray):
        """Compute (A_jac, B_jac) as numpy arrays at z_star."""
        from control.jacobian import compute_jacobian_np
        return compute_jacobian_np(
            self.model.predictor, self.model.action_encoder,
            z_star, self.device,
        )

    def save_checkpoint(self, tag='latest'):
        path = self.save_dir / f'checkpoint_{tag}.pt'
        torch.save({'epoch': self.epoch, 'global_step': self.global_step,
                    'model_state': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'config': self.model.get_config_dict()}, path)
