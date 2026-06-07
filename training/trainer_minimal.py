"""Minimal JEPA-control trainer — a small, debuggable baseline.

Implements exactly the boxed objective:

    L = L_1step + beta(e) * L_multi  +  lambda_inv * L_inv
                                       + lambda_fp  * L_fp
                                       + lambda_dynSIG * L_dynSIG

    L_1step(t) = ||f(z_t, c_t) - sg[z_{t+1}]||^2,  averaged over t = 0..H-1
    zhat_0 = z_0;  zhat_{t+1} = f(zhat_t, c_t)     (open-loop rollout, no re-encode)
    L_multi(t) = ||zhat_t - sg[z_t]||^2,           averaged over t = 1..H
    beta(e)    = min(1, e / multistep_warmup_epochs)

    L_inv = ||psi([z_0, z_H]) - mean(u)/scale||^2          (H-step endpoints)
    L_fp  = ||f(z*, 0) - z*||^2                            (z* = encoder(obs_eq))
    L_dynSIG = dynsigreg_loss(z_0, Sigma_target)
        Sigma_target = (1-alpha) * normalize(W_T) + alpha * I
        W_T = sum_{k<T_g} A^k B Sigma_c B^T (A^T)^k,  Sigma_c = I
        EMA: Sigma_target <- beta * Sigma_target + (1-beta) * Sigma_new

Everything else (spectral matching, PBH loss, local-linearization loss,
instability-margin loss, temporal consistency, mirror, anchor, state
supervision, windowed predictors, EMA target encoder, encoder freezing, ...)
is intentionally OMITTED. Per the governing philosophy: first make the latent
dynamics predictive, action-sensitive, and locally anchored at equilibrium —
only add control-theoretic regularizers later if diagnostics (training.diagnostics)
show a specific failure.
"""
from __future__ import annotations
import csv, random, time
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class MinimalTrainer:
    def __init__(self, model, config_dict, save_dir='checkpoints', device=None, seed=42):
        self.model = model
        self.cfg   = config_dict
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or (torch.device('cuda') if torch.cuda.is_available()
                                  else torch.device('cpu'))
        self.seed = seed
        set_all_seeds(seed)
        self.model.to(self.device)

        assert getattr(model.config, 'predictor_window', 1) == 1, \
            'MinimalTrainer requires a Markovian predictor (predictor_window=1). ' \
            'Windowed predictors need LQR/PBH/dynSIG computed on the augmented ' \
            'state xi_t = [z_{t-2};z_{t-1};z_t], which this trainer does not support.'

        # ── Loss weights (spec §2: only these four are active) ───────────────
        self.lambda_inv    = float(self.cfg.get('lambda_inv',    1.0))
        self.lambda_fp     = float(self.cfg.get('lambda_fp',     1.0))
        self.lambda_dynSIG = float(self.cfg.get('lambda_dynSIG', 1.0))
        self.inv_action_scale = float(self.cfg.get('inv_action_scale', 1.0))

        # ── Mixed JEPA prediction loss schedule (spec §3) ────────────────────
        self.multistep_warmup_epochs = int(self.cfg.get('multistep_warmup_epochs', 50))

        # ── dynSIG (spec §6) ──────────────────────────────────────────────────
        self.dynSIG_T_g   = int(  self.cfg.get('dynSIG_T_g',   5))
        self.dynSIG_alpha = float(self.cfg.get('dynSIG_alpha', 0.1))
        self.dynSIG_beta  = float(self.cfg.get('dynSIG_beta',  0.95))
        self.sigreg_num_slices = int(self.cfg.get('sigreg_num_slices', 128))
        self.sigreg_num_points = int(self.cfg.get('sigreg_num_points', 17))
        self.jacobian_every = int(self.cfg.get('jacobian_every', 50))
        # Start isotropic (Sigma_target = I, i.e. ordinary SIGreg) so dynSIG is
        # active from epoch 0; the EMA gradually shifts it toward the
        # Gramian-based target once the Jacobian becomes available (spec §6:
        # "for the first run, it is acceptable to start with ordinary SIGReg").
        d_lat = model.config.latent_dim
        self._Sigma_target: torch.Tensor = torch.eye(d_lat, device=self.device)
        self._A_jac_cache: Optional[torch.Tensor] = None
        self._B_jac_cache: Optional[torch.Tensor] = None

        # ── Optimizer ─────────────────────────────────────────────────────────
        lr           = float(self.cfg.get('lr', 1e-4))
        weight_decay = float(self.cfg.get('weight_decay', 1e-4))
        param_groups = [
            {'params': model.encoder.parameters()},
            {'params': model.action_encoder.parameters(), 'weight_decay': 0.0},
            {'params': model.predictor.parameters()},
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)

        # Inverse-dynamics head: psi([z_0, z_H]) -> mean(u). Self-supervised —
        # only needs actions (always available), trains the encoder/predictor
        # to be action-sensitive without state labels.
        self.inv_head = nn.Sequential(
            nn.Linear(2 * d_lat, d_lat), nn.ReLU(), nn.Linear(d_lat, 1)
        ).to(self.device)
        self.optimizer.add_param_group({'params': self.inv_head.parameters()})

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.cfg.get('epochs', 100)),
            eta_min=lr * 0.1,
        )

        self._z_star: Optional[torch.Tensor] = None      # encoder(obs_eq).detach(), (d,)
        self._obs_eq: Optional[torch.Tensor] = None       # (1, 3*FS, h, w)
        self._is_residual = bool(getattr(model.config, 'predictor_residual', False))

        self.log_path = self.save_dir / 'training_log.csv'
        self._init_csv_log()
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.epoch = 0

    # ── Equilibrium anchor ────────────────────────────────────────────────────
    def set_obs_eq(self, obs_eq_np: 'np.ndarray') -> None:
        """Pass the exact equilibrium observation (H,W,3 uint8) to anchor z*."""
        obs = torch.from_numpy(obs_eq_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_stack = getattr(self.model.config, 'frame_stack', 1)
        if frame_stack > 1:
            obs = obs.repeat(1, frame_stack, 1, 1)
        self._obs_eq = obs.to(self.device)
        self._refresh_z_star()

    @torch.no_grad()
    def _refresh_z_star(self) -> None:
        if self._obs_eq is None:
            return
        self._z_star = self.model.encoder(self._obs_eq).squeeze(0).detach()
        self.model.set_predictor_z_star(self._z_star)

    # ── beta(e) warmup schedule (spec §3) ────────────────────────────────────
    def _beta(self) -> float:
        if self.multistep_warmup_epochs <= 0:
            return 1.0
        return min(1.0, self.epoch / self.multistep_warmup_epochs)

    # ── CSV logging ──────────────────────────────────────────────────────────
    def _init_csv_log(self):
        if self.log_path.exists():
            return
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'split', 'step', 'total_loss', 'one_step_loss',
                 'multi_loss', 'beta', 'inv_loss', 'fp_loss', 'dynSIG_loss'])

    def _log_csv(self, epoch, split, step, info):
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, split, step,
                info.get('total_loss', ''), info.get('one_step_loss', ''),
                info.get('multi_loss', ''), info.get('beta', ''),
                info.get('inv_loss', ''), info.get('fp_loss', ''),
                info.get('dynSIG_loss', ''),
            ])

    def _log_wandb(self, epoch, tr, val):
        """Log per-epoch train/val metrics to wandb. No-op unless a wandb run
        is active (run_experiment.py --wandb initialises one)."""
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        log = {f'train/{k}': v for k, v in tr.items()}
        log.update({f'val/{k}': v for k, v in val.items()})
        log['lr'] = self.optimizer.param_groups[0]['lr']
        wandb.log(log, step=epoch)

    # ── Loss computation ──────────────────────────────────────────────────────
    def _compute_loss(self, batch, is_train=True):
        # batch keys: obs_seq (B,H+1,C,h,w), actions (B,H,1)
        obs_seq = batch['obs_seq'].to(self.device)
        actions = batch['actions'].to(self.device)
        B, H1, C, h, w = obs_seq.shape
        H = H1 - 1
        d = self.model.config.latent_dim

        # Encode the whole trajectory with the online encoder. z_t (context) keeps
        # its gradient; sg[] is applied at use-site via .detach() — standard JEPA
        # teacher-forcing convention, no EMA target encoder (kept minimal).
        obs_flat = obs_seq.reshape(B * H1, C, h, w)
        z_flat   = self.model.encoder(obs_flat)
        z_all    = z_flat.view(B, H1, d)             # (B, H+1, d)

        # Encode the action sequence once.
        u_flat = actions.reshape(B * H, -1)
        c_flat = self.model.action_encoder(u_flat)
        d_a    = c_flat.shape[-1]
        c_seq  = c_flat.view(B, H, d_a)               # (B, H, d_a)

        # ── L_1step: teacher-forced one-step prediction ──────────────────────
        one_step_loss = torch.zeros((), device=self.device)
        for t in range(H):
            z_pred = self.model.predictor(z_all[:, t], c_seq[:, t])
            one_step_loss = one_step_loss + F.mse_loss(z_pred, z_all[:, t + 1].detach())
        one_step_loss = one_step_loss / H

        # ── L_multi: open-loop rollout from z_0, never re-encoding ───────────
        multi_loss = torch.zeros((), device=self.device)
        z_hat = z_all[:, 0]
        for t in range(H):
            z_hat = self.model.predictor(z_hat, c_seq[:, t])
            multi_loss = multi_loss + F.mse_loss(z_hat, z_all[:, t + 1].detach())
        multi_loss = multi_loss / H

        beta = self._beta()
        pred_loss  = one_step_loss + beta * multi_loss
        total_loss = pred_loss
        info = {
            'one_step_loss': one_step_loss.item(),
            'multi_loss': multi_loss.item(),
            'beta': beta,
            'pred_loss': pred_loss.item(),
        }

        # ── L_inv: H-step endpoint inverse dynamics (spec §4) ─────────────────
        if self.lambda_inv > 0:
            scale  = self.inv_action_scale
            z_pair = torch.cat([z_all[:, 0], z_all[:, H]], dim=-1)   # (B, 2d)
            u_mean = actions.mean(dim=1) / scale                      # (B, 1)
            u_hat  = self.inv_head(z_pair)
            inv_loss = F.mse_loss(u_hat, u_mean)
            total_loss = total_loss + self.lambda_inv * inv_loss
            info['inv_loss'] = inv_loss.item()

        # ── L_fp: fixed-point loss (spec §5) ──────────────────────────────────
        # If using the residual predictor, f(z*,0)=z* holds by construction —
        # L_fp is then a pure (zero-gradient) diagnostic, logged but not optimized.
        if self._z_star is not None:
            d_a_zero = torch.zeros(1, d_a, device=self.device, dtype=z_all.dtype)
            z_star_in = self._z_star.unsqueeze(0)
            fp_pred = self.model.predictor(z_star_in, d_a_zero)
            if self._is_residual:
                with torch.no_grad():
                    fp_loss = F.mse_loss(fp_pred, z_star_in)
            else:
                fp_loss = F.mse_loss(fp_pred, z_star_in)
                total_loss = total_loss + self.lambda_fp * fp_loss
            info['fp_loss'] = fp_loss.item()

        # ── L_dynSIG: dynamics-aware SIGReg (spec §6) ─────────────────────────
        if self.lambda_dynSIG > 0:
            from losses.dyn_sigreg import dynsigreg_loss
            dyn_loss = dynsigreg_loss(
                z_all[:, 0], self._Sigma_target.detach(),
                num_slices=self.sigreg_num_slices,
                num_points=self.sigreg_num_points,
            )
            total_loss = total_loss + self.lambda_dynSIG * dyn_loss
            info['dynSIG_loss'] = dyn_loss.item()

        info['total_loss'] = total_loss.item()
        return total_loss, info

    # ── Jacobian / Sigma_target update (spec §6, every jacobian_every steps) ─
    def _update_jacobian_and_sigma_target(self):
        if self._z_star is None:
            return
        from control.jacobian import compute_jacobian_torch
        from losses.dyn_sigreg import compute_controllability_gramian, build_sigma_target
        try:
            A, B = compute_jacobian_torch(self.model, self._z_star, self.device)
            self._A_jac_cache = A.detach()
            self._B_jac_cache = B.detach()
            with torch.no_grad():
                W_T = compute_controllability_gramian(
                    self._A_jac_cache.float(), self._B_jac_cache.float(),
                    T_g=self.dynSIG_T_g,
                )
                Sigma_new = build_sigma_target(W_T, alpha=self.dynSIG_alpha)
                self._Sigma_target = (
                    self.dynSIG_beta * self._Sigma_target
                    + (1 - self.dynSIG_beta) * Sigma_new
                )
        except Exception as exc:
            import warnings
            warnings.warn(f'[MinimalTrainer] Jacobian/Sigma_target update failed: {exc}')

    # ── Train / val epochs ────────────────────────────────────────────────────
    def train_epoch(self, train_loader):
        self.model.train()
        metrics = {}
        for batch in tqdm(train_loader, desc=f'Epoch {self.epoch} [train]',
                          leave=False, dynamic_ncols=True):
            self._refresh_z_star()
            self.optimizer.zero_grad()
            loss, info = self._compute_loss(batch, is_train=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.lambda_dynSIG > 0 and self.global_step % self.jacobian_every == 0:
                self._update_jacobian_and_sigma_target()

            for k, v in info.items():
                if isinstance(v, (int, float)):
                    metrics.setdefault(k, []).append(v)
            self.global_step += 1
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    @torch.no_grad()
    def val_epoch(self, val_loader):
        self.model.eval()
        self._refresh_z_star()
        metrics = {}
        for batch in tqdm(val_loader, desc=f'Epoch {self.epoch} [val]',
                          leave=False, dynamic_ncols=True):
            _, info = self._compute_loss(batch, is_train=False)
            for k, v in info.items():
                if isinstance(v, (int, float)):
                    metrics.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def fit(self, train_loader, val_loader, epochs=None, checkpoint_every=10,
            resume_from=None):
        if epochs is None:
            epochs = int(self.cfg.get('epochs', 100))
        history = {'train': [], 'val': []}
        best_state = None

        start_epoch = 0
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            start_epoch = self.epoch + 1
            print(f'[train] Resuming from epoch {start_epoch}')

        for epoch in range(start_epoch, epochs):
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
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self._log_csv(epoch, 'train', self.global_step, tr)
            self._log_csv(epoch, 'val',   self.global_step, val)
            self._log_wandb(epoch, tr, val)
            if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                self.save_checkpoint(tag=f'epoch{epoch+1:04d}')
            dt = time.time() - t0
            print(f'[Epoch {epoch+1:3d}/{epochs}]'
                  f'  train={tr.get("total_loss",0):.4f}'
                  f'  val={val_loss:.4f}'
                  f'  1step={tr.get("one_step_loss",0):.4f}'
                  f'  multi={tr.get("multi_loss",0):.4f}'
                  f'  beta={tr.get("beta",0):.2f}'
                  f'  inv={tr.get("inv_loss",0):.4f}'
                  f'  fp={tr.get("fp_loss",0):.4f}'
                  f'  dynSIG={tr.get("dynSIG_loss",0):.4f}'
                  f'  lr={self.optimizer.param_groups[0]["lr"]:.2e}'
                  f'  dt={dt:.1f}s')

        self.final_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            print(f'[train] Using best-val model  val={self.best_val_loss:.4f}')
        return history

    def get_jacobian(self, z_star: np.ndarray):
        from control.jacobian import compute_jacobian_np
        return compute_jacobian_np(self.model, z_star, self.device)

    def save_checkpoint(self, tag='latest'):
        path = self.save_dir / f'checkpoint_{tag}.pt'
        torch.save({'epoch': self.epoch, 'global_step': self.global_step,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'config': self.model.get_config_dict()}, path)
        if tag != 'latest':
            latest = self.save_dir / 'checkpoint_latest.pt'
            import shutil
            shutil.copy2(path, latest)
        print(f'[ckpt] saved → {path}')

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.epoch = ckpt.get('epoch', 0)
        self.global_step = ckpt.get('global_step', 0)
        print(f'[ckpt] resumed from {path}  (epoch={self.epoch}, step={self.global_step})')
