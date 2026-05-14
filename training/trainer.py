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
        self.lambda_fp    = float(self.cfg.get('lambda_fp',    0.0))
        self.lambda_anchor     = float(self.cfg.get('lambda_anchor',     0.0))
        self.lambda_enc_anchor = float(self.cfg.get('lambda_enc_anchor', 0.0))
        self.lambda_inv   = float(self.cfg.get('lambda_inv',   0.0))
        self.inv_action_scale = float(self.cfg.get('inv_action_scale', 1.0))
        self.use_vicreg    = bool(self.cfg.get('use_vicreg', False))
        self.vicreg_lambda = float(self.cfg.get('vicreg_lambda', 25.0))
        self.vicreg_nu     = float(self.cfg.get('vicreg_nu',      1.0))
        self.lambda_sigreg      = float(self.cfg.get('lambda_sigreg', 0.0))
        self.sigreg_num_slices  = int(self.cfg.get('sigreg_num_slices', 128))
        self.sigreg_num_points  = int(self.cfg.get('sigreg_num_points', 17))
        self.ema_momentum  = float(self.cfg.get('ema_momentum',  0.996))
        self.state_encoder_grad_scale = float(
            self.cfg.get('state_encoder_grad_scale', 1.0))
        self.jacobian_every = int(self.cfg.get('jacobian_every', 50))

        lr           = float(self.cfg.get('lr', 1e-4))
        weight_decay = float(self.cfg.get('weight_decay', 1e-4))
        predictor_lr_mult = float(self.cfg.get('predictor_lr_mult', 1.0))
        self.optimizer = torch.optim.Adam([
            {'params': model.encoder.parameters()},
            {'params': model.action_encoder.parameters()},
            {'params': model.predictor.parameters(),
             'lr': lr * predictor_lr_mult},
        ], lr=lr, weight_decay=weight_decay)

        # State head: needed for anchor loss (Option C) or state supervision.
        _needs_state_head = (
            self.lambda_state > 0
            or float(self.cfg.get('warmup_lambda_state', 0.0)) > 0
            or self.lambda_anchor > 0
        )
        if _needs_state_head:
            d_lat = model.config.latent_dim
            self.state_head = nn.Linear(d_lat, 4).to(self.device)
            self.optimizer.add_param_group({'params': self.state_head.parameters()})
        else:
            self.state_head = None

        # Inverse dynamics head: ψ(z_t, z_{t+1}) → u_t.
        # Trains encoder (when unfrozen) to capture action-relevant features
        # without requiring state labels — only actions, which are always available.
        # After encoder freeze, keeps ψ calibrated as a persistent diagnostic.
        if self.lambda_inv > 0:
            d_lat = model.config.latent_dim
            self.inv_head = nn.Sequential(
                nn.Linear(2 * d_lat, d_lat), nn.ReLU(), nn.Linear(d_lat, 1)
            ).to(self.device)
            self.optimizer.add_param_group({'params': self.inv_head.parameters()})
        else:
            self.inv_head = None

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.cfg.get('epochs', 100)),
            eta_min=lr * 0.1,
        )

        self.true_unstable_eigs = gt.unstable_eigenvalues if gt is not None else None
        # z* anchor: set by set_obs_eq() to encoder(obs_eq); falls back to batch EMA
        self._z_star_ema: Optional[torch.Tensor] = None
        self._obs_eq: Optional[torch.Tensor] = None  # (1,3,h,w) equilibrium image

        self.log_path = self.save_dir / 'training_log.csv'
        self._init_csv_log()
        self.best_val_loss = float('inf')
        self.global_step   = 0
        self.epoch         = 0

    def set_obs_eq(self, obs_eq_np: 'np.ndarray') -> None:
        """Pass the exact equilibrium observation (H,W,3 uint8) to anchor z*."""
        import numpy as np
        obs = torch.from_numpy(obs_eq_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        self._obs_eq = obs.to(self.device)

    def _get_z_star_exact(self) -> Optional[torch.Tensor]:
        """Return encoder(obs_eq).detach() if obs_eq is available, else None."""
        if self._obs_eq is None:
            return None
        with torch.no_grad():
            return self.model.encoder(self._obs_eq).squeeze(0)

    # ── CSV logging ──────────────────────────────────────────────────────────
    def _init_csv_log(self):
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'split', 'step', 'total_loss', 'pred_loss',
                 'vicreg', 'state_loss', 'fp_loss', 'spec_loss', 'pbh_loss'])

    def _log_csv(self, epoch, split, step, info):
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, split, step,
                info.get('total_loss', ''), info.get('pred_loss', ''),
                info.get('vicreg_total', ''), info.get('state_loss', ''),
                info.get('fp_loss', ''), info.get('spec_loss', ''), info.get('pbh_loss', ''),
            ])

    # ── Loss computation ──────────────────────────────────────────────────────
    def _compute_loss(self, batch, is_train=True):
        # batch keys: obs_seq (B,H+1,3,h,w), actions (B,H,1), states (B,H+1,4)
        obs_seq = batch['obs_seq'].to(self.device)   # (B, H+1, 3, h, w)
        actions = batch['actions'].to(self.device)   # (B, H, 1)
        B, H1, C, h, w = obs_seq.shape
        H = H1 - 1

        # Online encoder: z_all used for prediction input, state_head, Jacobian
        obs_flat = obs_seq.view(B * H1, C, h, w)
        z_flat   = self.model.encoder(obs_flat)        # (B*(H+1), d)
        d        = z_flat.shape[-1]
        z_all    = z_flat.view(B, H1, d)               # (B, H+1, d)

        # z* = encoder(obs_eq) if available (exact), else EMA over near-eq batch samples.
        # Using the exact equilibrium image eliminates the train/eval z* mismatch that
        # caused high fp_err at eval despite low fp_loss during training.
        if self._obs_eq is not None:
            self._z_star_ema = self._get_z_star_exact()
        elif is_train:
            if 'states' in batch:
                states_b = batch['states'][:, 0].to(self.device).float()  # (B, 4)
                eq_mask = states_b.abs().max(dim=1).values < 0.05
                if eq_mask.sum() > 0:
                    z0_eq = z_all[:, 0][eq_mask].mean(dim=0).detach()
                    if self._z_star_ema is None:
                        self._z_star_ema = z0_eq.clone()
                    else:
                        self._z_star_ema = 0.99 * self._z_star_ema + 0.01 * z0_eq
            else:
                z0_mean = z_all[:, 0].mean(dim=0).detach()
                if self._z_star_ema is None:
                    self._z_star_ema = z0_mean.clone()
                else:
                    self._z_star_ema = 0.99 * self._z_star_ema + 0.01 * z0_mean

        # Stop-gradient targets from the online encoder.
        # Using the same encoder for inputs (z_t) and targets (z_{t+1}) keeps
        # predictor(z*, 0) ≈ z* at equilibrium, minimising fp_error and thus
        # the constant offset that inflates linearisation residual near z*.
        z_targets = z_all[:, 1:].detach()              # (B, H, d)

        # Multi-step unrolled prediction loss
        z_curr = z_all[:, 0]
        pred_loss = torch.zeros(1, device=self.device)
        for k in range(H):
            u_k   = actions[:, k]
            a_k   = self.model.action_encoder(u_k)
            z_hat = self.model.predictor(z_curr, a_k)
            pred_loss = pred_loss + F.mse_loss(z_hat, z_targets[:, k])
            z_curr = z_hat
        pred_loss = pred_loss / H

        total_loss = self.lambda_pred * pred_loss
        info = {'pred_loss': pred_loss.item()}

        # VICReg collapse prevention on online encoder outputs
        if self.use_vicreg:
            from losses.prediction import vicreg_collapse_loss
            vic_loss, vic_info = vicreg_collapse_loss(
                z_all[:, 0],
                lambda_var=self.vicreg_lambda,
                nu_cov=self.vicreg_nu,
            )
            total_loss = total_loss + vic_loss
            info.update(vic_info)
            info['vicreg_total'] = vic_loss.item()

        # SIGreg: Sketched Isotropic Gaussian Regularisation (LeJEPA, 2025).
        # Enforces z ~ N(0,I) via Epps-Pulley test on random 1-D projections.
        # Prevents collapse without stop-gradient or teacher networks.
        if self.lambda_sigreg > 0:
            from losses.sigreg import sigreg_loss
            sig_loss = sigreg_loss(
                z_all[:, 0],
                num_slices=self.sigreg_num_slices,
                num_points=self.sigreg_num_points,
            )
            total_loss = total_loss + self.lambda_sigreg * sig_loss
            info['sigreg_loss'] = sig_loss.item()

        # Multi-step inverse dynamics: ψ(z_0, z_H) → mean(u_0…u_{H-1}).
        # Using the full-horizon gap (H steps apart) instead of consecutive pairs
        # makes the "copy" shortcut impossible: z_0 and z_H differ substantially
        # even when individual steps are small, satisfying the persistent-excitation
        # condition that RichID requires for collapse prevention.
        if self.lambda_inv > 0 and self.inv_head is not None:
            scale  = self.inv_action_scale
            z_pair = torch.cat([z_all[:, 0], z_all[:, H]], dim=-1)  # (B, 2d)
            u_mean = actions.mean(dim=1) / scale                      # (B, 1)
            u_hat  = self.inv_head(z_pair)                            # (B, 1)
            inv_loss = F.mse_loss(u_hat, u_mean)
            total_loss = total_loss + self.lambda_inv * inv_loss
            info['inv_loss'] = inv_loss.item()

        # Encoder anchor: push encoder(obs_eq) toward the origin.
        # During warmup (encoder trainable) gradients flow into encoder params,
        # initialising z* near 0 so all downstream controllers have a known
        # target. After encoder freeze the gradient is zero — completely inert.
        if is_train and self.lambda_enc_anchor > 0 and self._obs_eq is not None:
            z_eq_grad = self.model.encoder(self._obs_eq).squeeze(0)  # grad enabled
            enc_anchor_loss = z_eq_grad.pow(2).mean()
            total_loss = total_loss + self.lambda_enc_anchor * enc_anchor_loss
            info['enc_anchor_loss'] = enc_anchor_loss.item()

        # State reconstruction with gradient mixing.
        # Theta is the key visual cue (pole angle) — weight it 100x vs x/xdot.
        # Cart x is hard to estimate from image; xdot/thetadot less reliable.
        # Encoder receives alpha fraction of state gradient; state_head gets full gradient.
        # All H+1 trajectory frames contribute state loss for denser supervision.
        if self.lambda_state > 0 and self.state_head is not None and 'states' in batch:
            states = batch['states'].to(self.device).float()  # (B, H+1, 4)
            # x gets 50x weight so encoder must encode cart position, not just theta
            w      = torch.tensor([50., 0.1, 100., 1.], device=self.device)
            alpha  = self.state_encoder_grad_scale
            state_loss = torch.zeros(1, device=self.device)
            for k in range(H + 1):
                z_mix = alpha * z_all[:, k] + (1 - alpha) * z_all[:, k].detach()
                state_loss = state_loss + (
                    w * (self.state_head(z_mix) - states[:, k]).pow(2)
                ).mean()
            state_loss = state_loss / (H + 1)
            # Anchor: state_head(z*) must decode to zero — equilibrium latent = zero state.
            # Prevents the drifted fixed-point issue where state_head(z*) shows theta != 0.
            if self._z_star_ema is not None:
                z_eq_mix = alpha * self._z_star_ema + (1 - alpha) * self._z_star_ema.detach()
                sh_eq = self.state_head(z_eq_mix.unsqueeze(0))  # (1, 4)
                state_loss = state_loss + (w * sh_eq.pow(2)).mean()
            total_loss = total_loss + self.lambda_state * state_loss
            info['state_loss'] = state_loss.item()

        # Equilibrium anchor: state_head(z*) must decode to the zero physical state.
        # Only the single equilibrium point is supervised — no trajectory labels needed.
        # During the predictor phase (encoder frozen) this is the only gradient signal
        # for state_head, ensuring the observer stays calibrated at the fixed point.
        if self.lambda_anchor > 0 and self.state_head is not None and self._z_star_ema is not None:
            w_anchor = torch.tensor([50., 0.1, 100., 1.], device=self.device)
            sh_at_zstar = self.state_head(self._z_star_ema.unsqueeze(0))  # (1, 4)
            anchor_loss = (w_anchor * sh_at_zstar.pow(2)).mean()
            total_loss = total_loss + self.lambda_anchor * anchor_loss
            info['anchor_loss'] = anchor_loss.item()

        # Fixed-point loss: predictor should map z* to itself under zero action.
        # Directly penalises the phantom drift that corrupts MPC plans.
        if is_train and self.lambda_fp > 0 and self._z_star_ema is not None:
            a_zero = self.model.action_encoder(
                torch.zeros(1, 1, device=self.device))
            z_star_pred = self.model.predictor(
                self._z_star_ema.unsqueeze(0), a_zero)
            fp_loss = F.mse_loss(z_star_pred,
                                 self._z_star_ema.unsqueeze(0).detach())
            total_loss = total_loss + self.lambda_fp * fp_loss
            info['fp_loss'] = fp_loss.item()

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

        # Warm-up: encoder-only phase (pred disabled) to seed encoder representation
        # before pred_loss locks the encoder into a dynamics-blind latent space.
        warmup_epochs       = int(self.cfg.get('warmup_epochs', 0))
        # After warmup, optionally freeze the encoder for N epochs so the predictor
        # learns dynamics on the theta-encoding latent space without disturbing it.
        freeze_enc_epochs   = int(self.cfg.get('freeze_encoder_epochs', 0))
        _encoder_frozen     = False
        _saved_lambdas = (
            self.lambda_pred, self.lambda_state, self.lambda_spec,
            self.lambda_PBH, self.lambda_fp, self.state_encoder_grad_scale,
        )

        for epoch in range(epochs):
            self.epoch = epoch

            # Switch lambdas / encoder freeze based on training phase
            if warmup_epochs > 0:
                if epoch < warmup_epochs:
                    if epoch == 0:
                        active = []
                        if float(self.cfg.get('warmup_lambda_state', 0.0)) > 0:
                            active.append('state-supervision')
                        if self.lambda_inv > 0:
                            active.append('inv-dynamics')
                        if self.lambda_anchor > 0:
                            active.append('anchor')
                        print(f'[train] Warm-up phase: {warmup_epochs} epochs of '
                              f'{" + ".join(active) if active else "encoder-only"} '
                              f'(pred disabled)')
                    self.lambda_pred  = 0.0
                    self.lambda_state = float(self.cfg.get('warmup_lambda_state', 1.0))
                    self.lambda_spec  = 0.0
                    self.lambda_PBH   = 0.0
                    self.lambda_fp    = 0.0
                    self.state_encoder_grad_scale = 1.0
                elif epoch == warmup_epochs:
                    (self.lambda_pred, self.lambda_state, self.lambda_spec,
                     self.lambda_PBH, self.lambda_fp,
                     self.state_encoder_grad_scale) = _saved_lambdas
                    print(f'[train] Warm-up complete — switching to full loss at epoch {epoch+1}')
                    if freeze_enc_epochs > 0:
                        for p in self.model.encoder.parameters():
                            p.requires_grad_(False)
                        _encoder_frozen = True
                        print(f'[train] Encoder frozen for {freeze_enc_epochs} epochs '
                              f'(epochs {epoch+1}–{epoch+freeze_enc_epochs})')
                elif _encoder_frozen and epoch == warmup_epochs + freeze_enc_epochs:
                    for p in self.model.encoder.parameters():
                        p.requires_grad_(True)
                    _encoder_frozen = False
                    # Optionally scale encoder LR down at unfreeze to prevent
                    # the encoder from overshooting and causing pred oscillation.
                    enc_lr_mult = float(self.cfg.get('encoder_lr_unfreeze_mult', 1.0))
                    if enc_lr_mult != 1.0:
                        self.optimizer.param_groups[0]['lr'] *= enc_lr_mult
                    print(f'[train] Encoder unfrozen at epoch {epoch+1}'
                          f'  (lr_mult={enc_lr_mult})')

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
            state_str  = f"  state={tr.get('state_loss',     0):.4f}" if 'state_loss'     in tr else ''
            inv_str    = f"  inv={tr.get('inv_loss',       0):.4f}" if 'inv_loss'       in tr else ''
            ea_str     = f"  ea={tr.get('enc_anchor_loss', 0):.4f}" if 'enc_anchor_loss' in tr else ''
            fp_str     = f"  fp={tr.get('fp_loss',         0):.4f}" if 'fp_loss'         in tr else ''
            spec_str   = f"  spec={tr.get('spec_loss',     0):.4f}" if 'spec_loss'       in tr else ''
            anchor_str = f"  anc={tr.get('anchor_loss',   0):.4f}" if 'anchor_loss'     in tr else ''
            sig_str    = f"  sig={tr.get('sigreg_loss',   0):.4f}" if 'sigreg_loss'     in tr else ''
            print(f'[Epoch {epoch+1:3d}/{epochs}]'
                  f'  train={tr.get("total_loss",0):.4f}'
                  f'  val={val_loss:.4f}'
                  f'  pred={tr.get("pred_loss",0):.4f}'
                  f'{state_str}{inv_str}{ea_str}{fp_str}{spec_str}{anchor_str}{sig_str}'
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
