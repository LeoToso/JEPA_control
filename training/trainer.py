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
        # When True, SIGreg operates on the full trajectory (B*(H+1), d) rather than
        # z_0 only. Needed when use_target_encoder=False: the Epps-Pulley gradient
        # vanishes at z=0 (sin(ωy)→0), so having H+1 gradient paths instead of 1
        # provides enough signal to bootstrap encoder diversity before pred collapses.
        self.sigreg_all_steps   = bool(self.cfg.get('sigreg_all_steps', False))
        # ── Dynamics-aware regularization ─────────────────────────────────────
        self.lambda_dynSIG  = float(self.cfg.get('lambda_dynSIG', 0.0))
        self.dynSIG_T_g     = int(  self.cfg.get('dynSIG_T_g',    5))
        self.dynSIG_alpha   = float(self.cfg.get('dynSIG_alpha',  0.1))
        self.dynSIG_beta    = float(self.cfg.get('dynSIG_beta',   0.95))
        self.dynSIG_near_eq_radius = float(self.cfg.get('dynSIG_near_eq_radius', 0.0))
        self.lambda_varfloor = float(self.cfg.get('lambda_varfloor', 0.0))
        self.lambda_temp    = float(self.cfg.get('lambda_temp',   0.0))
        self.temp_near_eq_radius = float(self.cfg.get('temp_near_eq_radius', 0.5))
        # Mirror antisymmetry: encoder(flip(obs)) + encoder(obs) ≈ 2·z*
        # Cartpole horizontal flip = negate all state components (x,ẋ,θ,θ̇→-x,-ẋ,-θ,-θ̇).
        # Purely self-supervised — no state labels needed.
        self.lambda_mirror  = float(self.cfg.get('lambda_mirror', 0.0))
        # Initialize Sigma_target to I so dynSIG/varfloor are active from epoch 0
        # (isotropic = standard SIGreg), then gradually shift to Gramian target once
        # A_jac is available.  Without this, both losses skip for the first
        # jacobian_every epochs (Sigma_target is None) → no collapse prevention.
        _d = model.encoder.latent_dim
        if self.lambda_dynSIG > 0 or self.lambda_varfloor > 0:
            self._Sigma_target: Optional[torch.Tensor] = torch.eye(_d, device=self.device)
        else:
            self._Sigma_target: Optional[torch.Tensor] = None  # EMA of Gramian-based target cov
        self.ema_momentum  = float(self.cfg.get('ema_momentum',  0.996))
        # EMA target encoder: use target_encoder for z_rest targets instead of
        # stop-grad online encoder.  Prevents collapse because targets lag behind
        # the online encoder (via EMA), so pred_loss stays non-zero even when the
        # online encoder starts to collapse.
        self.use_target_encoder = bool(self.cfg.get('use_target_encoder', False))
        self.detach_targets = bool(self.cfg.get('detach_targets', True))
        self.target_encoder_momentum = float(
            self.cfg.get('target_encoder_momentum', self.ema_momentum))
        self.state_encoder_grad_scale = float(
            self.cfg.get('state_encoder_grad_scale', 1.0))
        self.jacobian_every = int(self.cfg.get('jacobian_every', 50))
        self.predictor_window = int(self.cfg.get('predictor_window', 1))

        lr           = float(self.cfg.get('lr', 1e-4))
        weight_decay = float(self.cfg.get('weight_decay', 1e-4))
        predictor_lr_mult = float(self.cfg.get('predictor_lr_mult', 1.0))
        # Augmentation: apply noise to online encoder input during training.
        # EMA target encoder receives clean observations → prediction target remains
        # meaningful even near equilibrium where obs_t ≈ obs_{t+1}, preventing
        # the collapsed-encoder trivial solution (BYOL/I-JEPA standard mechanism).
        self.aug_noise_std = float(self.cfg.get('aug_noise_std', 0.0))
        param_groups = [
            {'params': model.encoder.parameters()},
            {'params': model.action_encoder.parameters(), 'weight_decay': 0.0},
            {'params': model.predictor.parameters(),
             'lr': lr * predictor_lr_mult},
        ]
        self.optimizer = torch.optim.Adam(
            param_groups, lr=lr, weight_decay=weight_decay)

        self._amp_enabled = (self.device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)

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

        # Local inverse dynamics head (local_pair / local_window modes).
        # inv_frames=2 (paper/SMWM): ψ(z_t, z_{t+1}) → u_t  — 2-layer MLP, H pairs.
        # inv_frames=3 (default):    ψ(z_{t-1}, z_t, z_{t+1}) → u_t — 4-layer MLP, H-1 triplets.
        # This head receives intermediate latent states.  For endpoint-only reconstruction
        # (no intermediate states) see EndpointActionDecoder and lambda_action_reconstruction.
        self.inv_frames = int(self.cfg.get('inv_frames', 3))
        if self.lambda_inv > 0:
            d_lat = model.config.latent_dim
            inv_hidden = int(self.cfg.get('inv_hidden_dim', d_lat))
            in_dim = self.inv_frames * d_lat
            if self.inv_frames == 2:
                self.inv_head = nn.Sequential(
                    nn.Linear(in_dim, inv_hidden), nn.ReLU(),
                    nn.Linear(inv_hidden, 1),
                ).to(self.device)
            else:
                self.inv_head = nn.Sequential(
                    nn.Linear(in_dim, inv_hidden), nn.ReLU(),
                    nn.Linear(inv_hidden, inv_hidden), nn.ReLU(),
                    nn.Linear(inv_hidden, inv_hidden), nn.ReLU(),
                    nn.Linear(inv_hidden, 1),
                ).to(self.device)
            self.optimizer.add_param_group({'params': self.inv_head.parameters()})
        else:
            self.inv_head = None

        # Endpoint action decoder (endpoint_sequence / local_pair / local_window modes).
        # Distinct from inv_head: receives ONLY (z_start, z_end) — no intermediate latents —
        # testing whether endpoint representations encode the full action-induced trajectory.
        # Gradient flows through both encoder calls (z_start and z_end) into the online encoder.
        self.lambda_action_reconstruction = float(
            self.cfg.get('lambda_action_reconstruction', 0.0))
        self.action_reconstruction_mode = str(
            self.cfg.get('action_reconstruction_mode', 'endpoint_sequence'))
        self.action_reconstruction_horizon = int(
            self.cfg.get('action_reconstruction_horizon', 5))
        self.ar_action_scale = float(
            self.cfg.get('action_reconstruction_action_scale', 1.0))

        if self.lambda_action_reconstruction > 0:
            from models.endpoint_action_decoder import EndpointActionDecoder
            d_lat = model.config.latent_dim
            # local_pair always reconstructs a single action; other modes use configured H_act
            _H_act = (1 if self.action_reconstruction_mode == 'local_pair'
                      else self.action_reconstruction_horizon)
            _ar_hidden = int(self.cfg.get('action_reconstruction_hidden_dim', 256))
            _ar_layers = int(self.cfg.get('action_reconstruction_n_layers', 3))
            _ar_delta  = bool(self.cfg.get('action_reconstruction_use_delta', True))
            self.endpoint_action_decoder = EndpointActionDecoder(
                latent_dim=d_lat, action_dim=1, H_act=_H_act,
                hidden_dim=_ar_hidden, n_layers=_ar_layers, use_delta=_ar_delta,
            ).to(self.device)
            self.optimizer.add_param_group(
                {'params': self.endpoint_action_decoder.parameters(), 'weight_decay': 0.0})
            # Physical-state endpoint decoder: trained alongside the latent decoder using
            # ground-truth states as inputs.  Its val loss upper-bounds what endpoint
            # representations can possibly achieve, since states contain more information
            # than latents.  Its gradient never reaches the encoder.
            self.phys_endpoint_decoder = EndpointActionDecoder(
                latent_dim=4,   # cartpole physical state dim
                action_dim=1, H_act=_H_act,
                hidden_dim=_ar_hidden, n_layers=_ar_layers, use_delta=_ar_delta,
            ).to(self.device)
            self.optimizer.add_param_group(
                {'params': self.phys_endpoint_decoder.parameters(), 'weight_decay': 0.0})
        else:
            self.endpoint_action_decoder = None
            self.phys_endpoint_decoder   = None

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.cfg.get('epochs', 100)),
            eta_min=lr * 0.1,
        )

        self.true_unstable_eigs  = gt.unstable_eigenvalues  if gt is not None else None
        self.true_marginal_eigs  = gt.marginal_eigenvalues  if gt is not None else np.array([])
        self.n_marginal_modes    = int(gt.n_marginal)       if gt is not None else 0
        self.lambda_spec_marginal = float(self.cfg.get('lambda_spec_marginal', 0.0))

        # ── Eigenvector-equation spec loss (L_spec_eig) ──────────────────────
        # Directly enforces A_jac v̂_u ≈ λ* v̂_u where v̂_u is the latent image
        # of the GT dominant unstable eigenvector (estimated from encoder sensitivity).
        # More targeted than L_spec (which hunts over all 32 eigenvalues).
        self.lambda_spec_eig         = float(self.cfg.get('lambda_spec_eig',         0.0))
        self.spec_eig_epsilon        = float(self.cfg.get('spec_eig_epsilon',        0.15))
        self.spec_eig_warmup_epochs  = int(  self.cfg.get('spec_eig_warmup_epochs',  0))
        self._obs_v_u: Optional[torch.Tensor] = None  # obs at x* + ε·v_u_physical

        # ── Data-driven local linearization + instability losses ──────────────
        # Replaces GT-eigenvalue spectral matching with two observation-only terms:
        #   L_local:    f(z_t,u_t) ≈ z*+A(z_t-z*)+Bu_t  on self-loop samples
        #   L_unstable: 1+η ≤ ρ(A) ≤ ρ_max  (data-driven bounds, no GT needed)
        self.lambda_local    = float(self.cfg.get('lambda_local',    0.0))
        self.lambda_unstable = float(self.cfg.get('lambda_unstable', 0.0))
        self.unstable_eta    = float(self.cfg.get('unstable_eta',    0.05))
        self.local_state_threshold = float(self.cfg.get('local_state_threshold', 0.05))
        self.rho_max_estimate = float(self.cfg.get('rho_max_init', 1.3))
        self._rho_buffer: list = []          # near-eq growth rates, reset each epoch
        self._A_jac_cache: Optional[torch.Tensor] = None   # stop-grad, updated every jacobian_every steps
        self._B_jac_cache: Optional[torch.Tensor] = None
        self._B_eff_cache: Optional[torch.Tensor] = None   # B_eff = B_jac @ W_enc (d×1), scalar action
        # z* anchor: set by set_obs_eq() to encoder(obs_eq); falls back to batch EMA
        self._z_star_ema: Optional[torch.Tensor] = None
        self._obs_eq: Optional[torch.Tensor] = None  # (1,3,h,w) equilibrium image

        self.log_path = self.save_dir / 'training_log.csv'
        self._init_csv_log()
        self.best_val_loss = float('inf')
        self.global_step   = 0
        self.epoch         = 0

    def set_obs_eq(self, obs_eq_np: 'np.ndarray') -> None:
        """Pass the exact equilibrium observation (H,W,3 uint8) to anchor z*.

        With frame_stack > 1, the same frame is duplicated to fill all channels,
        matching the training-time convention (prev=curr at episode start).
        """
        import numpy as np
        obs = torch.from_numpy(obs_eq_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_stack = getattr(self.model.config, 'frame_stack', 1)
        if frame_stack > 1:
            obs = obs.repeat(1, frame_stack, 1, 1)   # (1, 3*FS, h, w)
        self._obs_eq = obs.to(self.device)

    def set_obs_unstable_dir(self, obs_np: 'np.ndarray') -> None:
        """Set the observation at x* + ε·v_u_physical for L_spec_eig.

        The image is rendered at the perturbed state (position part of the GT
        dominant unstable eigenvector only — velocities are not visible in a
        single frame).  Called once before training, alongside set_obs_eq().
        """
        obs = torch.from_numpy(obs_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_stack = getattr(self.model.config, 'frame_stack', 1)
        if frame_stack > 1:
            obs = obs.repeat(1, frame_stack, 1, 1)
        self._obs_v_u = obs.to(self.device)

    def _get_z_star_exact(self) -> Optional[torch.Tensor]:
        """Return encoder(obs_eq).detach() if obs_eq is available, else None."""
        if self._obs_eq is None:
            return None
        with torch.no_grad():
            return self.model.encoder(self._obs_eq).squeeze(0)

    # ── CSV logging ──────────────────────────────────────────────────────────
    def _init_csv_log(self):
        if self.log_path.exists():
            return   # preserve existing log when resuming
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['epoch', 'split', 'step', 'total_loss', 'pred_loss',
                 'vicreg', 'state_loss', 'fp_loss', 'local_loss',
                 'unstable_loss', 'spec_loss', 'pbh_loss',
                 'dynSIG_loss', 'varfloor_loss', 'temp_loss'])

    def _log_csv(self, epoch, split, step, info):
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, split, step,
                info.get('total_loss', ''), info.get('pred_loss', ''),
                info.get('vicreg_total', ''), info.get('state_loss', ''),
                info.get('fp_loss', ''), info.get('local_loss', ''),
                info.get('unstable_loss', ''), info.get('spec_loss', ''),
                info.get('pbh_loss', ''),
                info.get('dynSIG_loss', ''), info.get('varfloor_loss', ''),
                info.get('temp_loss', ''),
            ])

    # ── Loss computation ──────────────────────────────────────────────────────
    def _compute_loss(self, batch, is_train=True):
        # batch keys: obs_seq (B,H+1,3,h,w), actions (B,H,1), states (B,H+1,4)
        obs_seq = batch['obs_seq'].to(self.device)   # (B, H+1, 3, h, w)
        actions = batch['actions'].to(self.device)   # (B, H, 1)
        B, H1, C, h, w = obs_seq.shape
        H = H1 - 1

        # Clean z_0 for predictor input: encoder sees unmodified pixels, matching
        # the eval-time distribution exactly (no train/eval mismatch for dynamics).
        # Gradient from pred_loss flows back through this clean path into the encoder.
        obs_0 = obs_seq[:, 0]
        z_0   = self.model.encoder(obs_0)   # (B, d) — clean, gradient flows here

        # Augmented z_0 for collapse-prevention losses (varfloor, dynSIG) only.
        # Adding pixel noise to the collapse-prevention path keeps those gradients
        # non-trivial even near equilibrium, without polluting the predictor's
        # training distribution.  At eval (is_train=False) z_0_aug == z_0.
        if self.aug_noise_std > 0 and is_train:
            z_0_aug = self.model.encoder(
                (obs_0 + self.aug_noise_std * torch.randn_like(obs_0)).clamp(0., 1.)
            )
        else:
            z_0_aug = z_0
        d   = z_0.shape[-1]
        obs_rest = obs_seq[:, 1:].contiguous().view(B * H, C, h, w)
        if self.detach_targets:
            with torch.no_grad():
                _target_enc = (self.model.target_encoder
                               if self.use_target_encoder
                               else self.model.encoder)
                z_rest = _target_enc(obs_rest).view(B, H, d)
        else:
            from torch.utils.checkpoint import checkpoint
            z_rest = checkpoint(self.model.encoder, obs_rest,
                                use_reentrant=False).view(B, H, d)
        z_all = torch.cat([z_0.unsqueeze(1), z_rest], dim=1)  # (B, H+1, d)

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

        z_targets = z_all[:, 1:].detach() if self.detach_targets else z_all[:, 1:]

        # Multi-step unrolled prediction loss with optional window context.
        # Convention (matches CEM): window = [z_{t-W+1},...,z_t], [u_{t-W+1},...,u_t]
        # where u_t is the action APPLIED at z_t to produce z_{t+1}.
        W = self.predictor_window
        z_win_buf = [z_0] * W                                             # W copies of z_0
        u_win_buf = [torch.zeros(B, 1, device=self.device)] * (W - 1)    # W-1 padding zeros

        pred_loss = torch.zeros(1, device=self.device)
        for k in range(H):
            u_k = actions[:, k]                                # (B, 1)
            u_win_buf.append(u_k)                              # current action into window first
            z_stack = torch.stack(z_win_buf[-W:], dim=1)      # (B, W, d)
            u_stack = torch.stack(u_win_buf[-W:], dim=1)      # (B, W, 1)
            z_hat = self.model.predict(z_stack, u_stack)       # (B, d)
            pred_loss = pred_loss + F.mse_loss(z_hat, z_targets[:, k])
            z_win_buf.append(z_hat)
        pred_loss = pred_loss / H

        total_loss = self.lambda_pred * pred_loss
        info = {'pred_loss': pred_loss.item()}

        # Collapse diagnostic: mean step-size and norm of latent trajectory.
        with torch.no_grad():
            info['z_step'] = (z_all[:, 1:] - z_all[:, :-1]).norm(dim=-1).mean().item()
            info['z_norm'] = z_all.norm(dim=-1).mean().item()

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
            z_sig = z_all.reshape(-1, d) if self.sigreg_all_steps else z_all[:, 0]
            sig_loss = sigreg_loss(
                z_sig,
                num_slices=self.sigreg_num_slices,
                num_points=self.sigreg_num_points,
            )
            total_loss = total_loss + self.lambda_sigreg * sig_loss
            info['sigreg_loss'] = sig_loss.item()

        # L_dynSIG: dynamics-aware SIGreg.
        # Target covariance = controllability Gramian of local linearization (A, B),
        # valid only where the linearization holds — near equilibrium.
        # When dynSIG_near_eq_radius > 0, restrict to ||z_t - z*|| < radius.
        # Uses z_0_aug so the collapse-prevention gradient is non-trivial even when
        # online and target encoder embeddings coincide at equilibrium.
        if self.lambda_dynSIG > 0 and self._Sigma_target is not None:
            from losses.dyn_sigreg import dynsigreg_loss
            z0_dyn = z_0_aug
            if self.dynSIG_near_eq_radius > 0 and self._z_star_ema is not None:
                dz_ne = (z0_dyn - self._z_star_ema.detach()).norm(dim=1)
                mask_ne = dz_ne < self.dynSIG_near_eq_radius
                z0_dyn = z0_dyn[mask_ne]
            if z0_dyn.shape[0] >= 4:
                dyn_loss = dynsigreg_loss(
                    z0_dyn,
                    self._Sigma_target.detach(),
                    num_slices=self.sigreg_num_slices,
                    num_points=self.sigreg_num_points,
                )
                total_loss = total_loss + self.lambda_dynSIG * dyn_loss
                info['dynSIG_loss'] = dyn_loss.item()

        # L_varfloor: per-dimension variance floor derived from Gramian diagonal.
        # Fixes the vanishing-gradient failure mode of the Epps-Pulley CF test:
        # at perfect collapse (z_centered ≈ 0) the CF gradient is zero, but the
        # sqrt(var + eps) formulation keeps the gradient finite and growing.
        # Uses z_0_aug (augmented) so the variance floor applies to the slightly
        # perturbed distribution — same path as dynSIG for consistency.
        if self.lambda_varfloor > 0 and self._Sigma_target is not None:
            from losses.dyn_sigreg import gramian_varfloor_loss
            vf_loss = gramian_varfloor_loss(
                z_0_aug, self._Sigma_target.detach())
            total_loss = total_loss + self.lambda_varfloor * vf_loss
            info['varfloor_loss'] = vf_loss.item()

        # L_temp: temporal covariance consistency on near-equilibrium samples.
        # Enforces Sigma_1_res ≈ A Sigma_0 where Sigma_1_res removes B*c_t,
        # training encoder + predictor to preserve local spectral structure.
        if (is_train and self.lambda_temp > 0
                and self._A_jac_cache is not None
                and self._z_star_ema is not None):
            from losses.dyn_sigreg import temporal_consistency_loss
            z_star_d = self._z_star_ema.detach()
            dz_t   = z_all[:, 0] - z_star_d.unsqueeze(0)   # (B, d)
            dz_tp1 = z_all[:, 1] - z_star_d.unsqueeze(0)   # (B, d)
            near_eq = dz_t.norm(dim=1) < self.temp_near_eq_radius
            if near_eq.sum() >= 4:
                c_ne = self.model.action_encoder(actions[near_eq, 0])  # (N, m)
                temp_loss = temporal_consistency_loss(
                    dz_t[near_eq], dz_tp1[near_eq],
                    self._A_jac_cache, self._B_jac_cache,
                    c_t=c_ne,
                )
                total_loss = total_loss + self.lambda_temp * temp_loss
                info['temp_loss'] = temp_loss.item()

        # Inverse dynamics loss.
        # inv_frames=2: forward pair  ψ(z_t, z_{t+1}) → u_t   (SMWM / paper)
        # inv_frames=N (odd, N≥3): centered window of N frames → u_center
        #   e.g. N=3: (z_{t-1}, z_t, z_{t+1}) → u_t
        #        N=5: (z_{t-2}, z_{t-1}, z_t, z_{t+1}, z_{t+2}) → u_t
        # Wider windows give the IDM more context to subtract state-dependent
        # drift and isolate the action effect, at the cost of fewer valid targets.
        if self.lambda_inv > 0 and self.inv_head is not None:
            scale = self.inv_action_scale
            F_inv = self.inv_frames
            if F_inv == 2:
                # Forward pair (even window, non-centered)
                z_windows = torch.cat([z_all[:, :-1], z_all[:, 1:]], dim=-1)  # (B, H, 2d)
                u_tgt  = actions[:, :, 0] / scale                              # (B, H)
                n_pred = H
            else:
                # Centered odd window: predict u_t from [z_{t-k},...,z_{t+k}]
                half   = F_inv // 2
                n_pred = H + 2 - F_inv           # = H - 2*half + 1
                z_windows = torch.cat(
                    [z_all[:, i:i + n_pred] for i in range(F_inv)], dim=-1
                )                                # (B, n_pred, F_inv*d)
                u_tgt = actions[:, half:half + n_pred, 0] / scale  # (B, n_pred)
            u_hat = self.inv_head(
                z_windows.reshape(B * n_pred, F_inv * d)
            ).reshape(B, n_pred)
            inv_loss   = F.mse_loss(u_hat, u_tgt)
            total_loss = total_loss + self.lambda_inv * inv_loss
            info['inv_loss'] = inv_loss.item()

        # Endpoint action reconstruction (endpoint_sequence | local_pair | local_window).
        # Unlike inv_head, the decoder receives ONLY (z_start, z_end) — no intermediate
        # latent states.  Both endpoints are re-encoded here with grad enabled so the
        # loss shapes the encoder even when detach_targets=True (which would stop grad
        # through z_rest).
        if self.lambda_action_reconstruction > 0 and self.endpoint_action_decoder is not None:
            mode  = self.action_reconstruction_mode
            H_act = self.endpoint_action_decoder.H_act

            if mode == 'local_pair':
                # Random adjacent pair: (z_t, z_{t+1}) → u_t
                k = int(torch.randint(0, H, (1,)).item())
                obs_s   = obs_seq[:, k]          # (B, C, h, w)
                obs_e   = obs_seq[:, k + 1]      # (B, C, h, w)
                act_tgt = actions[:, k:k+1] / self.ar_action_scale   # (B, 1, 1)
                _start  = k                      # for physical decoder lookup below
                _end    = k + 1
            else:
                # local_window / endpoint_sequence: random subwindow of length H_act
                # start_idx ~ Uniform{0, …, H - H_act}
                max_start = H - H_act
                _start  = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
                _end    = _start + H_act
                obs_s   = obs_seq[:, _start]     # (B, C, h, w)
                obs_e   = obs_seq[:, _end]       # (B, C, h, w)
                act_tgt = actions[:, _start:_end] / self.ar_action_scale  # (B, H_act, 1)

            # Explicit re-encode with grad enabled.  z_rest is under no_grad when
            # detach_targets=True, so we cannot reuse z_all for z_end.
            z_s = self.model.encoder(obs_s)   # (B, d)  grad enabled → flows to encoder
            z_e = self.model.encoder(obs_e)   # (B, d)  grad enabled → flows to encoder
            action_hat = self.endpoint_action_decoder(z_s, z_e)  # (B, H_act, 1)
            ep_act_loss = F.mse_loss(action_hat, act_tgt)
            total_loss  = total_loss + self.lambda_action_reconstruction * ep_act_loss
            info['endpoint_action_loss'] = ep_act_loss.item()

            with torch.no_grad():
                info['endpoint_action_mae'] = (action_hat - act_tgt).abs().mean().item()
                ss_res = ((action_hat - act_tgt) ** 2).sum()
                ss_tot = ((act_tgt - act_tgt.mean()) ** 2).sum() + 1e-12
                info['endpoint_action_r2'] = float(1.0 - ss_res / ss_tot)
                # Baseline MSE: predict zero (= mean of normalized actions for random data)
                baseline_mse = (act_tgt ** 2).mean()
                info['endpoint_action_baseline_mse']    = baseline_mse.item()
                info['endpoint_action_normalized_mse']  = float(
                    ep_act_loss / (baseline_mse + 1e-12))
                # Per-step MSE: error profile along the reconstructed sequence
                step_mse = ((action_hat - act_tgt) ** 2).mean(dim=(0, 2))  # (H_act,)
                for _k in range(H_act):
                    info[f'action_mse_step_{_k}'] = step_mse[_k].item()

            # Physical-state endpoint decoder: diagnostic upper bound.
            # Uses ground-truth states (not encoder outputs) → gradient never reaches encoder.
            # Trained jointly so val loss tracks a meaningful ceiling each epoch.
            if 'states' in batch and self.phys_endpoint_decoder is not None:
                states_b = batch['states'].to(self.device).float()  # (B, H+1, 4)
                s_s = states_b[:, _start]   # (B, 4)
                s_e = states_b[:, _end]     # (B, 4)
                phys_hat = self.phys_endpoint_decoder(s_s, s_e)     # (B, H_act, 1)
                phys_ep_loss = F.mse_loss(phys_hat, act_tgt)
                total_loss   = total_loss + self.lambda_action_reconstruction * phys_ep_loss
                info['endpoint_phys_action_loss'] = phys_ep_loss.item()

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
            # Configurable per-dim weights [x, xdot, theta, thetadot].
            # Default: prioritise position & angle (visually prominent).
            # For IDM to learn, set high weights on xdot & thetadot
            # (state_loss_weights: [1, 100, 1, 100] in config).
            _w_cfg = self.cfg.get('state_loss_weights', [50., 0.1, 100., 1.])
            w      = torch.tensor(_w_cfg, dtype=torch.float32, device=self.device)
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

        # Fixed-point loss (legacy, kept for backward compat — subsumed by L_local).
        if is_train and self.lambda_fp > 0 and self._z_star_ema is not None:
            W = self.predictor_window
            z_star_win = self._z_star_ema.unsqueeze(0).unsqueeze(0).expand(1, W, -1)  # (1, W, d)
            u_zero_win = torch.zeros(1, W, 1, device=self.device)
            z_star_pred = self.model.predict(z_star_win, u_zero_win)
            fp_loss = F.mse_loss(z_star_pred, self._z_star_ema.unsqueeze(0).detach())
            total_loss = total_loss + self.lambda_fp * fp_loss
            info['fp_loss'] = fp_loss.item()

        # Mirror antisymmetry loss: encoder(flip(obs)) + encoder(obs) ≈ 2·z*
        # Horizontal flip of the cartpole image negates all state components
        # (x,ẋ,θ,θ̇) → (-x,-ẋ,-θ,-θ̇). Enforcing this antisymmetry around z*
        # breaks the |θ| vs θ sign degeneracy without any state labels.
        if is_train and self.lambda_mirror > 0 and self._z_star_ema is not None:
            obs_0_flip = torch.flip(obs_0, dims=[-1])          # flip width axis
            z_flip = self.model.encoder(obs_0_flip)            # (B, d)
            target = 2.0 * self._z_star_ema.detach().unsqueeze(0).expand(B, -1)
            mirror_loss = F.mse_loss(z_0 + z_flip, target)
            total_loss = total_loss + self.lambda_mirror * mirror_loss
            info['mirror_loss'] = mirror_loss.item()
            info['mirror_zstar_norm'] = self._z_star_ema.detach().norm().item()
            info['mirror_residual_norm'] = (z_0 + z_flip - target).detach().norm(dim=-1).mean().item()

        # ── spec_eig via JVP: fires every training step after warmup ─────────
        # Computes A_jac @ v̂_u in ONE forward pass (forward-mode JVP), avoiding
        # the expensive d×d Jacobian.  Fires every step → 10× more gradient
        # pressure vs. the old every-jacobian_every version.
        # Gradient flows through BOTH predictor (via JVP create_graph) AND encoder
        # (via v_u_hat) so they cooperate: encoder aligns v̂_u with A_jac's
        # dominant mode; predictor pushes that mode toward λ*.
        if (is_train and self.lambda_spec_eig > 0
                and self.epoch >= self.spec_eig_warmup_epochs
                and self.gt is not None
                and self._obs_v_u is not None
                and self._z_star_ema is not None):
            try:
                z_pert_ge  = self.model.encoder(self._obs_v_u).squeeze(0)   # (d,) grad on
                z_star_sg  = self._z_star_ema.detach()
                v_u_lat    = z_pert_ge - z_star_sg
                v_norm_sg  = v_u_lat.norm().detach()   # detach norm — no grad through scale
                if v_norm_sg > 1e-4:
                    v_u_hat  = v_u_lat / v_norm_sg     # (d,) grad flows to encoder
                    W_w      = self.predictor_window
                    d_w      = v_u_hat.shape[0]
                    # History window filled with z* (constant, detached)
                    z_hist   = (z_star_sg.unsqueeze(0).unsqueeze(1)
                                .expand(1, W_w - 1, d_w).clone())   # (1,W-1,d)
                    u_zero   = torch.zeros(1, W_w, 1, device=self.device)
                    z_base   = z_star_sg.unsqueeze(0)   # (1, d) — JVP base point
                    v_tang   = v_u_hat.unsqueeze(0)     # (1, d) — tangent (has encoder grad)

                    def _pred_last(z_l):
                        z_l_unsq = z_l.unsqueeze(1)     # (1, 1, d)
                        z_win = (torch.cat([z_hist, z_l_unsq], dim=1)
                                 if W_w > 1 else z_l_unsq)
                        return self.model.predict(z_win, u_zero)   # (1, d)

                    # JVP: df/dz_last @ v_tang = A_jac @ v̂_u  (one forward pass)
                    _, Av_batch = torch.autograd.functional.jvp(
                        _pred_last, z_base, v_tang, create_graph=True,
                    )
                    Av     = Av_batch.squeeze(0)         # (d,) grad → pred + encoder
                    lam_gt = torch.tensor(
                        float(np.real(self.gt.dominant_unstable_eigenvalue)),
                        dtype=Av.dtype, device=Av.device,
                    )
                    residual      = Av - lam_gt * v_u_hat   # (d,)
                    spec_eig_loss = (residual ** 2).sum()
                    total_loss    = total_loss + self.lambda_spec_eig * spec_eig_loss
                    info['spec_eig_loss'] = spec_eig_loss.item()
            except Exception as exc:
                warnings.warn(f'spec_eig JVP failed: {exc}')

        # ── Local linearization loss (L_local) ───────────────────────────────
        # Applied every step to self-loop (near-equilibrium) samples.
        # Forces f(z_t,u_t) ≈ z* + A(z_t-z*) + B·u_t using stop-grad A, B.
        # Subsumes fp_loss (z_t=z*, u_t=0 case) and directly trains B's direction.
        if (is_train and self.lambda_local > 0
                and self._A_jac_cache is not None
                and self._z_star_ema is not None
                and 'states' in batch):
            states_t = batch['states'][:, 0].to(self.device).float()
            sl_mask = states_t.abs().max(dim=1).values < self.local_state_threshold
            if sl_mask.sum() > 0:
                z_sl  = z_all[sl_mask, 0]    # (N, d)
                u_sl  = actions[sl_mask, 0]   # (N, 1)
                z_star_sg = self._z_star_ema.detach()   # (d,)
                A_sg = self._A_jac_cache                # (d, d) already detached
                B_sg = self._B_jac_cache                # (d, 1) already detached
                dz   = z_sl - z_star_sg.unsqueeze(0)   # (N, d)
                linear_pred = (z_star_sg.unsqueeze(0)
                               + torch.mm(dz, A_sg.T)       # A·(z-z*)
                               + torch.mm(u_sl, B_sg.T))    # B·u
                a_enc_sl = self.model.action_encoder(u_sl)
                z_pred_sl = self.model.predictor(z_sl, a_enc_sl)
                local_loss = F.mse_loss(z_pred_sl, linear_pred.detach())
                total_loss = total_loss + self.lambda_local * local_loss
                info['local_loss'] = local_loss.item()

        # ── Jacobian regularisation every jacobian_every steps ────────────────
        _needs_jac = (
            self.lambda_unstable > 0
            or self.lambda_dynSIG > 0
            or self.lambda_temp > 0
            or (self.lambda_spec > 0
                and self.true_unstable_eigs is not None
                and len(self.true_unstable_eigs) > 0)
            or (self.lambda_PBH > 0
                and self.true_unstable_eigs is not None
                and len(self.true_unstable_eigs) > 0)
        )
        if is_train and self.global_step % self.jacobian_every == 0 and _needs_jac:
            try:
                from control.jacobian import compute_jacobian_torch
                z_star_t = (self._z_star_ema if self._z_star_ema is not None
                            else z_all[:, 0].mean(dim=0).detach())
                A_jac, B_jac = compute_jacobian_torch(
                    self.model, z_star_t, self.device,
                )
                # B_eff: effective (d×1) B for raw scalar action, differentiable.
                # B_jac = ∂f/∂c (d×m); B_eff = B_jac @ W_enc where c = W_enc @ u.
                # Gradient flows through both B_jac (predictor) and W_enc (action encoder).
                if hasattr(self.model.action_encoder, 'W'):
                    B_eff_torch = B_jac @ self.model.action_encoder.W.weight  # (d, 1)
                else:
                    B_eff_torch = B_jac  # identity encoder: B_jac already (d, 1)

                # Cache for L_local / L_temp (used every step between Jacobian updates)
                self._A_jac_cache = A_jac.detach()
                self._B_jac_cache = B_jac.detach()
                self._B_eff_cache = B_eff_torch.detach()

                # Update Sigma_target via EMA for L_dynSIG.
                # Use B_eff (d×1) for the Gramian: correct for scalar-input control.
                if self.lambda_dynSIG > 0:
                    from losses.dyn_sigreg import (compute_controllability_gramian,
                                                   build_sigma_target)
                    with torch.no_grad():
                        W_T_new = compute_controllability_gramian(
                            self._A_jac_cache.float(),
                            self._B_eff_cache.float(),   # scalar B, not wide B_jac
                            T_g=self.dynSIG_T_g,
                        )
                        Sigma_new = build_sigma_target(W_T_new, alpha=self.dynSIG_alpha)
                        if self._Sigma_target is None:
                            self._Sigma_target = Sigma_new
                        else:
                            self._Sigma_target = (
                                self.dynSIG_beta * self._Sigma_target
                                + (1 - self.dynSIG_beta) * Sigma_new
                            )

                eigvals = torch.linalg.eigvals(A_jac)

                # L_unstable: data-driven instability margin.
                # No GT eigenvalues needed — only requires knowing the equilibrium
                # is unstable (η>0) and a data-derived upper bound ρ_max.
                if self.lambda_unstable > 0:
                    rho_jac = torch.max(torch.abs(eigvals))
                    lower   = torch.relu(1.0 + self.unstable_eta - rho_jac) ** 2
                    upper   = torch.relu(rho_jac - self.rho_max_estimate) ** 2
                    unstable_loss = lower + upper
                    total_loss = total_loss + self.lambda_unstable * unstable_loss
                    info['unstable_loss'] = unstable_loss.item()

                # L_spec (legacy GT-eigenvalue matching, backward compat)
                if (self.lambda_spec > 0
                        and self.true_unstable_eigs is not None
                        and len(self.true_unstable_eigs) > 0):
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
                    target_rho   = float(max(np.abs(self.true_unstable_eigs)))
                    rho_jac_t    = torch.max(torch.abs(eigvals))
                    target_rho_t = torch.tensor(target_rho, dtype=rho_jac_t.dtype,
                                                device=rho_jac_t.device)
                    spec_loss   = spec_loss + 2.0 * (rho_jac_t - target_rho_t) ** 2
                    total_loss  = total_loss + self.lambda_spec * spec_loss
                    info['spec_loss'] = spec_loss.item()

                    if self.lambda_spec_marginal > 0 and self.n_marginal_modes > 0:
                        dist_to_one = (eigvals.real - 1.0) ** 2 + eigvals.imag ** 2
                        sorted_idx  = torch.argsort(dist_to_one)
                        k = min(self.n_marginal_modes, len(sorted_idx))
                        marginal_loss = dist_to_one[sorted_idx[:k]].sum()
                        total_loss = total_loss + self.lambda_spec_marginal * marginal_loss
                        info['spec_marginal_loss'] = marginal_loss.item()

                # PBH: cosine alignment of B_eff with the unstable eigenvectors of A_jac.
                #
                # Replaces the log-barrier -log(sigma_min([λI-A, B])):
                #   - Log-barrier couples A-shaping (spec) with B-alignment (PBH), causing
                #     competing gradients and a plateau when spec converges and [λI-A] is
                #     nearly singular (making sigma_min extremely sensitive to tiny B changes).
                #   - Cosine loss is O(1), first-order through B_eff, and fully decoupled:
                #     spec shapes A's eigenvalue, PBH shapes B's direction independently.
                #
                # Loss = 1 - cos²(B_eff, V_u) ∈ [0,1].  0 = perfectly aligned, 1 = orthogonal.
                if (self.lambda_PBH > 0
                        and self.true_unstable_eigs is not None
                        and len(self.true_unstable_eigs) > 0):
                    with torch.no_grad():
                        eig_vals_r, eig_vecs_r = torch.linalg.eig(A_jac.detach().float())
                        unstable_mask = eig_vals_r.abs() >= 0.95
                        if unstable_mask.sum() == 0:
                            unstable_mask = eig_vals_r.abs() >= eig_vals_r.abs().max() * 0.99
                        v_u = eig_vecs_r[:, unstable_mask].real          # (d, n_unstable)
                        v_u = v_u / (v_u.norm(dim=0, keepdim=True) + 1e-8)

                    b_eff = B_eff_torch.to(dtype=torch.float32).flatten()  # (d,)
                    b_norm = b_eff / (b_eff.norm() + 1e-8)
                    cos_sq = (v_u.T @ b_norm.unsqueeze(1)) ** 2            # (n_unstable, 1)
                    pbh_loss = 1.0 - cos_sq.max()                          # 0=aligned, 1=orthogonal
                    total_loss = total_loss + self.lambda_PBH * pbh_loss
                    info['pbh_loss'] = pbh_loss.item()
            except Exception as exc:
                warnings.warn(f'Jacobian regularisation failed: {exc}')

        # Accumulate near-equilibrium growth rates for ρ_max estimation.
        # Collected every training step; ρ_max updated at epoch boundary in fit().
        if is_train and self.lambda_unstable > 0 and self._z_star_ema is not None:
            with torch.no_grad():
                z_star_d = self._z_star_ema.detach()
                dz_t  = (z_all[:, 0] - z_star_d.unsqueeze(0)).norm(dim=1)
                dz_t1 = (z_all[:, 1] - z_star_d.unsqueeze(0)).norm(dim=1)
                small_u  = actions[:, 0].abs().squeeze(-1) < 1.0
                near_eq  = (dz_t > 0.02) & (dz_t < 0.5) & small_u
                if near_eq.sum() > 0:
                    r_vals = dz_t1[near_eq] / (dz_t[near_eq] + 1e-6)
                    self._rho_buffer.extend(r_vals.cpu().tolist())

        info['total_loss'] = total_loss.item()
        return total_loss, info

    # ── Train / val epochs ────────────────────────────────────────────────────
    def train_epoch(self, train_loader):
        self.model.train()
        metrics = {}
        # Build param list once per epoch, not per step
        _params = list(self.model.parameters())
        if self.state_head is not None:
            _params += list(self.state_head.parameters())
        if self.endpoint_action_decoder is not None:
            _params += list(self.endpoint_action_decoder.parameters())
        if self.phys_endpoint_decoder is not None:
            _params += list(self.phys_endpoint_decoder.parameters())

        _t_data = _t_gpu = _n = 0.0
        _t_batch = time.time()
        for batch in tqdm(train_loader, desc=f'Epoch {self.epoch} [train]',
                          leave=False, dynamic_ncols=True):
            _t_data += time.time() - _t_batch
            _t0 = time.time()

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                loss, info = self._compute_loss(batch, is_train=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(_params, max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.use_target_encoder:
                self.model.update_target_encoder(self.target_encoder_momentum)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            _t_gpu += time.time() - _t0
            _n += 1

            for k, v in info.items():
                if isinstance(v, (int, float)):
                    metrics.setdefault(k, []).append(v)
            self.global_step += 1
            _t_batch = time.time()

        if _n > 0:
            print(f'  [timing] data={_t_data/_n*1e3:.1f}ms/batch  '
                  f'gpu={_t_gpu/_n*1e3:.1f}ms/batch  '
                  f'({_n} batches)')
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    @torch.no_grad()
    def val_epoch(self, val_loader):
        self.model.eval()
        metrics = {}
        for batch in tqdm(val_loader, desc=f'Epoch {self.epoch} [val]',
                          leave=False, dynamic_ncols=True):
            with torch.cuda.amp.autocast(enabled=self._amp_enabled):
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
        best_state      = None
        _best_val_epoch = 0
        _spec_conv_epoch = None   # first epoch where spec_loss < 0.01
        spec_thresh     = float(self.cfg.get('spec_converge_thresh', 0.01))

        # Resume from checkpoint if requested
        start_epoch = 0
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            start_epoch = self.epoch + 1
            print(f'[train] Resuming from epoch {start_epoch}')

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

        for epoch in range(start_epoch, epochs):
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

            # Update ρ_max from growth rates collected during the previous epoch.
            if self.lambda_unstable > 0 and len(self._rho_buffer) > 10:
                r_arr = np.array(self._rho_buffer)
                new_rho_max = float(np.clip(r_arr.mean() + 2.0 * r_arr.std(),
                                            1.0 + self.unstable_eta + 0.01, 1.6))
                print(f'[train] ρ_max: {self.rho_max_estimate:.4f}'
                      f' → {new_rho_max:.4f}  (n={len(self._rho_buffer)})')
                self.rho_max_estimate = new_rho_max
                self._rho_buffer = []

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
            if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
                self.save_checkpoint(tag=f'epoch{epoch+1:04d}')
            dt = time.time() - t0
            state_str  = f"  state={tr.get('state_loss',     0):.4f}" if 'state_loss'     in tr else ''
            inv_str    = f"  inv={tr.get('inv_loss',       0):.4f}" if 'inv_loss'       in tr else ''
            ea_str     = f"  ea={tr.get('enc_anchor_loss', 0):.4f}" if 'enc_anchor_loss' in tr else ''
            fp_str       = f"  fp={tr.get('fp_loss',         0):.4f}" if 'fp_loss'         in tr else ''
            local_str    = f"  local={tr.get('local_loss',   0):.4f}" if 'local_loss'     in tr else ''
            unstable_str = f"  ρ={tr.get('unstable_loss',   0):.4f}" if 'unstable_loss'  in tr else ''
            spec_str     = (f"  spec={tr.get('spec_loss', 0):.4f}"
                            + (f"+m{tr.get('spec_marginal_loss', 0):.4f}"
                               if 'spec_marginal_loss' in tr else '')) if 'spec_loss' in tr else ''
            spec_eig_str = f"  spec_eig={tr.get('spec_eig_loss', 0):.4f}" if 'spec_eig_loss' in tr else ''
            anchor_str   = f"  anc={tr.get('anchor_loss',   0):.4f}" if 'anchor_loss'    in tr else ''
            sig_str      = f"  sig={tr.get('sigreg_loss',   0):.4f}" if 'sigreg_loss'    in tr else ''
            zmove_str    = (f"  dz={tr.get('z_step', 0):.4f}"
                            f"(|z|={tr.get('z_norm', 0):.3f})"
                            ) if 'z_step' in tr else ''
            dynsig_str   = f"  dynSIG={tr.get('dynSIG_loss', 0):.4f}" if 'dynSIG_loss'   in tr else ''
            varfloor_str = f"  vf={tr.get('varfloor_loss',  0):.4f}" if 'varfloor_loss'  in tr else ''
            temp_str     = f"  temp={tr.get('temp_loss',     0):.4f}" if 'temp_loss'      in tr else ''
            pbh_str      = f"  pbh={tr.get('pbh_loss',       0):.4f}" if 'pbh_loss'       in tr else ''
            mirror_str   = (f"  mir={tr.get('mirror_loss', 0):.4f}"
                            f"(z*={tr.get('mirror_zstar_norm', 0):.3f}"
                            f",res={tr.get('mirror_residual_norm', 0):.3f})"
                            ) if 'mirror_loss' in tr else ''
            ep_act_str   = (f"  ep_act={tr.get('endpoint_action_loss', 0):.4f}"
                            + (f"(phys={tr.get('endpoint_phys_action_loss', 0):.4f})"
                               if 'endpoint_phys_action_loss' in tr else '')
                            + (f" nMSE={tr.get('endpoint_action_normalized_mse', 0):.3f}"
                               if 'endpoint_action_normalized_mse' in tr else '')
                            ) if 'endpoint_action_loss' in tr else ''
            print(f'[Epoch {epoch+1:3d}/{epochs}]'
                  f'  train={tr.get("total_loss",0):.4f}'
                  f'  val={val_loss:.4f}'
                  f'  pred={tr.get("pred_loss",0):.4f}'
                  f'{state_str}{inv_str}{ea_str}{fp_str}{local_str}{unstable_str}{spec_str}{spec_eig_str}{anchor_str}{sig_str}{dynsig_str}{varfloor_str}{temp_str}{pbh_str}{mirror_str}{ep_act_str}{zmove_str}'
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
        #   • No spectral reg (lambda_spec=0): always use best-val — spec is irrelevant.
        #   • Spectral reg converged at or before best-val epoch → best-val model
        #     has both low val loss AND correct spectral properties.
        #   • Otherwise spectral convergence lags best-val → keep final-epoch weights.
        no_spec = (self.lambda_spec <= 0 and float(self.cfg.get('lambda_spec_marginal', 0)) <= 0
                   and float(self.cfg.get('lambda_spec_eig', 0)) <= 0)
        if no_spec or (_spec_conv_epoch is not None and _spec_conv_epoch <= _best_val_epoch):
            self.model.load_state_dict({k: v.to(self.device)
                                         for k, v in best_state.items()})
            spec_note = 'no spec reg' if no_spec else f'spec converged epoch {_spec_conv_epoch+1}'
            print(f'[train] Using best-val model  '
                  f'epoch={_best_val_epoch+1}  val={self.best_val_loss:.4f}  '
                  f'({spec_note})')
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
            self.model, z_star, self.device,
        )

    def save_checkpoint(self, tag='latest'):
        path = self.save_dir / f'checkpoint_{tag}.pt'
        payload = {'epoch': self.epoch, 'global_step': self.global_step,
                   'model_state': self.model.state_dict(),
                   'optimizer_state': self.optimizer.state_dict(),
                   'scheduler_state': self.scheduler.state_dict(),
                   'scaler_state': self.scaler.state_dict(),
                   'best_val_loss': self.best_val_loss,
                   'config': self.model.get_config_dict()}
        if self.inv_head is not None:
            payload['inv_head_state'] = self.inv_head.state_dict()
        if self.endpoint_action_decoder is not None:
            payload['endpoint_action_decoder_state'] = self.endpoint_action_decoder.state_dict()
        if self.phys_endpoint_decoder is not None:
            payload['phys_endpoint_decoder_state'] = self.phys_endpoint_decoder.state_dict()
        torch.save(payload, path)
        # Always keep a 'latest' copy for easy resume
        if tag != 'latest':
            latest = self.save_dir / 'checkpoint_latest.pt'
            import shutil
            shutil.copy2(path, latest)
        print(f'[ckpt] saved → {path}')

    def load_checkpoint(self, path):
        """Load model, optimizer, scheduler state from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        # Optimizer param groups may differ across configs (e.g. endpoint decoder added
        # mid-run); load gracefully rather than crashing on group-count mismatch.
        try:
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
        except (ValueError, KeyError) as _exc:
            warnings.warn(f'[ckpt] optimizer state mismatch ({_exc}) — starting optimizer fresh')
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.epoch       = ckpt.get('epoch', 0)
        self.global_step = ckpt.get('global_step', 0)
        if 'scaler_state' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state'])
        if 'inv_head_state' in ckpt and self.inv_head is not None:
            self.inv_head.load_state_dict(ckpt['inv_head_state'])
        if 'endpoint_action_decoder_state' in ckpt and self.endpoint_action_decoder is not None:
            self.endpoint_action_decoder.load_state_dict(ckpt['endpoint_action_decoder_state'])
        if 'phys_endpoint_decoder_state' in ckpt and self.phys_endpoint_decoder is not None:
            self.phys_endpoint_decoder.load_state_dict(ckpt['phys_endpoint_decoder_state'])
        print(f'[ckpt] resumed from {path}  (epoch={self.epoch}, step={self.global_step})')
