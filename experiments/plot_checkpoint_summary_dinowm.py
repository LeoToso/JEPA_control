#!/usr/bin/env python
"""Single-checkpoint 5-panel summary figure for CartPole DINO-WM.

Panels (left → right):
  1. Phase portrait  — GT (green) vs learned (plasma) vector field on θ–θ̇ plane
  2. H-step prediction error  — ||mean_patch(f_H(z_t,0)) − mean_patch(z_{t+H}^GT)||
  3. Planning cost  — log10||mean_patch(f_H(z_t,0)) − mean_patch(z_goal)||², zero action
  4. Latent norm divergence  — encoded-GT vs predictor-rollout mean-patch norms over time
  5. Cosine alignment  — cos(Δz_GT, Δz_pred) vs rollout step

Usage:
  python experiments/plot_checkpoint_summary_dinowm.py \\
      --dino-wm-dir ~/dino_wm \\
      --ckpt  /home/dz2478/dino_wm/outputs/2026-09-08/17-46-00/checkpoints/model_179.pth \\
      --data  /mnt/t7shield/dinowm_cartpole \\
      --title "DINO-WM" \\
      --out   results/summary_dinowm.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn.functional as F

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

# ── dino-wm constants ─────────────────────────────────────────────────────────
FRAMESKIP   = 5
D_VIS       = 384
D_PROP_OUT  = 10
D_ACT_OUT   = 10
D_TOTAL     = D_VIS + D_PROP_OUT + D_ACT_OUT   # 404
ACT_IN_DIM  = 1
PROP_IN_DIM = 4
IMG_SIZE    = 112
_NORM_MEAN  = torch.tensor([0.5, 0.5, 0.5])
_NORM_STD   = torch.tensor([0.5, 0.5, 0.5])
ACTION_LB, ACTION_UB = -10., 10.
MACRO_DT    = 0.02 * FRAMESKIP   # 0.1 s


# ── dino-wm utilities (self-contained) ────────────────────────────────────────

def setup_dinowm(dino_wm_dir):
    p = str(Path(dino_wm_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def make_env(seed=0):
    from envs.cartpole_visual import ContinuousCartpoleVisual
    return ContinuousCartpoleVisual(
        frame_skip=FRAMESKIP, image_size=IMG_SIZE,
        action_range=(-10., 10.),
        mass_cart=1.0, mass_pole=0.1,
        pole_length=0.5, gravity=9.8, dt=0.02,
        theta_threshold=1.2,
        seed=seed)


def load_action_stats(data_path, device):
    p = Path(data_path)
    actions     = torch.load(p / 'actions.pth').float()
    seq_lengths = torch.load(p / 'seq_lengths.pth')
    all_acts = [actions[i, :int(seq_lengths[i])] for i in range(len(seq_lengths))]
    all_acts = torch.vstack(all_acts)
    mean = all_acts.mean(0).to(device)
    std  = all_acts.std(0).to(device).clamp(min=1e-6)
    print(f'[action stats]  mean={mean.item():.4f}  std={std.item():.4f}')
    return mean, std


def load_model(ckpt_path, dino_wm_dir, device):
    setup_dinowm(dino_wm_dir)
    from models.dino import DinoV2Encoder
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    parts = {}
    for k in ['predictor', 'proprio_encoder', 'action_encoder']:
        parts[k] = ckpt[k].to(device).eval()
    parts['encoder'] = DinoV2Encoder(
        name='dinov2_vits14', feature_key='x_norm_patchtokens').to(device).eval()
    n_params = sum(p.numel() for p in parts['predictor'].parameters())
    print(f'[model] predictor params={n_params/1e6:.1f}M')
    return parts


def preprocess_obs(obs_hwc, device):
    x = torch.from_numpy(obs_hwc.copy()).float().permute(2, 0, 1) / 255.0
    if x.shape[1] != IMG_SIZE or x.shape[2] != IMG_SIZE:
        x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                          mode='bilinear', align_corners=False).squeeze(0)
    x = (x - _NORM_MEAN[:, None, None]) / _NORM_STD[:, None, None]
    return x.unsqueeze(0).to(device)


@torch.no_grad()
def encode_frame(parts, obs_hwc, state, act_norm_np, device):
    """Returns (1, N, D_TOTAL) tensor (mean across patch dim → (D_TOTAL,) via [0].mean(0))."""
    vis  = parts['encoder'](preprocess_obs(obs_hwc, device))
    N    = vis.shape[1]
    _pe  = next(parts['proprio_encoder'].parameters()).dtype
    prop = parts['proprio_encoder'](
        torch.as_tensor(state[:PROP_IN_DIM].astype(np.float32),
                        device=device).to(_pe).unsqueeze(0).unsqueeze(0))
    _ae  = next(parts['action_encoder'].parameters()).dtype
    act  = parts['action_encoder'](
        torch.as_tensor(act_norm_np[:ACT_IN_DIM].astype(np.float32),
                        device=device).to(_ae).unsqueeze(0).unsqueeze(0))
    return torch.cat([vis, prop[:, :1].expand(-1, N, -1),
                      act[:, :1].expand(-1, N, -1)], dim=2)


@torch.no_grad()
def predictor_step(parts, z_ctx, a_real, act_mean, act_std):
    """One predictor step. Returns (z_next (1,N,D), new_z_ctx (1,T,N,D))."""
    _, T, N, D = z_ctx.shape
    ae = parts['action_encoder']
    ae_dtype = next(ae.parameters()).dtype
    a_n = float((a_real - float(act_mean[0].cpu())) / float(act_std[0].cpu()))
    a_emb = ae(torch.tensor([[[a_n]]], device=z_ctx.device, dtype=ae_dtype))
    a_tile = a_emb.expand(1, N, -1)
    z_last = torch.cat([z_ctx[:, -1:, :, :D_VIS+D_PROP_OUT],
                        a_tile.unsqueeze(1)], dim=-1)
    z_in   = torch.cat([z_ctx[:, :-1], z_last], dim=1)
    z_nf   = parts['predictor'](z_in.reshape(1, T*N, D)).reshape(1, T, N, D)[:, -1]
    return z_nf, torch.cat([z_ctx[:, 1:], z_nf.unsqueeze(1)], dim=1)


def _initial_z_ctx(parts, obs_hwc, state, num_hist, device, act_mean, act_std):
    zero_norm = np.zeros(ACT_IN_DIM, dtype=np.float32)
    frame = encode_frame(parts, obs_hwc, state, zero_norm, device)
    return frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()


# ── ridge probe ───────────────────────────────────────────────────────────────

class RidgeProbe:
    """Linear probe: state ≈ W @ z_mean_patch + b, pure numpy ridge regression."""

    def fit(self, Z: np.ndarray, X: np.ndarray, alpha: float = 1e-3):
        N, D = Z.shape
        Z_aug = np.concatenate([Z, np.ones((N, 1))], axis=1)
        A = Z_aug.T @ Z_aug
        A += alpha * np.eye(D + 1)
        A[-1, -1] = 0.
        self.W = np.linalg.solve(A, Z_aug.T @ X)
        return self

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        Z = np.atleast_2d(Z)
        return np.concatenate([Z, np.ones((len(Z), 1))], axis=1) @ self.W


def build_probe(parts, act_mean, act_std, num_hist, device, n_samples=200, seed=123):
    rng = np.random.RandomState(seed)
    states = rng.uniform(
        [-.2, -.5, -.15, -1.], [.2, .5, .15, 1.],
        size=(n_samples, 4)).astype(np.float32)
    env = make_env(seed=seed)
    zero_norm = np.zeros(ACT_IN_DIM, dtype=np.float32)
    zs = []
    for s in states:
        obs, _, _ = env.reset_to_state(s)
        frame = encode_frame(parts, obs, s, zero_norm, device)
        zs.append(frame[0].mean(0).cpu().numpy())
    env.close()
    return RidgeProbe().fit(np.array(zs), states, 1e-3)


# ── GT unstable eigenvector ───────────────────────────────────────────────────

def gt_unstable_eigvec(eps=1e-4):
    """FD linearise GT CartPole at origin → most unstable eigenvector (in θ–θ̇)."""
    A = np.zeros((4, 4), dtype=np.float64)
    env = make_env(seed=42)
    x0 = np.zeros(4, np.float32)
    for j in range(4):
        d = np.zeros(4, np.float32); d[j] = eps
        env.reset_to_state((x0 + d).astype(np.float64)); _, xp, _, _, _ = env.step(0.)
        env.reset_to_state((x0 - d).astype(np.float64)); _, xm, _, _, _ = env.step(0.)
        A[:, j] = (xp - xm) / (2. * eps)
    env.close()
    vals, vecs = np.linalg.eig(A)
    order = np.argsort(np.abs(vals.real))[::-1]
    v = vecs[:, order[0]].real
    return v / (np.linalg.norm(v) + 1e-12)


# ── rollout helpers ───────────────────────────────────────────────────────────

def gt_rollout_latents(parts, state0, n_steps, act_mean, act_std,
                       num_hist, device, action=0.):
    """Step GT env n_steps times, encode every frame → mean-patch latents array (n_steps+1, D)."""
    env = make_env(seed=0)
    zero_norm = np.zeros(ACT_IN_DIM, dtype=np.float32)
    obs, state, _ = env.reset_to_state(state0.astype(np.float64))
    frame = encode_frame(parts, obs, state, zero_norm, device)
    latents = [frame[0].mean(0).cpu().numpy()]
    am = float(act_mean[0].cpu()); as_ = float(act_std[0].cpu())
    for _ in range(n_steps):
        obs, state, _, done, _ = env.step(action)
        a_n = np.array([(action - am) / as_], dtype=np.float32)
        frame = encode_frame(parts, obs, state, a_n, device)
        latents.append(frame[0].mean(0).cpu().numpy())
        if done:
            break
    env.close()
    return np.array(latents)


def pred_rollout_latents(parts, state0, n_steps, act_mean, act_std,
                         num_hist, device, action=0.):
    """Encode initial frame only, then roll predictor forward n_steps → mean-patch latents (n_steps+1, D)."""
    env = make_env(seed=0)
    zero_norm = np.zeros(ACT_IN_DIM, dtype=np.float32)
    obs, state, _ = env.reset_to_state(state0.astype(np.float64))
    env.close()
    frame = encode_frame(parts, obs, state, zero_norm, device)
    z_ctx = frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()
    latents = [z_ctx[0, -1].mean(0).cpu().numpy()]
    for _ in range(n_steps):
        z_next, z_ctx = predictor_step(parts, z_ctx, action, act_mean, act_std)
        latents.append(z_next[0].mean(0).cpu().numpy())
    return np.array(latents)


# ── Panel 1: phase portrait ───────────────────────────────────────────────────

def panel_phase_portrait(ax, parts, probe, act_mean, act_std, num_hist, device,
                         theta_max, rate_max, n_theta, n_rate):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)
    theta_mesh, rate_mesh = np.meshgrid(theta_deg, rate_deg)
    gt_u = np.zeros_like(theta_mesh); gt_v = np.zeros_like(theta_mesh)
    lr_u = np.zeros_like(theta_mesh); lr_v = np.zeros_like(theta_mesh)
    speed = np.zeros_like(theta_mesh)

    env = make_env(seed=991)
    zero_norm = np.zeros(ACT_IN_DIM, dtype=np.float32)
    for idx in np.ndindex(theta_mesh.shape):
        state = np.array([0., 0., np.deg2rad(theta_mesh[idx]),
                          np.deg2rad(rate_mesh[idx])], dtype=np.float32)
        obs, s0, _ = env.reset_to_state(state)
        _, s1, _, _, _ = env.step(0.)
        gt_u[idx] = np.rad2deg(s1[2] - s0[2])
        gt_v[idx] = np.rad2deg(s1[3] - s0[3])

        frame = encode_frame(parts, obs, state, zero_norm, device)
        z_ctx = frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()
        z_next, _ = predictor_step(parts, z_ctx, 0., act_mean, act_std)
        z0_m  = frame[0].mean(0).cpu().numpy()
        z1_m  = z_next[0].mean(0).cpu().numpy()
        dec   = probe(np.stack([z0_m, z1_m]))
        lr_u[idx] = np.rad2deg(dec[1, 2] - dec[0, 2])
        lr_v[idx] = np.rad2deg(dec[1, 3] - dec[0, 3])
        speed[idx] = np.hypot(lr_u[idx], lr_v[idx])
    env.close()

    frac = 0.45
    dx = 2 * theta_max / max(n_theta - 1, 1)
    dy = 2 * rate_max  / max(n_rate  - 1, 1)

    def _normalise(u, v):
        n = np.sqrt(u**2 + v**2 + 1e-14)
        return u / n, v / n

    gt_un, gt_vn = _normalise(gt_u, gt_v)
    lr_un, lr_vn = _normalise(lr_u, lr_v)

    ax.quiver(theta_mesh, rate_mesh,
              gt_un * frac * dx, gt_vn * frac * dy,
              color='#2ecc40', alpha=.7, pivot='mid',
              scale=1, scale_units='xy', angles='xy',
              width=.002, headwidth=4, headlength=4)
    ax.quiver(theta_mesh, rate_mesh,
              lr_un * frac * dx, lr_vn * frac * dy,
              color='black', alpha=0.75, pivot='mid',
              scale=1, scale_units='xy', angles='xy',
              width=.0025, headwidth=4, headlength=4)
    ax.scatter([0.], [0.], marker='*', s=200, color='red', zorder=5)
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
    ax.set_ylabel(r'$\dot{\theta}$ [deg/s]', fontsize=14)
    ax.legend(handles=[
        Line2D([0], [0], color='#2ecc40', lw=4, label='GT'),
        Line2D([0], [0], color='black',   lw=3, label='Learned'),
        Line2D([0], [0], marker='*', color='red', ls='None',
               ms=13, label='Equilibrium'),
    ], fontsize=14, loc='upper right')
    ax.tick_params(labelsize=14)
    ax.grid(alpha=.2)


# ── Panel 2: H-step prediction error ─────────────────────────────────────────

def panel_pred_error(ax, fig, parts, act_mean, act_std, num_hist, device,
                     theta_max, rate_max, n_theta, n_rate, H):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)
    err = np.zeros((n_rate, n_theta))

    am = float(act_mean[0].cpu()); as_ = float(act_std[0].cpu())
    env = make_env(seed=993)
    zero_norm = np.zeros(ACT_IN_DIM, dtype=np.float32)
    for ri, r_deg in enumerate(rate_deg):
        for ti, t_deg in enumerate(theta_deg):
            state = np.array([0., 0., np.deg2rad(t_deg), np.deg2rad(r_deg)],
                             dtype=np.float32)
            obs, s, _ = env.reset_to_state(state)

            # Predictor: roll H steps from initial encode
            frame = encode_frame(parts, obs, s, zero_norm, device)
            z_ctx = frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()
            for _ in range(H):
                z_next, z_ctx = predictor_step(parts, z_ctx, 0., act_mean, act_std)
            z_pred_H = z_next[0].mean(0).cpu().numpy()

            # GT: step env H times, encode final frame
            env.reset_to_state(state)
            obs_t, s_t = obs, s
            done = False
            for _ in range(H):
                obs_next, s_next, _, done, _ = env.step(0.)
                obs_t, s_t = obs_next, s_next
                if done:
                    break
            a_n = np.zeros(ACT_IN_DIM, dtype=np.float32)
            z_gt_H = encode_frame(parts, obs_t, s_t, a_n, device)[0].mean(0).cpu().numpy()

            err[ri, ti] = float(np.linalg.norm(z_pred_H - z_gt_H))
    env.close()

    im = ax.pcolormesh(theta_deg, rate_deg, err, cmap='YlOrRd', shading='auto', vmin=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\|\hat{z}_H - z_H^{\rm GT}\|$', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax.contour(theta_deg, rate_deg, err, levels=6,
               colors='k', linewidths=0.5, alpha=0.35)
    ax.scatter([0.], [0.], marker='*', s=200, color='lime', zorder=5,
               edgecolors='darkgreen', lw=0.8, label='Equilibrium')
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
    ax.set_ylabel(r'$\dot{\theta}$ [deg/s]', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.15)


# ── Panel 3: planning cost ────────────────────────────────────────────────────

def panel_planning_cost(ax, fig, parts, act_mean, act_std, num_hist, device,
                        theta_max, rate_max, n_theta, n_rate, H):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)

    # Encode goal state (origin)
    env_g = make_env(seed=999)
    goal_obs, goal_state, _ = env_g.reset_to_state(np.zeros(4, np.float32))
    env_g.close()
    zero_norm = np.zeros(ACT_IN_DIM, dtype=np.float32)
    goal_frame = encode_frame(parts, goal_obs, goal_state, zero_norm, device)
    z_goal_mean = goal_frame[0].mean(0).cpu().numpy()

    cost = np.zeros((n_rate, n_theta))
    env = make_env(seed=992)
    for ri, r_deg in enumerate(rate_deg):
        for ti, t_deg in enumerate(theta_deg):
            state = np.array([0., 0., np.deg2rad(t_deg), np.deg2rad(r_deg)],
                             dtype=np.float32)
            obs, s, _ = env.reset_to_state(state)
            frame = encode_frame(parts, obs, s, zero_norm, device)
            z_ctx = frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()
            for _ in range(H):
                z_next, z_ctx = predictor_step(parts, z_ctx, 0., act_mean, act_std)
            z_H = z_next[0].mean(0).cpu().numpy()
            diff = z_H - z_goal_mean
            cost[ri, ti] = float(np.dot(diff, diff))
    env.close()

    log_cost = np.log10(np.maximum(cost, 1e-12))
    im = ax.pcolormesh(theta_deg, rate_deg, log_cost, cmap='RdYlBu_r', shading='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\log_{10}\|\hat{z}_H - z_{\rm goal}\|^2$', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax.contour(theta_deg, rate_deg, log_cost, levels=8,
               colors='k', linewidths=0.5, alpha=0.35)
    ax.scatter([0.], [0.], marker='*', s=200, color='lime', zorder=5,
               edgecolors='darkgreen', lw=0.8, label='Equilibrium')
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
    ax.set_ylabel(r'$\dot{\theta}$ [deg/s]', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.15)


# ── Panel 4: latent norm divergence ──────────────────────────────────────────

def panel_latent_norm(ax, parts, act_mean, act_std, num_hist, device,
                      theta0_rad, n_steps):
    state0 = np.array([0., 0., theta0_rad, 0.], dtype=np.float32)
    gt_lat   = gt_rollout_latents(parts, state0, n_steps, act_mean, act_std,
                                   num_hist, device)
    pred_lat = pred_rollout_latents(parts, state0, n_steps, act_mean, act_std,
                                    num_hist, device)

    n = min(len(gt_lat), len(pred_lat))
    gt_norm   = np.linalg.norm(gt_lat[:n],   axis=1)
    pred_norm = np.linalg.norm(pred_lat[:n], axis=1)
    time = np.arange(n) * MACRO_DT

    ax.plot(time, gt_norm,   color='steelblue',   lw=2,   label='encoded GT')
    ax.plot(time, pred_norm, '--', color='darkorange', lw=2, label='predicted')
    ax.set_xlabel('time [s]', fontsize=14)
    ax.set_ylabel('mean-patch latent norm', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.25)


# ── Panel 5: cosine alignment ─────────────────────────────────────────────────

def panel_cosine_alignment(ax, parts, act_mean, act_std, num_hist, device,
                           v_u, H_max=20, n_starts=6, alpha_max=0.25):
    eps_vals = np.linspace(0.02, alpha_max, n_starts)
    color = '#2166ac'
    cos_curves = []

    for eps in eps_vals:
        x0 = (eps * v_u).astype(np.float32)
        gt_lat   = gt_rollout_latents(parts, x0, H_max, act_mean, act_std,
                                      num_hist, device)
        pred_lat = pred_rollout_latents(parts, x0, H_max, act_mean, act_std,
                                        num_hist, device)
        n = min(len(gt_lat), len(pred_lat))
        dz_gt   = gt_lat[:n]   - gt_lat[0][None]
        dz_pred = pred_lat[:n] - pred_lat[0][None]
        dot  = (dz_gt * dz_pred).sum(axis=1)
        norm = (np.linalg.norm(dz_gt,   axis=1) *
                np.linalg.norm(dz_pred, axis=1))
        cos = np.where(norm > 1e-6, dot / np.where(norm > 1e-6, norm, 1.), np.nan)
        cos_curves.append(cos)

    max_len = max(len(c) for c in cos_curves)
    Hs = np.arange(max_len) * MACRO_DT
    mat = np.array([np.pad(c.astype(float), (0, max_len - len(c)),
                           constant_values=np.nan)
                    for c in cos_curves])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat,  axis=0)

    ax.plot(Hs, mean, color=color, lw=2.2)
    ax.fill_between(Hs,
                    np.clip(mean - std, -1, 1),
                    np.clip(mean + std, -1, 1),
                    color=color, alpha=0.15)
    ax.axhline(0., color='gray', ls='--', lw=0.8)
    ax.set_ylim(-0.25, 1.15)
    ax.set_xlabel('Rollout time [s]', fontsize=14)
    ax.set_ylabel(r'$\cos(\Delta z_{GT},\, \Delta z_{pred})$', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.25)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dino-wm-dir', default=str(Path.home() / 'dino_wm'))
    p.add_argument('--ckpt',  required=True, help='Checkpoint path (.pth)')
    p.add_argument('--data',  default='/mnt/t7shield/dinowm_cartpole',
                   help='Dataset dir for action statistics')
    p.add_argument('--num-hist', type=int, default=3)
    p.add_argument('--title', required=True, help='Figure suptitle')
    p.add_argument('--out',   required=True, help='Output image path')

    # Panel 1: phase portrait
    p.add_argument('--phase-theta-max', type=float, default=5.,
                   help='Half-range θ [deg]')
    p.add_argument('--phase-rate-max',  type=float, default=20.,
                   help='Half-range θ̇ [deg/s]')
    p.add_argument('--phase-theta-pts', type=int,   default=13)
    p.add_argument('--phase-rate-pts',  type=int,   default=11)
    p.add_argument('--probe-samples',   type=int,   default=200)

    # Panels 2+3: prediction error and planning cost
    p.add_argument('--pred-theta-max',  type=float, default=25.)
    p.add_argument('--pred-rate-max',   type=float, default=100.)
    p.add_argument('--pred-theta-pts',  type=int,   default=13)
    p.add_argument('--pred-rate-pts',   type=int,   default=11)
    p.add_argument('--horizon',         type=int,   default=3,
                   help='H for panels 2 and 3')

    # Panel 4: latent norm
    p.add_argument('--theta0',     type=float, default=0.1,
                   help='Initial θ [rad] for latent norm panel')
    p.add_argument('--norm-steps', type=int,   default=20)

    # Panel 5: cosine alignment
    p.add_argument('--n-steps',   type=int,   default=20,
                   help='H_max for cosine alignment panel')
    p.add_argument('--n-starts',  type=int,   default=6,
                   help='Number of perturbation magnitudes')
    p.add_argument('--alpha-max', type=float, default=0.25,
                   help='Max perturbation magnitude along unstable eigenvector')

    p.add_argument('--device', default='cuda')
    p.add_argument('--skip-panels', type=int, nargs='+', default=[],
                   metavar='N',
                   help='Panel numbers to skip (1=phase portrait, '
                        '2=pred error, 3=planning cost, '
                        '4=latent norm, 5=cosine alignment)')
    args = p.parse_args()

    device = torch.device(args.device)

    print('[load] model …')
    parts = load_model(args.ckpt, args.dino_wm_dir, device)
    act_mean, act_std = load_action_stats(args.data, device)

    print('[probe] fitting ridge probe …')
    probe = build_probe(parts, act_mean, act_std, args.num_hist,
                        device, args.probe_samples)

    print('[GT] computing unstable eigenvector …')
    v_u = gt_unstable_eigvec()

    skip   = set(args.skip_panels)
    active = [i for i in (1, 2, 3, 4, 5) if i not in skip]
    n_panels = len(active)
    fig, axes_all = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6.5))
    if n_panels == 1:
        axes_all = [axes_all]
    ax_iter = iter(axes_all)
    if 1 not in skip:
        print('[panel 1] phase portrait …')
        panel_phase_portrait(
            next(ax_iter), parts, probe, act_mean, act_std, args.num_hist, device,
            args.phase_theta_max, args.phase_rate_max,
            args.phase_theta_pts, args.phase_rate_pts)

    if 2 not in skip:
        print(f'[panel 2] H={args.horizon} prediction error …')
        panel_pred_error(
            next(ax_iter), fig, parts, act_mean, act_std, args.num_hist, device,
            args.pred_theta_max, args.pred_rate_max,
            args.pred_theta_pts, args.pred_rate_pts, args.horizon)

    if 3 not in skip:
        print(f'[panel 3] planning cost H={args.horizon} …')
        panel_planning_cost(
            next(ax_iter), fig, parts, act_mean, act_std, args.num_hist, device,
            args.pred_theta_max, args.pred_rate_max,
            args.pred_theta_pts, args.pred_rate_pts, args.horizon)

    if 4 not in skip:
        print('[panel 4] latent norm divergence …')
        panel_latent_norm(
            next(ax_iter), parts, act_mean, act_std, args.num_hist, device,
            args.theta0, args.norm_steps)

    if 5 not in skip:
        print('[panel 5] cosine alignment …')
        panel_cosine_alignment(
            next(ax_iter), parts, act_mean, act_std, args.num_hist, device,
            v_u, H_max=args.n_steps, n_starts=args.n_starts, alpha_max=args.alpha_max)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'[done] {out}')


if __name__ == '__main__':
    main()

