#!/usr/bin/env python
"""Local stability probe for CartPole DINO-WM checkpoint.

Panel 1  – Vector-field comparison
           GT vs learned one-step transitions on a θ–θ̇ grid, decoded back to
           physical space via a ridge probe fitted on encoded states.

Panel 2  – Empirical Region of Attraction (ROA)
           Grid of (θ, θ̇) initial conditions; each run closed-loop on the GT
           environment using the LQR gain from the learned latent linearisation.
           Background shows learned-LQR success; dashed contour = GT-LQR boundary.

Panel 3  – Lyapunov decrease condition
           Quadratic Lyapunov V(x) = xᵀ P x, P from GT physical DARE (Q=I₄,R=1).
           Apply one step of learned-LQR on GT env; plot ΔV = V(x') − V(x).

Usage
-----
  python experiments/probe_local_stability_dinowm.py \\
      --dino-wm-dir ~/dino_wm \\
      --ckpt /home/dz2478/dino_wm/outputs/2026-09-08/17-46-00/checkpoints/model_179.pth \\
      --data /mnt/t7shield/dinowm_cartpole \\
      --title "DINO-WM" \\
      --out results/stability_dinowm.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F

# ── shared dino-wm constants ─────────────────────────────────────────────────
FRAMESKIP  = 5
D_VIS      = 384
D_PROP_OUT = 10
D_ACT_OUT  = 10
D_TOTAL    = D_VIS + D_PROP_OUT + D_ACT_OUT
ACT_IN_DIM = 1
PROP_IN_DIM = 4
IMG_SIZE   = 112
_NORM_MEAN = torch.tensor([0.5, 0.5, 0.5])
_NORM_STD  = torch.tensor([0.5, 0.5, 0.5])
ACTION_LB, ACTION_UB = -10., 10.


# ── dino-wm utilities ────────────────────────────────────────────────────────

def setup_dinowm(dino_wm_dir):
    p = str(Path(dino_wm_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def make_env(seed=0):
    from envs.cartpole_visual import ContinuousCartpoleVisual
    return ContinuousCartpoleVisual(frame_skip=FRAMESKIP, image_size=IMG_SIZE, seed=seed)


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
    print(f'[model] predictor params={sum(p.numel() for p in parts["predictor"].parameters())/1e6:.1f}M')
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
    """Returns (1, N, D_TOTAL) tensor."""
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


def build_goal_ctx(parts, goal_state, num_hist, device, act_mean, act_std):
    env = make_env(seed=0)
    gobs, _, _ = env.reset_to_state(goal_state.copy())
    env.close()
    zero_norm = np.zeros(ACT_IN_DIM, dtype=np.float32)
    gframe = encode_frame(parts, gobs, goal_state, zero_norm, device)
    z_goal_ctx  = gframe.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()
    z_goal_mean = z_goal_ctx[0, -1].mean(0).cpu().numpy()
    return z_goal_ctx, z_goal_mean


# ── ridge probe ───────────────────────────────────────────────────────────────

class RidgeProbe:
    """Linear probe: state ≈ W @ z_mean + b, fitted via ridge regression."""

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


def build_probe(parts, act_mean, act_std, num_hist, device, n_samples=300, seed=7):
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


# ── GT physical linearisation ─────────────────────────────────────────────────

def gt_physical_linearization(eps=1e-3, action_scale=10.):
    env = make_env(seed=42)
    x0  = np.zeros(4, np.float32)
    A   = np.zeros((4, 4)); B = np.zeros((4, 1))
    for j in range(4):
        d = np.zeros(4, np.float32); d[j] = eps
        env.reset_to_state((x0+d).astype(np.float64)); _, xp, _, _, _ = env.step(0.)
        env.reset_to_state((x0-d).astype(np.float64)); _, xm, _, _, _ = env.step(0.)
        A[:, j] = (xp - xm) / (2.*eps)
    ua = float(eps * action_scale)
    env.reset_to_state(x0.astype(np.float64)); _, xp, _, _, _ = env.step( ua)
    env.reset_to_state(x0.astype(np.float64)); _, xm, _, _, _ = env.step(-ua)
    B[:, 0] = (xp - xm) / (2.*ua)
    env.close()
    return A, B


def solve_dare(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


# ── LQR episode ───────────────────────────────────────────────────────────────

def lqr_episode(parts, K_z, z_eq, initial_state, n_steps,
                threshold, hold_steps, act_mean, act_std, num_hist, device):
    env   = make_env(seed=0)
    obs, state, _ = env.reset_to_state(np.asarray(initial_state, np.float64))
    zero_norm = np.zeros(ACT_IN_DIM, np.float32)
    frame = encode_frame(parts, obs, state, zero_norm, device)
    z_ctx = frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()

    states = [state.copy()]; terminated = False
    am, as_ = float(act_mean[0].cpu()), float(act_std[0].cpu())
    for _ in range(n_steps):
        z_mean = z_ctx[0, -1].mean(0).cpu().numpy()
        u_norm = float(-(K_z @ (z_mean - z_eq)).item())
        u_real = float(np.clip(u_norm * as_ + am, ACTION_LB, ACTION_UB))
        obs, state, _, done, _ = env.step(u_real)
        a_norm_np = np.array([(u_real - am) / as_], dtype=np.float32)
        new_f = encode_frame(parts, obs, state, a_norm_np, device)
        z_ctx = torch.cat([z_ctx[:, 1:], new_f.unsqueeze(1)], dim=1)
        states.append(state.copy())
        if done: terminated = True; break
    env.close()

    states = np.array(states)
    errors = np.linalg.norm(states, axis=1)
    tail   = errors[-hold_steps:]
    held   = bool(not terminated and len(tail) == hold_steps
                  and np.all(tail < threshold))
    return held, states


# ── Panel 1: vector-field comparison ─────────────────────────────────────────

def panel_vector_field(ax, parts, probe, act_mean, act_std, num_hist, device,
                       theta_max, rate_max, n_theta, n_rate):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)
    theta_mesh, rate_mesh = np.meshgrid(theta_deg, rate_deg)
    gt_u = np.zeros_like(theta_mesh); gt_v = np.zeros_like(theta_mesh)
    lr_u = np.zeros_like(theta_mesh); lr_v = np.zeros_like(theta_mesh)
    err_norms = []

    zero_norm = np.zeros(ACT_IN_DIM, np.float32)
    env = make_env(seed=11)
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
        err_norms.append(np.hypot(gt_u[idx] - lr_u[idx], gt_v[idx] - lr_v[idx]))
    env.close()

    gt_mag = np.hypot(gt_u, gt_v)
    norm_err = float(np.mean(err_norms) / max(np.mean(gt_mag), 1e-8))
    frac = 0.45
    dx = 2*theta_max / max(n_theta-1, 1)
    dy = 2*rate_max  / max(n_rate-1,  1)

    def _q(u, v, **kw):
        n = np.sqrt(u**2 + v**2 + 1e-14)
        ax.quiver(theta_mesh, rate_mesh, u/n*frac*dx, v/n*frac*dy,
                  pivot='mid', scale=1, scale_units='xy', angles='xy',
                  width=.0025, headwidth=4, headlength=4, **kw)

    _q(gt_u, gt_v, color='#2ecc40', alpha=.75)
    _q(lr_u, lr_v, color='darkorange', alpha=.75)
    ax.scatter([0.], [0.], marker='*', s=200, color='red', zorder=5)
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=11)
    ax.set_ylabel(r'$\dot\theta$ [deg/s]', fontsize=11)
    ax.legend(handles=[
        Line2D([0],[0], color='#2ecc40',   lw=3, label='GT'),
        Line2D([0],[0], color='darkorange', lw=3, label='Learned'),
    ], fontsize=9, loc='upper right')
    ax.set_title(f'Local vector-field comparison\nnorm. error = {norm_err:.3f}', fontsize=11)
    ax.grid(alpha=.2)


# ── Panel 2: empirical ROA ────────────────────────────────────────────────────

def panel_roa(ax, parts, K_z, z_eq, K_phys,
              act_mean, act_std, num_hist, device,
              theta_max, rate_max, n_theta, n_rate,
              n_steps, threshold, hold_steps):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)
    theta_mesh, rate_mesh = np.meshgrid(theta_deg, rate_deg)

    success_map = np.zeros(theta_mesh.shape, dtype=bool)
    print(f'  ROA grid {n_theta}×{n_rate} = {n_theta*n_rate} points …', flush=True)
    am, as_ = float(act_mean[0].cpu()), float(act_std[0].cpu())
    for idx in np.ndindex(theta_mesh.shape):
        state = np.array([0., 0., np.deg2rad(theta_mesh[idx]),
                          np.deg2rad(rate_mesh[idx])], dtype=np.float32)
        held, _ = lqr_episode(parts, K_z, z_eq, state, n_steps,
                              threshold, hold_steps, act_mean, act_std, num_hist, device)
        success_map[idx] = held

    ax.pcolormesh(theta_deg, rate_deg, success_map.astype(float),
                  cmap='RdYlGn', vmin=0, vmax=1, shading='auto', alpha=.85)

    # GT-LQR boundary
    gt_ok = np.zeros_like(success_map, dtype=float)
    env = make_env(seed=0)
    for idx in np.ndindex(theta_mesh.shape):
        state = np.array([0., 0., np.deg2rad(theta_mesh[idx]),
                          np.deg2rad(rate_mesh[idx])], dtype=np.float32)
        obs, s, _ = env.reset_to_state(state.astype(np.float64))
        ss = [s.copy()]; term = False
        for _ in range(n_steps):
            u = float(np.clip(-(K_phys @ s), ACTION_LB, ACTION_UB))
            _, s, _, done, _ = env.step(u)
            ss.append(s.copy())
            if done: term = True; break
        errs = np.linalg.norm(np.array(ss), axis=1)
        tail = errs[-hold_steps:]
        gt_ok[idx] = float(not term and len(tail) == hold_steps
                           and np.all(tail < threshold))
    env.close()
    try:
        ax.contour(theta_deg, rate_deg, gt_ok, levels=[.5],
                   colors='k', linewidths=1.5, linestyles='--')
    except Exception:
        pass

    ax.scatter([0.], [0.], marker='*', s=200, color='blue', zorder=5)
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=11)
    ax.set_ylabel(r'$\dot\theta$ [deg/s]', fontsize=11)
    ax.legend(handles=[
        mpatches.Patch(color='green', alpha=.85, label='Learned LQR stabilises'),
        mpatches.Patch(color='red',   alpha=.85, label='Learned LQR fails'),
        Line2D([0],[0], color='k', lw=1.5, ls='--', label='GT LQR boundary'),
    ], fontsize=8, loc='upper right')
    sr = float(np.mean(success_map))
    ax.set_title(f'Empirical ROA  (learned LQR on GT env)\nsuccess rate = {sr:.1%}', fontsize=11)
    ax.grid(alpha=.15)


# ── Panel 3: Lyapunov decrease ────────────────────────────────────────────────

def panel_lyapunov(ax, parts, K_z, z_eq, P_phys,
                   act_mean, act_std, num_hist, device,
                   theta_max, rate_max, n_theta, n_rate):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)
    theta_mesh, rate_mesh = np.meshgrid(theta_deg, rate_deg)
    dV_map = np.zeros(theta_mesh.shape)

    zero_norm = np.zeros(ACT_IN_DIM, np.float32)
    am, as_ = float(act_mean[0].cpu()), float(act_std[0].cpu())
    for idx in np.ndindex(theta_mesh.shape):
        state = np.array([0., 0., np.deg2rad(theta_mesh[idx]),
                          np.deg2rad(rate_mesh[idx])], dtype=np.float32)
        V0 = float(state @ P_phys @ state)
        env = make_env(seed=0)
        obs, s0, _ = env.reset_to_state(state.astype(np.float64))
        frame = encode_frame(parts, obs, state, zero_norm, device)
        z_mean = frame[0].mean(0).cpu().numpy()
        u_norm = float(-(K_z @ (z_mean - z_eq)).item())
        u_real = float(np.clip(u_norm * as_ + am, ACTION_LB, ACTION_UB))
        _, s1, _, _, _ = env.step(u_real)
        env.close()
        dV_map[idx] = float(s1.astype(np.float64) @ P_phys @ s1.astype(np.float64)) - V0

    vabs = np.percentile(np.abs(dV_map), 95)
    im = ax.pcolormesh(theta_deg, rate_deg, dV_map, cmap='RdBu_r',
                       shading='auto', vmin=-vabs, vmax=vabs)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=r'$\Delta V = V(x^\prime) - V(x)$')
    ax.contour(theta_deg, rate_deg, dV_map, levels=[0.], colors='k', linewidths=1.2)
    ax.scatter([0.], [0.], marker='*', s=200, color='lime', zorder=5,
               edgecolors='darkgreen', lw=0.8)
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=11)
    ax.set_ylabel(r'$\dot\theta$ [deg/s]', fontsize=11)
    frac_dec = float(np.mean(dV_map < 0))
    ax.set_title(r'Lyapunov decrease  $\Delta V < 0$' + '\n'
                 f'(GT DARE matrix, learned LQR policy)  {frac_dec:.1%} of grid',
                 fontsize=11)
    ax.grid(alpha=.15)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dino-wm-dir', default=str(Path.home()/'dino_wm'))
    p.add_argument('--ckpt',  required=True)
    p.add_argument('--data',  default='/mnt/t7shield/dinowm_cartpole')
    p.add_argument('--num-hist', type=int, default=3)
    p.add_argument('--title', required=True)
    p.add_argument('--out',   required=True)

    p.add_argument('--vf-theta-max',  type=float, default=8.)
    p.add_argument('--vf-rate-max',   type=float, default=30.)
    p.add_argument('--vf-theta-pts',  type=int,   default=11)
    p.add_argument('--vf-rate-pts',   type=int,   default=9)

    p.add_argument('--roa-theta-max', type=float, default=25.)
    p.add_argument('--roa-rate-max',  type=float, default=80.)
    p.add_argument('--roa-theta-pts', type=int,   default=17)
    p.add_argument('--roa-rate-pts',  type=int,   default=13)

    p.add_argument('--lyap-theta-max', type=float, default=15.)
    p.add_argument('--lyap-rate-max',  type=float, default=50.)
    p.add_argument('--lyap-theta-pts', type=int,   default=17)
    p.add_argument('--lyap-rate-pts',  type=int,   default=13)

    p.add_argument('--n-steps',     type=int,   default=60)
    p.add_argument('--threshold',   type=float, default=0.3)
    p.add_argument('--hold-steps',  type=int,   default=10)
    p.add_argument('--q-scale',     type=float, default=1.0)
    p.add_argument('--r-scale',     type=float, default=1.0)
    p.add_argument('--probe-samples', type=int, default=300)
    p.add_argument('--eps-z',       type=float, default=1e-3)
    p.add_argument('--eps-u',       type=float, default=1e-2)
    p.add_argument('--chunk-size',  type=int,   default=64)
    p.add_argument('--device',      default='cuda')
    p.add_argument('--skip-panels', type=int, nargs='+', default=[],
                   metavar='N', help='1=vector field  2=ROA  3=Lyapunov')
    args = p.parse_args()

    device = torch.device(args.device)

    print('[load] model …')
    parts = load_model(args.ckpt, args.dino_wm_dir, device)
    act_mean, act_std = load_action_stats(args.data, device)

    print('[probe] fitting ridge probe …')
    probe = build_probe(parts, act_mean, act_std, args.num_hist,
                        device, args.probe_samples)

    print('[jacobians] learned latent linearisation at equilibrium …')
    # Import jacobians_fd from lqr script
    from lqr_dinowm_cartpole import jacobians_fd
    goal_state  = np.zeros(4, np.float32)
    z_goal_ctx, z_eq = build_goal_ctx(parts, goal_state, args.num_hist,
                                       device, act_mean, act_std)
    A_z, B_z = jacobians_fd(parts, z_goal_ctx, args.num_hist,
                             eps_z=args.eps_z, eps_u=args.eps_u,
                             chunk_size=args.chunk_size)
    d = A_z.shape[0]
    Q_z = args.q_scale * np.eye(d)
    R_z = np.array([[args.r_scale]])
    print('[DARE] learned latent …')
    K_z, _ = solve_dare(A_z, B_z, Q_z, R_z)
    rho_cl = np.max(np.abs(np.linalg.eigvals(A_z - B_z @ K_z)))
    print(f'  ρ(A_z − B_z K_z) = {rho_cl:.4f}')

    print('[GT] physical linearisation …')
    A_p, B_p = gt_physical_linearization()
    K_phys, P_phys = solve_dare(A_p, B_p, args.q_scale*np.eye(4),
                                np.array([[args.r_scale]]))
    rho_p = np.max(np.abs(np.linalg.eigvals(A_p - B_p @ K_phys)))
    print(f'  ρ(A_p − B_p K_p) = {rho_p:.4f}')

    skip   = set(args.skip_panels)
    active = [i for i in (1, 2, 3) if i not in skip]
    fig, axes_all = plt.subplots(1, len(active), figsize=(7*len(active), 6))
    if len(active) == 1:
        axes_all = [axes_all]
    ax_iter = iter(axes_all)
    fig.suptitle(args.title, fontsize=16, fontweight='bold', y=1.02)

    if 1 not in skip:
        print('[panel 1] vector field …')
        panel_vector_field(next(ax_iter), parts, probe, act_mean, act_std,
                           args.num_hist, device,
                           args.vf_theta_max, args.vf_rate_max,
                           args.vf_theta_pts, args.vf_rate_pts)

    if 2 not in skip:
        print('[panel 2] empirical ROA …')
        panel_roa(next(ax_iter), parts, K_z, z_eq, K_phys,
                  act_mean, act_std, args.num_hist, device,
                  args.roa_theta_max, args.roa_rate_max,
                  args.roa_theta_pts, args.roa_rate_pts,
                  args.n_steps, args.threshold, args.hold_steps)

    if 3 not in skip:
        print('[panel 3] Lyapunov condition …')
        panel_lyapunov(next(ax_iter), parts, K_z, z_eq, P_phys,
                       act_mean, act_std, args.num_hist, device,
                       args.lyap_theta_max, args.lyap_rate_max,
                       args.lyap_theta_pts, args.lyap_rate_pts)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'[done] {out}')


if __name__ == '__main__':
    main()

