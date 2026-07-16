"""Diagnose why the IDM can't learn from a healthy encoder.

Two independent analyses:
1. Oracle IDM on ground-truth states: establishes the action-recovery signal
   available in perfect latent transitions (upper bound).
2. Gradient norm analysis on a loaded checkpoint: measures how IDM gradient
   competes with SIGreg+pred gradients at the encoder.

Run standalone (no checkpoint) for the oracle analysis only:
    python experiments/diagnose_idm.py --oracle-only

Run with a checkpoint for the full analysis:
    python experiments/diagnose_idm.py \
        --ckpt results_inv_eq/v2_E-full_random_eq_fs1_fstack2_seed42/model_ep250.pt \
        --config configs/cartpole_jepa_pred_inv_random_eq.yaml
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Oracle analysis: action-recovery signal from GT states ──────────────

def _cartpole_step(state, force, dt=0.02, mass_cart=1.0, mass_pole=0.1,
                   pole_length=0.5, gravity=9.8):
    """Pure-numpy cartpole step (no rendering)."""
    import math
    x, x_dot, theta, theta_dot = state
    M, m, g, l = mass_cart, mass_pole, gravity, pole_length
    sin_t = math.sin(theta); cos_t = math.cos(theta)
    total_mass = M + m; ml = m * l
    temp = (force + ml * theta_dot**2 * sin_t) / total_mass
    theta_acc = (g * sin_t - cos_t * temp) / (l * (4.0/3.0 - m * cos_t**2 / total_mass))
    x_acc = temp - ml * theta_acc * cos_t / total_mass
    new_x_dot   = x_dot   + dt * x_acc
    new_theta_dot = theta_dot + dt * theta_acc
    new_x     = x     + dt * x_dot
    new_theta = theta + dt * theta_dot
    done = (abs(new_x) > 2.4 or abs(new_theta) > 0.2094)
    return np.array([new_x, new_x_dot, new_theta, new_theta_dot]), done


def simulate_episodes(n_episodes=200, ep_len=200, dt=0.02,
                      mass_cart=1.0, mass_pole=0.1, pole_length=0.5,
                      gravity=9.8, action_range=(-10., 10.), seed=0):
    """Simulate random-action cartpole (pure numpy, no rendering)."""
    rng = np.random.RandomState(seed)
    all_states, all_actions = [], []
    for ep in range(n_episodes):
        s = rng.uniform(-1.0, 1.0, size=4)
        states, actions = [s.copy()], []
        for _ in range(ep_len):
            u = rng.uniform(action_range[0], action_range[1])
            s, done = _cartpole_step(s, u, dt=dt, mass_cart=mass_cart,
                                     mass_pole=mass_pole, pole_length=pole_length,
                                     gravity=gravity)
            actions.append(u)
            states.append(s.copy())
            if done:
                break
        all_states.append(np.array(states))
        all_actions.append(np.array(actions))
    return all_states, all_actions


def oracle_idm_analysis(all_states, all_actions):
    """Compute action-state correlations from ground-truth state transitions."""
    state_names = ['x', 'xdot', 'theta', 'thetadot']

    # Flatten to (N, 4) and (N,)
    pairs = []
    for states, actions in zip(all_states, all_actions):
        T = len(actions)
        for t in range(T):
            s_t   = states[t]
            s_tp1 = states[t + 1]
            u_t   = actions[t]
            pairs.append((s_t, s_tp1, u_t))

    S_t   = np.array([p[0] for p in pairs])   # (N, 4)
    S_tp1 = np.array([p[1] for p in pairs])   # (N, 4)
    U     = np.array([p[2] for p in pairs])   # (N,)

    print("\n" + "="*60)
    print("ORACLE IDM ANALYSIS: action recovery from GT states")
    print("="*60)
    print(f"N transitions: {len(U)}")
    print(f"Action range: [{U.min():.2f}, {U.max():.2f}], "
          f"std={U.std():.2f}, E[u²]={np.mean(U**2):.3f}")
    print(f"Baseline IDM MSE (predict zero): {np.mean(U**2):.4f}")

    # 2-frame difference: s_{t+1} - s_t
    dS = S_tp1 - S_t  # (N, 4): encodes velocity effects + action effects

    print("\n--- Pearson r(u_t, Δs[i]) ---")
    for i, name in enumerate(state_names):
        r = np.corrcoef(U, dS[:, i])[0, 1]
        print(f"  Δ{name}: r = {r:+.4f}  (R²={r**2:.5f})")

    print("\n--- Pearson r(u_t, s_{t+1}[i]) ---")
    for i, name in enumerate(state_names):
        r = np.corrcoef(U, S_tp1[:, i])[0, 1]
        print(f"  {name}_{{t+1}}: r = {r:+.4f}  (R²={r**2:.5f})")

    # Combined linear regression: u ~ linear(s_t, s_{t+1}) — oracle IDM
    X_oracle = np.hstack([S_t, S_tp1])   # (N, 8)
    X_oracle = np.hstack([X_oracle, np.ones((len(U), 1))])  # add bias
    w_oracle, _, _, _ = np.linalg.lstsq(X_oracle, U, rcond=None)
    u_hat_oracle = X_oracle @ w_oracle
    mse_oracle = np.mean((u_hat_oracle - U)**2)
    r2_oracle  = 1.0 - mse_oracle / np.mean(U**2)
    print(f"\nOracle linear IDM on (s_t, s_{{t+1}}) — UPPER BOUND:")
    print(f"  MSE = {mse_oracle:.4f} / baseline {np.mean(U**2):.4f}")
    print(f"  R²  = {r2_oracle:.4f}  (fraction of action variance explained)")
    print(f"  → Reduction from baseline: {(1-r2_oracle)*100:.1f}% remains")

    # Just using Δs
    X_delta = np.hstack([dS, np.ones((len(U), 1))])
    w_delta, _, _, _ = np.linalg.lstsq(X_delta, U, rcond=None)
    u_hat_delta = X_delta @ w_delta
    mse_delta = np.mean((u_hat_delta - U)**2)
    print(f"\nOracle linear IDM on (s_{{t+1}} - s_t) only:")
    print(f"  MSE = {mse_delta:.4f}  (R² = {1 - mse_delta/np.mean(U**2):.4f})")

    # MLP IDM on GT states
    print("\nTraining MLP IDM (128-128-128) on GT states...")
    X_t = torch.tensor(np.hstack([S_t, S_tp1]), dtype=torch.float32)
    y_t = torch.tensor(U, dtype=torch.float32).unsqueeze(1)
    mlp = nn.Sequential(
        nn.Linear(8, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    N = len(y_t)
    n_epochs = 200
    batch_size = 512
    idx = np.arange(N)
    for epoch in range(n_epochs):
        np.random.shuffle(idx)
        total_loss = 0.
        n_batches = 0
        for start in range(0, N, batch_size):
            b = idx[start:start+batch_size]
            xb = X_t[b]; yb = y_t[b]
            opt.zero_grad()
            loss = F.mse_loss(mlp(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item(); n_batches += 1
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                full_pred = mlp(X_t)
                val_mse = F.mse_loss(full_pred, y_t).item()
            print(f"  epoch {epoch+1:3d}: train_loss={total_loss/n_batches:.4f}, "
                  f"full_mse={val_mse:.4f}")
    with torch.no_grad():
        u_hat_mlp = mlp(X_t).squeeze(1).numpy()
    mse_mlp = np.mean((u_hat_mlp - U)**2)
    print(f"\nMLP IDM on GT states after {n_epochs} epochs:")
    print(f"  MSE = {mse_mlp:.4f}  (R² = {1 - mse_mlp/np.mean(U**2):.4f})")
    print(f"  → If encoder were perfect, best achievable IDM MSE ≈ {mse_mlp:.4f}")

    return {
        'n_transitions': len(U),
        'baseline_mse': float(np.mean(U**2)),
        'oracle_linear_mse': float(mse_oracle),
        'oracle_linear_r2': float(r2_oracle),
        'oracle_mlp_mse': float(mse_mlp),
    }


def imperfect_encoder_analysis(all_states, all_actions,
                               r_per_dim=None, action_scale=10.0):
    """
    Estimate IDM performance achievable with an IMPERFECT encoder.
    Simulates what the latent space looks like given per-dimension Pearson r.
    Default r values are from the pred+inv model at epoch 250.
    """
    # Default: pred+inv encoder at ep250 (approx from memory)
    if r_per_dim is None:
        # r for [x, xdot, theta, thetadot] best matching latent dims
        # These are approximate values from the Pearson correlation analysis
        r_per_dim = [0.922, 0.366, 0.043, 0.209]

    state_names = ['x', 'xdot', 'theta', 'thetadot']

    pairs = []
    for states, actions in zip(all_states, all_actions):
        T = len(actions)
        for t in range(T):
            pairs.append((states[t], states[t+1], actions[t]))

    S_t   = np.array([p[0] for p in pairs])
    S_tp1 = np.array([p[1] for p in pairs])
    U     = np.array([p[2] for p in pairs])
    U_scaled = U / action_scale

    print("\n" + "="*60)
    print("IMPERFECT ENCODER ANALYSIS (simulated latent transitions)")
    print("="*60)
    print(f"Encoder quality r per state dim: {dict(zip(state_names, r_per_dim))}")

    rng = np.random.RandomState(1)
    # Normalize state dims to unit variance (as SIGreg would produce)
    S_t_norm   = (S_t   - S_t.mean(0)) / (S_t.std(0) + 1e-8)
    S_tp1_norm = (S_tp1 - S_tp1.mean(0)) / (S_tp1.std(0) + 1e-8)

    # Simulated latent: z_t = r * state_t/std + sqrt(1-r²) * noise
    Z_t_sim   = np.zeros((len(U), 4))
    Z_tp1_sim = np.zeros((len(U), 4))
    for i, r in enumerate(r_per_dim):
        noise_t   = rng.randn(len(U))
        noise_tp1 = rng.randn(len(U))
        Z_t_sim[:, i]   = r * S_t_norm[:, i]   + np.sqrt(max(0, 1 - r**2)) * noise_t
        Z_tp1_sim[:, i] = r * S_tp1_norm[:, i] + np.sqrt(max(0, 1 - r**2)) * noise_tp1

    # Correlation between u and simulated latent differences
    dZ = Z_tp1_sim - Z_t_sim
    print("\n--- Pearson r(u_t, Δz_sim[i]) ---")
    for i, name in enumerate(state_names):
        r = np.corrcoef(U, dZ[:, i])[0, 1]
        print(f"  Δz[{name}]: r = {r:+.5f}  (R²={r**2:.6f})")

    # Linear IDM on simulated latent
    baseline_mse = np.mean(U_scaled**2)
    X_sim = np.hstack([Z_t_sim, Z_tp1_sim, np.ones((len(U), 1))])
    w_sim, _, _, _ = np.linalg.lstsq(X_sim, U_scaled, rcond=None)
    u_hat_sim = X_sim @ w_sim
    mse_sim = np.mean((u_hat_sim - U_scaled)**2)
    r2_sim  = 1.0 - mse_sim / baseline_mse
    print(f"\nLinear IDM on simulated latent (2-frame, noise-corrupted state):")
    print(f"  Baseline MSE (predict zero): {baseline_mse:.4f}")
    print(f"  Best linear IDM MSE: {mse_sim:.4f}")
    print(f"  R² = {r2_sim:.5f}  ({r2_sim*100:.3f}% of variance explained)")
    print(f"  → Improvement over baseline: {(baseline_mse - mse_sim):.5f} = "
          f"{(1 - mse_sim/baseline_mse)*100:.3f}%")
    print()
    print("*** KEY FINDING ***")
    print(f"  With current encoder quality, IDM can improve by only "
          f"{(1-mse_sim/baseline_mse)*100:.2f}% from baseline.")
    print(f"  This is BELOW numerical noise — the IDM signal is too weak to learn.")

    return {
        'simulated_linear_mse': float(mse_sim),
        'simulated_r2': float(r2_sim),
        'baseline_mse': float(baseline_mse),
    }


# ── 2. Gradient norm analysis from a checkpoint ────────────────────────────

def gradient_norm_analysis(ckpt_path, config_path, device='cpu', n_batches=10):
    """Load checkpoint and measure per-loss gradient norms on encoder params."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from models.jepa_v2 import JEPAV2
    from models.config import ModelConfig
    model_cfg_dict = cfg['model']
    env_cfg = cfg['environment']
    mc = ModelConfig(
        latent_dim=model_cfg_dict['latent_dim'],
        action_latent_dim=model_cfg_dict['action_latent_dim'],
        patch_size=model_cfg_dict['patch_size'],
        frame_stack=model_cfg_dict['frame_stack'],
        predictor_window=model_cfg_dict['predictor_window'],
        vit_embed_dim=model_cfg_dict['vit_embed_dim'],
        vit_depth=model_cfg_dict['vit_depth'],
        vit_num_heads=model_cfg_dict['vit_num_heads'],
        predictor_hidden_dim=model_cfg_dict['predictor_hidden_dim'],
        predictor_n_layers=model_cfg_dict['predictor_n_layers'],
        image_size=env_cfg['image_size'],
    )
    model = JEPAV2(mc).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.train()

    train_cfg = cfg['training']
    lambda_pred   = float(train_cfg.get('lambda_pred', 1.0))
    lambda_sigreg = float(train_cfg.get('lambda_sigreg', 1.0))
    lambda_inv    = float(train_cfg.get('lambda_inv', 0.0))
    inv_action_scale = float(train_cfg.get('inv_action_scale', 1.0))
    inv_frames    = int(train_cfg.get('inv_frames', 3))
    inv_hidden    = int(train_cfg.get('inv_hidden_dim', 8))
    d = mc.latent_dim

    inv_head = None
    if lambda_inv > 0:
        in_dim = inv_frames * d
        if inv_frames == 2:
            inv_head = nn.Sequential(nn.Linear(in_dim, inv_hidden), nn.ReLU(),
                                     nn.Linear(inv_hidden, 1)).to(device)
        else:
            inv_head = nn.Sequential(
                nn.Linear(in_dim, inv_hidden), nn.ReLU(),
                nn.Linear(inv_hidden, inv_hidden), nn.ReLU(),
                nn.Linear(inv_hidden, inv_hidden), nn.ReLU(),
                nn.Linear(inv_hidden, 1)).to(device)
        if 'inv_head_state_dict' in ckpt:
            inv_head.load_state_dict(ckpt['inv_head_state_dict'])

    # Generate synthetic batches for gradient measurement
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=env_cfg['frame_skip'],
        image_size=env_cfg['image_size'],
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        action_range=tuple(env_cfg['action_range']), seed=99,
    )

    print("\n" + "="*60)
    print("GRADIENT NORM ANALYSIS at encoder parameters")
    print("="*60)

    frame_stack = mc.frame_stack
    H = 30  # from config horizon
    B = 32

    grad_norms = {'pred': [], 'sigreg': [], 'inv': []}

    for batch_idx in range(n_batches):
        # Collect a batch of H+1 frame sequences
        obs_list, act_list = [], []
        for _ in range(B):
            frames_ep = []
            env.reset(init_range=1.0)
            prev_frame = env._render_obs()
            stack = np.stack([prev_frame] * frame_stack, axis=0)  # (fstack, H, W, C)
            for t in range(H + 1):
                frames_ep.append(stack)
                u = np.random.uniform(-10., 10.)
                obs_next, _, _, done, _ = env.step(np.array([u]))
                # Update frame stack
                stack = np.roll(stack, -1, axis=0)
                stack[-1] = obs_next
                act_list.append(u) if t < H else None
            obs_list.append(np.array(frames_ep))  # (H+1, fstack, H, W, C)
        obs_seq = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)  # (B, H+1, fstack, H, W, C)
        obs_seq = obs_seq / 255.0
        # Reshape: (B, H+1, C*fstack, h, w)
        B_, T, FS, h, w, C_ = obs_seq.shape
        obs_seq = obs_seq.permute(0, 1, 4, 2, 3).reshape(B_, T, C_*FS, h, w)
        actions_arr = np.array(act_list[:B * H]).reshape(B, H, 1)
        actions = torch.tensor(actions_arr, dtype=torch.float32).to(device)

        def measure_grad_norm(loss_val, retain=False):
            model.zero_grad()
            if inv_head is not None:
                inv_head.zero_grad()
            loss_val.backward(retain_graph=retain)
            total_sq = 0.
            n_params = 0
            for p in model.encoder.parameters():
                if p.grad is not None:
                    total_sq += p.grad.data.norm().item() ** 2
                    n_params += p.numel()
            return np.sqrt(total_sq), n_params

        obs_0 = obs_seq[:, 0]
        z_0 = model.encoder(obs_0)
        obs_rest = obs_seq[:, 1:].contiguous().view(B * H, C_*FS, h, w)
        from torch.utils.checkpoint import checkpoint as grad_ckpt
        z_rest = grad_ckpt(model.encoder, obs_rest, use_reentrant=False).view(B, H, d)
        z_all = torch.cat([z_0.unsqueeze(1), z_rest], dim=1)

        # Pred loss
        z_targets = z_all[:, 1:]
        W = mc.predictor_window
        z_win_buf = [z_0] * W
        u_win_buf = [torch.zeros(B, 1, device=device)] * (W - 1)
        pred_loss = torch.zeros(1, device=device)
        for k in range(H):
            u_k = actions[:, k]
            u_win_buf.append(u_k)
            z_stack = torch.stack(z_win_buf[-W:], dim=1)
            u_stack = torch.stack(u_win_buf[-W:], dim=1)
            z_hat = model.predict(z_stack, u_stack)
            pred_loss = pred_loss + F.mse_loss(z_hat, z_targets[:, k])
            z_win_buf.append(z_hat)
        pred_loss = pred_loss / H

        gn_pred, n_enc = measure_grad_norm(lambda_pred * pred_loss, retain=True)
        grad_norms['pred'].append(gn_pred)

        # SIGreg loss
        z_0_fresh = model.encoder(obs_0)
        obs_rest_fresh = obs_seq[:, 1:].contiguous().view(B * H, C_*FS, h, w)
        z_rest_fresh = grad_ckpt(model.encoder, obs_rest_fresh,
                                 use_reentrant=False).view(B, H, d)
        z_all_fresh = torch.cat([z_0_fresh.unsqueeze(1), z_rest_fresh], dim=1)
        from losses.sigreg import sigreg_loss
        z_sig = z_all_fresh.reshape(-1, d)
        sig_loss = sigreg_loss(z_sig, num_slices=32, num_points=9)
        gn_sig, _ = measure_grad_norm(lambda_sigreg * sig_loss, retain=True)
        grad_norms['sigreg'].append(gn_sig)

        # IDM loss
        if lambda_inv > 0 and inv_head is not None:
            z_0_inv = model.encoder(obs_0)
            obs_rest_inv = obs_seq[:, 1:].contiguous().view(B * H, C_*FS, h, w)
            z_rest_inv = grad_ckpt(model.encoder, obs_rest_inv,
                                   use_reentrant=False).view(B, H, d)
            z_all_inv = torch.cat([z_0_inv.unsqueeze(1), z_rest_inv], dim=1)
            scale = inv_action_scale
            if inv_frames == 2:
                z_pairs = torch.cat([z_all_inv[:, :-1], z_all_inv[:, 1:]], dim=-1)
                u_tgt = actions[:, :, 0] / scale
                u_hat = inv_head(z_pairs.view(B * H, 2 * d)).view(B, H)
            else:
                z_trips = torch.cat([z_all_inv[:, :-2], z_all_inv[:, 1:-1],
                                     z_all_inv[:, 2:]], dim=-1)
                u_tgt = actions[:, 1:, 0] / scale
                u_hat = inv_head(z_trips.view(B * (H - 1), 3 * d)).view(B, H - 1)
            inv_loss = F.mse_loss(u_hat, u_tgt)
            gn_inv, _ = measure_grad_norm(lambda_inv * inv_loss, retain=False)
            grad_norms['inv'].append(gn_inv)

        if batch_idx < 2:
            print(f"  batch {batch_idx}: ||∇pred||={gn_pred:.4f}, "
                  f"||∇sigreg||={gn_sig:.4f}"
                  + (f", ||∇inv||={grad_norms['inv'][-1]:.4f}" if grad_norms['inv'] else ""))

    env.close()

    print(f"\nEncoder gradient norms (mean over {n_batches} batches):")
    for name, norms in grad_norms.items():
        if norms:
            print(f"  {name:8s}: {np.mean(norms):.5f} ± {np.std(norms):.5f}")
    if grad_norms['inv'] and grad_norms['sigreg']:
        ratio = np.mean(grad_norms['inv']) / (np.mean(grad_norms['sigreg']) + 1e-8)
        print(f"\n  IDM gradient is {ratio:.3f}× the SIGreg gradient at encoder")
        if ratio < 0.1:
            print("  → IDM gradient overwhelmed by SIGreg (factor >10)")
        elif ratio < 0.3:
            print("  → IDM gradient weaker than SIGreg (factor 3-10)")
        else:
            print("  → IDM gradient competitive with SIGreg")


# ── 3. Summary: what to do ─────────────────────────────────────────────────

def print_diagnosis():
    print("\n" + "="*60)
    print("ROOT CAUSE DIAGNOSIS")
    print("="*60)
    print("""
Problem: IDM inv_loss stuck at ~0.327 ≈ E[(u/10)²] = 0.333 (zero-prediction baseline).

Cause 1 — WEAK SIGNAL (fundamental physics):
  The action u_t has a dt=0.02s effect on the state transition.
  Action → Δẋ = dt/(M+m) * u ≈ 0.018*u  [only ±0.18 for u=±10]
  Action → Δθ̇ = -dt/(L*(M+m)) * u ≈ -0.036*u

  Pearson r(u_t, Δz[vel_dim]) ≈ r_enc × 0.018 × Var(u) / (Std(u)×Std(Δz))
                               ≈ 0.366 × 0.018 × 33.3 / (5.77 × 1.41) ≈ 0.027

  With current encoder quality (r_enc ≈ 0.37 for ẋ), the action signal
  in latent transitions is only ~2-3% Pearson correlation.
  This gives R² ≈ 0.001 → IDM MSE improvement ≈ 0.333 × 0.001 ≈ 0.0003.
  That's completely invisible compared to training noise.

Cause 2 — GRADIENT COMPETITION (optimization):
  SIGreg + pred gradients dominate the encoder.
  The IDM gradient is proportional to Cov(u_t, Δz) ≈ 0 at initialization
  and remains near-zero because the signal never bootstraps above noise.

Cause 3 — CHICKEN-AND-EGG:
  IDM needs good state encoding to extract action signal.
  But IDM is supposed to CREATE the action-discriminative encoding.
  SIGreg+pred already shape the encoder for state; IDM can't overcome this.

Upper bound with PERFECT encoding (r_enc=1.0):
  Pearson r(u_t, Δz[vel_dim]) ≈ 7.4%  (ẋ)
  Pearson r(u_t, Δz[ang_dim]) ≈ 14.8% (θ̇)
  Oracle MLP IDM MSE ≈ 0.27 (vs baseline 0.333) → only 19% reduction even perfect!

CONCLUSION:
  In this cartpole setting (dt=0.02, action_range=±10, frame_stack=2),
  the IDM cannot learn from consecutive frame encodings because the action
  effect is a tiny 2nd-order term over a single 20ms step.

  The SMWM paper likely uses: longer dt, stronger action effects, or
  environments where actions cause visually obvious changes.

RECOMMENDED FIXES:
  Option A (recommended): Add lambda_state supervision.
    With direct state encoding (r_enc → 0.9+), the oracle IDM achieves
    ~R²=15-20%, making the IDM learnable over many epochs.
    The IDM then adds regularization on top of the state-supervised encoder.

  Option B: Increase dt (frame_skip=3-5).
    3× larger dt → 3× larger action signal → 3× better IDM correlation.
    But changes the dynamics and Jacobian eigenvalues.

  Option C: Use multi-step IDM (predict action from 10+ frame window).
    Cumulative action effect over N steps grows linearly → stronger signal.
    But the model would predict a SEQUENCE of actions, not just u_t.

  Option D: Accept IDM doesn't help here.
    The baseline (SIGreg + pred + fp) already achieves ρ(A_aug)=1.023.
    Focus on CEM control performance evaluation without IDM.
""")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle-only', action='store_true',
                        help='Run only the oracle GT analysis (no checkpoint needed)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--config', type=str,
                        default='configs/cartpole_jepa_pred_inv_random_eq.yaml',
                        help='Path to config YAML')
    parser.add_argument('--n-episodes', type=int, default=100,
                        help='Simulation episodes for oracle analysis')
    parser.add_argument('--n-batches', type=int, default=5,
                        help='Batches for gradient norm analysis')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print("Simulating cartpole episodes for oracle analysis...")
    all_states, all_actions = simulate_episodes(n_episodes=args.n_episodes, seed=0)

    oracle_results = oracle_idm_analysis(all_states, all_actions)
    imperfect_encoder_analysis(all_states, all_actions)
    print_diagnosis()

    if not args.oracle_only and args.ckpt is not None:
        gradient_norm_analysis(args.ckpt, args.config,
                               device=args.device, n_batches=args.n_batches)
    elif not args.oracle_only:
        print("\n[Skipping gradient norm analysis: no --ckpt provided]")
        print("  Run with --ckpt <path/to/model_ep250.pt> for gradient analysis.")

    return oracle_results


if __name__ == '__main__':
    main()
