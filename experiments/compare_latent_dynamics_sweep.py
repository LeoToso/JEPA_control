"""Sweep compare_latent_dynamics metrics across all checkpoints in a directory.

For each checkpoint computes:
  - rho(A_aug)           spectral radius of the augmented Jacobian at z*
  - ||B_eff||            effective action influence norm
  - max |r(state_i, z)| Pearson correlation of each physical state var with latent

Outputs a summary CSV and a 2-panel plot (rho + Pearson r vs epoch).

Usage:
  python experiments/compare_latent_dynamics_sweep.py \\
      --ckpt-dir results/jepa_sf_w3_fs5/checkpoints \\
      --config   configs/cartpole_jepa_sf_w3_fs5.yaml \\
      --out      results/jepa_sf_w3_fs5/rho_sweep.png \\
      --csv      results/jepa_sf_w3_fs5/rho_sweep.csv \\
      --device   cuda
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml


def _epoch_from_name(p: Path) -> int:
    for part in p.stem.split('_'):
        if part.startswith('epoch'):
            return int(part[5:])
    return -1


def _load_model(ckpt_path: str, cfg: dict, device):
    from models.jepa import make_jepa
    env_cfg   = cfg['environment']
    model_cfg = dict(cfg['model'])

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw.get('model_state', raw.get('model_state_dict', raw))
    if isinstance(raw, dict) and 'config' in raw:
        for k in ('latent_dim', 'action_latent_dim', 'action_encoder',
                  'encoder_type', 'patch_size', 'frame_stack',
                  'vit_embed_dim', 'vit_depth', 'vit_num_heads',
                  'predictor_type', 'predictor_hidden_dim', 'predictor_n_layers',
                  'predictor_window', 'predictor_embed_dim', 'predictor_depth',
                  'predictor_num_heads', 'predictor_mlp_ratio'):
            if k in raw['config']:
                model_cfg[k] = raw['config'][k]

    model = make_jepa(
        variant='E-full',
        latent_dim=int(model_cfg['latent_dim']),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 1)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        encoder_type=model_cfg.get('encoder_type', 'vit'),
        image_size=int(env_cfg['image_size']),
        patch_size=int(model_cfg.get('patch_size', 8)),
        frame_stack=int(model_cfg.get('frame_stack', 1)),
        vit_embed_dim=int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg.get('vit_depth', 4)),
        vit_num_heads=int(model_cfg.get('vit_num_heads', 4)),
        predictor_type=model_cfg.get('predictor_type', 'mlp'),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 64)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 3)),
        predictor_window=int(model_cfg.get('predictor_window', 1)),
        predictor_embed_dim=int(model_cfg.get('predictor_embed_dim', 128)),
        predictor_depth=int(model_cfg.get('predictor_depth', 4)),
        predictor_num_heads=int(model_cfg.get('predictor_num_heads', 4)),
        predictor_mlp_ratio=float(model_cfg.get('predictor_mlp_ratio', 4.0)),
    )
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    frame_stack = int(model_cfg.get('frame_stack', 1))
    return model, frame_stack


def _collect_rollouts(env, model, frame_stack, device,
                      n_rollouts=40, rollout_len=50, theta_range=0.5):
    """Return (zs, states) arrays for Pearson r computation."""
    zs, states = [], []
    rng = np.random.RandomState(0)
    for _ in range(n_rollouts):
        theta0 = rng.uniform(-theta_range, theta_range)
        x0 = np.array([rng.uniform(-0.3, 0.3), 0.0, theta0, 0.0], dtype=np.float32)
        obs, state, _ = env.reset_to_state(x0)
        prev_obs = obs.copy()
        for _ in range(rollout_len):
            curr = torch.from_numpy(obs).float().permute(2, 0, 1)[None].to(device) / 255.0
            if frame_stack > 1:
                prev = torch.from_numpy(prev_obs).float().permute(2, 0, 1)[None].to(device) / 255.0
                inp = torch.cat([prev, curr], dim=1)
            else:
                inp = curr
            with torch.no_grad():
                z = model.encoder(inp).cpu().numpy()[0]
            zs.append(z)
            states.append(state.copy())
            action = env.sample_action()
            obs_next, state_next, _, done, _ = env.step(action)
            prev_obs = obs.copy()
            obs, state = obs_next, state_next
            if done:
                break
    return np.array(zs), np.array(states)


def _pearson_max(zs, states):
    """Return max |r| per state variable across all latent dims."""
    n_state = states.shape[1]
    maxr = np.zeros(n_state)
    for j in range(n_state):
        for i in range(zs.shape[1]):
            xi = zs[:, i] - zs[:, i].mean()
            sj = states[:, j] - states[:, j].mean()
            denom = np.linalg.norm(xi) * np.linalg.norm(sj) + 1e-12
            r = abs(float(np.dot(xi, sj) / denom))
            if r > maxr[j]:
                maxr[j] = r
    return maxr  # (4,)  [x, xdot, theta, thetadot]


def _run_one(ckpt_path, cfg, device, n_rollouts, rollout_len):
    model, frame_stack = _load_model(str(ckpt_path), cfg, device)
    env_cfg = cfg['environment']

    from envs.cartpole_visual import ContinuousCartpoleVisual
    frame_skip = int(env_cfg.get('frame_skip', 1))
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip,
        image_size=int(env_cfg['image_size']),
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'], seed=0,
    )

    # z_star from equilibrium observation
    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    obs_eq_t = torch.from_numpy(obs_eq).float().permute(2, 0, 1)[None].to(device) / 255.0
    if frame_stack > 1:
        obs_eq_t = torch.cat([obs_eq_t, obs_eq_t], dim=1)
    with torch.no_grad():
        z_star = model.encoder(obs_eq_t).cpu().numpy()[0]

    # Jacobian at z*
    rho = b_norm = None
    try:
        from control.jacobian import compute_augmented_jacobian_np
        A_aug, B_aug = compute_augmented_jacobian_np(model, z_star, device)
        rho = float(np.max(np.abs(np.linalg.eigvals(A_aug))))
        if hasattr(model.action_encoder, 'W'):
            W_enc = model.action_encoder.W.weight.detach().cpu().numpy()
            B_eff = B_aug @ W_enc
        else:
            B_eff = B_aug
        b_norm = float(np.linalg.norm(B_eff[:len(z_star)]))
    except Exception as exc:
        print(f'    Jacobian failed: {exc}')

    # Pearson r from rollouts
    zs, states = _collect_rollouts(env, model, frame_stack, device,
                                   n_rollouts=n_rollouts, rollout_len=rollout_len)
    maxr = _pearson_max(zs, states) if len(zs) > 10 else np.zeros(4)

    env.close()
    return {
        'rho':       rho,
        'b_norm':    b_norm,
        'r_x':       float(maxr[0]),
        'r_xdot':    float(maxr[1]),
        'r_theta':   float(maxr[2]),
        'r_thetadot': float(maxr[3]),
        'n_rollout_steps': len(zs),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir',    required=True)
    p.add_argument('--config',      required=True)
    p.add_argument('--out',         default=None)
    p.add_argument('--csv',         default=None)
    p.add_argument('--n-rollouts',  type=int, default=40)
    p.add_argument('--rollout-len', type=int, default=50)
    p.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpts = sorted(Path(args.ckpt_dir).glob('checkpoint_epoch*.pt'), key=_epoch_from_name)
    if not ckpts:
        print(f'No checkpoints found in {args.ckpt_dir}')
        return
    print(f'Found {len(ckpts)} checkpoints  ({_epoch_from_name(ckpts[0])}–{_epoch_from_name(ckpts[-1])})')

    # GT rho
    frame_skip = int(cfg['environment'].get('frame_skip', 1))
    try:
        from ground_truth.cartpole_gt import CartpoleGroundTruth
        _gt = CartpoleGroundTruth()
        A_fs = np.linalg.matrix_power(_gt.A_star, frame_skip)
        rho_gt = float(np.max(np.abs(np.linalg.eigvals(A_fs))))
        print(f'[GT] rho(A_star^{frame_skip})={rho_gt:.4f}')
    except Exception:
        rho_gt = None

    print(f'\n{"epoch":>6}  {"rho(A_aug)":>12}  {"||B||":>8}  '
          f'{"r(x)":>6}  {"r(ẋ)":>6}  {"r(θ)":>6}  {"r(θ̇)":>7}')
    print('-' * 65)

    rows = []
    for ckpt in ckpts:
        epoch = _epoch_from_name(ckpt)
        print(f'  ep{epoch:04d} ...', end='', flush=True)
        try:
            m = _run_one(ckpt, cfg, device, args.n_rollouts, args.rollout_len)
        except Exception as exc:
            print(f' ERROR: {exc}')
            continue
        rho_str = f'{m["rho"]:.4f}' if m['rho'] is not None else '  N/A '
        rho_color = '✓' if m['rho'] is not None and m['rho'] > 1.0 else '✗'
        print(f'\r{epoch:>6}  {rho_str:>12} {rho_color}  {m["b_norm"]:>8.4f}  '
              f'{m["r_x"]:>6.3f}  {m["r_xdot"]:>6.3f}  {m["r_theta"]:>6.3f}  '
              f'{m["r_thetadot"]:>7.3f}')
        rows.append({'epoch': epoch, **m})

    if args.csv:
        out_csv = Path(args.csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nCSV → {out_csv}')

    if args.out and rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = [r['epoch'] for r in rows]
        rhos   = [r['rho']   for r in rows]
        r_x    = [r['r_x']       for r in rows]
        r_xd   = [r['r_xdot']    for r in rows]
        r_th   = [r['r_theta']   for r in rows]
        r_thd  = [r['r_thetadot'] for r in rows]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 1: rho(A_aug) vs epoch
        ax = axes[0]
        ax.plot(epochs, rhos, 'b-o', markersize=4, label='rho(A_aug) learned')
        ax.axhline(1.0, color='red', lw=1.2, linestyle='--', label='stability boundary')
        if rho_gt is not None:
            ax.axhline(rho_gt, color='green', lw=1.2, linestyle=':', label=f'GT rho^{frame_skip}={rho_gt:.3f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Spectral radius ρ(A_aug)')
        ax.set_title('Learned dynamics instability over training')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 2: Pearson r vs epoch
        ax = axes[1]
        ax.plot(epochs, r_x,   label='r(x)',   color='C0')
        ax.plot(epochs, r_xd,  label='r(ẋ)',   color='C1', linestyle='--')
        ax.plot(epochs, r_th,  label='r(θ)',   color='C2', linewidth=2)
        ax.plot(epochs, r_thd, label='r(θ̇)',  color='C3', linestyle='--')
        ax.axhline(0.5, color='gray', lw=0.8, linestyle=':', label='r=0.5 threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('max |Pearson r|')
        ax.set_title('State encoding quality over training')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.suptitle(f'Latent Dynamics Sweep — {Path(args.ckpt_dir).parent.name}', fontsize=11)
        fig.tight_layout()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches='tight')
        print(f'Plot → {out}')


if __name__ == '__main__':
    main()
