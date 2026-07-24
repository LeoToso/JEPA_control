"""Sweep diag_predictor_divergence across all checkpoints in a directory.

For each checkpoint runs a passive (u=0) rollout from theta0 and records:
  - max ||z_real - z*||   : how far the encoder moves z as the pole falls
  - final ||z_pred - z*|| : how far the predictor thinks z moved (should match above)
  - max ||z_pred - z_real||: worst-case predictor error
  - encoder_tracks        : 1 if max||z_real-z*|| > 2x initial ||z0-z*||

Outputs a summary CSV + 2-panel plot.

Usage:
  python experiments/diag_predictor_divergence_sweep.py \\
      --ckpt-dir results/jepa_v6_state_control/checkpoints \\
      --config   configs/cartpole_jepa_state_control.yaml \\
      --theta0   0.10 --steps 30 \\
      --out      results/jepa_v6_state_control/diag_divergence_sweep.png \\
      --csv      results/jepa_v6_state_control/diag_divergence_sweep.csv \\
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
    stem = p.stem
    for part in stem.split('_'):
        if part.startswith('epoch'):
            return int(part[5:])
    return -1


def _make_model(model_cfg, env_cfg, device):
    from models.jepa import make_jepa
    return make_jepa(
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
        predictor_window=int(model_cfg.get('predictor_window', 5)),
        predictor_embed_dim=int(model_cfg.get('predictor_embed_dim', 128)),
        predictor_depth=int(model_cfg.get('predictor_depth', 4)),
        predictor_num_heads=int(model_cfg.get('predictor_num_heads', 4)),
        predictor_mlp_ratio=float(model_cfg.get('predictor_mlp_ratio', 4.0)),
    )


def _run_one(ckpt_path, cfg, device, theta0, max_steps):
    """Return dict of divergence metrics for one checkpoint."""
    env_cfg   = cfg['environment']
    model_cfg = dict(cfg['model'])

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw.get('model_state', raw.get('model_state_dict', raw))
    if isinstance(raw, dict) and 'config' in raw:
        _ckpt_cfg = raw['config']
        for k in ('latent_dim', 'action_latent_dim', 'action_encoder',
                  'encoder_type', 'patch_size', 'frame_stack',
                  'vit_embed_dim', 'vit_depth', 'vit_num_heads',
                  'predictor_type', 'predictor_hidden_dim', 'predictor_n_layers',
                  'predictor_window', 'predictor_embed_dim', 'predictor_depth',
                  'predictor_num_heads', 'predictor_mlp_ratio'):
            if k in _ckpt_cfg:
                model_cfg[k] = _ckpt_cfg[k]

    model = _make_model(model_cfg, env_cfg, device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    frame_stack = int(model_cfg.get('frame_stack', 1))
    W = getattr(model.config, 'predictor_window', 1)

    from envs.cartpole_visual import ContinuousCartpoleVisual
    frame_skip = int(env_cfg.get('frame_skip', 1))
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip, image_size=int(env_cfg['image_size']),
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'], seed=0,
    )

    def encode(obs_np, prev_obs_np=None):
        curr = torch.from_numpy(obs_np).float().permute(2, 0, 1)[None].to(device) / 255.0
        if frame_stack > 1:
            prev = (curr if prev_obs_np is None else
                    torch.from_numpy(prev_obs_np).float().permute(2, 0, 1)[None].to(device) / 255.0)
            inp = torch.cat([prev, curr], dim=1)
        else:
            inp = curr
        with torch.no_grad():
            return model.encoder(inp).cpu().numpy()[0]

    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    z_star = encode(obs_eq)

    x0 = np.array([0., 0., theta0, 0.], dtype=np.float32)
    obs, state_phys, _ = env.reset_to_state(x0)

    z0 = encode(obs, None)
    real_z   = [z0.copy()]
    real_phys = [state_phys.copy()]
    pred_z   = [z0.copy()]

    for step in range(1, max_steps + 1):
        obs_next, state_next, _, done, _ = env.step(0.0)
        z_real = encode(obs_next, obs)

        past = real_z[:]
        while len(past) < W:
            past = [past[0]] + past
        win_entries = past[-W:]
        z_win_real = torch.stack(
            [torch.tensor(zz, device=device).float().unsqueeze(0)
             for zz in win_entries], dim=1)
        u_win_zero = torch.zeros(1, W, 1, device=device)
        with torch.no_grad():
            z_pred_next = model.predict(z_win_real, u_win_zero).cpu().numpy()[0]

        real_z.append(z_real.copy())
        real_phys.append(state_next.copy())
        pred_z.append(z_pred_next.copy())

        obs = obs_next
        state_phys = state_next
        if done:
            break

    env.close()

    Z  = np.array(real_z)
    Zp = np.array(pred_z)
    Xp = np.array(real_phys)

    dist_real = np.linalg.norm(Z  - z_star, axis=1)
    dist_pred = np.linalg.norm(Zp - z_star, axis=1)
    pred_err  = np.linalg.norm(Zp - Z,     axis=1)

    return {
        'n_steps':          len(Z),
        'final_theta':      float(Xp[-1, 2]),
        'init_z_dist':      float(dist_real[0]),
        'max_z_real':       float(dist_real.max()),
        'final_z_pred':     float(dist_pred[-1]),
        'max_pred_err':     float(pred_err[1:].max()) if len(pred_err) > 1 else 0.0,
        'final_pred_err':   float(pred_err[-1]),
        # ratio: how much does the encoder amplify vs predictor amplify?
        'encoder_amplif':   float(dist_real.max() / max(dist_real[0], 1e-6)),
        'pred_amplif':      float(dist_pred.max() / max(dist_pred[1], 1e-6)),
        'dist_real':        dist_real.tolist(),
        'dist_pred':        dist_pred.tolist(),
        'thetas':           Xp[:, 2].tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir', required=True)
    p.add_argument('--config',   required=True)
    p.add_argument('--theta0',   type=float, default=0.10)
    p.add_argument('--steps',    type=int,   default=30)
    p.add_argument('--step',     type=int,   default=1,
                   help='Evaluate every N-th checkpoint')
    p.add_argument('--out',      default=None)
    p.add_argument('--csv',      default=None)
    p.add_argument('--device',   default=None)
    args = p.parse_args()

    device   = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.out)  if args.out  else ckpt_dir.parent / 'diag_divergence_sweep.png'
    csv_path = Path(args.csv)  if args.csv  else ckpt_dir.parent / 'diag_divergence_sweep.csv'

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpts = sorted(
        [p_ for p_ in ckpt_dir.glob('checkpoint_epoch*.pt')],
        key=_epoch_from_name)[::args.step]
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
    print(f'Found {len(ckpts)} checkpoints  '
          f'(ep{_epoch_from_name(ckpts[0])}–ep{_epoch_from_name(ckpts[-1])})')
    print(f'theta0={args.theta0}  max_steps={args.steps}')
    print(f'\n{"epoch":>6}  {"n_steps":>7}  {"θ_final":>8}  '
          f'{"z0_dist":>8}  {"max_zreal":>10}  {"enc_amp":>8}  '
          f'{"zpred_fin":>10}  {"pred_amp":>9}  {"max_perr":>9}')
    print('-' * 85)

    rows = []
    traj_data = []
    for ckpt_path in ckpts:
        epoch = _epoch_from_name(ckpt_path)
        try:
            m = _run_one(ckpt_path, cfg, device, args.theta0, args.steps)
        except Exception as e:
            print(f'ep{epoch:04d}  ERROR: {e}')
            continue

        print(f'{epoch:>6}  {m["n_steps"]:>7}  {m["final_theta"]:>8.4f}  '
              f'{m["init_z_dist"]:>8.4f}  {m["max_z_real"]:>10.4f}  '
              f'{m["encoder_amplif"]:>8.3f}x  '
              f'{m["final_z_pred"]:>10.4f}  {m["pred_amplif"]:>8.3f}x  '
              f'{m["max_pred_err"]:>9.4f}')

        rows.append({'epoch': epoch, **{k: v for k, v in m.items()
                                        if k not in ('dist_real', 'dist_pred', 'thetas')}})
        traj_data.append({'epoch': epoch, **m})

    if not rows:
        print('No results.')
        return

    # CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f'\nCSV → {csv_path}')

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs      = [r['epoch']        for r in rows]
    enc_amp     = [r['encoder_amplif'] for r in rows]
    pred_amp    = [r['pred_amplif']  for r in rows]
    max_perr    = [r['max_pred_err'] for r in rows]
    max_zreal   = [r['max_z_real']   for r in rows]
    zpred_fin   = [r['final_z_pred'] for r in rows]

    cmap = plt.cm.viridis
    n    = len(traj_data)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Panel 1: encoder amplification vs predictor amplification over epochs
    ax = axes[0]
    ax.plot(epochs, enc_amp,  'o-', color='steelblue', ms=4, lw=1.5, label='encoder amp (max||z_real-z*||/init)')
    ax.plot(epochs, pred_amp, 's--', color='tomato',   ms=4, lw=1.5, label='predictor amp (max||z_pred-z*||/init)')
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Amplification factor')
    ax.set_title('Encoder vs Predictor Amplification')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: max predictor error over epochs
    ax = axes[1]
    ax.plot(epochs, max_perr, 'o-', color='darkorange', ms=4, lw=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('max ||z_pred - z_real||')
    ax.set_title('Max Predictor Error (passive rollout)')
    ax.grid(alpha=0.3)

    # Panel 3: per-checkpoint trajectory of ||z_real-z*|| and ||z_pred-z*||
    ax = axes[2]
    for i, td in enumerate(traj_data):
        c = cmap(i / max(n - 1, 1))
        steps_r = list(range(len(td['dist_real'])))
        steps_p = list(range(len(td['dist_pred'])))
        ax.plot(steps_r, td['dist_real'], '-',  color=c, alpha=0.7, lw=1.2)
        ax.plot(steps_p, td['dist_pred'], '--', color=c, alpha=0.5, lw=1.0)
    # legend proxy
    ax.plot([], [], 'k-',  label='||z_real - z*||')
    ax.plot([], [], 'k--', label='||z_pred - z*||')
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=epochs[0], vmax=epochs[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Epoch')
    ax.set_xlabel('Step'); ax.set_ylabel('||z - z*||')
    ax.set_title(f'Latent Trajectory (θ₀={args.theta0} rad, solid=real, dashed=pred)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'Predictor Divergence Sweep — {ckpt_dir.parent.name}', fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f'Plot → {out_path}')


if __name__ == '__main__':
    main()
