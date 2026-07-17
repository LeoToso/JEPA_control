"""Sweep all checkpoints in one or more directories and rank by Pearson correlation.

For each checkpoint_epoch*.pt found, computes:
  - max |r| with each physical state variable (x, xdot, theta, thetadot)
  - sum of max |r| across all 4 vars  (used as ranking score)
  - rho(A_aug)  — spectral radius of learned linearization

Outputs:
  - per-model CSV:  <out_dir>/<label>_pearson_sweep.csv
  - combined table printed to stdout
  - best checkpoint per model printed at the end

Usage:
  python experiments/eval_checkpoints.py \\
      --ckpt_dirs results_inv_fs3_eq_v6/.../checkpoints \\
                  results_pred_fs3_eq_v1/.../checkpoints \\
      --cfgs      configs/cartpole_jepa_pred_inv_fs3_random_eq.yaml \\
                  configs/cartpole_jepa_pred_fs3_random_eq.yaml \\
      --labels    "pred+IDM" "pred-only" \\
      --out_dir   results_comparison
"""
from __future__ import annotations
import argparse
import csv
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


# ── Reuse helpers from compare_latent_dynamics ────────────────────────────────

def _to_tensor(obs_hwc, device):
    return torch.from_numpy(obs_hwc).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


def _stack_obs(curr, prev, frame_stack, device):
    c = _to_tensor(curr, device)
    if frame_stack > 1:
        p = _to_tensor(prev, device)
        return torch.cat([p, c], dim=1)
    return c


def load_model(ckpt_path, cfg_yaml, device):
    from models.jepa import make_jepa, JEPAConfig
    import yaml

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt

    full_cfg = {}
    if cfg_yaml and Path(cfg_yaml).exists():
        with open(cfg_yaml) as f:
            full_cfg = yaml.safe_load(f)

    model_cfg_yaml = full_cfg.get('model', {})

    if isinstance(ckpt, dict) and 'config' in ckpt:
        arch = ckpt['config']
    else:
        arch = {k: model_cfg_yaml.get(k, v) for k, v in {
            'latent_dim': 8, 'action_latent_dim': 1, 'action_encoder': 'linear',
            'encoder_type': 'vit', 'image_size': 64, 'patch_size': 8, 'frame_stack': 2,
            'vit_embed_dim': 128, 'vit_depth': 4, 'vit_num_heads': 4,
            'predictor_window': 3, 'predictor_hidden_dim': 64, 'predictor_n_layers': 2,
        }.items()}

    frame_stack = int(arch.get('frame_stack', 2))

    cfg = JEPAConfig()
    for k, v in arch.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    from models.jepa import JEPAModel
    model = JEPAModel(cfg)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, frame_stack


def collect_rollouts(env, model, frame_stack, device, n_rollouts=40, rollout_len=50,
                     theta_range=0.6, seed=0):
    zs, states = [], []
    rng = np.random.RandomState(seed)
    for _ in range(n_rollouts):
        theta0 = rng.uniform(-theta_range, theta_range)
        x0 = np.array([rng.uniform(-0.3, 0.3), 0.0, theta0, 0.0], dtype=np.float32)
        obs, state, _ = env.reset_to_state(x0)
        prev_obs = obs.copy()
        for _ in range(rollout_len):
            obs_t = _stack_obs(obs, prev_obs, frame_stack, device)
            with torch.no_grad():
                z = model.encoder(obs_t).cpu().numpy()[0]
            zs.append(z)
            states.append(state.copy())
            action = env.sample_action()
            obs_next, state_next, _, done, _ = env.step(action)
            prev_obs = obs.copy()
            obs, state = obs_next, state_next
            if done:
                break
    return np.array(zs), np.array(states)


def pearson_max_r(zs, states):
    """Return max |r| per state variable (x, xdot, theta, thetadot)."""
    n_state = states.shape[1]
    d = zs.shape[1]
    max_r = np.zeros(n_state)
    for j in range(n_state):
        for i in range(d):
            xi, sj = zs[:, i], states[:, j]
            xi_c = xi - xi.mean(); sj_c = sj - sj.mean()
            denom = np.linalg.norm(xi_c) * np.linalg.norm(sj_c) + 1e-12
            r = abs(float(np.dot(xi_c, sj_c) / denom))
            if r > max_r[j]:
                max_r[j] = r
    return max_r


def compute_rho(model, obs_eq, frame_stack, device):
    from control.jacobian import compute_augmented_jacobian_np
    obs_eq_t = _to_tensor(obs_eq, device)
    if frame_stack > 1:
        obs_eq_t = torch.cat([obs_eq_t, obs_eq_t], dim=1)
    with torch.no_grad():
        z_star = model.encoder(obs_eq_t).cpu().numpy()[0]
    try:
        A_aug, _ = compute_augmented_jacobian_np(model, z_star, device)
        return float(np.max(np.abs(np.linalg.eigvals(A_aug))))
    except Exception:
        return float('nan')


def epoch_from_name(path: Path) -> int:
    m = re.search(r'epoch(\d+)', path.stem)
    return int(m.group(1)) if m else -1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dirs', nargs='+', required=True,
                   help='Checkpoint directories (one per model)')
    p.add_argument('--cfgs', nargs='+', required=True,
                   help='Config yaml paths (matched by position to --ckpt_dirs)')
    p.add_argument('--labels', nargs='+', default=None)
    p.add_argument('--out_dir', default='results_comparison')
    p.add_argument('--n_rollouts', type=int, default=40)
    p.add_argument('--rollout_len', type=int, default=50)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--stride', type=int, default=1,
                   help='Evaluate every Nth checkpoint (default: all)')
    args = p.parse_args()

    n_models = len(args.ckpt_dirs)
    if len(args.cfgs) != n_models:
        raise ValueError('--ckpt_dirs and --cfgs must have the same length')
    labels = args.labels or [f'model{i}' for i in range(n_models)]
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from envs.cartpole_visual import ContinuousCartpoleVisual

    STATE_NAMES = ['x', 'xdot', 'theta', 'thetadot']
    SCORE_COL   = 'sum_max_r'

    all_results = {}  # label -> list of row dicts

    for ckpt_dir, cfg_yaml, label in zip(args.ckpt_dirs, args.cfgs, labels):
        import yaml as _yaml
        with open(cfg_yaml) as _f:
            _full_cfg = _yaml.safe_load(_f)
        _env_cfg = _full_cfg.get('environment', {})
        _frame_skip = int(_env_cfg.get('frame_skip', 1))
        _image_size = int(_env_cfg.get('image_size', 64))
        print(f'\n[{label}]  env: frame_skip={_frame_skip}  image_size={_image_size}')
        env = ContinuousCartpoleVisual(
            frame_skip=_frame_skip,
            image_size=_image_size,
            mass_cart=_env_cfg.get('mass_cart', 1.0),
            mass_pole=_env_cfg.get('mass_pole', 0.1),
            pole_length=_env_cfg.get('pole_length', 0.5),
            gravity=_env_cfg.get('gravity', 9.8),
            dt=_env_cfg.get('dt', 0.02),
            action_range=(_env_cfg.get('action_range', [-10, 10])[0],
                          _env_cfg.get('action_range', [-10, 10])[1]),
        )
        obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
        ckpt_dir = Path(ckpt_dir)
        ckpts = sorted(
            [f for f in ckpt_dir.glob('checkpoint_epoch*.pt')],
            key=epoch_from_name
        )
        if args.stride > 1:
            ckpts = ckpts[::args.stride]

        print(f'\n[{label}]  {len(ckpts)} checkpoints in {ckpt_dir}')

        rows = []
        for ckpt_path in ckpts:
            epoch = epoch_from_name(ckpt_path)
            print(f'  epoch {epoch:4d} ...', end='', flush=True)

            try:
                model, frame_stack = load_model(str(ckpt_path), cfg_yaml, device)
                zs, states = collect_rollouts(
                    env, model, frame_stack, device,
                    n_rollouts=args.n_rollouts, rollout_len=args.rollout_len)
                max_r = pearson_max_r(zs, states)
                rho   = compute_rho(model, obs_eq, frame_stack, device)
                score = float(max_r.sum())
                row = {'epoch': epoch, 'rho': rho, SCORE_COL: score}
                for name, r in zip(STATE_NAMES, max_r):
                    row[f'r_{name}'] = float(r)
                print(f'  x={max_r[0]:.3f}  xdot={max_r[1]:.3f}  '
                      f'theta={max_r[2]:.3f}  thetadot={max_r[3]:.3f}  '
                      f'sum={score:.3f}  rho={rho:.4f}')
            except Exception as exc:
                print(f'  ERROR: {exc}')
                row = {'epoch': epoch, 'rho': float('nan'), SCORE_COL: float('nan')}
                for name in STATE_NAMES:
                    row[f'r_{name}'] = float('nan')

            rows.append(row)

        all_results[label] = rows

        # Save CSV
        csv_path = out_dir / f'{label.replace("+", "_").replace(" ", "_")}_pearson_sweep.csv'
        fieldnames = ['epoch'] + [f'r_{n}' for n in STATE_NAMES] + [SCORE_COL, 'rho']
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f'  → saved {csv_path}')
        env.close()

    # ── Print combined summary table ──────────────────────────────────────────
    print('\n' + '='*90)
    print(f'{"epoch":>6}  ' +
          '  '.join(f'{lb:>30}' for lb in labels))
    print(f'{"":>6}  ' +
          '  '.join(f'{"x":>5} {"xd":>5} {"th":>5} {"thd":>5} {"sum":>5} {"rho":>6}'
                    for _ in labels))
    print('-'*90)

    all_epochs = sorted(set(
        r['epoch'] for rows in all_results.values() for r in rows
    ))
    for ep in all_epochs:
        row_str = f'{ep:6d}  '
        for label in labels:
            rows = all_results[label]
            match = next((r for r in rows if r['epoch'] == ep), None)
            if match:
                row_str += (
                    f'{match["r_x"]:5.3f} {match["r_xdot"]:5.3f} '
                    f'{match["r_theta"]:5.3f} {match["r_thetadot"]:5.3f} '
                    f'{match[SCORE_COL]:5.3f} {match["rho"]:6.4f}  '
                )
            else:
                row_str += ' ' * 38
        print(row_str)

    # ── Best checkpoint per model ─────────────────────────────────────────────
    print('\n' + '='*90)
    print('BEST CHECKPOINT PER MODEL (by sum of max |r|):')
    for label in labels:
        rows = [r for r in all_results[label] if not np.isnan(r[SCORE_COL])]
        if not rows:
            print(f'  {label}: no valid checkpoints')
            continue
        best = max(rows, key=lambda r: r[SCORE_COL])
        print(f'  {label}:  epoch={best["epoch"]}  '
              f'x={best["r_x"]:.3f}  xdot={best["r_xdot"]:.3f}  '
              f'theta={best["r_theta"]:.3f}  thetadot={best["r_thetadot"]:.3f}  '
              f'sum={best[SCORE_COL]:.3f}  rho={best["rho"]:.4f}')


if __name__ == '__main__':
    main()
