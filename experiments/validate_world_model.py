"""
World-model rollout accuracy test.

For a grid of initial physical states, rolls out K steps through:
  obs -> encoder -> predictor -> state_head -> predicted_state

and compares against the ground-truth cartpole physics.

Usage (run on the server where the checkpoints live):
  python experiments/validate_world_model.py \
      --checkpoint results/v2_E-full_mixed_fs1_seed42/checkpoints/checkpoint_epoch0200.pt \
      --state-head  results/v2_E-full_mixed_fs1_seed42/state_head.pt \
      --config      configs/cartpole_v2_fullspec.yaml \
      --steps 10
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.jepa import JEPAModel, JEPAConfig
from envs.cartpole_visual import ContinuousCartpoleVisual


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, cfg: dict, device: torch.device) -> JEPAModel:
    model = JEPAModel(JEPAConfig.from_dict(cfg['model'])).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # accept both full checkpoints and bare state-dicts
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    ep = ckpt.get('epoch', '?')
    print(f'[model] loaded  epoch={ep}  path={ckpt_path}')
    return model


def load_state_head(sh_path: str, latent_dim: int, device: torch.device):
    head = torch.nn.Linear(latent_dim, 4).to(device)
    head.load_state_dict(torch.load(sh_path, map_location=device))
    head.eval()
    print(f'[head] loaded  path={sh_path}')
    return head


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """HWC uint8 -> (1, C, H, W) float32 in [0,1]."""
    t = torch.from_numpy(obs).float() / 255.0          # (H, W, C)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)


@torch.no_grad()
def rollout_model(model, head, env_cfg: dict, init_state: np.ndarray,
                  actions: np.ndarray, device: torch.device):
    """
    Roll out the world model for len(actions) steps from init_state.
    Returns predicted states shape (K+1, 4) alongside GT states (K+1, 4).
    """
    env = ContinuousCartpoleVisual(
        frame_skip=1,
        image_size=env_cfg['image_size'],
        mass_cart=env_cfg['mass_cart'],
        mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'],
        gravity=env_cfg['gravity'],
        dt=env_cfg['dt'],
        action_range=tuple(env_cfg['action_range']),
    )

    obs, gt_state, _ = env.reset_to_state(init_state)
    obs_t = obs_to_tensor(obs, device)

    W = getattr(model.config, 'predictor_window', 1)
    d = model.config.latent_dim

    # encode first observation
    z0 = model.encoder(obs_t)  # (1, d)

    # build window history (all z0)
    z_window = z0.unsqueeze(1).expand(1, W, d).clone()  # (1, W, d)

    pred_states = [head(z0).cpu().numpy()[0]]
    gt_states   = [gt_state.copy()]

    for u in actions:
        # predictor step
        u_t = torch.tensor([[[float(u)]]], dtype=torch.float32, device=device)  # (1,1,1)
        u_win = torch.zeros(1, W, 1, device=device)
        u_win[:, -1, :] = u_t[:, 0, :]

        z_next = model.predict(z_window, u_win)           # (1, d)
        pred_states.append(head(z_next).cpu().numpy()[0])

        # advance window
        z_window = torch.cat([z_window[:, 1:], z_next.unsqueeze(1)], dim=1)

        # GT step
        obs, gt_state, _, done, _ = env.step(u)
        gt_states.append(gt_state.copy())
        if done:
            break

    env.close()
    return np.array(pred_states), np.array(gt_states)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--state-head', required=True)
    p.add_argument('--config',     default='configs/cartpole_v2_fullspec.yaml')
    p.add_argument('--steps',      type=int, default=10)
    p.add_argument('--device',     default='cpu')
    args = p.parse_args()

    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg['environment']

    model = load_model(args.checkpoint, cfg, device)
    head  = load_state_head(args.state_head, model.config.latent_dim, device)

    # Grid of initial states: vary theta (pole angle) and theta_dot (angular vel)
    # x=0, x_dot=0, theta=*, theta_dot=*
    thetas     = [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20]   # radians
    theta_dots = [-0.5,   0.0,   0.5]
    K = args.steps

    state_names = ['x', 'ẋ', 'θ', 'θ̇']
    errs_by_step = np.zeros((K,))   # mean abs error per step across all trials
    n_trials = 0

    print(f'\n{"="*70}')
    print(f'WORLD MODEL ROLLOUT ACCURACY  ({K} steps, u=0 everywhere)')
    print(f'{"="*70}')
    print(f'{"Init state":30s}  ' + '  '.join(f'step{k+1:02d}' for k in range(min(K,5))))
    print('-'*70)

    all_abs_errs = []

    for theta in thetas:
        for theta_dot in theta_dots:
            init = np.array([0.0, 0.0, theta, theta_dot], dtype=np.float32)
            actions = np.zeros(K)   # zero control — tests open-loop dynamics

            pred, gt = rollout_model(model, head, env_cfg, init, actions, device)

            abs_err = np.abs(pred - gt)       # (K+1, 4)
            all_abs_errs.append(abs_err[1:])  # drop step-0 (just encoder error)

            per_step_norm = np.linalg.norm(abs_err[1:], axis=1)  # (K,)

            tag = f'θ={theta:+.2f} θ̇={theta_dot:+.1f}'
            step_strs = '  '.join(f'{e:.4f}' for e in per_step_norm[:5])
            print(f'{tag:30s}  {step_strs}')

            n_trials += 1

    all_abs_errs = np.array(all_abs_errs)  # (N_trials, K, 4)
    mean_abs = all_abs_errs.mean(axis=0)   # (K, 4)

    print(f'\n{"="*70}')
    print('MEAN ABSOLUTE ERROR PER STATE COMPONENT AND STEP')
    print(f'{"Step":>6}  ' + '  '.join(f'{n:>8}' for n in state_names))
    print('-'*70)
    for k in range(K):
        row = '  '.join(f'{mean_abs[k, i]:8.4f}' for i in range(4))
        print(f'  {k+1:3d}    {row}')

    print(f'\n{"="*70}')
    print('SUMMARY')
    print(f'  Mean ||error|| step 1:   {np.linalg.norm(mean_abs[0]):.4f}')
    print(f'  Mean ||error|| step 5:   {np.linalg.norm(mean_abs[4]):.4f}' if K >= 5 else '')
    print(f'  Mean ||error|| step {K}:  {np.linalg.norm(mean_abs[-1]):.4f}')

    # Per-component summary
    print(f'\n  Dominant error source at step 1: '
          f'{state_names[np.argmax(mean_abs[0])]}  '
          f'(err={np.max(mean_abs[0]):.4f})')
    print(f'  Dominant error source at step {K}: '
          f'{state_names[np.argmax(mean_abs[-1])]}  '
          f'(err={np.max(mean_abs[-1]):.4f})')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
