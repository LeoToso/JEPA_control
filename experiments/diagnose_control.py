"""Diagnose sign alignment and latent-space cost issues without retraining.

Run on the server after a training run:
  python experiments/diagnose_control.py --results results/v2_E-full_mixed_fs1_seed42
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results/v2_E-full_mixed_fs1_seed42')
    p.add_argument('--config',  default='configs/cartpole_v2.yaml')
    args = p.parse_args()

    out_dir = Path(args.results)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg   = cfg['environment']
    model_cfg = cfg['model']

    # ── Load model ────────────────────────────────────────────────────────────
    from models.jepa import make_jepa
    model = make_jepa(
        variant='E-full',
        latent_dim=model_cfg['latent_dim'],
        action_latent_dim=model_cfg['action_latent_dim'],
        image_size=env_cfg['image_size'],
        patch_size=model_cfg.get('patch_size', 8),
        vit_embed_dim=model_cfg.get('vit_embed_dim', 128),
        vit_depth=model_cfg.get('vit_depth', 4),
        vit_num_heads=model_cfg.get('vit_num_heads', 4),
        predictor_hidden_dim=model_cfg['predictor_hidden_dim'],
    )
    model.load_state_dict(
        torch.load(out_dir / 'model_final.pt', map_location=device), strict=False)
    model.to(device).eval()

    d = model_cfg['latent_dim']
    state_head = torch.nn.Linear(d, 4).to(device)
    sh_path = out_dir / 'state_head.pt'
    if sh_path.exists():
        state_head.load_state_dict(torch.load(sh_path, map_location=device))
        state_head.eval()
        print('[ok] state_head loaded')
    else:
        print('[warn] no state_head.pt found — state estimation tests skipped')
        state_head = None

    z_star = np.load(out_dir / 'z_star.npy')
    A_jac  = np.load(out_dir / 'A_jac.npy')
    B_jac  = np.load(out_dir / 'B_jac.npy')
    z_star_t = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)

    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=env_cfg.get('frame_skip', 1),
        image_size=env_cfg['image_size'],
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'], seed=0,
    )

    print('\n' + '='*60)
    print('SIGN ALIGNMENT: state_head at swept θ values')
    print('='*60)
    print(f'{"θ_phys":>8}  {"x_hat":>8}  {"ẋ_hat":>8}  {"θ_hat":>8}  {"θ̇_hat":>8}  sign_ok')
    thetas = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
    z_at = {}
    for th in thetas:
        obs, _, _ = env.reset_to_state(np.array([0., 0., th, 0.], dtype=np.float32))
        obs_t = torch.from_numpy(obs).float().permute(2,0,1)[None].to(device) / 255.
        with torch.no_grad():
            z = model.encoder(obs_t)
            z_at[th] = z.cpu().numpy()[0]
            if state_head is not None:
                x_hat = state_head(z).cpu().numpy()[0]
                sign_ok = ('✓' if np.sign(x_hat[2]) == np.sign(th) or abs(th) < 1e-6
                           else '✗ WRONG')
                print(f'{th:>8.2f}  {x_hat[0]:>8.3f}  {x_hat[1]:>8.3f}'
                      f'  {x_hat[2]:>8.3f}  {x_hat[3]:>8.3f}  {sign_ok}')

    print('\n' + '='*60)
    print('B_JAC SIGN CHECK: applying u=+1 from z* — does θ_hat decrease?')
    print('(physical: positive force on cart → pole tilts negative)')
    print('='*60)
    with torch.no_grad():
        a_pos = model.action_encoder(torch.ones(1, 1, device=device))
        a_neg = model.action_encoder(-torch.ones(1, 1, device=device))
        z_after_pos = model.predictor(z_star_t, a_pos)
        z_after_neg = model.predictor(z_star_t, a_neg)
        if state_head is not None:
            x_pos = state_head(z_after_pos).cpu().numpy()[0]
            x_neg = state_head(z_after_neg).cpu().numpy()[0]
            x_eq  = state_head(z_star_t).cpu().numpy()[0]
            print(f'state_head(z*)            : θ={x_eq[2]:.4f}  (ideal 0)')
            print(f'state_head(f(z*, u=+1))   : θ={x_pos[2]:.4f}  '
                  f'Δθ={x_pos[2]-x_eq[2]:+.4f}  (expect negative)')
            print(f'state_head(f(z*, u=-1))   : θ={x_neg[2]:.4f}  '
                  f'Δθ={x_neg[2]-x_eq[2]:+.4f}  (expect positive)')
            if x_pos[2] < x_eq[2]:
                print('  → sign CORRECT: u>0 reduces θ in latent space')
            else:
                print('  → sign WRONG: u>0 INCREASES θ — MPC will push wrong way!')

    print('\n' + '='*60)
    print('PREDICTOR DRIFT from z* with u=0')
    print('='*60)
    with torch.no_grad():
        a_zero = model.action_encoder(torch.zeros(1, 1, device=device))
        z_cur  = z_star_t.clone()
        for k in range(10):
            z_cur = model.predictor(z_cur, a_zero)
            dist  = float(torch.norm(z_cur - z_star_t))
            state_str = ''
            if state_head is not None:
                x_k = state_head(z_cur).cpu().numpy()[0]
                state_str = (f'  [x={x_k[0]:+.3f}  ẋ={x_k[1]:+.3f}'
                             f'  θ={x_k[2]:+.3f}  θ̇={x_k[3]:+.3f}]')
            print(f'  step {k+1:2d}: ||z-z*||={dist:.4f}{state_str}')

    print('\n' + '='*60)
    print('ENCODER SENSITIVITY: does z change meaningfully with θ?')
    print('(if ||z(θ) - z(0)|| is tiny, the encoder ignores θ)')
    print('='*60)
    z0 = z_at.get(0.0, None)
    if z0 is not None:
        for th in thetas:
            dz = np.linalg.norm(z_at[th] - z0)
            print(f'  θ={th:+.2f}  ||z(θ)-z(0)||={dz:.4f}')

    print('\n' + '='*60)
    print('Q_LQR ALIGNMENT: does the cost gradient point the right way?')
    print('(apply +ε in the θ direction of latent space → cost should increase)')
    print('='*60)
    W = state_head.weight.detach().cpu().numpy()  # (4, d)
    Q_phys = np.diag([10., 0.1, 100., 0.1])
    Q_lqr  = W.T @ Q_phys @ W + 0.01 * np.eye(d)
    Q_lqr *= d / (np.trace(Q_lqr) + 1e-12)
    # Direction in latent space that corresponds to θ > 0
    theta_dir = W[2]  # row 2 of state_head = θ mapping
    theta_dir /= np.linalg.norm(theta_dir) + 1e-12
    dz_theta = 0.1 * theta_dir
    cost_at_z_star = float(dz_theta @ Q_lqr @ dz_theta)
    print(f'||W[θ]|| = {np.linalg.norm(W[2]):.4f}')
    print(f'Cost for δz in θ direction: {cost_at_z_star:.6f}  (should be > 0)')

    # What action does grad MPC prefer from z*?
    print('\n' + '='*60)
    print('GRAD MPC: single plan from z* (should output u ≈ 0 at equilibrium)')
    print('='*60)
    Q_t  = torch.tensor(Q_lqr, dtype=torch.float32, device=device)
    R_t  = torch.tensor(0.01 * np.eye(1), dtype=torch.float32, device=device)
    Qf_t = 10. * Q_t
    H    = int(cfg['mpc'].get('horizon', 10))
    lr   = float(cfg['mpc'].get('grad_lr', 0.05))
    n_iter = int(cfg['mpc'].get('grad_n_iter', 100))

    u_seq = torch.zeros(H, 1, device=device, requires_grad=True)
    opt   = torch.optim.Adam([u_seq], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        z = z_star_t.clone()
        cost = torch.zeros(1, device=device)
        for k in range(H):
            u_k = torch.clamp(u_seq[k], -10., 10.).unsqueeze(0)
            a_k = model.action_encoder(u_k)
            dz  = z - z_star_t
            cost = cost + (dz @ Q_t @ dz.T).squeeze()
            cost = cost + (u_k @ R_t @ u_k.T).squeeze()
            z   = model.predictor(z, a_k)
        dz_f = z - z_star_t
        cost = cost + (dz_f @ Qf_t @ dz_f.T).squeeze()
        cost.backward()
        opt.step()
    u_plan = u_seq.detach().cpu().numpy().flatten()
    print(f'Planned actions (H={H}): {np.round(u_plan, 3)}')
    print(f'Mean |u|: {np.mean(np.abs(u_plan)):.4f}  '
          f'(should be near 0 when starting at equilibrium)')
    print(f'Sign of u[0]: {np.sign(u_plan[0])}  (arbitrary but should not be consistently wrong)')

    env.close()
    print('\nDone.')


if __name__ == '__main__':
    main()
