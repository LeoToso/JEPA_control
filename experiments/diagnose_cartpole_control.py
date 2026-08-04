"""Diagnose whether cartpole control failures come from physics or learned dynamics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from control.lqr import solve_discrete_lqr
from data.dataset import load_discrete_dataset_meta
from experiments.evaluate_cartpole_control import (
    build_model, equilibrium_latent, make_env, physical_action_jacobian)
from control.jacobian import compute_jacobian_np


def physical_jacobian(env, eps_x=1e-4, eps_u=1e-3):
    A = np.zeros((4, 4)); B = np.zeros((4, 1))
    for j in range(4):
        dx = np.zeros(4); dx[j] = eps_x
        env.reset_to_state(dx); _, xp, _, _, _ = env.step(0.0)
        env.reset_to_state(-dx); _, xm, _, _, _ = env.step(0.0)
        A[:, j] = (xp - xm) / (2 * eps_x)
    env.reset_to_state(np.zeros(4)); _, xp, _, _, _ = env.step(eps_u)
    env.reset_to_state(np.zeros(4)); _, xm, _, _, _ = env.step(-eps_u)
    B[:, 0] = (xp - xm) / (2 * eps_u)
    return A, B


def oracle_eval(env, K, trials, steps, init_scale, seed, threshold, hold=10):
    rng = np.random.RandomState(seed)
    success, terminated, costs, sat = [], [], [], []
    Q = np.diag([1., 1., 10., 1.])
    for _ in range(trials):
        _, x, _ = env.reset_to_state(rng.uniform(-init_scale, init_scale, 4))
        xs, us, done = [], [], False
        for _ in range(steps):
            u = float(np.clip((-K @ x).item(), env.action_low, env.action_high))
            xs.append(x.copy()); us.append(u)
            _, x, _, done, _ = env.step(u)
            if done:
                break
        errs = np.linalg.norm(np.vstack(xs + [x]), axis=1)
        success.append(bool(not done and np.all(errs[-min(hold, len(errs)):] < threshold)))
        terminated.append(done)
        stage = sum(float(xx @ Q @ xx + .01 * uu * uu) for xx, uu in zip(xs, us))
        costs.append(stage + (1e4 if done else 0.0))
        sat.append(np.mean(np.abs(us) >= .99 * env.action_high))
    return dict(success_rate=float(np.mean(success)),
                termination_rate=float(np.mean(terminated)),
                mean_cost=float(np.mean(costs)), saturation_rate=float(np.mean(sat)))


def empirical_latent_B(model, env, device, action_scale, forces=(0.5, 1., 3.)):
    obs0, _, _ = env.reset_to_state(np.zeros(4))
    x0 = torch.from_numpy(obs0).float().permute(2, 0, 1)[None].to(device) / 255.
    with torch.no_grad(): z0 = model.encode_obs(x0, x0)
    rows = []
    for force in forces:
        encoded = []
        predicted = []
        for sign in (1., -1.):
            env.reset_to_state(np.zeros(4))
            obs1, _, _, _, _ = env.step(sign * force)
            x1 = torch.from_numpy(obs1).float().permute(2, 0, 1)[None].to(device) / 255.
            u = torch.tensor([[[sign * force / action_scale]]], device=device)
            with torch.no_grad():
                encoded.append(model.encode_obs(x1, x0).cpu().numpy()[0])
                predicted.append(model.predict(z0[:, None], u).cpu().numpy()[0])
        B_real = (encoded[0] - encoded[1]) / (2 * force)
        B_pred = (predicted[0] - predicted[1]) / (2 * force)
        denom = np.linalg.norm(B_real) * np.linalg.norm(B_pred)
        rows.append((force, np.linalg.norm(B_real), np.linalg.norm(B_pred),
                     float(B_real @ B_pred / denom) if denom > 1e-12 else np.nan,
                     np.linalg.norm(B_pred - B_real) / max(np.linalg.norm(B_real), 1e-12)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--trials', type=int, default=50)
    p.add_argument('--steps', type=int, default=200); p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    with open(a.config) as f: cfg = yaml.safe_load(f)
    env = make_env(cfg, a.seed); ctrl = cfg.get('control', {})
    Ap, Bp = physical_jacobian(env)
    K, _, poles = solve_discrete_lqr(Ap, Bp, np.diag([1., 1., 10., 1.]), np.array([[.01]]))
    print(f'[oracle] rho(A)={max(abs(np.linalg.eigvals(Ap))):.4f}  '
          f'rho(A-BK)={max(abs(poles)):.4f}  ||B||={np.linalg.norm(Bp):.5f}')
    print('[oracle]', oracle_eval(env, K, a.trials, a.steps,
          float(ctrl.get('init_scale', .05)), a.seed,
          float(ctrl.get('stabilization_threshold', .1))))

    ck = torch.load(a.checkpoint, map_location=device)
    model, _ = build_model(cfg['model'], ck, device)
    zstar = equilibrium_latent(model, env, device)
    Al, Be = compute_jacobian_np(model, zstar, device=device)
    scale = float(load_discrete_dataset_meta(a.data).get('action_scale', 1.))
    Bl = physical_action_jacobian(model, Be, scale, device)
    print(f'[learned] rho(A)={max(abs(np.linalg.eigvals(Al))):.4f}  ||B_raw||={np.linalg.norm(Bl):.5f}')
    print('[action test] force  ||B_real||  ||B_pred||  cosine  relative_error')
    for row in empirical_latent_B(model, env, device, scale):
        print(f'              {row[0]:4.1f}   {row[1]:10.5f}  {row[2]:10.5f}  {row[3]:+7.3f}  {row[4]:10.3f}')
    env.close()


if __name__ == '__main__': main()
