#!/usr/bin/env python
"""Isolate visual state estimation from dynamics in CartPole control.

Fits a frozen post-hoc ridge decoder on the checkpoint encoder, then runs CEM
with the exact nonlinear CartPole equations.  An oracle-state CEM baseline uses
the same planner and initial conditions, so any gap is attributable to visual
state estimation rather than the learned predictor or CEM hyperparameters.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from data.dataset import load_discrete_dataset_meta
from experiments.evaluate_cartpole_control import build_model, make_env
from experiments.probe_cartpole_nonlinear_instability import (
    decode, encode_split, fit_ridge_probe, obs_tensor, probe_metrics,
)


class GroundTruthCEM:
    """Batched CEM using the environment's exact nonlinear Euler dynamics."""

    def __init__(self, cfg, horizon, n_samples, n_elites, n_iter, init_std,
                 warm_start_std, terminal_cost_mult, action_lb, action_ub,
                 device):
        e = cfg['environment']
        self.M = float(e.get('mass_cart', 1.0))
        self.m = float(e.get('mass_pole', 0.1))
        self.l = float(e.get('pole_length', 0.5))
        self.g = float(e.get('gravity', 9.8))
        self.dt = float(e.get('dt', 0.02))
        self.frame_skip = int(e.get('frame_skip', 1))
        self.friction_cart = float(e.get('friction_cart', 0.0))
        self.friction_pole = float(e.get('friction_pole', 0.0))
        self.horizon = int(horizon)
        self.n_samples = int(n_samples)
        self.n_elites = min(int(n_elites), self.n_samples)
        self.n_iter = int(n_iter)
        self.init_std = float(init_std)
        self.warm_start_std = float(warm_start_std)
        self.terminal_cost_mult = float(terminal_cost_mult)
        self.action_lb, self.action_ub = float(action_lb), float(action_ub)
        self.device = device
        self.Q = torch.tensor([1., 1., 10., 1.], device=device)
        self.R = .01
        self.prev_mu = None

    def reset(self):
        self.prev_mu = None

    def _physics_step(self, state, force):
        x, x_dot, theta, theta_dot = state.unbind(-1)
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        total_mass, ml = self.M + self.m, self.m * self.l
        temp = (force - self.friction_cart * x_dot
                + ml * theta_dot.square() * sin_t) / total_mass
        theta_acc = (self.g * sin_t - cos_t * temp
                     - self.friction_pole * theta_dot / ml) / (
                         self.l * (4. / 3. - self.m * cos_t.square() / total_mass))
        x_acc = temp - ml * theta_acc * cos_t / total_mass
        return torch.stack((
            x + self.dt * x_dot,
            x_dot + self.dt * x_acc,
            theta + self.dt * theta_dot,
            theta_dot + self.dt * theta_acc,
        ), dim=-1)

    def _macro_step(self, state, force):
        for _ in range(self.frame_skip):
            state = self._physics_step(state, force)
        return state

    def _cost(self, x0, actions):
        state = x0.expand(actions.shape[0], -1)
        cost = torch.zeros(actions.shape[0], device=self.device)
        for t in range(self.horizon):
            u = actions[:, t]
            cost += (state.square() * self.Q).sum(-1) + self.R * u.square()
            state = self._macro_step(state, u)
        return cost + self.terminal_cost_mult * (state.square() * self.Q).sum(-1)

    def plan_state(self, state):
        x0 = torch.as_tensor(state, dtype=torch.float32,
                             device=self.device).reshape(1, 4)
        if self.prev_mu is None:
            mu = torch.zeros(self.horizon, device=self.device)
            sigma = torch.full_like(mu, self.init_std)
        else:
            mu = torch.cat((self.prev_mu[1:], torch.zeros(1, device=self.device)))
            # Preserve the previous solution locally instead of discarding it
            # with full exploratory variance at every receding-horizon step.
            sigma = torch.full_like(mu, self.warm_start_std)
        with torch.no_grad():
            for _ in range(self.n_iter):
                actions = (mu + sigma * torch.randn(
                    self.n_samples, self.horizon, device=self.device)).clamp(
                        self.action_lb, self.action_ub)
                elite = actions[torch.argsort(self._cost(x0, actions))[:self.n_elites]]
                mu = elite.mean(0)
                sigma = elite.std(0).clamp(min=.05)
        self.prev_mu = mu.detach()
        return float(mu[0].clamp(self.action_lb, self.action_ub).item())


def encode_online(model, obs, prev_obs, device):
    with torch.no_grad():
        return model.encode_obs(obs_tensor(obs, device),
                                obs_tensor(prev_obs, device)).cpu().numpy()[0]


def run_trial(model, planner, env, x0, mode, ridge_params, z_eq_decoded,
              device, steps, stabilization_threshold, settling_threshold,
              success_hold_steps, failure_penalty):
    planner.reset()
    obs, state, _ = env.reset_to_state(x0)
    prev_obs = obs
    states, actions = [], []
    terminated = False
    w, b, state_mean, state_std = ridge_params
    for _ in range(steps):
        states.append(state.copy())
        if mode == 'oracle_gt_cem':
            state_est = state
        else:
            z = encode_online(model, obs, prev_obs, device)
            state_est = (decode(z[None], w, b, state_mean, state_std)[0]
                         - z_eq_decoded)
        action = planner.plan_state(state_est)
        actions.append([action])
        old_obs = obs
        obs, state, _, done, _ = env.step(action)
        prev_obs = old_obs
        if done:
            terminated = True
            break

    states = np.asarray(states)
    actions = np.asarray(actions)
    errors = np.linalg.norm(states, axis=1)
    metric_errors = np.concatenate((errors, [np.linalg.norm(state)]))
    hold = min(max(int(success_hold_steps), 1), len(metric_errors))
    stabilized = bool(not terminated and np.all(
        metric_errors[-hold:] < stabilization_threshold))
    frac_stable = float(np.mean(metric_errors < settling_threshold))
    Q = np.diag([1., 1., 10., 1.])
    cost = sum(float(x @ Q @ x + .01 * u @ u)
               for x, u in zip(states, actions))
    if terminated:
        cost += float(failure_penalty)
    return {'success': stabilized, 'terminated': terminated,
            'fraction_stable': frac_stable, 'cost': cost,
            'steps': int(len(actions))}


def summarize(rows):
    return {
        'success_rate': float(np.mean([r['success'] for r in rows])),
        'termination_rate': float(np.mean([r['terminated'] for r in rows])),
        'mean_fraction_stable': float(np.mean([r['fraction_stable'] for r in rows])),
        'mean_cost': float(np.mean([r['cost'] for r in rows])),
        'mean_episode_length': float(np.mean([r['steps'] for r in rows])),
        'n_trials': len(rows),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--probe-train-samples', type=int, default=4000)
    p.add_argument('--probe-test-samples', type=int, default=1500)
    p.add_argument('--probe-batch-size', type=int, default=128)
    p.add_argument('--ridge', type=float, default=1e-3)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--cem-horizon', type=int, default=15)
    p.add_argument('--cem-samples', type=int, default=512)
    p.add_argument('--cem-elites', type=int, default=64)
    p.add_argument('--cem-iters', type=int, default=5)
    p.add_argument('--cem-init-std', type=float, default=3.0)
    p.add_argument('--cem-warm-start-std', type=float, default=0.5)
    p.add_argument('--cem-terminal-mult', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, _ = build_model(cfg['model'], checkpoint, device)
    meta = load_discrete_dataset_meta(args.data)
    state_mean = np.asarray(meta.get('state_mean', np.zeros(4)))
    state_std = np.asarray(meta.get('state_std', np.ones(4)))
    print('[probe] fitting frozen ridge state decoder ...')
    z_train, s_train = encode_split(model, args.data, 'train',
                                    args.probe_train_samples,
                                    args.probe_batch_size, args.seed, device)
    z_test, s_test = encode_split(model, args.data, 'test',
                                  args.probe_test_samples,
                                  args.probe_batch_size, args.seed + 1, device)
    w, b = fit_ridge_probe(z_train, s_train, state_mean, state_std, args.ridge)
    metrics = probe_metrics(
        s_test, decode(z_test, w, b, state_mean, state_std))
    print('[state probe] R2=' + ', '.join(f'{v:.3f}' for v in metrics['r2']))

    env_eq = make_env(cfg, args.seed + 999)
    eq_obs, _, _ = env_eq.reset_to_state(np.zeros(4, dtype=np.float32))
    z_eq = encode_online(model, eq_obs, eq_obs, device)
    z_eq_decoded = decode(z_eq[None], w, b, state_mean, state_std)[0]
    env_eq.close()
    print('[equilibrium probe] ' + np.array2string(z_eq_decoded, precision=4))

    bounds = cfg['environment'].get('action_range', [-10, 10])
    ctrl = cfg.get('control', {})
    planner = GroundTruthCEM(
        cfg, args.cem_horizon, args.cem_samples, args.cem_elites,
        args.cem_iters, args.cem_init_std, args.cem_warm_start_std,
        args.cem_terminal_mult, bounds[0], bounds[1], device)
    ridge_params = (w, b, state_mean, state_std)
    rng = np.random.RandomState(args.seed)
    initial_states = [rng.uniform(-float(ctrl.get('init_scale', .05)),
                                  float(ctrl.get('init_scale', .05)), 4)
                      .astype(np.float32) for _ in range(args.trials)]
    results = {'checkpoint': args.checkpoint, 'posthoc_state_probe': metrics,
               'cem': {'horizon': args.cem_horizon,
                       'samples': args.cem_samples, 'elites': args.cem_elites,
                       'iterations': args.cem_iters,
                       'init_std': args.cem_init_std,
                       'warm_start_std': args.cem_warm_start_std,
                       'terminal_cost_multiplier': args.cem_terminal_mult},
               'controllers': {}}
    for mode in ('oracle_gt_cem', 'encoded_gt_cem'):
        env = make_env(cfg, args.seed)
        rows = []
        for trial, x0 in enumerate(initial_states):
            # Give oracle and encoded CEM identical candidate noise per trial.
            torch.manual_seed(args.seed + trial)
            rows.append(run_trial(
                model, planner, env, x0, mode, ridge_params, z_eq_decoded,
                device, args.steps,
                float(ctrl.get('stabilization_threshold', .1)),
                float(ctrl.get('settling_threshold', .05)),
                int(ctrl.get('success_hold_steps', 10)),
                float(ctrl.get('failure_penalty', 1e4))))
        env.close()
        results['controllers'][mode] = summarize(rows)
        row = results['controllers'][mode]
        print(f'[{mode}] success={row["success_rate"]:.1%}  '
              f'terminated={row["termination_rate"]:.1%}  '
              f'stable={row["mean_fraction_stable"]:.1%}  '
              f'cost={row["mean_cost"]:.3f}')

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
