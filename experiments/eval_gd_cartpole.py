#!/usr/bin/env python
"""Gradient-based planning for CartPole SMWM checkpoints.

Optimises a horizon-H action sequence by backpropagating through the latent
predictor to minimise the squared distance to a goal latent:

    L(u) = ||z_H(u) - z*||²          (mode='last')
or
    L(u) = mean_t ||z_t(u) - z*||²   (mode='all')

Uses Adam on the action sequence with optional Gaussian noise injection after
each gradient step, following the GD planner in

    github.com/qw3rtman/robust-world-model-planning  (planning/gd.py)

Replanning is MPC-style: plan H steps, execute K steps, replan.

Usage
-----
  python experiments/compare_gd_smwm.py \\
      --ckpts  /mnt/t7shield/.../model_final.pt \\
      --cfgs   configs/cartpole_jepa_ibot_projector_ar_1step.yaml \\
      --trials 50 \\
      --planning-horizon 10  --executed-steps 1 \\
      --gd-steps 50  --lr 0.05  --action-noise 0.01 \\
      --objective last \\
      --output results/gd_ibot_ar.json
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

from experiments.probe_utils import make_env
from cartpole_cem_utils import encode_online, summarize
from models.jepa_world_model import JEPAWorldModel


# ── label helper ──────────────────────────────────────────────────────────────

def checkpoint_label(checkpoint):
    path = Path(checkpoint)
    run_name = (path.parent.parent.name
                if path.parent.name == 'checkpoints' else path.parent.name)
    r = run_name.lower()
    if 'sigreg_ms' in r or 'sigreg-ms' in r:  return 'SIGReg MS'
    if 'sigreg' in r:                           return 'SIGReg'
    if '_sr_' in r or r.endswith('_sr'):        return 'State recon.'
    if 'ar_ms' in r or 'ar-ms' in r:           return 'AR MS'
    if 'dinov2' in r:                           return 'DINOv2'
    if 'ibot' in r:                             return 'iBOT'
    if 'noproprio' in r or 'no_proprio' in r:  return run_name + ' (no proprio)'
    return run_name


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path, config_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg  = dict(checkpoint['model_config'])
    model      = JEPAWorldModel(model_cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    state_mean   = np.asarray(checkpoint.get('state_mean', np.zeros(4)), dtype=np.float32)
    state_std    = np.asarray(checkpoint.get('state_std',  np.ones(4)),  dtype=np.float32)
    action_scale = float(checkpoint.get('action_scale', 10.))
    return cfg, checkpoint, model, state_mean, state_std, action_scale


# ── gradient-based planner ────────────────────────────────────────────────────

class CartPoleGDPlanner:
    """Gradient-based action-sequence optimiser for CartPole.

    At each replan:
      1. Start from u = zeros (or warm-start from previous plan)
      2. For gd_steps iterations:
           a. Roll out z0 under u through the predictor
           b. Compute loss L(u) in latent space
           c. Adam step on u
           d. Inject Gaussian noise: u ← u + N(0, action_noise²)
      3. Execute the first executed_steps actions.

    Parameters
    ----------
    objective : 'last' or 'all'
        'last'  — minimise ||z_H - z*||²  (terminal cost, like paper CEM)
        'all'   — minimise mean_t ||z_t - z*||²  (all-timesteps cost)
    warm_start : bool
        Shift the previous planned sequence to initialise the next replan.
    """

    def __init__(self, model, action_scale, horizon, executed_steps,
                 gd_steps, lr, action_noise,
                 action_lb, action_ub, device,
                 objective='last', warm_start=True,
                 optimizer_type='adam'):
        self.model          = model
        self.action_scale   = float(action_scale)
        self.horizon        = int(horizon)
        self.executed_steps = min(int(executed_steps), self.horizon)
        self.gd_steps       = int(gd_steps)
        self.lr             = float(lr)
        self.action_noise   = float(action_noise)
        self.action_lb      = float(action_lb)
        self.action_ub      = float(action_ub)
        self.device         = device
        self.objective      = objective
        self.warm_start     = warm_start
        self.optimizer_type = optimizer_type
        self._prev_u        = None  # for warm-start

    def _rollout(self, z0, u):
        """Roll out latent state under action sequence u.

        Parameters
        ----------
        z0 : (1, D) tensor
        u  : (H,) tensor  (raw action units, NOT normalised)

        Returns
        -------
        zs : (1, H, D) tensor  — latent after each action step
        """
        z  = z0
        zs = []
        for t in range(self.horizon):
            a_norm = (u[t:t+1] / self.action_scale).reshape(1, 1)  # (1, 1)
            a_ctx  = self.model.expand_action(a_norm).unsqueeze(1)  # (1, 1, ctx)
            z      = self.model.predict(z[:, None], a_ctx)[:, 0]    # (1, D)
            zs.append(z)
        return torch.stack(zs, dim=1)   # (1, H, D)

    def _make_optimizer(self, u):
        if self.optimizer_type == 'sgd':
            return torch.optim.SGD([u], lr=self.lr)
        elif self.optimizer_type == 'momentum':
            return torch.optim.SGD([u], lr=self.lr, momentum=0.9)
        elif self.optimizer_type == 'adam':
            return torch.optim.Adam([u], lr=self.lr)
        elif self.optimizer_type == 'adamw':
            return torch.optim.AdamW([u], lr=self.lr)
        raise ValueError(f'Unknown optimizer_type: {self.optimizer_type}')

    def plan_sequence(self, z0, z_goal):
        """Optimise and return (executed_steps,) numpy action array."""
        # Initialise action sequence
        if self.warm_start and self._prev_u is not None:
            # shift left by executed_steps, zero-pad remainder
            prev = self._prev_u.detach()
            shifted = torch.cat([
                prev[self.executed_steps:],
                torch.zeros(self.executed_steps, device=self.device,
                            dtype=prev.dtype)
            ], dim=0)
            u = shifted.clone().requires_grad_(True)
        else:
            u = torch.zeros(self.horizon, device=self.device,
                            requires_grad=True)

        optimizer = self._make_optimizer(u)

        for _ in range(self.gd_steps):
            optimizer.zero_grad()
            zs = self._rollout(z0, u)            # (1, H, D)
            if self.objective == 'last':
                loss = (zs[:, -1] - z_goal).pow(2).sum()
            else:   # 'all'
                loss = (zs - z_goal.unsqueeze(1)).pow(2).sum(-1).mean()
            loss.backward()
            optimizer.step()
            # Noise injection as in gd.py / gradcem.py
            with torch.no_grad():
                u += torch.randn_like(u) * self.action_noise

        with torch.no_grad():
            u_clamped = u.clamp(self.action_lb, self.action_ub)

        self._prev_u = u_clamped.detach()
        return u_clamped[:self.executed_steps].cpu().numpy()


# ── trial evaluation ──────────────────────────────────────────────────────────

def evaluate_trial_gd(env, planner, model, initial_state, goal_state, z_goal,
                      state_mean, state_std, primitive_budget,
                      stabilization_threshold, settling_threshold,
                      success_hold_steps, device):
    """Run one closed-loop gradient-based planning episode.

    Matches the interface of evaluate_trial in cartpole_cem_utils.py so the
    same summarize() function can be used for both CEM and GD results.
    """
    obs, state, _ = env.reset_to_state(initial_state)
    prev_obs = obs
    frame_skip   = int(env.frame_skip)
    macro_budget = int(np.ceil(primitive_budget / frame_skip))

    states  = [state.copy()]
    actions = []
    latent_goal_errors = []
    terminated = False
    replans    = 0

    planner._prev_u = None   # reset warm-start at episode start

    while len(actions) < macro_budget and not terminated:
        z = encode_online(model, obs, prev_obs, state,
                          state_mean, state_std, device)
        latent_goal_errors.append(
            float(torch.linalg.vector_norm(z - z_goal)))
        sequence = planner.plan_sequence(z, z_goal)
        replans += 1
        remaining = macro_budget - len(actions)
        for action in sequence[:remaining]:
            old_obs = obs
            obs, state, _, done, _ = env.step(float(action))
            prev_obs = old_obs
            actions.append(float(action))
            states.append(state.copy())
            if done:
                terminated = True
                break

    z_final = encode_online(model, obs, prev_obs, state,
                            state_mean, state_std, device)
    latent_goal_errors.append(
        float(torch.linalg.vector_norm(z_final - z_goal)))

    states  = np.asarray(states)
    actions = np.asarray(actions)
    goal    = np.asarray(goal_state, dtype=np.float64)
    errors  = np.linalg.norm(states - goal[None], axis=1)

    stable_mask = errors[1:] < settling_threshold
    stabilization_macro_steps     = int(np.sum(stable_mask))
    stabilization_primitive_steps = int(stabilization_macro_steps * frame_skip)

    hold        = min(max(int(success_hold_steps), 1), len(errors))
    success     = bool(not terminated and errors[-1] < stabilization_threshold)
    held_stable = bool(not terminated and
                       np.all(errors[-hold:] < stabilization_threshold))

    return {
        'initial_state': np.asarray(initial_state).tolist(),
        'goal_state':    goal.tolist(),
        'success':       success,
        'held_stable':   held_stable,
        'terminated':    terminated,
        'replans':       replans,
        'macro_steps':   int(len(actions)),
        'primitive_steps': int(len(actions) * frame_skip),
        'final_error':   float(errors[-1]),
        'max_error':     float(errors.max()),
        'fraction_stable': (float(np.mean(stable_mask))
                            if len(stable_mask) else 0.),
        'stabilization_macro_steps':               stabilization_macro_steps,
        'stabilization_primitive_steps_equivalent': stabilization_primitive_steps,
        'action_rms':    float(np.sqrt(np.mean(actions ** 2)))
                         if len(actions) else 0.,
        'latent_goal_errors_at_replans': latent_goal_errors,
        'actions':  actions.tolist(),
        'states':   states.tolist(),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Gradient-based latent-goal planning for CartPole SMWM.')
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--cfgs',  nargs='+', required=True)
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--primitive-budget', type=int, default=300,
                   help='Total primitive env steps per trial')
    p.add_argument('--planning-horizon', type=int, default=10,
                   help='Action sequence length H to optimise')
    p.add_argument('--executed-steps', type=int, default=1,
                   help='Steps to execute before replanning (MPC)')
    p.add_argument('--gd-steps', type=int, default=50,
                   help='Gradient-descent iterations per replan')
    p.add_argument('--lr', type=float, default=0.05,
                   help='Adam learning rate for action optimisation')
    p.add_argument('--action-noise', type=float, default=0.01,
                   help='Gaussian noise injected after each GD step')
    p.add_argument('--objective', choices=['last', 'all'], default='last',
                   help='"last": terminal cost only; "all": mean over horizon')
    p.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd', 'momentum'],
                   default='adam')
    p.add_argument('--warm-start', action='store_true', default=True,
                   help='Shift previous plan to initialise next replan')
    p.add_argument('--no-warm-start', dest='warm_start', action='store_false')
    p.add_argument('--success-threshold', type=float, default=None,
                   help='Physical-state norm for success (default: from config)')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    if len(args.cfgs) == 1:
        args.cfgs *= len(args.ckpts)
    if len(args.cfgs) != len(args.ckpts):
        raise ValueError('--cfgs must have length 1 or match --ckpts')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    with open(args.cfgs[0]) as f:
        ref_cfg = yaml.safe_load(f)
    ctrl = ref_cfg.get('control', {})
    success_threshold = (float(args.success_threshold)
                         if args.success_threshold is not None
                         else float(ctrl.get('stabilization_threshold', 0.1)))
    init_scale        = float(ctrl.get('init_scale', 0.05))
    success_hold_steps = int(ctrl.get('success_hold_steps', 10))

    rng = np.random.RandomState(args.seed)
    initial_states = [rng.uniform(-init_scale, init_scale, 4).astype(np.float32)
                      for _ in range(args.trials)]

    all_results = []

    for ckpt, cfg_path in zip(args.ckpts, args.cfgs):
        label = checkpoint_label(ckpt)
        print(f'\n[{label}] loading {ckpt}')
        cfg, checkpoint, model, state_mean, state_std, action_scale = \
            load_model(ckpt, cfg_path, device)

        if 'environment' not in cfg:
            model_cfg = dict(checkpoint['model_config'])
            cfg['environment'] = {
                'frame_skip':   int(model_cfg.get('frame_skip', 5)),
                'image_size':   int(model_cfg.get('image_size', 128)),
                'action_range': [-10, 10],
                'mass_cart': 1.0, 'mass_pole': 0.1,
                'pole_length': 0.5, 'gravity': 9.8, 'dt': 0.02,
            }

        bounds      = cfg['environment'].get('action_range', [-10., 10.])
        frame_skip  = int(cfg['environment'].get('frame_skip', 1))
        macro_budget = int(np.ceil(args.primitive_budget / frame_skip))

        planner = CartPoleGDPlanner(
            model=model,
            action_scale=action_scale,
            horizon=args.planning_horizon,
            executed_steps=args.executed_steps,
            gd_steps=args.gd_steps,
            lr=args.lr,
            action_noise=args.action_noise,
            action_lb=bounds[0],
            action_ub=bounds[1],
            device=device,
            objective=args.objective,
            warm_start=args.warm_start,
            optimizer_type=args.optimizer,
        )
        print(f'[{label}] GD planner  H={planner.horizon}  K={planner.executed_steps}  '
              f'gd_steps={planner.gd_steps}  lr={planner.lr}  '
              f'noise={planner.action_noise}  objective={planner.objective}  '
              f'frame_skip={frame_skip}  macro_budget={macro_budget}')

        # Encode goal (upright equilibrium)
        goal_state  = np.zeros(4, dtype=np.float32)
        goal_env    = make_env(cfg, args.seed + 999)
        goal_obs, _, _ = goal_env.reset_to_state(goal_state)
        z_goal = encode_online(model, goal_obs, goal_obs, goal_state,
                               state_mean, state_std, device)
        goal_env.close()
        print(f'[{label}] z_goal encoded  dim={z_goal.shape[-1]}')

        rows      = []
        n_success = 0
        for trial, x0 in enumerate(initial_states):
            torch.manual_seed(args.seed + trial)
            env = make_env(cfg, args.seed + trial)
            row = evaluate_trial_gd(
                env, planner, model, x0, goal_state, z_goal,
                state_mean, state_std, args.primitive_budget,
                success_threshold, success_threshold,
                success_hold_steps, device)
            env.close()
            n_success += int(row['success'])
            rows.append(row)
            print(f'[{label}] trial {trial:03d}  '
                  f'success={row["success"]}  held={row["held_stable"]}  '
                  f'term={row["terminated"]}  final={row["final_error"]:.5f}  '
                  f'zgoal={row["latent_goal_errors_at_replans"][-1]:.4f}')

        summary = summarize(rows)
        print(f'[{label}] SUMMARY  '
              f'success={summary["success_rate"]:.1%}  '
              f'held={summary["held_stable_rate"]:.1%}  '
              f'term={summary["termination_rate"]:.1%}  '
              f'final={summary["mean_final_error"]:.5f}  '
              f'Lbar={summary["mean_stabilization_macro_steps"]:.2f}  '
              f'invL2={summary["inverse_squared_mean_stabilization_macro_length"]}')

        all_results.append({
            'label':      label,
            'checkpoint': ckpt,
            'summary':    summary,
            'trials':     rows,
        })

    # Comparison table
    print('\n' + '=' * 72)
    print(f'{"Model":<30} {"Success":>8} {"Held":>8} '
          f'{"Term":>6} {"FinalErr":>10} {"Lbar":>8}')
    print('-' * 72)
    for r in all_results:
        s = r['summary']
        print(f'{r["label"]:<30} '
              f'{s["success_rate"]:>8.1%} '
              f'{s["held_stable_rate"]:>8.1%} '
              f'{s["termination_rate"]:>6.1%} '
              f'{s["mean_final_error"]:>10.5f} '
              f'{s["mean_stabilization_macro_steps"]:>8.2f}')
    print('=' * 72)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planner':           'gradient_descent',
            'planning_horizon':  args.planning_horizon,
            'executed_steps':    args.executed_steps,
            'gd_steps':          args.gd_steps,
            'lr':                args.lr,
            'action_noise':      args.action_noise,
            'objective':         args.objective,
            'optimizer':         args.optimizer,
            'warm_start':        args.warm_start,
            'primitive_budget':  args.primitive_budget,
            'success_threshold': success_threshold,
            'seed':              args.seed,
            'n_trials':          args.trials,
        },
        'models': all_results,
    }, indent=2))
    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()
