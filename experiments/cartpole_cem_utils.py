"""CartPole CEM planner utilities for compare_paper_cem_smwm.py.

Mirrors experiments/evaluate_cartpole_paper_cem_learned.py from the
agent/cartpole-sensorimotor-world-model branch, extended to support
checkpoints that do not use proprioceptive input (use_proprio=False).

Provides:
  PaperLearnedCEM  — CEM planner with 1-D action space and latent goal cost
  encode_online    — encode (obs, prev_obs, state) without a bundle dict
  evaluate_trial   — run one closed-loop CEM episode
  summarize        — aggregate a list of trial dicts into summary statistics
"""
from __future__ import annotations

import numpy as np
import torch


# ── observation → tensor ──────────────────────────────────────────────────────

def _obs_t(obs, device):
    return (torch.from_numpy(np.asarray(obs)).float()
            .permute(2, 0, 1).unsqueeze(0).to(device) / 255.)


def _build_obs_input(model, obs, prev_obs, device):
    """Return the observation tensor the encoder expects.

    frame_stack > 1: obs may be a list of frame_stack HxWxC arrays (oldest
                     first).  A bare array is repeated frame_stack times
                     (correct for goal / init encoding).
    use_frame_diff:  single obs + prev_obs handled downstream in model.encode.
    """
    fs = int(model.frame_stack)
    if fs > 1:
        frames = list(obs) if isinstance(obs, (list, tuple)) else [obs] * fs
        return torch.cat([_obs_t(f, device) for f in frames], dim=1)
    return _obs_t(obs, device)


def make_frame_buffer(model, initial_obs):
    """Return a list of frame_stack copies of initial_obs (oldest first)."""
    return [initial_obs] * int(model.frame_stack)


def push_frame(frame_buffer, new_obs):
    """Append new_obs and drop the oldest frame in-place."""
    frame_buffer.pop(0)
    frame_buffer.append(new_obs)


# ── encoding ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_online(model, obs, prev_obs, state, state_mean, state_std, device):
    """Encode a single (obs, prev_obs, state) triple without a bundle dict.

    obs may be a list of frame_stack arrays when model.frame_stack > 1.
    Works for both use_proprio=True and use_proprio=False checkpoints.
    """
    proprio = None
    if model.use_proprio:
        norm = (np.asarray(state, dtype=np.float32) - state_mean) / state_std
        if model.proprio_indices is not None:
            norm = norm[model.proprio_indices]
        proprio = torch.as_tensor(norm, dtype=torch.float32,
                                  device=device).reshape(1, -1)
    obs_in  = _build_obs_input(model, obs, prev_obs, device)
    prev_in = (_obs_t(prev_obs, device)
               if model.use_frame_diff and model.frame_stack == 1 else None)
    return model.encode(obs_in, prev_in, proprio)


# ── CEM planner ───────────────────────────────────────────────────────────────

class PaperLearnedCEM:
    """CEM with terminal latent goal distance, matching Eq. (8).

    Actions are kept as 1-D tensors (pop, horizon) — same shape as the
    original evaluate_cartpole_paper_cem_learned.py on the reference branch.
    """

    def __init__(self, model, action_scale, horizon, executed_steps,
                 population, elites, iterations, initial_variance_scale,
                 action_lb, action_ub, device):
        self.model = model
        self.action_scale = float(action_scale)
        self.horizon = int(horizon)
        self.executed_steps = min(int(executed_steps), self.horizon)
        self.population = int(population)
        self.elites = min(int(elites), self.population)
        self.iterations = int(iterations)
        self.initial_variance_scale = float(initial_variance_scale)
        self.action_lb = float(action_lb)
        self.action_ub = float(action_ub)
        self.device = device

    @torch.no_grad()
    def _terminal_goal_cost(self, z0, z_goal, actions):
        """
        actions : (pop, horizon)  raw action values
        Returns : (pop,) squared latent distance at horizon
        """
        z = z0.expand(actions.shape[0], -1)
        for t in range(self.horizon):
            a_raw = (actions[:, t] / self.action_scale).reshape(-1, 1)  # (pop, 1)
            a_ctx = self.model.expand_action(a_raw).unsqueeze(1)        # (pop, 1, ctx)
            z = self.model.predict(z[:, None], a_ctx)[:, 0]
        return (z - z_goal).square().sum(-1)

    @torch.no_grad()
    def plan_sequence(self, z0, z_goal):
        """Return (executed_steps,) numpy array of raw actions to execute."""
        mean = torch.zeros(self.horizon, device=self.device)
        variance = torch.full_like(mean, self.initial_variance_scale)
        for _ in range(self.iterations):
            actions = (mean + variance.sqrt() *
                       torch.randn(self.population, self.horizon,
                                   device=self.device))
            actions.clamp_(self.action_lb, self.action_ub)
            costs = self._terminal_goal_cost(z0, z_goal, actions)
            elite = actions[torch.argsort(costs)[:self.elites]]
            mean = elite.mean(0)
            variance = elite.var(0, unbiased=False).clamp(min=1e-6)
        return mean[:self.executed_steps].clamp(
            self.action_lb, self.action_ub).cpu().numpy()


# ── trial evaluation ──────────────────────────────────────────────────────────

def evaluate_trial(env, planner, model, initial_state, goal_state, z_goal,
                   state_mean, state_std, primitive_budget,
                   stabilization_threshold, settling_threshold,
                   success_hold_steps, device):
    """Run one closed-loop CEM episode.

    primitive_budget is divided by env.frame_skip to obtain the macro-step
    budget, matching the original implementation on the reference branch.
    """
    obs, state, _ = env.reset_to_state(initial_state)
    frame_buf = make_frame_buffer(model, obs)
    frame_skip = int(env.frame_skip)
    macro_budget = int(np.ceil(primitive_budget / frame_skip))

    states  = [state.copy()]
    actions = []
    latent_goal_errors = []
    terminated = False
    replans = 0

    while len(actions) < macro_budget and not terminated:
        z = encode_online(model, frame_buf, obs, state,
                          state_mean, state_std, device)
        latent_goal_errors.append(float(torch.linalg.vector_norm(z - z_goal)))
        sequence = planner.plan_sequence(z, z_goal)
        replans += 1
        remaining = macro_budget - len(actions)
        for action in sequence[:remaining]:
            obs, state, _, done, _ = env.step(float(action))
            push_frame(frame_buf, obs)
            actions.append(float(action))
            states.append(state.copy())
            if done:
                terminated = True
                break

    # Record terminal latent distance.
    z_final = encode_online(model, frame_buf, obs, state,
                            state_mean, state_std, device)
    latent_goal_errors.append(float(torch.linalg.vector_norm(z_final - z_goal)))

    states  = np.asarray(states)
    actions = np.asarray(actions)
    errors  = np.linalg.norm(states - goal_state[None], axis=1)

    # Exclude the initial (pre-action) state from the stability count.
    stable_mask = errors[1:] < settling_threshold
    stabilization_macro_steps = int(np.sum(stable_mask))
    stabilization_primitive_steps = int(stabilization_macro_steps * frame_skip)

    hold = min(max(int(success_hold_steps), 1), len(errors))
    success     = bool(not terminated and errors[-1] < stabilization_threshold)
    held_stable = bool(not terminated and
                       np.all(errors[-hold:] < stabilization_threshold))

    return {
        'initial_state': np.asarray(initial_state).tolist(),
        'goal_state':    np.asarray(goal_state).tolist(),
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
        'stabilization_macro_steps': stabilization_macro_steps,
        'stabilization_primitive_steps_equivalent': stabilization_primitive_steps,
        'action_rms':    float(np.sqrt(np.mean(actions ** 2)))
                         if len(actions) else 0.,
        'latent_goal_errors_at_replans': latent_goal_errors,
        'actions':       actions.tolist(),
        'states':        states.tolist(),
    }


# ── summary ───────────────────────────────────────────────────────────────────

def summarize(rows):
    """Aggregate a list of evaluate_trial dicts into summary statistics."""
    mean_macro     = float(np.mean(
        [r['stabilization_macro_steps'] for r in rows]))
    mean_primitive = float(np.mean(
        [r['stabilization_primitive_steps_equivalent'] for r in rows]))
    return {
        'success_rate':         float(np.mean([r['success']     for r in rows])),
        'held_stable_rate':     float(np.mean([r['held_stable'] for r in rows])),
        'termination_rate':     float(np.mean([r['terminated']  for r in rows])),
        'mean_final_error':     float(np.mean([r['final_error'] for r in rows])),
        'mean_fraction_stable': float(np.mean([r['fraction_stable'] for r in rows])),
        'mean_stabilization_macro_steps':               mean_macro,
        'mean_stabilization_primitive_steps_equivalent': mean_primitive,
        'inverse_squared_mean_stabilization_macro_length': (
            float(1. / mean_macro ** 2) if mean_macro > 0. else None),
        'inverse_squared_mean_stabilization_primitive_length': (
            float(1. / mean_primitive ** 2) if mean_primitive > 0. else None),
        'mean_action_rms': float(np.mean([r['action_rms'] for r in rows])),
        'mean_final_latent_goal_error': float(np.mean(
            [r['latent_goal_errors_at_replans'][-1] for r in rows])),
        'mean_replans': float(np.mean([r['replans'] for r in rows])),
        'n_trials': len(rows),
    }

