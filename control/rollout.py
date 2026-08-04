"""Closed-loop MPC rollout on the real cartpole environment."""
from __future__ import annotations
import warnings
from typing import Optional, Dict, List
import numpy as np
import torch


def rollout_latent_mpc(encoder, mpc, env, x0, T=200,
                       stabilization_threshold=0.1, settling_threshold=0.05,
                       success_hold_steps=10,
                       device=None, z_star=None,
                       save_frames=False, save_all_obs=False,
                       frame_stack=1, use_frame_diff=False):
    if device is None:
        try:    device = next(encoder.parameters()).device
        except: device = torch.device('cpu')
    encoder.eval()

    d_mpc   = mpc.A.shape[0]
    x_star       = np.zeros(4)
    settling_time= T
    terminated = False
    # W_aug is resolved after the first encoder call once d_lat is known.
    W_aug   = None   # window size for augmented state (1 = no augmentation)
    z_history: List = []   # ring buffer, newest last; length <= W_aug

    if z_star is None:
        z_star = np.zeros(d_mpc)

    # Reset warm-start state so each episode begins with a fresh prior
    if hasattr(mpc, 'reset'):
        mpc.reset()

    obs, state, _ = env.reset_to_state(x0)
    states: List     = []
    latent_states: List = []
    actions: List    = []
    frames: List     = []
    all_obs: List    = []
    pending: List    = []
    z_t = None
    prev_obs_t = None   # for frame stacking: previous step's observation tensor

    t = 0
    while t < T:
        states.append(state.copy())
        if save_all_obs:
            all_obs.append(obs.copy())
        obs_t = torch.from_numpy(obs).float().permute(2, 0, 1)[None].to(device) / 255.0
        prev = prev_obs_t if prev_obs_t is not None else obs_t
        if use_frame_diff:
            enc_input = torch.cat([prev, obs_t, obs_t - prev], dim=1)
        elif frame_stack > 1:
            enc_input = torch.cat([prev, obs_t], dim=1)
        else:
            enc_input = obs_t
        with torch.no_grad():
            z_t = encoder(enc_input).cpu().numpy()[0]
        latent_states.append(z_t.copy())

        # Resolve augmentation window on first step
        if W_aug is None:
            d_lat = len(z_t)
            if d_mpc % d_lat == 0:
                W_aug = d_mpc // d_lat   # e.g. 24//8=3
            else:
                W_aug = 1
            # Initialise history with copies of first z
            z_history = [z_t.copy()] * W_aug

        z_history.append(z_t.copy())

        if len(pending) == 0:
            if W_aug > 1:
                # Augmented state: [z_t, z_{t-1}, ..., z_{t-W+1}], newest first
                hist = z_history[-W_aug:]
                s_t  = np.concatenate(hist[::-1])   # reverse so newest is first
                chunk, pred_zs = mpc.plan(s_t, z_star)
            else:
                chunk, pred_zs = mpc.plan(z_t, z_star)
            pending = list(chunk)
            if save_frames:
                frames.append({'obs': obs.copy(), 'pred_zs': pred_zs, 't': t})

        prev_obs_t = obs_t
        u_vec  = pending.pop(0)
        u_s    = float(np.clip(u_vec[0], mpc.action_lb, mpc.action_ub))
        actions.append(np.array([u_s]))
        obs, state, _, done, _ = env.step(u_s)
        t += 1
        if done:
            terminated = True
            for _ in range(T - t):
                states.append(state.copy())
                latent_states.append(z_t.copy())
                actions.append(np.array([0.0]))
            break

    states_arr  = np.array(states)
    lat_arr     = np.array(latent_states)
    actions_arr = np.array(actions)
    n_real      = t
    # states_arr[k] is before action k; state is after the final action.
    # Include that true terminal state in terminal and hold metrics.
    errors = np.linalg.norm(states_arr[:n_real] - x_star, axis=1)
    metric_errors = np.concatenate(
        [errors, [float(np.linalg.norm(state - x_star))]])
    final_error = float(metric_errors[-1])
    frac_stable = float(np.mean(metric_errors < settling_threshold))
    # Success must be sustained, not a single lucky final threshold crossing.
    hold = min(max(int(success_hold_steps), 1), len(metric_errors))
    stabilized = bool((not terminated)
                      and np.all(metric_errors[-hold:] < stabilization_threshold))
    # Settling time is the first sample after which the trajectory stays settled.
    settled = metric_errors < settling_threshold
    suffix_all = np.logical_and.accumulate(settled[::-1])[::-1]
    settled_idx = np.flatnonzero(suffix_all)
    if settled_idx.size:
        settling_time = int(settled_idx[0])
    return {
        'states':       states_arr,
        'latent_states':lat_arr,
        'actions':      actions_arr,
        'frames':       frames,
        'all_obs':      all_obs,
        'final_state_error': final_error,
        'stabilized':   stabilized,
        'settling_time':settling_time,
        'done_at':      n_real,
        'terminated':   terminated,
        'fraction_stable': frac_stable,
    }


def evaluate_stabilization_mpc(encoder, mpc, env, n_trials=100, T=200,
                                init_scale=0.05, stabilization_threshold=0.1,
                                settling_threshold=0.05, success_hold_steps=10,
                                failure_penalty=1.0e4, seed=0,
                                device=None, z_star=None, vis_trial=0,
                                frame_stack=1, use_frame_diff=False):
    rng = np.random.RandomState(seed)
    successes, ep_lengths, frac_stables, costs = [], [], [], []
    vis_result = None
    Q_phys = np.diag([1.0, 1.0, 10.0, 1.0])
    R_phys = 0.01 * np.eye(1)

    for trial in range(n_trials):
        x0   = rng.uniform(-init_scale, init_scale, 4).astype(np.float32)
        save = (trial == vis_trial)
        try:
            result = rollout_latent_mpc(
                encoder=encoder, mpc=mpc, env=env, x0=x0, T=T,
                stabilization_threshold=stabilization_threshold,
                settling_threshold=settling_threshold,
                success_hold_steps=success_hold_steps,
                device=device, z_star=z_star,
                save_frames=save, save_all_obs=save,
                frame_stack=frame_stack, use_frame_diff=use_frame_diff,
            )
            if save:
                vis_result = result
            successes.append(result['stabilized'])
            ep_lengths.append(result['done_at'])
            frac_stables.append(result['fraction_stable'])
            xs, us = result['states'], result['actions']
            # Score the full requested horizon. Padding a terminated trajectory
            # avoids the old bias where crashing early produced a deceptively low cost.
            cost = sum(float(xs[k] @ Q_phys @ xs[k] + us[k] @ R_phys @ us[k])
                       for k in range(min(T, len(xs), len(us))))
            if result['terminated']:
                cost += float(failure_penalty)
            costs.append(cost)
        except Exception as exc:
            warnings.warn(f'MPC trial {trial} failed: {exc}')
            successes.append(False)
            ep_lengths.append(0)
            frac_stables.append(0.0)
            costs.append(float(failure_penalty))

    ep_arr = np.array(ep_lengths, dtype=float)
    inv_sq = np.mean(1.0 / np.maximum(ep_arr, 1) ** 2)
    return {
        'success_rate':            float(np.mean(successes)),
        'mean_episode_length':     float(np.mean(ep_arr)),
        'mean_inv_sq_ep_length':   float(inv_sq),
        'mean_fraction_stable':    float(np.mean(frac_stables)),
        'mean_cost':               float(np.mean(costs)) if costs else float('nan'),
        'n_trials':                n_trials,
        'vis_result':              vis_result,
    }
