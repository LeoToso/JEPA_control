"""Closed-loop MPC rollout on the real cartpole environment."""
from __future__ import annotations
import warnings
from typing import Optional, Dict, List
import numpy as np
import torch


def rollout_latent_mpc(encoder, mpc, env, x0, T=200,
                       stabilization_threshold=0.1, settling_threshold=0.05,
                       device=None, z_star=None,
                       save_frames=False, save_all_obs=False,
                       frame_stack=1):
    if device is None:
        try:    device = next(encoder.parameters()).device
        except: device = torch.device('cpu')
    encoder.eval()

    d_mpc   = mpc.A.shape[0]
    d_lat   = d_mpc // 2 if d_mpc % 2 == 0 else d_mpc
    if z_star is None:
        z_star = np.zeros(d_lat)
    x_star       = np.zeros(4)
    settling_time= T
    use_aug_state= False

    obs, state, _ = env.reset_to_state(x0)
    states: List     = []
    latent_states: List = []
    actions: List    = []
    frames: List     = []
    all_obs: List    = []
    pending: List    = []
    z_t = np.zeros(d_lat)
    z_prev = None
    prev_obs_t = None   # for frame stacking: previous step's observation tensor

    t = 0
    while t < T:
        states.append(state.copy())
        if save_all_obs:
            all_obs.append(obs.copy())
        obs_t = torch.from_numpy(obs).float().permute(2, 0, 1)[None].to(device) / 255.0
        if frame_stack > 1:
            enc_input = torch.cat(
                [prev_obs_t if prev_obs_t is not None else obs_t, obs_t], dim=1)
        else:
            enc_input = obs_t
        with torch.no_grad():
            z_t = encoder(enc_input).cpu().numpy()[0]
        latent_states.append(z_t.copy())

        if t == 0:
            use_aug_state = (d_mpc == 2 * len(z_t))

        if len(pending) == 0:
            if use_aug_state:
                z_p   = z_t if z_prev is None else z_prev
                s_t   = np.concatenate([z_t, z_p])
                s_star= np.concatenate([z_star, z_star])
                chunk, pred_zs = mpc.plan(s_t, s_star)
            else:
                chunk, pred_zs = mpc.plan(z_t, z_star)
            pending = list(chunk)
            if save_frames:
                frames.append({'obs': obs.copy(), 'pred_zs': pred_zs, 't': t})

        z_prev = z_t
        prev_obs_t = obs_t
        u_vec  = pending.pop(0)
        u_s    = float(np.clip(u_vec[0], mpc.action_lb, mpc.action_ub))
        actions.append(np.array([u_s]))
        obs, state, _, done, _ = env.step(u_s)
        if np.linalg.norm(state - x_star) < settling_threshold and settling_time == T:
            settling_time = t
        t += 1
        if done:
            for _ in range(T - t):
                states.append(state.copy())
                latent_states.append(z_t.copy())
                actions.append(np.array([0.0]))
            break

    states_arr  = np.array(states)
    lat_arr     = np.array(latent_states)
    actions_arr = np.array(actions)
    n_real      = t
    final_error = float(np.linalg.norm(states_arr[n_real - 1] - x_star))
    frac_stable = float(np.mean(np.linalg.norm(states_arr[:n_real] - x_star, axis=1)
                                < settling_threshold))
    return {
        'states':       states_arr,
        'latent_states':lat_arr,
        'actions':      actions_arr,
        'frames':       frames,
        'all_obs':      all_obs,
        'final_state_error': final_error,
        'stabilized':   bool(final_error < stabilization_threshold),
        'settling_time':settling_time,
        'done_at':      n_real,
        'fraction_stable': frac_stable,
    }


def evaluate_stabilization_mpc(encoder, mpc, env, n_trials=100, T=200,
                                init_scale=0.05, stabilization_threshold=0.1,
                                settling_threshold=0.05, seed=0,
                                device=None, z_star=None, vis_trial=0,
                                frame_stack=1):
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
                device=device, z_star=z_star,
                save_frames=save, save_all_obs=save,
                frame_stack=frame_stack,
            )
            if save:
                vis_result = result
            successes.append(result['stabilized'])
            ep_lengths.append(result['done_at'])
            frac_stables.append(result['fraction_stable'])
            xs, us = result['states'], result['actions']
            cost = sum(float(xs[k] @ Q_phys @ xs[k] + us[k] @ R_phys @ us[k])
                       for k in range(result['done_at']))
            costs.append(cost)
        except Exception as exc:
            warnings.warn(f'MPC trial {trial} failed: {exc}')
            successes.append(False)
            ep_lengths.append(0)
            frac_stables.append(0.0)

    return {
        'success_rate':       float(np.mean(successes)),
        'mean_episode_length':float(np.mean(ep_lengths)),
        'mean_fraction_stable':float(np.mean(frac_stables)),
        'mean_cost':          float(np.mean(costs)) if costs else float('nan'),
        'n_trials':           n_trials,
        'vis_result':         vis_result,
    }
