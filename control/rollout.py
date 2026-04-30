"""True-system closed-loop rollout: latent LQR and latent MPC variants."""
from __future__ import annotations
import warnings
from typing import Optional, Dict, Any, List
import numpy as np
import torch

def rollout_latent_lqr(encoder,A_hat,B_hat,K_hat,env,x0,T=200,
                       stabilization_threshold=0.1,settling_threshold=0.05,
                       device=None,z_star=None):
    """Roll out u_t = -K * aug_state on the real environment.

    If K_hat has shape (d_u, 2*d) it is treated as an augmented-state gain:
      aug_state = [z_t - z_star; z_{t-1} - z_star]
    giving the controller implicit velocity information (z_t - z_{t-1}).
    Otherwise the standard u_t = -K(z_t - z_star) is used.
    """
    if device is None:
        try:
            device=next(encoder.parameters()).device
        except StopIteration:
            device=torch.device('cpu')
    encoder.eval()
    obs,state,_=env.reset_to_state(x0)
    states,latent_states,actions,latent_actions=[],[],[],[]
    d=A_hat.shape[0]
    d_u=B_hat.shape[1]
    use_aug=(K_hat.shape[1]==2*d)
    if z_star is None:
        z_star=np.zeros(d)
    settling_time=T
    x_star=np.zeros(4)
    z_prev=None
    for t in range(T):
        states.append(state.copy())
        obs_t=torch.from_numpy(obs).float().permute(2,0,1)[None].to(device)/255.0
        with torch.no_grad():
            z_t=encoder(obs_t).cpu().numpy()[0]
        latent_states.append(z_t.copy())
        if use_aug:
            if z_prev is None:
                z_prev=z_t
            aug=np.concatenate([z_t-z_star,z_prev-z_star])
            a_t=-K_hat@aug
            z_prev=z_t
        else:
            a_t=-K_hat@(z_t-z_star)
        latent_actions.append(a_t.copy())
        try:
            u_scalar=float(np.clip(a_t[0],-10.0,10.0))
        except Exception:
            u_scalar=0.0
        actions.append(np.array([u_scalar]))
        obs,state,_,done,_=env.step(u_scalar)
        if np.linalg.norm(state-x_star)<settling_threshold and settling_time==T:
            settling_time=t
        if done:
            for _ in range(T-t-1):
                states.append(state.copy())
                latent_states.append(z_t.copy())
                actions.append(np.array([0.0]))
                latent_actions.append(np.zeros(d_u))
            break
    states=np.array(states)
    latent_states=np.array(latent_states)
    actions=np.array(actions)
    latent_actions=np.array(latent_actions)
    final_error=float(np.linalg.norm(states[-1]-x_star))
    return {'states':states,'latent_states':latent_states,'actions':actions,
            'latent_actions':latent_actions,'final_state_error':final_error,
            'stabilized':bool(final_error<stabilization_threshold),
            'settling_time':settling_time}


def evaluate_stabilization(encoder,A_hat,B_hat,K_hat,env,n_trials=100,T=200,
                           init_scale=0.2,Q_lqr=None,R_lqr=None,
                           stabilization_threshold=0.1,settling_threshold=0.05,
                           seed=0,device=None,z_star=None):
    rng=np.random.RandomState(seed)
    d=A_hat.shape[0]
    d_u=B_hat.shape[1]
    Q_lqr_default=np.diag([1.0,1.0,10.0,1.0]) if Q_lqr is None else Q_lqr
    R_lqr_default=0.01*np.eye(1) if R_lqr is None else R_lqr
    successes,settling_times,final_errors,true_costs,latent_costs,all_results=[],[],[],[],[],[]
    for trial in range(n_trials):
        x0=rng.uniform(-init_scale,init_scale,size=4).astype(np.float32)
        try:
            result=rollout_latent_lqr(
                encoder=encoder,A_hat=A_hat,B_hat=B_hat,K_hat=K_hat,
                env=env,x0=x0,T=T,
                stabilization_threshold=stabilization_threshold,
                settling_threshold=settling_threshold,
                device=device,z_star=z_star)
            successes.append(result['stabilized'])
            settling_times.append(result['settling_time'])
            final_errors.append(result['final_state_error'])
            xs=result['states']
            us=result['actions']
            true_cost=sum(float(xs[t]@Q_lqr_default@xs[t]+us[t]@R_lqr_default@us[t])
                          for t in range(len(xs)))
            true_costs.append(true_cost)
            zs=result['latent_states']
            as_=result['latent_actions']
            Q_lat=np.eye(d)
            R_lat=0.01*np.eye(d_u)
            latent_cost=sum(float(zs[t]@Q_lat@zs[t]+as_[t]@R_lat@as_[t])
                            for t in range(len(zs)))
            latent_costs.append(latent_cost)
            all_results.append(result)
        except Exception as exc:
            warnings.warn(f'Trial {trial} failed: {exc}')
            successes.append(False)
            settling_times.append(T)
            final_errors.append(float('nan'))
    return {'success_rate':float(np.mean(successes)),
            'mean_settling_time':float(np.mean(settling_times)),
            'mean_final_error':float(np.nanmean(final_errors)),
            'true_lqr_cost':float(np.mean(true_costs)) if true_costs else float('nan'),
            'latent_lqr_cost':float(np.mean(latent_costs)) if latent_costs else float('nan'),
            'n_trials':n_trials,'all_results':all_results}


# ─────────────────────────────────────────────────────────────────────────────
# MPC rollout
# ─────────────────────────────────────────────────────────────────────────────

def rollout_latent_mpc(
    encoder,
    mpc,
    env,
    x0: np.ndarray,
    T: int = 200,
    stabilization_threshold: float = 0.1,
    settling_threshold: float = 0.05,
    device=None,
    z_star: Optional[np.ndarray] = None,
    save_frames: bool = False,
) -> Dict:
    """Roll out a LatentMPC controller on the real environment.

    At each re-planning event (every mpc.chunk_size steps) the controller
    encodes the current observation, plans H steps ahead using the learned
    linear dynamics, and deploys the first chunk_size actions before
    re-planning.

    Parameters
    ----------
    save_frames : if True, stores {'obs', 'pred_states', 't'} for every
                  re-planning event (used for visualization).
    """
    if device is None:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
    encoder.eval()

    d_mpc = mpc.A.shape[0]
    # Detect augmented-state MPC: A is (2d × 2d) built from 2nd-order dynamics.
    # In that case we maintain z_prev and pass [z_t, z_{t-1}] as the MPC state.
    d_lat = d_mpc // 2 if d_mpc % 2 == 0 else d_mpc   # actual latent dim
    use_aug_state = False  # determined after first encode
    if z_star is None:
        z_star = np.zeros(d_lat)
    x_star = np.zeros(4)
    settling_time = T

    obs, state, _ = env.reset_to_state(x0)
    states: List[np.ndarray] = []
    latent_states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    frames: List[Dict] = []

    pending: List[np.ndarray] = []
    z_t = np.zeros(d_lat)
    z_prev: Optional[np.ndarray] = None

    t = 0
    while t < T:
        states.append(state.copy())

        obs_t = torch.from_numpy(obs).float().permute(2, 0, 1)[None].to(device) / 255.0
        with torch.no_grad():
            z_t = encoder(obs_t).cpu().numpy()[0]
        latent_states.append(z_t.copy())

        # Determine if MPC expects augmented state [z_t, z_{t-1}] on first step.
        if t == 0:
            use_aug_state = (d_mpc == 2 * len(z_t))

        # Re-plan when the action buffer is exhausted.
        if len(pending) == 0:
            if use_aug_state:
                z_p = z_t if z_prev is None else z_prev
                s_t = np.concatenate([z_t, z_p])
                s_star = np.concatenate([z_star, z_star])
                chunk, pred_zs = mpc.plan(s_t, s_star)
            else:
                chunk, pred_zs = mpc.plan(z_t, z_star)
            pending = list(chunk)
            if save_frames:
                frames.append({'obs': obs.copy(), 'pred_states': pred_zs, 't': t})

        z_prev = z_t

        u_vec = pending.pop(0)
        u_scalar = float(np.clip(u_vec[0], mpc.action_lb, mpc.action_ub))
        actions.append(np.array([u_scalar]))

        obs, state, _, done, _ = env.step(u_scalar)
        if np.linalg.norm(state - x_star) < settling_threshold and settling_time == T:
            settling_time = t

        t += 1
        if done:
            for _ in range(T - t):
                states.append(state.copy())
                latent_states.append(z_t.copy())
                actions.append(np.array([0.0]))
            break

    states_arr = np.array(states)
    latent_states_arr = np.array(latent_states)
    actions_arr = np.array(actions)

    n_real = t  # number of real (non-padded) timesteps
    final_error = float(np.linalg.norm(states_arr[n_real - 1] - x_star))

    # Fraction of real timesteps where ||state|| < settling_threshold.
    real_errors = np.linalg.norm(states_arr[:n_real] - x_star, axis=1)
    fraction_stable = float(np.mean(real_errors < settling_threshold))

    return {
        'states': states_arr,
        'latent_states': latent_states_arr,
        'actions': actions_arr,
        'frames': frames,
        'final_state_error': final_error,
        'stabilized': bool(final_error < stabilization_threshold),
        'settling_time': settling_time,
        'done_at': n_real,           # episode length before done / T
        'fraction_stable': fraction_stable,  # fraction of time near equilibrium
    }


def evaluate_stabilization_mpc(
    encoder,
    mpc,
    env,
    n_trials: int = 100,
    T: int = 200,
    init_scale: float = 0.05,
    stabilization_threshold: float = 0.1,
    settling_threshold: float = 0.05,
    seed: int = 0,
    device=None,
    z_star: Optional[np.ndarray] = None,
    vis_trial: int = 0,
) -> Dict:
    """Evaluate a LatentMPC controller over multiple random initial conditions.

    Parameters
    ----------
    vis_trial : index of the trial for which frames are saved (for visualization).
                Set to -1 to disable frame saving entirely.
    """
    rng = np.random.RandomState(seed)
    d = mpc.A.shape[0]
    d_u = mpc.B.shape[1]
    Q_phys = np.diag([1.0, 1.0, 10.0, 1.0])
    R_phys = 0.01 * np.eye(1)

    successes, settling_times, final_errors, true_costs = [], [], [], []
    done_ats, frac_stables = [], []
    vis_result = None

    for trial in range(n_trials):
        x0 = rng.uniform(-init_scale, init_scale, size=4).astype(np.float32)
        save = (trial == vis_trial)
        try:
            result = rollout_latent_mpc(
                encoder=encoder, mpc=mpc, env=env, x0=x0, T=T,
                stabilization_threshold=stabilization_threshold,
                settling_threshold=settling_threshold,
                device=device, z_star=z_star,
                save_frames=save,
            )
            if save:
                vis_result = result
            successes.append(result['stabilized'])
            settling_times.append(result['settling_time'])
            final_errors.append(result['final_state_error'])
            done_ats.append(result['done_at'])
            frac_stables.append(result['fraction_stable'])
            xs, us = result['states'], result['actions']
            true_costs.append(
                sum(float(xs[k] @ Q_phys @ xs[k] + us[k] @ R_phys @ us[k])
                    for k in range(result['done_at']))
            )
        except Exception as exc:
            warnings.warn(f'MPC trial {trial} failed: {exc}')
            successes.append(False)
            settling_times.append(T)
            final_errors.append(float('nan'))
            done_ats.append(0)
            frac_stables.append(0.0)

    return {
        'success_rate': float(np.mean(successes)),
        'mean_settling_time': float(np.mean(settling_times)),
        'mean_final_error': float(np.nanmean(final_errors)),
        'mean_episode_length': float(np.mean(done_ats)),  # avg steps before done
        'mean_fraction_stable': float(np.mean(frac_stables)),  # avg time near eq.
        'true_lqr_cost': float(np.mean(true_costs)) if true_costs else float('nan'),
        'n_trials': n_trials,
        'horizon': mpc.horizon,
        'chunk_size': mpc.chunk_size,
        'vis_result': vis_result,
    }
