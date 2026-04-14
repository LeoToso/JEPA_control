"""True-system closed-loop rollout with latent LQR controller."""
from __future__ import annotations
import warnings
from typing import Optional,Dict,Any
import numpy as np
import torch

def rollout_latent_lqr(encoder,A_hat,B_hat,K_hat,env,x0,T=200,use_observer=False,L_hat=None,C_hat=None,stabilization_threshold=0.1,settling_threshold=0.05,device=None):
    if device is None:
        try:
            device=next(encoder.parameters()).device
        except StopIteration:
            device=torch.device('cpu')
    encoder.eval()
    obs,state,_=env.reset_to_state(x0)
    states,latent_states,actions,latent_actions,observer_errors=[],[],[],[],[]
    d=A_hat.shape[0]
    d_u=B_hat.shape[1]
    if use_observer and L_hat is not None and C_hat is not None:
        from control.observer import LuenbergerObserver
        observer=LuenbergerObserver(A_hat,B_hat,C_hat,L_hat)
    else:
        observer=None
        use_observer=False
    settling_time=T
    x_star=np.zeros(4)
    for t in range(T):
        states.append(state.copy())
        obs_t=torch.from_numpy(obs).float().permute(2,0,1)[None].to(device)/255.0
        with torch.no_grad():
            z_t=encoder(obs_t).cpu().numpy()[0]
        latent_states.append(z_t.copy())
        if use_observer and observer is not None:
            y_t=C_hat@z_t
            z_hat_t=observer.update(y_t,np.zeros(d_u))
            a_t=-K_hat@z_hat_t
        else:
            a_t=-K_hat@z_t
        latent_actions.append(a_t.copy())
        try:
            u_scalar=float(np.clip(a_t[0],-10.0,10.0))
        except Exception:
            u_scalar=0.0
        actions.append(np.array([u_scalar]))
        if use_observer and observer is not None:
            observer.state=A_hat@observer.state+B_hat@a_t+L_hat@(C_hat@z_t-C_hat@observer.state)
        obs,state,_,done,_=env.step(u_scalar)
        if np.linalg.norm(state-x_star)<settling_threshold and settling_time==T:
            settling_time=t
        if use_observer and observer is not None:
            observer_errors.append(float(np.linalg.norm(z_t-observer.state)))
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
    result={'states':states,'latent_states':latent_states,'actions':actions,'latent_actions':latent_actions,'final_state_error':final_error,'stabilized':bool(final_error<stabilization_threshold),'settling_time':settling_time}
    if observer_errors:
        result['observer_error_final']=float(observer_errors[-1])
        result['observer_error_mean']=float(np.mean(observer_errors))
    return result

def evaluate_stabilization(encoder,A_hat,B_hat,K_hat,env,n_trials=100,T=200,use_observer=False,L_hat=None,C_hat=None,init_scale=0.2,Q_lqr=None,R_lqr=None,stabilization_threshold=0.1,settling_threshold=0.05,seed=0,device=None):
    rng=np.random.RandomState(seed)
    d=A_hat.shape[0]
    d_u=B_hat.shape[1]
    Q_lqr_default=np.diag([1.0,1.0,10.0,1.0]) if Q_lqr is None else Q_lqr
    R_lqr_default=0.01*np.eye(1) if R_lqr is None else R_lqr
    successes,settling_times,final_errors,true_costs,latent_costs,all_results=[],[],[],[],[],[]
    for trial in range(n_trials):
        x0=rng.uniform(-init_scale,init_scale,size=4).astype(np.float32)
        try:
            result=rollout_latent_lqr(encoder=encoder,A_hat=A_hat,B_hat=B_hat,K_hat=K_hat,env=env,x0=x0,T=T,use_observer=use_observer,L_hat=L_hat,C_hat=C_hat,stabilization_threshold=stabilization_threshold,settling_threshold=settling_threshold,device=device)
            successes.append(result['stabilized'])
            settling_times.append(result['settling_time'])
            final_errors.append(result['final_state_error'])
            xs=result['states']
            us=result['actions']
            true_cost=sum(float(xs[t]@Q_lqr_default@xs[t]+us[t]@R_lqr_default@us[t]) for t in range(len(xs)))
            true_costs.append(true_cost)
            zs=result['latent_states']
            as_=result['latent_actions']
            Q_lat=np.eye(d)
            R_lat=0.01*np.eye(d_u)
            latent_cost=sum(float(zs[t]@Q_lat@zs[t]+as_[t]@R_lat@as_[t]) for t in range(len(zs)))
            latent_costs.append(latent_cost)
            all_results.append(result)
        except Exception as exc:
            warnings.warn(f'Trial {trial} failed: {exc}')
            successes.append(False)
            settling_times.append(T)
            final_errors.append(float('nan'))
    return {'success_rate':float(np.mean(successes)),'mean_settling_time':float(np.mean(settling_times)),'mean_final_error':float(np.nanmean(final_errors)),'true_lqr_cost':float(np.mean(true_costs)) if true_costs else float('nan'),'latent_lqr_cost':float(np.mean(latent_costs)) if latent_costs else float('nan'),'n_trials':n_trials,'all_results':all_results}
