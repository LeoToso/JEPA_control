"""PBH probes P2.1, P2.2, P2.3."""
from __future__ import annotations
import warnings
from typing import Dict,Any,Optional
import numpy as np
import scipy.linalg

def P2_1_pbh_stabilizability(A_hat,B_hat,delta_tol=0.05):
    d=A_hat.shape[0]
    eigvals=scipy.linalg.eigvals(A_hat)
    near_unstable=eigvals[np.abs(eigvals)>=(1.0-delta_tol)]
    if len(near_unstable)==0:
        return {'mu_S':float('nan'),'pbh_values':{},'is_stabilizable':True,'n_near_unstable':0}
    pbh_values={}
    sigma_mins=[]
    for lam in near_unstable:
        lam_r=lam.real
        M_S=np.hstack([lam_r*np.eye(d)-A_hat,B_hat])
        sv=np.linalg.svd(M_S,compute_uv=False)
        sigma_min=float(sv[-1])
        pbh_values[complex(lam)]=sigma_min
        sigma_mins.append(sigma_min)
    mu_S=float(min(sigma_mins))
    return {'mu_S':mu_S,'pbh_values':{str(k):v for k,v in pbh_values.items()},'is_stabilizable':bool(mu_S>1e-4),'n_near_unstable':len(near_unstable)}

def P2_2_pbh_detectability(A_hat,C_hat,encoder=None,gt=None,delta_tol=0.05,epsilon=0.01,n_test=100,device=None):
    d=A_hat.shape[0]
    eigvals=scipy.linalg.eigvals(A_hat)
    near_unstable=eigvals[np.abs(eigvals)>=(1.0-delta_tol)]
    result={}
    if len(near_unstable)==0:
        result.update({'mu_D':float('nan'),'is_detectable':True,'n_near_unstable':0})
    else:
        sigma_mins=[]
        for lam in near_unstable:
            lam_r=lam.real
            M_D=np.vstack([lam_r*np.eye(d)-A_hat,C_hat])
            sv=np.linalg.svd(M_D,compute_uv=False)
            sigma_mins.append(float(sv[-1]))
        result['mu_D']=float(min(sigma_mins))
        result['is_detectable']=bool(result['mu_D']>1e-4)
        result['n_near_unstable']=len(near_unstable)
    if encoder is not None and gt is not None:
        try:
            import torch
            v1=gt.dominant_right_eigenvector
            if v1 is None:
                result['MSR_unstable']=float('nan')
                return result
            v1=np.real(v1)
            v1=v1/(np.linalg.norm(v1)+1e-12)
            rng=np.random.RandomState(42)
            x_bases=rng.uniform(-0.05,0.05,size=(n_test,4)).astype(np.float32)
            x_perturbed=x_bases+epsilon*v1.astype(np.float32)[np.newaxis,:]
            from envs.cartpole_visual import ContinuousCartpoleVisual
            env=ContinuousCartpoleVisual()
            z_bases=[]
            z_perturbed=[]
            encoder.eval()
            for i in range(n_test):
                obs_b,_,_=env.reset_to_state(x_bases[i])
                obs_p,_,_=env.reset_to_state(x_perturbed[i])
                obs_b_t=torch.from_numpy(obs_b).float().permute(2,0,1)[None]/255.0
                obs_p_t=torch.from_numpy(obs_p).float().permute(2,0,1)[None]/255.0
                if device is not None:
                    obs_b_t=obs_b_t.to(device)
                    obs_p_t=obs_p_t.to(device)
                with torch.no_grad():
                    z_b=encoder(obs_b_t).cpu().numpy()[0]
                    z_p=encoder(obs_p_t).cpu().numpy()[0]
                z_bases.append(z_b)
                z_perturbed.append(z_p)
            env.close()
            z_bases=np.array(z_bases)
            z_perturbed=np.array(z_perturbed)
            dz=np.linalg.norm(z_perturbed-z_bases,axis=-1)
            result['MSR_unstable']=float(np.mean(dz)/(epsilon+1e-12))
        except Exception as exc:
            warnings.warn(f'MSR computation failed: {exc}')
            result['MSR_unstable']=float('nan')
    return result

def P2_3_separation_principle(A_hat,B_hat,C_hat,encoder,env,n_trials=100,T=200,stabilization_threshold=0.1,settling_threshold=0.05):
    """Fully-observed latent LQR probe: u_t = -K(z_t - z_star), no observer."""
    import torch
    from control.lqr import solve_discrete_lqr,pre_stabilize_A
    from control.rollout import rollout_latent_lqr
    d=A_hat.shape[0]
    d_u=B_hat.shape[1]
    Q_lqr=np.eye(d)
    R_lqr=0.01*np.eye(d_u)
    try:
        A_dare,_=pre_stabilize_A(A_hat,[],tol=0.05,target=0.9)
        K_hat,P_hat,cl_eigs=solve_discrete_lqr(A_dare,B_hat,Q_lqr,R_lqr)
        if float(np.max(np.abs(cl_eigs)))>=1.0:
            raise ValueError('LQR unstable on pre-stabilised system')
    except Exception as exc:
        warnings.warn(f'LQR design failed: {exc}')
        return {'stabilization_success_rate':0.0,'mean_settling_time':T}
    try:
        device=next(encoder.parameters()).device
    except StopIteration:
        device=torch.device('cpu')
    obs_eq,_,_=env.reset_to_state(np.zeros(4))
    obs_eq_t=torch.from_numpy(obs_eq).float().permute(2,0,1)[None].to(device)/255.0
    with torch.no_grad():
        z_star=encoder(obs_eq_t).cpu().numpy()[0]
    rng=np.random.RandomState(0)
    success=[]
    settling_times=[]
    for trial in range(n_trials):
        x0=rng.uniform(-0.05,0.05,size=4)
        try:
            result=rollout_latent_lqr(encoder=encoder,A_hat=A_hat,B_hat=B_hat,
                                      K_hat=K_hat,env=env,x0=x0,T=T,
                                      z_star=z_star,device=device)
            success.append(result['stabilized'])
            settling_times.append(result['settling_time'])
        except Exception:
            success.append(False)
            settling_times.append(T)
    return {'stabilization_success_rate':float(np.mean(success)),
            'mean_settling_time':float(np.mean(settling_times))}
