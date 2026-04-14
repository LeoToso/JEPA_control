"""Kalman decomposition probes P3.1, P3.2, P3.3, P3.4."""
from __future__ import annotations
import warnings
from typing import Dict,Any,Optional
import numpy as np
import scipy.linalg

def _controllability_matrix(A,B):
    d=A.shape[0]
    cols=[B]
    Ak=np.eye(d)
    for _ in range(d-1):
        Ak=Ak@A
        cols.append(Ak@B)
    return np.hstack(cols)

def _observability_matrix(A,C):
    rows=[C]
    Ak=np.eye(A.shape[0])
    for _ in range(A.shape[0]-1):
        Ak=Ak@A
        rows.append(C@Ak)
    return np.vstack(rows)

def _effective_rank(M,threshold):
    sv=np.linalg.svd(M,compute_uv=False)
    return int(np.sum(sv>threshold))

def P3_1_kalman_decomposition(A_hat,B_hat,C_hat,epsilon_rank=1e-3):
    d=A_hat.shape[0]
    R=_controllability_matrix(A_hat,B_hat)
    O=_observability_matrix(A_hat,C_hat)
    r_C=_effective_rank(R,epsilon_rank)
    r_O=_effective_rank(O,epsilon_rank)
    OR=O@R
    dim_Xco=_effective_rank(OR,epsilon_rank)
    efficiency_ratio=dim_Xco/d
    U_c,s_c,_=np.linalg.svd(R,full_matrices=True)
    U_cbar=U_c[:,r_C:]
    _,s_o,Vh_o=np.linalg.svd(O,full_matrices=True)
    V_obar=Vh_o[r_O:,:].T
    phantom_spectral_radius=float('nan')
    has_phantom_instability=False
    if U_cbar.shape[1]>0 and V_obar.shape[1]>0:
        try:
            A_unc=U_cbar.T@A_hat@U_cbar
            eigs_unc=scipy.linalg.eigvals(A_unc)
            phantom_spectral_radius=float(np.max(np.abs(eigs_unc)))
            has_phantom_instability=bool(phantom_spectral_radius>1.0)
        except Exception:
            pass
    return {'r_controllable':r_C,'r_observable':r_O,'dim_Xco':dim_Xco,'efficiency_ratio':float(efficiency_ratio),'phantom_spectral_radius':phantom_spectral_radius,'has_phantom_instability':has_phantom_instability,'latent_dim':d}

def P3_2_controllable_subspace_alignment(A_hat,B_hat,A_star,B_star,paired_data):
    R_z=_controllability_matrix(A_hat,B_hat)
    R_star=_controllability_matrix(A_star,B_star)
    U_z,s_z,_=np.linalg.svd(R_z,full_matrices=False)
    U_star,s_star,_=np.linalg.svd(R_star,full_matrices=False)
    eps=1e-3
    r_z=int(np.sum(s_z>eps*s_z[0]))
    r_star=int(np.sum(s_star>eps*s_star[0]))
    U_z=U_z[:,:r_z]
    U_star=U_star[:,:r_star]
    result={'r_controllable_latent':r_z,'r_controllable_true':r_star}
    xs=paired_data.get('states')
    zs=paired_data.get('latent_states')
    if xs is None or zs is None:
        result['subspace_alignment_error']=float('nan')
        result['principal_angles']=[]
        return result
    W_T,_,_,_=np.linalg.lstsq(xs,zs,rcond=1e-10)
    W=W_T.T
    WU_star=W@U_star
    M=U_z.T@WU_star
    sv_M=np.linalg.svd(M,compute_uv=False)
    sv_clipped=np.clip(sv_M,-1.0,1.0)
    principal_angles=np.arccos(sv_clipped)
    eta_C=float(np.mean(1.0-np.cos(principal_angles)))
    result['principal_angles']=principal_angles.tolist()
    result['subspace_alignment_error']=eta_C
    return result

def P3_3_manifold_geometry(A_hat,rollout_data,A_star=None,n_rollouts=50):
    eigvals,eigvecs=scipy.linalg.eig(A_hat)
    stable_mask=np.abs(eigvals)<1.0
    unstable_mask=~stable_mask
    result={}
    try:
        V=eigvecs
        V_inv=np.linalg.inv(V)
        P_u=np.real(V[:,unstable_mask]@V_inv[unstable_mask,:])
        P_s=np.real(V[:,stable_mask]@V_inv[stable_mask,:])
    except np.linalg.LinAlgError:
        P_u=np.zeros_like(A_hat)
        P_s=np.eye(A_hat.shape[0])
    lz=rollout_data.get('latent_states')
    if lz is None:
        result['fitted_unstable_rate']=float('nan')
        result['fitted_stable_rate']=float('nan')
        return result
    lz=lz[:n_rollouts]
    if lz.ndim==2:
        lz=lz[np.newaxis]
    z_star=np.zeros(lz.shape[-1])
    dz=lz-z_star
    dz_u=np.einsum('ij,ntj->nti',P_u,dz)
    dz_s=np.einsum('ij,ntj->nti',P_s,dz)
    norm_u=np.linalg.norm(dz_u,axis=-1).mean(axis=0)
    norm_s=np.linalg.norm(dz_s,axis=-1).mean(axis=0)
    T=len(norm_u)
    ts=np.arange(T,dtype=float)
    def fit_rate(norms):
        log_n=np.log(np.maximum(norms,1e-10))
        mask=np.isfinite(log_n)&(norms>1e-10)
        if mask.sum()<3:
            return float('nan')
        slope,_=np.polyfit(ts[mask],log_n[mask],1)
        return float(np.exp(slope))
    result['fitted_unstable_rate']=fit_rate(norm_u)
    result['fitted_stable_rate']=fit_rate(norm_s)
    if A_star is not None:
        eigs_star=scipy.linalg.eigvals(A_star)
        unstable_true=eigs_star[np.abs(eigs_star)>1.0]
        stable_true=eigs_star[np.abs(eigs_star)<1.0]
        rho_u_star=float(np.max(np.abs(unstable_true))) if len(unstable_true)>0 else 1.0
        rho_s_star=float(np.max(np.abs(stable_true))) if len(stable_true)>0 else 0.0
        r_u=result['fitted_unstable_rate']
        r_s=result['fitted_stable_rate']
        result['Delta_u']=float(abs(r_u-rho_u_star)) if not np.isnan(r_u) else float('nan')
        result['Delta_s']=float(abs(r_s-rho_s_star)) if not np.isnan(r_s) else float('nan')
    return result

def P3_4_phantom_instability(A_hat,B_hat,C_hat,encoder,env,n_initial=20,T=500):
    from control.lqr import solve_discrete_lqr
    d=A_hat.shape[0]
    d_u=B_hat.shape[1]
    try:
        K_hat,_,_=solve_discrete_lqr(A_hat,B_hat,np.eye(d),0.01*np.eye(d_u))
    except Exception as exc:
        warnings.warn(f'LQR failed in P3.4: {exc}')
        return {'phantom_growth':False}
    R_ctrl=_controllability_matrix(A_hat,B_hat)
    O_obs=_observability_matrix(A_hat,C_hat)
    eps=1e-3
    U_c,s_c,_=np.linalg.svd(R_ctrl,full_matrices=True)
    r_C=int(np.sum(s_c>eps*s_c[0]))
    U_cbar=U_c[:,r_C:]
    _,s_o,Vh_o=np.linalg.svd(O_obs,full_matrices=True)
    r_O=int(np.sum(s_o>eps*s_o[0]))
    P_co=U_c[:,:r_C]@U_c[:,:r_C].T
    P_phantom=U_cbar@U_cbar.T if U_cbar.shape[1]>0 else np.zeros((d,d))
    rng=np.random.RandomState(42)
    import torch
    co_ni,co_nf,ph_ni,ph_nf=[],[],[],[]
    encoder.eval()
    device=next(encoder.parameters()).device
    for trial in range(n_initial):
        x0=rng.uniform(-0.1,0.1,size=4)
        obs,state,_=env.reset_to_state(x0)
        z_traj=[]
        for t in range(T):
            obs_t=torch.from_numpy(obs).float().permute(2,0,1)[None].to(device)/255.0
            with torch.no_grad():
                z_t=encoder(obs_t).cpu().numpy()[0]
            z_traj.append(z_t)
            a_t=-K_hat@z_t
            u_t=float(np.clip(a_t[0] if d_u==1 else np.dot(a_t,np.ones(d_u)),-10,10))
            obs,state,_,done,_=env.step(u_t)
            if done:
                break
        z_traj=np.array(z_traj)
        if len(z_traj)<2:
            continue
        co_ni.append(float(np.linalg.norm(P_co@z_traj[0])))
        co_nf.append(float(np.linalg.norm(P_co@z_traj[-1])))
        ph_ni.append(float(np.linalg.norm(P_phantom@z_traj[0])))
        ph_nf.append(float(np.linalg.norm(P_phantom@z_traj[-1])))
    if not co_ni:
        return {'phantom_growth':False}
    return {'Xco_norm_initial':float(np.mean(co_ni)),'Xco_norm_final':float(np.mean(co_nf)),'Xcbarcbar_norm_initial':float(np.mean(ph_ni)),'Xcbarcbar_norm_final':float(np.mean(ph_nf)),'phantom_growth':bool(np.mean(ph_nf)>np.mean(ph_ni))}
