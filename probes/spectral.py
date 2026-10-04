"""Spectral probes P1.1, P1.2, P1.3."""
from __future__ import annotations
import warnings
from typing import Optional,Dict,Any
import numpy as np
import scipy.linalg
import scipy.signal

def P1_1_eigenvalue_recovery(A_hat,A_star,epsilon_lambda=0.05):
    eigvals_hat=scipy.linalg.eigvals(A_hat)
    eigvals_star=scipy.linalg.eigvals(A_star)
    rho_hat=float(np.max(np.abs(eigvals_hat)))
    rho_star=float(np.max(np.abs(eigvals_star)))
    unstable_true=eigvals_star[np.abs(eigvals_star)>1.0]
    unstable_hat=eigvals_hat[np.abs(eigvals_hat)>1.0]
    ell_star=len(unstable_true)
    n_unstable_latent=len(unstable_hat)
    if ell_star==0:
        warnings.warn('No unstable true eigenvalues found.')
        return {'delta_lambda':float('nan'),'UMR':float('nan'),'spectral_radius_error':abs(rho_hat-rho_star),'n_unstable_latent':n_unstable_latent,'n_unstable_true':0}
    dists=[]
    within_eps=0
    for lam_star in unstable_true:
        d=np.min(np.abs(eigvals_hat-lam_star))
        dists.append(float(d))
        if d<epsilon_lambda:
            within_eps+=1
    return {'delta_lambda':float(np.mean(dists)),'UMR':float(within_eps/ell_star),'spectral_radius_error':float(abs(rho_hat-rho_star)),'n_unstable_latent':n_unstable_latent,'n_unstable_true':ell_star,'individual_distances':dists}

def P1_2_jordan_block(A_hat,rollout_data=None,eps_cluster=0.05):
    d=A_hat.shape[0]
    eigvals,eigvecs=scipy.linalg.eig(A_hat)
    try:
        cond=float(np.linalg.cond(eigvecs))
    except Exception:
        cond=float('inf')
    result={'eigvec_condition_number':cond,'rank_deficiency_k1':0,'rank_deficiency_k2':0,'better_fit':None}
    used=np.zeros(len(eigvals),dtype=bool)
    max_cluster_size=0
    best_cluster_center=None
    for i,lam in enumerate(eigvals):
        if used[i]:
            continue
        cluster=[i]
        for j in range(i+1,len(eigvals)):
            if not used[j] and abs(eigvals[j]-lam)<eps_cluster:
                cluster.append(j)
                used[j]=True
        used[i]=True
        if len(cluster)>max_cluster_size:
            max_cluster_size=len(cluster)
            best_cluster_center=lam
    if best_cluster_center is not None and max_cluster_size>=2:
        I_d=np.eye(d)
        M1=A_hat-best_cluster_center.real*I_d
        r1=np.linalg.matrix_rank(M1,tol=1e-6)
        result['rank_deficiency_k1']=d-r1
        M2=M1@M1
        r2=np.linalg.matrix_rank(M2,tol=1e-6)
        result['rank_deficiency_k2']=d-r2
    if rollout_data is not None and 'latent_states' in rollout_data:
        zs=rollout_data['latent_states']
        if zs.ndim==2:
            zs=zs[np.newaxis]
        z_star=zs.mean(axis=(0,1),keepdims=True)
        norms=np.linalg.norm(zs-z_star,axis=-1)
        mean_norms=norms.mean(axis=0)
        T=len(mean_norms)
        ts=np.arange(T,dtype=float)
        mean_norms=np.maximum(mean_norms,1e-10)
        unstable_mask=np.abs(eigvals)>=1.0
        if np.any(unstable_mask):
            lam_dom=eigvals[np.argmax(np.abs(eigvals[unstable_mask]))]
            r=float(np.abs(lam_dom))
            log_norms=np.log(mean_norms)
            log_r=np.log(r+1e-12)
            c2_log=np.mean(log_norms-log_r*ts)
            exp_pred=np.exp(c2_log)*r**ts
            res_exp=np.mean((mean_norms-exp_pred)**2)
            t_r=ts*r**ts+1e-12
            c1=np.sum(mean_norms*t_r)/(np.sum(t_r**2)+1e-12)
            poly_pred=c1*t_r
            res_poly=np.mean((mean_norms-poly_pred)**2)
            result['better_fit']='polynomial' if res_poly<res_exp else 'exponential'
            result['residual_exponential']=float(res_exp)
            result['residual_polynomial']=float(res_poly)
    return result

def P1_3_marginal_mode_frequency(rollout_data,A_hat,A_star=None,n_rollouts=50,dt=0.02):
    lz=rollout_data.get('latent_states')
    ys=rollout_data.get('states')
    result={}
    if lz is not None and len(lz)>0:
        lz=lz[:n_rollouts]
        T=lz.shape[1]
        z_star=np.zeros(lz.shape[-1])
        norms=np.linalg.norm(lz-z_star,axis=-1)
        mean_norms=norms.mean(axis=0)
        freqs=np.fft.rfftfreq(T,d=dt)
        psd=np.abs(np.fft.rfft(mean_norms))**2
        omega_z_peak=float(freqs[1+np.argmax(psd[1:])]) if len(freqs)>1 else 0.0
        ts=np.arange(T,dtype=float)
        log_norms=np.log(np.maximum(mean_norms,1e-10))
        mask=np.isfinite(log_norms)
        if mask.sum()>2:
            slope,_=np.polyfit(ts[mask],log_norms[mask],1)
            r_z=float(np.exp(slope))
        else:
            r_z=float('nan')
        result['omega_z_peak']=omega_z_peak
        result['growth_rate_latent']=r_z
    if ys is not None and len(ys)>0:
        ys=ys[:n_rollouts]
        T=ys.shape[1]
        pole_angles=ys[:,:,2]
        mean_angle=np.mean(np.abs(pole_angles),axis=0)
        freqs=np.fft.rfftfreq(T,d=dt)
        psd=np.abs(np.fft.rfft(mean_angle))**2
        omega_y_peak=float(freqs[1+np.argmax(psd[1:])]) if len(freqs)>1 else 0.0
        ts=np.arange(T,dtype=float)
        log_angle=np.log(np.maximum(mean_angle,1e-10))
        mask=np.isfinite(log_angle)
        if mask.sum()>2:
            slope,_=np.polyfit(ts[mask],log_angle[mask],1)
            r_true=float(np.exp(slope))
        else:
            r_true=float('nan')
        result['omega_y_peak']=omega_y_peak
        result['growth_rate_true']=r_true
    if 'omega_z_peak' in result and 'omega_y_peak' in result:
        result['delta_omega']=float(abs(result['omega_z_peak']-result['omega_y_peak']))
    if A_star is not None:
        eigs_star=scipy.linalg.eigvals(A_star)
        rho_star=float(np.max(np.abs(eigs_star)))
        result['growth_rate_true_analytic']=rho_star
    return result

