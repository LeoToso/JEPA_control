"""Transmission zero probes P4.1, P4.2, P4.3, P4.4."""
from __future__ import annotations
import warnings
from typing import Dict,Any,Optional
import numpy as np
import scipy.linalg

def _transmission_zeros(A,B,C,D=None):
    n=A.shape[0]
    p,m=C.shape[0],B.shape[1]
    if D is None:
        D=np.zeros((p,m))
    nr,nc=n+p,n+m
    N_mat=np.zeros((nr,nc))
    M_mat=np.zeros((nr,nc))
    N_mat[:n,:n]=np.eye(n)
    M_mat[:n,:n]=A
    M_mat[:n,n:]=B
    M_mat[n:,:n]=-C
    M_mat[n:,n:]=-D
    try:
        zeros_all=scipy.linalg.eigvals(M_mat,N_mat)
        finite_mask=np.isfinite(zeros_all)
        zeros_finite=zeros_all[finite_mask]
        return zeros_finite[np.abs(zeros_finite)<1e6]
    except Exception:
        return np.array([],dtype=complex)

def P4_1_transmission_zeros(A_hat,B_hat,C_hat,A_star,B_star,C_star,D_hat=None,D_star=None):
    latent_zeros=_transmission_zeros(A_hat,B_hat,C_hat,D_hat)
    true_zeros=_transmission_zeros(A_star,B_star,C_star,D_star)
    latent_nmp=latent_zeros[np.abs(latent_zeros)>1.0]
    true_nmp=true_zeros[np.abs(true_zeros)>1.0]
    nmp_count_match=len(latent_nmp)==len(true_nmp)
    if len(latent_zeros)>0 and len(true_zeros)>0:
        dists=[float(np.min(np.abs(latent_zeros-zt))) for zt in true_zeros]
        Delta_z=float(np.mean(dists))
    else:
        Delta_z=float('nan')
    return {'latent_zeros':latent_zeros.tolist() if len(latent_zeros)>0 else [],'true_zeros':true_zeros.tolist() if len(true_zeros)>0 else [],'latent_NMP_count':len(latent_nmp),'true_NMP_count':len(true_nmp),'NMP_count_match':nmp_count_match,'zero_matching_distance':Delta_z}

def P4_2_step_response_undershoot(A_hat,B_hat,C_hat,encoder,env,T=100,n_trials=20):
    d=A_hat.shape[0]
    d_u=B_hat.shape[1]
    p=C_hat.shape[0]
    a_step=np.ones((d_u,))
    z=np.zeros(d)
    y_latent=np.zeros((T,p))
    for t in range(T):
        y_latent[t]=C_hat@z
        z=A_hat@z+B_hat@a_step
    UR_latent_list=[]
    for i in range(p):
        y=y_latent[:,i]
        y0,yT=y[0],y[-1]
        if abs(yT-y0)<1e-10:
            UR_latent_list.append(float('nan'))
        else:
            UR_latent_list.append(float((np.min(y)-y0)/(yT-y0)))
    UR_latent=float(np.nanmean(UR_latent_list))
    NMP_detected_latent=bool(UR_latent<-0.01) if not np.isnan(UR_latent) else False
    import torch
    encoder.eval()
    device=next(encoder.parameters()).device
    UR_true_list=[]
    for trial in range(n_trials):
        obs,state,_=env.reset()
        y_traj=[]
        for t in range(T):
            y_traj.append(float(state[2]))
            obs,state,_,done,_=env.step(1.0)
            if done:
                break
        if len(y_traj)<2:
            continue
        y_arr=np.array(y_traj)
        y0,yT=y_arr[0],y_arr[-1]
        if abs(yT-y0)<1e-10:
            continue
        UR_true_list.append(float((np.min(y_arr)-y0)/(yT-y0)))
    UR_true=float(np.mean(UR_true_list)) if UR_true_list else float('nan')
    NMP_detected_true=bool(UR_true<-0.01) if not np.isnan(UR_true) else False
    return {'UR_latent':UR_latent,'UR_true':UR_true,'sign_match':bool(np.sign(UR_latent)==np.sign(UR_true)) if not (np.isnan(UR_latent) or np.isnan(UR_true)) else False,'NMP_detected_latent':NMP_detected_latent,'NMP_detected_true':NMP_detected_true}

def P4_3_markov_parameters(A_hat,B_hat,C_hat,A_star,B_star,C_star,encoder,env,K_markov=8,delta=0.1,epsilon_H=0.01):
    d=A_hat.shape[0]
    p=C_hat.shape[0]
    d_u=B_hat.shape[1]
    p_star=C_star.shape[0]
    m_star=B_star.shape[1]
    H_latent=np.zeros((K_markov,p,d_u))
    Ak=np.eye(d)
    for k in range(K_markov):
        if k==0:
            H_latent[k]=C_hat@B_hat
        else:
            Ak=Ak@A_hat
            H_latent[k]=C_hat@Ak@B_hat
    H_true=np.zeros((K_markov,p_star,m_star))
    Ak_star=np.eye(A_star.shape[0])
    for k in range(K_markov):
        if k==0:
            H_true[k]=C_star@B_star
        else:
            Ak_star=Ak_star@A_star
            H_true[k]=C_star@Ak_star@B_star
    def relative_degree(H_seq,eps):
        for k in range(len(H_seq)):
            if np.linalg.norm(H_seq[k])>eps:
                return k+1
        return len(H_seq)
    r_hat=relative_degree(H_latent,epsilon_H)
    r_star=relative_degree(H_true,epsilon_H)
    p_min=min(p,p_star)
    d_u_min=min(d_u,m_star)
    Delta_H_num=sum(np.linalg.norm(H_latent[k,:p_min,:d_u_min]-H_true[k,:p_min,:d_u_min],'fro') for k in range(K_markov))
    Delta_H_den=sum(np.linalg.norm(H_true[k],'fro') for k in range(K_markov))+1e-12
    return {'markov_sequence_error':float(Delta_H_num/Delta_H_den),'relative_degree_latent':r_hat,'relative_degree_true':r_star,'relative_degree_match':bool(r_hat==r_star),'markov_params_latent':H_latent.tolist(),'markov_params_true':H_true.tolist()}

def P4_4_zero_direction_alignment(A_hat,B_hat,C_hat,A_star,B_star,C_star,paired_data):
    n=A_star.shape[0]
    d=A_hat.shape[0]
    m=B_hat.shape[1]
    m_star=B_star.shape[1]
    latent_zeros=_transmission_zeros(A_hat,B_hat,C_hat)
    latent_nmp=latent_zeros[np.abs(latent_zeros)>1.0]
    true_zeros=_transmission_zeros(A_star,B_star,C_star)
    true_nmp=true_zeros[np.abs(true_zeros)>1.0]
    if len(latent_nmp)==0 or len(true_nmp)==0:
        return {'zero_direction_angles':[],'mean_zero_direction_angle':float('nan')}
    angles=[]
    for i,z_hat_i in enumerate(latent_nmp[:len(true_nmp)]):
        z_star_i=true_nmp[i] if i<len(true_nmp) else true_nmp[0]
        try:
            nr_z=d+C_hat.shape[0]
            nc_z=d+m
            P_z=np.zeros((nr_z,nc_z),dtype=complex)
            P_z[:d,:d]=z_hat_i*np.eye(d)-A_hat
            P_z[:d,d:]=-B_hat
            P_z[d:,:d]=C_hat
            _,sv_z,Vh_z=np.linalg.svd(P_z)
            null_z=Vh_z[-1,:].conj()
            u_null_z=null_z[d:].real
            nr_s=n+C_star.shape[0]
            nc_s=n+m_star
            P_s=np.zeros((nr_s,nc_s),dtype=complex)
            P_s[:n,:n]=z_star_i*np.eye(n)-A_star
            P_s[:n,n:]=-B_star
            P_s[n:,:n]=C_star
            _,sv_s,Vh_s=np.linalg.svd(P_s)
            null_s=Vh_s[-1,:].conj()
            u_null_s=null_s[n:].real
            n_z=u_null_z/(np.linalg.norm(u_null_z)+1e-12)
            n_s=u_null_s/(np.linalg.norm(u_null_s)+1e-12)
            cos_a=float(np.abs(np.dot(n_z[:min(len(n_z),len(n_s))],n_s[:min(len(n_z),len(n_s))])))
            cos_a=np.clip(cos_a,0.0,1.0)
            angles.append(float(np.arccos(cos_a)))
        except Exception:
            angles.append(float('nan'))
    valid_angles=[a for a in angles if not np.isnan(a)]
    return {'zero_direction_angles':angles,'mean_zero_direction_angle':float(np.mean(valid_angles)) if valid_angles else float('nan')}
