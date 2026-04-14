"""Non-minimum-phase (NMP) zero preservation loss."""
from __future__ import annotations
import warnings
import numpy as np
import torch
import scipy.linalg
from losses.pbh import mini_batch_dmdc
from typing import Optional

def _compute_latent_zeros_numpy(A_hat,B_hat,C_hat,D_hat=None):
    if D_hat is None:
        D_hat=np.zeros((C_hat.shape[0],B_hat.shape[1]))
    n=A_hat.shape[0]
    p,m=C_hat.shape[0],B_hat.shape[1]
    N_mat=np.zeros((n+p,n+m))
    M_mat=np.zeros((n+p,n+m))
    N_mat[:n,:n]=np.eye(n)
    M_mat[:n,:n]=A_hat
    M_mat[:n,n:n+m]=B_hat
    M_mat[n:n+p,:n]=-C_hat
    M_mat[n:n+p,n:n+m]=-D_hat
    try:
        zeros_all=scipy.linalg.eigvals(M_mat,N_mat)
        finite_mask=np.isfinite(zeros_all)
        zeros_finite=zeros_all[finite_mask]
        reasonable_mask=np.abs(zeros_finite)<1e6
        return zeros_finite[reasonable_mask]
    except Exception:
        return np.array([],dtype=complex)

def nmp_zero_loss(z,a,z_next,C_hat,true_nmp_zeros,eps=1e-8):
    if len(true_nmp_zeros)==0:
        return torch.tensor(0.0,device=z.device,requires_grad=True),{}
    A_hat_t,B_hat_t=mini_batch_dmdc(z,a,z_next)
    A_np=A_hat_t.detach().cpu().numpy()
    B_np=B_hat_t.detach().cpu().numpy()
    latent_zeros=_compute_latent_zeros_numpy(A_np,B_np,C_hat)
    if len(latent_zeros)==0:
        info={"nmp_loss":0.0,"latent_nmp_count":0,"true_nmp_count":len(true_nmp_zeros)}
        return torch.tensor(0.0,device=z.device,requires_grad=True),info
    true_zeros_t=torch.tensor(np.array([[z.real,z.imag] for z in true_nmp_zeros]),dtype=torch.float32,device=z.device)
    latent_zeros_t=torch.tensor(np.array([[z.real,z.imag] for z in latent_zeros]),dtype=torch.float32,device=z.device)
    loss_terms=[]
    for true_z in true_zeros_t:
        diffs=latent_zeros_t-true_z.unsqueeze(0)
        dist_sq=(diffs**2).sum(dim=-1)
        loss_terms.append(dist_sq.min())
    loss=torch.stack(loss_terms).sum()
    latent_nmp=[z for z in latent_zeros if abs(z)>1.0]
    info={"nmp_loss":loss.item(),"latent_nmp_count":len(latent_nmp),"true_nmp_count":len(true_nmp_zeros)}
    return loss,info
