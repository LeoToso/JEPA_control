"""Spectral regularisation loss."""
from __future__ import annotations
import torch
import numpy as np
from losses.pbh import mini_batch_dmdc

def spectral_matching_loss(z,a,z_next,true_unstable_eigenvalues,eps=1e-8):
    A_hat,_=mini_batch_dmdc(z,a,z_next)
    eigvals=torch.linalg.eig(A_hat).eigenvalues
    true_eigs=torch.tensor(true_unstable_eigenvalues,dtype=eigvals.dtype,device=eigvals.device)
    if true_eigs.numel()==0:
        return torch.tensor(0.0,device=z.device,requires_grad=True),{}
    loss_terms=[]
    min_dists=[]
    for lam_star in true_eigs:
        diff=eigvals-lam_star
        dist_sq=diff.real**2+diff.imag**2
        min_dist_sq=dist_sq.min()
        loss_terms.append(min_dist_sq)
        min_dists.append(min_dist_sq.item())
    loss=torch.stack(loss_terms).sum()
    info={"spec_loss":loss.item(),"mean_min_dist":float(np.mean(min_dists))}
    return loss,info
