"""PBH stabilisability loss."""
from __future__ import annotations
import warnings
from typing import Optional,Tuple
import torch
import torch.nn.functional as F

def mini_batch_dmdc(z,a,z_next,rcond=1e-5):
    d=z.shape[1]
    d_u=a.shape[1]
    X=torch.cat([z,a],dim=-1)
    Y=z_next
    # Use normal equations via solve — reliably autodiff on all backends.
    # lstsq has inconsistent autograd support on CUDA.
    XtX=X.T@X+rcond*torch.eye(d+d_u,device=X.device,dtype=X.dtype)
    XtY=X.T@Y
    sol=torch.linalg.solve(XtX,XtY)
    A_hat=sol[:d,:].T
    B_hat=sol[d:,:].T
    return A_hat,B_hat

def pbh_stabilizability_loss(z,a,z_next,delta_tol=0.05,eps=1e-6):
    A_hat,B_hat=mini_batch_dmdc(z,a,z_next)
    with torch.no_grad():
        eigvals=torch.linalg.eigvals(A_hat.detach())
        near_unstable_mask=eigvals.abs()>=(1.0-delta_tol)
        near_unstable=eigvals[near_unstable_mask]
    if near_unstable.numel()==0:
        info={"pbh_loss":0.0,"n_unstable":0,"mu_S":float("nan")}
        return torch.tensor(0.0,device=z.device,requires_grad=True),info
    d=A_hat.shape[0]
    loss_terms=[]
    sigma_mins=[]
    for lam in near_unstable:
        lam_r=lam.real.to(A_hat.dtype)
        M_S=torch.cat([lam_r*torch.eye(d,device=A_hat.device,dtype=A_hat.dtype)-A_hat,B_hat],dim=-1)
        sv=torch.linalg.svdvals(M_S)
        sigma_min=sv[-1]
        sigma_mins.append(sigma_min.item())
        # Log-barrier: maximises sigma_min, O(1) scale regardless of initialisation.
        term=-torch.log(sigma_min+eps)
        loss_terms.append(term)
    # Mean over modes so loss scale is independent of how many modes are near-unstable.
    loss=torch.stack(loss_terms).mean()
    info={"pbh_loss":loss.item(),"n_unstable":len(near_unstable),"mu_S":float(min(sigma_mins)) if sigma_mins else float("nan")}
    return loss,info
