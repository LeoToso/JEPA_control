"""JEPA prediction loss with stop-gradient + optional VICReg regularisation."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def jepa_prediction_loss(z_hat,z_next_sg):
    return F.mse_loss(z_hat,z_next_sg)

def vicreg_loss(z,z_next,lambda_var=25.0,mu_cov=25.0,nu_inv=1.0,eps=1e-4):
    B,d=z.shape
    inv_loss=F.mse_loss(z,z_next)
    std_z=torch.sqrt(z.var(dim=0)+eps)
    std_z_next=torch.sqrt(z_next.var(dim=0)+eps)
    var_loss=(torch.mean(F.relu(1.0-std_z))+torch.mean(F.relu(1.0-std_z_next)))/2.0
    z_c=z-z.mean(dim=0)
    z_next_c=z_next-z_next.mean(dim=0)
    cov_z=(z_c.T@z_c)/(B-1)
    cov_z_next=(z_next_c.T@z_next_c)/(B-1)
    off_diag_sq=lambda m:(m**2).sum()-(torch.diag(m)**2).sum()
    cov_loss=(off_diag_sq(cov_z)+off_diag_sq(cov_z_next))/(2.0*d)
    loss=mu_cov*inv_loss+lambda_var*var_loss+nu_inv*cov_loss
    info={"vicreg_inv":inv_loss.item(),"vicreg_var":var_loss.item(),"vicreg_cov":cov_loss.item()}
    return loss,info

def vicreg_collapse_loss(z, lambda_var=25.0, nu_cov=1.0, eps=1e-4):
    """Variance + covariance regularization on a single embedding set.

    No invariance term — that caused encoder collapse when applied between
    online frames whose targets were themselves changing.  Variance prevents
    mode collapse; covariance decorrelates latent dimensions.
    """
    B, d = z.shape
    std_z    = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std_z))
    z_c      = z - z.mean(dim=0)
    cov      = (z_c.T @ z_c) / (B - 1)
    cov_loss = ((cov ** 2).sum() - (torch.diag(cov) ** 2).sum()) / d
    info = {'vicreg_var': var_loss.item(), 'vicreg_cov': cov_loss.item()}
    return lambda_var * var_loss + nu_cov * cov_loss, info


def combined_prediction_loss(outputs,use_vicreg=True,vicreg_lambda=25.0,vicreg_mu=25.0,vicreg_nu=1.0):
    z_hat=outputs["z_hat"]
    z_next_sg=outputs["z_next_sg"]
    z_t=outputs["z_t"]
    z_next=outputs["z_next"]
    pred_loss=jepa_prediction_loss(z_hat,z_next_sg)
    info={"pred_loss":pred_loss.item()}
    if use_vicreg:
        vic_loss,vic_info=vicreg_loss(z_t,z_next,lambda_var=vicreg_lambda,mu_cov=vicreg_mu,nu_inv=vicreg_nu)
        total=pred_loss+vic_loss
        info.update(vic_info)
        info["vicreg_total"]=vic_loss.item()
    else:
        total=pred_loss
    info["total_pred"]=total.item()
    return total,info
