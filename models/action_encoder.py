"""Action encoders psi_theta: u(m,)->a(d_u,)."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearActionEncoder(nn.Module):
    """Linear lifting: a=Wu, W in R^{d_u x m}."""
    def __init__(self,action_dim=1,latent_action_dim=32):
        super().__init__()
        self.action_dim=action_dim
        self.latent_action_dim=latent_action_dim
        self.W=nn.Linear(action_dim,latent_action_dim,bias=False)
        nn.init.normal_(self.W.weight,mean=0.0,std=1.0/(action_dim**0.5))
        self.register_buffer('W_pinv',None)

    def forward(self,u):
        return self.W(u)

    @torch.no_grad()
    def compute_pseudoinverse(self):
        W=self.W.weight.data
        WtW=W.T@W
        try:
            WtW_inv=torch.linalg.inv(WtW)
        except Exception:
            WtW_inv=torch.linalg.pinv(WtW)
        self.W_pinv=WtW_inv@W.T

    def decode(self,a):
        if self.W_pinv is None:
            self.compute_pseudoinverse()
        return a@self.W_pinv.T

    def reconstruction_error(self,u):
        if self.W_pinv is None:
            self.compute_pseudoinverse()
        a=self(u)
        u_hat=self.decode(a)
        return torch.norm(u-u_hat,dim=-1)

    @property
    def condition_number(self):
        W=self.W.weight.data
        sv=torch.linalg.svdvals(W)
        sv_pos=sv[sv>1e-12]
        if len(sv_pos)==0:
            return float('inf')
        return float(sv_pos.max()/sv_pos.min())

class MLPActionEncoder(nn.Module):
    """Encoder-decoder pair for action lifting."""
    def __init__(self,action_dim=1,latent_action_dim=32,hidden_dim=128):
        super().__init__()
        self.action_dim=action_dim
        self.latent_action_dim=latent_action_dim
        self.encoder=nn.Sequential(
            nn.Linear(action_dim,hidden_dim),nn.ELU(),
            nn.Linear(hidden_dim,hidden_dim),nn.ELU(),
            nn.Linear(hidden_dim,latent_action_dim),
        )
        self.decoder=nn.Sequential(
            nn.Linear(latent_action_dim,hidden_dim),nn.ELU(),
            nn.Linear(hidden_dim,hidden_dim),nn.ELU(),
            nn.Linear(hidden_dim,action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,u):
        return self.encoder(u)

    def decode(self,a):
        return self.decoder(a)

    def reconstruction_loss(self,u):
        a=self.encoder(u)
        u_hat=self.decoder(a)
        return F.mse_loss(u_hat,u)

    def lipschitz_estimate(self,u_samples,eps=1e-3):
        with torch.no_grad():
            a_samples=self.encoder(u_samples)
            lips=[]
            for i in range(min(len(a_samples),200)):
                a0=a_samples[i]
                delta=torch.randn_like(a0)*eps
                a1=a0+delta
                u0=self.decode(a0.unsqueeze(0)).squeeze(0)
                u1=self.decode(a1.unsqueeze(0)).squeeze(0)
                ratio=(torch.norm(u1-u0)/(torch.norm(delta)+1e-12)).item()
                lips.append(ratio)
        return float(max(lips)) if lips else float('nan')

class IdentityActionEncoder(nn.Module):
    """Pass-through: a=u (no lifting, d_u=m)."""
    def __init__(self,action_dim=1):
        super().__init__()
        self.action_dim=action_dim
        self.latent_action_dim=action_dim
    def forward(self,u):
        return u
    def decode(self,a):
        return a

def make_action_encoder(variant,action_dim=1,latent_action_dim=32):
    if variant=='none':
        return IdentityActionEncoder(action_dim=action_dim)
    elif variant=='linear':
        return LinearActionEncoder(action_dim=action_dim,latent_action_dim=latent_action_dim)
    elif variant=='mlp':
        return MLPActionEncoder(action_dim=action_dim,latent_action_dim=latent_action_dim)
    else:
        raise ValueError(f'Unknown action encoder variant: {variant!r}')

