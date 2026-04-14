"""Temporal curvature (straightening) regularisation."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def temporal_curvature_loss(z_seq,eps=1e-8):
    if isinstance(z_seq,(list,tuple)):
        assert len(z_seq)==3
        z_t,z_t1,z_t2=z_seq
        v1=z_t1-z_t
        v2=z_t2-z_t1
        cos_sim=F.cosine_similarity(v1,v2,dim=-1,eps=eps)
        return (1.0-cos_sim).mean()
    if z_seq.dim()==2:
        z_seq=z_seq.unsqueeze(0)
    B,T,d=z_seq.shape
    if T<3:
        return torch.tensor(0.0,device=z_seq.device)
    v1=z_seq[:,1:T-1,:]-z_seq[:,0:T-2,:]
    v2=z_seq[:,2:T,:]-z_seq[:,1:T-1,:]
    v1_flat=v1.reshape(-1,d)
    v2_flat=v2.reshape(-1,d)
    cos_sim=F.cosine_similarity(v1_flat,v2_flat,dim=-1,eps=eps)
    return (1.0-cos_sim).mean()

def straightening_loss_from_outputs(outputs_t,outputs_t1,outputs_t2):
    z_t=outputs_t["z_t"]
    z_t1=outputs_t1["z_t"]
    z_t2=outputs_t2["z_t"]
    return temporal_curvature_loss([z_t,z_t1,z_t2])
