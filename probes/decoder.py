"""Action decoder fidelity probe D1."""
from __future__ import annotations
import warnings
from typing import Dict,Any,Optional
import numpy as np
import torch

def D1_action_decoder_fidelity(action_encoder,env=None,n_test=1000,lqr_actions=None,action_low=-5.0,action_high=5.0,epsilon_oom=0.01,device=None):
    from models.action_encoder import LinearActionEncoder,MLPActionEncoder,IdentityActionEncoder
    if device is None:
        try:
            device=next(action_encoder.parameters()).device
        except StopIteration:
            device=torch.device('cpu')
    rng=np.random.RandomState(42)
    u_test=rng.uniform(action_low,action_high,size=(n_test,action_encoder.action_dim)).astype(np.float32)
    u_tensor=torch.from_numpy(u_test).to(device)
    result={}
    if isinstance(action_encoder,IdentityActionEncoder):
        result['reconstruction_error']=0.0
        result['decoder_lipschitz']=1.0
        result['out_of_image_rate']=0.0
        return result
    elif isinstance(action_encoder,LinearActionEncoder):
        action_encoder.compute_pseudoinverse()
        with torch.no_grad():
            a_test=action_encoder(u_tensor)
            u_recon=action_encoder.decode(a_test)
        errors=torch.norm(u_tensor-u_recon,dim=-1).cpu().numpy()
        result['reconstruction_error']=float(np.mean(errors))
        W_pinv=action_encoder.W_pinv
        sv=torch.linalg.svdvals(W_pinv)
        result['decoder_lipschitz']=float(sv.max().item())
        result['condition_number_W']=action_encoder.condition_number
        W=action_encoder.W.weight.data
        check=W_pinv@W
        I_m=torch.eye(action_encoder.action_dim,device=device)
        result['pinv_identity_error']=float(torch.norm(check-I_m).item())
        result['out_of_image_rate']=0.0
    elif isinstance(action_encoder,MLPActionEncoder):
        with torch.no_grad():
            a_test=action_encoder(u_tensor)
            u_recon=action_encoder.decode(a_test)
        errors=torch.norm(u_tensor-u_recon,dim=-1).cpu().numpy()
        result['reconstruction_error']=float(np.mean(errors))
        if lqr_actions is not None:
            lqr_t=torch.from_numpy(lqr_actions.astype(np.float32)).to(device)
            L_Psi=action_encoder.lipschitz_estimate(lqr_t)
        else:
            L_Psi=action_encoder.lipschitz_estimate(u_tensor[:200])
        result['decoder_lipschitz']=L_Psi
        if lqr_actions is not None:
            lqr_t=torch.from_numpy(lqr_actions.astype(np.float32)).to(device)
            with torch.no_grad():
                a_lqr=action_encoder(lqr_t)
                u_lqr_recon=action_encoder.decode(a_lqr)
                a_lqr_recon=action_encoder(u_lqr_recon)
                oom_errors=torch.norm(a_lqr-a_lqr_recon,dim=-1)
            result['out_of_image_rate']=float((oom_errors>epsilon_oom).float().mean().item())
        else:
            result['out_of_image_rate']=float('nan')
    else:
        warnings.warn(f'Unknown action encoder type: {type(action_encoder)}')
        result['reconstruction_error']=float('nan')
        result['decoder_lipschitz']=float('nan')
        result['out_of_image_rate']=float('nan')
    return result
