"""Full JEPA world model combining encoder + predictor + action encoder."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Optional,Dict,Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import VisualEncoder
from models.predictor import MLPPredictor
from models.action_encoder import make_action_encoder,IdentityActionEncoder

@dataclass
class JEPAConfig:
    variant:str='E-noact'
    latent_dim:int=32
    action_dim:int=1
    action_latent_dim:int=32
    encoder_channels:list=field(default_factory=lambda:[32,64,128,256])
    predictor_hidden_dim:int=256
    image_size:int=64
    normalize_latent:bool=True
    lambda_pred:float=1.0
    lambda_PBH:float=0.0
    lambda_spec:float=0.0
    lambda_NMP:float=0.0
    lambda_curv:float=0.0
    use_vicreg:bool=True
    vicreg_lambda:float=25.0
    vicreg_mu:float=25.0
    vicreg_nu:float=1.0

    @classmethod
    def from_variant(cls,variant,**kwargs):
        presets={
            'E-noact':dict(lambda_pred=1.0,lambda_PBH=0.0,lambda_spec=0.0,action_latent_dim=1),
            'E-spec':dict(lambda_pred=1.0,lambda_PBH=0.0,lambda_spec=1e-2,action_latent_dim=1),
            'E-PBH':dict(lambda_pred=1.0,lambda_PBH=1e-2,lambda_spec=0.0,action_latent_dim=1),
            'E-both-r':dict(lambda_pred=1.0,lambda_PBH=1e-2,lambda_spec=1e-2,action_latent_dim=1),
            'E-lift':dict(lambda_pred=1.0,lambda_PBH=0.0,lambda_spec=0.0,action_latent_dim=32),
            'E-full':dict(lambda_pred=1.0,lambda_PBH=1e-2,lambda_spec=1e-2,action_latent_dim=32),
        }
        if variant not in presets:
            raise ValueError(f'Unknown variant {variant!r}. Choose from {list(presets)}')
        cfg=cls(variant=variant)
        for k,v in presets[variant].items():
            setattr(cfg,k,v)
        for k,v in kwargs.items():
            setattr(cfg,k,v)
        return cfg

    @property
    def action_encoder_type(self):
        if self.variant in ('E-lift','E-full'):
            return 'linear'
        return 'none'

class JEPAModel(nn.Module):
    """Full JEPA world model."""
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.encoder=VisualEncoder(
            latent_dim=config.latent_dim,
            channels=config.encoder_channels,
            image_size=config.image_size,
            normalize=config.normalize_latent,
        )
        self.action_encoder=make_action_encoder(
            variant=config.action_encoder_type,
            action_dim=config.action_dim,
            latent_action_dim=config.action_latent_dim,
        )
        self.predictor=MLPPredictor(
            latent_dim=config.latent_dim,
            action_dim=self.action_encoder.latent_action_dim,
            hidden_dim=config.predictor_hidden_dim,
        )

    @property
    def latent_dim(self):
        return self.config.latent_dim

    @property
    def action_latent_dim(self):
        return self.action_encoder.latent_action_dim

    def forward(self,obs,action,next_obs):
        z_t=self.encoder(obs)
        a_t=self.action_encoder(action)
        z_hat=self.predictor(z_t,a_t)
        with torch.no_grad():
            z_next_sg=self.encoder(next_obs)
        z_next=self.encoder(next_obs)
        return {'z_t':z_t,'a_t':a_t,'z_hat':z_hat,'z_next':z_next,'z_next_sg':z_next_sg}

    @torch.no_grad()
    def encode_batch(self,obs,batch_size=256):
        device=next(self.parameters()).device
        zs=[]
        for i in range(0,len(obs),batch_size):
            chunk=obs[i:i+batch_size].to(device)
            zs.append(self.encoder(chunk).cpu())
        return torch.cat(zs,dim=0)

    @torch.no_grad()
    def encode_actions_batch(self,actions,batch_size=1024):
        device=next(self.parameters()).device
        aes=[]
        for i in range(0,len(actions),batch_size):
            chunk=actions[i:i+batch_size].to(device)
            aes.append(self.action_encoder(chunk).cpu())
        return torch.cat(aes,dim=0)

    def decode_action(self,a):
        return self.action_encoder.decode(a)

    def get_config_dict(self):
        return {'variant':self.config.variant,'latent_dim':self.config.latent_dim,
                'action_dim':self.config.action_dim,'action_latent_dim':self.config.action_latent_dim,
                'action_encoder_type':self.config.action_encoder_type,
                'lambda_pred':self.config.lambda_pred,'lambda_PBH':self.config.lambda_PBH,
                'lambda_spec':self.config.lambda_spec}

def make_jepa(variant,**kwargs):
    config=JEPAConfig.from_variant(variant,**kwargs)
    return JEPAModel(config)
