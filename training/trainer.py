"""Training loop for all JEPA encoder variants."""
from __future__ import annotations
import csv,os,random,time,warnings
from pathlib import Path
from typing import Dict,Optional,Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.jepa import JEPAModel,JEPAConfig
from losses.prediction import combined_prediction_loss
from losses.pbh import pbh_stabilizability_loss
from losses.spectral import spectral_matching_loss
from losses.straightening import temporal_curvature_loss
from losses.nmp import nmp_zero_loss

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SlidingWindowDMDc:
    def __init__(self,window_size=1000,refit_every=50):
        self.window_size=window_size
        self.refit_every=refit_every
        self.Z_buf,self.A_buf,self.Z_next_buf=[],[],[]
        self.step_count=0
        self.A_hat=None
        self.B_hat=None
    def update(self,z,a,z_next):
        self.Z_buf.append(z)
        self.A_buf.append(a)
        self.Z_next_buf.append(z_next)
        if len(self.Z_buf)>self.window_size:
            self.Z_buf.pop(0)
            self.A_buf.pop(0)
            self.Z_next_buf.pop(0)
        self.step_count+=1
        if self.step_count%self.refit_every==0 and len(self.Z_buf)>32:
            self._refit()
    def _refit(self):
        from identification.dmdc import fit_dmdc
        Z=np.array(self.Z_buf)
        A=np.array(self.A_buf)
        Z_next=np.array(self.Z_next_buf)
        try:
            self.A_hat,self.B_hat=fit_dmdc(Z,A,Z_next)
        except Exception:
            pass

class Trainer:
    def __init__(self,model,config_dict,gt=None,save_dir='checkpoints',device=None,seed=42):
        self.model=model
        self.cfg=config_dict
        self.gt=gt
        self.save_dir=Path(save_dir)
        self.save_dir.mkdir(parents=True,exist_ok=True)
        self.device=device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.seed=seed
        set_all_seeds(seed)
        self.model.to(self.device)
        self.lambda_pred=float(self.cfg.get('lambda_pred',1.0))
        self.lambda_PBH=float(self.cfg.get('lambda_PBH',0.0))
        self.lambda_spec=float(self.cfg.get('lambda_spec',0.0))
        self.lambda_NMP=float(self.cfg.get('lambda_NMP',0.0))
        self.lambda_curv=float(self.cfg.get('lambda_curv',0.0))
        self.use_vicreg=bool(self.cfg.get('use_vicreg',True))
        self.vicreg_lambda=float(self.cfg.get('vicreg_lambda',25.0))
        self.vicreg_mu=float(self.cfg.get('vicreg_mu',25.0))
        self.vicreg_nu=float(self.cfg.get('vicreg_nu',1.0))
        lr=float(self.cfg.get('lr',1e-4))
        self.optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=int(self.cfg.get('epochs',100)),eta_min=lr*0.1)
        self.sliding_dmdc=SlidingWindowDMDc(int(self.cfg.get('dmdc_window',1000)),int(self.cfg.get('dmdc_refit_every',50)))
        self.true_unstable_eigs=None
        self.true_nmp_zeros=None
        self.C_hat_for_nmp=None
        if gt is not None:
            self.true_unstable_eigs=gt.unstable_eigenvalues
            self.true_nmp_zeros=gt.nmp_zeros
        self.log_path=self.save_dir/'training_log.csv'
        self._init_csv_log()
        self.best_val_loss=float('inf')
        self.global_step=0
        self.epoch=0

    def _init_csv_log(self):
        with open(self.log_path,'w',newline='') as f:
            csv.writer(f).writerow(['epoch','split','step','total_loss','pred_loss','pbh_loss','spec_loss','nmp_loss','curv_loss','vicreg_var','vicreg_cov'])

    def _log_csv(self,epoch,split,step,info):
        with open(self.log_path,'a',newline='') as f:
            csv.writer(f).writerow([epoch,split,step,info.get('total_loss',''),info.get('total_pred',''),info.get('pbh_loss',''),info.get('spec_loss',''),info.get('nmp_loss',''),info.get('curv_loss',''),info.get('vicreg_var',''),info.get('vicreg_cov','')])

    def _compute_loss(self,batch,is_train=True):
        obs=batch['obs'].to(self.device)
        action=batch['action'].to(self.device)
        next_obs=batch['next_obs'].to(self.device)
        outputs=self.model(obs,action,next_obs)
        z_t=outputs['z_t']
        a_t=outputs['a_t']
        z_next=outputs['z_next']
        pred_loss,pred_info=combined_prediction_loss(outputs,use_vicreg=self.use_vicreg,vicreg_lambda=self.vicreg_lambda,vicreg_mu=self.vicreg_mu,vicreg_nu=self.vicreg_nu)
        total_loss=self.lambda_pred*pred_loss
        info=dict(pred_info)
        info['total_loss']=total_loss.item()
        from models.action_encoder import MLPActionEncoder
        if isinstance(self.model.action_encoder,MLPActionEncoder):
            recon_loss=self.model.action_encoder.reconstruction_loss(action)
            total_loss=total_loss+recon_loss
            info['action_recon_loss']=recon_loss.item()
        if self.lambda_PBH>0 and is_train:
            try:
                _,pbh_info=pbh_stabilizability_loss(z_t.detach(),a_t.detach(),z_next.detach())
                pbh_loss_grad,_=pbh_stabilizability_loss(z_t,a_t,z_next)
                total_loss=total_loss+self.lambda_PBH*pbh_loss_grad
                info.update(pbh_info)
            except Exception as exc:
                warnings.warn(f'PBH loss failed: {exc}')
        if self.lambda_spec>0 and self.true_unstable_eigs is not None and is_train:
            try:
                spec_loss,spec_info=spectral_matching_loss(z_t,a_t,z_next,self.true_unstable_eigs)
                total_loss=total_loss+self.lambda_spec*spec_loss
                info.update(spec_info)
            except Exception as exc:
                warnings.warn(f'Spectral loss failed: {exc}')
        if self.lambda_NMP>0 and self.true_nmp_zeros is not None and self.C_hat_for_nmp is not None and is_train:
            try:
                nmp_l,nmp_info=nmp_zero_loss(z_t,a_t,z_next,self.C_hat_for_nmp,self.true_nmp_zeros)
                total_loss=total_loss+self.lambda_NMP*nmp_l
                info.update(nmp_info)
            except Exception as exc:
                warnings.warn(f'NMP loss failed: {exc}')
        info['total_loss']=total_loss.item()
        if is_train:
            with torch.no_grad():
                self.sliding_dmdc.update(z_t.detach().cpu().numpy(),a_t.detach().cpu().numpy(),z_next.detach().cpu().numpy())
        return total_loss,info

    def train_epoch(self,train_loader):
        self.model.train()
        total_metrics={}
        for batch in tqdm(train_loader,desc=f'Epoch {self.epoch} [train]',leave=False,dynamic_ncols=True):
            self.optimizer.zero_grad()
            loss,info=self._compute_loss(batch,is_train=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1.0)
            self.optimizer.step()
            for k,v in info.items():
                if isinstance(v,(int,float)):
                    total_metrics.setdefault(k,[]).append(v)
            self.global_step+=1
        return {k:float(np.mean(v)) for k,v in total_metrics.items()}

    @torch.no_grad()
    def val_epoch(self,val_loader):
        self.model.eval()
        total_metrics={}
        for batch in tqdm(val_loader,desc=f'Epoch {self.epoch} [val]',leave=False,dynamic_ncols=True):
            _,info=self._compute_loss(batch,is_train=False)
            for k,v in info.items():
                if isinstance(v,(int,float)):
                    total_metrics.setdefault(k,[]).append(v)
        return {k:float(np.mean(v)) for k,v in total_metrics.items()}

    def save_checkpoint(self,tag='latest'):
        path=self.save_dir/f'checkpoint_{tag}.pt'
        torch.save({'epoch':self.epoch,'global_step':self.global_step,'model_state':self.model.state_dict(),'optimizer_state':self.optimizer.state_dict(),'best_val_loss':self.best_val_loss,'config':self.model.get_config_dict()},path)

    def load_checkpoint(self,path):
        ckpt=torch.load(path,map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.epoch=ckpt.get('epoch',0)
        self.global_step=ckpt.get('global_step',0)
        self.best_val_loss=ckpt.get('best_val_loss',float('inf'))

    def fit(self,train_loader,val_loader,epochs=None,checkpoint_every=10):
        if epochs is None:
            epochs=int(self.cfg.get('epochs',100))
        history={'train':[],'val':[]}
        for epoch in range(epochs):
            self.epoch=epoch
            t0=time.time()
            train_metrics=self.train_epoch(train_loader)
            val_metrics=self.val_epoch(val_loader)
            self.scheduler.step()
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)
            val_loss=val_metrics.get('total_loss',float('inf'))
            if val_loss<self.best_val_loss:
                self.best_val_loss=val_loss
                self.save_checkpoint('best')
            if (epoch+1)%checkpoint_every==0:
                self.save_checkpoint(f'epoch_{epoch+1:04d}')
            self.save_checkpoint('latest')
            self._log_csv(epoch,'train',self.global_step,train_metrics)
            self._log_csv(epoch,'val',self.global_step,val_metrics)
            dt=time.time()-t0
            print(f'[Epoch {epoch+1:3d}/{epochs}] train_loss={train_metrics.get("total_loss",0):.4f}  val_loss={val_loss:.4f}  lr={self.optimizer.param_groups[0]["lr"]:.2e}  dt={dt:.1f}s')
        return history

    def post_training_dmdc(self,full_loader,Y_loader=None):
        from identification.dmdc import DMDcFitter
        self.model.eval()
        Z_list,A_list,Z_next_list=[],[],[]
        with torch.no_grad():
            for batch in tqdm(full_loader,desc='DMDc encoding'):
                obs=batch['obs'].to(self.device)
                action=batch['action'].to(self.device)
                next_obs=batch['next_obs'].to(self.device)
                z=self.model.encoder(obs).cpu().numpy()
                a=self.model.action_encoder(action).cpu().numpy()
                z_next=self.model.encoder(next_obs).cpu().numpy()
                Z_list.append(z)
                A_list.append(a)
                Z_next_list.append(z_next)
        Z=np.vstack(Z_list)
        A=np.vstack(A_list)
        Z_next=np.vstack(Z_next_list)
        fitter=DMDcFitter(use_proximal=True)
        fitter.fit(Z,A,Z_next)
        print(fitter.summary())
        return fitter.A_hat,fitter.B_hat,fitter
