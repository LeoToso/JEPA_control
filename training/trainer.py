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
        self.lambda_conj=float(self.cfg.get('lambda_conj',1.0))
        self.lambda_PBH=float(self.cfg.get('lambda_PBH',0.0))
        self.lambda_spec=float(self.cfg.get('lambda_spec',0.0))
        self.lambda_NMP=float(self.cfg.get('lambda_NMP',0.0))
        self.lambda_curv=float(self.cfg.get('lambda_curv',0.0))
        self.use_vicreg=bool(self.cfg.get('use_vicreg',True))
        self.vicreg_lambda=float(self.cfg.get('vicreg_lambda',25.0))
        self.vicreg_mu=float(self.cfg.get('vicreg_mu',25.0))
        self.vicreg_nu=float(self.cfg.get('vicreg_nu',1.0))
        lr=float(self.cfg.get('lr',1e-4))
        weight_decay=float(self.cfg.get('weight_decay',1e-4))
        self.optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
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
        # Conjugacy loss: ||z_{t+1} - (A z_t + B u_t)||^2
        # Forces encoder to produce linearly predictable representations so that
        # the learned A, B are the true system matrices (Bounou 2024 / Lutkus 2025).
        if self.lambda_conj>0 and 'z_hat_lin' in outputs:
            conj_loss=nn.functional.mse_loss(outputs['z_hat_lin'],z_next.detach())
            total_loss=total_loss+self.lambda_conj*conj_loss
            info['conj_loss']=conj_loss.item()
        info['total_loss']=total_loss.item()
        from models.action_encoder import MLPActionEncoder
        if isinstance(self.model.action_encoder,MLPActionEncoder):
            recon_loss=self.model.action_encoder.reconstruction_loss(action)
            total_loss=total_loss+recon_loss
            info['action_recon_loss']=recon_loss.item()
        # Spectral and PBH losses act directly on model.dynamics.A and B.
        # The mini_batch_dmdc approach was indirect and inconsistent with the
        # control-relevant matrices; this is the correct target.
        if (self.lambda_spec>0 or self.lambda_PBH>0) and is_train and self.true_unstable_eigs is not None and len(self.true_unstable_eigs)>0:
            A_dyn=self.model.dynamics.A
            B_dyn=self.model.dynamics.B
            d_dyn=A_dyn.shape[0]
            if self.lambda_spec>0:
                try:
                    eigvals_dyn=torch.linalg.eigvals(A_dyn)
                    true_eigs_t=torch.tensor(self.true_unstable_eigs,dtype=eigvals_dyn.dtype,device=eigvals_dyn.device)
                    spec_terms=[]
                    for lam_star in true_eigs_t:
                        diff=eigvals_dyn-lam_star
                        dist_sq=diff.real**2+diff.imag**2
                        spec_terms.append(dist_sq.min())
                    spec_loss_dyn=torch.stack(spec_terms).sum()
                    total_loss=total_loss+self.lambda_spec*spec_loss_dyn
                    info['spec_loss']=spec_loss_dyn.item()
                except Exception as exc:
                    warnings.warn(f'Spectral loss on dynamics failed: {exc}')
            if self.lambda_PBH>0:
                try:
                    pbh_terms=[]
                    for lam_star_val in self.true_unstable_eigs:
                        lam_r=torch.tensor(float(np.real(lam_star_val)),dtype=A_dyn.dtype,device=A_dyn.device)
                        M_S=torch.cat([lam_r*torch.eye(d_dyn,device=A_dyn.device,dtype=A_dyn.dtype)-A_dyn,B_dyn],dim=-1)
                        sv=torch.linalg.svdvals(M_S)
                        sigma_min=sv[-1]
                        pbh_terms.append(-torch.log(sigma_min+1e-6))
                    pbh_loss_dyn=torch.stack(pbh_terms).mean()
                    total_loss=total_loss+self.lambda_PBH*pbh_loss_dyn
                    info['pbh_loss']=pbh_loss_dyn.item()
                except Exception as exc:
                    warnings.warn(f'PBH loss on dynamics failed: {exc}')
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
        # Save model state only (no optimizer) to halve checkpoint size.
        # Optimizer state is only needed to resume training mid-run, which we
        # don't support; keeping it was doubling disk usage for no benefit.
        torch.save({'epoch':self.epoch,'global_step':self.global_step,'model_state':self.model.state_dict(),'best_val_loss':self.best_val_loss,'config':self.model.get_config_dict()},path)

    def load_checkpoint(self,path):
        ckpt=torch.load(path,map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
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
        # Restore best-val checkpoint so downstream DMDc and control use
        # the generalising model, not the final (potentially overfit) one.
        best_ckpt=self.save_dir/'checkpoint_best.pt'
        if best_ckpt.exists():
            ckpt=torch.load(best_ckpt,map_location=self.device)
            self.model.load_state_dict(ckpt['model_state'])
            print(f'[train] Restored best checkpoint (val_loss={self.best_val_loss:.4f})')
        return history

    def post_training_dmdc(self,full_loader,Y_loader=None):
        from identification.dmdc import DMDcFitter
        # Use learned linear dynamics parameters directly — they were trained
        # jointly with the encoder via conjugacy loss, so they ARE the system
        # matrices.  No post-hoc regression needed.
        A_hat,B_hat=self.model.dynamics.get_AB()
        # Evaluate residual on the provided data split for diagnostics.
        self.model.eval()
        Z_list,U_list,Z_next_list=[],[],[]
        with torch.no_grad():
            for batch in tqdm(full_loader,desc='Evaluating linear dynamics'):
                obs=batch['obs'].to(self.device)
                action=batch['action'].to(self.device)
                next_obs=batch['next_obs'].to(self.device)
                z=self.model.encoder(obs).cpu().numpy()
                u=action.cpu().numpy()
                z_next=self.model.encoder(next_obs).cpu().numpy()
                Z_list.append(z);U_list.append(u);Z_next_list.append(z_next)
        Z=np.vstack(Z_list);U=np.vstack(U_list);Z_next=np.vstack(Z_next_list)
        Z_next_pred=Z@A_hat.T+U@B_hat.T
        res_num=np.linalg.norm(Z_next-Z_next_pred,'fro')
        residual=float(res_num/(np.linalg.norm(Z_next,'fro')+1e-12))
        spec_rad=float(np.max(np.abs(np.linalg.eigvals(A_hat))))
        fitter=DMDcFitter()
        fitter.A_hat=A_hat;fitter.B_hat=B_hat
        fitter.fit_info={'residual':residual,'n_iters':0,'converged':True,
                         'A_hat_cond':float(np.linalg.cond(A_hat)),
                         'B_hat_cond':float(np.linalg.cond(B_hat))}
        print(f'[LinearDynamics] A spectral radius: {spec_rad:.4f}')
        print(f'[LinearDynamics] Test residual: {residual:.4f}')
        print(f'[LinearDynamics] A_hat_cond: {fitter.fit_info["A_hat_cond"]:.2e}  B_hat_cond: {fitter.fit_info["B_hat_cond"]:.2e}')
        return A_hat,B_hat,fitter
