"""Dataset generation and loading for JEPA cartpole experiments."""
from __future__ import annotations
import os,warnings
from pathlib import Path
from typing import Dict,Optional,Tuple
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

def _compute_lqr_gain():
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    import scipy.linalg
    gt=CartpoleGroundTruth()
    A,B=gt.A_star,gt.B_star
    Q=np.diag([1.0,1.0,10.0,1.0])
    R=np.array([[0.01]])
    try:
        P=scipy.linalg.solve_discrete_are(A,B,Q,R)
        K=np.linalg.inv(R+B.T@P@B)@B.T@P@A
    except Exception:
        warnings.warn('DARE failed; falling back to pole-placement gain.')
        K=np.array([[0.0,0.0,10.0,0.0]])
    return K

def _collect_transitions(env,n_transitions,mode,lqr_gain,action_low,action_high,init_range,lqr_noise_std,rng):
    obs_list,state_list,action_list,next_obs_list,next_state_list=[],[],[],[],[]
    obs,state,_=env.reset(init_range=init_range)
    steps_since_reset=0
    collected=0
    while collected<n_transitions:
        if mode=='random':
            u=float(rng.uniform(action_low,action_high))
        elif mode=='lqr':
            u_lqr=float(np.clip(float(lqr_gain@state),action_low,action_high))
            u=float(np.clip(u_lqr+float(rng.normal(0.0,lqr_noise_std)),action_low,action_high))
        else:
            raise ValueError(f'Unknown mode: {mode}')
        next_obs,next_state,_,done,_=env.step(u)
        obs_list.append(obs.copy())
        state_list.append(state.copy())
        action_list.append(np.array([u],dtype=np.float32))
        next_obs_list.append(next_obs.copy())
        next_state_list.append(next_state.copy())
        collected+=1
        steps_since_reset+=1
        if done or steps_since_reset>200:
            obs,state,_=env.reset(init_range=init_range)
            steps_since_reset=0
        else:
            obs,state=next_obs,next_state
    return {'obs':np.stack(obs_list).astype(np.uint8),'states':np.stack(state_list).astype(np.float32),'actions':np.stack(action_list).astype(np.float32),'next_obs':np.stack(next_obs_list).astype(np.uint8),'next_states':np.stack(next_state_list).astype(np.float32)}

def generate_dataset(dataset_type='random',n_transitions=50000,frame_skip=1,save_path=None,seed=42,train_frac=0.8,val_frac=0.1,action_range=(-5.0,5.0),init_range=0.1,lqr_noise_std=0.1,image_size=64):
    from envs.cartpole_visual import ContinuousCartpoleVisual
    rng=np.random.RandomState(seed)
    lqr_gain=_compute_lqr_gain() if dataset_type in ('lqr','mixed') else None
    env=ContinuousCartpoleVisual(frame_skip=frame_skip,image_size=image_size,action_range=(-10.0,10.0),seed=seed)
    action_low,action_high=action_range
    if dataset_type=='mixed':
        n_random=n_transitions//2
        n_lqr=n_transitions-n_random
        data_rand=_collect_transitions(env,n_random,'random',lqr_gain,action_low,action_high,init_range=0.1,lqr_noise_std=lqr_noise_std,rng=rng)
        data_lqr=_collect_transitions(env,n_lqr,'lqr',lqr_gain,action_low,action_high,init_range=0.05,lqr_noise_std=lqr_noise_std,rng=rng)
        data={key:np.concatenate([data_rand[key],data_lqr[key]],axis=0) for key in data_rand}
    else:
        init_r=0.05 if dataset_type=='lqr' else init_range
        data=_collect_transitions(env,n_transitions,dataset_type,lqr_gain,action_low,action_high,init_range=init_r,lqr_noise_std=lqr_noise_std,rng=rng)
    env.close()
    N=data['obs'].shape[0]
    perm=rng.permutation(N)
    for key in data:
        data[key]=data[key][perm]
    actions=data['actions']
    action_cov=np.cov(actions.T)
    if action_cov.ndim==0:
        action_cov=float(action_cov)
        kappa=1.0
    else:
        eigvals=np.linalg.eigvalsh(action_cov)
        eigvals_pos=np.maximum(eigvals,1e-12)
        kappa=float(eigvals_pos.max()/eigvals_pos.min())
    n_train=int(N*train_frac)
    n_val=int(N*val_frac)
    splits={'train':np.arange(0,n_train),'val':np.arange(n_train,n_train+n_val),'test':np.arange(n_train+n_val,N)}
    data['splits']=splits
    data['action_cov']=np.atleast_2d(action_cov)
    data['action_cov_condition_number']=kappa
    if save_path is not None:
        _save_hdf5(data,save_path,splits)
    return data

def _save_hdf5(data,path,splits):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.',exist_ok=True)
    with h5py.File(path,'w') as f:
        for key in ['obs','states','actions','next_obs','next_states']:
            f.create_dataset(key,data=data[key],compression='gzip',compression_opts=4)
        f.create_dataset('action_cov',data=data['action_cov'])
        f.attrs['action_cov_condition_number']=float(data['action_cov_condition_number'])
        grp=f.create_group('splits')
        for split_name,idx in splits.items():
            grp.create_dataset(split_name,data=idx)

def load_dataset(path):
    data={}
    with h5py.File(path,'r') as f:
        for key in ['obs','states','actions','next_obs','next_states','action_cov']:
            if key in f:
                data[key]=f[key][:]
        data['action_cov_condition_number']=float(f.attrs.get('action_cov_condition_number',1.0))
        splits={}
        if 'splits' in f:
            for split_name in f['splits']:
                splits[split_name]=f['splits'][split_name][:]
        data['splits']=splits
    return data

class TransitionDataset(Dataset):
    def __init__(self,data,split='train'):
        splits=data.get('splits',{})
        idx=splits[split] if splits and split in splits else np.arange(len(data['obs']))
        self.obs=data['obs'][idx]
        self.states=data['states'][idx]
        self.actions=data['actions'][idx]
        self.next_obs=data['next_obs'][idx]
        self.next_states=data['next_states'][idx]
    def __len__(self):
        return len(self.obs)
    def __getitem__(self,idx):
        obs=torch.from_numpy(self.obs[idx]).float().permute(2,0,1)/255.0
        next_obs=torch.from_numpy(self.next_obs[idx]).float().permute(2,0,1)/255.0
        return {'obs':obs,'state':torch.from_numpy(self.states[idx]),'action':torch.from_numpy(self.actions[idx]),'next_obs':next_obs,'next_state':torch.from_numpy(self.next_states[idx])}

def make_dataloaders(data,batch_size=256,num_workers=0):
    loaders={}
    for split in ('train','val','test'):
        if split in data.get('splits',{}):
            ds=TransitionDataset(data,split=split)
            loaders[split]=DataLoader(ds,batch_size=batch_size,shuffle=(split=='train'),num_workers=num_workers,pin_memory=True,drop_last=(split=='train'))
    return loaders
