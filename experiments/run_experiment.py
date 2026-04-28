"""Single experiment runner."""
from __future__ import annotations
import os,sys,json,time,warnings,random
from pathlib import Path
from typing import Dict,Any,Optional
# Ensure project root is on sys.path regardless of working directory.
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import yaml

def run_single_experiment(encoder_variant,dataset_name,frame_skip=1,seed=42,config_path='configs/cartpole.yaml',data_dir='data',results_dir='results',device=None,skip_if_exists=True,eval_only=False):
    exp_name=f'{encoder_variant}_{dataset_name}_fs{frame_skip}_seed{seed}'
    out_dir=Path(results_dir)/exp_name
    out_dir.mkdir(parents=True,exist_ok=True)
    results_file=out_dir/'results.json'
    if skip_if_exists and results_file.exists():
        print(f'[skip] {exp_name} already exists.')
        with open(results_file) as f:
            return json.load(f)
    print(f'\n{"="*60}\nEXPERIMENT: {exp_name}\n{"="*60}')
    t_start=time.time()
    if device is None:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with open(config_path) as f:
        cfg=yaml.safe_load(f)
    env_cfg=cfg['environment']
    model_cfg=cfg['model']
    train_cfg=cfg['training']
    probe_cfg=cfg['probes']
    ctrl_cfg=cfg['control']
    env_cfg['frame_skip']=frame_skip
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    gt=CartpoleGroundTruth(mass_cart=env_cfg['mass_cart'],mass_pole=env_cfg['mass_pole'],pole_length=env_cfg['pole_length'],gravity=env_cfg['gravity'],dt=env_cfg['dt']*frame_skip)
    print(f'\n[GT] {gt.summary()}')
    val_result=gt.validate_linearization()
    print(f'[GT] Linearisation validation: {val_result}')
    from data.dataset import load_dataset,make_dataloaders,generate_dataset
    h5_path=Path(data_dir)/f'cartpole_{dataset_name}_fs{frame_skip}_seed{seed}.h5'
    if h5_path.exists():
        print(f'\n[data] Loading {h5_path}')
        data=load_dataset(str(h5_path))
    else:
        print(f'\n[data] Generating {dataset_name} dataset...')
        data=generate_dataset(dataset_type=dataset_name,n_transitions=cfg['data']['n_random'],frame_skip=frame_skip,save_path=str(h5_path),seed=seed,image_size=env_cfg['image_size'])
    kappa=data.get('action_cov_condition_number',float('nan'))
    print(f'[data] Action covariance condition number: {kappa:.2f}')
    if kappa>1000:
        warnings.warn('kappa > 1000: B_hat estimate may be unreliable!')
    loaders=make_dataloaders(data,batch_size=train_cfg['batch_size'])
    from models.jepa import make_jepa,JEPAConfig
    model=make_jepa(encoder_variant,latent_dim=model_cfg['latent_dim'],action_latent_dim=model_cfg['action_latent_dim'],encoder_channels=model_cfg['encoder_channels'],predictor_hidden_dim=model_cfg['predictor_hidden_dim'],image_size=env_cfg['image_size'])
    model.to(device)
    n_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n[model] {encoder_variant} - {n_params:,} trainable parameters')
    from training.trainer import Trainer
    # lambda_spec and lambda_PBH now act directly on model.dynamics.A/B
    # (not mini_batch_dmdc), so the effective gradient scale is ~1 per unit.
    # Use 1.0 for spectral (matches conjugacy loss scale) and 0.5 for PBH.
    variant_weights={'E-noact':dict(lambda_pred=1.0,lambda_PBH=0.0,lambda_spec=0.0),'E-spec':dict(lambda_pred=1.0,lambda_PBH=0.0,lambda_spec=1.0),'E-PBH':dict(lambda_pred=1.0,lambda_PBH=1.0,lambda_spec=0.0),'E-both-r':dict(lambda_pred=1.0,lambda_PBH=1.0,lambda_spec=1.0),'E-lift':dict(lambda_pred=1.0,lambda_PBH=0.0,lambda_spec=0.0),'E-full':dict(lambda_pred=1.0,lambda_PBH=1.0,lambda_spec=1.0)}
    train_cfg_exp=dict(train_cfg)
    train_cfg_exp.update(variant_weights.get(encoder_variant,{}))
    trainer=Trainer(model=model,config_dict=train_cfg_exp,gt=gt,save_dir=str(out_dir/'checkpoints'),device=device,seed=seed)
    # --eval-only: load saved model weights and skip training.
    # Useful when re-running just DMDc/probes/control after a hyperparameter change.
    saved_model=out_dir/'model_final.pt'
    if eval_only and saved_model.exists():
        print(f'[train] --eval-only: loading {saved_model}')
        # strict=False: W_pinv may be present in old checkpoints (saved after
        # compute_pseudoinverse()); it is ignored here and recomputed below.
        model.load_state_dict(torch.load(saved_model,map_location=device),strict=False)
        history={'train':[],'val':[]}
    else:
        print(f'\n[train] Starting training for {train_cfg_exp["epochs"]} epochs...')
        history=trainer.fit(loaders['train'],loaders['val'],epochs=train_cfg_exp['epochs'],checkpoint_every=train_cfg_exp.get('checkpoint_every',10))
    from models.action_encoder import LinearActionEncoder
    if isinstance(model.action_encoder,LinearActionEncoder):
        model.action_encoder.compute_pseudoinverse()
        ipe=float(torch.norm(model.action_encoder.W_pinv@model.action_encoder.W.weight.data-torch.eye(1,device=device)).item())
        print(f'[model] W_pinv @ W identity error: {ipe:.2e}')
    print('\n[DMDc] Fitting latent system on test set...')
    A_hat,B_hat,dmdc_fitter=trainer.post_training_dmdc(loaders['test'])
    # Fully-observed latent model: C_hat = I_d (latent state IS the observation).
    d=A_hat.shape[0]
    C_hat=np.eye(d)
    print(f'[DMDc] Using C_hat = I_{d} (fully-observed latent model)')
    print('\n[rollout] Generating probe rollout data...')
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env=ContinuousCartpoleVisual(frame_skip=frame_skip,image_size=env_cfg['image_size'],mass_cart=env_cfg['mass_cart'],mass_pole=env_cfg['mass_pole'],pole_length=env_cfg['pole_length'],gravity=env_cfg['gravity'],dt=env_cfg['dt'],seed=seed)
    rollout_data=_generate_rollout_data(model,env,gt,n_rollouts=probe_cfg['n_rollouts_probes'],T=probe_cfg['T_rollout'],device=device,A_hat=A_hat,B_hat=B_hat,C_hat=C_hat)
    paired_data=_generate_paired_data(model,env,n_pairs=1000,device=device)
    print('\n[probes] Running probe suite...')
    from probes.suite import run_all_probes
    probe_results=run_all_probes(model=model,encoder_variant=encoder_variant,dataset=data,gt=gt,rollout_data=rollout_data,paired_data=paired_data,env=env,device=device,config=probe_cfg)
    print('\n[control] Running control validation...')
    from control.lqr import solve_discrete_lqr
    from control.rollout import evaluate_stabilization
    d_u=B_hat.shape[1]
    # Fully-observed latent model: Q = I_d, uniform cost on all latent dimensions.
    Q_lqr=np.eye(d)
    R_lqr=float(ctrl_cfg.get('R_lqr',0.01))*np.eye(d_u)
    print(f'[control] Q_lqr=I_{d} (fully-observed)  R={R_lqr[0,0]:.4f}')
    # Compute z_star: latent encoding of the upright equilibrium image.
    # The LQR control law u = -K(z - z_star) requires this offset so the
    # controller drives the system to the physical equilibrium, not z=0.
    model.eval()
    obs_eq,_,_=env.reset_to_state(np.zeros(4))
    obs_eq_t=torch.from_numpy(obs_eq).float().permute(2,0,1)[None].to(device)/255.0
    with torch.no_grad():
        z_star=model.encoder(obs_eq_t).cpu().numpy()[0]
    print(f'[control] z_star norm: {np.linalg.norm(z_star):.3f}')
    ctrl_results={}
    try:
        from control.lqr import pre_stabilize_A
        # Pre-stabilise: deflate phantom eigenvalues before DARE so K_hat doesn't blow up.
        A_dare,n_def=pre_stabilize_A(A_hat,gt.unstable_eigenvalues,tol=0.05,target=0.9)
        if n_def>0:
            rho_dare=float(np.max(np.abs(np.linalg.eigvals(A_dare))))
            print(f'[control] Pre-stabilised {n_def} phantom mode(s); A_dare rho={rho_dare:.4f}')
        # Solve DARE on the pre-stabilised system; cl_eigs are on A_dare.
        K_hat,P_hat,cl_eigs_dare=solve_discrete_lqr(A_dare,B_hat,Q_lqr,R_lqr)
        max_cl_dare=float(np.max(np.abs(cl_eigs_dare)))
        print(f'[control] Max |cl_eig| (pre-stab A): {max_cl_dare:.4f}')
        print(f'[control] K_hat norm: {np.linalg.norm(K_hat):.3f}')
        if max_cl_dare>=1.0:
            warnings.warn(
                f'LQR failed on pre-stabilised system (max|cl_eig|={max_cl_dare:.4f} >= 1).\n'
                f'Likely cause: near-uncontrollable physical unstable mode (mu_S={probe_results.get("P2_1",{}).get("mu_S","?")}).\n'
                f'Try: --dataset mixed.'
            )
            ctrl_results={'success_rate':0.0,'error':'lqr_unstable','max_cl_eig_dare':max_cl_dare,'n_trials':probe_cfg['n_trials_control']}
        else:
            # Sign check on K_hat before building augmented gain.
            model.eval()
            sign_votes=[]
            for _theta,_label in [(+0.05,'right'),(-0.05,'left')]:
                _obs,_,_=env.reset_to_state(np.array([0.,0.,_theta,0.]))
                _obs_t=torch.from_numpy(_obs).float().permute(2,0,1)[None].to(device)/255.0
                with torch.no_grad():
                    _z=model.encoder(_obs_t).cpu().numpy()[0]
                _dz=_z-z_star
                _u=float(-K_hat@_dz)
                _sign_ok=(_u>0 if _theta>0 else _u<0)
                sign_votes.append(_sign_ok)
                print(f'[control] Sign diag  theta={_theta:+.3f}({_label}): u={_u:.4f}  sign_ok={_sign_ok}')
            n_ok=sum(sign_votes)
            if n_ok==0:
                K_hat=-K_hat
                print('[control] SIGN FLIP: negating K_hat')
            elif n_ok==1:
                print('[control] WARNING: mixed sign votes (1/2) — no flip')
            else:
                print('[control] Sign checks passed (2/2)')
            # PD augmented gain: u_t = -K(z_t-z*) - γ·K(z_t-z_{t-1})
            # Equivalent to K_aug = [(1+γ)·K, -γ·K] on aug state [dz_t; dz_{t-1}].
            # DARE on singular A_aug gives K₂≈0 so we construct the gain manually.
            gamma_pd=float(ctrl_cfg.get('pd_gamma',5.0))
            K_aug=np.hstack([(1+gamma_pd)*K_hat,-gamma_pd*K_hat])
            print(f'[control] PD gain: gamma={gamma_pd}, K_aug norm={np.linalg.norm(K_aug):.3f}')
            ctrl_results=evaluate_stabilization(encoder=model.encoder,A_hat=A_hat,B_hat=B_hat,K_hat=K_aug,env=env,n_trials=probe_cfg['n_trials_control'],T=probe_cfg['T_rollout'],seed=seed,device=device,z_star=z_star,init_scale=0.05)
            print(f'[control] Success rate: {ctrl_results["success_rate"]:.3f}')
    except Exception as exc:
        warnings.warn(f'Control validation failed: {exc}')
        ctrl_results={'error':str(exc)}
    env.close()
    results={'experiment':{'encoder_variant':encoder_variant,'dataset_name':dataset_name,'frame_skip':frame_skip,'seed':seed,'exp_name':exp_name},'gt_validation':val_result,'data_quality':{'action_cov_condition_number':kappa,'n_train':int(len(data['splits'].get('train',[]))),'n_test':int(len(data['splits'].get('test',[])))}, 'dmdc':{'residual':float(dmdc_fitter.fit_info.get('residual',float('nan'))),'A_hat_spectral_radius':float(np.max(np.abs(np.linalg.eigvals(A_hat)))),'n_iters':int(dmdc_fitter.fit_info.get('n_iters',0))},'probes':probe_results,'control':ctrl_results,'training_time_s':time.time()-t_start,'n_params':n_params}
    def _make_serializable(obj):
        if isinstance(obj,dict):
            return {k:_make_serializable(v) for k,v in obj.items()}
        elif isinstance(obj,(list,tuple)):
            return [_make_serializable(v) for v in obj]
        elif isinstance(obj,np.ndarray):
            return obj.tolist()
        elif isinstance(obj,(np.integer,np.floating)):
            return float(obj)
        elif isinstance(obj,complex):
            return {'real':float(obj.real),'imag':float(obj.imag)}
        elif isinstance(obj,bool):
            return bool(obj)
        return obj
    results_serializable=_make_serializable(results)
    if 'control' in results_serializable and 'all_results' in results_serializable['control']:
        del results_serializable['control']['all_results']
    with open(results_file,'w') as f:
        json.dump(results_serializable,f,indent=2)
    np.save(out_dir/'A_hat.npy',A_hat)
    np.save(out_dir/'B_hat.npy',B_hat)
    np.save(out_dir/'C_hat.npy',C_hat)
    torch.save(model.state_dict(),out_dir/'model_final.pt')
    elapsed=time.time()-t_start
    print(f'\n[done] {exp_name} completed in {elapsed:.1f}s')
    return results

def _generate_rollout_data(model,env,gt,n_rollouts,T,device,A_hat,B_hat,C_hat):
    model.eval()
    all_states=[]
    all_latent_states=[]
    rng=np.random.RandomState(99)
    for i in range(n_rollouts):
        x0=rng.uniform(-0.05,0.05,size=4)
        obs,state,_=env.reset_to_state(x0)
        traj_states=[]
        traj_z=[]
        for t in range(T):
            traj_states.append(state.copy())
            obs_t=torch.from_numpy(obs).float().permute(2,0,1)[None].to(device)/255.0
            with torch.no_grad():
                z=model.encoder(obs_t).cpu().numpy()[0]
            traj_z.append(z)
            obs,state,_,done,_=env.step(0.0)
            if done:
                for _ in range(T-t-1):
                    traj_states.append(state.copy())
                    traj_z.append(z.copy())
                break
        all_states.append(np.array(traj_states))
        all_latent_states.append(np.array(traj_z))
    return {'states':np.array(all_states),'latent_states':np.array(all_latent_states),'A_hat':A_hat,'B_hat':B_hat,'C_hat':C_hat}

def _generate_paired_data(model,env,n_pairs,device):
    model.eval()
    rng=np.random.RandomState(77)
    states_list=[]
    latent_list=[]
    for _ in range(n_pairs):
        x0=rng.uniform(-0.1,0.1,size=4)
        obs,state,_=env.reset_to_state(x0)
        obs_t=torch.from_numpy(obs).float().permute(2,0,1)[None].to(device)/255.0
        with torch.no_grad():
            z=model.encoder(obs_t).cpu().numpy()[0]
        states_list.append(state.copy())
        latent_list.append(z)
    return {'states':np.array(states_list),'latent_states':np.array(latent_list)}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',default='E-noact',choices=['E-noact','E-spec','E-PBH','E-both-r','E-lift','E-full'])
    parser.add_argument('--dataset',default='random',choices=['random','lqr','mixed'])
    parser.add_argument('--frame_skip',type=int,default=1,choices=[1,5,10])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--config',default='configs/cartpole.yaml')
    parser.add_argument('--data_dir',default='data')
    parser.add_argument('--results_dir',default='results')
    parser.add_argument('--eval-only',action='store_true',help='Load model_final.pt and skip training; only re-run DMDc/probes/control')
    parser.add_argument('--force',action='store_true',help='Re-run even if results.json already exists')
    args=parser.parse_args()
    results=run_single_experiment(encoder_variant=args.variant,dataset_name=args.dataset,frame_skip=args.frame_skip,seed=args.seed,config_path=args.config,data_dir=args.data_dir,results_dir=args.results_dir,skip_if_exists=not args.force,eval_only=args.eval_only)
    print(f'\nControl success rate: {results["control"].get("success_rate","N/A")}')
