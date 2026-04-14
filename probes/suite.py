"""Master probe suite."""
from __future__ import annotations
import time,warnings
from typing import Dict,Any,Optional
import numpy as np

def run_all_probes(model,encoder_variant,dataset,gt,rollout_data,paired_data,env=None,device=None,config=None):
    from probes.spectral import P1_1_eigenvalue_recovery,P1_2_jordan_block,P1_3_marginal_mode_frequency
    from probes.pbh import P2_1_pbh_stabilizability,P2_2_pbh_detectability,P2_3_separation_principle
    from probes.kalman import P3_1_kalman_decomposition,P3_2_controllable_subspace_alignment,P3_3_manifold_geometry,P3_4_phantom_instability
    from probes.zeros import P4_1_transmission_zeros,P4_2_step_response_undershoot,P4_3_markov_parameters,P4_4_zero_direction_alignment
    from probes.decoder import D1_action_decoder_fidelity
    if config is None:
        config={}
    delta_tol=config.get('delta_tol',0.05)
    epsilon_lambda=config.get('epsilon_lambda',0.05)
    epsilon_rank=config.get('epsilon_rank',1e-3)
    A_hat=rollout_data.get('A_hat')
    B_hat=rollout_data.get('B_hat')
    C_hat=rollout_data.get('C_hat')
    A_star=gt.A_star
    B_star=gt.B_star
    C_star=gt.C_star
    results={'encoder_variant':encoder_variant,'timestamp':time.time()}
    def run_probe(name,fn,*args,**kwargs):
        try:
            t0=time.time()
            out=fn(*args,**kwargs)
            results[name]=out
            results[name]['_runtime_s']=time.time()-t0
        except Exception as exc:
            warnings.warn(f'Probe {name} failed: {exc}')
            results[name]={'error':str(exc)}
    if A_hat is not None:
        run_probe('P1_1',P1_1_eigenvalue_recovery,A_hat,A_star,epsilon_lambda)
        run_probe('P1_2',P1_2_jordan_block,A_hat,rollout_data)
        run_probe('P1_3',P1_3_marginal_mode_frequency,rollout_data,A_hat,A_star)
    else:
        warnings.warn('A_hat not in rollout_data; skipping P1 probes.')
    if A_hat is not None and B_hat is not None:
        run_probe('P2_1',P2_1_pbh_stabilizability,A_hat,B_hat,delta_tol)
        if C_hat is not None:
            run_probe('P2_2',P2_2_pbh_detectability,A_hat,C_hat,encoder=model.encoder if model is not None else None,gt=gt,delta_tol=delta_tol,device=device)
            if env is not None and model is not None:
                run_probe('P2_3',P2_3_separation_principle,A_hat,B_hat,C_hat,model.encoder,env)
    if A_hat is not None and B_hat is not None and C_hat is not None:
        run_probe('P3_1',P3_1_kalman_decomposition,A_hat,B_hat,C_hat,epsilon_rank)
        run_probe('P3_2',P3_2_controllable_subspace_alignment,A_hat,B_hat,A_star,B_star,paired_data)
        run_probe('P3_3',P3_3_manifold_geometry,A_hat,rollout_data,A_star)
        if env is not None and model is not None:
            run_probe('P3_4',P3_4_phantom_instability,A_hat,B_hat,C_hat,model.encoder,env)
    if A_hat is not None and B_hat is not None and C_hat is not None:
        run_probe('P4_1',P4_1_transmission_zeros,A_hat,B_hat,C_hat,A_star,B_star,C_star)
        if env is not None and model is not None:
            run_probe('P4_2',P4_2_step_response_undershoot,A_hat,B_hat,C_hat,model.encoder,env)
            run_probe('P4_3',P4_3_markov_parameters,A_hat,B_hat,C_hat,A_star,B_star,C_star,model.encoder,env)
        run_probe('P4_4',P4_4_zero_direction_alignment,A_hat,B_hat,C_hat,A_star,B_star,C_star,paired_data)
    if model is not None:
        run_probe('D1',D1_action_decoder_fidelity,model.action_encoder,device=device)
    _print_summary(results)
    return results

def _print_summary(results):
    print('\n'+'='*60)
    print(f"PROBE SUITE RESULTS  -  variant: {results.get('encoder_variant','?')}")
    print('='*60)
    key_metrics=[('P1_1','delta_lambda','Eigenvalue matching distance'),('P1_1','UMR','Unstable mode recall'),('P1_1','spectral_radius_error','Spectral radius error'),('P2_1','mu_S','PBH stabilisability mu_S'),('P2_1','is_stabilizable','Is stabilisable'),('P2_2','mu_D','PBH detectability mu_D'),('P2_3','stabilization_success_rate','Control success rate'),('P3_1','efficiency_ratio','Kalman efficiency ratio'),('P3_1','has_phantom_instability','Has phantom instability'),('P4_1','latent_NMP_count','Latent NMP zeros'),('P4_1','NMP_count_match','NMP count match'),('P4_2','UR_latent','Undershoot ratio (latent)'),('P4_2','NMP_detected_latent','NMP detected (latent)'),('P4_3','relative_degree_match','Relative degree match'),('D1','reconstruction_error','Action recon error')]
    for probe_key,metric,label in key_metrics:
        probe_res=results.get(probe_key,{})
        if 'error' in probe_res:
            val='ERROR'
        else:
            val=probe_res.get(metric,'N/A')
            if isinstance(val,float):
                val=f'{val:.4f}'
            elif isinstance(val,bool):
                val=str(val)
        print(f'  {label:<40s}: {val}')
    print('='*60+'\n')

def flatten_results(results,prefix=''):
    flat={}
    for k,v in results.items():
        key=f'{prefix}{k}' if prefix else k
        if isinstance(v,dict):
            flat.update(flatten_results(v,prefix=f'{key}/'))
        elif isinstance(v,(int,float,bool)):
            flat[key]=float(v)
        elif isinstance(v,(list,np.ndarray)) and len(v)==1:
            flat[key]=float(v[0])
    return flat
