"""Aggregation and correlation analysis for experiment results."""
from __future__ import annotations
import json,warnings
from pathlib import Path
from typing import Dict,List,Optional
import numpy as np
import pandas as pd
import scipy.stats

def load_all_results(results_dir='results'):
    results_dir=Path(results_dir)
    rows=[]
    for results_file in sorted(results_dir.glob('*/results.json')):
        try:
            with open(results_file) as f:
                data=json.load(f)
        except Exception as exc:
            warnings.warn(f'Failed to load {results_file}: {exc}')
            continue
        row={}
        exp=data.get('experiment',{})
        row['encoder_variant']=exp.get('encoder_variant','')
        row['dataset_name']=exp.get('dataset_name','')
        row['frame_skip']=exp.get('frame_skip',-1)
        row['seed']=exp.get('seed',-1)
        row['exp_name']=exp.get('exp_name','')
        dq=data.get('data_quality',{})
        row['action_cov_kappa']=dq.get('action_cov_condition_number',float('nan'))
        dmdc=data.get('dmdc',{})
        row['dmdc_residual']=dmdc.get('residual',float('nan'))
        row['A_hat_spectral_radius']=dmdc.get('A_hat_spectral_radius',float('nan'))
        ctrl=data.get('control',{})
        row['stabilization_success_rate']=ctrl.get('success_rate',float('nan'))
        row['mean_settling_time']=ctrl.get('mean_settling_time',float('nan'))
        row['mean_final_error']=ctrl.get('mean_final_error',float('nan'))
        row['true_lqr_cost']=ctrl.get('true_lqr_cost',float('nan'))
        probes=data.get('probes',{})
        row.update(_flatten_dict(probes,prefix='probe/'))
        rows.append(row)
    if not rows:
        warnings.warn(f'No results found in {results_dir}')
        return pd.DataFrame()
    return pd.DataFrame(rows)

def _flatten_dict(d,prefix=''):
    flat={}
    for k,v in d.items():
        key=f'{prefix}{k}'
        if isinstance(v,dict):
            flat.update(_flatten_dict(v,prefix=f'{key}/'))
        elif isinstance(v,(int,float,bool)) and not isinstance(v,complex):
            flat[key]=float(v)
        elif isinstance(v,list) and len(v)==1 and isinstance(v[0],(int,float)):
            flat[key]=float(v[0])
    return flat

def compute_correlation_matrix(df,target_col='stabilization_success_rate',min_valid=10):
    if target_col not in df.columns:
        warnings.warn(f'Target column {target_col!r} not in DataFrame.')
        return pd.DataFrame()
    target=df[target_col].values
    probe_cols=[c for c in df.columns if c.startswith('probe/') and not c.endswith('/error')]
    rows=[]
    for col in probe_cols:
        vals=df[col].values.astype(float)
        valid=np.isfinite(vals)&np.isfinite(target)
        n_valid=int(valid.sum())
        if n_valid<min_valid:
            continue
        r,p=scipy.stats.spearmanr(vals[valid],target[valid])
        rows.append({'metric':col,'spearman_r':float(r),'p_value':float(p),'n_valid':n_valid})
    if not rows:
        return pd.DataFrame()
    corr_df=pd.DataFrame(rows)
    corr_df['abs_r']=corr_df['spearman_r'].abs()
    corr_df=corr_df.sort_values('abs_r',ascending=False).drop('abs_r',axis=1).reset_index(drop=True)
    return corr_df

def summarize_by_variant(df):
    key_cols=['stabilization_success_rate','mean_settling_time','mean_final_error','dmdc_residual','A_hat_spectral_radius','probe/P1_1/delta_lambda','probe/P1_1/UMR','probe/P2_1/mu_S','probe/P3_1/efficiency_ratio','probe/P4_1/latent_NMP_count','probe/P4_2/UR_latent','probe/D1/reconstruction_error']
    available=[c for c in key_cols if c in df.columns]
    summary_rows=[]
    for variant,grp in df.groupby('encoder_variant'):
        row={'encoder_variant':variant,'n_experiments':len(grp)}
        for col in available:
            vals=grp[col].dropna()
            row[f'{col}_mean']=float(vals.mean()) if len(vals)>0 else float('nan')
            row[f'{col}_std']=float(vals.std()) if len(vals)>1 else float('nan')
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).set_index('encoder_variant')

def test_lifting_vs_regularization(df,dataset='random',frame_skip=1):
    mask=(df['dataset_name']==dataset)&(df['frame_skip']==frame_skip)
    df_sub=df[mask]
    def _get(variant,col):
        return df_sub[df_sub['encoder_variant']==variant][col].dropna().values
    result={}
    pairs=[('E-PBH','E-lift'),('E-PBH','E-full'),('E-spec','E-lift')]
    for v1,v2 in pairs:
        sr1=_get(v1,'stabilization_success_rate')
        sr2=_get(v2,'stabilization_success_rate')
        key=f'{v1}_vs_{v2}'
        if len(sr1)>=2 and len(sr2)>=2:
            stat,p=scipy.stats.mannwhitneyu(sr1,sr2,alternative='two-sided')
            result[key]={f'{v1}_mean_sr':float(np.mean(sr1)),f'{v2}_mean_sr':float(np.mean(sr2)),'mann_whitney_u':float(stat),'p_value':float(p),'significant_0.05':bool(p<0.05)}
        else:
            result[key]={'error':'insufficient data'}
    for metric in ['probe/P2_1/mu_S','probe/P1_1/UMR','probe/P4_2/sign_match']:
        if metric not in df.columns:
            continue
        for v in ['E-PBH','E-lift','E-full']:
            vals=_get(v,metric)
            if len(vals)>0:
                result.setdefault(f'metric_{metric}',{})[v]=float(np.nanmean(vals))
    return result

def pivot_success_rate(df):
    if 'stabilization_success_rate' not in df.columns:
        return pd.DataFrame()
    return df.pivot_table(values='stabilization_success_rate',index='encoder_variant',columns=['dataset_name','frame_skip'],aggfunc='mean')

