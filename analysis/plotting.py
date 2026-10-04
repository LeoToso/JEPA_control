"""Visualisation functions for JEPA control experiment results."""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict,List,Optional,Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
try:
    import seaborn as sns
    HAS_SEABORN=True
except ImportError:
    HAS_SEABORN=False
VARIANT_COLORS={'E-noact':'#1f77b4','E-spec':'#ff7f0e','E-PBH':'#2ca02c','E-both-r':'#d62728','E-lift':'#9467bd','E-full':'#8c564b'}
VARIANT_ORDER=['E-noact','E-spec','E-PBH','E-both-r','E-lift','E-full']

def _save(fig,save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(save_path,dpi=150,bbox_inches='tight')
    return fig

def plot_probe_correlation_heatmap(results_df,save_path=None,top_k=20):
    from analysis.metrics import compute_correlation_matrix
    corr_df=compute_correlation_matrix(results_df)
    if corr_df.empty:
        warnings.warn('No correlation data available.')
        fig,ax=plt.subplots()
        ax.text(0.5,0.5,'No data',ha='center')
        return _save(fig,save_path)
    top=corr_df.head(top_k)
    metrics=[m.replace('probe/','').replace('/','\n') for m in top['metric']]
    rs=top['spearman_r'].values
    ps=top['p_value'].values
    colors=['#d62728' if r>0 else '#1f77b4' for r in rs]
    alpha=[1.0 if p<0.05 else 0.4 for p in ps]
    fig,ax=plt.subplots(figsize=(10,max(4,top_k*0.35)))
    bars=ax.barh(range(len(metrics)),rs,color=colors)
    for bar,a in zip(bars,alpha):
        bar.set_alpha(a)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics,fontsize=8)
    ax.axvline(0,color='k',linewidth=0.8)
    ax.set_xlabel('Spearman correlation with stabilization success rate')
    ax.set_title('Probe metric correlations\n(opaque=p<0.05, blue=negative, red=positive)')
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig,save_path)

def plot_eigenvalue_comparison(A_hat_dict,A_star,save_path=None):
    import scipy.linalg
    variants=[v for v in VARIANT_ORDER if v in A_hat_dict]
    ncols=min(3,len(variants)+1)
    nrows=(len(variants)+1+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(4*ncols,4*nrows))
    axes=np.array(axes).flatten()
    theta=np.linspace(0,2*np.pi,300)
    def _plot_eigs(ax,eigs,color,label,marker='o',zorder=3):
        stable=eigs[np.abs(eigs)<1.0]
        unstable=eigs[np.abs(eigs)>=1.0]
        ax.scatter(stable.real,stable.imag,c=color,marker=marker,s=50,alpha=0.7,label=f'{label} stable',zorder=zorder)
        ax.scatter(unstable.real,unstable.imag,c='red',marker=marker,s=80,alpha=1.0,label=f'{label} unstable',edgecolors='k',zorder=zorder+1)
    eigs_star=scipy.linalg.eigvals(A_star)
    _plot_eigs(axes[0],eigs_star,'black','A*',marker='*')
    axes[0].plot(np.cos(theta),np.sin(theta),'k--',linewidth=0.8,alpha=0.5)
    axes[0].set_title('Ground truth A*',fontsize=10)
    axes[0].set_aspect('equal')
    axes[0].legend(fontsize=7)
    axes[0].set_xlim(-1.5,1.5)
    axes[0].set_ylim(-1.5,1.5)
    for ax_i,variant in enumerate(variants):
        ax=axes[ax_i+1]
        A_hat=A_hat_dict[variant]
        eigs=scipy.linalg.eigvals(A_hat)
        color=VARIANT_COLORS.get(variant,'gray')
        _plot_eigs(ax,eigs,color,variant)
        ax.plot(np.cos(theta),np.sin(theta),'k--',linewidth=0.8,alpha=0.5)
        ax.set_title(variant,fontsize=10,color=color)
        ax.set_aspect('equal')
        ax.set_xlim(-1.5,1.5)
        ax.set_ylim(-1.5,1.5)
        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
    for ax in axes[len(variants)+1:]:
        ax.set_visible(False)
    fig.suptitle('Eigenvalue Comparison (A_hat vs A*)',fontsize=12)
    fig.tight_layout()
    return _save(fig,save_path)

def plot_pbh_index_vs_success(results_df,save_path=None):
    mu_col='probe/P2_1/mu_S'
    sr_col='stabilization_success_rate'
    if mu_col not in results_df.columns or sr_col not in results_df.columns:
        fig,ax=plt.subplots()
        ax.text(0.5,0.5,'Data not available',ha='center')
        return _save(fig,save_path)
    fig,ax=plt.subplots(figsize=(8,6))
    for variant in VARIANT_ORDER:
        mask=results_df['encoder_variant']==variant
        sub=results_df[mask]
        mu=sub[mu_col].values.astype(float)
        sr=sub[sr_col].values.astype(float)
        valid=np.isfinite(mu)&np.isfinite(sr)
        ax.scatter(mu[valid],sr[valid],color=VARIANT_COLORS.get(variant,'gray'),label=variant,s=60,alpha=0.8,edgecolors='k',linewidths=0.5)
    ax.set_xlabel('PBH stabilisability index μ_S',fontsize=12)
    ax.set_ylabel('Stabilization success rate',fontsize=12)
    ax.set_title('PBH Index vs Control Performance',fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05,1.05)
    ax.axhline(0.5,color='gray',linestyle='--',alpha=0.5)
    fig.tight_layout()
    return _save(fig,save_path)

def plot_step_response_comparison(step_response_dict,save_path=None):
    variants=[v for v in VARIANT_ORDER if v in step_response_dict]
    if not variants:
        fig,ax=plt.subplots()
        ax.text(0.5,0.5,'No step response data',ha='center')
        return _save(fig,save_path)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    ax_lat,ax_true=axes
    for variant in variants:
        data=step_response_dict[variant]
        color=VARIANT_COLORS.get(variant,'gray')
        if 'latent' in data and data['latent'] is not None:
            y=np.array(data['latent'])
            if y.ndim>1:
                y=y[:,0]
            ax_lat.plot(range(len(y)),y,color=color,label=variant,alpha=0.8)
        if 'true' in data and data['true'] is not None:
            y=np.array(data['true'])
            ax_true.plot(range(len(y)),y,color=color,label=variant,alpha=0.8)
    for ax,title in [(ax_lat,'Latent step response (C_hat z_t)'),(ax_true,'True step response (pole angle)')]:
        ax.set_xlabel('Time step')
        ax.set_ylabel('Output')
        ax.set_title(title,fontsize=11)
        ax.legend(fontsize=8)
        ax.axhline(0,color='k',linewidth=0.5)
    fig.suptitle('Step Response: NMP Undershoot Detection',fontsize=13)
    fig.tight_layout()
    return _save(fig,save_path)

def plot_manifold_geometry(rollout_data_dict,save_path=None):
    from sklearn.decomposition import PCA
    variants=[v for v in VARIANT_ORDER if v in rollout_data_dict]
    if not variants:
        fig,ax=plt.subplots()
        ax.text(0.5,0.5,'No rollout data',ha='center')
        return _save(fig,save_path)
    ncols=min(3,len(variants))
    nrows=(len(variants)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(4*ncols,4*nrows))
    axes=np.array(axes).flatten()
    for ax_i,variant in enumerate(variants):
        ax=axes[ax_i]
        data=rollout_data_dict[variant]
        lz=data.get('latent_states')
        if lz is None:
            ax.text(0.5,0.5,'N/A',ha='center')
            continue
        N,T,d=lz.shape
        Z=lz.reshape(-1,d)
        t_idx=np.tile(np.arange(T),N)
        pca=PCA(n_components=2)
        try:
            Z2=pca.fit_transform(Z)
        except Exception:
            ax.text(0.5,0.5,'PCA failed',ha='center')
            continue
        sc=ax.scatter(Z2[:,0],Z2[:,1],c=t_idx,cmap='viridis',s=2,alpha=0.5)
        ax.set_title(variant,fontsize=10,color=VARIANT_COLORS.get(variant,'gray'))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(sc,ax=ax,label='t')
    for ax in axes[len(variants):]:
        ax.set_visible(False)
    fig.suptitle('Latent Trajectory Manifold (Zero-Control Rollouts)',fontsize=12)
    fig.tight_layout()
    return _save(fig,save_path)

plot_latent_trajectory_pca=plot_manifold_geometry

def plot_success_rate_grid(results_df,save_path=None):
    from analysis.metrics import pivot_success_rate
    pivot=pivot_success_rate(results_df)
    if pivot.empty:
        fig,ax=plt.subplots()
        ax.text(0.5,0.5,'No data',ha='center')
        return _save(fig,save_path)
    variants_present=[v for v in VARIANT_ORDER if v in pivot.index]
    pivot=pivot.reindex(variants_present)
    fig,ax=plt.subplots(figsize=(max(8,pivot.shape[1]*1.2),max(4,len(variants_present)*0.8)))
    if HAS_SEABORN:
        sns.heatmap(pivot.astype(float),ax=ax,vmin=0,vmax=1,cmap='RdYlGn',annot=True,fmt='.2f',linewidths=0.5,cbar_kws={'label':'Success rate'})
    else:
        im=ax.imshow(pivot.values.astype(float),vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{d}\nfs={fs}' for d,fs in pivot.columns],fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index,fontsize=9)
        plt.colorbar(im,ax=ax,label='Success rate')
    ax.set_title('Stabilisation Success Rate by Variant x Dataset x Frame-skip',fontsize=12)
    ax.set_ylabel('Encoder variant')
    ax.set_xlabel('(Dataset, Frame-skip)')
    fig.tight_layout()
    return _save(fig,save_path)

def plot_training_curves(log_csv_path,save_path=None):
    try:
        df=pd.read_csv(log_csv_path)
    except Exception as exc:
        warnings.warn(f'Could not load training log: {exc}')
        fig,ax=plt.subplots()
        ax.text(0.5,0.5,'No log data',ha='center')
        return _save(fig,save_path)
    train=df[df['split']=='train'].groupby('epoch')['total_loss'].mean()
    val=df[df['split']=='val'].groupby('epoch')['total_loss'].mean()
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(train.index,train.values,label='Train',color='#1f77b4')
    ax.plot(val.index,val.values,label='Val',color='#ff7f0e')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    return _save(fig,save_path)

def generate_all_plots(results_df,output_dir='figures',A_hat_dict=None,A_star=None,rollout_data_dict=None,step_response_dict=None):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    plot_probe_correlation_heatmap(results_df,save_path=str(out/'probe_correlations.png'))
    plot_pbh_index_vs_success(results_df,save_path=str(out/'pbh_vs_success.png'))
    plot_success_rate_grid(results_df,save_path=str(out/'success_rate_grid.png'))
    if A_hat_dict is not None and A_star is not None:
        plot_eigenvalue_comparison(A_hat_dict,A_star,save_path=str(out/'eigenvalues.png'))
    if rollout_data_dict is not None:
        plot_manifold_geometry(rollout_data_dict,save_path=str(out/'manifold_geometry.png'))
    if step_response_dict is not None:
        plot_step_response_comparison(step_response_dict,save_path=str(out/'step_response.png'))
    print(f'[plots] All figures saved to {out}')

