"""Full experiment grid runner."""
from __future__ import annotations
import argparse,json,os,time
from multiprocessing import Pool,cpu_count
from pathlib import Path
from typing import List,Tuple
VARIANTS=['E-noact','E-spec','E-PBH','E-both-r','E-lift','E-full']
DATASETS=['random','lqr','mixed']
FRAME_SKIPS=[1,5,10]
SEEDS=[0,1,2]

def _worker(args):
    variant,dataset,frame_skip,seed,config_path,data_dir,results_dir,skip_if_exists=args
    from experiments.run_experiment import run_single_experiment
    try:
        result=run_single_experiment(encoder_variant=variant,dataset_name=dataset,frame_skip=frame_skip,seed=seed,config_path=config_path,data_dir=data_dir,results_dir=results_dir,skip_if_exists=skip_if_exists)
        return {'status':'ok','exp':f'{variant}_{dataset}_fs{frame_skip}_seed{seed}','success_rate':result.get('control',{}).get('success_rate',float('nan'))}
    except Exception as exc:
        return {'status':'error','exp':f'{variant}_{dataset}_fs{frame_skip}_seed{seed}','error':str(exc)}

def build_grid(variants=VARIANTS,datasets=DATASETS,frame_skips=FRAME_SKIPS,seeds=SEEDS):
    grid=[]
    for v in variants:
        for d in datasets:
            for fs in frame_skips:
                for s in seeds:
                    grid.append((v,d,fs,s))
    return grid

def run_all(n_workers=1,resume=True,dry_run=False,config_path='configs/cartpole.yaml',data_dir='data',results_dir='results',variants=VARIANTS,datasets=DATASETS,frame_skips=FRAME_SKIPS,seeds=SEEDS):
    grid=build_grid(variants,datasets,frame_skips,seeds)
    total=len(grid)
    print(f'Experiment grid: {total} experiments ({len(variants)} variants x {len(datasets)} datasets x {len(frame_skips)} frame_skips x {len(seeds)} seeds)')
    if dry_run:
        for i,(v,d,fs,s) in enumerate(grid):
            exp_name=f'{v}_{d}_fs{fs}_seed{s}'
            exists=(Path(results_dir)/exp_name/'results.json').exists()
            print(f'  {i+1:3d}. {"[EXISTS]" if exists else "[PENDING]"} {exp_name}')
        return
    worker_args=[(v,d,fs,s,config_path,data_dir,results_dir,resume) for v,d,fs,s in grid]
    t0=time.time()
    all_results=[]
    if n_workers>1:
        effective_workers=min(n_workers,cpu_count(),total)
        print(f'Running with {effective_workers} parallel workers...')
        with Pool(processes=effective_workers) as pool:
            for i,result in enumerate(pool.imap_unordered(_worker,worker_args)):
                all_results.append(result)
                print(f'  [{i+1}/{total}] {result["status"].upper()} {result["exp"]} (success_rate={result.get("success_rate","?")})')
    else:
        print('Running sequentially...')
        for i,args in enumerate(worker_args):
            print(f'\n[{i+1}/{total}] {args[0]}_{args[1]}_fs{args[2]}_seed{args[3]}')
            result=_worker(args)
            all_results.append(result)
            print(f'  -> {result["status"].upper()}, success_rate={result.get("success_rate","?")}')
    elapsed=time.time()-t0
    n_ok=sum(1 for r in all_results if r['status']=='ok')
    n_err=sum(1 for r in all_results if r['status']=='error')
    print(f'\n{"="*60}\nGRID COMPLETE in {elapsed:.0f}s\n  OK: {n_ok} | Errors: {n_err}\n{"="*60}')
    summary_path=Path(results_dir)/'grid_summary.json'
    summary_path.parent.mkdir(parents=True,exist_ok=True)
    with open(summary_path,'w') as f:
        json.dump(all_results,f,indent=2)
    print(f'Summary saved to {summary_path}')
    if n_err>0:
        print('\nFailed experiments:')
        for r in all_results:
            if r['status']=='error':
                print(f'  {r["exp"]}: {r.get("error","")}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Run full JEPA control experiment grid')
    parser.add_argument('--n_workers',type=int,default=1)
    parser.add_argument('--resume',action='store_true',default=True)
    parser.add_argument('--no_resume',dest='resume',action='store_false')
    parser.add_argument('--dry_run',action='store_true')
    parser.add_argument('--config',default='configs/cartpole.yaml')
    parser.add_argument('--data_dir',default='data')
    parser.add_argument('--results_dir',default='results')
    parser.add_argument('--variants',nargs='+',default=VARIANTS)
    parser.add_argument('--datasets',nargs='+',default=DATASETS)
    parser.add_argument('--frame_skips',nargs='+',type=int,default=FRAME_SKIPS)
    parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    args=parser.parse_args()
    run_all(n_workers=args.n_workers,resume=args.resume,dry_run=args.dry_run,config_path=args.config,data_dir=args.data_dir,results_dir=args.results_dir,variants=args.variants,datasets=args.datasets,frame_skips=args.frame_skips,seeds=args.seeds)
