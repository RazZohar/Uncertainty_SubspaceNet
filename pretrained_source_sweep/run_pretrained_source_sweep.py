#!/usr/bin/env python3
"""Load one existing native checkpoint and evaluate a source-angle sweep.

Evaluation only: no optimizer, trainer, training dataset, or checkpoint writes.
--checkpoint_path selects the exact pretrained weights; --config_path selects
its original model configuration. --dry_run checks inputs without loading torch.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
EXPERIMENTS=['fixed_anchor','symmetric_separation','fixed_separation','coherence','power_imbalance']


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=True));temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def cases_for(args,system):
    cases=[]
    moving=list(args.moving_values) if args.moving_values is not None else np.arange(args.moving_start,args.moving_stop+args.moving_step*.1,args.moving_step).tolist()+[args.moving_stop]
    if any(not -90<=v<=90 for v in moving):raise ValueError("Moving angles must lie in [-90,90].")
    if args.dense_near_anchor:
        moving+= [args.fixed_source_deg+x for x in [-5,-3,-2,-1,-.5,0,.5,1,2,3,5]]
    moving=sorted(set(round(float(v),8) for v in moving if -90<=v<=90))
    for snr in args.snr_values or [float(system['snr'])]:
        for exp in args.experiments:
            if exp=='fixed_anchor':
                points=[(args.fixed_source_deg,v,v,None,0.) for v in moving]
            elif exp=='symmetric_separation':
                points=[(args.center_deg-v/2,args.center_deg+v/2,v,None,0.) for v in args.separation_values]
            elif exp=='fixed_separation':
                points=[(v-args.control_separation/2,v+args.control_separation/2,v,None,0.) for v in args.center_values]
            elif exp=='coherence':
                points=[(*args.probe_pair,v,v,0.) for v in args.coherence_values]
            else:
                points=[(*args.probe_pair,v,None,v) for v in args.power_imbalance_values]
            for a,b,x,rho,power in points:
                if not(-90<=a<=90 and -90<=b<=90):raise ValueError(f'Angles outside [-90,90]: {a,b}')
                sep=abs(b-a);du=abs(np.sin(np.deg2rad(a))-np.sin(np.deg2rad(b)))
                status='coincident' if sep<1e-8 else ('endfire' if max(abs(a),abs(b))>=90-1e-8 else 'regular')
                c=dict(experiment=exp,x=float(x),theta_anchor_deg=float(a),theta_moving_deg=float(b),
                    separation_deg=float(sep),spatial_separation=float(du),
                    wrapped_spatial_separation=float(min(du,2-du)),case_status=status,
                    snr_parameter=float(snr),rho_override=rho,power_imbalance_db=float(power),
                    source_nature=system.get('signal_nature','non-coherent'))
                text=json.dumps(c,sort_keys=True)
                c['case_id']=exp+'_'+hashlib.sha256(text.encode()).hexdigest()[:12]
                # Stable seed independent of model/stage/order/resuming/sharding.
                c['test_seed']=(args.seed+int(hashlib.sha256(text.encode()).hexdigest()[:8],16))%(2**32-1)
                cases.append(c)
    return cases


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint_path',type=Path,help='Exact existing native .pth file')
    p.add_argument('--config_path',type=Path,help='Original config.json; defaults to checkpoint parent.parent/config.json')
    p.add_argument('--model_type',choices=['multi_subarray','transmusic','data_driven_complex'],default='multi_subarray')
    p.add_argument('--results_dir',type=Path,default=Path('sweep_results/pretrained_source_sweep'))
    p.add_argument('--include_esprit',action='store_true')
    p.add_argument('--experiments',nargs='+',choices=EXPERIMENTS,default=['fixed_anchor'])
    p.add_argument('--fixed_source_deg',type=float,default=40.)
    p.add_argument('--moving_start',type=float,default=-90.);p.add_argument('--moving_stop',type=float,default=90.)
    p.add_argument('--moving_step',type=float,default=5.);p.add_argument('--moving_values',type=float,nargs='+')
    p.add_argument('--dense_near_anchor',action='store_true')
    p.add_argument('--center_deg',type=float,default=0.)
    p.add_argument('--separation_values',type=float,nargs='+',default=[1,2,3,5,8,12,20,30])
    p.add_argument('--control_separation',type=float,default=10.)
    p.add_argument('--center_values',type=float,nargs='+',default=[-80,-60,-40,-20,0,20,40,60,80])
    p.add_argument('--coherence_values',type=float,nargs='+',default=[0,.5,.9,.99,1.])
    p.add_argument('--probe_pair',type=float,nargs=2,default=[40.,35.])
    p.add_argument('--power_imbalance_values',type=float,nargs='+',default=[0,5,10,15,20])
    p.add_argument('--snr_values',type=float,nargs='+')
    p.add_argument('--trials',type=int,default=5000);p.add_argument('--batch_size',type=int,default=256)
    p.add_argument('--uq_chunk_size',type=int,default=32)
    p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--device',default='auto');p.add_argument('--tau',type=int,default=8)
    p.add_argument('--num_angle_bins',type=int,default=360)
    p.add_argument('--covariance_policy',choices=['report','project'],default='report')
    p.add_argument('--eigenvalue_floor',type=float,default=1e-9,help='rad^2; used only with explicit project policy')
    p.add_argument('--ellipse_moving_values',type=float,nargs='+',default=[-40.,35.,45.,70.])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry_run',action='store_true');p.add_argument('--plots_only',action='store_true')
    p.add_argument('--case_index',type=int,help='Optional independent SLURM-array case index; use separate results_dir per task')
    p.add_argument('--wandb_project',default='multi-subarrays-doa');p.add_argument('--log_eval_to_wandb',action='store_true')
    p.add_argument('--wandb_mode',choices=['online','offline','disabled'],default='offline')
    args=p.parse_args(argv)
    if args.plots_only:return args
    if args.trials<3 or args.batch_size<1 or args.moving_step<=0 or args.uq_chunk_size<1:p.error('Invalid trials/batch/step/chunk size.')
    if not -90<=args.fixed_source_deg<=90 or args.moving_start<-90 or args.moving_stop>90 or args.moving_start>args.moving_stop:p.error('Invalid angular bounds.')
    if any(not 0<=r<=1 for r in args.coherence_values):p.error('Coherence must lie in [0,1].')
    if any(s<=0 for s in args.separation_values) or args.control_separation<=0:p.error('Separations must be positive.')
    if args.eigenvalue_floor<=0:p.error('Floor must be positive.')
    if not args.checkpoint_path:p.error('--checkpoint_path is required.')
    return args


def main(argv=None):
    args=parse_args(argv)
    if args.plots_only:
        from inter_source_tools.plotting import plot_results
        plot_results(args.results_dir.resolve());return
    results=args.results_dir.resolve();results.mkdir(parents=True,exist_ok=True)
    checkpoint_path=args.checkpoint_path.resolve()
    if not checkpoint_path.is_file():raise FileNotFoundError(f'Pretrained checkpoint does not exist: {checkpoint_path}')
    config_path=(args.config_path or checkpoint_path.parent.parent/'config.json').resolve()
    if not config_path.is_file():raise FileNotFoundError(f'Original model config not found: {config_path}. Pass --config_path explicitly.')
    cfg=json.loads(config_path.read_text())
    syscfg=cfg['subarray_config'][0]['system_model']
    if cfg['L']!=1 or syscfg['M']!=2:
        raise ValueError('Select an existing L=1,M=2 checkpoint and its original config for this two-source sweep. An M=3 model cannot be relabeled as M=2.')
    if syscfg['signal_type']!='NarrowBand':raise ValueError('This sweep requires NarrowBand.')
    stage=checkpoint_path.stem
    if stage.startswith('best_'):stage=stage[5:]
    elif stage.endswith('_completed'):stage=stage[:-10]
    targets=[dict(model_type=args.model_type,stage=stage,label=args.model_type+'__'+re.sub(r'[^A-Za-z0-9_-]+','_',stage),checkpoint=str(checkpoint_path))]
    if args.include_esprit:targets.append(dict(model_type='esprit',stage='analytic',label='esprit__analytic',checkpoint=None))
    cases=cases_for(args,syscfg)
    if args.case_index is not None:
        if not 0<=args.case_index<len(cases):raise ValueError('case_index out of range.')
        cases=[cases[args.case_index]]
    plan=dict(mode='pretrained_evaluation',checkpoint_path=str(checkpoint_path),config_path=str(config_path),
              configuration=cfg,targets=targets,cases=cases,case_count=len(cases),
              total_trial_predictions=len(cases)*args.trials*len(targets),
              source_snr_note='Native source amplitude=10**(snr/10); preserve the training convention.')
    atomic_json(results/'sweep_plan.json',plan)
    print(f'Pretrained checkpoint: {checkpoint_path}',flush=True)
    print(f'Original configuration: {config_path}',flush=True)
    print(f'Plan: {len(cases)} cases x {len(targets)} targets x {args.trials} trials',flush=True)
    if args.dry_run:
        print('Input files and configuration checked; neural weight compatibility is checked on the real run.')
        print('Plan saved:',results/'sweep_plan.json');return
    os.environ['WANDB_MODE']=args.wandb_mode
    for target in targets:
        if target['checkpoint']:target['sha256']=digest(target['checkpoint'])
    from inter_source_tools.native import NativePredictor,generate_case
    from inter_source_tools.statistics import evaluate,save_table
    from inter_source_tools.plotting import plot_results
    import torch
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    sources={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'run_pretrained_source_sweep.py',*sorted((ROOT/'inter_source_tools').glob('*.py')),
        ROOT/'src/models.py',ROOT/'src/uncertainty_block.py',ROOT/'src/signal_creation.py',ROOT/'src/system_model.py',ROOT/'src/multi_subarrays_model.py',ROOT/'src/criterions.py'] if p.exists()}
    signature=dict(configuration=cfg,targets=targets,cases=cases,trials=args.trials,batch_size=args.batch_size,
        covariance_policy=args.covariance_policy,floor=args.eigenvalue_floor,source_hashes=sources,
        torch=torch.__version__,device=device,uq_chunk_size=args.uq_chunk_size,tau=args.tau,num_angle_bins=args.num_angle_bins,
        ellipse_moving_values=args.ellipse_moving_values)
    manifest_path=results/'manifest.json'
    if manifest_path.exists():
        if not args.resume:raise FileExistsError('Existing results: use --resume or a new results_dir.')
        if json.loads(manifest_path.read_text())!=signature:raise ValueError('Resume signature differs (checkpoint/code/config/cases/settings). Use a new results_dir.')
    else:atomic_json(manifest_path,signature)
    predictors={t['label']:NativePredictor(t,cfg,config_path,args,device) for t in targets}
    wandb_run=None
    if args.log_eval_to_wandb:
        import wandb
        wandb_run=wandb.init(project=args.wandb_project,name='inter_source_evaluation',config=signature)
    rows=[]
    for index,case in enumerate(cases):
        print(f"[{index+1}/{len(cases)}] {case['experiment']} theta={case['theta_anchor_deg']:g},{case['theta_moving_deg']:g} status={case['case_status']}",flush=True)
        pending=[t for t in targets if not (args.resume and (results/'cases'/case['case_id']/(t['label']+'.json')).exists())]
        if pending:x,generator_info=generate_case(case,syscfg,args.trials)
        for target in targets:
            label=target['label'];base=results/'cases'/case['case_id']/label
            if target not in pending:
                rows.append(json.loads(base.with_suffix('.json').read_text()));continue
            pred,cov=predictors[label](x)
            truth=np.deg2rad([case['theta_anchor_deg'],case['theta_moving_deg']])
            metrics,arrays=evaluate(pred,truth,cov,args.covariance_policy,args.eigenvalue_floor,case['case_status'])
            row={**case,**{k:target[k] for k in ['label','model_type','stage']},**generator_info,**metrics}
            base.parent.mkdir(parents=True,exist_ok=True)
            temp=base.with_suffix('.npz.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,theta_hat_rad=pred,theta_true_rad=truth,covariance_rad2=cov,**arrays)
            temp.replace(base.with_suffix('.npz'))
            # JSON is the completion marker written only after raw predictions.
            atomic_json(base.with_suffix('.json'),row);rows.append(row)
            save_table(results/'benchmark_results.csv',rows)
            if wandb_run:wandb_run.log({k:v for k,v in row.items() if isinstance(v,(int,float))}|{'target':label,'experiment':case['experiment']})
            print(f"  {label}: RMSE(anchor,moving)=({metrics['RMSE_anchor_deg']:.4g},{metrics['RMSE_moving_deg']:.4g}) raw invalid covariance={metrics['raw_invalid_fraction']:.3%}",flush=True)
        save_table(results/'benchmark_results.csv',rows)
    if wandb_run:wandb_run.finish()
    plot_results(results)
    print('Completed:',results,flush=True)


if __name__=='__main__':main()
