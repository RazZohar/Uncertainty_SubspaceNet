"""Checkpoint selection, shared-input validation, and comparison exports."""
import csv
import json
from pathlib import Path
import re
import numpy as np
from .statistics import save_table,align

MODEL_NAMES={'multi_subarray':'SubspaceNet','transmusic':'TransMUSIC','data_driven_complex':'DataDrivenComplex','esprit':'ESPRIT'}
# Defaults are those of the supplied native SystemModelParams.
SIM_DEFAULTS=dict(modulation='Gaussian',eta=0,bias=0,sv_noise_var=0,snr=10,signal_nature='non-coherent',signal_type='NarrowBand')
SIM_KEYS=['N','M','T','signal_type','signal_nature','modulation','snr','eta','bias','sv_noise_var']

def sim_key(cfg):
    s=cfg['subarray_config'][0]['system_model']
    return {k:s.get(k,SIM_DEFAULTS.get(k)) for k in SIM_KEYS}

def validate_cfg(cfg,path):
    if cfg.get('L')!=1 or len(cfg.get('subarray_config',[]))!=1:raise ValueError(f'{path}: select an L=1 configuration.')
    s=cfg['subarray_config'][0]['system_model']
    if s['M']!=2:raise ValueError(f'{path}: this two-source sweep requires an M=2 checkpoint; do not relabel M=3 weights.')
    if s['signal_type']!='NarrowBand':raise ValueError('Only NarrowBand is supported.')
    if s['N']<=s['M'] or s['T']<2:raise ValueError('Require N>M and T>=2.')

def resolve_targets(args):
    if args.models_json:
        content=json.loads(args.models_json.read_text())
        entries=content['models'] if isinstance(content,dict) else content
    else:
        param='subarray_config_0_system_model_M'
        entries=[]
        for model in args.model_types:
            run=args.pretrained_root/model/('sweep_'+param)/(param+'_2')
            entries.append(dict(model_type=model,checkpoint_path=str(run/model/args.checkpoint_name),config_path=str(run/'config.json')))
    if not entries:raise ValueError('At least one pretrained model is required.')
    targets=[]
    for entry in entries:
        model=entry['model_type']
        if model not in ['multi_subarray','transmusic','data_driven_complex']:raise ValueError(f'Unsupported model: {model}')
        cp=Path(entry['checkpoint_path']).resolve()
        config_path=Path(entry.get('config_path',cp.parent.parent/'config.json')).resolve()
        if not cp.is_file():raise FileNotFoundError(f'Missing pretrained checkpoint: {cp}')
        if not config_path.is_file():raise FileNotFoundError(f'Missing original config: {config_path}. Use --models_json to supply its actual location.')
        cfg=json.loads(config_path.read_text());validate_cfg(cfg,config_path)
        stage=cp.stem.removeprefix('best_').removesuffix('_completed')
        label=entry.get('label',model+'__'+stage)
        if not re.fullmatch(r'[A-Za-z0-9_-]+',label):raise ValueError('Labels must use letters, digits, underscores or hyphens.')
        targets.append(dict(model_type=model,stage=stage,label=label,checkpoint=str(cp),config_path=str(config_path),configuration=cfg))
    if len(set(t['label'] for t in targets))!=len(targets):raise ValueError('Model labels must be unique.')
    simulation=json.loads(args.simulation_config.read_text()) if args.simulation_config else targets[0]['configuration']
    validate_cfg(simulation,args.simulation_config or 'first model config')
    for t in targets:
        own,common=sim_key(t['configuration']),sim_key(simulation)
        if any(own[k]!=common[k] for k in ['N','M','T']):raise ValueError('All pretrained models must match the common simulation N,M,T.')
        differing=[k for k in SIM_KEYS if own[k]!=common[k]]
        if differing and not args.simulation_config:
            raise ValueError(f"{t['label']}: simulator settings differ: {differing}. Select matched checkpoints or supply an explicit --simulation_config for a documented distribution-shift experiment.")
        t['evaluation_shift_fields']=differing
    if args.include_esprit:
        targets.append(dict(model_type='esprit',stage='analytic',label='esprit__analytic',checkpoint=None,config_path=None,configuration=simulation,evaluation_shift_fields=[]))
    return targets,simulation


def save_summaries(root,rows):
    save_table(root/'benchmark_results.csv',rows)
    groups={}
    for row in rows:groups.setdefault(row['label'],[]).append(row)
    for label,part in groups.items():save_table(root/'models'/label/'benchmark_results.csv',part)


def write_trials(root,case,target,pred,truth,cov,arrays):
    matched,raw,swapped=align(pred,truth,cov)
    used=arrays['covariance_used_rad2'];e=arrays['error_rad'];degree=180/np.pi
    path=root/'trial_results'/case['case_id']/(target['label']+'.csv');path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.csv.tmp')
    fields=['case_id','model','trial','theta_anchor_deg','theta_moving_deg','estimate_anchor_deg','estimate_moving_deg',
            'error_anchor_deg','error_moving_deg','raw_C11_deg2','raw_C12_deg2','raw_C21_deg2','raw_C22_deg2',
            'used_C11_deg2','used_C12_deg2','used_C22_deg2','raw_spd','valid_covariance','permuted','NEES_full','NEES_diagonal','NEES_flipped']
    with temporary.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(fields)
        for i in range(len(pred)):
            writer.writerow([case['case_id'],target['label'],i,*np.rad2deg(truth),*np.rad2deg(matched[i]),*np.rad2deg(e[i]),
                *(raw[i].reshape(-1)*degree**2),used[i,0,0]*degree**2,used[i,0,1]*degree**2,used[i,1,1]*degree**2,
                int(arrays['raw_spd'][i]),int(arrays['valid_covariance'][i]),int(swapped[i]),
                arrays['nees_full'][i],arrays['nees_diagonal'][i],arrays['nees_flipped'][i]])
    temporary.replace(path)


def export_all(root):
    from .plotting import plot_results
    from .comparison_plots import plot_comparison
    plot_results(root)
    plot_comparison(root)
