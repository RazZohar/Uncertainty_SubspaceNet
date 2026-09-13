"""Real-result CSV, PNG and matching PGFPlots exports (no placeholder results)."""
import csv
import json
import re
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from .statistics import save_table

COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
TEXCOLORS=['blue!75!black','orange!85!black','green!50!black','magenta!70!black']
XLABEL={'fixed_anchor':r'Moving source angle (deg)', 'symmetric_separation':r'Angular separation (deg)',
        'fixed_separation':r'Pair center (deg)', 'coherence':r'Waveform correlation',
        'power_imbalance':r'Moving-source attenuation (dB)'}

def slug(s):return re.sub(r'[^A-Za-z0-9_-]+','_',str(s))
def num(row,key):
    try:return float(row[key])
    except (KeyError,ValueError,TypeError):return float('nan')
def tex_escape(s):return str(s).replace('_',r'\_').replace('%',r'\%')


def panels():
    return [
        ('Periodic RMSE (deg)',[('RMSE_anchor_deg','Anchor'),('RMSE_moving_deg','Moving')],None),
        ('Error correlation',[('rho_empirical_paired','Empirical'),('rho_predicted_paired','Predicted')],0),
        ('Joint 95% coverage (valid)',[(f'joint95_valid_{v}',v.title()) for v in ['full','diagonal','flipped']],.95),
        ('Normalized ANEES (valid)',[(f'ANEES_valid_{v}',v.title()) for v in ['full','diagonal','flipped']],1),
        ('Gap std. / empirical std.',[(f'gap_std_ratio_valid_{v}',v.title()) for v in ['full','diagonal','flipped']],1),
        ('Invalid covariance fraction',[('raw_invalid_fraction','Raw invalid'),('projection_fraction','Projected')],0)]


def export_group(root,rows,key,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    exp,snr,label=key;stem=slug(f'{exp}_snr_{snr}_{label}')
    rows=sorted(rows,key=lambda r:num(r,'x'))
    # Degenerate source identities and endpoint chart singularities are retained
    # in raw exports but not interpreted as regular Gaussian UQ cases.
    display=[]
    for row in rows:
        r=dict(row)
        if r['case_status']!='regular':
            for key in r:
                if any(key.startswith(p) for p in ['rho_','ANEES_','log_ANEES_','joint95_','gap_','midpoint_','NLL_','paired_NLL_']):r[key]=float('nan')
        display.append(r)
    data=Path('numerical_results')/(stem+'.csv');save_table(root/data,display)
    x=np.array([num(r,'x') for r in display]);specs=panels()
    fig,axs=plt.subplots(2,3,figsize=(12,6.8),layout='constrained')
    latex=[r'\begin{tikzpicture}',r'\begin{groupplot}[group style={group size=3 by 2,horizontal sep=1.15cm,vertical sep=1.4cm},',
           r'width=0.31\textwidth,height=0.24\textwidth,grid=major,unbounded coords=jump,',
           r'tick label style={font=\scriptsize},label style={font=\scriptsize},',
           r'legend style={font=\tiny,draw=none,fill=white,at={(0.5,1.02)},anchor=south,legend columns=2}]']
    for ax,(ylabel,series,ref) in zip(axs.flat,specs):
        extra=',ymin=0,ymax=1' if ylabel=='Invalid covariance fraction' else ''
        latex.append(r'\nextgroupplot['+extra.lstrip(',')+(',' if extra else '')+'xlabel={'+XLABEL[exp]+r'},ylabel={'+tex_escape(ylabel)+'}]')
        for j,(field,name) in enumerate(series):
            y=np.array([num(r,field) for r in display])
            ax.plot(x,y,marker='o',markersize=2.6,lw=1.4,color=COLORS[j],label=name)
            latex += [r'\addplot+['+TEXCOLORS[j]+r',mark=*,mark size=0.9pt] table[col sep=comma,x=x,y='+field+'] {'+data.as_posix()+'};',r'\addlegendentry{'+name+'}']
        if ref is not None:
            ax.axhline(ref,color='0.4',ls=':',lw=1)
            latex.append(r'\addplot[gray,dotted,forget plot] coordinates {('+str(x.min())+','+str(ref)+') ('+str(x.max())+','+str(ref)+')};')
        if exp=='fixed_anchor':
            anchor=num(rows[0],'theta_anchor_deg');ax.axvline(anchor,color='0.7',ls='--',lw=.7)
        ax.set(xlabel=XLABEL[exp],ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7)
        if ylabel=='Invalid covariance fraction':ax.set_ylim(0,1)
    title=f'{label} | {exp} | native SNR parameter={snr}; covariance policy={rows[0]["covariance_policy"]}'
    if manifest.get('synthetic_fixture'):title='SYNTHETIC SOFTWARE TEST — '+title
    fig.suptitle(title,fontsize=10)
    fig.savefig(root/'figures'/(stem+'.png'),dpi=180);fig.savefig(root/'figures'/(stem+'.pdf'));plt.close(fig)
    latex += [r'\end{groupplot}',r'\end{tikzpicture}']
    fragment=Path('plotting_latex_code')/(stem+'.tex');(root/fragment).write_text('\n'.join(latex)+'\n')
    caption=(f'{exp.replace("_"," ").capitalize()} for {label.replace("_"," ")}. '
        'Full, diagonal, and sign-flipped covariance use identical point estimates and marginal variances. '
        'Covariance metrics use the same valid covariance subset; the final panel reports raw invalid outputs. '
        'Coincident-angle and endfire cases are excluded from regular Gaussian uncertainty curves. '
        'The ideal normalized ANEES and gap standard-deviation ratio equal one.')
    master=[r'\documentclass{article}',r'\usepackage[margin=1cm]{geometry}',r'\usepackage{pgfplots}',r'\usepgfplotslibrary{groupplots}',
        r'\pgfplotsset{compat=1.18}',r'\begin{document}',r'\begin{figure}[p]',r'\centering',r'\input{'+fragment.as_posix()+'}',
        r'\caption{'+tex_escape(caption)+'}',r'\end{figure}',r'\end{document}']
    (root/(stem+'_standalone.tex')).write_text('\n'.join(master)+'\n')
    likelihood_plot(root,display,stem,data,exp)
    # Spatial-frequency plot keeps the two sides of the anchor separate.
    if exp=='fixed_anchor':
        fig,ax=plt.subplots(figsize=(5.6,3.6),layout='constrained')
        lines=[r'\begin{tikzpicture}',r'\begin{axis}[width=0.65\textwidth,xlabel={$|\sin\theta_2-\sin\theta_1|$},ylabel={Anchor RMSE (deg)},grid=major,legend pos=north west]']
        for j,(condition,name) in enumerate([(lambda r:num(r,'theta_moving_deg')<num(r,'theta_anchor_deg'),'Below anchor'),(lambda r:num(r,'theta_moving_deg')>num(r,'theta_anchor_deg'),'Above anchor')]):
            part=[r for r in display if condition(r) and r['case_status']=='regular']
            if not part:continue
            part=sorted(part,key=lambda r:num(r,'spatial_separation'))
            path=Path('numerical_results')/(stem+f'_branch{j}.csv');save_table(root/path,part)
            ax.scatter([num(r,'spatial_separation') for r in part],[num(r,'RMSE_anchor_deg') for r in part],s=18,color=COLORS[j],label=name)
            lines += [r'\addplot[only marks,mark=*,mark size=1.2pt,'+TEXCOLORS[j]+r'] table[col sep=comma,x=spatial_separation,y=RMSE_anchor_deg] {'+path.as_posix()+'};',r'\addlegendentry{'+name+'}']
        ax.set(xlabel=r'$|\sin\theta_2-\sin\theta_1|$',ylabel='Anchor RMSE (deg)');ax.legend();ax.grid(alpha=.2)
        fig.savefig(root/'figures'/(stem+'_spatial.png'),dpi=180);plt.close(fig)
        lines += [r'\end{axis}',r'\end{tikzpicture}'];(root/'plotting_latex_code'/(stem+'_spatial.tex')).write_text('\n'.join(lines))
        ellipse_plots(root,rows,stem,manifest)
    return dict(stem=stem,experiment=exp,label=label,snr=snr,latex=fragment.as_posix(),data=data.as_posix())


def ellipse_points(cov,center):
    vals,vec=np.linalg.eigh(cov)
    if not np.isfinite(vals).all() or vals.min()<=0:return None
    angle=np.linspace(0,2*np.pi,201)
    return (vec@np.diag(np.sqrt(vals*chi2.ppf(.95,2)))@np.array([np.cos(angle),np.sin(angle)])).T+center


def ellipse_plots(root,rows,stem,manifest):
    import matplotlib.pyplot as plt
    requested=manifest.get('ellipse_moving_values',[-40,35,45,70])
    selected=[r for r in rows if r['case_status']=='regular' and any(abs(num(r,'theta_moving_deg')-v)<1e-7 for v in requested)]
    for row in selected:
        base=root/'cases'/row['case_id']/row['label']
        if not base.with_suffix('.npz').exists():continue
        with np.load(base.with_suffix('.npz')) as f:
            mask=f['valid_covariance'];error=f['error_rad'][mask]*180/np.pi;cov=f['covariance_used_rad2'][mask]*(180/np.pi)**2
        if len(error)<3:continue
        angle=num(row,'theta_moving_deg');name=stem+'_ellipse_'+slug(angle)
        # The cloud and empirical covariance use the SAME eligible trials.
        keep=np.linspace(0,len(error)-1,min(len(error),800),dtype=int)
        scatter_path=Path('numerical_results')/(name+'_cloud.csv')
        save_table(root/scatter_path,[dict(anchor=e[0],moving=e[1]) for e in error[keep]])
        avg=cov.mean(0)
        curves=[('Empirical',np.cov(error,rowvar=False),error.mean(0)),('Full',avg,np.zeros(2)),('Diagonal',np.diag(np.diag(avg)),np.zeros(2))]
        fig,ax=plt.subplots(figsize=(4.4,4.4),layout='constrained');ax.scatter(error[keep,0],error[keep,1],s=3,c='.55',alpha=.25)
        lines=[r'\begin{tikzpicture}',r'\begin{axis}[width=0.43\textwidth,axis equal image,xlabel={Anchor error (deg)},ylabel={Moving error (deg)},grid=major,legend style={font=\scriptsize},legend pos=north west]',
               r'\addplot[only marks,mark size=0.4pt,gray,opacity=0.3,forget plot] table[col sep=comma,x=anchor,y=moving] {'+scatter_path.as_posix()+'};']
        for j,(label,c,center) in enumerate(curves):
            points=ellipse_points(c,center)
            if points is None:continue
            path=Path('numerical_results')/(name+'_'+label.lower()+'.csv')
            save_table(root/path,[dict(anchor=e[0],moving=e[1]) for e in points])
            ax.plot(points[:,0],points[:,1],color=COLORS[j],label=label)
            lines += [r'\addplot[thick,no marks,'+TEXCOLORS[j]+r'] table[col sep=comma,x=anchor,y=moving] {'+path.as_posix()+'};',r'\addlegendentry{'+label+'}']
        ax.set(xlabel='Anchor error (deg)',ylabel='Moving error (deg)',title=f'Moving={angle:g} deg; valid {len(error)}/{int(num(row,"K"))}')
        ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(root/'figures'/(name+'.png'),dpi=180);fig.savefig(root/'figures'/(name+'.pdf'));plt.close(fig)
        lines += [r'\end{axis}',r'\end{tikzpicture}'];(root/'plotting_latex_code'/(name+'.tex')).write_text('\n'.join(lines))


def plot_results(root):
    root=Path(root)
    with (root/'benchmark_results.csv').open() as f:rows=list(csv.DictReader(f))
    manifest=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else {}
    for directory in ['figures','numerical_results','plotting_latex_code']:(root/directory).mkdir(exist_ok=True)
    groups={}
    for r in rows:groups.setdefault((r['experiment'],r['snr_parameter'],r['label']),[]).append(r)
    index=[export_group(root,rs,key,manifest) for key,rs in groups.items()]
    (root/'figure_index.json').write_text(json.dumps(index,indent=2))
    (root/'plotting_latex_code'/'README.txt').write_text('Compile *_standalone.tex from the RESULTS ROOT, or copy numerical_results/ and plotting_latex_code/ to your paper root. Load pgfplots and its groupplots library. Ellipses show Gaussian contours of mean covariance, not measured 95% coverage; joint coverage uses per-trial covariance. Degenerate points are kept in benchmark_results.csv but masked from regular UQ curves. No automatic success claim is made.\n')
    print(f'Wrote {len(index)} figure groups, native-result CSVs and matching PGFPlots to {root}',flush=True)


def likelihood_plot(root,rows,stem,data,exp):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5.6,3.6),layout='constrained')
    lines=[r'\begin{tikzpicture}',r'\begin{axis}[width=0.65\textwidth,xlabel={'+XLABEL[exp]+r'},ylabel={NLL gain of full covariance},grid=major,unbounded coords=jump,legend pos=north east]']
    for j,name in enumerate(['diagonal','flipped']):
        field='paired_NLL_gain_vs_'+name;se='paired_NLL_gain_SE_vs_'+name
        x=[num(r,'x') for r in rows];y=[num(r,field) for r in rows]
        ax.errorbar(x,y,yerr=[1.96*num(r,se) for r in rows],color=COLORS[j],marker='o',ms=3,lw=1,label='vs. '+name)
        lines += [r'\addplot+['+TEXCOLORS[j]+r',mark=*,mark size=1pt,error bars/.cd,y dir=both,y explicit] table[col sep=comma,x=x,y='+field+r',y error expr=1.96*\thisrow{'+se+'}] {'+data.as_posix()+'};',r'\addlegendentry{vs. '+name+'}']
    ax.axhline(0,color='.5',ls=':');ax.set(xlabel=XLABEL[exp],ylabel='NLL gain of full covariance')
    ax.grid(alpha=.2);ax.legend();fig.savefig(root/'figures'/(stem+'_nll_gain.png'),dpi=180);plt.close(fig)
    lines += [r'\end{axis}',r'\end{tikzpicture}'];(root/'plotting_latex_code'/(stem+'_nll_gain.tex')).write_text('\n'.join(lines))
