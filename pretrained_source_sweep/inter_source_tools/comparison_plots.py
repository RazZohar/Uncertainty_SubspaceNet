"""Overlay pretrained methods and compare each correlation to its own errors."""
import csv
import json
from pathlib import Path
import numpy as np
from .statistics import save_table
from .plotting import slug,num,tex_escape,XLABEL,COLORS,TEXCOLORS
from .comparison import MODEL_NAMES

SPECS=[('Fixed/first source RMSE (deg)','RMSE_anchor_deg',None),
       ('Moving/second source RMSE (deg)','RMSE_moving_deg',None),
       ('Joint 95% coverage (valid)','joint95_valid_full',.95),
       ('Normalized ANEES (valid)','ANEES_valid_full',1),
       ('Gap std. / empirical std.','gap_std_ratio_valid_full',1),
       ('Invalid covariance fraction','raw_invalid_fraction',0)]


def plot_comparison(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    with (root/'benchmark_results.csv').open() as f:rows=list(csv.DictReader(f))
    metadata=json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else {}
    prefix='SOFTWARE TEST FIXTURE — ' if metadata.get('synthetic_fixture') else ''
    groups={}
    for row in rows:groups.setdefault((row['experiment'],row['snr_parameter']),[]).append(row)
    master=[r'% Load pgfplots and its groupplots library in the paper preamble.']
    for (exp,snr),part in groups.items():
        labels=list(dict.fromkeys(r['label'] for r in part))
        conditions={}
        for r in part:conditions.setdefault(r['case_id'],{})[r['label']]=r
        ordered=sorted(conditions.values(),key=lambda c:num(next(iter(c.values())),'x'))
        wide=[];names={}
        for condition in ordered:
            first=next(iter(condition.values()));out={k:first[k] for k in ['case_id','x','theta_anchor_deg','theta_moving_deg','separation_deg','spatial_separation','case_status']}
            for j,label in enumerate(labels):
                r=condition.get(label,{})
                names[label]=MODEL_NAMES.get(r.get('model_type'),label)+' / '+r.get('stage','')
                for _,field,_ in SPECS:
                    value=num(r,field)
                    if first['case_status']!='regular' and field not in ['RMSE_anchor_deg','RMSE_moving_deg','raw_invalid_fraction']:value=np.nan
                    out[f'm{j}_{field}']=value
                for field in ['rho_empirical_paired','rho_predicted_paired']:
                    out[f'm{j}_{field}']=num(r,field) if first['case_status']=='regular' else np.nan
            wide.append(out)
        stem=slug('all_models_'+exp+'_snr_'+snr);relative=Path('numerical_results')/(stem+'.csv')
        save_table(root/relative,wide)
        x=np.array([num(r,'x') for r in wide])
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        tex=[r'\begin{tikzpicture}',r'\begin{groupplot}[group style={group size=3 by 2,horizontal sep=1.1cm,vertical sep=1.5cm},width=0.29\textwidth,height=0.24\textwidth,grid=major,unbounded coords=jump,tick label style={font=\scriptsize},label style={font=\scriptsize},legend style={font=\tiny,legend columns=2,draw=none,at={(0.5,1.02)},anchor=south}]']
        for ax,(ylabel,field,ref) in zip(axes.flat,SPECS):
            bounds=',ymin=0,ymax=1' if field in ['raw_invalid_fraction','joint95_valid_full'] else ''
            tex.append(r'\nextgroupplot[xlabel={'+XLABEL[exp]+'},ylabel={'+tex_escape(ylabel)+'}'+bounds+']')
            for j,label in enumerate(labels):
                col=f'm{j}_{field}';color=COLORS[j%len(COLORS)]
                y=np.array([num(r,col) for r in wide])
                ax.plot(x,y,color=color,label=names[label],marker='o',markersize=2.2,lw=1.3)
                tex += [r'\addplot+['+TEXCOLORS[j%len(TEXCOLORS)]+r',mark=*,mark size=0.8pt] table[col sep=comma,x=x,y='+col+'] {'+relative.as_posix()+'};',r'\addlegendentry{'+tex_escape(names[label])+'}']
            if ref is not None:
                ax.axhline(ref,color='.4',linestyle=':',lw=1)
                tex.append(r'\addplot[gray,dotted,forget plot] coordinates {('+str(x.min())+','+str(ref)+') ('+str(x.max())+','+str(ref)+')};')
            ax.set(xlabel=XLABEL[exp],ylabel=ylabel);ax.grid(alpha=.2)
            if bounds:ax.set_ylim(0,1)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(prefix+f'{exp}: pretrained methods, native SNR parameter={snr}',fontsize=11)
        fig.savefig(root/'figures'/(stem+'.png'),dpi=180);fig.savefig(root/'figures'/(stem+'.pdf'));plt.close(fig)
        tex += [r'\end{groupplot}',r'\end{tikzpicture}']
        fragment=Path('plotting_latex_code')/(stem+'.tex');(root/fragment).write_text('\n'.join(tex))
        caption=('Pretrained estimators evaluated on identical observations. Point errors use the native modulo-pi convention. '
            'Covariance scores are conditional on each estimator producing valid covariance; invalid fractions are shown explicitly. '
            'Coincident-source and endfire points are omitted from regular Gaussian uncertainty curves. Conditional scores must be interpreted together with invalidity.')
        master += [r'\begin{figure*}[p]\centering',r'\input{'+fragment.as_posix()+'}',r'\caption{'+caption+'}',r'\end{figure*}']
        # Each model is compared to its OWN empirical error correlation.
        nrows=(len(labels)+1)//2
        fig,axes=plt.subplots(nrows,2,figsize=(9,3.2*nrows),squeeze=False,layout='constrained')
        tex=[r'\begin{tikzpicture}',r'\begin{groupplot}[group style={group size=2 by '+str(nrows)+r',horizontal sep=1.5cm,vertical sep=1.5cm},width=0.43\textwidth,height=0.27\textwidth,grid=major,unbounded coords=jump,ymin=-1,ymax=1,tick label style={font=\scriptsize},title style={font=\scriptsize},legend style={font=\tiny,draw=none}]']
        for j,label in enumerate(labels):
            ax=axes.flat[j];tex.append(r'\nextgroupplot[title={'+tex_escape(names[label])+r'},xlabel={'+XLABEL[exp]+r'},ylabel={Error correlation}]')
            for k,(field,name) in enumerate([('rho_empirical_paired','Empirical'),('rho_predicted_paired','Predicted')]):
                col=f'm{j}_{field}';ax.plot(x,[num(r,col) for r in wide],color=COLORS[k],label=name,linestyle='-' if k==0 else '--',marker='o',ms=2)
                tex += [r'\addplot+['+TEXCOLORS[k]+(','+'dashed' if k else '')+r',mark=*,mark size=0.8pt] table[col sep=comma,x=x,y='+col+'] {'+relative.as_posix()+'};',r'\addlegendentry{'+name+'}']
            ax.set(title=names[label],xlabel=XLABEL[exp],ylabel='Error correlation',ylim=(-1,1));ax.grid(alpha=.2);ax.legend(fontsize=8)
        if prefix:fig.suptitle(prefix)
        for j in range(len(labels),nrows*2):axes.flat[j].set_visible(False)
        fig.savefig(root/'figures'/(stem+'_correlation.png'),dpi=180);fig.savefig(root/'figures'/(stem+'_correlation.pdf'));plt.close(fig)
        tex += [r'\end{groupplot}',r'\end{tikzpicture}'];frag2=Path('plotting_latex_code')/(stem+'_correlation.tex');(root/frag2).write_text('\n'.join(tex))
        master += [r'\begin{figure*}[p]\centering',r'\input{'+frag2.as_posix()+'}',r'\caption{Predicted versus empirical error correlation for each estimator, evaluated on its own valid covariance subset. Each empirical curve is computed from that estimator\textquotesingle s own errors.}',r'\end{figure*}']
    master_path=Path('plotting_latex_code/all_benchmark_figures_pgfplots.tex');(root/master_path).write_text('\n'.join(master))
    (root/'all_benchmark_figures_standalone.tex').write_text('\n'.join([r'\documentclass{article}',r'\usepackage[margin=1cm]{geometry}',r'\usepackage{pgfplots}',r'\usepgfplotslibrary{groupplots}',r'\pgfplotsset{compat=1.18}',r'\begin{document}',r'\input{'+master_path.as_posix()+'}',r'\end{document}']))
    print('Combined CSV and comparison figures saved to',root,flush=True)
