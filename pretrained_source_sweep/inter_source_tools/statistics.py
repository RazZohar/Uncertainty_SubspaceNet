"""Per-source and joint diagnostics; no silent negative-ANEES clipping."""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import chi2,norm


def save_table(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def periodic(e):return (e+np.pi/2)%np.pi-np.pi/2


def align(pred,truth,cov):
    pred=pred.copy();cov=cov.copy()
    reverse=np.sum(periodic(pred[:,::-1]-truth)**2,1)<np.sum(periodic(pred-truth)**2,1)
    pred[reverse]=pred[reverse,::-1];cov[reverse]=cov[reverse,::-1,::-1]
    return pred,cov,reverse


def safe_correlation(c):
    den=c[0,0]*c[1,1]
    return float(c[0,1]/np.sqrt(den)) if den>0 else float('nan')


def wilson(p,n):
    if n==0:return float('nan'),float('nan')
    z=norm.ppf(.975);d=1+z*z/n;center=(p+z*z/(2*n))/d
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return float(center-half),float(center+half)


def evaluate(pred,truth,cov,policy='report',floor=1e-9,case_status='regular'):
    k=len(pred)
    if pred.shape!=(k,2) or cov.shape!=(k,2,2):raise ValueError('Expected DoA [K,2] and covariance [K,2,2].')
    pred,cov,swapped=align(pred,np.asarray(truth),cov)
    e=periodic(pred-truth)
    if not np.isfinite(e).all():raise ValueError('Nonfinite point estimates: no samples are silently discarded.')
    degree=180/np.pi
    ec=np.cov(e,rowvar=False,ddof=1);moment=e.T@e/k
    raw=(cov+cov.transpose(0,2,1))/2
    finite=np.isfinite(raw).all(axis=(1,2))
    eig=np.full((k,2),np.nan)
    eig[finite]=np.linalg.eigvalsh(raw[finite])
    spd=finite & (eig[:,0]>0)
    used=raw.copy();corrected=np.zeros(k,bool)
    if policy=='project':
        if not finite.all():raise ValueError('Cannot project nonfinite covariance.')
        val,vec=np.linalg.eigh(raw)
        used=(vec*np.maximum(val,floor)[:,None,:])@vec.transpose(0,2,1)
        corrected=val[:,0]<floor
        valid=np.ones(k,bool)
    else:valid=spd
    n=int(valid.sum())
    row=dict(K=k,K_valid_covariance=n,raw_invalid_fraction=float(1-spd.mean()),
        nonfinite_covariance_fraction=float(1-finite.mean()),
        raw_min_eigenvalue=float(np.nanmin(eig)) if finite.any() else float('nan'),
        raw_max_asymmetry=float(np.nanmax(abs(cov-cov.transpose(0,2,1)))) if finite.any() else float('nan'),
        projection_fraction=float(corrected.mean()),
        covariance_policy=policy,
        RMSE_anchor_deg=float(np.sqrt(np.mean(e[:,0]**2))*degree),
        RMSE_moving_deg=float(np.sqrt(np.mean(e[:,1]**2))*degree),
        RMSPE_deg=float(np.mean(np.sqrt(np.mean(e**2,axis=1)))*degree),
        bias_anchor_deg=float(e[:,0].mean()*degree),bias_moving_deg=float(e[:,1].mean()*degree),
        physical_RMSE_anchor_deg=float(np.sqrt(np.mean((pred[:,0]-truth[0])**2))*degree),
        physical_RMSE_moving_deg=float(np.sqrt(np.mean((pred[:,1]-truth[1])**2))*degree),
        chart_cross_fraction=float(np.any(abs(pred-truth)>np.pi/2,axis=1).mean()),
        permutation_fraction=float(swapped.mean()),rho_empirical_all=safe_correlation(ec),
        pair_resolution_failure_fraction=float(np.any(abs(e)>=abs(truth[1]-truth[0])/2,axis=1).mean()) if abs(truth[1]-truth[0])>1e-8 else float('nan'),
        raw_var_anchor_mean_deg2=float(np.nanmean(raw[:,0,0]))*degree**2,
        raw_var_moving_mean_deg2=float(np.nanmean(raw[:,1,1]))*degree**2,
        raw_cov12_mean_deg2=float(np.nanmean(raw[:,0,1]))*degree**2)
    row['raw_rho_mean_covariance']=safe_correlation(np.nanmean(raw,axis=0))
    # All covariance ablations are evaluated on the same eligible trial subset.
    # Invalid outputs count as failures for *_operational_all coverage.
    cv=used[valid];ev=e[valid]
    variants={'full':cv,'diagonal':np.zeros_like(cv),'flipped':cv.copy()}
    if n:
        variants['diagonal'][:,0,0]=cv[:,0,0];variants['diagonal'][:,1,1]=cv[:,1,1]
        variants['flipped'][:,0,1]*=-1;variants['flipped'][:,1,0]*=-1
    arrays=dict(error_rad=e,covariance_used_rad2=used,valid_covariance=valid,raw_spd=spd,
                empirical_covariance_all_rad2=ec,empirical_second_moment_all_rad2=moment)
    empirical_paired=np.cov(ev,rowvar=False,ddof=1) if n>1 else np.full((2,2),np.nan)
    predicted_paired=cv.mean(0) if n else np.full((2,2),np.nan)
    for label,matrix in [('empirical_all',ec),('empirical_paired',empirical_paired),('predicted_paired',predicted_paired)]:
        for i,j in [(0,0),(0,1),(1,1)]:
            row[f'{label}_C{i+1}{j+1}_deg2']=float(matrix[i,j])*degree**2
    row['rho_empirical_paired']=safe_correlation(np.cov(ev,rowvar=False,ddof=1)) if n>1 else float('nan')
    row['rho_predicted_paired']=safe_correlation(cv.mean(0)) if n else float('nan')
    for i,name in enumerate(['anchor','moving']):
        row['empirical_std_'+name+'_paired_deg']=float(np.std(ev[:,i],ddof=1)*degree) if n>1 else float('nan')
        row['predicted_std_'+name+'_paired_deg']=float(np.sqrt(cv[:,i,i].mean())*degree) if n else float('nan')
    nll_values={}
    for name,c in variants.items():
        metrics={key:float('nan') for key in ['ANEES_valid','log_ANEES_valid','NLL_valid',
            'joint95_valid','joint95_valid_low','joint95_valid_high','joint95_operational_all',
            'joint95_operational_low','joint95_operational_high','area95_deg2_valid',
            'gap_std_ratio_valid','midpoint_std_ratio_valid','gap95_valid','midpoint95_valid',
            'marginal95_anchor_valid','marginal95_moving_valid']}
        nees=np.full(k,np.nan)
        if n:
            l=np.linalg.cholesky(c)
            w=np.linalg.solve(l,ev[...,None])[...,0]
            q=np.sum(w*w,axis=1);nees[valid]=q
            logdet=2*np.log(np.diagonal(l,axis1=-2,axis2=-1)).sum(1)
            nll_values[name]=.5*(q+logdet+2*np.log(2*np.pi))
            metrics.update(ANEES_valid=float(q.mean()/2),log_ANEES_valid=float(np.log(q.mean()/2)) if q.mean()>0 else float('-inf'),
                NLL_valid=float(np.mean(.5*(q+logdet+2*np.log(2*np.pi)))),
                joint95_valid=float(np.mean(q<=chi2.ppf(.95,2))),
                joint95_operational_all=float(np.sum(q<=chi2.ppf(.95,2))/k),
                area95_deg2_valid=float(np.mean(np.pi*chi2.ppf(.95,2)*np.exp(logdet/2)))*degree**2)
            metrics['joint95_valid_low'],metrics['joint95_valid_high']=wilson(metrics['joint95_valid'],n)
            metrics['joint95_operational_low'],metrics['joint95_operational_high']=wilson(metrics['joint95_operational_all'],k)
            for key,h in [('gap',np.array([-1.,1.])),('midpoint',np.array([.5,.5]))]:
                error=ev@h;v=np.einsum('i,bij,j->b',h,c,h)
                metrics[key+'_std_ratio_valid']=float(np.sqrt(np.mean(v)/np.var(error,ddof=1))) if n>1 and np.var(error)>0 else float('nan')
                metrics[key+'95_valid']=float(np.mean(abs(error)<=norm.ppf(.975)*np.sqrt(v)))
            marg=abs(ev)<=norm.ppf(.975)*np.sqrt(np.diagonal(c,axis1=-2,axis2=-1))
            metrics['marginal95_anchor_valid'],metrics['marginal95_moving_valid']=map(float,marg.mean(0))
        else:
            metrics['joint95_operational_all']=0.
            metrics['joint95_operational_low'],metrics['joint95_operational_high']=wilson(0,k)
        row.update({key+'_'+name:value for key,value in metrics.items()})
        arrays['nees_'+name]=nees
    for name in ['diagonal','flipped']:
        diff=nll_values[name]-nll_values['full'] if n else np.array([])
        row['paired_NLL_gain_vs_'+name]=float(diff.mean()) if n else float('nan')
        row['paired_NLL_gain_SE_vs_'+name]=float(diff.std(ddof=1)/np.sqrt(n)) if n>1 else float('nan')
    return row,arrays
