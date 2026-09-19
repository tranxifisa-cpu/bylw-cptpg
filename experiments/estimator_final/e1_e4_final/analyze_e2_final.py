from __future__ import annotations
from pathlib import Path
import argparse, json, math
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL={'hybrid':'Proposed estimator','v5_raw':'CPT-PG v5 plug-in'}
MARK={'hybrid':'o','v5_raw':'s'}
PERIODS=[20,50,90,150]; BUDGETS=[32,64,128,256,512,1024,2048]; DIMS=[2,4,8,10]; DIM_BUDGETS=[64,128,256,512,1024,2048]
NAMES={20:'Stable',50:'Post-change',90:'Mid-drift',150:'Post-drift stationary'}
COORD=2

def ref_stats(path):
    r=np.load(path)['estimates']; return r.mean(0),r.var(0,ddof=1)/len(r)

def ci_mean(x):
    x=np.asarray(x,float); se=x.std(ddof=1)/np.sqrt(len(x)); return x.mean()-1.96*se,x.mean()+1.96*se

def bootstrap_ci(x,seed=0,B=4000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);n=len(x)
    vals=np.empty(B)
    for b in range(B): vals[b]=x[rng.integers(0,n,n)].mean()
    return np.quantile(vals,[.025,.975])

def save(fig,folder,name):
    for ext in ('png','pdf','svg'): fig.savefig(folder/f'{name}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--figures',type=Path,required=True);ap.add_argument('--tables',type=Path,required=True);a=ap.parse_args()
    a.figures.mkdir(parents=True,exist_ok=True);a.tables.mkdir(parents=True,exist_ok=True)
    rows=[]
    refs={}
    for p in PERIODS:
        ref,rv=ref_stats(a.results/f'period_{p:03d}'/'reference.npz');refs[p]=(ref,rv)
        for B in BUDGETS:
            z=np.load(a.results/f'period_{p:03d}'/f'budget_{B:04d}.npz'); methods=list(z['methods']);ec=z['expected_costs']
            for mi,m in enumerate(methods):
                x=z['estimates'][:,mi,:];mean=x.mean(0);var=x.var(0,ddof=1);n=len(x)
                bias=mean[COORD]-ref[COORD];se=np.sqrt(var[COORD]/n+rv[COORD]);raw=np.mean(np.sum((x-ref)**2,axis=1));adj=max(raw-rv.sum(),0.0)
                rows.append(dict(period=p,period_name=NAMES[p],budget=B,method=m,repetitions=n,expected_cost=float(ec[mi]),coordinate_bias=float(bias),bias_low=float(bias-1.96*se),bias_high=float(bias+1.96*se),variance_trace=float(var.sum()),mse_raw=float(raw),mse_reference_adjusted=float(adj),reference_variance=float(rv.sum()),actual_cost=float(z['calls'][:,mi].mean())))
    tab=pd.DataFrame(rows);tab.to_csv(a.tables/'e2_bias_mse_all.csv',index=False)

    # Replication variance + studentization at stable period.
    z=np.load(a.results/'replication'/'outputs.npz');des=json.loads((a.results/'replication'/'design.json').read_text());methods=list(z['methods']);tr=des['trials'];mx=des['max_M'];x=z['estimates'].reshape(tr,mx,2,10);ref,rv=refs[20]
    moments=[];student=[];qq=[]
    for M in des['M_grid']:
        means=x[:,:M].mean(1)
        for mi,m in enumerate(methods):
            mm=means[:,mi,:]; moments.append(dict(M=M,method=m,variance_trace=float(mm.var(0,ddof=1).sum()),mse=float(max(np.mean(np.sum((mm-ref)**2,axis=1))-rv.sum(),0.0))))
            if M>1:
                sd=x[:,:M,mi,COORD].std(1,ddof=1);t=np.sqrt(M)*(means[:,mi,COORD]-ref[COORD])/sd
                student.append(dict(M=M,method=m,coverage95=float(np.mean(np.abs(t)<=stats.t.ppf(.975,M-1))),t_mean=float(t.mean()),t_sd=float(t.std(ddof=1)),ks=float(stats.kstest(t,'norm').statistic)))
                if M==64:
                    for v in t: qq.append(dict(method=m,t=float(v)))
    mt=pd.DataFrame(moments);mt.to_csv(a.tables/'e2_replication.csv',index=False);pd.DataFrame(student).to_csv(a.tables/'e2_studentized.csv',index=False)

    # True dimension runs.
    drows=[]
    for d in DIMS:
        r,rvd=ref_stats(a.results/'dimension'/f'd_{d:02d}'/'reference.npz')
        for B in DIM_BUDGETS:
            z=np.load(a.results/'dimension'/f'd_{d:02d}'/f'budget_{B:04d}.npz')
            for mi,m in enumerate(list(z['methods'])):
                xx=z['estimates'][:,mi,:];raw=np.mean(np.sum((xx-r)**2,axis=1));drows.append(dict(dimension=d,budget=B,method=m,mse=float(max(raw-rvd.sum(),0.0)),variance_trace=float(xx.var(0,ddof=1).sum()),expected_cost=float(z['expected_costs'][mi])))
    dt=pd.DataFrame(drows);dt.to_csv(a.tables/'e2_dimension_budget.csv',index=False)

    # Four-period standard-budget MSE table with mean squared-error-difference CI.
    std=[]
    for p in PERIODS:
        B=512;ref,rv=refs[p];z=np.load(a.results/f'period_{p:03d}'/f'budget_{B:04d}.npz');est=z['estimates']
        rawerr=np.sum((est-ref[None,None,:])**2,axis=2)
        hmean=max(float(rawerr[:,0].mean()-rv.sum()),0.0);vmean=max(float(rawerr[:,1].mean()-rv.sum()),0.0)
        diff=rawerr[:,0]-rawerr[:,1]
        lo,hi=bootstrap_ci(diff,seed=2026+p)
        std.append(dict(period=p,market_period=NAMES[p],budget=B,hybrid_mse=hmean,v5_mse=vmean,ratio=float(hmean/vmean),difference=float(diff.mean()),difference_ci_low=float(lo),difference_ci_high=float(hi)))
    stdf=pd.DataFrame(std);stdf.to_csv(a.tables/'e2_mse_standard_budget.csv',index=False)

    # Figure 3 style four panels.
    fig,axs=plt.subplots(2,2,figsize=(12.6,8.7));
    ax=axs[0,0]
    for m in methods:
        d=tab[(tab.period==20)&(tab.method==m)].sort_values('expected_cost')
        ax.errorbar(d.expected_cost,d.coordinate_bias,yerr=[d.coordinate_bias-d.bias_low,d.bias_high-d.coordinate_bias],marker=MARK[m],capsize=3,lw=1.7,label=LABEL[m])
    ax.axhline(0,color='0.25',ls=':',lw=1);ax.set_xscale('log',base=2);ax.set_xlabel('Expected trajectory budget $\\mathsf{B}$');ax.set_ylabel('Bias in the momentum coordinate');ax.set_title('A  Finite-sample bias',loc='left',fontweight='bold');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)

    ax=axs[0,1]
    for m in methods:
        d=mt[mt.method==m].sort_values('M');ax.loglog(d.M,d.variance_trace,marker=MARK[m],lw=1.7,label=LABEL[m])
    base=mt[mt.method=='hybrid'].sort_values('M'); guide=1.12*base.variance_trace.iloc[0]/base.M
    ax.loglog(base.M,guide,ls=':',color='0.25',lw=1.5,label='$M^{-1}$ rate guide')
    ax.set_xlabel('Independent complete outputs $M$');ax.set_ylabel('Trace covariance');ax.set_title('B  Variance and $M^{-1}$ scaling',loc='left',fontweight='bold');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)

    ax=axs[1,0]
    qdf=pd.DataFrame(qq)
    for m in methods:
        vals=np.sort(qdf[qdf.method==m].t.to_numpy());q=stats.norm.ppf((np.arange(len(vals))+.5)/len(vals));ax.plot(q,vals,marker=MARK[m],markersize=2.8,lw=1,label=LABEL[m])
    lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],ls=':',color='0.25',lw=1)
    ax.set_xlabel('Standard-normal quantiles');ax.set_ylabel('Studentized quantiles');ax.set_title('C  Studentized normality',loc='left',fontweight='bold');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=8)

    ax=axs[1,1]
    # Proposed estimator emphasized; v5 shown faintly to preserve the direct comparator without crowding.
    for d in DIMS:
        a1=dt[(dt.dimension==d)&(dt.method=='hybrid')].sort_values('expected_cost');ax.loglog(a1.expected_cost,a1.mse,marker='o',ms=3.5,lw=1.4,label=f'Proposed, d={d}')
    guide_x=np.array(DIM_BUDGETS,float); y0=dt[(dt.dimension==10)&(dt.method=='hybrid')].sort_values('expected_cost').mse.iloc[0]; ax.loglog(guide_x,1.25*y0*DIM_BUDGETS[0]/guide_x,ls=':',color='0.25',lw=1.4,label='$\\mathsf{B}^{-1}$ rate guide')
    ax.set_xlabel('Expected trajectory budget $\\mathsf{B}$');ax.set_ylabel('Gradient MSE');ax.set_title('D  Fixed-dimensional budget scaling',loc='left',fontweight='bold');ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=7,ncol=2)
    fig.tight_layout();save(fig,a.figures,'e2_estimator_diagnostics')

    variance_slopes={}
    for m in methods:
        dd=mt[mt.method==m].sort_values('M');variance_slopes[m]=float(np.polyfit(np.log(dd.M),np.log(dd.variance_trace),1)[0])
    dim_slopes={}
    for d in DIMS:
        dd=dt[(dt.dimension==d)&(dt.method=='hybrid')].sort_values('expected_cost');dim_slopes[str(d)]=float(np.polyfit(np.log(dd.expected_cost),np.log(dd.mse),1)[0])
    summary={'standard_budget':std,'studentized_M64':[r for r in student if r['M']==64], 'mse_better_periods_at_512':int((stdf.hybrid_mse<stdf.v5_mse).sum()),'variance_loglog_slopes':variance_slopes,'hybrid_budget_mse_slopes_by_dimension':dim_slopes}
    (a.tables/'e2_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(stdf.to_string(index=False));print(pd.DataFrame(summary['studentized_M64']).to_string(index=False))
if __name__=='__main__': main()
