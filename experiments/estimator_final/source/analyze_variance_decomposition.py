from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL={'hybrid':'Proposed hybrid','v5_raw':'CPT-PG v5 baseline'}
METHODS=['hybrid','v5_raw']
DIM=10

def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(path.with_suffix('.'+ext),dpi=240,bbox_inches='tight')
    plt.close(fig)

def arrays(ep,method):
    f=ep[ep.method==method].sort_values(['replicate','episode'])
    reps=np.sort(f.replicate.unique());ks=np.sort(f.episode.unique())
    if len(f)!=len(reps)*len(ks): raise ValueError('incomplete grid')
    theta=f[[f'theta_{j}' for j in range(DIM)]].to_numpy(float).reshape(len(reps),len(ks),DIM)
    q2=f.q_squared.to_numpy(float).reshape(len(reps),len(ks))
    wealth=f.wealth.to_numpy(float).reshape(len(reps),len(ks))
    return dict(reps=reps,episodes=ks,theta=theta,q2=q2,wealth=wealth)

def curves(arr):
    return dict(theta_trace_variance=np.var(arr['theta'],axis=0,ddof=1).sum(axis=1),
                q2_variance=np.var(arr['q2'],axis=0,ddof=1),
                wealth_variance=np.var(arr['wealth'],axis=0,ddof=1),
                mean_q2=np.mean(arr['q2'],axis=0),mean_wealth=np.mean(arr['wealth'],axis=0))

def dispersion_frame(aa):
    rows=[]
    for m,a in aa.items():
        c=curves(a)
        for i,k in enumerate(a['episodes']): rows.append(dict(method=m,episode=int(k),n=len(a['reps']),**{x:float(y[i]) for x,y in c.items()}))
    return pd.DataFrame(rows)

def sampled_summary(a,idx,metric,mode):
    if metric=='theta_trace_variance': curve=np.var(a['theta'][idx],axis=0,ddof=1).sum(axis=1)
    elif metric=='q2_variance': curve=np.var(a['q2'][idx],axis=0,ddof=1)
    elif metric=='wealth_variance': curve=np.var(a['wealth'][idx],axis=0,ddof=1)
    else: raise ValueError(metric)
    return float(curve[-1] if mode=='final' else curve.mean())

def bootstrap_ratios(aa,seed=20260916,B=4000):
    nrep=len(aa['hybrid']['reps']);rng=np.random.default_rng(seed);rows=[]
    for metric in ['theta_trace_variance','q2_variance','wealth_variance']:
      for mode in ['final','time_average']:
        ids=np.arange(nrep);h=sampled_summary(aa['hybrid'],ids,metric,mode);v=sampled_summary(aa['v5_raw'],ids,metric,mode);ratio=h/v
        boots=np.empty(B)
        for b in range(B):
            ix=rng.integers(0,nrep,size=nrep);hh=sampled_summary(aa['hybrid'],ix,metric,mode);vv=sampled_summary(aa['v5_raw'],ix,metric,mode);boots[b]=hh/vv
        lo,hi=np.quantile(boots,[.025,.975])
        rows.append(dict(metric=metric,summary=mode,hybrid=h,v5=v,ratio=ratio,ratio_ci_low=lo,ratio_ci_high=hi,percent_change=(ratio-1)*100))
    return pd.DataFrame(rows)

def terminal_wealth_metrics(daily):
    rows=[]
    for (m,r),f in daily.groupby(['method','replicate']):
        f=f.sort_values('day');ret=f.net_return.to_numpy(float);sd=ret.std(ddof=1);terminal=float(f.wealth.iloc[-1])
        rows.append(dict(method=m,replicate=int(r),terminal_wealth=terminal,cumulative_return=terminal-1,
            annualized_return=terminal**(252/len(f))-1,annualized_volatility=sd*np.sqrt(252),
            sharpe=(np.sqrt(252)*ret.mean()/sd if sd>0 else np.nan),max_drawdown=float(f.drawdown.max()),
            total_turnover=float(f.turnover.sum()),cumulative_fee=float(f.fee.sum())))
    return pd.DataFrame(rows)

def wealth_dispersion(ws,nrep,seed=20260917,B=5000):
    rng=np.random.default_rng(seed);rows=[]
    for metric in ['terminal_wealth','cumulative_return','annualized_return','annualized_volatility','sharpe','max_drawdown','total_turnover']:
        h=ws[ws.method=='hybrid'].sort_values('replicate')[metric].to_numpy(float);v=ws[ws.method=='v5_raw'].sort_values('replicate')[metric].to_numpy(float)
        hv=np.var(h,ddof=1);vv=np.var(v,ddof=1);ratio=hv/vv;boots=np.empty(B)
        for b in range(B):
            ix=rng.integers(0,nrep,size=nrep);boots[b]=np.var(h[ix],ddof=1)/np.var(v[ix],ddof=1)
        lo,hi=np.quantile(boots,[.025,.975])
        rows.append(dict(metric=metric,hybrid_mean=np.mean(h),v5_mean=np.mean(v),hybrid_variance=hv,v5_variance=vv,
                         variance_ratio=ratio,ratio_ci_low=lo,ratio_ci_high=hi,percent_change=(ratio-1)*100))
    return pd.DataFrame(rows)

def plot_experiment(disp,out,title):
    fig,axs=plt.subplots(2,2,figsize=(13,9));spec=[('theta_trace_variance','Trace covariance of learned $\\theta_k$'),('q2_variance','Variance of common-CRN diagnostic $Q_k^2$'),('wealth_variance','Variance of episode-end wealth')]
    for ax,(col,ttl) in zip(axs.ravel()[:3],spec):
        for m in METHODS:
            f=disp[disp.method==m];ax.plot(f.episode,f[col],label=LABEL[m],lw=2)
        ax.set_yscale('symlog',linthresh=1e-12);ax.set_xlabel('Learning episode');ax.set_title(ttl);ax.grid(alpha=.2)
    axs[0,0].legend();ax=axs[1,1]
    for col,label in [('theta_trace_variance','theta trace'),('q2_variance','Q²'),('wealth_variance','wealth')]:
        p=disp.pivot(index='episode',columns='method',values=col);ax.plot(p.index,p.hybrid/p.v5_raw,label=label,lw=1.8)
    ax.axhline(1,ls='--',lw=1);ax.set_yscale('log');ax.set_xlabel('Learning episode');ax.set_ylabel('Hybrid variance / v5 variance');ax.set_title('Variance ratio over time');ax.grid(alpha=.2);ax.legend()
    fig.suptitle(title,fontsize=14);fig.tight_layout(rect=[0,0,1,.95]);save(fig,out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);a=ap.parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for ex in ['A','B']:
        inp=a.input_root/f'experiment_{ex}';ep=pd.read_csv(inp/'episodes.csv');dy=pd.read_csv(inp/'daily.csv');aa={m:arrays(ep,m) for m in METHODS};nrep=len(aa['hybrid']['reps'])
        if not np.array_equal(aa['hybrid']['reps'],aa['v5_raw']['reps']):raise ValueError('replicate pairing mismatch')
        disp=dispersion_frame(aa);rat=bootstrap_ratios(aa,seed=20260916+(0 if ex=='A' else 1));ws=terminal_wealth_metrics(dy);wd=wealth_dispersion(ws,nrep,seed=20260918+(0 if ex=='A' else 1))
        disp.to_csv(a.output_root/f'experiment_{ex}_dispersion_by_episode.csv',index=False);rat.to_csv(a.output_root/f'experiment_{ex}_variance_ratio_summary.csv',index=False);ws.to_csv(a.output_root/f'experiment_{ex}_wealth_metrics.csv',index=False);wd.to_csv(a.output_root/f'experiment_{ex}_wealth_variance_summary.csv',index=False)
        title=('Experiment A: estimator-randomness-induced online variability\nFixed execution-return path, action RNG and evaluator; 64 estimator RNG replications' if ex=='A' else 'Experiment B: market-path-induced online variability\nFixed estimator RNG, action RNG and evaluator; 64 execution-return paths')
        plot_experiment(disp,a.output_root/f'experiment_{ex}_variance_decomposition',title);tmp=rat.copy();tmp.insert(0,'experiment',ex);summaries.append(tmp)
    pd.concat(summaries,ignore_index=True).to_csv(a.output_root/'variance_ratio_summary_all.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(13,5.2),sharey=True)
    for ax,ex in zip(axs,['A','B']):
        r=pd.read_csv(a.output_root/f'experiment_{ex}_variance_ratio_summary.csv');r=r[r.summary=='time_average'];x=np.arange(3);vals=r.ratio.to_numpy();lo=vals-r.ratio_ci_low.to_numpy();hi=r.ratio_ci_high.to_numpy()-vals
        ax.errorbar(x,vals,yerr=[lo,hi],fmt='o',capsize=5);ax.axhline(1,ls='--',lw=1);ax.set_xticks(x,['theta trace','Q²','wealth']);ax.set_yscale('log');ax.grid(axis='y',alpha=.2);ax.set_title('A: vary estimator RNG' if ex=='A' else 'B: vary market path')
    axs[0].set_ylabel('Hybrid / v5 time-averaged variance (95% paired bootstrap CI)');fig.suptitle('Separated sources of online variability');fig.tight_layout(rect=[0,0,1,.93]);save(fig,a.output_root/'variance_decomposition_summary')
    lines=['# Online variance decomposition','', 'Experiment A holds the realized execution-return path, execution-policy random numbers, and evaluation random numbers fixed; only the training-estimator random seed changes. Experiment B holds the estimator, execution-policy, and evaluation random numbers fixed; only the semi-synthetic realized execution-return path changes. The diagnostic evaluator uses common random numbers and therefore does not contribute an independent source of cross-replication noise.','']
    for ex in ['A','B']:
        lines += [f'## Experiment {ex}',pd.read_csv(a.output_root/f'experiment_{ex}_variance_ratio_summary.csv').to_markdown(index=False),'','Terminal wealth-metric variance:',pd.read_csv(a.output_root/f'experiment_{ex}_wealth_variance_summary.csv').to_markdown(index=False),'']
    (a.output_root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':main()
