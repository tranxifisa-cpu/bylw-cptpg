from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL={'online':'Online update','frozen':'Frozen control'}
STYLE={'online':'-','frozen':'--'}
CHECK=[40,60,120,180,240,300]

def boot(x,seed=0,B=10000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);n=len(x);v=np.empty(B)
    for i in range(B):v[i]=x[rng.integers(0,n,n)].mean()
    return np.quantile(v,[.025,.975])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--figures',type=Path,required=True);ap.add_argument('--tables',type=Path,required=True);a=ap.parse_args();a.figures.mkdir(parents=True,exist_ok=True);a.tables.mkdir(parents=True,exist_ok=True)
    fs=sorted((a.results/'raw').glob('seed_[0-9][0-9][0-9].csv'));D=pd.concat([pd.read_csv(p) for p in fs],ignore_index=True);n=D.seed.nunique();D.to_csv(a.tables/'e3_slow_all.csv',index=False)
    # summaries
    post=D[D.episode>=41].groupby(['seed','method']).agg(res=('projected_residual_term','sum'),dlr=('dlr_term','sum')).reset_index()
    summary={'seeds':int(n)}
    for met in ['res','dlr']:
        p=post.pivot(index='seed',columns='method',values=met);d=p.online-p.frozen;lo,hi=boot(d,2026+len(summary))
        summary['post_'+met]=dict(online=float(p.online.mean()),frozen=float(p.frozen.mean()),relative_improvement=float(1-p.online.mean()/p.frozen.mean()),difference=float(d.mean()),ci_low=float(lo),ci_high=float(hi))
    cps=[]
    for m in LABEL:
        for k in CHECK:
            x=D[(D.method==m)&(D.episode==k)]
            cps.append(dict(method=m,K=k,avg_res=float(x.average_projected_residual.mean()),avg_dlr=float(x.average_dynamic_local_regret.mean()),avg_state_var=float(x.average_state_variation.mean()),avg_exog_var=float(x.average_exogenous_variation.mean())))
    summary['checkpoints']=cps
    # empirical tail slopes based on increments after episode 120
    slopes={}
    for m in LABEL:
        g=D[(D.method==m)&(D.episode>=121)].groupby('episode')[['cumulative_projected_residual','dynamic_local_regret']].mean();t=g.index.to_numpy()-120
        rr=g.cumulative_projected_residual.to_numpy()-g.cumulative_projected_residual.iloc[0]+1e-12;dd=g.dynamic_local_regret.to_numpy()-g.dynamic_local_regret.iloc[0]+1e-12;keep=t>=60
        slopes[m]=dict(residual=float(np.polyfit(np.log(t[keep]),np.log(np.maximum(rr[keep],1e-12)),1)[0]),dlr=float(np.polyfit(np.log(t[keep]),np.log(np.maximum(dd[keep],1e-12)),1)[0]))
    summary['tail_loglog_slopes']=slopes
    (a.tables/'e3_slow_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame(cps).to_csv(a.tables/'e3_slow_checkpoints.csv',index=False)

    # main 2x2
    fig,axs=plt.subplots(2,2,figsize=(12.8,8.5))
    panels=[('cumulative_projected_residual','A  Cumulative squared projected residual',r'$\sum_{k\leq K}\widehat\zeta_k$'),('average_projected_residual','B  Average squared projected residual',r'$K^{-1}\sum_{k\leq K}\widehat\zeta_k$'),('dynamic_local_regret','C  Cumulative DLR',r'$\widehat{\mathrm{DLR}}(K)$'),('average_dynamic_local_regret','D  Average DLR',r'$\widehat{\mathrm{DLR}}(K)/K$')]
    for ax,(col,title,ylab) in zip(axs.ravel(),panels):
        for m in LABEL:
            g=D[D.method==m].groupby('episode')[col].agg(['mean','std']);x=g.index.to_numpy();mean=g['mean'].to_numpy();se=g['std'].to_numpy()/np.sqrt(n)
            ax.plot(x,mean,STYLE[m],lw=1.9,label=LABEL[m]);ax.fill_between(x,mean-1.96*se,mean+1.96*se,alpha=.10)
        for v in [40.5,60.5,120.5]:ax.axvline(v,color='.45',ls=':',lw=.8)
        ax.set_xlim(1,300);ax.set_xlabel('Learning episode $K$');ax.set_ylabel(ylab);ax.set_title(title,loc='left',fontweight='bold');ax.grid(alpha=.15)
    axs[0,0].legend(frameon=False,fontsize=8);fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(a.figures/f'e3_slow_continuous.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

    # diagnostic state variation figure, separate from main evidence
    fig,ax=plt.subplots(figsize=(7.4,4.7))
    g=D[D.method=='online'].groupby('episode')[['state_step_norm','average_state_variation','average_exogenous_variation']].mean()
    ax.plot(g.index,g.state_step_norm,label='Per-episode state-step proxy');ax.plot(g.index,g.average_state_variation,label='Average state variation proxy');ax.plot(g.index,g.average_exogenous_variation,label='Average exogenous variation proxy')
    for v in [40.5,60.5,120.5]:ax.axvline(v,color='.45',ls=':',lw=.8)
    ax.set_xlabel('Episode $K$');ax.set_ylabel('Variation proxy');ax.set_title('Slow-variation construction diagnostic',loc='left',fontweight='bold');ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(a.figures/f'e3_slow_variation_diagnostic.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
