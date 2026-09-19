from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_e4_slow import ETA_GRID,HORIZON_GRID,GAMMA_GRID,VARTTHETA_GRID


def boot(x,seed=0,B=5000):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);n=len(x)
    vals=np.empty(B)
    for b in range(B): vals[b]=x[rng.integers(0,n,n)].mean()
    return np.quantile(vals,[.025,.975])

def save(fig,folder,name):
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'{name}.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--figures',type=Path,required=True);ap.add_argument('--tables',type=Path,required=True);a=ap.parse_args()
    a.figures.mkdir(parents=True,exist_ok=True);a.tables.mkdir(parents=True,exist_ok=True)
    D=pd.read_csv(a.results/'seed_metrics.csv');rows=[]
    for keys,x in D.groupby(['panel','key1','key2'],dropna=False):
        lo,hi=boot(x.delta_track,seed=202609+len(rows));rl,rh=boot(x.delta_residual,seed=202709+len(rows))
        rows.append(dict(panel=keys[0],key1=keys[1],key2=keys[2],mean=float(x.delta_track.mean()),ci_low=float(lo),ci_high=float(hi),
                         residual_mean=float(x.delta_residual.mean()),residual_ci_low=float(rl),residual_ci_high=float(rh),seeds=int(x.seed.nunique())))
    A=pd.DataFrame(rows);A.to_csv(a.tables/'e4_robustness_summary.csv',index=False)

    fig,axs=plt.subplots(1,3,figsize=(15.0,4.5))
    # A reference rates
    r=A[A.panel=='reference'];etas=list(ETA_GRID);mat=np.full((len(etas),len(etas)),np.nan)
    for _,q in r.iterrows():mat[etas.index(q.key2),etas.index(q.key1)]=100*q['mean']
    im=axs[0].imshow(mat,origin='lower',aspect='auto');axs[0].set_xticks(range(len(etas)),[f'{x:g}' for x in etas]);axs[0].set_yticks(range(len(etas)),[f'{x:g}' for x in etas]);axs[0].set_xlabel(r'$\eta_+$');axs[0].set_ylabel(r'$\eta_-$');axs[0].set_title('A  Reference adaptation',loc='left',fontweight='bold');axs[0].plot(etas.index(.20),etas.index(.05),marker='*',ms=12,mfc='none',mec='black',mew=1.3)
    cb=fig.colorbar(im,ax=axs[0],fraction=.046,pad=.04);cb.set_label('Post-change DLR reduction (%)')
    # B horizon
    h=A[A.panel=='horizon'].sort_values('key1');x=h.key1.to_numpy();y=100*h['mean'].to_numpy();lo=100*h.ci_low.to_numpy();hi=100*h.ci_high.to_numpy()
    axs[1].plot(x,y,marker='o',lw=1.8);axs[1].fill_between(x,lo,hi,alpha=.14);axs[1].axhline(0,color='.3',ls=':',lw=1);axs[1].axvline(5,color='.45',ls=':',lw=.9);axs[1].set_xscale('log');axs[1].set_xticks(x,[str(int(v)) for v in x]);axs[1].set_xlabel('Episode horizon $h$');axs[1].set_ylabel('Post-change DLR reduction (%)');axs[1].set_title('B  Episode horizon',loc='left',fontweight='bold');axs[1].grid(alpha=.15)
    # C optimizer
    o=A[A.panel=='optimizer'];gs=list(GAMMA_GRID);vs=list(VARTTHETA_GRID);mat=np.full((len(vs),len(gs)),np.nan)
    for _,q in o.iterrows():mat[vs.index(q.key2),gs.index(q.key1)]=100*q['mean']
    im2=axs[2].imshow(mat,origin='lower',aspect='auto');axs[2].set_xticks(range(len(gs)),[f'{x:g}' for x in gs]);axs[2].set_yticks(range(len(vs)),[f'{x:g}' for x in vs]);axs[2].set_xlabel(r'$\gamma$');axs[2].set_ylabel(r'$\vartheta$');axs[2].set_title('C  Optimizer scale',loc='left',fontweight='bold');axs[2].plot(gs.index(.08),vs.index(.05),marker='*',ms=12,mfc='none',mec='black',mew=1.3)
    cb2=fig.colorbar(im2,ax=axs[2],fraction=.046,pad=.04);cb2.set_label('Post-change DLR reduction (%)')
    fig.tight_layout();save(fig,a.figures,'e4_robustness')

    rdef=r[(np.isclose(r.key1,.20))&(np.isclose(r.key2,.05))].iloc[0]
    odef=o[(np.isclose(o.key1,.08))&(np.isclose(o.key2,.05))].iloc[0]
    summary=dict(reference_positive=int((r['mean']>0).sum()),reference_total=int(len(r)),reference_resolved=int((r.ci_low>0).sum()),
                 asymmetric_positive=int((r[r.key1>r.key2]['mean']>0).sum()),asymmetric_total=int((r.key1>r.key2).sum()),
                 default_reference=dict(mean=float(rdef['mean']),ci_low=float(rdef.ci_low),ci_high=float(rdef.ci_high)),
                 horizon_positive=[int(v) for v in h.loc[h['mean']>0,'key1']],horizon_resolved=[int(v) for v in h.loc[h.ci_low>0,'key1']],
                 optimizer_positive=int((o['mean']>0).sum()),optimizer_total=int(len(o)),optimizer_resolved=int((o.ci_low>0).sum()),
                 default_optimizer=dict(mean=float(odef['mean']),ci_low=float(odef.ci_low),ci_high=float(odef.ci_high)))
    (a.tables/'e4_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
