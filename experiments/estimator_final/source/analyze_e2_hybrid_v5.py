from pathlib import Path
import json,math
import numpy as np,pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'e2_results'; FIG=ROOT/'figures'; TAB=ROOT/'tables';FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
LABEL={'hybrid':'Proposed: centered LOO base + raw correction','v5_raw':'CPT-PG v5 (adapted raw split plug-in)'}
MARK={'hybrid':'o','v5_raw':'s'}
DAYS=[0,150,290,350];BUDGETS=[32,64,128,256,512,1024,2048];DIMS=[2,4,8,10]

def save(fig,name):
    for e in ('png','pdf','svg'):fig.savefig(FIG/f'{name}.{e}',dpi=220,bbox_inches='tight')
    plt.close(fig)

def reference(day):
    r=np.load(RES/f'day_{day:03d}'/'reference.npz')['estimates'];return r.mean(0),r.var(0,ddof=1)/len(r)
rows=[]
for day in DAYS:
    ref,refvar=reference(day)
    for B in BUDGETS:
        z=np.load(RES/f'day_{day:03d}'/f'budget_{B:04d}.npz');methods=list(z['methods']);ec=z['expected_costs']
        for mi,m in enumerate(methods):
            x=z['estimates'][:,mi,:];n=len(x);mean=x.mean(0);var=x.var(0,ddof=1)
            se=math.sqrt(var[2]/n+refvar[2]);bias=mean[2]-ref[2]
            raw=np.mean(np.sum((x-ref)**2,axis=1));adj=raw-refvar.sum()
            rows.append(dict(day=day,budget=B,method=m,repetitions=n,expected_cost=float(ec[mi]),coordinate_bias=float(bias),
                bias_low=float(bias-1.96*se),bias_high=float(bias+1.96*se),variance_trace=float(var.sum()),
                mse_raw=float(raw),mse_reference_adjusted=float(adj),reference_variance=float(refvar.sum()),actual_cost=float(z['calls'][:,mi].mean())))
tab=pd.DataFrame(rows);tab.to_csv(TAB/'e2_bias_mse.csv',index=False)
# Moment pool
z=np.load(RES/'replication'/'outputs.npz');des=json.loads((RES/'replication'/'design.json').read_text());methods=list(z['methods']);tr=des['trials'];mx=des['max_M'];x=z['estimates'].reshape(tr,mx,2,10);ref,refvar=reference(0)
mom=[];qq=[];student=[]
for M in des['M_grid']:
    means=x[:,:M].mean(1)
    for mi,m in enumerate(methods):
        for d in DIMS:
            mm=means[:,mi,:d];raw=np.mean(np.sum((mm-ref[:d])**2,axis=1));mom.append(dict(M=M,dimension=d,method=m,variance_trace=float(mm.var(0,ddof=1).sum()),mse_reference_adjusted=float(raw-refvar[:d].sum()),mse_raw=float(raw)))
        if M>1:
            sd=x[:,:M,mi,2].std(1,ddof=1);t=np.sqrt(M)*(means[:,mi,2]-ref[2])/sd
            student.append(dict(M=M,method=m,coverage95=float(np.mean(np.abs(t)<=stats.t.ppf(.975,M-1))),t_mean=float(t.mean()),t_sd=float(t.std(ddof=1)),ks=float(stats.kstest(t,'norm').statistic)))
            for val in t:qq.append(dict(M=M,method=m,t=float(val)))
mt=pd.DataFrame(mom);mt.to_csv(TAB/'e2_replication_dimension.csv',index=False);pd.DataFrame(student).to_csv(TAB/'e2_studentized.csv',index=False);qt=pd.DataFrame(qq)
# composite 4 panels
fig,axs=plt.subplots(2,2,figsize=(13,9));
ax=axs[0,0]
for m in methods:
    d=tab[(tab.day==0)&(tab.method==m)].sort_values('expected_cost'); ax.errorbar(d.expected_cost,d.coordinate_bias,yerr=[d.coordinate_bias-d.bias_low,d.bias_high-d.coordinate_bias],marker=MARK[m],capsize=3,label=LABEL[m])
ax.axhline(0,ls=':',lw=1);ax.set_xscale('log',base=2);ax.set_title('A  Bias at a fixed market state');ax.set_xlabel('Expected trajectory budget B');ax.set_ylabel('Signed bias: momentum coordinate');ax.grid(alpha=.2);ax.legend(fontsize=8)
ax=axs[0,1]
for m in methods:
    d=mt[(mt.method==m)&(mt.dimension==10)].sort_values('M');ax.loglog(d.M,d.variance_trace,marker=MARK[m],label=LABEL[m])
base=mt[(mt.method=='hybrid')&(mt.dimension==10)].sort_values('M');ax.loglog(base.M,base.variance_trace.iloc[0]/base.M,ls=':',label='1/M slope guide');ax.set_title('B  Finite-replication variance');ax.set_xlabel('Independent complete replications M');ax.set_ylabel('Trace of empirical covariance');ax.grid(alpha=.2);ax.legend(fontsize=8)
ax=axs[1,0]
for m in methods:
    vals=np.sort(qt[(qt.method==m)&(qt.M==64)].t.to_numpy());q=stats.norm.ppf((np.arange(len(vals))+.5)/len(vals));ax.plot(q,vals,marker=MARK[m],markersize=3,lw=.8,label=LABEL[m])
lims=ax.get_xlim();lo=min(lims[0],ax.get_ylim()[0]);hi=max(lims[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],ls=':',lw=1);ax.set_title('C  Studentized normal-quantile diagnostic');ax.set_xlabel('Standard normal quantiles');ax.set_ylabel('Studentized estimator quantiles');ax.grid(alpha=.2);ax.legend(fontsize=8)
ax=axs[1,1]
for d in DIMS:
    for m,ls in [('hybrid','-'),('v5_raw','--')]:
        a=mt[(mt.method==m)&(mt.dimension==d)].sort_values('M');ax.loglog(a.M,a.mse_reference_adjusted,marker=MARK[m],ls=ls,markersize=4,label=f'{"Proposed" if m=="hybrid" else "v5"}, d={d}')
ax.set_title('D  Dimension and sampling budget');ax.set_xlabel('Independent complete replications M');ax.set_ylabel('Gradient MSE (reference-adjusted)');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
fig.suptitle('Estimator experiment E2: proposed hybrid vs CPT-PG v5 baseline\n30 stocks + cash, 5-day paths; fresh estimator samples; common independent numerical reference',fontsize=14);fig.tight_layout(rect=[0,0,1,.95]);save(fig,'E2_four_panels_hybrid_vs_v5')
# MSE contexts plot
fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True);names={0:'Uptrend',150:'Sideways',290:'Shock',350:'Style shift'}
for ax,day in zip(axs.ravel(),DAYS):
    for m in methods:
        d=tab[(tab.day==day)&(tab.method==m)].sort_values('expected_cost');ax.loglog(d.expected_cost,d.mse_reference_adjusted,marker=MARK[m],label=LABEL[m])
    ax.set_title(names[day]);ax.set_xlabel('Expected trajectory budget');ax.set_ylabel('Gradient MSE');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8);fig.suptitle('Full-gradient MSE across four fixed market contexts');fig.tight_layout(rect=[0,0,1,.96]);save(fig,'E2_MSE_four_contexts_hybrid_vs_v5')
# concise comparison tables
p=tab.pivot(index=['day','budget'],columns='method',values='mse_reference_adjusted').reset_index();p['proposed_over_v5']=p.hybrid/p.v5_raw;p.to_csv(TAB/'e2_mse_ratio.csv',index=False)
print(p[p.budget==512].to_string(index=False));print(pd.DataFrame(student).query('M==64').to_string(index=False))
