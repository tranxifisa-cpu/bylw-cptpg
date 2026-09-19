from pathlib import Path
import json,math
import numpy as np,pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'online_results';FIG=ROOT/'figures';TAB=ROOT/'tables';FIG.mkdir(exist_ok=True);TAB.mkdir(exist_ok=True)
LABEL={'hybrid':'Proposed hybrid','v5_raw':'CPT-PG v5 baseline'};MARK={'hybrid':'o','v5_raw':'s'}

def save(fig,name):
    for e in ('png','pdf','svg'):fig.savefig(FIG/f'{name}.{e}',dpi=220,bbox_inches='tight')
    plt.close(fig)

def ci_mean(x):
    x=np.asarray(x,float);m=x.mean();se=x.std(ddof=1)/np.sqrt(len(x));q=stats.t.ppf(.975,len(x)-1);return m,m-q*se,m+q*se

ep=pd.read_csv(RES/'episodes.csv');dy=pd.read_csv(RES/'daily.csv');ws=pd.read_csv(RES/'wealth_summary.csv')
# tracking summary at K=100
trackcols=['new_window_dlr','average_new_window_dlr','cumulative_residual_squared','average_residual_squared']
final=ep[ep.episode==ep.episode.max()][['method','seed']+trackcols];final.to_csv(TAB/'online_tracking_seedwise_final.csv',index=False)
rows=[]
for c in trackcols:
    p=final.pivot(index='seed',columns='method',values=c);d=p.hybrid-p.v5_raw;m,lo,hi=ci_mean(d)
    rows.append(dict(metric=c,proposed_mean=float(p.hybrid.mean()),v5_mean=float(p.v5_raw.mean()),paired_difference=float(m),ci_low=float(lo),ci_high=float(hi),paired_t_p=float(stats.ttest_rel(p.hybrid,p.v5_raw).pvalue)))
pd.DataFrame(rows).to_csv(TAB/'online_tracking_comparison.csv',index=False)
# wealth metrics and paired comparisons
wealthcols=['terminal_wealth','cumulative_return','annualized_return','annualized_volatility','sharpe','max_drawdown','total_turnover','cumulative_fee','mean_cash']
comp=[]
for c in wealthcols:
    p=ws.pivot(index='seed',columns='method',values=c);d=p.hybrid-p.v5_raw;m,lo,hi=ci_mean(d)
    comp.append(dict(metric=c,proposed_mean=float(p.hybrid.mean()),proposed_sd=float(p.hybrid.std(ddof=1)),v5_mean=float(p.v5_raw.mean()),v5_sd=float(p.v5_raw.std(ddof=1)),paired_difference=float(m),ci_low=float(lo),ci_high=float(hi),paired_t_p=float(stats.ttest_rel(p.hybrid,p.v5_raw).pvalue)))
pd.DataFrame(comp).to_csv(TAB/'wealth_metric_comparison.csv',index=False);ws.to_csv(TAB/'wealth_seedwise.csv',index=False)
# tracking plot four panels
fig,axs=plt.subplots(2,2,figsize=(13,9));spec=[('new_window_dlr','A  New window DLR: cumulative'),('average_new_window_dlr','B  New window DLR / K'),('cumulative_residual_squared','C  Periodwise residual squared: cumulative'),('average_residual_squared','D  Mean periodwise residual squared')]
for ax,(col,title) in zip(axs.ravel(),spec):
    for m in ['hybrid','v5_raw']:
        g=ep[ep.method==m].groupby('episode')[col].agg(['mean','std','count']);x=g.index.to_numpy();mean=g['mean'].to_numpy();se=g['std'].to_numpy()/np.sqrt(g['count'].to_numpy());q=stats.t.ppf(.975,5)
        ax.plot(x,mean,label=LABEL[m]);ax.fill_between(x,mean-q*se,mean+q*se,alpha=.15)
    for b in (30,58,70):ax.axvline(b,ls=':',lw=.8,alpha=.6)
    ax.set_title(title);ax.set_xlabel('Learning episode K (5 trading days)');ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axs[0,0].legend();fig.suptitle('Online tracking comparison: same semi-synthetic market paths, six paired seeds\nCommon unbiased evaluator; new DLR squares period residuals before window averaging',fontsize=14);fig.tight_layout(rect=[0,0,1,.95]);save(fig,'online_tracking_hybrid_vs_v5')
# wealth trajectory plot
fig,ax=plt.subplots(figsize=(11,5.8))
for m in ['hybrid','v5_raw']:
    g=dy[dy.method==m].groupby('day').wealth.agg(['mean','std','count']);x=g.index;mean=g['mean'];se=g['std']/np.sqrt(g['count']);q=stats.t.ppf(.975,5)
    ax.plot(x,mean,label=LABEL[m],lw=2);ax.fill_between(x,mean-q*se,mean+q*se,alpha=.15)
ax.axhline(1,ls=':',lw=1);ax.set_xlabel('Trading day');ax.set_ylabel('Wealth');ax.set_title('Wealth trajectories: six paired seeds');ax.grid(alpha=.2);ax.legend();save(fig,'online_wealth_trajectory')
# wealth metrics 2x3
metrics=[('cumulative_return','Cumulative return'),('annualized_return','Annualized return'),('sharpe','Sharpe ratio'),('max_drawdown','Maximum drawdown'),('annualized_volatility','Annualized volatility'),('total_turnover','Total turnover')]
fig,axs=plt.subplots(2,3,figsize=(14,8))
for ax,(col,title) in zip(axs.ravel(),metrics):
    vals=[];errs=[]
    for m in ['hybrid','v5_raw']:
        x=ws.loc[ws.method==m,col].to_numpy();vals.append(x.mean());errs.append(stats.t.ppf(.975,5)*x.std(ddof=1)/np.sqrt(len(x)))
    ax.bar([0,1],vals,yerr=errs,capsize=4);ax.set_xticks([0,1],['Proposed','CPT-PG v5']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
fig.suptitle('Portfolio performance metrics (mean ± 95% t interval across six paired seeds)');fig.tight_layout(rect=[0,0,1,.95]);save(fig,'online_wealth_metrics')
# short markdown summary
track=pd.DataFrame(rows);wc=pd.DataFrame(comp)
lines=['# Hybrid vs CPT-PG v5: online summary','', 'All runs use the same 30-stock/cash semi-synthetic environment, 500 trading days, six paired seeds. Training budget is 512 expected complete paths per update. Evaluation is common across methods: centered unbiased split estimator, 1536 expected paths per repeat, 4 repeats, two independent streams per episode.','', '## Final tracking metrics',track.to_markdown(index=False),'','## Wealth metrics',wc.to_markdown(index=False)]
(ROOT/'ONLINE_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print(wc[['metric','proposed_mean','v5_mean','paired_difference','ci_low','ci_high']].to_string(index=False));print(track.to_string(index=False))
