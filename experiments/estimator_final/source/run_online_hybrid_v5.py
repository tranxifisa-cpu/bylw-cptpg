from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import argparse,json,time,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from mvp_cpt_pg import paper_experiments as pe
from mvp_cpt_pg.paper_market import PaperMarket,PaperPanel
from estimators import chunked_draw,v5_weight
from loo_controls import loo_components
from evaluation_estimator import evaluate_fixed

EXPONENT=1.5; R=2**(-EXPONENT); KAPPA=2*(1-R)/(1-2*R)
SEEDS=(29,147,3141,42,3407,592)
METHODS=('hybrid','v5_raw')
LABELS={'hybrid':'Centered LOO base + raw random correction','v5_raw':'CPT-PG v5 adapted (raw split plug-in)'}

def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def load_market(): return PaperMarket(PaperPanel.load(ROOT/'inputs/paper_hs300.csv','20230301','20260529'),True)
def loo_design(budget):
    for n in range(int(budget),1,-1):
        rho=n**-.5;cost=n+rho*KAPPA*n
        if cost<=budget:return dict(n=n,rho=rho,expected_cost=cost)
    raise ValueError

def hybrid_estimate(law,theta,rng,budget=512):
    d=loo_design(budget); n=d['n'];rho=d['rho']
    # derive three independent child streams from one fresh seed drawn from the framework stream
    ss=np.random.SeedSequence(int(rng.integers(0,2**63-1)))
    rb,rc,rr=[np.random.default_rng(x) for x in ss.spawn(3)]
    u,g=chunked_draw(law,theta,rb,n);parts=loo_components(u,g,law.cpt)
    base=parts[0]-parts[1:].sum(axis=0);corr=np.zeros_like(base);level=0;calls=n
    if rc.random()<rho:
        level=int(rc.geometric(1-R));N=n*(1<<level);u,g=chunked_draw(law,theta,rr,N);h=N//2
        rf=loo_components(u,g,law.cpt)[0];r1=loo_components(u[:h],g[:h],law.cpt)[0];r2=loo_components(u[h:],g[h:],law.cpt)[0]
        q=(1-R)*R**(level-1);corr=(rf-.5*(r1+r2))/(rho*q);calls+=N
    return base+corr,dict(calls=calls,level=level,expected=d['expected_cost'],n=n,rho=rho,base_norm=float(np.linalg.norm(base)),correction_norm=float(np.linalg.norm(corr)))

def v5_estimate(law,theta,rng,budget=512):
    n=budget//8;m=budget-n
    a,_=chunked_draw(law,theta,rng,n);b,g=chunked_draw(law,theta,rng,m)
    z=np.mean(v5_weight(a,b,law.cpt)[:,None]*g,axis=0)
    return z,dict(calls=budget,level=0,expected=float(budget),n=n,m=m,base_norm=float(np.linalg.norm(z)),correction_norm=0.)

def add_tracking(ep,c):
    out=[]
    for _,f in ep.groupby(['method','seed'],sort=False):
        f=f.sort_values('episode').copy();q2=[]
        for _,r in f.iterrows():
            th=r[[f'theta_{i}' for i in range(c.dimension)]].to_numpy(float)
            g=r[[f'diagnostic_g1_{i}' for i in range(c.dimension)]].to_numpy(float)
            q=(pe.projected_update(th,g,c)-th)/c.gamma;q2.append(float(q@q))
        q2=np.array(q2);win=[]
        for i in range(len(q2)):
            x=q2[max(0,i+1-c.window):i+1][::-1];w=c.rho**np.arange(len(x));win.append(float(np.average(x,weights=w)))
        k=np.arange(1,len(f)+1);win=np.array(win)
        f['period_residual_squared']=q2
        f['cumulative_residual_squared']=np.cumsum(q2)
        f['average_residual_squared']=np.cumsum(q2)/k
        f['square_first_window_residual']=win
        f['new_window_dlr']=np.cumsum(win)
        f['average_new_window_dlr']=np.cumsum(win)/k
        out.append(f)
    return pd.concat(out,ignore_index=True)

def wealth_summary(daily):
    rows=[]
    for (m,s),f in daily.groupby(['method','seed'],sort=False):
        f=f.sort_values('day');r=f.net_return.to_numpy(float);sd=r.std(ddof=1)
        terminal=float(f.wealth.iloc[-1]);ann=terminal**(252/len(f))-1;vol=sd*np.sqrt(252)
        rows.append(dict(method=m,seed=s,terminal_wealth=terminal,cumulative_return=terminal-1,annualized_return=ann,
                         annualized_volatility=vol,sharpe=(np.sqrt(252)*r.mean()/sd if sd>0 else np.nan),
                         max_drawdown=float(f.drawdown.max()),total_turnover=float(f.turnover.sum()),
                         cumulative_fee=float(f.fee.sum()),mean_cash=float(f.cash.mean())))
    return pd.DataFrame(rows)

def run_one(job):
    method,seed,output=job;dst=Path(output)/f'{method}_seed_{seed}';dst.mkdir(parents=True,exist_ok=False)
    c=pe.PaperConfig(trajectory_budget=512,n=64,evaluation_n=1024,evaluation_m=512)
    accounting=[];evalacct=[]
    def train(law,theta,rng,_method,_estimator):
        z,info=(hybrid_estimate(law,theta,rng,512) if method=='hybrid' else v5_estimate(law,theta,rng,512));accounting.append(info);return z,info['calls']
    def evaluate(law,theta,rng,_method):
        g,info=evaluate_fixed(law,theta,rng,estimator='unbiased',budget=1536,repeats=4);evalacct.append(info);return info['objective_plugin'],g
    oldt,olde=pe.estimate_gradient,pe.evaluate;st=time.time()
    try:
        pe.estimate_gradient=train;pe.evaluate=evaluate
        ep,dy=pe.run_online(load_market(),c,[seed],['dynamic'],500,common_probe=False)
    finally: pe.estimate_gradient=oldt;pe.evaluate=olde
    ep['method']=method;dy['method']=method
    assert len(accounting)==len(ep) and len(evalacct)==2*len(ep)
    ep['estimator_level']=[x['level'] for x in accounting];ep['expected_training_paths']=[x['expected'] for x in accounting]
    ep['base_norm']=[x['base_norm'] for x in accounting];ep['correction_norm']=[x['correction_norm'] for x in accounting]
    ep['evaluation_actual_paths']=[evalacct[2*i]['actual_paths']+evalacct[2*i+1]['actual_paths'] for i in range(len(ep))]
    ep['evaluation_expected_paths']=[evalacct[2*i]['expected_paths']+evalacct[2*i+1]['expected_paths'] for i in range(len(ep))]
    ep=add_tracking(ep,c)
    ep.to_csv(dst/'episodes.csv',index=False);dy.to_csv(dst/'daily.csv',index=False)
    wealth_summary(dy).to_csv(dst/'wealth_summary.csv',index=False)
    dump(dst/'completion.json',dict(status='complete',method=method,seed=seed,runtime_seconds=time.time()-st,
         training_expected_total=float(sum(x['expected'] for x in accounting)),training_actual_total=int(sum(x['calls'] for x in accounting))))
    print('ONLINE',method,seed,'done',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    if a.output.exists():ap.error('fresh output required');a.output.mkdir(parents=True)
    a.output.mkdir(parents=True)
    dump(a.output/'protocol.json',dict(methods=METHODS,labels=LABELS,seeds=SEEDS,steps=500,horizon=5,training_expected_budget=512,
      v5=dict(inner=64,outer=448,kernel='raw v5'),hybrid={**loo_design(512),'base=':'centered LOO','correction':'raw random MLMC'},
      evaluation=dict(kind='common centered unbiased split estimator',budget_per_repeat=1536,repeats=4,streams=2),
      tracking='square residual first, then 5-episode rho=0.9 scalar window average; cumulative and /K both reported'))
    jobs=[(m,s,str(a.output.resolve())) for m in METHODS for s in SEEDS];st=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(run_one,j) for j in jobs]):f.result()
    for name in ('episodes.csv','daily.csv','wealth_summary.csv'):
        pd.concat([pd.read_csv(a.output/f'{m}_seed_{s}'/name) for m,s,_ in jobs],ignore_index=True).to_csv(a.output/name,index=False)
    dump(a.output/'completion.json',dict(status='complete',jobs=len(jobs),runtime_seconds=time.time()-st))
if __name__=='__main__':main()
