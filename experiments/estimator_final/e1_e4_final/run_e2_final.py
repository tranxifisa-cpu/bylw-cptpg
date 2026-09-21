from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse, json, math, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT=Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(ROOT.parents[2]))
from mvp_cpt_pg.cpt_objective import CPTPreference
from estimators import chunked_draw, v5_weight, quantile_weight
from loo_controls import loo_components
from split_debias import design_for_budget, estimate as split_estimate
from common_slow_continuous import fixed_context_slow_law, SlowVariationSpec, slow_period_names

SEED=2026091801
BUDGETS=(32,64,128,256,512,1024,2048)
PERIODS=(20,50,90,150)
MGRID=(1,2,4,8,16,32,64)
DIMS=(2,4,8,10)
DIM_BUDGETS=(64,128,256,512,1024,2048)
EXPONENT=1.5
R=2**(-EXPONENT)
KAPPA=2*(1-R)/(1-2*R)
METHODS=('hybrid','v5_raw')
COORD=2


def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def law_for(period,dim=10,budget=512):
    return fixed_context_slow_law(period,dimension=dim,budget=budget,cpt=CPTPreference(),spec=SlowVariationSpec())

def loo_design(budget):
    for n in range(int(budget),1,-1):
        rho=n**-.5; cost=n+rho*KAPPA*n
        if cost<=budget: return dict(n=n,rho=rho,expected_cost=cost)
    raise ValueError('budget too small')

def hybrid_once(law,theta,budget,ss):
    d=loo_design(budget);n=d['n'];rho=d['rho']
    rg_base,rg_ctrl,rg_corr=[np.random.default_rng(x) for x in ss.spawn(3)]
    u,g=chunked_draw(law,theta,rg_base,n)
    pieces=loo_components(u,g,law.cpt)
    base=pieces[0]-pieces[1:].sum(axis=0)
    corr=np.zeros_like(base);level=0;calls=n
    if rg_ctrl.random()<rho:
        level=int(rg_ctrl.geometric(1-R));N=n*(1<<level)
        u,g=chunked_draw(law,theta,rg_corr,N);h=N//2
        raw_full=loo_components(u,g,law.cpt)[0]
        raw_h1=loo_components(u[:h],g[:h],law.cpt)[0]
        raw_h2=loo_components(u[h:],g[h:],law.cpt)[0]
        q=(1-R)*R**(level-1)
        corr=(raw_full-.5*(raw_h1+raw_h2))/(rho*q);calls+=N
    return base+corr,calls,level

def v5_once(law,theta,budget,ss):
    rng=np.random.default_rng(ss)
    n=max(1,int(round(budget*.125))); n=min(n,budget-1);m=budget-n
    a,_=chunked_draw(law,theta,rng,n); b,g=chunked_draw(law,theta,rng,m)
    z=np.mean(v5_weight(a,b,law.cpt)[:,None]*g,axis=0)
    return z,budget,0

def reference(law,count=64,budget=32768,seed=0):
    theta=np.zeros(law.config.dimension); d=design_for_budget(budget,.125,EXPONENT)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,99,seed,law.config.dimension])); vals=[]
    for _ in range(count):
        def draw(r,n): return chunked_draw(law,theta,r,n)
        def stat(a,b,g): return np.mean(quantile_weight(a,b,law.cpt)[:,None]*g,axis=0)
        z,_=split_estimate(draw,stat,rng,d); vals.append(z)
    return np.array(vals)

def context_job(args):
    period,reps,out=args; out=Path(out)/f'period_{period:03d}';out.mkdir(parents=True,exist_ok=False)
    law=law_for(period,10);theta=np.zeros(10)
    ref=reference(law,count=32,budget=16384,seed=period)
    np.savez_compressed(out/'reference.npz',estimates=ref)
    for B in BUDGETS:
        lawB=law_for(period,10,B)
        est=np.empty((reps,2,10));calls=np.empty((reps,2),int);lev=np.zeros((reps,2),int)
        for i in range(reps):
            ss1,ss2=np.random.SeedSequence([SEED,period,B,i]).spawn(2)
            est[i,0],calls[i,0],lev[i,0]=hybrid_once(lawB,theta,B,ss1)
            est[i,1],calls[i,1],lev[i,1]=v5_once(lawB,theta,B,ss2)
        np.savez_compressed(out/f'budget_{B:04d}.npz',estimates=est,calls=calls,levels=lev,methods=np.array(METHODS),
                            expected_costs=np.array([loo_design(B)['expected_cost'],B],float))
        print('E2 context',period,B,'done',flush=True)
    return period

def moment_job(args):
    trials,maxM,out=args; out=Path(out)/'replication';out.mkdir(parents=True,exist_ok=False)
    period=20;B=128;law=law_for(period,10,B);theta=np.zeros(10);n=trials*maxM
    est=np.empty((n,2,10));calls=np.empty((n,2),int)
    for i in range(n):
        ss1,ss2=np.random.SeedSequence([SEED,777,i]).spawn(2)
        est[i,0],calls[i,0],_=hybrid_once(law,theta,B,ss1)
        est[i,1],calls[i,1],_=v5_once(law,theta,B,ss2)
    np.savez_compressed(out/'outputs.npz',estimates=est,calls=calls,methods=np.array(METHODS),
                        expected_costs=np.array([loo_design(B)['expected_cost'],B],float))
    dump(out/'design.json',dict(trials=trials,max_M=maxM,M_grid=MGRID,unit_budget=B,coordinate=COORD,period=period))
    print('E2 replication pool done',flush=True)

def dimension_job(args):
    d,reps,out=args; out=Path(out)/'dimension'/f'd_{d:02d}';out.mkdir(parents=True,exist_ok=False)
    period=20;theta=np.zeros(d);law=law_for(period,d,512)
    ref=reference(law,count=24,budget=8192,seed=1000+d)
    np.savez_compressed(out/'reference.npz',estimates=ref)
    for B in DIM_BUDGETS:
        lawB=law_for(period,d,B)
        est=np.empty((reps,2,d));calls=np.empty((reps,2),int)
        for i in range(reps):
            ss1,ss2=np.random.SeedSequence([SEED,888,d,B,i]).spawn(2)
            est[i,0],calls[i,0],_=hybrid_once(lawB,theta,B,ss1)
            est[i,1],calls[i,1],_=v5_once(lawB,theta,B,ss2)
        np.savez_compressed(out/f'budget_{B:04d}.npz',estimates=est,calls=calls,methods=np.array(METHODS),
                            expected_costs=np.array([loo_design(B)['expected_cost'],B],float))
        print('E2 dimension',d,B,'done',flush=True)
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--repetitions',type=int,default=600)
    ap.add_argument('--trials',type=int,default=400);ap.add_argument('--max-M',type=int,default=64);ap.add_argument('--dim-repetitions',type=int,default=250);ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args();
    if a.output.exists(): ap.error('fresh output required')
    a.output.mkdir(parents=True)
    dump(a.output/'protocol.json',dict(seed=SEED,methods=METHODS,budgets=BUDGETS,periods=PERIODS,period_names=slow_period_names(),repetitions=a.repetitions,
        trials=a.trials,max_M=a.max_M,dimensions=DIMS,dimension_budgets=DIM_BUDGETS,dimension_repetitions=a.dim_repetitions,
        cpt_value='(x^2+eps_v^2)^(alpha/2)-eps_v^alpha',
        environment='shared E2-E3 slow-variation semi-synthetic 30-stock+cash setup; time-invariant empirical A-share factor-block law; E2 fixes standardized account state at four pre-specified targets',stages=dict(stable='episodes 1-40: momentum',abrupt='episodes 41-60: reversal',gradual='episodes 61-120: finite interpolation to terminal law',tail='episodes 121-300: terminal return law fixed'),
        baseline='CPT-PG v5 adapted raw independent split; inner fraction 1/8, no centering, no correction',
        proposed='centered LOO base + raw independent randomized correction; expected-cost matched',hard_level_cap=None))
    st=time.time()
    jobs=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for p in PERIODS: jobs.append(ex.submit(context_job,(p,a.repetitions,str(a.output))))
        jobs.append(ex.submit(moment_job,(a.trials,a.max_M,str(a.output))))
        for d in DIMS: jobs.append(ex.submit(dimension_job,(d,a.dim_repetitions,str(a.output))))
        for f in as_completed(jobs): f.result()
    dump(a.output/'completion.json',dict(status='complete',runtime_seconds=time.time()-st))
if __name__=='__main__': main()
