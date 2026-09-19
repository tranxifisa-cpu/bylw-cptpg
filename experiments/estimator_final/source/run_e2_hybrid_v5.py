from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, json, math, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(ROOT/'vendor'))
from mvp_cpt_pg.paper_market import PaperMarket,PaperPanel
from mvp_cpt_pg.paper_experiments import EpisodeLaw,PaperConfig
from mvp_cpt_pg.cpt_objective import CPTPreference
from estimators import chunked_draw, v5_weight
from loo_controls import loo_components
from split_debias import design_for_budget, estimate as split_estimate

SEED=2026091631
BUDGETS=(32,64,128,256,512,1024,2048)
DAYS=(0,150,290,350)
MGRID=(1,2,4,8,16,32,64)
DIMS=(2,4,8,10)
EXPONENT=1.5
R=2**(-EXPONENT)
KAPPA=2*(1-R)/(1-2*R)
METHODS=('hybrid','v5_raw')
COORD=2

def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def load_market(): return PaperMarket(PaperPanel.load(ROOT/'inputs/paper_hs300.csv','20230301','20260529'),True)
def load_law(day,dim=10):
    panel=PaperPanel.load(ROOT/'inputs/paper_hs300.csv','20230301','20260529')
    market=PaperMarket(panel,True); prev=np.zeros(len(panel.codes)); prev[0]=1.
    return EpisodeLaw(market,day,1.,1.,prev,PaperConfig(dimension=dim),CPTPreference())

def loo_design(budget):
    for n in range(int(budget),1,-1):
        rho=n**-.5; cost=n+rho*KAPPA*n
        if cost<=budget: return dict(n=n,rho=rho,expected_cost=cost)
    raise ValueError('budget too small')

def hybrid_once(law,theta,budget,ss):
    d=loo_design(budget); n=d['n'];rho=d['rho']
    rg_base,rg_ctrl,rg_corr=[np.random.default_rng(x) for x in ss.spawn(3)]
    u,g=chunked_draw(law,theta,rg_base,n)
    pieces=loo_components(u,g,law.cpt)
    base=pieces[0]-pieces[1:].sum(axis=0)
    corr=np.zeros_like(base); level=0; calls=n
    if rg_ctrl.random()<rho:
        level=int(rg_ctrl.geometric(1-R)); N=n*(1<<level)
        u,g=chunked_draw(law,theta,rg_corr,N); h=N//2
        rf=loo_components(u,g,law.cpt)[0]
        r1=loo_components(u[:h],g[:h],law.cpt)[0]
        r2=loo_components(u[h:],g[h:],law.cpt)[0]
        q=(1-R)*R**(level-1)
        corr=(rf-.5*(r1+r2))/(rho*q); calls+=N
    return base+corr,calls,level

def v5_once(law,theta,budget,ss):
    rng=np.random.default_rng(ss)
    n=max(1,int(round(budget*.125))); n=min(n,budget-1);m=budget-n
    a,_=chunked_draw(law,theta,rng,n); b,g=chunked_draw(law,theta,rng,m)
    z=np.mean(v5_weight(a,b,law.cpt)[:,None]*g,axis=0)
    return z,budget,0

def reference(law,count=64,budget=32768,seed=0):
    # Independent centered split + random correction. Used only as high-precision numerical reference.
    theta=np.zeros(law.config.dimension); d=design_for_budget(budget,.125,EXPONENT)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,99,seed])); vals=[]
    from estimators import quantile_weight
    for _ in range(count):
        def draw(r,n): return chunked_draw(law,theta,r,n)
        def stat(a,b,g): return np.mean(quantile_weight(a,b,law.cpt)[:,None]*g,axis=0)
        z,_=split_estimate(draw,stat,rng,d); vals.append(z)
    return np.array(vals)

def context_job(args):
    day,reps,out,reference_dir,regenerate_reference=args; out=Path(out)/f'day_{day:03d}';out.mkdir(parents=True,exist_ok=False)
    law=load_law(day);theta=np.zeros(10)
    reference_dir=Path(reference_dir); reference_dir.mkdir(parents=True,exist_ok=True)
    ref_path=reference_dir/f'day_{day:03d}_reference.npz'
    if ref_path.exists() and not regenerate_reference:
        ref=np.load(ref_path)['estimates']
    else:
        ref=reference(law,count=64,budget=32768,seed=day)
        np.savez_compressed(ref_path,estimates=ref)
    np.savez_compressed(out/'reference.npz',estimates=ref)
    for B in BUDGETS:
        est=np.empty((reps,2,10)); calls=np.empty((reps,2),int); lev=np.zeros((reps,2),int)
        for i in range(reps):
            base=np.random.SeedSequence([SEED,day,B,i])
            ss1,ss2=base.spawn(2)
            est[i,0],calls[i,0],lev[i,0]=hybrid_once(law,theta,B,ss1)
            est[i,1],calls[i,1],lev[i,1]=v5_once(law,theta,B,ss2)
        np.savez_compressed(out/f'budget_{B:04d}.npz',estimates=est,calls=calls,levels=lev,methods=np.array(METHODS),
                            expected_costs=np.array([loo_design(B)['expected_cost'],B],float))
        print('E2',day,B,'done',flush=True)
    return day

def moment_job(args):
    trials,maxM,out=args; out=Path(out)/'replication';out.mkdir(parents=True,exist_ok=False)
    law=load_law(0);theta=np.zeros(10);B=128;n=trials*maxM
    est=np.empty((n,2,10));calls=np.empty((n,2),int)
    for i in range(n):
        ss1,ss2=np.random.SeedSequence([SEED,777,i]).spawn(2)
        est[i,0],calls[i,0],_=hybrid_once(law,theta,B,ss1)
        est[i,1],calls[i,1],_=v5_once(law,theta,B,ss2)
    np.savez_compressed(out/'outputs.npz',estimates=est,calls=calls,methods=np.array(METHODS),
                        expected_costs=np.array([loo_design(B)['expected_cost'],B],float))
    dump(out/'design.json',dict(trials=trials,max_M=maxM,M_grid=MGRID,dimensions=DIMS,unit_budget=B,coordinate=COORD))
    print('E2 replication pool done',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--repetitions',type=int,default=600)
    ap.add_argument('--trials',type=int,default=400);ap.add_argument('--max-M',type=int,default=64);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--reference-dir',type=Path,default=ROOT/'reference_inputs')
    ap.add_argument('--regenerate-reference',action='store_true')
    a=ap.parse_args();
    if a.output.exists(): ap.error('fresh output required')
    a.output.mkdir(parents=True)
    dump(a.output/'protocol.json',dict(seed=SEED,methods=METHODS,budgets=BUDGETS,days=DAYS,repetitions=a.repetitions,trials=a.trials,max_M=a.max_M,
        baseline='CPT-PG v5 adapted raw independent split; inner fraction 1/8, no centering, no correction',
        proposed='centered LOO base + RAW independent random correction; expected-cost matched',hard_level_cap=None))
    st=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs=[ex.submit(context_job,(d,a.repetitions,str(a.output),str(a.reference_dir.resolve()),a.regenerate_reference)) for d in DAYS]
        fs.append(ex.submit(moment_job,(a.trials,a.max_M,str(a.output))))
        for f in as_completed(fs): f.result()
    dump(a.output/'completion.json',dict(status='complete',runtime_seconds=time.time()-st))
if __name__=='__main__':main()
