"""Actual CPT E2: bias, finite-replication variance, studentized CLT, dimension-risk.
No prescribed estimator noise, level cap, correction deletion, or tuned rankings.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np
import pandas as pd
from scipy.stats import binom
from mvp_cpt_pg.paper_market import PaperMarket,PaperPanel
from mvp_cpt_pg.paper_experiments import EpisodeLaw,PaperConfig
from mvp_cpt_pg.cpt_objective import CPTPreference,pooled_gradient
from mvp_cpt_pg.controlled_cpt import quantile_weight
from estimators import chunked_draw,v5_weight
from split_debias import design_for_budget,correction
SEED=2026091607
BUDGETS=(32,64,128,256,512,1024,2048)
DAYS=(0,150,290,350)
M_GRID=(1,2,4,8,16,32,64)
DIMS=(2,4,8,10)
METHODS=('loo_corrected','split_corrected','loo_plugin','split_centered','split_raw','split_corrected_raw')
EXPONENT=1.5
R=2.**(-EXPONENT)
KAPPA=2*(1-R)/(1-2*R)
COORD=2

def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

def load_law(day,dim=10):
    panel=PaperPanel.load(ROOT/'inputs/paper_hs300.csv','20230301','20260529')
    market=PaperMarket(panel,True);prev=np.zeros(len(panel.codes));prev[0]=1.
    return EpisodeLaw(market,day,1.,1.,prev,PaperConfig(dimension=dim),CPTPreference())

def loo_design(budget):
    for n in range(int(budget),1,-1):
        rho=n**(-.5);cost=n+rho*KAPPA*n
        if cost<=budget:return dict(n=n,rho=rho,expected_cost=cost)
    raise ValueError('Budget too small')

def pair_stat(a,b,g,cpt):
    return np.array([np.mean(fn(a,b,cpt)[:,None]*g,axis=0) for fn in (v5_weight,quantile_weight)])

def sample_family(law,budget,count,seed):
    """Independent complete outputs. Common base paths pair different methods.
    LOO and split corrections use fresh independent paths and RNGs. Within split,
    A/B are disjoint; fresh C full/halves share D exactly as in the uploaded code.
    """
    theta=np.zeros(law.config.dimension)
    sd=design_for_budget(budget,.125,EXPONENT);ld=loo_design(budget)
    rngs=[np.random.default_rng(np.random.SeedSequence([SEED,*seed,i])) for i in range(1,6)]
    rng_base,rng_lctrl,rng_lpath,rng_sctrl,rng_spath=rngs
    values=np.empty((count,len(METHODS),law.config.dimension))
    calls=np.full((count,len(METHODS)),budget,dtype=np.int64)
    levels=np.zeros((count,2),dtype=np.int64)
    basevalues=np.empty((count,3,law.config.dimension));adjustments=np.zeros_like(basevalues)
    nsplit=max(1,int(budget*.125));chunksize=max(1,min(128,16384//budget))
    t=time.perf_counter()
    for start in range(0,count,chunksize):
        stop=min(count,start+chunksize)
        u,g=chunked_draw(law,theta,rng_base,(stop-start)*budget)
        u=u.reshape(stop-start,budget,2);g=g.reshape(stop-start,budget,law.config.dimension)
        for b in range(stop-start):
            row=start+b;uu=u[b];gg=g[b]
            basel=pooled_gradient(uu[:ld['n']],gg[:ld['n']],law.cpt)
            bases=pair_stat(uu[:sd.n],uu[sd.n:sd.n+sd.m],gg[sd.n:sd.n+sd.m],law.cpt)
            dl=np.zeros_like(basel);ds=np.zeros_like(bases)
            calls[row,0]=ld['n'];calls[row,1]=calls[row,5]=sd.n+sd.m
            if rng_lctrl.random()<ld['rho']:
                level=int(rng_lctrl.geometric(1-R));levels[row,0]=level;size=ld['n']*(1<<level)
                lu,lg=chunked_draw(law,theta,rng_lpath,size);half=size//2
                delta=pooled_gradient(lu,lg,law.cpt)-.5*(pooled_gradient(lu[:half],lg[:half],law.cpt)+pooled_gradient(lu[half:],lg[half:],law.cpt))
                dl=delta/(ld['rho']*((1-R)*R**(level-1)));calls[row,0]+=size
            if rng_sctrl.random()<sd.rho:
                level=int(rng_sctrl.geometric(1-sd.r));levels[row,1]=level;size=sd.n*(1<<level)
                su,_=chunked_draw(law,theta,rng_spath,size);ou,og=chunked_draw(law,theta,rng_spath,sd.mc)
                delta=correction(su,ou,og,lambda a,b,g:pair_stat(a,b,g,law.cpt))
                ds=delta/(sd.rho*sd.probability(level));calls[row,1]+=size+sd.mc;calls[row,5]=calls[row,1]
            bp=pair_stat(uu[:nsplit],uu[nsplit:],gg[nsplit:],law.cpt)
            values[row]=np.array([basel+dl,bases[1]+ds[1],pooled_gradient(uu,gg,law.cpt),bp[1],bp[0],bases[0]+ds[0]])
            basevalues[row]=np.array([basel,bases[1],bases[0]]);adjustments[row]=np.array([dl,ds[1],ds[0]])
    expected=np.array([ld['expected_cost'],sd.expected_cost,budget,budget,budget,sd.expected_cost])
    if not np.isfinite(values).all():raise FloatingPointError('Invalid estimator; no outputs deleted')
    return dict(estimates=values,calls=calls,levels=levels,bases=basevalues,adjustments=adjustments,
        expected_costs=expected,methods=np.array(METHODS),elapsed=np.array(time.perf_counter()-t))

def reference_draw(law,count,budget,seed):
    sd=design_for_budget(budget,.125,EXPONENT);theta=np.zeros(law.config.dimension)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,*seed,101]))
    ctrl=np.random.default_rng(np.random.SeedSequence([SEED,*seed,102]))
    corrng=np.random.default_rng(np.random.SeedSequence([SEED,*seed,103]))
    vals=[];calls=[];levels=[]
    for i in range(count):
        a,_=chunked_draw(law,theta,rng,sd.n);b,g=chunked_draw(law,theta,rng,sd.m)
        z=pair_stat(a,b,g,law.cpt)[1];cost=sd.n+sd.m;level=0
        if ctrl.random()<sd.rho:
            level=int(ctrl.geometric(1-sd.r));nc=sd.n*(1<<level)
            a,_=chunked_draw(law,theta,corrng,nc);b,g=chunked_draw(law,theta,corrng,sd.mc)
            z+=correction(a,b,g,lambda a,b,g:pair_stat(a,b,g,law.cpt)[1])/(sd.rho*sd.probability(level));cost+=nc+sd.mc
        vals.append(z);calls.append(cost);levels.append(level)
    return dict(estimates=np.array(vals),calls=np.array(calls),levels=np.array(levels),expected_cost=np.array(sd.expected_cost))

def context_job(args):
    day,reps,out=args;out=Path(out)/f'day_{day:03d}';out.mkdir(parents=True,exist_ok=False)
    law=load_law(day)
    dump(out/'context.json',dict(day=day,date=str(law.market.panel.dates[day]),regime=law.market.regime(day),
       theta=[0.]*10,wealth=1.,reference=1.,previous=law.previous.tolist(),config=asdict(law.config),preference=asdict(law.cpt)))
    for B in BUDGETS:
        data=sample_family(law,B,reps,[10,day,B]);np.savez_compressed(out/f'budget_{B:04d}.npz',**data)
        print(f'day={day} B={B} fresh={reps} done {float(data["elapsed"]):.1f}s',flush=True)
    for tier,budget,nrep in [('high',32768,64),('low',8192,32)]:
        rr=reference_draw(law,nrep,budget,[20,day,budget]);np.savez_compressed(out/f'reference_{tier}.npz',**rr)
        print(f'day={day} reference {tier} done paths={rr["calls"].sum()}',flush=True)
    return dict(day=day,status='complete')

def moment_job(args):
    trials,maxM,out=args;out=Path(out)/'replication';out.mkdir(parents=True,exist_ok=False)
    data=sample_family(load_law(0),128,trials*maxM,[30,0,128]);np.savez_compressed(out/'iid_complete_outputs.npz',**data)
    dump(out/'design.json',dict(trials=trials,max_M=maxM,M_grid=[m for m in M_GRID if m<=maxM],dimensions=DIMS,
       day=0,unit_budget=128,studentized_coordinate_zero_based=COORD,
       note='At each M, trial groups are disjoint. Different M share prefixes. Complete outputs are freshly drawn.'))
    print(f'CLT/moment pool fresh={trials*maxM} done {float(data["elapsed"]):.1f}s',flush=True)
    return dict(task='moments',status='complete')

def exact_checks():
    cpt=CPTPreference();rng=np.random.default_rng(SEED+33);errs=[]
    for n in (2,3,8,21):
        u=rng.integers(0,5,(n,2))/10;g=rng.normal(size=(n,10))
        direct=np.mean([quantile_weight(np.delete(u,i,axis=0),u[i:i+1],cpt)[0]*g[i] for i in range(n)],axis=0)
        errs.append(float(np.max(abs(direct-pooled_gradient(u,g,cpt)))))
    tel=[]
    def U(n,k,p):
        indicator=np.r_[np.ones(k),np.zeros(n-k)]
        return pooled_gradient(np.c_[indicator,np.zeros(n)],indicator-p,cpt)
    for p in (.2,.5,.8):
        for half in (2,4,8,16):
            mn=sum(binom.pmf(k,half,p)*U(half,k,p) for k in range(half+1))
            m2=sum(binom.pmf(k,2*half,p)*U(2*half,k,p) for k in range(2*half+1));delta=0.
            for i in range(half+1):
                for j in range(half+1):
                    delta+=binom.pmf(i,half,p)*binom.pmf(j,half,p)*(U(2*half,i+j,p)-.5*(U(half,i,p)+U(half,j,p)))
            tel.append(dict(p=p,half=half,error=float(abs(delta-(m2-mn))),base_bias=float(mn-cpt.weight_prime(p)*p*(1-p))))
    law=load_law(0);full=law.draw(np.zeros(10),np.random.default_rng(SEED+55),61);de=[]
    for d in DIMS:
        lo=EpisodeLaw(law.market,law.day,law.wealth,law.reference,law.previous,replace(law.config,dimension=d),law.cpt)
        small=lo.draw(np.zeros(d),np.random.default_rng(SEED+55),61)
        de.append(dict(d=d,utilities_error=float(np.max(abs(full[0]-small[0]))),score_error=float(np.max(abs(full[1][:,:d]-small[1])))))
    assert max(errs)<1e-12 and max(x['error'] for x in tel)<1e-12
    assert max(x['utilities_error']+x['score_error'] for x in de)<1e-12
    return dict(sorted_loo_max_errors=errs,bernoulli_telescoping=tel,nested_dimensions=de)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=600);p.add_argument('--trials',type=int,default=400)
    p.add_argument('--max-M',type=int,default=64);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Use a fresh result directory')
    if min(a.repetitions,a.trials)<3 or a.max_M not in M_GRID:p.error('Invalid replication configuration')
    a.output.mkdir(parents=True,exist_ok=True)
    proto=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,budgets=BUDGETS,days=DAYS,methods=METHODS,
       repetitions=a.repetitions,trials=a.trials,max_M=a.max_M,dimensions=DIMS,coordinate=COORD,
       split_fraction=.125,exponent=EXPONENT,hard_level_cap=None,checks=exact_checks(),
       designs={b:dict(loo=loo_design(b),split={**asdict(design_for_budget(b)),'expected_cost':design_for_budget(b).expected_cost}) for b in BUDGETS},
       source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.rglob('*.py')},
       input_sha256=hashlib.sha256((ROOT/'inputs/paper_hs300.csv').read_bytes()).hexdigest(),
       scope='English E2 four tasks, current 30-stock/cash law. All estimates fresh. No prescribed bias, variance or normal samples. '
       'Dimensions use existing nested d=2,4,8,10 models at theta=0, not a minimax lower-bound test. All corrections retained.')
    dump(a.output/'protocol.json',proto);t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(context_job,(day,a.repetitions,str(a.output.resolve()))) for day in DAYS]
        futures.append(pool.submit(moment_job,(a.trials,a.max_M,str(a.output.resolve()))))
        for f in as_completed(futures):f.result()
    dump(a.output/'completion.json',dict(status='complete',runtime_seconds=time.perf_counter()-t,
       primary_estimator_vectors=4*len(BUDGETS)*a.repetitions*len(METHODS)+a.trials*a.max_M*len(METHODS)))
if __name__=='__main__':main()
