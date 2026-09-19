"""A bounded follow-up using the existing eight multi-asset conditional laws.

Both raw-v5 and centered kernels use identical sampled blocks for comparison.
Every individual estimator uses independent A, B, C, D path groups internally.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np
import pandas as pd
from scipy.stats import binom
from mvp_cpt_pg.paper_market import PaperMarket,PaperPanel
from mvp_cpt_pg.paper_experiments import EpisodeLaw,PaperConfig
from mvp_cpt_pg.cpt_objective import CPTPreference
from estimators import v5_weight,quantile_weight,chunked_draw
from split_debias import SplitDesign,design_for_budget,estimate,correction

SEED=2026091611
BUDGETS=(128,512,2048)
METHODS=('split_debias_raw','split_debias_centered')


def write_json(path,data): Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False))
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_market(): return PaperMarket(PaperPanel.load(ROOT/'inputs/paper_hs300.csv','20230301','20260529'),True)


def pair_stat(inner,outer,scores,cpt):
    return np.array([np.mean(fn(inner,outer,cpt)[:,None]*scores,axis=0)
                     for fn in (v5_weight,quantile_weight)])


def exact_checks():
    cpt=CPTPreference(); rows=[]
    for p in (.2,.5,.8):
        for n in (1,2,4,8,16):
            # In this finite model the outer expectation is integrated exactly.
            outer=np.array([[0.,0.],[1.,0.]])
            outer_score=np.array([-p,1-p])
            outer_prob=np.array([1-p,p])
            def stat(inner,ignored_outer,ignored_score):
                return np.array([np.dot(outer_prob,fn(inner,outer,cpt)*outer_score)
                                 for fn in (v5_weight,quantile_weight)])
            means=[]
            for size in (n,2*n):
                total=np.zeros(2)
                for k in range(size+1):
                    inner=np.c_[np.r_[np.ones(k),np.zeros(size-k)],np.zeros(size)]
                    total+=binom.pmf(k,size,p)*stat(inner,None,None)
                means.append(total)
            expected_delta=np.zeros(2);second=np.zeros(2)
            for j in range(n+1):
                for k in range(n+1):
                    inner=np.c_[np.r_[np.ones(j),np.zeros(n-j),np.ones(k),np.zeros(n-k)],np.zeros(2*n)]
                    delta=correction(inner,None,None,stat)
                    prob=binom.pmf(j,n,p)*binom.pmf(k,n,p)
                    expected_delta+=prob*delta;second+=prob*delta**2
            err=float(np.max(np.abs(expected_delta-(means[1]-means[0]))))
            assert err<2e-13
            assert np.max(np.abs(means[0]-means[0][0]))<2e-13
            rows.append(dict(p=p,n=n,expectation_telescoping_error=err,
                bias_raw=float(means[0][0]-cpt.weight_prime(p)*p*(1-p)),
                delta_second_moment=second.tolist()))
    rng=np.random.default_rng(SEED+99);role_calls=[]
    def draw(random,n):
        role_calls.append(n);return random.random((n,2)),random.normal(size=(n,10))
    design=SplitDesign(8,48,activation=1.)
    z,info=estimate(draw,lambda a,b,g:pair_stat(a,b,g,cpt),rng,design)
    assert len(role_calls)==4 and role_calls==list(info['role_counts'].values())
    assert sum(role_calls)==info['calls'] and z.shape==(2,10)
    assert info['level']>=1
    return dict(exact_bernoulli=rows,independent_call_roles=info['role_counts'],
        note='Finite-sum and accounting diagnostics, not experimental proofs of unbiasedness.')


def summarize(values,refs):
    mean_ref=refs.mean(axis=0);noise=np.var(refs,axis=0,ddof=1).sum()/len(refs)
    var=np.var(values,axis=0,ddof=1).sum()
    raw=np.sum((values-mean_ref)**2,axis=1)
    return dict(mse_raw=float(raw.mean()),mse_corrected=float(raw.mean()-noise),
        reference_noise=float(noise),variance=float(var),
        squared_bias_corrected=float(np.sum((values.mean(axis=0)-mean_ref)**2)-var/len(values)-noise))


def run_case(job):
    case,reps,destination=job;out=Path(destination)/f'case_{case:02d}';out.mkdir(exist_ok=False)
    ctx=json.loads((ROOT/f'prior_results/case_{case:02d}/case.json').read_text())
    state=ctx['context'];theta=np.array(ctx['theta'])
    law=EpisodeLaw(load_market(),state['day'],state['wealth'],state['reference'],
        np.array(state['previous']),PaperConfig(**ctx['config']),CPTPreference(**ctx['preference']))
    refs=np.load(ROOT/f'prior_results/case_{case:02d}/reference.npz')['estimates']
    draw=lambda random,n:chunked_draw(law,theta,random,n)
    stat=lambda a,b,g:pair_stat(a,b,g,law.cpt)
    rows=[]
    for budget in BUDGETS:
        design=design_for_budget(budget,.125)
        rng=np.random.default_rng(np.random.SeedSequence([SEED,case,budget]))
        values=np.empty((reps,2,10));bases=np.empty_like(values);corrs=np.empty_like(values)
        calls=[];levels=[];roles=[]
        for rep in range(reps):
            z,info=estimate(draw,stat,rng,design)
            values[rep]=z;bases[rep]=info['base'];corrs[rep]=info['adjustment']
            calls.append(info['calls']);levels.append(info['level']);roles.append(list(info['role_counts'].values()))
        assert np.isfinite(values).all()
        np.savez_compressed(out/f'budget_{budget}.npz',estimates=values,bases=bases,corrections=corrs,
            calls=calls,levels=levels,role_counts=roles,methods=METHODS)
        for i,method in enumerate(METHODS):
            rows.append(dict(case=case,budget=budget,method=method,repetitions=reps,
                **summarize(values[:,i],refs),n=design.n,m=design.m,correction_m=design.mc,
                activation=design.rho,expected_calls=design.expected_cost,observed_calls=float(np.mean(calls)),
                max_calls=max(calls),p99_calls=float(np.quantile(calls,.99)),max_level=max(levels),
                activations=int(np.sum(np.array(levels)>0))))
        print(f'Independent split: case {case}, budget {budget}, {reps} completed',flush=True)
        pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    if case%2==0:
        layer=[];rng=np.random.default_rng(np.random.SeedSequence([SEED,case,987]))
        for N in (32,64,128,256,512):
            vals=[]
            for _ in range(200):
                a,_=draw(rng,N);b,g=draw(rng,128)
                vals.append(correction(a,b,g,stat))
            vals=np.array(vals)
            np.savez_compressed(out/f'layer_{N}.npz',differences=vals)
            for i,method in enumerate(METHODS):
                second=float(np.mean(np.sum(vals[:,i]**2,axis=1)))
                layer.append(dict(case=case,method=method,N=N,outer=128,second_moment=second))
        pd.DataFrame(layer).to_csv(out/'layers.csv',index=False)
    return case


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'results')
    ap.add_argument('--repetitions',type=int,default=600)
    ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args()
    if a.repetitions<2 or a.workers<1:ap.error('Need >=2 repetitions and >=1 worker')
    if a.output.exists() and any(a.output.iterdir()):ap.error('Use a fresh result directory')
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,budgets=BUDGETS,
        inner_fraction=.125,exponent=1.5,activation='(n+m)^(-1/2)',correction_outer='m',
        designs={B:{**asdict(design_for_budget(B)), 'expected_cost':design_for_budget(B).expected_cost} for B in BUDGETS},
        methods=METHODS,repetitions=a.repetitions,max_level=None,
        checks=exact_checks(),source_sha256={str(p.relative_to(ROOT)):digest(p) for p in ROOT.rglob('*.py')},
        previous_summary_sha256=digest(ROOT/'prior_results/estimator_summary.csv'),
        scope='Exploratory new construction, same eight conditional multi-asset laws and old independent reference. '
        'Four independent sample roles A/B/C/D; C-full and C-halves share D internally. No proposed-parameter tuning '
        'after this protocol. Mean error corrected for reference variance. No exact truth or optimal allocation claimed.')
    write_json(a.output/'protocol.json',protocol)
    started=time.perf_counter()
    jobs=[(i,a.repetitions,str(a.output.resolve())) for i in range(8)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(run_case,j) for j in jobs]):f.result()
    pd.concat([pd.read_csv(a.output/f'case_{i:02d}/summary.csv') for i in range(8)],ignore_index=True).to_csv(a.output/'summary.csv',index=False)
    pd.concat([pd.read_csv(a.output/f'case_{i:02d}/layers.csv') for i in range(0,8,2)],ignore_index=True).to_csv(a.output/'layers.csv',index=False)
    write_json(a.output/'completion.json',dict(status='complete',runtime_seconds=time.perf_counter()-started,
        vector_estimates=8*3*2*a.repetitions))


if __name__=='__main__':main()
