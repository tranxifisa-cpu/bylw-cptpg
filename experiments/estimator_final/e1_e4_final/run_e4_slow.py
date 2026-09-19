from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'

import argparse, json, time, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

from common_slow_continuous import SlowVariationSpec, load_slow_market, slow_config
from run_e3_slow_continuous import State, initial_state, run_method, add_metrics, TRAIN_BUDGET

E4_SEED_BASE=2000
ETA_GRID=(0.0,0.05,0.10,0.20,0.40)
HORIZON_GRID=(2,5,10,25)
GAMMA_GRID=(0.01,0.02,0.05,0.08)
VARTTHETA_GRID=(0.005,0.01,0.03,0.05)
L_DIAG=0.31
EVAL_BUDGET=768
EVAL_REPEATS=1
TOTAL_DAYS=1500


def admissible(g,v):
    return float(g)*L_DIAG <= float(v)+1e-15


def make_spec(horizon:int)->SlowVariationSpec:
    h=int(horizon)
    if TOTAL_DAYS % h or 200 % h or 300 % h or 600 % h:
        raise ValueError('horizon must divide all market-time boundaries')
    return SlowVariationSpec(episodes=TOTAL_DAYS//h,horizon=h,stable_end=200//h,
                             abrupt_end=300//h,drift_end=600//h,state_power=.75)


def run_pair(seed:int, *, horizon:int=5, gamma:float=.08, vartheta:float=.05,
             eta_gain:float=.20, eta_loss:float=.05):
    spec=make_spec(horizon);market=load_slow_market(spec)
    config=slow_config(10,TRAIN_BUDGET,gamma,vartheta,horizon=horizon,
                       eta_gain=eta_gain,eta_loss=eta_loss)
    states={'online':initial_state(market,config.dimension),'frozen':initial_state(market,config.dimension)}
    rows=[]
    for episode in range(1,spec.episodes+1):
        execution=market.execution_episode(seed,episode,config.horizon)
        ns,r,_=run_method('online',states['online'],market,execution,config,seed,episode,EVAL_BUDGET,EVAL_REPEATS,'unbiased')
        states['online']=ns;rows.append(r)
        if episode<=spec.stable_end:
            states['frozen']=ns.copy();fr=r.copy();fr['method']='frozen';rows.append(fr)
        else:
            fs,fr,_=run_method('frozen',states['frozen'],market,execution,config,seed,episode,EVAL_BUDGET,EVAL_REPEATS,'unbiased')
            states['frozen']=fs;rows.append(fr)
    D=add_metrics(pd.DataFrame(rows),config.dimension)
    post=D[D.episode>spec.stable_end]
    totals=post.groupby('method').agg(dlr=('dlr_term','sum'),residual=('projected_residual_term','sum')).astype(float)
    online_dlr=float(totals.loc['online','dlr']);frozen_dlr=float(totals.loc['frozen','dlr'])
    online_res=float(totals.loc['online','residual']);frozen_res=float(totals.loc['frozen','residual'])
    return dict(online_dlr=online_dlr,frozen_dlr=frozen_dlr,
                delta_track=float(1-online_dlr/frozen_dlr),
                online_residual=online_res,frozen_residual=frozen_res,
                delta_residual=float(1-online_res/frozen_res),episodes=spec.episodes)


def one(job):
    panel,key1,key2,seed=job
    kw={}
    if panel=='reference': kw=dict(eta_gain=float(key1),eta_loss=float(key2))
    elif panel=='horizon': kw=dict(horizon=int(key1))
    elif panel=='optimizer': kw=dict(gamma=float(key1),vartheta=float(key2))
    else: raise ValueError(panel)
    out=run_pair(seed,**kw)
    out.update(panel=panel,key1=float(key1),key2=(np.nan if key2 is None else float(key2)),seed=int(seed))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seeds',type=int,default=4);ap.add_argument('--workers',type=int,default=16)
    a=ap.parse_args()
    if a.output.exists():ap.error('fresh output required')
    a.output.mkdir(parents=True)
    optimizer_pairs=[(g,v) for g in GAMMA_GRID for v in VARTTHETA_GRID if admissible(g,v)]
    protocol=dict(seed_base=E4_SEED_BASE,seeds=a.seeds,baseline=dict(gamma=.08,vartheta=.05,eta_gain=.20,eta_loss=.05,horizon=5),
                  reference_grid=ETA_GRID,horizon_grid=HORIZON_GRID,
                  optimizer_grid=dict(gamma=GAMMA_GRID,vartheta=VARTTHETA_GRID,L_diag=L_DIAG,admissible=optimizer_pairs),
                  metric='Delta_track = 1 - post-change DLR_online / post-change DLR_frozen',
                  environment='same continuous slow-variation setup as E3; state gain tied to elapsed market time; total market time 1500 days',
                  evaluation=dict(mode='unbiased',budget=EVAL_BUDGET,repeats_per_stream=EVAL_REPEATS,independent_streams=2),
                  role='robustness only; E4 does not select the default configuration')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    jobs=[]
    for eg in ETA_GRID:
        for el in ETA_GRID:
            for s in range(a.seeds):jobs.append(('reference',eg,el,E4_SEED_BASE+s))
    for h in HORIZON_GRID:
        for s in range(a.seeds):jobs.append(('horizon',h,None,E4_SEED_BASE+s))
    for g,v in optimizer_pairs:
        for s in range(a.seeds):jobs.append(('optimizer',g,v,E4_SEED_BASE+s))
    st=time.time();rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as ex:
        fs=[ex.submit(one,j) for j in jobs]
        for f in as_completed(fs):
            r=f.result();rows.append(r);print('E4',r['panel'],r['key1'],r['key2'],r['seed'],'done',flush=True)
    pd.DataFrame(rows).sort_values(['panel','key1','key2','seed'],na_position='last').to_csv(a.output/'seed_metrics.csv',index=False)
    (a.output/'completion.json').write_text(json.dumps(dict(status='complete',runtime_seconds=time.time()-st,jobs=len(rows)),indent=2),encoding='utf-8')

if __name__=='__main__':main()
