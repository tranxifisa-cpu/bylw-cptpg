from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'

import argparse, json, math, time, sys, multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))

from mvp_cpt_pg.paper_experiments import action_map,next_wealth,policy,policy_features,projected_update,update_reference
from common_slow_continuous import SlowVariationSpec, SlowEpisodeLaw, load_slow_market, slow_config, relax_state
from evaluation_estimator import evaluate_gradient
from run_e2_final import loo_design, R
from estimators import chunked_draw
from loo_controls import loo_components
from mvp_cpt_pg.controlled_cpt import quantile_weight

ROOT_SEED=2026092307
TRAIN_BUDGET=512
MIN_TRAIN_REPEATS=4
WINDOW=5
RHO=0.9

@dataclass
class State:
    theta: np.ndarray
    wealth: float
    reference: float
    previous: np.ndarray
    def copy(self):
        return State(self.theta.copy(),float(self.wealth),float(self.reference),self.previous.copy())


def initial_state(market,dimension):
    p=np.zeros(len(market.panel.codes)); p[0]=1.0
    return State(np.zeros(dimension),1.0,1.0,p)


def training_repeats(k:int,horizon:int=5)->int:
    equivalent_k=max(1,int(math.ceil(k*int(horizon)/5.0)))
    return max(MIN_TRAIN_REPEATS,math.isqrt(equivalent_k-1)+1)


def proposed_gradient(law,theta,seed,repeats):
    """Batch M independent complete unbiased outputs without changing the estimator law.

    The original implementation called law.draw separately for every base output.
    Here all M independent base trajectory blocks are drawn in one vectorized call,
    while Bernoulli activations and randomized correction levels remain independent.
    This is a compute-only reorganization: the centered LOO base and raw correction
    formulas, activation probability, level law, and expected trajectory cost are unchanged.
    """
    d=loo_design(TRAIN_BUDGET); n=d['n']; rho=d['rho']
    ss_base,ss_ctrl,ss_corr=np.random.SeedSequence(seed).spawn(3)
    rg_base=np.random.default_rng(ss_base); rg_ctrl=np.random.default_rng(ss_ctrl); rg_corr=np.random.default_rng(ss_corr)
    u,g=chunked_draw(law,theta,rg_base,n*repeats)
    u=u.reshape(repeats,n,2); g=g.reshape(repeats,n,len(theta))
    vals=[]; calls=n*repeats; maxlevel=0
    active=rg_ctrl.random(repeats)<rho
    levels=np.zeros(repeats,dtype=int)
    if active.any():
        levels[active]=rg_ctrl.geometric(1-R,size=int(active.sum()))
    for j in range(repeats):
        pieces=loo_components(u[j],g[j],law.cpt)
        value=pieces[0]-pieces[1:].sum(axis=0)
        level=int(levels[j])
        if level>0:
            N=n*(1<<level)
            uc,gc=chunked_draw(law,theta,rg_corr,N); h=N//2
            rf=loo_components(uc,gc,law.cpt)[0]
            r1=loo_components(uc[:h],gc[:h],law.cpt)[0]
            r2=loo_components(uc[h:],gc[h:],law.cpt)[0]
            q=(1-R)*R**(level-1)
            value=value+(rf-.5*(r1+r2))/(rho*q)
            calls+=N; maxlevel=max(maxlevel,level)
        vals.append(value)
    return np.mean(vals,axis=0),calls,maxlevel


def _plugin_eval_stream(law,theta,seed,count):
    rng=np.random.default_rng(np.random.SeedSequence(seed))
    inner,_=chunked_draw(law,theta,rng,count)
    outer,scores=chunked_draw(law,theta,rng,count)
    g=np.mean(quantile_weight(inner,outer,law.cpt)[:,None]*scores,axis=0)
    return g,2*count

def evaluate_period(law,theta,seed,episode,budget,repeats,mode='unbiased'):
    if mode=='unbiased':
        g1,i1=evaluate_gradient(law,theta,[ROOT_SEED,seed,episode,70],budget=budget,repeats=repeats)
        g2,i2=evaluate_gradient(law,theta,[ROOT_SEED,seed,episode,71],budget=budget,repeats=repeats)
        return g1,g2,dict(actual=int(i1['actual_paths']+i2['actual_paths']),expected=float(i1['expected_paths']+i2['expected_paths']),gap=float(np.linalg.norm(g1-g2)),maxlevel=int(max(i1['maximum_level'],i2['maximum_level'])))
    if mode=='plugin':
        # Diagnostic-only high-precision independent plug-in streams.  Training remains unchanged.
        g1,c1=_plugin_eval_stream(law,theta,[ROOT_SEED,seed,episode,70],budget)
        g2,c2=_plugin_eval_stream(law,theta,[ROOT_SEED,seed,episode,71],budget)
        return g1,g2,dict(actual=c1+c2,expected=float(c1+c2),gap=float(np.linalg.norm(g1-g2)),maxlevel=0)
    raise ValueError(mode)


def execute_raw(market,execution,config,state,seed,episode):
    theta=state.theta
    wealth=float(state.wealth);reference=float(state.reference);previous=state.previous.copy()
    daily=[]
    for j in range(config.horizon):
        feat=policy_features(execution['features'][j],previous,config.dimension)
        rng=np.random.default_rng(np.random.SeedSequence([ROOT_SEED,seed,episode,j,40]))
        latent,_=policy(theta,feat,rng)
        target=action_map(previous,latent,execution['tradable'][j],config.trade_fraction)
        oldw=wealth
        wealth,fee,turn=next_wealth(wealth,previous,target,execution['returns'][j],config)
        reference=float(update_reference(wealth,reference,config))
        daily.append(dict(offset=j,date=str(execution['dates'][j]),raw_wealth=float(wealth),
                          raw_reference=float(reference),raw_cash=float(target[0]),
                          raw_net_return=float(wealth/oldw-1),turnover=float(turn),fee=float(fee)))
        previous=target
    return float(wealth),float(reference),previous,daily


def run_method(method,state,market,execution,config,seed,episode,eval_budget,eval_repeats,eval_mode):
    theta=state.theta.copy()
    law=SlowEpisodeLaw(market,episode,state.wealth,state.reference,state.previous.copy(),config)
    g1,g2,einfo=evaluate_period(law,theta,seed,episode,eval_budget,eval_repeats,eval_mode)
    gmean=.5*(g1+g2)
    q1=(projected_update(theta,g1,config)-theta)/config.gamma
    q2=(projected_update(theta,g2,config)-theta)/config.gamma
    qmean=(projected_update(theta,gmean,config)-theta)/config.gamma
    if method=='online':
        M=training_repeats(episode,config.horizon)
        tg,calls,maxlevel=proposed_gradient(law,theta,[ROOT_SEED,seed,episode,90],M)
        newtheta=projected_update(theta,tg,config)
    elif method=='frozen':
        M=0;calls=0;maxlevel=0;tg=np.zeros_like(theta);newtheta=theta
    else: raise ValueError(method)

    raww,rawr,rawp,daily=execute_raw(market,execution,config,state,seed,episode)
    a=market.spec.state_gain(episode)
    wealth,reference,previous=relax_state(state.wealth,state.reference,state.previous,raww,rawr,rawp,a)
    newstate=State(newtheta,wealth,reference,previous)
    co,mu,sc=market.spec.parameters(episode)
    if episode==1:
        co0,mu0,sc0=co,mu,sc
    else:
        co0,mu0,sc0=market.spec.parameters(episode-1)
    exog_step=float(np.linalg.norm(co-co0)+abs(mu-mu0)+abs(sc-sc0))
    state_step=float(abs(np.log(wealth/state.wealth))+abs(reference-state.reference)+np.abs(previous-state.previous).sum())
    raw_state_step=float(abs(np.log(raww/state.wealth))+abs(rawr-state.reference)+np.abs(rawp-state.previous).sum())
    row=dict(seed=seed,method=method,episode=episode,stage=market.spec.stage(episode),
             state_gain=a,wealth_start=state.wealth,reference_start=state.reference,cash_start=state.previous[0],
             raw_wealth_end=raww,raw_reference_end=rawr,raw_cash_end=rawp[0],
             wealth_end=wealth,reference_end=reference,cash_end=previous[0],
             state_step_norm=state_step,raw_state_step_norm=raw_state_step,exogenous_step_norm=exog_step,
             theta_norm=float(np.linalg.norm(theta)),update_norm=float(np.linalg.norm(newtheta-theta)),
             training_repeats=M,training_trajectories=int(calls),training_max_level=maxlevel,
             training_expected_paths=float(M*loo_design(TRAIN_BUDGET)['expected_cost']),
             evaluation_actual_paths=einfo['actual'],evaluation_expected_paths=einfo['expected'],
             evaluation_stream_gap=einfo['gap'],evaluation_max_level=einfo['maxlevel'],
             execution_block_start=int(execution['start']))
    for j in range(config.dimension):
        row[f'theta_{j}']=theta[j];row[f'q1_{j}']=q1[j];row[f'q2_{j}']=q2[j];row[f'q_{j}']=qmean[j]
        row[f'eval_g1_{j}']=g1[j];row[f'eval_g2_{j}']=g2[j];row[f'train_g_{j}']=tg[j]
    return newstate,row,daily


def add_metrics(df,dimension):
    out=[];W=float(np.sum(RHO**np.arange(WINDOW)))
    for method,g in df.groupby('method',sort=False):
        g=g.sort_values('episode').copy()
        q1=g[[f'q1_{j}' for j in range(dimension)]].to_numpy(float)
        q2=g[[f'q2_{j}' for j in range(dimension)]].to_numpy(float)
        zeta=np.einsum('ij,ij->i',q1,q2)
        g['projected_residual_term']=zeta
        g['cumulative_projected_residual']=np.cumsum(zeta)
        g['average_projected_residual']=np.cumsum(zeta)/np.arange(1,len(zeta)+1)
        terms=[]
        for i in range(len(g)):
            a=q1[max(0,i-WINDOW+1):i+1][::-1];b=q2[max(0,i-WINDOW+1):i+1][::-1]
            ww=RHO**np.arange(len(a))
            s1=np.sum(a*ww[:,None],axis=0)/W;s2=np.sum(b*ww[:,None],axis=0)/W
            terms.append(float(s1@s2))
        terms=np.asarray(terms)
        g['dlr_term']=terms;g['dynamic_local_regret']=np.cumsum(terms);g['average_dynamic_local_regret']=np.cumsum(terms)/np.arange(1,len(terms)+1)
        g['cumulative_state_variation']=g.state_step_norm.cumsum()
        g['average_state_variation']=g.cumulative_state_variation/np.arange(1,len(g)+1)
        g['cumulative_exogenous_variation']=g.exogenous_step_norm.cumsum()
        g['average_exogenous_variation']=g.cumulative_exogenous_variation/np.arange(1,len(g)+1)
        out.append(g)
    return pd.concat(out,ignore_index=True)


def run_seed(args):
    seed,out,gamma,vartheta,eval_budget,eval_repeats,eval_mode=args
    spec=SlowVariationSpec();market=load_slow_market(spec);config=slow_config(10,TRAIN_BUDGET,gamma,vartheta)
    states={'online':initial_state(market,config.dimension),'frozen':initial_state(market,config.dimension)}
    rows=[];daily=[]
    for episode in range(1,spec.episodes+1):
        if episode==1 or episode%20==0:
            print('slow E3 seed',seed,'episode',episode,'start',flush=True)
        execution=market.execution_episode(seed,episode,config.horizon)
        ns,r,dd=run_method('online',states['online'],market,execution,config,seed,episode,eval_budget,eval_repeats,eval_mode)
        states['online']=ns;rows.append(r)
        daily.extend(dict(seed=seed,method='online',episode=episode,stage=r['stage'],**x) for x in dd)
        if episode<=spec.stable_end:
            states['frozen']=ns.copy();fr=r.copy();fr['method']='frozen';rows.append(fr)
            daily.extend(dict(seed=seed,method='frozen',episode=episode,stage=r['stage'],**x) for x in dd)
        else:
            fs,fr,fd=run_method('frozen',states['frozen'],market,execution,config,seed,episode,eval_budget,eval_repeats,eval_mode)
            states['frozen']=fs;rows.append(fr)
            daily.extend(dict(seed=seed,method='frozen',episode=episode,stage=fr['stage'],**x) for x in fd)
    frame=add_metrics(pd.DataFrame(rows),config.dimension)
    out=Path(out);(out/'raw').mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/'raw'/f'seed_{seed:03d}.csv',index=False)
    pd.DataFrame(daily).to_csv(out/'raw'/f'seed_{seed:03d}_daily.csv',index=False)
    print('slow E3 seed',seed,'done',flush=True)
    return seed


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seeds',type=int,default=20);ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--gamma',type=float,default=.08);ap.add_argument('--vartheta',type=float,default=.05);ap.add_argument('--eval-budget',type=int,default=1024);ap.add_argument('--eval-repeats',type=int,default=2);ap.add_argument('--eval-mode',choices=['unbiased','plugin'],default='unbiased')
    a=ap.parse_args()
    if a.output.exists():ap.error('fresh output required')
    a.output.mkdir(parents=True)
    spec=SlowVariationSpec()
    protocol=dict(seeds=a.seeds,episodes=spec.episodes,horizon=spec.horizon,methods=['online','frozen'],
                  optimizer=dict(gamma=a.gamma,vartheta=a.vartheta,source='frozen from prior independent calibration; not retuned on slow environment'),
                  sampling=dict(schedule='M_k=max(4,ceil(sqrt(k)))',budget_per_complete_output=TRAIN_BUDGET),
                  evaluation=dict(mode=a.eval_mode,independent_streams=2,budget=a.eval_budget,repeats_per_stream=a.eval_repeats),
                  dlr=dict(window=WINDOW,rho=RHO,definition='temporal average of residual vectors, then squared; estimated by independent cross streams'),
                  environment=dict(kind='continuous slow-variation semi-synthetic environment',
                                   factor_law='time-invariant empirical A-share block distribution',
                                   exogenous='episodes 1-40 stable momentum; 41-60 reversal; 61-120 finite interpolation; 121-300 fixed terminal law',
                                   endogenous='wealth/reference/holdings never reset; episode transition under-relaxed by alpha_k=k^{-3/4}',
                                   state_gain='alpha_k=k^{-0.75}; sum alpha_k=O(K^0.25)=o(K) under bounded raw transitions'),
                  comparator='same path through episode 40; theta frozen thereafter, while its endogenous state continues under the same slow transition rule')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    st=time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('spawn')) as ex:
        fs=[ex.submit(run_seed,(s,str(a.output),a.gamma,a.vartheta,a.eval_budget,a.eval_repeats,a.eval_mode)) for s in range(a.seeds)]
        for f in as_completed(fs):f.result()
    (a.output/'completion.json').write_text(json.dumps(dict(status='complete',runtime_seconds=time.time()-st),indent=2),encoding='utf-8')

if __name__=='__main__':main()
