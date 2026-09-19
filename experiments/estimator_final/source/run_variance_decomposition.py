from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, json, time, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from mvp_cpt_pg import paper_experiments as pe
from run_online_hybrid_v5 import load_market, hybrid_estimate, v5_estimate
from evaluation_estimator import evaluate_fixed

METHODS=('hybrid','v5_raw')
LABELS={'hybrid':'Centered LOO base + raw random correction',
        'v5_raw':'CPT-PG v5 adapted (raw split plug-in)'}
FIXED_EXECUTION_SEED=424242
FIXED_ESTIMATOR_SEED=700041
FIXED_ACTION_SEED=515151
FIXED_EVAL_SEED=616161
EVAL_BUDGET=768
N_REPLICATIONS=64
STEPS=500
TRAIN_BUDGET=512


def rng_for(master:int, episode:int, stream:int):
    return np.random.default_rng(np.random.SeedSequence([int(master),int(episode),int(stream)]))


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')


def estimator(method, law, theta, seed, episode):
    rng=rng_for(seed,episode,10)
    if method=='hybrid':
        return hybrid_estimate(law,theta,rng,TRAIN_BUDGET)
    if method=='v5_raw':
        return v5_estimate(law,theta,rng,TRAIN_BUDGET)
    raise ValueError(method)


def diagnostic(law, theta, episode):
    # Deliberately fixed common-random-number evaluator. Its randomness does not vary
    # across the replications within either decomposition experiment.
    rng=rng_for(FIXED_EVAL_SEED,episode,20)
    g,info=evaluate_fixed(law,theta,rng,estimator='unbiased',budget=EVAL_BUDGET,repeats=1)
    return g,info


def run_path(method:str, execution_seed:int, estimator_seed:int, replicate:int, experiment:str, output:str):
    market=load_market()
    c=pe.PaperConfig(trajectory_budget=TRAIN_BUDGET,n=64,evaluation_n=1024,evaluation_m=512)
    execution=market.execution_returns(execution_seed)
    theta=np.zeros(c.dimension)
    wealth=reference=1.0
    previous=np.zeros(len(market.panel.codes)); previous[0]=1.0
    peak=1.0
    rows=[]; daily=[]; total_train=0; total_eval=0; max_level=0
    st=time.time()
    for episode,day in enumerate(range(0,STEPS,c.horizon),1):
        law=pe.EpisodeLaw(market,day,wealth,reference,previous.copy(),c)
        gradient,train_info=estimator(method,law,theta,estimator_seed,episode)
        diag,eval_info=diagnostic(law,theta,episode)
        q=(pe.projected_update(theta,diag,c)-theta)/c.gamma
        theta_before=theta.copy()
        # Hold action random numbers fixed in BOTH experiments. Thus A varies only
        # estimator RNG; B varies only the realized execution-return path.
        for offset in range(c.horizon):
            index=day+offset
            features=pe.policy_features(market.panel.features[index],previous,c.dimension)
            target,_=pe.policy(theta,features,rng_for(FIXED_ACTION_SEED,index,40))
            target=pe.action_map(previous,target,market.panel.tradable[index],c.trade_fraction)
            old=wealth
            wealth,fee,turnover=pe.next_wealth(wealth,previous,target,execution[index],c)
            reference=float(pe.update_reference(wealth,reference,c)); peak=max(peak,wealth)
            daily.append(dict(experiment=experiment,method=method,replicate=replicate,
                execution_seed=execution_seed,estimator_seed=estimator_seed,episode=episode,
                day=index,wealth=float(wealth),net_return=float(wealth/old-1),
                drawdown=float(1-wealth/peak),turnover=float(turnover),fee=float(fee),cash=float(target[0])))
            previous=target
        theta=pe.projected_update(theta,gradient,c)
        total_train+=int(train_info['calls']); total_eval+=int(eval_info['actual_paths'])
        max_level=max(max_level,int(train_info.get('level',0)))
        rec=dict(experiment=experiment,method=method,replicate=replicate,execution_seed=execution_seed,
            estimator_seed=estimator_seed,episode=episode,day=day,wealth=float(wealth),reference=float(reference),
            q_squared=float(q@q),gradient_squared=float(gradient@gradient),
            update_norm=float(np.linalg.norm(theta-theta_before)),theta_norm=float(np.linalg.norm(theta)),
            training_paths=int(train_info['calls']),training_level=int(train_info.get('level',0)),
            evaluation_paths=int(eval_info['actual_paths']))
        for j in range(c.dimension):
            rec[f'theta_{j}']=float(theta[j])
            rec[f'gradient_{j}']=float(gradient[j])
            rec[f'q_{j}']=float(q[j])
        rows.append(rec)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    stem=f'{experiment}_{method}_{replicate:03d}'
    pd.DataFrame(rows).to_csv(out/f'{stem}_episodes.csv',index=False)
    pd.DataFrame(daily).to_csv(out/f'{stem}_daily.csv',index=False)
    dump(out/f'{stem}_completion.json',dict(status='complete',experiment=experiment,method=method,
        replicate=replicate,execution_seed=execution_seed,estimator_seed=estimator_seed,
        runtime_seconds=time.time()-st,total_training_paths=total_train,total_evaluation_paths=total_eval,
        maximum_training_level=max_level))
    return stem


def make_jobs(experiment, output, nrep):
    jobs=[]
    if experiment=='A':
        # Fixed realized market path; only estimator master seed changes.
        estimator_seeds=[1000003+7919*i for i in range(nrep)]
        for method in METHODS:
            for r,s in enumerate(estimator_seeds):
                jobs.append((method,FIXED_EXECUTION_SEED,s,r,'A',output))
    elif experiment=='B':
        # Fixed estimator random stream; only realized market path changes.
        execution_seeds=[2000003+6151*i for i in range(nrep)]
        for method in METHODS:
            for r,s in enumerate(execution_seeds):
                jobs.append((method,s,FIXED_ESTIMATOR_SEED,r,'B',output))
    else: raise ValueError(experiment)
    return jobs


def run_experiment(experiment, output:Path, workers:int, nrep:int):
    if output.exists():
        raise FileExistsError(f'fresh output required: {output}')
    output.mkdir(parents=True)
    protocol=dict(experiment=experiment,n_replications=nrep,methods=METHODS,labels=LABELS,steps=STEPS,
        episodes=STEPS//5,training_expected_budget=TRAIN_BUDGET,evaluation_budget=EVAL_BUDGET,
        evaluation_repeats=1,fixed_action_seed=FIXED_ACTION_SEED,fixed_evaluation_seed=FIXED_EVAL_SEED,
        isolation=(
          'Fixed execution-return path and fixed action/evaluation random numbers; estimator master seed varies.'
          if experiment=='A' else
          'Fixed estimator/action/evaluation random numbers; only execution-return path seed varies.'),
        fixed_execution_seed=(FIXED_EXECUTION_SEED if experiment=='A' else None),
        fixed_estimator_seed=(FIXED_ESTIMATOR_SEED if experiment=='B' else None),
        fixed_estimator_seed_selection=('For Experiment B, seed 700041 was selected before outcome generation from candidate seeds 700000-700499 solely because its 100-episode hybrid realized path count averaged 513.62, closest to the 512 target among the screened schedules with a moderate maximum level; no tracking or wealth outcome was used.' if experiment=='B' else None))
    dump(output/'protocol.json',protocol)
    jobs=make_jobs(experiment,str(output.resolve()),nrep); st=time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fut=[ex.submit(run_path,*j) for j in jobs]
        for i,f in enumerate(as_completed(fut),1):
            f.result()
            if i%16==0 or i==len(fut): print(experiment,i,'/',len(fut),flush=True)
    episodes=pd.concat([pd.read_csv(p) for p in sorted(output.glob(f'{experiment}_*_episodes.csv'))],ignore_index=True)
    daily=pd.concat([pd.read_csv(p) for p in sorted(output.glob(f'{experiment}_*_daily.csv'))],ignore_index=True)
    episodes.to_csv(output/'episodes.csv',index=False); daily.to_csv(output/'daily.csv',index=False)
    dump(output/'completion.json',dict(status='complete',jobs=len(jobs),runtime_seconds=time.time()-st))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=8); ap.add_argument('--replications',type=int,default=N_REPLICATIONS)
    ap.add_argument('--experiments',nargs='+',choices=['A','B'],default=['A','B']); a=ap.parse_args()
    a.output_root.mkdir(parents=True,exist_ok=True)
    for e in a.experiments:
        run_experiment(e,a.output_root/f'experiment_{e}',a.workers,a.replications)
if __name__=='__main__': main()
