import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse, numpy as np, json, sys
from run_e2_hybrid_v5 import load_law,hybrid_once,v5_once,loo_design,SEED,METHODS

def work(args):
    start,stop,B=args; law=load_law(0);theta=np.zeros(10);n=stop-start
    est=np.empty((n,2,10));calls=np.empty((n,2),int)
    for jj,i in enumerate(range(start,stop)):
        ss1,ss2=np.random.SeedSequence([SEED,777,i]).spawn(2)
        est[jj,0],calls[jj,0],_=hybrid_once(law,theta,B,ss1)
        est[jj,1],calls[jj,1],_=v5_once(law,theta,B,ss2)
    return start,est,calls

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--trials',type=int,default=200);ap.add_argument('--max-M',type=int,default=64);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    if a.output.exists():
        import shutil;shutil.rmtree(a.output)
    a.output.mkdir(parents=True)
    total=a.trials*a.max_M;edges=np.linspace(0,total,a.workers+1,dtype=int);parts=[]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        fs=[ex.submit(work,(int(edges[j]),int(edges[j+1]),128)) for j in range(a.workers)]
        for f in as_completed(fs):parts.append(f.result())
    parts.sort(); est=np.concatenate([x[1] for x in parts]);calls=np.concatenate([x[2] for x in parts])
    np.savez_compressed(a.output/'outputs.npz',estimates=est,calls=calls,methods=np.array(METHODS),expected_costs=np.array([loo_design(128)['expected_cost'],128.]))
    (a.output/'design.json').write_text(json.dumps(dict(trials=a.trials,max_M=a.max_M,M_grid=[1,2,4,8,16,32,64],dimensions=[2,4,8,10],unit_budget=128,coordinate=2),indent=2))
if __name__=='__main__':main()
