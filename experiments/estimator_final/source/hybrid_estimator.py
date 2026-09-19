"""Pilot-free centered-LOO base plus RAW independent random correction.

This function does not change the user's online update or market model. It is
provided for a later controlled integration. Strict unbiasedness and finite
variance require the fixed-target assumptions in proofs/CENTERING_EXTENSION.md.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from estimators import chunked_draw
from loo_controls import loo_components
from run_e2 import loo_design,R

@dataclass(frozen=True)
class HybridResult:
    gradient: np.ndarray
    base: np.ndarray
    correction: np.ndarray
    base_n: int
    activated: bool
    level: int
    trajectories: int
    expected_trajectories: float


def estimate_hybrid(law, theta: np.ndarray, budget: int,
                    seed: int | np.random.SeedSequence) -> HybridResult:
    """Estimate the fixed conditional gradient with independent random streams.

    budget counts expected COMPLETE paths, not a hard maximum. There is no
    random-level cap. Repeated calls must receive distinct seeds/SeedSequences.
    """
    theta=np.asarray(theta,dtype=float)
    if theta.shape!=(law.config.dimension,) or not np.isfinite(theta).all():
        raise ValueError('theta must be a finite vector of the actor dimension')
    if not isinstance(budget,(int,np.integer)):
        raise TypeError('budget must be an integer expected-path allowance')
    design=loo_design(int(budget));n=design['n'];rho=design['rho']
    ss=seed if isinstance(seed,np.random.SeedSequence) else np.random.SeedSequence(seed)
    rg_base,rg_ctrl,rg_corr=[np.random.default_rng(x) for x in ss.spawn(3)]
    u,g=chunked_draw(law,theta,rg_base,n)
    pieces=loo_components(u,g,law.cpt)
    base=pieces[0]-pieces[1:].sum(axis=0)
    correction=np.zeros_like(base);level=0;paths=n
    activated=bool(rg_ctrl.random()<rho)
    if activated:
        level=int(rg_ctrl.geometric(1-R));N=n*(1<<level)
        u,g=chunked_draw(law,theta,rg_corr,N);half=N//2
        raw_full=loo_components(u,g,law.cpt)[0]
        raw_h1=loo_components(u[:half],g[:half],law.cpt)[0]
        raw_h2=loo_components(u[half:],g[half:],law.cpt)[0]
        q=(1-R)*R**(level-1)
        correction=(raw_full-.5*(raw_h1+raw_h2))/(rho*q)
        paths+=N
    result=base+correction
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite output: do not clip or silently discard this replication')
    return HybridResult(result,base,correction,n,activated,level,paths,design['expected_cost'])
