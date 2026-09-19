"""Independent probability/score blocks, plus an independent debiasing block.

A: n inner paths; B: m outer paths. Fresh C/D supply the correction.
Within the correction, C's full sample and halves MUST share the same D.
No level caps, correction clipping, redraws or completion censoring.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SplitDesign:
    n: int
    m: int
    exponent: float = 1.5
    activation: float | None = None
    correction_outer: int | None = None

    def __post_init__(self):
        for name, value in [('n',self.n),('m',self.m),('correction_outer',self.mc)]:
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:
                raise ValueError(f'{name} must be a positive integer')
        if not 1 < self.exponent < 2 or not 0 < self.rho <= 1:
            raise ValueError('Need 1 < exponent < 2 and 0 < activation <= 1')

    @property
    def mc(self): return self.m if self.correction_outer is None else self.correction_outer

    @property
    def rho(self): return (self.n+self.m)**(-.5) if self.activation is None else self.activation

    @property
    def r(self): return 2.**(-self.exponent)

    @property
    def kappa(self): return 2*(1-self.r)/(1-2*self.r)

    @property
    def expected_cost(self): return self.n+self.m+self.rho*(self.kappa*self.n+self.mc)

    def probability(self,level): return (1-self.r)*self.r**(level-1)


def design_for_budget(budget,inner_fraction=.125,exponent=1.5):
    if not math.isfinite(budget) or not 0<inner_fraction<1:
        raise ValueError('Finite budget and interior fraction required')
    for total in range(math.floor(budget),1,-1):
        n=max(1,min(total-1,math.floor(total*inner_fraction)))
        design=SplitDesign(n,total-n,exponent)
        if design.expected_cost<=budget: return design
    raise ValueError('Budget too small')


def correction(inner,outer,scores,statistic):
    if len(inner)<2 or len(inner)%2:
        raise ValueError('Correction inner block must split into two equal halves')
    half=len(inner)//2
    return (statistic(inner,outer,scores)
            -.5*(statistic(inner[:half],outer,scores)+statistic(inner[half:],outer,scores)))


def estimate(draw,statistic,rng,design):
    """statistic(inner_utility, outer_utility, outer_score) may return a vector
    or a stack of vectors for paired raw/centered comparisons. draw calls must
    return fresh iid paths at the same frozen context. Costs count full paths.
    """
    inner,_=draw(rng,design.n)
    outer,scores=draw(rng,design.m)
    base=statistic(inner,outer,scores)
    delta=np.zeros_like(base);adjustment=np.zeros_like(base)
    active=bool(rng.random()<design.rho)
    level=0;inner_c=outer_c=0
    if active:
        level=int(rng.geometric(1-design.r))
        inner_c=design.n*(1<<level);outer_c=design.mc
        new_inner,_=draw(rng,inner_c)
        new_outer,new_scores=draw(rng,outer_c)
        delta=correction(new_inner,new_outer,new_scores,statistic)
        adjustment=delta/(design.rho*design.probability(level))
    return base+adjustment,dict(base=np.asarray(base).tolist(),delta=np.asarray(delta).tolist(),
        adjustment=np.asarray(adjustment).tolist(),active=active,level=level,
        calls=design.n+design.m+inner_c+outer_c,expected_calls=design.expected_cost,
        role_counts=dict(A=design.n,B=design.m,C=inner_c,D=outer_c))
