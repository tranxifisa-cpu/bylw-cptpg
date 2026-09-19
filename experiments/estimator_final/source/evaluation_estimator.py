"""One common centered evaluator, independent of training for every method.

Unbiasedness is a property of the gradient, under PROOF_ZH.md's assumptions.
It does not pass through squaring or the nonlinear residual map.
"""
from run_split_validation import np
from estimators import chunked_draw, quantile_weight
from split_debias import design_for_budget, estimate
from mvp_cpt_pg import paper_experiments as pe


def evaluate_fixed(law, theta, rng, *, estimator='unbiased', budget=1536, repeats=4):
    if estimator not in ('unbiased','plugin') or repeats < 1:
        raise ValueError('Choose unbiased/plugin and positive repeats')
    design = design_for_budget(budget, inner_fraction=2/3)
    total = np.zeros(len(theta)); values=[]; costs=[]; levels=[]; roles=[]
    # The objective remains a plug-in diagnostic, evaluated using existing A paths.
    objective_values=[]
    for _ in range(repeats):
        first = [True]
        def draw(random, count):
            utilities, scores = chunked_draw(law, theta, random, count)
            if first[0]:
                objective_values.append(pe.objective(utilities, law.cpt)); first[0]=False
            return utilities, scores
        def stat(a,b,g): return np.mean(quantile_weight(a,b,law.cpt)[:,None]*g, axis=0)
        if estimator == 'unbiased':
            g, info = estimate(draw, stat, rng, design)
        else:
            # Match the legacy inner/outer proportion and the nominal expected cost.
            n = int(budget*2/3); m = int(budget)-n
            if min(n,m)<1: raise ValueError('Budget too small')
            a,_ = draw(rng,n); b,s = draw(rng,m); g = stat(a,b,s)
            info = dict(calls=n+m, expected_calls=n+m, level=0,
                        role_counts=dict(A=n,B=m,C=0,D=0))
        total += g; values.append(g); costs.append(info['calls']); levels.append(info['level'])
        roles.append(list(info['role_counts'].values()))
    return total/repeats, dict(
        estimator=estimator, repeats=repeats, nominal_budget_per_repeat=budget,
        expected_paths=repeats*(design.expected_cost if estimator=='unbiased' else budget),
        actual_paths=int(sum(costs)), active_corrections=int(np.count_nonzero(levels)),
        maximum_level=int(max(levels)), role_counts=np.sum(roles,axis=0).tolist(),
        objective_plugin=float(np.mean(objective_values)), replicate_gradients=np.asarray(values).tolist())
