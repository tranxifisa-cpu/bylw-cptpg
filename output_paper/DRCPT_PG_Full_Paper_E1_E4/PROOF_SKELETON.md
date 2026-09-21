# Proof Skeleton

## Scope

This ledger tracks the current theorem-bearing manuscript chain:

- `sections/03_problem_setup.tex`
- `sections/04_assumptions.tex`
- `sections/05_algorithm.tex`
- `sections/06_main_results.tex`
- `appendix/A_notation.tex`
- `appendix/B_reference_and_gradient.tex`
- `appendix/C_estimator.tex`
- `appendix/D_online.tex`
- `appendix/E_minimax.tex`

The main text now states claim-level results; proof-only lemmas are kept in the appendices.  E1--E7 are populated and use the same estimator, residual, and reference definitions as the theory sections.

## Dependency DAG

1. `ass:trajectory` + the explicit Dirichlet actor imply positive compact concentration parameters and finite score moments.
2. `ass:likelihood` identifies the policy score as the complete explicit parameter derivative of the frozen conditional trajectory law.
3. Steps 1--2 + `ass:cpt-regularity` imply CPT smoothness and `thm:gradient-identity`.
4. `thm:gradient-identity` + the second-order expansion and Hoeffding decomposition imply `lem:level-cancellation`.
5. `lem:level-cancellation` + leave-one-out independence imply `lem:centering-expectation`.
6. The two estimator lemmas + the untruncated randomized level law imply exact unbiasedness, finite variance, and finite expected cost in `thm:estimator`; iid replication gives `thm:clt`.
7. CPT smoothness + projection geometry imply the appendix lemma `lem:projected-ascent`.
8. `thm:estimator` + `lem:projected-ascent` + the problem-defined budgets `beta_k` and `V_K` imply `thm:online-residual`.
9. `thm:online-residual` + the square-first window definition imply `cor:dynamic-local-regret` by nonnegative reweighting and reindexing.
10. The local geometry condition `eq:error-bound` converts residual control to `cor:stationary-tracking`; together with gradient drift it also yields the appendix lemma `lem:stationary-set-drift`.
11. `thm:online-residual` + sublinear objective variation + vanishing Cesaro estimator/simulator error imply `cor:vanishing-tracking`.
12. The complete-trajectory oracle experiment in `def:minimax-oracle` + the realizable Bernoulli-return submodel gives the minimax lower bound; `thm:estimator` under the same expected trajectory budget gives the upper bound in `thm:minimax`.

## Standing-assumption ledger

| Standing condition | Main role | How it is checked in the manuscript | If the global form is unavailable |
|---|---|---|---|
| `ass:trajectory` compact policy and trajectory domain | uniform score moments, bounded objective, episode-uniform smoothness constants | policy domain, horizon/assets, feature bounds and analyzed trajectory ranges | fixed-target arguments remain local; online constants become local or high-probability |
| `ass:likelihood` policy-mediated likelihood structure | validates the trajectory-score derivative used by the gradient estimator | simulator interface: after the action is fixed, the market kernel and feasibility map have no extra theta input | add the direct likelihood/pathwise derivative; otherwise the present score estimates only the policy-mediated component |
| `ass:cpt-regularity` smooth regularized CPT weighting | gradient identity, Hessian/smoothness control, second-order LOO remainder | analytic derivatives of the chosen regularized weight maps | weaker identities may survive, but the present finite-sample remainder/variance proof needs replacement tail or margin control |

Result-specific conditions are not standing assumptions: reference-isolated coupling is local to `prop:reference-sensitivity`; `eq:error-bound` is local to stationary-set interpretation; and the minimax information source is defined by `def:minimax-oracle`.

## Canonical quantified statements

### Frozen-target estimator

For every fixed conditional target satisfying the standing assumptions, every integer (n\ge2), activation probability (\varrho\in(0,1]), exponent (b\in(1,2)), and (M\ge1) independent complete outputs,

- (E[Z_n^{db}]=g);
- (operatorname{tr}\operatorname{Cov}(Z_n^{db})\le C_{var}(n)<\infty);
- (E[\operatorname{cost}(Z_n^{db})]=C_{cost}(n,\varrho,b)<\infty);
- (E\|M^{-1}\sum_m Z_{n,m}^{db}-g\|^2\le C_{var}(n)/M).

### Online residual

For every (K\ge1) and fixed (\gamma,\vartheta>0) with (\gamma L\le\vartheta),

[
\mathcal E_K
\le
\frac{8(2B_v+E\mathcal V_K)}{\vartheta\gamma K}
+
\frac{10C_g}{\vartheta^3K}
\sum_{k=1}^K
\left(\frac{C_{var}(n)}{M_k}+E\beta_k^2\right).
]

With the square-first window statistic,
[
\mathcal R_{\rho,w}(K)
\le
W_{\rho,w}\sum_{k=1}^K\zeta_k,
\qquad
W_{\rho,w}=\sum_{j=0}^{w-1}\rho^j.
]
Thus fixed (w,\rho) preserve the vanishing-rate conclusion when (\mathcal E_K\to0).

### Minimax rate

For fixed dimension and fixed regularization/model-class constants under `def:minimax-oracle`,
[
c_{lb}/\mathsf B\le\mathfrak R_{\mathsf B}\le C_{ub}/\mathsf B
]
for all sufficiently large expected complete-trajectory budgets (\mathsf B).

## Micro-claim inventory

- Compact theta and bounded features keep Dirichlet concentrations in a positive compact interval.
- Dirichlet density derivatives admit an integrable log-polynomial envelope.
- Policy-mediated likelihood structure turns the retained latent-action score into the required likelihood-ratio derivative.
- Reference recursion is ((1-\eta_{min}))-Lipschitz under the reference-isolated common-wealth comparison.
- The regularized CPT score identity follows from likelihood differentiation, dominated convergence, Fubini, and score centering.
- Raw LOO admits a first-order influence term plus an (O_{L^2}(n^{-1})) remainder.
- Full/half coupling cancels the common influence term, giving (E\|D_{n,\ell}\|^2=O(n^{-2}4^{-\ell})).
- Centered and raw LOO share the same finite-sample mean; that mean differs from the target by (O(n^{-1})).
- Untruncated randomized corrections telescope from the finite-sample mean to the exact frozen-target gradient.
- (b<2) gives finite correction variance and (b>1) gives finite expected correction cost.
- Normalized direction is (\vartheta^{-1})-Lipschitz; projection geometry yields the one-step ascent inequality.
- Moving-objective increments telescope and are controlled by (2B_v+\mathcal V_K).
- Estimator MSE plus moving-objective variation yields the average squared projected-residual bound.
- Square-first DLR averages already-squared contemporaneous residuals; reindexing gives the fixed-window factor (W_{\rho,w}).
- The local error bound, when imposed, converts residuals to stationary-set distance and controls set drift.
- The one-stock Bernoulli submodel is realizable by the stated actor/reference recursion and supplies the fixed-dimensional minimax lower bound.
- Cost-matched replication of the exact-unbiased estimator supplies the minimax upper bound.

## Limit-order map

- (M\to\infty): fixed target, fixed ((n,\varrho,b)), fixed dimension.
- (K\to\infty): fixed (n,\gamma,\vartheta,w,\rho); standing constants uniform over episodes; (E\mathcal V_K=o(K)) and Cesaro estimator/simulator error vanish.
- (mathsf B\to\infty): fixed dimension, regularization, and oracle model-class constants.
