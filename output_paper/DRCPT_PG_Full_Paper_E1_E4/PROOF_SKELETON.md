# Proof Skeleton

## Scope

Audited theorem-bearing files:

- `sections/03_problem_setup.tex`
- `sections/04_assumptions.tex`
- `sections/05_algorithm.tex`
- `sections/06_main_results.tex`
- `appendix/A_notation.tex`
- `appendix/B_reference_and_gradient.tex`
- `appendix/C_estimator.tex`
- `appendix/D_online.tex`
- `appendix/E_minimax.tex`

Current revision: the full source was read for consistency; the mathematical obligation ledger below covers the theorem-bearing files. Introduction/related work and E1–E4 are populated, E5–E7 remain design sections, and Discussion is still the original placeholder. This precision repair does not fill or rewrite them.

## Dependency DAG

The dependency graph is acyclic.

1. `ass:trajectory` + explicit Dirichlet actor -> `lem:dirichlet-moments`.
2. `lem:dirichlet-moments` + `ass:cpt-regularity` -> `lem:cpt-smoothness` and `thm:gradient-identity`.
3. `thm:gradient-identity` + Taylor expansion + Hoeffding decomposition -> `lem:level-cancellation`.
4. `lem:level-cancellation` + leave-one-out independence -> `lem:centering-expectation`.
5. `lem:level-cancellation` + `lem:centering-expectation` + randomized level law -> `thm:estimator`.
6. `thm:estimator` -> `thm:clt`.
7. `lem:cpt-smoothness` + projection geometry -> `lem:projected-ascent`.
8. `thm:estimator` + `ass:online-variation` + `lem:projected-ascent` -> `thm:online-residual`.
9. `thm:online-residual` + discounted residual-vector averaging + Jensen -> `cor:dynamic-local-regret`.
10. `thm:online-residual` + `ass:error-bound` -> `cor:stationary-tracking` and `lem:stationary-set-drift`.
11. `thm:online-residual` + sublinear variation/MSE conditions -> `cor:vanishing-tracking`.
12. `ass:minimax-oracle` + realizable Bernoulli-return submodel -> minimax lower bound; `thm:estimator` + expected-cost calibration -> minimax upper bound -> `thm:minimax`.

## Assumption Ledger

| Result | Required assumptions / inputs | Discharge location |
|---|---|---|
| Score moments and likelihood-ratio identities | bounded factor features; compact parameter set; finite horizon/assets; theta-independent market kernel and feasibility map | Appendix A, `lem:dirichlet-moments` |
| CPT smoothness | score moments; fixed C3 regularized weighting functions; bounded value range | Appendix A, `lem:cpt-smoothness` |
| Reference sensitivity | bounded CPT derivatives; reference-isolated coupling | Appendix B |
| Gradient identity | likelihood-ratio identities; bounded CPT derivative; finite integration range | Appendix B |
| Full/half cancellation | C3 weight regularity; fourth moments of Bernoulli empirical survival; score second moment | Appendix C |
| Exact unbiasedness | centered/raw equal mean; raw full/half telescoping; no hard level truncation; independent base/correction | Appendix C |
| Finite variance | level second moment O(4^-ell); level pmf exponent b<2; base variance O(1/n) | Appendix C |
| Finite expected cost | level pmf exponent b>1 | Appendix C |
| CLT | iid complete outputs; finite second moment; fixed target/design | Appendix C |
| One-step projected ascent | L-smooth objective; closed convex Theta; gamma L <= vartheta | Appendix D |
| Online residual | one-step ascent; estimator MSE; objective variation budget | Appendix D |
| Dynamic local regret | nonnegative exponential weights summing to one; convexity of squared norm | Appendix D |
| Stationary-set tracking | residual error bound with uniform kappa | Appendix D |
| Minimax lower bound | exact-trajectory oracle; fixed d; realizable Bernoulli-return submodel; positive endpoint slope | Appendix E |
| Minimax upper bound | uniform estimator constants over model class; expected trajectory cost | Appendix E |

## Canonical Quantified Statements

### Estimator theorem
For every fixed conditional target satisfying Assumptions 4.1-4.2, every integer n>=2, activation probability varrho in (0,1], exponent b in (1,2), and M>=1 independent complete outputs,

- E[Z_n^db] = g;
- tr Cov(Z_n^db) <= C_var(n) < infinity;
- E[cost(Z_n^db)] = C_cost(n,varrho,b) < infinity;
- E||M^{-1} sum_m Z_{n,m}^db - g||^2 <= C_var(n)/M.

All constants are uniform over episodes only when the primitive bounds in the standing assumptions are uniform over episodes.

### Online residual theorem
For every K>=1, fixed gamma,vartheta>0 satisfying gamma L<=vartheta, and the supplied deterministic replication schedule M_k>=1,

E_K <= 8(2 B_v + E V_K)/(vartheta gamma K)
      + 10 C_g/(vartheta^3 K) sum_k [C_var(n)/M_k + E beta_k^2].

### Minimax theorem
For fixed dimension d and fixed regularization/model-class constants, there exist c_lb,C_ub,B_0 in (0,infinity) such that for every budget B>=B_0,

c_lb/B <= R_B <= C_ub/B.

No dimension-uniform d/B claim and no online minimax-regret claim is made.

## Micro-Claim Inventory

- MC-01: compact theta + bounded q -> all Dirichlet concentrations lie in a positive compact interval.
- MC-02: the Dirichlet density derivatives admit a common integrable log-polynomial envelope.
- MC-03: MC-02 -> first/second likelihood-ratio differentiation for bounded trajectory functionals.
- MC-04: reference recursion is (1-eta_min)-Lipschitz in the reference coordinate under common wealth paths.
- MC-05: shifted-power values are globally Lipschitz with the displayed constant.
- MC-06: integrated survival-function difference is bounded by coupled E|V-V'|.
- MC-07: CPT score identity follows by likelihood-ratio differentiation, DCT, Fubini, and score centering.
- MC-08: raw LOO Taylor expansion decomposes into a linear influence term, a degenerate ordered U-statistic, and a quadratic remainder.
- MC-09: the degenerate ordered U-statistic has L2 norm O(n^-1).
- MC-10: the Taylor remainder has L2 norm O(n^-1).
- MC-11: the common influence term cancels exactly in full-vs-half coupling, giving E||D_{n,ell}||^2=O(n^-2 4^-ell).
- MC-12: centered and raw LOO statistics have the same finite-sample mean because the centering coefficient excludes trajectory i and E G_i=0.
- MC-13: randomized raw corrections telescope from mu_n to g and are absolutely integrable.
- MC-14: b<2 gives finite correction second moment; b>1 gives finite expected correction cost.
- MC-15: iid averaging gives 1/M MSE and the fixed-target multivariate CLT.
- MC-16: normalized direction map is vartheta^-1-Lipschitz as a Euclidean projection onto the unit ball.
- MC-17: projection VI + L-smoothness + MC-16 -> one-step projected-ascent inequality.
- MC-18: moving-objective increments telescope pathwise and are bounded by 2 B_v + V_K.
- MC-19: MC-17 + MC-18 + true-gradient MSE -> average projected-residual bound.
- MC-20: Jensen applied to the full-window exponential weights gives each DLR term an upper bound by the corresponding weighted residual squares; reindexing yields `Reg_{rho,w}(K) <= sum_k zeta_k`.
- MC-21: error bound converts Q_k residuals to distances from S_k.
- MC-22: normalized residual maps inherit gradient drift, and MC-21 converts it to Hausdorff drift of stationary sets.
- MC-23: Bernoulli-return one-stock submodel is realizable by the stated Dirichlet actor and reference recursion.
- MC-24: gradient map g(p) has derivative bounded below on a fixed interior p-interval.
- MC-25: stopped adaptive transcript KL equals expected call count times one-call Bernoulli KL in the two-point submodel.
- MC-26: Pinsker + nearest-target testing gives the Omega(B^-1) lower bound.
- MC-27: finite expected estimator cost + 1/M risk gives the O(B^-1) upper bound.

## Limit-Order Map

- M -> infinity: fixed target, fixed (n,varrho,b), fixed d; CLT and MSE scaling.
- K -> infinity: fixed n, fixed gamma,vartheta, fixed finite w and rho; standing constants uniform over episodes; requires E V_K=o(K) and Cesaro estimator/simulator error -> 0.
- B -> infinity: fixed d, fixed regularization, fixed model-class constants; minimax expected complete-trajectory budget.


## Precision-repair delta (2026-09-19)

- I1–I2: define the theorem-visible survival, score and population gradient before use.
- I3–I4: use G_x for a trajectory score and a single T_call for the oracle count.
- I5–I7: explicit consecutive episodes, model/induced-law distinction, and uniform conditional true/simulator likelihood assumptions.
- I8: remove unused B_J; B_v and all endpoint constants unchanged.
- I9: one 16-row semantic-chain table, with (z_k,F_k,r_k) restored; estimator internals retain their local definitions.
- I10: four metric names are distinct; every displayed metric definition is unchanged.

MC-02, MC-03, MC-07 through MC-19 now explicitly inherit the simulator scope of Assumption trajectory. No dependency edge is reversed or added from an online result to the estimator. The fixed-target minimax chain remains separate from online tracking.

The new full-source review may report preexisting obligations outside I1–I10. Consult PROOF_AUDIT.md for the scoped/full-paper verdict distinction; this ledger does not assert that such observations were silently repaired.
