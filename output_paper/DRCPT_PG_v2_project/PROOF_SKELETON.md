# DRCPT-PG Proof Skeleton

## Scope

- Canonical source: `main.tex`.
- Source SHA-256: `F77A224B17BB22EC0CFDFFC56B1E7CF6E198E130FADC4A026A451256C7F216D6`.
- Audit date: 2026-09-09.
- The source manuscript was not edited.

## Mathematical Flow

1. Episode `k` starts from market state, normalized wealth `X`, reference `r`, previous portfolio `omega`, and parameter `theta`.
2. The Dirichlet factor policy maps bounded asset features `q` through `z = theta^T q` and `a = exp(z)` to a distribution on the long-only simplex.
3. The execution layer uses the policy mean; the learning layer simulates stochastic horizon-`h` trajectories from the episode-start state.
4. Each trajectory produces terminal relative wealth `Y_k = X_k - r_{k,h}`.
5. The gain and loss survival distributions of `v_+(Y_k)` and `v_-(Y_k)` define the regularized CPT objective `J_k(theta,r_k)`.
6. A score-function identity expresses `nabla J_k` as `E[psi_k(tau) G_k(tau)]`.
7. A proposed randomized multilevel estimator targets this gradient; its averaged output determines a normalized projected update.
8. Market and reference drift move the objective and its stationary set. Discounted recent gradients define dynamic local regret.

## Dependency Graph

| Result | Required inputs | Proof dependency | Audit status |
|---|---|---|---|
| Path dependence, Proposition 4.1 | asymmetric reference recursion | algebra | Verified algebraically and in Lean |
| CPT gradient identity, Theorem 4.2 | bounded integrability, smooth weights, DQM | differentiation under integral, score identity | Conditionally valid |
| Plug-in bias, Proposition 4.3 | cross-fitting, empirical survival, smooth derivative | second-order Taylor remainder | Incomplete assumptions |
| Debiased estimator, Proposition 4.3 | concrete coupled levels and summability | randomized telescoping | Abstract logic valid; construction incomplete |
| Variance bound, Theorem 4.4 | independent replications, finite second moment | variance of a mean | Conditional |
| CLT, Theorem 4.5 | fixed dimension, conditional iid, `2+xi` moment | multivariate CLT and Slutsky | Conditional |
| Stationary-set drift, Lemma 4.6 | gradient drift, uniform error bound | directed Hausdorff argument | Valid after symmetric drift clarification |
| Projected ascent, Lemma A.1 | smooth objective, Euclidean projection | claimed projection loss is quadratic | False as written; Lean counterexample |
| Dynamic local regret, Theorem 4.7 | projected ascent plus alignment and estimator MSE | telescoping | Not established |
| Static limit, Corollary 4.8 | identical episode objectives, valid regret theorem | specialization | Not established from current theorem |
| Set tracking, Corollary 4.9 | error bound plus raw-gradient control | squaring and averaging | First inequality valid; final control unsupported |
| Minimax lower bound, Theorem 4.10 | CPT-realizable statistical submodel | Assouad reduction | Model embedding missing |

## Lean Scope

`formal/DRCPTProofs/Basic.lean` mechanically checks:

- the two-path reference difference;
- gain-side and loss-side reference recursions;
- the squared error-bound implication;
- a boundary counterexample to the projected-ascent lemma;
- a cancellation counterexample showing that a small smoothed gradient need not make each raw gradient small.

The measure-theoretic, asymptotic, and stochastic-process claims remain paper proofs. Their validity depends on the missing hypotheses listed in `PROOF_AUDIT.md`.
