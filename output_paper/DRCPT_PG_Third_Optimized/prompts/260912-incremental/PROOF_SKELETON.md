# Incremental obligation ledger

Target: DRCPT_PG_Third_Optimized.tex, SHA256 AF2E82C28ED85785F933B4E7E295C02A3085D814458F74B8D1F320A474433E4D.

Baseline snapshot matches the previously audited third manuscript byte for byte (740C762EC88A3A1B618BF08EF193B87F1A43331E0B401756CA50F7146B59A362). The advisor snapshot differs from the current main.tex only by one trailing blank line.

## Dependency and assumption ledger

| Obligation | Necessary inputs | Incremental assessment |
|---|---|---|
| Lemma C.1: Dirichlet moments and domination | compact parameter neighborhood; bounded parameter-independent features; finite horizon/assets; parameter-independent conditional market kernels | Concentration, log-moment and dominating-law calculations retained. No pointwise score bound is claimed. |
| Theorem 5.2: CPT gradient and smoothness | C.1; bounded utilities on the analysis domain; fixed C3 regularized weights | Likelihood differentiation and threshold Fubini bounds retained. Value uses w; score multiplier uses w derivative. |
| Proposition 5.1: reference mechanism | scalar recursion; common wealth/score coupling; value Lipschitz modulus; 5.2 for gradient sensitivity | Branch conditions and coupling restriction retained. Gradient identity does not depend on this proposition, so forward citation creates no logical cycle. |
| Proposition 5.3: antithetic estimator | 5.2; independent inner/outer samples; full/half estimates share identical outer score; fixed regularization | Taylor, fourth-moment and summability arguments retained; line 1065 has an undefined legacy W derivative. |
| Theorem 5.4: MSE | 5.3; independent complete replications; separate simulator discrepancy | Sampling variance, simulation bias, cap bias retained. Shared-inner plug-in variance uses total variance and Efron-Stein. |
| Theorem 5.5: CLT | finite second moments; iid complete replications at a fixed conditional context | Limit center is simulator gradient for untruncated estimates, cap-specific mean for fixed caps. No cross-episode CLT. |
| Lemma 5.6: set drift | gradient variation; same parameter domain/step/normalization; optional uniform residual error bound | Conditional set-distance result only. No proof that general CPT objectives satisfy the error bound. |
| Lemma D.1: projected ascent | smoothness; closed convex domain; positive normalization floor; gamma L <= a0 | Inner-product macro and all uses are consistent after their joint change; constants retained. |
| Theorem 5.7: online residual bound | D.1; bounded objective; value variation; conditional direction MSE | Pathwise telescope remains valid for endogenous contexts. Reference decomposition separately requires coupling. |
| Corollary 5.8: static/sublinear-variation limit | uniform constants; average MSE vanishes; complete objective fixed or sublinear variation | Average expected residual limit only; fixed simulation/cap/sample errors do not imply a zero limit. |
| Corollary 5.9: set tracking | 5.7 plus uniform residual error bound | Average squared distance follows conditionally. Window cancellation has no converse implication. |
| Theorem 5.10: minimax | exact-trajectory oracle; fixed dimension/domain; expected observation budget; positive endpoint slope | Realizable Bernoulli/Beta submodel, stopped KL, testing bound and cost-calibrated upper bound retained. Independent of the online theorem. |

## Core object inventory

Twelve grouped semantic objects retained: state, policy parameter, Dirichlet policy, true/simulation trajectory laws, reference recursion, terminal relative outcome, true/simulation CPT objectives, their gradients, estimator, normalized projected update, stationary set, residual/window criterion. Grouping is for interface preservation, not an exhaustive count of every mathematical symbol.

## Formal verification scope

Inherited Core.lean was executed successfully using lake env lean. Its rational identities do not prove the real-analysis results above. In particular averageDistanceFromResidual merely restates its hypothesis.

AuditAlgebra.lean was executed successfully: cubic multiplier coefficient identity, affine/quadratic antithetic cancellation, and a window-cancellation counterexample. No sorry/admit/custom axioms are used. This verifies these exact rational statements only. CPT differentiation, uniform error bounds, stochastic telescoping, CLT and minimax are not fully formalized in Lean.
