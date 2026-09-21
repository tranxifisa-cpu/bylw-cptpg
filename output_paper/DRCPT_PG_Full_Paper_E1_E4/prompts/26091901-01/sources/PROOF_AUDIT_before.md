# Proof Audit

Target: current Problem Setup / Assumptions / Algorithm / Main Results / Appendices A-E
Verdict: WARN
Claim status: core proof chain appears coherent; notation/model-interface repairs are required before notation gate can pass
Reviewer backend: local-executor-fallback
Reviewer model: GPT-5.6 Sol
Review independence: none
Acceptance status: provisional-evidence-only

## Notation Gate Scorecard (current manuscript; preliminary full-core census)

Core semantic objects retained: 13/13 (100%)
Undefined symbols: 3
Symbol collisions: 1
One-use definitions: 2/41 (4.9%)
Maximum parallel representations of one object: 4
Maximum alias-chain depth: 2
Maximum active nonstandard symbols in one proof step: 13

The gate is **not ready** because undefined symbols and a same-base-letter collision remain.

## Main finding

The mathematical dependency direction is now much stronger than the old advisor draft: true and simulated objectives are separated, the projected residual rather than a raw gradient is the constrained stationarity object, DLR is downstream of the residual, and the minimax experiment is separated from the online theorem.  The remaining problems are primarily symbol/interface discipline, plus one assumption-scope point that should be made explicit because it feeds the estimator and online chains.

## Issues

### I1: gradient theorem introduces two undefined shorthands

- Severity: MINOR (notation blocker)
- Category: MISSING_DERIVATION / UNCLEAR
- Location: `sections/06_main_results.tex`, gradient identity theorem
- Problem: the theorem writes `S_theta^pm=S_{k,theta}^pm` although `S_{k,theta}^pm` was never defined; it also uses `G_theta` although only `G_{k,theta}` was defined globally.
- Minimal repair: at the theorem start write directly
  `S_theta^pm(y):=S_{P_k,z_k,r_k,theta}^pm(y)` and `G_theta:=G_{k,theta}`.
  Do not create the intermediate alias `S_{k,theta}`.

### I2: `g` appears in the level-cancellation lemma before it is defined

- Severity: MINOR (notation blocker)
- Category: UNCLEAR
- Location: `sections/06_main_results.tex`, Lemma `level-cancellation`
- Problem: `U_n^raw-g` is displayed before the manuscript defines local `g:=nabla J(theta)`; that definition appears only in the following centering lemma.
- Minimal repair: define `g:=nabla J(theta)` in the level-cancellation lemma statement (or once at the beginning of the estimator subsection).

### I3: trajectory-score component uses the same base letter as the true gradient

- Severity: MINOR (notation blocker)
- Category: NORMALIZATION_MISMATCH / UNCLEAR
- Location: `appendix/C_estimator.tex`, opening paragraph
- Problem: a realization is written `x=(v^+,v^-,g_x)` and then `G_x=g_x`, while `g` denotes the true gradient throughout the appendix.  The subscript changes the meaning of the base glyph.
- Minimal repair: write the realization directly as `x=(v^+,v^-,G_x)` and delete `g_x`.

### I4: the minimax call count has two persistent representations

- Severity: MINOR
- Category: UNCLEAR
- Location: Assumption `minimax-oracle` vs `appendix/E_minimax.tex`
- Problem: the assumption uses `T_call`; the proof switches to `mathsf T_call` for the same stopping count.
- Minimal repair: use `T_call` everywhere.  If stopping-time status matters, state in prose that this data-dependent call count is a stopping time for the oracle filtration.

### I5: episode/calendar interface is almost explicit but still misses the consecutive-episode relation

- Severity: MINOR
- Category: MODEL_INTERFACE / UNCLEAR
- Location: `sections/03_problem_setup.tex`
- Problem: `r_k:=r_{t_k}` is now explicit, but the manuscript never states `t_{k+1}=t_k+h` for consecutive episodes.  The algorithm returns the next episode's wealth/reference/holdings as if this relation were understood.
- Minimal repair: after defining episode horizon, add one sentence: `For consecutive episodes, t_{k+1}=t_k+h.`

### I6: `P` alternates between “conditional law/model” and the theta-indexed trajectory law

- Severity: MAJOR clarity issue
- Category: MODEL_INTERFACE / UNCLEAR
- Location: `sections/03_problem_setup.tex`, generic CPT objective
- Problem: the prose calls `P` a conditional trajectory law but then writes `P_theta{...}`.  Elsewhere `P_k` behaves like the true episode environment/model, which together with the policy induces a theta-dependent augmented trajectory law.  The current wording obscures which object is parameter independent, a fact needed by likelihood differentiation.
- Minimal repair: keep the existing symbols but say explicitly: `For a frozen episode environment/model P and context (z,r), let P_theta denote the induced augmented trajectory law under pi_theta.`  Then `P_k` and `tilde P_k` are the true and simulator episode models, and `P_{k,theta}` is induced only when needed.  Do not add another persistent model symbol.

### I7: standing likelihood/domain assumption does not explicitly say that the same regularity holds for frozen simulators

- Severity: MAJOR
- Category: HIDDEN_ASSUMPTION
- Impact: global to estimator -> online chain
- Location: Assumption `trajectory`
- Problem: Appendix C conditions on the frozen simulator and the online theorem invokes the estimator under `tilde P_k`.  Assumption 3.1 states bounds on the analyzed domain and a parameter-independent market kernel, but does not explicitly quantify over both the true and frozen-simulator models.  The historical audited version did make this scope explicit.
- Minimal repair: if intended by the model, state that the bounds and parameter-independence conditions hold uniformly for every analyzed true law `P_k` and frozen simulator `tilde P_k`.  If that is not intended, the estimator theorem must be stated conditionally on simulator-specific moment assumptions instead.

### I8: `B_J` is a dead symbol

- Severity: MINOR
- Category: exposition / one-use definition
- Location: `sections/06_main_results.tex`, end of online-residual theorem
- Problem: `B_J:=sup |J_k|` is defined but never used; the displayed theorem uses the already proved bound `B_v`.
- Minimal repair: delete the `B_J` sentence.  Do not replace `2B_v` unless a different bound is mathematically intended.

### I9: Appendix A is no longer advisor-style “main notation”

- Severity: MAJOR exposition issue
- Category: notation load
- Location: `appendix/A_notation.tex`, two notation tables
- Problem: the current appendix lists proof-local estimator internals (`nu_l`, `D_{n,l}`, `C_var(n)`, full minimax objects) while omitting the crucial conditioning bridge `(z_k,F_k)`.  It also repeats `rho,w` conceptually through the DLR row and the estimator/error table.  This obscures the actual chain of objects.
- Source comparison: the advisor-style correction preserved a single concise table (16 rows) and deliberately moved moment constants, Hessians, and multilevel-sampling details to their definition sites.  The TPI paper likewise introduces the framework notation first and local proof notation immediately before the proof where it is used.
- Minimal repair: return Appendix A to one concise table ordered by the semantic chain.  Keep only state/reference, context/information, policy, outcome, true/sim objectives, true/sim gradients, debiased update estimator, projected residual, stationary set, DLR, simulator/variation errors, and minimax budget.  Define `n,varrho,b,nu_l,D,Z,C_var` locally in Algorithm/Main Results instead of the global table unless a symbol is used across multiple sections.

### I10: terminology for the first-order objects should be frozen to four names

- Severity: MINOR
- Category: exposition
- Location: setup/results/experiments
- Problem: nearby prose alternates among “projected residual”, “periodwise residual square”, “stationarity criterion”, and “first-order tracking error”.  These phrases refer to different aggregation levels but can read as aliases.
- Minimal repair: use exactly:
  1. `projected residual` for `Q_k`;
  2. `squared projected residual` for `zeta_k`;
  3. `average expected squared projected residual` for `E_K`;
  4. `dynamic local regret (DLR)` for `Reg_{rho,w}`.
  Use “tracking” only as interpretation, not as a fifth mathematical noun.

## Correctness checks that currently pass

- `r_k:=r_{t_k}` removes the older daily/episode reference ambiguity at the start of an episode.
- The minimax proof uses `theta^[1]` for a coordinate and does not reuse online `theta_1`.
- Market/simulator gradient error `beta_k` and objective variation `V_K` are separated rather than overloading one drift symbol.
- The current projected-ascent lemma is formulated with the projected residual and feasible projection inequality; it is not the false boundary argument in the old advisor draft.
- DLR is one-way controlled by the sum of squared projected residuals; the proof does not reverse Jensen.
- The stationary-set distance conclusion is explicitly conditional on the separate error-bound assumption.
- The minimax lower and upper bounds use the same expected complete-trajectory budget.
- No duplicate LaTeX labels were found in the current core files.

## Recommended repair order

1. Clarify `P` versus induced `P_theta` and true/simulator scope of Assumption 3.1.
2. Fix the three undefined shorthands (`S_{k,theta}`, `G_theta`, early `g`).
3. Remove `g_x`, unify `T_call`, add `t_{k+1}=t_k+h`, delete `B_J`.
4. Replace the two Appendix-A notation tables with one advisor-style core table ordered by the semantic chain.
5. Run a second symbol census and theorem-hypothesis pass.
6. Only after the notation gate reaches zero undefined symbols/collisions, compile and visually inspect the appendix transition and key theorem pages.

## Remaining risks

- This is a local same-family audit, not a cross-family acceptance review.
- The simulator-scope issue requires author confirmation if the intended simulator class is broader than the true model class.
- A full exhaustive glyph census of every proof-local scalar constant has not yet been used as an acceptance gate; the current scorecard is for the persistent/core symbol layer plus theorem-visible shorthands.
