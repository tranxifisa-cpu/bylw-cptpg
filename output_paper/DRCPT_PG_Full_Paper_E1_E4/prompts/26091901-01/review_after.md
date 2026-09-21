# Fresh final mathematical re-review

Date: 2026-09-19. Reviewer: `/root/proof_review_final`, a fresh read-only Codex reviewer. `review_independence: same-family`; `acceptance_status: provisional`. This is a mathematical source review, not a cross-family acceptance, formal verification, execution audit of experiments, or validation of cited papers.

## Split verdict

**Authorized I1–I10 patch: PASS, provisional.** All ten original issues are closed in the final inspected source. One small I9 regression found during this review—the disappearance of the description of `m_u` as market state—was reported to the parent and restored in the existing Appendix A row. I reread that restoration. No remaining mathematical regression attributable to the authorized patch was found.

**Full manuscript: WARN; not a global acceptance.** The 17 stated mathematical results have coherent derivations when read with the manuscript's standing assumptions, fixed-target conditioning convention, and globally declared feasible parameter domain. No verified counterexample to that core chain was found. Preexisting experiment-to-theorem gaps and notation problems remain below. In particular, the proof-orchestrator whole-artifact notation gate is **AUDIT_FAILED**, so this report does not mark the complete manuscript `READY_FOR_USER`.

The first verdict does not certify every preexisting sentence. The second does not reopen unrelated material for editing. No manuscript file was edited by this reviewer.

## Inputs and scope

I read the actual complete contents of `main.tex`, all seven `sections/*.tex`, and all five `appendix/*.tex` in:

`/workspace/scratch/7f23b07a728e/work/english_audit_fix/DRCPT_PG_Intro_Related_Refs_FullDiff_Final/final_package_after/paper/DRCPT_PG_Full_Paper_E1_E4/`.

I also read the supplied `project_sources/04-proof_checker.md`, `tmp/source_review/orchestrator/SKILL.md`, and its `references/notation-audit.md`. The original root `tmp/proof_baseline/PROOF_AUDIT(1).md` and `PROOF_SKELETON(2).md` were used exclusively as the issue inventory and scope specification. Their favorable conclusions were not used as proof evidence. No previous generated review was read.

The baseline for change attribution was `tmp/proof_baseline/paper/DRCPT_PG_Full_Paper_E1_E4/`. I inspected the source diff. Changes are the requested model/likelihood clarifications, local symbol repairs, call-count unification, episode relation, deletion of the dead bound, notation-table replacement, and first-order terminology changes. The analytic calculations and numerical experiment values were preserved. The restored `m_u` explanation is a repair of the I9 table rewrite, not a new modeling assumption.

The mathematical census comprises six assumptions, two propositions, seven lemmas, five theorems, and three corollaries. The last four categories total **17 results**.

## Closure of original issues

| Issue | Source evidence and independent check | Status |
|---|---|---|
| I1 | `sections/06_main_results.tex:63` defines the survival shorthand directly from `S_{P_k,z_k,r_k,theta}` and sets `G_theta:=G_{k,theta}`. Both have the correct fixed-target type; the nonexistent intermediate survival alias is gone. | Closed |
| I2 | The level-cancellation lemma at line 79 fixes the target and parameter and defines `g:=nabla J(theta)` before the expansion. This is the same target used by Appendix C. | Closed |
| I3 | `appendix/C_estimator.tex:5` uses `x=(v^+,v^-,G_x)` directly. Its score coordinate is no longer denoted by lower-case `g_x`; `g` remains the target gradient. | Closed |
| I4 | Assumption `minimax-oracle`, minimax risk, and Appendix E use `T_call`. Appendix E:90 explicitly identifies its stopping-time role; the adaptive information calculation uses the same expected count. | Closed |
| I5 | Setup line 5 explicitly states `t_{k+1}=t_k+h`. The execution algorithm's consecutive episodes now have an explicit calendar interface. | Closed |
| I6 | Setup lines 83–94 distinguish a frozen model `P` from its policy-induced law `P_theta` and explicitly hold the true and simulator models fixed during differentiation. Appendix A's likelihood argument now has the stated intended interpretation. | Closed |
| I7 | Assumption `trajectory`, line 16, explicitly applies its uniform bounds and parameter independence to every analyzed true model and frozen simulator, conditional on start information/context. This is an explicit strengthening/clarification of the admissible simulator class, not a theorem that arbitrary simulators satisfy it. Every downstream estimator application is conditional on that class. | Closed |
| I8 | The unused `B_J` sentence is deleted; the online theorem still uses `2B_v`. Appendix A proves `|J_k|<=B_v`, and the telescope uses its two endpoints, so the retained constant is correct. | Closed |
| I9 | Appendix A has one 16-row semantic-chain table, including `z_k,F_k,r_k`. Multilevel parameters/constants remain defined locally. The final state row explicitly retains `m_u` as market state after the restoration requested during this review. | Closed |
| I10 | The setup explicitly distinguishes projected residual `Q_k`, squared projected residual `zeta_k`, average expected squared projected residual `E_K`, and dynamic local regret. Main result titles and affected E3 prose/captions agree with these aggregation levels. Remaining uses of “tracking” describe interpretation. | Closed |

Assumption delta: I7 explicitly restricts simulators to the same uniform regularity/bounded-domain class and freezes them as theta varies. I5 explicitly restricts the calendar relation for consecutive episodes. I6 disambiguates the existing intended likelihood model. No optimization bound, estimator target, exponent, numerical result, or optional error-bound hypothesis was silently changed.

## Dependency graph and hypothesis ledger

The verified dependency order is:

1. State/reference/action maps and parameter-independent frozen models define a theta-independent trajectory functional under a theta-dependent policy law.
2. Compact parameters, bounded features, finite horizon/assets, and augmented latent actions give score moments and likelihood differentiation.
3. Those likelihood identities and bounded regularized weight derivatives give smoothness and the CPT score-gradient identity.
4. The gradient identity, iid conditional trajectories, Taylor expansion, and Hoeffding decomposition give cancellation and centered-base variance.
5. Centering and the remainder bound give the finite-sample mean limit; independent randomized correction gives exact unbiasedness, finite variance/cost, and iid CLT.
6. Smoothness and projection give the one-step inequality. Conditional simulator unbiasedness plus the simulator error bound give true-gradient MSE. Those facts plus the moving-objective telescope give the online bound.
7. Jensen/reindexing give DLR; the optional error bound gives stationary-set distance and drift; the stated Cesaro limits give vanishing averages.
8. The separate fixed-target oracle experiment gives a realizable two-point lower bound and a cost-matched estimator upper bound.

There is no dependency from the online theorem back to its estimator or from the minimax lower bound back to the minimax theorem. Forward references to Appendix A are presentation order, not logical cycles.

| Result / application | Hypotheses actually used and discharge | Checked conclusion |
|---|---|---|
| `prop:path-dependence` | Explicit branch inequalities select the two reference-update branches; `eta_+,eta_-<1`; strictly increasing signed regularized deterministic value. | The reference subtraction is exact; nonzero terminal reference difference implies different deterministic CPT values. |
| `prop:reference-sensitivity` | Uniform finite outcome bound and bounded weight derivative; optional same-wealth-path coupling; fixed horizon. The common-path property is assumed, not inferred for arbitrary reference-sensitive maps. | Piecewise contraction, value Lipschitz bound, survival coupling, and triangle split prove exactly the displayed reference-only bound. |
| `lem:dirichlet-moments` | Compact Theta plus uniformly bounded features bounds concentrations away from zero; finite N,h; latent proposals retained; true and simulator kernels parameter independent by repaired assumption. | All needed positive-order score moments and an integrable Hessian bound; first and second likelihood derivatives. |
| `lem:cpt-smoothness` | Previous lemma on bounded indicators, C1/C2 weight bounds, finite integration interval, convex Theta. | Uniform `B_g,L`; Hessian bound implies the stated Lipschitz gradient; both CPT integrals lie in `[0,B_v]`. |
| `lem:residual-stationarity` | Global setup convention `theta in Theta`; Theta nonempty/closed/convex; positive gamma and floor. | Projection variational inequality and positive rescaling give the equivalence. No out-of-domain theta counterexample is relevant to its manuscript application. |
| `thm:gradient-identity` | Parameter-independent trajectory functional; likelihood differentiation; bounded weight derivative; finite score first moment. | Differentiation and Fubini are justified; mean-zero score permits deterministic survival centering, for true and simulator objectives. |
| `lem:level-cancellation` | Conditional iid summaries; finite score second moment; C3 weights; all sample/half sizes >=2. | Ordered-kernel projections are correct; degenerate variance and Taylor remainder are O(n^-2); exact influence cancellation gives the factor 9. Efron–Stein proves centered-base O(n^-1) covariance. |
| `lem:centering-expectation` | Same standing assumptions and fixed-target convention; leave-one-out independence; mean-zero score; previous remainder expansion. | Both means equal mu_n and `||mu_n-g||<=sqrt(C_rem)/n`. |
| `thm:estimator` | Same fixed-target hypotheses; independent base/correction; positive activation; `1<b<2`; untruncated levels. | Absolute L1 summability makes telescoping legitimate. Correct Xi/Upsilon sums give finite variance and expected trajectory cost, then MSE/M. |
| `thm:clt` | Fixed target, fixed n/varrho/b, iid complete outputs, finite covariance. Studentization separately assumes nonsingularity. | Scalar CLT plus Cramer–Wold; LLN for products requires only finite second moments. Fixed dimension turns entrywise consistency into operator-norm consistency; continuous inverse square root and Slutsky apply with probability tending to one. |
| `lem:projected-ascent` | Feasible x, nonempty closed convex Theta, L-smoothness, finite gradient estimate, `gamma L<=vartheta`, bounded true gradient. | Projection inequality, Young's inequality, displacement comparison, and normalized-map Lipschitzness yield the exact 1/8 and 5/4 constants. |
| `thm:online-residual` | All preceding uniform bounds; fixed/frozen F_k-measurable target/context; independent conditional outputs; declared M_k schedule; simulator error; finite expected objective variation. | Conditional cross term is zero. The pathwise moving-objective identity and endpoint bound prove the displayed coefficients 8 and 10. The algorithm's iterates stay feasible. |
| `cor:dynamic-local-regret` | Nonnegative normalized window weights, zero prehistory. | Jensen followed by summing each residual's coefficient, which is <=1, proves the pathwise bound and its expected version. |
| `cor:stationary-tracking` | Optional error bound at every actual iterate, with uniform kappa. | Squaring and averaging proves the claim. Geometry is an assumption, not a consequence of smoothness. |
| `lem:stationary-set-drift` | Same uniform error bound, common Theta/gamma/floor, nonempty compact stationary sets. | Residual perturbation <=gradient perturbation/floor; both directed set distances give the Hausdorff bound. |
| `cor:vanishing-tracking` | Fixed estimator design and gamma/floor; episode-uniform constants; stated o(K) variation and Cesaro simulation/replication conditions. | All terms in the expected-average bound vanish. No almost-sure convergence or convergence to one stationary point is claimed. |
| `thm:minimax` | Fixed dimension, common model constants, interior theta0, explicit submodel inclusion, positive weight derivative at zero, exact fresh complete-trajectory oracle, expected budget and no extra observations. | The Bernoulli model is realizable; derivative separation, stopped transcript KL, testing reduction, and fixed-design estimator upper bound target the same gradient and budget. |

The centering and estimator statements rely on the section's *standing* statistical assumptions and Appendix A's fixed-target convention, even though they do not repeat every assumption label in each statement. This is an explicit reading convention of the manuscript, not a claim that those results hold for arbitrary data or arbitrary simulators. Likewise, the displayed online MSE bound is read for the declared replication schedule; an arbitrary random, history-dependent `M_k` would require an expectation around its inverse in the unconditional bound.

## Analytic, inequality, mode, and constant checks

- **Likelihood domination.** On a compact parameter neighborhood, the action density and its first two derivatives are bounded by a constant times `prod omega_i^(a_- -1) (1+sum |log omega_i|^2)`. Positive `a_-`, finite N,h, bounded features, and parameter-independent probability kernels make the finite product integrable. This is an actual common envelope, stronger than a bare uniform moment assertion. It also handles atoms or singular market kernels because differentiation acts on latent action densities.
- **CPT integrals.** Indicators are bounded; score moments and C1/C2 bounds dominate derivatives uniformly in the threshold over a finite interval. Absolute integrability justifies Fubini. Atoms and threshold ties do not require differentiating the indicators pathwise.
- **Reference coupling.** The nonnegative indicator-difference integral allows Tonelli. The contraction remains valid when references straddle the branch point; the two affine pieces agree there. Eta_min=0 gives a nonexpansive bound, not spurious strict contraction.
- **LOO expansion.** The ordered kernel has one zero projection and a generally nonzero second projection h1; subtracting h1 makes it degenerate in both arguments. Covariances sharing only one index vanish. Reversed ordered pairs are counted. The Bernoulli fourth moment is O((n-1)^-2); n>=2 converts this uniformly to O(n^-2). G_i is independent of the empirical leave-one-out survival, not of its own indicator; the latter is only bounded by one in the estimate.
- **Telescoping.** The sum of level L1 norms is finite, so expectation and randomized level sum can be interchanged. The bias limit is a norm limit of deterministic means; it is not an unproved almost-sure limit. Xi is finite only below b=2; expected cost is finite only above b=1. The independence assumption removes the base/correction covariance.
- **Projected ascent.** The comparison point x is feasible. The inequality `uv<=u^2/(4 gamma)+gamma v^2` has the correct direction. Projection nonexpansiveness compares actual and population displacements. The normalization is the unit-ball projection of g/floor, yielding the claimed Lipschitz constant even at zero and at the threshold.
- **Moving target.** The telescope is pathwise, so endogenous next-episode objectives need not be independent of the update noise. Conditional unbiasedness is used only for the true/simulator gradient cross term. All subsequent conclusions are expectation or pathwise statements as labeled.
- **Minimax interchange.** The t-integral derivative is dominated by bounded m(t) times an integrable `t^a |log t|` envelope on a compact positive-a interval. The p derivative is dominated by bounded C1/C2 factors. The stopped log-likelihood increments are bounded on the fixed interior Bernoulli interval; summing their absolute values has expectation <=a constant times E T_call. Thus the stopping-time calculation is justified without an optional-stopping leap.
- **Minimax constants.** The gradient separation lower bound is `2 c_* delta`; Pinsker at KL<=1/8 gives TV<=1/4; the resulting lower risk constant is `3 c_*^2 delta_*^2/(8 B)`. The upper construction uses `M=floor(B/C_cost)` and, for B>=2 C_cost, `M>=B/(2 C_cost)`. No deterministic hard budget is substituted for the authorized expected budget.

Uniformity: C_rem, C_U and C_bias are uniform over targets satisfying the fixed common score/weight/outcome bounds, and hence over the repaired true/simulator class. They may depend on the compact parameter set, d, h, N, feature bound, B_v, and fixed regularization. C_var further depends on n,varrho,b. None of these constants is claimed uniform as regularization vanishes, b approaches an endpoint, activation vanishes, dimension grows, or the parameter domain expands. K limits hold these constants and the step/floor fixed; M limits fix target/design; the minimax B limit fixes the dimension and class constants. The O(K^-1/2) replication calculation and O(K^1/4) deterministic power sum are correct.

## Boundary and counterexample attempts

| Attempt | Result |
|---|---|
| One dimension, optimum on the boundary, or zero true gradient | The projection characterization and ascent proof survive because the comparison point is feasible. A raw-gradient stationarity criterion is never substituted. |
| Singular estimator covariance / score identically zero | The unstudentized CLT remains possibly degenerate; nonsingularity is explicitly required only for studentization. |
| n=2, first correction level | Half sizes remain at least two; the `n/(n-1)` comparisons remain bounded. |
| Weight linearity, C2=0 | Taylor/projection terms simplify; minimax explicitly treats C2=0 when selecting p_max. |
| Atoms and deterministic outcomes | Survival-score differentiation remains valid under likelihood domination. The deterministic path-dependence proof uses endpoint-normalized weights correctly. |
| b=1 or b=2; activation approaching zero | Divergence can occur at the endpoints, which are excluded. No uniform endpoint efficiency is claimed. |
| Identical stationary sets, zero gradient drift | Both directed Hausdorff bounds vanish as required. |
| Adaptive queries and unbounded stopping count | The expected-budget KL proof still works because the per-call Bernoulli information does not depend on the chosen policy and the stopping count is integrable. |
| Small state motion with a discontinuous state-to-objective map | Breaks the proposed inference from state increments alone to objective variation; this is the preexisting E3 bridge issue below, not a counterexample to the theorem that assumes objective variation. |
| Unbiased gradient estimates passed through normalization | Does not generally give unbiased projected residual estimates; an explicit algebraic example is given below. |

No verified counterexample to the core estimator, online, or minimax theorem under its stated/conventional hypotheses was found.

## Proof-orchestrator notation scorecard

**Census scope:** persistent semantic interface families in setup/main results/Appendix A, with their uses checked throughout all reviewed source. This is a declared 41-family census, not a claim of exhaustive counting of every proof-local scalar. Standard operators, bound variables, summation indices, generic dummy coordinates, and unnamed bound constants are excluded. The 41 families are:

`s, m, X, r, omega, tilde-omega, z, F, theta, Theta, pi, P, tilde-P, P_theta, Y, v_+/v_-, w_+^eps/w_-^eps, J, tilde-J, g, tilde-g, G, hat-g, M, Q, gamma, vartheta, S_stationary, zeta, E_K, R_DLR, rho, w_window, beta, V_variation, T_call, B_budget, Z_db, S_survival, W_window, bar-Q`.

Aliases with suppressed fixed context are grouped into their semantic family for definition-payoff counting. Glyph collisions are nevertheless counted under the skill's strict same-base-glyph rule; distinct subscripts do not excuse changed meaning. The activity metric examines the largest touched theorem step, including design/bound symbols necessary to read it.

```text
Core semantic objects retained: 13/13 (100%)
Undefined symbols: 0
Symbol collisions: 6
One-use definitions: 1/41 (2.4%)
Maximum parallel representations of one object: 4
Maximum alias-chain depth: 2
Maximum active nonstandard symbols in one proof step: 15
```

The thirteen retained semantic groups are state/reference, conditioning/context, parameter/policy, true/simulator models and induced law, outcome/value maps, objectives, gradients, trajectory score, update estimator, projected residual, stationary set, residual/DLR aggregation, and simulator/variation bridges. The minimax budget remains explicit as well.

The six collision families, all preexisting rather than introduced by I1–I10, are:

1. `w`: probability-weight maps versus window length.
2. `r`: reference wealth versus Appendix C's Taylor remainder `r_sigma(e)`.
3. `s`: state `s_u`, displacement `s=x^+-x`, and experimental shock-scale `s_k`.
4. `v`: value functions versus realized scalar values and non-reference variation `v_k^nr`.
5. `tilde-omega`: latent proposal versus raw terminal executed holdings in E3's episode transition.
6. calligraphic `R`: DLR total versus E4's ratio of two DLR totals.

These are primarily notation-policy blockers, not evidence that the associated calculations are numerically wrong. Renaming them would be outside the authorized original repairs. Extra families outside the 41-family census include the preexisting `alpha` preference-exponent/state-gain reuse and `b` level-exponent/factor-loading reuse. The empirical formula's `mu_k,b_k,f_{i,u},s_k` also lacks explicit mathematical definitions/domains in the inspected source; these four items are outside the declared core census and are not hidden by the zero core-undefined count.

The one-use interface definition is the explicitly introduced induced-law notation `P_theta`, directly used in the following survival formula; retaining it is justified by I6's essential model/law distinction. The four score representations are the full functional `G_{k,theta}`, frozen shorthand `G_theta`, sampled score `G_i`, and realization coordinate `G_x`. Their roles are mathematically intelligible, and some are local sampled evaluations rather than independent persistent aliases; under the conservative lexical score they exceed the skill's threshold. The two-layer objective alias chain is generic `J(P,z,r;theta)` to `J_k` to locally suppressed `J`; its context restriction is explicit. The maximum active-load count is the online theorem's `E_K,B_v,V_K,vartheta,gamma,C_g,C_var,n,M_k,beta_k,hat-g_k,g_k,tilde-g_k,theta_k,L` across its statement and displayed bound. This is a readability warning; removing necessary assumptions/constants merely to lower it would be inappropriate.

The three originally undefined shorthands and the specific `g_x` collision are repaired. The strict full-artifact zero-collision condition is not met. No global `READY_FOR_USER` label is warranted.

**Top-down derivation structure: PASS.** The substantive reference, score-gradient, Hoeffding/cancellation, estimator, projected-ascent, and minimax proofs identify their target, establish the necessary intermediate claims in dependency order, and explicitly return to their displayed conclusion. The online argument obtains the target MSE first, then the sufficient one-step/telescope bounds, and recombines them. Short Jensen, squaring, and projection-equivalence arguments do not require additional scaffolding. This structural pass is separate from the global notation failure.

## Preexisting issues outside the authorized repair scope

### O1 — E3 state-step budget is not an established objective-variation budget

- Status: **UNJUSTIFIED**, impact LOCAL to empirical theorem alignment, category HIDDEN_ASSUMPTION / SCOPE_OVERCLAIM; MAJOR.
- Location: `sections/07_experiments.tex:103–120`, and the interpretation at lines 137–139.
- The power sum and the damped state-step description do not prove `E V_K=o(K)`. The theorem assumes variation of the entire frozen objective over Theta. An additional uniform continuity/Lipschitz bridge from start context to objective, bounded relevant state domain/log-wealth increments, treatment of factor-context changes, and the simulator-error Cesaro condition would be needed to verify the theorem for that construction.
- Counterexample to the *generic implication*: a bounded objective can switch between two different functions when a state crosses a discontinuity, while the state moves by arbitrarily small alternating increments. Thus an o(K) state-step budget alone does not control objective variation. This is not asserted to be the actual experimental implementation.
- The cross-episode damping is also a distinct transition from directly carrying raw terminal states in the displayed execution algorithm; its relationship to the episode model should be made explicit in a future experiment-interface revision.
- Minimal future fix: prove/check the missing bridge for the implemented model, or explicitly describe E3 as a finite-horizon diagnostic without treating state-step control as hypothesis verification. Do not modify within this patch.

### O2 — Independent evaluation streams remove self-square variance, but do not by themselves identify the population projected residual

- Status: **UNJUSTIFIED**, impact LOCAL, category NORMALIZATION_MISMATCH / MISSING_DERIVATION; MAJOR.
- Location: `sections/07_experiments.tex:126`.
- For independent identically distributed residual estimates, their cross product estimates the squared norm of their mean. If each is obtained by putting an unbiased gradient estimate through normalization and projection, that mean need not equal Q(g). The prose correctly removes the ordinary self-square noise term but supplies no bridge for the remaining nonlinear bias.
- Algebraically verified example: d=1, theta=0, gamma=floor=1, Theta=[-2,2], true g=1/2, and independent gradient estimates equal to -1/2 or 3/2 with equal probabilities. They are unbiased. Their projected residuals equal -1/2 or 1, so the expected cross product is `(1/4)^2=1/16`, whereas the true squared projected residual is `(1/2)^2=1/4`.
- This example refutes an automatic unbiasedness inference, not the manuscript's estimator theorem or the actual code, which was not inspected here. A consistency/error analysis or a description of a separate population-residual evaluation method would close the empirical bridge.

### O3 — Strict whole-artifact notation and experiment reproducibility remain incomplete

- Status UNCLEAR, impact COSMETIC/LOCAL, category UNCLEAR / MODEL_INTERFACE; MINOR individually, blocking under the chosen notation gate.
- The collision and undefined experiment-parameter inventory is recorded above. The declared core census should not be advertised as a zero-undefined, zero-collision census of every symbol in the paper.
- The scalar diagnostic `L_diag=0.31` in E4 is explicitly named and numerically given, but the inspected manuscript does not establish it as an upper bound on the theorem's uniform smoothness constant L. Thus the E4 grid filter is a diagnostic rule unless independently certified; it does not by itself verify `gamma L<=vartheta`.
- Preserve these unrelated items in this authorized patch. Their eventual repair requires a separate full-notation/experiment-interface task.

### O4 — Scope of the completed empirical manuscript

E5–E7 and Discussion remain placeholders. The introduction's planned evidence/contribution language should not be read as evidence that those studies are completed. This is preexisting and has no bearing on the validity of the 17 mathematical derivations reviewed here. No external citation verification or experimental recomputation was performed.

## Final source fingerprint

All paths below are relative to the current paper root; SHA-256 hashes identify the final source reviewed, including the restored market-state wording.

| File | SHA-256 |
|---|---|
| main.tex | 18cdee67f089373a87a2d01e8fe11248dc5387bac947aadfd786e47a926bf3f9 |
| sections/01_introduction.tex | 27c9952fec31136ade20b97adfbebc4fa0cb6264f6dccdfcb38c7d639601c205 |
| sections/02_related_work.tex | a105de27b5c41b547d284ba17df9d3e488581f35797723801d31ae16b544d4ea |
| sections/03_problem_setup.tex | 6b0c61aa562ea1454fddc30be2e36b1c7d4a7d553b3cbb7ad7a8b9bb9f9a0e89 |
| sections/04_assumptions.tex | 349f9ba305e7c3d09f1918fc722cb398e3165573f77688e61d415c1a56a76dfd |
| sections/05_algorithm.tex | 9a8d1b74132034cdc479da8fcc8d04718d5e7a3c197a7bffa7ada946960c21dc |
| sections/06_main_results.tex | 62b711c20c33354b61171e0f6d3c266424d0324ce174665732aeae8df1375702 |
| sections/07_experiments.tex | b3cde21a0361ab53acd9c5ed1b702839791fdad5e9ee279be92209d862ed2c47 |
| appendix/A_notation.tex | 96f841d9c69927bce82efa0ce62a4a32b7baf8483873e81fe3c1ad48c0da19ca |
| appendix/B_reference_and_gradient.tex | 38265975a3c2dbc279d70a06f97e8daf9bfce83795f99ac3e4d36a5dbb00f111 |
| appendix/C_estimator.tex | 647727ff19eeaf7369026b08810f258c77359e275baa3068a8b907cda019189e |
| appendix/D_online.tex | ca338671e464de4658e3185bd6488a37c56a68a0359a037415a31242a5466853 |
| appendix/E_minimax.tex | 884600dd85466ad9394d6f82a08d3915148d3d153d36392ce11ab75041f181e1 |

Proof-checker acceptance accounting: no open FATAL/CRITICAL issue was found in the authorized repairs or core conditional theorem chain; hypotheses, interchanges, constants, modes, and boundary attempts were checked above. Whole-manuscript acceptance remains withheld because the preexisting empirical bridges and strict notation gate are unresolved. This is same-family provisional review evidence only.
