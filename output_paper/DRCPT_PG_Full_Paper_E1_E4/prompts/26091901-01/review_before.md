# Fresh adversarial review of the current ten-issue audit

review_independence: same-family  
acceptance_status: provisional  
review_stage: before authorized I1–I10 repairs  
reviewer_task: /root/proof_review_before  
date: 2026-09-19  
verdict: WARN — the core arguments are coherent on the stated bounded, fixed-target domain, but the simulator scope must be made explicit and the recorded notation/interface defects remain open. This is not a final semantic acceptance.

## Scope and provenance

I read the supplied `04-proof_checker.md` and `proof-audit-rubric.md`, the current root `PROOF_AUDIT(1).md` and `PROOF_SKELETON(2).md`, and all nine designated theorem-bearing files. I did not use the historical nested `paper/audit` material as evidence of correctness. I did not edit the manuscript or code. After the parent began implementing repairs, all remaining paper reads used the verbatim snapshot at `/workspace/scratch/7f23b07a728e/tmp/proof_baseline/paper/DRCPT_PG_Full_Paper_E1_E4/`. File locations below are relative to that paper root; line numbers refer to this before snapshot. The unchanged implementation was read under the working package's `code/final_e1_e4/` solely to assess the minimal I7 scope repair.

No new FATAL or CRITICAL defect was established in the designated proof chain. I7 is an assumption-scope gap with global downstream importance; the safe repair is a stated uniform true/simulator hypothesis, not an assertion that any fitted simulator automatically satisfies it. Other I1–I10 fixes can preserve every displayed mathematical formula except local definitions and symbols. The out-of-scope observations below must remain separate from the authorized issue ledger.

## Verification of I1–I10

| ID | Status / impact / severity | Location | Verification and minimum repair |
|---|---|---|---|
| I1 | UNCLEAR / COSMETIC / MINOR | `sections/06_main_results.tex:63–75` | Confirmed: `S_{k,theta}^±` has no definition and local `G_theta` is not identified. Define `S_theta^±(y):=S_{P_k,z_k,r_k,theta}^±(y)` and `G_theta:=G_{k,theta}` directly. No intermediate `S_{k,theta}` alias is needed. The simulator identity uses the corresponding induced simulator trajectory law. |
| I2 | UNCLEAR / COSMETIC / MINOR | `sections/06_main_results.tex:78–109` | Confirmed: `g` appears in the cancellation statement before the following centering statement defines it. Introduce the frozen target and `g:=∇J(theta)` before its first use, with target/context fixed. This is the same `g` used in Appendix C. |
| I3 | UNCLEAR / COSMETIC / MINOR | `appendix/C_estimator.tex:3–5` | Confirmed: `x=(v^+,v^-,g_x)` unnecessarily reuses the gradient glyph for a score component. Write `x=(v^+,v^-,G_x)` and retain `I_{x,sigma}`. This changes no random variable or proof. |
| I4 | UNCLEAR / COSMETIC / MINOR | `sections/04_assumptions.tex:71–75`; `appendix/E_minimax.tex:90–97` | Confirmed: `T_call` becomes `mathsf T_call` in the stopped-KL calculation. Use `T_call` consistently and identify it as an oracle-filtration stopping time. Pre-call measurability of `{T_call≥j}` is the precise property the KL argument uses. |
| I5 | UNCLEAR / COSMETIC / MINOR | `sections/03_problem_setup.tex:5`; Algorithm 2 return line | Confirmed: `t_{k+1}=t_k+h` is implicit in the terminal-to-next-start handoff. Add that relation for consecutive episodes. This does not authorize changing the separately reported E3 state-relaxation experiment. |
| I6 | UNCLEAR / GLOBAL interface / MAJOR clarity | `sections/03_problem_setup.tex:83–104` | Confirmed: `P` is called a trajectory law and immediately indexed by theta. Say that `P` is a frozen episode environment/model and `P_theta` is its induced augmented trajectory law under the policy, with context fixed. `P_k` and `tilde P_k` remain the true and simulator models. The environment's parameter independence and the induced trajectory law's parameter dependence must not be conflated. |
| I7 | UNDERSTATED / GLOBAL / MAJOR | `sections/04_assumptions.tex:5–17`; `appendix/B_reference_and_gradient.tex:94`; `appendix/D_online.tex:66–79` | Confirmed: the simulator identity and uniform conditional MSE invoke the trajectory regularity without explicit simulator quantification. State common conditional bounds and parameter independence uniformly for every analyzed true and frozen-simulator model. Implementation evidence and the necessary scope limitations are detailed below. |
| I8 | UNCLEAR / COSMETIC / MINOR | `sections/06_main_results.tex:195` | Confirmed: `B_J` is unused. Delete its sentence. The `2B_v` endpoint term is correct because each gain/loss integral belongs to `[0,B_v]`, so their difference belongs to `[-B_v,B_v]`. |
| I9 | UNCLEAR / COSMETIC / exposition issue, not a proof failure | `appendix/A_notation.tex:5–68` | Confirmed: the two tables mix persistent objects and local derivation internals and omit the frozen `(F_k,z_k,r_k)` bridge. One concise semantic-chain table is safe provided local definitions of `n,varrho,b,nu_l,D,Z,C_var` remain at their actual sites. Keep the true/simulator model distinction and the score/gradient distinction. The existing audit's MAJOR label describes presentation impact, not an invalid theorem. |
| I10 | UNCLEAR / COSMETIC / MINOR | Setup/results/Appendix A, D and relevant experiment wording | Confirmed: freeze the four levels as projected residual `Q_k`, squared projected residual `zeta_k`, average expected squared projected residual `E_K`, and dynamic local regret `Reg_{rho,w}`. Tracking can describe an interpretation or the separate stationary-set distance result. A name repair must not accidentally replace squared quantities by unsquared ones. |

No algebraic counterexample applies to I1–I6 or I8–I10: these are notation, model-interface, or presentation defects. I7 admits a verified failure construction if simulators are allowed outside the stated likelihood class, described next.

## I7: minimum safe repair and implementation evidence

### Explicit assumption recommended

The smallest mathematically sufficient change is to say, within Assumption `trajectory`, that conditional on the frozen episode-start information and context, its bounds and parameter-independence conditions hold with common finite deterministic constants for every analyzed true model `P_k` and every frozen simulator `tilde P_k`, uniformly over the queried `theta∈Theta`. Frozen model choice may depend on the prior history; when differentiating, that choice and its fitted parameters are held fixed. Every explicit theta dependence of the induced augmented law then comes from the same Dirichlet actor.

This is a scope clarification/strengthening, not a claim that an arbitrary learned simulator has been proved regular. A supplementary sentence may state that the score, smoothness, and estimator bounds derived below apply to both true and simulator induced trajectory laws under those common constants. No new persistent model symbol is needed. There is no mathematical need to change the algorithm or replace the standing bounded-domain hypothesis by simulator-specific moment assumptions for the implementations inspected.

### Source evidence

All code locations here are relative to the package's `code/final_e1_e4/`:

* `vendor/mvp_cpt_pg/paper_market.py:262–273`: factor features are clipped and scaled to `[-1,1]`; the remaining cycle feature uses `tanh`. `paper_experiments.py:84–89` replaces one coordinate by holdings in the simplex and forms a product of bounded features. Thus all actor feature components remain bounded by one, and `||q||₂≤sqrt(d)` for the supported dimensions.
* `vendor/mvp_cpt_pg/paper_experiments.py:92–101`: the actor concentrations are exactly `exp(features @ theta)`. The code computes a Dirichlet score from latent log weights before applying feasibility/inertia. `:136–143` applies a map depending on previous holdings, latent action, tradability, and a fixed trade fraction, without an explicit theta input. `:146–149` projects the update into the specified parameter ball.
* `common_slow_continuous.py:97–115`: empirical factor-block selection and the return kernel have no theta argument. Gaussian innovations pass through `0.08*tanh(...)`, so they do not produce unbounded returns. `:117–128` uses the same return law for execution.
* `common_slow_continuous.py:141–161`: a simulator draw freezes episode, market, wealth, reference, previous holdings, and configuration; it invokes the actor, the same bounded return kernel, the parameter-independent action map, and the same wealth/reference updates. `run_e3_slow_continuous.py:126–145` constructs the frozen law before drawing training and evaluation outputs.
* `common_semisynthetic.py:93–104` uses the same bounded-return form for the chronological variant. `vendor/mvp_cpt_pg/paper_experiments.py:171–206` shows that both semisynthetic and empirical historical simulator sampling occurs without an explicit theta-dependent environment kernel.

For a fixed bounded start context and finite `h`, let `r_bar=.08` and let `c_tc` be the transaction coefficient. Both executed and previous portfolios lie in the simplex. Since inertia yields `||omega_new−omega_old||₁≤2χ`,

`1−r_bar−2χ c_tc ≤ X_{u+1}/X_u ≤ 1+r_bar`.

At the default `χ=.2,c_tc=.001`, the lower bound is `.9196>0` and the upper bound is `1.08`. Even the general checked configuration `c_tc<.25,χ≤1` has positive lower bound greater than `.42`. Hence, over an episode, `0<X≤X_start(1.08)^h`. The reference update is a convex combination of its old value and new wealth, so `r≤max(r_start,X_start(1.08)^h)`. The regularized value maps are bounded on the resulting compact outcome interval. Compact theta and bounded features give concentrations in `[exp(−R_Theta sqrt(d)),exp(R_Theta sqrt(d))]`, hence the Appendix A envelopes and score moments.

This establishes compatibility of the finite, bounded-context implementation with the intended assumption. It does not prove fixed constants over an unlimited run: bounded per-day return multipliers permit wealth to grow, and the E3 log-state relaxation with gains `k^(−3/4)` does not impose a deterministic global wealth cap. The manuscript already says its `K` limits require episode-uniform constants (`appendix/A_notation.tex:3`). Keep that conditional asymptotic scope; do not claim that the simulator code itself proves uniformity in `K` or `E V_K=o(K)`.

### Verified counterexample if I7 is not imposed

Take `d=1`, `Theta=[−1,1]`, `h=N=1`, `X_start=r_start=1`, identity feasibility, `χ=1`, `eta_+=eta_-=0`, and zero features for both assets. The latent risky weight `U` is then Uniform(0,1), independent of theta, and the recorded policy score is identically zero. Let the true environment have zero return. It satisfies the stated true-law assumptions. Now let an unrestricted simulator use `R=a ξ`, where `a>0` is fixed and small and, **inside its market kernel**, `ξ~Bernoulli(1/2+theta/4)` independently of `U`. Use identity probability weights, permitted by the regularity assumption, and the prescribed increasing regularized gain value `v`.

All simulator outcomes remain bounded, but its CPT objective is `(1/2+theta/4) E[v(aU)]`. Therefore its actual gradient is `E[v(aU)]/4>0`, whereas every recorded-score LOO/debiased output equals zero. Thus the simulator gradient identity and simulator unbiasedness fail if parameter independence is only assumed for true models. This is an algebraically verified illustration of the missing scope, not a counterexample to the repaired hypothesis or to the inspected simulator implementation.

## Complete proof-obligation pass

| Result | Hypotheses actually used at applications | Review result |
|---|---|---|
| Dirichlet score moments / two likelihood derivatives | Finite `h,N`, compact theta neighborhood, bounded parameter-independent features, positive Dirichlet concentrations, theta-independent nonpolicy kernels, bounded theta-independent trajectory functional | Appendix A's common density/derivative envelope is integrable; iterated kernel integration is valid. Conditional true/simulator scope needs I7. |
| Uniform gradient/smoothness | Previous likelihood identities; bounded indicators; finite threshold interval; bounded first/second weight derivatives; convex theta | Gradient and Hessian constants `2B_v C1 sqrt(C_G2)` and `2B_v{C2 C_G2+C1(C_G2+C_H)}` are valid. Uniform Hessian bound implies Lipschitz gradient on line segments. |
| Projection/stationarity equivalence | Feasible theta, nonempty closed convex set, positive gamma and floor | Projection variational inequality gives both directions; scaling by the positive normalization denominator preserves the inequality. Feasibility is part of the surrounding definition. |
| Same terminal wealth proposition | Branch inequalities and strict monotonicity of signed regularized value | Direct expansions cancel the `r0` coefficient exactly; deterministic normalized CPT equals signed value. |
| Reference sensitivity | Same wealth paths under explicit reference-isolated coupling; value parameters fixed; bounded normalized weight derivative | Piecewise reference map is continuous and has slopes `1−eta_±`. The value derivative bound is conservative and correct. Coupled survival inequality and variation triangle inequality apply. |
| Score-gradient identity | Bounded-indicator likelihood identity; finite first score moment; fixed context and environment | Threshold derivative and Fubini steps are dominated. Mean-zero score follows by differentiating `E_theta[1]=1`. I1/I6/I7 are the only needed repairs in this chain. |
| Full/half cancellation | IID complete conditional trajectories; score second moment; bounded first three weight derivatives; `n≥2` | Hoeffding first-order terms are exactly batch-additive. Both projections of the ordered degenerate kernel vanish. Fourth moment of the Bernoulli empirical mean is `O((n−1)^(−2))`. Minkowski produces factor `9` exactly. |
| Centered base variance | Same IID law; bounded/Lipschitz LOO scalar coefficient; score second moment | Replacement bound and vector Efron–Stein yield `C_U/n`; no bounded-score assumption is used. |
| Centered/raw equal means and finite-n bias | Mean-zero independent held-out score; integrability; previous expansion | Centering has zero expectation; Jensen/Cauchy–Schwarz gives `sqrt(C_rem)/n`. No circular dependence on the final unbiased estimator. |
| Exact debiasing, variance, cost | Independent base/correction; independent activation/level; untruncated levels; `1<b<2`; preceding cancellation and bias | Absolute first-moment summability justifies telescoping; squared correction moments and expected cost are finite. Geometric closed forms and the `1/M` MSE are correct. |
| Fixed-target CLT / studentization | Fixed design and law; IID complete outputs; finite covariance; nonsingularity only for studentization; fixed dimension | Scalar CLTs plus Cramer–Wold, entrywise covariance LLN, and Slutsky suffice. Singular population covariance is allowed in the unstudentized statement. See minor finite-sample definition convention below. |
| One-step ascent | L-smooth objective on convex domain; feasible x; finite estimated gradient; bounded true gradient; `gamma L≤vartheta` | Every inequality direction and all coefficients in Appendix D check. The normalization map is exactly the unit-ball projection of `g/vartheta`; it is `1/vartheta`-Lipschitz. |
| Online average expected squared residual | Previous ascent; F_k-measurable target/context and parameter; conditional IID complete outputs; uniform conditional variance bound; simulator-gradient error; objective variation | Conditional mean-zero error cancels the cross term. The moving-objective telescope is pathwise. Factors `8` and `10` and the endpoint `2B_v` are correct. I7 is essential at the estimator application. |
| DLR control | Nonnegative window weights summing to one; zero-padding before episode 1 | Jensen and finite reindexing give the pathwise inequality in the claimed direction, including `K<w`. Expectation is taken only afterward. |
| Stationary-set distance | Explicit optional uniform residual error bound | Squaring is legitimate because both sides are nonnegative. Averaging and expectation give exactly the displayed conditional conclusion. |
| Stationary-set drift | Both episodes satisfy the same optional error bound; same gamma/floor/set; uniform gradient drift | Projection and normalization Lipschitz bounds give the stated constant; the two directed Hausdorff bounds are both proved. |
| Vanishing residual/DLR | Fixed estimator design, gamma/floor and uniform constants as K grows; sublinear expected variation; Cesaro sampling/bias condition | Every term vanishes under those conditions. The `sum k^(−1/2)≤2sqrt(K)−1` bound includes K=1. No high-probability or almost-sure convergence is claimed. |
| Fixed-d minimax lower bound | Class contains the explicit admissible one-stock submodel; fixed positive d_W and upsilon; interior p interval; adaptive oracle queries; expected stopping budget; no extra information | Gradient separation, uniform Bernoulli KL bound, bounded stopped log-likelihood increments, Pinsker, and testing reduction give the displayed `3c_*²delta_*²/(8B)`. Adaptive queries add no p information through the common action kernel. |
| Fixed-d minimax upper bound | Same oracle and expected budget; uniform estimator constants over the class; fixed estimator design | `M=floor(B/C_cost)` has expected cost ≤B and `M≥B/(2C_cost)` for large B. This matches the lower-bound experiment. |

### Inequality checks with constants

1. For the degenerate ordered U-statistic, only identical or reversed pairs survive the covariance sum. At most `2n(n−1)` pair interactions remain; a uniform kernel second moment yields `O(1/(n(n−1)))≤O(1/n²)` for every `n≥2`.
2. The Taylor term is bounded by `C3 e²/2`; Cauchy–Schwarz over the finite threshold interval and independence of `G_i` from its leave-one-out sample use only `E||G||²` and the Bernoulli fourth moment. Dependence across different remainder summands is handled by Jensen, not incorrectly discarded.
3. `||D_l||_{L2}≤sqrt(C_rem)/(n2^l)+2sqrt(C_rem)/(n2^l)=3sqrt(C_rem)/(n2^l)`. Squaring gives `9C_rem n^(−2)4^(−l)`.
4. With `nu_l=(1−2^(−b))2^(−b(l−1))`, the variance series is `1/[4(1−2^(−b))(1−2^(b−2))]`; the cost series is `2(1−2^(−b))/(1−2^(1−b))`.
5. Young's inequality in the ascent proof removes `c_g||e||||s||` by `c_g||s||²/(4gamma)+c_g gamma||e||²`. Using `gamma L≤c_g` leaves `c_g||s||²/(4gamma)`. The displacement comparison then gives `c_g gamma||Q||²/8−5c_g gamma||e||²/4`. Replacing `c_g` by the floor in the positive term and its upper bound in the negative term preserves the lower-bound direction.
6. Dividing the summed ascent inequality by `vartheta gamma K/8` produces `10 C_g/(vartheta³ K)` in front of squared gradient error; no gamma factor remains there.
7. The minimax first-coordinate derivative satisfies `g'(p)≥(d_W−2C2p)A_v`, including `C2=0`; the testing separation is `D≥2c_*delta`, and `D²(1−TV)/8≥3c_*²delta²/8` when `TV≤1/4`.

## Interchanges, probability modes, and uniformity

* Appendix A likelihood differentiation: a compact theta neighborhood supplies a single integrable Dirichlet product envelope times squared log terms. Feature bounds hold pathwise; the theta-independent environment kernels are integrated conditionally. This is stronger than merely having a uniform numerical moment bound.
* Appendix A/B differentiation of threshold integrals: bounded first/second survival derivatives and bounded weight derivatives dominate the integrands on a finite interval. Fubini for the score-gradient identity uses `2B_v C1 E||G||<∞`.
* Reference sensitivity: Tonelli is applied to a nonnegative indicator-difference integrand. Its pathwise integral equals `|V−V'|`.
* Appendix C kernel expectations and Taylor bounds: bounded indicators, finite threshold length, and finite score first/second moments provide absolute integrability. The randomized-level first-moment sum is finite, legitimizing expectation/sum interchange. Second moments and costs use nonnegative sums, hence Tonelli.
* Appendix E: near theta zero, concentrations are bounded away from zero, so `t^(a_-) |log t|` is an integrable derivative envelope. Differentiation in p is dominated by a bounded multiple of `m(t)(−t log t)`. In the stopped transcript, absolute log increments are bounded on the fixed interior Bernoulli interval; expected absolute stopped sum is bounded by a constant times `E T_call`, so Fubini/conditional expectation is valid.
* Fixed-target identities, variance bounds, and CLTs are conditional on the frozen target. Appendix D converts conditional MSE to unconditional expectation through the tower property; the simulator bias vector is frozen/measurable. The DLR inequality is pathwise. Online conclusions and stationary distance conclusions are in expectation, not almost surely or with high probability. No mode upgrade was found.
* Constants in the estimator expansion can be chosen uniformly from `B_v,C1,C2,C3` and a uniform score second moment. They are uniform over `n≥2`, levels, queried parameters, and analyzed episodes only if the standing primitive constants are uniform. They may depend on fixed horizon, asset count, parameter-domain radius, feature bound, and regularization. The explicit design factors expose divergence when `varrho→0`, `b↓1`, or `b↑2`. CLT fixes all design parameters. Minimax fixes dimension and model-class constants while B grows. K-asymptotics do not follow from bounds allowed to grow with K.

## Counterexample and edge-case pass

| Test | Outcome |
|---|---|
| Zero features / zero score | Score moments and all estimator identities reduce to zero without a singularity. Unstudentized CLT allows zero covariance; studentized theorem expressly excludes it. |
| Deterministic terminal value | Centered statistic vanishes; raw statistic is a fixed coefficient times mean score; coupled correction cancels that linear term exactly. No contradiction to zero target gradient. |
| Two-point outcomes / atoms at threshold | Indicators are bounded and likelihood differentiation acts on the law; no outcome density or pathwise differentiability is required. LOO Bernoulli fourth-moment bound remains valid. |
| Smallest base n=2 | Half batches at first correction have size 2. Factors `(n−1)^(−2)` remain bounded by `4/n²`; no denominator is zero. |
| b approaching endpoints | Expected correction cost diverges at b=1; second-moment upper series diverges at b=2. The open interval `(1,2)` correctly excludes both. |
| Activation probability approaching zero | Variance term has explicit `1/varrho`; theorem only fixes positive varrho. No unsupported uniformity at zero. |
| Boundary stationary point, Theta=[0,1], J(theta)=theta | At theta=1 the gradient is nonzero but projected residual is zero. The stated stationarity and ascent lemmas handle this correctly. |
| g=0 with arbitrary estimated gradient | Residual term is zero and the ascent lower bound permits a noise penalty. No division by zero because the normalization floor is positive. |
| Singleton convex Theta | Projection always returns the unique feasible point; residual, DLR, and stationary distance vanish. The minimax assumption separately requires an interior point and therefore does not claim this degenerate case. |
| rho=1, w=1, or K<w | Window mass remains positive and Jensen/reindexing still hold; for w=1 DLR is exactly the sum of squared residuals. |
| eta_+=eta_-=0 | Reference contraction factor is one, as expected. The path-dependence proposition's branch premises can become impossible rather than yielding a false conclusion. |
| eta_+ approaching one, chi approaching zero | Reference minimax signal constant can approach zero but both parameters are fixed and strictly within the specified range for the theorem. No parameter-uniform positive lower constant is claimed. |
| Affine probability weights (C2=0) | Taylor remainders disappear and the minimax p-interval choice explicitly handles C2=0. |
| Adaptive random oracle stopping | Finite expected calls and bounded interior-Bernoulli log increments suffice for the stopped KL sum. No bounded stopping time is secretly used. |
| Simulator with theta-dependent return law | Verified I7 failure construction above; excluded by the minimum repair and absent from the inspected implementation. |
| Unbounded K with bounded per-day returns | A bounded multiplier alone does not establish K-uniform wealth/value bounds. This is a limitation of empirical extrapolation, not a counterexample to a theorem that explicitly assumes uniform bounds. |

No numeric search was needed to certify these algebraic edge tests. Only the theta-dependent simulator construction is labeled a verified failure; the remaining tests either pass or establish scope limitations.

## Dependency graph and hypothesis delta

The actual acyclic dependency order is:

1. Policy/context/model and CPT definitions + trajectory/CPT assumptions.
2. Dirichlet moments and dominated likelihood identities.
3. CPT smoothness and score-gradient identity (both derive directly from step 2; neither depends on the estimator).
4. LOO Hoeffding expansion, remainder control, and Efron–Stein variance.
5. Equal means/bias → exact telescoping, finite variance, finite expected cost → fixed-target CLT.
6. Smoothness + projection geometry → one-step ascent.
7. Conditional simulator estimator bounds + simulator bias + one-step ascent + objective variation → online average expected squared projected residual.
8. Residual bound → DLR; with an additional error bound → stationary-set distance. Stationary-set drift follows directly from projection Lipschitzness plus the error bound.
9. Minimax lower bound uses the explicit Bernoulli submodel, not the online theorem. Its upper bound uses the fixed-target estimator from step 5 under the same oracle budget.

No semantic cycle or minimax-to-estimator dependency was found. I7 adds explicit simulator coverage to step 1 and thus discharges the simulator applications in steps 3, 5, and 7. It does not require changing any rate, inequality constant, probability mode, or geometric condition. I6 explains which object is held fixed at step 2. I1–I5 and I8–I10 affect names, definitions, or exposition only.

## Separate out-of-scope observations

These are not additional authorized manuscript edits. They should not be silently folded into I1–I10.

1. **OOS-1: finite-sample studentization convention.** MINOR; UNCLEAR / COSMETIC; `sections/06_main_results.tex:153–157`, `appendix/C_estimator.tex:235–238`. The empirical inverse square root is written without defining it on the event of singular empirical covariance. A nonsingular population covariance only makes that event vanish asymptotically; it need not make its probability zero for every M. The usual harmless convention is to set the statistic to zero, or any fixed value, on the singular event. The proof already works on the event of positive empirical eigenvalues and establishes its probability tends to one. This is a formal definition convention, not a failure of the claimed asymptotic normal law. Counterexample status: elementary finite-sample singularity is possible, e.g. M≤d; no counterexample to the asymptotic theorem.

2. **OOS-2: experimental bounded-domain extrapolation.** MAJOR if promoted to an unconditional asymptotic claim, otherwise a scope limitation; `sections/07_experiments.tex:103–120` and `common_slow_continuous.py:199–215`. The relaxed log-wealth recursion has increments bounded by a constant times alpha_k, but sum alpha_k diverges. A deterministic positive raw multiplier yields `log X_k=log X_0+c sum_{j<k}alpha_j`, which is unbounded. Thus bounded returns and diminishing gains alone do not prove a globally bounded state domain or absolute state increments `O(alpha_k)` with a K-independent coefficient. The existing theorem remains conditional on uniform primitive constants and sublinear objective variation. The I7 repair should avoid asserting experimental verification of those infinite-horizon assumptions. Counterexample status: verified for the claimed implication from bounded multiplicative raw transitions to global boundedness; not a counterexample to the conditional theorem.

3. **OOS-3: positive deterministic replication convention.** No established defect under the algorithm's ordinary reading: `{M_k}` is supplied as a schedule. If authors later allow data-dependent positive F_k-measurable replication counts, unconditional MSE and online displays need `E[C_var(n)/M_k]`, rather than a random expression on the right of an unconditional expectation. If counts depend on the outputs being averaged, IID averaging and centering also need a stopping analysis. This is a candidate extension issue only, not an authorized fix or a counterexample to the current prescribed schedule.

4. **OOS-4: E3 transition interface.** The experiment expressly adds an episode-boundary diminishing-gain state relaxation, while the formal Algorithm 2 propagates raw terminal wealth/reference/holdings. Adding the authorized consecutivity relation I5 does not by itself reconcile these two transition conventions. The online proof is stated via frozen objectives and variation budgets and can be applied under appropriate assumptions; a separate explanation of the experimental transition is needed for an exact algorithm-implementation claim. This is an existing exposition/interface question outside the designated theorem-bearing scope, not a reason to modify the experiment during these repairs.

## Input fingerprints

| Relative input | SHA-256 |
|---|---|
| sections/03_problem_setup.tex | 4b02611618e4ebb2b05145378256e3880ff2022e74223a6900fb43a7bc03922d |
| sections/04_assumptions.tex | 19a508ba0033ef6b37d0dee15ac95a7ed24f032609d3507ede2bd081cdd54d84 |
| sections/05_algorithm.tex | 9a8d1b74132034cdc479da8fcc8d04718d5e7a3c197a7bffa7ada946960c21dc |
| sections/06_main_results.tex | c091ce8c453175dcb8f4323c0d9246f1239b70fb13667f29633a3eeb7be61f87 |
| appendix/A_notation.tex | d18ce9b7c7277340af34dd2a9176f94811dd808c4727f8e78548087f4232aa92 |
| appendix/B_reference_and_gradient.tex | 38265975a3c2dbc279d70a06f97e8daf9bfce83795f99ac3e4d36a5dbb00f111 |
| appendix/C_estimator.tex | 975f7d6731669219a81e34433e9b77014402edde5ad044d390473b57099310b2 |
| appendix/D_online.tex | 58ff2268b44f2d47ccdf618884d32b0d7628765b717c12c783bb1d1fbcc71ea2 |
| appendix/E_minimax.tex | 78da97e2c6325367677171a569b2b45ba910de11d540199e4895aac4b418c41d |

The semantic review remains same-family and provisional. Final repair closure requires a fresh review of the actual changed files and deterministic compilation; neither is certified by this before report.
