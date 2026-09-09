# DRCPT-PG Mathematical Audit

## Verdict

**FAIL in the current form.** The modeling chain is coherent and several local arguments are repairable. The headline dynamic-local-regret theorem, its downstream corollaries, and the minimax-optimality claim are not established by the present assumptions and proofs.

The original `main.tex` remains unchanged. This report distinguishes validated parts from proposed claims.

## Critical Findings

### P01. The projected-ascent lemma is false as stated

Source: `main.tex:772-787`.

Take `Theta=[0,1]`, `J(theta)=theta`, `theta=1`, `d=1`, and a small positive `gamma`. Projection returns `theta_next=1`, so the objective does not increase. The claimed right side contains the positive first-order term `gamma <grad J,d>` and only a quadratic loss. For sufficiently small `gamma`, it is strictly larger than `J(theta)`. Compactness and nonexpansiveness do not make the projection loss second order at an outward-pointing boundary direction.

This invalidates the one-step inequality used at `main.tex:791-816`, so Theorem 4.7 is not proved.

Repair options:

1. Remove parameter projection and assume the iterates remain in an open bounded analysis region. Then the ordinary smooth-ascent inequality applies directly.
2. Keep projection and formulate the result with the projected gradient mapping and a feasible update direction satisfying the correct variational inequality.

### P02. The Dirichlet policy does not satisfy the samplewise bounded-score assumption

Source: `main.tex:240-246` and `main.tex:297-311`.

Dirichlet draws can approach the simplex boundary. Its log-density score contains `log omega_i`, which is unbounded as `omega_i` approaches zero. Bounding `theta` and the features bounds the concentration parameters, yet it does not produce a pathwise bound on every sampled score.

Repair: replace the almost-sure score bound with a uniform `2+xi` moment condition, keep concentrations uniformly away from zero, and propagate the resulting moment constants through the estimator and regret analysis. An interior-truncated action law is another route, but it changes the policy.

### P03. The minimax lower-bound subclass is not shown to be CPT-realizable

Source: `main.tex:503-519` and `main.tex:850-872`.

The proof builds arbitrary product distributions for a vector `Z`. The class was described as trajectory laws whose debiased replication has that distribution, but no portfolio MDP, CPT objective, policy, or multilevel construction is provided that realizes the hypercube family. Assouad's calculation supports a generic bounded-mean estimation problem. It does not yet support a CPT gradient lower bound.

Repair: construct an explicit finite-horizon CPT policy submodel whose gradient replications realize the testing family, or narrow the theorem to estimating the mean of abstract debiased replications.

### P04. The multilevel estimator is under-specified

Source: `main.tex:266-274`, `main.tex:351-363`, and `main.tex:682-725`.

The symbols `T_{k,l}`, cross-fitting, the fixed score replication, and the nested full/half-sample coupling are not defined at an executable mathematical level. The crucial level-difference variance rate is assumed. If adjacent levels use independent score trajectories, that variance need not decay.

Repair: define every sample index, fold, shared random variable, half-sample coupling, and level statistic. Prove the decay rate under explicit smoothness and moments, including the base level's finite second moment.

### P05. Dynamic local regret does not control each raw gradient

Source: `main.tex:224-234` and `main.tex:494-500`.

The discounted average can be small through cancellation. With a two-period equal-weight window and alternating gradients `1,-1,1,-1`, the smoothed gradient is zero after initialization while every raw squared gradient equals one. The proof correctly establishes the one-way Jensen bound from raw gradients to smoothed gradients. It later uses the dynamic-regret theorem as though it controlled the raw-gradient average needed by the stationary-set error bound.

Repair: state dynamic local regret as the primary smoothed diagnostic. Add a separate theorem for the raw squared-gradient average, then use that theorem for distance to `S_k`. Do not derive raw-gradient control by reversing Jensen's inequality.

## Major Findings

### P06. The plug-in bias proof needs stronger regularity

Source: `main.tex:249-252` and `main.tex:682-689`.

The displayed second-order remainder for `dot w` requires a Lipschitz second derivative of `w`, or an equivalent bounded third derivative/local modulus condition. Twice continuous differentiability with bounded first and second derivatives is insufficient for the stated quadratic remainder uniformly in `z`.

### P07. One drift symbol covers three different errors

Source: `main.tex:254-264`, `main.tex:407-410`, and `main.tex:816`.

`delta_k` is used for adjacent-episode market drift, finite-horizon simulation mismatch, and within-episode law drift. These quantities have different conditioning and enter bounds differently.

Repair: define separate `delta_market,k`, `delta_sim,k`, and `delta_within,k` budgets.

### P08. Constants and step-size scaling are hidden

Source: `main.tex:468-480` and `main.tex:789-827`.

Dividing the one-step estimate by `gamma` introduces step-size dependence into estimator and drift terms unless constants explicitly absorb fixed `gamma`. The theorem should display this dependence and declare how constants depend on `h,w,rho,d,epsilon,a0`.

### P09. The static limit needs a fixed episode-start law

Source: `main.tex:486-492` and `main.tex:830-837`.

A stationary market and fixed reference do not alone imply `J_k=J`. Episode-start wealth and previous portfolio remain state variables and can change. The corollary needs identical start-state distributions, a continuing-state objective, or a reset convention.

### P10. Endpoint regularization changes normalization

Source: `main.tex:194-210`.

The map `w_epsilon(p)=w(epsilon+(1-2epsilon)p)` gives `w_epsilon(0)=w(epsilon)>0` and `w_epsilon(1)=w(1-epsilon)<1`. This introduces an artificial baseline in the CPT integrals and may make the objective depend on the chosen integration ceiling.

Repair: use the normalized transform

`[w(epsilon+(1-2epsilon)p)-w(epsilon)]/[w(1-epsilon)-w(epsilon)]`

or state and analyze the shifted objective explicitly.

### P11. Decreasing-step convergence is asserted without a theorem

Source: `main.tex:491` and `main.tex:836`.

The phrase “usual stochastic-approximation conditions” does not settle moving objectives, projection, normalized directions, estimator moments, or almost-sure versus expected convergence. A separate theorem is required.

### P12. Variance and minimax statements use inconsistent budgets

Source: `main.tex:426-443` and `main.tex:503-519`.

The variance bound counts `M` debiased replications, each with random computational cost. The minimax theorem counts `M` abstract observations. Rate optimality under a common trajectory or computation budget is not demonstrated, and dimension dependence is hidden in `C_db`.

## Valid or Conditionally Valid Components

- The two-path reference update algebra is correct. Lean verifies the displayed difference exactly.
- The reference mechanism generates path dependence whenever terminal references differ and the smoothed value mapping is strictly monotone on the relevant outcomes.
- The CPT score-gradient identity has the correct structural form under differentiability in quadratic mean, fixed episode-start information, and sufficient domination or moment assumptions.
- Randomized telescoping gives an unbiased limit estimator once the coupled level statistics, base integrability, absolute summability, and independence are explicitly established.
- Averaging independent finite-variance replications yields the stated `1/M` variance reduction.
- The conditional multivariate CLT is standard for fixed dimension under conditional iid sampling, nonsingular covariance, and a `2+xi` moment.
- The stationary-set drift lemma follows from uniform gradient drift and the error-bound condition, with a symmetric adjacent-episode drift statement.
- The pointwise error-bound implication `dist^2 <= kappa^2 ||grad J||^2` is correct and Lean verified in a scalar abstraction.

## Plain-Language Walkthrough

### Problem setup

At the start of each episode, the system knows current market factors, current wealth, the investor's reference wealth, the previous allocation, and the current factor coefficients. It chooses a long-only allocation over cash and stocks. The next market return and trading cost change wealth. The observed wealth then adjusts the reference point at different gain-side and loss-side speeds.

### Objective

The algorithm evaluates the full distribution of terminal wealth relative to the path-dependent terminal reference. Positive and negative outcomes pass through different value functions. Their probabilities pass through nonlinear weighting functions. The result is one episode-specific CPT objective `J_k`. Market changes and reference updates make this objective move over episodes.

### Policy and learning

The parameter `theta` says how strongly each factor matters. Factor scores become positive Dirichlet concentrations. The real execution path uses the Dirichlet mean for a stable portfolio. Simulated learning paths use Dirichlet samples so their log-probability scores reveal how changing `theta` changes the trajectory distribution.

### Estimator

The CPT multiplier assigns each simulated trajectory a behavioral marginal value based on its location in the gain and loss distributions. Multiplying this value by the policy score gives a policy-gradient contribution. The draft proposes cross-fitting and randomized multilevel debiasing because empirical ranks introduce nonlinear plug-in bias. The general debiasing idea is sound, while the exact coupled estimator and its rate assumptions still need a full definition and proof.

### Main conclusion

The intended theorem says the average discounted recent-gradient magnitude is bounded by optimization error, Monte Carlo variance, market drift, reference drift, and truncation bias. This is the right conceptual decomposition for a moving objective. The current proof cannot establish it because the projected-ascent step is false. A corrected update analysis is required before this theorem can serve as the thesis foundation.

### What the theorem would mean after repair

It would show that the algorithm stays near locally stationary behavior when the market and reference move slowly enough and the gradient estimator is accurate enough. It would not promise global CPT maximization, monotone objective values, or maximum wealth.

## Recommended Revision Order

1. Decide between an unprojected bounded-domain analysis and a projected-gradient-mapping analysis.
2. Replace samplewise bounded Dirichlet score by explicit uniform moment assumptions.
3. Define the cross-fitted multilevel statistics and coupling completely.
4. Separate market, simulation, and within-episode drift.
5. Prove raw-gradient control separately from dynamic local regret.
6. Repair the regularization endpoints and specify the utility smoothing polynomial.
7. Restate the static limit with a fixed objective definition.
8. Recast or reconstruct the minimax lower bound under a common information budget.

## Verification Evidence

- Independent proof audit: reviewer `01a08679-a962-7621-b959-2370b2e80647`.
- Lean project: `formal/`.
- Command: `lake build`.
- Result: success, four jobs, no unresolved goals.
- Lean proves local algebra and counterexamples; it does not formalize measure theory or stochastic asymptotics in this audit.
