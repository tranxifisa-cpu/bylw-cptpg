# Leave-one-out reuse plus independent randomized correction

## Verdict and source

The complete untruncated construction is unbiased for the fixed regularized CPT gradient under the assumptions below. The leave-one-out statistic alone is generally biased. These two statements refer to different random variables.

This review checks Section 18 (RC-1--RC-10) of the user's uploaded `background/previous_merged_proofs.md` against `vendor/mvp_cpt_pg/cpt_objective.py::pooled_gradient`. The original Chinese source section is included as `LOO_SOURCE_SECTION18.md`. The new experiment runner implements RC-3 directly because the earlier standalone `unbiased_estimator.py` was not present in the uploaded package. This is a formula-level implementation, not a claim of executing that unavailable old script.

## Fixed target

Condition on the episode-start information, simulator, policy parameter, preference parameters, starting wealth, reference and holdings. Independent complete trajectories yield

\[
 X_i=(V_i^+,V_i^-,G_i),\quad 0\le V_i^\sigma\le B_v,
 \quad E G_i=0,\quad E\|G_i\|^2\le M_2.
\]

The correct score is that of the latent Dirichlet proposals. The inertial action map is parameter-independent. The CPT score-gradient identity is assumed valid. Fixed endpoint-regularized TK weights have bounded first, second and third derivatives. Every expectation below is conditional on this fixed target.

Define

\[
 S_\sigma(z)=P(V^\sigma>z),\quad I_i^\sigma(z)=1\{V_i^\sigma>z\},
 \quad f_\sigma(p,I)=W_\sigma'(p)(I-p),
\]

and the genuine leave-one-out statistic

\[
 U_n={1\over n}\sum_{i=1}^n G_i\sum_\sigma s_\sigma
 \int_0^{B_v}f_\sigma(\widehat S_{-i,n}^\sigma(z),I_i^\sigma(z))\,dz,
 \quad \widehat S_{-i,n}^\sigma={1\over n-1}\sum_{j\ne i}I_j^\sigma.
\]

Each outer path is independent of the other paths used for its empirical probability. Different summands of U_n are nevertheless dependent. In particular, their sample variance cannot be treated as the variance of n independent complete estimator replications.

The nonlinear W' leaves a finite-n plug-in bias. Excluding the outer path does not imply E[W'(S_hat)]=W'(S).

## Why the multilevel correction has enough cancellation

Write D_1=C_1+C_2 and D_2=C_3+2C_2, where C_j bounds |W^(j)| over both sides. A second-order expansion in the empirical survival error, followed by the ordered-pair Hoeffding decomposition, gives

\[
 U_n-g={1\over n}\sum_{i=1}^n H(X_i)+R_n,
 \qquad E H=0,\qquad E\|R_n\|^2\le C_R/n^2,
\]

\[
 C_R=B_v^2M_2(32D_1^2+8D_2^2).
\]

Here is the dependence argument, rather than an iid assumption about LOO summands. The linear empirical-CDF term has the ordered kernel

\[
 K(x,y)=G_x\sum_\sigma s_\sigma\int
 f_{\sigma,p}(S_\sigma,I_x^\sigma)(I_y^\sigma-S_\sigma)dz.
\]

E[K(X,Y)|X]=0. Subtracting h(Y)=E[K(X,Y)|Y] makes the remaining ordered kernel degenerate in both arguments. Only identical or reversed ordered index pairs contribute to its covariance, giving an O(n^-2) remainder second moment. The Taylor remainder has the same order because E(S_hat-S)^4 <= (n-1)^-2 and the outer score is independent of its leave-one-out empirical CDF. These are the two parts of RC-4.

For N=n2^l, evaluate the full sample and its two halves on the SAME fresh trajectories:

\[
 \Delta_{n,l}=U_N-\tfrac12(U_{N/2}^{(1)}+U_{N/2}^{(2)}).
\]

The H terms cancel path by path, not merely in expectation. The L2 triangle inequality on the three remainder terms yields

\[
 E\|\Delta_{n,l}\|^2\le {9C_R\over n^2}\,2^{-2l}.
\]

This cancellation would not hold if the three full/half estimators used independent paths.

## Unbiasedness

Choose n>=2, activation probability 0<rho<=1, and 1<a<2. Put r=2^-a and q_l=(1-r)r^(l-1), l>=1. Independently generate A~Bernoulli(rho). When active, generate L~q and a fresh full/half correction batch, independent of the base sample. Output

\[
 Z_n=U_n^{(0)}+{A\over\rho q_L}\Delta_{n,L}.
\]

Let mu_n=E U_n. Since E Delta_(n,l)=mu_(n2^l)-mu_(n2^(l-1)), and the previous second-moment bound implies absolute L1 summability,

\[
 \begin{aligned}
 E Z_n
 &=\mu_n+\sum_{l=1}^\infty E\Delta_{n,l}\\
 &=\mu_n+\lim_{L\to\infty}(\mu_{n2^L}-\mu_n)=g.
 \end{aligned}
\]

This is exact unbiasedness of every complete output. It is not an inference from a Monte Carlo interval containing zero. The no-hard-cap and no-selective-discard conditions are part of this construction. Each sampled level and each completed output has finite cost almost surely; the maximum possible cost is not bounded.

## Finite variance, CLT and budget

Define

\[
 Q_a=\sum_{l\ge1}{4^{-l}\over q_l}
 ={1\over4(1-r)(1-1/(4r))},\quad
 \kappa_a=\sum_{l\ge1}q_l2^l={2(1-r)\over1-2r}.
\]

Both are finite for 1<a<2. Efron--Stein for the whole LOO statistic gives

\[
 \operatorname{trCov}(U_n)\le C_U/n,
 \quad C_U=8B_v^2M_2(4C_1^2+D_1^2).
\]

Independence of base and correction therefore implies, for M independent COMPLETE outputs,

\[
 E\|\overline Z_{M,n}-g\|^2
 \le {1\over M}\left({C_U\over n}
       +{9C_RQ_a\over\rho n^2}\right),
 \qquad E\operatorname{cost}(Z_n)=n(1+\rho\kappa_a).
\]

For fixed n, rho, a and target, the finite-second-moment iid CLT gives sqrt(M)(Z_bar-g) converging to N(0,Sigma_n). Coordinate studentization requires positive coordinate variance. Minimax optimality additionally requires a valid lower bound on the same model class and the same information/cost budget; it does not follow from this CLT or from the four plots.

## Does reuse help relative to a matched independent split?

For the influence functions H_out and H_in in the source proof, at a fixed allocation fraction nu,

\[
 \Sigma_{\rm split}(\nu)-\operatorname{Cov}(H_{\rm out}+H_{\rm in})
 =\operatorname{Cov}\left(
 \sqrt{\nu\over1-\nu}H_{\rm out}
 -\sqrt{1-\nu\over\nu}H_{\rm in}\right)\succeq0.
\]

This is an asymptotic covariance comparison for matched centered kernels under the source assumptions. With rho_n=n^-1/2 the correction is asymptotically negligible on the sqrt(n) scale and its cost fraction vanishes. It does not establish strict finite-budget superiority over every split allocation or over the raw-kernel v5 estimator.

## Checks actually performed in this delivery

The runner checks the sorted LOO implementation against explicit leave-one-out evaluation, including ties and vector scores. A Bernoulli model checks the finite full/half telescoping identity by enumeration. Random levels have no cap and all generated corrections are retained. These executable checks verify formulas and bookkeeping; the infinite-level expectation argument is the mathematical argument above. No Lean formalization was performed.
