# Centering the complete corrected LOO estimator

This is a new extension for this experiment, not text copied from the user's manuscript. The base model, regularized TK weights, mean-zero score assumption, and full/half LOO correction are unchanged from `original_LOO_PROOF_REVIEW.md` and the uploaded code.

## 1. Zero-mean controls at every finite sample size

For each signed gain/loss side s, define

C_{n,s} = (1/n) sum_i G_i sign_s integral W_s'(S_hat_{-i,s}(z)) S_hat_{-i,s}(z) dz.

The integrand coefficient for path i depends only on the other paths. Conditional on the frozen target it is independent of G_i, and E G_i=0. Hence E C_{n,s}=0 for every n>=2. The raw LOO statistic U_{n,0} and the centered statistic

U_{n,lambda} = U_{n,0} - lambda_+ C_{n,+} - lambda_- C_{n,-}

have the same finite-n expectation for any fixed coefficients. The usual centered statistic sets both coefficients to 1; raw sets both to 0. Centering alone does not remove plug-in bias.

## 2. Apply the same control construction to the complete random correction

Let H_0=A Delta_L^{raw}/(rho q_L). For each side let

H_{C,s}=A [C_{n2^L,s} - (C_{n2^{L-1},s}^{half1}+C_{n2^{L-1},s}^{half2})/2]/(rho q_L).

Each H_{C,s} has mean zero under the summability conditions. Thus

Z_lambda = U_{n,0}+H_0 - sum_s lambda_s (C_{n,s}^{base}+H_{C,s})

is unbiased for g. Here corrections are independent of the base, and full/half statistics share their correction trajectories. No hard level cap or selective omission is permitted.

For bounded |lambda_s|<=4, the finite-moment and summability proof extends. One may alternatively observe that each new complete output is a bounded linear combination of finite-second-moment raw and side-control complete outputs. The level difference retains its second-order cancellation. The bounded control coefficients do not clip the estimator output or random level.

## 3. Independent pilot fitting preserves unbiasedness

Let P be a pilot dataset independent of all production base samples, correction samples, activations and levels. Fit bounded coefficients lambda_hat(P), then freeze them. Conditional on P,

E[Z_{lambda_hat} | P] = g.

Therefore the unconditional estimator is also unbiased. Finite second moments remain valid because coefficient norms are uniformly bounded. A pilot-derived intercept must NOT be subtracted from production estimates: the control's population mean is zero, but subtracting a nonzero empirical pilot intercept would change the target. This implementation uses no intercept in the production correction.

## 4. Optimize the complete output, not only the base term

Write R=Z_raw and C=sum_s(C_{n,s}^{base}+H_{C,s}). For a scalar coefficient,

V(lambda)=tr Cov(R) - 2 lambda E[(R-E R)^T C] + lambda^2 E||C||^2.

The population optimum is lambda*=E[(R-E R)^T C]/E||C||^2 when the denominator is positive. It follows that

V(lambda)-V(lambda*) = E||C||^2 (lambda-lambda*)^2.

Since the complete output is unbiased, V(lambda) is also its MSE. For two controls, replace this by the quadratic V(lambda)=V0-2 lambda^T b+lambda^T Q lambda. The population optimum need not be reached by a finite pilot, and independent fitting preserves unbiasedness, not a guaranteed variance reduction.

The implemented pilot fit uses the sample covariance trace objective with a relative 1e-3 ridge towards coefficient 1 and fixed bounds [-4,4]. These controls were specified before test sampling. No numerical reference gradient is used in fitting.

## 5. Additional candidate: different base and correction coefficients

Because both base controls and correction controls separately have mean zero, one may use

Z = U_raw - lambda_B C_B + H_raw - lambda_H C_H.

The two coefficients need not match to preserve unbiasedness. Base and correction are independent, so population variance is the sum of their two variance quadratics. The code fits each using the independent pilot; nonactivation zeros remain part of the correction distribution. A prespecified fallback sets lambda_H=1 if fewer than eight corrections occur in the pilot. This candidate is additional to the requested scalar and gain/loss centering, and all its test outcomes are reported.

## 6. Pilot cost is not free

Let c be the expected cost of one complete output, P the total expected pilot cost, V0 the fixed-centering variance and V1 the learned-centering variance. Averaging M production outputs costs P+Mc and has conditional MSE V1/M. At the same cost, fixed centering can use about M+P/c outputs and has MSE V0/(M+P/c). The learned design is cost-effective only when

M > P/[c(V0/V1-1)], provided V1<V0.

This calculation requires a common fixed target and reusable coefficients. A parameter/state/market change does not automatically permit reusing the coefficients with the same variance benefit; unbiasedness can persist for a fixed or predictable coefficient under a correctly conditioned new draw, but the variance advantage must be tested again.

## References

- Source LOO construction: proofs/original_LOO_PROOF_REVIEW.md and source/vendor/mvp_cpt_pg/cpt_objective.py.
- General policy-gradient control-variate perspective: Greensmith, Bartlett and Baxter (2004), *Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning*, JMLR 5:1471-1530. https://www.jmlr.org/papers/v5/greensmith04a.html . Its setting is not this CPT model; the identities above are derived for this experiment.

## 7. Follow-up: a fixed, pilot-free hybrid

Primary-study diagnostics suggested testing `lambda_B=1, lambda_H=0`: center the base, keep the correction raw. This was not one of the primary predeclared candidates. It is evaluated using a separate prospectively recorded confirmation protocol and a new master random seed.

Z_hybrid = U_centered + H_raw.

Since E U_centered = E U_raw = mu_n, and E H_raw = g-mu_n, E Z_hybrid = g. The corrected raw and centered constructions already imply the needed second moments, and independence of base and correction is unchanged. Thus fixed coefficients 1 and 0 require no pilot observations at production time.

The exact population MSE comparison to the old fixed-centered method is especially simple:

MSE(Z_hybrid)-MSE(Z_fixed) = tr Cov(H_raw)-tr Cov(H_centered).

Both use exactly the SAME centered base and exactly the SAME expected path count. The hybrid is better precisely when its raw correction has smaller variance; the inequality is not universally guaranteed by centering. This is the condition tested in the independent confirmation samples.
