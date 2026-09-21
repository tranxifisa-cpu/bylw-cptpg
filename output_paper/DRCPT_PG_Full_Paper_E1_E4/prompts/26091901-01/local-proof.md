# Local proof of the repaired model-to-estimator interface

Status: LOCAL_PROVED for the I1–I10 repair obligation under the explicitly stated assumptions. This is not unrestricted full-paper semantic acceptance.

## Target and sufficient subgoals

The substantive obligation I7 is to justify applying the existing score-gradient and estimator results conditionally under each frozen simulator. It is enough to establish (a) theta-independent nonpolicy kernels under that simulator, (b) a common integrable envelope for the policy likelihood derivatives, and (c) the finite moments/weight regularity already used by the LOO estimator. These are exactly the premises of the existing Appendix A–C arguments; no theorem rate or estimator needs replacement.

## A. Hold the right object fixed (I6–I7)

The episode environment/model P, context (z,r), and fitted simulator parameters are frozen. P_theta is the augmented trajectory law induced by pi_theta in that environment. Thus P is parameter-independent while the trajectory distribution is not. Assumption trajectory now explicitly quantifies its bounds and parameter-independence conditions uniformly over both analyzed true and frozen-simulator models, conditional on episode-start information/context. This is an explicit assumption-scope clarification/strengthening, not empirical certification of every simulator.

## B. Differentiate under the frozen conditional law

Compact theta and bounded features give concentrations in a common positive compact interval. On a compact neighborhood, the Dirichlet density and its first two parameter derivatives have an integrable envelope proportional to

\[
\prod_{i=0}^N\widetilde\omega_i^{a_- -1}
\left(1+\sum_{i=0}^N|\log\widetilde\omega_i|^2\right),\qquad a_->0.
\]

The finite product over h latent actions stays integrable; the theta-independent market kernels integrate to one iteratively. This provides actual domination, not merely a numerical moment bound. DCT gives the bounded-functional likelihood derivative and mean-zero score, including under the simulator.

The indicator survival derivative is bounded by sqrt(C_G,2). Bounded weight derivatives on [0,1] and the finite threshold interval justify differentiating the CPT integral. Fubini is justified by 2 B_v C_1 E||G||<infinity. The repaired local names S_theta and G_theta identify exactly the survival and score in this calculation (I1).

## C. Return to the estimator and online target

The unchanged LOO proof uses independent held-out scores, a C3 Taylor remainder, a degenerate ordered U-statistic, and a Bernoulli fourth moment. All those hypotheses are now explicitly available for the frozen simulator with the common constants. The corrected definition g=nabla J(theta) precedes the full/half expansion (I2), while G_x remains its distinct score coordinate (I3). Existing cancellation gives E||D_n,l||^2 <= 9 C_rem n^-2 4^-l; 1<b<2 gives finite correction variance and expected cost. Absolute first-moment summability justifies telescoping. Therefore E[ghat_k|F_k]=tilde g_k and conditional MSE <= C_var(n)/M_k under the supplied replication schedule.

The cross term between this conditional zero-mean error and the frozen vector tilde g_k-g_k is zero. The existing projected-ascent lemma and moving-objective telescope then give the original online bound with exactly the original constants 8, 10 and 2 B_v. No B_J alias is needed (I8). This closes the original I7 target.

## Remaining interface obligations

I4 only identifies the already assumed oracle stopping count; the event {T_call>=j} is known before call j, and bounded Bernoulli log increments with E T_call<infinity justify the stopped KL sum. I5 states the consecutive-episode boundary relation used by Algorithm 2. I9 retains all thirteen semantic groups above in a 16-row main table while leaving local definitions untouched. I10 distinguishes Q, its square, average expectation, and DLR without changing any metric formula or experimental value.

## Counterexample check for the repaired assumption

An unrestricted simulator could use zero policy features (hence G=0) but a theta-dependent Bernoulli return probability p(theta)=1/2+theta/4. Then its CPT objective under identity weights is p(theta) E[v(aU)], whose derivative is positive, whereas the recorded-score estimator is zero. The new parameter-independence requirement expressly excludes that verified construction. It illustrates why the scope addition is necessary; it does not refute the corrected assumption or the frozen simulator used in the experiments.

Top-down derivation structure: PASS
