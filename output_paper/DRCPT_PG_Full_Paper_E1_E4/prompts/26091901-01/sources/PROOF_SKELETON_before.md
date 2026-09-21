# Proof Skeleton and Symbol Flow

Target: current manuscript in `final_package_after/paper/DRCPT_PG_Full_Paper_E1_E4/`.
Scope: Problem Setup -> Assumptions -> Algorithm -> Main Results -> Appendices A-E.

## Semantic center

The paper has two coupled chains.

**Execution / behavioral chain**

\[
(\mathcal F_k,z_k,r_k,\theta_k,P_k)
\to \pi_{\theta_k}
\to (\omega_u,X_u,r_u)_{u=t_k}^{t_k+h}
\to Y_k
\to J_k(\theta,r_k)
\to g_k(\theta)
\to Q_k(\theta),\mathsf S_k,\zeta_k,\mathcal R_{\rho,w}(K).
\]

**Learning / simulator chain**

\[
(\mathcal F_k,z_k,r_k,\theta_k,\widetilde P_k)
\to \widetilde J_k
\to \widetilde g_k
\to Z_{k,m}^{\rm db}
\to \widehat g_k^{\rm db}
\to \theta_{k+1}.
\]

The bridges are

\[
\|\widetilde g_k(\theta_k)-g_k(\theta_k)\|\le \beta_k,
\qquad
\mathcal V_K=\sum_k \sup_\theta |J_{k+1}-J_k|.
\]

These two bridges feed the online residual theorem.  The optional error bound maps the residual to distance from the stationary set.  The minimax section is a separate fixed-target information experiment.

## Dependency DAG

1. Definitions: state / reference recursion / Dirichlet policy / CPT objective / projected residual / DLR.
2. Assumptions 3.1-3.2 -> Appendix A score moments and objective smoothness.
3. Appendix A + fixed target -> gradient identity.
4. Gradient identity + LOO construction -> level cancellation -> centering -> unbiased estimator / finite variance / finite expected cost -> CLT.
5. Smoothness + projected update -> one-step projected ascent.
6. One-step ascent + estimator MSE + objective variation -> average projected-residual theorem.
7. Residual theorem -> DLR corollary.
8. Optional error bound -> stationary-set distance and set drift.
9. Fixed-target oracle assumption + estimator theorem -> minimax lower/upper rate.

No semantic cycle was found in this dependency graph.

## Assumption ledger

| Result | Required assumptions / facts | Current status |
|---|---|---|
| score moments / likelihood differentiation | compact Theta, bounded features, finite h,N, parameter-independent environment/action map | stated for analyzed domain; simulator scope needs clarification |
| CPT smoothness | score moments + C3 regularized weights | stated |
| gradient identity | score moments + bounded weight derivative | stated |
| level cancellation | C3 weight derivative + score second moment | stated |
| exact debiasing | level second-moment / bias decay + independent base/correction | stated/proved |
| CLT | iid complete outputs + finite covariance; nonsingularity only for studentization | stated |
| projected ascent | L-smooth objective, convex Theta, gamma L <= vartheta | stated |
| online residual | projected ascent + simulator-gradient error + objective variation | stated, subject to simulator regularity clarification |
| DLR | residual definition + Jensen/reindexing | stated/proved |
| stationary-set tracking | optional residual error bound | stated |
| minimax | fixed-d model class, exact trajectory oracle, expected call budget, realizable Bernoulli submodel | stated/proved |
