# Frozen target and semantic inventory

The target is the source in Rebased_99afee22, not the older attachment called FullDiff_Final(1).
Only I1–I10 from the source package root audit are authorized edits.

## Core semantic objects to retain (13 groups)

1. Episode state and frozen information/context: s_u, X_u, omega_u, z_k, F_k.
2. Dynamic reference r_u, its recursion Phi, and start reference r_k=r_{t_k}.
3. Dirichlet policy pi_theta, concentrations, latent proposals, and executed action map.
4. True and simulator episode models P_k, tilde P_k and induced trajectory laws.
5. Relative outcome Y and regularized gain/loss CPT transformations.
6. True/simulator objectives J_k, tilde J_k.
7. Objective gradients g_k, tilde g_k and trajectory score G_{k,theta}.
8. LOO statistics, raw multilevel correction, complete debiased output and averaged estimate.
9. Normalized projected parameter update and residual Q_k.
10. Squared projected residual zeta_k, its average expectation E_K, and DLR.
11. Stationary set S_k and optional distance/error-bound consequence.
12. Simulator error beta_k and objective variation V_K bridging learning and execution.
13. Fixed-target minimax oracle, stopping count T_call, budget B and risk.

## Repaired theorem-visible definitions

- S_theta^sigma(y): scalar in [0,1], directly S_{P_k,z_k,r_k,theta}^sigma(y), fixed true target; use the analogous frozen simulator model for that identity.
- G_theta: trajectory-dependent vector in R^d, directly G_{k,theta}.
- g: fixed-target vector in R^d, nabla J(theta), defined before the LOO cancellation expansion.
- G_x: score component of a trajectory-summary realization x=(v^+,v^-,G_x); not the population gradient.
- T_call: nonnegative integer-valued stopping count, the same object in the oracle assumption and stopped-KL proof.

No estimator-design variable was lost when the appendix table was shortened: n, varrho, b, nu_l, D, Z, C_var are already defined in Algorithm/Main Results at their use sites.

## Scope of all limits

M grows with target, n, varrho, b and d fixed. K grows only under episode-uniform primitive constants and the stated variation/error conditions. B grows at fixed d, regularization and model-class constants. None of these limits is replaced by a different asymptotic regime in this patch.
