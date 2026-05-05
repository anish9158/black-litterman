# Black-Litterman model — conceptual primer

## Goal
Combine an equilibrium view of expected returns with subjective or data-driven opinions ("views"), while staying numerically disciplined and diversification-aware.

## Core idea (intuition)
Investors anchor on a consensus prior (typically implied equilibrium excess returns scaled from observed covariances and market capitalization weights — CAPM-linked reasoning). Views adjust that prior in Bayesian fashion. The posterior expected returns shrink toward equilibrium unless views are convincing (via confidence parameters).

## Key ingredients
**Prior (Π or π):** vector of equilibrium expected excess returns aligned with covariance Σ and market weights \(w_{\text{mkt}}\). Common construction: \(\Pi = \delta \Sigma w_{\text{mkt}}\) using a scalar risk-aversion \(\delta\) (alternative normalizations exist).

**Views (Q, P, Ω):** Express opinions as \(P \, \mu \approx Q\). Matrix \(P\) picks linear combinations of assets (often per-asset forecasts or relative forecasts). Ω is covariance of view noise — encodes uncertainty. Smaller Ω means "stronger / more precise" views; larger Ω means weaker views that barely move Π.

**Posterior formulas:** Posterior blends prior and views through Σ and Ω. Practical implementations compute new mean \(\mu_{\text{BL}}\) plus updated covariance nuances depending on parametrization. **τ (tau)** scales uncertainty of the equilibrium prior versus data — raises/lowers responsiveness to Ω.

Compact matrix form often written:

\[\mu_{\text{BL}} = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} \big[(τΣ)^{-1}\pi + P^T Ω^{-1} Q\big]\]

Implementations vary in whether τ multiplies Σ only in the equilibrium block; stay consistent with the code path you ship.

### Posterior covariance (memory aid)
Variants include \(\Sigma_{\mu} = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1}\), used when sampling or stress-testing uncertainty in μ rather than fixing a point estimate for mean–variance.

### Numerical stability
Matrix inverses amplify noise. **Regularised Σ**, ridge on unstable directions, or **factor-structure** covariance approximations help when \(\Omega^{-1}\) entries are huge (very tight views).

### Bayesian shrinkage intuition
The posterior pulls μ toward **π** when Ω is diffuse or when the equilibrium block dominates; it moves toward satisfying **PQ**-like forecasts when Ω is tight and P is well-conditioned.

## Why practitioners use BL
Mitigates extreme mean-variance corner solutions by anchoring allocations; allows structured incorporation of discretionary or model forecasts; maps naturally to Bayesian thinking about confidence.

## Common pitfalls (interview-worthy)
Mis-specified Ω dominates outcomes; \(\tau\) is not uniquely identifiable historically and is often calibrated; unstable Σ inversion → regularization/shrinkage; rows of P must match the economic meaning of each view (absolute vs relative).

## Link to ML in this codebase
Treat **Q** as predictive expected returns, **Ω** (often diagonal) from residual dispersion heuristics, and **P** as identity-like picks for simple per-name views — a practical bridge between XGBoost-style signals and the BL update.
