# Black-Litterman model — conceptual primer

## Goal
Combine an equilibrium view of expected returns with subjective or data-driven opinions ("views"), while staying numerically disciplined and diversification-aware.

## Core idea (intuition)
Investors anchor on a consensus prior (typically implied equilibrium excess returns scaled from observed covariances and market capitalization weights — CAPM-linked reasoning). Views adjust that prior in Bayesian fashion. The posterior expected returns shrink toward equilibrium unless views are convincing (via confidence parameters).

## Key ingredients
**Prior (Π or π):** vector of equilibrium expected excess returns aligned with covariance Σ and market weights \(w_{\text{mkt}}\). Common construction: \(\Pi = \delta \Sigma w_{\text{mkt}}\) using a scalar risk-aversion \(\delta\) (alternative normalizations exist).

**Views (Q, P, Ω):** Express opinions as \(P \, \mu \approx Q\). Matrix \(P\) picks linear combinations of assets (often per-asset forecasts or relative forecasts). Ω is covariance of view noise — encodes uncertainty. Smaller Ω means "stronger / more precise" views; larger Ω means weaker views that barely move Π.

**Posterior formulas:** Posterior blends prior and views through Σ and Ω. Practical implementations compute new mean \(\mu_{\text{BL}}\) plus updated covariance nuances depending on parametrization. **τ (tau)** scales uncertainty of the equilibrium prior versus data — raises/lowers responsiveness to Ω.

## Why practitioners use BL
Mitigates extreme mean-variance corner solutions by anchoring allocations; allows structured incorporation of discretionary or model forecasts; maps naturally to Bayesian thinking about confidence.

## Common pitfalls (interview-worthy)
Mis-specified Ω dominates outcomes; \(\tau\) is not uniquely identifiable historically and is often calibrated; unstable Σ inversion → regularization/shrinkage; views must align with economic meaning of P rows.
