# Glossary — BL portfolio / ML / RAG (this project)

**Black-Litterman (BL):** Bayesian-style blend of equilibrium expected returns (prior) and explicit views (signals) with calibrated uncertainty, yielding a posterior mean of returns used in optimisation.

**Prior / equilibrium (π or Π):** Baseline expected excess returns implied from market weights and Σ (CAPM-scaled construction in many implementations). Anchors the optimiser so weights do not chase tiny μ noise.

**Views (Q):** Vector of expected outcomes the model wants to reflect. In this project **Q** entries come from **XGBoost** out-of-sample predictions of returns (or their summary statistics), not from human analyst spreadsheets.

**Pick matrix (P):** Each row selects which assets a view applies to. A simple design is one row per asset with a 1 in that column (absolute view on that name). Relative views use +1 and −1 in two columns ("A beats B").

**Omega (Ω):** View uncertainty covariance; often diagonal. **Ω[i,i]** large → view i is fuzzy → posterior barely moves. **Ω[i,i]** small → view is treated as informative. Residual variance from the view model is a common heuristic to build Ω.

**Tau (τ):** Scales how uncertain the **prior** is relative to views. Higher τ often means the prior is diffuse and views can pull μ more; lower τ makes the equilibrium stickier. Typical reported ranges in practice: small positive decimals (calibration-dependent).

**Shrinkage (Σ):** Sample covariance is noisy; shrinkage (e.g. Ledoit–Wolf toward a structured target) stabilises optimisation and matrix inversions.

**Mean–variance optimisation:** Maximise μᵀw − (λ/2) wᵀΣw with constraints (long-only, caps, budget). **λ** is risk aversion.

**CVXPY:** Python modelling layer for convex programs; used here for constrained weight solving.

**Rolling / walk-forward test:** Refit features and models on past data, predict and trade forward, then slide the window. Reduces look-ahead bias versus one-shot in-sample fit.

**View multiplier (s):** Scales Q (e.g. Q → s·Q) to study how allocation changes when conviction in ML views is dialled up or down (**sensitivity analysis**).

**Sharpe ratio:** Mean excess return divided by volatility (often annualised). Higher indicates more return per unit of risk in the sample window.

**Maximum drawdown:** Peak-to-trough loss on a cumulative curve; measures path risk.

**Embeddings:** Dense vectors representing text meaning; used to match user questions to knowledge-base **chunks** in FAISS.

**Topic guard / closed ecosystem:** Filters or prompts that block off-domain questions so the assistant does not hallucinate unrelated facts.
