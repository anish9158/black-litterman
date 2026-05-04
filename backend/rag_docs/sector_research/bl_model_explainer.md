# Black-Litterman Model — Technical Explainer

## Core Formula
The Black-Litterman model combines the market equilibrium return vector (π) with investor views (Q) to produce a posterior return vector (μ):

  μ = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹π + PᵀΩ⁻¹Q]

Where:
- **π** = implied equilibrium returns from CAPM (market cap weighted)
- **τ** = scalar uncertainty in the prior (typically 0.05–0.15)
- **Σ** = covariance matrix of asset returns
- **P** = pick matrix (which assets each view applies to)
- **Q** = view return vector (predicted by XGBoost in this project)
- **Ω** = diagonal uncertainty matrix for each view (residual variance from XGBoost)

## XGBoost Views Generation
For each stock i in the portfolio:
1. Features are computed: price, SMA20, SMA50, 21-day momentum, RSI, trailing PE, price-to-book
2. An XGBoost regressor is trained on the train window (last 80% of history)
3. The model predicts daily returns on the test window
4. Q[i] = mean predicted return annualised; Ω[i,i] = residual variance from training

## Covariance Shrinkage
The raw sample covariance matrix is regularised:
  Σ_shrunk = (1-α)·Σ + α·diag(Σ)   where α = 0.1

This reduces extreme off-diagonal correlations that inflate estimation error.

## CVXPY Optimisation
Portfolio weights w are chosen to minimise:
  minimise:  w'Σw - (1/λ)·μ'w
  subject to: sum(w) = 1, w ≥ 0, w ≤ 0.4 (per-asset cap)

Where λ = risk_aversion parameter (default 2.5).

## Key Parameters
| Parameter     | Default | Meaning                              |
|---------------|---------|--------------------------------------|
| tau           | 0.10    | Prior uncertainty scalar             |
| risk_aversion | 2.5     | Controls return vs risk trade-off    |
| shrinkage α   | 0.10    | Covariance regularisation strength   |
| view blend    | 0.8     | Weight of XGBoost vs equilibrium view|

## Fama-French 5 Factor Model
Additionally computed as a risk decomposition:
- **Mkt-RF**: Market excess return
- **SMB**: Small minus Big (size premium)
- **HML**: High minus Low (value premium)
- **RMW**: Robust minus Weak (profitability premium)
- **CMA**: Conservative minus Aggressive (investment premium)

Computed from NIFTY50 constituents monthly data using market cap, book-to-market ratio, operating margin, and asset growth.

## Sensitivity Analysis
The view multiplier s scales Q: Q_scaled = s · Q
- At s=0.5: model is conservative, closer to market-cap weights
- At s=1.0: default calibration
- At s=1.5: model is aggressive, concentrated in highest-view stocks
ICICIBANK and BAJFINANCE gain weight as s increases (highest XGBoost confidence).
HDFCBANK and HINDUNILVR decrease (model discounts their relative views at higher confidence).
