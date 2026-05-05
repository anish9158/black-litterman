# Views: P matrix, Q vector, Ω, and the sensitivity multiplier s

## Why structure matters (P)
Each **view row** asserts something about combinations of unknown expected asset returns \(\mu_i\).

**Diagonal pick row:** One asset only — natural when each stock owns an independent scalar forecast \(Q_k \approx \mu_i\).

**Relative row:** \(\mu_i - \mu_j \approx \text{spread}\)— picks +1 / −1 in same row; captures pair relationships but couples uncertainty because model errors correlate.

Poorly conditioned P interacts badly when concatenated alongside dense Σ inversion — regularisation pathways matter.

## Q from machine learning versus human analyst
Traditional BL examples use qualitative views (“Tech outperforms value by 100 bps/year”). Quant variants replace them with statistically estimated expectations (here **XGBoost** mean predictions). Transparency trade-off: deterministic math once Q fixed, stochastic training noise propagates indirectly.

## Building Ω intuitively (diagonal first pass)
Assume independent view errors: Ω diagonal.**Large** variance → sceptical Bayesian update leaves posterior near π.**Tiny** diagonal → dangerously overconfident—optimiser hunts corners.

Residual variance heuristic: fit error dispersion from cross-validated residuals supplies scale for diagonal entries aligning noise with empirical model uncertainty.

Off-diagonal ω_{ij}: needed if views share overlapping information (duplicate signal across correlated tickers)—often simplified away for prototyping.

## Sensitivity multiplier scaling Q → s · Q (no Ω auto-rescale assumption)
Increasing **s** amplifies disagreement between equilibrium and predictive layer while Ω unchanged mechanically treats predictions as bolder relative magnitude. Observed behaviours: allocations migrate toward strongest predicted outperformers absent binding caps—in line with exploratory academic sensitivity charts.

Interpretation caveat: multiplying Q without widening Ω asserts equal confidence harder—paired stress should sometimes scale Ω proportional to s squared for coherence; projects often omit for transparent scenario toggling.
