# Mean-variance optimisation (Markowitz) primer

## One-period objective
Estimate expected excess returns \(\mu\) and covariance \(\Sigma\) for assets \(i=1..N\). Choose weights \(w\) to maximize \(\mu^{\top}w - (\lambda/2)\, w^{\top}\Sigma w\) subject to \(\sum_i w_i=1\) and optional long-only or sector caps.

Risk aversion \(\lambda\) trades return vs volatility: higher \(\lambda\) shrinks exposures.

## Convexity & solvers
With linear inequality constraints problem is convex; **CVXPY** (disciplined convex programming) is a standard Python toolchain with reliable solvers (ECOS, OSQP variants, etc.). Non-convex penalties require different methods.

## Instability hallmark
Tiny changes in \(\mu\) can swing corner solutions wildly because optimiser hunts max Sharpe extremes. Mitigations: Bayesian / shrinkage estimates, turnover penalties, heuristic caps, blended priors (**Black-Litterman** rationale).
