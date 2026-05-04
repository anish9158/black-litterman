# Covariance estimation, shrinkage, and CVXPY in optimisation

Empirical covariance from limited windows is noisy; eigenvalue instability breaks quadratic optimisers.

**Ledoit-Wolf shrink** blends sample covariance toward structured targets tightening condition numbers.

Numerical safeguards: jitter on diagonal regularization symmetric decomposition fallbacks mitigate inversion blow-ups.
