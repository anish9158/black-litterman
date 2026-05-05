# Optimisation constraints and economic meaning (long-only, caps, budget)

## Budget constraint
** ∑_i w_i = 1 ** with **w_i ≥ 0** implements a **long-only** fully invested portfolio. Partial cash can be modelled by adding a synthetic cash asset with near-zero variance or explicit slack variable.

## Per-name caps
Upper bounds like **w_i ≤ 0.40** avoid single-name dominance from estimation error or view spikes. Tight caps reduce concentration but may miss genuine edge if model quality very high (unlikely with noisy returns).

## Risk term and objective shape
Quadratic risk **wᵀΣw** plus linear return **μᵀw** form convex objective with linear inequality constraints — solvers return global optimum under stated Σ.

## Shadow effects of constraints
Binding caps create **kinked** efficient frontiers; small μ perturbations may not move weights until another constraint releases.

## Connection to BL output
Posterior **μ_BL** feeds directly into this stage; mis-scaled μ relative to Σ magnitude mis-ranks assets—normalisation heuristics sometimes adjust μ scale to historical vol bands.
