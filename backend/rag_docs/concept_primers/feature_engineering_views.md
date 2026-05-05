# Feature engineering — technical and fundamental signals for ML views

## Role in this project
**XGBoost** models ingest tabular histories per ticker. Predicted returns (or residuals) supply **Black-Litterman views (Q)**. Good features stabilize signal and interpretability (**SHAP** later explains which drove each prediction).

## Price-based technical features (typical family)
**Returns / momentum:** Short-horizon past returns summarise trend persistence; polynomial expansions capture mild non-linear interactions when sample size permits.

**Simple moving averages (SMA):** SMA20 versus SMA50 style ratios encode medium vs slower trend regimes; crossings are classic regime proxies on daily data.

**RSI:** Bounded oscillator (~0–100) reflecting recent up vs down magnitude; extremes flag stretched conditions (not deterministic reversal signals alone).

## Fundamental ratios (firm-level proxies)
When available from vendor or cached fundamentals: **price-to-earnings**, **price-to-book** proxies value vs growth extremes within the universe. Levels differ by sector — models learn cross-section relative to contemporaneous peers rather than naive global thresholds.

## Polynomial feature expansion (optional degrees)
Interactions between core inputs can help tree ensembles separate regimes; curse of dimensionality and overfitting escalate if polynomial degree rises without ample training depth or strict regularisation (tree depth limits, penalties, dropout not used in boosted trees).

## Leakage controls (engineering discipline)
Training labels align with horizons **after** the feature timestamp. Feature rows must exclude same-bar future prices. Rolling backtests reinforce temporal ordering.

## From prediction to BL view mapping
Forecast mean at decision day → yearly-scaled incremental return intuition for entry in **Q**. Training residual variance motivates **Ω** diagonal entries conveying model-driven confidence.
