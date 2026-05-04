# Rolling-window backtesting design

Purpose: emulate production decisions where models retrain periodically on expanding or sliding histories then predict next period unseen returns avoiding single static hindsight fit.

Procedure sketch: sliding training window \(T_{\text{train}}\) trains feature pipeline + boosted models predicting next-step returns; optimise portfolio solving constrained problem with updated Σ (often Ledoit-Wolf shrinkage stabilisation); accumulate realised P&L; roll forward skipping overlap leakage.

Sensitivity sweeps sweep meta-parameters (e.g. view confidence multiplier scaling Ω) illustrating robustness regimes.
