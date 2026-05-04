# Backtest Results — Black-Litterman with XGBoost Views

## Overview
A 5-year rolling walk-forward backtest was run on NIFTY 50 stocks using:
- Training window: 150 trading days
- Test window: 30 trading days
- XGBoost views: enabled (per-stock return predictions)
- Benchmark: NIFTY 50 equal-weight index (^NSEI)

## Performance Summary

| Model             | Annual Return | Sharpe Ratio | Max Drawdown | vs Benchmark |
|-------------------|---------------|--------------|--------------|--------------|
| All 50 With Views | **41.0%**     | **2.36**     | -10.3%       | +20.1pp      |
| Top 15 With Views | 26.8%         | 1.45         | -13.0%       | +5.9pp       |
| Top 10 With Views | 23.4%         | 1.16         | -13.5%       | +2.4pp       |
| NIFTY 50 (^NSEI)  | ~20.9%        | —            | —            | baseline     |

## Key Findings
1. **More stocks = better diversification**: The All-50 model significantly outperforms the Top-10 and Top-15 models because XGBoost can exploit diverse signals across the full universe.
2. **XGBoost views are the main alpha source**: The "With Views" label means XGBoost-generated return predictions were active. Turning off views (using equilibrium Pi only) reduces performance to approximately benchmark level.
3. **Concentration risk in small universes**: The Top-10 model barely outperforms the benchmark (+2.4pp) and has higher drawdown, showing that excessive concentration reduces diversification benefits.
4. **Sharpe ratio of 2.36**: The All-50 model's Sharpe ratio significantly exceeds typical equity fund benchmarks (0.5–1.0), indicating excellent risk-adjusted performance during the backtest period.

## Sensitivity to View Multiplier
At s=1.0 (default), the key allocations are:
- ICICIBANK.NS: 17.6%
- HDFCBANK.NS: 14.8%
- BAJFINANCE.NS: 13.0%
- HINDUNILVR.NS: 5.9%
- DIVISLAB.NS: 7.9%

30 out of 50 NIFTY tickers receive zero allocation — the BL model effectively identifies which stocks deserve capital.

## Note on Live Performance
The live API (black-litterman.onrender.com) on a 1-year period showed:
- Annual return: +4.68% (portfolio) vs -15.95% (benchmark NIFTY)
- Outperformance: +20.6pp
- Sharpe: -0.10 (negative due to short horizon and recent market conditions)
- Final cumulative return: 1.026x

The difference from the 5-year backtest reflects that 2024-2025 was a challenging year for Indian equities, but the model still significantly outperformed the index.
