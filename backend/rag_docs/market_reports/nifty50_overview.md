# NIFTY 50 Index — Overview & Benchmark Context

## What is NIFTY 50?
The NIFTY 50 (ticker: ^NSEI) is the flagship broad-based stock market index of the National Stock Exchange of India (NSE). It tracks the performance of the 50 largest and most liquid Indian companies listed on the NSE, representing approximately 65% of the total market capitalisation of the exchange.

## Benchmark Role in This Project
In this Black-Litterman portfolio optimizer, the NIFTY 50 is used as:
- The **buy-and-hold benchmark** against which portfolio performance is measured.
- The **market-cap equilibrium proxy** — NIFTY constituent weights imply the market's consensus expected returns (the π vector in Black-Litterman).
- The **universe** for the All-50 backtest model (all NIFTY 50 stocks are included as candidate assets).

## Historical Performance
- Long-run annualised return (20-year): approximately 14–16% per year.
- 5-year annualised return (2020–2025): approximately 18–22% per year.
- This project's All-50 XGBoost model achieved +41% annualised return vs ~21% for the benchmark in the backtest period, a +20pp outperformance.

## Sector Composition (approximate, 2024)
| Sector              | Weight |
|---------------------|--------|
| Financial Services  | 33%    |
| IT                  | 14%    |
| Oil & Gas           | 12%    |
| Consumer Goods      | 9%     |
| Automobiles         | 6%     |
| Pharma              | 5%     |
| Metals              | 4%     |
| Telecom             | 4%     |
| Others              | 13%    |

## Key Index Facts
- Base year: 1995 (base value: 1000)
- Rebalancing: Semi-annual (March and September)
- Weighting method: Free-float market capitalisation
- Managed by: NSE Indices Limited

## Why Black-Litterman Outperforms Simple Index Investment
The Black-Litterman model improves upon passive index investing by:
1. **Views integration** — XGBoost predictions shift the portfolio away from pure market-cap weighting toward assets with better forward-looking return expectations.
2. **Covariance shrinkage** — reduces estimation error in the covariance matrix, leading to more diversified and stable allocations.
3. **Convex optimisation** — explicitly minimises portfolio variance for a given expected return target, subject to long-only and budget constraints.
