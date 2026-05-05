# Risk metrics and how to read backtest summaries

## Annualised return vs annualised volatility
**Annualised return** scales average per-period arithmetic or log returns assuming ~252 equity trading days. **Volatility** is standard deviation of returns scaled similarly. These are descriptive sample statistics—not guaranteed forward forecasts.

## Sharpe ratio (working definition)
**(Mean − risk−free)** divided by volatility. If the risk−free approximation is imperfect in emerging markets proxies, interpretations remain rank-wise comparable within the study if held constant across strategies.

Elevated Sharpes in short samples reflect favourable regimes; scepticism required when extrapolating to future stress.

## Benchmark excess return
Portfolio **minus benchmark** arithmetic return over the horizon (or equivalently attribution forms). Naming matters: "beat the benchmark by X%" specifies whether arithmetic or geometric compounding framing was used internally.

## Maximum drawdown
Largest proportional drop from rolling peak of cumulative equity curve. Helps communicate pain path even when annualised volatility looks moderate.

## Sensitivity traces (confidence in views)
If raising the **view multiplier** concentrates weights into predicted winners, diversification falls and drawdown tails can widen even if headline mean return rises historically.

## Honest disclaimers when presenting to reviewers
Historical optimisation does not subtract transaction taxes, liquidity impact, dividend reinvestment quirks, survivorship cleanliness. Production deployment requires stress tests plus latency around data ingestion.
