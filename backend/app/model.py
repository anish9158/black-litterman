import datetime as dt
import time
from typing import List, Dict, Any

import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf

NIFTY50 = [
    "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS",
    "BHARTIARTL.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCLIFE.NS", "HDFCBANK.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS", "INDUSINDBK.NS", "INFY.NS",
    "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "WIPRO.NS"
]

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS"]

def batch_download(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    data = yf.download(tickers, period=period, progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.dropna(axis=1, how="all").ffill().dropna()
    return close

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def compute_equilibrium_returns(cov_matrix: np.ndarray, market_weights: np.ndarray, risk_aversion: float) -> np.ndarray:
    return risk_aversion * cov_matrix @ market_weights

def bl_update(Pi: np.ndarray, tau: float, cov_matrix: np.ndarray, P: np.ndarray, Q: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    middle = np.linalg.inv(P @ (tau * cov_matrix) @ P.T + Omega)
    posterior = Pi + (tau * cov_matrix) @ P.T @ middle @ (Q - P @ Pi)
    return posterior

def optimize_portfolio(expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_aversion: float) -> np.ndarray:
    n = len(expected_returns)
    w = cp.Variable(n)
    objective = cp.Maximize(expected_returns @ w - risk_aversion * cp.quad_form(w, cov_matrix))
    constraints = [cp.sum(w) == 1, w >= 0, w <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        return np.ones(n) / n
    weights = np.maximum(np.asarray(w.value).ravel(), 0)
    return weights / weights.sum()

def calculate_metrics(port_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0677) -> Dict[str, float]:
    if port_returns.empty:
        return {}
    annual_ret = float(port_returns.mean() * 252)
    annual_vol = float(port_returns.std() * np.sqrt(252))
    sharpe = float((annual_ret - risk_free_rate) / annual_vol) if annual_vol else 0.0
    cum = (1 + port_returns).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    bench = benchmark_returns.reindex(port_returns.index).dropna()
    bench_ret = float(bench.mean() * 252) if not bench.empty else 0.0
    return {"annual_return": annual_ret, "annual_volatility": annual_vol, "sharpe_ratio": sharpe, "max_drawdown": max_dd, "benchmark_annual_return": bench_ret, "final_cumulative_return": float(cum.iloc[-1])}

def run_black_litterman(tickers: List[str] | None = None, period: str = "1y", risk_aversion: float = 2.5, tau: float = 0.10) -> Dict[str, Any]:
    tickers = tickers or DEFAULT_TICKERS
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    prices = batch_download(tickers, period=period)
    returns = compute_returns(prices)
    tickers = list(returns.columns)
    n = len(tickers)
    if n < 2:
        raise ValueError("At least two valid tickers are required.")
    cov = returns.cov().values * 252
    market_weights = np.ones(n) / n
    Pi = compute_equilibrium_returns(cov, market_weights, risk_aversion)
    # Simple grounded view: recent 30-day annualized mean return as Q.
    recent = returns.tail(min(30, len(returns))).mean().values * 252
    P = np.eye(n)
    Omega = np.diag(np.diag(tau * cov))
    posterior = bl_update(Pi, tau, cov, P, recent, Omega)
    weights = optimize_portfolio(posterior, cov, risk_aversion)
    port_returns = returns @ weights
    benchmark_returns = returns.mean(axis=1)
    metrics = calculate_metrics(port_returns, benchmark_returns)
    allocation = [{"ticker": t, "weight": float(w)} for t, w in sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)]
    curve = [{"date": str(idx.date()), "value": float(val)} for idx, val in (1 + port_returns).cumprod().items()]
    return {"tickers": tickers, "allocation": allocation, "metrics": metrics, "cumulative_returns": curve, "generated_at": dt.datetime.utcnow().isoformat() + "Z"}
