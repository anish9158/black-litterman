"""
Black-Litterman model with full two-inverse formula and covariance shrinkage.
Faithfully converted from notebook cell 4. XGBoost views hook added.
"""
import datetime as dt
from typing import List, Dict, Any, Optional

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
    "ULTRACEMCO.NS", "WIPRO.NS",
]

DEFAULT_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS",
]


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


def compute_equilibrium_returns(
    cov_matrix: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    return risk_aversion * cov_matrix @ market_weights


def shrink_covariance(cov: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Diagonal shrinkage: (1-α)*Σ + α*diag(Σ) from notebook cell 5."""
    return (1 - alpha) * cov + alpha * np.diag(np.diag(cov))


def bl_update(
    Pi: np.ndarray,
    tau: float,
    cov_matrix: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
) -> tuple:
    """
    Full two-inverse Black-Litterman formula from notebook cell 4.
    Returns (posterior_mean, posterior_covariance).
    """
    inv_tau = np.linalg.inv(tau * cov_matrix)
    inv_O = np.linalg.inv(Omega)
    M = np.linalg.inv(inv_tau + P.T @ inv_O @ P)
    mu = M @ (inv_tau @ Pi + P.T @ inv_O @ Q)
    return mu, M


def optimize_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    n = len(expected_returns)
    w = cp.Variable(n)
    objective = cp.Minimize(
        risk_aversion * cp.quad_form(w, cov_matrix) - expected_returns @ w
    )
    constraints = [cp.sum(w) == 1, w >= 0, w <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        return np.ones(n) / n
    weights = np.maximum(np.asarray(w.value).ravel(), 0)
    total = weights.sum()
    return weights / total if total > 0 else np.ones(n) / n


def calculate_metrics(
    port_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0677,
) -> Dict[str, float]:
    if port_returns.empty:
        return {}
    annual_ret = float(port_returns.mean() * 252)
    annual_vol = float(port_returns.std() * np.sqrt(252))
    sharpe = float((annual_ret - risk_free_rate) / annual_vol) if annual_vol else 0.0
    cum = (1 + port_returns).cumprod()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    bench = benchmark_returns.reindex(port_returns.index).dropna()
    bench_ret = float(bench.mean() * 252) if not bench.empty else 0.0
    return {
        "annual_return": annual_ret,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "benchmark_annual_return": bench_ret,
        "final_cumulative_return": float(cum.iloc[-1]),
    }


def run_black_litterman(
    tickers: Optional[List[str]] = None,
    period: str = "1y",
    risk_aversion: float = 2.5,
    tau: float = 0.10,
    use_xgb_views: bool = False,
) -> Dict[str, Any]:
    tickers = tickers or DEFAULT_TICKERS
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    prices = batch_download(tickers, period=period)
    returns = compute_returns(prices)
    tickers = list(returns.columns)
    n = len(tickers)
    if n < 2:
        raise ValueError("At least two valid tickers are required.")

    # Shrinkage covariance (annualised)
    raw_cov = returns.cov().values * 252
    cov = shrink_covariance(raw_cov, alpha=0.1)

    market_weights = np.ones(n) / n
    Pi = compute_equilibrium_returns(cov, market_weights, risk_aversion)

    P = np.eye(n)

    if use_xgb_views:
        from .views import generate_views  # deferred import to avoid slow startup
        train_window = min(150, len(returns) - 30)
        test_window = min(30, len(returns) - train_window)
        if train_window > 0 and test_window > 0:
            train_dates = returns.index[:train_window]
            test_dates = returns.index[train_window: train_window + test_window]
            Q, res_vars = generate_views(prices, returns, train_dates, test_dates)
            # View-shrinkage: blend prediction toward equilibrium
            Q = 0.8 * Q + 0.2 * Pi
            Omega = np.diag(res_vars)
        else:
            Q = returns.tail(30).mean().values * 252
            Omega = np.diag(np.diag(tau * cov))
    else:
        Q = returns.tail(min(30, len(returns))).mean().values * 252
        Omega = np.diag(np.diag(tau * cov))

    posterior_mu, _ = bl_update(Pi, tau, cov, P, Q, Omega)
    weights = optimize_portfolio(posterior_mu, cov, risk_aversion)

    port_returns = returns @ weights
    benchmark_returns = returns.mean(axis=1)
    metrics = calculate_metrics(port_returns, benchmark_returns)
    allocation = [
        {"ticker": t, "weight": float(w)}
        for t, w in sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)
    ]
    curve = [
        {"date": str(idx.date()), "value": float(val)}
        for idx, val in (1 + port_returns).cumprod().items()
    ]
    return {
        "tickers": tickers,
        "allocation": allocation,
        "metrics": metrics,
        "cumulative_returns": curve,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
    }
