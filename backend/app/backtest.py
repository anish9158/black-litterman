"""
Rolling-window backtest, sensitivity analysis, and async job store.
Faithfully converted from notebook cells 5, 6, and 7.
"""
import asyncio
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .features import get_fundamental_features, generate_asset_features, generate_poly_features
from .model import (
    batch_download,
    compute_returns,
    compute_equilibrium_returns,
    shrink_covariance,
    bl_update,
    optimize_portfolio,
    calculate_metrics,
    NIFTY50,
)
from .views import train_xgb_model

# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
_JOBS: Dict[str, Dict[str, Any]] = {}


def create_job() -> str:
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {"status": "running", "result": None, "error": None}
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _JOBS.get(job_id)


def _finish_job(job_id: str, result: Any) -> None:
    if job_id in _JOBS:
        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["result"] = result


def _fail_job(job_id: str, error: str) -> None:
    if job_id in _JOBS:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"] = error


# ---------------------------------------------------------------------------
# Rolling-window backtest (notebook cell 5 + 6)
# ---------------------------------------------------------------------------

def _rolling_backtest_sync(
    tickers: Optional[List[str]],
    period: str,
    window_train: int,
    window_test: int,
    risk_aversion: float,
    tau: float,
    use_views: bool,
) -> Dict[str, Any]:
    """Synchronous implementation; run inside asyncio.to_thread to avoid blocking."""
    tickers = tickers or NIFTY50[:15]
    tickers = [t.strip().upper() for t in tickers if t.strip()]

    stocks = batch_download(tickers, period=period)
    returns = compute_returns(stocks)
    tickers = list(returns.columns)
    n_assets = len(tickers)
    if n_assets < 2:
        raise ValueError("Need at least two valid tickers.")

    market_weights = np.ones(n_assets) / n_assets
    P = np.eye(n_assets)
    dates = returns.index
    total = len(dates)

    port_rets: List[pd.Series] = []
    weights_hist: List[np.ndarray] = []
    weights_dates: List[Any] = []

    # Fundamental features fetched once
    fund_dict = {t: get_fundamental_features(t) for t in tickers}

    step = window_test
    for start in range(0, total - window_train - window_test, step):
        train_dates = dates[start: start + window_train]
        test_dates = dates[start + window_train: start + window_train + window_test]
        R_train = returns.loc[train_dates]

        preds = np.zeros((len(test_dates), n_assets))
        res_vars = np.full(n_assets, 0.05)

        for i, ticker in enumerate(tickers):
            try:
                prices_tr = stocks[ticker].loc[train_dates]
                prices_ts = stocks[ticker].loc[test_dates]
                fdata = fund_dict[ticker]
                X_tr_raw = generate_asset_features(ticker, prices_tr, fdata)
                X_ts_raw = generate_asset_features(ticker, prices_ts, fdata)
                X_tr = generate_poly_features(X_tr_raw, degree=1)
                X_ts = generate_poly_features(X_ts_raw, degree=1)
                if use_views:
                    m = train_xgb_model(X_tr, R_train[ticker])
                    preds[:, i] = m.predict(X_ts)
                    resid = R_train[ticker].values - m.predict(X_tr)
                    res_vars[i] = float(np.var(resid))
            except Exception:
                preds[:, i] = R_train[ticker].mean() if ticker in R_train else 0.0

        res_vars = np.maximum(res_vars, 1e-4)
        Omega = np.diag(res_vars)
        Sigma = shrink_covariance(R_train.cov().values, alpha=0.1)
        Pi = compute_equilibrium_returns(Sigma, market_weights, risk_aversion)

        if use_views:
            Q = preds.mean(axis=0)
            Q = 0.8 * Q + 0.2 * Pi
        else:
            Q = Pi

        mu_post, _ = bl_update(Pi, tau, Sigma, P, Q, Omega)
        w_opt = optimize_portfolio(mu_post, Sigma, risk_aversion)

        sub = returns.loc[test_dates]
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        arr = sub.values @ w_opt
        ret = pd.Series(arr, index=sub.index)
        port_rets.append(ret)
        weights_hist.append(w_opt)
        weights_dates.append(test_dates[0])

    if not port_rets:
        raise ValueError("Backtest produced no windows — period too short.")

    portfolio_returns = pd.concat(port_rets)
    benchmark_series = returns.mean(axis=1).reindex(portfolio_returns.index).dropna()
    metrics = calculate_metrics(portfolio_returns, benchmark_series)

    # Capture last-window params for sensitivity analysis
    last_train_start = total - window_train - window_test
    last_train_dates = dates[max(0, last_train_start): total - window_test]
    R_last = returns.loc[last_train_dates]
    cov_last = shrink_covariance(R_last.cov().values, alpha=0.1)
    Pi_last = compute_equilibrium_returns(cov_last, market_weights, risk_aversion)

    preds_last = np.zeros((window_test, n_assets))
    for i, ticker in enumerate(tickers):
        try:
            prices_tr = stocks[ticker].loc[last_train_dates]
            prices_ts = stocks[ticker].iloc[-window_test:]
            fdata = fund_dict[ticker]
            X_tr_raw = generate_asset_features(ticker, prices_tr, fdata)
            X_ts_raw = generate_asset_features(ticker, prices_ts, fdata)
            X_tr = generate_poly_features(X_tr_raw, degree=1)
            X_ts = generate_poly_features(X_ts_raw, degree=1)
            if use_views:
                m = train_xgb_model(X_tr, R_last[ticker])
                preds_last[:, i] = m.predict(X_ts)
        except Exception:
            preds_last[:, i] = R_last[ticker].mean() if ticker in R_last else 0.0

    Q_last = preds_last.mean(axis=0)

    cumulative = (1 + portfolio_returns).cumprod()
    cumulative_series = [
        {"date": str(idx.date()), "value": round(float(v), 6)}
        for idx, v in cumulative.items()
    ]
    weights_history = [
        {
            "date": str(d.date()) if hasattr(d, "date") else str(d),
            "weights": {tickers[j]: round(float(weights_hist[k][j]), 6) for j in range(n_assets)},
        }
        for k, d in enumerate(weights_dates)
    ]

    return {
        "tickers": tickers,
        "cumulative_returns": cumulative_series,
        "metrics": metrics,
        "weights_history": weights_history,
        # Stored for sensitivity endpoint
        "_last_params": {
            "Pi": Pi_last.tolist(),
            "Q_last": Q_last.tolist(),
            "cov": cov_last.tolist(),
            "Omega_diag": np.diag(np.diag(np.diag(res_vars))).diagonal().tolist(),
            "tickers": tickers,
            "risk_aversion": risk_aversion,
            "tau": tau,
        },
    }


async def run_backtest_async(
    job_id: str,
    tickers: Optional[List[str]],
    period: str,
    window_train: int,
    window_test: int,
    risk_aversion: float,
    tau: float,
    use_views: bool,
) -> None:
    """Launch backtest in a thread so FastAPI event loop stays responsive."""
    try:
        result = await asyncio.to_thread(
            _rolling_backtest_sync,
            tickers,
            period,
            window_train,
            window_test,
            risk_aversion,
            tau,
            use_views,
        )
        _finish_job(job_id, result)
    except Exception as exc:
        _fail_job(job_id, str(exc))


# ---------------------------------------------------------------------------
# Sensitivity analysis (notebook cells 7 and 8) — fast, runs in-request
# ---------------------------------------------------------------------------

def run_sensitivity(
    tickers: Optional[List[str]] = None,
    period: str = "2y",
    risk_aversion: float = 2.5,
    tau: float = 0.10,
    multipliers: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    View-multiplier sensitivity sweep from notebook cell 7.
    Computes Pi and Q from fresh data; no prior backtest required.
    """
    if multipliers is None:
        multipliers = list(np.linspace(0.5, 1.5, 11).round(4))

    tickers = tickers or NIFTY50[:15]
    tickers = [t.strip().upper() for t in tickers if t.strip()]

    stocks = batch_download(tickers, period=period)
    returns = compute_returns(stocks)
    tickers = list(returns.columns)
    n = len(tickers)
    if n < 2:
        raise ValueError("Need at least two valid tickers.")

    market_weights = np.ones(n) / n
    P = np.eye(n)
    cov = shrink_covariance(returns.cov().values, alpha=0.1)
    Pi = compute_equilibrium_returns(cov, market_weights, risk_aversion)

    # Simple Q: recent 30-day mean annualised
    Q = returns.tail(30).mean().values * 252
    Omega = np.diag(np.maximum(np.diag(tau * cov), 1e-4))

    rows = []
    for s in multipliers:
        mu_s, _ = bl_update(Pi, tau, cov, P, float(s) * Q, Omega)
        w_s = optimize_portfolio(mu_s, cov, risk_aversion)
        rows.append({"multiplier": round(float(s), 4), "weights": {tickers[j]: round(float(w_s[j]), 6) for j in range(n)}})

    return {"tickers": tickers, "sensitivity": rows}
