"""
Feature engineering for Black-Litterman views.
Faithfully converted from notebook cells 0, 3, and 4.
"""
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import yfinance as yf

_INFO_CACHE: Dict[str, Any] = {}
_CACHE_DIR = Path(__file__).resolve().parents[1] / "yf_cache"


def fetch_info(ticker: str, pause: float = 0.3) -> Dict[str, Any]:
    """Fetch yfinance ticker info with disk + memory caching."""
    if ticker in _INFO_CACHE:
        return _INFO_CACHE[ticker]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{ticker}.pkl"
    if cache_path.exists():
        try:
            data = pickle.loads(cache_path.read_bytes())
            _INFO_CACHE[ticker] = data
            return data
        except Exception:
            pass
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    try:
        cache_path.write_bytes(pickle.dumps(info))
    except Exception:
        pass
    time.sleep(pause)
    _INFO_CACHE[ticker] = info
    return info


def get_fundamental_features(ticker: str) -> Dict[str, float]:
    """Return trailing PE and price-to-book from yfinance info."""
    info = fetch_info(ticker)
    return {
        "trailingPE": float(info.get("trailingPE") or np.nan),
        "priceToBook": float(info.get("priceToBook") or np.nan),
    }


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index from notebook cell 3."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.rolling(window).mean()
    avg_l = loss.rolling(window).mean()
    rs = avg_g / avg_l
    return (100 - 100 / (1 + rs)).fillna(50)


def generate_asset_features(
    ticker: str,
    prices: pd.Series,
    fdata: Dict[str, float],
) -> pd.DataFrame:
    """
    Build per-stock feature DataFrame from notebook cell 3:
    price, SMA20, SMA50, 21-day momentum, daily return, RSI, PE, P/B.
    """
    df = pd.DataFrame(index=prices.index)
    df["price"] = prices
    df["SMA20"] = prices.rolling(20, min_periods=1).mean()
    df["SMA50"] = prices.rolling(50, min_periods=1).mean()
    df["mom21"] = prices.pct_change(21).fillna(0)
    df["daily_return"] = prices.pct_change().fillna(0)
    df["RSI"] = compute_rsi(prices)
    df["trailingPE"] = fdata.get("trailingPE", np.nan)
    df["priceToBook"] = fdata.get("priceToBook", np.nan)
    return df


def generate_poly_features(df: pd.DataFrame, degree: int = 1) -> pd.DataFrame:
    """Wrap sklearn PolynomialFeatures from notebook cell 4."""
    clean = df.fillna(0)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    arr = poly.fit_transform(clean)
    return pd.DataFrame(arr, index=clean.index, columns=poly.get_feature_names_out())
