"""
Fama-French 5-factor model computed from NIFTY data.
Faithfully converted from notebook cell 1.
"""
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .features import fetch_info
from .model import NIFTY50


def compute_ff_factors(
    tickers: Optional[List[str]] = None,
    risk_free_rate: float = 0.0667,
) -> Dict[str, float]:
    """
    Compute Fama-French 5-factor model components from NIFTY constituent data.

    Returns a dict with keys: Mkt-RF, SMB, HML, RMW, CMA, RF.
    Uses monthly data for the trailing 1-year period.
    """
    universe = tickers or NIFTY50

    price_data = yf.download(
        universe, period="1y", interval="1mo", progress=False, auto_adjust=False
    )["Close"]
    # Keep tickers that actually downloaded
    available = [t for t in universe if t in price_data.columns]
    price_data = price_data[available].dropna(how="all")

    monthly_returns = price_data.pct_change().dropna()
    avg_monthly_return = monthly_returns.mean()

    data_list = []
    for ticker in available:
        info = fetch_info(ticker)
        market_cap = info.get("marketCap", np.nan)
        book_value = info.get("bookValue", np.nan)
        price = (
            float(price_data[ticker].iloc[-1])
            if ticker in price_data.columns
            else np.nan
        )
        btom = (
            book_value / price
            if pd.notnull(book_value) and pd.notnull(price) and price != 0
            else np.nan
        )
        op_margin = info.get("operatingMargins", np.nan)

        # Asset growth from balance sheet
        asset_growth = np.nan
        try:
            bs = yf.Ticker(ticker).balance_sheet
            if not bs.empty and "Total Assets" in bs.index and bs.shape[1] >= 2:
                dates = bs.columns.sort_values(ascending=False)
                latest = bs.loc["Total Assets", dates[0]]
                prev = bs.loc["Total Assets", dates[1]]
                if prev and prev != 0:
                    asset_growth = float((latest - prev) / prev)
        except Exception:
            pass

        ret = float(avg_monthly_return.get(ticker, np.nan))
        data_list.append(
            {
                "Ticker": ticker,
                "MarketCap": market_cap,
                "BookValue": book_value,
                "BookToMarket": btom,
                "OpMargin": op_margin,
                "AssetGrowth": asset_growth,
                "Return": ret,
            }
        )

    df = pd.DataFrame(data_list).dropna(
        subset=["MarketCap", "BookToMarket", "Return", "OpMargin", "AssetGrowth"]
    )

    if df.empty:
        return {
            "Mkt-RF": 0.0,
            "SMB": 0.0,
            "HML": 0.0,
            "RMW": 0.0,
            "CMA": 0.0,
            "RF": risk_free_rate,
            "note": "Insufficient data to compute factors",
        }

    df["Size"] = np.where(df["MarketCap"] < df["MarketCap"].median(), "Small", "Big")
    df["Value"] = np.where(
        df["BookToMarket"] > df["BookToMarket"].median(), "High", "Low"
    )
    df["Profitability"] = np.where(
        df["OpMargin"] > df["OpMargin"].median(), "Robust", "Weak"
    )
    df["Investment"] = np.where(
        df["AssetGrowth"] < df["AssetGrowth"].median(), "Conservative", "Aggressive"
    )

    def _spread(df: pd.DataFrame, col: str, val_a: str, val_b: str) -> float:
        a = df[df[col] == val_a]["Return"].mean()
        b = df[df[col] == val_b]["Return"].mean()
        return float(a - b)

    smb = _spread(df, "Size", "Small", "Big")
    hml = _spread(df, "Value", "High", "Low")
    rmw = _spread(df, "Profitability", "Robust", "Weak")
    cma = _spread(df, "Investment", "Conservative", "Aggressive")
    mkt_rf = float(df["Return"].mean()) - risk_free_rate

    return {
        "Mkt-RF": round(mkt_rf, 6),
        "SMB": round(smb, 6),
        "HML": round(hml, 6),
        "RMW": round(rmw, 6),
        "CMA": round(cma, 6),
        "RF": risk_free_rate,
    }
