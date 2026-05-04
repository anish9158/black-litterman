"""
News sentiment scoring using yfinance headlines + Groq LLM zero-shot classification.

Pipeline per ticker:
  1. Fetch recent headlines via yfinance Ticker.news
  2. If GROQ_API_KEY present  → ask Groq to rate overall sentiment as float in [-1, 1]
     Else                     → simple keyword heuristic
  3. Map score → signal: "bullish" | "neutral" | "bearish"
"""
import json
import os
import time
from typing import Dict, List, Any

import yfinance as yf

from .config import load_settings

settings = load_settings()

_POSITIVE_WORDS = {
    "surge", "rally", "gain", "profit", "growth", "beat", "strong",
    "upgrade", "buy", "rise", "record", "high", "outperform", "boost",
    "expand", "winning", "positive", "exceed",
}
_NEGATIVE_WORDS = {
    "fall", "drop", "loss", "miss", "weak", "downgrade", "sell",
    "decline", "low", "cut", "risk", "fear", "crash", "warning",
    "underperform", "loss", "negative", "concern",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_sentiment(headlines: List[str]) -> float:
    """Fast keyword heuristic. Returns float in [-1, 1]."""
    score = 0.0
    count = 0
    for h in headlines:
        words = set(h.lower().split())
        pos = len(words & _POSITIVE_WORDS)
        neg = len(words & _NEGATIVE_WORDS)
        if pos + neg > 0:
            score += (pos - neg) / (pos + neg)
            count += 1
    return round(score / count, 4) if count > 0 else 0.0


def _llm_sentiment(headlines: List[str], api_key: str, model: str) -> float:
    """
    Zero-shot Groq classification of a batch of headlines.
    Returns float in [-1, 1]. Falls back to keyword score on parse error.
    """
    try:
        from langchain_openai import ChatOpenAI

        sample = headlines[:12]
        headline_text = "\n".join(f"- {h}" for h in sample)
        prompt = (
            "Rate the overall investor sentiment for a stock based on these recent headlines.\n"
            "Reply with ONLY a JSON object, e.g.: {\"score\": 0.4}\n"
            "score must be a float from -1.0 (very bearish) to 1.0 (very bullish). 0.0 is neutral.\n\n"
            f"Headlines:\n{headline_text}\n\nJSON:"
        )

        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
        )
        raw = llm.invoke([("human", prompt)]).content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        return float(max(-1.0, min(1.0, data.get("score", 0.0))))
    except Exception:
        return _keyword_sentiment(headlines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_stock_sentiment(ticker: str) -> Dict[str, Any]:
    """
    Fetch sentiment for a single ticker.

    Returns
    -------
    dict with keys: ticker, score, signal, headlines_count, (optional) error
    """
    try:
        news_items = yf.Ticker(ticker).news or []
        headlines = [
            item.get("content", {}).get("title") or item.get("title", "")
            for item in news_items[:15]
        ]
        headlines = [h for h in headlines if h]

        if not headlines:
            return {
                "ticker": ticker,
                "score": 0.0,
                "signal": "neutral",
                "headlines_count": 0,
                "headlines": [],
            }

        api_key = os.getenv("GROQ_TOKEN") or settings.groq_api_key
        score = (
            _llm_sentiment(headlines, api_key, settings.groq_model)
            if api_key
            else _keyword_sentiment(headlines)
        )
        signal = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"

        return {
            "ticker": ticker,
            "score": round(score, 4),
            "signal": signal,
            "headlines_count": len(headlines),
            "headlines": headlines[:5],  # return preview
        }
    except Exception as exc:
        return {
            "ticker": ticker,
            "score": 0.0,
            "signal": "neutral",
            "headlines_count": 0,
            "headlines": [],
            "error": str(exc),
        }


def get_portfolio_sentiment(tickers: List[str]) -> Dict[str, Any]:
    """
    Fetch sentiment for every ticker in the list.

    Returns
    -------
    {
        "sentiment": {ticker: {...}},
        "portfolio_score": float,
        "portfolio_signal": "bullish"|"neutral"|"bearish",
    }
    """
    results: Dict[str, Any] = {}
    for ticker in tickers:
        results[ticker] = get_stock_sentiment(ticker)
        time.sleep(0.2)  # polite pause between yfinance calls

    scores = [v["score"] for v in results.values()]
    avg = sum(scores) / len(scores) if scores else 0.0
    portfolio_signal = "bullish" if avg > 0.1 else "bearish" if avg < -0.1 else "neutral"

    return {
        "sentiment": results,
        "portfolio_score": round(avg, 4),
        "portfolio_signal": portfolio_signal,
    }
