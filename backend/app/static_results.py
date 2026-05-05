"""
Pre-computed results from the original Black-Litterman notebook.
These are served instantly — no Yahoo Finance calls, no computation.
"""

NOTEBOOK_RESULTS = {
    "benchmark": {
        "name": "NIFTY 50 (^NSEI)",
        "annual_return": 0.2093,
        "description": (
            "The NIFTY 50 is a market-capitalisation-weighted index of the 50 largest "
            "companies listed on the National Stock Exchange of India. It is the standard "
            "benchmark for large-cap Indian equity portfolios."
        ),
    },
    "models": [
        {
            "name": "All 50 With Views",
            "annual_return": 0.41015,
            "annual_volatility": 0.144877,
            "sharpe_ratio": 2.363724,
            "max_drawdown": -0.102846,
            "excess_return_vs_benchmark": 0.200862,
        },
        {
            "name": "Top 15 With Views",
            "annual_return": 0.268175,
            "annual_volatility": 0.137963,
            "sharpe_ratio": 1.4531,
            "max_drawdown": -0.130333,
            "excess_return_vs_benchmark": 0.058887,
        },
        {
            "name": "Top 10 With Views",
            "annual_return": 0.233745,
            "annual_volatility": 0.143096,
            "sharpe_ratio": 1.160372,
            "max_drawdown": -0.135435,
            "excess_return_vs_benchmark": 0.024457,
        },
        {
            "name": "Benchmark (NIFTY 50)",
            "annual_return": 0.2093,
            "annual_volatility": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "excess_return_vs_benchmark": 0.0,
        },
    ],
    # Final portfolio weights at view multiplier s=1.5 (All-50 model)
    "weights": [
        {"ticker": "ICICIBANK.NS",  "weight": 0.2063},
        {"ticker": "BAJFINANCE.NS", "weight": 0.1976},
        {"ticker": "HDFCBANK.NS",   "weight": 0.1183},
        {"ticker": "DIVISLAB.NS",   "weight": 0.0958},
        {"ticker": "BHARTIARTL.NS", "weight": 0.0751},
        {"ticker": "SHREECEM.NS",   "weight": 0.0498},
        {"ticker": "KOTAKBANK.NS",  "weight": 0.0415},
        {"ticker": "TECHM.NS",      "weight": 0.0399},
        {"ticker": "SUNPHARMA.NS",  "weight": 0.0380},
        {"ticker": "CIPLA.NS",      "weight": 0.0256},
        {"ticker": "HINDUNILVR.NS", "weight": 0.0248},
        {"ticker": "INFY.NS",       "weight": 0.0220},
        {"ticker": "MARUTI.NS",     "weight": 0.0216},
        {"ticker": "HCLTECH.NS",    "weight": 0.0196},
        {"ticker": "HDFCLIFE.NS",   "weight": 0.0169},
        {"ticker": "BAJAJFINSV.NS", "weight": 0.0072},
    ],
    # Sensitivity analysis: how key allocations shift as the view multiplier changes
    "sensitivity": {
        "multipliers": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        "series": [
            {
                "ticker": "ICICIBANK.NS",
                "weights": [0.1386, 0.1463, 0.1539, 0.1615, 0.1690, 0.1761,
                            0.1822, 0.1883, 0.1945, 0.2004, 0.2063],
            },
            {
                "ticker": "BAJFINANCE.NS",
                "weights": [0.0588, 0.0731, 0.0874, 0.1017, 0.1160, 0.1303,
                            0.1445, 0.1588, 0.1730, 0.1856, 0.1976],
            },
            {
                "ticker": "HDFCBANK.NS",
                "weights": [0.1722, 0.1675, 0.1628, 0.1581, 0.1533, 0.1482,
                            0.1424, 0.1366, 0.1308, 0.1252, 0.1183],
            },
            {
                "ticker": "DIVISLAB.NS",
                "weights": [0.0615, 0.0648, 0.0682, 0.0718, 0.0754, 0.0789,
                            0.0822, 0.0856, 0.0889, 0.0924, 0.0958],
            },
            {
                "ticker": "BHARTIARTL.NS",
                "weights": [0.0383, 0.0418, 0.0453, 0.0488, 0.0523, 0.0558,
                            0.0595, 0.0631, 0.0668, 0.0709, 0.0751],
            },
        ],
    },
    "notes": (
        "Results are from a ~5-year rolling-window backtest (July 2020–July 2025) using the "
        "Black-Litterman model with XGBoost-generated views. The 'All 50 With Views' "
        "portfolio achieves a Sharpe ratio of 2.36 vs the NIFTY 50 benchmark, "
        "representing ~20% excess annualised return."
    ),
}
