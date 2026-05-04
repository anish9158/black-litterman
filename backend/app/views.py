"""
XGBoost-based return views for Black-Litterman.
Faithfully converted from notebook cells 4 and 5.
"""
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import get_fundamental_features, generate_asset_features, generate_poly_features


def train_xgb_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """XGBoost regressor config from notebook cell 4."""
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def generate_views(
    stocks_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train per-stock XGBoost on train_dates, predict on test_dates.

    Returns
    -------
    Q         : shape (n_assets,) — mean predicted return over test window
    res_vars  : shape (n_assets,) — per-stock residual variance (diagonal of Omega)
    """
    n_assets = returns_df.shape[1]
    n_test = len(test_dates)
    preds = np.zeros((n_test, n_assets))
    res_vars = np.full(n_assets, 0.05)

    for i, ticker in enumerate(returns_df.columns):
        try:
            prices_tr = stocks_df[ticker].loc[train_dates]
            prices_ts = stocks_df[ticker].loc[test_dates]
            fdata = get_fundamental_features(ticker)
            X_tr_raw = generate_asset_features(ticker, prices_tr, fdata)
            X_ts_raw = generate_asset_features(ticker, prices_ts, fdata)
            X_tr = generate_poly_features(X_tr_raw, degree=1)
            X_ts = generate_poly_features(X_ts_raw, degree=1)
            r_train = returns_df[ticker].loc[train_dates]
            model = train_xgb_model(X_tr, r_train)
            preds[:, i] = model.predict(X_ts)
            resid = r_train.values - model.predict(X_tr)
            res_vars[i] = float(np.var(resid))
        except Exception:
            # Fall back to simple mean return for this asset
            preds[:, i] = returns_df[ticker].loc[train_dates].mean()

    res_vars = np.maximum(res_vars, 1e-4)
    Q = preds.mean(axis=0)
    return Q, res_vars


def explain_views(
    tickers: List[str],
    period: str = "1y",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Train per-stock XGBoost on the most recent data window and compute
    SHAP feature importances for each ticker.

    Returns
    -------
    Dict mapping ticker → list of top-10 feature dicts:
        {feature, shap_value, mean_abs_shap, feature_value}
    Falls back to XGBoost built-in gain importances if shap is unavailable.
    """
    # Function-level import avoids circular dependency (model imports views)
    from .model import batch_download

    prices = batch_download(tickers, period)
    if prices.empty:
        return {}

    returns = prices.pct_change().dropna()
    valid = [t for t in tickers if t in returns.columns and t in prices.columns]

    result: Dict[str, List[Dict[str, Any]]] = {}
    for ticker in valid:
        try:
            prices_t = prices[ticker].dropna()
            fdata = get_fundamental_features(ticker)
            X_raw = generate_asset_features(ticker, prices_t, fdata)
            X = generate_poly_features(X_raw, degree=1)
            r = returns[ticker]

            # Align indices
            common_idx = X.index.intersection(r.index)
            X = X.loc[common_idx]
            r = r.loc[common_idx]

            split = int(len(X) * 0.8)
            if split < 20 or len(X) - split < 2:
                continue

            X_train, X_test = X.iloc[:split], X.iloc[split:]
            r_train = r.iloc[:split]
            model = train_xgb_model(X_train, r_train)
            feat_names = list(X.columns)

            try:
                import shap  # optional dependency
                explainer = shap.TreeExplainer(model)
                shap_matrix = explainer.shap_values(X_test)  # (n_test, n_feats)
                mean_abs = np.abs(shap_matrix).mean(axis=0)
                mean_signed = shap_matrix.mean(axis=0)
            except Exception:
                # Fallback: use XGBoost gain importance as shap_value proxy
                imp = model.get_booster().get_score(importance_type="gain")
                mean_abs = np.array([imp.get(f"f{i}", 0.0) for i in range(len(feat_names))])
                max_imp = mean_abs.max() or 1.0
                mean_abs = mean_abs / max_imp * 0.01  # normalise to small floats
                mean_signed = mean_abs.copy()

            top_idx = np.argsort(mean_abs)[::-1][:10]
            last_row = X_test.iloc[-1]
            top_feats = [
                {
                    "feature": feat_names[idx],
                    "shap_value": round(float(mean_signed[idx]), 6),
                    "mean_abs_shap": round(float(mean_abs[idx]), 6),
                    "feature_value": round(float(last_row.iloc[idx]), 4),
                }
                for idx in top_idx
            ]
            result[ticker] = top_feats
        except Exception:
            continue

    return result
