"""
XGBoost-based return views for Black-Litterman.
Faithfully converted from notebook cells 4 and 5.
"""
from typing import Tuple

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
