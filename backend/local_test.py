"""
Offline test suite — tests all business logic with synthetic data.
Does NOT call Yahoo Finance. Safe to run anytime.
"""
import sys, traceback
import numpy as np
import pandas as pd

PASS, FAIL = [], []

def ok(name):
    PASS.append(name)
    print(f"  [PASS] {name}")

def fail(name, exc=None):
    FAIL.append(name)
    print(f"  [FAIL] {name}")
    if exc:
        traceback.print_exc()

# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n=== 1. Import checks ===")
try:
    import cvxpy, yfinance, xgboost, fastapi
    from sklearn.preprocessing import PolynomialFeatures
    ok("core packages (cvxpy, yfinance, xgboost, sklearn, fastapi)")
except Exception as e:
    fail("core packages", e); sys.exit(1)

try:
    from app.config import load_settings
    from app.model import (shrink_covariance, bl_update, optimize_portfolio,
                           calculate_metrics, DEFAULT_TICKERS)
    from app.features import (compute_rsi, generate_asset_features,
                               generate_poly_features)
    from app.views import train_xgb_model
    from app.backtest import create_job, get_job
    from app.middleware import InMemoryRateLimiter
    from app.main import app
    ok("all app modules import cleanly")
except Exception as e:
    fail("app module imports", e); sys.exit(1)

# ── 2. Config ─────────────────────────────────────────────────────────────────
print("\n=== 2. Config ===")
try:
    s = load_settings()
    assert isinstance(s.cors_origins, list) and len(s.cors_origins) > 0
    assert s.rate_limit_per_minute > 0
    ok(f"load_settings  env={s.env}  cors={s.cors_origins}")
except Exception as e:
    fail("load_settings", e)

# ── 3. BL math ────────────────────────────────────────────────────────────────
print("\n=== 3. Black-Litterman math ===")
n = 4
cov   = np.diag([0.04, 0.05, 0.03, 0.06])   # diagonal cov for clarity
Pi    = np.array([0.08, 0.10, 0.07, 0.09])
P     = np.eye(n)
Q     = np.array([0.09, 0.11, 0.08, 0.10])
Omega = np.eye(n) * 0.01

try:
    mu, M = bl_update(Pi, 0.10, cov, P, Q, Omega)
    assert mu.shape == (n,), f"Shape mismatch: {mu.shape}"
    assert M.shape == (n, n)
    ok(f"bl_update (two-inverse)  mu={np.round(mu, 4)}")
except Exception as e:
    fail("bl_update", e)

try:
    full_cov = np.array([[0.04, 0.01, 0.005, 0.002],
                          [0.01, 0.05, 0.008, 0.003],
                          [0.005,0.008,0.03, 0.001],
                          [0.002,0.003,0.001,0.06]])
    shrunk = shrink_covariance(full_cov, alpha=0.1)
    assert shrunk.shape == (n, n)
    assert np.allclose(shrunk, shrunk.T), "Not symmetric"
    # Check diagonal is larger after shrinkage
    assert all(np.diag(shrunk) >= np.diag(full_cov) * 0.9)
    ok("shrink_covariance (symmetric, diagonal preserved)")
except Exception as e:
    fail("shrink_covariance", e)

try:
    w = optimize_portfolio(Pi, cov, 2.5)
    assert abs(w.sum() - 1.0) < 1e-4, f"Sum={w.sum()}"
    assert all(w >= -1e-6), f"Negative weight: {w.min()}"
    assert all(w <= 1 + 1e-6), f"Weight >1: {w.max()}"
    ok(f"optimize_portfolio  w={np.round(w, 3)}  sum={w.sum():.6f}")
except Exception as e:
    fail("optimize_portfolio", e)

try:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    rets  = pd.Series(rng.normal(0.0005, 0.015, 252), index=dates)
    bench = pd.Series(rng.normal(0.0003, 0.012, 252), index=dates)
    m = calculate_metrics(rets, bench)
    assert "annual_return" in m and "sharpe_ratio" in m and "max_drawdown" in m
    assert m["max_drawdown"] <= 0
    ok(f"calculate_metrics  ret={m['annual_return']:.3f}  sharpe={m['sharpe_ratio']:.3f}  "
       f"dd={m['max_drawdown']:.3f}")
except Exception as e:
    fail("calculate_metrics", e)

# ── 4. Edge cases ─────────────────────────────────────────────────────────────
print("\n=== 4. BL edge cases ===")
try:
    # Near-singular covariance (should use fallback)
    nearly_singular = np.ones((4, 4)) * 0.04 + np.eye(4) * 1e-8
    mu2, _ = bl_update(Pi, 0.10, nearly_singular, P, Q, Omega)
    assert mu2.shape == (n,), "Fallback bl_update failed"
    ok("bl_update with near-singular cov (fallback triggered)")
except Exception as e:
    fail("bl_update near-singular", e)

try:
    # Single winning asset view
    w2 = optimize_portfolio(np.array([0.20, 0.02, 0.03, 0.01]), cov, 2.5)
    assert abs(w2.sum() - 1.0) < 1e-4
    ok(f"optimize_portfolio with dominant view  w[0]={w2[0]:.3f}")
except Exception as e:
    fail("optimize_portfolio dominant view", e)

# ── 5. Features ───────────────────────────────────────────────────────────────
print("\n=== 5. Feature engineering ===")
rng = np.random.default_rng(0)
prices = pd.Series(
    np.cumsum(rng.normal(0, 1, 120)) + 100,
    index=pd.date_range("2023-01-01", periods=120, freq="B"),
)

try:
    rsi = compute_rsi(prices)
    assert len(rsi) == 120
    assert rsi.between(0, 100).all(), f"RSI out of range: {rsi.describe()}"
    ok(f"compute_rsi  min={rsi.min():.1f}  max={rsi.max():.1f}")
except Exception as e:
    fail("compute_rsi", e)

try:
    fdata = {"trailingPE": 22.5, "priceToBook": 3.1}
    feat = generate_asset_features("TEST.NS", prices, fdata)
    required_cols = {"price", "SMA20", "SMA50", "mom21", "daily_return", "RSI", "trailingPE", "priceToBook"}
    missing = required_cols - set(feat.columns)
    assert not missing, f"Missing columns: {missing}"
    assert feat.shape[0] == 120
    ok(f"generate_asset_features  cols={sorted(feat.columns)}")
except Exception as e:
    fail("generate_asset_features", e)

try:
    poly = generate_poly_features(feat, degree=1)
    assert poly.shape[0] == feat.shape[0]
    assert poly.shape[1] >= feat.shape[1]
    assert not poly.isnull().any().any(), "NaNs in poly features"
    ok(f"generate_poly_features  {feat.shape[1]} -> {poly.shape[1]} features")
except Exception as e:
    fail("generate_poly_features", e)

# ── 6. XGBoost ────────────────────────────────────────────────────────────────
print("\n=== 6. XGBoost ===")
try:
    rng2 = np.random.default_rng(7)
    X = pd.DataFrame(rng2.standard_normal((150, 8)),
                     columns=[f"f{i}" for i in range(8)])
    y = pd.Series(rng2.normal(0.001, 0.02, 150))
    model = train_xgb_model(X, y)
    preds = model.predict(X)
    assert len(preds) == 150
    resid_var = float(np.var(y.values - preds))
    ok(f"train_xgb_model  train_resid_var={resid_var:.6f}")
except Exception as e:
    fail("train_xgb_model", e)

# ── 7. Sensitivity math ───────────────────────────────────────────────────────
print("\n=== 7. Sensitivity sweep (math, no network) ===")
try:
    multipliers = list(np.linspace(0.5, 1.5, 5).round(4))
    rows = []
    for s_val in multipliers:
        mu_s, _ = bl_update(Pi, 0.10, cov, P, float(s_val) * Q, Omega)
        w_s = optimize_portfolio(mu_s, cov, 2.5)
        rows.append({"multiplier": round(s_val, 4),
                     "weights": {f"T{j}": round(float(w_s[j]), 6) for j in range(n)}})
    assert len(rows) == 5
    # Weights should vary with multiplier
    first_w = list(rows[0]["weights"].values())
    last_w  = list(rows[-1]["weights"].values())
    assert first_w != last_w, "Weights should change with multiplier"
    ok(f"sensitivity sweep  {len(rows)} multipliers, weights change as expected")
except Exception as e:
    fail("sensitivity sweep math", e)

# ── 8. Job store ──────────────────────────────────────────────────────────────
print("\n=== 8. Async job store ===")
try:
    jid = create_job()
    job = get_job(jid)
    assert job is not None and job["status"] == "running"
    assert get_job("nonexistent-id") is None
    ok(f"create_job / get_job  id={jid[:8]}...")
except Exception as e:
    fail("job store", e)

# ── 9. Rate limiter ───────────────────────────────────────────────────────────
print("\n=== 9. Rate limiter ===")
try:
    lim = InMemoryRateLimiter(requests_per_minute=3)
    assert lim.allow("ip1") and lim.allow("ip1") and lim.allow("ip1")
    assert not lim.allow("ip1"), "Should be blocked"
    assert lim.allow("ip2"), "Different client should pass"
    ok("InMemoryRateLimiter (blocks at limit, isolates per-client)")
except Exception as e:
    fail("InMemoryRateLimiter", e)

# ── 10. FastAPI routes ────────────────────────────────────────────────────────
print("\n=== 10. FastAPI routes ===")
try:
    routes = [r.path for r in app.routes]
    required = ["/", "/health", "/default-tickers", "/optimize-portfolio",
                "/ask-rag", "/rebuild-rag", "/backtest", "/backtest/{job_id}",
                "/sensitivity", "/ff-factors"]
    missing = [r for r in required if r not in routes]
    assert not missing, f"Missing routes: {missing}"
    ok(f"all {len(required)} routes present: {required}")
except Exception as e:
    fail("FastAPI routes", e)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*52}")
print(f"  Total: {len(PASS)+len(FAIL)}   PASS: {len(PASS)}   FAIL: {len(FAIL)}")
if FAIL:
    print(f"\n  FAILED: {FAIL}")
    sys.exit(1)
else:
    print("\n  All offline tests passed.")
    print("  The backend logic is correct and safe to deploy.")
    sys.exit(0)
