from typing import List, Optional
import asyncio

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import load_settings
from .middleware import (
    InMemoryRateLimiter,
    RequestSizeLimitMiddleware,
    get_rate_limit_dependency,
)
from .model import run_black_litterman, DEFAULT_TICKERS
from .rag import ask_rag, get_vectorstore
from .backtest import (
    create_job,
    get_job,
    run_backtest_async,
    run_sensitivity,
)
from .fama_french import compute_ff_factors

app = FastAPI(title="Black-Litterman RAG API", version="2.0.0")
settings = load_settings()
rate_limiter = InMemoryRateLimiter(requests_per_minute=settings.rate_limit_per_minute)
enforce_rate_limit = get_rate_limit_dependency(rate_limiter)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=settings.max_request_size_bytes,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    tickers: Optional[List[str]] = Field(default=None)
    period: str = "1y"
    risk_aversion: float = 2.5
    tau: float = 0.10
    use_xgb_views: bool = False


class ChatRequest(BaseModel):
    question: str
    k: int = 5
    history: Optional[List[dict]] = None


class BacktestRequest(BaseModel):
    tickers: Optional[List[str]] = None
    period: str = "5y"
    window_train: int = 150
    window_test: int = 30
    risk_aversion: float = 2.5
    tau: float = 0.10
    use_views: bool = True


class SensitivityRequest(BaseModel):
    tickers: Optional[List[str]] = None
    period: str = "2y"
    risk_aversion: float = 2.5
    tau: float = 0.10
    multipliers: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Existing endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"message": "Black-Litterman RAG API is live", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.env}


@app.get("/default-tickers")
def default_tickers():
    return {"tickers": DEFAULT_TICKERS}


@app.post("/optimize-portfolio")
def optimize_portfolio(req: OptimizeRequest, _: None = Depends(enforce_rate_limit)):
    try:
        return run_black_litterman(
            req.tickers,
            req.period,
            req.risk_aversion,
            req.tau,
            req.use_xgb_views,
        )
    except Exception as exc:
        detail = str(exc) if not settings.is_production else "Optimization failed"
        raise HTTPException(status_code=400, detail=detail)


@app.post("/ask-rag")
def rag_chat(req: ChatRequest, _: None = Depends(enforce_rate_limit)):
    try:
        return ask_rag(req.question, req.k, req.history)
    except Exception as exc:
        detail = str(exc) if not settings.is_production else "RAG request failed"
        raise HTTPException(status_code=400, detail=detail)


@app.post("/rebuild-rag")
def rebuild_rag(_: None = Depends(enforce_rate_limit)):
    store = get_vectorstore(force_rebuild=True)
    return {"status": "rebuilt", "chunks": store.index.ntotal}


# ---------------------------------------------------------------------------
# New endpoints
# ---------------------------------------------------------------------------

@app.post("/backtest")
async def start_backtest(req: BacktestRequest, _: None = Depends(enforce_rate_limit)):
    """Start a rolling-window backtest in the background. Returns a job_id to poll."""
    job_id = create_job()
    asyncio.create_task(
        run_backtest_async(
            job_id=job_id,
            tickers=req.tickers,
            period=req.period,
            window_train=req.window_train,
            window_test=req.window_test,
            risk_aversion=req.risk_aversion,
            tau=req.tau,
            use_views=req.use_views,
        )
    )
    return {"job_id": job_id}


@app.get("/backtest/{job_id}")
def poll_backtest(job_id: str):
    """Poll the status of a running or completed backtest job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    response: dict = {"status": job["status"]}
    if job["status"] == "done" and job["result"] is not None:
        result = dict(job["result"])
        result.pop("_last_params", None)  # strip internal state before sending
        response["result"] = result
    elif job["status"] == "error":
        response["error"] = job["error"]
    return response


@app.post("/sensitivity")
def sensitivity_analysis(req: SensitivityRequest, _: None = Depends(enforce_rate_limit)):
    """
    View-multiplier sensitivity sweep. Fast, runs in-request.
    Returns per-ticker weights at each multiplier value.
    """
    try:
        return run_sensitivity(
            tickers=req.tickers,
            period=req.period,
            risk_aversion=req.risk_aversion,
            tau=req.tau,
            multipliers=req.multipliers,
        )
    except Exception as exc:
        detail = str(exc) if not settings.is_production else "Sensitivity analysis failed"
        raise HTTPException(status_code=400, detail=detail)


@app.get("/ff-factors")
def fama_french_factors(_: None = Depends(enforce_rate_limit)):
    """Compute and return Fama-French 5 factors (Mkt-RF, SMB, HML, RMW, CMA, RF)."""
    try:
        factors = compute_ff_factors()
        return {"factors": factors}
    except Exception as exc:
        detail = str(exc) if not settings.is_production else "FF factors computation failed"
        raise HTTPException(status_code=400, detail=detail)


@app.exception_handler(Exception)
async def handle_unexpected_exception(_, exc: Exception):
    if settings.is_production:
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    return JSONResponse(status_code=500, content={"detail": str(exc)})
