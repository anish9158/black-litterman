import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .config import load_settings
from .middleware import (
    InMemoryRateLimiter,
    RequestSizeLimitMiddleware,
    get_rate_limit_dependency,
)
from .model import run_black_litterman, DEFAULT_TICKERS
from .rag import ask_rag, ask_rag_stream, generate_narrative, get_vectorstore
from .backtest import create_job, get_job, run_backtest_async, run_sensitivity
from .fama_french import compute_ff_factors

app = FastAPI(title="Black-Litterman RAG API", version="3.0.0")
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


class ExplainRequest(BaseModel):
    tickers: Optional[List[str]] = None
    period: str = "1y"


class SentimentRequest(BaseModel):
    tickers: Optional[List[str]] = None


class NLOptimizeRequest(BaseModel):
    prompt: str


class ReportRequest(BaseModel):
    tickers: List[str]
    allocation: List[Dict[str, Any]]
    metrics: Dict[str, float]
    narrative: str = ""
    shap_data: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Core endpoints
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
        result = run_black_litterman(
            req.tickers,
            req.period,
            req.risk_aversion,
            req.tau,
            req.use_xgb_views,
        )
        # Attach LLM narrative (fails silently — never blocks the response)
        try:
            result["narrative"] = generate_narrative(
                allocation=result.get("allocation", []),
                metrics=result.get("metrics", {}),
                tickers=result.get("tickers", []),
            )
        except Exception:
            result["narrative"] = ""
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/ask-rag")
def rag_chat(req: ChatRequest, _: None = Depends(enforce_rate_limit)):
    try:
        return ask_rag(req.question, req.k, req.history)
    except Exception as exc:
        detail = str(exc) if not settings.is_production else "RAG request failed"
        raise HTTPException(status_code=400, detail=detail)


@app.post("/ask-rag-stream")
def rag_chat_stream(req: ChatRequest, _: None = Depends(enforce_rate_limit)):
    """Server-Sent Events streaming endpoint for the RAG chatbot."""
    return StreamingResponse(
        ask_rag_stream(req.question, req.k, req.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/rebuild-rag")
def rebuild_rag(_: None = Depends(enforce_rate_limit)):
    store = get_vectorstore(force_rebuild=True)
    return {"status": "rebuilt", "chunks": store.index.ntotal}


# ---------------------------------------------------------------------------
# Backtest endpoints
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
        result.pop("_last_params", None)
        response["result"] = result
    elif job["status"] == "error":
        response["error"] = job["error"]
    return response


@app.post("/sensitivity")
def sensitivity_analysis(req: SensitivityRequest, _: None = Depends(enforce_rate_limit)):
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
    try:
        factors = compute_ff_factors()
        return {"factors": factors}
    except Exception as exc:
        detail = str(exc) if not settings.is_production else "FF factors computation failed"
        raise HTTPException(status_code=400, detail=detail)


# ---------------------------------------------------------------------------
# AI Enhancement endpoints
# ---------------------------------------------------------------------------

@app.post("/explain-views")
def explain_views_endpoint(req: ExplainRequest, _: None = Depends(enforce_rate_limit)):
    """
    Train per-stock XGBoost models and return SHAP feature importances.
    Shows which technical/fundamental features drove each stock's return view.
    """
    try:
        from .views import explain_views
        tickers = req.tickers or DEFAULT_TICKERS
        explanations = explain_views(tickers, req.period)
        return {"tickers": list(explanations.keys()), "explanations": explanations}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/sentiment")
def portfolio_sentiment(req: SentimentRequest, _: None = Depends(enforce_rate_limit)):
    """
    Score each ticker's recent news sentiment via Groq zero-shot classification
    (falls back to keyword heuristic when no API key is set).
    """
    try:
        from .sentiment import get_portfolio_sentiment
        tickers = req.tickers or DEFAULT_TICKERS
        return get_portfolio_sentiment(tickers)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/nl-optimize")
def nl_optimize(req: NLOptimizeRequest, _: None = Depends(enforce_rate_limit)):
    """
    Parse a natural-language portfolio request via Groq and run optimization.

    Example prompts:
      "Optimise RELIANCE, TCS, INFY over 2 years with XGBoost views"
      "What if I only use banking stocks for 6 months?"
    """
    import os as _os

    api_key = _os.getenv("GROQ_TOKEN") or settings.groq_api_key
    if not api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY required for natural language parsing")

    # Step 1: parse the intent with Groq
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=settings.groq_model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
        )
        system = (
            "Extract portfolio optimisation parameters from the user's request.\n"
            "Return ONLY a valid JSON object with these optional keys:\n"
            '  "tickers": ["LIST.NS", ...]  — comma-separated NSE ticker symbols\n'
            '  "period": "1y"               — one of: "1mo","3mo","6mo","1y","2y","5y"\n'
            '  "use_xgb_views": false        — boolean\n'
            '  "risk_aversion": 2.5          — positive float\n'
            "Include only the fields explicitly mentioned. Omit all others.\n"
            "If tickers are Indian company names, convert to .NS Yahoo Finance tickers.\n"
            "Return ONLY the JSON object, no explanation."
        )
        raw = llm.invoke([
            ("system", system),
            ("human", req.prompt),
        ]).content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed: dict = json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse your request: {exc}")

    # Step 2: run optimization with the parsed parameters
    opt_req = OptimizeRequest(
        tickers=parsed.get("tickers") or DEFAULT_TICKERS,
        period=parsed.get("period", "1y"),
        risk_aversion=float(parsed.get("risk_aversion", 2.5)),
        tau=float(parsed.get("tau", 0.10)),
        use_xgb_views=bool(parsed.get("use_xgb_views", False)),
    )
    try:
        result = run_black_litterman(
            opt_req.tickers, opt_req.period, opt_req.risk_aversion,
            opt_req.tau, opt_req.use_xgb_views,
        )
        result["parsed_params"] = parsed
        result["narrative"] = generate_narrative(
            allocation=result.get("allocation", []),
            metrics=result.get("metrics", {}),
            tickers=result.get("tickers", []),
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/generate-report")
def generate_report_endpoint(req: ReportRequest, _: None = Depends(enforce_rate_limit)):
    """
    Generate and return a downloadable PDF report from already-computed results.
    Pass the allocation, metrics, and optionally narrative + SHAP data.
    """
    try:
        from .report import generate_report
        pdf_bytes = generate_report(
            tickers=req.tickers,
            allocation=req.allocation,
            metrics=req.metrics,
            narrative=req.narrative,
            shap_data=req.shap_data,
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=portfolio_report.pdf"},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def handle_unexpected_exception(_, exc: Exception):
    if settings.is_production:
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    return JSONResponse(status_code=500, content={"detail": str(exc)})
