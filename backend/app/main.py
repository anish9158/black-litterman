from typing import List, Optional
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

app = FastAPI(title="Black-Litterman RAG API", version="1.0.0")
settings = load_settings()
rate_limiter = InMemoryRateLimiter(requests_per_minute=settings.rate_limit_per_minute)
enforce_rate_limit = get_rate_limit_dependency(rate_limiter)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=settings.max_request_size_bytes,
)

class OptimizeRequest(BaseModel):
    tickers: Optional[List[str]] = Field(default=None)
    period: str = "1y"
    risk_aversion: float = 2.5
    tau: float = 0.10

class ChatRequest(BaseModel):
    question: str
    k: int = 5

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
        return run_black_litterman(req.tickers, req.period, req.risk_aversion, req.tau)
    except Exception as exc:
        detail = str(exc) if not settings.is_production else "Optimization failed"
        raise HTTPException(status_code=400, detail=detail)

@app.post("/ask-rag")
def rag_chat(req: ChatRequest, _: None = Depends(enforce_rate_limit)):
    try:
        return ask_rag(req.question, req.k)
    except Exception as exc:
        detail = str(exc) if not settings.is_production else "RAG request failed"
        raise HTTPException(status_code=400, detail=detail)

@app.post("/rebuild-rag")
def rebuild_rag(_: None = Depends(enforce_rate_limit)):
    store = get_vectorstore(force_rebuild=True)
    return {"status": "rebuilt", "chunks": store.index.ntotal}


@app.exception_handler(Exception)
async def handle_unexpected_exception(_, exc: Exception):
    if settings.is_production:
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    return JSONResponse(status_code=500, content={"detail": str(exc)})
