# Project architecture — data, models, API, and UI (end-to-end map)

## Data layer
Market prices and corporate actions ingested via **yfinance** (rate limits possible). Local disk caching mitigates repeated pulls. Fundamentals may be merged when available for feature construction.

## Machine learning layer
Per-ticker or pooled designs possible; this project emphasises **XGBoost** regressors producing **views** used inside **Black-Litterman** updates. **SHAP** optional layer explains feature pushes on predictions.

## Quant core
**Covariance estimate + shrinkage → BL posterior μ → CVXPY quadratic program** yields weights. **Rolling backtest** simulates refits across time. **Sensitivity** sweeps meta-parameters like view multiplier.

## LLM layer
**Groq-hosted** chat model formats narrative explanations and can generate follow-up prompts. **RAG** grounds answers on local markdown / notebook-derived knowledge rather than open web.

## API surface (FastAPI)
REST JSON routes for optimisation snapshots, streaming chat, static notebook metrics, knowledge-base chunk listing, etc. CORS configured for separate front-end host.

## Front-end
React + Tailwind + shadcn-style components: chat page, dashboard with precomputed charts, knowledge base browser.

## Deployment notes
Environment variables hold API keys never committed. Rebuild FAISS when `rag_docs` files change (mtime check) on server boot paths.
