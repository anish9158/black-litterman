# AI-Powered Black-Litterman Portfolio Intelligence Platform

Full-stack application combining **deterministic Black–Litterman optimisation** (CVXPY), **optional XGBoost-generated views**, **SHAP explainability** for those views, and a **source-grounded RAG assistant** over project documentation (FAISS + HuggingFace embeddings, Groq LLM). The frontend is React + TypeScript (Vite); the backend is FastAPI.

## Project Overview

The platform supports:

- Portfolio optimisation and metrics via `run_black_litterman` (equilibrium + views, covariance shrinkage, constrained solve).
- **Notebook-cached results** (`NOTEBOOK_RESULTS` / `GET /notebook-results`) for dashboard charts and tables without recomputing the full notebook on every page load.
- **Live-style flows**: backtest jobs, sensitivity, Fama–French factors, NL-driven optimise (`/nl-optimize`), PDF report generation, and optional sentiment / explain-views endpoints where implemented in `main.py`.
- **RAG chat** (`/ask-rag`, `/ask-rag-stream`) with retrieval scores, topic guard, streaming, and grounded starter prompts (`GET /rag-suggested-prompts`).
- **Knowledge Base** metadata (`GET /knowledge-base`) and **System Status** checks (`GET /system-check`).

This is an engineering / research stack, not a retail product: behaviour is bounded by what is implemented in this repository (no claims beyond the code paths described below).

## AI Engineering Features

| Area | Implementation (high level) |
|------|-----------------------------|
| **Embeddings & retrieval** | `sentence-transformers/all-MiniLM-L6-v2` via LangChain `HuggingFaceEmbeddings`; FAISS index under `backend/rag_index/`; ingestion from `backend/rag_docs/`. |
| **LLM inference** | Groq-compatible OpenAI client (`langchain_openai.ChatOpenAI`) with base URL from settings; model from `GROQ_MODEL` / config. |
| **RAG grounding** | System prompt restricts answers to retrieved chunks; off-topic string guard; optional retrieval-only mode without API key. |
| **Views** | XGBoost regressors on engineered features; views feed Black–Litterman when enabled. |
| **Explainability** | SHAP TreeExplainer for XGBoost view models where wired through optimisation / report flows. |
| **Observability** | Chat API returns token usage and latency (non-stream: provider metadata; stream: character-based estimates + wall-clock latency). |
| **Audit** | `log_llm_call` → `backend/logs/llm_calls.jsonl` (NDJSON, `prompt_hash`, optional tokens/latency/route/run_id/status). |

## Staged AI Pipeline

A **reference list** of stages (aligned with modules and routes) is served as static JSON:

```http
GET /ai-pipeline-trace
```

Use this for dashboards or external docs; it does not execute the pipeline.

## Architecture

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  React (Vite)   │────▶│  FastAPI backend │────▶│  Optimiser / RAG    │
│  Chat, Dashboard│     │  REST + SSE      │     │  model.py, rag.py,  │
│  Knowledge Base │     │  rate + size cap │     │  backtest, FAISS    │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
         │                                               │
         └────────────────── VITE_API_URL ──────────────┘
```

- **Frontend**: routes in `frontend/src/App.tsx` — Chat, Dashboard, Knowledge Base, System Status.
- **Backend**: `backend/app/main.py` mounts routes; `rag.py` handles index build, `ask_rag`, streaming, narratives, grounded suggestions.

## Frontend Pages

| Route | Purpose |
|-------|---------|
| `/chat` | Streaming source-grounded assistant; sources, L² distances, token/latency badges, follow-up chips. |
| `/dashboard` | Cached notebook metrics, charts, AI engineering narrative, pipeline trace, observability copy. |
| `/knowledge-base` | Chunk listing from `GET /knowledge-base`. |
| `/system-check` | Operational checks (`GET /system-check`) — RAG index, Groq config, dashboard data. |

## Backend API Routes

| Method | Path | Role |
|--------|------|------|
| GET | `/` | Health message |
| GET | `/health` | Liveness |
| GET | `/system-check` | Deep smoke checks |
| GET | `/ai-pipeline-trace` | Static pipeline JSON |
| GET | `/notebook-results` | Cached dashboard payload |
| GET | `/knowledge-base` | RAG chunk metadata |
| GET | `/rag-suggested-prompts` | Grounded starter questions |
| GET | `/default-tickers` | Default universe |
| GET | `/backtest/{job_id}` | Poll async backtest |
| GET | `/ff-factors` | Fama–French factors |
| POST | `/optimize-portfolio` | BL optimise (+ optional narrative) |
| POST | `/ask-rag` | RAG Q&A (JSON) |
| POST | `/ask-rag-stream` | RAG SSE |
| POST | `/rebuild-rag` | Rebuild FAISS index |
| POST | `/backtest` | Start backtest job |
| POST | `/sensitivity` | Sensitivity sweep |
| POST | `/explain-views` | View explanation helper |
| POST | `/sentiment` | Sentiment endpoint (if configured) |
| POST | `/nl-optimize` | NL → params → optimise + narrative |
| POST | `/generate-report` | PDF report |

## RAG and Source Grounding

- Documents are loaded from `backend/rag_docs/` (including nested `concept_primers/`, etc.), chunked, embedded, and stored in FAISS.
- Responses are constrained to **retrieved context** in the chat system prompt; the UI shows **source labels**, previews, and **FAISS L² embedding distance** (lower = closer in embedding space for that retrieval batch).
- `/rag-suggested-prompts` mines grounded questions from diversified retrieval plus filters (topic + term overlap).
- Without `GROQ_API_KEY` / `GROQ_TOKEN`, chat falls back to retrieval snippets without full generation.

## Token and Latency Observability

- **`POST /ask-rag`**: response includes `token_usage` (from Groq response metadata when available) and `latency_ms` for the handler’s LLM call window.
- **`POST /ask-rag-stream`**: final SSE event includes **estimated** prompt/completion/total tokens (character heuristic) and `latency_ms` for the streaming generation window; the Chat UI displays these.
- No central metrics server: values are **per-request** in API/ UI only.

## LLM Audit Logging

`backend/app/llm_audit.py` appends one JSON object per line to **`backend/logs/llm_calls.jsonl`** (directory gitignored).

**Always written (when logging succeeds):** `stage`, ISO `timestamp`, `provider`, `model`, `prompt_hash` (SHA-256 of prompt material), `input_artifacts`, `output_artifact`.

**Optional (when supplied):** `input_tokens`, `output_tokens`, `total_tokens`, `latency_ms`, `route`, `run_id`, `status`, `error_message`.

API keys are never logged. Logging failures are swallowed so user flows never depend on disk I/O.

## Deterministic Optimisation vs LLM Reasoning

- **Deterministic core**: returns, covariances, Black–Litterman posterior, and CVXPY solution are **numeric and reproducible** given the same data and flags (subject to external data vendor behaviour).
- **LLM layers**: Groq is used for **natural-language narrative** (`generate_narrative`), **RAG answers and suggestions**, and **NL parameter parsing** (`/nl-optimize`). These outputs are **not** substitute proof of performance; evaluation uses the optimisation and backtest metrics, not prose.

## Local Development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Copy `frontend/.env.example` → `frontend/.env` and set `VITE_API_URL` to the backend (e.g. `http://localhost:8000`).

### Environment variables

Backend (`backend/.env.example`): `GROQ_API_KEY`, `GROQ_MODEL`, `APP_ENV`, `CORS_ORIGINS`, `MAX_REQUEST_SIZE_BYTES`, `RATE_LIMIT_PER_MINUTE`, etc.

Frontend: `VITE_API_URL`

## Deployment

- Frontend: `.github/workflows/frontend-vercel.yml` (Vercel-oriented).
- Backend: `.github/workflows/backend-render.yml`, `backend/apprunner.yaml`
- Runbooks: `docs/deployment.md`, `docs/operations.md`

## Validation

```bash
python scripts/validate_portfolio_project.py
make validate   # POSIX; requires `python3` in PATH
```

## Helper scripts

- `scripts/deploy-backend-staging.ps1`
- `scripts/deploy-frontend-vercel.ps1`
- `scripts/smoke-check.ps1`
