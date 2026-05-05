# Black-Litterman RAG Full-Stack App

Production-oriented setup:

- Frontend: React + TypeScript on Vercel
- Backend: FastAPI in Docker / Render (see `.github/workflows/`)

## Product scope

**Primary deliverable:** NIFTY-style **Black-Litterman portfolio optimisation**, **live backtests/sensitivity**, and a **closed-domain RAG assistant** grounded on project docs. This app is **not** a staged marketing-copy testing pipeline. Validator + `llm_calls.jsonl`-style audit apply only **thinly**, for portfolio/RAG reproducibility—not campaign rubrics or compliance decks.

## Local Development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set `frontend/.env` from `frontend/.env.example` and point `VITE_API_URL` to backend.

## Environment Variables

Backend (see `backend/.env.example`):

- `GROQ_API_KEY`
- `GROQ_MODEL`
- `APP_ENV`
- `CORS_ORIGINS`
- `MAX_REQUEST_SIZE_BYTES`
- `RATE_LIMIT_PER_MINUTE`

Frontend:

- `VITE_API_URL`

## Deployments

- Frontend workflows: `.github/workflows/frontend-vercel.yml`
- Backend workflows: `.github/workflows/backend-render.yml`
- App Runner config: `backend/apprunner.yaml`
- Deployment runbook: `docs/deployment.md`
- Operations runbook: `docs/operations.md`

## Validation

Static checks (`NOTEBOOK_RESULTS` shape, `rag_docs` presence, optional FAISS index):

```bash
python scripts/validate_portfolio_project.py
make validate   # POSIX, requires make + python3 in PATH as `python3`
```

## LLM audit log

RAG, streaming RAG follow-ups, and portfolio narrative Groq calls append NDJSON rows to **`backend/logs/llm_calls.jsonl`** with `stage`, ISO `timestamp`, `provider`, `model`, **`prompt_hash`**, `input_artifacts`, `output_artifact`. Keys are never logged; the log directory is `.gitignore`d.

## Helper Scripts

- `scripts/deploy-backend-staging.ps1`
- `scripts/deploy-frontend-vercel.ps1`
- `scripts/smoke-check.ps1`
- `scripts/validate_portfolio_project.py`

## Main API Routes

- `GET /health`
- `GET /system-check` — dashboard payload, RAG FAISS + sample retrieval, Groq key (same checks as the frontend System Check page)
- `GET /default-tickers`
- `POST /optimize-portfolio`
- `POST /ask-rag`
- `POST /rebuild-rag`
