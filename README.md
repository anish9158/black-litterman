# Black-Litterman RAG Full-Stack App

Production-oriented setup:

- Frontend: React + TypeScript on Vercel
- Backend: FastAPI in Docker on AWS App Runner

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
- Backend workflows: `.github/workflows/backend-aws.yml`
- App Runner config: `backend/apprunner.yaml`
- Deployment runbook: `docs/deployment.md`
- Operations runbook: `docs/operations.md`

## Helper Scripts

- `scripts/deploy-backend-staging.ps1`
- `scripts/deploy-frontend-vercel.ps1`
- `scripts/smoke-check.ps1`

## Main API Routes

- `GET /health`
- `GET /system-check` — dashboard payload, RAG FAISS + sample retrieval, Groq key (same checks as the frontend System Check page)
- `GET /default-tickers`
- `POST /optimize-portfolio`
- `POST /ask-rag`
- `POST /rebuild-rag`
