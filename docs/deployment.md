# Deployment Guide

## Environments

- `staging`: frontend preview on Vercel + staging backend on AWS App Runner.
- `production`: frontend production on Vercel + production backend on AWS App Runner.

## Required Secrets

### AWS

- `GROQ_API_KEY` in AWS Secrets Manager.
- IAM role for GitHub OIDC with access to ECR and App Runner.
- App Runner service ARN for staging and production.

### GitHub Actions

- `AWS_ROLE_TO_ASSUME`
- `ECR_REGISTRY_URI` (e.g., `123456789012.dkr.ecr.us-east-1.amazonaws.com`)
- `APP_RUNNER_SERVICE_ARN`
- `VERCEL_TOKEN`

## Backend (AWS App Runner)

1. Create ECR repo: `black-litterman-rag-backend`.
2. Create App Runner service from ECR image.
3. Set env vars:
   - `APP_ENV=production`
   - `CORS_ORIGINS=https://<your-vercel-domain>`
   - `GROQ_MODEL=llama-3.3-70b-versatile`
   - `MAX_REQUEST_SIZE_BYTES=1048576`
   - `RATE_LIMIT_PER_MINUTE=60`
4. Attach `GROQ_API_KEY` from Secrets Manager.
5. Set health check path to `/health`.

## Frontend (Vercel)

1. Import repo and set root directory to `frontend`.
2. Set environment variables:
   - Preview: `VITE_API_URL=https://<staging-backend-url>`
   - Production: `VITE_API_URL=https://<production-backend-url>`
3. Confirm build command is `npm run build` and output dir is `dist`.

## Staging Rollout

1. Run backend deploy workflow (or `scripts/deploy-backend-staging.ps1`).
2. Run frontend preview deploy workflow.
3. Run `scripts/smoke-check.ps1 -BackendBaseUrl <staging-url>`.
4. Validate charts, optimizer response, and chatbot response in preview frontend.

## Production Rollout

1. Deploy backend image to production App Runner service.
2. Update Vercel production `VITE_API_URL` if needed.
3. Trigger Vercel production deployment.
4. Run smoke checks against production backend.
5. Monitor CloudWatch metrics for 30 minutes before sign-off.
