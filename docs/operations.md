# Operations Runbook

## Alerts

Configure CloudWatch alarms for:

- `5XXErrorRate > 2%` for 5 minutes
- `P95Latency > 3000 ms` for 10 minutes
- `MemoryUtilization > 85%` for 10 minutes
- Health check failures on `/health`

## Synthetic Checks

Run every 5 minutes:

- `GET /health`
- `POST /optimize-portfolio` with known ticker set
- `POST /ask-rag` with a static prompt

Use `scripts/smoke-check.ps1` for manual verification.

## Incident Response

1. Check App Runner deployment history for recent rollouts.
2. Inspect CloudWatch logs for stack traces or upstream timeout errors.
3. Roll back to previous ECR image tag if issue is deployment-related.
4. If Groq API outage is suspected, verify fallback retrieval-only behavior.
5. Post-incident: add missing alerts/tests before next release.

## Capacity Guidance

- Keep min instances at `1` to reduce cold starts.
- Scale max instances based on sustained CPU/latency pressure.
- Protect `/rebuild-rag` behind admin policy in production environments.
