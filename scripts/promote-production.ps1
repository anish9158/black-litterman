param(
  [Parameter(Mandatory = $true)][string]$ProductionBackendBaseUrl
)

$ErrorActionPreference = "Stop"

Write-Host "Running production smoke checks..."
.\scripts\smoke-check.ps1 -BackendBaseUrl $ProductionBackendBaseUrl

Write-Host "Production cutover checks complete."
Write-Host "Next steps:"
Write-Host "1) Confirm Vercel production deployment is live."
Write-Host "2) Validate CloudWatch alarms are green for 30 minutes."
Write-Host "3) Announce release completion."
