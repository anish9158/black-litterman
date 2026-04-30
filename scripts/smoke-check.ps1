param(
  [Parameter(Mandatory = $true)][string]$BackendBaseUrl
)

$ErrorActionPreference = "Stop"

Write-Host "Checking health endpoint..."
Invoke-RestMethod -Method Get -Uri "$BackendBaseUrl/health" | Out-Null

Write-Host "Checking optimize endpoint..."
$optimizeBody = @{
  tickers = @("RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS")
  period = "6mo"
  risk_aversion = 2.5
  tau = 0.1
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$BackendBaseUrl/optimize-portfolio" -Body $optimizeBody -ContentType "application/json" | Out-Null

Write-Host "Checking rag endpoint..."
$ragBody = @{
  question = "What does this app do?"
  k = 3
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$BackendBaseUrl/ask-rag" -Body $ragBody -ContentType "application/json" | Out-Null

Write-Host "Smoke checks passed."
