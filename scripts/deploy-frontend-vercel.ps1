param(
  [Parameter(Mandatory = $true)][ValidateSet("preview", "production")][string]$Environment,
  [Parameter(Mandatory = $true)][string]$VercelToken
)

$ErrorActionPreference = "Stop"
$frontendPath = "./frontend"

Push-Location $frontendPath
try {
  npm ci
  npm run build
  npm i -g vercel

  if ($Environment -eq "preview") {
    vercel pull --yes --environment=preview --token=$VercelToken
    vercel build --token=$VercelToken
    vercel deploy --prebuilt --token=$VercelToken
  }
  else {
    vercel pull --yes --environment=production --token=$VercelToken
    vercel build --prod --token=$VercelToken
    vercel deploy --prebuilt --prod --token=$VercelToken
  }
}
finally {
  Pop-Location
}
