param(
  [Parameter(Mandatory = $true)][string]$AwsRegion,
  [Parameter(Mandatory = $true)][string]$AwsAccountId,
  [Parameter(Mandatory = $true)][string]$EcrRepository,
  [Parameter(Mandatory = $true)][string]$ImageTag,
  [Parameter(Mandatory = $true)][string]$AppRunnerServiceArn
)

$ErrorActionPreference = "Stop"

$imageUri = "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com/$EcrRepository`:$ImageTag"

Write-Host "Logging into ECR..."
aws ecr get-login-password --region $AwsRegion | docker login --username AWS --password-stdin "$AwsAccountId.dkr.ecr.$AwsRegion.amazonaws.com"

Write-Host "Building backend image..."
docker build -t "$EcrRepository`:$ImageTag" "./backend"
docker tag "$EcrRepository`:$ImageTag" $imageUri
docker push $imageUri

Write-Host "Triggering App Runner deployment..."
aws apprunner start-deployment --service-arn $AppRunnerServiceArn --region $AwsRegion

Write-Host "Staging deployment requested successfully."
