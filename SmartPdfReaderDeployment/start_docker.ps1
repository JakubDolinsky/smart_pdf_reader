$ErrorActionPreference = "Stop"

$dockerDesktopExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
if (-not (Test-Path $dockerDesktopExe)) {
  Write-Error "Deployment failed: Docker Desktop not found at $dockerDesktopExe. Install Docker Desktop (bootstrap_env.ps1) and retry."
  exit 1
}

Write-Host "Ensuring Docker Desktop is running..."
& docker info 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Starting Docker Desktop..."
  Start-Process $dockerDesktopExe
}

Write-Host "Waiting for Docker engine..."
$timeoutSec = 300
$start = Get-Date
while ($true) {
  & docker info 2>$null | Out-Null
  if ($LASTEXITCODE -eq 0) { break }
  if (((Get-Date) - $start).TotalSeconds -ge $timeoutSec) {
    Write-Error "Deployment failed: Docker did not become ready within ${timeoutSec}s. Stopping deployment."
    exit 1
  }
  Start-Sleep -Seconds 5
}

Write-Host "Running hello-world..."
docker run --rm hello-world
if ($LASTEXITCODE -ne 0) {
  Write-Error "Deployment failed: docker hello-world test failed. Stopping deployment."
  exit 1
}

Write-Host "Docker OK."
exit 0
