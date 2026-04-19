$ErrorActionPreference = "Stop"

function Get-DockerDesktopExe {
  $candidates = @(
    "${env:ProgramFiles}\Docker\Docker\Docker Desktop.exe",
    "C:\Program Files\Docker\Docker\Docker Desktop.exe",
    "${env:ProgramFiles(x86)}\Docker\Docker\Docker Desktop.exe"
  )
  foreach ($p in $candidates) {
    if ($p -and (Test-Path $p)) { return $p }
  }
  return $null
}

Write-Host "Ensuring Docker engine is available..."
& docker info 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
  Write-Host "Docker engine already running."
} else {
  $dockerDesktopExe = Get-DockerDesktopExe
  if (-not $dockerDesktopExe) {
    Write-Error "Deployment failed: Docker Desktop not found under Program Files. Install Docker Desktop (README step 1 / bootstrap_env.ps1) and retry."
    exit 1
  }
  Write-Host "Starting Docker Desktop from: $dockerDesktopExe"
  Start-Process $dockerDesktopExe
}

Write-Host "Waiting for Docker engine..."
$timeoutSec = 300
$start = Get-Date
while ($true) {
  & docker info 2>$null | Out-Null
  if ($LASTEXITCODE -eq 0) { break }
  if (((Get-Date) - $start).TotalSeconds -ge $timeoutSec) {
    Write-Error "Deployment failed: Docker did not become ready within ${timeoutSec}s. Open Docker Desktop once, finish setup, enable WSL2 if prompted, then retry."
    exit 1
  }
  Start-Sleep -Seconds 5
}

Write-Host "Running hello-world..."
& docker run --rm hello-world
if ($LASTEXITCODE -ne 0) {
  Write-Error "Deployment failed: docker hello-world test failed. Stopping deployment."
  exit 1
}

Write-Host "Docker OK."
exit 0
