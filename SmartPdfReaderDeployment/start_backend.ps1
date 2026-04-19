$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "common.ps1")
Assert-RepoRoot $repoRoot

if (-not $env:MSSQL_SA_PASSWORD) {
  Write-Error "Set MSSQL_SA_PASSWORD first, e.g. `$env:MSSQL_SA_PASSWORD = 'Your_strong_password123'"
  exit 1
}
if ($env:MSSQL_SA_PASSWORD.Length -lt 8) {
  Write-Error "MSSQL_SA_PASSWORD must be at least 8 characters (SQL Server policy). Use a strong password."
  exit 1
}

$env:OLLAMA_HOST = "http://ollama:11434"
$composeFile = Join-Path $repoRoot "SmartPdfReaderDeployment/docker-compose.yml"

function Invoke-SqlCmd {
  param(
    [Parameter(Mandatory = $true)][string]$Password,
    [Parameter(Mandatory = $true)][string]$Query
  )

  $cmds = @(
    "/opt/mssql-tools18/bin/sqlcmd",
    "/opt/mssql-tools/bin/sqlcmd",
    "sqlcmd"
  )

  foreach ($sqlcmd in $cmds) {
    # docker/sqlcmd often prints transient readiness errors to stderr while SQL is starting.
    # With $ErrorActionPreference='Stop', PowerShell would treat native stderr as a terminating error.
    & {
      $ErrorActionPreference = "Continue"
      & docker compose -f $composeFile exec -T mssql $sqlcmd -S localhost -U sa -P $Password -C -b -Q $Query 2>$null | Out-Null
    }
    if ($LASTEXITCODE -eq 0) {
      return $true
    }
  }
  return $false
}

function Test-SaLogin {
  param([Parameter(Mandatory = $true)][string]$Password)
  return (Invoke-SqlCmd -Password $Password -Query "SELECT 1;")
}

Write-Host "Starting data stores and Ollama..."
Invoke-Compose $repoRoot @("up", "-d", "--build", "qdrant", "mssql", "ollama")

Write-Host "Waiting for SQL Server to be ready..."
$maxAttempts = 30
$attempt = 0
$ready = $false

while (-not $ready -and $attempt -lt $maxAttempts) {
  if (Test-SaLogin -Password $env:MSSQL_SA_PASSWORD) {
    $ready = $true
    break
  }
  Start-Sleep -Seconds 2
  $attempt++
}

if (-not $ready) {
  $portOpen = Test-NetConnection -ComputerName "127.0.0.1" -Port 1433 -InformationLevel Quiet -WarningAction SilentlyContinue
  if ($portOpen) {
    Write-Host ""
    Write-Error "SQL Server did not accept 'sa' login in time (port 1433 is open). Wrong MSSQL_SA_PASSWORD for this SQL volume is likely. Re-run with the original password or reset volumes: docker compose -f SmartPdfReaderDeployment/docker-compose.yml down -v"
  } else {
    Write-Error "SQL Server did not become ready in time. Check logs: .\SmartPdfReaderDeployment\logs.ps1 mssql"
  }
  exit 1
}

Write-Host "SQL Server is ready."

Write-Host "Preparing models and database (one-shot containers)..."
Invoke-Compose $repoRoot @("--profile", "setup", "run", "--rm", "model_prep")

Write-Host "Waiting for Ollama server to be ready (before model pull)..."
$ollamaReady = $false
for ($i = 0; $i -lt 60; $i++) {
  try {
    $r = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3
    if ($r.StatusCode -eq 200) { $ollamaReady = $true; break }
  } catch { }
  Start-Sleep -Seconds 2
}
if (-not $ollamaReady) {
  Write-Error "Ollama did not become ready in time. Check logs: .\SmartPdfReaderDeployment\logs.ps1 ollama"
  exit 1
}

Invoke-Compose $repoRoot @("--profile", "setup", "run", "--rm", "ollama_pull")

Write-Host "Starting RAG API and SmartPdfReaderApi..."
Invoke-Compose $repoRoot @("up", "-d", "--build", "rag", "smartpdfreaderapi")

function Test-Url([string]$url) {
  try {
    $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
    return ($r.StatusCode -eq 200)
  } catch { return $false }
}

Write-Host "Waiting for services to become healthy (first run can take 15-30+ minutes while images build and models download)..."
$healthTimeoutMin = 45
if ($env:BACKEND_HEALTH_TIMEOUT_MINUTES) {
  $parsed = 0
  if ([int]::TryParse($env:BACKEND_HEALTH_TIMEOUT_MINUTES, [ref]$parsed) -and $parsed -gt 0) {
    $healthTimeoutMin = $parsed
  }
}
$deadline = (Get-Date).AddMinutes($healthTimeoutMin)
while ((Get-Date) -lt $deadline) {
  $q = Test-Url "http://localhost:6333/healthz"
  $r = Test-Url "http://localhost:8000/docs"
  $a = Test-Url "http://localhost:5000/swagger"
  $o = Test-Url "http://localhost:11434/api/tags"
  if ($q -and $r -and $a -and $o) {
    Write-Host "BACKEND READY"
    exit 0
  }
  Start-Sleep -Seconds 3
}

Write-Error ("Backend did not become ready within {0} minutes. Increase BACKEND_HEALTH_TIMEOUT_MINUTES if needed. Run .\\SmartPdfReaderDeployment\\logs.ps1 for details." -f $healthTimeoutMin)
exit 1
