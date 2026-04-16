$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "common.ps1")
Assert-RepoRoot $repoRoot

if (-not $env:MSSQL_SA_PASSWORD) {
  Write-Error "Set MSSQL_SA_PASSWORD first, e.g. `$env:MSSQL_SA_PASSWORD = 'Your_strong_password123'"
  exit 1
}

$env:OLLAMA_HOST = "http://ollama:11434"
$composeFile = Join-Path $repoRoot "SmartPdfReaderDeployment/docker-compose.yml"

Write-Host "Starting data stores and Ollama..."
Invoke-Compose $repoRoot @("up", "-d", "--build", "qdrant", "mssql", "ollama")

Write-Host "Waiting for SQL Server to accept connections..."
$sqlReady = $false
for ($i = 0; $i -lt 40; $i++) {
  if (Test-NetConnection -ComputerName "127.0.0.1" -Port 1433 -InformationLevel Quiet -WarningAction SilentlyContinue) {
    $sqlReady = $true
    break
  }
  Start-Sleep -Seconds 3
}
if (-not $sqlReady) {
  Write-Error "SQL Server did not become reachable on port 1433. Check logs: .\SmartPdfReaderDeployment\logs.ps1 mssql"
  exit 1
}

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
    & docker compose -f $composeFile exec -T mssql $sqlcmd -S localhost -U sa -P $Password -C -b -Q $Query 2>$null
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

# Password drift handling for existing mssql_data volume:
# - The DB is persisted in the mssql_data volume. The sa password is "locked in" on first init.
# - If you forgot the original password, the only recovery is to delete the volume and reinitialize (DESTROYS SQL data).
$loginOk = Test-SaLogin -Password $env:MSSQL_SA_PASSWORD

if (-not $loginOk) {
  Write-Host ""
  Write-Error "SQL login failed for user 'sa' with the provided MSSQL_SA_PASSWORD."
  Write-Host "This usually means the SQL Server volume was initialized earlier with a different password."
  Write-Host ""
  Write-Host "Fix:"
  Write-Host "  - Re-run with the ORIGINAL password, OR reset volumes to create an empty DB (DESTROYS SQL data):"
  Write-Host "      docker compose -f SmartPdfReaderDeployment/docker-compose.yml down -v"
  Write-Host ""
  exit 1
}

Write-Host "Preparing models and database (one-shot containers)..."
Invoke-Compose $repoRoot @("--profile", "setup", "run", "--rm", "model_prep")
Invoke-Compose $repoRoot @("--profile", "setup", "run", "--rm", "ollama_pull")
Invoke-Compose $repoRoot @("--profile", "setup", "run", "--rm", "api_migrations")

Write-Host "Starting RAG API and SmartPdfReaderApi..."
Invoke-Compose $repoRoot @("up", "-d", "--build", "rag", "smartpdfreaderapi")

function Test-Url([string]$url) {
  try {
    $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
    return ($r.StatusCode -eq 200)
  } catch { return $false }
}

Write-Host "Waiting for services to become healthy..."
$deadline = (Get-Date).AddMinutes(5)
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

Write-Error "Backend did not become ready in time. Run .\SmartPdfReaderDeployment\logs.ps1 for details."
exit 1
