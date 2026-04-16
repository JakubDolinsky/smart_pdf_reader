$ErrorActionPreference = "Stop"

# Compose file path relative to repository root
$script:COMPOSE_FILE = "SmartPdfReaderDeployment/docker-compose.yml"

function Assert-RepoRoot {
  param([string]$RepoRoot)
  if (-not (Test-Path (Join-Path $RepoRoot "SmartPdfReaderDeployment\docker-compose.yml"))) {
    throw "Run from repository root (expected SmartPdfReaderDeployment\docker-compose.yml under current directory)."
  }
}

function Invoke-Compose {
  param([string]$RepoRoot, [string[]]$ComposeArgs)
  Push-Location $RepoRoot
  try {
    & docker compose -f $script:COMPOSE_FILE @ComposeArgs
    if ($LASTEXITCODE -ne 0) {
      throw "docker compose failed (exit $LASTEXITCODE): docker compose -f $($script:COMPOSE_FILE) $($ComposeArgs -join ' ')"
    }
  } finally {
    Pop-Location
  }
}
