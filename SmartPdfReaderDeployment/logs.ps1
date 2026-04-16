param(
  [Parameter(Position = 0)]
  [string]$Service = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "common.ps1")
Assert-RepoRoot $repoRoot

Push-Location $repoRoot
try {
  if ([string]::IsNullOrWhiteSpace($Service)) {
    & docker compose -f $script:COMPOSE_FILE logs --tail 200
  } else {
    & docker compose -f $script:COMPOSE_FILE logs --tail 200 $Service
  }
  if ($LASTEXITCODE -ne 0) { exit 1 }
  exit 0
} catch {
  Write-Error $_
  exit 1
} finally {
  Pop-Location
}
