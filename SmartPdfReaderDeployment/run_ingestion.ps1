$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "common.ps1")
Assert-RepoRoot $repoRoot

Write-Host "Running ingestion (this may take a while)..."
Invoke-Compose $repoRoot @("run", "--rm", "rag", "python", "-m", "AI_module.application.ingestion.ingestion")
Write-Host "Ingestion finished."
exit 0
