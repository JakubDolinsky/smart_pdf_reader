$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "common.ps1")
Assert-RepoRoot $repoRoot

Write-Host "Stopping Docker stack..."
Invoke-Compose $repoRoot @("down")

Write-Host "Stopping DesktopClient (if running)..."
Get-Process -Name "DesktopClient" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "Stopped."
exit 0
