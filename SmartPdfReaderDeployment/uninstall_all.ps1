$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "common.ps1")
Assert-RepoRoot $repoRoot

function Require-Admin {
  $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
  if (-not $isAdmin) {
    Write-Error "Run PowerShell as Administrator. This script removes system-wide tools."
    exit 1
  }
}

Require-Admin

Write-Host ""
Write-Host "FULL UNINSTALL - pre-deployment state" -ForegroundColor Red
Write-Host "This will:" -ForegroundColor Yellow
Write-Host "  - Stop DesktopClient and remove this project's Docker stack and volumes"
Write-Host "  - Stop Docker Desktop and uninstall Docker Desktop (if installed via Chocolatey)"
Write-Host "  - Uninstall Git, .NET 9 SDK (if installed via Chocolatey)"
Write-Host "  - Remove Chocolatey (uninstall packages above, then delete install folder)"
Write-Host ""
Write-Host "WSL is NOT removed." -ForegroundColor Green
Write-Host "Some tools installed outside Chocolatey must be removed manually." -ForegroundColor Yellow
Write-Host ""

$confirmation = Read-Host "Type YES to continue, anything else to cancel"
if ($confirmation -ne "YES") {
  Write-Host "Cancelled."
  exit 0
}

# --- 1) Project Docker state ---
Write-Host ""
Write-Host "[1/5] Stopping DesktopClient..."
Get-Process -Name "DesktopClient" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "[2/5] docker compose down -v (project)..."
Push-Location $repoRoot
try {
  if (Get-Command docker -ErrorAction SilentlyContinue) {
    & {
      $ErrorActionPreference = "Continue"
      & docker compose -f $script:COMPOSE_FILE down -v
      if ($LASTEXITCODE -ne 0) {
        Write-Warning "docker compose down -v failed (exit=$LASTEXITCODE). Continuing uninstall. If Docker Desktop is stopped, this is expected."
      }
    }
  }
} finally {
  Pop-Location
}

# --- 2) Docker Desktop ---
Write-Host "[3/5] Stopping Docker Desktop processes..."
$dockerProcs = @("Docker Desktop", "com.docker.backend", "Docker")
foreach ($name in $dockerProcs) {
  Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

if (Get-Command choco -ErrorAction SilentlyContinue) {
  Write-Host "Uninstalling docker-desktop via Chocolatey (if present)..."
  & choco uninstall -y docker-desktop 2>$null
}

# --- 3) Git + .NET (Chocolatey packages from bootstrap) ---
if (Get-Command choco -ErrorAction SilentlyContinue) {
  Write-Host "[4/5] Uninstalling git, dotnet-9.0-sdk via Chocolatey (if present)..."
  & choco uninstall -y git 2>$null
  & choco uninstall -y dotnet-9.0-sdk 2>$null
} else {
  Write-Host "[4/5] Chocolatey not found - skip choco uninstall of git/dotnet (remove manually if needed)."
}

# --- 4) Chocolatey itself (WSL left untouched) ---
Write-Host "[5/5] Removing Chocolatey installation folder..."
$chocoPath = "$env:ProgramData\chocolatey"
if (Test-Path $chocoPath) {
  Remove-Item -Path $chocoPath -Recurse -Force -ErrorAction SilentlyContinue
  Write-Host "Removed: $chocoPath"
} else {
  Write-Host "Chocolatey folder not found at $chocoPath (already removed or never installed)."
}
Write-Host "If choco remains on PATH, remove it under Settings > Environment Variables."

Write-Host ""
Write-Host "DONE. Reboot recommended." -ForegroundColor Green
Write-Host "Machine should be near pre-deployment for tools installed via this pipeline; verify Apps & Features for anything left."
exit 0
