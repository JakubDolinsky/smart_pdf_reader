$ErrorActionPreference = "Stop"

function Fail([string]$msg) {
  Write-Error $msg
  exit 1
}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
  Fail "Run PowerShell as Administrator (required for WSL and Chocolatey package installs)."
}

if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
  Fail "Chocolatey is not installed. Install it first (README section 9, step A: Before cloning), then run this script again."
}

$restartNeeded = $false

# --- WSL (platform only where supported; avoids forcing a default Linux distro) ---
& wsl --status *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Installing WSL platform..."
  # --no-distribution: WSL without a default distro (Windows 11+); reduces interactive prompts.
  & wsl --install --no-distribution
  if ($LASTEXITCODE -ne 0) {
    Write-Host "wsl --install --no-distribution failed; trying wsl --install ..."
    & wsl --install
  }
  # May fail until reboot; non-fatal — user reboots before Docker/WSL2 steps.
  & wsl --set-default-version 2 2>$null
  $restartNeeded = $true
}

# --- Git, Docker Desktop, .NET SDK (Chocolatey; install Chocolatey itself is README step A) ---
Write-Host "Installing / updating Git, Docker Desktop, .NET 9 SDK via Chocolatey..."
& choco install -y git docker-desktop dotnet-9.0-sdk
if ($LASTEXITCODE -ne 0) { Fail "choco install failed." }

Write-Host ""
Write-Host "RESTART REQUIRED"
if ($restartNeeded) {
  Write-Host "(Reboot after first-time WSL install, then continue deployment.)"
}
exit 0
