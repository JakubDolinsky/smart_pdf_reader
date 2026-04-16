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

# --- WSL ---
& wsl --status *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Installing WSL..."
  & wsl --install
  & wsl --set-default-version 2
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
