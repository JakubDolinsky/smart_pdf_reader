$ErrorActionPreference = "Stop"

function Fail([string]$msg) {
  Write-Error $msg
  exit 1
}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
  Fail "Run PowerShell as Administrator (required for WSL/Chocolatey installs)."
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

# --- Chocolatey ---
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
  Write-Host "Installing Chocolatey..."
  Set-ExecutionPolicy Bypass -Scope Process -Force
  [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
  Invoke-Expression ((New-Object System.Net.WebClient).DownloadString("https://community.chocolatey.org/install.ps1"))
  $restartNeeded = $true
}

# --- Git, Docker Desktop, .NET SDK ---
Write-Host "Installing / updating Git, Docker Desktop, .NET 9 SDK via Chocolatey..."
& choco install -y git docker-desktop dotnet-9.0-sdk
if ($LASTEXITCODE -ne 0) { Fail "choco install failed." }

Write-Host ""
Write-Host "RESTART REQUIRED"
if ($restartNeeded) {
  Write-Host "(Reboot after first-time WSL or Chocolatey install, then continue deployment.)"
}
exit 0
