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
$wslCmd = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wslCmd) {
  Write-Host "WSL command not found. Enabling Windows optional features for WSL2..."
  & dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart | Out-Host
  & dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart | Out-Host
  $restartNeeded = $true
} else {
  $wslStatusExit = 1
  try {
    & wsl --status *> $null
    $wslStatusExit = $LASTEXITCODE
  } catch {
    $wslStatusExit = 1
  }

  if ($wslStatusExit -ne 0) {
    Write-Host "Installing WSL platform..."
    # --no-distribution: WSL without a default distro (Windows 11+); reduces interactive prompts.
    & {
      $ErrorActionPreference = "Continue"
      & wsl --install --no-distribution
      if ($LASTEXITCODE -ne 0) {
        Write-Host "wsl --install --no-distribution failed; trying wsl --install ..."
        & wsl --install
      }
      # May fail until reboot; non-fatal — user reboots before Docker/WSL2 steps.
      & wsl --set-default-version 2 2>$null
    }
    $restartNeeded = $true
  }
}

# --- Git, Docker Desktop, .NET SDK (Chocolatey; install Chocolatey itself is README step A) ---
Write-Host "Installing / updating Git, Docker Desktop, .NET 9 SDK..."

function Ensure-ChocoCommunitySource {
  # Some environments have Chocolatey installed but the community feed disabled/missing.
  $sourceList = & choco source list 2>$null
  if (-not $sourceList) { return }

  $hasCommunity = ($sourceList | Select-String -SimpleMatch "https://community.chocolatey.org/api/v2/" -Quiet)
  if (-not $hasCommunity) {
    Write-Host "Adding Chocolatey Community source..."
    & choco source add -n="chocolatey-community" -s "https://community.chocolatey.org/api/v2/" | Out-Host
  }

  # Ensure any source pointing at community is enabled.
  $communityNames = @()
  foreach ($line in $sourceList) {
    if ($line -match '^\s*\S+\s+https://community\.chocolatey\.org/api/v2/') {
      $communityNames += ($line -split '\s+')[0]
    }
  }
  $communityNames += "chocolatey-community"
  $communityNames = $communityNames | Select-Object -Unique
  foreach ($n in $communityNames) {
    & choco source enable -n="$n" 2>$null | Out-Null
  }
}

function Install-WithChoco([string[]]$packages) {
  Ensure-ChocoCommunitySource
  & choco install -y @packages
  return ($LASTEXITCODE -eq 0)
}

Write-Host "Installing Git + .NET 9 SDK via Chocolatey..."
if (-not (Install-WithChoco @("git","dotnet-9.0-sdk"))) {
  Fail "choco install failed for git/dotnet-9.0-sdk."
}

Write-Host "Installing Docker Desktop..."
if (-not (Install-WithChoco @("docker-desktop"))) {
  Write-Host "Chocolatey could not install 'docker-desktop' (package not found or source unavailable). Trying winget..."
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    & winget install --id Docker.DockerDesktop -e --source winget
    if ($LASTEXITCODE -ne 0) {
      Fail "Failed to install Docker Desktop via winget as well."
    }
  } else {
    Fail "Chocolatey could not find 'docker-desktop' and winget is not available. Install Docker Desktop manually, then rerun: https://www.docker.com/products/docker-desktop/"
  }
}

Write-Host ""
Write-Host "RESTART REQUIRED"
if ($restartNeeded) {
  Write-Host "(Reboot after first-time WSL install, then continue deployment.)"
}
exit 0
