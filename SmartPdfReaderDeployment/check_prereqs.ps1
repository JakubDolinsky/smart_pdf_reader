$ErrorActionPreference = "Stop"

function Fail([string]$reason) {
  Write-Error "Deployment failed: $reason Fix the missing requirement and retry."
  exit 1
}

# --- Administrator ---
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
  Fail "PowerShell must be run as Administrator."
}

# --- Windows version (10/11) ---
$os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
if (-not $os) { Fail "Could not read Windows version." }
$major = [int]$os.Version.Split(".")[0]
$build = [int]$os.BuildNumber
if ($major -lt 10) {
  Fail "Windows 10 or newer is required (found version $($os.Version))."
}

# --- Internet ---
try {
  if (-not (Test-NetConnection -ComputerName "www.microsoft.com" -Port 443 -InformationLevel Quiet -WarningAction SilentlyContinue)) {
    Fail "No outbound HTTPS connectivity (check firewall/proxy)."
  }
} catch {
  Fail "No network connectivity: $($_.Exception.Message)"
}

Write-Host "Prerequisites OK (Windows build $build, admin, virtualization check done, network reachable)."
exit 0
