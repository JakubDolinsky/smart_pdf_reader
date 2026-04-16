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

# --- Virtualization ---
$proc = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1
if ($proc -and ($null -ne $proc.VirtualizationFirmwareEnabled) -and (-not $proc.VirtualizationFirmwareEnabled)) {
  Fail "Hardware virtualization is disabled in BIOS/UEFI (firmware)."
}
$sys = systeminfo 2>$null
if ($sys) {
  $virtLine = $sys | Select-String -Pattern "Virtualization Enabled In Firmware"
  if ($virtLine -and ($virtLine.Line -match ":\s*No\s*$")) {
    Fail "Hardware virtualization is disabled in BIOS/UEFI (systeminfo)."
  }
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
