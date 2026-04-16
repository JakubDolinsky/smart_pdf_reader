$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
  $exe = Join-Path $PWD "DesktopClient\DesktopClient\bin\Release\net9.0-windows\DesktopClient.exe"
  if (-not (Test-Path $exe)) {
    Write-Host "Building DesktopClient (Release)..."
    dotnet build ".\DesktopClient\DesktopClient\DesktopClient.csproj" -c Release
    if ($LASTEXITCODE -ne 0) { throw "dotnet build failed (exit $LASTEXITCODE)" }
  }
  if (-not (Test-Path $exe)) { throw "DesktopClient.exe not found after build: $exe" }
  Start-Process $exe
  exit 0
} catch {
  Write-Error $_
  exit 1
} finally {
  Pop-Location
}
