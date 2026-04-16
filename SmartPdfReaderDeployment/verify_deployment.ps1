$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot "common.ps1")
Assert-RepoRoot $repoRoot

function Test-Url([string]$url) {
  try {
    $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
    return @{ Ok = ($r.StatusCode -eq 200); Detail = "HTTP $($r.StatusCode)" }
  } catch {
    return @{ Ok = $false; Detail = $_.Exception.Message }
  }
}

$results = [ordered]@{}
$results["Qdrant"]       = Test-Url "http://localhost:6333/healthz"
$results["RAG API"]      = Test-Url "http://localhost:8000/docs"
$results["SmartPdf API"] = Test-Url "http://localhost:5000/swagger"
$results["Ollama"]       = Test-Url "http://localhost:11434/api/tags"

Write-Host "Service checks:"
$allOk = $true
foreach ($name in $results.Keys) {
  $st = if ($results[$name].Ok) { "OK" } else { "FAIL" }
  Write-Host "  $name : $st  ($($results[$name].Detail))"
  if (-not $results[$name].Ok) { $allOk = $false }
}

if ($allOk) {
  Write-Host "ALL SYSTEMS OK"
  exit 0
}

$failed = ($results.Keys | Where-Object { -not $results[$_].Ok }) -join ", "
Write-Host "FAIL: $failed"
exit 1
