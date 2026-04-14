param(
  [switch]$WithOllamaInDocker = $true,
  [switch]$RunIngestion = $false
)

$ErrorActionPreference = "Stop"

function Require-Command($name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Missing command '$name'. Install it and reopen the terminal."
  }
}

Require-Command git
Require-Command docker

Write-Host "== smart_pdf_reader clean machine bootstrap =="

if (-not $env:MSSQL_SA_PASSWORD) {
  # SQL Server requires a reasonably strong password; generate a deterministic-ish default.
  $env:MSSQL_SA_PASSWORD = "SmartPdfReader_SA_" + ([Guid]::NewGuid().ToString("N").Substring(0,12)) + "!"
  Write-Host "MSSQL_SA_PASSWORD was not set. Generated one for this session:"
  Write-Host "  $env:MSSQL_SA_PASSWORD"
}

if ($WithOllamaInDocker) {
  $env:OLLAMA_HOST = "http://ollama:11434"
  Write-Host "Using Ollama in Docker (OLLAMA_HOST=$env:OLLAMA_HOST)"
} else {
  if (-not $env:OLLAMA_HOST) {
    $env:OLLAMA_HOST = "http://host.docker.internal:11434"
  }
  Write-Host "Using host Ollama (OLLAMA_HOST=$env:OLLAMA_HOST)"
}

Write-Host ""
Write-Host "[1/6] docker compose build"
docker compose build

Write-Host ""
Write-Host "[2/6] Pre-download AI models (embedding + reranker)"
docker compose run --rm model_prep

if ($WithOllamaInDocker) {
  Write-Host ""
  Write-Host "[3/6] Start backend services (including Ollama)"
  docker compose --profile ollama up -d qdrant mssql ollama rag smartpdfreaderapi

  Write-Host ""
  Write-Host "[4/6] Pull Ollama LLM images into Docker volume"
  docker compose --profile ollama run --rm ollama_pull
} else {
  Write-Host ""
  Write-Host "[3/6] Start backend services (host Ollama)"
  docker compose up -d qdrant mssql rag smartpdfreaderapi

  Write-Host ""
  Write-Host "[4/6] Ensure host Ollama has the default LLM (phi3:mini — same as AI_module/config.py → llm_model)"
  Write-Host "Run on host:"
  Write-Host "  ollama pull phi3:mini"
}

Write-Host ""
Write-Host "[5/6] Apply SQL migrations"
docker compose run --rm api_migrations

if ($RunIngestion) {
  Write-Host ""
  Write-Host "[6/6] Run ingestion (make sure PDFs exist in AI_module/data/pdfs)"
  docker compose run --rm rag python -m AI_module.application.ingestion.ingestion
} else {
  Write-Host ""
  Write-Host "[6/6] Ingestion is not run (pass -RunIngestion to run it)."
}

Write-Host ""
Write-Host "Backend is ready:"
Write-Host "  RAG FastAPI:        http://localhost:8000/docs"
Write-Host "  SmartPdfReaderApi:  http://localhost:5000/swagger"
Write-Host ""
Write-Host "Next: run DesktopClient on Windows and point it to http://localhost:5000 (default)."
