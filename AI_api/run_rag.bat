@echo off
setlocal
REM Run RAG FastAPI. Use for the .NET API, integration tests, or local development.
REM Port 8000 (set ChatService:FastApiBaseUrl / RAG URL to http://localhost:8000 in config).

cd /d "%~dp0\.."

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo Starting RAG FastAPI on http://localhost:8000 (Ctrl+C to stop)...
uvicorn AI_api.main:app --host 0.0.0.0 --port 8000

endlocal
