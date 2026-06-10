@echo off
setlocal
REM Start app DB (Qdrant),Ollama, RAG FastAPI, and SmartPdfReaderApi web API.
REM Run from repo root: start_all.bat

cd /d "%~dp0"

echo [1/4] Starting app DB (Docker Qdrant)...
call AI_module\dev_tools\start_app_db.bat
if %errorlevel% neq 0 (
    echo start_app_db failed. Exiting.
    exit /b %errorlevel%
)

echo [2/4] Starting Ollama on http://localhost:11434 ...

echo Checking if Ollama is already running...

curl http://localhost:11434/api/tags >nul 2>&1
if not errorlevel 1 (
    echo Ollama is already running. Skipping startup.
    goto ollama_ready
)

where ollama >nul 2>&1
if errorlevel 1 (
    echo ERROR: Ollama executable was not found in PATH.
    echo Install Ollama and ensure "ollama" command is available.
    exit /b 1
)

start "Ollama" cmd /k "ollama serve"

echo Waiting for Ollama to become ready...

set /a OLLAMA_WAIT_SECONDS=60
set /a OLLAMA_ELAPSED=0

:wait_ollama
curl http://localhost:11434/api/tags >nul 2>&1

if not errorlevel 1 (
    echo Ollama is ready.
    goto ollama_ready
)

if %OLLAMA_ELAPSED% GEQ %OLLAMA_WAIT_SECONDS% (
    echo ERROR: Ollama did not become ready within %OLLAMA_WAIT_SECONDS% seconds.
    echo Check the Ollama window for startup errors.
    exit /b 1
)

timeout /t 2 >nul
set /a OLLAMA_ELAPSED+=2
goto wait_ollama

:ollama_ready

echo [3/4] Starting RAG FastAPI on http://localhost:8000 ...
start "RAG" cmd /k "cd /d "%~dp0" && AI_api\run_rag.bat"

echo [4/4] Starting SmartPdfReaderApi on http://localhost:5000 ...
start "SmartPdfReaderApi" cmd /k "cd /d "%~dp0\SmartPdfReaderApi\SmartPdfReaderApi" && dotnet run"

echo.
echo All services started. RAG and Web API run in separate windows; close those windows or run stop_all.bat to stop them.
endlocal
exit /b 0
