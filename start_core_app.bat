@echo off
setlocal
REM Start app DB (Qdrant), RAG FastAPI, and SmartPdfReaderApi web API.
REM Run from repo root: start_all.bat

cd /d "%~dp0"

echo [1/3] Starting app DB (Docker Qdrant)...
call AI_module\dev_tools\start_app_db.bat
if %errorlevel% neq 0 (
    echo start_app_db failed. Exiting.
    exit /b %errorlevel%
)

echo [2/3] Starting RAG FastAPI on http://localhost:8000 ...
start "RAG" cmd /k "cd /d "%~dp0" && AI_api\run_rag.bat"

echo [3/3] Starting SmartPdfReaderApi on http://localhost:5000 ...
start "SmartPdfReaderApi" cmd /k "cd /d "%~dp0\SmartPdfReaderApi\SmartPdfReaderApi" && dotnet run"

echo.
echo All services started. RAG and Web API run in separate windows; close those windows or run stop_all.bat to stop them.
endlocal
exit /b 0
