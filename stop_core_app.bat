@echo off
setlocal
REM Stop SmartPdfReaderApi (port 5000), RAG (port 8000), and app DB (Qdrant).
REM Run from repo root: stop_all.bat

echo [1/3] Stopping SmartPdfReaderApi (port 5000)...
set "PORT=5000"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr /L "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
    if not errorlevel 1 (
        echo Stopped process on port 5000 - PID %%a
        goto :api_done
    )
)
echo No process found listening on port 5000.
:api_done

echo [2/3] Stopping RAG (port 8000)...
call "%~dp0AI_api\stop_rag.bat"

echo [3/3] Stopping app DB (Qdrant)...
call "%~dp0AI_module\dev_tools\stop_app_db.bat"

echo.
echo All services stopped.
endlocal
exit /b 0
