@echo off
setlocal
REM Stop RAG FastAPI (process listening on port 8000).

set "PORT=8000"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000" ^| findstr /L "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
    if not errorlevel 1 (
        echo Stopped process on port 8000 - PID %%a
        goto :done
    )
)
echo No process found listening on port 8000.
:done
endlocal
