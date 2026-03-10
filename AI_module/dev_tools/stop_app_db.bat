@echo off
REM -------------------------------
REM Stop Qdrant container and optionally Docker Desktop
REM Usage: call AI_module\dev_tools\stop_app_db.bat
REM -------------------------------

setlocal

REM --- Stop Qdrant container ---
docker ps -a --format "{{.Names}}" | findstr /i "qdrant_local" >nul
if %errorlevel%==0 (
    echo Stopping Qdrant container...
    docker stop qdrant_local >nul 2>&1
    echo Qdrant container stopped.
) else (
    echo Qdrant container not found or already stopped.
)

REM --- Optional: stop Docker Desktop ---
REM echo Stopping Docker Desktop...
REM taskkill /IM "Docker Desktop.exe" /F >nul 2>&1
REM echo Docker Desktop stopped.

endlocal
exit /b 0