@echo off
REM -------------------------------
REM Start Docker Desktop (if needed) and Qdrant container
REM Usage: run from repo root or AI_module: call AI_module\dev_tools\start_app_db.bat
REM -------------------------------

setlocal EnableDelayedExpansion

REM --- Check if Docker is running ---
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not running. Attempting to start Docker Desktop...
    if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
        start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        echo Waiting for Docker to start...
        set /a count=0
        :wait_docker
        timeout /t 5 >nul
        docker info >nul 2>&1
        if %errorlevel%==0 goto docker_ready
        set /a count+=1
        if !count! lss 18 goto wait_docker
        echo Docker did not start in time. Start it manually.
        exit /b 1
    ) else (
        echo Docker Desktop not found. Start it manually.
        exit /b 1
    )
)
:docker_ready
echo Docker is running.

REM --- Start or create Qdrant container ---
docker start qdrant_local >nul 2>&1
if %errorlevel% neq 0 (
    echo Creating Qdrant container...
    docker run -d -p 6333:6333 --name qdrant_local qdrant/qdrant
    if %errorlevel% neq 0 (
        echo Failed to start Qdrant container.
        docker ps -a
        exit /b 1
    )
) else (
    echo Qdrant container started.
)

REM --- Wait for Qdrant health ---
set /a attempts=0
:wait_qdrant
set /a attempts+=1
powershell -NoProfile -Command ^
"try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:6333/healthz' -UseBasicParsing -TimeoutSec 3; exit [int]($r.StatusCode -ne 200) } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 goto qdrant_ready
if !attempts! geq 18 (
    echo Qdrant did not respond in time.
    docker ps -a
    exit /b 1
)
timeout /t 5 >nul
goto wait_qdrant

:qdrant_ready
echo Qdrant is ready at http://127.0.0.1:6333
endlocal
exit /b 0