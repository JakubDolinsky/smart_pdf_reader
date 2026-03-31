@echo off
setlocal
REM Try to add a new migration (if model changed), then apply all migrations to the database.
REM If there are no model changes, "add" fails and we only run "database update".

cd /d "%~dp0.."

REM Unique migration name to avoid "name already used" (EF requires unique class names)
set "MIGRATION_NAME=Migration_%RANDOM%_%RANDOM%"

echo Attempting to add migration: %MIGRATION_NAME%
dotnet ef migrations add "%MIGRATION_NAME%" --project Data --startup-project SmartPdfReaderApi\SmartPdfReaderApi.csproj
if errorlevel 1 (
    echo No new migration added - model unchanged or add failed. Applying existing migrations...
) else (
    echo New migration added. Applying migrations to database...
)

dotnet ef database update --project Data --startup-project SmartPdfReaderApi\SmartPdfReaderApi.csproj
if errorlevel 1 (
    echo Database update failed.
    exit /b 1
)
echo Done.
exit /b 0
