@echo off
REM Run ingestion: chunk PDFs from config PDF_INPUT_DIR, embed, and load into vector DB.
REM Usage: from repo root: AI_module\dev_tools\run_ingestion.bat
REM Or double-click, or run from CMD from any directory (script cd's to repo root).

cd /d "%~dp0..\.."
python -m AI_module.application.ingestion.ingestion
exit /b %errorlevel%
