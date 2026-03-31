@echo off
REM Download embedding model into AI_module\data\models\... (needs internet once).
cd /d "%~dp0..\.."
python AI_module\dev_tools\download_embedding_model.py
exit /b %errorlevel%
