@echo off
cd /d "%~dp0"

call venv\Scripts\activate.bat
python -m src.app
REM pause
exit /b %ERRORLEVEL%