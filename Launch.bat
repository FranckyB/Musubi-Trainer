@echo off
setlocal
cd /d "%~dp0"

where pyw >nul 2>nul
if %errorlevel%==0 (
	start "" pyw "%~dp0main.py"
	exit /b 0
)

where pythonw >nul 2>nul
if %errorlevel%==0 (
	start "" pythonw "%~dp0main.py"
	exit /b 0
)

rem Fallback: run minimized if only console Python is available.
start "Musubi-Trainer" /min python "%~dp0main.py"