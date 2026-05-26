@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m venv venv
) else (
    python -m venv venv
)

if not exist "venv\Scripts\python.exe" (
    echo Failed to create venv
    exit /b 1
)

"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\python.exe" -m pip install -r requirements.txt

if %errorlevel%==0 (
    echo.
    echo App venv ready. Launch with Launch.bat
) else (
    echo.
    echo Failed to install one or more app dependencies.
    exit /b 1
)
