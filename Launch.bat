@echo off
setlocal
cd /d "%~dp0"

set "PS_EXE=powershell"
where %PS_EXE% >nul 2>nul
if not %errorlevel%==0 set "PS_EXE=pwsh"

%PS_EXE% -NoProfile -ExecutionPolicy Bypass -Command "$running = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(?i)(python|pyw)(\.exe)?$' -and $_.CommandLine -match '(?i)(^|\s)-m\s+src\.app(\s|$)' }; if ($running) { exit 1 } else { exit 0 }" >nul 2>nul
if %errorlevel%==1 (
	%PS_EXE% -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Windows.Forms; [void][System.Windows.Forms.MessageBox]::Show('Musubi Trainer is already running.','Musubi Trainer',[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Information)" >nul 2>nul
	echo Musubi Trainer is already running.
	exit /b 0
)

if exist "venv\Scripts\pythonw.exe" (
	start "" "venv\Scripts\pythonw.exe" -m src.app
	exit /b 0
)

if exist "venv\Scripts\python.exe" (
	start "Musubi-Trainer" /min "venv\Scripts\python.exe" -m src.app
	exit /b 0
)

where pyw >nul 2>nul
if %errorlevel%==0 (
	start "" pyw -m src.app
	exit /b 0
)

where pythonw >nul 2>nul
if %errorlevel%==0 (
	start "" pythonw -m src.app
	exit /b 0
)

rem Fallback: run minimized if only console Python is available.
start "Musubi-Trainer" /min python -m src.app