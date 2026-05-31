@echo off
cd /d "%~dp0"

set "LOCK_FILE=%CD%\.musubi-trainer.lock"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$lockFile = Join-Path $PWD.Path '.musubi-trainer.lock';" ^
  "function Test-PidRunning($procId) { try { Get-Process -Id $procId -EA Stop | Out-Null; return $true } catch { return $false } };" ^
  "if (Test-Path $lockFile) {" ^
  "  $storedPid = [int](Get-Content $lockFile -EA SilentlyContinue);" ^
  "  if ($storedPid -and (Test-PidRunning $storedPid)) {" ^
  "    Add-Type -AssemblyName System.Windows.Forms;" ^
  "    [System.Windows.Forms.MessageBox]::Show('Musubi Trainer is already running.','Musubi Trainer',0,64) | Out-Null;" ^
  "    exit 1" ^
  "  }" ^
  "}" ^
  "exit 0"

if errorlevel 1 exit /b 1

for /f %%I in ('powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"ProcessId=$PID\").ParentProcessId"') do set "LAUNCHER_PID=%%I"
if not defined LAUNCHER_PID set "LAUNCHER_PID=%RANDOM%"

> "%LOCK_FILE%" echo %LAUNCHER_PID%

call venv\Scripts\activate.bat
python -m src.app
set "APP_EXIT=%ERRORLEVEL%"

del /f /q "%LOCK_FILE%" >nul 2>nul
pause
exit /b %APP_EXIT%