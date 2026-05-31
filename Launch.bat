@echo off
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command ^
  "$lockFile = Join-Path $PWD.Path '.musubi-trainer.lock';" ^
  "function Test-PidRunning($pid) { try { $p = Get-Process -Id $pid -EA Stop; return $true } catch { return $false } };" ^
  "if (Test-Path $lockFile) {" ^
  "  $storedPid = [int](Get-Content $lockFile -EA SilentlyContinue);" ^
  "  if ($storedPid -and (Test-PidRunning $storedPid)) {" ^
  "    Add-Type -AssemblyName System.Windows.Forms;" ^
  "    [System.Windows.Forms.MessageBox]::Show('Musubi Trainer is already running.','Musubi Trainer',0,64) | Out-Null;" ^
  "    exit" ^
  "  }" ^
  "}" ^
  "$py = @('venv\Scripts\pythonw.exe','venv\Scripts\python.exe') | Where-Object { Test-Path $_ } | Select-Object -First 1;" ^
  "if (-not $py) { $py = 'pythonw' }" ^
  "$p = Start-Process $py '-m src.app' -PassThru -WindowStyle Hidden;" ^
  "Set-Content $lockFile $p.Id;" ^
  "$p.WaitForExit();" ^
  "Remove-Item $lockFile -EA SilentlyContinue"