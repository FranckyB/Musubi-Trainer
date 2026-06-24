@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem -----------------------------------------------------------------------------
rem Unified setup for a single shared venv used by:
rem - Musubi-Trainer (this repo)
rem
rem Usage examples:
rem   Setup.bat
rem   Setup.bat --cuda cu128
rem   Setup.bat --sage-wheel D:\ComfyUI\SageAttention\dist\sageattention-2.2.0-cp311-cp311-win_amd64.whl
rem If --cuda is omitted, the script asks interactively (default: cu130 recommended).
rem -----------------------------------------------------------------------------

set "CUDA_TAG=cu130"
set "CUDA_ARG_PROVIDED=0"
set "SAGE_WHEEL="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--cuda" (
	set "CUDA_TAG=%~2"
	set "CUDA_ARG_PROVIDED=1"
	shift
	shift
	goto parse_args
)
if /I "%~1"=="--sage-wheel" (
	set "SAGE_WHEEL=%~2"
	shift
	shift
	goto parse_args
)
echo Unknown argument: %~1
exit /b 1

:args_done
if "%CUDA_ARG_PROVIDED%"=="0" (
	echo.
	echo Select CUDA wheel profile for PyTorch:
	echo   [1] cu130 ^(recommended^)
	echo   [2] cu128
	echo   [3] cu124
	set "CUDA_CHOICE="
	set /p CUDA_CHOICE="Choose 1/2/3 (Enter for 1): "
	if "!CUDA_CHOICE!"=="" set "CUDA_CHOICE=1"

	if "!CUDA_CHOICE!"=="1" (
		set "CUDA_TAG=cu130"
	) else if "!CUDA_CHOICE!"=="2" (
		set "CUDA_TAG=cu128"
	) else if "!CUDA_CHOICE!"=="3" (
		set "CUDA_TAG=cu124"
	) else (
		echo Invalid choice "!CUDA_CHOICE!". Defaulting to recommended cu130.
		set "CUDA_TAG=cu130"
	)
)

if /I "%CUDA_TAG%"=="cu124" (
	set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124"
) else if /I "%CUDA_TAG%"=="cu128" (
	set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
) else if /I "%CUDA_TAG%"=="cu130" (
	set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130"
) else (
	echo Unsupported --cuda value "%CUDA_TAG%". Use cu124, cu128, or cu130.
	exit /b 1
)

where py >nul 2>nul
if %errorlevel%==0 (
	py -3.11 -m venv venv
) else (
	python -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)"
	if errorlevel 1 (
		echo Python 3.11 is required. Install Python 3.11 or use the py launcher.
		exit /b 1
	)
	python -m venv venv
)

if not exist "venv\Scripts\python.exe" (
	echo Failed to create venv.
	exit /b 1
)

set "PY=venv\Scripts\python.exe"

"%PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)"
if errorlevel 1 (
	echo Created venv is not Python 3.11. Please install/use Python 3.11.
	exit /b 1
)

"%PY%" -c "import struct; raise SystemExit(0 if struct.calcsize('P')*8==64 else 1)"
if errorlevel 1 (
	echo Created venv is not 64-bit Python. Please install Python 3.11 x64 and run Setup.bat again.
	exit /b 1
)

echo.
echo [1/5] Upgrading pip/wheel and installing compatible setuptools ^(^<82 for torch cu130^) ...
"%PY%" -m pip install --upgrade pip wheel "setuptools>=70,<82"
if errorlevel 1 exit /b 1

echo.
echo [2/5] Installing Torch stack for %CUDA_TAG%...
set "SKIP_TORCH_INSTALL=0"
"%PY%" -c "import torch, torchvision, torchaudio, sys; tag='+' + '%CUDA_TAG%'; sys.exit(0 if ((tag in getattr(torch, '__version__', '')) and (tag in getattr(torchvision, '__version__', '')) and (tag in getattr(torchaudio, '__version__', ''))) else 1)"
if not errorlevel 1 (
	echo   Torch stack already matches %CUDA_TAG%. Skipping reinstall.
	set "SKIP_TORCH_INSTALL=1"
)
if "%SKIP_TORCH_INSTALL%"=="0" (
	"%PY%" -m pip install --upgrade torch torchvision torchaudio --index-url %TORCH_INDEX_URL%
	if errorlevel 1 exit /b 1
)

echo.
echo [3/6] Installing Triton ^(Windows build^) for Torch Compile...
"%PY%" -m pip install --upgrade triton-windows==3.6.0.post25
if errorlevel 1 (
	echo Failed to install triton-windows.
	echo Torch Compile requires Triton on this setup.
	exit /b 1
)

echo.
echo [4/6] Installing unified Python dependencies...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo [5/6] Installing SageAttention wheel (optional)...
if not "%SAGE_WHEEL%"=="" (
	if exist "%SAGE_WHEEL%" (
		echo   Using explicit wheel: %SAGE_WHEEL%
		"%PY%" -m pip install "%SAGE_WHEEL%"
		if errorlevel 1 exit /b 1
	) else (
		echo   Provided --sage-wheel path not found: %SAGE_WHEEL%
	)
) else (
	set "AUTO_SAGE="
	set "SAGE_TMP_ROOT=%TEMP%"
	if "!SAGE_TMP_ROOT!"=="" set "SAGE_TMP_ROOT=%LOCALAPPDATA%\Temp"
	if "!SAGE_TMP_ROOT!"=="" set "SAGE_TMP_ROOT=%cd%"
	set "SAGE_TMP_ROOT=!SAGE_TMP_ROOT:\\=\!"
	set "SAGE_TMP_ROOT=!SAGE_TMP_ROOT:"=!"
	set "SAGE_TMP_DIR=!SAGE_TMP_ROOT!\musubi-trainer-sage"
	if not exist "!SAGE_TMP_DIR!" (
		mkdir "!SAGE_TMP_DIR!" 2>nul
	)
	if not exist "!SAGE_TMP_DIR!" (
		echo   Warning: could not create temp dir at !SAGE_TMP_DIR!
		set "SAGE_TMP_DIR=%cd%\musubi-trainer-sage"
		if not exist "!SAGE_TMP_DIR!" mkdir "!SAGE_TMP_DIR!" 2>nul
	)

	echo   Looking up matching wheel from sdbds/SageAttention-for-windows releases...
	set "SAGE_DL_OUT=!SAGE_TMP_DIR!\sage-download.out.txt"
	set "SAGE_DL_ERR=!SAGE_TMP_DIR!\sage-download.err.txt"
	if exist "!SAGE_DL_OUT!" del /f /q "!SAGE_DL_OUT!" >nul 2>nul
	if exist "!SAGE_DL_ERR!" del /f /q "!SAGE_DL_ERR!" >nul 2>nul

	"%PY%" "scripts\download_sageattention_wheel.py" --out-dir "!SAGE_TMP_DIR!" 1>"!SAGE_DL_OUT!" 2>"!SAGE_DL_ERR!"
	set "SAGE_DL_RC=!errorlevel!"
	if "!SAGE_DL_RC!"=="0" (
		for /f "usebackq delims=" %%P in ("!SAGE_DL_OUT!") do (
			if "!AUTO_SAGE!"=="" set "AUTO_SAGE=%%P"
		)
	) else (
		echo   Auto-download helper failed with exit code !SAGE_DL_RC!.
		if exist "!SAGE_DL_ERR!" (
			for /f "usebackq delims=" %%L in ("!SAGE_DL_ERR!") do echo   [sage-helper] %%L
		)
	)

	if not "!AUTO_SAGE!"=="" (
		echo   Downloaded matching wheel: !AUTO_SAGE!
		"%PY%" -m pip install "!AUTO_SAGE!"
		if errorlevel 1 exit /b 1
	) else (
		echo   Could not auto-download a matching SageAttention wheel.
		echo   Trying local fallback path D:\ComfyUI\SageAttention\dist\sageattention-*.whl ...
		set "AUTO_SAGE="
		for %%F in ("D:\ComfyUI\SageAttention\dist\sageattention-*.whl") do (
			if "!AUTO_SAGE!"=="" set "AUTO_SAGE=%%~fF"
		)
		if not "!AUTO_SAGE!"=="" (
			echo   Auto-detected !AUTO_SAGE!
			"%PY%" -m pip install "!AUTO_SAGE!"
			if errorlevel 1 exit /b 1
		) else (
			echo   No SageAttention wheel found. Skipping.
			echo   Tip: use --sage-wheel PATH_TO_WHL for manual install.
		)
	)
)

echo.
echo [6/6] Summary
echo   Unified venv python: %cd%\venv\Scripts\python.exe
echo   CUDA profile: %CUDA_TAG%
echo.
echo Setup complete.
echo Musubi-Trainer will use this venv automatically:
echo   %cd%\venv\Scripts\python.exe

pause
exit /b 0
