@echo off
setlocal
echo ======================================================================
echo   CAMERA-VISION-TRACKER - REAL-TIME PERCEPTION CORE
echo   Built by Parshwa
echo   Thank you for downloading and using this software.
echo   https://github.com/anonymous986
echo ======================================================================

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

REM Create virtual environment if missing
if not exist "venv" (
    echo [SETUP] Creating isolated Python virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Check and install dependencies
python -c "import ultralytics, cv2, torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo [SETUP] Installing required dependencies...
    pip install -r requirements.txt
)

echo [LAUNCH] Starting Camera Vision Tracker...
python main.py %*
pause
