@echo off
setlocal
echo ========================================================
echo     AudioSync AI - Terminal Karaoke & Vocal Isolator
echo ========================================================

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
python -c "import faster_whisper, soundfile, scipy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [SETUP] Installing required dependencies...
    pip install -r requirements.txt
)

echo [LAUNCH] Starting AudioSync AI...
python main.py %*
pause
