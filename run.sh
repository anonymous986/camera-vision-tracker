#!/usr/bin/env bash
set -e

echo "======================================================================"
echo "  CAMERA-VISION-TRACKER - REAL-TIME PERCEPTION CORE"
echo "  Built by Parshwa"
echo "  Thank you for downloading and using this software."
echo "  https://github.com/anonymous986"
echo "======================================================================"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "[SETUP] Creating isolated Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

if ! python3 -c "import ultralytics, cv2, torch" &> /dev/null; then
    echo "[SETUP] Installing required dependencies..."
    pip install -r requirements.txt
fi

echo "[LAUNCH] Starting Camera Vision Tracker..."
python3 main.py "$@"
