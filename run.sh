#!/usr/bin/env bash
set -e

echo "========================================================"
echo "    AudioSync AI - Terminal Karaoke & Vocal Isolator"
echo "========================================================"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "[SETUP] Creating isolated Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

python3 -c "import faster_whisper, soundfile, scipy" 2>/dev/null || {
    echo "[SETUP] Installing dependencies..."
    pip install -r requirements.txt
}

echo "[LAUNCH] Starting AudioSync AI..."
python3 main.py "$@"
