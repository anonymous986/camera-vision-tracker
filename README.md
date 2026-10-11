<div align="center">

# camera-vision-tracker

**Real-time multi-class object detection and DIoU motion tracking for webcam streams.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c)](https://pytorch.org)
[![Author](https://img.shields.io/badge/Author-Parshwa-00f2fe)](https://github.com/anonymous986)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-OpenImages-00ffff)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An asynchronous computer vision pipeline running multi-class object detection, human face clustering, and motion-compensated tracking on live video streams at 60+ FPS.

</div>

---

## Features

- **High-Vocabulary Recognition**: Uses YOLOv8 with 600+ categories (Open Images V7) to recognize everyday tools, electronics, furniture, vehicles, and people.
- **Calibrated Detection Sensitivity**: Tuned confidence thresholding (`conf=0.10`) with soft suppression, capturing smaller desktop objects (pens, glasses, keys, earbuds) that standard thresholds discard.
- **Velocity-Aware DIoU Tracker**: Tracks objects across frames with velocity prediction and occlusion coasting. Matches across semantic synonyms to prevent ID switching.
- **Native Face Retention**: Clusters human bodies and faces directly without requiring external ONNX weights.
- **Hardware Negotiation**: Automatically probes DirectShow camera indices (0, 1, 2) on Windows with seamless fallback.

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/anonymous986/camera-vision-tracker.git
cd camera-vision-tracker
```

### 2. Run with One Command

**Windows:**
```cmd
run.bat
```

**Linux / macOS:**
```bash
chmod +x run.sh && ./run.sh
```

---

## Manual Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

---

## Hotkeys & Options

- **`q`**: Quit the application.
- **`s`**: Save a snapshot screenshot to the current directory.
- **`c`**: Reset motion tracking history.

### CLI Flags

```bash
# Specify camera index (default: 0)
python main.py --cam 1

# Stream from a local video file
python main.py --video "path/to/video.mp4"

# Adjust detection threshold (default: 0.10)
python main.py --conf 0.15
```

---

## Performance

Tested on NVIDIA RTX 5050 Laptop GPU / CUDA 12.x:
- **Frame Rate**: ~60 FPS @ 1280x720
- **Inference Latency**: ~15 ms per frame

---

## Author & Acknowledgments

**Built by Parshwa**  
*Thank you for downloading and using this project.*  
Official Profile: [github.com/anonymous986](https://github.com/anonymous986)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
