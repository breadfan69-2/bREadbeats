# bREadbeats

Real-time audio-to-motion app for e-stim control.
It listens to audio, detects rhythm, and generates smooth TCode output.

## What it does
- Captures live audio input
- Detects beat timing and intensity
- Maps rhythm to stroke movement
- Sends motion data to compatible devices

## Innovations (high-level)
- Fast beat-driven mapping tuned for real-time response
- Stable motion shaping to reduce jitter between beats (coming soon)
- Practical live controls for adjustment during playback (coming soon)

## Quick start
```bash
pip install -r requirements.txt
python run.py
```

## Build (Windows)
```bash
.venv/Scripts/pyinstaller.exe bREadbeats.spec
```

## Requirements
- Windows 10+
- Python 3.10+
- Compatible TCode receiver
