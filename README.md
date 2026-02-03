# bREadbeats

Audio beat detection to restim controller.

Based on [Breadbeats by breadfan-69](https://github.com/breadfan-69/Breadbeats) - motion patterns and stroke logic adapted from that project.

## Features

- **Real-time audio capture** - Captures system audio ("What-U-Hear") using sounddevice
- **Beat detection** - Uses spectral flux + peak energy detection (adjustable mix)
- **Stroke patterns** - Multiple modes:
  - Mode 1: Simple Circle
  - Mode 2: Figure-8
  - Mode 3: Random Arc
  - Mode 4: User (slider-controlled)
- **TCP connection to restim** - Sends T-code commands (L0/L1 for alpha/beta)
- **Mountain-range spectrum visualizer** - Pretty matplotlib display
- **Alpha/Beta position display** - Real-time circular position indicator
- **Jitter & Creep** - Micro-movements when idle
- **Full slider control** - All parameters adjustable in real-time

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Start restim and enable TCP listening on port 12347
2. Run bREadbeats:
   ```bash
   python run.py
   ```
3. Click "Connect" to connect to restim
4. Click "Start" to begin audio capture
5. Click "Play" to start sending commands
6. Adjust sliders to tune the response

## T-Code Format

Commands sent to restim follow the format:
- `L0xxxxIyyy` - Alpha axis position (0000-9999) over yyy milliseconds
- `L1xxxxIyyy` - Beta axis position (0000-9999) over yyy milliseconds

Example: `L05000I250 L15000I250` moves both axes to center over 250ms

## Files

- `run.py` - Entry point
- `main.py` - Qt GUI application
- `config.py` - Configuration dataclasses
- `audio_engine.py` - Audio capture and beat detection
- `network_engine.py` - TCP connection to restim
- `stroke_mapper.py` - Beat-to-stroke conversion

## Requirements

- Python 3.10+
- Windows (for WASAPI loopback audio capture)
- restim running with TCP enabled
