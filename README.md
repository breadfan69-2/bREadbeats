# bREadbeats

Real-time audio-reactive TCode generator for e-stim devices. Captures system audio, detects beats, and generates smooth stroke patterns.

![bREadbeats](https://img.shields.io/badge/License-Non--Commercial-blue) ![Python](https://img.shields.io/badge/Python-3.10+-green)

## Features

- **Beat Detection** - Spectral flux + peak energy with auto-adjusting parameters
- **4 Stroke Modes** - Circle, Spiral, Teardrop, User-controlled
- **Tempo Tracking** - Downbeat detection with pattern matching
- **Frequency Mapping** - Maps audio frequencies to P0/F0 TCode axes
- **Visualizers** - Spectrum, Mountain Range, Bar Graph, Phosphor displays
- **Preset System** - 5 slots for saving/loading configurations
- **Jitter & Creep** - Smooth idle motion when no beats

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

1. Start restim with TCP enabled (port 12347)
2. Connect → Start → Play

## System Requirements

- Windows 10+ (WASAPI loopback audio)
- Python 3.10+
- restim application

## Files

| File | Purpose |
|------|---------|
| main.py | GUI and application logic |
| audio_engine.py | Audio capture and beat detection |
| stroke_mapper.py | Beat-to-stroke pattern conversion |
| network_engine.py | TCP/TCode communication |
| config.py | Configuration dataclasses |

## License

Non-commercial use only. See [LICENSE](LICENSE) for details.

---

*Created with assistance from Claude AI*
