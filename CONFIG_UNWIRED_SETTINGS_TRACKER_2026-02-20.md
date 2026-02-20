# Config Settings Not Wired (Tracker)

Updated: 2026-02-20

## Confirmed not wired

| Setting key | Status | Evidence | Notes |
|---|---|---|---|
| `stroke.freq_depth_factor` | Removed from schema/config/UI (2026-02-20) | Previously only found in `config.py` dataclass definitions; no runtime reads in `main.py`, `audio_engine.py`, `beat_intelligence.py`, `stroke_mapper.py` | Safely removed per dead-wiring cleanup scope |
| `stroke.downbeat_high_band_relax` | Removed from schema/config/UI (2026-02-20) | Previously found in `config.py` and `main.py` UI slider wiring, but not consumed by runtime gate logic in `beat_intelligence.py` | Safely removed per dead-wiring cleanup scope |
| `stroke.high_band_window_frames` | Retained intentionally (keep-list, 2026-02-20) | Value is set via UI in `main.py` and read by `BeatIntelligence._get_high_band_presence_status()`, but that helper is not currently used by live gate cascade in `build_decision` | Kept by decision; still a future wiring candidate |
| `stroke.high_tip_freq_low_hz` | Retained intentionally (keep-list, 2026-02-20) | Found in `config.py` and `main.py` UI/state updates, but not consumed by runtime gate logic in `beat_intelligence.py` | Kept by decision; still a future wiring candidate |
| `stroke.high_tip_freq_high_hz` | Retained intentionally (keep-list, 2026-02-20) | Found in `config.py` and `main.py` UI/state updates, but not consumed by runtime gate logic in `beat_intelligence.py` | Kept by decision; still a future wiring candidate |
| `stroke.mid_bass_freq_high_hz` | Retained intentionally (keep-list, 2026-02-20) | Found in `config.py` and `main.py` UI/state updates, but not consumed by runtime analysis/gate logic in `beat_intelligence.py` | Kept by decision; still a future wiring candidate |
| `stroke.mid_bass_freq_low_hz` | Retained intentionally (keep-list, 2026-02-20) | Found in `config.py` and `main.py` UI/state updates, but not consumed by runtime analysis/gate logic in `beat_intelligence.py` | Kept by decision; still a future wiring candidate |
| `stroke.minimum_depth` | Removed from schema/config/UI (2026-02-20) | Previously found in `config.py` and forced/reset in `main.py`, but no runtime motion logic reads detected | Safely removed per dead-wiring cleanup scope |
| `stroke.phase_advance` | Removed from schema/config/persist wiring (2026-02-20) | Previously found in `config.py` and `close_persist_wiring.py`; no runtime motion/gating reads detected | Safely removed per dead-wiring cleanup scope |
| `stroke.single_stroke_bpm_cutoff` | Removed from schema/config (2026-02-20) | Previously defined in `config.py`; no runtime cadence logic reads this key (cadence derived elsewhere) | Safely removed per dead-wiring cleanup scope |
| `stroke.stroke_fullness` | Removed from schema/config (2026-02-20) | Previously found in `config.py` only; no runtime reads detected in motion path | Safely removed per dead-wiring cleanup scope |
| `stroke.stroke_max` | Removed from schema/config/UI (2026-02-20) | Previously set via UI in `main.py` and defined in `config.py`, but no runtime motion reads detected | Safely removed per dead-wiring cleanup scope |
| `stroke.stroke_min` | Removed from schema/config/UI (2026-02-20) | Previously set via UI in `main.py` and defined in `config.py`, but no runtime motion reads detected | Safely removed per dead-wiring cleanup scope |

## How to use this file

- Add one row per audited setting.
- Use statuses: `Confirmed not wired`, `Retained intentionally`, `Removed`, `Partially wired`, `Wired`, `Unknown`.
- Keep notes short and include the runtime module where behavior should exist.
