# bREadbeats Agent Reference Document

## Purpose
This document serves as a canonical reference for future AI agents working on bREadbeats. It explains the core architecture, key features, and their intended behaviors. **Review this document before making changes** to avoid accidentally breaking critical functionality.

---

## Program Overview

**bREadbeats** is a real-time audio-reactive TCode generator for restim devices. It captures system audio, detects beats and tempo, and generates smooth circular/arc stroke patterns in the alpha/beta plane. Additionally, it monitors dominant audio frequencies and translates them to TCode values for pulse frequency (P0) and carrier frequency (F0).

### Core Signal Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐
│  Audio Input    │───▶│  Beat Detection │───▶│  StrokeMapper   │───▶│ TCP/TCode   │
│  (WASAPI loop)  │    │  (AudioEngine)  │    │ (Patterns/Idle) │    │ to restim   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────┘
        │                       │                      │
        ▼                       ▼                      ▼
   Spectrum FFT         Tempo Tracking          Alpha/Beta coords
   Dominant Freq        Downbeat Detection      Volume/Duration
        │
        ▼
┌─────────────────┐
│  P0/F0 TCode    │──▶ Pulse & Carrier frequency commands
│  Frequency Map  │
└─────────────────┘
```

### Key Files
- **main.py** - GUI (PyQt6), wiring, P0/F0 computation, visualizers
- **audio_engine.py** - Audio capture, FFT, beat/tempo detection, downbeat tracking
- **stroke_mapper.py** - Beat→stroke conversion, 4 stroke modes, jitter/creep
- **network_engine.py** - TCP connection to restim, TCodeCommand class
- **config.py** - All configuration dataclasses and defaults

---

## Critical Features & Behaviors

### 1. Stroke Modes [1-4] ⚠️ HANDLE WITH CARE

**Reference commit:** `097fea9` (latest, most correct)

Stroke modes are extremely sensitive to code changes. All 4 modes use circular coordinates around (0,0) in the alpha/beta plane. They generate **smooth arcs**, NOT straight lines.

| Mode | Name | Behavior |
|------|------|----------|
| 1 | Circle | Standard circle trace, 2π per beat |
| 2 | Spiral | Archimedean spiral, N revolutions per cycle |
| 3 | Teardrop | Piriform curve, 1/4 phase advance (slower) |
| 4 | User | Elliptical motion with flux/peak response via axis weights |

**Arc Resolution Formula (CRITICAL):**
```python
n_points = max(8, int(beat_interval_ms / 10))  # 1 point per 10ms
# For downbeats:
n_points = max(16, int(measure_duration_ms / 20))  # 1 point per 20ms
```
DO NOT change this to `/ 200` or similar - it breaks smooth motion.

**Axis Weights:**
- Modes 1-3: Scales axis amplitude (0=off, 1=normal, 2=double)
- Mode 4: Controls flux vs peak response (0=flux, 1=balanced, 2=peak)

### 2. Downbeat Detection & Tracking

**Reference commit:** `ef7c72e`

Energy-based downbeat detection (replaces simple counter):
- Accumulates beat energy at each measure position (1-4)
- Older measures decay at 0.85x so recent music dominates
- After 2 full measures, strongest position = beat 1
- Zero queuing/blocking - pure arithmetic inline

**On downbeat:**
- Full measure-length stroke loop (~4x beat duration)
- Extended arc duration for emphasis
- Prints `⬇ DOWNBEAT` in logs

### 3. Jitter (Micro-Circles When Idle)

**Correct behavior - reference commits:** `059be93`, `0809eb7`, `2430684`  
**AVOID behavior from:** `291428c` (random movements - incorrect)

**Correct implementation:**
- **Sinusoidal** micro-circles (NOT random movements)
- Advance jitter_angle based on intensity * 0.15 (slower, smoother)
- Amplitude controls circle size (0.01-0.1 range)
- Creep radius pulled inward by jitter amplitude to prevent clipping at ±1.0

```python
# CORRECT jitter pattern
jitter_speed = jitter_cfg.intensity * 0.15
self.state.jitter_angle += jitter_speed
alpha_target = base_alpha + np.cos(self.state.jitter_angle) * jitter_r
beta_target = base_beta + np.sin(self.state.jitter_angle) * jitter_r
```

### 4. Creep (Slow Drift When Idle)

**Reference commit:** `003e307`

**Behavior:**
- Tempo-synced rotation: speed=1.0 moves 1/4 circle per beat
- When no BPM: slowly oscillates toward center (not rotating)
- Creep radius pulled inward by jitter amplitude (`0.98 - jitter_r`)
- Smooth creep reset to (0, -0.5) after beat strokes complete over 400ms

```python
# Creep angle increment per update (at 60fps)
angle_increment = (np.pi / 2.0) / updates_per_beat * creep_cfg.speed
```

### 5. Beat Detection & Tempo Prediction

**Reference commit:** `5ab4469`

**Features:**
- Combined spectral flux + peak energy detection
- Butterworth bandpass filter for bass isolation (30-200Hz default)
- Weighted tempo averaging (recent beats weighted 0.5-1.5)
- Exponential smoothing (factor 0.7) for stable BPM
- 2000ms timeout resets tempo tracking (preserves last known BPM)

**Flux-based stroke behavior:**
- Low flux (<threshold): Only full strokes on downbeats
- High flux (≥threshold): Full strokes on every beat
- Flux scaling weight affects stroke size (0.5-1.5 range)

### 6. Idle Suppression / Silence Handling

**Reference commit:** `9d53193`

**Thresholds (lowered for sensitive detection):**
```python
quiet_flux_thresh = cfg.flux_threshold * 0.03
quiet_energy_thresh = beat_cfg.peak_floor * 0.3
```

**Behavior:**
- Fade-out over 2 seconds when truly silent
- Reset tempo/downbeat after `silence_reset_ms` (default 400ms)
- Idle motion suppressed when `_fade_intensity < 0.01`

### 7. Frequency Detection → P0/F0 TCode & Stroke Depth

**Reference commit:** `710d63e` (range slider wiring), `097fea9` (Hz*67 formula)

**P0/F0 Pipeline:**
1. Monitor dominant frequency in configurable Hz range (30-22050 Hz sliders)
2. Normalize to 0-1 based on monitor min/max
3. Apply freq_weight to scale effect (0.5 + (norm - 0.5) * weight)
4. Convert to TCode:
   - **P0**: `tcode_val = int(freq_hz * 67)` (0-150 Hz → 0-9999 TCode)
   - **F0**: `tcode_val = int((display - 500) * 10)` (500-1500 display → 0-9999 TCode)

**Stroke Depth Frequency Mapping:**
1. **depth_freq_range_slider** (30-22050 Hz) defines which frequencies affect stroke depth
2. **freq_depth_factor** slider (0-2.0) controls how much frequency affects depth
3. Frequency analysis: Lower frequencies (bass) → deeper strokes, Higher frequencies → shallower strokes
4. Formula in stroke_mapper.py:
   ```python
   freq_factor = self._freq_to_factor(event.frequency)  # 0-1 based on depth_freq range
   depth = minimum_depth + (1 - minimum_depth) * (1 - freq_depth_factor * freq_factor)
   ```

**Range Slider Wiring (for all freq bands):**
- Visualizer bands drag → update config → update sliders
- Slider changes → update config → update visualizer bands

**Frequency bands on visualizer:**
- Red (50% height): Beat detection band
- Green (40% height): Stroke depth band  
- Blue (33% height): P0 TCode band
- Cyan (25% height): F0 TCode band

---

## Volume Control

**Reference commit:** `097fea9` (removed silence volume factor)

Volume is controlled ONLY by:
1. Main volume slider
2. Fade intensity (silence fade-out)

The `_tcode_silence_volume_factor` was REMOVED - do not re-add it.

```python
volume = self.get_volume() * fade  # CORRECT
# NOT: volume = self.get_volume() * silence_factor * fade
```

---

## Arc Return Animation

After a beat stroke completes, smooth arc return over 400ms:
- Uses quadratic Bezier curve for curved path (not straight line)
- Returns to (0, -0.5) for non-spiral modes, (0, 0) for spiral
- Minimum 200ms per command for smooth motion

---

## Key Commits Reference Table

| Commit | Feature | Notes |
|--------|---------|-------|
| `097fea9` | Stroke modes reference | Arc resolution restored, volume simplified |
| `87214dc` | Strokemapper behavior | Perfect stroke controller, but patterns were not correct |
| `ef7c72e` | Downbeat detection | Energy-based, replaces simple counter |
| `059be93` | Jitter restoration | Sinusoidal micro-circles (correct) |
| `0809eb7` | Jitter full circles | Creep radius pulled inward, no clipping |
| `2430684` | Jitter size/teardrop | Much smaller circles, 1/4 phase for teardrop |
| `291428c` | Jitter (AVOID) | Random movements - INCORRECT, do not use |
| `003e307` | Creep reset | Smooth reset to (0, -0.5), custom presets |
| `5ab4469` | Beat/tempo prediction | Flux-based adaptive stroking |
| `9d53193` | Idle suppression | Lower thresholds for sensitive detection |
| `710d63e` | Freq range sliders | Butterworth filter, PyQtGraph migration |
| `058d94a` | BETA reference | Basic program function reference |

---

## UI Components

### Range Sliders
Dual-handle sliders for min/max pairs:
- **Beat detection freq** (30-22050 Hz) → Red band on visualizer → Butterworth filter range
- **Stroke depth freq** (30-22050 Hz) → Green band on visualizer → Controls stroke depth based on bass/treble content  
- **Pulse monitor freq** (30-22050 Hz) → Blue band on visualizer → Audio range for P0 TCode generation
- **Pulse TCode freq** (0-150 Hz) → P0 output range (Hz*67 → TCode 0-9999)
- **Carrier monitor freq** (30-22050 Hz) → Cyan band on visualizer → Audio range for F0 TCode generation  
- **Carrier TCode freq** (500-1500) → F0 output range ((display-500)*10 → TCode 0-9999)

### Preset System
5-slot custom presets storing all settings:
- Left-click: Load preset
- Right-click: Save preset
- Blue button = has saved preset
- Green border = currently active

---

## Real-Time Data Requirement ⚠️ CRITICAL

**NEVER send cached or queued data — ALWAYS use live values.**

TCode commands must always reflect the current state at the moment they are computed, not stale/queued values from earlier. This ensures responsive, real-time control.

**Current Implementation:**
- GUI widget values (sliders, checkboxes) are synced to cached variables every 100ms in `_update_display()`
- `_compute_and_attach_tcode()` reads from these cached values (thread-safe)
- Each command is computed fresh with current slider positions

**Rules:**
1. Never batch/queue commands for later sending
2. Never compute TCode values ahead of time and store them
3. Always read current slider/config values when generating each command
4. Cached values exist ONLY for thread safety (GUI thread → audio thread), not for time-shifting

```python
# CORRECT: Read current values each time
p0_val = int(tcode_min + norm * (tcode_max - tcode_min))  # Uses current cached slider values

# WRONG: Pre-compute and queue
queued_commands.append(pre_computed_p0)  # NEVER queue commands
```

---

## Common Pitfalls to Avoid

1. **Sending cached/queued data** - Always compute live; never pre-compute or queue commands
2. **Breaking arc resolution** - Don't change `/ 10` or `/ 20` divisors
3. **Adding silence volume factor back** - Volume is only slider + fade
4. **Changing jitter to random** - Keep sinusoidal micro-circles
5. **Missing wiring for new sliders** - Always wire both directions (visualizer↔config↔slider)
6. **Modifying stroke mode math** - Check against commit `097fea9` first
7. **Breaking creep radius** - Must be `0.98 - jitter_r` to prevent clipping

---

## Testing Checklist

Before committing changes:
- [ ] All 4 stroke modes produce smooth arcs (not lines)
- [ ] Jitter creates small circles (not random jumps)
- [ ] Creep rotates smoothly (not jerky)
- [ ] Downbeat detection triggers extended strokes
- [ ] P0/F0 display shows correct Hz values
- [ ] Preset load/save works for all settings
- [ ] Range slider dragging updates visualizer bands
- [ ] Silence properly fades out and resets tempo
- [ ] No volume factor beyond slider and fade
- [ ] All TCode values are computed live (no queued/cached commands)
- [ ] Frequency-to-depth mapping works (bass = deeper strokes)
- [ ] All 6 range sliders properly wired to config and visualizers

---

## How to Verify Behavior

1. **Stroke modes**: Enable each mode, play music, watch position dot trace smooth curves
2. **Downbeat**: Watch BPM display for `[1] ⬇` indicator on beat 1
3. **Jitter**: Let audio go silent, enable jitter, observe smooth micro-circles
4. **Creep**: Enable creep, verify slow rotation follows tempo
5. **P0/F0**: Enable pulse checkbox, verify Hz display updates with music

---

*Document created: 2026-02-07*  
*Reference commit for document: `097fea9`*
