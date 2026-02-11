# bREadbeats Agent Reference Document

## Purpose
This document serves as a canonical reference for future AI agents working on bREadbeats. It explains the core architecture, key features, and their intended behaviors. **Review this document before making changes** to avoid accidentally breaking critical functionality.

⚠️ **IMPORTANT:** This document is part of the codebase. **Always commit and push AGENT_REFERENCE.md** whenever you update it during feature work. Keeping it in sync with code changes ensures future agents have accurate information.

---

## Program Overview

**bREadbeats** is a real-time audio-reactive TCode generator for restim devices. It captures system audio, detects beats and tempo, and generates smooth circular/arc stroke patterns in the alpha/beta plane. Additionally, it monitors dominant audio frequencies and translates them to TCode values for pulse frequency (P0) and carrier frequency (C0), and feeds the same dominant frequency into StrokeMapper to scale stroke depth (bass → deeper, treble → shallower).

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
│  P0/C0 TCode    │──▶ Pulse & Carrier frequency commands
│  Frequency Map  │
└─────────────────┘
```

### Key Files
- **main.py** - GUI (PyQt6), wiring, P0/C0 computation, visualizers
- **audio_engine.py** - Audio capture, FFT, beat/tempo detection (classic + z-score dual-path), downbeat tracking, dominant-frequency estimation per frame; contains `ZScorePeakDetector` class
- **stroke_mapper.py** - Beat→stroke conversion, 4 stroke modes, jitter/creep, depth scaling from dominant frequency via `_freq_to_factor`
- **network_engine.py** - TCP connection to restim, TCodeCommand class
- **config.py** - All configuration dataclasses and defaults

### Dependencies

**Runtime Requirements** (`requirements.txt`):
- **PyQt6** ≥6.5.0 - GUI framework (main.py, visualization widgets)
- **numpy** ≥1.24.0 - Audio signal processing
- **scipy** ≥1.10.0 - Butterworth bandpass filter (`scipy.signal.butter`, `sosfilt`, `sosfilt_zi`)
- **sounddevice** ≥0.4.6 - Audio input capture (PortAudio wrapper)
- **pyaudiowpatch** ≥0.2.12 - Audio device enumeration (Windows WASAPI support)
- **pyqtgraph** ≥0.13.0 - Real-time waveform/spectrum/beat visualization
- **comtypes** ≥1.2.0 - Windows COM interface for WASAPI loopback audio
- **aubio** ≥0.4.9 - Optional: Additional beat detection (falls back gracefully if missing)
- **Pillow** ≥10.0.0 - Image processing
- **pictex** ≥0.1.0 - Splash screen styled text rendering

**Development/Build** (`requirements-dev.txt`):
- **PyInstaller** ≥6.0.0 - Standalone exe packaging (development only)

**NOT INCLUDED** (legacy/unused):
- ~~matplotlib~~ - Replaced by pyqtgraph for real-time plotting
- ~~moderngl/moderngl-window~~ - Never used in codebase
- ~~python-dateutil, six, fonttools, cycler, kiwisolver, contourpy~~ - matplotlib dependencies (removed with matplotlib)

### Installation

```bash
# Core runtime
pip install -r requirements.txt

# For development/building exe
pip install -r requirements-dev.txt
```

---

## TCode Protocol Reference (restim Electrostimulation)

bREadbeats communicates with **restim** (electrostimulation software) via TCP using the **TCode** protocol. Each command is a space-separated string of axis commands with interpolation times.

### TCode Format
```
<AXIS><VALUE>I<DURATION_MS>\n
```
Example: `L04999I100 L14999I100 V09999I100 P00500I250 C05000I900\n`

### Axis Reference Table

| Axis | Name | TCode Range | Real-World Units | Feeling / Effect |
|------|------|-------------|-------------------|-----------------|
| **L0** | Alpha (vertical) | 0000-9999 | Position -1.0 to 1.0 | Stroke position — vertical axis of stimulation pattern |
| **L1** | Beta (horizontal) | 0000-9999 | Position -1.0 to 1.0 | Stroke position — horizontal axis of stimulation pattern |
| **V0** | Volume | 0000-9999 | 0.0 to 1.0 | Overall stimulation intensity |
| **P0** | Pulse Frequency | 0000-9999 | ~1.5-150 Hz (÷67) | Rate of pulses. Higher = buzzier/faster. Lower = thuddy/slower. |
| **C0** | Carrier Frequency | 0000-9999 | 500-1500 display (val/10+500) | Carrier wave frequency. Affects texture/character of stimulation. |
| **P1** | Pulse Width | 0000-9999 | Device-dependent | **Higher = stronger, smoother** feeling. Lower = more prickly, lighter (less strong because narrower pulse). Width of a single pulse within the wavelet. |
| **P3** | Rise Time | 0000-9999 | Device-dependent | **Higher = smoother, gentler** feeling. Lower = harsher but still pleasant. Controls how quickly each pulse ramps up. Less dramatic difference than P1. |

### Coordinate System
- **restim** uses L0=vertical, L1=horizontal (swapped & negated from bREadbeats internal alpha/beta)
- **Rotation:** 90° clockwise transform applied in `TCodeCommand.to_tcode()`
- **Interpolation:** Each axis includes `I<ms>` for smooth transitions

### Currently Implemented Axes
- ✅ **L0, L1** — Stroke position (alpha/beta from StrokeMapper)
- ✅ **V0** — Volume (from audio RMS)
- ✅ **P0** — Pulse frequency (music-reactive via dominant frequency or dot speed)
- ✅ **C0** — Carrier frequency (music-reactive, same input modes as P0)
- ✅ **P1** — Pulse width (music-reactive, maps RMS energy to pulse width)
- ✅ **P3** — Rise time (music-reactive, maps spectral centroid to rise time)

### Music→Parameter Mapping Philosophy

Each TCode axis maps an audio feature to a physical sensation. The goal is **musical expressiveness** — the stimulation should "feel like" the music.

| Parameter | Best Audio Source | Mapping Rationale |
|-----------|-------------------|-------------------|
| **P0** (Pulse Freq) | Dominant frequency in band | Bass notes → low pulse rate (thuddy), treble → high rate (buzzy) |
| **C0** (Carrier Freq) | Dominant frequency in band | Similar to P0 but controls carrier wave texture |
| **P1** (Pulse Width) | RMS energy / volume level | Loud passages → wider pulses (stronger, smoother). Quiet → narrow (lighter, prickly). Intensity follows loudness naturally. |
| **P3** (Rise Time) | Spectral centroid / brightness | Bright/treble-heavy → short rise (exciting, snappy). Warm/bassy → long rise (flowing, smooth). Musical brightness maps to stimulation sharpness. |

### Input Modes (shared across frequency-reactive axes)
1. **Hz (dominant freq)** — Extracts dominant frequency from a configurable frequency band via FFT peak detection
2. **Speed (dot movement)** — Uses the velocity of the stroke pattern dot in alpha/beta space
3. **Volume (RMS energy)** — Uses audio loudness level
4. **Brightness (spectral centroid)** — Uses spectral centroid as brightness measure

All modes use **250ms sliding window averaging** for smooth transitions, with configurable weight and invert options.

---

## Key Design Principles

### 1 BPM Precision Matters

**bREadbeats is an auto-adjusting metronome, not just a music visualizer.** The rotating position dot completes one full cycle per beat — if the target BPM is off by even 1 BPM, the dot will visibly drift out of sync with the music over a few measures. This is fundamentally different from music listening, where ±1 BPM is imperceptible.

**Rule:** Never dismiss 1 BPM differences as "close enough." All tempo tracking, auto-alignment, and BPM feedback systems must aim for exact integer BPM lock. When in doubt, prioritize precision over speed of convergence.

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

**Reference commit:** `ef7c72e` (energy-based) + Latest (pattern matching)

Energy-based downbeat detection (replaces simple counter):
- Accumulates beat energy at each measure position (1-4)
- Older measures decay at 0.85x so recent music dominates
- After 2 full measures, strongest position = beat 1
- Zero queuing/blocking - pure arithmetic inline

**Downbeat Validation (Pattern Matching):**
- Energy-based detection identifies downbeat candidate
- If pattern matching enabled: validate timing against predicted downbeat
- Predicted time = last confirmed downbeat + (measure_interval = beat_interval × 4)
- Phase error = |detected_time - predicted_time| in milliseconds
- Accept downbeat if phase_error ≤ 100ms default tolerance
- Log shows `[Downbeat]` if accepted or `[Downbeat REJECTED]` if timing off

**On accepted downbeat:**
- Full measure-length stroke loop (~4x beat duration)
- Extended arc duration for emphasis
- **If tempo LOCKED:** 25% amplitude boost (1.25× multiplier) on radius and stroke_len
- Prints `⬇ DOWNBEAT` with pattern match counter and phase error

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

**Reference commit:** Latest (beat detection gain fix + tempo lock)

**⚠️ CRITICAL FIX: Audio Gain in Butterworth Path**
When using Butterworth filter (time-domain path), audio gain MUST be applied:
```python
band_energy = np.sqrt(np.mean(beat_mono ** 2))
band_energy = band_energy * self.config.audio.gain  # ← APPLY HERE (line ~360)
```
If gain is NOT applied in Butterworth path, beat detection will fail completely (zero beats) even when audio is present. The FFT fallback path also applies gain, but Butterworth is the PRIMARY path and must have it.

**Single Gain Application Rule:**
- Apply gain ONCE in Butterworth section
- FFT fallback comment states "gain already applied, no need to apply again"
- Double-application causes excessive band_energy (~0.5+), breaking detection thresholds

**Peak Floor Slider (Critical for Initial Beat Detection):**
- Range: 0.01-0.14 (was 0.0-0.8 before fix)
- Reset value: 0.08 (must stay at 0.08)
- Typical band_energy after gain: 0.08-0.15
- **If reset > 0.08, beats won't be detected initially because peak_floor > band_energy**

**Beat Detection Features:**
- **Two-path detection system** (classic + z-score, see below)
- Combined spectral flux + peak energy detection
- Butterworth bandpass filter for bass isolation (30-200Hz default)
- Weighted tempo averaging (recent beats weighted 0.5-1.5)
- Exponential smoothing (factor 0.7) for stable BPM
- 2000ms timeout resets tempo tracking (preserves last known BPM)

**Z-Score Adaptive Peak Detection (Brakel, 2014) — Multi-Band:**
`_detect_beat()` uses two parallel detection paths — a beat fires if EITHER triggers:

| Path | How it works | Strengths |
|------|-------------|-----------|
| **Classic** | peak_floor + rise_sens + threshold multiplier | User-tunable, works with auto-ranging metrics |
| **Z-Score (multi-band)** | 4 sub-band z-score detectors, OR logic | Self-adapting, tracks instrument hopping |

**Multi-Band Z-Score System (4 sub-bands):**

Each band gets its own `ZScorePeakDetector(lag=30, threshold=2.5, influence=0.05)`:

| Band | Frequency Range | Typical Source |
|------|----------------|----------------|
| `sub_bass` | 30–100 Hz | Kick drum, sub-bass |
| `low_mid` | 100–500 Hz | Bass guitar, toms |
| `mid` | 500–2000 Hz | Snare, vocals |
| `high` | 2000–16000 Hz | Hi-hats, cymbals |

```python
# Per-band state (created in __init__)
self._zscore_bands = [('sub_bass',30,100), ('low_mid',100,500), ('mid',500,2000), ('high',2000,16000)]
self._zscore_detectors = {name: ZScorePeakDetector(...) for name, _, _ in self._zscore_bands}
self._band_energies = {name: 0.0 ...}        # Current RMS energy per band
self._band_zscore_signals = {name: 0 ...}     # 1=fired, 0=quiet per band
self._band_fire_history = {name: [] ...}      # Rolling fire count (60 frames ≈ 1s)
self._primary_beat_band = 'sub_bass'          # The "winning" band
```

**`_update_multiband_zscore(spectrum)`** — called every audio frame:
1. Extracts per-band RMS energy from FFT spectrum bins
2. Applies audio gain to each band's energy
3. Feeds each band's energy to its `ZScorePeakDetector`
4. Appends fire/quiet to rolling history (60 frames ≈ 1s window)
5. Selects primary band = highest fire count with hysteresis (+2 fires to switch)

**How the paths now combine:**
1. Multi-band z-score runs on EVERY frame via `_update_multiband_zscore(spectrum)`
2. Classic path: peak_floor → rise_sens → threshold check → fires if energy exceeds
3. Z-score path: fires if **ANY** band's z-score signals +1 AND overall energy > 1.1× avg
4. Beat detected if **either** path fires (after refractory guard)
5. Log shows source + band info: `[Z] bands=band=high fired=high,sub_bass`
6. All band detectors reset on tempo reset (silence) for fresh baseline

**Band Switching Behavior:**
- Primary band tracks whichever instrument is most rhythmically consistent
- Example: hi-hats fire `high` band steadily → primary = `high`
- When kick drum enters, `sub_bass` fires more → primary switches to `sub_bass`
- Hysteresis: new band must have 2+ more fires than current to switch
- Log: `[MultiBand] Primary band switched | band=sub_bass fires=8`

**Downbeat Pattern Matching (Tempo Lock System):**
New strict validation ensures downbeats match predicted tempo:
```python
pattern_match_tolerance_ms: float = 100.0  # Max ±100ms deviation
consecutive_match_threshold: int = 3       # 3 consecutive matches = LOCKED
downbeat_pattern_enabled: bool = True      # Enable/disable
```
When enabled:
1. First downbeat establishes baseline predicted time
2. Next downbeats must occur within ±100ms of prediction
3. After 3 consecutive matches: **tempo is LOCKED** (gives 25% stroke boost)
4. Counter resets if error exceeds tolerance
5. Log shows `[Downbeat]` if accepted or `[Downbeat REJECTED]` if error too large

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

### 7. Frequency Detection → P0/C0 TCode & Stroke Depth

**Reference commit:** `710d63e` (range slider wiring), `097fea9` (Hz*67 formula)

**P0/C0 Pipeline:**
1. Monitor dominant frequency in configurable Hz range (30-22050 Hz sliders)
2. Normalize to 0-1 based on monitor min/max
3. Apply freq_weight to scale effect (0.5 + (norm - 0.5) * weight)
4. Convert to TCode:
   - **P0**: `tcode_val = int(freq_hz * 67)` (0-150 Hz → 0-9999 TCode)
   - **C0**: `tcode_val = int((display - 500) * 10)` (500-1500 display → 0-9999 TCode)
     - Note: restim uses C0 for carrier frequency (not F0)

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
- Cyan (25% height): C0 TCode band (carrier frequency)

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
| Latest | Beat detection gain + tempo lock | CRITICAL: Gain in Butterworth path (~line 360), pattern matching (±100ms tolerance), 25% downbeat boost when locked |
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

### UI Naming Convention — NO TCode Tags
**IMPORTANT:** Never use TCode axis tags (P0, C0, P1, P2, P3, L0, L1, V0, etc.) in user-facing UI text.
Always use the human-readable function name instead:
| Internal / TCode | UI Label |
|-----------------|----------|
| P0 | Pulse Freq |
| C0 | Carrier Freq |
| P1 | Pulse Width |
| P2 | Interval Random |
| P3 | Rise Time |
| L0/L1 | (stroke axes — not user-labeled) |
| V0 | (vibration — not user-labeled) |

TCode tags are fine in code comments, log output, and agent/developer docs — just not in anything the user sees (labels, dialogs, tooltips, group box titles, checkbox text, slider names).

### Window & Layout
- **Default size:** `resize(1100, 950)`, minimum `setMinimumSize(400, 300)`
- **Main layout:** Vertical `QSplitter` — top half is visualizers (min 220px), bottom half is settings tabs only
- **Presets row** (presets panel + "Whip the Llama" button) is **outside** the splitter, locked at the bottom of `main_layout` — never compresses
- **Splitter ratio:** Stretch factors `5:2` (~72% visualizers, ~28% tabs)
- **Visualizer panels** (`_create_spectrum_panel`, `_create_position_panel`) are plain `QWidget` containers with zero margins — NOT `QGroupBox` (groupboxes were removed for cleaner look)
- **Connection/Controls panels** are also plain `QWidget` (no `QGroupBox` borders) for minimal chrome
- **Controls layout:** Grid row 0 has Start/Play buttons, volume slider, freq display stack (110px), then a right-side vertical stack (100px) with traffic light → BPM label → beat/downbeat indicators
- **Settings panels** use `CollapsibleGroupBox` (custom windowshade widget, 12px bold title, 30px click area) inside `NoWheelScrollArea` tabs

### Position Visualizer
- **PositionCanvas** displays alpha/beta as a dot on a circular field
- **90° CCW rotation** is applied after the user rotation transform: `x_display = -y_rot`, `y_display = x_rot`
- Background color `#3d3d3d` matches the window background
- Circle pen `#555555`, crosshair pen `#4a4a4a` (softened, non-distracting)
- Trail uses 20-point scatter with decaying alpha

### Range Sliders
Dual-handle sliders for min/max pairs:
- **Beat detection freq** (30-22050 Hz) → Red band on visualizer → Butterworth filter range
- **Stroke depth freq** (30-22050 Hz) → Green band on visualizer → Controls stroke depth based on bass/treble content  
- **Pulse Freq monitor** (30-22050 Hz) → Blue band on visualizer → Audio range for pulse freq generation
- **Pulse Freq sent value** (0-9999) → Direct TCode output range for P0
- **Carrier Freq monitor** (30-22050 Hz) → Cyan band on visualizer → Audio range for carrier freq generation  
- **Carrier Freq sent value** (0-9999) → Direct TCode output range for C0
- **Pulse Width sent value** (0-9999) → Direct TCode output range for P1
- **Rise Time sent value** (0-9999) → Direct TCode output range for P3

### Preset System
5-slot custom presets storing all settings:
- Left-click: Load preset
- Right-click: Save preset
- Blue button = has saved preset
- Green border = currently active

### Metric Auto-Ranging System

The metric auto-ranging system uses feedback-driven parameter adjustment. Each metric monitors a specific audio characteristic and adjusts its parameter to optimize beat detection.

**Enabled Metrics (toggled via Auto-Adjust checkboxes in GUI):**

| Metric | Parameter Adjusted | Feedback Method |
|--------|-------------------|-----------------|
| Energy Margin | peak_floor | `compute_energy_margin_feedback()` — tracks valley history, adjusts floor relative to average valley |
| Audio Amp | audio_amp | `compute_audio_amp_feedback()` — scales gain to keep energy in optimal detection range |
| Flux Balance | flux_mult | `compute_flux_balance_feedback()` — balances spectral flux scaling |
| Target BPS | peak_floor (indirect) | `compute_bps_feedback()` — raises/lowers floor to match target beats-per-second |

**Note:** The UI displays BPM (beats per minute) but the audio engine works in BPS internally. Conversion happens at the GUI layer in `_on_target_bpm_change()` (÷60) and `_on_bpm_tolerance_change()` (÷60). Widget names: `target_bpm_spin` (range 30–240), `bpm_tolerance_spin` (range 3–60), `bpm_actual_label`.

**Settling System:**
- Each metric tracks consecutive "in-zone" checks
- After 12 consecutive in-zone checks (~30s), metric transitions to SETTLED state
- Out-of-zone: decrement counter by 3 (not hard reset) for faster recovery
- Hysteresis: 2 consecutive out-of-zone checks before triggering adjustment

**Traffic Light Widget** (located in the Controls panel, right-side vertical stack above BPM display):
- 🔴 Red = Metrics actively adjusting
- 🟡 Yellow = Mixed state (some settled, some adjusting)
- 🟢 Green = All enabled metrics settled

**Global Toggle:** Master "Enable Auto-Adjust" checkbox controls all 4 individual metric checkboxes via `_on_metrics_global_toggle()`.

**Parameter Definitions:**
- **flux_mult (Flux Multiplier)**: Controls overall beat sensitivity by scaling spectral flux threshold. Lower = less sensitive (fewer false beats).
- **peak_decay**: Exponential decay rate applied to spectrum peaks frame-to-frame. Lower = faster decay = easier differentiation between troughs and peaks.
- **rise_sens (Rise Sensitivity)**: Size of the rise (amplitude distance between peak and valley) considered significant for beat detection. Lower = more sensitive.
- **peak_floor**: Valley height threshold in spectrum. Audio energy below this floor is ignored. Higher = only strong beats detected; lower = catches quieter beats.

**Step Spinbox Precision:**
- 4 decimal places (enables 0.0001 step size)
- Width: 85px (accommodates 4 decimals)
- Allows fine tuning of small parameters without rounding errors

**P0/C0 Sliding Window Averaging (250ms Rolling Window):**
Smooth frequency display by accumulating time-weighted samples:
```python
self._p0_freq_window: deque = deque()  # (timestamp, norm_weighted) tuples
self._c0_freq_window: deque = deque()
self._freq_window_ms: float = 250.0    # milliseconds
```
Removes samples older than 250ms, averages remaining. Reduces jitter while keeping real-time responsiveness.

**Butterworth Filter Re-initialization:**
When `_on_freq_band_change()` is called (by manual slider), the Butterworth
bandpass filter is re-initialized via `audio_engine._init_butterworth_filter()` so the
actual beat detection filter matches the displayed band.

---

## Restim TCode Protocol Reference

**Source:** Analyzed from [diglet48/restim](https://github.com/diglet48/restim) codebase (Feb 2026).

### TCode Command Format (restim `net/tcode.py`)

```
{axis_id}{value:04d}         ← instant move
{axis_id}{value:04d}I{ms}   ← interpolated move over ms milliseconds
```
- `axis_id` = 2-char string (e.g. `L0`, `V0`, `P0`, `C0`)
- `value` = 0000-9999 (internally normalized to 0.0-1.0 via `float(value) / 10^len(value)`)
- `I{ms}` = interpolation duration in milliseconds (restim converts to seconds: `interval / 1000.0`)
- Commands separated by whitespace or newline
- Restim routing: `route.axis.add(route.remap(cmd.value), cmd.interval / 1000.0)`

### Restim Default Axis Mapping (from `funscript_kit.py`)

| TCode Axis | Restim Internal Axis | Default Range | Description |
|------------|---------------------|---------------|-------------|
| **L0** | POSITION_ALPHA | -1 to 1 | Position alpha (our alpha, rotated) |
| **L1** | POSITION_BETA | -1 to 1 | Position beta (our beta, rotated) |
| L2 | POSITION_GAMMA | -1 to 1 | 3rd axis (4-phase only, not used by us) |
| **V0** | VOLUME_API | 0 to 1 | Volume — one of 4 volume multipliers |
| **C0** | CARRIER_FREQUENCY | 500 to 1000 Hz | Carrier frequency |
| **P0** | PULSE_FREQUENCY | 0 to 100 Hz | Pulse frequency |
| P1 | PULSE_WIDTH | 4 to 10 cycles | Pulse width in carrier cycles |
| P2 | PULSE_INTERVAL_RANDOM | 0 to 1 | Randomize inter-pulse interval |
| P3 | PULSE_RISE_TIME | 2 to 20 cycles | Pulse rise time in carrier cycles |

**bREadbeats currently sends:** L0, L1, V0, P0, C0, P1, P3 (all correct).

### Range Mismatch Notes ⚠️

Our output ranges are wider than restim's defaults — values beyond restim's `limit_max` are clipped:

| Axis | Our Output Range | Restim Default Range | Clipping Risk |
|------|-----------------|---------------------|---------------|
| **P0** | 0-150 Hz (Hz×67 → 0-9999) | 0-100 Hz | Values >100 Hz clipped unless user widens in restim preferences |
| **C0** | 500-1500 Hz ((display-500)×10 → 0-9999) | 500-1000 Hz | Values >1000 Hz clipped unless user widens in restim preferences |

Users can change these limits in restim's Settings → Funscript tab → T-Code axis configuration.

### No Start/Stop TCode Commands Exist

Restim has **no TCode-level start/stop or session commands**. Behavior is implicit:
- **Start:** First TCode command arriving = active
- **Stop:** Send V0 to 0, or stop sending commands (restim's inactivity timer reduces volume)
- **Duration:** Built into every command via `I{ms}` suffix — no separate duration signal needed

The `RequestSignalStart`/`RequestSignalStop` messages in restim's codebase are **FOCStim hardware protobuf commands** (serial/TCP to microcontroller), NOT TCode. They are irrelevant to our TCP TCode connection.

### Restim's Volume Stack

Restim multiplies 4 volume sources together — our V0 sets only the `api` volume:

```
final_volume = api (V0) × master (restim slider) × inactivity × external
```

- **api** — set by our V0 TCode commands
- **master** — user's restim volume slider (independent)
- **inactivity** — auto-reduces if position unchanged for N seconds (configurable)
- **external** — from Buttplug/other sources
- **Volume ramp** — slow start to prevent sudden power on restim's end

Even if we send V0=9999, restim's master/ramp/inactivity still apply.

### Axes We Don't Send Yet (Potential Future Features)

| Axis | What It Does | Integration Idea |
|------|-------------|-----------------|
| **P2** (Pulse Interval Random) | Randomize inter-pulse interval (0-1). "Tingly sensation, less painful on fast moves, slows numbing" | Map to flux variance or constant low value |

### Restim's Built-in FunscriptExpander

Restim's `serialproxy.py` has a `FunscriptExpander` that converts 1D L0 commands into 2D alpha/beta semicircular arcs automatically. This is conceptually similar to our stroke modes but much simpler (fixed semicircle). Our 4-mode stroke system is far more sophisticated — no action needed, but good to know restim has this concept for Buttplug/serial inputs.

### NeoStim/NeoDK Hardware Limits (If Targeting)

| Parameter | Min | Max |
|-----------|-----|-----|
| Pulse frequency | 1 Hz | 100 Hz |
| Carrier frequency | 500 Hz | 3000 Hz |
| Duty cycle | 0% | 90% |

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

1. **Forgetting audio gain in Butterworth path** ⚠️ MOST CRITICAL
   - Location: audio_engine.py ~line 360
   - MUST have: `band_energy = band_energy * self.config.audio.gain`
   - Symptom: Zero beats detected even when audio is present
   - Fix: Add gain application in Butterworth section, NOT in FFT fallback

2. **Setting peak_floor reset/range too high**
   - Reset: MUST stay at 0.08 (NOT 0.2!)
   - Range: Keep 0.01-0.14 (NOT 0.0-0.8)
   - Symptom: Beats don't detect initially because floor > band_energy
   - Band_energy typical: 0.08-0.15 with default gain

3. **Sending cached/queued data** - Always compute live; never pre-compute

4. **Breaking arc resolution** - Don't change `/ 10` or `/ 20` divisors

5. **Adding silence volume factor back** - Volume is only slider + fade

6. **Changing jitter to random** - Keep sinusoidal micro-circles

7. **Missing wiring for new sliders** - Always wire both directions (visualizer↔config↔slider)

8. **Modifying stroke mode math** - Check against commit `097fea9` first

9. **Breaking creep radius** - Must be `0.98 - jitter_r` to prevent clipping

10. **Disabling downbeat pattern matching**
    - Keep `downbeat_pattern_enabled = True` in config
    - Disabling breaks entire tempo lock system
    - Pattern matching requires tolerance (default 100ms)

11. **Changing tempo lock boost factor**
    - Currently 1.25 (25% stronger on downbeats)
    - Lower values (~1.1) won't feel emphatic
    - Code in stroke_mapper.py ~line 229

12. **Removing P0/C0 sliding window**
    - Window smoothing prevents jittery Hz display
    - Removing causes values to jump erratically
    - 250ms window balances smoothness + responsiveness

13. **Modifying step spinbox decimals**
    - Keep at 4 decimals (was 3)
    - 3 decimals loses fine-tuning precision
    - 4 decimals allows 0.0001 step size

---

## Testing Checklist

Before committing changes:
- [ ] All 4 stroke modes produce smooth arcs (not lines)
- [ ] Jitter creates small circles (not random jumps)
- [ ] Creep rotates smoothly (not jerky)
- [ ] Downbeat detection triggers extended strokes
- [ ] P0/C0 display shows correct Hz values
- [ ] Preset load/save works for all settings and checkboxes in all tabs
- [ ] Range slider dragging updates visualizer bands
- [ ] Silence properly fades out and resets tempo
- [ ] No volume factor beyond slider and fade
- [ ] All TCode values are computed live (no queued/cached commands)
- [ ] Frequency-to-depth mapping works (bass = deeper strokes)
- [ ] All 6 range sliders properly wired to config and visualizers
- [ ] All tcode commands are always sent at 4 digit format 0001 - 9999

---

## How to Verify Behavior

1. **Beat Detection** (CRITICAL TO VERIFY FIRST)
   - Play music, watch console for `[Beat] energy=X.XXXX` logs
   - Should see beats within first 2 seconds of audio
   - If ZERO beats: Check audio_engine.py line ~360 - verify gain is applied in Butterworth path
   - Verify band_energy shows 0.08-0.15 range (NOT 0.01-0.05 or >0.2)

2. **Stroke modes**
   - Enable each mode (1-4), play music
   - Watch position dot trace smooth curves (NOT straight lines)
   - Arcs should be continuous and flowing

3. **Downbeat Detection**
   - Watch console for `[Downbeat]` messages
   - Each should include pattern match status and phase error
   - Log shows `LOCKED` when 3 consecutive downbeats match predicted timing

4. **Tempo Lock Boost**
   - Play steady music with consistent beat
   - Once tempo locks, downbeat strokes appear noticeably larger/more emphatic
   - Log shows `tempo=LOCKED+BOOST` in downbeat messages

5. **Jitter**
   - Let audio go quiet, enable jitter
   - Observe smooth micro-circles (NOT random jumps)
   - Circles should be sinusoidal, ~0.01-0.1 amplitude

6. **Creep**
   - Enable creep, verify slow rotation follows tempo
   - Should rotate smoothly, not jerkily

7. **P0/C0 Display**
   - Enable pulse checkbox
   - Verify Hz display is smooth (not jittery)
   - 250ms sliding window should prevent value jumps

8. **Peak Floor Reset**
    - Enable auto-adjust on peak_floor
    - Verify slider resets to 0.08 (NOT 0.2 or higher)
    - If reset wrong, auto-adjust icon shows but beats don't appear

---

## Research & Recommended Future Improvements

### Peak Detection Optimization Research

The following academic and engineering resources provide pathways for improved beat detection and frequency band selection:

#### Source 1: Smoothed Z-Score Algorithm (Brakel, 2014) — HIGH PRIORITY
**Source:** https://stackoverflow.com/a/22640362

**Algorithm Overview:**
```python
# Parameters:
#   lag       = rolling window size (number of past samples)
#   threshold = number of std devs to trigger signal (e.g., 3.5)
#   influence = how much peaks affect rolling stats (0=ignore, 1=full)

# For each new data point:
if abs(new_value - rolling_mean) > threshold * rolling_std:
    signal = +1 (peak) or -1 (valley)
    filtered_value = influence * new_value + (1 - influence) * prev_filtered
else:
    signal = 0
    filtered_value = new_value

rolling_mean = mean(filtered_values[i-lag:i])
rolling_std  = std(filtered_values[i-lag:i])
```

**Recommended Integration Paths:**

**Option A — Band Quality Scoring** (low risk, improves band selection)
1. For each candidate band (scanning in 50Hz steps across 30-22050Hz):
   - Maintain a z-score detector on that band's energy history
   - Count how many z-score signals (peaks) it produces over the last N samples
   - Score = signal_count / total_samples (regularity metric)
2. Band with highest regular signal rate = best for beat detection
3. Parameters: `lag=16` (half our 32-sample history), `threshold=2.5`, `influence=0`
4. **Impact:** Would replace CV-based scoring in `find_consistent_frequency_band()`; drops from 2-3 frames to detect best band to <1 frame

**Option B — Self-Tuning Beat Detection** (higher risk, replaces core peak_floor)
1. Feed `band_energy` values into a single z-score detector each frame
2. When `signal == +1`: that's a beat (energy spike above adaptive threshold)
3. The rolling mean/std automatically adapts to current audio level
4. `peak_floor` becomes unnecessary — z-score threshold replaces it
5. Parameters: `lag=30` (~0.5s at 60fps), `threshold=3.0`, `influence=0.1`
6. **Impact:** Eliminates manual peak_floor tuning; system self-adjusts to any audio level

**Streaming Implementation (available for both options):**
```python
class ZScorePeakDetector:
    """Real-time z-score peak detector for streaming data."""
    def __init__(self, lag=30, threshold=3.0, influence=0.1):
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.buffer = []
        self.filtered = []
        self.mean = 0.0
        self.std = 0.0
        self.initialized = False
    
    def update(self, value: float) -> int:
        """Feed one value, returns +1 (peak), -1 (valley), or 0."""
        self.buffer.append(value)
        if len(self.buffer) < self.lag:
            self.filtered.append(value)
            return 0
        if not self.initialized:
            self.mean = np.mean(self.buffer[:self.lag])
            self.std = np.std(self.buffer[:self.lag])
            self.filtered.append(value)
            self.initialized = True
            return 0
        
        if abs(value - self.mean) > self.threshold * self.std:
            signal = 1 if value > self.mean else -1
            filt = self.influence * value + (1 - self.influence) * self.filtered[-1]
        else:
            signal = 0
            filt = value
        
        self.filtered.append(filt)
        # Update rolling stats from filtered buffer
        window = self.filtered[-self.lag:]
        self.mean = np.mean(window)
        self.std = np.std(window)
        
        # Keep memory bounded
        if len(self.buffer) > self.lag * 3:
            self.buffer = self.buffer[-self.lag * 2:]
            self.filtered = self.filtered[-self.lag * 2:]
        
        return signal
```

#### Source 2: Derivative-Based Peak Detection (UMD) — MEDIUM PRIORITY
**Source:** https://terpconnect.umd.edu/~toh/spectrum/PeakFindingandMeasurement.htm

**Key Concepts:**
- Smooth the first derivative, then find downward zero-crossings
- Two thresholds: **SlopeThreshold** (rejects broad/narrow features) and **AmpThreshold** (rejects small peaks)
- Optimal SlopeThreshold ≈ `0.7 × WidthPoints^-2`

**Integration Strategy:**
- **Auto-calculate `rise_sens`:** Compute optimal rise_sens from typical beat width in samples: `0.7 / width_samples²`
- **Segmented detection:** Use different parameters for different frequency ranges (low frequencies need wider smoothing, high frequencies need narrower)
- **Improve `find_consistent_frequency_band()`:** Use derivative zero-crossings to count peaks per band instead of raw energy variance

#### Source 3: NI Quadratic Fit Peak Detection — LOW PRIORITY (validational)
**Source:** https://www.ni.com/en/support/documentation/supplemental/06/peak-detection-using-labview-and-measurement-studio.html

**Key Takeaways:**
- Fits parabola to groups of `width` points, checks concavity for peak/valley
- Returns fractional index locations (sub-sample accuracy)
- Multi-block processing with retained internal state between calls
- **Key advice:** "Smooth first, then detect with width=3" — validates our Butterworth-filter-first approach
- **Potential:** Interpolation before detection improves accuracy; could interpolate FFT bins for finer frequency resolution

**Note:** Not worth implementing separately — our current Butterworth + threshold approach is already similar in principle.

#### Source 4: MathWorks Trough Detection — VALIDATIONAL ONLY
**Source:** https://www.mathworks.com/matlabcentral/answers/2042461

Validates windowed smoothing + first derivative sign change is standard for real-time trough/peak detection in streaming signals. No novel techniques beyond Sources 1-2.

---

### Implementation Roadmap (Recommended Order)

1. **Phase 1 — Z-Score Band Scoring (Option A)** — ✅ IMPLEMENTED
   - `ZScorePeakDetector` class in audio_engine.py
   - Multi-band z-score detection with 4 sub-bands

2. **Phase 2 — Z-Score Beat Detection (Option B)** — ✅ IMPLEMENTED
   - Dual-path detection: classic threshold (PATH 1) + z-score (PATH 2)
   - Metric auto-ranging replaces manual peak_floor tuning

3. **Phase 3 — Auto-tuned rise_sens** — MEDIUM PRIORITY
   - Use derivative-based formula to compute optimal rise_sens from beat width
   - Could integrate with existing metric auto-ranging system
   - Incremental improvement; lower priority than Phases 1-2

---

## Latest Session Changes (2026-02-11)

### Auto-Align BPM Rounding Bug Fix

**Problem:** Auto-align target BPM spinbox was not updating visually when stable tempo detected.

**Root Cause:** Logic used `round(current_target + 0.3)` to increment target BPM. When `current_target = 120`, `round(120 + 0.3) = 120` (same value), so spinbox change signal didn't fire—PyQt6 optimizes away unchanged value updates.

**Solution:** Changed to direct integer stepping with directional clamping (main.py lines 6278-6287):
```python
# OLD (buggy):
new_target = round(current_target + rate)  # rate = 0.3; rounds to same int
self.target_bpm_spin.setValue(new_target)  # no update if value unchanged

# NEW (working):
if current_bpm > current_target:
    new_target = current_target + 1  # Step up by 1 BPM
elif current_bpm < current_target:
    new_target = current_target - 1  # Step down by 1 BPM
else:
    return  # Already aligned
self.target_bpm_spin.setValue(new_target)  # Always triggers update
```

**Effect:** Spinbox now visibly updates 1 BPM per cycle when tempo is stable, allowing users to see auto-alignment in action.

**Testing:** App launched, auto-align toggle verified working, spinbox updates confirmed.

---

### Start/Stop vs Play/Pause TCode Pipeline Refactor

**Problem:** Unclear separation between button states; Play/Pause toggled `sending_enabled` flag but Start/Stop didn't properly enable the TCode pipeline. Volume ramp response felt disconnected from button press.

**Solution:** Refactored control logic with clear intent separation (main.py lines 5512-5559):

| Event | `sending_enabled` | TCode Sent | Action |
|-------|-------------------|-----------|--------|
| **Start** | → True | V0=0 | Enables TCode TCP pipeline, ready to send commands |
| **Play** | (unchanged) | Ramps V0 linearly 0→set over 1.3s | Starts volume fade-in, no pipeline enable |
| **Pause** | (unchanged) | V0=0 immediately | Zeros volume instantly, keeps pipeline alive |
| **Stop** | → False | V0=0 | Disables TCode pipeline, stops all audio engines |

**Implementation Details:**
- `_on_start_stop()`: Sets `self._sending_enabled = True` on Start; `False` on Stop
- `_on_play_pause()`: Calls `volume_ramp()` for Play (starts fade-in), calls `send_immediate("V0 0")` for Pause
- Volume ramp duration: Increased from 0.8s → **1.3s** for smoother fade-in subjective feel
- No more Play/Pause toggling `sending_enabled`; only Start/Stop manage pipeline state

**Effect:** Clear state machine—pipeline enable/disable separated from volume control. Volume response immediate to Pause button. Start button reliably prepares TCode system.

**Testing:** App launched, all 4 button states tested, volume ramp timing confirmed.

---

### Default Beat Detection Frequency Range: 30-200 Hz → 30-150 Hz

**Change:** Updated `config.py` line 32:
```python
# OLD:
freq_high: float = 200.0

# NEW:
freq_high: float = 150.0
```

**Rationale:** Factory default now focuses on bass frequencies (30-150 Hz) rather than extending into mid-range (200 Hz). Aligns with haptic toy frequency response sweet spot and typical music kick drum range.

**Behavior:**
- **Fresh installs:** Start with 30-150 Hz band (per `config.py` default)
- **Existing users:** Unaffected—use saved `~/.bREadbeats/config.json` from last session
- **EXE builds:** Use `config.py` defaults if user config file absent

**Testing:** Deleted local config file, verified fresh app launch shows 30-150 Hz slider defaults.

---

### Per-Slider Frequency Range Indicator Toggles

**Problem:** Users wanted granular control over frequency range indicator visibility rather than all-or-nothing menu toggle.

**Solution:** Moved range indicator toggles from menu bar to individual per-slider checkboxes:
- Added "Show" checkbox next to Beat Band, Depth Band, P0 Band, and F0 Band range sliders
- Each checkbox independently controls its corresponding frequency band overlay on all 4 visualizers
- Removed "Show (Frequency) Range Selection Indicators" from Options menu
- Individual toggle methods: `_on_toggle_beat_band()`, `_on_toggle_depth_band()`, `_on_toggle_p0_band()`, `_on_toggle_f0_band()`

**Implementation:** Each checkbox calls its respective toggle method which updates all 4 canvas classes (SpectrumCanvas, MountainRangeCanvas, BarGraphCanvas, PhosphorCanvas) to show/hide the specific band overlay and label.

### Advanced Controls Dialog

**Added "Advanced Controls..." menu option** that opens a warning dialog about experimental features:
- Dialog includes warning message: "DON'T BORK YOUR BEATS - These are experimental features that may behave oddly..."
- "Unlock Advanced Controls" checkbox to enable access
- Placeholder for future advanced settings that might be risky for casual users
- Implementation: `_on_advanced_controls()` method opens `QDialog` with warning text and unlock toggle

### Auto-Align Target BPM Feature

**New automatic BPM alignment system** to gradually sync user's target BPM with detected audio tempo:
- **"Auto-align target with sensed BPM"** checkbox in Beat Detection tab
- **Configurable stability threshold** via spinbox (range 2-10, default 5) for "stable reading count"
- When enabled, target BPM slowly aligns with sensed BPM when tempo is stable for N consecutive readings
- Methods: `_on_auto_align_toggle()`, `_on_auto_align_checks_change()`
- Prevents constant BPM chasing by requiring sustained stable tempo before adjustment

**Auto-Alignment Logic:**
```python
# In _update_display(), check if conditions met for auto-alignment
if (self._auto_align_enabled and 
    self._bpm_stable_count >= self._auto_align_checks and 
    abs(current_bpm - target_bpm) > 1.0):
    # Gradually move target toward sensed BPM
    new_target = target_bpm + (current_bpm - target_bpm) * 0.1
    self.target_bpm_spin.setValue(new_target)
```

### Volume Fade Fix - Immediate TCode Commands

**Problem:** Volume fade to zero on pause was queued behind other commands, causing delayed response.

**Solution:** Added `send_immediate()` method to NetworkEngine that bypasses queue and sending_enabled check:
```python
def send_immediate(self, command_str: str) -> bool:
    """Send command immediately, bypassing queue and sending_enabled check.
    Used for critical commands like volume fade on pause."""
```

**Volume Behavior Fix:**
- Modified `_on_play_pause()` to use `send_immediate()` for volume fade commands
- When pausing: immediate V0 fade to zero (not queued)
- When resuming: normal queued volume ramp up
- Ensures responsive volume control regardless of queue state

### NetworkEngine send_immediate() Method

**Purpose:** Bypass normal command queue for time-critical operations like shutdown/volume fade.

**Implementation:**
- Checks connection status but ignores `self._sending_enabled` flag
- Sends directly to TCP socket without queuing
- Used for volume fade on pause and application shutdown commands
- Returns boolean success status

---

## Latest Session Changes (2026-02-09 Continued)

### Critical Fix: Beat Detector Refractory Period
**Problem:** Beat detector was firing 4-6 detections per musical beat within 250ms (old 50ms cooldown), causing:
- 77% stroke loss rate (strokes skipped due to min_interval guard)
- Inflated BPS metrics (reports 2.4 when real rate is 0.08)
- Metrics receiving garbage data, unable to tune correctly

**Solution Implemented:**
```python
# In audio_engine._detect_beat(), line ~620
refractory_s = self.config.stroke.min_interval_ms / 1000.0  # e.g. 300ms → 0.3s
if current_time - self._last_beat_time < refractory_s:
    return False
```
Changed from hardcoded `min_beat_interval = 0.05s` (max 20 BPS) to dynamic `min_interval_ms` (default 300ms = max 3.3 BPS).

**Impact:** Beat clusters eliminated at the source. Strokes now fire consistently at musical intervals (~600ms for 100 BPM). No more rapid cascades. Real BPS metrics now match observed stroke rate.

### Metric Settling System & Traffic Light
**Traffic Light Widget (re-added)**
- Red = Metrics actively adjusting
- Yellow = Mixed state (some settled, some adjusting)
- Green = All enabled metrics settled (stable for N consecutive checks)

**Per-Metric Settled Detection**
- Each metric tracks consecutive "in-zone" checks
- After 12 consecutive in-zone checks (~30s), metric transitions to SETTLED state
- When metric goes out-of-zone: decrement counter by 3 (not hard reset to 0) → recovers faster
- Hysteresis: require 2 consecutive out-of-zone checks before triggering adjustment
- Check intervals increased: 1100ms → 2500ms for sensitivity/audio_amp, 500ms → 1000ms for flux_balance

**Auto-Range Enable Bug Fixed**
- New `_sync_metric_checkboxes_to_engine()` method syncs checkbox states to engine after engine creation
- Metrics now enable on startup without requiring manual toggle

### Amplitude Proportionality
Two critical clamps added to prevent parameters from dropping too low when gain increases:

1. **Peak Floor Clamp** (in `compute_energy_margin_feedback`, line ~1177)
   ```python
   avg_valley = np.mean(self.valley_history)
   peak_floor = max(avg_valley, self.config.audio.audio_amp * 0.10)  # At least 10% of gain
   ```
   Max raised from 0.28 → 1.2 → 2.0 to match expected valleys (~1.9)

2. **Flux Mult Clamp** (in main.py metric callback, line ~3761)
   ```python
   flux_mult = max(value, self.config.audio.audio_amp * 0.15)  # At least 15% of gain
   ```
   Prevents flux_mult from dropping excessively when audio_amp increases

**Purpose:** When audio_amp cranks up to 3.7+, these clamps ensure peak_floor and flux_mult scale proportionally, preventing beat detection from becoming over-sensitive.

### Stability Threshold Relaxed
- Changed from 0.15 → 0.28 in config.py
- Real music with humanistic beat variations naturally has CV ~0.20-0.30
- At 0.15, tempo never reached "stable" state and never locked
- At 0.28, BPM locks properly, enabling sensitivity excess suppression and downbeat boost

### Metric Behavior Improvements
1. **Reversed Prevention** — target_bps lowering of peak_floor suppressed when valley tracking wants it raised (at >80% of avg valley)
2. **Audio Amp De-escalation** — if BPS > 2× target for 2 consecutive checks, lowers audio_amp at half the raise rate
3. **Predicted Beat Matching** — sensitivity excess suppressed if tempo is locked (beats match predictions, so current sensitivity is correct)

---

## GUI Enhancements

### Indicator Visibility Split (Feb 10, 2026)

**Problem:** Single "Show Indicators" menu option toggled both 3-bar peak indicators AND 4x frequency range indicators together, with no independent control.

**Solution:** Split into two separate menu options with different default visibility:
- **"Show Peak Indicators"** — Controls 3 vertical bars (peak_actual_bar, peak_floor_bar, peak_decay_bar) — **Visible on startup** (checked)
- **"Show Range Indicators"** — Controls 4 frequency band overlays (beat_band, depth_band, p0_band, f0_band + labels) — **Hidden on startup** (unchecked)

**Implementation Details:**

*Indicator Method Refactor (All 4 Canvas Classes: SpectrumCanvas, MountainRangeCanvas, BarGraphCanvas, PhosphorCanvas)*
```python
# OLD (line ~450-470):
def set_indicators_visible(self, visible: bool):
    """Show or hide all frequency band indicators and labels"""
    self.beat_band.setVisible(visible)
    # ... range indicators ...
    self.peak_actual_bar.setVisible(visible)  # Peak bars hidden together with ranges
    
# NEW (split into two methods):
def set_peak_indicators_visible(self, visible: bool):
    """Show or hide 3-bar peak indicators"""
    if hasattr(self, 'peak_actual_bar'):
        self.peak_actual_bar.setVisible(visible)
    if hasattr(self, 'peak_floor_bar'):
        self.peak_floor_bar.setVisible(visible)
    if hasattr(self, 'peak_decay_bar'):
        self.peak_decay_bar.setVisible(visible)

def set_range_indicators_visible(self, visible: bool):
    """Show or hide frequency range band indicators and labels"""
    self.beat_band.setVisible(visible)
    self.depth_band.setVisible(visible)
    self.p0_band.setVisible(visible)
    self.f0_band.setVisible(visible)
    # ... label visibility ...
```

*Menu Items (main.py line ~2654)*
```python
self.show_peak_indicators_action = options_menu.addAction("Show Peak Indicators")
self.show_peak_indicators_action.setCheckable(True)
self.show_peak_indicators_action.setChecked(True)  # Default: visible
self.show_peak_indicators_action.triggered.connect(self._on_show_peak_indicators_menu_toggle)

self.show_range_indicators_action = options_menu.addAction("Show Range Indicators")
self.show_range_indicators_action.setCheckable(True)
self.show_range_indicators_action.setChecked(False)  # Default: hidden
self.show_range_indicators_action.triggered.connect(self._on_show_range_indicators_menu_toggle)
```

*Startup Initialization (main.py line ~2117)*
```python
# Changed from: self._on_hide_indicators_toggle(0)
self._on_show_peak_indicators_toggle(True)      # Peak visible
self._on_show_range_indicators_toggle(False)    # Range hidden
```

*Toggle Handlers (main.py line ~3757)*
```python
def _on_show_peak_indicators_toggle(self, checked: bool):
    for canvas in [self.spectrum_canvas, self.mountain_canvas, self.bar_canvas, self.phosphor_canvas]:
        if hasattr(canvas, 'set_peak_indicators_visible'):
            canvas.set_peak_indicators_visible(checked)

def _on_show_range_indicators_toggle(self, checked: bool):
    for canvas in [self.spectrum_canvas, self.mountain_canvas, self.bar_canvas, self.phosphor_canvas]:
        if hasattr(canvas, 'set_range_indicators_visible'):
            canvas.set_range_indicators_visible(checked)
```

**Bug Fixed:** Range indicators were not disappearing when unchecked because SpectrumCanvas (Waterfall visualizer) still had the old combined `set_indicators_visible()` method. Applied split method refactor to all 4 canvas classes.

**Verification:** App launches without errors; menu items appear in Options menu; both toggles work correctly on all 4 spectrum visualizers.

---

## Documentation Additions

**New Reference Documents Created (Feb 10, 2026):**

1. **GUI_CONTROLS_REFERENCE.md** — Comprehensive tabulated reference of all 43 GUI controls organized by tab, including ranges, defaults, tooltips, and effects on stroke generation
2. **GUI_CONTROL_CATALOG.md** — Detailed widget inventory organized by panel/section with line numbers, types, default values, and signal connections

These documents capture the full control landscape for documentation and future maintenance. Not auto-updated; manually maintain sync with main.py.

---

## Spectrogram & Frequency Slider Overhaul (2026-02-11)

### Log-Frequency Spectrogram Display

**Problem:** All 4 spectrum visualizers used linear frequency axis (`np.linspace` resampling), which compressed bass frequencies into a few pixels while spreading out inaudible high frequencies. Dynamic range normalization clipped at [-4, 0] dB (via `(spectrum + 4) / 4`), discarding the bottom third of useful signal.

**Changes Applied to All 4 Canvas Classes** (SpectrumCanvas, MountainRangeCanvas, BarGraphCanvas, PhosphorCanvas):

#### 1. Dynamic Range Fix
```python
# OLD: clipped to [-4, 0] log10 range — lost bottom 33%
spectrum = np.clip((spectrum + 4) / 4, 0, 1)

# NEW: full [-6, 0] log10 range preserved
spectrum = np.clip((spectrum + 6) / 6, 0, 1)
```

#### 2. Log-Frequency Resampling
```python
# OLD: linear spacing — bass compressed, treble stretched
indices = np.linspace(0, len(spectrum) - 1, self.num_bins).astype(int)
resampled = spectrum[indices]

# NEW: logarithmic spacing — perceptually correct
n = len(spectrum)
log_indices = np.logspace(0, np.log10(n - 1), self.num_bins).astype(int)
log_indices = np.clip(log_indices, 0, n - 1)
resampled = spectrum[log_indices]
```

#### 3. Log Hz↔Bin Conversion (all 4 classes)
Frequency band overlays (beat band, depth band, P0 band, C0 band) must map Hz positions correctly onto the log-frequency display:

```python
def _hz_to_bin(self, hz: float) -> int:
    """Convert Hz to display bin index in log-frequency space."""
    if not hasattr(self, '_sample_rate') or self._sample_rate <= 0:
        return 0
    nyquist = self._sample_rate / 2.0
    n = getattr(self, '_fft_bins', self.num_bins)  # FFT bins from spectrum
    linear_idx = hz / nyquist * (n - 1)
    linear_idx = max(1, min(linear_idx, n - 1))
    log_bin = np.log10(linear_idx) / np.log10(n - 1) * self.num_bins
    return int(np.clip(log_bin, 0, self.num_bins - 1))

def _bin_to_hz(self, bin_idx: int) -> float:
    """Convert display bin index back to Hz in log-frequency space."""
    if not hasattr(self, '_sample_rate') or self._sample_rate <= 0:
        return 0.0
    nyquist = self._sample_rate / 2.0
    n = getattr(self, '_fft_bins', self.num_bins)
    linear_idx = 10 ** ((bin_idx / self.num_bins) * np.log10(n - 1))
    return linear_idx / (n - 1) * nyquist
```

#### 4. `_fft_bins` Tracking
Each canvas `__init__` initializes `self._fft_bins = 513` (default for 1024-sample FFT with `np.fft.rfft`). Updated every frame in `update_spectrum()`:
```python
self._fft_bins = len(spectrum)  # before resampling
```
This ensures `_hz_to_bin()` maps correctly even before first audio data arrives.

#### 5. `set_frequency_band()` Updated
All 4 classes now route through `_hz_to_bin()` instead of linear multiplication:
```python
# OLD: linear position
low_bin = int(low_norm * self.num_bins)

# NEW: Hz → log bin
hz_low = low_norm * nyquist
low_bin = self._hz_to_bin(hz_low)
```

#### 6. `set_sample_rate()` Wiring Fix
Added missing `set_sample_rate()` calls for `bar_canvas` and `phosphor_canvas` in `BREadbeatsWindow._on_sample_rate_changed()` (line ~3404). Previously only `spectrum_canvas` and `mountain_canvas` received sample rate updates, causing incorrect Hz↔bin mapping on the other two visualizers.

### Log-Scale Range Sliders

**Problem:** After switching to log-frequency display, the linear Hz range sliders no longer matched the visual. Moving the left handle barely changed position on the spectrogram (compressing low-Hz movement), while the right handle moved too fast (stretching high-Hz movement).

**Solution:** Added `log_scale` parameter to `RangeSlider` and `RangeSliderWithLabel`:

```python
class RangeSlider(QWidget):
    def __init__(self, ..., log_scale: bool = False):
        self._log_scale = log_scale
    
    def _val_to_pos(self, val):
        if self._log_scale and self._max_val > self._min_val and self._min_val > 0:
            log_min = np.log10(self._min_val)
            log_max = np.log10(self._max_val)
            log_val = np.log10(max(val, self._min_val))
            frac = (log_val - log_min) / (log_max - log_min)
            return int(self._margin + frac * usable)
        # ... linear fallback

    def _pos_to_val(self, pos):
        if self._log_scale and self._max_val > self._min_val and self._min_val > 0:
            log_min = np.log10(self._min_val)
            log_max = np.log10(self._max_val)
            log_val = log_min + frac * (log_max - log_min)
            return 10 ** log_val
        # ... linear fallback
```

**Enabled on all 6 frequency Hz sliders** (all have `log_scale=True`):
- `freq_range_slider` — Beat detection frequency band
- `pulse_freq_range_slider` — Pulse Freq monitor band
- `f0_freq_range_slider` — Carrier Freq monitor band
- `p1_monitor_range_slider` — Pulse Width monitor band
- `p3_monitor_range_slider` — Rise Time monitor band
- `depth_freq_range_slider` — Stroke depth frequency band

**NOT enabled** on TCode "Sent Value" sliders (linear 0-9999 range, not frequency-based).

### Updated Testing Checklist Items
- [ ] Spectrogram shows log-frequency distribution (bass frequencies get more visual space)
- [ ] Frequency band overlays align correctly with spectral content in log space
- [ ] Range slider handle movement matches visual position on spectrogram
- [ ] All 4 visualizers display consistent frequency mapping
- [ ] Band positions correct at startup before first audio frame (_fft_bins = 513 default)

---

*Document created: 2026-02-07*  
*Last updated: 2026-02-11 (log-frequency spectrogram, dynamic range fix, log-scale range sliders)*  
*All implementations verified with running program - beat detection working, steady stroke generation, no burst clusters, BPS metrics accurate, metrics reaching settled state, traffic light reaching green or yellow, indicator toggles functioning correctly, spectrogram showing log-frequency distribution with correct band overlay positions.*
*Current branch: main*
*Repository: https://github.com/breadfan69-2/bREadbeats*
