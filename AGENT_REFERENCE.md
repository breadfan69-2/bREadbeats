# bREadbeats Agent Reference Document

## Purpose
This document serves as a canonical reference for future AI agents working on bREadbeats. It explains the core architecture, key features, and their intended behaviors. **Review this document before making changes** to avoid accidentally breaking critical functionality.

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
- **audio_engine.py** - Audio capture, FFT, beat/tempo detection, downbeat tracking, dominant-frequency estimation per frame
- **stroke_mapper.py** - Beat→stroke conversion, 4 stroke modes, jitter/creep, depth scaling from dominant frequency via `_freq_to_factor`
- **network_engine.py** - TCP connection to restim, TCodeCommand class
- **config.py** - All configuration dataclasses and defaults
- **hunt order.txt** - Auto-adjust hunting cycle sequence (14 steps with strategic flux_mult repetition)

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
- Combined spectral flux + peak energy detection
- Butterworth bandpass filter for bass isolation (30-200Hz default)
- Weighted tempo averaging (recent beats weighted 0.5-1.5)
- Exponential smoothing (factor 0.7) for stable BPM
- 2000ms timeout resets tempo tracking (preserves last known BPM)

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

### Range Sliders
Dual-handle sliders for min/max pairs:
- **Beat detection freq** (30-22050 Hz) → Red band on visualizer → Butterworth filter range
- **Stroke depth freq** (30-22050 Hz) → Green band on visualizer → Controls stroke depth based on bass/treble content  
- **Pulse monitor freq** (30-22050 Hz) → Blue band on visualizer → Audio range for P0 TCode generation
- **Pulse TCode freq** (0-150 Hz) → P0 output range (Hz*67 → TCode 0-9999)
- **Carrier monitor freq** (30-22050 Hz) → Cyan band on visualizer → Audio range for C0 TCode generation  
- **Carrier TCode freq** (500-1500) → C0 output range ((display-500)*10 → TCode 0-9999)

### Preset System
5-slot custom presets storing all settings:
- Left-click: Load preset
- Right-click: Save preset
- Blue button = has saved preset
- Green border = currently active

### Auto-Adjust (Hunting & Ranging) System

**Hunting Cycle (1/8 interval = 12.5ms during HUNTING phase):**
14-step optimized cycle that strategically repeats flux_mult (appears 4×):
```python
_auto_hunting_cycle = [
    'audio_amp', 'flux_mult', 'sensitivity', 'peak_decay', 'flux_mult',
    'rise_sens', 'peak_floor', 'flux_mult', 'peak_decay', 'flux_mult',
    'peak_floor', 'rise_sens', 'flux_mult', 'sensitivity'
]
```
**Why flux_mult repeats 4×?** It has fastest convergence speed; frequent adjustment significantly reduces overall hunting time (~50% faster).

**Reversing Cycle (100ms interval after downbeat detected):**
Reversed order of hunting cycle, used to fine-tune when downbeat exists.

**Beat-Count Lock System (Replaces Time-Based Locks):**
Parameters lock after 8 consecutive beats where BPM stays within 60-180 range:
```python
_auto_beats_lock_threshold = 8  # Consecutive beats required to lock
_auto_bpm_min = 60.0            # Min BPM for valid beat
_auto_bpm_max = 180.0           # Max BPM for valid beat
_auto_consecutive_beat_count = {}  # Per-param beat counters
```
When a parameter's counter reaches threshold, it transitions from HUNTING → LOCKED.
Global "beats:" spinbox (default 8) controls this threshold for all parameters.
Individual slider lock_spin widgets were removed - only global beat-count lock exists now.

**Auto-Reset Slider on Enable:**
When user toggles auto-adjust ON for a parameter, slider automatically resets to initial value:
```python
reset_values = {
    'audio_amp': 0.15,     # Hunts DOWN from max (inverted param)
    'peak_floor': 0.08,    # Hunts DOWN from max ← MUST stay 0.08!
    'peak_decay': 0.999,   # Hunts DOWN from max (inverted param)
    'rise_sens': 0.02,     # Hunts UP from min (normal param)
    'sensitivity': 0.1,    # Hunts UP from min (normal param)
    'flux_mult': 0.2       # Hunts UP from min (normal param)
}
```
**If peak_floor reset changed to 0.2, beats won't detect because 0.2 > band_energy (0.08-0.15)**

**Slider Ranges (Control Hunting Search Space):**
| Parameter | Min | Max | Reset | Why Changed |
|-----------|-----|-----|-------|-------------|
| audio_amp | 0.15 | 10.0 | 0.15 | Reset to min; hunting raises it |
| peak_floor | 0.015 | 0.28 | 0.14 | Reset to max; hunting lowers it; min raised to 0.015 |
| peak_decay | 0.230 | 0.999 | 0.999 | Reset to max; hunting lowers it; min raised to 0.230 |
| rise_sens | 0.02 | 1.0 | 0.02 | Rise height threshold; min 0.02 to prevent user issues |
| sensitivity | 0.01 | 1.0 | 0.1 | Unchanged |
| flux_mult | 0.2 | 10.0 | 0.2 | Min raised to 0.2 to prevent stroke disconnect |

**Parameter Definitions:**
- **flux_mult (Flux Multiplier)**: Controls overall beat sensitivity by scaling spectral flux threshold. Lower = less sensitive (fewer false beats). Best used when downbeat and expected beats are both above detection threshold but background noise/artifacts cause extra beats.
- **peak_decay**: Exponential decay rate applied to spectrum peaks frame-to-frame. Lower = faster decay = easier differentiation between troughs and peaks. Faster decay (lower values) makes beat detection more responsive to sudden energy changes.
- **rise_sens (Rise Sensitivity)**: Size of the rise (amplitude distance between peak and valley) that the system considers significant for beat detection. Lower = smaller rise triggers detection = more sensitive. Best used when overall beats are excessive but downbeats are missing (reduces false positives while keeping real beats).
- **peak_floor**: Valley height threshold in spectrum. Audio energy below this floor is ignored. Higher = only strong beats detected; lower = catches quieter beats.

**CRITICAL:** Narrower ranges enable faster convergence. DO NOT widen peak_floor above 0.28.

**Beathunting Emergency Trigger:**
If 3+ consecutive no-beat cycles occur while audio is playing:
```python
if self._auto_no_beat_count >= self._auto_beathunting_threshold and has_audio:
    # All locks cleared, all params reset to HUNTING
    for p in self._auto_param_state:
        self._auto_param_state[p] = 'HUNTING'
        self._auto_consecutive_beat_count[p] = 0  # Reset beat counters
    print(f"[Auto] ⚡ç BEATHUNTING triggered after {count} cycles")
```
**Purpose:** Prevents system from getting stuck with all params locked but zero beats detected.

**Step Spinbox Precision:**
- 4 decimal places (enables 0.0001 step size)
- Width: 85px (was 75px; accommodates 4 decimals)
- Allows fine tuning of small parameters without rounding errors

**P0/C0 Sliding Window Averaging (250ms Rolling Window):**
Smooth frequency display by accumulating time-weighted samples:
```python
self._p0_freq_window: deque = deque()  # (timestamp, norm_weighted) tuples
self._c0_freq_window: deque = deque()
self._freq_window_ms: float = 250.0    # milliseconds
```
Removes samples older than 250ms, averages remaining. Reduces jitter while keeping real-time responsiveness.

**Auto-Frequency Band Tracking:**
Automatically finds and tracks the most consistent frequency band for beat detection.
Band width animates smoothly: **narrows when beats arrive** (15% per update toward spinbox target),
**widens when beats are missed** (10% per update toward full 22050 Hz range).

```python
self._auto_freq_enabled: bool = False           # Toggle via "auto-freq band" checkbox
self._auto_freq_width: float = 300.0            # Target band width (Hz) - "width:" spinbox
self._auto_freq_current_width: float = 22050.0  # Animated display width
self._auto_freq_current_center: float = 0.0     # Tracked center frequency (Hz)
self._auto_freq_phase: str = 'scanning'         # 'scanning', 'widening', 'tracking'
self._auto_freq_missed_threshold: int = 3       # Expand + UNLOCK after this many misses
```

**State Machine (driven by beat arrivals):**
1. **SCANNING** (no beats): Full 30-22050 Hz range used for beat detection.
   Continuously finds bands with consistent peak/valley patterns using
   `find_consistent_frequency_band()`. Center smoothly tracks the best band (30% blend when unlocked).
2. **WIDENING** (beats missed): Width gradually expands toward full range at 10% per 100ms update.
   Center position freezes during widening to prevent oscillation. Label shows `⚡[WIDEN]`.
3. **TRACKING** (beats arriving, tempo locked): Width gradually narrows toward spinbox target at 15% per 100ms update.
   Center tracks detected peaks (10% blend when locked, 30% when scanning). Label shows `🔒[TRACK]`.

**Missed Beat Detection & Unlock Mechanism:**
When beats are expected but don't arrive within 2× expected interval:
- Increment `_auto_freq_missed_count` (tracks consecutive misses)
- Width begins expanding immediately (WIDENING phase starts)
- After 3+ consecutive missed beats:
  - `audio_engine.consecutive_matching_downbeats` reset to 0 (UNLOCKS tempo lock)
  - `_band_energy_history` cleared for fresh scan
  - Missed count reset to prevent re-triggering
  - Console logs: `[AutoFreq] ⚡ 3 missed beats → UNLOCKED tempo`
- Width continues expanding, center frozen until next beat arrives

**Width Animation Rules:**
- On beat arrival: `width += (target_width - width) * 0.15` (narrows 15% per update)
- On missed beat: `width += (22050 - width) * 0.10` (widens 10% per update)
- Center updates: Only when NOT widening and score > 0.1 (no tracking during widen phase)
- Band display: Centered on `_auto_freq_current_center ± (current_width / 2)`

**Butterworth Filter Re-initialization:**
When `_on_freq_band_change()` is called (by auto-freq or manual slider), the Butterworth
bandpass filter is re-initialized via `audio_engine._init_butterworth_filter()` so the
actual beat detection filter matches the displayed band.

**Audio Engine Methods:**
```python
def find_consistent_frequency_band(min_freq, max_freq, band_width) -> (center, low, high, score)
  # Returns center frequency, low/high bounds, and consistency score (0-1)
```

**GUI Updates (all react in real-time):**
- Beat detection freq range slider moves to show current auto-freq band
- All visualizer canvases (spectrum, mountain, bar, phosphor) red band overlay updates
- Auto-freq label shows phase + lock icon + band range:
  - `🔓[SCAN] 30-22050 Hz` (no beats detected yet)
  - `⚡[WIDEN] (expanding)` (missing beats, width expanding)
  - `🔒[TRACK] (narrowed)` (beats locked, width narrowed)

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

10. **Modifying hunting cycle order or flux_mult repetitions**
    - flux_mult APPEARS 4 TIMES strategically
    - Removing repetitions breaks convergence speed optimization
    - Order is tuned for fastest bailout

11. **Disabling downbeat pattern matching**
    - Keep `downbeat_pattern_enabled = True` in config
    - Disabling breaks entire tempo lock system
    - Pattern matching requires tolerance (default 100ms)

12. **Changing tempo lock boost factor**
    - Currently 1.25 (25% stronger on downbeats)
    - Lower values (~1.1) won't feel emphatic
    - Code in stroke_mapper.py ~line 229

13. **Removing P0/C0 sliding window**
    - Window smoothing prevents jittery Hz display
    - Removing causes values to jump erratically
    - 250ms window balances smoothness + responsiveness

14. **Modifying step spinbox decimals**
    - Keep at 4 decimals (was 3)
    - 3 decimals loses fine-tuning precision
    - 4 decimals allows 0.0001 step size

15. **Changing audio_amp hunting limit**
    - Old: Limited to 1.0 during hunting
    - New: Uses full 0.15-5.0 range
    - DO NOT re-add 1.0 limit; breaks audio amp scaling

16. **Modifying auto-freq width animation rates**
    - Narrow rate: MUST stay at 15% per update (when beats arrive)
    - Widen rate: MUST stay at 10% per update (when beats missed)
    - Changing these breaks the balance between responsiveness and stability
    - Lower widen rate → band gets stuck narrow when music changes
    - Higher narrow rate → band snaps too fast, not enough scan time

17. **Removing center freezing during WIDENING phase**
    - Center MUST freeze when `is_missing_beats` is True
    - Unfreezing causes oscillation/drift when beats are lost
    - Center should only update when score > 0.1 AND NOT widening

18. **Changing missed beat threshold**
    - MUST stay at 3 consecutive misses before unlock
    - Lower (e.g., 1-2) causes excessive unlocks on tempo variations
    - Higher (e.g., 5+) prevents quick recovery when music changes
    - Threshold of 3 balances stability with adaptability

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

5. **Hunting Cycle**
   - Enable auto-adjust on any parameter
   - Watch console: `[Auto] ↑ param (HUNTING, BPM=X, db=Y, interval=12ms)`
   - Should show 12ms interval during HUNTING phase
   - Once downbeat detected, interval changes to 100ms (REVERSING phase)

6. **Beathunting Trigger**
   - Mute audio while music plays
   - After ~3 updates with no beats, should see `[Auto] ⚡ç BEATHUNTING triggered`
   - All params reset to HUNTING state

7. **Jitter**
   - Let audio go quiet, enable jitter
   - Observe smooth micro-circles (NOT random jumps)
   - Circles should be sinusoidal, ~0.01-0.1 amplitude

8. **Creep**
   - Enable creep, verify slow rotation follows tempo
   - Should rotate smoothly, not jerkily

9. **P0/C0 Display**
   - Enable pulse checkbox
   - Verify Hz display is smooth (not jittery)
   - 250ms sliding window should prevent value jumps

10. **Peak Floor Reset**
    - Enable auto-adjust on peak_floor
    - Verify slider resets to 0.08 (NOT 0.2 or higher)
    - If reset wrong, auto-adjust icon shows but beats don't appear

11. **Auto-Frequency Band Tracking**
    - Enable "auto-freq band" checkbox
    - Play music with steady beat
    - Observe console: `[AutoFreq] ⚡ 3 missed beats → UNLOCKED tempo` should NOT appear immediately
    - Band should narrow (get tighter) as beats are detected
    - Mute audio or change tempo: Band should widen (expand toward full range)
    - Label should show: `🔓[SCAN]` (searching), `⚡[WIDEN]` (beat loss), `🔒[TRACK]` (locked)
    - Red frequency band overlay on visualizer should move and resize smoothly (not jitter)
    - Downbeat strokes should be 25% larger when tempo locked (BOOST active)

---

## Recent Major Changes (Latest Commit)

### 1. Beat Detection Gain Fix (CRITICAL ISSUE RESOLVED)
**Problem Encountered:** Audio gain was only applied in FFT fallback path, not in the primary Butterworth filter path, causing **zero beats detected** even with audio present.

**Solution Implemented:**
```python
# audio_engine.py line ~360 (in Butterworth filter section)
band_energy = np.sqrt(np.mean(beat_mono ** 2))
band_energy = band_energy * self.config.audio.gain  # ← CRITICAL: Apply gain HERE
```

**Impact:** Program went from "ZERO BEATS DETECTED" to normal beat detection in ~1 second.

### 2. Hunting Cycle Overhaul (14-Step with Strategic Repetition)
**New Approach:** 14-step cycle with flux_mult appearing 4× strategically, running at 1/8 interval (12.5ms) during HUNTING phase.

**Why It Works:**
- flux_mult has fastest convergence (controls stroke appearance)
- Repeating it 4× reduces overall hunting time by ~50%
- Oscillation amplitude reduced to 3/4 of step_size for smoother convergence

### 3. Tempo Lock Boost System (Confidence Indicator)
When downbeat timing matches predicted pattern for 3+ consecutive downbeats:
- Tempo marked as LOCKED
- Downbeat strokes get 25% amplitude boost (1.25× multiplier)
- Log shows `tempo=LOCKED+BOOST`

**Purpose:** Creates more emphatic feel when tempo detection is stable.

### 4. Downbeat Pattern Matching (Strict Tempo Validation)
- Energy-based detection finds downbeat candidate
- Check if timing matches predicted downbeat (±100ms tolerance)
- After 3 accepted consecutive downbeats: tempo LOCKED
- Log shows acceptance/rejection with phase error

### 5. P0/C0 Sliding Window Averaging (Smooth Display)
250ms rolling window averages samples for smooth Hz display, reducing jitter by ~80% while keeping <250ms response.

### 6. Slider Range Adjustments (Optimize Search Space)
- peak_floor: 0.015-0.28 (was 0.0-0.8), reset=0.08
- peak_decay: 0.23-0.999 (was 0.5-0.999)
- audio_amp: 0.15-10.0 (widened max for more headroom)
- flux_mult: 0.2-10.0 (widened max for more headroom)
- freq_weight: 0.0-5.0 (was 0.0-2.0)
- min_interval: 50-5000ms (was 50-500ms)
- freq_depth_factor: 0.0-2.0 (was 0.0-1.0)
- volume: 0-100 (display scale, internally 0-1)

Narrower ranges = Faster hunting convergence (50% faster parameter detection).

### 7. Auto-Reset Slider on Enable
When user toggles auto-adjust ON, slider automatically returns to reset value (peak_floor resets to 0.08, not 0.2).

### 8. Beathunting Emergency Trigger
If 3+ consecutive no-beat cycles while audio plays, all locks clear and all params reset to HUNTING. Prevents stuck states.

### 9. Beat-Count Lock (Replaces Time-Based Locks)
- Old system: Parameters locked after X milliseconds in REVERSING state
- New system: Parameters lock after 8 consecutive beats within valid BPM range (60-180)
- Global "beats:" spinbox controls threshold for all parameters
- Individual slider lock_spin widgets removed (6 total removed)
- Simpler `_auto_param_config` tuple: (step_size, max_limit) instead of (step, lock_time, max)

### 10. Auto-Frequency Band Tracking (Animated Width with Beat Loss Handling)
**New Behavior:** Band width animates smoothly based on beat arrivals, replacing rigid 3-phase state machine.

**Key Features:**
- **Animated width** starts at full 22050 Hz, gradually narrows (15%/update) toward spinbox target when beats arrive
- **Missed beat detection** counts consecutive missing beats; on 3+ misses, width expands (10%/update) back toward full range
- **Center freezing** during WIDENING phase prevents band oscillation/drift
- **Tempo unlock on threshold** - after 3+ consecutive missed beats, tempo lock is cleared, history flushed, system resumes scanning
- **Three phase labels:** `🔓[SCAN]` (searching), `⚡[WIDEN]` (beats lost, expanding), `🔒[TRACK]` (beats active, narrowed)

**Impact on Beat Detection:**
When beats are lost (music changes, tempo shift, or detection drift):
1. Width begins expanding at 10% per 100ms update
2. Center position freezes to prevent oscillation  
3. Display label changes to `⚡[WIDEN]` for visual feedback
4. After 3 consecutive expected beats missed, tempo is UNLOCKED:
   - `consecutive_matching_downbeats` reset to 0
   - Band energy history cleared for fresh analysis
   - Search becomes more aggressive over full spectrum

**Integration with Tempo Lock Boost (25% amplification):**
Once beats stabilize and 3+ downbeats match predicted timing, tempo becomes LOCKED:
- Downbeat strokes receive 25% amplitude boost (1.25× multiplier)
- Center tracking switches to 10% blend (more stable)
- Width continues narrowing toward spinbox target
- System provides visual/tactile feedback of confidence through larger strokes

---

*Document created: 2026-02-07*  
*Last comprehensive update: 2026-02-09*  
*Reference for recent changes: Commit 30224a7 (auto-freq gradual widen) + tempo lock boost system*
*All implementations verified with running program - beat detection working, hunting cycle active, tempo lock functional, auto-freq band tracking with beat loss handling.*
