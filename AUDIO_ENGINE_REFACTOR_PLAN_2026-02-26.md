# AudioEngine Decomposition — Detailed Audit & Execution Plan

**Date:** 2026-02-26 (updated 2026-02-27)  
**Target file:** `audio_engine.py` (2,851 lines, 53 methods, 217 `self.*` attributes)  
**Existing sub-package:** `audio_modules/` (10 modules, ~1,498 lines already extracted)

---

### Update Log (2026-02-27)

**What changed since the plan was written:**

1. **File grew slightly** — 2,828 → 2,851 lines, 48 → 53 methods, ~205 → ~217 attrs. Callback is now 501 lines (was 485). Growth is from:
   - `VolumeNormalizer` integration (gain compensation applied in callback before RMS).
   - New methods: `_reference_bpm_for_onset_filters`, `_effective_phase_accept_window_s`, `_is_raw_onset_acceptable` (onset-phase gating), `_compute_teaching_features` (teaching capture features).
   - `_build_shadow_feature_frame` expanded.

2. **Two new `audio_modules/` modules already extracted:**
   - `volume_normalizer.py` (182 lines) — polls Windows master volume via pycaw, exposes `get_compensation_gain()`. Already wired into `AudioEngine.__init__` and `_audio_callback_pyaudio`.
   - `adaptive_lead.py` (111 lines) — EMA-based phase-error observer for learned beat lead offset.

3. **Silence gate thresholds lowered** — `beat_intelligence.py` now uses -70/-65 dBFS (was -55/-48) so the gate exits reliably at low Windows volumes. `beat_intelligence.py` also gained `_silence_fade_in_rate` (0.008/frame) for gradual music-return fade. These changes don't affect the refactor plan since they're in `beat_intelligence.py`, not `audio_engine.py`.

4. **Existing module line counts shifted** — most shrank slightly due to cleanups:
   - `contracts.py`: 105 → 89 (added `BusState`/`BusDecision`/`EngineDecision`, removed some fields)
   - `feature_extractors.py`: 192 → 153
   - `tempo_tracker.py`: 302 → 250
   - `event_detector.py`: 349 → 301
   - Other modules: minor shrinkage.

5. **API surface violations still unfixed** — all Phase 0 items remain needed:
   - `main.py` still pokes `_spectrum_skip_frames`, `_aggressive_tempo_snap_enabled`, `_init_butterworth_filter`
   - `tcode_wiring.py` still reads `_band_energies` directly (both `sub_bass` and `mid`)
   - `beat_intelligence.py` still writes `silence_gate_active` directly
   - `beat_intelligence.py` now imports `RMS_DB_FLOOR`, `rms_to_dbfs`, `silence_threshold_to_dbfs` from `audio_engine` (Phase 0f target)

**Plan impact:** The refactor phases are **still valid and in the right order**. Phase 7 (Audio I/O) is reduced slightly since `VolumeNormalizer` is already extracted. New methods (`_reference_bpm_for_onset_filters`, `_is_raw_onset_acceptable`, `_compute_teaching_features`) should go into:
- `_reference_bpm_for_onset_filters`, `_effective_phase_accept_window_s`, `_is_raw_onset_acceptable` → Phase 5 (Metronome), since they are onset-phase gating helpers used by the metronome pipeline.
- `_compute_teaching_features` → stays in AudioEngine or a future `teaching.py` module (not in scope of this plan).

---

## 1  Current State Summary

### 1.1  The `AudioEngine` class today

| Stat | Value |
|------|-------|
| Total lines | 2,851 (was 2,828) |
| Methods | 53 (on `AudioEngine`) + 3 on `ZScorePeakDetector` + 1 on `BeatEvent` |
| Instance attributes (`self.*`) | ~217 distinct names (was ~205) |
| Longest method | `_audio_callback_pyaudio` — **501 lines** (was 485) |
| `__init__` | **~300 lines** of attribute initialization |
| Deepest nesting | 34 spaces (8+ indent levels) |

### 1.2  What AudioEngine does (responsibility map)

The single class conflates **six distinct concerns**:

| Concern | Key methods | Lines | Shared state (self.*) |
|---------|-------------|-------|-----------------------|
| **A. Audio I/O** | `start`, `stop`, `_start_loopback_capture`, `_start_input_capture`, `_init_butterworth_filter` | ~150 | `pyaudio`, `stream`, `running`, `config.audio.*`, `_butter_sos/zi` |
| **B. Signal processing (FFT, multi-band)** | FFT loop inside `_audio_callback_pyaudio`, `_filter_frequency_band`, `_compute_spectral_flux`, `_update_multiband_zscore`, `_build_shadow_feature_frame` | ~280 | `prev_spectrum`, `fft_size`, `hop_size`, `_hanning_window`, `_fft_input_buffer`, `_band_energies`, `_zscore_detectors`, `_band_fire_history`, etc. |
| **C. Beat detection** | `_detect_beat`, plus valley tracking, z-score path, classic path | ~135 | `energy_history`, `flux_history`, `peak_envelope`, `_valley_history`, `_primary_beat_band`, `_last_beat_time` |
| **D. Tempo / metronome** | `_estimate_tempo_acf`, `_advance_metronome`, `_nudge_metronome_phase`, `_reset_acf_metronome`, `_update_tempo_tracking`, `_predict_next_beat`, `_validate_downbeat_against_pattern`, `_reset_downbeat_pattern`, `_estimate_onset_bpm`, `_compute_tempo_lock_state`, `_sync_tempo_tracker_state`, `get_tempo_info`, `_reference_bpm_for_onset_filters`, `_effective_phase_accept_window_s`, `_is_raw_onset_acceptable` | ~750 | `_onset_buffer`, `_acf_*`, `_metronome_*`, `beat_intervals`, `beat_times`, `smoothed_tempo`, `phase_error_ms`, `consecutive_matching_downbeats`, `measure_energy_accum`, `_last_accepted_raw_onset_time`, etc. |
| **E. Auto-ranging / metrics** | `enable_metric_autoranging`, `compute_energy_margin_feedback`, `compute_audio_amp_feedback`, `set_metric_response_speed`, `get_metric_states`, `_effective_metric_*`, `_scaled_metric_*` | ~220 | `_metric_*`, `_valley_history`, `_energy_margin_history`, `_audio_amp_*` |
| **F. Session stats / telemetry** | `_reset_session_stats`, `_update_session_stats`, `_compute_persistence_stats`, `_session_summary_payload`, `_log_shutdown_summary`, `_record_shadow_telemetry` | ~190 | `_session_*`, `_shadow_telemetry` |

The remaining ~300 lines are `__init__` setup shared across all six concerns, plus syncopation detection logic, teaching features (`_compute_teaching_features`), volume normalization delegation, and the main 501-line callback that wires everything together.

### 1.3  Already-extracted modules in `audio_modules/`

| Module | Lines | What it does |
|--------|-------|-------------|
| `contracts.py` | 89 | Shared dataclasses: `FeatureFrame`, `TempoState`, `TriggerDecision`, `FrontendFrame`, `BusState`, `BusDecision`, `EngineDecision` |
| `feature_extractors.py` | 153 | Pure functions: `compute_bass_dominance`, `positive_spectral_flux`, `rolling_percentile_norm`, `slice_spectrum_band`, `compute_multiband_energies`, etc. |
| `tempo_tracker.py` | 250 | `TempoTracker` class + free functions (`build_acf_octave_candidates`, `select_acf_octave_candidate`, `dedup_window_seconds`, `metronome_phase_error_s`, etc.) |
| `event_detector.py` | 301 | `EventDetector` class — the new-trigger fusion pipeline |
| `telemetry_tuning.py` | 121 | `TelemetryTuning`, `TriggerTelemetry` — shadow telemetry bookkeeping |
| `audioflux_adapter.py` | 105 | `AudioFluxAdapter` — optional audioflux sidecar features |
| `signal_frontend.py` | 82 | `SignalFrontend` — FFT + band extraction (appears **unused** by AudioEngine today) |
| `replay_harness.py` | 104 | Offline replay tooling |
| `volume_normalizer.py` | 182 | **NEW** — `VolumeNormalizer` — polls Windows master volume via pycaw and exposes a compensation gain so all processing sees "100%-equivalent" signal levels. Already wired into `AudioEngine.__init__` and `_audio_callback_pyaudio`. |
| `adaptive_lead.py` | 111 | **NEW** — `AdaptiveLead` — EMA-based observer of phase error to nudge predicted beat times ahead of the audio by a learned lead offset. |

**Key observation:** `signal_frontend.py` already implements the FFT + band extraction pipeline that is *duplicated* inside `_audio_callback_pyaudio`. This is the natural landing zone for concern **B**.

**New observation (2026-02-27):** `VolumeNormalizer` is already cleanly extracted and wired into AudioEngine. The callback now applies `vol_gain` before RMS computation, which means silence-gate dBFS thresholds work reliably regardless of Windows volume. This is a win — one less thing to extract in Phase 7 (Audio I/O).

### 1.4  External consumers (public API surface)

These are the attributes/methods that code outside `audio_engine.py` touches:

| Consumer | Attributes / methods used |
|----------|--------------------------|
| `main.py` | `start`, `stop`, `smoothed_tempo`, `stable_tempo`, `beat_intervals`, `beat_times`, `beat_position_in_measure`, `beats_per_measure`, `measure_energy_accum`, `measure_beat_counts`, `tempo_tracking_enabled`, `set_zscore_threshold`, `enable_metric_autoranging`, `set_metric_response_speed`, `_init_butterworth_filter`, `_spectrum_skip_frames`, `_aggressive_tempo_snap_enabled` |
| `event_handlers.py` | `get_spectrum`, `get_waveform`, `get_tempo_info`, `get_metric_states`, `compute_energy_margin_feedback`, `compute_audio_amp_feedback` |
| `beat_intelligence.py` | `silence_gate_active` (write), `get_spectrum` |
| `tcode_wiring.py` | `_band_energies` (reads `sub_bass` and `mid` keys directly) |

**Problem areas in API surface (still present as of 2026-02-27):**
- `main.py` reaches into private attrs (`_aggressive_tempo_snap_enabled`, `_spectrum_skip_frames`, `_init_butterworth_filter`) — these need proper accessors or config delegation. **Still unfixed.**
- `tcode_wiring.py` reads `_band_energies` directly (both `sub_bass` and `mid`) — needs a public accessor. **Still unfixed.**
- `beat_intelligence.py` writes `silence_gate_active` as a cross-module control signal — should become a method call. **Still unfixed.**

**New cross-module dependency (2026-02-27):**
- `beat_intelligence.py` now imports `RMS_DB_FLOOR`, `rms_to_dbfs`, `silence_threshold_to_dbfs` from `audio_engine`. These free functions should move to `audio_modules/contracts.py` or a new `audio_modules/utils.py` (per Phase 0f). Plan unchanged here.

---

## 2  Proposed Module Architecture

```
audio_engine.py            (facade: ~300-400 lines)
    AudioEngine            — thin orchestrator, owns audio I/O lifecycle
    BeatEvent              — stays here (or moves to contracts.py)
    ZScorePeakDetector     — stays here (or moves to audio_modules/peak_detector.py)

audio_modules/
    contracts.py           — (existing) FeatureFrame, TempoState, TriggerDecision, etc.
    feature_extractors.py  — (existing) pure spectral functions
    signal_frontend.py     — (existing, will be wired in) FFT pipeline
    tempo_tracker.py       — (existing) ACF, metronome helpers
    volume_normalizer.py   — (existing, already wired) Windows volume compensation
    adaptive_lead.py       — (existing) phase-error EMA → learned beat lead offset

    beat_detector.py       — NEW: classic + z-score beat detection logic
    metronome.py           — NEW: internal metronome phase accumulator + downbeat
    syncopation.py         — NEW: off-beat detection logic
    auto_ranging.py        — NEW: metric feedback (peak_floor, audio_amp)
    session_stats.py       — NEW: session telemetry + shutdown summary
    audio_io.py            — NEW: PyAudio/WASAPI setup, Butterworth init
```

### 2.1  Module-by-module breakdown

#### `audio_modules/audio_io.py` (~150 lines)
**Extracts from:** `start`, `stop`, `_start_loopback_capture`, `_start_input_capture`, `_init_butterworth_filter`

Contains a class `AudioIO` or free functions that:
- Initialize PyAudio + WASAPI loopback or input capture
- Initialize Butterworth filter coefficients
- Return an open stream + actual sample rate/channels
- Clean shutdown of stream + PyAudio

**State moved:** `pyaudio`, `stream`, `_butter_sos`, `_butter_zi`, `_use_butterworth`, `_highpass_hz`

**Interface to AudioEngine:**
```python
class AudioIO:
    def open(config: AudioConfig, callback) -> AudioStreamInfo
    def close()
    def init_butterworth(config) -> tuple[sos, zi] | None
```

#### `audio_modules/beat_detector.py` (~200 lines)
**Extracts from:** `_detect_beat` (135 lines) + valley tracking + z-score combining logic

Contains a class `BeatDetector` that:
- Receives per-frame `(energy, flux, band_zscore_signals, primary_band)` 
- Runs the classic path (peak_floor + sensitivity + rise check)
- Runs the z-score path (primary band + any-band)
- Enforces refractory period
- Tracks valley history for the auto-ranging metric
- Returns a `BeatDetectorResult(is_beat, source, valley_update)`

**State moved:** `energy_history`, `flux_history`, `_last_beat_time` (refractory), `_valley_history`, `_prev_energy_for_valley`, `_energy_was_falling`

**Key benefit:** Becomes independently unit-testable. Can write tests like:
```python
def test_refractory_suppresses_double_trigger():
    bd = BeatDetector(config)
    assert bd.detect(energy=0.5, flux=0.3, ...).is_beat == True
    assert bd.detect(energy=0.6, flux=0.4, ...).is_beat == False  # within refractory
```

#### `audio_modules/metronome.py` (~250 lines)  
**Extracts from:** `_advance_metronome` (179L), `_nudge_metronome_phase` (21L), `_reset_acf_metronome` (18L), `_estimate_tempo_acf` (121L), `_estimate_onset_bpm` (9L), `_predict_next_beat` (14L), `_compute_tempo_lock_state` (29L), `_validate_downbeat_against_pattern` (67L), `_reset_downbeat_pattern` (8L)

Contains a class `Metronome` that:
- Maintains the internal ACF-driven metronome clock
- Owns onset buffer, ACF estimation, BPM smoothing
- Tracks phase, beat/downbeat firing, energy-based downbeat detection
- Validates downbeats against predicted pattern
- Exposes tempo lock state with hysteresis

**State moved:** `_onset_buffer`, `_onset_buffer_max`, `_onset_callback_count`, `_onset_first_time`, `_fps_calibration_times`, `_fps_calibration_window`, `_acf_*` (all 7), `_metronome_*` (all 14), `_tempo_lock_*` (all 6), `beat_intervals`, `beat_times`, `smoothed_tempo`, `last_known_tempo`, `stable_tempo`, `beat_stability`, `beat_position_in_measure`, `beats_per_measure`, `measure_energy_accum`, `measure_beat_counts`, `downbeat_position`, `downbeat_confidence`, `consecutive_matching_downbeats`, `last_predicted_downbeat_time`, `phase_error_ms`, `is_downbeat`, `last_beat_time`, `predicted_next_beat`, `predicted_next_beat_mono`

**This is the largest extraction** (~60+ attributes). The key design decision: Metronome owns ALL tempo/downbeat state. AudioEngine delegates to it.

**Interface to AudioEngine:**
```python
class Metronome:
    def feed_onset(flux: float, current_time: float)
    def advance(now: float, band_energy: float) -> MetronomeTick
    def nudge_phase(onset_strength: float)
    def reset()
    def get_tempo_info() -> dict
    @property
    def tempo_lock_state -> bool
    @property
    def bpm -> float
    # etc.
```

#### `audio_modules/syncopation.py` (~80 lines)
**Extracts from:** Syncopation detection block inside `_audio_callback_pyaudio` (~80 lines) + syncopation tracking in `_advance_metronome`

Contains a class `SyncopationDetector` that:
- Tracks off-beat onsets relative to metronome phase
- Implements armed/streak/confirmed state machine
- Handles predictive drop-off logic

**State moved:** `_syncopation_*` (all 6), `_any_band_onset`

#### `audio_modules/auto_ranging.py` (~220 lines)
**Extracts from:** `enable_metric_autoranging`, `compute_energy_margin_feedback` (92L), `compute_audio_amp_feedback` (84L), `set_metric_response_speed`, `get_metric_states`, all `_effective_metric_*` / `_scaled_metric_*` helpers

Contains a class `AutoRanging` that:
- Owns all metric state (settled flags, hysteresis counters)
- Computes peak_floor valley-based feedback
- Computes audio_amp beat-presence feedback
- Reports metric states (ADJUSTING/SETTLED)

**State moved:** `_metric_*` (all 8), `_energy_margin_*` (all 4), `_audio_amp_*` (all 3), `_metric_response_speed`

#### `audio_modules/session_stats.py` (~190 lines)
**Extracts from:** `_reset_session_stats`, `_update_session_stats`, `_compute_persistence_stats`, `_session_summary_payload`, `_log_shutdown_summary`, `_record_shadow_telemetry`

Contains a class `SessionStats` that:
- Tracks frame counts, min/max/sum for rms_db, band_energy, flux
- Computes persistence stats (flux/peak high ratios)
- Generates shutdown summary payload
- Delegates shadow telemetry to existing `TelemetryTuning`

**State moved:** `_session_*` (all 14), `_shadow_telemetry`

---

## 3  The Callback Rewrite Strategy

The 485-line `_audio_callback_pyaudio` is the central challenge. Today it does everything inline. After extraction, it becomes a **~100-line orchestrator**:

```python
def _audio_callback_pyaudio(self, in_data, frame_count, time_info, status):
    if not self.running:
        return (in_data, pyaudio.paContinue)

    # 1. Decode audio → mono (stays inline, ~10 lines)
    mono, beat_mono = self._decode_audio(in_data)

    # 2. FFT pipeline → spectrum, band_energy, flux (delegated)
    frame = self._signal_frontend.process(mono, beat_mono)
    if frame is None:
        return (in_data, pyaudio.paContinue)

    # 3. Multi-band z-score update
    self._multiband.update(frame.spectrum)

    # 4. Silence gate check (~5 lines inline)
    silence_veto = frame.raw_rms_db < -96.0 or self.silence_gate_active

    # 5. Beat detection (delegated)
    raw_beat = self._beat_detector.detect(frame, self._multiband, silence_veto)

    # 6. ACF + Metronome (delegated)
    tick = self._metronome.advance(frame, raw_beat, silence_veto)

    # 7. Syncopation (delegated)
    syncopated = self._syncopation.check(tick, self._multiband, silence_veto)

    # 8. Build BeatEvent + notify callback (~30 lines)
    event = self._build_event(frame, tick, raw_beat, syncopated)
    self._session_stats.update(frame)
    self.beat_callback(event)

    return (in_data, pyaudio.paContinue)
```

---

## 4  Execution Plan — Phase-by-Phase

### Ground rules
1. **One extraction per commit.** Each phase produces a working, test-passing codebase.
2. **Backward-compatible re-exports.** After each extraction, `audio_engine.py` re-exports the moved symbols so external `from audio_engine import X` still works.
3. **Run `python run.py` + `pytest tests/ -q`** after every phase.
4. **No functional changes.** Pure mechanical moves. Bug fixes or improvements are separate commits.
5. **Real-time callback discipline.** Keep callback path allocation-light and lock-minimal; no blocking I/O, no file writes, and no new heavy object construction in `_audio_callback_pyaudio`.
6. **Parity before replacement.** For signal/tempo critical moves, run shadow parity checks against golden captures before switching live code paths.
7. **Golden replay gate each phase.** After every phase, run offline capture replay and compare core counters (beat/downbeat/syncopation counts, tempo lock transitions, median BPM drift) against baseline.

---

### Phase 0: Preparation (Low risk)
**Goal:** Fix API surface violations before moving code.

| Step | Action | Risk |
|------|--------|------|
| 0a | Add `get_band_energies() -> dict` method to expose `_band_energies` publicly. Update `tcode_wiring.py` to use it. | Trivial |
| 0a.1 | Update `beat_intelligence.py` to read band energies through `get_band_energies()` (with fallback compatibility path during migration). | Trivial |
| 0b | Add `set_silence_gate(active: bool)` method. Update `beat_intelligence.py` to call it instead of writing `engine.silence_gate_active` directly. | Trivial |
| 0b.1 | Keep temporary compatibility shim: preserve readable/writable `silence_gate_active` for at least one full phase after introducing `set_silence_gate()`. | Trivial |
| 0c | Replace `main.py` direct access to `_aggressive_tempo_snap_enabled`, `_spectrum_skip_frames`, `_init_butterworth_filter` with proper public methods/properties. | Trivial |
| 0d | Move `BeatEvent` to `audio_modules/contracts.py`. Add re-export in `audio_engine.py`: `from audio_modules.contracts import BeatEvent`. All external imports unchanged. | Safe — import path preserved |
| 0e | Move `ZScorePeakDetector` to `audio_modules/peak_detector.py`. Import it back inside `audio_engine.py`. | Safe |
| 0f | Move `rms_to_dbfs`, `silence_threshold_to_dbfs`, `RMS_DB_FLOOR` to `audio_modules/contracts.py` (or a new `audio_modules/utils.py`). Add re-export from `audio_engine.py`. | Safe — `beat_intelligence.py` already imports via `from audio_engine import ...` |

**Phase 0 exit criteria (added):**
- `main.py`, `tcode_wiring.py`, and `beat_intelligence.py` no longer require private AudioEngine members for routine operation.
- Compatibility aliases (`silence_gate_active`, optional `_band_energies` fallback) still function for one transition phase.

**Commit:** "Phase 0: Clean up AudioEngine public API surface"

---

### Phase 1: Extract Session Stats (Lowest coupling)
**Goal:** Move `_session_*` state + 6 methods → `audio_modules/session_stats.py`

| Step | Action |
|------|--------|
| 1a | Create `audio_modules/session_stats.py` with class `SessionStats`. |
| 1b | Move methods: `_reset_session_stats`, `_update_session_stats`, `_compute_persistence_stats`, `_session_summary_payload`, `_log_shutdown_summary`, `_record_shadow_telemetry`. |
| 1c | In `AudioEngine.__init__`, replace 14 `self._session_*` attrs with `self._session_stats = SessionStats()`. |
| 1d | Update `_audio_callback_pyaudio` to call `self._session_stats.update(...)` and `self._session_stats.record_shadow_telemetry(...)`. |
| 1e | Update `stop()` to call `self._session_stats.log_shutdown_summary()`. |

**Lines moved:** ~190  
**Risk:** Very Low — session stats have no coupling to beat/tempo logic.  
**Commit:** "Extract SessionStats from AudioEngine (audio_modules/session_stats.py)"

---

### Phase 2: Extract Auto-Ranging (Low coupling)
**Goal:** Move all `_metric_*` state + 8 methods → `audio_modules/auto_ranging.py`

| Step | Action |
|------|--------|
| 2a | Create `audio_modules/auto_ranging.py` with class `AutoRanging`. |
| 2b | Move methods: `enable_metric_autoranging`, `compute_energy_margin_feedback`, `compute_audio_amp_feedback`, `set_metric_response_speed`, `get_metric_states`, `_effective_metric_*` (4 helpers), `_scaled_metric_*` (2 helpers). |
| 2c | In `__init__`, replace ~20 metric attrs with `self._auto_ranging = AutoRanging(config)`. |
| 2d | Add delegating methods on AudioEngine for the public API (`compute_energy_margin_feedback`, etc.) that forward to `self._auto_ranging`. |

**Lines moved:** ~220  
**Risk:** Low — auto-ranging reads `last_beat_time`, `beat_times`, `config.beat.peak_floor` (all readable via config/params). One special coupling: `_valley_history` is populated inside `_detect_beat` — the detector will need to push valley data to auto-ranging.  
**Commit:** "Extract AutoRanging from AudioEngine (audio_modules/auto_ranging.py)"

---

### Phase 3: Extract Beat Detector (Medium coupling)
**Goal:** Move `_detect_beat` + state → `audio_modules/beat_detector.py`

| Step | Action |
|------|--------|
| 3a | Create `audio_modules/beat_detector.py` with class `BeatDetector`. |
| 3b | Move `_detect_beat` body. It receives frame data + band signals as parameters (no self.* references to AudioEngine). |
| 3c | Move `energy_history`, `flux_history`, `_last_beat_time`, `peak_envelope`, valley tracking state. |
| 3d | Beat detector returns a result struct containing `is_beat`, `source`, and optionally a valley value for auto-ranging. |
| 3e | Update callback to call `self._beat_detector.detect(...)`. |

**Lines moved:** ~200 (including state init)  
**Risk:** Medium — `_detect_beat` currently calls `_update_tempo_tracking` on beat. After extraction, the callback will call tempo tracking explicitly when BeatDetector reports a beat.  
**Commit:** "Extract BeatDetector from AudioEngine (audio_modules/beat_detector.py)"

---

### Phase 4: Extract Syncopation Detector (Low coupling)
**Goal:** Move syncopation state machine → `audio_modules/syncopation.py`

| Step | Action |
|------|--------|
| 4a | Create `audio_modules/syncopation.py` with class `SyncopationDetector`. |
| 4b | Move the two syncopation blocks from `_audio_callback_pyaudio` (~80 lines). |
| 4c | Move the per-beat syncopation tracking from `_advance_metronome`. |
| 4d | SyncopationDetector receives metronome phase, band signals, config as parameters. |

**Lines moved:** ~80  
**Risk:** Low — syncopation reads metronome phase + band signals, which are passed as arguments.  
**Commit:** "Extract SyncopationDetector from AudioEngine (audio_modules/syncopation.py)"

---

### Phase 5: Extract Metronome (Highest coupling — largest extraction)
**Goal:** Move ACF estimation + metronome + downbeat + tempo tracking → `audio_modules/metronome.py`

This is the most impactful extraction. It touches ~65 instance attributes and ~15 methods.

| Step | Action |
|------|--------|
| 5a | Create `audio_modules/metronome.py` with class `Metronome`. |
| 5b | Move methods in dependency order: utility functions first, then `_estimate_tempo_acf`, `_advance_metronome`, `_update_tempo_tracking`, `_predict_next_beat`, `_validate_downbeat_against_pattern`, `_reference_bpm_for_onset_filters`, `_effective_phase_accept_window_s`, `_is_raw_onset_acceptable`, etc. |
| 5c | Move all `_onset_*`, `_acf_*`, `_metronome_*`, `_tempo_lock_*`, `beat_intervals`, `beat_times`, `smoothed_tempo`, `_last_accepted_raw_onset_time`, etc. into `Metronome.__init__`. |
| 5d | Metronome wraps existing `TempoTracker` internally. |
| 5e | Add delegating properties on `AudioEngine` for backwards-compat: `engine.smoothed_tempo` → `engine._metronome.smoothed_tempo`. |
| 5f | `get_tempo_info()` delegates to `self._metronome.get_tempo_info()`. |

**Lines moved:** ~750 (was ~700; now includes onset-phase gating helpers)  
**Risk:** Medium-High — this is the core beat/tempo pipeline. Requires careful testing.  

**Time note (revised):** budget **4–6 hours** for extraction + stabilization, not including optional polish.

**Safety strategy:**
1. Create the `Metronome` class and copy all methods + state.
2. Have `AudioEngine.__init__` create `self._metronome = Metronome(config)`.
3. Replace each method call one at a time (e.g., first `_estimate_tempo_acf`, test, then `_advance_metronome`, test, etc.).
4. After all delegated, remove old methods from AudioEngine.

**Commit:** "Extract Metronome from AudioEngine (audio_modules/metronome.py)"

---

### Phase 6: Wire up SignalFrontend (Already exists)
**Goal:** Replace the inline FFT loop in `_audio_callback_pyaudio` with `SignalFrontend`

| Step | Action |
|------|--------|
| 6a | Review/update `audio_modules/signal_frontend.py` to match the exact FFT logic in the callback (hanning window, hop_size, spectrum skip, Butterworth). |
| 6a.1 | Add shadow A/B mode: run inline FFT and `SignalFrontend` in parallel for replay sessions; log per-frame deltas for spectrum energy, flux, and band energies before cutover. |
| 6b | Create `self._signal_frontend = SignalFrontend(config)` in `__init__`. |
| 6c | Replace the ~50-line FFT loop in `_audio_callback_pyaudio` with `frame = self._signal_frontend.process(mono, beat_mono)`. |
| 6d | Also replace `get_spectrum()` / `get_waveform()` with reads from SignalFrontend. |

**Phase 6 cutover gate (added):**
- Replay parity within agreed tolerance across golden captures before enabling live replacement path.
- Callback timing remains within baseline budget (no sustained regressions under typical frame rate/load).

**Lines moved:** ~80 from callback + ~30 from helpers  
**Risk:** Medium — the FFT pipeline is performance-critical (runs at ~43 fps in real-time callback). Must verify identical output.  
**Commit:** "Wire SignalFrontend into AudioEngine callback (replaces inline FFT)"

---

### Phase 7: Extract Audio I/O (Optional, polish)
**Goal:** Move PyAudio setup/teardown → `audio_modules/audio_io.py`

**Note (2026-02-27):** `VolumeNormalizer` is already extracted to `audio_modules/volume_normalizer.py` and fully wired. This phase now only needs to move PyAudio/WASAPI stream management and Butterworth filter init — the volume compensation piece is done.

| Step | Action |
|------|--------|
| 7a | Create helper functions in `audio_modules/audio_io.py` for device enumeration, stream opening, Butterworth init. |
| 7b | `AudioEngine.start()` becomes: stream_info = `audio_io.open_stream(config, callback)`. |
| 7c | `AudioEngine.stop()` becomes: `audio_io.close_stream(stream_info)`. |
| 7d | VolumeNormalizer is already extracted — no work needed. |

**Lines moved:** ~100 (was ~150 in original plan; volume normalization already done)  
**Risk:** Low — but touches hardware init, so test on actual audio device.  
**Commit:** "Extract audio I/O helpers (audio_modules/audio_io.py)"

---

## 5  Final Architecture (Post-Refactor)

```
audio_engine.py  (~300-400 lines)
├── AudioEngine class
│   ├── __init__: creates sub-components
│   ├── start() / stop(): delegates to AudioIO
│   ├── _audio_callback_pyaudio: ~100-line orchestrator
│   ├── Delegating properties for backward-compat
│   └── silence_gate_active (cross-module control)
├── BeatEvent re-export (from contracts.py)
└── rms_to_dbfs re-export (from contracts.py)

audio_modules/
├── contracts.py       (existing + BeatEvent, RMS_DB_FLOOR, rms_to_dbfs)
├── feature_extractors.py  (existing, unchanged)
├── signal_frontend.py (existing, wired into callback)
├── tempo_tracker.py   (existing, unchanged)
├── event_detector.py  (existing, unchanged)
├── telemetry_tuning.py (existing, unchanged)
├── audioflux_adapter.py (existing, unchanged)
├── replay_harness.py  (existing, unchanged)
├── volume_normalizer.py (existing, already wired — no changes needed)
├── adaptive_lead.py   (existing — phase-error EMA for beat lead offset)
├── peak_detector.py   (NEW: ZScorePeakDetector)
├── beat_detector.py   (NEW: classic + z-score detection)
├── metronome.py       (NEW: ACF + phase accumulator + downbeat)
├── syncopation.py     (NEW: off-beat detection)
├── auto_ranging.py    (NEW: metric feedback system)
├── session_stats.py   (NEW: session telemetry)
└── audio_io.py        (NEW: PyAudio/WASAPI lifecycle, Butterworth init)
```

**Expected line counts after refactor:**

| File | Est. lines |
|------|-----------|
| `audio_engine.py` | ~300-400 |
| `audio_modules/metronome.py` | ~450 |
| `audio_modules/beat_detector.py` | ~200 |
| `audio_modules/auto_ranging.py` | ~220 |
| `audio_modules/session_stats.py` | ~190 |
| `audio_modules/audio_io.py` | ~100 (reduced: VolumeNormalizer already done) |
| `audio_modules/syncopation.py` | ~80 |
| `audio_modules/peak_detector.py` | ~80 |
| `audio_modules/volume_normalizer.py` | 182 (already exists) |
| `audio_modules/adaptive_lead.py` | 111 (already exists) |

---

## 6  Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Real-time callback performance regression | Phase 6 (SignalFrontend) is isolated. Benchmark FFT frame time before/after. Expect <0.5ms overhead from delegation. |
| Shared state leaks between components | Each extracted class owns its state. AudioEngine passes data via method arguments, not shared mutable refs. |
| Backward-compat breakage (import paths) | Every moved symbol gets a re-export in `audio_engine.py`. External imports unchanged. |
| Backward-compat breakage (private attr migration) | Keep compatibility shims for one phase: preserve `silence_gate_active` and transitional `_band_energies` fallback while callers move to public accessors. |
| Metronome extraction (Phase 5) breaks tempo | Execute sub-method by sub-method with tests after each. Use the existing `test_audio_engine_tempo_lock_hysteresis.py` as regression gate. |
| Test coverage gap | Existing tests are mostly integration-level. Each new module should get a focused unit test file. Minimum: `test_beat_detector.py`, `test_metronome.py`. |
| Silent behavioral drift during mechanical extraction | Add golden-capture replay diff after each phase: compare beat/downbeat/syncopation totals, tempo lock transitions, and BPM drift statistics to baseline. |

---

## 7  What NOT to Do

1. **Don't change algorithms during refactor.** No "while we're here" beat detection improvements. Separate commits.
2. **Don't break the BeatEvent contract.** It's consumed by 8+ files and 9 test files.
3. **Don't make the callback async.** PyAudio callbacks must return synchronously.
4. **Don't over-abstract.** The goal is splitting a god-class, not building a plugin framework.
5. **Don't refactor `_update_tempo_tracking` and `_advance_metronome` simultaneously.** They share downbeat state — extract as a unit into Metronome.

---

## 8  Effort Estimate

| Phase | Est. time | Difficulty |
|-------|-----------|-----------|
| Phase 0 (API cleanup) | 30 min | Easy |
| Phase 1 (Session stats) | 30 min | Easy |
| Phase 2 (Auto-ranging) | 45 min | Easy |
| Phase 3 (Beat detector) | 1 hour | Medium |
| Phase 4 (Syncopation) | 30 min | Easy |
| Phase 5 (Metronome) | 4-6 hours | Hard |
| Phase 6 (SignalFrontend) | 1 hour | Medium |
| Phase 7 (Audio I/O) | 30 min | Easy |
| **Total** | **~9-12 hours** | |

Phases execute in order. Each produces a shippable state. You can stop after any phase and have a better codebase than before.
