# AudioEngine Refactor Handoff (2026-02-27)

## Current Status Snapshot

Refactor progress is **well underway** and currently at **Phase 5 (partial)**.

### Completed Phases

- **Phase 0 — API cleanup and safety shims**
  - Added public accessors/mutators in `audio_engine.py` (band energies, silence gate, spectrum skip, aggressive snap, Butterworth reinit).
  - Rewired consumers (`main.py`, `beat_intelligence.py`, `tcode_wiring.py`) off private members with compatibility fallback paths.
  - Moved `BeatEvent`, `RMS_DB_FLOOR`, `rms_to_dbfs`, `silence_threshold_to_dbfs` into `audio_modules/contracts.py`.

- **Phase 1 — Session stats extraction**
  - Added `audio_modules/session_stats.py` (`SessionStats`).
  - `AudioEngine` delegates session update/shutdown summary/shadow telemetry.

- **Phase 2 — Auto-ranging extraction**
  - Added `audio_modules/auto_ranging.py` (`AutoRanging`).
  - `AudioEngine` metric public API delegates to this component.

- **Phase 3 — Beat detector extraction**
  - Added `audio_modules/beat_detector.py` (`BeatDetector`).
  - `_detect_beat` now delegates to component with a compatibility fallback path for `__new__` test harnesses.

- **Phase 4 — Syncopation extraction**
  - Added `audio_modules/syncopation.py` (`SyncopationDetector`).
  - Callback and metronome beat-boundary syncopation logic now routed through module.

### Phase 5 Progress (partial)

- Added `audio_modules/metronome.py` with `MetronomeController`.
- `audio_engine.py` now delegates these methods to controller:
  - `_sync_tempo_tracker_state`
  - `_compute_tempo_lock_state`
  - `_reference_bpm_for_onset_filters`
  - `_effective_phase_accept_window_s`
  - `_is_raw_onset_acceptable`
  - `_estimate_tempo_acf`
  - `_estimate_onset_bpm`
  - `_advance_metronome`
  - `_nudge_metronome_phase`
  - `_reset_acf_metronome`
- Added lazy helper `_metronome_ctrl()` to support tests that instantiate `AudioEngine` via `__new__`.

## Remaining Work

## 1) Finish Phase 5 (Metronome extraction)

### Still in `audio_engine.py` and should move/delegate

- `_update_tempo_tracking`
- `_predict_next_beat`
- `_validate_downbeat_against_pattern`
- `_reset_downbeat_pattern`
- `get_tempo_info`

### Key constraint

Do **not** change runtime behavior while moving methods. Preserve:

- lock/unlock hysteresis behavior,
- downbeat confidence and pattern matching behavior,
- tempo timeout/recovery interactions,
- event payload fields consumed by UI/beat intelligence/stroke mapper.

### Suggested approach

1. Copy methods into `MetronomeController` first.
2. Make thin delegating wrappers in `AudioEngine`.
3. Keep compatibility aliases/fields in `AudioEngine` while callers still use direct attributes.
4. Remove duplicate logic from `AudioEngine` only after tests pass.

## 2) Phase 6 (Signal frontend integration)

- Wire `audio_modules/signal_frontend.py` into callback and replace inline FFT/frame construction path.
- Preserve callback timing and visualizer parity.
- Ensure `get_spectrum()` and `get_waveform()` remain stable.

## 3) Phase 7 (Audio I/O extraction)

- Extract `start/stop/_start_loopback_capture/_start_input_capture/_init_butterworth_filter` to `audio_modules/audio_io.py`.
- Keep current behavior for WASAPI loopback/input mode and sample-rate wiring.

## High-Risk Areas / Watchouts

- **`__new__` test setups**: several tests bypass `__init__` and directly set fields; any new delegation must tolerate missing component objects via lazy creation or fallback.
- **Metronome state ownership**: avoid splitting downbeat state between old/new logic.
- **No algorithm drift during extraction**: keep this mechanical.
- **Realtime callback discipline**: no extra heavy allocations in hot path.

## Validation Gate (after each extraction slice)

Run at minimum:

- `python -m pytest tests/test_audio_engine_tempo_lock_hysteresis.py -q`
- `python -m pytest tests/test_phase2_readiness_silence.py -q`
- `python -m pytest tests/test_phase6_bpm_jitter.py -q`

Then run broader gate:

- `python -m pytest tests/ -x -q --tb=short`

## Files Introduced During This Refactor Sequence

- `audio_modules/session_stats.py`
- `audio_modules/auto_ranging.py`
- `audio_modules/beat_detector.py`
- `audio_modules/syncopation.py`
- `audio_modules/metronome.py`

## Practical Next Step for Next Agent

Start with **Phase 5 completion** only (one PR chunk):

- Move/delegate `_predict_next_beat`, `_validate_downbeat_against_pattern`, `_reset_downbeat_pattern`, `get_tempo_info`.
- Run the tempo-focused tests above.
- Stop and hand off before touching Phase 6.
