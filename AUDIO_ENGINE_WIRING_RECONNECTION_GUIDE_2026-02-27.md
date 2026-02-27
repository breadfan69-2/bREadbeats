# AudioEngine Wiring & Reconnection Guide (Pre-Refactor)

Date: 2026-02-27  
Workspace: `bREadbeats-master`

## Purpose
This document maps all current `AudioEngine` wiring paths so refactor/replacement work can reconnect behavior without missing hidden dependencies.

---

## 1) Runtime Topology (Current)

`main.py` owns lifecycle and constructs `AudioEngine(config, beat_callback)`.

### Main chain
1. `BREadbeatsWindow._start_engines()` creates `AudioEngine(self.config, self._audio_callback)` and calls `start()`.
2. `AudioEngine` runs PyAudio stream callback (`_audio_callback_pyaudio`) on audio thread.
3. Audio thread emits `BeatEvent` via callback to `main._audio_callback` (delegated to `event_handlers.audio_callback`).
4. `event_handlers.audio_callback`:
   - emits Qt signal `beat_detected` (GUI-thread handling),
   - emits Qt signal `spectrum_ready` (GUI-thread spectrum/waveform update),
   - runs motion generation (`StrokeMapper.process_beat`) and network send path (`network_engine.send_command`) on audio thread.

### Supporting modules in active path
- `event_handlers.py`: transport control, callback fanout, UI updates, metric feedback timer path.
- `stroke_mapper.py`: motion command synthesis from `BeatEvent` + audio-engine side reads.
- `beat_intelligence.py`: gate/decision engine used by `StrokeMapper`; writes silence gate back into audio engine.
- `tcode_wiring.py`: P0/C0/P1/P3 computation from command + spectrum + audio-engine band internals.
- `dialog_builders.py`: FFT diagnostics reads live engine sample-rate/fft-size.

---

## 2) Lifecycle Wiring (Start/Play/Stop)

## Start
- UI `Start` triggers `event_handlers.on_start_stop(..., checked=True)`.
- Calls `win._start_engines()` (`main.py`):
  - set audio device config (`device_index`, `is_loopback`),
  - instantiate/start `AudioEngine`,
  - set metric response speed,
  - sync metric checkbox state to engine,
  - instantiate `StrokeMapper(..., audio_engine=win.audio_engine)`.
- Transport sending is enabled immediately (`set_transport_sending(..., True)`), but volume zero until `Play` path ramps.

## Play
- UI `Play` triggers `event_handlers.on_play_pause(..., checked=True)`.
- Recreates `StrokeMapper` (important: fresh intelligence/motion state on each play).
- Enables warmup gate and starts volume ramp.

## Stop
- UI `Stop` triggers `event_handlers.on_start_stop(..., checked=False)`.
- Sends zero-volume command, disables transport sending, calls `win._stop_engines()`:
  - `audio_engine.stop()` then `win.audio_engine = None`,
  - `stroke_mapper = None`.

Refactor implication: preserve start/play/stop ordering and side effects (especially warmup + zero-volume + mapper recreation behavior).

---

## 3) Thread Boundaries (Critical)

- `AudioEngine._audio_callback_pyaudio`: audio thread.
- `event_handlers.audio_callback`: called from audio thread.
- `signals.beat_detected` / `signals.spectrum_ready`: handoff to GUI thread.
- `event_handlers.on_beat`, `on_spectrum`, `do_spectrum_update`, `update_display`: GUI thread.

Refactor implication: keep thread-safe API for `get_spectrum()`/`get_waveform()` (copy under lock) and avoid introducing GUI access in audio thread.

---

## 4) AudioEngine External Contract (What Callers Expect)

## Constructor/Lifecycle
- `AudioEngine(config, beat_callback)`
- `start()`
- `stop()`

## Public methods currently used externally
- `set_zscore_threshold(threshold)`
- `enable_metric_autoranging(metric, enable)`
- `set_metric_response_speed(speed)`
- `compute_energy_margin_feedback(band_energy, callback=...)`
- `compute_audio_amp_feedback(now, callback=...)`
- `get_metric_states()`
- `get_tempo_info()`
- `get_spectrum()`
- `get_waveform()`

## Public/semipublic attributes read or written externally
- `tempo_tracking_enabled` (write)
- `smoothed_tempo` (write/reset)
- `stable_tempo` (write/reset)
- `beat_intervals` (clear)
- `beat_times` (clear/read by internal metric logic)
- `beats_per_measure` (write)
- `measure_energy_accum` (write/reset)
- `measure_beat_counts` (write/reset)
- `beat_position_in_measure` (write/reset)
- `predicted_next_beat_mono` (read)
- `silence_gate_active` (write from BeatIntelligence)
- `fft_size` (read by diagnostics)
- `config.audio.sample_rate` (read by diagnostics)

## Private members currently consumed by other modules (high-risk coupling)
- `_band_energies` (read in `tcode_wiring.py`, `beat_intelligence.py`)
- `_band_zscore_signals` (read in `stroke_mapper.py`)
- `_estimate_frequency(...)` (called in `stroke_mapper.py`)
- `_metronome_bpm` (read in `stroke_mapper.py`, `beat_intelligence.py`)
- `_metronome_phase` (read in `beat_intelligence.py`)
- `_spectrum_skip_frames` (written in `main.py`)
- `_init_butterworth_filter()` (called in `main.py`)
- `_aggressive_tempo_snap_enabled` (written in `main.py`)

Refactor implication: either preserve these names temporarily or introduce adapter/getter APIs and migrate all consumers.

---

## 5) BeatEvent Contract (Downstream Consumers)

`AudioEngine` emits `BeatEvent` with fields consumed across stack:
- `timestamp`
- `monotonic_timestamp`
- `is_beat`
- `is_downbeat`
- `is_syncopated`
- `intensity`
- `frequency`
- `spectral_flux`
- `peak_energy`
- `bpm`
- `metronome_bpm`
- `acf_confidence`
- `tempo_locked`
- `tempo_reset`
- `phase_error_ms`
- `beat_band`
- `fired_bands`
- `beat_features`
- `raw_rms`
- `raw_rms_db`

Refactor implication: preserve schema and types; several gates and UI indicators depend on these directly.

---

## 6) Data/Control Paths You Must Reconnect

## A) Audio capture → beat event path
`AudioEngine._audio_callback_pyaudio`  
→ compute spectrum/energy/flux/tempo/gates  
→ create `BeatEvent`  
→ invoke callback (`beat_callback(event)`).

## B) Audio capture → visualization path
`AudioEngine` stores latest `spectrum_data` + `waveform_data` (thread-safe locks).  
`event_handlers.audio_callback` reads via `get_spectrum()`/`get_waveform()` and emits `spectrum_ready`.  
GUI timer (`do_spectrum_update`) updates visible canvas.

## C) Beat event → motion + network path
`event_handlers.audio_callback`  
→ `stroke_mapper.process_beat(event)`  
→ `tcode_wiring.compute_and_attach_tcode(..., spectrum)`  
→ `apply_volume_ramp(...)`  
→ `network_engine.send_command(cmd)`.

## D) Silence feedback loop (important closed loop)
`AudioEngine` computes raw RMS and applies hard silence veto.  
`BeatIntelligence.build_decision()` computes adaptive silence gate and writes `audio_engine.silence_gate_active`.  
Next audio frame: `AudioEngine` suppresses beats/metronome/syncopation if gate active.

This loop is explicitly relied on to prevent phantom beats at noise floor.

## E) Metric auto-adjust loop
GUI timer (`update_display`) calls:
- `audio_engine.compute_audio_amp_feedback(now, callback=win._on_metric_feedback)`
- `audio_engine.get_metric_states()`

Beat events also call:
- `audio_engine.compute_energy_margin_feedback(event.peak_energy, callback=...)`

Refactor implication: preserve callback shape and call cadence assumptions.

---

## 7) Config Wiring Paths

## Config -> engine init/runtime
Used heavily by `AudioEngine` internals:
- `config.audio.*` (`sample_rate`, `buffer_size`, `channels`, `device_index`, `gain`, `fft_size`, `spectrum_skip_frames`, `is_loopback`, `visualizer_enabled`, `highpass_filter_hz`, `use_butterworth`, `volume_normalize`)
- `config.beat.*` (tempo/tracking/gating/trigger bus/syncopation and many advanced tuning fields)
- `config.auto_adjust.metric_response_speed`

## UI -> config -> immediate engine mutation paths
- Z-score threshold slider -> `set_zscore_threshold(...)`
- Spectrum skip menu -> writes `audio_engine._spectrum_skip_frames`
- Beat frequency band slider -> calls `_init_butterworth_filter()` after config update
- Tempo tracking toggle -> writes/clears tempo fields
- Time signature change -> writes/reset measure tracking fields
- Generic tempo params -> `setattr(audio_engine, engine_attr, value)`
- Aggressive snap toggle -> writes `_aggressive_tempo_snap_enabled`

Refactor implication: these live mutations are part of expected UX and should remain hot-swappable while running.

---

## 8) File-by-File Wiring Map

## `main.py`
- Owns `self.audio_engine` lifecycle.
- Creates engine in `_start_engines()`, destroys in `_stop_engines()`.
- Forwards callback to `event_handlers.audio_callback` through `_audio_callback`.
- Pushes runtime config changes into engine (mix of public and private members).

## `event_handlers.py`
- Start/stop/play orchestration uses `win._start_engines()` / `win._stop_engines()`.
- Audio callback reads spectrum/waveform, emits Qt signals, executes stroke/network path.
- GUI beat handler queries tempo info and metric feedback.
- GUI timer polls metric feedback continuously.

## `stroke_mapper.py`
- Constructed with `audio_engine`; passes it to `BeatIntelligence`.
- Reads predicted beat timing and metronome tempo.
- Reads private band z-score signals and private frequency estimator.
- Uses live spectrum for fill motion modulation.

## `beat_intelligence.py`
- Reads spectrum and private band energies.
- Reads metronome state for journey timing (`_metronome_bpm`, `_metronome_phase`).
- Writes `audio_engine.silence_gate_active` each frame.

## `tcode_wiring.py`
- Uses private `_band_energies` for P0/C0 band modes.
- Uses event frequency/spectrum to derive P0/C0/P1/P3 command tags.

## `dialog_builders.py`
- FFT diagnostics reads `audio_engine.fft_size` and `audio_engine.config.audio.sample_rate` if engine exists.

---

## 9) Reconnection Checklist (Refactor Execution)

## Phase 1: Preserve external interface
- Keep `AudioEngine(config, beat_callback)` signature.
- Keep `BeatEvent` schema unchanged.
- Keep these methods callable: `start`, `stop`, `get_spectrum`, `get_waveform`, `get_tempo_info`, metric methods.

## Phase 2: Add compatibility adapter for private couplings
Provide temporary shim accessors or mirrored fields for:
- `_band_energies`
- `_band_zscore_signals`
- `_estimate_frequency`
- `_metronome_bpm`
- `_metronome_phase`
- `_spectrum_skip_frames`
- `_init_butterworth_filter`
- `_aggressive_tempo_snap_enabled`

Then migrate callers to public APIs incrementally.

## Phase 3: Reconnect lifecycle and thread behavior
- Ensure callback still runs on audio thread.
- Keep signal handoff and lock/copy semantics for spectrum/waveform.
- Preserve start/play/stop order and mapper recreation-on-play behavior.

## Phase 4: Reconnect feedback loops
- Silence closed-loop writeback (`BeatIntelligence` -> `AudioEngine.silence_gate_active`).
- Metric feedback callbacks and polling cadence from GUI timer.

## Phase 5: Validate critical behaviors
- Startup with selected audio device (loopback/input).
- Beat indicator + downbeat indicator behavior.
- Spectrum/waveform visuals update normally.
- TCode path still attaches P0/C0/P1/P3 and sends commands.
- Silence-to-music transitions do not generate phantom beats.

---

## 10) Recommended Contract Hardening (Post-Reconnect)

After refactor reconnect succeeds, reduce hidden coupling by introducing public APIs:
- `get_band_energies()`
- `get_band_zscore_signals()`
- `get_metronome_state()` (bpm, phase, predicted next beat)
- `estimate_frequency_in_band(spectrum, low, high)` (public wrapper)
- `set_spectrum_skip_frames(value)`
- `rebuild_band_filter()`
- `set_aggressive_tempo_snap_enabled(enabled)`

Then replace direct private-member reads/writes in `main.py`, `stroke_mapper.py`, `beat_intelligence.py`, and `tcode_wiring.py`.

---

## 11) Quick Regression Targets (Existing Tests)

Relevant current tests covering contract-sensitive areas:
- `tests/test_stroke_mapper_contract.py`
- `tests/test_phase2_readiness_silence.py`
- `tests/test_phase3_gates.py`
- `tests/test_audio_engine_tempo_lock_hysteresis.py`

Run these first during refactor reconnect before broader suites.

---

## Bottom Line
The highest-risk reconnection points are **private member couplings** and the **silence feedback loop**. If those are preserved (or intentionally shimmed), most UI/motion/network behavior should remain stable while internals are refactored.