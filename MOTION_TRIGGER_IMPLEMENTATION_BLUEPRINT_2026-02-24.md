# Motion Trigger System Implementation Blueprint (2026-02-24)

## 1) Purpose

This document is the execution plan for implementing the new motion-trigger stack and extracting `audio_engine.py` into focused modules.

It operationalizes:
- `MOTION_TRIGGER_UPGRADE_PLAN_2026-02-24.md`
- `AUDIOFLUX_INTEGRATION_FINDINGS_2026-02-24.md`
- current runtime behavior in `audio_engine.py` + `beat_intelligence.py`

Guiding rule: **no risky rewrite**. Use incremental extraction with feature flags and A/B telemetry.

---

## 2) Non-Negotiable Constraints

1. **Realtime safety first**
   - Callback path cannot block on optional work.
   - Any new computation must be bounded and fail-open.

2. **Behavior parity when disabled**
   - With new flags off, output must match current baseline.

3. **Single beat owner at event time**
   - Keep metronome ownership when locked.
   - Raw detector remains fallback / confidence source.

4. **Windows freeze safety**
   - Optional libs (e.g., `audioflux`) must never break startup.

---

## 3) Target Module Map

## New modules (incremental)

1. `audio_modules/signal_frontend.py`
- Mono conversion, buffering, windowing, FFT frame production.
- Output: `FrontendFrame` with spectrum, beat_spectrum/band_energy inputs.

2. `audio_modules/feature_extractors.py`
- Spectral flux, band energies, deltas, HFC proxy, RMS-derived cues.
- Output: `FeatureFrame`.

3. `audio_modules/tempo_tracker.py`
- ACF estimation, tempo fusion, metronome phase, downbeat validation.
- Output: `TempoState`.

4. `audio_modules/event_detector.py`
- Confidence-fused beat scoring + hysteresis/refractory.
- Output: `TriggerDecision` (`raw_onset`, `beat_score`, `is_candidate`, class confidences).

5. `audio_modules/audioflux_adapter.py` (optional)
- Lazy import, stride-based compute, bounded ring buffer, fail-open.
- Output: optional sidecar fields (`af_*`).

6. `audio_modules/telemetry_tuning.py`
- Side-by-side legacy/new detector metrics and runtime timing summaries.

## Existing modules kept as owners
- `audio_engine.py` remains orchestration shell and callback owner.
- `beat_intelligence.py` remains motion-policy owner.

---

## 4) Data Contracts (Stage 0 deliverable)

Create dataclasses in `audio_modules/contracts.py`:

1. `FrontendFrame`
- `mono_time`, `wall_time`, `spectrum`, `band_energy`, `spectral_flux`, `raw_rms`, `raw_rms_db`

2. `FeatureFrame`
- Base cues: `flux_norm`, `energy_norm`, `energy_delta`, `flux_delta`, `hfc_proxy`
- Band cues: `sub_bass`, `low_mid`, `mid`, `high`, `bass_dominance`
- Optional sidecar cues: `af_entropy`, `af_flatness`, `af_hfc`, `af_novelty`, `af_rms`, `af_onset_conf`

3. `TempoState`
- `metronome_bpm`, `acf_confidence`, `tempo_locked`, `phase_error_ms`, `is_downbeat`, `beat_phase`

4. `TriggerDecision`
- `beat_score`, `raw_onset_conf`, `is_beat_candidate`, `kick_like_conf`, `hat_like_conf`, `mixed_conf`, `reason_codes`

5. `EngineDecision`
- final callback-facing decision (`is_beat`, `is_downbeat`, `beat_band`, `fired_bands`, confidence context)

---

## 5) Feature Flags to Add

Add these toggles to config (default conservative):

- `beat.new_trigger_fusion_enabled = False`
- `beat.new_trigger_telemetry_enabled = True`
- `beat.new_trigger_shadow_mode = True`  (compute new path, do not own firing)
- `beat.bass_dominance_weighting_enabled = False`
- `beat.transient_classification_enabled = False`
- `beat.audioflux_enabled = False`
- `beat.audioflux_emit_onset_confidence = True`
- `beat.audioflux_frame_stride = 2`
- `beat.audioflux_fft_size = 1024`

---

## 6) Confidence Fusion Spec (initial)

## Normalized cues (0..1)
- `c_flux`: novelty / flux-based cue
- `c_band_spike`: multi-band z-score cue strength
- `c_energy_delta`: short-timescale RMS/band derivative
- `c_phase_align`: proximity to expected beat phase (when metronome active)
- `c_sidecar`: optional `af_onset_conf`

## Base weighted score

`beat_score = w_flux*c_flux + w_band*c_band_spike + w_delta*c_energy_delta + w_phase*c_phase_align + w_sidecar*c_sidecar`

Initial defaults:
- `w_flux=0.28`
- `w_band=0.30`
- `w_delta=0.17`
- `w_phase=0.20`
- `w_sidecar=0.05` (0 when unavailable)

## Trigger thresholds
- Arm threshold: `T_on = 0.62`
- Sustain/release threshold: `T_off = 0.45`
- Refractory: inherit existing adaptive beat refractory guard

## Dominance adaptive weighting
- If bass-dominant, shift weight from high-band transient cues to low-band envelope cues.
- If treble-dominant, shift opposite direction but keep bass floor weight non-zero.

---

## 7) Transient Classifier Spec (initial)

Compute three confidences from short windows around candidate peaks:

- `kick_like_conf`
  - high low-band energy, lower HFC proportion, stronger envelope body
- `hat_like_conf`
  - strong HFC / high-band spike, sharp attack, short decay
- `mixed_conf`
  - balanced low/high evidence

Use in phase 1 of rollout only as telemetry and policy hint (not ownership).

---

## 8) Step-by-Step Implementation Sequence

## Stage 0 — Contracts + wiring shell (no behavior change)
1. Add `audio_modules/contracts.py`.
2. Add `audio_modules/__init__.py`.
3. Add adapter interfaces with no-op implementations.
4. Thread new config flags through config loader/defaults.

Exit criteria:
- Build/tests unchanged.
- Runtime behavior unchanged.

## Stage 1 — Extract pure feature functions
1. Move pure helpers from `audio_engine.py` into `feature_extractors.py`:
   - flux normalization and cue normalization helpers
   - band-energy aggregation helpers
   - bass dominance computation
2. Keep callback call sites in `audio_engine.py` but delegate computation.

Exit criteria:
- Beat count parity within tolerance in baseline tracks.
- No callback timing regression.

## Stage 2 — Tempo tracker extraction
1. Move ACF + metronome update logic into `tempo_tracker.py` state object.
2. Keep public outputs identical to current event payload fields.
3. Preserve existing lock/downbeat semantics.

Exit criteria:
- Existing BPM jitter tests continue passing.
- Downbeat flags and lock behavior parity on replay traces.

## Stage 3 — New detector in shadow mode
1. Implement `event_detector.py` beat score path.
2. Compute decision in callback but do not own firing when `shadow_mode=True`.
3. Log side-by-side outcomes:
   - `legacy_fire`, `new_fire`, `agreement`, `new_score`, `cue_vector`.

Exit criteria:
- Agreement rate target reached on representative set.
- No over-trigger spikes in silence windows.

## Stage 4 — Controlled ownership switch
1. Enable `new_trigger_fusion_enabled=True` in dev profile only.
2. Keep metronome as final owner when lock is valid.
3. Use new detector for raw onset candidate path.
4. Keep immediate rollback flag.

Exit criteria:
- Better miss/false-positive profile than legacy baseline.
- Tempo stability not degraded.

## Stage 5 — Optional audioflux sidecar
1. Implement `audioflux_adapter.py` with strict fail-open behavior.
2. Inject `af_*` cues into fusion only as soft modifiers.
3. Validate dev/unfrozen then frozen app behavior.

Exit criteria:
- No startup failures without package.
- Sidecar CPU budget under target.

## Stage 6 — Motion policy enrichment
1. Thread classifier + bass dominance context to `beat_intelligence.py`.
2. Apply conservative depth/radius modifiers.
3. Keep existing gate hierarchy and safety clamps.

Exit criteria:
- Improved kick-vs-hat expressiveness without cadence instability.

---

## 9) Legacy Gate Reduction Plan

Treat old auto-tuning/gating sections as three categories:

1. **Keep as hard safety**
- silence veto
- refractory
- metronome lock ownership

2. **Freeze to observe-only during migration**
- auto-adjust loops that mutate thresholds continuously
- target BPS metric writes under lock conditions

3. **Retire after A/B evidence**
- any gate that contributes no measurable precision/recall gain and increases jitter.

Retirement rule: remove only after telemetry proves no regression on full track set.

---

## 10) Telemetry & A/B Schema

For each frame (or decimated sample), log:
- `time_mono`, `raw_rms_db`, `band_energy`, `spectral_flux`
- legacy decision flags
- new `beat_score`, cue components, threshold state
- metronome state (`bpm`, `acf_conf`, `phase_error_ms`, `tempo_locked`)
- `fired_bands`, `bass_dominance`, transient class confidences
- callback compute times (`frontend_ms`, `tempo_ms`, `detector_ms`, `sidecar_ms`)

For each track/session summary:
- agreement %, misses, extra fires, silence false fires, bpm jitter stats, octave flips, relock time

---

## 11) Test Plan

## Unit tests (new)
- `tests/test_event_detector_fusion.py`
- `tests/test_bass_dominance_weighting.py`
- `tests/test_transient_classifier.py`
- `tests/test_audioflux_adapter_failopen.py`

## Regression tests (existing + extended)
- keep `tests/test_phase6_bpm_jitter.py`
- add replay comparison fixtures for legacy vs shadow mode

## Runtime checks
- callback timing budget checks
- silence false-trigger checks
- lock-hold and relock behavior checks

---

## 12) Definition of Done

1. Module extraction complete with orchestration still in `audio_engine.py`.
2. New detector can run in shadow mode and active mode by flag.
3. No startup/runtime failures when `audioflux` missing.
4. Quantified improvement in trigger quality on the representative track pack.
5. No regression in tempo stability and no callback overrun increase.

---

## 13) Rollback Strategy

- Fast rollback: set `new_trigger_fusion_enabled=False`.
- Sidecar rollback: set `audioflux_enabled=False`.
- If instability observed, force legacy ownership and keep shadow telemetry for diagnosis.

---

## 14) Immediate Next Implementation Tasks

1. Create `audio_modules/contracts.py` and feature-flag config additions.
2. Extract pure feature helpers into `feature_extractors.py`.
3. Introduce `event_detector.py` in shadow mode with telemetry only.
4. Build replay harness for baseline/new comparison on saved track set.

This order gives maximum safety and fastest signal on whether the new trigger logic is outperforming legacy gating.
