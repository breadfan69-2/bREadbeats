# Motion Trigger Implementation Handoff (2026-02-24)

Companion to:
- `MOTION_TRIGGER_IMPLEMENTATION_BLUEPRINT_2026-02-24.md`

---

## 1) Current Status Snapshot

Progress is currently at **Stage 6 complete** with Stage 4 ownership switch and Stage 5 sidecar integration completed behind flags.

What is done:
- Stage 0 scaffold complete (`audio_modules` package, contracts, no-op module shells).
- Stage 1 pure feature extraction complete enough for migration baseline:
  - spectrum band slicing
  - spectral flux math
  - dominant frequency estimation
  - multiband energy aggregation helpers
  - band fire-history primary selection helper
  - teaching feature normalization/offbeat/confidence helpers
  - bass-dominance helper
- Stage 2 helper extraction integrated:
  - reference BPM selector
  - phase-accept window math
  - raw-onset dedup and metronome phase-error helpers
  - ACF octave candidate build/select helpers
  - onset-BPM interval estimator helper
  - runtime `TempoTracker` state mirror and sync wiring
  - ACF/onset BPM fusion adapter method in `TempoTracker`
  - metronome phase-step helper extraction (`step_metronome_phase`)
  - ACF smoothing/jump-gating helper extraction (`smooth_acf_bpm_with_jump_gating`)
  - shadow telemetry scaffold wired for legacy/new agreement counters and ACF smoothing decision tags
- Stage 3 complete in shadow mode:
  - `event_detector.py` confidence fusion path is running in callback
  - side-by-side telemetry captures `legacy_fire/new_fire/agreement` and cue means
- Stage 4 complete (controlled ownership switch):
  - metronome remains final owner when lock path is active
  - raw-onset fallback owner can switch to fusion detector when `new_trigger_fusion_enabled=True` and `new_trigger_shadow_mode=False`
  - immediate rollback preserved via flags
- Stage 5 complete:
  - `audioflux_adapter.py` now provides fail-open, bounded-buffer, stride-based optional `af_*` cues
  - `audio_engine.py` threads sidecar cues into `FeatureFrame` as soft modifiers only
  - callback timing telemetry buckets are emitted in shadow telemetry (`frontend_ms`, `tempo_ms`, `detector_ms`, `sidecar_ms`)
- Stage 6 complete:
  - transient classifier + bass-dominance context are threaded into `beat_intelligence.py`
  - transient confidence hints are now flag-gated (`transient_classification_enabled`) for baseline parity when disabled
  - dedicated Stage 6-oriented tests added for transient classifier and bass-dominance weighting

Validation status:
- Repeated focused regression run after each slice:
  - `python -m unittest tests.test_telemetry_tuning tests.test_tempo_tracker_helpers tests.test_event_detector_fusion tests.test_shadow_replay_harness tests.test_phase6_bpm_jitter tests.test_audioflux_adapter_failopen`
- Result at handoff: **passing (27 focused tests in latest Stage 5 run)**.

---

## 2) Files Changed So Far

Primary edited files:
- `audio_engine.py`
- `config.py`
- `audio_modules/feature_extractors.py`
- `audio_modules/tempo_tracker.py`
- `audio_modules/contracts.py`
- `audio_modules/__init__.py`
- `audio_modules/signal_frontend.py`
- `audio_modules/event_detector.py`
- `audio_modules/audioflux_adapter.py`
- `audio_modules/telemetry_tuning.py`

Planning/docs added:
- `MOTION_TRIGGER_IMPLEMENTATION_BLUEPRINT_2026-02-24.md`
- `MOTION_TRIGGER_UPGRADE_PLAN_2026-02-24.md`
- `AUDIOFLUX_INTEGRATION_FINDINGS_2026-02-24.md`

---

## 3) Architecture Delta Implemented

### 3.1 `audio_engine.py` now delegates pure math to modules
`audio_engine.py` imports helper functions from:
- `audio_modules.feature_extractors`
- `audio_modules.tempo_tracker`

This is still orchestration-only refactor work (no ownership flip).

### 3.2 `TempoTracker` is now a real runtime mirror
`TempoTracker` now stores `TempoState` and is synced each callback via `AudioEngine._sync_tempo_tracker_state(...)`.

`AudioEngine.get_tempo_info()` now reads mirrored state with safe fallback to existing engine fields.

### 3.3 Controlled ownership switch is now flag-gated
- Metronome ownership is unchanged and still authoritative when active/locked.
- Legacy and new raw-onset candidates are both evaluated with shared acceptance gating.
- Ownership selection for raw fallback is controlled by:
  - `new_trigger_fusion_enabled`
  - `new_trigger_shadow_mode`
- Shadow telemetry continues to compare against legacy path even when fusion ownership is enabled.

---

## 4) Known Safe Invariants

These must remain true during next steps:
1. With new features disabled, behavior should match baseline.
2. `audio_engine.py` remains callback owner until explicit Stage 3/4 switch.
3. `TempoTracker` extraction should be incremental and adapter-first (avoid sudden state relocation).
4. Keep silence veto and refractory semantics unchanged.
5. Keep metronome/downbeat ownership semantics unchanged until dedicated migration step.

---

## 5) Recommended Next Steps (Ordered)

## Stage 5 next slices
1. Keep sidecar optional/fail-open and validate startup behavior with and without `audioflux` installed.
2. Tune sidecar cue normalization and confirm it remains soft influence only.
3. Add callback timing telemetry buckets for sidecar cost (`sidecar_ms`) before any heavier extraction.

---

## 6) Testing Protocol at Each Slice

Run after each meaningful change:

```bash
python -m unittest tests.test_config_persistence tests.test_config_migration tests.test_phase6_bpm_jitter
```

Optional broader run before ownership changes:

```bash
python -m unittest
```

If any behavior-sensitive change is made to metronome logic, compare:
- lock behavior,
- jitter profile,
- downbeat consistency,
- silence behavior.

---

## 7) Rollback Procedure

If a migration step causes instability:
1. Revert only the latest slice touching tempo logic.
2. Keep pure helper extractions that are already validated.
3. Preserve `audio_engine.py` ownership and use fallback values in `get_tempo_info()`.
4. Re-run focused tests above before proceeding.

---

## 8) Open Items / TODO

- `audioflux_adapter.py` remains a fail-open optional sidecar and currently uses lightweight local feature approximations rather than direct native `audioflux` transform objects.
- Future optional enhancement: broaden replay-fixture packs for larger A/B telemetry baselines.

---

## 9) Handoff Intent

This handoff preserves a stable baseline while progressively relocating logic from monolithic `audio_engine.py` into modules, with Stage 4 ownership controls and Stage 5 sidecar integration now active behind conservative flags.

The next contributor should continue with Stage 5 timing/packaging validation and then proceed to Stage 6 policy enrichment.
