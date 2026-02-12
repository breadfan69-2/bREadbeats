# bREadbeats Repair Proposal

## Objectives
- Restore full config persistence and enum correctness.
- Fix initial auto-connect retry logic in NetworkEngine.
- Outline maintainability refactors to reduce risk and ease testing.

## Quick Wins (target for immediate patch)
1) Config load/hydration
- Hydrate all saved sections: pulse_width, rise_time, device_limits, log_level.
- Coerce enums on load: BeatDetectionType, StrokeMode.
- Keep defaults intact for missing fields; preserve existing save path logic.

2) Network auto-reconnect
- Ensure the worker retries even after the first connection failure by not gating on _was_connected (or initialize it on first attempt).
- Add lightweight logging for first-failure retry to aid diagnosis.

## Optional Near-Term Improvements
- Validate loaded config ranges using BEAT_RANGE_LIMITS to guard against invalid persisted values.
- Add a small version tag in the saved config to allow future migrations.

## Structural Refactor (incremental)
- Split main.py into focused modules:
  - ui_app.py (Qt app bootstrap + splash)
  - ui_windows.py (BREadbeatsWindow + widgets/layout)
  - ui_plot.py (SpectrumCanvas)
  - config_io.py (save/load helpers, enum coercion, migrations)
- Carve utility units out of audio_engine.py (e.g., z-score detector, ACF/metronome helpers) to enable unit tests without Qt.
- Keep public interfaces stable to avoid ripples; refactor with thin shim imports from main.py to new modules.

## Test Plan (post-fix)
- Manual: save settings with device_limits + pulse_width + rise_time; restart and confirm values and prompt state persist.
- Network: start with auto_connect true against a closed port, observe retries and eventual connect once server appears.
- Smoke: run app, verify spectrum draws, beats fire strokes, and no regressions in startup logs.

## Risks & Mitigations
- Config migrations: mitigate by defaulting missing fields and keeping coercion lenient.
- Reconnect loops: keep retry interval bounded (reuse reconnect_delay_ms) and log failures once per backoff cycle.
