# Session Summary — 2026-02-13

## What changed
- Fixed visual orientation for `Waterfall` and `MountainRange` canvases so motion/peaks render in expected direction.
- Tuned `MountainRange` vertical mapping to avoid over-compressed low-end display and use a practical dB display window.
- Added FFT visualization normalization in `audio_engine.py` (visual path) to stabilize display scaling.
- Decoupled visual peak/flux bar rendering from beat-detection thresholds; visual bars now derive from spectrum dynamics (GUI-side).
- Made `event.frequency` band-aware using `stroke.depth_freq_low/high`, so depth band selection directly affects stroke depth mapping input.
- Added mode-safe radius cap behavior in `stroke_mapper.py`:
  - `stroke_fullness` acts as baseline radius ceiling.
  - `freq_depth_factor` + depth can expand toward edge.
  - Applied consistently to Simple Circle, Teardrop, Spiral, and Syncopated arcs.
- Fixed bass motion gate behavior so `motion_freq_cutoff` is only enforced when `strict_bass_motion_gate_enabled` is ON.
- Moved **Motion Band Cutoff** UI control to Advanced Controls next to **Bass Gating** and removed duplicate from Beat Detection Levels.
- Set Bass Gating default to OFF:
  - `config.py` default updated.
  - `config.json` current workspace value updated.

## Verification status
- Python cache folders cleared (`__pycache__`, `.pytest_cache`).
- App restarted successfully with latest code.
- Runtime logs show healthy startup and active processing.

## Next step
- User validation pass in live app; after confirmation, commit and push all session changes.
