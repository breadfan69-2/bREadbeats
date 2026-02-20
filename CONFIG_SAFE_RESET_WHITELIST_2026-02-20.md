# Safe Reset Whitelist (Config)

Generated: 2026-02-20

Purpose: identify settings that are safe to ignore/reset first before touching active runtime controls.

## Tier 1 — Safe to ignore/remove first

These are outside current default schema (not canonical config keys):

- `app_run_count`
- `beat.teaching_profile_path`
- `stroke.overall_amp_fill_sustain_frames`

## Tier 2 — Legacy/back-compat knobs

These exist for compatibility and are not primary tuning controls:

- `stroke.flux_depth_boost_enabled`
- `stroke.thump_enabled`

## Tier 3 — Unread in runtime test session

Unread during instrumented runtime pass (186 tests). These are lower priority for reset sweeps:

- `auto_adjust.enabled_params.audio_amp`
- `auto_adjust.enabled_params.flux_mult`
- `auto_adjust.enabled_params.peak_decay`
- `auto_adjust.enabled_params.peak_floor`
- `auto_adjust.enabled_params.rise_sens`
- `auto_adjust.enabled_params.sensitivity`
- `stroke.orbit_geometry.beat.center_y`
- `stroke.orbit_geometry.beat.max_radius`
- `stroke.orbit_geometry.beat.park_radius`
- `stroke.orbit_geometry.creep.center_y`
- `stroke.orbit_geometry.creep.max_radius`
- `stroke.orbit_geometry.creep.park_radius`
- `stroke.orbit_geometry.downbeat.center_y`
- `stroke.orbit_geometry.downbeat.max_radius`
- `stroke.orbit_geometry.downbeat.park_radius`
- `stroke.orbit_geometry.start.center_y`
- `stroke.orbit_geometry.start.max_radius`
- `stroke.orbit_geometry.start.park_radius`
- `stroke.orbit_geometry.syncopation.center_y`
- `stroke.orbit_geometry.syncopation.max_radius`
- `stroke.orbit_geometry.syncopation.park_radius`

## Practical reset order

1. Remove/ignore Tier 1 keys from tuning decisions.
2. Leave Tier 2 at defaults unless you have a legacy preset dependency.
3. Reset active runtime keys first (all keys not listed above).
4. Only revisit Tier 3 if behavior still looks off after active-key reset.

## Evidence sources

- Runtime read audit: `config_runtime_read_audit.json`
- Runtime summary: `CONFIG_RUNTIME_READ_AUDIT_2026-02-20.md`
- Tuned diff report: `CONFIG_SETTINGS_DIFF_TUNED_HOME_PRESYNC.md`
- Legacy notes: `CONFIG_LEGACY_DEAD_SETTINGS_AUDIT_2026-02-20.md`
