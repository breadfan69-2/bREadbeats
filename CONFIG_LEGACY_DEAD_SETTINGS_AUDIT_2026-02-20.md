# Config Legacy/Dead Settings Audit

Generated: 2026-02-20

Scope: Runtime scan across core app modules (excluding tests/tools/archive scripts) plus schema comparison.

## Confirmed legacy/back-compat settings

These are explicitly marked as legacy/back-compat in code comments:

- `stroke.flux_depth_boost_enabled` (legacy/internal, UI toggle removed)
- `stroke.thump_enabled` (kept for preset compatibility)

Reference: [config.py](config.py#L131-L134), [config.py](config.py#L171)

## High-confidence likely dead (no runtime hits)

The following keys had no direct attribute access and no quoted-key access in runtime modules:

- `auto_adjust.auto_range_enabled`
- `auto_adjust.consec_beats`
- `auto_adjust.cooldown_sec`
- `auto_adjust.step_audio_amp`
- `auto_adjust.step_flux_mult`
- `auto_adjust.step_peak_decay`
- `auto_adjust.step_peak_floor`
- `auto_adjust.step_rise_sens`
- `auto_adjust.step_sensitivity`
- `auto_adjust.threshold_sec`
- `base_radius`
- `beat.amplification`
- `beat.syncopation_arc_size`
- `beat.syncopation_speed`
- `stroke.bass_jitter_size_influence_percent`
- `stroke.bass_jitter_speed_influence_percent`
- `stroke.beats_between_strokes`
- `stroke.bpm_cutoff_2_to_4`
- `stroke.bpm_cutoff_4_to_8`
- `stroke.cadence_cutoff_bias_bpm`
- `stroke.combo_power`
- `stroke.combo_speed`
- `stroke.dbfs_reference_window_ms`
- `stroke.downbeat_jitter_vector_percent`
- `stroke.flux_depth_factor`
- `stroke.freq_depth_factor`
- `stroke.geometry_sink_start_intensity`
- `stroke.new_gate_priority_enabled`
- `stroke.noise_burst_enabled`
- `stroke.noise_burst_flux_multiplier`
- `stroke.noise_burst_magnitude`
- `stroke.noise_burst_scale`
- `stroke.noise_primary_mode`
- `stroke.overall_activity_guard_enabled`
- `stroke.overall_low_energy_threshold`
- `stroke.overall_low_flux_threshold`
- `stroke.silence_energy_multiplier`
- `stroke.silence_flux_multiplier`
- `stroke.silence_multiplier_locked`
- `stroke.single_stroke_bpm_cutoff`
- `stroke.stroke_fullness`
- `stroke.thump_enabled`

## Not dead (had runtime hits)

These came up in earlier broad candidates but do have runtime references:

- `jitter.size`
- `volume`

## Extra keys present in tuned config (not in current schema)

From [config.tuned_home_presync.json](config.tuned_home_presync.json):

- `app_run_count`
- `beat.teaching_profile_path`
- `stroke.overall_amp_fill_sustain_frames`

## Notes

- This is static analysis and may miss reflection-style usage outside scanned runtime modules.
- For reset planning, treat the high-confidence list as "safe to deprioritize" first, then verify behavior after a test build.
- Full machine-readable scan artifacts: [config_usage_audit.json](config_usage_audit.json), [config_strict_dead.txt](config_strict_dead.txt)