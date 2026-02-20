# bREadbeats Diff-Only Settings (Tuned Home Presync)

Generated: 2026-02-20

Source config: `config.tuned_home_presync.json`

This report lists only values that differ from default schema values, plus missing/extra keys.

Summary: 91 changed, 6 missing, 3 extra

## Changed values

| Setting | tuned value | Default |
|---|---:|---:|
| audio.device_index | 16 | null |
| audio.gain | 6.2 | 1.0 |
| audio.sample_rate | 48000 | 44100 |
| auto_adjust.metric_response_speed | 1.16 | 1.0 |
| beat.aggressive_snap_confidence | 0.61 | 0.55 |
| beat.aggressive_snap_max_bpm_jump_ratio | 0.09 | 0.12 |
| beat.aggressive_tempo_snap_enabled | true | false |
| beat.beat_dedup_fraction | 0.28 | 0.22 |
| beat.center_jitter_flux_guard_enabled | true | false |
| beat.flux_multiplier | 10.0 | 1.0 |
| beat.freq_high | 212.0 | 150.0 |
| beat.motion_freq_cutoff | 180.0 | 500.0 |
| beat.peak_decay | 0.443 | 0.9 |
| beat.peak_floor | 0.668 | 0.08 |
| beat.phase_accept_window_ms | 67.0 | 85.0 |
| beat.phase_snap_weight | 0.8 | 0.3 |
| beat.rise_sensitivity | 0.4 | 0.5 |
| beat.scheduled_lead_ms | 6 | 0 |
| beat.sensitivity | 0.48 | 0.5 |
| beat.silence_reset_ms | 179 | 400 |
| beat.stability_threshold | 0.13 | 0.28 |
| beat.strict_bass_motion_gate_enabled | true | false |
| beat.syncopation_arc_size | 0.82 | 0.5 |
| beat.syncopation_band | "low_mid" | "any" |
| beat.syncopation_bpm_limit | 130.0 | 160.0 |
| beat.syncopation_speed | 1.0 | 0.5 |
| beat.syncopation_window | 0.09 | 0.15 |
| beat.teaching_apply_in_circle_mode | true | false |
| beat.teaching_learning_strength | 0.2 | 0.55 |
| beat.teaching_metronome_relaxed_confidence | 0.25 | 0.14 |
| beat.teaching_min_confidence | 0.18 | 0.12 |
| beat.teaching_no_motion_bias | 0.72 | 1.0 |
| beat.teaching_rule_fit_path | "defaults/learning/rule_fit.tranquilizer_blend.json" | "datasets/rule_fit.json" |
| beat.teaching_stroke_ready_grace_ms | 2662.0 | 450.0 |
| beat.tempo_timeout_ms | 1100 | 2000 |
| carrier_freq.monitor_freq_max | 4000.0 | 200.0 |
| device_limits.p0_freq_max | 100.0 | 0.0 |
| device_limits.p0_freq_min | 1.0 | 0.0 |
| device_limits.prompted | true | false |
| jitter.size | 0.012 | 0.024 |
| pulse_freq.monitor_freq_max | 4000.0 | 200.0 |
| stroke.bass_jitter_size_influence_percent | 0.0 | 100.0 |
| stroke.beat_fill_bin_high | 2 | 512 |
| stroke.beat_overall_amp_fill_required | 0.24 | 0.7 |
| stroke.block_mid_trigger_high_hz | 4000.0 | 2000.0 |
| stroke.block_mid_trigger_low_hz | 170.0 | 100.0 |
| stroke.block_mid_trigger_range_enabled | false | true |
| stroke.bpm_cutoff_2_to_4 | 135.0 | 60.0 |
| stroke.bpm_cutoff_4_to_8 | 150.0 | 180.0 |
| stroke.combo_depth | 1.0000000000000002 | 1.0 |
| stroke.combo_power | 1.0000000000000004 | 1.0 |
| stroke.combo_reaction | 1.0300000000000002 | 1.0 |
| stroke.combo_size | 1.0800000000000003 | 1.0 |
| stroke.combo_speed | 1.0200000000000002 | 1.0 |
| stroke.combo_texture | 0.94 | 1.0 |
| stroke.depth_freq_high | 22050.0 | 200.0 |
| stroke.downbeat_fill_bin_high | 246 | 512 |
| stroke.downbeat_fill_bin_low | 30 | 0 |
| stroke.downbeat_high_band_relax | 0.898 | 0.9 |
| stroke.downbeat_low_band_relax | 0.589 | 0.85 |
| stroke.downbeat_overall_amp_fill_required | 0.06 | 0.75 |
| stroke.dual_band_high_db_min | -21.0 | -30.0 |
| stroke.dual_band_sub_bass_db_min | -22.2 | -15.0 |
| stroke.flux_depth_factor | 0.12 | 0.0 |
| stroke.flux_threshold | 0.068 | 0.03 |
| stroke.freq_depth_factor | 0.07 | 0.3 |
| stroke.high_band_floor_threshold | 0.031 | 0.06 |
| stroke.high_band_mean_threshold | 0.079 | 0.12 |
| stroke.high_band_occupancy_threshold | 0.412 | 0.55 |
| stroke.low_band_activity_threshold | 0.209 | 0.2 |
| stroke.low_band_window_frames | 15 | 18 |
| stroke.min_interval_ms | 150 | 260 |
| stroke.noise_burst_enabled | false | true |
| stroke.noise_burst_flux_multiplier | 3.32 | 2.0 |
| stroke.noise_burst_magnitude | 2.12 | 1.0 |
| stroke.noise_burst_scale | 0.0699999999999997 | 0.35 |
| stroke.noise_primary_mode | true | false |
| stroke.orbit_geometry.creep.center_y | 0.4 | 0.1 |
| stroke.overall_amp_fill_required_scale | 0.8499999999999994 | 0.5 |
| stroke.overall_amp_fill_target | 0.11 | 0.5 |
| stroke.overall_amp_fill_tolerance | 0.63 | 0.5 |
| stroke.overall_low_energy_threshold | 0.18 | 0.14 |
| stroke.overall_low_flux_threshold | 0.16 | 0.06 |
| stroke.phase_advance | 1.0 | 0.25 |
| stroke.post_silence_ramp_seconds | 2.7 | 3.0 |
| stroke.post_silence_vol_reduction | 0.47 | 0.15 |
| stroke.silence_close_threshold | 0.008 | 0.048 |
| stroke.silence_threshold | 0.002 | 0.04 |
| stroke.stroke_fullness | 0.98 | 0.7 |
| stroke.stroke_min | 0.0 | 0.2 |
| stroke.syncopation_overall_amp_fill_required | 0.42 | 0.6 |

## Missing canonical keys in tuned config

| Setting | Expected default |
|---|---:|
| stroke.beat_overall_amp_fill_sustain_frames | 3 |
| stroke.downbeat_overall_amp_fill_sustain_frames | 3 |
| stroke.orbit_geometry.start.center_y | 0.2 |
| stroke.orbit_geometry.start.max_radius | 0.92 |
| stroke.orbit_geometry.start.park_radius | 0.7 |
| stroke.syncopation_overall_amp_fill_sustain_frames | 3 |

## Extra keys in tuned config

| Setting | tuned value |
|---|---:|
| app_run_count | 89 |
| beat.teaching_profile_path | "profile.tranquilizer.json" |
| stroke.overall_amp_fill_sustain_frames | 3 |
