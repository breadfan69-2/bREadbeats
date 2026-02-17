# GUI Controls Refactor Audit — Post-StrokeMapper Deletion

**Date:** 2026-02-17  
**Status:** Research / Report — no code changes  
**Companion doc:** `STROKEMAPPER_AUTOPSY_AND_REFACTOR_ROADMAP.md`  
**Authoritative engine state:** Orbital geometry parks at $(0, -0.7)$, uses S-curve interpolation for 1/2/4/8-beat journeys.

---

## Roadmap Review — Quick Verdict

The `STROKEMAPPER_AUTOPSY_AND_REFACTOR_ROADMAP.md` is **solid and accurate**. Specific notes:

| Section | Verdict |
|---------|---------|
| §1 Executive Summary (Muscle/Intelligence/Plumbing split) | **Correct.** The 3-layer breakdown matches actual code structure. |
| §2 Signal Processing Autopsy | **Correct.** 4-band EMA at α=0.2, flux rise factor, spectrum fill ratio — all confirmed present and well-described. |
| §3 Logic Gate Inventory | **Correct and matches GATE_MAP.md.** The 14 gates are real. The PORT/DELETE tags are sound. |
| §4 Redundancy Report | **Correct.** ~1,800 lines dead weight is realistic. The teaching/learning note ("[keep][needs own module]") is wise. |
| §5 Clean Break Plan | **Good architecture.** `beat_intelligence.py` is the right extraction target. The 6-phase plan is sequential and safe. |
| §6 FFT-to-Radius (Bloom) | **Good formula.** Bass fill 70/30 weighting and the $x^{1.5}$ curve are sensible starting points. May need tuning in practice. |
| §7 Treble Elevator | **Good formula.** The $x^{2.0}$ curve requiring strong treble for noticeable lift is appropriate — prevents over-reaction to faint highs. |
| §8 Refactor Roadmap for Codex | **Well-sequenced.** Extract → Wire → Simplify → Delete → Gut → Archive is the correct order. |

**One flag:** The roadmap says "1/2/4/8-beat journeys" but the file header says "2/4/8 beat journeys" — confirm whether 1-beat syncopation strokes are in the orbital engine yet. The roadmap assumes they are.

---

## Table of Contents

1. [Control Disposition Summary](#1-control-disposition-summary)
2. [Main Controls Panel](#2-main-controls-panel)
3. [Beat Detection Tab](#3-beat-detection-tab)
4. [Stroke Settings Tab](#4-stroke-settings-tab)
5. [Effects / Axis Tab](#5-effects--axis-tab)
6. [Trigger Settings Dialog — Full Breakdown](#6-trigger-settings-dialog)
7. [Geometry Rest State Popout](#7-geometry-rest-state-popout)
8. [Tempo Tracking Popout](#8-tempo-tracking-popout)
9. [Pulse / Carrier / TCode Tab](#9-pulse--carrier--tcode-tab)
10. [Config Keys With No GUI (Hidden/Code-Only)](#10-config-keys-with-no-gui)
11. [New Controls Needed Post-Refactor](#11-new-controls-needed-post-refactor)
12. [Implementation Checklist](#12-implementation-checklist)

---

## 1. Control Disposition Summary

| Disposition | Count | Meaning |
|-------------|-------|---------|
| **KEEP** | ~55 | Signal-processing tuning, tempo, audio config — feeds `beat_intelligence.py` |
| **DELETE** | ~30 | Legacy drawing, trajectory shapes, thump, noise burst random patterns, BPM cadence thinning |
| **REPURPOSE** | ~10 | Existing controls that map to new orbital concepts (bloom, lift, journey) |
| **NEW** | ~5 | Controls the new orbital engine needs that don't exist yet |

---

## 2. Main Controls Panel

*Always visible above tabs. Line ref: main.py L7764+*

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Power | `stroke.combo_power` | **KEEP** | Master intensity scaler — applies to bloom depth |
| Depth | `stroke.combo_depth` | **REPURPOSE** | Was stroke depth → becomes **bloom reach** (max orbit radius) |
| Speed | `stroke.combo_speed` | **KEEP** | Journey speed multiplier — still useful |
| Texture | `stroke.combo_texture` | **REPURPOSE** | Was shape texture → becomes **S-curve sharpness** or orbit smoothness |
| Reaction | `stroke.combo_reaction` | **KEEP** | Beat reactivity — how aggressively bloom responds to beats |
| Tempo lock required | `beat.tempo_lock_required` | **KEEP** | Essential gate — port to `beat_intelligence` |
| Sensitivity (fill gate scale) | `stroke.overall_amp_fill_required_scale` | **KEEP** | Master sensitivity scaler for fill gates |
| Stroke mode combo | `stroke.mode` | **DELETE** | Hidden, locked to SIMPLE_CIRCLE. Orbital engine has one mode. |
| combo_size | `stroke.combo_size` | **DELETE** | No GUI widget, only in preset save. Orbital engine has fixed geometry. |

---

## 3. Beat Detection Tab

*Line ref: main.py L8477+*

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Detection Type combo | (internal) | **KEEP** | Still chooses Peak Energy / Spectral Flux / Combined |
| Auto-Levels group (all) | `auto_adjust.*` | **KEEP** | Auto-tuning is upstream of stroke mapper — independent |
| Butterworth bandpass | `audio.use_butterworth` | **KEEP** | Audio pipeline config |
| Freq Range (Hz) | `beat.freq_low/high` | **KEEP** | Beat detection freq band |
| Audio Amplification | `audio.gain` | **KEEP** | Input gain |
| Sensitivity | `beat.sensitivity` | **KEEP** | Beat detection sensitivity |
| Z-Score Sens | (z-score threshold) | **KEEP** | Beat detection z-score |
| Flux Multiplier | `beat.flux_multiplier` | **KEEP** | Beat detection flux scaling |
| Depth (peak floor) | `beat.peak_floor` | **KEEP** | Beat detection floor |
| Peak Decay | `beat.peak_decay` | **KEEP** | Beat detection decay |
| Rise Sensitivity | `beat.rise_sensitivity` | **KEEP** | Beat detection attack |

**Beat Detection tab verdict: ALL KEEP.** Nothing here is about drawing or stroke_mapper geometry.

---

## 4. Stroke Settings Tab

*Line ref: main.py L9174+*

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Stroke Min/Max (RangeSlider) | `stroke.stroke_min/max` | **DELETE** | Fixed by S-curve geometry. Orbital radius is set by bloom, not min/max range. |
| Stroke Fullness | `stroke.stroke_fullness` | **DELETE** | Replaced by bloom formula. "Roundness" of arc is meaningless in orbital model. |

**Stroke Settings tab verdict: ENTIRE TAB DELETABLE.** This tab will be empty. Replace contents with new orbital controls (see §11) or merge remaining controls into another tab.

---

## 5. Effects / Axis Tab

*Line ref: main.py L9205+*

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Jitter (checkbox) | `jitter.enabled` | **REPURPOSE** | Jitter → "orbit wobble" or micro-bloom pulse. Keep the enable toggle. |
| Circle Size (jitter.amplitude) | `jitter.amplitude` | **REPURPOSE** | Maps to micro-bloom pulse amplitude |
| Circle Speed (jitter.intensity) | `jitter.intensity` | **REPURPOSE** | Maps to micro-bloom pulse frequency |
| Creep (checkbox) | `creep.enabled` | **DELETE** | Creep is now the 8-beat journey — it's always "on" when amplitude is low. No toggle needed. |
| Creep Speed | `creep.speed` | **DELETE** | 8-beat journey speed is fixed by BPM. |
| Alpha Weight | `alpha_weight` | **KEEP** | Vertical axis weight — still applies to orbital output |
| Beta Weight | `beta_weight` | **KEEP** | Horizontal axis weight — still applies to orbital output |
| Volume Reduction Limit | `stroke.vol_reduction_limit` | **KEEP** | Volume envelope, not drawing |

---

## 6. Trigger Settings Dialog — Full Breakdown

*The big one. Line ref: main.py L4639+*

### 6.1 Syncopation / Double-Stroke Group

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Enable syncopation | `beat.syncopation_enabled` | **KEEP** | Controls whether 1-beat S-curve journeys fire |
| Detection band combo | `beat.syncopation_band` | **KEEP** | Which freq band detects off-beats |
| Off-beat window | `beat.syncopation_window` | **KEEP** | Timing window for off-beat detection |
| BPM limit | `beat.syncopation_bpm_limit` | **KEEP** | Disable off-beats above this tempo |
| Syncopation arc size | `beat.syncopation_arc_size` | **DELETE** | "Arc size" is legacy drawing. New engine: 1-beat journey has fixed geometry. Bloom magnitude controls perceived "size". |
| Syncopation speed | `beat.syncopation_speed` | **DELETE** | Journey speed is BPM-locked in new engine (1 beat = 1 beat). |

### 6.2 Amplitude Gate Group

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Full stroke threshold (enter) | `stroke.amplitude_gate_high` | **KEEP** | RMS threshold for HIGH motion mode → port to `beat_intelligence` |
| Creep threshold (exit) | `stroke.amplitude_gate_low` | **KEEP** | RMS threshold for LOW motion mode → port to `beat_intelligence` |
| Full-stroke dwell bias | `stroke.full_stroke_dwell_bias` | **DELETE** | Prevented rapid mode flip. New engine: silence gate hysteresis replaces this. |
| Require bass z-score bands | `beat.strict_bass_motion_gate_enabled` | **KEEP** | Bass-only beat gating → port to `beat_intelligence` |
| Allow motion only below (Hz) | `beat.motion_freq_cutoff` | **KEEP** | Frequency ceiling for motion triggers |
| Enable overall amp + fill gate | `stroke.overall_amp_fill_gate_enabled` | **KEEP** | Master enable for fill gates |
| New Gate Priority | `stroke.new_gate_priority_enabled` | **DELETE** | This was a transition shim to bypass the old `overall_activity_guard`. Post-refactor, the simplified 5-gate cascade makes this irrelevant. |
| Overall amp target | `stroke.overall_amp_fill_target` | **KEEP** | Feeds spectrum_fullness gate |
| Overall amp tolerance | `stroke.overall_amp_fill_tolerance` | **KEEP** | Feeds spectrum_fullness gate |
| Downbeat fill required | `stroke.downbeat_overall_amp_fill_required` | **KEEP** | Per-trigger fill threshold |
| Beat fill required | `stroke.beat_overall_amp_fill_required` | **KEEP** | Per-trigger fill threshold |
| Syncopation fill required | `stroke.syncopation_overall_amp_fill_required` | **KEEP** | Per-trigger fill threshold |
| Fill bin ranges (6 spins) | `stroke.*_fill_bin_{low,high}` | **KEEP** | Per-trigger FFT bin ranges for fill calc |
| Enable dual-band dB gate | `stroke.dual_band_db_gate_enabled` | **KEEP** | → port to `beat_intelligence` |
| Dual-band sub-bass min (dB) | `stroke.dual_band_sub_bass_db_min` | **KEEP** | Sub-bass dB floor |
| Dual-band high min (dB) | `stroke.dual_band_high_db_min` | **KEEP** | High-band dB floor |
| Enable high-tip fullness gate | `stroke.high_tip_fullness_enabled` | **KEEP** | Treble detail requirement |
| High-tip range Hz (low/high) | `stroke.high_tip_freq_low_hz/high_hz` | **KEEP** | Treble check band |
| High-tip dB min | `stroke.high_tip_db_min` | **KEEP** | Treble floor |
| High-tip occupancy | `stroke.high_tip_occupancy_threshold` | **KEEP** | Treble richness |
| Block mid trigger range | `stroke.block_mid_trigger_range_enabled` | **KEEP** | Vocal filtering gate |
| Mid block low/high Hz | `stroke.block_mid_trigger_low_hz/high_hz` | **KEEP** | Mid-range block band |

### 6.3 Noise Burst Group — **ENTIRE GROUP DELETE**

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Enable noise burst | `stroke.noise_burst_enabled` | **DELETE** | Random micro-patterns replaced by deterministic orbital model |
| Burst flux multiplier | `stroke.noise_burst_flux_multiplier` | **DELETE** | No random patterns |
| Burst magnitude | `stroke.noise_burst_magnitude` | **DELETE** | No random patterns |
| Burst scale | `stroke.noise_burst_scale` | **DELETE** | No random patterns |
| Downbeat jitter blend (%) | `stroke.downbeat_jitter_vector_percent` | **GRAY** | Could feed micro-bloom pulse direction if jitter is repurposed. Leave disconnected for now. |
| Bass→jitter speed (%) | `stroke.bass_jitter_speed_influence_percent` | **REPURPOSE** | Maps bass pitch to orbit speed variation → bloom modulator. Port. |
| Bass→jitter size (%) | `stroke.bass_jitter_size_influence_percent` | **DELETE** | Jitter "size" concept doesn't map to orbital model. |

### 6.4 Stroke Timing Group — **ENTIRE GROUP DELETE**

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Fallback beats/stroke | `stroke.beats_between_strokes` | **DELETE** | New engine has fixed 1/2/4/8 mapping |
| Auto cutoff 2→4 (BPM) | `stroke.bpm_cutoff_2_to_4` | **DELETE** | Fixed mapping |
| Auto cutoff 4→8 (BPM) | `stroke.bpm_cutoff_4_to_8` | **DELETE** | Fixed mapping |
| Cadence cutoff bias | `stroke.cadence_cutoff_bias_bpm` | **DELETE** | Fixed mapping |
| Scheduled lead (ms) | `beat.scheduled_lead_ms` | **KEEP** | Pre-fire lead — still useful for latency compensation |

### 6.5 Noise vs Metronome Priority — **DELETE**

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Noise-primary mode | `stroke.noise_primary_mode` | **DELETE** | Simplified trigger hierarchy replaces this |

### 6.6 Post-Silence Volume Ramp — **KEEP ALL**

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Volume reduction (%) | `stroke.post_silence_vol_reduction` | **KEEP** | Track-change behavior → port to `beat_intelligence` |
| Ramp duration (seconds) | `stroke.post_silence_ramp_seconds` | **KEEP** | Recovery time |
| Fade max drop points | `stroke.silence_fade_drop_points` | **KEEP** | Max volume dip |

### 6.7 Flux Sensitivity Group

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Flux threshold | `stroke.flux_threshold` | **KEEP** | Core flux gate → port to `beat_intelligence` |
| Flux Scaling (size) | `stroke.flux_scaling_weight` | **REPURPOSE** | Was flux→stroke size. Now becomes flux→bloom scale weight. |
| **Phase Advance** | `stroke.phase_advance` | **DELETE** | S-curve handles pacing. No manual phase advance needed. |
| Low-band drop ratio | `stroke.flux_drop_ratio` | **KEEP** | Flux drop guard for bass dropout → creep fallback |
| Enable low-band drop guard | `stroke.low_band_drop_guard_enabled` | **KEEP** | Bass dropout behavior |
| Low-band gate window (frames) | `stroke.low_band_window_frames` | **KEEP** | Bass gate window |
| Low-band mean threshold | `stroke.low_band_activity_threshold` | **KEEP** | Bass presence floor |
| Low-band Δ threshold | `stroke.low_band_delta_threshold` | **KEEP** | Bass change sensitivity |
| Low-band variance threshold | `stroke.low_band_variance_threshold` | **KEEP** | Bass stability |
| Low-band fullness occupancy | `stroke.low_band_fullness_occupancy_threshold` | **KEEP** | Bass richness |
| Low:high mean ratio min | `stroke.low_band_to_high_ratio_min` | **KEEP** | Bass/treble ratio |
| Enable mid-bass support | `stroke.mid_bass_support_enabled` | **KEEP** | Mid-bass fallback |
| Mid-bass range Hz | `stroke.mid_bass_freq_low_hz/high_hz` | **KEEP** | Mid-bass band |
| Mid-bass activity min | `stroke.mid_bass_activity_threshold` | **KEEP** | Mid-bass floor |
| Mid-bass occupancy | `stroke.mid_bass_occupancy_threshold` | **KEEP** | Mid-bass richness |
| Downbeat gate relax | `stroke.downbeat_low_band_relax` | **KEEP** | Downbeat bass easement |
| Require upper-band presence | `stroke.high_band_gate_enabled` | **KEEP** | Treble gate enable |
| Include mid in upper gate | `stroke.high_band_include_mid` | **KEEP** | Treble includes mid |
| High-band gate window (frames) | `stroke.high_band_window_frames` | **KEEP** | Treble gate window |
| High-band mean threshold | `stroke.high_band_mean_threshold` | **KEEP** | Treble floor |
| High-band fill floor | `stroke.high_band_floor_threshold` | **KEEP** | Treble spectrum floor |
| High-band occupancy threshold | `stroke.high_band_occupancy_threshold` | **KEEP** | Treble richness |
| High-band Δ threshold | `stroke.high_band_delta_threshold` | **KEEP** | Treble change sensitivity |
| High-band variance threshold | `stroke.high_band_variance_threshold` | **KEEP** | Treble stability |
| Upper pattern window (beats) | `stroke.high_band_pattern_window_beats` | **KEEP** | Treble pattern memory |
| Upper pattern min hits | `stroke.high_band_pattern_min_hits` | **KEEP** | Treble pattern threshold |
| Downbeat high-band relax | `stroke.downbeat_high_band_relax` | **KEEP** | Downbeat treble easement |
| Block strokes when overall quiet | `stroke.overall_activity_guard_enabled` | **DELETE** | Subsumed by silence gate in new 5-gate cascade |
| Overall low flux threshold | `stroke.overall_low_flux_threshold` | **DELETE** | Subsumed by silence gate |
| Overall low energy threshold | `stroke.overall_low_energy_threshold` | **DELETE** | Subsumed by silence gate |
| Block center+jitter reset | `beat.center_jitter_flux_guard_enabled` | **KEEP** | Prevents jitter reset during activity |
| Center reset guard Δflux | `beat.center_jitter_flux_delta_threshold` | **KEEP** | Activity change threshold |
| Center reset guard avg | `beat.center_jitter_flux_avg_threshold` | **KEEP** | Activity average threshold |

---

## 7. Geometry Rest State Popout

*Line ref: main.py L4200+*

| Control | Config Key | Verdict | Reason |
|---------|------------|---------|--------|
| Rest Y Offset | `stroke.geometry_y_offset` | **REPURPOSE** | Was manual Y offset. New engine parks at fixed $(0, -0.7)$. This could become a **park position override** or be deleted if park is always $-0.7$. |
| Sink Start Intensity | `stroke.geometry_sink_start_intensity` | **DELETE** | Legacy intensity→position mapping. Orbital engine doesn't "sink" based on intensity. Bloom handles this now. |

---

## 8. Tempo Tracking Popout

*Line ref: main.py L4156+, L9293+*

**ALL KEEP.** Every control in this popout is about tempo detection, metronome PLL, phase locking — none of it is about drawing. These feed directly into `beat_intelligence.evaluate_stroke_readiness()`.

Key controls preserved:
- ACF interval, BPM alpha slow/fast, PLL window/base/conf gains
- Fusion min/max ACF weight
- Beat de-dup fraction, Phase accept window
- Target-BPS lock gate (checkbox + confidence + downbeat matches)
- Aggressive tempo snap (checkbox + confidence/phase error/BPM jump/min matches)

---

## 9. Pulse / Carrier / TCode Tab

**ALL KEEP.** The Pulse Frequency, Carrier Frequency, Pulse Width, and Rise Time groups are TCode output mapping — completely independent of stroke_mapper geometry.

---

## 10. Config Keys With No GUI (Hidden/Code-Only)

These exist in `config.py` / `config.json` but have no visible control:

| Config Key | Verdict | Reason |
|------------|---------|--------|
| `stroke.continuation_arcs_enabled` | **DELETE** | Orbital engine is always continuous |
| `stroke.thump_enabled` | **DELETE** | Landing feel is S-curve native |
| `stroke.rhythmic_phrasing_enabled` | **DELETE** | Replaced by orbital S-curve natively |
| `stroke.rhythmic_phrase_min_intensity` | **DELETE** | Part of rhythmic phrasing |
| `stroke.rhythmic_phrase_energy_low/high` | **DELETE** | Part of rhythmic phrasing |
| `stroke.rhythmic_phrase_ease_mode` | **DELETE** | Part of rhythmic phrasing |
| `stroke.rhythmic_phrase_explode_radius_min/max` | **DELETE** | Part of rhythmic phrasing |
| `stroke.single_stroke_bpm_cutoff` | **DELETE** | Fixed 1/2/4/8 mapping |
| `stroke.combo_size` | **DELETE** | No GUI, orbital geometry is fixed |
| `stroke.minimum_depth` | **DELETE** | Legacy depth concept |
| `stroke.freq_depth_factor` | **DELETE** | Legacy freq→depth mapping |
| `stroke.depth_freq_low/high` | **DELETE** | Legacy freq→depth band |
| `stroke.flux_depth_factor` | **DELETE** | Legacy flux→depth |
| `stroke.flux_depth_boost_enabled` | **DELETE** | Legacy flux→depth |
| `stroke.min_interval_ms` | **DELETE** | Fixed journey durations |
| `stroke.silence_flux_multiplier` | **KEEP** | Silence detection tuning |
| `stroke.silence_energy_multiplier` | **KEEP** | Silence detection tuning |
| `stroke.silence_multiplier_locked` | **KEEP** | Silence detection lock |
| `beat.teaching_*` (all 10+ keys) | **KEEP** (separate module) | Will move to own module per roadmap |

---

## 11. New Controls Needed Post-Refactor

These controls don't exist yet but are needed for the orbital engine + `beat_intelligence` system:

| Proposed Control | Purpose | Suggested Location |
|------------------|---------|-------------------|
| **Bloom Max Radius** | Maximum orbit expansion on loud bass ($0.0$–$0.5$) | Main Controls (replace "Depth" label) or new Orbital tab |
| **Treble Lift Max** | Maximum Y center lift on bright treble ($0.0$–$0.5$) | New Orbital tab or Effects tab |
| **Bloom Curve Exponent** | Non-linearity of bass→radius mapping (default $1.5$) | Advanced/Trigger Settings |
| **Treble Lift Curve** | Non-linearity of treble→lift mapping (default $2.0$) | Advanced/Trigger Settings |
| **No-Beat Timeout (ms)** | Time before orbit decays to park when no beats fire (default $2000$ms) | Advanced/Trigger Settings or Tempo Tracking |

---

## 12. Implementation Checklist

### Phase A: Delete Dead Controls (do first — reduces UI clutter)

- [ ] Remove Stroke Settings tab contents (`stroke_min/max`, `stroke_fullness`)
- [ ] Remove Noise Burst group (6 controls) — save `bass_jitter_speed_influence_percent` for repurpose
- [ ] Remove Stroke Timing group (5 controls) — save `scheduled_lead_ms` and move it
- [ ] Remove Noise vs Metronome Priority group (1 control)
- [ ] Remove from Amplitude Gate: `full_stroke_dwell_bias`, `new_gate_priority_enabled`
- [ ] Remove from Flux Sensitivity: `phase_advance`, overall activity guard (checkbox + 2 sliders)
- [ ] Remove from Syncopation: `syncopation_arc_size`, `syncopation_speed`
- [ ] Remove Geometry popout: `geometry_sink_start_intensity`
- [ ] Hide/remove `mode_combo` (already hidden — delete the widget entirely)
- [ ] Remove Creep checkbox + speed slider from Effects tab

### Phase B: Repurpose Surviving Controls

- [ ] Rename "Depth" (combo_depth) → "Bloom Reach" in Main Controls
- [ ] Rename "Texture" (combo_texture) → label TBD (S-curve smoothness?)
- [ ] Rename "Jitter" group → "Orbit Wobble" or "Micro Bloom"
- [ ] Rename `geometry_y_offset` → "Park Y Override" or delete if park is always $-0.7$
- [ ] Rename `flux_scaling_weight` → "Flux → Bloom Weight"
- [ ] Move `bass_jitter_speed_influence_percent` to a "Bloom Modulators" sub-group
- [ ] Move `scheduled_lead_ms` to Tempo Tracking popout or Beat Detection tab

### Phase C: Add New Orbital Controls

- [ ] Add "Bloom Max Radius" slider (re-skin Stroke Settings tab → "Orbital Settings")
- [ ] Add "Treble Lift Max" slider
- [ ] Add "Bloom Curve" advanced slider
- [ ] Add "Treble Lift Curve" advanced slider
- [ ] Add "No-Beat Timeout (ms)" spin

### Phase D: Clean Config Keys

- [ ] Remove all deleted config keys from `config.py` dataclass
- [ ] Remove from `config.json` defaults
- [ ] Remove from preset save/load in `main.py` (L7900+ area)
- [ ] Add new config keys for bloom/lift/timeout
- [ ] Migration: bump `config.version` and handle old configs gracefully

---

## Appendix: Full DELETE List (Config Keys)

```
stroke.mode                              → hardcoded, widget already hidden
stroke.stroke_min                        → fixed by S-curve
stroke.stroke_max                        → fixed by S-curve
stroke.stroke_fullness                   → replaced by bloom
stroke.minimum_depth                     → legacy
stroke.freq_depth_factor                 → legacy
stroke.depth_freq_low                    → legacy
stroke.depth_freq_high                   → legacy
stroke.flux_depth_factor                 → legacy
stroke.flux_depth_boost_enabled          → legacy
stroke.min_interval_ms                   → fixed journey durations
stroke.combo_size                        → no GUI, orbital geometry fixed
stroke.phase_advance                     → S-curve handles pacing
stroke.full_stroke_dwell_bias            → silence gate hysteresis replaces
stroke.geometry_sink_start_intensity     → bloom handles this
stroke.rhythmic_phrasing_enabled         → S-curve native
stroke.rhythmic_phrase_min_intensity     → S-curve native
stroke.rhythmic_phrase_energy_low        → S-curve native
stroke.rhythmic_phrase_energy_high       → S-curve native
stroke.rhythmic_phrase_ease_mode         → S-curve native
stroke.rhythmic_phrase_explode_radius_min → S-curve native
stroke.rhythmic_phrase_explode_radius_max → S-curve native
stroke.single_stroke_bpm_cutoff         → fixed 1/2/4/8 mapping
stroke.bpm_cutoff_2_to_4                → fixed 1/2/4/8 mapping
stroke.bpm_cutoff_4_to_8                → fixed 1/2/4/8 mapping
stroke.beats_between_strokes            → fixed 1/2/4/8 mapping
stroke.cadence_cutoff_bias_bpm          → fixed 1/2/4/8 mapping
stroke.continuation_arcs_enabled        → orbital always continuous
stroke.thump_enabled                    → S-curve landing native
stroke.noise_burst_enabled              → no random patterns
stroke.noise_burst_flux_multiplier      → no random patterns
stroke.noise_burst_magnitude            → no random patterns
stroke.noise_burst_scale                → no random patterns
stroke.noise_primary_mode               → simplified trigger hierarchy
stroke.new_gate_priority_enabled        → old transition shim
stroke.overall_activity_guard_enabled   → subsumed by silence gate
stroke.overall_low_flux_threshold       → subsumed by silence gate
stroke.overall_low_energy_threshold     → subsumed by silence gate
stroke.bass_jitter_size_influence_percent → doesn't map to orbital
stroke.downbeat_jitter_vector_percent   → gray area, disconnect for now
beat.syncopation_arc_size               → fixed geometry
beat.syncopation_speed                  → BPM-locked
```

## Appendix: Full KEEP List (Config Keys — Port to beat_intelligence)

```
stroke.amplitude_gate_high              → RMS mode-switch (high)
stroke.amplitude_gate_low               → RMS mode-switch (low)
stroke.flux_threshold                   → Base flux threshold
stroke.flux_scaling_weight              → Flux→bloom scale (repurpose)
stroke.flux_drop_ratio                  → Bass dropout guard
stroke.silence_threshold                → Silence noise floor
stroke.silence_flux_multiplier          → Silence detection
stroke.silence_energy_multiplier        → Silence detection
stroke.silence_multiplier_locked        → Silence lock
stroke.low_band_*                       → All bass gate controls (10 keys)
stroke.mid_bass_*                       → All mid-bass fallback controls (5 keys)
stroke.high_band_*                      → All treble gate controls (11 keys)
stroke.dual_band_*                      → Both dual-band dB gates (3 keys)
stroke.high_tip_*                       → All high-tip fullness controls (5 keys)
stroke.block_mid_trigger_*              → Mid-range block controls (3 keys)
stroke.overall_amp_fill_*               → Fill gate controls (8 keys)
stroke.*_fill_bin_*                     → Per-trigger bin ranges (6 keys)
stroke.downbeat_low_band_relax          → Downbeat bass easement
stroke.downbeat_high_band_relax         → Downbeat treble easement
stroke.low_band_drop_guard_enabled      → Bass dropout behavior
stroke.post_silence_*                   → Post-silence ramp (3 keys)
stroke.vol_reduction_limit              → Volume envelope
stroke.bass_jitter_speed_influence_percent → Repurpose as bloom modulator
beat.syncopation_enabled                → Enable 1-beat journeys
beat.syncopation_band                   → Off-beat detection band
beat.syncopation_window                 → Off-beat timing window
beat.syncopation_bpm_limit              → Off-beat tempo ceiling
beat.strict_bass_motion_gate_enabled    → Bass-only filtering
beat.motion_freq_cutoff                 → Frequency ceiling
beat.tempo_lock_required                → Tempo confidence gate
beat.teaching_metronome_relaxed_confidence → Min metronome confidence
beat.center_jitter_flux_*               → Center reset guards (3 keys)
beat.scheduled_lead_ms                  → Pre-fire latency compensation
beat.teaching_*                         → All teaching/learning keys (own module)
All tempo tracking keys                 → Untouched
All audio.* keys                        → Untouched
All auto_adjust.* keys                  → Untouched
All pulse_freq/carrier_freq/pulse_width/rise_time keys → Untouched
All device_limits.* keys                → Untouched
alpha_weight, beta_weight, volume       → Untouched
jitter.enabled, jitter.intensity, jitter.amplitude → Repurpose as micro-bloom
```
