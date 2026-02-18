# BeatIntelligence Port Audit — 2026-02-17

**Source of truth:** commit `ab2b35a` (last pre-refactor, `stroke_mapper.py` at 4,257 lines)  
**Refactor commit:** `24ae407` ("Refactor stroke mapper to decision-orbital adapter")  
**Current state:** `beat_intelligence.py` at 276 lines  

---

## Summary

The refactor preserved the *architecture* of intelligence (correct class name, broadly correct method
signatures) but stripped almost all of the *substance*. What is active today is roughly 15% of what
ran before: silence gate, 4-band EMA, single-frame bass check, and BPM-based journey timing.

Missing components are grouped below by severity.

---

## 1. CRITICAL — Missing, causing wrong behaviour today

### 1.1 Rolling history deques (all missing)

The old code maintained per-frame rolling windows (typically 18 frames) for:

| Deque | Purpose |
|---|---|
| `_recent_low_band_values` | 18-frame bass occupancy history |
| `_recent_high_band_values` | 18-frame treble occupancy history |
| `_recent_mid_bass_values` | 18-frame 200–400 Hz support history |
| `_recent_high_band_beat_hits` | Per-beat boolean treble-hit pattern |
| `_recent_flux_values` | Short-window flux EMA for center-jitter guard |

`BeatIntelligence` has none of these. Every gate that checks "is this consistently present over
time" is impossible without them. Current gates evaluate only the instantaneous EMA value.

**Effect:** The occupancy-based bass and treble gates always pass or always fail based on the last
frame only. The system cannot distinguish a sustained bass bed from an isolated bass spike.

---

### 1.2 FluxTracker (missing)

Old implementation: `_update_flux_history` + `_get_flux_rise_factor`

```python
# Stored (timestamp, spectral_flux) tuples over a 250ms window
def _get_flux_rise_factor(self) -> float:
    oldest_flux = self._flux_history[0][1]
    newest_flux = self._flux_history[-1][1]
    rise = max(0.0, newest_flux - oldest_flux)
    return min(1.0, rise / 0.1)  # 0–1 urgency
```

Current code uses raw `event.spectral_flux` instantaneously against a threshold. The bloom
response has no "urgency ramp" — a fast-rising spectrum and a flat one get the same radius.

**Effect:** bloom radius is underresponsive to percussive hits and fast transients.

---

### 1.3 Beat hierarchy guard — `_has_recent_beats` (missing)

The old gate enforced:
- Syncopation only fires after a recent beat/downbeat within ~900ms
- Beats only fire after a confirmed downbeat
- A "tempo-reset motion hold" timer prevents false-triggering after lock loss

Current `classify_trigger` fires syncopation on any `is_syncopated == True` event with no
history check. There is no downbeat-before-beat requirement.

**Effect:** Syncopation and beat strokes fire at the very start of a track, before any
rhythmic context is established. They also fire during lock-loss transients.

---

### 1.4 No-beat timeout → park decay (missing)

If no beats arrive for ~2 seconds, position should decay back to the park position. The
autopsy document explicitly flagged this as **NECESSARY**.

Nothing in `StrokeMapper.process_beat()` or `BeatIntelligence` tracks time-since-last-beat
and enforces decay. If the audio engine stops sending events, the last rendered position
is frozen indefinitely.

---

## 2. IMPORTANT — Gate cascade gutted entirely

### 2.1 `_is_low_band_full_enough` (missing)

The most complex single gate. Evaluates:
- Mean of 18-frame low-band history ≥ threshold
- Occupancy (frames above floor) ≥ 62%
- Low/high ratio ≥ 0.58 (prevents mid/treble-only material from firing)
- Optional mid-bass support check (200–400 Hz occupancy) when treble is absent

Current replacement (`_strict_bass_motion_allowed`): checks only `event.beat_band` and
`event.fired_bands`, which is a single-frame event field, not a history gate.

**Effect:** Vocal-only passages, isolated wah pedal hits, and high-frequency percussion
all pass the bass gate and trigger strokes. The old gate specifically filtered these.

---

### 2.2 `_passes_dual_band_db_gate` (missing)

Requires both ends of the spectrum to be above configurable dB floors simultaneously:
- `config.stroke.dual_band_sub_bass_db_min` (default −15 dB)
- `config.stroke.dual_band_high_db_min` (default −30 dB)

Also includes a "tip fullness" sub-gate checking 3.5–16 kHz occupancy over an 18-frame window.

Controlled by `config.stroke.dual_band_db_gate_enabled`. Not present in any form.

**Effect:** Mono-band content (e.g., pure bass rumble with no treble shimmer, or treble
shimmer with no bass) triggers full strokes. The old gate required a "full spectrum" to
confirm the music is actually playing and not just a transient artefact.

---

### 2.3 `_is_mid_trigger_blocked` (missing)

Configurable Hz-range suppressor. When a beat event's `frequency` falls inside
`[block_mid_trigger_low_hz, block_mid_trigger_high_hz]` (defaults: 100–2000 Hz), the
trigger is suppressed. Designed to filter vocal fundamental and formant-range beats.

Controlled by `config.stroke.block_mid_trigger_range_enabled`. Not present.

**Effect:** Vocal melody beats, guitar fundamental hits, and snare hits that fire in the
mid range all trigger strokes when they should not.

---

### 2.4 `_get_spectrum_fill_ratio` + `_passes_overall_amp_fill_gate` (missing)

The most sophisticated gate. Reads the live FFT spectrum from `audio_engine.get_spectrum()`,
normalises to peak, and checks what fraction of bins (within configurable bin windows per
phase) are above a threshold.

Features:
- Per-phase bin windows: `downbeat_fill_bin_low/high`, `beat_fill_bin_low/high`,
  `syncopation_fill_bin_low/high` in `config.stroke`
- Auto-adapting EMA offset per phase that adjusts `fill_required` up/down based on
  recent pass rate (target 58% pass rate, controlled by `overall_amp_fill_auto_*` config keys)
- Near-silence detection that disables the EMA adaptation during fadeouts

Gone entirely. No spectrum density check of any kind is present.

**Effect:** Single-sine tones, sparse synths, and near-silence transients all pass
the fill gate and trigger strokes. The old gate specifically prevented this.

---

### 2.5 `_update_motion_mode` — FULL_STROKE vs CREEP_MICRO (missing)

500ms hysteresis amplitude-level switch:
- `rms_envelope > amplitude_gate_high` (~0.15 default) → FULL_STROKE mode
- `rms_envelope < amplitude_gate_low` (~0.05 default) → CREEP_MICRO mode

Current code: `interval_beats_for_trigger` maps trigger kind to beats (1/2/4/8) with no
amplitude-level mode concept. Everything goes through the same mapping regardless of
whether the audio is quiet or loud.

**Effect:** Quiet passages trigger the same stroke intensity/duration as loud passages.
The "creep during quiet, stroke during loud" behaviour that was the original design intent
is absent.

---

## 3. LEARNING PIPELINE — Missing in full

### 3.1 `_build_runtime_feature_values` (missing)

Maps a `BeatEvent` to the 13 input features the model expects. Includes fallback logic
for when `event.beat_features` dict is absent:

```python
{
    'rms':                  from event.beat_features['energy_mean'] or event.peak_energy
    'log_energy':           log10(rms + eps)
    'spectral_flux':        from event.beat_features['flux_mean'] or event.spectral_flux
    'flux_delta':           flux_peak - flux_mean
    'sub_bass_energy':      self._sub_bass_energy
    'low_mid_energy':       self._low_mid_energy
    'mid_energy':           self._mid_energy
    'high_energy':          self._high_energy
    'low_high_ratio':       (sub_bass + low_mid) / (high + eps)
    'spectral_centroid_hz': event.frequency
    'spectral_bandwidth_hz':event.beat_features['freq_delta']
    'spectral_rolloff_hz':  centroid + 0.5 * bandwidth
    'spectral_flatness':    0.35 + 0.50 * (1 - energy_norm)
}
```

Without this, inference is impossible even if the model is loaded.

---

### 3.2 `_predict_learning_targets` (missing)

Full inference pipeline:
1. Build feature dict via `_build_runtime_feature_values`
2. Z-score normalise each feature against `_learning_norm_mean` / `_learning_norm_std`
3. For each target: `value = intercept + sum(coef[f] * normalised[f] for f in features)`
4. Clamp outputs to valid ranges:
   - `arc_size`: 0–1
   - `arc_duration_frac`: 0.1–4.0
   - `jitter_mix`: 0–1
   - `creep_mix`: 0–1
   - `gate_strictness`: 0–1
   - `burst_prob`: 0–1
5. Apply `cadence_rule` to derive `beats_between_strokes` (quiet→4, mid→2, active→1)
   based on a weighted combination of normalised RMS and flux

The `cadence_rule.beats_between_strokes` output is a **third pacing dimension** that the
new interval-beats mapping has no equivalent for. This is the mechanism by which the model
slows down stroking during quiet passages and speeds it up during energetic ones — entirely
absent from the current system.

---

### 3.3 `_try_load_learning_model` (missing — incomplete in guide)

The old loader:
- Iterated `_candidate_learning_model_paths()` (multiple fallback paths)
- Validated `status == 'ok'`
- Extracted and stored: `feature_columns`, `normalization.mean`, `normalization.std`,
  `models` (per-target intercept + coefficients), and `cadence_rule`
- Set `_learning_model_loaded = True` and logged path on success

Current `BeatIntelligence` has no `_learning_model_loaded` flag, no stored model fields,
and no loader at all. The guide's Step 6 skeleton is correct in shape but must also
extract and store `cadence_rule` — that field is missing from the guide's version.

---

## 4. GRAY AREA — Autopsy said PORT, never ported

### 4.1 `_stabilize_unlocked_bpm` + `_cap_bpm_to_last_locked`

During tempo lock loss, BPM can spike wildly. Old code:
- Capped orbit speed to `_last_locked_bpm` to prevent sudden spin acceleration
- Applied a per-frame jump ratio limit (default 12% per frame) before lock

Current `effective_bpm` clips to [40, 240] but has no memory of the last locked value.

**Effect:** When metronome loses lock between beats, creep orbit speed can jump abruptly.

---

### 4.2 `_update_bass_jitter_drive` (disconnected stub)

Maps bass frequency (30–220 Hz) to jitter speed multiplier (depth 0.03–0.075). Lower
bass pitch → slower jitter pace, higher bass pitch → faster jitter pace. Smoothed with
EMA to prevent frame-to-frame flicker.

Autopsy verdict: port but leave disconnected until a T-Code aux output is wired. Not
ported at all — no stub, no field.

---

### 4.3 Flux drop guard (missing)

When low-band flux drops suddenly (significant downward delta), the old code fell back
to creep mode to let the sound "settle." Autopsy verdict: PORT (yes).

Not present. No flux drop detection of any kind.

---

### 4.4 Post-silence volume ramp (missing)

After silence gate opens (audio resumes), the old code ramped volume up over ~500ms to
prevent jarring re-entry. Autopsy verdict: PORT.

Current `build_decision` sets `volume=0` during silence and lets `StrokeMapper.process_beat`
set full `get_volume()` immediately on re-open. No ramp.

---

## 5. What IS correctly ported

| Component | Status |
|---|---|
| 4-band EMA energy tracking (`update_band_energies`) | ✓ Correct |
| RMS envelope with attack/release (`update_envelope`) | ✓ Correct |
| Silence deadzone hysteresis gate (`update_silence_deadzone_gate`) | ✓ Correct |
| Basic trigger classification — syncopation/beat/downbeat/creep | ✓ Correct |
| Tempo confidence check (`_tempo_ready_for_motion`) | ✓ Correct |
| Single-frame strict bass gate (`_strict_bass_motion_allowed`) | ✓ Partial (no history) |
| Journey progress / S-curve timing (`update_journey_progress`) | ✓ Correct |
| Treble lift EMA with landing guard (`compute_treble_lift`) | ✓ Correct |
| Sub-bass bloom radius formula (`compute_radius_bloom_from_sub_bass`) | ✓ Correct |
| Journey continuation between discrete events (`build_decision`) | ✓ Correct |

---

## 6. Implementation priority order

| Priority | Item | Effort |
|---|---|---|
| 1 | Rolling history deques (all 5) — foundation for all gate logic | Low — just `collections.deque` and per-frame `.append()` calls |
| 2 | FluxTracker (250ms deque + rise factor) | Low |
| 3 | `_has_recent_beats` + beat-hierarchy guards | Low |
| 4 | No-beat timeout → park decay | Low |
| 5 | `_is_low_band_full_enough` | Medium — depends on deques |
| 6 | `_passes_dual_band_db_gate` | Medium |
| 7 | `_is_mid_trigger_blocked` | Low |
| 8 | `_passes_overall_amp_fill_gate` + `_get_spectrum_fill_ratio` | High — most complex |
| 9 | `_update_motion_mode` (FULL_STROKE/CREEP_MICRO) | Medium |
| 10 | `_build_runtime_feature_values` | Low |
| 11 | `_predict_learning_targets` (incl. cadence_rule) | Medium |
| 12 | `_try_load_learning_model` (complete, incl. cadence_rule field) | Low |
| 13 | `_cap_bpm_to_last_locked` + `_stabilize_unlocked_bpm` | Low |
| 14 | Post-silence volume ramp | Low |
| 15 | `_update_bass_jitter_drive` (stub, leave disconnected) | Low |
| 16 | Flux drop → creep fallback guard | Low |

Items 1–4 are the foundation. Ship those first; items 5–9 depend on the deques being populated.
Items 10–12 are the learning pipeline and can be done independently once 1–4 are in.

---

## 7. Reference commit for source code

All original implementations are available at git commit `ab2b35a`:

```
git show ab2b35a:stroke_mapper.py | Select-String "def _update_flux_history" -Context 20
```

Key line numbers in that commit (for reference, may shift on checkout):

| Method | Approx. line |
|---|---|
| `_update_flux_history` + `_get_flux_rise_factor` | 758 |
| `_has_recent_beats` | 791 |
| `_try_load_learning_model` | 1050 |
| `_build_runtime_feature_values` | 1083 |
| `_predict_learning_targets` | 1120 |
| `_update_motion_mode` | 1368 |
| `_stabilize_unlocked_bpm` | 1458 |
| `_is_low_band_full_enough` | 1540 |
| `_passes_dual_band_db_gate` | 1606 |
| `_is_mid_trigger_blocked` | 1669 |
| `_get_spectrum_fill_ratio` | 1770 |
| `_passes_overall_amp_fill_gate` | 1808 |
| `_get_high_band_activity` | 1844 |
| `_get_high_band_presence_status` | 1880 |
| `_get_high_band_pattern_status` | 1912 |
| `_update_bass_jitter_drive` | ~1960 |
