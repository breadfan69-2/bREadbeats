# Volume-Dependency Audit — bREadbeats
**Date:** 2026-02-25  
**Scope:** Every place where raw absolute audio levels affect motion output  
**Status:** Thorough line-by-line audit of beat_intelligence.py, stroke_mapper.py, audio_engine.py, event_detector.py, feature_extractors.py

---

## Legend
- **VOLUME-DEPENDENT** — Raw absolute audio level directly affects motion; will behave differently at low vs high OS volume
- **VOLUME-INDEPENDENT** — Uses rolling normalization, ratios, or self-relative comparison; immune to OS volume changes
- **PARTIALLY DEPENDENT** — Uses ratio-based comparison but against a raw baseline that is itself volume-dependent

---

## Finding #1: `_transient_motion_profile` — raw `spectral_flux` threshold ⚠️ VOLUME-DEPENDENT

**File:** beat_intelligence.py **Lines:** 1272–1273

```python
flux_now = float(np.clip(getattr(event, "spectral_flux", 0.0) or 0.0, 0.0, 8.0))
full_spectrum_active = bool(flux_now >= min_flux_for_full or float(np.clip(energy_fullness, 0.0, 1.0)) >= min_fullness_for_full)
```

**Impact:** `flux_now` is raw `event.spectral_flux` (absolute FFT-derived spectral flux). It's compared against a fixed threshold `min_flux_for_full` (default: 0.15). At low OS volume, raw spectral flux is proportionally smaller, so `flux_now >= 0.15` will fail more often.

**What happens:** `full_spectrum_active` being `False` means `has_kick` is forced `False` (line 1279), which means kick-like beats fall through to `hat_only` → park+bounce (no full motion). **This is a direct volume→motion suppression path.**

**The `energy_fullness` fallback (line 1273)** partially mitigates this because `energy_fullness` IS volume-normalized. But for tracks where `energy_fullness < 0.34` and flux is the only path to `full_spectrum_active`, low volume = no full motion.

**Fix:** Normalize `flux_now` against the rolling P95 of `_recent_flux_values` before comparing to `min_flux_for_full`, the same way flux_boost does it in `compute_radius_bloom_from_sub_bass`. Example:
```python
raw_flux_now = float(np.clip(getattr(event, "spectral_flux", 0.0) or 0.0, 0.0, 8.0))
flux_history = list(self._recent_flux_values)
if len(flux_history) >= 10:
    p95 = float(np.percentile(flux_history, 95))
    flux_now = float(np.clip(raw_flux_now / max(p95, 1e-9), 0.0, 1.0))
else:
    flux_now = raw_flux_now
```

---

## Finding #2: `_populate_rolling_deques` — `_recent_flux_values` stores RAW flux ⚠️ VOLUME-DEPENDENT (affects multiple downstream paths)

**File:** beat_intelligence.py **Line:** 1087

```python
self._recent_flux_values.append(float(getattr(event, "spectral_flux", 0.0) or 0.0))
```

**Impact:** `_recent_flux_values` stores raw absolute `spectral_flux` values. This deque is used in:

1. **Phrase commitment flux-drop detection** (lines 1889–1891) — compares `current_flux_mean` to `self._phrase_flux_baseline * 0.35`. **PARTIALLY DEPENDENT**: uses ratio to self-baseline, so *within* a phrase it's volume-invariant. But phrase entry baseline (line 1879) is set from raw flux mean, and if OS volume changes mid-session the ratio breaks.

2. **Phrase renewal** (lines 1905–1907) — same ratio-to-baseline pattern. **PARTIALLY DEPENDENT** for same reason.

3. **Tempo-unlock hold flux spike/drop cancellation** (lines 683–694) — compares `current_flux` (mean of last 4 raw values) to `baseline * 2.0` (spike) or `baseline * 0.25` (drop). **PARTIALLY DEPENDENT**: ratio-based but baseline is raw.

4. **`_volume_normalized_flux`** (line 262) — uses this deque's P95 for normalization. This is the **correct** usage — it creates volume independence for learning features.

5. **`compute_radius_bloom_from_sub_bass`** flux_boost (line 1374) — uses P95 from this deque. **Already fixed.**

**Root issue:** The deque itself is FINE — storing raw values and computing rolling percentiles from them is the correct approach. The individual consumers that use the raw values as ratios-to-own-baseline are mostly OK. The issue is Finding #1 above which reads `event.spectral_flux` directly without normalizing.

**Verdict:** The deque population is correct. The P95 values ARE stable at low volume because they track the same proportionally-reduced signals. Ratio-based comparisons (phrase, unlock hold) are volume-invariant. No fix needed for the deque itself.

---

## Finding #3: `_passes_overall_amp_fill_gate` — `event.intensity` pre-check ⚠️ MOSTLY VOLUME-INDEPENDENT (but verify)

**File:** beat_intelligence.py **Lines:** 974–979

```python
intensity = float(getattr(event, 'intensity', 0.0) or 0.0)
target = float(getattr(cfg, 'overall_amp_fill_target', 0.5))
tolerance = float(getattr(cfg, 'overall_amp_fill_tolerance', 0.5))

if intensity < (target - tolerance):
    self._fill_pass_consecutive[trigger_kind] = 0
    return False
```

**How `intensity` is computed (audio_engine.py L1459):**
```python
intensity=min(1.0, band_energy / max(0.0001, self.peak_envelope)),
```

**Verdict:** `intensity` is `band_energy / peak_envelope` — a ratio of current energy to recent peak energy. `peak_envelope` tracks with decay, so this is **self-relative and volume-independent**. With default `target=0.5` and `tolerance=0.5`, the threshold is `0.0`, meaning this pre-check only rejects when `intensity < 0.0` — effectively never fires. **No fix needed unless config is non-default.**

---

## Finding #4: `_get_spectrum_fill_ratio` (dBFS mode) — threshold relative to `_dbfs_reference_max` ✅ VOLUME-INDEPENDENT

**File:** beat_intelligence.py **Lines:** 937–944

```python
reference_max = max(self._dbfs_reference_max, 1e-10)
linear_threshold = reference_max * (10.0 ** (dbfs_threshold / 20.0))
filled = float(np.sum(band >= linear_threshold))
```

**Verdict:** The dBFS threshold is applied relative to `_dbfs_reference_max` (the rolling recent-max spectrum magnitude). This is a ratio-based comparison — the reference scales with volume, so the fill ratio is **volume-independent**. No fix needed.

---

## Finding #5: `_journey_start_intensity` — stored but never read ✅ NO IMPACT

**File:** beat_intelligence.py **Line:** 1531

```python
self._journey_start_intensity = float(getattr(event, 'intensity', 0.0) or 0.0)
```

**Verdict:** This value is stored at journey start but is **never read** by any downstream code. It's dead state. No motion impact, no fix needed.

---

## Finding #6: `compute_radius_bloom_from_sub_bass` — `sub_bass_energy` base calculation ✅ VOLUME-INDEPENDENT

**File:** beat_intelligence.py **Lines:** 1362–1370

```python
sub_bass = float(np.clip(self.energies.sub_bass, 0.0, 1.0))
low_mid = float(np.clip(self.energies.low_mid, 0.0, 1.0))

weighted_bass = (sub_bass * 0.70) + (low_mid * 0.30)
bass_fill = float(np.clip(max(sub_bass, weighted_bass), 0.0, 1.0))
bass_power = bass_fill ** 2.0
```

**Verdict:** `self.energies.sub_bass` and `self.energies.low_mid` are the EMA-smoothed outputs of `update_band_energies()` which normalizes against rolling P95 (lines 1101–1121). These are **already volume-independent**. No fix needed.

---

## Finding #7: `update_envelope` and `get_overall_amplitude` — raw RMS envelope for silence detection ⚠️ MOSTLY OK BUT EDGE CASE

**File:** beat_intelligence.py **Lines:** 1124–1130

```python
def update_envelope(self, event: BeatEvent) -> None:
    target = self._event_rms_db(event)
    alpha = self.rms_attack if target >= self.rms_envelope else self.rms_release
    self.rms_envelope += (target - self.rms_envelope) * alpha

def get_overall_amplitude(self, event: BeatEvent) -> float:
    raw_rms_db = self._event_rms_db(event)
```

**And silence detection (lines 1134–1155):**
```python
def update_silence_deadzone_gate(self, overall_amplitude: float, ...):
    ...
    open_threshold = silence_threshold_to_dbfs(open_threshold_raw, default_linear=0.001)
    close_threshold = silence_threshold_to_dbfs(close_threshold_raw, default_linear=0.003)
    ...
    level_db = self._coerce_amplitude_db(overall_amplitude)
```

**Verdict:** Silence detection uses absolute dBFS thresholds (configurable, defaults ~-66dB enter, ~-58dB exit). At very low OS volume, these thresholds may falsely trigger silence even during audible music. **This is intentionally absolute** — silence detection *should* use absolute levels because there genuinely is no signal at those levels. However, if a user has OS volume at 5% but music playing, the silence gate will incorrectly activate and suppress all motion.

**Impact:** False silence → `silence_active=True` → no motion output, fade to park/zero.

**Fix consideration:** The silence thresholds are already configurable. A dynamic silence gate that adapts to the rolling noise floor would be more robust, but this is a design-level decision. The current behavior is intentional for actual silence detection. Users with very low OS volumes should adjust `silence_threshold` config.

---

## Finding #8: `_build_runtime_feature_values` — band energies NOT volume-calibrated for learning ⚠️ PARTIALLY DEPENDENT

**File:** beat_intelligence.py **Lines:** 419–426

```python
sub = float(self.energies.sub_bass)
low = float(self.energies.low_mid)
mid = float(self.energies.mid)
high = float(self.energies.high)

# Derived features
low_high_ratio = sub / max(high, 1e-10)
energy_norm = self._dbfs_to_unit(energy_mean)
```

**The band energies used here:** `self.energies.*` are the P95-normalized values (from `update_band_energies`). These are **volume-independent** (0-1 range).

**However, the feature dict sends them as-is to the learning model:**
```python
"sub_bass_energy": sub,
"low_mid_energy": low,
"mid_energy": mid,
"high_energy": high,
```

**Verdict:** The band energies passed to the learning model are **already volume-normalized** (P95 normalization in `update_band_energies`). The `rms` and `spectral_flux` features are separately volume-calibrated (lines 393–394). **No fix needed.**

---

## Finding #9: Phrase commitment — flux baseline set from raw flux values ⚠️ PARTIALLY DEPENDENT

**File:** beat_intelligence.py **Line:** 1879

```python
self._phrase_flux_baseline = float(np.mean(recent_flux)) if len(recent_flux) >= 4 else 0.0
```

And comparisons at lines 1890-1891, 1907:
```python
if (self._phrase_flux_baseline > 1e-6
        and current_flux_mean < (self._phrase_flux_baseline * self._phrase_flux_drop_ratio)):
```

**Verdict:** This is **ratio-to-self** — the baseline and current values are both raw flux from the same volume context. As long as OS volume doesn't change *during* a phrase commitment, these comparisons are volume-independent (a 35% drop is a 35% drop regardless of absolute level). **No fix needed** unless there's a mid-session volume change concern.

---

## Finding #10: Tempo-unlock hold — flux baseline set from raw flux values ⚠️ PARTIALLY DEPENDENT (same pattern as #9)

**File:** beat_intelligence.py **Line:** 718

```python
self._tempo_unlock_hold_flux_baseline = float(np.mean(recent_flux)) if len(recent_flux) >= 4 else 0.1
```

And comparisons at lines 687-691:
```python
if current_flux > baseline * self._tempo_unlock_hold_flux_spike_ratio:
    self._tempo_unlock_hold_active = False
if current_flux < baseline * self._tempo_unlock_hold_flux_drop_ratio:
    self._tempo_unlock_hold_active = False
```

**Verdict:** Same ratio-to-self pattern. **Volume-independent** within a hold episode. No fix needed.

---

## Finding #11: `event_detector.py` — all scores use volume-normalized features ✅ VOLUME-INDEPENDENT

**File:** audio_modules/event_detector.py **Lines:** 133–220

The `_score_single_bus` method operates on `FeatureFrame` fields (`flux_norm`, `energy_delta`, etc.) which are already normalized in `_build_shadow_feature_frame` against rolling P95 references. The z-score computation (line 153) uses its own history mean/std. All bus energies come from `FeatureFrame` which contains ratio-normalized values.

**Verdict:** The entire event detector pipeline is **volume-independent**. No fix needed.

---

## Finding #12: `feature_extractors.py` — `compute_multiband_energies` returns raw energy × gain ⚠️ VOLUME-DEPENDENT AT SOURCE

**File:** audio_modules/feature_extractors.py **Lines:** 151–169

```python
energies[name] = float(np.sqrt(np.mean(band_slice ** 2))) * float(gain)
```

**Verdict:** This returns raw RMS-of-spectrum values scaled by gain. These are the raw values that feed into `audio_engine._band_energies`, which are then normalized by `beat_intelligence.update_band_energies()` against P95. The raw values here are **intentionally volume-dependent** — normalization happens downstream. **No fix needed.**

---

## Finding #13: `stroke_mapper.py` — does NOT read raw audio values ✅ CLEAN

**File:** stroke_mapper.py (all 1376 lines)

Stroke mapper only reads from `BeatDecision` fields: `radius_bloom`, `energy_fullness`, `silence_active`, `silence_fade`, `post_silence_ramp`, `trigger_kind`, `interval_beats`, `park_bounce_only`, `park_bounce_gain`, `learning.*`, `session_intensity`, `journey_completion`. 

None of these are raw audio values — they're all processed outputs from `BeatIntelligence`. `energy_fullness` and `radius_bloom` are already P95-normalized upstream. The only audio-adjacent interaction is `_compute_bass_jitter_offsets` which reads `event.frequency` (just the dominant frequency in Hz, not amplitude).

**Verdict:** Stroke mapper is **fully volume-independent**. No fix needed.

---

## Summary of Actionable Findings

| # | Severity | File | Line(s) | Issue | Fix |
|---|----------|------|---------|-------|-----|
| **1** | **HIGH** | beat_intelligence.py | 1272-1273 | `flux_now` compared to fixed threshold `min_flux_for_full` using raw `event.spectral_flux` | Normalize against rolling P95 of `_recent_flux_values` |
| 7 | LOW | beat_intelligence.py | 1134-1155 | Silence gate uses absolute dBFS thresholds; false positive at very low OS volume | Consider adaptive noise floor, or document config workaround |
| 5 | NONE | beat_intelligence.py | 1531 | `_journey_start_intensity` stored but never used | Dead code, remove if desired |

**Finding #1 is the smoking gun** for the user's reported issue. At low OS volume:
1. Raw `spectral_flux` is proportionally smaller
2. `flux_now >= 0.15` (default `min_flux_for_full`) fails
3. `full_spectrum_active` = False
4. `has_kick` = False (even if kick_conf is high)
5. Profile falls through to `"hat_only"` → park+bounce only
6. Motion is suppressed despite music playing

Everything else (band energies, radius_bloom bass path, amp_gate, fill gate dBFS mode, learning features) is already properly volume-normalized.
