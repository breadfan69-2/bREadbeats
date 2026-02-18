# BeatIntelligence Port Audit v2 — 2026-02-17

**Source of truth:** commit `ab2b35a` (`stroke_mapper.py` at ~4,257 lines)  
**Refactor commit:** `24ae407` ("Refactor stroke mapper to decision-orbital adapter")  
**Current state:** `beat_intelligence.py` at 276 lines, `stroke_mapper.py` at ~195 lines  
**Supersedes:** `BEAT_INTELLIGENCE_PORT_AUDIT_2026-02-17.md` (v1 — correct but incomplete)

---

## What changed from v1

v1 was accurate on all 16 items. This v2 adds:

- **5 additional missing items** found by method-level diff against `ab2b35a`
- **Exact `__init__` state** for every deque and instance variable that must be restored
- **Full source snippets** for every missing method (v1 had some but not all)
- **`_update_stroke_readiness` detail** — the full traffic-light + grace-period logic is far more complex than `_tempo_ready_for_motion` and was not adequately covered
- **`_update_learning_adapter`** — the runtime teaching layer that reads model predictions and blends them into motion parameters; entirely absent from v1
- **`_apply_fade` + post-silence ramp** — v1 mentioned post-silence ramp but didn't document the fade intensity tracker that drives it
- **Silence fade-out tracker** (`_fade_intensity`, `_consecutive_silent_count`, `_silence_reset_armed`, tempo tracking reset on prolonged silence) — the old silence system was a two-part mechanism (deadzone gate + fade tracker), and only one part was ported
- **Flux drop guard** inline in `process_beat()` — v1 mentioned it but didn't show the full implementation
- **Auto-fill adaptation** — the adaptive threshold controller that adjusts `fill_required` per phase based on pass rate; 6 config fields and ~60 lines of logic

---

## Summary Table — All Missing Components

| # | Component | Severity | Effort | v1 covered? |
|---|-----------|----------|--------|-------------|
| 1 | Rolling history deques (5 deques) | CRITICAL | Low | ✓ |
| 2 | FluxTracker (250ms deque + rise factor) | CRITICAL | Low | ✓ |
| 3 | `_has_recent_beats` + beat hierarchy guards | CRITICAL | Low | ✓ |
| 4 | No-beat timeout → park decay | CRITICAL | Low | ✓ |
| 5 | `_is_low_band_full_enough` | IMPORTANT | Medium | ✓ |
| 6 | `_passes_dual_band_db_gate` | IMPORTANT | Medium | ✓ |
| 7 | `_is_mid_trigger_blocked` | IMPORTANT | Low | ✓ |
| 8 | `_get_spectrum_fill_ratio` + `_passes_overall_amp_fill_gate` | IMPORTANT | High | ✓ |
| 9 | `_update_motion_mode` (FULL_STROKE/CREEP_MICRO) | IMPORTANT | Medium | ✓ |
| 10 | `_build_runtime_feature_values` | LEARNING | Low | ✓ |
| 11 | `_predict_learning_targets` (incl. `cadence_rule`) | LEARNING | Medium | ✓ |
| 12 | `_try_load_learning_model` (complete, incl. `cadence_rule` field) | LEARNING | Low | ✓ |
| 13 | `_cap_bpm_to_last_locked` + `_stabilize_unlocked_bpm` | GRAY | Low | ✓ |
| 14 | Post-silence volume ramp | GRAY | Low | ✓ |
| 15 | `_update_bass_jitter_drive` (stub, leave disconnected) | GRAY | Low | ✓ |
| 16 | Flux drop → creep fallback guard | GRAY | Low | ✓ |
| **17** | **`_update_stroke_readiness` (full traffic-light logic + grace period)** | **CRITICAL** | **Medium** | **NO** |
| **18** | **`_update_learning_adapter` (runtime teaching blend)** | **LEARNING** | **Medium** | **NO** |
| **19** | **Silence fade-out tracker (`_fade_intensity` + tempo reset on prolonged silence)** | **IMPORTANT** | **Low** | **NO** |
| **20** | **Auto-fill adaptation (`_update_auto_fill_required` + per-phase EMA offset)** | **IMPORTANT** | **Medium** | **NO** |
| **21** | **`_get_high_band_activity`, `_get_mid_bass_activity`, `_get_high_band_presence_status`, `_get_high_band_pattern_status`** | **IMPORTANT** | **Medium** | **Partial** |

---

## 1. CRITICAL — Rolling History Deques

### What's missing

All 5 deques from the old `__init__`:

```python
self._recent_flux_values: deque = deque(maxlen=60)           # ~1s flux history for center-reset guard
self._recent_low_band_values: deque = deque(maxlen=60)       # ~1s low-band for beat gating
self._recent_high_band_values: deque = deque(maxlen=60)      # ~1s high-band for treble gate
self._recent_mid_bass_values: deque = deque(maxlen=60)       # ~1s 200–400Hz support
self._recent_high_band_beat_hits: deque = deque(maxlen=16)   # per-beat treble hit boolean
```

### How they were populated

In `process_beat()`, after calling the continuous trackers:

```python
self._recent_flux_values.append(event.spectral_flux)
low_band_activity = self._get_low_band_activity(event)
high_band_activity = self._get_high_band_activity(event)
mid_bass_activity = self._get_mid_bass_activity(event)
self._recent_low_band_values.append(low_band_activity)
self._recent_high_band_values.append(high_band_activity)
self._recent_mid_bass_values.append(mid_bass_activity)
```

And for the beat-hit deque, inside the FULL_STROKE beat gate cascade:

```python
high_beat_hit = (
    ('high' in fired_bands)
    or (include_mid_high_gate and ('mid' in fired_bands))
    or (beat_band == 'high')
    or (include_mid_high_gate and beat_band == 'mid')
)
self._recent_high_band_beat_hits.append(bool(high_beat_hit))
```

### Where to put them

Add to `BeatIntelligence.__init__`. Populate in `build_decision()` after band energies are updated.

---

## 2. CRITICAL — FluxTracker

### Old source (lines 758–771)

```python
def _update_flux_history(self, event: BeatEvent) -> None:
    now = event.timestamp
    self._flux_history.append((now, event.spectral_flux))
    cutoff = now - self._flux_rise_window_ms / 1000.0
    while self._flux_history and self._flux_history[0][0] < cutoff:
        self._flux_history.popleft()

def _get_flux_rise_factor(self) -> float:
    if len(self._flux_history) < 2:
        return 0.0
    oldest_flux = self._flux_history[0][1]
    newest_flux = self._flux_history[-1][1]
    rise = max(0.0, newest_flux - oldest_flux)
    return min(1.0, rise / 0.1)
```

### Init state

```python
self._flux_history: deque = deque()
self._flux_rise_window_ms: float = 250.0
self._flux_stroke_factor: float = 1.0
```

### Effect when missing

`compute_radius_bloom_from_sub_bass()` uses raw `event.spectral_flux` against a threshold — no urgency ramp. A fast-rising spectrum and a flat one get the same radius. Bloom is underresponsive to percussive hits.

### Integration point

The flux rise factor should modulate the bloom radius computation. In the old code, `_flux_stroke_factor` was:

```python
flux_ratio = event.spectral_flux / max(cfg.flux_threshold, 0.001)
flux_ratio = np.clip(flux_ratio, 0.2, 3.0)
base_factor = 0.5 + (flux_ratio / 3.0)
self._flux_stroke_factor = 1.0 + (base_factor - 1.0) * scaling_weight
```

---

## 3. CRITICAL — `_has_recent_beats` + Beat Hierarchy Guards

### Old source (lines 791–810)

```python
def _has_recent_beats(self, now=None, window_s=0.9) -> bool:
    current = now if now is not None else time.perf_counter()
    beat_recent = (
        self._last_any_beat_time > 0
        and (current - self._last_any_beat_time) <= window_s
    )
    reset_hold_active = current < self._tempo_reset_motion_hold_until
    return bool(beat_recent or reset_hold_active)

def _arm_tempo_reset_motion_hold(self, now: float) -> None:
    self._last_any_beat_time = now
    self._tempo_reset_motion_hold_until = max(
        self._tempo_reset_motion_hold_until,
        now + self._tempo_reset_motion_hold_s,
    )
```

### Init state

```python
self._last_any_beat_time: float = 0.0
self._last_confirmed_beat_time: float = 0.0
self._last_downbeat_call_time: float = 0.0
self._last_beat_or_downbeat_call_time: float = 0.0
self._last_downbeat_stroke_time: float = 0.0
self._downbeat_chain_active: bool = False
self._downbeat_chain_last_time: float = 0.0
self._tempo_reset_motion_hold_s: float = 1.8
self._tempo_reset_motion_hold_until: float = 0.0
```

### Gate cascade usage

In the old `process_beat()`, the hierarchy was enforced:

1. **Syncopation requires recent beat/downbeat call** within `prereq_window_s` (~2.5 beat periods)
2. **Syncopation requires recent downbeat stroke** within `prereq_window_s`
3. **Beat requires recent downbeat stroke** before firing, checked with `has_recent_downbeat_stroke`
4. **`_arm_tempo_reset_motion_hold`** prevents false-triggering for 1.8s after `tempo_reset` events
5. **`_has_recent_beats`** guards creep edge continuation and arc launch phase selection

### What current code does wrong

`classify_trigger()` fires syncopation on any `is_syncopated == True` event with **no history check**. There is no downbeat-before-beat requirement. No `tempo_reset` hold.

---

## 4. CRITICAL — No-Beat Timeout → Park Decay

### Old logic (in `process_beat()`)

```python
if (self._last_any_beat_time > 0
        and (now - self._last_any_beat_time) > self._no_beat_timeout_s
        and self._trajectory is not None):
    # Check center-reset flux guard first
    hold_center_reset = False
    if bool(getattr(beat_cfg, 'center_jitter_flux_guard_enabled', False)):
        hold_center_reset, recent_avg, recent_delta = self._is_center_reset_flux_guard_active()

    if not hold_center_reset:
        self._pending_arc_event = None
        self._last_downbeat_stroke_time = 0.0
        self._downbeat_chain_active = False
        # Generate park return arc or hard-reset to park
        if not getattr(self._trajectory, 'is_park_return', False):
            transitioned = self._generate_park_return_arc()
            if not transitioned:
                self._trajectory = None
                self.state.creep_reset_active = True
```

### Init state

```python
self._no_beat_timeout_s: float = 2.0
```

### How to port

Track `_last_any_beat_time` in `build_decision()`. When `now - _last_any_beat_time > 2.0s` and a journey is active, force decay to park. The center-reset flux guard (§2's deque) can optionally hold this.

---

## 5. IMPORTANT — `_is_low_band_full_enough`

### Old source (lines 1540–1604)

Full method evaluates:
- 18-frame window from `_recent_low_band_values`
- Mean ≥ configurable threshold (default 0.20), relaxed by `downbeat_low_band_relax` for downbeats
- **Occupancy**: fraction of frames above floor (0.70 × threshold) must be ≥ `low_band_fullness_occupancy_threshold` (default 0.62)
- **Low/high ratio**: `mean_low / mean_high ≥ low_band_to_high_ratio_min` (default 0.58) — prevents mid/treble-only content from passing
- **Mid-bass support** fallback: when treble isn't full enough, checks 200–400 Hz occupancy via `_recent_mid_bass_values`
- If deque has < 8 frames, falls back to single-frame `_get_low_band_activity()` check

### What current code has

`_strict_bass_motion_allowed()` checks only `event.beat_band` and `event.fired_bands` — single-frame event fields, not a history gate.

---

## 6. IMPORTANT — `_passes_dual_band_db_gate`

### Old source (lines 1606–1668)

- Requires sub-bass energy ≥ `dual_band_sub_bass_db_min` (default −15 dB) AND high energy ≥ `dual_band_high_db_min` (default −30 dB)
- **Event frequency fallback**: if band energy is near zero but event frequency is in the right range, infers energy from peak
- **High-tip fullness sub-gate**: checks 3.5–16 kHz band using `_recent_high_band_values` deque
  - Mean above `high_tip_db_min` linear threshold
  - Occupancy above `high_tip_occupancy_threshold`
  - OR dominant tip frequency detected in current event
- Controlled by `dual_band_db_gate_enabled` (default False) and `high_tip_fullness_enabled` (default True)
- **Learning relax**: when `_learning_relax_phase1_gates` is True, gate always passes

---

## 7. IMPORTANT — `_is_mid_trigger_blocked`

### Old source (lines 1669–1683)

```python
def _is_mid_trigger_blocked(self, event: BeatEvent) -> bool:
    if self._learning_enabled and self._learning_relax_phase1_gates:
        return False
    cfg = self.config.stroke
    if not bool(getattr(cfg, 'block_mid_trigger_range_enabled', False)):
        return False
    low_hz = float(getattr(cfg, 'block_mid_trigger_low_hz', 100.0) or 100.0)
    high_hz = float(getattr(cfg, 'block_mid_trigger_high_hz', 2000.0) or 2000.0)
    if high_hz <= low_hz:
        high_hz = low_hz + 1.0
    freq = float(getattr(event, 'frequency', 0.0) or 0.0)
    return bool(low_hz <= freq <= high_hz)
```

Simple but essential. Prevents vocal-range and guitar-fundamental beats from triggering strokes.

---

## 8. IMPORTANT — `_get_spectrum_fill_ratio` + `_passes_overall_amp_fill_gate`

### `_get_spectrum_fill_ratio` (lines 1770–1806)

Reads live FFT from `audio_engine.get_spectrum()`, normalizes to peak, counts bins above threshold in a per-phase configurable bin window:

```python
spectrum = self.audio_engine.get_spectrum()
magnitudes = np.abs(np.asarray(spectrum, dtype=float))
peak = float(np.max(magnitudes))
# Per-phase bin windows from config:
low_bin = config.stroke.{phase}_fill_bin_low
high_bin = config.stroke.{phase}_fill_bin_high
# Slice, normalize, count active bins above threshold
norm = magnitudes[low_bin:high_bin+1] / peak
active_bins = norm >= active_floor  # default 0.02
active = norm[active_bins]
return float(np.sum(active >= threshold) / max(1, active.size))
```

### `_passes_overall_amp_fill_gate` (lines 1808–1842)

- Gate enabled by `overall_amp_fill_gate_enabled`
- Checks `event.intensity >= target - tolerance`
- Computes `fill_ratio` via `_get_spectrum_fill_ratio()`
- Compares against `_get_overall_amp_fill_required(phase)` (which includes auto-adapt offset)
- **Near-silence detection**: when flux and energy are close to silence thresholds, disables adaptive EMA updates to prevent drift during fadeouts
- Calls `_update_auto_fill_required()` and `_maybe_log_auto_fill_status()`

---

## 9. IMPORTANT — `_update_motion_mode` (FULL_STROKE / CREEP_MICRO)

### Old source (lines 1368–1406)

```python
def _update_motion_mode(self) -> None:
    now = time.time()
    cfg = self.config.stroke
    dwell_bias = float(getattr(cfg, 'full_stroke_dwell_bias', 0.0) or 0.0)
    gate_high = float(cfg.amplitude_gate_high) - dwell_bias
    gate_low = float(cfg.amplitude_gate_low) + dwell_bias
    gate_high = float(np.clip(gate_high, 0.005, 0.95))
    gate_low = float(np.clip(gate_low, 0.001, 0.94))
    if gate_low >= gate_high:
        midpoint = (gate_low + gate_high) * 0.5
        gate_high = min(0.95, midpoint + 0.001)
        gate_low = max(0.001, midpoint - 0.001)
    # 500ms minimum dwell
    if now - self._mode_switch_time < 0.5:
        return
    old = self._motion_mode
    if self._motion_mode == MotionMode.CREEP_MICRO:
        if self._rms_envelope > gate_high:
            self._motion_mode = MotionMode.FULL_STROKE
            self._mode_switch_time = now
    else:
        if self._rms_envelope < gate_low:
            self._motion_mode = MotionMode.CREEP_MICRO
            self._mode_switch_time = now
```

### Init state

```python
self._motion_mode: str = MotionMode.CREEP_MICRO  # start quiet
self._mode_switch_time: float = 0.0
```

### What current code does

No motion mode concept at all. `interval_beats_for_trigger()` maps trigger kind directly to beats (1/2/4/8) without amplitude level.

---

## 10–12. LEARNING PIPELINE

### 10. `_build_runtime_feature_values` (line 1083)

Maps a `BeatEvent` to the 13 features the fitted model expects:

```python
{
    'rms':                   from event.beat_features['energy_mean'] or event.peak_energy
    'log_energy':            log10(rms + eps)
    'spectral_flux':         from event.beat_features['flux_mean'] or event.spectral_flux
    'flux_delta':            flux_peak - flux_mean
    'sub_bass_energy':       self._sub_bass_energy
    'low_mid_energy':        self._low_mid_energy
    'mid_energy':            self._mid_energy
    'high_energy':           self._high_energy
    'low_high_ratio':        (sub_bass + low_mid + eps) / (high + eps)
    'spectral_centroid_hz':  event.frequency
    'spectral_bandwidth_hz': event.beat_features['freq_delta']
    'spectral_rolloff_hz':   centroid + 0.5 * bandwidth
    'spectral_flatness':     0.35 + 0.50 * (1 - energy_norm)
}
```

### 11. `_predict_learning_targets` (line 1120)

Full inference:
1. Build feature dict
2. Z-score normalize each feature against `_learning_norm_mean` / `_learning_norm_std`
3. For each target: `value = intercept + sum(coef[f] * normalized[f] for f in features)`
4. Clamp: `arc_size` 0–1, `arc_duration_frac` 0.1–4.0, `jitter_mix` 0–1, `creep_mix` 0–1, `gate_strictness` 0–1, `burst_prob` 0–1
5. **`cadence_rule`**: derives `beats_between_strokes` from weighted RMS + flux against `quiet_threshold` / `mid_threshold`

The `cadence_rule.beats_between_strokes` is a **third pacing dimension** absent from current code.

### 12. `_try_load_learning_model` (line 1050)

Old loader must extract AND store:
- `feature_columns`
- `normalization.mean` / `normalization.std`
- `models` (per-target intercept + coefficients)
- **`cadence_rule`** ← missing from the current `_try_load_learning_model` in `stroke_mapper.py` and from the BAKED guide's Step 6

Init state:

```python
self._learning_model_loaded: bool = False
self._learning_model_path: str = ""
self._learning_model: dict = {}
self._learning_norm_mean: dict[str, float] = {}
self._learning_norm_std: dict[str, float] = {}
self._learning_cadence_rule: dict = {}
self._learning_feature_columns: list[str] = []
```

---

## 13. GRAY — `_cap_bpm_to_last_locked` + `_stabilize_unlocked_bpm`

### Old source (lines 1450–1480)

```python
def _cap_bpm_to_last_locked(self, bpm: float) -> float:
    if bpm <= 0.0:
        return 0.0
    if self._last_locked_bpm > 0.0:
        return float(min(bpm, self._last_locked_bpm))
    return float(bpm)

def _stabilize_unlocked_bpm(self, bpm: float, event=None) -> float:
    if bpm <= 0.0:
        return 0.0
    if bool(getattr(event, 'tempo_locked', False)):
        self._last_unlocked_motion_bpm = 0.0
        return float(bpm)
    no_lock_ceiling = 140.0
    jump_ratio = float(np.clip(
        getattr(beat_cfg, 'aggressive_snap_max_bpm_jump_ratio', 0.12), 0.03, 0.35))
    clamped = float(min(bpm, no_lock_ceiling))
    if self._last_unlocked_motion_bpm > 0.0:
        allowed_up = self._last_unlocked_motion_bpm * (1.0 + jump_ratio)
        clamped = float(min(clamped, allowed_up))
    self._last_unlocked_motion_bpm = clamped
    return clamped
```

### Init state

```python
self._last_locked_bpm: float = 0.0
self._last_unlocked_motion_bpm: float = 0.0
```

### Current code

`effective_bpm()` clips to [40, 240] but has no memory of last locked value and no jump-ratio limiter.

---

## 14. GRAY — Post-Silence Volume Ramp

### Old logic (in `_apply_fade`, lines 2689–2717)

When silence gate closes (audio resumes), the old code:
1. Detected `_was_silent` → `_post_silence_ramp_active = True`
2. Started volume at `(1 - post_silence_vol_reduction)` and linearly ramped to `1.0` over `post_silence_ramp_seconds`

```python
if self._post_silence_ramp_active:
    elapsed = time.perf_counter() - self._post_silence_ramp_start
    ramp_dur = max(0.5, cfg.post_silence_ramp_seconds)
    if elapsed >= ramp_dur:
        self._post_silence_ramp_active = False
    else:
        reduction = cfg.post_silence_vol_reduction
        ramp_mult = (1.0 - reduction) + reduction * (elapsed / ramp_dur)
        cmd.volume *= ramp_mult
```

### Init state

```python
self._post_silence_ramp_active: bool = False
self._post_silence_ramp_start: float = 0.0
self._was_silent: bool = False
```

### Current code

`build_decision` sets `silence_active=True` → StrokeMapper sets `volume=0`. On re-open, immediately uses `get_volume()`. No ramp.

---

## 15. GRAY — `_update_bass_jitter_drive` (stub)

### Old source (lines 1931–1968)

Maps bass frequency 30–220 Hz to jitter speed multiplier (depth 0.03–0.075). Smoothed with EMA.

### Init state

```python
self._bass_jitter_speed_mult: float = 1.0
self._bass_jitter_attack: float = 0.25
self._bass_jitter_release: float = 0.06
```

Port as stub, leave disconnected until T-Code aux output is wired.

---

## 16. GRAY — Flux Drop → Creep Fallback Guard

### Old logic (inline in `process_beat()`)

```python
if len(self._recent_flux_values) >= 30:
    if bool(getattr(cfg, 'low_band_drop_guard_enabled', True)):
        recent_avg = sum(list(self._recent_low_band_values)[-15:]) / 15.0
        older_avg = sum(list(self._recent_low_band_values)[:15]) / 15.0
        flux_drop_ratio = cfg.flux_drop_ratio  # default 0.25
        min_high_band = float(getattr(cfg, 'low_band_activity_threshold', 0.20))
        if older_avg >= min_high_band and recent_avg < older_avg * flux_drop_ratio:
            if not recent_beats_active:
                if self._motion_mode == MotionMode.FULL_STROKE:
                    if not traj_active:
                        self._motion_mode = MotionMode.CREEP_MICRO
```

When low-band energy drops sharply (to <25% of recent average), force back to creep. Requires `_recent_low_band_values` deque.

---

## 17. **NEW** CRITICAL — `_update_stroke_readiness` (Full Traffic-Light Logic)

**Not covered in v1.**

### What current code has

`_tempo_ready_for_motion()` in `BeatIntelligence`: checks only `event.tempo_locked` or `acf_confidence >= relaxed_threshold`. This is a **simplified subset**.

### What the old code had (lines 1230–1367)

The full `_update_stroke_readiness()` was a **138-line method** that implemented:

1. **Traffic light evaluation** from `audio_engine.get_metric_states()`:
   - `traffic_green` = all metrics SETTLED
   - `traffic_yellow` = some but not all SETTLED
   - `traffic_was_green` tracked with 3s expiry
   
2. **Metronome confidence levels**:
   - `metro_green` = `acf_confidence >= 0.25` and BPM > 0
   - `metro_yellow` = `acf_confidence >= 0.05` and BPM > 0
   - `metro_relaxed` = `acf_confidence >= teaching_metronome_relaxed_confidence` and BPM > 0
   
3. **Multiple readiness paths**:
   - Both green
   - One green + one yellow
   - Both yellow (only if previously had lights, or beat/downbeat confirms)
   - Recovery: traffic recently green + metronome yellow/green
   - Stable: metronome green > 2s → override any traffic state
   - Metronome-first: `metronome_ready` alone when teaching rules active

4. **Grace period** (1300ms default from `teaching_stroke_ready_grace_ms`): when conditions drop, strokes continue for grace period before reverting
   
5. **`_stroke_gate_block_streak`** counter + `_stroke_finish_beats` allowance

### Init state

```python
self._stroke_ready: bool = False
self._stroke_ready_lost_time: float = 0.0
self._stroke_grace_ms: float = 450.0  # up to 1300.0
self._stroke_gate_block_streak: int = 0
self._stroke_finish_beats: int = 1
self._traffic_was_green: bool = False
self._traffic_left_green_time: float = 0.0
self._metro_green_since: float = 0.0
self._prev_had_any_light: bool = False
self._ignore_traffic_lights: bool = False (from config)
self._metronome_relaxed_confidence: float = 0.14 (from config)
```

### Impact

The current `_tempo_ready_for_motion` is a boolean check. The old system was a **stateful evaluator** with hysteresis and grace periods. Without it, strokes cut out instantly on brief confidence dips and never get the traffic-light boost that helps with metric settling.

---

## 18. **NEW** LEARNING — `_update_learning_adapter` (Runtime Teaching Blend)

**Not covered in v1.**

### Old source (lines 923–1020)

This was the **bridge method** between model inference and motion parameters. Called once per beat when `_learning_enabled` and `event.is_beat`:

```python
def _update_learning_adapter(self, event: BeatEvent) -> None:
    if not self._learning_enabled or not getattr(event, 'is_beat', False):
        return
    # Skip in SIMPLE_CIRCLE unless apply_in_circle_mode
    if config.stroke.mode == StrokeMode.SIMPLE_CIRCLE and not self._learning_apply_in_circle_mode:
        # Reset all learned fields to neutral
        return

    features = getattr(event, 'beat_features', None) or {}
    confidence = features.get('confidence', event.acf_confidence)
    if confidence < self._learning_min_confidence:
        return

    if self._learning_model_loaded:
        prediction = self._predict_learning_targets(event)
        if prediction:
            cadence_divisor = prediction['beats_between_strokes']
            gate_strictness = prediction['gate_strictness'] * self._learning_no_motion_bias
            arc_size = prediction['arc_size']
            arc_duration = prediction['arc_duration_frac']
            burst_prob = prediction['burst_prob']
            jitter_mix = prediction['jitter_mix']

            # Gate strictness → force higher cadence divisor
            if gate_strictness > 0.92: cadence_divisor = max(cadence_divisor, 8)
            elif gate_strictness > 0.78: cadence_divisor = max(cadence_divisor, 4)

            target_divisor = cadence_divisor
            target_radius_mult = 0.72 + 0.58 * arc_size   # 0.70–1.30
            target_lead_ms = (0.35 - gate_strictness) * 10.0 + (1.0 - arc_duration) * 2.0
            burst_drive = (0.65 * burst_prob) + (0.35 * jitter_mix)
            target_sync_size = 1.0 + 0.30 * burst_drive    # 1.0–1.35
            target_sync_speed = 1.0 + 0.24 * burst_drive   # 1.0–1.25
    else:
        # Heuristic fallback from raw beat features
        # (energy_norm, flux_norm, offbeat_score → divisor, radius, lead)
        ...

    # EMA blend into learned fields at _learning_strength rate
    strength = self._learning_strength
    self._learned_divisor_hint = round(divisor * strength + ...)
    self._learned_radius_mult = ... * strength + (1-strength) * prev
    self._learned_lead_ms = ... * strength + (1-strength) * prev
    self._learned_sync_size_mult = ...
    self._learned_sync_speed_mult = ...
```

### Outputs used elsewhere

- `_learned_divisor_hint` → modifies beats-per-stroke in cadence selection
- `_learned_radius_mult` → scales arc radius
- `_learned_lead_ms` → adjusts predictive arc timing in `_get_effective_lead_seconds()`
- `_learned_sync_size_mult` → scales syncopation arc size
- `_learned_sync_speed_mult` → scales syncopation arc speed

### Init state

```python
self._learned_divisor_hint: int = 1
self._learned_radius_mult: float = 1.0
self._learned_lead_ms: float = 0.0
self._learned_sync_size_mult: float = 1.0
self._learned_sync_speed_mult: float = 1.0
self._edge_follow_radius: float = 0.85
self._learning_strength: float = 0.55
self._learning_min_confidence: float = 0.12
self._learning_no_motion_bias: float = 1.0
```

---

## 19. **NEW** IMPORTANT — Silence Fade-Out Tracker

**Not covered in v1.** v1 covered the silence deadzone gate (which IS ported) but not the **separate fade intensity tracker** that gradually dims output during silence and triggers tempo reset.

### Old logic (in `process_beat()`, inline)

```python
quiet_flux_thresh = cfg.flux_threshold * cfg.silence_flux_multiplier
quiet_energy_thresh = beat_cfg.peak_floor * cfg.silence_energy_multiplier
fade_duration = 2.0
silence_reset_threshold = beat_cfg.silence_reset_ms / 1000.0
consecutive_silent_required = 10

is_truly_silent = (event.spectral_flux < quiet_flux_thresh
                   and event.peak_energy < quiet_energy_thresh)
if is_truly_silent:
    self._consecutive_silent_count += 1
    if self._consecutive_silent_count >= consecutive_silent_required:
        if self._fade_intensity > 0.0:
            if self._last_quiet_time == 0.0:
                self._last_quiet_time = now
            elapsed = now - self._last_quiet_time
            self._fade_intensity = max(0.0, 1.0 - (elapsed / fade_duration))
            # After silence_reset_ms, reset tempo tracking
            if self.audio_engine and elapsed > silence_reset_threshold and self._silence_reset_armed:
                self.audio_engine.reset_tempo_tracking()
                self._silence_reset_armed = False
else:
    self._consecutive_silent_count = 0
    self._silence_reset_armed = True
    # Detect silence → sound transition: start post-silence ramp
    if self._was_silent and self._fade_intensity < 0.5:
        self._post_silence_ramp_active = True
        self._post_silence_ramp_start = now
        self._was_silent = False
    self._fade_intensity = min(1.0, self._fade_intensity + 0.1)
    self._last_quiet_time = 0.0
```

### Init state

```python
self._fade_intensity: float = 1.0
self._last_quiet_time: float = 0.0
self._consecutive_silent_count: int = 0
self._silence_reset_armed: bool = True
```

### Difference from silence deadzone gate

The deadzone gate is a **boolean** open/close with hysteresis. The fade tracker is a **continuous 0–1 multiplier** that gradually dims volume/intensity over 2 seconds AND triggers `audio_engine.reset_tempo_tracking()` after `silence_reset_ms`. These are complementary — both should exist.

### Current code

Only the deadzone gate exists. No fade intensity multiplier, no tempo reset on prolonged silence, no consecutive-frame counter. The `_apply_fade` method that consumed `_fade_intensity` is also gone.

---

## 20. **NEW** IMPORTANT — Auto-Fill Adaptation

**Not covered in v1.**

### What it does

An adaptive controller that raises/lowers the `fill_required` threshold per trigger phase (beat / downbeat / syncopation) to maintain a target pass rate (~58%). Without it, the fill gate either passes everything (too permissive) or blocks everything (too strict) depending on the source material.

### Old source (lines 1705–1760)

```python
def _update_auto_fill_required(self, phase: str, fill_pass: bool) -> None:
    if not self._auto_fill_enabled:
        return
    phase_state = self._auto_fill_state.get(phase)
    pass_value = 1.0 if fill_pass else 0.0
    ema_now = ema_prev + (pass_value - ema_prev) * self._auto_fill_ema_alpha
    phase_state['ema'] = ema_now

    error = phase_state['ema'] - self._auto_fill_target_pass_rate
    if abs(error) <= self._auto_fill_deadband:
        return
    step = self._auto_fill_step * min(2.0, abs(error) / max(deadband, 1e-6))
    if error > 0.0:  # passing too often → raise threshold
        offset += step
    else:             # failing too often → lower threshold
        offset -= step
    phase_state['offset'] = np.clip(offset, -max_offset, max_offset)

def _get_overall_amp_fill_required(self, phase: str) -> float:
    base_required = self._get_overall_amp_fill_required_base(phase)
    if not self._auto_fill_enabled:
        return base_required
    offset = self._auto_fill_state[phase]['offset']
    return np.clip(base_required + offset, min_required, max_required)
```

### Init state

```python
self._auto_fill_enabled: bool = True  # from config
self._auto_fill_target_pass_rate: float = 0.58
self._auto_fill_ema_alpha: float = 0.12
self._auto_fill_deadband: float = 0.06
self._auto_fill_step: float = 0.02
self._auto_fill_max_offset: float = 0.35
self._auto_fill_min_required: float = 0.05
self._auto_fill_max_required: float = 0.98
self._auto_fill_state = {
    'beat': {'ema': 0.58, 'offset': 0.0},
    'downbeat': {'ema': 0.58, 'offset': 0.0},
    'syncopation': {'ema': 0.58, 'offset': 0.0},
}
```

### Why it matters

Without auto-fill, a fixed `fill_required` threshold will be:
- Too strict for sparse electronic music → blocks all strokes
- Too loose for dense orchestral → triggers on everything

---

## 21. **NEW** IMPORTANT — High-Band Activity & Pattern Helper Methods

**Partially mentioned in v1 under gate §2.1 but source not shown.**

### `_get_high_band_activity` (line 1844)

```python
def _get_high_band_activity(self, event: BeatEvent) -> float:
    include_mid = bool(getattr(cfg, 'high_band_include_mid', True))
    activity = float(max(0.0, (self._mid_energy + self._high_energy) if include_mid else self._high_energy))
    if activity > 1e-6:
        return activity
    # Fallback from event fields
    ...
```

### `_get_mid_bass_activity` (line 1862)

```python
def _get_mid_bass_activity(self, event: BeatEvent) -> float:
    low_hz = float(getattr(cfg, 'mid_bass_freq_low_hz', 200.0))
    high_hz = float(getattr(cfg, 'mid_bass_freq_high_hz', 400.0))
    freq = float(getattr(event, 'frequency', 0.0))
    peak = float(getattr(event, 'peak_energy', 0.0))
    if low_hz <= freq <= high_hz:
        return peak * 0.60
    if beat_band == 'low_mid' and (low_hz * 0.75) <= freq <= (high_hz * 1.25):
        return peak * 0.40
    return 0.0
```

### `_get_high_band_presence_status` (line 1880)

Evaluates 18-frame window from `_recent_high_band_values`:
- Mean ≥ threshold (0.12)
- Occupancy (frames ≥ floor 0.06) ≥ threshold (0.55)
- Delta or variance above thresholds
- Downbeat relaxation factor

### `_get_high_band_pattern_status` (line 1912)

Checks last N beats from `_recent_high_band_beat_hits`:
- Window of `high_band_pattern_window_beats` (default 5)
- Hit count ≥ `high_band_pattern_min_hits` (default 3)

### `_get_low_band_gate_status` (line 1515)

18-frame window from `_recent_low_band_values`:
- Mean ≥ threshold
- Delta or variance above thresholds
- Downbeat relaxation

All of these feed into the FULL_STROKE gate cascade in `process_beat()`. Without them, the high-band and low-band gates cannot function.

---

## What IS Correctly Ported (confirmed same as v1)

| Component | Status |
|---|---|
| 4-band EMA energy tracking (`update_band_energies`) | ✓ Correct |
| RMS envelope with attack/release (`update_envelope`) | ✓ Correct |
| Silence deadzone hysteresis gate (`update_silence_deadzone_gate`) | ✓ Correct |
| Basic trigger classification — syncopation/beat/downbeat/creep | ✓ Correct (but no hierarchy guards) |
| Tempo confidence check (`_tempo_ready_for_motion`) | ✓ Simplified (no traffic lights) |
| Single-frame strict bass gate (`_strict_bass_motion_allowed`) | ✓ Partial (no history) |
| Journey progress / S-curve timing (`update_journey_progress`) | ✓ Correct |
| Treble lift EMA with landing guard (`compute_treble_lift`) | ✓ Correct |
| Sub-bass bloom radius formula (`compute_radius_bloom_from_sub_bass`) | ✓ Correct |
| Journey continuation between discrete events (`build_decision`) | ✓ Correct |

---

## Implementation Priority Order (Revised)

| Priority | Item | Effort | Depends on |
|---|---|---|---|
| 1 | Rolling history deques (all 5) | Low | — |
| 2 | FluxTracker (250ms deque + rise factor) | Low | — |
| 3 | `_has_recent_beats` + beat-hierarchy guards | Low | — |
| 4 | No-beat timeout → park decay | Low | #3 |
| 5 | `_update_stroke_readiness` (full traffic-light + grace) | Medium | — |
| 6 | `_update_motion_mode` (FULL_STROKE/CREEP_MICRO) | Medium | — |
| 7 | `_get_low_band_activity` + `_get_high_band_activity` + `_get_mid_bass_activity` helper methods | Low | — |
| 8 | `_is_low_band_full_enough` | Medium | #1, #7 |
| 9 | `_get_high_band_presence_status` + `_get_high_band_pattern_status` | Medium | #1 |
| 10 | `_passes_dual_band_db_gate` | Medium | #1 |
| 11 | `_is_mid_trigger_blocked` | Low | — |
| 12 | `_get_spectrum_fill_ratio` + `_passes_overall_amp_fill_gate` | High | #7 |
| 13 | Auto-fill adaptation (`_update_auto_fill_required`) | Medium | #12 |
| 14 | Silence fade-out tracker (`_fade_intensity` + tempo reset) | Low | — |
| 15 | Post-silence volume ramp | Low | #14 |
| 16 | Flux drop → creep fallback guard | Low | #1, #6 |
| 17 | `_cap_bpm_to_last_locked` + `_stabilize_unlocked_bpm` | Low | — |
| 18 | `_build_runtime_feature_values` | Low | — |
| 19 | `_predict_learning_targets` (incl. `cadence_rule`) | Medium | #18 |
| 20 | `_try_load_learning_model` (complete, incl. `cadence_rule` field) | Low | — |
| 21 | `_update_learning_adapter` (runtime teaching blend) | Medium | #19, #20 |
| 22 | `_update_bass_jitter_drive` (stub, disconnected) | Low | — |

**Foundation tier (1–6):** Ship first. Everything else depends on the deques and mode concept.  
**Gate tier (7–13):** Core intelligence gates. Ship after foundation.  
**Silence tier (14–16):** Polish + safety nets.  
**BPM tier (17):** Stabilization during lock loss.  
**Learning tier (18–21):** Independent of gates, can be done in parallel.  
**Stub tier (22):** Low priority, disconnected.

---

## Reference Commit

All original implementations at `ab2b35a`:

```
git show ab2b35a:stroke_mapper.py
```

Key line numbers:

| Method | Line |
|---|---|
| `__init__` (all deque/state declarations) | 102–427 |
| `_update_flux_history` + `_get_flux_rise_factor` | 758–771 |
| `_is_center_reset_flux_guard_active` | 773–789 |
| `_has_recent_beats` + `_arm_tempo_reset_motion_hold` | 791–810 |
| `_update_learning_adapter` | 923–1020 |
| `_try_load_learning_model` | 1050–1082 |
| `_build_runtime_feature_values` | 1083–1118 |
| `_predict_learning_targets` | 1120–1165 |
| `_update_envelope` | 1222–1229 |
| `_update_stroke_readiness` | 1230–1367 |
| `_update_motion_mode` | 1368–1406 |
| `_get_reliable_metronome_bpm` | 1432–1448 |
| `_cap_bpm_to_last_locked` | 1450–1457 |
| `_stabilize_unlocked_bpm` | 1458–1482 |
| `_update_band_energies` | 1483–1493 |
| `_get_low_band_activity` | 1494–1514 |
| `_get_low_band_gate_status` | 1515–1539 |
| `_is_low_band_full_enough` | 1540–1604 |
| `_passes_dual_band_db_gate` | 1606–1668 |
| `_is_mid_trigger_blocked` | 1669–1683 |
| `_get_overall_amp_fill_required_base` | 1685–1704 |
| `_update_auto_fill_required` | 1705–1730 |
| `_get_overall_amp_fill_required` | 1760–1769 |
| `_get_spectrum_fill_ratio` | 1770–1806 |
| `_passes_overall_amp_fill_gate` | 1808–1842 |
| `_get_high_band_activity` | 1844–1861 |
| `_get_mid_bass_activity` | 1862–1879 |
| `_get_high_band_presence_status` | 1880–1911 |
| `_get_high_band_pattern_status` | 1912–1930 |
| `_update_bass_jitter_drive` | 1931–1968 |
| `process_beat` (full gate cascade + fade tracker + no-beat timeout) | 2047–2688 |
| `_apply_fade` (post-silence ramp) | 2689–2717 |
