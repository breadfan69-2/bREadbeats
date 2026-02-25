# Learning Outputs Complete Teardown Audit — 2026-02-25

Every wire traced from definition → setting → consumption. All line numbers verified.

---

## 1. `LearningOutputs` Dataclass Definition

**File:** [beat_intelligence.py](beat_intelligence.py#L27-L36)

```python
@dataclass
class LearningOutputs:
    """Bounded modifier outputs from the learning adapter."""
    divisor_hint: int = 1              # beats-between-strokes hint from cadence_rule
    radius_mult: float = 1.0           # scale arc radius
    lead_ms: float = 0.0               # predictive arc timing offset
    sync_size_mult: float = 1.0        # syncopation arc size multiplier
    sync_speed_mult: float = 1.0       # syncopation arc speed multiplier
    gate_bias: float = 0.0             # gate strictness bias (-1..+1)
    active: bool = False               # whether learning produced valid output
```

**7 fields total.** Embedded in `BeatDecision` at [beat_intelligence.py L54](beat_intelligence.py#L54):
```python
    learning: LearningOutputs = field(default_factory=LearningOutputs)
```

---

## 2. Every Place `LearningOutputs` Fields Are SET

### 2a. Instance field initialization — [beat_intelligence.py L192](beat_intelligence.py#L192)
```python
        self._learning_outputs: LearningOutputs = LearningOutputs()
```

### 2b. Default return (no learning) — [beat_intelligence.py L504](beat_intelligence.py#L504)
```python
        outputs = LearningOutputs()
        if not self._learning_enabled or not self._learning_model_loaded:
            self._learning_outputs = outputs
            return outputs
```

### 2c. Active prediction outputs — [beat_intelligence.py L561-L570](beat_intelligence.py#L561-L570)
```python
        outputs = LearningOutputs(
            divisor_hint=cadence_beats,
            radius_mult=radius_mult,
            lead_ms=lead_ms,
            sync_size_mult=sync_size_mult,
            sync_speed_mult=sync_speed_mult,
            gate_bias=gate_bias,
            active=True,
        )
```

### 2d. EMA-smoothed write-back — [beat_intelligence.py L582-L591](beat_intelligence.py#L582-L591)
```python
        outputs = LearningOutputs(
            divisor_hint=self._learned_divisor_hint,
            radius_mult=float(np.clip(self._learned_radius_mult, 0.3, 2.5)),
            lead_ms=float(np.clip(self._learned_lead_ms, 0.0, 100.0)),
            sync_size_mult=float(np.clip(self._learned_sync_size_mult, 0.5, 2.0)),
            sync_speed_mult=float(np.clip(self._learned_sync_speed_mult, 0.3, 3.0)),
            gate_bias=float(np.clip(self._learned_gate_bias, -1.0, 1.0)),
            active=True,
        )
        self._learning_outputs = outputs
```

### 2e. Silence/no-confidence early return — [beat_intelligence.py L1762](beat_intelligence.py#L1762)
```python
                learning=LearningOutputs(),
```
(inside a `BeatDecision` returned when silence triggers an early path)

### 2f. Final `build_decision` return — [beat_intelligence.py L1992](beat_intelligence.py#L1992)
```python
            learning=learning,
```
(the `learning` local comes from `self._update_learning_adapter(event)` at [L1766](beat_intelligence.py#L1766))

---

## 3. Every Place `LearningOutputs` Fields Are READ/Consumed

### 3a. In `build_decision` (beat_intelligence.py)

#### `learning.active` + `_committed_divisor_hint` — [L1943-L1944](beat_intelligence.py#L1943-L1944)
```python
        if learning.active and self._committed_divisor_hint > 1:
            interval_beats = max(interval_beats, self._committed_divisor_hint)
```
**Effect:** Overrides `interval_beats` to be at least the learned cadence divisor (e.g., every 2nd or 4th beat).

#### `learning` passed through to `BeatDecision` — [L1992](beat_intelligence.py#L1992)
```python
            learning=learning,
```
The entire `LearningOutputs` object is embedded in the returned `BeatDecision`.

### 3b. In `_passes_overall_amp_fill_gate` (beat_intelligence.py)

#### `self._learning_outputs.active` + `self._learning_outputs.gate_bias` — [L985-L988](beat_intelligence.py#L985-L988)
```python
        # Apply learning gate_bias: negative bias lowers the bar (more motion),
        # positive bias raises it (less motion).  Scaled to ±20% of required.
        if self._learning_outputs.active and abs(self._learning_outputs.gate_bias) > 1e-3:
            bias_shift = float(self._learning_outputs.gate_bias * 0.20 * required)
            required = float(np.clip(required + bias_shift, 0.02, 0.99))
```
**Effect:** Shifts the fill-gate `required` threshold by up to ±20%.  Negative `gate_bias` → easier to pass (more motion); positive → harder (less motion).

### 3c. In stroke_mapper.py

#### `learning.active` + `learning.radius_mult` — [stroke_mapper.py L770-L771](stroke_mapper.py#L770-L771)
```python
                        if learning.active:
                            self._journey_learning_mult = float(np.clip(learning.radius_mult, 0.3, 2.5))
```
**Effect:** Latches the radius multiplier at journey start. Applied to `bloom_target_radius` at [L788-L789](stroke_mapper.py#L788-L789):
```python
                bloom_target_radius = float(type_park_radius + ((type_bloom - type_park_radius) * learning_mult))
```

#### `learning.sync_size_mult` — [stroke_mapper.py L773-L779](stroke_mapper.py#L773-L779)
```python
                            if decision.trigger_kind == "syncopation":
                                sync_size = float(np.clip(learning.sync_size_mult, 0.5, 2.0))
                                self._journey_max_radius = float(np.clip(
                                    self._journey_max_radius * sync_size,
                                    self._journey_park_radius,
                                    1.0,
                                ))
```
**Effect:** For syncopation triggers, scales `_journey_max_radius` by the learning sync_size_mult (0.5–2.0×).

#### `learning.active` (else branch) — [stroke_mapper.py L781](stroke_mapper.py#L781)
```python
                        else:
                            self._journey_learning_mult = 1.0
```

### 3d. Fields READ NOWHERE

| Field | Consumed? |
|---|---|
| `divisor_hint` | Only via `_committed_divisor_hint` inside `build_decision` (indirect — latched from `_learned_divisor_hint` in `update_journey_progress`) |
| `radius_mult` | stroke_mapper.py L771 (journey latch) |
| `lead_ms` | **NOT consumed anywhere** — computed but never read by any consumer |
| `sync_size_mult` | stroke_mapper.py L774 (syncopation max_radius scaling) |
| `sync_speed_mult` | **NOT consumed anywhere** — computed but never read by any consumer |
| `gate_bias` | beat_intelligence.py L987 (`_passes_overall_amp_fill_gate`) |
| `active` | beat_intelligence.py L987, L1943; stroke_mapper.py L770 |

**Dead fields: `lead_ms` and `sync_speed_mult` are computed in `_update_learning_adapter` but never consumed downstream.**

---

## 4. `_committed_divisor_hint` / `_learned_divisor_hint`

### Initialization — [beat_intelligence.py L194-L195](beat_intelligence.py#L194-L195)
```python
        self._learned_divisor_hint: int = 1
        self._committed_divisor_hint: int = 1   # only applied at journey start
```

### SET: `_learned_divisor_hint` — [beat_intelligence.py L574](beat_intelligence.py#L574)
```python
        self._learned_divisor_hint = cadence_beats  # discrete, no smoothing
```

### SET: `_learned_divisor_hint` into LearningOutputs — [beat_intelligence.py L583](beat_intelligence.py#L583)
```python
            divisor_hint=self._learned_divisor_hint,
```

### LATCH: `_committed_divisor_hint` ← `_learned_divisor_hint` — [beat_intelligence.py L1580](beat_intelligence.py#L1580)
```python
            self._committed_divisor_hint = self._learned_divisor_hint
```
Inside `update_journey_progress`, triggered only when a new journey actually starts.

### CONSUMED: `_committed_divisor_hint` in `build_decision` — [beat_intelligence.py L1943-L1944](beat_intelligence.py#L1943-L1944)
```python
        if learning.active and self._committed_divisor_hint > 1:
            interval_beats = max(interval_beats, self._committed_divisor_hint)
```

---

## 5. `learning.speed_mult` and `learning.radius_mult` in stroke_mapper.py

### `learning.radius_mult` consumed — [stroke_mapper.py L771](stroke_mapper.py#L771)
```python
                            self._journey_learning_mult = float(np.clip(learning.radius_mult, 0.3, 2.5))
```
Stored in `_journey_learning_mult` (initialized at [L60](stroke_mapper.py#L60)), then applied at [L788-L789](stroke_mapper.py#L788-L789):
```python
                learning_mult = self._journey_learning_mult
                ...
                bloom_target_radius = float(type_park_radius + ((type_bloom - type_park_radius) * learning_mult))
```

### `speed_mult` in stroke_mapper.py — **NOT a learning field**
The only `speed_mult` in stroke_mapper.py is at [L1151-L1154](stroke_mapper.py#L1151-L1154):
```python
        speed_mult = float(np.clip(1.0 + delta, 0.5, 1.5))
        size_mult = float(np.clip(1.0 + delta, 0.5, 1.5))

        jitter_speed = max(0.0, base_speed * speed_mult)
```
This is bass-frequency-derived jitter speed, **not from learning**.

### `learning.sync_speed_mult` — **DEAD**
Computed in `_update_learning_adapter` ([L550-L551](beat_intelligence.py#L550-L551), [L587](beat_intelligence.py#L587)) but **never consumed** in stroke_mapper.py or anywhere else.

---

## 6. `_learning_strength` Field

### Definition — [beat_intelligence.py L183](beat_intelligence.py#L183)
```python
        self._learning_strength: float = 0.55
```

### SET from config — [stroke_mapper.py L182](stroke_mapper.py#L182)
```python
        self._learning_strength = float(getattr(self.config.beat, "teaching_learning_strength", 0.0) or 0.0)
```

### SET via configure — [stroke_mapper.py L207](stroke_mapper.py#L207)
```python
        self._learning_strength = float(learning_strength)
```

### Forwarded to BI — [stroke_mapper.py L220](stroke_mapper.py#L220)
```python
            strength=self._learning_strength,
```

### SET in BI.configure_learning — [beat_intelligence.py L321](beat_intelligence.py#L321)
```python
        self._learning_strength = float(np.clip(strength, 0.0, 1.0))
```

### READ in `_update_learning_adapter` — [beat_intelligence.py L528](beat_intelligence.py#L528)
```python
        strength = self._learning_strength
```

### How it scales outputs (in `_update_learning_adapter`)
All blended outputs are multiplied by `strength`:

| Output | Formula | Line |
|---|---|---|
| `radius_mult` | `1.0 + strength * (arc_size * 2.0 - 1.0)` | [L544](beat_intelligence.py#L544) |
| `sync_size_mult` | `1.0 + strength * (arc_size - 0.5)` | [L547](beat_intelligence.py#L547) |
| `sync_speed_mult` | `1.0 + strength * (1.0/max(arc_dur_frac, 0.1) - 1.0)` | [L550](beat_intelligence.py#L550) |
| `gate_bias` | `strength * (gate_strict - 0.5) * 2.0 * no_motion_bias` | [L554](beat_intelligence.py#L554) |
| `lead_ms` | `strength * jitter_mix * 50.0` | [L558](beat_intelligence.py#L558) |

When `strength = 0`: all outputs collapse to neutral (1.0, 0.0, etc.).  
When `strength = 1`: full prediction influence.

### Config source — [config.py L119](config.py#L119)
```python
    teaching_learning_strength: float = 0.59
```

### GUI reads/writes — [main.py L3428](main.py#L3428), [L3458](main.py#L3458), [L5209](main.py#L5209), [L5262](main.py#L5262), [L5278](main.py#L5278), [L5315](main.py#L5315), [L7056](main.py#L7056), [L7090](main.py#L7090)

---

## 7. `_learning_no_motion_bias` Field

### Definition — [beat_intelligence.py L185](beat_intelligence.py#L185)
```python
        self._learning_no_motion_bias: float = 1.0
```

### SET from config — [stroke_mapper.py L184](stroke_mapper.py#L184)
```python
        self._learning_no_motion_bias = float(getattr(self.config.beat, "teaching_no_motion_bias", 1.0) or 1.0)
```

### SET via configure — [stroke_mapper.py L209](stroke_mapper.py#L209)
```python
        self._learning_no_motion_bias = float(no_motion_bias)
```

### Forwarded to BI — [stroke_mapper.py L222](stroke_mapper.py#L222)
```python
            no_motion_bias=self._learning_no_motion_bias,
```

### SET in BI.configure_learning — [beat_intelligence.py L323](beat_intelligence.py#L323)
```python
        self._learning_no_motion_bias = float(np.clip(no_motion_bias, 0.25, 3.0))
```

### READ in `_update_learning_adapter` — [beat_intelligence.py L529](beat_intelligence.py#L529)
```python
        no_motion_bias = self._learning_no_motion_bias
```

### Consumed at — [beat_intelligence.py L554](beat_intelligence.py#L554)
```python
        gate_bias = strength * (gate_strict - 0.5) * 2.0 * no_motion_bias
```
**Effect:** Amplifies/attenuates `gate_bias` — higher `no_motion_bias` (>1.0) makes learning gate adjustments stronger (more aggressive holdback or loosening). Range clamp: 0.25–3.0.

---

## 8. `learning.gate_bias` → `_passes_overall_amp_fill_gate`

Full gate modification path — [beat_intelligence.py L985-L989](beat_intelligence.py#L985-L989):
```python
        # Apply learning gate_bias: negative bias lowers the bar (more motion),
        # positive bias raises it (less motion).  Scaled to ±20% of required.
        if self._learning_outputs.active and abs(self._learning_outputs.gate_bias) > 1e-3:
            bias_shift = float(self._learning_outputs.gate_bias * 0.20 * required)
            required = float(np.clip(required + bias_shift, 0.02, 0.99))
```

**Flow:**
1. `gate_strictness` predicted (0.0–1.0) → centred around 0.5 → `gate_bias` = `strength * (gate_strict - 0.5) * 2.0 * no_motion_bias`
2. `gate_bias` clipped to [-1.0, +1.0]
3. EMA smoothed into `_learned_gate_bias`
4. Written into `self._learning_outputs.gate_bias`
5. In `_passes_overall_amp_fill_gate`: `required += gate_bias * 0.20 * required`
   - `gate_bias = -1.0` → `required` drops by 20% (easier to pass)
   - `gate_bias = +1.0` → `required` rises by 20% (harder to pass)
   - Final `required` clipped to [0.02, 0.99]

### Where `_passes_overall_amp_fill_gate` is called — [beat_intelligence.py L1825](beat_intelligence.py#L1825):
```python
            elif not self._passes_overall_amp_fill_gate(event, raw_trigger_kind):
```
(inside `build_decision`, after silence/readiness checks)

---

## 9. `learning.active` — All Checks

| File | Line | Context |
|---|---|---|
| [beat_intelligence.py](beat_intelligence.py#L987) | 987 | `if self._learning_outputs.active and abs(self._learning_outputs.gate_bias) > 1e-3:` |
| [beat_intelligence.py](beat_intelligence.py#L1943) | 1943 | `if learning.active and self._committed_divisor_hint > 1:` |
| [stroke_mapper.py](stroke_mapper.py#L770) | 770 | `if learning.active:` (journey start radius latch) |

---

## 10. `_recent_flux_values` Deque

### Definition — [beat_intelligence.py L110](beat_intelligence.py#L110)
```python
        self._recent_flux_values: deque = deque(maxlen=60)
```
**maxlen = 60** (~1 second at 60fps)

### Populated — [beat_intelligence.py L1087](beat_intelligence.py#L1087)
```python
        self._recent_flux_values.append(float(getattr(event, "spectral_flux", 0.0) or 0.0))
```

### All READ Sites

| Line | Method | Usage |
|---|---|---|
| [L262](beat_intelligence.py#L262) | `_volume_normalized_flux` | `flux_history = list(self._recent_flux_values)` — P95 for volume normalization |
| [L683](beat_intelligence.py#L683) | `_update_tempo_unlock_hold` | `recent_flux = list(self._recent_flux_values)` — flux spike/drop detection |
| [L718](beat_intelligence.py#L718) | `_update_tempo_unlock_hold` | `recent_flux = list(self._recent_flux_values)` — baseline capture |
| [L1374](beat_intelligence.py#L1374) | `compute_radius_bloom_from_sub_bass` | `flux_history = list(self._recent_flux_values)` — P95 for flux normalization |
| [L1878](beat_intelligence.py#L1878) | `build_decision` (phrase entry) | `recent_flux = list(self._recent_flux_values)` — phrase flux baseline |
| [L1888](beat_intelligence.py#L1888) | `build_decision` (phrase drop check) | `recent_flux = list(self._recent_flux_values)` — drop cancellation |
| [L1904](beat_intelligence.py#L1904) | `build_decision` (phrase renewal) | `recent_flux = list(self._recent_flux_values)` — renewal check |
| [L2008](beat_intelligence.py#L2008) | `snapshot_gate_state` | `flux_vals = list(self._recent_flux_values)` — mean/std/delta for CSV |

---

## 11. String Search Results Across All Python Files

### `jitter_mix`

| File | Line | Context |
|---|---|---|
| [beat_intelligence.py](beat_intelligence.py#L534) | 534 | `jitter_mix = float(np.clip(predictions.get("jitter_mix", 0.0), 0.0, 1.0))` |
| [beat_intelligence.py](beat_intelligence.py#L558) | 558 | `lead_ms = strength * jitter_mix * 50.0  # 0..50ms predictive offset` |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L17) | 17 | Target column definition |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L81) | 81 | `jitter_mix = np.clip(0.15 * speed_n + 0.85 * jerk_n, 0.0, 1.0)` |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L95) | 95 | `"jitter_mix": float(jitter_mix[i]),` |
| [local_learning/fit_rules.py](local_learning/fit_rules.py#L31) | 31 | Target column list |

### `creep_mix`

| File | Line | Context |
|---|---|---|
| [beat_intelligence.py](beat_intelligence.py#L535) | 535 | `creep_mix = float(np.clip(predictions.get("creep_mix", 0.5), 0.0, 1.0))` |
| [beat_intelligence.py](beat_intelligence.py#L553) | 553 | `# creep_mix drives no-motion holdback: higher creep = less motion` (comment only — **creep_mix itself is unused after assignment; only gate_strict feeds gate_bias**) |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L18) | 18 | Target column definition |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L82) | 82 | `creep_mix = np.clip(0.70 * (1.0 - speed_n) + 0.30 * (1.0 - radius), 0.0, 1.0)` |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L96) | 96 | `"creep_mix": float(creep_mix[i]),` |
| [local_learning/fit_rules.py](local_learning/fit_rules.py#L32) | 32 | Target column list |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L31) | 31 | Test target columns |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L41) | 41 | Test rule definition |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L209) | 209 | `self.assertIn("creep_mix", preds)` |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L237) | 237 | `self.assertAlmostEqual(preds["creep_mix"], 0.6, places=4)` |

### `burst_prob`

| File | Line | Context |
|---|---|---|
| [beat_intelligence.py](beat_intelligence.py#L537) | 537 | `burst_prob = float(np.clip(predictions.get("burst_prob", 0.2), 0.0, 1.0))` (**assigned but NEVER consumed**) |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L20) | 20 | Target column definition |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L84) | 84 | `burst_prob = np.clip(0.70 * jerk_n + 0.30 * ang_n, 0.0, 1.0)` |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L98) | 98 | `"burst_prob": float(burst_prob[i]),` |
| [local_learning/fit_rules.py](local_learning/fit_rules.py#L34) | 34 | Target column list |

### `arc_size`

| File | Line | Context |
|---|---|---|
| [beat_intelligence.py](beat_intelligence.py#L532) | 532 | `arc_size = float(np.clip(predictions.get("arc_size", 0.5), 0.0, 1.0))` |
| [beat_intelligence.py](beat_intelligence.py#L544) | 544 | `radius_mult = 1.0 + strength * (arc_size * 2.0 - 1.0)` |
| [beat_intelligence.py](beat_intelligence.py#L547) | 547 | `sync_size_mult = 1.0 + strength * (arc_size - 0.5)` |
| [config.py](config.py#L111) | 111 | `syncopation_arc_size: float = 0.82` (different meaning — config param, not learning) |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L15) | 15 | Target column definition |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L79) | 79 | `arc_size = np.clip(0.15 + 0.75 * speed_n, 0.0, 1.0)` |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L93) | 93 | `"arc_size": float(arc_size[i]),` |
| [local_learning/fit_rules.py](local_learning/fit_rules.py#L29) | 29 | Target column list |
| [local_learning/fit_rules.py](local_learning/fit_rules.py#L193) | 193 | `speed_test = ds.y["arc_size"][test_idx]` |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L31) | 31 | Test target columns |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L37) | 37 | Test rule definition |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L208) | 208 | `self.assertIn("arc_size", preds)` |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L236) | 236 | `self.assertAlmostEqual(preds["arc_size"], 0.5, places=4)` |

### `arc_duration_frac`

| File | Line | Context |
|---|---|---|
| [beat_intelligence.py](beat_intelligence.py#L533) | 533 | `arc_dur_frac = float(np.clip(predictions.get("arc_duration_frac", 1.0), 0.1, 4.0))` |
| [beat_intelligence.py](beat_intelligence.py#L550) | 550 | `sync_speed_mult = 1.0 + strength * (1.0 / max(arc_dur_frac, 0.1) - 1.0)` |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L16) | 16 | Target column definition |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L80) | 80 | Formula |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L94) | 94 | Dict output |
| [local_learning/fit_rules.py](local_learning/fit_rules.py#L30) | 30 | Target column list |

### `gate_strictness`

| File | Line | Context |
|---|---|---|
| [beat_intelligence.py](beat_intelligence.py#L536) | 536 | `gate_strict = float(np.clip(predictions.get("gate_strictness", 0.5), 0.0, 1.0))` |
| [beat_intelligence.py](beat_intelligence.py#L554) | 554 | `gate_bias = strength * (gate_strict - 0.5) * 2.0 * no_motion_bias` |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L19) | 19 | Target column definition |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L83) | 83 | Formula |
| [local_learning/extract_motion_targets.py](local_learning/extract_motion_targets.py#L97) | 97 | Dict output |
| [local_learning/fit_rules.py](local_learning/fit_rules.py#L33) | 33 | Target column list |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L31) | 31 | Test target columns |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L45) | 45 | Test rule definition |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L210) | 210 | `self.assertIn("gate_strictness", preds)` |

### `radius_mult`

| File | Line | Context |
|---|---|---|
| [beat_intelligence.py](beat_intelligence.py#L30) | 30 | `radius_mult: float = 1.0` (dataclass field) |
| [beat_intelligence.py](beat_intelligence.py#L196) | 196 | `self._learned_radius_mult: float = 1.0` |
| [beat_intelligence.py](beat_intelligence.py#L544-L545) | 544–545 | Computed: `radius_mult = 1.0 + strength * ...` then clipped |
| [beat_intelligence.py](beat_intelligence.py#L563) | 563 | `radius_mult=radius_mult,` (first LearningOutputs) |
| [beat_intelligence.py](beat_intelligence.py#L575) | 575 | EMA smooth: `self._learned_radius_mult += alpha_radius * (outputs.radius_mult - ...)` |
| [beat_intelligence.py](beat_intelligence.py#L584) | 584 | `radius_mult=float(np.clip(self._learned_radius_mult, 0.3, 2.5)),` (smoothed output) |
| [beat_intelligence.py](beat_intelligence.py#L1231) | 1231 | `"""Return (profile_kind, radius_mult, hat_only_limited, park_bounce_gain).` (docstring — different context: `_transient_motion_profile`) |
| [beat_intelligence.py](beat_intelligence.py#L1771) | 1771 | `motion_profile, motion_radius_mult, ...` (return from `_transient_motion_profile`, NOT learning) |
| [beat_intelligence.py](beat_intelligence.py#L1952) | 1952 | `radius_bloom = float(np.clip(base + (span * motion_radius_mult), base, 1.0))` (motion profile, NOT learning) |
| [stroke_mapper.py](stroke_mapper.py#L771) | 771 | `self._journey_learning_mult = float(np.clip(learning.radius_mult, 0.3, 2.5))` |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L330-L331) | 330–331 | Range assertions |
| [tests/test_phase5_learning.py](tests/test_phase5_learning.py#L348) | 348 | Neutral-at-zero-strength assertion |

### `speed_mult`

| File | Line | Context |
|---|---|---|
| [beat_intelligence.py](beat_intelligence.py#L33) | 33 | `sync_speed_mult: float = 1.0` (dataclass field) |
| [beat_intelligence.py](beat_intelligence.py#L199) | 199 | `self._learned_sync_speed_mult: float = 1.0` |
| [beat_intelligence.py](beat_intelligence.py#L550-L551) | 550–551 | Computed and clipped |
| [beat_intelligence.py](beat_intelligence.py#L566) | 566 | `sync_speed_mult=sync_speed_mult,` (first output) |
| [beat_intelligence.py](beat_intelligence.py#L578) | 578 | EMA smooth |
| [beat_intelligence.py](beat_intelligence.py#L587) | 587 | `sync_speed_mult=float(np.clip(..., 0.3, 3.0)),` (smoothed output) |
| [stroke_mapper.py](stroke_mapper.py#L1151) | 1151 | `speed_mult = float(np.clip(1.0 + delta, 0.5, 1.5))` (**NOT learning** — bass jitter) |
| [stroke_mapper.py](stroke_mapper.py#L1154) | 1154 | `jitter_speed = max(0.0, base_speed * speed_mult)` (**NOT learning**) |

---

## Summary: Dead & Live Wires

### LIVE wires (actually affect behavior)
| Field | Consumer | Effect |
|---|---|---|
| `active` | BI L987, L1943; SM L770 | Guards all learning reads |
| `radius_mult` | SM L771 | Scales orbit radius at journey start |
| `sync_size_mult` | SM L774 | Scales syncopation max_radius |
| `gate_bias` | BI L987-L988 | ±20% shift on fill-gate `required` |
| `divisor_hint` | → `_committed_divisor_hint` → BI L1944 | Overrides `interval_beats` |

### DEAD wires (computed but never consumed anywhere)
| Field | Computed at | Status |
|---|---|---|
| `lead_ms` | BI L558, L564, L585 | **DEAD** — no consumer |
| `sync_speed_mult` | BI L550-L551, L566, L578, L587 | **DEAD** — no consumer |
| `burst_prob` | BI L537 | **DEAD** — predicted, never stored in LearningOutputs, never consumed |
| `creep_mix` | BI L535 | **DEAD** — predicted, never used after gate_bias computation (only `gate_strictness` feeds `gate_bias`) |

### Intermediate predictions (consumed within `_update_learning_adapter` only, not exposed)
| Prediction | Usage |
|---|---|
| `arc_size` | Drives `radius_mult` and `sync_size_mult` |
| `arc_duration_frac` | Drives `sync_speed_mult` (which is dead) |
| `jitter_mix` | Drives `lead_ms` (which is dead) |
| `creep_mix` | Unused (comment says it drives holdback, but no code uses it) |
| `gate_strictness` | Drives `gate_bias` |
| `burst_prob` | Completely unused |
