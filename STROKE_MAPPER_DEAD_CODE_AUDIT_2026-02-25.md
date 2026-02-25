# Stroke Mapper Dead Code Audit — 2026-02-25

Audit of `stroke_mapper.py` (1 551 lines) for dead code: unused classes,
methods, instance variables, parameters, and vestigial logic.

---

## 1. Fully Dead Class

| Item | Line | Evidence |
|------|------|----------|
| `MotionMode` class (and its constants `FULL_STROKE`, `CREEP_MICRO`) | 32-34 | Defined but **never referenced anywhere** in the entire codebase — not in `stroke_mapper.py`, `main.py`, tests, or any other file. |

---

## 2. Fully Dead Methods (defined, never called)

| Method | Line | Evidence |
|--------|------|----------|
| `_s_curve()` | 1365 | Static method. Zero callers anywhere in the codebase. Superseded by `_quintic_ease`. |
| `_sine_ease_with_velocity()` | 1381 | Static method. Zero callers. Docstring describes it as the "core curve" but nothing invokes it. |
| `_s_curve_with_initial_velocity()` | 1431 | Static method. Docstring labels it "Legacy cubic Hermite easing (kept for test compatibility)" but **no tests call it either**. Zero callers. |
| `_compute_initial_speed_slope()` | 1462 | Instance method. Zero callers. Was the companion to the above easing functions; all three became dead together when the motion pipeline switched to purely angle-driven progression. |

---

## 3. Fully Dead Instance Variables (written, never read)

| Variable | Init Line | Evidence |
|----------|----------|----------|
| `_active_interval_beats` | 63 | Set in `__init__` and updated per-frame at line 344 (`= decision.interval_beats`), but **never read** anywhere — not internally, not from `main.py`, not from tests. |
| `_lazy_glide_active` | 90 | Set in `__init__` and per-frame at line 346, but **never read** in `stroke_mapper.py` or externally. (The identically-named field in `BeatIntelligence` is used; this copy on StrokeMapper is not.) |
| `_journey_initial_speed_slope` | 86 | Set in `__init__` only. Never read or updated after init. Was the runtime store for `_compute_initial_speed_slope`'s output; both are dead. |
| `_journey_nominal_angular_speed` | 87 | Set in `__init__` only. Never read or updated after init. |
| `_idle_radius` | 102 | Set in `__init__` (`= self._min_radius`). Never read. Superseded by `_park_idle_radius`. |
| `_silence_decay_per_beat` | 103 | Set in `__init__` (`= 0.40`). Never read. Decay is now handled by `BeatIntelligence.silence_fade`. |
| `_anchor_angle` | 156 | Set in `__init__` only. Never read. The anchor angle is computed inline in `_compute_landing_rotation` using `(π/2) * self._anchor_sign` instead. |
| `_learning_model` | 216 | Declared `Optional[dict]`, loaded by `_try_load_learning_model()`, set/cleared in `configure_learning()` — but **never read** for any decision. The actual model lives in `BeatIntelligence._learning_model`. This is a vestigial parallel copy. |
| `_learning_isolation_mode` | 211 | Set in `__init__` and `configure_learning()` but never read. Not forwarded to `_sync_learning_to_intelligence()` or used anywhere. |
| `_learning_apply_in_circle_mode` | 210 | Same pattern as `_learning_isolation_mode`: set but never read. |
| `_post_wait_reentry_progress` | 182 | Set in `__init__` only. Never read or updated. |
| `_post_wait_reentry_beats_remaining` | 183 | Set in `__init__` only. Never read or updated. |

### Settle-system variables (partially dead)

`_settle_active` is used as a boolean flag (set `False` at lines 697, 784; read at line 804), but the following companion variables are **never read after initialization**:

| Variable | Init Line | Evidence |
|----------|----------|----------|
| `_settle_elapsed` | 132 | Never read or updated beyond init. |
| `_settle_start_angle` | 133 | Never read beyond init. |
| `_settle_decay_rate` | 134 | Never read beyond init. |

### Crossfade-system variables (partially dead)

`_crossfade_active` is written to (`= False` at line 695) but never read as a condition. The following companion variables are **write-only**:

| Variable | Init Line | Evidence |
|----------|----------|----------|
| `_crossfade_elapsed` | 139 | Never read beyond init. |
| `_crossfade_duration` | 140 | Never read beyond init. |
| `_crossfade_from_angle` | 141 | Never read beyond init. |
| `_crossfade_from_center_y` | 142 | Never read beyond init. |
| `_crossfade_from_radius` | 143 | Never read beyond init. |

---

## 4. Dead Instance Variable (always zero, used but effectless)

| Variable | Init Line | Evidence |
|----------|----------|----------|
| `_center_x_offset` | 188 | Initialized to `0.0`, explicitly set to `0.0` in the wander code path (line 1167), and decayed via `*= decay` in the silence path (line 1171) — but since it is **never set to a non-zero value**, the decay and the `x_cap` computation in `_radius_cap_for_center` (line 1503) are no-ops. This is remnant infrastructure from removed X-axis wander. |

---

## 5. Dead Method (only called but does nothing useful)

| Method | Line | Evidence |
|--------|------|----------|
| `_try_load_learning_model()` | 265 | Loads JSON into `self._learning_model`, but since `_learning_model` is never read (see §3), the entire method is effectless. The real model loading now happens in `BeatIntelligence._try_load_learning_model()`. |

---

## 6. Unused Parameters

| Method | Parameter | Line | Evidence |
|--------|-----------|------|----------|
| `configure_geometry_rest_state` | `y_offset` | 221 | Accepted from caller (`main.py` line 2853) but the method body ignores it — `_park_y` is unconditionally hard-coded to `0.20`. |
| `configure_geometry_rest_state` | `sink_start_intensity` | 221 | Default `0.25`, never passed by any caller, and completely unused in the method body. |

---

## 7. Stored-but-Never-Read Constructor Parameter

| Attribute | Line | Evidence |
|-----------|------|----------|
| `self.send_callback` | 48 | Accepted as `__init__` parameter and stored, but StrokeMapper **never calls it**. The caller (`main.py`) passes `self._send_command_direct` but the actual command dispatch happens through the returned `TCodeCommand` from `process_beat`, not through this callback. |

---

## 8. Vestigial / No-Op Logic Block

| Location | Lines | Description |
|----------|-------|-------------|
| §8 Entry journey gating | 1083-1091 | The conditional block ultimately reaches a `pass` statement (line 1091) and modifies nothing. The `_post_silence_entry_done` flag is set/reset but has **no downstream consumer** — no code path branches on it to alter alpha/beta, radius, or any output. The gating described in the comment ("force creep otherwise") is never implemented. |
| `_last_trigger_kind` writes | 64, 345 | Set per-frame but only externally read in one test (`test_stroke_mapper_contract.py:706`) which *writes* to it. No production code reads it from the StrokeMapper instance. (Keyboard teacher has its own identically-named field.) |

---

## 9. Summary

| Category | Count |
|----------|-------|
| Dead class | 1 (`MotionMode`) |
| Dead methods | 5 (`_s_curve`, `_sine_ease_with_velocity`, `_s_curve_with_initial_velocity`, `_compute_initial_speed_slope`, `_try_load_learning_model`) |
| Dead instance variables | 22 |
| Unused parameters | 2 |
| Stored-but-never-read attribute | 1 (`send_callback`) |
| No-op logic block | 2 |
| **Total dead items** | **~33** |

### Estimated removable lines

Removing all of the above would eliminate roughly **250-280 lines** from the
file's 1 551-line total (~17%).

---

*No code was modified during this audit.*
