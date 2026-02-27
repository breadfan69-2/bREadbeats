# BeatIntelligence Refactor Plan — 2026-02-27

**Source of truth:** today's working tree (1,939 lines)  
**Reference:** BEAT_INTELLIGENCE_WIRING_AUDIT_2026-02-27.md  
**Goal:** Clean dead code, remove disabled paths, simplify structure — no new features.

---

## Current State Summary

`beat_intelligence.py` is 1,939 lines. It grew organically during the port from the old
stroke_mapper monolith and accumulated:
- Dead recovery path (disabled at line 1637, immediately neutralized every frame)
- Tombstone comments for removed gates (§5, §6) that serve no purpose
- `_recovery_radius_bloom` field that is written but never read externally
- `session_intensity` / `session_arc` machinery that is computed but only consumed
  by `keyboard_teacher.py` for CSV capture — not used for any runtime behavior
- Overly defensive `is_recovering` state machine that was disabled during testing and
  never re-enabled — the code sets it True then immediately sets it False 2 lines later

---

## Phase 0: Dead Code Removal (do first, no behavior change)

These removals are guaranteed safe because the wiring audit confirms no external consumer.

### 0.1 Remove disabled recovery path
The `is_recovering` state machine in `build_decision()` does this every frame:
```python
# line 1607-1617: sets is_recovering = True, _was_silence_active = False
# line 1639-1640: DISABLED FOR TESTING — immediately sets is_recovering = False
```
Net effect: `is_recovering` is **always False** at the point where it matters.
The entire recovery path in `update_journey_progress()` (lines 1375-1387, 1410-1415)
is dead code that can never execute in production.

**Remove:**
- `self.is_recovering` field
- `self._was_silence_active` field
- `self._recovery_radius_bloom` field
- `recovery_start` variable in `build_decision()`
- `force_start` parameter from `update_journey_progress()` (always False)
- Recovery-specific branches in `update_journey_progress()` (lines 1375-1387, 1410-1415)
- The `_was_silence_active` / `is_recovering` block in `build_decision()` (1607-1617)
- The disabled-for-testing block (1637-1640)
- `"gs_is_recovering"` from `snapshot_gate_state()`

**Test impact:** If any test asserts `is_recovering`, update it. The field was always False.

### 0.2 Remove session_intensity computation (keep field for now)
`session_intensity` is computed via `_session_intensity_ema` and returned in `BeatDecision`.
`keyboard_teacher.py` (line 88) captures `session_intensity` from decisions into CSV
(`dec_session_intensity` column), and `test_keyboard_teacher.py` mocks it.
However, **no runtime behavior** depends on the value — it is purely telemetry.

**Remove:**
- `self._session_intensity_ema` field
- Session arc EMA block in `build_decision()` (~line 1582-1585)
- `session_arc_enabled` / `session_arc_ema_alpha` config reads
- `"gs_session_intensity"` from `snapshot_gate_state()`

**Keep (or default):**
- `session_intensity` field on `BeatDecision` — set to `energy_fullness` (or 0.5)
  so `keyboard_teacher.py` doesn't break. Can be fully removed in a follow-up
  that also updates `keyboard_teacher.py` and its test.
- `energy_fullness` — that IS consumed by stroke_mapper.

### 0.3 Remove tombstone comments
Lines 768-770 are empty section markers for gates that were removed long ago:
```python
# ── Phase 3 §5: Low-band fullness gate (removed) ───────────────────
# ── Phase 3 §6: Dual-band dB gate (removed) ──────────────────────
```
And line 1137-1138:
```python
# _strict_bass_motion_allowed removed — gate was disabled by default
# and is no longer part of the gate chain.
```
Delete these — they have no informational value in the current codebase.

### 0.4 ~~Remove~~ Keep `_lazy_glide_active` on BeatDecision
`self._lazy_glide_active` is set inside `update_journey_progress()` and exposed
on `BeatDecision.lazy_glide_active`.

**Correction:** `stroke_mapper.py` line 954 reads `decision.lazy_glide_active` in
`_is_upcoming_beat_expected()` to suppress anticipation motion during lazy-glide
tails — this is a **live, load-bearing read**. `keyboard_teacher.py` line 87 also
captures it for CSV telemetry.

**Action:** Keep both the internal logic AND the `BeatDecision` field. No removal.

---

## Phase 1: Structural Simplification (safe refactors, same behavior)

### 1.1 Flatten the gate chain
The beat-family gate chain in `build_decision()` (lines ~1686-1716) tests three
gates in sequence: `stroke_ready`, `flux_drop`, `spectral_fill`. The
"gate-fail-preserve" logic that lets a running journey survive N consecutive
failures is correct and load-bearing — **keep it**.

But the branching is deeply nested. Extract a helper:

```python
def _evaluate_beat_gates(self, event, raw_trigger_kind, stroke_ready, flux_drop) -> tuple[str, str]:
    """Returns (trigger_kind, gate_fail_reason)."""
```

This makes `build_decision()` read as a flat pipeline instead of nested if/elif.

### 1.2 Extract phrase commitment to a method
The phrase commitment block (~lines 1720-1795) is ~70 lines of self-contained
state machine logic. Extract to:

```python
def _update_phrase_commitment(self, trigger_kind, silence_active, gate_fail_reason, is_beat_event) -> str:
    """Returns possibly-overridden trigger_kind."""
```

### 1.3 Consolidate `_event_rms_db` / `_coerce_amplitude_db` / `get_overall_amplitude`
Three methods that all convert amplitude to dBFS with slightly different input
assumptions. Currently called from different spots. Unify into one robust
converter with clear semantics:
- `_to_db(value) -> float` — handles None, 0, linear 0-1, raw dBFS, NaN/Inf

### 1.4 Remove `_get_mid_bass_activity` / `_get_mid_band_activity` trivial wrappers
These are one-line methods that just return `self.energies.low_mid` or
`self.energies.mid` clamped to 0-1. They're only called from
`_populate_rolling_deques()`. Inline them.

**Note:** `test_phase1_foundations.py` line 201 directly calls `_get_mid_bass_activity`,
so that test must be updated when inlining.

---

## Phase 2: Contract Tightening

### 2.1 Make `audio_engine` access explicit
Currently `build_decision()` and helpers reach into `audio_engine` for:
- `get_spectrum()` — fill gate
- `get_band_energies()` / `_band_energies` — band update
- `silence_gate_active` — feedback write
- `predicted_next_beat_mono` — journey timing
- `_metronome_bpm` / `_metronome_phase` — beat timing

Create a thin `AudioSnapshot` dataclass populated at the top of `build_decision()`:
```python
@dataclass
class AudioSnapshot:
    spectrum: np.ndarray | None
    band_energies: dict[str, float]
    predicted_next_beat_mono: float
    metronome_bpm: float
    metronome_phase: float
```
Pass it down instead of `self.audio_engine` everywhere. This makes the data
flow explicit and removes direct coupling to AudioEngine internals.

The **one remaining side effect** (writing `silence_gate_active` back) stays at
the top level of `build_decision()`. Note: AudioEngine already exposes
`set_silence_gate()` (line 1444); the current code at line 1601 prefers this
method with a fallback to direct attribute write.

### 2.2 Remove `set_audio_engine()` per-frame call
`StrokeMapper` calls `self._intelligence.set_audio_engine(self.audio_engine)` every
frame before `build_decision()`. Since the audio engine never changes after
construction (confirmed by wiring audit), remove the per-frame call and require
it at construction only.

### 2.3 Type-annotate `BeatEvent` access (staged rollout)
Replace `getattr(event, "field", default)` patterns with a typed protocol
or explicit attribute access, but do it in small, subsystem-focused passes so
regressions localize quickly.

Proposed order:
1. Silence + envelope inputs
2. Readiness + tempo unlock hold inputs
3. Gate chain inputs (flux/fill/spectrum)
4. Journey/phrase timing inputs

For each pass:
- convert only that subsystem’s field access,
- run the phase contract tests,
- land before starting the next subsystem.

This preserves momentum while avoiding one large ~50-site edit that is harder to
debug if behavior shifts.

Suggested per-pass validation commands:
```bash
# 2.3a — silence/envelope
python -m pytest tests/test_phase1_foundations.py tests/test_phase2_readiness_silence.py -x -q --tb=short

# 2.3b — readiness/unlock-hold
python -m pytest tests/test_phase2_readiness_silence.py tests/test_phase6_bpm_jitter.py -x -q --tb=short

# 2.3c — gates (flux/fill/spectrum)
python -m pytest tests/test_phase3_gates.py tests/test_stroke_mapper_contract.py -x -q --tb=short

# 2.3d — journey/phrase timing
python -m pytest tests/test_phase1_foundations.py tests/test_phase6_bpm_jitter.py tests/test_stroke_mapper_contract.py -x -q --tb=short

# optional full contract sweep before merge
python -m pytest tests/test_phase1_foundations.py tests/test_phase2_readiness_silence.py \
  tests/test_phase3_gates.py tests/test_phase5_learning.py tests/test_phase6_bpm_jitter.py \
  tests/test_stroke_mapper_contract.py -x -q --tb=short
```

---

## Phase 3: Test Validation

### 3.1 Run existing phase tests after each removal
```
python -m pytest tests/test_phase1_foundations.py tests/test_phase2_readiness_silence.py \
  tests/test_phase3_gates.py tests/test_phase5_learning.py tests/test_phase6_bpm_jitter.py \
  tests/test_stroke_mapper_contract.py -x -q --tb=short
```

### 3.2 Add missing coverage
- Phrase commitment: no dedicated test exists — add one that verifies:
  - fill→beat transition locks for 8 beats
  - flux crash cancels early
  - renewal extends on sustained energy
- `snapshot_gate_state()`: verify all keys present and types correct

---

## Execution Order

| Step | What | Risk | Lines removed |
|------|------|------|--------------|
| 0.1 | Dead recovery path | Zero — always-False path | ~40 |
| 0.2 | Session_intensity computation | Low — keyboard_teacher captures field | ~10 |
| 0.3 | Tombstone comments | Zero | ~6 |
| 0.4 | Keep lazy_glide (has live consumer) | N/A | 0 |
| 1.1 | Extract gate chain helper | Low — same logic | 0 (net) |
| 1.2 | Extract phrase commitment | Low — same logic | 0 (net) |
| 1.3 | Consolidate dB converters | Low | ~15 |
| 1.4 | Inline trivial activity wrappers | Zero | ~12 |
| 2.1 | AudioSnapshot dataclass | Medium — touches many methods | 0 (net) |
| 2.2 | Remove per-frame set_audio_engine | Low | ~5 |
| 2.3a | Type BeatEvent access: silence/envelope | Low | 0 (net) |
| 2.3b | Type BeatEvent access: readiness/unlock-hold | Low | 0 (net) |
| 2.3c | Type BeatEvent access: gates (flux/fill/spectrum) | Medium | 0 (net) |
| 2.3d | Type BeatEvent access: journey/phrase timing | Medium | 0 (net) |

**Estimated net reduction:** ~60-70 lines of dead code removed in Phase 0.
**Estimated structural improvement:** `build_decision()` is currently ~288 lines
(lines 1570-1857) and would shrink to ~140 after extracting helpers in Phase 1.

---

## What NOT to Change

These are load-bearing and confirmed live by the wiring audit:

1. `build_decision()` call ordering (band → envelope → flux → silence → gates → journey)
2. `silence_gate_active` feedback write to AudioEngine (via `set_silence_gate()` method)
3. `BeatDecision` field names consumed by StrokeMapper (`trigger_kind`, `interval_beats`,
   `radius_bloom`, `silence_active`, `journey_completion`, `silence_fade`,
   `post_silence_ramp`, `lazy_glide_active`, `gate_fail`, `energy_fullness`)
4. `snapshot_gate_state()` — consumed by keyboard teacher
5. `configure_learning()` API — called from StrokeMapper
6. Phrase commitment logic — prevents beat/fill thrash
7. Adaptive lead pipeline-latency compensation
8. Auto-fill adaptation EMA (targets 58% pass rate)
9. Transient motion profile (kick/hat detection)
10. BPM stabilization (last-locked memory + jump limiter + EMA smooth)
