# Channel-Bus Isolation Implementation Plan (2026-02-24)

## 1) Purpose

This document defines how to implement **frequency channel-bus isolation** to prevent cross-band interference in beat detection and motion policy.

Primary objective:
- Split processing into independent buses (`sub_bass`, `low_mid`, `mid`, `high`) with per-bus state.
- Fuse bus outputs only at decision time with explicit weighting.

---

## 2) Core Isolation Model

## 2.1 Core idea
Each bus has its own runtime state and detector lifecycle:
- own envelope
- own noise floor
- own z-score history
- own thresholds
- own refractory timer

No bus writes directly into another bus state.

## 2.2 Isolation rules
1. **Per-bus detectors**
   - Independent statistics and refractory guards.
2. **Per-bus gating**
   - A bus contributes only if it passes its own onset + sustain checks.
3. **Cross-bus bleed suppression**
   - Soft masks prevent high-band transients from inflating bass confidence and vice versa.
4. **Fusion-stage only coupling**
   - Global score combines bus decisions via explicit weights and dominance logic.
   - No shared mutable thresholds between buses.

---

## 3) Data Contracts

Add the following to `audio_modules/contracts.py`.

## 3.1 `BusState`
Per bus mutable runtime state:
- `name: str`
- `env: float`
- `noise_floor: float`
- `z_hist: list[float]` or bounded deque
- `last_onset_mono: float`
- `refractory_ms: float`
- `arm_threshold: float`
- `release_threshold: float`
- `sustain_frames: int`
- `active_frames: int`
- `inactive_frames: int`

## 3.2 `BusDecision`
Per frame per-bus output:
- `name: str`
- `onset_conf: float` (0..1)
- `sustain_conf: float` (0..1)
- `passed_onset_gate: bool`
- `passed_sustain_gate: bool`
- `in_refractory: bool`
- `eligible: bool`
- `reason_codes: list[str]`

## 3.3 `TriggerDecision` extension
Add bus-scoped outputs (non-breaking additions):
- `bus_scores: dict[str, float]`
- `bus_pass: dict[str, bool]`
- `bus_reason_codes: dict[str, list[str]]`

---

## 4) Detector Behavior

Implement in `audio_modules/event_detector.py`.

## 4.1 Per-bus scoring first
For each bus (`sub_bass`, `low_mid`, `mid`, `high`):
1. Compute isolated cues:
   - local flux cue
   - local energy delta cue
   - local z-score spike cue
2. Apply bus-only thresholds and refractory.
3. Emit `BusDecision` + numeric bus score.

## 4.2 Cross-bus bleed suppression (soft masks)
Apply masks before fusion, examples:
- If `high` has sharp transient but `sub_bass` local cues are weak, cap bass bus contribution.
- If bass envelope is dominant and `high` is weak/noisy, damp high-only spike influence.

Suggested form:
- `masked_score = raw_score * mask_factor`
- `mask_factor` in `[0.35, 1.0]` from dominance + coherence conditions.

## 4.3 Fusion stage (single owner)
Global `beat_score` from masked bus scores:

`beat_score = w_sub*sub + w_low*low + w_mid*mid + w_high*high + optional_phase + optional_sidecar`

Rules:
- Weights are explicit, config-driven.
- Dominance logic modifies **weights only**, not other buses’ thresholds.
- Final arm/release hysteresis remains global, but input is bus-isolated.

---

## 5) Integration Points

## 5.1 `audio_modules/feature_extractors.py`
- Ensure stable per-band normalized outputs suitable for isolated bus processing.
- Keep per-band cues independent (no pre-fused band score returned to detector).

## 5.2 `audio_modules/contracts.py`
- Add `BusState` and `BusDecision`.
- Extend `TriggerDecision` with bus-level diagnostics.

## 5.3 `audio_modules/event_detector.py`
- Introduce internal `self._bus_states: dict[str, BusState]`.
- Compute `bus_scores` first.
- Run bleed suppression masks.
- Compute final `beat_score` from masked bus outputs.

## 5.4 `beat_intelligence.py`
- Map bus-level context to policy (e.g., kick-like vs hat-like confidence).
- Keep policy layer read-only with respect to bus thresholds (no feedback mutation).

---

## 6) Recommended Implementation Order

## Stage 1 — Contracts + plumbing (no behavior flip)
1. Add `BusState` and `BusDecision` dataclasses.
2. Extend `TriggerDecision` with bus debug fields.
3. Keep legacy scoring path as owner.

Exit:
- tests pass
- behavior unchanged when new path disabled

## Stage 2 — Isolated bus scoring in detector (shadow only)
1. Add per-bus state map and bus scoring function.
2. Emit bus diagnostics in shadow telemetry.
3. Do **not** switch ownership yet.

Exit:
- no runtime regressions
- telemetry contains bus pass/fail and masked scores

## Stage 3 — Anti-bleed masking + fusion refinement (shadow only)
1. Add soft masks and dominance-aware weighting.
2. Compare with legacy via replay + live telemetry.

Exit:
- reduced false cross-band promotions in replay metrics

## Stage 4 — Controlled ownership switch
1. Feature-flag enable isolated fusion owner when:
   - `new_trigger_fusion_enabled=True`
   - `new_trigger_shadow_mode=False`
2. Keep rollback instant by flipping flags.

Exit:
- acceptance metrics met (see Section 9)

---

## 7) Test Plan (Anti-Interference Focus)

Add/extend tests in `tests/test_event_detector_fusion.py` and replay harness tests.

Required cases:
1. **High-hat spike does not promote bass bus**
   - High transient only; bass buses must stay below pass threshold.
2. **Kick does not force high bus**
   - Strong bass transient only; high bus must remain low unless high evidence exists.
3. **Mixed transient preserves both appropriately**
   - Legit mixed content passes both with expected weighting.
4. **Refractory is per-bus**
   - Re-trigger block applies only to bus in refractory; other buses remain eligible.
5. **Global score derives from bus fusion only**
   - No direct cross-bus state mutation.

Optional replay assertions:
- fewer false extra-fires during high-hat-only passages
- no increase in missed kick entries

---

## 8) Telemetry / A-B Rollout

In shadow telemetry (`audio_modules/telemetry_tuning.py` + engine wiring), record:
- per-bus raw score
- per-bus masked score
- per-bus pass/fail
- per-bus reason codes
- final fused score
- legacy vs new agreement

Rollout policy:
1. shadow mode first on representative tracks
2. compare misses/extras by content type (kick-heavy, hat-heavy, mixed)
3. switch owner only after stable gains

---

## 9) Acceptance Criteria

1. **Isolation correctness**
   - Bus thresholds/refractory/history are independent.
2. **Interference reduction**
   - Hat-only spikes no longer inflate bass confidence enough to pass bass gates.
3. **No regression in core behavior**
   - silence handling and tempo lock paths remain stable.
4. **Feature-flag safety**
   - immediate rollback to legacy path without code changes.
5. **Test coverage**
   - anti-interference tests are green in CI/local suites.

---

## 10) Execution Checklist

1. Add contracts (`BusState`, `BusDecision`, `TriggerDecision` bus fields).
2. Add detector bus-state initialization/reset.
3. Implement per-bus scoring + per-bus gating.
4. Implement soft bleed suppression masks.
5. Implement final fusion from bus scores.
6. Wire bus telemetry in shadow mode.
7. Add anti-interference tests.
8. Run focused suites:
   - `tests.test_event_detector_fusion`
   - `tests.test_shadow_replay_harness`
   - `tests.test_phase6_bpm_jitter`
   - `tests.test_stroke_mapper_contract`
9. Enable staged A/B runs with flags.

---

## 11) Notes for Safe Adoption

- Keep this as an incremental migration, not a rewrite.
- Preserve existing metronome ownership semantics while validating bus isolation.
- Do not let policy-level modules mutate detector bus thresholds during runtime.
- Favor explicit config parameters over adaptive shared globals.
