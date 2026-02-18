# Porting Decisions — What NOT to Hard-Port (and What To Do Instead)

Date: 2026-02-17  
Primary input: `BEAT_INTELLIGENCE_PORT_AUDIT_v2_2026-02-17.md`  
Reference source behavior: commit `ab2b35a` (`stroke_mapper.py` monolith)

---

## Executive Summary

The audit is directionally correct: several legacy behaviors are missing and materially affect motion quality.  
However, **not all missing items should be copied 1:1** into the new split architecture (`beat_intelligence.py` + `stroke_mapper.py` + `audio_engine.py` + UI/config).

### Principle

Hard-port only what is:
1. Pure local state logic,
2. Independent of old monolith coupling,
3. Still semantically valid with current signal model.

Adapt/rebuild (do not hard-port verbatim) what is:
1. Tightly coupled to old method ordering,
2. Dependent on old side effects or hidden invariants,
3. Mixing control policy with rendering/output concerns,
4. Based on old feature semantics that no longer match current runtime fields.

---

## Architecture Delta (Why direct copy can break behavior)

### Old (`ab2b35a`)

Single `StrokeMapper.process_beat()` contained:
- signal tracking,
- beat gating,
- traffic-light readiness,
- learning inference,
- mode switching,
- silence fade + tempo reset,
- command shaping.

All of those shared mutable state and implicit call ordering.

### Current

Responsibilities are split:
- `audio_engine.py`: beat detection / metric states / lock confidence / auto-adjust,
- `beat_intelligence.py`: gating + decision model,
- `stroke_mapper.py`: trajectory and command mapping,
- `main.py`: UI wiring and metric toggles.

**Implication:** old methods that depended on being called in one giant function can misbehave if pasted directly.

---

## Classification of Audit Items

## Safe to hard-port (mostly as-is)

These are low-coupling and should be ported directly with minimal adaptation:

- #1 Rolling deques
- #2 FluxTracker primitives
- #3 Recent-beat hierarchy guards (core logic)
- #4 No-beat timeout → park decay
- #7 Mid-trigger block helper
- #21 High/low/mid activity helper methods

These improve functionality with low architectural risk.

## Should NOT be hard-ported verbatim

The following items should be adapted, not copied line-for-line:

- #5 `_is_low_band_full_enough`
- #6 `_passes_dual_band_db_gate`
- #8 `_get_spectrum_fill_ratio` + `_passes_overall_amp_fill_gate`
- #9 `_update_motion_mode`
- #10–12 learning pipeline pieces (`_build_runtime_feature_values`, `_predict_learning_targets`, `_try_load_learning_model`)
- #13 `_cap_bpm_to_last_locked` + `_stabilize_unlocked_bpm`
- #14 post-silence volume ramp
- #15 `_update_bass_jitter_drive`
- #16 flux-drop creep fallback guard
- #17 `_update_stroke_readiness`
- #18 `_update_learning_adapter`
- #19 silence fade-out tracker
- #20 auto-fill adaptation controller

Details below.

---

## Item-by-item: do not hard-port and replacement plan

## #17 `_update_stroke_readiness` (CRITICAL)

### Why not hard-port verbatim

Old code depended on monolithic timing and direct inspection of traffic-light internals from inside mapper flow. In the split architecture, readiness is now consumed in `beat_intelligence.py` while metric state ownership lives in `audio_engine.py` and toggles in `main.py`.

Copying old code risks:
- duplicate readiness truth sources,
- contradictory state transitions with `_tempo_ready_for_motion`,
- hidden dependence on old per-frame invocation ordering.

### What to do instead

Implement a **dedicated readiness state machine** inside `BeatIntelligence`:

- Inputs:
  - `event.tempo_locked`, `event.acf_confidence`, effective BPM,
  - optional `audio_engine.get_metric_states()` snapshot injected as dependency.
- Internal state:
  - green/yellow history, grace timer, block streak, recent-ready timer.
- Output:
  - single boolean `stroke_ready`, plus reason code for diagnostics.

Keep `_tempo_ready_for_motion()` as a lightweight wrapper around this state machine.

### Functional parity goal

Replicate outcomes (hysteresis + grace) without replicating old call graph.

---

## #18 `_update_learning_adapter` (LEARNING)

### Why not hard-port verbatim

Old method blended prediction outputs directly into monolith-local knobs that no longer exist with the same semantics. Direct port would couple learning internals to outdated pacing and sync multipliers.

### What to do instead

Create a **LearningAdapter module** with explicit API:

- Input: normalized feature payload + confidence + mode context,
- Output struct: `divisor_hint`, `radius_mult`, `lead_ms`, `sync_size_mult`, `sync_speed_mult`, `gate_bias`.

Then let `BeatIntelligence` consume this output as optional modifiers at well-defined integration points.

### Functional parity goal

Keep runtime learning influence, but via explicit typed outputs and bounded blending.

---

## #20 auto-fill adaptation controller

### Why not hard-port verbatim

Old controller updates were interwoven with old gate execution timing and near-silence heuristics embedded in a single function. In split code, those updates can fire at different cadence and create offset drift.

### What to do instead

Implement auto-fill adaptation as a **standalone policy object**:

- Phase-keyed state (`beat/downbeat/syncopation`),
- Called only on discrete trigger evaluations,
- Explicit update contract: `(phase, gate_passed, context)`.

Use bounded offsets and pause updates when silence/tempo-unready.

### Functional parity goal

Preserve adaptive pass-rate targeting, avoid frame-rate/order dependence.

---

## #19 silence fade-out tracker + tempo reset

### Why not hard-port verbatim

Old fade tracker was intertwined with command-level fade application (`_apply_fade`) and direct tempo reset side effects. Current volume behavior is split between decision flags and mapper output.

### What to do instead

Split into two explicit mechanisms:

1. **SilenceDecayState** in `BeatIntelligence`:
   - tracks prolonged silence confidence,
   - emits `silence_fade` scalar 0..1,
   - emits `request_tempo_reset` event when threshold crossed.

2. **Command post-processor** in `StrokeMapper`:
   - applies fade scalar deterministically to output volume/depth.

Tempo reset should be edge-triggered and debounced.

### Functional parity goal

Restore gradual fade and reset behavior without hidden side-effect coupling.

---

## #14 post-silence ramp

### Why not hard-port verbatim

Old ramp assumed the same object owned silence detection and output. That assumption is no longer true.

### What to do instead

Trigger ramp from a clean event: `silence_state` transition `ACTIVE -> INACTIVE`.  
Apply ramp in mapper output stage with explicit duration and multiplier bounds.

### Functional parity goal

Smooth re-entry after silence with deterministic ownership.

---

## #9 `_update_motion_mode`

### Why not hard-port verbatim

The old FULL_STROKE/CREEP mode was entangled with legacy gate stack and trigger cadence rules. Current decision model uses trigger-kind mapping and journey continuity; naive reintroduction can fight that logic.

### What to do instead

Introduce a small **ModeResolver**:

- Input: RMS envelope + amplitude thresholds + dwell bias,
- Output: mode enum used only to influence cadence mapping / gate softness,
- No direct command side effects.

Mode should be advisory and observable in debug telemetry.

### Functional parity goal

Recover amplitude-responsive pacing while preserving current orbital journey logic.

---

## #13 BPM cap/stabilize helpers

### Why not hard-port verbatim

Old helpers were written around monolith-local tempo lifecycle and assumptions about lock transitions. Current lock handling is already split with detector/metric logic in `audio_engine.py`.

### What to do instead

Implement BPM stabilization near tempo source (`audio_engine.py`) or pass stabilized BPM in event payload.  
`BeatIntelligence` should consume a single `effective_motion_bpm` field, not manage lock memory itself.

### Functional parity goal

Avoid duplicated tempo policy and diverging BPM interpretations.

---

## #10–12 Learning model methods (raw port caution)

### Why not hard-port verbatim

The feature names and derivation assumptions in old code were tied to old beat feature payload shape. Directly porting can silently produce poor predictions if feature semantics drifted.

### What to do instead

- Keep loader (`_try_load_learning_model`) but validate schema version and required features.
- Build feature extraction from current canonical event fields.
- Add explicit fallback behavior when features are missing.
- Preserve `cadence_rule` support, but behind validation.

### Functional parity goal

Retain learning capabilities with safe schema-aware compatibility.

---

## #8 spectrum fill ratio + overall fill gate

### Why not hard-port verbatim

Old method reached directly into `audio_engine.get_spectrum()` and assumed spectrum normalization/timing semantics. In split architecture, this is high-coupling and vulnerable to analyzer update-rate mismatch.

### What to do instead

Expose a compact, precomputed **spectrum summary** from audio engine (e.g., per-band occupancy metrics) in the beat event.  
Gate logic should consume these stable summaries rather than raw FFT arrays.

### Functional parity goal

Keep spectral fullness gating with deterministic data contracts.

---

## #6 dual-band dB gate and #5 low-band fullness gate

### Why not hard-port verbatim

Old versions mixed historical deques, per-event fallbacks, and specific dB assumptions that may not match current normalized energy scales.

### What to do instead

Re-implement gate criteria with calibration layer:

- normalize all energy comparisons to one canonical scale,
- centralize thresholds in config with migration defaults,
- keep deque-based occupancy checks,
- add debug counters for pass/fail reasons.

### Functional parity goal

Equivalent selectivity without scale mismatches.

---

## #16 flux-drop creep fallback guard

### Why not hard-port verbatim

Old guard depended on legacy mode state transitions and specific window slicing assumptions.

### What to do instead

Implement as a **guard rule** in decision stage:

- evaluate drop ratio from normalized low-band rolling stats,
- only downgrade trigger confidence/mode when no recent confirmed beats,
- never force hard state jumps mid-journey.

### Functional parity goal

Protect against false sustained motion after bass collapse, without abrupt discontinuity.

---

## #15 `_update_bass_jitter_drive`

### Why not hard-port verbatim

This is effectively a future/output-path concern and currently disconnected from the active command protocol.

### What to do instead

Keep as optional placeholder behind feature flag, with no runtime influence until downstream path exists.

### Functional parity goal

No regression; avoid adding dead complexity now.

---

## Recommended Replacement Architecture (functional equivalence)

## 1) Decision state package in `BeatIntelligence`

Add explicit state blocks:
- `ReadinessState`
- `SilenceState`
- `GateHistoryState`
- `AdaptiveFillState`

All transitions happen in one deterministic update order.

## 2) Data contract from `audio_engine`

Expose stable event fields instead of live pulls:
- `effective_motion_bpm`
- metric traffic snapshot
- precomputed spectrum occupancy summaries

## 3) Mapper post-processing stage

`StrokeMapper` should only:
- map decision → trajectory,
- apply post-process scalars (fade/ramp) from decision payload,
- avoid owning detector policy.

## 4) Learning adapter boundary

Learning outputs must be bounded modifiers, never direct command writers.

---

## Migration Policy: hard-port vs adapt-port matrix

| Category | Policy |
|---|---|
| Deques / local counters / simple helpers | Hard-port with minor naming changes |
| Multi-system state machines | Adapt-port into dedicated component |
| Methods that read live FFT directly | Replace with summarized engine contract |
| Methods with side effects across tempo + output | Split into event + post-process |
| Learning inference and blending | Adapt-port with schema/version validation |

---

## Functional acceptance criteria (to prove parity without hard-porting)

1. **Readiness hysteresis:** brief confidence dips do not immediately kill motion.  
2. **Silence behavior:** prolonged silence fades smoothly and can request tempo reset once per silence episode.  
3. **Re-entry:** audio return ramps in over configured duration.  
4. **Gates:** low-band/high-band/fill gates produce stable pass-rate ranges across at least two contrasting tracks.  
5. **Learning:** missing/partial model fields fail safe (no crash, bounded defaults).  
6. **Trajectory continuity:** guard/fallback logic does not introduce hard discontinuities.

---

## Implementation sequence aligned to this decision

1. Hard-port safe foundations (#1, #2, #3, #4, #7, #21).  
2. Add readiness and silence state machines (adapted #17, #19, #14).  
3. Rebuild gate family with canonical scale contract (adapted #5, #6, #8, #16, #20).  
4. Add mode resolver (adapted #9).  
5. Add learning adapter boundary (adapted #10, #11, #12, #18).  
6. Keep bass jitter behind feature flag (adapted #15).  
7. Consolidate BPM stabilization near detector source (adapted #13).

---

## Bottom line

The audit’s missing-functionality findings are correct.  
The safest path is **hybrid porting**:
- hard-port low-coupling primitives,
- adapt-port high-coupling policy/state machines.

That preserves behavior while respecting the current split architecture and avoids reintroducing monolith fragility.
