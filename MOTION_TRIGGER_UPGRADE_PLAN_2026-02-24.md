# Motion Trigger Upgrade Findings & Plan (2026-02-24)

## 1) Executive Summary

This document consolidates:
- deep external research on waveform/signal-change detection,
- your existing `AUDIOFLUX_INTEGRATION_FINDINGS_2026-02-24.md`,
- and current in-repo architecture in `audio_engine.py` and `beat_intelligence.py`.

### Primary recommendation
Keep your current real-time ownership model (multi-band z-score + raw onset + ACF/metronome + downbeat sync logic), and upgrade it to a **confidence-fused multi-cue trigger stack** instead of relying mainly on binary OR logic between classic threshold and z-score paths.

### Why this is the best path
- Preserves your proven low-latency behavior and sync logic.
- Improves sensitivity to subtle changes without increasing false triggers in silence.
- Adds explicit **bass-vs-treble dominance logic** and **transient class awareness** (kick-like vs hat-like behavior).
- Strengthens tempo stability through confidence-weighted fusion and stricter jump governance.
- Fits your plan to split `audio_engine` into modules without risky full rewrites.

---

## 2) Goals (from user requirements)

You asked for:
1. **More sensitivity / earlier notice of signal changes**.
2. **Bass-aware behavior** (when bass notes are stronger than treble).
3. **Beat pickup from both hi-hat and kick drum**.
4. **More stable tempo recognition**.
5. **No code changes yet**; produce detailed findings + plan for module refactor.

This plan addresses all five explicitly.

---

## 3) Current System Strengths (what to keep)

From `audio_engine.py` + `beat_intelligence.py`:

1. **Streaming FFT + low-latency callback path already robust**
   - Efficient callback in `_audio_callback_pyaudio`
   - Existing spectrum/band extraction + flux + energy tracking

2. **Multi-band adaptive peak detection exists today**
   - Per-band z-score detectors (`sub_bass`, `low_mid`, `mid`, `high`)
   - Primary-band switching with hysteresis
   - Fired-bands telemetry passed in `BeatEvent`

3. **Tempo subsystem is advanced**
   - ACF onset buffer tempo estimation
   - Onset-tempo fusion into metronome BPM
   - PLL-like phase nudging and downbeat pattern checks
   - Tempo lock semantics + confidence propagation to decisions

4. **Motion policy layer already has sophisticated gating**
   - Low-band fullness gating
   - Dual-band dB gating
   - Spectrum fill gate and auto-adjust offsets
   - Readiness/hold systems to prevent cadence chaos

5. **Good test foothold exists**
   - `tests/test_phase6_bpm_jitter.py` validates BPM stabilization behavior.

These are excellent foundations; the upgrade should be an augmentation, not replacement.

---

## 4) Key Gaps vs Desired Behavior

## Gap A: Signal-change sensitivity is still conservative in some transitions
Current beat decision in `_detect_beat` is still dominated by:
- classic threshold checks,
- per-band z-score spikes,
- and a single energy sanity multiplier.

Result: misses can occur in nuanced transitions (texture shifts, tonal attacks, gentle section changes).

## Gap B: Bass-vs-treble dominance is not explicit enough in trigger fusion
You track low/high histories in `BeatIntelligence`, but trigger ownership is not yet using a dedicated **bass dominance index** and **transient type confidence** as first-class fusion signals.

## Gap C: Hi-hat vs kick adaptation is present but can be more immediate
Primary band selection currently relies heavily on rolling fire-history score. That is stable, but can lag at fast arrangement changes (kick enters suddenly after hat-led passage).

## Gap D: Tempo stability is strong but still vulnerable to octave/jump edge cases
ACF + fusion is already good, but further improvements can come from:
- stronger confidence-conditioned update gates,
- stricter consistency windows before adopting major BPM shifts,
- and drift detector hooks for section-change handling.

---

## 5) External Findings Applied to This Project

## 5.1 Onset / change detectors most relevant
From librosa, aubio, madmom, Essentia references:
- **Spectral flux**: baseline novelty curve for transients.
- **SuperFlux-style novelty**: vibrato/tremolo suppression improves onset robustness.
- **Complex/phase-domain onset cues**: better for tonal/polyphonic changes.
- **HFC (High Frequency Content)**: strong cue for hi-hat/cymbal transients.
- **RMS/energy derivative**: catches broader rises where transients are soft.

Practical use here: combine these as **weighted cues**, not independent trigger owners.

## 5.2 Tempo stability methods relevant
- ACF periodicity (already in place).
- Dynamic fusion of onset-interval tempo + ACF tempo by confidence (already partially in place).
- Stronger anti-octave acceptance windows.
- Optional online drift/change detection (ADWIN/Page-Hinkley) to detect regime shifts and temporarily relax/tighten update rules.

## 5.3 Change-point detection role
`ruptures` is excellent offline for section segmentation and analysis; for real-time use, prefer lightweight online detectors (Page-Hinkley/ADWIN) on compact features.

## 5.4 audioflux role (confirmed by your findings doc)
Strong fit as **optional sidecar feature enrichment**:
- `entropy`, `flatness`, `hfc`, `novelty`, `rms`, optional onset confidence.
- Do not replace core metronome/downbeat engine.
- Keep callback-safe optional import and fail-open behavior.

---

## 6) Target Upgrade Architecture (No code yet)

Design principle: **separate signal extraction from decision policy**.

## Proposed module boundaries

1. `signal_frontend`
- Audio buffering, mono mix, windowing, FFT staging.
- No trigger logic.

2. `feature_extractors`
- Base features: band energies, flux, RMS, deltas, HFC proxy, centroid/rolloff/flatness.
- Optional sidecar: `audioflux_adapter` (rate-limited, fail-open).

3. `tempo_tracker`
- ACF estimator, onset-tempo estimator, fusion, PLL phase, downbeat pattern validation, lock confidence.

4. `event_detector`
- Multi-cue confidence fusion to produce beat/downbeat/syncopation events.
- Refractory, hysteresis, prominence constraints.

5. `motion_policy`
- Existing `BeatIntelligence` gating/policy and journey logic.
- Consumes event confidence and feature context, does not touch raw DSP.

6. `telemetry_tuning`
- Metrics, counters, pass/fail rates, jitter stats, hold-state dwell, calibration suggestions.

---

## 7) Trigger Fusion Upgrade (core technical plan)

## 7.1 Replace binary OR with weighted confidence score
Current:
- `is_beat = classic_beat or zscore_beat`

Target:
- `beat_score = Σ w_i * cue_i`
- cues include:
  - novelty cue (flux/SuperFlux-like)
  - band-spike cue (per-band zscore strength)
  - energy derivative cue (`ΔRMS`, `Δband_energy`)
  - phase-alignment cue (distance to metronome beat boundary)
  - optional sidecar onset confidence

Then trigger with:
- high threshold to arm,
- lower threshold to sustain/release (hysteresis),
- prominence + minimum inter-onset distance.

## 7.2 Add dual-timescale onset channels
- **Fast channel** (~high band transients): catches hats/clicks.
- **Slow channel** (~low band envelope): catches kick/bass body.

Final trigger confidence uses max/weighted blend depending on dominance context.

## 7.3 Add explicit transient classification
Per frame / near peak:
- `kick_like_confidence`
- `hat_like_confidence`
- `mixed_confidence`

Derived from:
- band energy ratios,
- HFC vs low-band envelope,
- attack slope and crest.

Usage:
- kick-like events can drive deeper/larger motion.
- hat-like events still count as beat cues with lighter motion weighting.

---

## 8) Bass-vs-Treble Dominance Plan

Define a stable dominance metric:

- `bass_energy = sub_bass + low_mid`
- `treble_energy = high + k*mid` (small mid contribution)
- `bass_dominance = bass_energy / (treble_energy + ε)`

Then use it in 3 places:

1. **Trigger cue weighting**
- If `bass_dominance` high: increase low-band channel weight.
- If low: allow high-band transients to dominate cue weighting.

2. **Primary source arbitration**
- Keep existing primary-band hysteresis, but allow immediate override when instantaneous confidence gap is large enough.

3. **Motion mapping context**
- Bass-dominant beats scale depth/radius more.
- Treble-dominant beats can keep cadence but with restrained depth.

This directly satisfies “process when bass notes are stronger than treble” while still respecting hi-hat-driven timing.

---

## 9) Tempo Stability Upgrade Plan

## 9.1 Keep existing ACF + onset fusion foundation
Do not remove current metronome ownership model.

## 9.2 Strengthen BPM adoption rules
- Apply confidence-weighted jump limits before committing new BPM.
- Require a short consistency window for large BPM deltas.
- Maintain lock-hold during brief confidence dips (already present conceptually).

## 9.3 Improve octave ambiguity handling
- Evaluate 1x/2x/0.5x candidates against:
  - recent lock BPM,
  - phase error trend,
  - beat interval consistency.
- Only switch octave class when confidence and consistency jointly agree.

## 9.4 Add section-shift awareness
- Optional online drift detector over compact features (`flux`, `RMS`, `band ratios`).
- On detected regime shift:
  - temporarily relax stale-lock assumptions,
  - speed adaptation for 1–2 bars,
  - then re-tighten stability guards.

---

## 10) audioflux Sidecar Plan (Aligned with your findings)

Use audioflux only as optional enrichment.

## Runtime sidecar fields (first rollout)
- `af_entropy`
- `af_flatness`
- `af_hfc`
- `af_novelty`
- `af_rms`
- `af_onset_conf` (if available)

## Runtime behavior
- lazy import;
- stride/rate-limited compute (not every callback);
- bounded ring buffer;
- fail-open fallback (no crash, no startup break).

## Usage in decisions
- contribute to confidence score in borderline states only;
- never hard-own beat firing in initial rollout.

## Offline use
- enrich `local_learning` extraction and retrain rules.

---

## 11) Refactor Plan (staged, no code yet)

## Stage 0 — Interface freeze (design-only)
Deliver module contracts first:
- data classes for feature frame, tempo state, event confidence, motion context,
- clear dependency direction (`event_detector` depends on `feature_extractors` + `tempo_tracker`, not vice versa).

## Stage 1 — Extract pure feature functions
Move non-stateful calculations first (band stats, deltas, novelty helpers) into `feature_extractors`.
No behavior changes.

## Stage 2 — Isolate tempo tracker
Lift ACF/metronome/downbeat state machine into `tempo_tracker` module.
Preserve existing outputs and thresholds.

## Stage 3 — Introduce fusion layer behind feature flag
Implement `beat_score` path in parallel to legacy `_detect_beat`.
A/B compare both outputs through telemetry before switching default.

## Stage 4 — Add audioflux sidecar (optional)
Integrate sidecar adapter and inject optional cues into fusion only.
Fail-open on import/runtime issues.

## Stage 5 — Policy-layer tuning
Map transient class + bass dominance into motion policy with conservative clamps.
Keep existing gate hierarchy.

---

## 12) Validation & Acceptance Criteria

## Functional invariants
- With new features disabled, behavior must match baseline.
- Missing `audioflux` must not affect startup or runtime stability.

## Detection quality metrics
- Lower miss rate on low-amplitude transitions.
- Better hi-hat-only tracking without overfiring.
- Better kick-entry response after treble-led sections.

## Tempo metrics
- Reduced BPM jitter (frame-to-frame variance).
- Faster relock after section changes.
- Fewer octave flip events per minute.

## Runtime safety
- No callback overruns/dropouts attributable to new path.
- Sidecar compute budget:
  - mean < 2 ms,
  - p95 < 5 ms.

## Suggested A/B test sets
- kick-heavy EDM
- hat-driven techno
- bass-light acoustic
- dense polyphonic mix
- low-level ambient + sudden transitions

---

## 13) Risk Register & Mitigations

1. **Over-sensitivity / chatter**
- Mitigate with hysteresis, prominence, refractory constraints, and silence veto priority.

2. **Tempo instability from extra cues**
- Keep tempo ownership in `tempo_tracker`; event detector consumes tempo, not vice versa.

3. **audioflux packaging risk on Windows/PyInstaller**
- Optional import and freeze-safe fallback are mandatory.

4. **Refactor regressions during module split**
- Extract in small steps with golden-output telemetry + existing BPM jitter tests.

---

## 14) Immediate Next Deliverables (design artifacts)

1. Module interface spec document (types + responsibilities).
2. Event fusion math spec (cue definitions, normalization, thresholds).
3. Migration checklist mapping old symbols -> new modules.
4. A/B telemetry schema for side-by-side detector comparison.
5. Tuning playbook (safe default ranges and rollback points).

---

## 15) Final Recommendation

Proceed with a **modular confidence-fusion upgrade**:
- keep core real-time beat/metronome/downbeat architecture,
- add multi-cue sensitivity and transient-class awareness,
- add explicit bass-dominance logic,
- strengthen tempo adoption rules,
- integrate audioflux as optional sidecar enrichment.

This gives the highest upside for your requested behavior with the lowest risk to your current stable system.
