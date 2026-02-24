# AUDIOFLUX Integration Findings & Implementation Guide (2026-02-24)

## 1) Executive Summary

`audioflux` is a strong fit for **feature enrichment** in bREadbeats, but not a strong candidate to replace your existing real-time beat/meter core.

Your current pipeline in `audio_engine.py` already provides:
- low-latency streaming FFT + spectral flux
- multi-band z-score onset firing
- ACF tempo estimation + internal metronome phase tracking
- custom downbeat pattern matching and syncopation logic

`audioflux` is most valuable as a sidecar for:
- richer spectral descriptors (entropy/flatness/hfc/novelty/rms)
- optional onset confidence channel
- optional harmonic/percussive separation (HPSS)
- offline feature extraction for learning data improvement

---

## 2) What audioflux provides (relevant subset)

Based on repository/docs review:

### Transform layer
- `BFT`, `CQT`, `VQT`, `CWT`, `ST`, `FST`, etc.
- Multiple scale types: linear, mel, bark, erb, octave, log.

### Feature layer
- `Spectral` family: centroid, spread, skewness, kurtosis, entropy, crest, slope, rms, energy, hfc, sf/sd, novelty, etc.
- `XXCC` family: MFCC/BFCC/GTCC/CQCC style cepstral features.
- `Temporal`: RMS/energy/ZCR in frame domain.

### MIR layer
- `Onset`
- `Pitch` methods (YIN/CEP/PEF/NCF/HPS/LHS/STFT/FFP)
- `HPSS`

### Tooling
- `FeatureExtractor` for batch extraction across transforms.

---

## 3) Fit against current bREadbeats architecture

## Strong fit (use)
1. **Beat-window feature quality boost**
   - Your `BeatEvent.beat_features` generation in `audio_engine.py` can be enhanced with robust descriptors:
   - `flatness`, `entropy`, `hfc`, `novelty`, `spectral mean/variance`, optional `zcr`.
   - Expected value: better adaptive decision quality in `beat_intelligence.py`, especially across genre changes.

2. **Secondary onset confidence (not trigger owner)**
   - Keep your current trigger ownership (`_detect_beat`, z-score, metronome).
   - Add audioflux onset score as a confidence feature for gating/tie-breaks.

3. **Offline learning pipeline upgrade**
   - In `local_learning/`, use audioflux batch extraction to improve rule-fit datasets.
   - Low runtime risk, high model quality upside.

## Medium fit (evaluate)
4. **HPSS pre-conditioning**
   - Potentially improve onset robustness on dense harmonic material.
   - Cost: extra CPU, may be too heavy in callback if run too often.

## Low fit (skip for now)
5. **Replacing metronome/downbeat/phase stack**
   - Your current code is deeply integrated with sync logic and movement timing.
   - Replacement risk is high for minimal guaranteed gain.

---

## 4) Windows + packaging constraints (important)

Your build is PyInstaller-based (`bREadbeats.spec`) and aggressively excludes heavy libs.

Risks:
- `audioflux` is a native-extension package (DLL loading behavior matters).
- Packaging may fail unless dynamic libs and submodules are explicitly included.
- Runtime startup may fail on systems missing bundled DLL dependencies.

Practical guidance:
1. Prefer `pip install audioflux` in dev first; avoid source-build on Windows.
2. Keep integration optional and guarded:
   - if import fails, run current pipeline unchanged.
3. For frozen app, add explicit hidden imports and binaries collection for audioflux libs.
4. Validate both:
   - unfrozen (`python main.py`)
   - frozen (PyInstaller EXE on clean machine).

---

## 5) Recommended integration strategy (phased)

## Phase 0 — Feasibility spike (1–2 days)
Goal: prove import/runtime stability and measure CPU impact.

- Add optional dependency in local dev env.
- Create a sidecar adapter that computes features every N frames (not every callback).
- Log:
  - sidecar compute time (ms)
  - feature availability ratio
  - callback overruns/dropouts

Exit criteria:
- no crashes
- no audible glitches
- < ~2 ms average extra callback time (or run sidecar at reduced cadence)

## Phase 1 — Feature sidecar (safe runtime path)
Goal: enrich `BeatEvent.beat_features` only.

- Add `audioflux_adapter.py` with:
  - lazy import
  - ring buffer for mono samples
  - configurable compute interval + FFT params
  - returns dict of extra features
- In `audio_engine.py`, merge sidecar output into existing `beat_features`.
- No changes to `is_beat` ownership.

## Phase 2 — Confidence fusion (gated)
Goal: improve hard cases with minimal behavior change.

- Add one scalar `audioflux_onset_confidence` to decision logic in `beat_intelligence.py`.
- Use as tie-breaker/soft gate only when baseline confidence is borderline.
- Keep existing lock and safety gates as the source of truth.

## Phase 3 — Offline learning expansion
Goal: improve learned rules without callback cost.

- Extend `local_learning/extract_audio_features.py` to include audioflux descriptors.
- Refit rules and compare outcome metrics.

---

## 6) Concrete code touchpoints

## New file
- `audioflux_adapter.py`
  - Owns optional import and feature extraction state.

## Existing files
1. `config.py`
   - Add toggles:
     - `audioflux_enabled: bool = False`
     - `audioflux_frame_stride: int = 2` (or 4)
     - `audioflux_fft_size: int = 1024`
     - `audioflux_emit_onset_confidence: bool = True`

2. `audio_engine.py`
   - Initialize adapter once in `__init__`.
   - Feed mono frame data in callback.
   - Pull latest extracted features and merge into `beat_features`.

3. `beat_intelligence.py`
   - Read optional fields from `event.beat_features`.
   - Blend into confidence/gating with strict clamps and fallback defaults.

4. `requirements.txt`
   - Add optional dependency note for `audioflux` (or move to optional extras doc).

5. `bREadbeats.spec` / `main.spec`
   - Add hidden imports/collect binaries for `audioflux` during freeze.
   - Keep feature behind config so missing package does not break startup.

---

## 7) Suggested adapter contract

```python
# audioflux_adapter.py
from typing import Optional, Dict

class AudioFluxAdapter:
    def __init__(self, sample_rate: int, fft_size: int, stride: int, enabled: bool):
        ...

    @property
    def available(self) -> bool:
        ...

    def push_audio(self, mono_frame) -> None:
        ...

    def get_latest_features(self) -> Optional[Dict[str, float]]:
        # returns None if not ready/unavailable
        # e.g. {
        #   'af_entropy': ..., 'af_flatness': ..., 'af_hfc': ...,
        #   'af_novelty': ..., 'af_rms': ..., 'af_onset_conf': ...
        # }
        ...
```

Design rules:
- Never raise from callback path.
- If unavailable/error: return `None`, continue baseline pipeline.
- Rate-limit compute (stride/windowing).

---

## 8) Performance guardrails

1. Keep processing decoupled from every callback
   - Compute every `N` frames.
2. Use bounded buffers only.
3. Log per-call cost with moving average.
4. Hard-disable sidecar on repeated exceptions.

Target budgets (initial):
- Mean sidecar compute < 2 ms
- P95 sidecar compute < 5 ms
- No callback starvation/glitches

---

## 9) Validation plan

## Functional
- Verify app behavior unchanged when `audioflux_enabled=False`.
- Verify startup succeeds when `audioflux` is absent.
- Verify feature fields appear when enabled.

## Behavioral
- Compare A/B runs on representative tracks:
  - beat trigger count stability
  - false positives during silence
  - syncopation consistency
  - tempo lock time and drop-outs

## Performance
- Capture callback timing stats before/after.
- CPU delta across low/mid/high complexity audio.

## Packaging
- Build EXE and test on clean Windows machine.
- Confirm optional fallback if DLL load fails.

---

## 10) Recommended feature set for first rollout

Start with these 6 fields only:
- `af_entropy`
- `af_flatness`
- `af_hfc`
- `af_novelty`
- `af_rms`
- `af_onset_conf` (optional)

Why:
- high signal-to-complexity ratio
- directly useful for current gating/learning
- limited CPU footprint relative to larger transforms

---

## 11) Decision summary

Implement now:
- optional audioflux sidecar (Phase 1)
- offline learning feature enrichment (Phase 3)

Defer:
- replacing core beat/metronome/downbeat engine
- heavy transform usage in strict callback hot path

This approach gives measurable quality upside while preserving your current low-latency architecture and minimizing Windows packaging risk.
