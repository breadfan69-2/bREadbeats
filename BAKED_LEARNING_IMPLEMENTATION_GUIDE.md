# Baking rule_fit + Profile into bREadbeats (Always-Active, No User Loading)

## Goal

Ship a release build where the fitted rule model (`rule_fit.json`) and the learning
profile (`profile.release.json`) are compiled directly into the executable and activate
automatically at startup. The end user never sees a load dialog, file picker, or error
about missing files.

---

## Step 1 — Choose and place the release asset files

Create a stable folder inside the repo:

```
defaults/
  learning/
    rule_fit.release.json
    profile.release.json
```

These are the **only** two filenames the runtime will ever look for when loading the
baked defaults. Never glob for newest-file here; use deterministic names.

**Where to get them:**

| File | Source |
|---|---|
| `rule_fit.release.json` | Copy your best trained `rule_fit.json` from `datasets/` or a training run |
| `profile.release.json` | Copy the best `profile*.json` from `D:\breadbeats_datasets\blends\` or wherever you keep snapshots |

Both files are standard UTF-8 JSON. They must be committed to the repo so PyInstaller
can find them.

---

## Step 2 — Add a PyInstaller-safe path resolver

In `main.py`, replace the hard-coded dev path in `_apply_release_learning_defaults()`.
Add a small helper near the top of the method (or as a module-level function) that
resolves the `defaults/learning/` folder in both frozen and source-run contexts:

```python
def _get_bundled_defaults_dir() -> Path | None:
    """Return path to defaults/learning/ whether frozen (PyInstaller) or source run."""
    import sys
    candidates = []

    # 1. PyInstaller frozen bundle
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        candidates.append(Path(sys._MEIPASS) / "defaults" / "learning")

    # 2. Alongside the EXE (user may place files next to exe for override)
    if getattr(sys, 'frozen', False):
        candidates.append(Path(sys.executable).parent / "defaults" / "learning")

    # 3. Repo-relative (dev / source run)
    candidates.append(Path(__file__).resolve().parent / "defaults" / "learning")

    for c in candidates:
        if c.exists():
            return c
    return None
```

---

## Step 3 — Rewrite `_apply_release_learning_defaults()`

Replace the entire body that looks at `D:\breadbeats_datasets\blends` with:

```python
def _apply_release_learning_defaults(self) -> None:
    defaults_dir = _get_bundled_defaults_dir()
    if defaults_dir is None:
        print("[Learning] No bundled defaults folder found — skipping.")
        return

    profile_path = defaults_dir / "profile.release.json"
    rule_fit_path = defaults_dir / "rule_fit.release.json"

    selected_profile  = profile_path  if profile_path.exists()  else None
    selected_rule_fit = rule_fit_path if rule_fit_path.exists() else None

    # --- Apply profile learning config fields (same logic as before) ---
    if selected_profile is not None:
        try:
            payload = json.loads(selected_profile.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[Learning] Failed reading bundled profile: {exc}")
            payload = {}

        if isinstance(payload, dict):
            learning_cfg = payload.get("learning", {}) or {}
            model_cfg    = payload.get("model",    {}) or {}

            bool_keys = {
                "teaching_learning_enabled",
                "teaching_use_fitted_rules",
                "teaching_apply_in_circle_mode",
                "teaching_isolation_mode",
            }
            float_keys = {
                "teaching_learning_strength",
                "teaching_min_confidence",
                "teaching_no_motion_bias",
            }

            for key in bool_keys:
                if key in learning_cfg:
                    setattr(self.config.beat, key, bool(learning_cfg[key]))
            for key in float_keys:
                if key in learning_cfg:
                    try:
                        setattr(self.config.beat, key, float(learning_cfg[key]))
                    except Exception:
                        pass

            # Profile may embed its own rule_fit path — honour it only if it
            # points to an existing file; otherwise fall back to bundled.
            raw_rf = model_cfg.get("rule_fit") or learning_cfg.get("teaching_rule_fit_path")
            if isinstance(raw_rf, str) and raw_rf.strip():
                candidate = Path(raw_rf.strip())
                if not candidate.is_absolute():
                    candidate = selected_profile.parent / candidate
                if candidate.exists():
                    selected_rule_fit = candidate

    # --- Apply paths and force learning on ---
    if selected_profile is not None:
        setattr(self.config.beat, 'teaching_profile_path', str(selected_profile))
    if selected_rule_fit is not None:
        self.config.beat.teaching_rule_fit_path = str(selected_rule_fit)

    self.config.beat.teaching_learning_enabled  = True
    self.config.beat.teaching_use_fitted_rules  = True

    source = "bundled"
    print(
        f"[Learning] Release defaults applied — source={source}"
        f" profile={selected_profile or '(none)'}"
        f" rule_fit={selected_rule_fit or '(none)'}"
    )
```

The rest of startup calls `_apply_learning_config_to_mapper()` which drives
`mapper._try_load_learning_model()` — **but that method must be implemented first**
(see Step 6).

---

## Step 4 — Bundle the assets in `bREadbeats.spec`

Open `bREadbeats.spec` and update the `datas` list:

```python
datas=[
    # ... existing entries ...
    ('defaults/learning/rule_fit.release.json',  'defaults/learning'),
    ('defaults/learning/profile.release.json',   'defaults/learning'),
],
```

Both paths are relative to the spec file (repo root). PyInstaller will copy them into
the `_MEIPASS` temp folder so `_get_bundled_defaults_dir()` finds them at runtime.

---

## Step 5 — Verify `_apply_learning_config_to_mapper()` is called at startup

In `__init__` (around line 3300 in `main.py`) these two calls already exist:

```python
self._apply_release_learning_defaults()  # must come first
self._apply_learning_config_to_mapper()  # pushes paths into StrokeMapper
```

`_apply_learning_config_to_mapper()` sets `mapper._learning_rule_fit_path` and then
calls `mapper._try_load_learning_model()`. That method **does not currently exist** on
the refactored `StrokeMapper` — it was removed during the decision-only refactor. It
must be re-implemented (Step 6) before the path wiring has any effect.

---

## Step 6 — Implement `_try_load_learning_model()` in `StrokeMapper`

The refactored `StrokeMapper` (post-autopsy) has no model loader. Add this method:

```python
def _try_load_learning_model(self) -> None:
    """Load rule_fit.json from _learning_rule_fit_path into _rule_fit_model."""
    import json
    path_str = getattr(self, '_learning_rule_fit_path', '')
    if not path_str:
        self._rule_fit_model = None
        return
    try:
        from pathlib import Path
        data = json.loads(Path(path_str).read_text(encoding='utf-8'))
        if data.get('status') != 'ok':
            raise ValueError(f"rule_fit status not ok: {data.get('status')}")
        self._rule_fit_model = data
        features = data.get('feature_columns', [])
        targets  = data.get('target_columns', [])
        print(f"[Learning] Loaded rule_fit: {len(features)} features → {len(targets)} targets")
    except Exception as exc:
        print(f"[Learning] Failed to load rule_fit model: {exc}")
        self._rule_fit_model = None
```

Also initialise the attribute in `__init__`:

```python
self._rule_fit_model: dict | None = None
self._learning_rule_fit_path: str = ''
self._learning_enabled: bool = False
self._learning_use_fitted_rules: bool = False
```

---

## Step 7 — Wire model inference into `BeatIntelligence.build_decision()`

Loading the model has no effect unless its outputs are applied. Currently nothing in
`BeatIntelligence` reads `_rule_fit_model`. The inference shape from `rule_fit.json` is:

```
normalize features → dot(coefficients) + intercept → motion parameter override
```

Targets from the model: `arc_size`, `arc_duration_frac`, `jitter_mix`, `creep_mix`,
`gate_strictness`, `burst_prob`.

Required additions:

**In `StrokeMapper`:** add a method that executes inference for one frame:

```python
def _apply_rule_fit(self, raw_features: dict[str, float]) -> dict[str, float]:
    """Run one inference pass and return a dict of override values, or {} on failure."""
    model = getattr(self, '_rule_fit_model', None)
    if model is None:
        return {}
    try:
        import numpy as np
        norm   = model['normalization']
        mean   = norm['mean']
        std    = norm['std']
        cols   = model['feature_columns']
        models = model['models']

        x_norm = np.array([
            (raw_features.get(c, mean[c]) - mean[c]) / max(std[c], 1e-8)
            for c in cols
        ], dtype=float)

        result = {}
        for target, spec in models.items():
            intercept = float(spec['intercept'])
            coefs     = [float(spec['coefficients'].get(c, 0.0)) for c in cols]
            result[target] = float(intercept + np.dot(coefs, x_norm))
        return result
    except Exception:
        return {}
```

**In `BeatIntelligence.build_decision()`:** after it assembles its raw energy values
but before it returns `BeatDecision`, call `_apply_rule_fit` (passed in or via a back-
reference) and blend the results into the decision fields. The cleanest seam is to give
`BeatIntelligence` a reference to the model via a setter:

```python
# In BeatIntelligence.__init__:
self._rule_fit_apply: Callable[[dict], dict] | None = None

# In StrokeMapper.__init__ (after self._intelligence is created):
self._intelligence._rule_fit_apply = self._apply_rule_fit

# In BeatIntelligence.build_decision(), after energies are computed:
if self._rule_fit_apply is not None:
    overrides = self._rule_fit_apply({
        'rms':                 self.energies.rms,
        'log_energy':          self.energies.log_energy,
        'spectral_flux':       self.energies.spectral_flux,
        'flux_delta':          self.energies.flux_delta,
        'sub_bass_energy':     self.energies.sub_bass,
        'low_mid_energy':      self.energies.low_mid,
        'mid_energy':          self.energies.mid,
        'high_energy':         self.energies.high,
        'low_high_ratio':      self.energies.low_high_ratio,
        'spectral_centroid_hz':  self.energies.spectral_centroid,
        'spectral_bandwidth_hz': self.energies.spectral_bandwidth,
        'spectral_rolloff_hz':   self.energies.spectral_rolloff,
        'spectral_flatness':     self.energies.spectral_flatness,
    })
    # apply overrides to decision fields (clamp to valid range)
    arc_size = float(np.clip(overrides.get('arc_size', decision.radius_bloom), 0.0, 1.0))
    decision = BeatDecision(
        trigger_kind=decision.trigger_kind,
        interval_beats=decision.interval_beats,
        radius_bloom=arc_size,
        silence_active=decision.silence_active,
        journey_completion=decision.journey_completion,
    )
```

Map the remaining targets (`jitter_mix`, `creep_mix`, `gate_strictness`, `burst_prob`)
to the corresponding config fields or local effect weights as appropriate.

> **Note:** The energy field names inside `BeatIntelligence` (e.g. `self.energies.rms`,
> `self.energies.spectral_flux`) must be confirmed against the actual `BeatIntelligence`
> dataclass before writing the dict literal above. Adjust key names to match exactly.

---

## Step 8 — Remove any remaining dependency on `D:\breadbeats_datasets`

Search `main.py` for:

```python
base_dir = Path(r"D:\breadbeats_datasets\blends")
```

Delete that line and all code that depends on `base_dir` within
`_apply_release_learning_defaults()`. After the rewrite in Step 3, nothing in that
function should reference an external absolute path.

Also remove the `_by_mtime_desc` inner function and the glob-based candidate
selection — all of that logic is replaced by the deterministic filename lookup in Step 3.

---

## What NOT to change

| Area | Reason |
|---|---|
| `_apply_learning_config_to_mapper()` | Already correct; pushes config → StrokeMapper |
| `learned_profile_slots.json` / preset buttons | Separate user-managed system; leave intact |
| `teaching_*` config defaults in `config.py` | Only `teaching_learning_enabled` and `teaching_use_fitted_rules` must default to `True`; already the case |
| Core stroke geometry / S-curve / journey logic | Out of scope; model overrides blend in at the decision level only |

---

## Validation checklist

```
1. python -m py_compile main.py beat_intelligence.py stroke_mapper.py
2. python -m pytest -q tests/test_stroke_mapper_contract.py
3. python run.py
   → look for: [Learning] Loaded rule_fit: 13 features → 6 targets
   → look for: [Learning] Release defaults applied — source=bundled profile=... rule_fit=...
4. Run PyInstaller build task
5. Launch dist/bREadbeats/bREadbeats.exe
   → same startup log lines, no D:\ reference, no file-not-found errors
```

---

## File summary

| Action | File |
|---|---|
| Create | `defaults/learning/rule_fit.release.json` |
| Create | `defaults/learning/profile.release.json` |
| Edit | `main.py` — replace `_apply_release_learning_defaults()` body |
| Edit | `main.py` — add `_get_bundled_defaults_dir()` helper |
| Edit | `bREadbeats.spec` — add two entries to `datas` |
| Edit | `stroke_mapper.py` — add `_try_load_learning_model()`, `_apply_rule_fit()`, and init attributes |
| Edit | `beat_intelligence.py` — add `_rule_fit_apply` hook and inference call in `build_decision()` |
| No change | `config.py` |
