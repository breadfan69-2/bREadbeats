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

The rest of startup calls `_apply_learning_config_to_mapper()` which already drives
`mapper._try_load_learning_model()`, so no additional wiring is needed.

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

After the StrokeMapper is constructed (later during engine start), it calls
`_try_load_learning_model()` with the path that was stored in
`config.beat.teaching_rule_fit_path`. No extra work needed beyond ensuring the paths
are set before the mapper is constructed.

If the order is inverted (mapper created before `_apply_release_learning_defaults` is
called) the mapper will also call `_try_load_learning_model()` again after each path
assignment via `_apply_learning_config_to_mapper()`, so either order is safe.

---

## Step 6 — Remove any remaining dependency on `D:\breadbeats_datasets`

Search `main.py` for:

```python
base_dir = Path(r"D:\breadbeats_datasets\blends")
```

Delete that line and all code that depends on `base_dir` within
`_apply_release_learning_defaults()`. After the rewrite in Step 3, nothing in that
function should reference an external absolute path.

---

## What NOT to change

| Area | Reason |
|---|---|
| `_apply_learning_config_to_mapper()` | Already correct; pushes config → StrokeMapper |
| `_try_load_learning_model()` in StrokeMapper/BeatIntelligence | Already reads from `_learning_rule_fit_path`; no change needed |
| `learned_profile_slots.json` / preset buttons | Separate user-managed system; leave intact |
| `teaching_*` config defaults in `config.py` | Only `teaching_learning_enabled` and `teaching_use_fitted_rules` must default to `True`; already the case |
| Stroke motion logic | Out of scope entirely |

---

## Validation checklist

```
1. python -m py_compile main.py
2. python -m pytest -q tests/test_stroke_mapper_contract.py
3. python run.py
   → look for: [Learning] Release defaults applied — source=bundled profile=... rule_fit=...
4. Run PyInstaller build task
5. Launch dist/bREadbeats/bREadbeats.exe
   → same startup log line, no D:\ reference, no file-not-found errors
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
| No change | `stroke_mapper.py`, `beat_intelligence.py`, `config.py` |
