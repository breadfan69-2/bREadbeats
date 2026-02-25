# Audio Engine Dead Code Audit — 2026-02-25

## Summary

`audio_engine.py` (2 933 lines) contains **9 dead methods** and **2 dead subsystems**
(BPS Feedback, Flux Balance) whose UI controls are permanently disabled. Removing
them will eliminate ~270 lines of method bodies, ~40 lines of `__init__` state, and
related dead branches in `main.py`.

---

## Dead Methods

### 1. `reset_tempo_tracking` — L183-L218  (36 lines)

**Why dead:** Public method, zero callers in any file. Tempo resets flow through
`_reset_acf_metronome` and `_reset_downbeat_pattern` instead.

**Removal:** Delete the method body. No callers to update.

---

### 2. `get_freq_band_bins` — L1517-L1525  (9 lines)

**Why dead:** Public method, zero callers. Was presumably used for a frequency-band
visualisation overlay that was removed or never shipped.

**Removal:** Delete the method body. No callers to update.

---

### 3. `compute_bps_feedback` — L2596-L2673  (78 lines)

**Why dead:** Public method, zero callers. Part of the dead **Target BPS** subsystem.
The feedback callback was never wired from `main.py`.

**Removal:** Delete together with the rest of the BPS subsystem (see §Below).

---

### 4. `set_target_bps` — L2675-L2677  (3 lines)

**Why dead:** Setter for `_target_bps`, never called from any file. Dead BPS API.

---

### 5. `set_bps_adjustment_speed` — L2679-L2681  (3 lines)

**Why dead:** Setter for `_bps_adjustment_speed`, never called. Dead BPS API.

---

### 6. `set_bps_tolerance` — L2683-L2685  (3 lines)

**Why dead:** Setter for `_target_bps_tolerance`, never called. Dead BPS API.

---

### 7. `compute_flux_balance_feedback` — L2730-L2822  (93 lines)

**Why dead:** Public method, zero callers. Part of the dead **Flux Balance** subsystem.
`enable_metric_autoranging('flux_balance', …)` is never invoked from `main.py` either,
so the enabling path is also dead.

**Removal:** Delete together with the rest of the Flux Balance subsystem (see §Below).

---

### 8. `_accept_raw_onset` — L702-L707  (6 lines)

**Why dead:** Private wrapper that calls `_is_raw_onset_acceptable` and then sets
`_last_accepted_raw_onset_time`. The audio callback at ~L1288 performs both of
these steps inline and never calls this wrapper.

**Removal:** Delete the method. No callers to update — `_is_raw_onset_acceptable`
is the one actually used.

---

### 9. `_compute_bass_dominance` — L1593-L1600  (8 lines)

**Why dead:** Private wrapper that delegates to the module-level
`compute_bass_dominance()`. The shadow-feature path at ~L602 calls the module
function directly; nothing calls `self._compute_bass_dominance()`.

**Removal:** Delete the method. No callers to update.

---

## Dead Subsystems (init state + UI)

Removing the dead methods above also lets you prune the associated `__init__`
state and disabled UI widgets.

### A. Target BPS Subsystem

**Dead `__init__` state** (audio_engine.py L419-L429):
```python
self._target_bps_enabled         # L420
self._target_bps                 # L421
self._target_bps_tolerance       # L422
self._bps_window_seconds         # L423
self._bps_beat_times             # L424
self._bps_adjustment_speed       # L425
self._bps_base_step              # L426
self._target_bps_lock_gate_enabled    # L427
self._target_bps_lock_gate_acf_conf   # L428
self._target_bps_lock_gate_downbeats  # L429
```

**Dead `enable_metric_autoranging` branch** (audio_engine.py L2495-L2499):
The `elif metric == 'target_bps':` block can be removed.

**Dead main.py UI widgets & handlers**:
| Item | Location |
|------|----------|
| `metric_target_bps_cb` checkbox (always disabled/unchecked) | L6510-L6513, L6674-L6678, L6740-L6741 |
| `target_bpm_spin` spinbox (disabled) | L6681-L6691 |
| `bpm_tolerance_spin` spinbox (disabled) | L6694-L6702 |
| `bpm_actual_label` (never updated for BPS) | L6706-L6707 |
| `auto_align_target_cb` checkbox (disabled) | L6709-L6713 |
| `auto_align_seconds_spin` spinbox (disabled) | L6715-L6724 |
| `_on_target_bpm_change` stub handler | L6571-L6573 |
| `_on_bpm_tolerance_change` stub handler | L6575-L6577 |
| `_on_target_bps_lock_gate_toggle` / `_on_target_bps_lock_gate_acf_conf_change` / `_on_target_bps_lock_gate_downbeats_change` (if present) | check L5842-L5847 |
| `_on_metric_feedback` branch `elif metric in ('peak_floor', 'target_bps', 'flux_balance'):` | L6568 — remove `'target_bps'` from the tuple (and `'flux_balance'`) |

**Dead preset save/load keys** (main.py):
| Key | Location |
|-----|----------|
| `target_bps_lock_gate_enabled` | L5741, L5842-L5843 |
| `target_bps_lock_gate_acf_conf` | L5742, L5844-L5845 |
| `target_bps_lock_gate_downbeats` | L5743, L5846-L5847 |

**Dead `_metric_settled_counts` / `_metric_settled_flags` entries** (audio_engine.py):
Remove the `'flux_balance'` key from both dicts at L405 and L411.

---

### B. Flux Balance Subsystem

**Dead `__init__` state** (audio_engine.py L388-L396):
```python
self._metric_flux_balance_enabled      # L389
self._flux_balance_check_interval_ms   # L390
self._last_flux_balance_check          # L391
self._flux_energy_ratios               # L392
self._flux_balance_target_low          # L393
self._flux_balance_target_high         # L394
self._flux_balance_step_pct            # L395
self._flux_balance_hysteresis_count    # L396
```

**Dead `enable_metric_autoranging` branch** (audio_engine.py L2486-L2493):
The `elif metric == 'flux_balance':` block can be removed.

**Dead `_on_metric_feedback` branch** in main.py L6568:
Remove `'flux_balance'` from the no-op tuple.

---

## Safe Deletion Order

Deletions have no inter-dependencies — they can be done in any order. The
recommended sequence minimises risk:

1. **`_accept_raw_onset`** and **`_compute_bass_dominance`**
   Simplest — standalone dead wrappers, no associated state.

2. **`reset_tempo_tracking`** and **`get_freq_band_bins`**
   Standalone methods, no associated state.

3. **Flux Balance subsystem**
   - Delete `compute_flux_balance_feedback` method
   - Delete `__init__` state fields (L388-L396)
   - Delete `elif metric == 'flux_balance':` branch in `enable_metric_autoranging`
   - Delete `'flux_balance'` from `_metric_settled_counts` / `_metric_settled_flags`
   - Remove `'flux_balance'` from the no-op tuple in main.py `_on_metric_feedback`

4. **BPS subsystem**
   - Delete `compute_bps_feedback`, `set_target_bps`, `set_bps_adjustment_speed`,
     `set_bps_tolerance` methods
   - Delete `__init__` state fields (L419-L429)
   - Delete `elif metric == 'target_bps':` branch in `enable_metric_autoranging`
   - Delete all disabled UI widgets and stub handlers in main.py (see table above)
   - Delete preset save/load keys in main.py (L5741-L5743, L5842-L5847)
   - Remove `'target_bps'` from the no-op tuple in main.py `_on_metric_feedback`

5. **Verify** — run the app and confirm no `AttributeError` at startup or during
   audio playback.

---

## What to Keep

- The `if __name__ == "__main__":` test harness at the bottom (L2913-L2933) is not
  technically dead code — it's a developer convenience entry-point. Keep or
  remove at your discretion.
- `_is_raw_onset_acceptable` (L709) is **actively used** (called ~L1288). Do NOT
  delete — only its dead wrapper `_accept_raw_onset` is removed.
- All other methods listed in the audit are actively called.

---

*Lines referenced are from the current file as of 2026-02-25.*
