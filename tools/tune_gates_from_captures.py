#!/usr/bin/env python3
"""Analyse keyboard-teaching captures to find optimal per-bus gate thresholds.

Reads all teaching CSVs, maps human directives to a binary label
(wanted_motion = True/False), then finds threshold boundaries for:
  - per-bus scores  (bf_bus_scores_*)
  - per-bus pass    (bf_bus_pass_*)
  - gate-state band energies  (gs_sub_bass, gs_low_mid, gs_mid, gs_high)
  - transient profile features (bf_kick_like_conf, bf_hat_like_conf, bf_bass_dominance)
  - overall fill & flux values

Outputs a summary table + suggested config adjustments.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAPTURE_DIR = ROOT / "teaching_captures" / "keyboard"


# ── Directive → label mapping ──────────────────────────────────────

# Old smooth model directives
OLD_MOTION_POSITIVE = {"more", "faster", "more+faster"}
OLD_MOTION_NEGATIVE = {"less", "slower", "less+slower", "less+faster"}  # less+faster = conflicted → negative
OLD_NEUTRAL = {"none", ""}

# New discrete model directives
NEW_MOTION_NEGATIVE = {"park"}


def label_row(row: dict) -> int | None:
    """Return 1 (wanted motion), 0 (wanted less/park), or None (ambiguous/neutral)."""
    directive = row.get("directive", "").strip()

    # New discrete format
    if "is_parked" in row:
        is_parked = row.get("is_parked", "0").strip()
        if is_parked == "1" or directive == "park":
            return 0
        else:
            return 1

    # Old smooth format
    if directive in OLD_MOTION_POSITIVE:
        return 1
    elif directive in OLD_MOTION_NEGATIVE:
        return 0
    else:
        return None  # neutral / no input — skip


def safe_float(val: str, default: float = float("nan")) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ── Feature extraction ─────────────────────────────────────────────

FEATURES_BUS_SCORES = [
    "bf_bus_scores_sub_bass",
    "bf_bus_scores_low_mid",
    "bf_bus_scores_mid",
    "bf_bus_scores_high",
]

FEATURES_GS_BANDS = [
    "gs_sub_bass",
    "gs_low_mid",
    "gs_mid",
    "gs_high",
]

FEATURES_TRANSIENT = [
    "bf_kick_like_conf",
    "bf_hat_like_conf",
    "bf_mixed_conf",
    "bf_bass_dominance",
]

FEATURES_FILL = [
    "gs_energy_fullness",
    "gs_flux_mean",
    "gs_flux_std",
    "gs_rms_envelope_db",
]

FEATURES_CONTEXT = [
    "spectral_flux",
    "raw_rms_db",
    "intensity",
    "bpm",
]

ALL_FEATURES = FEATURES_BUS_SCORES + FEATURES_GS_BANDS + FEATURES_TRANSIENT + FEATURES_FILL + FEATURES_CONTEXT


def load_sessions() -> list[dict]:
    """Load all keyboard teaching CSVs and return labelled rows."""
    rows = []
    session_dirs = sorted(CAPTURE_DIR.iterdir()) if CAPTURE_DIR.exists() else []
    for sd in session_dirs:
        csv_path = sd / "directives.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            session_rows = list(reader)
        n_labelled = 0
        for r in session_rows:
            lbl = label_row(r)
            if lbl is not None:
                r["_label"] = lbl
                r["_session"] = sd.name
                rows.append(r)
                n_labelled += 1
        total = len(session_rows)
        print(f"  {sd.name}: {total} rows, {n_labelled} labelled")
    return rows


def compute_stats(rows: list[dict]) -> None:
    """Print per-feature statistics split by label."""
    positive = [r for r in rows if r["_label"] == 1]
    negative = [r for r in rows if r["_label"] == 0]

    print(f"\n{'─' * 80}")
    print(f"  Total labelled rows: {len(rows)}  (motion={len(positive)}, park/less={len(negative)})")
    print(f"{'─' * 80}\n")

    print(f"{'Feature':<32s} {'Motion mean':>12s} {'Park mean':>12s} {'Δ':>8s} {'Motion p50':>12s} {'Park p50':>12s} {'Suggested':>12s}")
    print(f"{'─' * 32} {'─' * 12} {'─' * 12} {'─' * 8} {'─' * 12} {'─' * 12} {'─' * 12}")

    for feat in ALL_FEATURES:
        vals_pos = np.array([safe_float(r.get(feat, "")) for r in positive])
        vals_neg = np.array([safe_float(r.get(feat, "")) for r in negative])

        vals_pos = vals_pos[~np.isnan(vals_pos)]
        vals_neg = vals_neg[~np.isnan(vals_neg)]

        if len(vals_pos) < 5 or len(vals_neg) < 5:
            print(f"{feat:<32s} {'(insufficient data)':>56s}")
            continue

        m_pos = np.mean(vals_pos)
        m_neg = np.mean(vals_neg)
        p50_pos = np.median(vals_pos)
        p50_neg = np.median(vals_neg)
        delta = m_pos - m_neg

        # Simple midpoint suggestion for threshold
        suggested = (p50_pos + p50_neg) / 2.0

        print(f"{feat:<32s} {m_pos:>12.4f} {m_neg:>12.4f} {delta:>+8.4f} {p50_pos:>12.4f} {p50_neg:>12.4f} {suggested:>12.4f}")


def compute_gate_fail_analysis(rows: list[dict]) -> None:
    """Analyze which gates blocked motion when the human wanted it."""
    positive = [r for r in rows if r["_label"] == 1]
    negative = [r for r in rows if r["_label"] == 0]

    print(f"\n{'─' * 80}")
    print("  Gate fail analysis: what blocked beats when human wanted motion?")
    print(f"{'─' * 80}\n")

    # From beat_intelligence gate_fail
    gate_fail_key = "dec_gate_fail"
    gate_counts_pos = defaultdict(int)
    gate_counts_neg = defaultdict(int)

    for r in positive:
        gf = r.get(gate_fail_key, "").strip()
        gate_counts_pos[gf or "(open)"] += 1
    for r in negative:
        gf = r.get(gate_fail_key, "").strip()
        gate_counts_neg[gf or "(open)"] += 1

    all_gates = sorted(set(gate_counts_pos.keys()) | set(gate_counts_neg.keys()))
    print(f"{'Gate fail':<25s} {'Motion frames':>15s} {'Park frames':>15s}")
    print(f"{'─' * 25} {'─' * 15} {'─' * 15}")
    for g in all_gates:
        cp = gate_counts_pos.get(g, 0)
        cn = gate_counts_neg.get(g, 0)
        pct_p = 100 * cp / max(1, len(positive))
        pct_n = 100 * cn / max(1, len(negative))
        print(f"{g:<25s} {cp:>6d} ({pct_p:5.1f}%) {cn:>6d} ({pct_n:5.1f}%)")


def compute_bus_pass_analysis(rows: list[dict]) -> None:
    """Per-bus pass rates by label."""
    positive = [r for r in rows if r["_label"] == 1]
    negative = [r for r in rows if r["_label"] == 0]

    print(f"\n{'─' * 80}")
    print("  Per-bus pass rate by label")
    print(f"{'─' * 80}\n")

    buses = ["sub_bass", "low_mid", "mid", "high"]
    print(f"{'Bus':<15s} {'Motion pass%':>15s} {'Park pass%':>15s} {'Δ':>8s}")
    print(f"{'─' * 15} {'─' * 15} {'─' * 15} {'─' * 8}")

    for bus in buses:
        key = f"bf_bus_pass_{bus}"
        pos_pass = sum(1 for r in positive if r.get(key, "").strip().lower() in ("true", "1"))
        neg_pass = sum(1 for r in negative if r.get(key, "").strip().lower() in ("true", "1"))
        pos_total = sum(1 for r in positive if r.get(key, "").strip() != "")
        neg_total = sum(1 for r in negative if r.get(key, "").strip() != "")

        if pos_total == 0 or neg_total == 0:
            print(f"{bus:<15s} {'(no data)':>40s}")
            continue

        pct_pos = 100 * pos_pass / pos_total
        pct_neg = 100 * neg_pass / neg_total
        print(f"{bus:<15s} {pct_pos:>14.1f}% {pct_neg:>14.1f}% {pct_pos - pct_neg:>+7.1f}%")


def compute_bus_reason_analysis(rows: list[dict]) -> None:
    """Per-bus reason code frequency during motion-wanted frames."""
    positive = [r for r in rows if r["_label"] == 1]

    print(f"\n{'─' * 80}")
    print("  Per-bus rejection reasons when human wanted MOTION")
    print(f"{'─' * 80}\n")

    buses = ["sub_bass", "low_mid", "mid", "high"]
    for bus in buses:
        key = f"bf_bus_reason_codes_{bus}"
        reason_counts = defaultdict(int)
        total = 0
        for r in positive:
            val = r.get(key, "").strip()
            if not val or val.lower() in ("", "[]"):
                continue
            total += 1
            # Parse reason list — it's stored as Python list repr: "['refractory', 'below_gate']"
            cleaned = val.strip("[]'\"").replace("'", "").replace('"', '')
            for reason in cleaned.split(","):
                reason = reason.strip()
                if reason:
                    reason_counts[reason] += 1

        if total == 0:
            continue
        print(f"  {bus}:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason:<25s}  {count:>5d}  ({100 * count / total:5.1f}%)")
        print()


def compute_percentile_thresholds(rows: list[dict]) -> None:
    """Show percentile distributions for key features, split by label."""
    positive = [r for r in rows if r["_label"] == 1]
    negative = [r for r in rows if r["_label"] == 0]

    key_features = FEATURES_BUS_SCORES + FEATURES_GS_BANDS + ["bf_kick_like_conf", "bf_bass_dominance", "gs_energy_fullness"]

    print(f"\n{'─' * 80}")
    print("  Percentile distributions (p10 / p25 / p50 / p75 / p90)")
    print(f"{'─' * 80}\n")

    for feat in key_features:
        vals_pos = np.array([safe_float(r.get(feat, "")) for r in positive])
        vals_neg = np.array([safe_float(r.get(feat, "")) for r in negative])
        vals_pos = vals_pos[~np.isnan(vals_pos)]
        vals_neg = vals_neg[~np.isnan(vals_neg)]

        if len(vals_pos) < 10 or len(vals_neg) < 10:
            continue

        pctiles = [10, 25, 50, 75, 90]
        pos_pct = np.percentile(vals_pos, pctiles)
        neg_pct = np.percentile(vals_neg, pctiles)

        pct_str_pos = " / ".join(f"{v:.4f}" for v in pos_pct)
        pct_str_neg = " / ".join(f"{v:.4f}" for v in neg_pct)

        print(f"  {feat}:")
        print(f"    Motion: {pct_str_pos}")
        print(f"    Park:   {pct_str_neg}")
        # Optimal split point: maximize separation between p25 of motion and p75 of park
        split = (pos_pct[1] + neg_pct[3]) / 2.0  # midpoint of motion-p25 and park-p75
        print(f"    → Split (motion_p25 ↔ park_p75): {split:.4f}")
        print()


def main():
    print("=" * 80)
    print("  Gate Tuning from Keyboard Teaching Captures")
    print("=" * 80)
    print(f"\nCapture dir: {CAPTURE_DIR}\n")

    rows = load_sessions()
    if not rows:
        print("No labelled data found!")
        sys.exit(1)

    compute_stats(rows)
    compute_gate_fail_analysis(rows)
    compute_bus_pass_analysis(rows)
    compute_bus_reason_analysis(rows)
    compute_percentile_thresholds(rows)

    print("\n" + "=" * 80)
    print("  Done. Review the Δ column and percentile splits to adjust gate thresholds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
