"""FunScript Converter — 6-axis to 4-phase electrode math.

Pipeline: load multi-axis funscripts → unify timeline → mix to (α,β,γ)
→ tetrahedral projection → permute → e1-e4 + optional freq outputs → export.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt

import numpy as np

from pmv_funscript_io import FunscriptAction

# ---------------------------------------------------------------------------
# Tetrahedral basis vectors (regular tetrahedron inscribed in unit sphere)
# ---------------------------------------------------------------------------
_COEF_1 = 1.0
_COEF_2 = sqrt(8) / 3          # ≈ 0.9428
_COEF_3 = sqrt(2) / sqrt(3)    # ≈ 0.8165

TETRA_VERTICES = np.array([
    [ _COEF_1,         0.0,         0.0      ],   # v0
    [-_COEF_1 / 3,     _COEF_2,     0.0      ],   # v1
    [-_COEF_1 / 3,    -_COEF_2 / 2, _COEF_3  ],   # v2
    [-_COEF_1 / 3,    -_COEF_2 / 2, -_COEF_3 ],   # v3
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Electrode placement presets — permutation of (0,1,2,3)
# placement[i] = which basis-vector index electrode E(i+1) receives
# ---------------------------------------------------------------------------
PRESETS: dict[str, tuple[int, int, int, int]] = {
    "Triangle + Behind":            (0, 1, 2, 3),
    "Tip Sides + Base":             (1, 2, 3, 0),
    "Tip-Base + Bipolar Internal":  (0, 3, 1, 2),
}

DEFAULT_PRESET = "Triangle + Behind"

# ---------------------------------------------------------------------------
# Input axis names recognised by the converter
# ---------------------------------------------------------------------------
CONVERTER_INPUT_AXES = {"main", "surge", "sway", "twist", "roll", "pitch"}

# ---------------------------------------------------------------------------
# Weight defaults
# ---------------------------------------------------------------------------

@dataclass
class MixWeights:
    w_primary: float = 0.8
    w_secondary: float = 0.2
    w_twist: float = 0.3
    twist_phase: float = 0.7853981633974483  # π/4


@dataclass
class FreqConfig:
    enabled: bool = False
    freq_scale: float = 1.0
    carrier_scale: float = 1.0


# ---------------------------------------------------------------------------
# Timeline unification
# ---------------------------------------------------------------------------

def unify_timeline(
    axes: dict[str, list[FunscriptAction]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Unify multi-axis funscripts onto a single authoritative timeline.

    Returns (timestamps_ms, {axis_name: positions_0_100}).
    Missing axes are filled with 50 (neutral).
    """
    if not axes:
        empty = np.array([], dtype=np.float64)
        return empty, {}

    # Choose authoritative axis: stroke (main) if present, else most keyframes
    if "main" in axes and axes["main"]:
        auth_name = "main"
    else:
        auth_name = max(axes, key=lambda k: len(axes[k]))

    auth_actions = axes[auth_name]
    timestamps = np.array([a.at for a in auth_actions], dtype=np.float64)

    # Extend duration to cover all axes
    max_t = timestamps[-1] if len(timestamps) > 0 else 0.0
    for ax_actions in axes.values():
        if ax_actions:
            last_t = ax_actions[-1].at
            if last_t > max_t:
                max_t = last_t

    unified: dict[str, np.ndarray] = {}
    for axis_name in CONVERTER_INPUT_AXES:
        if axis_name in axes and axes[axis_name]:
            src = axes[axis_name]
            src_t = np.array([a.at for a in src], dtype=np.float64)
            src_p = np.array([a.pos for a in src], dtype=np.float64)
            unified[axis_name] = np.interp(timestamps, src_t, src_p)
        else:
            unified[axis_name] = np.full(len(timestamps), 50.0)

    return timestamps, unified


# ---------------------------------------------------------------------------
# 6→3 spatial mixing
# ---------------------------------------------------------------------------

def _norm(pos: np.ndarray) -> np.ndarray:
    """Funscript 0-100 → signed -1.0 to +1.0 (center at 50 → 0.0)."""
    return (pos / 100.0 - 0.5) * 2.0


def mix_to_3d(
    unified: dict[str, np.ndarray],
    weights: MixWeights | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map 6 input axes to 3D (α, β, γ) space.

    Returns three arrays of equal length.
    """
    w = weights or MixWeights()

    stroke = _norm(unified.get("main", np.array([50.0])))
    sway = _norm(unified.get("sway", np.array([50.0])))
    surge = _norm(unified.get("surge", np.array([50.0])))
    roll = _norm(unified.get("roll", np.array([50.0])))
    pitch = _norm(unified.get("pitch", np.array([50.0])))
    twist = _norm(unified.get("twist", np.array([50.0])))

    alpha = stroke
    beta = sway * w.w_primary + roll * w.w_secondary
    gamma = surge * w.w_primary + pitch * w.w_secondary

    # Twist: circular decomposition into β/γ plane
    beta = beta + twist * w.w_twist * cos(w.twist_phase)
    gamma = gamma + twist * w.w_twist * sin(w.twist_phase)

    # Clamp to [-1, +1]
    alpha = np.clip(alpha, -1.0, 1.0)
    beta = np.clip(beta, -1.0, 1.0)
    gamma = np.clip(gamma, -1.0, 1.0)

    return alpha, beta, gamma


# ---------------------------------------------------------------------------
# Tetrahedral projection
# ---------------------------------------------------------------------------

def tetrahedral_project(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Project (α,β,γ) points onto tetrahedral basis vectors.

    Returns array of shape (N, 4) with values in [0, 1].
    """
    points = np.column_stack([alpha, beta, gamma])           # (N, 3)
    raw = points @ TETRA_VERTICES.T                          # (N, 4)

    # Per-sample: shift min→0, then normalize max→1
    row_min = raw.min(axis=1, keepdims=True)
    shifted = raw - row_min
    row_max = shifted.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    normalized = shifted / row_max

    return normalized


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------

def apply_permutation(
    projected: np.ndarray,
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Reorder columns of projected (N,4) according to placement.

    placement[i] = which basis vector index electrode E(i+1) gets.
    """
    return projected[:, list(placement)]


# ---------------------------------------------------------------------------
# Frequency axis generation
# ---------------------------------------------------------------------------

def generate_freq_axes(
    unified: dict[str, np.ndarray],
    config: FreqConfig | None = None,
) -> dict[str, np.ndarray]:
    """Generate pulse_frequency and carrier_frequency from surge.

    Returns {axis_name: values_0_100} for enabled outputs.
    """
    cfg = config or FreqConfig()
    if not cfg.enabled:
        return {}

    surge = _norm(unified.get("surge", np.array([50.0])))
    result: dict[str, np.ndarray] = {}

    if cfg.freq_scale > 0:
        pf = np.abs(surge) * cfg.freq_scale
        result["pulse_frequency"] = np.clip(pf * 100.0, 0.0, 100.0)

    if cfg.carrier_scale > 0:
        cf = np.abs(surge) * cfg.carrier_scale
        result["carrier_frequency"] = np.clip(cf * 100.0, 0.0, 100.0)

    return result


# ---------------------------------------------------------------------------
# Main conversion entry point
# ---------------------------------------------------------------------------

def convert(
    axes: dict[str, list[FunscriptAction]],
    placement: tuple[int, int, int, int] | None = None,
    weights: MixWeights | None = None,
    freq_config: FreqConfig | None = None,
) -> dict[str, list[FunscriptAction]]:
    """Run the full 6-axis → 4-phase conversion pipeline.

    Returns dict with keys 'e1'–'e4' and optionally
    'pulse_frequency', 'carrier_frequency'.
    """
    if placement is None:
        placement = PRESETS[DEFAULT_PRESET]

    timestamps, unified = unify_timeline(axes)
    if len(timestamps) == 0:
        return {}

    alpha, beta, gamma = mix_to_3d(unified, weights)
    projected = tetrahedral_project(alpha, beta, gamma)
    permuted = apply_permutation(projected, placement)

    ts_int = timestamps.astype(np.int64)

    result: dict[str, list[FunscriptAction]] = {}
    for i, name in enumerate(["e1", "e2", "e3", "e4"]):
        positions = np.clip(np.round(permuted[:, i] * 100.0), 0, 100).astype(int)
        result[name] = [
            FunscriptAction(at=int(t), pos=int(p))
            for t, p in zip(ts_int, positions)
        ]

    # Optional frequency outputs
    freq_axes = generate_freq_axes(unified, freq_config)
    for fname, fvals in freq_axes.items():
        positions = np.clip(np.round(fvals), 0, 100).astype(int)
        result[fname] = [
            FunscriptAction(at=int(t), pos=int(p))
            for t, p in zip(ts_int, positions)
        ]

    return result
