"""FunScript Converter — 6-axis to 4-phase electrode math.

Pipeline: load multi-axis funscripts → unify timeline → mix to (α,β,γ)
→ tetrahedral projection → permute → e1-e4 + optional freq outputs → export.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt, pi

import numpy as np

from pmv_funscript_io import FunscriptAction
from pmv_axis_converter import apply_centered_output_limits

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
# Legacy wiring presets — permutation of (0,1,2,3)
# placement[i] = which decoded output index physical electrode E(i+1) receives
# ---------------------------------------------------------------------------
PRESETS: dict[str, tuple[int, int, int, int]] = {
    "Triangle + Behind":            (0, 1, 2, 3),
    "Tip Sides + Base":             (1, 2, 3, 0),
    "Tip-Base + Bipolar Internal":  (0, 3, 1, 2),
}

DEFAULT_PRESET = "Triangle + Behind"
IDENTITY_WIRING_MAP = PRESETS[DEFAULT_PRESET]

# ---------------------------------------------------------------------------
# Pair-position layout models (direct E1-E4 decoder)
# ---------------------------------------------------------------------------

DEFAULT_LAYOUT_MODEL = "Pair At Middle"

LAYOUT_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "Pair At Top": "Pair At Top",
    "Pair At Middle": "Pair At Middle",
    "Pair At Bottom / Rear": "Pair At Bottom / Rear",
}

LAYOUT_MODEL_ALIASES: dict[str, str] = {
    "Triangle + Behind": "Pair At Middle",
    "Tip Sides + Base": "Pair At Top",
    "Tip-Base + Bipolar Internal": "Pair At Bottom / Rear",
}

_POINT_CENTER = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
_POINT_A = np.array([1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
_POINT_B = np.array([1.0 / 3.0, 1.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
_POINT_C = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0 / 3.0], dtype=np.float64)
_POINT_D = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float64)
_POINT_AB = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
_POINT_BC = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
_POINT_CD = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
_POINT_ABC = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)
_POINT_ABD = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float64)
_POINT_ACD = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float64)
_POINT_BCD = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float64)

_SIDE_GAIN = 0.45
_ROT_GAIN = 0.35
_PULSE_SPEED_BIAS_GAIN = 0.15
_CARRIER_SPEED_BIAS_GAIN = 0.10
_CARRIER_POSITION_BIAS_GAIN = 0.15

LAYOUT_MODELS: dict[str, dict[str, np.ndarray]] = {
    "Pair At Top": {
        "top": _POINT_AB,
        "mid": _POINT_C,
        "low": _POINT_D,
        "side_left": _POINT_A,
        "side_right": _POINT_B,
        "rot_left": _POINT_ACD,
        "rot_right": _POINT_BCD,
    },
    "Pair At Middle": {
        "top": _POINT_A,
        "mid": _POINT_BC,
        "low": _POINT_D,
        "side_left": _POINT_B,
        "side_right": _POINT_C,
        "rot_left": _POINT_ABD,
        "rot_right": _POINT_ACD,
    },
    "Pair At Bottom / Rear": {
        "top": _POINT_A,
        "mid": _POINT_B,
        "low": _POINT_CD,
        "side_left": _POINT_C,
        "side_right": _POINT_D,
        "rot_left": _POINT_ABC,
        "rot_right": _POINT_ABD,
    },
}

_LAYOUT_CONTROL_ROTATIONS: dict[str, float] = {
    # Rotate the upstream alpha/gamma frame so pair position changes how
    # axial travel and rotational branch bias borrow from the same 3-DOF input.
    "Pair At Top": -pi / 6.0,
    "Pair At Middle": 0.0,
    "Pair At Bottom / Rear": pi / 6.0,
}

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
    pulse_surge_influence: float = 1.0
    pulse_speed_influence: float = _PULSE_SPEED_BIAS_GAIN
    pulse_center: float = 55.0    # center position for pulse_frequency (0-100)
    pulse_min: float = 20.0
    pulse_max: float = 80.0
    carrier_surge_influence: float = 1.0
    carrier_speed_influence: float = _CARRIER_SPEED_BIAS_GAIN
    carrier_center: float = 50.0  # center position for carrier_frequency (0-100)
    carrier_min: float = 40.0
    carrier_max: float = 60.0


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

    if len(timestamps) > 0 and max_t > timestamps[-1]:
        timestamps = np.append(timestamps, float(max_t))

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


def _compute_speed_signal(
    positions: np.ndarray,
    timestamps_ms: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a normalized 0-1 motion speed envelope from the main axis."""
    values = np.clip(np.asarray(positions, dtype=np.float64) / 100.0, 0.0, 1.0)
    if len(values) == 0:
        return values
    if len(values) == 1:
        return np.zeros_like(values)

    if timestamps_ms is None or len(timestamps_ms) != len(values):
        delta_t_sec = np.ones(len(values) - 1, dtype=np.float64)
    else:
        delta_t_sec = np.diff(np.asarray(timestamps_ms, dtype=np.float64)) / 1000.0
        delta_t_sec = np.where(delta_t_sec > 1e-9, delta_t_sec, 1e-9)

    velocity = np.abs(np.diff(values)) / delta_t_sec
    speed = np.empty_like(values)
    speed[0] = velocity[0]
    speed[-1] = velocity[-1]
    if len(values) > 2:
        speed[1:-1] = 0.5 * (velocity[:-1] + velocity[1:])

    peak = float(np.max(speed))
    if peak <= 1e-12:
        return np.zeros_like(values)
    return np.clip(speed / peak, 0.0, 1.0)


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

    # Preserve direction by scaling the combined 3D vector into the unit ball
    # instead of clipping each axis independently.
    points = np.column_stack([alpha, beta, gamma])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    scale = np.where(norms > 1.0, norms, 1.0)
    normalized_points = points / scale

    return normalized_points[:, 0], normalized_points[:, 1], normalized_points[:, 2]


def mix_to_layout_controls(
    unified: dict[str, np.ndarray],
    weights: MixWeights | None = None,
    layout_model: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map 6 input axes to axial/side/rotation layout controls.

    Pair-position layouts share one 3-DOF upstream motion source, but each
    layout rotates the axial/rotational frame differently so the accessory
    axes land in distinct places before direct E1-E4 decoding.
    """
    alpha, beta, gamma = mix_to_3d(unified, weights)
    resolved_layout = _resolve_layout_model(layout_model)
    theta = _LAYOUT_CONTROL_ROTATIONS[resolved_layout]

    axial = alpha * cos(theta) - gamma * sin(theta)
    side = beta
    rotation = alpha * sin(theta) + gamma * cos(theta)

    points = np.column_stack([axial, side, rotation])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    scale = np.where(norms > 1.0, norms, 1.0)
    normalized_points = points / scale
    return normalized_points[:, 0], normalized_points[:, 1], normalized_points[:, 2]


# ---------------------------------------------------------------------------
# Tetrahedral projection
# ---------------------------------------------------------------------------

def tetrahedral_project(
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Map 3D motion into direct fourphase coordinates.

    The target space is FOC-Stim's direct e1-e4 position manifold, not the
    legacy abc->e1234 helper space. Boundary directions should therefore hit
    direct landmarks like A, AB, and ABC exactly:

    * center -> [1, 1, 1, 1]
    * tetra vertex -> [1, 1/3, 1/3, 1/3]
    * edge direction -> [1, 1, 0, 0]
    * face direction -> [1, 1, 1, 0]

    We get that by:
      1. normalizing each 3D point to its direction on the unit sphere,
      2. projecting onto the tetrahedral basis,
      3. scaling each row by its own min/max span so boundary edge/face
         directions reach the expected direct fourphase landmarks,
      4. constraining onto the valid e1-e4 manifold,
      5. linearly interpolating from the direct center [1,1,1,1] by radius.
    """
    points = np.column_stack([alpha, beta, gamma])
    radius = np.linalg.norm(points, axis=1, keepdims=True)
    direction = np.divide(
        points,
        np.maximum(radius, 1e-12),
        out=np.zeros_like(points),
        where=radius > 1e-12,
    )
    raw = direction @ TETRA_VERTICES.T

    min_vals = raw.min(axis=1, keepdims=True)
    max_vals = raw.max(axis=1, keepdims=True)
    span = max_vals - min_vals
    scaled = np.divide(
        raw - min_vals,
        span,
        out=np.zeros_like(raw),
        where=span > 1e-12,
    )
    boundary = constrain_fourphase_coordinates(scaled)
    clamped_radius = np.clip(radius, 0.0, 1.0)
    return 1.0 + clamped_radius * (boundary - 1.0)


def constrain_fourphase_coordinates(projected: np.ndarray) -> np.ndarray:
    """Project 4-phase coordinates onto the valid FOC-Stim position manifold.

    FOC-Stim's direct fourphase path expects coordinates where at least one
    component is exactly 1 and the largest component does not exceed the sum
    of the other three. This matches Restim's constrain_4p_amplitudes /
    FOC-Stim's fourphase_constrain_coordinates behavior.
    """
    a = np.clip(projected[:, 0], 0.0, 1.0)
    b = np.clip(projected[:, 1], 0.0, 1.0)
    c = np.clip(projected[:, 2], 0.0, 1.0)
    d = np.clip(projected[:, 3], 0.0, 1.0)

    s_a = np.minimum(-a + b + c + d, 0.0) / -3.0
    s_b = np.minimum(a - b + c + d, 0.0) / -3.0
    s_c = np.minimum(a + b - c + d, 0.0) / -3.0
    s_d = np.minimum(a + b + c - d, 0.0) / -3.0

    a = a + s_b + s_c + s_d
    b = b + s_a + s_c + s_d
    c = c + s_a + s_b + s_d
    d = d + s_a + s_b + s_c

    constrained = np.column_stack([a, b, c, d])
    max_vals = constrained.max(axis=1)
    needs_shift = max_vals < 1.0
    constrained[needs_shift] += (1.0 - max_vals[needs_shift])[:, None]

    max_indices = np.argmax(constrained, axis=1)
    constrained[np.arange(len(constrained)), max_indices] = 1.0
    return constrained


def _resolve_layout_model(layout_model: str | None) -> str:
    if layout_model is None:
        return DEFAULT_LAYOUT_MODEL
    if layout_model in LAYOUT_MODELS:
        return layout_model
    if layout_model in LAYOUT_MODEL_ALIASES:
        return LAYOUT_MODEL_ALIASES[layout_model]
    raise ValueError(f"Unknown layout model: {layout_model}")


def _blend_rows(start: np.ndarray, end: np.ndarray, t: np.ndarray) -> np.ndarray:
    weights = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    if weights.size == 0:
        return np.empty((0, start.shape[0]), dtype=np.float64)
    return start[np.newaxis, :] * (1.0 - weights) + end[np.newaxis, :] * weights


def _pm_rows(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    t = (np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0) + 1.0) * 0.5
    return _blend_rows(left, right, t)


def _step_rows(points: np.ndarray, anchors: np.ndarray, gains: np.ndarray) -> np.ndarray:
    weight = np.clip(np.asarray(gains, dtype=np.float64), 0.0, 1.0).reshape(-1, 1)
    return constrain_fourphase_coordinates(points + weight * (anchors - points))


def decode_layout_controls(
    u: np.ndarray,
    s: np.ndarray,
    r: np.ndarray,
    layout_model: str | None = None,
) -> np.ndarray:
    """Decode axial/side/rotation controls into direct fourphase coordinates."""
    resolved_layout = _resolve_layout_model(layout_model)
    layout = LAYOUT_MODELS[resolved_layout]

    axial = np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0)
    side = np.clip(np.asarray(s, dtype=np.float64), -1.0, 1.0)
    rotation = np.clip(np.asarray(r, dtype=np.float64), -1.0, 1.0)
    if not (len(axial) == len(side) == len(rotation)):
        raise ValueError("Layout controls must have equal length")

    points = np.repeat(_POINT_CENTER[np.newaxis, :], len(axial), axis=0)

    pos_mask = axial >= 0.0
    mid_mask = (axial < 0.0) & (axial >= -0.5)
    low_mask = axial < -0.5

    if np.any(pos_mask):
        points[pos_mask] = _blend_rows(_POINT_CENTER, layout["top"], axial[pos_mask])
    if np.any(mid_mask):
        points[mid_mask] = _blend_rows(_POINT_CENTER, layout["mid"], -2.0 * axial[mid_mask])
    if np.any(low_mask):
        points[low_mask] = _blend_rows(layout["mid"], layout["low"], -2.0 * axial[low_mask] - 1.0)

    points = constrain_fourphase_coordinates(points)
    side_anchor = _pm_rows(side, layout["side_left"], layout["side_right"])
    points = _step_rows(points, side_anchor, _SIDE_GAIN * np.abs(side))
    rot_anchor = _pm_rows(rotation, layout["rot_left"], layout["rot_right"])
    return _step_rows(points, rot_anchor, _ROT_GAIN * np.abs(rotation))


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------

def apply_permutation(
    projected: np.ndarray,
    placement: tuple[int, int, int, int],
) -> np.ndarray:
    """Legacy alias for applying a post-decode wiring map.

    placement[i] = which basis vector index electrode E(i+1) gets.
    """
    return apply_wiring_map(projected, placement)


def apply_wiring_map(
    projected: np.ndarray,
    wiring_map: tuple[int, int, int, int],
) -> np.ndarray:
    """Reorder columns of direct e1-e4 output according to wiring_map."""
    return projected[:, list(wiring_map)]


# ---------------------------------------------------------------------------
# Frequency axis generation
# ---------------------------------------------------------------------------

def generate_freq_axes(
    unified: dict[str, np.ndarray],
    config: FreqConfig | None = None,
    timestamps_ms: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Generate pulse/carrier frequency axes from surge plus mild main cues.

    Each axis is centered at its configured center position (0-100 scale)
    and uses the same surge polarity. Pulse and carrier each get configurable
    surge and main-speed influence, while carrier also keeps a mild
    main-position tilt.

    Returns {axis_name: values_0_100} for enabled outputs.
    """
    cfg = config or FreqConfig()
    if not cfg.enabled:
        return {}

    sample_count = len(next(iter(unified.values()))) if unified else 1
    surge_values = np.asarray(unified.get("surge", np.full(sample_count, 50.0)), dtype=np.float64)
    main_values = np.asarray(unified.get("main", np.full(sample_count, 50.0)), dtype=np.float64)

    pulse_surge_influence = float(np.clip(cfg.pulse_surge_influence, 0.0, 2.0))
    pulse_speed_influence = float(np.clip(cfg.pulse_speed_influence, 0.0, 1.0))
    carrier_surge_influence = float(np.clip(cfg.carrier_surge_influence, 0.0, 2.0))
    carrier_speed_influence = float(np.clip(cfg.carrier_speed_influence, 0.0, 1.0))

    surge = _norm(surge_values)
    # Positive surge pulls both frequency axes downward before their
    # per-axis main-motion bias is applied.
    pulse_surge_signal = np.clip(0.5 - (0.5 * surge), 0.0, 1.0)
    carrier_surge_signal = pulse_surge_signal.copy()

    main_speed_signal = 0.5 + (0.5 * _compute_speed_signal(main_values, timestamps_ms))
    main_position_signal = np.clip(main_values / 100.0, 0.0, 1.0)

    pulse_base = np.clip(
        0.5
        + (pulse_surge_influence * (pulse_surge_signal - 0.5))
        + (pulse_speed_influence * (main_speed_signal - 0.5)),
        0.0,
        1.0,
    )
    carrier_base = np.clip(
        0.5
        + (carrier_surge_influence * (carrier_surge_signal - 0.5))
        + (carrier_speed_influence * (main_speed_signal - 0.5))
        + (_CARRIER_POSITION_BIAS_GAIN * (main_position_signal - 0.5)),
        0.0,
        1.0,
    )
    result: dict[str, np.ndarray] = {}

    if cfg.freq_scale > 0:
        pf_signal = np.clip(0.5 + ((pulse_base - 0.5) * cfg.freq_scale), 0.0, 1.0)
        result["pulse_frequency"] = apply_centered_output_limits(
            pf_signal,
            center=cfg.pulse_center,
            lower=cfg.pulse_min,
            upper=cfg.pulse_max,
        ) * 100.0

    if cfg.carrier_scale > 0:
        cf_signal = np.clip(0.5 + ((carrier_base - 0.5) * cfg.carrier_scale), 0.0, 1.0)
        result["carrier_frequency"] = apply_centered_output_limits(
            cf_signal,
            center=cfg.carrier_center,
            lower=cfg.carrier_min,
            upper=cfg.carrier_max,
        ) * 100.0
        # 'frequency' is identical to carrier_frequency
        result["frequency"] = result["carrier_frequency"].copy()

    return result


# ---------------------------------------------------------------------------
# Main conversion entry point
# ---------------------------------------------------------------------------

def convert(
    axes: dict[str, list[FunscriptAction]],
    placement: tuple[int, int, int, int] | None = None,
    weights: MixWeights | None = None,
    freq_config: FreqConfig | None = None,
    layout_model: str | None = None,
) -> dict[str, list[FunscriptAction]]:
    """Run the full 6-axis → direct fourphase conversion pipeline.

    placement is a post-decode wiring map for physical channel order.

    Returns dict with keys 'e1'–'e4' and optionally
    'pulse_frequency', 'carrier_frequency', 'frequency'.
    """
    if placement is None:
        placement = PRESETS[DEFAULT_PRESET]

    timestamps, unified = unify_timeline(axes)
    if len(timestamps) == 0:
        return {}

    axial, side, rotation = mix_to_layout_controls(unified, weights, layout_model)
    decoded = decode_layout_controls(axial, side, rotation, layout_model)
    permuted = apply_wiring_map(decoded, placement)

    ts_int = timestamps.astype(np.int64)

    result: dict[str, list[FunscriptAction]] = {}
    for i, name in enumerate(["e1", "e2", "e3", "e4"]):
        positions = np.clip(np.round(permuted[:, i] * 100.0), 0, 100).astype(int)
        result[name] = [
            FunscriptAction(at=int(t), pos=int(p))
            for t, p in zip(ts_int, positions)
        ]

    # Optional frequency outputs
    freq_axes = generate_freq_axes(unified, freq_config, timestamps)
    for fname, fvals in freq_axes.items():
        positions = np.clip(np.round(fvals), 0, 100).astype(int)
        result[fname] = [
            FunscriptAction(at=int(t), pos=int(p))
            for t, p in zip(ts_int, positions)
        ]

    return result
