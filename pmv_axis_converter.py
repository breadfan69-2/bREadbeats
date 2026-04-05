from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pmv_funscript_io import FunscriptAction


PRESET_CURVES: dict[str, list[tuple[float, float]]] = {
    "linear": [(0.0, 0.0), (1.0, 1.0)],
    "ease_in": [(0.0, 0.0), (0.5, 0.2), (1.0, 1.0)],
    "ease_out": [(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)],
    "bell": [(0.0, 0.0), (0.25, 0.3), (0.5, 1.0), (0.75, 0.3), (1.0, 0.0)],
}


@dataclass(slots=True)
class AxisConfig:
    direction_flip_probability: float = 0.0
    min_distance: float = 0.1
    speed_threshold_pct: float = 50.0
    prostate_algorithm: str = "standard"
    prostate_volume_mult: float = 1.5
    e1_curve: str = "linear"
    e2_curve: str = "ease_in"
    e3_curve: str = "ease_out"
    e4_curve: str = "bell"
    e_custom_points: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    e_phase_shift: dict[str, float] = field(default_factory=lambda: {"e1": 0.0, "e2": 0.0, "e3": 0.0, "e4": 0.0})
    e_min_segment_sec: float = 0.5
    frequency_ramp_ratio: float = 2.0
    pulse_frequency_ratio: float = 3.0
    volume_ramp_ratio: float = 20.0
    pulse_rise_ratio: float = 2.0
    pulse_width_ratio: float = 3.0
    rest_level: float = 0.4
    ramp_up_duration_sec: float = 1.0
    ramp_percent_per_hour: float = 15.0
    speed_window_sec: float = 5.0
    points_per_second: int = 25
    enabled_axes: set[str] = field(default_factory=lambda: {"main"})


@dataclass(slots=True)
class MultiAxisResult:
    axes: dict[str, list[FunscriptAction]]


def _report(
    progress_callback: Callable[[str, float], None] | None,
    message: str,
    percent: float,
) -> None:
    if progress_callback is None:
        return
    progress_callback(message, float(percent))


def _sorted_actions(actions: list[FunscriptAction]) -> list[FunscriptAction]:
    return sorted((FunscriptAction(int(a.at), int(a.pos)) for a in actions), key=lambda a: a.at)


def _mix(a: float, b: float, ratio: float) -> float:
    r = max(1.0, float(ratio))
    return float((float(a) * (r - 1.0) + float(b)) / r)


def _compute_speed(actions: list[FunscriptAction], window_sec: float = 5.0) -> np.ndarray:
    n = len(actions)
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([0.0], dtype=np.float64)

    at = np.array([float(a.at) for a in actions], dtype=np.float64)
    pos = np.array([float(a.pos) for a in actions], dtype=np.float64)
    window_ms = max(1.0, float(window_sec) * 1000.0)

    # O(n log n) via cumulative sum instead of O(n²) inner diff loop
    cumsum = np.concatenate(([0.0], np.cumsum(np.abs(np.diff(pos)))))

    speed = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        t_start = at[i] - window_ms
        j = int(np.searchsorted(at, t_start, side="left"))
        if i <= j:
            continue
        dt = max(1.0, at[i] - at[j])
        dp = cumsum[i] - cumsum[j]
        speed[i] = dp / dt

    speed[-1] = speed[-2] if n >= 2 else speed[-1]
    return np.clip(speed / 1.0, 0.0, 1.0)


def _radius_for_speed(speed_norm: float, min_distance: float, speed_threshold_pct: float) -> float:
    r_min = float(np.clip(min_distance, 0.05, 0.5))
    r_max = 0.5
    threshold = max(0.01, float(speed_threshold_pct) / 100.0)
    scale = float(np.clip(speed_norm / threshold, 0.0, 1.0))
    return float(r_min + ((r_max - r_min) * scale))


def _to_action(time_ms: int, x: float) -> FunscriptAction:
    return FunscriptAction(at=int(time_ms), pos=int(np.clip(round(x * 100.0), 0, 100)))


def _restim_segment_points(duration_ms: float, start_p: float, end_p: float) -> int:
    if abs(start_p - end_p) <= 1e-9:
        return 1
    if duration_ms <= 100.0:
        return 2
    if duration_ms <= 200.0:
        return 3
    if duration_ms <= 300.0:
        return 4
    if duration_ms <= 400.0:
        return 5
    return 6


def _convert_restim_style(
    main_actions: list[FunscriptAction],
    direction_flip_probability: float,
) -> tuple[list[FunscriptAction], list[FunscriptAction]]:
    alpha: list[FunscriptAction] = []
    beta: list[FunscriptAction] = []

    if len(main_actions) <= 0:
        return alpha, beta

    if len(main_actions) == 1:
        t0 = int(main_actions[0].at)
        return [_to_action(t0, 0.5)], [_to_action(t0, 0.5)]

    # Normalize positions to 0.0-1.0 so the circle math is relative to the
    # actual stroke range rather than absolute position.  This keeps alpha/beta
    # proportional when automap / centering compress the main axis.
    raw = np.array([float(a.pos) for a in main_actions], dtype=np.float64)
    lo, hi = float(np.min(raw)), float(np.max(raw))
    span = hi - lo
    if span < 1.0:
        norm_pos = np.full_like(raw, 0.5)
    else:
        norm_pos = (raw - lo) / span  # 0.0 to 1.0

    rng = np.random.default_rng(1337)
    direction = 1.0
    flip_prob = float(np.clip(direction_flip_probability, 0.0, 1.0))

    total_segments = len(main_actions) - 1

    for i in range(total_segments):
        start_t = float(main_actions[i].at)
        end_t = float(main_actions[i + 1].at)
        duration = max(1.0, end_t - start_t)

        start_p = float(norm_pos[i])
        end_p = float(norm_pos[i + 1])

        center = (end_p + start_p) / 2.0
        radius = (start_p - end_p) / 2.0

        if rng.random() < flip_prob:
            direction *= -1.0

        # Dense sampling: at least ~25 pts/sec, minimum from ReStim logic
        n_restim = _restim_segment_points(duration, start_p, end_p)
        n_rate = max(2, int(round(duration / 40.0)))
        n = max(n_restim, n_rate) + 1  # +1 for endpoint-inclusive count

        t = np.linspace(0.0, duration, n, endpoint=True)
        theta = np.linspace(0.0, np.pi, n, endpoint=True)

        x = center + (radius * np.cos(theta))
        y = (radius * direction * np.sin(theta)) + 0.5

        # Skip first point of non-first segments (duplicate of previous endpoint)
        k_start = 1 if i > 0 else 0

        for k in range(k_start, n):
            time_ms = int(round(start_t + float(t[k])))
            alpha.append(_to_action(time_ms, float(np.clip(x[k], 0.0, 1.0))))
            beta.append(_to_action(time_ms, float(np.clip(y[k], 0.0, 1.0))))

    return alpha, beta





def _response_lookup(points: list[tuple[float, float]], x: float) -> float:
    if not points:
        return float(np.clip(x, 0.0, 1.0))

    ordered = sorted((float(px), float(py)) for px, py in points)
    xs = np.array([p[0] for p in ordered], dtype=np.float64)
    ys = np.array([p[1] for p in ordered], dtype=np.float64)
    x_clamped = float(np.clip(x, 0.0, 1.0))
    return float(np.interp(x_clamped, xs, ys))


def _apply_response_curve(
    main_actions: list[FunscriptAction],
    curve_name: str,
    custom_points: list[tuple[float, float]] | None,
    phase_shift_pct: float = 0.0,
    min_segment_sec: float = 0.5,
) -> list[FunscriptAction]:
    points = custom_points if custom_points else PRESET_CURVES.get(str(curve_name), PRESET_CURVES["linear"])
    mapped: list[FunscriptAction] = []

    for action in main_actions:
        norm = float(np.clip(action.pos / 100.0, 0.0, 1.0))
        out = _response_lookup(points, norm)
        mapped.append(FunscriptAction(at=int(action.at), pos=int(np.clip(round(out * 100.0), 0, 100))))

    shift_ms = int(round(max(0.0, float(phase_shift_pct)) / 100.0 * max(0.0, float(min_segment_sec)) * 1000.0))
    if shift_ms > 0:
        mapped = [FunscriptAction(at=int(a.at + shift_ms), pos=int(a.pos)) for a in mapped]

    return mapped


def _convert_prostate_standard(
    alpha: list[FunscriptAction],
    beta: list[FunscriptAction],
    volume_mult: float,
) -> tuple[list[FunscriptAction], list[FunscriptAction]]:
    mult = float(np.clip(volume_mult, 1.0, 3.0))
    a_out: list[FunscriptAction] = []
    b_out: list[FunscriptAction] = []

    for a, b in zip(alpha, beta):
        ax = 50.0 + ((float(a.pos) - 50.0) * mult)
        by = 50.0 + ((float(b.pos) - 50.0) * mult)
        a_out.append(FunscriptAction(at=int(a.at), pos=int(np.clip(round(ax), 0, 100))))
        b_out.append(FunscriptAction(at=int(b.at), pos=int(np.clip(round(by), 0, 100))))

    return a_out, b_out


def _convert_prostate_tear_shaped(
    main_actions: list[FunscriptAction],
    speed_norm: np.ndarray,
    min_distance: float,
    volume_mult: float,
) -> tuple[list[FunscriptAction], list[FunscriptAction]]:
    alpha: list[FunscriptAction] = []
    beta: list[FunscriptAction] = []

    mult = float(np.clip(volume_mult, 1.0, 3.0))
    for i, action in enumerate(main_actions):
        theta = (float(action.pos) / 100.0) * (2.0 * np.pi)
        radius = _radius_for_speed(float(speed_norm[min(i, len(speed_norm) - 1)]), min_distance, 50.0)
        if 0.0 <= theta < (2.0 * np.pi / 3.0):
            shape = 1.0 - (0.5 * theta / (2.0 * np.pi / 3.0))
        elif (2.0 * np.pi / 3.0) <= theta < (4.0 * np.pi / 3.0):
            shape = 0.5
        else:
            shape = 0.5 + (0.5 * (theta - 4.0 * np.pi / 3.0) / (2.0 * np.pi / 3.0))
        r = float(np.clip(radius * shape * mult, 0.05, 0.5))
        x = 0.5 + (r * float(np.cos(theta)))
        y = 0.5 + (r * float(np.sin(theta)))
        alpha.append(_to_action(action.at, x))
        beta.append(_to_action(action.at, y))

    return alpha, beta


def _make_volume_ramp(t: np.ndarray, duration: float, ramp_percent_per_hour: float) -> np.ndarray:
    """Build a 4-point volume ramp matching edger's approach:
    0 → start_value (at 10s) → 1.0 (near end) → 0 (at end)."""
    if len(t) == 0:
        return np.array([], dtype=np.float64)
    file_duration_hours = max(1e-9, duration / 1000.0 / 3600.0)
    total_increase = (ramp_percent_per_hour / 100.0) * file_duration_hours
    start_value = max(0.0, 1.0 - total_increase)
    # Key points in ms
    t_start = float(t[0])
    t_10s = t_start + 10000.0
    t_peak = float(t[-1]) - 1.0 if len(t) >= 2 else float(t[-1])
    t_end = float(t[-1])
    # Build piecewise ramp (vectorised)
    ramp = np.piecewise(
        t,
        [
            t <= t_start,
            (t > t_start) & (t <= t_10s),
            (t > t_10s) & (t <= t_peak),
            (t > t_peak) & (t <= t_end),
            t > t_end,
        ],
        [
            0.0,
            lambda _t: start_value * ((_t - t_start) / max(1.0, t_10s - t_start)),
            lambda _t: start_value + (1.0 - start_value) * ((_t - t_10s) / max(1.0, t_peak - t_10s)),
            lambda _t: 1.0 - ((_t - t_peak) / max(1.0, t_end - t_peak)),
            0.0,
        ],
    )
    return np.clip(ramp, 0.0, 1.0)


def _detect_rest_and_ramp(
    values: np.ndarray,
    speed: np.ndarray,
    t: np.ndarray,
    rest_level: float,
    ramp_up_duration_ms: float,
) -> np.ndarray:
    """Apply rest-level reduction when speed is near zero, with smooth ramp-up
    after rest→active transitions (matching edger's combine_funscripts logic)."""
    is_rest = speed < 0.02
    result = values.copy()
    # Apply rest_level immediately to rest points
    result[is_rest] *= rest_level
    if ramp_up_duration_ms <= 0:
        return result
    # Find rest→active transition indices
    half_dur = ramp_up_duration_ms / 2.0
    transitions: list[float] = []
    for i in range(1, len(is_rest)):
        if is_rest[i - 1] and not is_rest[i]:
            transitions.append(float(t[i]))
    if not transitions:
        return result
    # Apply ramp-up around each transition (vectorised via searchsorted)
    trans = np.asarray(transitions, dtype=np.float64)
    # For each time point, find the nearest preceding transition within range
    idx = np.searchsorted(trans, t, side='right')  # first transition > t[i]
    for k in range(len(trans)):
        lo = trans[k] - half_dur
        hi = trans[k] + half_dur
        mask = (t >= lo) & (t <= hi)
        if not np.any(mask):
            continue
        diff = t[mask] - trans[k]
        progress = np.clip((diff + half_dur) / ramp_up_duration_ms, 0.0, 1.0)
        mult = rest_level + (1.0 - rest_level) * progress
        result[mask] = values[mask] * mult
    return np.clip(result, 0.0, 1.0)


def _generate_aux_axes(
    main_actions: list[FunscriptAction],
    speed_norm: np.ndarray,
    config: AxisConfig,
    duration_ms: int,
    alpha_actions: list[FunscriptAction] | None = None,
) -> dict[str, list[FunscriptAction]]:
    n = len(main_actions)
    if n == 0:
        return {}

    t = np.array([float(a.at) for a in main_actions], dtype=np.float64)
    duration = max(1.0, float(max(duration_ms, int(t[-1]))))

    # Proper volume ramp (edger-style 4-point curve, not linear t/duration)
    ramp = _make_volume_ramp(t, duration, config.ramp_percent_per_hour)
    ramp_inv = 1.0 - ramp
    speed_inv = 1.0 - speed_norm

    # Alpha signal for pulse_frequency (edger & restim both use alpha)
    if alpha_actions:
        alpha_norm = np.interp(
            t,
            [float(a.at) for a in alpha_actions],
            [float(a.pos) / 100.0 for a in alpha_actions],
        )
    else:
        # Fallback: use position as proxy when alpha unavailable
        alpha_norm = np.array([float(a.pos) / 100.0 for a in main_actions], dtype=np.float64)

    def _mix_vec(a: np.ndarray, b: np.ndarray, ratio: float) -> np.ndarray:
        r = max(1.0, float(ratio))
        return (a * (r - 1.0) + b) / r

    # --- frequency: combine(ramp, speed, ratio) ---
    freq = _mix_vec(ramp, speed_norm, config.frequency_ramp_ratio)

    # --- pulse_frequency: combine(speed, alpha, ratio) ---
    pulse_freq = _mix_vec(speed_norm, alpha_norm, config.pulse_frequency_ratio)

    # --- volume: combine(ramp, speed, ratio) + rest detection + ramp-up ---
    volume_raw = _mix_vec(ramp, speed_norm, config.volume_ramp_ratio)
    volume = _detect_rest_and_ramp(
        volume_raw, speed_norm, t,
        config.rest_level,
        config.ramp_up_duration_sec * 1000.0,
    )

    # --- pulse_rise: combine(ramp_inverted, speed_inverted, ratio) ---
    pulse_rise = _mix_vec(ramp_inv, speed_inv, config.pulse_rise_ratio)

    # --- pulse_width: combine(speed, inverted_position, ratio) ---
    inv_pos = 1.0 - np.array([a.pos for a in main_actions], dtype=np.float64) / 100.0
    pulse_width = _mix_vec(speed_norm, inv_pos, config.pulse_width_ratio)

    def to_actions(values: np.ndarray) -> list[FunscriptAction]:
        return [FunscriptAction(at=int(main_actions[i].at), pos=int(np.clip(round(values[i] * 100.0), 0, 100))) for i in range(n)]

    return {
        "frequency": to_actions(np.clip(freq, 0.0, 1.0)),
        "pulse_frequency": to_actions(np.clip(pulse_freq, 0.0, 1.0)),
        "volume": to_actions(np.clip(volume, 0.0, 1.0)),
        "pulse_rise": to_actions(np.clip(pulse_rise, 0.0, 1.0)),
        "pulse_width": to_actions(np.clip(pulse_width, 0.0, 1.0)),
    }


def convert_to_2d(
    main_actions: list[FunscriptAction],
    config: AxisConfig,
    duration_ms: int,
    progress_callback: Callable[[str, float], None] | None = None,
) -> MultiAxisResult:
    actions = _sorted_actions(main_actions)
    result: dict[str, list[FunscriptAction]] = {"main": actions}

    if not actions:
        return MultiAxisResult(axes=result)

    _report(progress_callback, "Computing speed timeline...", 10.0)
    speed_norm = _compute_speed(actions, config.speed_window_sec)

    need_2d = bool(config.enabled_axes.intersection({"alpha", "beta", "alpha_prostate", "beta_prostate"}))
    alpha: list[FunscriptAction] = []
    beta: list[FunscriptAction] = []

    if need_2d:
        _report(progress_callback, "Converting main axis to 2D...", 35.0)
        alpha, beta = _convert_restim_style(actions, config.direction_flip_probability)

        if "alpha" in config.enabled_axes:
            result["alpha"] = alpha
        if "beta" in config.enabled_axes:
            result["beta"] = beta

    if bool(config.enabled_axes.intersection({"alpha_prostate", "beta_prostate"})):
        _report(progress_callback, "Generating prostate axes...", 55.0)
        if str(config.prostate_algorithm).strip().lower() == "tear_shaped":
            p_alpha, p_beta = _convert_prostate_tear_shaped(actions, speed_norm, config.min_distance, config.prostate_volume_mult)
        else:
            base_a = alpha if alpha else _convert_restim_style(actions, config.direction_flip_probability)[0]
            base_b = beta if beta else _convert_restim_style(actions, config.direction_flip_probability)[1]
            p_alpha, p_beta = _convert_prostate_standard(base_a, base_b, config.prostate_volume_mult)

        if "alpha_prostate" in config.enabled_axes:
            result["alpha_prostate"] = p_alpha
        if "beta_prostate" in config.enabled_axes:
            result["beta_prostate"] = p_beta

    _report(progress_callback, "Applying response curves...", 70.0)
    for axis_name, curve_name in (("e1", config.e1_curve), ("e2", config.e2_curve), ("e3", config.e3_curve), ("e4", config.e4_curve)):
        if axis_name not in config.enabled_axes:
            continue
        result[axis_name] = _apply_response_curve(
            actions,
            curve_name,
            config.e_custom_points.get(axis_name),
            phase_shift_pct=float(config.e_phase_shift.get(axis_name, 0.0)),
            min_segment_sec=float(config.e_min_segment_sec),
        )

    _report(progress_callback, "Generating auxiliary axes...", 85.0)
    aux = _generate_aux_axes(actions, speed_norm, config, duration_ms, alpha_actions=alpha if alpha else None)
    for axis_name, axis_actions in aux.items():
        if axis_name in config.enabled_axes:
            result[axis_name] = axis_actions

    _report(progress_callback, "Axis conversion complete", 100.0)
    return MultiAxisResult(axes=result)


__all__ = [
    "AxisConfig",
    "MultiAxisResult",
    "PRESET_CURVES",
    "convert_to_2d",
]
