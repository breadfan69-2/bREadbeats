from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from pmv_funscript_io import FunscriptAction


PRESET_CURVES: dict[str, list[tuple[float, float]]] = {
    "linear": [(0.0, 0.0), (1.0, 1.0)],
    "ease_in": [(0.0, 0.0), (0.5, 0.2), (1.0, 1.0)],
    "ease_out": [(0.0, 0.0), (0.5, 0.8), (1.0, 1.0)],
    "bell": [(0.0, 0.0), (0.25, 0.3), (0.5, 1.0), (0.75, 0.3), (1.0, 0.0)],
    "inverted": [(0.0, 1.0), (1.0, 0.0)],
    "s_curve": [(0.0, 0.0), (0.2, 0.1), (0.5, 0.5), (0.8, 0.9), (1.0, 1.0)],
    "sharp_peak": [(0.0, 0.0), (0.4, 0.1), (0.5, 1.0), (0.6, 0.1), (1.0, 0.0)],
    "gentle_wave": [(0.0, 0.2), (0.25, 0.7), (0.5, 0.3), (0.75, 0.8), (1.0, 0.4)],
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
    pulse_freq_mode: int = 0  # 0=Ratio 1=Hz 2=Speed 3=BandEnergy 4=Hybrid 5=MotionEnvelope
    pulse_freq_band: str = "sub_bass"
    pulse_freq_weight: float = 1.0
    carrier_frequency_ratio: float = 3.0
    carrier_freq_mode: int = 1  # 0=Ratio 1=Hz 2=Speed 3=BandEnergy 4=Hybrid 5=MotionEnvelope
    carrier_freq_band: str = "mid"
    carrier_freq_weight: float = 1.0
    volume_ramp_ratio: float = 20.0
    pulse_rise_ratio: float = 2.0
    pulse_width_ratio: float = 3.0
    rest_level: float = 0.4
    ramp_up_duration_sec: float = 1.5
    ramp_percent_per_hour: float = 15.0
    speed_window_sec: float = 5.0
    points_per_second: int = 25
    pulse_freq_range_start: float = 30.0
    pulse_freq_range_end: float = 70.0
    smooth_frequency_sec: float = 2.0
    smooth_pulse_frequency_sec: float = 3.0
    smooth_carrier_frequency_sec: float = 3.0
    smooth_volume_sec: float = 3.0
    smooth_pulse_rise_sec: float = 3.0
    smooth_pulse_width_sec: float = 3.0
    enabled_axes: set[str] = field(default_factory=lambda: {"main"})
    alpha_beta_mode: str = "restim"  # "restim" or "orbital"
    orbital_blend: float = 0.0  # 0.0 = pure restim, 1.0 = pure orbital
    preview_tcode_mode: str = "threephase"  # "threephase" (L0/L1) or "fourphase" (E1-E4)


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

    # Use positions directly as 0.0-1.0 (pos / 100) — matches restim's
    # transform where beta amplitude is proportional to actual stroke size.
    pos_01 = np.array([float(a.pos) / 100.0 for a in main_actions], dtype=np.float64)

    rng = np.random.default_rng(1337)
    direction = 1.0
    flip_prob = float(np.clip(direction_flip_probability, 0.0, 1.0))

    total_segments = len(main_actions) - 1

    for i in range(total_segments):
        start_t = float(main_actions[i].at)
        end_t = float(main_actions[i + 1].at)
        duration = max(1.0, end_t - start_t)

        start_p = float(pos_01[i])
        end_p = float(pos_01[i + 1])

        center = (end_p + start_p) / 2.0
        radius = (start_p - end_p) / 2.0

        if rng.random() < flip_prob:
            direction *= -1.0

        n = _restim_segment_points(duration, start_p, end_p)

        t = np.linspace(0.0, duration, n, endpoint=False)
        theta = np.linspace(0.0, np.pi, n, endpoint=False)

        x = center + (radius * np.cos(theta))
        y = (radius * direction * np.sin(theta)) + 0.5

        for k in range(n):
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
    """Apply rest-level reduction when speed is near zero, with two-phase
    ramp-up after rest→active transitions:
      Phase 1 (fast):  0 → 90% of recovery in first 1/3 of duration
      Phase 2 (slow): 90 → 100% of recovery in remaining 2/3
    This gives a sharp initial return followed by a gentle final approach,
    matching the feel of hand-scripted funscripts."""
    is_rest = speed < 0.02
    result = values.copy()
    # Apply rest_level immediately to rest points
    result[is_rest] *= rest_level
    if ramp_up_duration_ms <= 0:
        return result
    # Find rest→active transition indices
    transitions: list[float] = []
    for i in range(1, len(is_rest)):
        if is_rest[i - 1] and not is_rest[i]:
            transitions.append(float(t[i]))
    if not transitions:
        return result
    # Two-phase ramp: fast phase = first 1/3, slow phase = remaining 2/3
    fast_dur = ramp_up_duration_ms / 3.0    # e.g. 500ms when total=1500
    total_dur = ramp_up_duration_ms          # e.g. 1500ms
    knee_frac = 0.9  # reach 90% of recovery at end of fast phase
    recovery = 1.0 - rest_level
    for k in range(len(transitions)):
        t0 = transitions[k]
        hi = t0 + total_dur
        mask = (t >= t0) & (t <= hi)
        if not np.any(mask):
            continue
        elapsed = t[mask] - t0
        # Piecewise: fast phase then slow phase
        in_fast = elapsed <= fast_dur
        progress = np.empty_like(elapsed)
        # Fast phase: linear 0 → knee_frac over [0, fast_dur]
        progress[in_fast] = knee_frac * elapsed[in_fast] / max(1.0, fast_dur)
        # Slow phase: linear knee_frac → 1.0 over [fast_dur, total_dur]
        slow_elapsed = elapsed[~in_fast] - fast_dur
        slow_dur = max(1.0, total_dur - fast_dur)
        progress[~in_fast] = knee_frac + (1.0 - knee_frac) * slow_elapsed / slow_dur
        progress = np.clip(progress, 0.0, 1.0)
        mult = rest_level + recovery * progress
        result[mask] = values[mask] * mult
    return np.clip(result, 0.0, 1.0)


def _generate_aux_axes(
    main_actions: list[FunscriptAction],
    speed_norm: np.ndarray,
    config: AxisConfig,
    duration_ms: int,
    alpha_actions: list[FunscriptAction] | None = None,
    audio_timeline: Any | None = None,
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

    # ── Audio-derived signals (interpolated to action timestamps) ──
    centroid_norm: np.ndarray | None = None
    band_cache: dict[str, np.ndarray] = {}
    if audio_timeline is not None:
        frame_t = np.asarray(audio_timeline.frame_times_ms, dtype=np.float64)
        # Spectral centroid normalised to 0-1 (centroid up to ~8000 Hz)
        raw_centroid = np.asarray(audio_timeline.spectral_centroid_per_frame, dtype=np.float64)
        c_max = max(float(np.max(raw_centroid)), 1.0) if len(raw_centroid) > 0 else 1.0
        centroid_interp = np.interp(t, frame_t, raw_centroid / c_max)
        centroid_norm = np.clip(centroid_interp, 0.0, 1.0)
        # Band energies — lazily interpolated per band on first use
        be = getattr(audio_timeline, "band_energies_per_frame", None)
        if isinstance(be, dict):
            for band_name, band_arr in be.items():
                raw = np.asarray(band_arr, dtype=np.float64)
                b_max = max(float(np.max(raw)), 1e-12) if len(raw) > 0 else 1e-12
                band_cache[band_name] = np.clip(np.interp(t, frame_t, raw / b_max), 0.0, 1.0)

    def _mix_vec(a: np.ndarray, b: np.ndarray, ratio: float) -> np.ndarray:
        r = max(1.0, float(ratio))
        return (a * (r - 1.0) + b) / r

    def _apply_weight(signal: np.ndarray, weight: float) -> np.ndarray:
        """Centre-weighted scaling: 0.5 + (signal - 0.5) * weight."""
        w = max(0.0, min(1.0, float(weight)))
        return np.clip(0.5 + (signal - 0.5) * w, 0.0, 1.0)

    def _mode_dispatch(mode: int, ratio: float, band: str, weight: float) -> np.ndarray:
        """Compute a normalised 0-1 signal based on the selected mode."""
        if mode == 1 and centroid_norm is not None:  # Hz
            return _apply_weight(centroid_norm, weight)
        if mode == 2:  # Speed
            return _apply_weight(speed_norm, weight)
        if mode == 3 and band in band_cache:  # Band Energy
            return _apply_weight(band_cache[band], weight)
        if mode == 4 and centroid_norm is not None:  # Hybrid
            base = _mix_vec(speed_norm, alpha_norm, ratio)
            return np.clip(base * (0.5 + centroid_norm * 0.5), 0.0, 1.0)
        if mode == 5:  # Motion Envelope (legacy frequency behavior)
            return _mix_vec(ramp, speed_norm, ratio)
        # Mode 0 (Ratio) or fallback when audio unavailable
        return _mix_vec(speed_norm, alpha_norm, ratio)

    # --- frequency: combine(ramp, speed, ratio) ---
    freq = _mix_vec(ramp, speed_norm, config.frequency_ramp_ratio)

    # --- pulse_frequency: mode-selectable ---
    pulse_freq = _mode_dispatch(
        config.pulse_freq_mode, config.pulse_frequency_ratio,
        config.pulse_freq_band, config.pulse_freq_weight,
    )

    # Apply time-based ramp to pulse_frequency: rescale so the average
    # trends from pulse_freq_range_start → pulse_freq_range_end (pos units)
    # over the duration.  The mode_dispatch 0-1 signal modulates within
    # a time-varying window centred on the ramp.
    pf_lo = config.pulse_freq_range_start / 100.0
    pf_hi = config.pulse_freq_range_end / 100.0
    progress_t = np.clip((t - t[0]) / max(1.0, t[-1] - t[0]), 0.0, 1.0)
    pf_centre = pf_lo + (pf_hi - pf_lo) * progress_t
    # Scale the 0-1 reactive signal as deviation around the centre
    pf_half_range = np.minimum(pf_centre, 1.0 - pf_centre)
    pulse_freq = pf_centre + (pulse_freq - 0.5) * 2.0 * pf_half_range

    # --- carrier_frequency: mode-selectable (new axis) ---
    carrier_freq = _mode_dispatch(
        config.carrier_freq_mode, config.carrier_frequency_ratio,
        config.carrier_freq_band, config.carrier_freq_weight,
    )

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

    # Cap to practical carrier-cycle ranges (pos 0-20 for width, 0-18 for rise)
    # and enforce physical constraint: rise can never exceed width.
    pulse_width = np.clip(pulse_width, 0.0, 1.0) * (20.0 / 100.0)
    pulse_rise = np.clip(pulse_rise, 0.0, 1.0) * (18.0 / 100.0)
    pulse_rise = np.minimum(pulse_rise, pulse_width)

    # --- Per-axis envelope smoothing ---
    # Compute median sample interval for kernel sizing
    if len(t) > 1:
        median_dt_sec = float(np.median(np.diff(t))) / 1000.0
    else:
        median_dt_sec = 1.0 / max(1, config.points_per_second)

    def _smooth(signal: np.ndarray, window_sec: float) -> np.ndarray:
        if window_sec <= 0 or len(signal) < 3:
            return signal
        kernel_n = max(1, int(round(window_sec / median_dt_sec)))
        if kernel_n < 2:
            return signal
        # Use a simple uniform moving-average (fast, no scipy needed)
        kernel = np.ones(kernel_n) / kernel_n
        padded = np.pad(signal, kernel_n // 2, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='same')[kernel_n // 2 : kernel_n // 2 + len(signal)]
        return smoothed

    freq = _smooth(freq, config.smooth_frequency_sec)
    pulse_freq = _smooth(pulse_freq, config.smooth_pulse_frequency_sec)
    carrier_freq = _smooth(carrier_freq, config.smooth_carrier_frequency_sec)
    volume = _smooth(volume, config.smooth_volume_sec)
    pulse_rise = _smooth(pulse_rise, config.smooth_pulse_rise_sec)
    pulse_width = _smooth(pulse_width, config.smooth_pulse_width_sec)
    # Re-enforce constraint after smoothing
    pulse_rise = np.minimum(pulse_rise, pulse_width)

    def to_actions(values: np.ndarray) -> list[FunscriptAction]:
        return [FunscriptAction(at=int(main_actions[i].at), pos=int(np.clip(round(values[i] * 100.0), 0, 100))) for i in range(n)]

    return {
        "frequency": to_actions(np.clip(freq, 0.0, 1.0)),
        "pulse_frequency": to_actions(np.clip(pulse_freq, 0.0, 1.0)),
        "carrier_frequency": to_actions(np.clip(carrier_freq, 0.0, 1.0)),
        "volume": to_actions(np.clip(volume, 0.0, 1.0)),
        "pulse_rise": to_actions(pulse_rise),
        "pulse_width": to_actions(pulse_width),
    }


def convert_to_2d(
    main_actions: list[FunscriptAction],
    config: AxisConfig,
    duration_ms: int,
    progress_callback: Callable[[str, float], None] | None = None,
    audio_timeline: Any | None = None,
) -> MultiAxisResult:
    actions = _sorted_actions(main_actions)
    result: dict[str, list[FunscriptAction]] = {"main": actions}

    if not actions:
        return MultiAxisResult(axes=result)

    _report(progress_callback, "Computing speed timeline...", 10.0)
    speed_norm = _compute_speed(actions, config.speed_window_sec)

    alpha: list[FunscriptAction] = []
    beta: list[FunscriptAction] = []

    # Always compute alpha/beta so TCode preview can send L0/L1 regardless of
    # which axis checkboxes are enabled in the UI.
    _report(progress_callback, "Converting main axis to 2D...", 35.0)
    alpha, beta = _convert_restim_style(actions, config.direction_flip_probability)
    result["alpha"] = alpha
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
    aux = _generate_aux_axes(actions, speed_norm, config, duration_ms, alpha_actions=alpha if alpha else None, audio_timeline=audio_timeline)
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
