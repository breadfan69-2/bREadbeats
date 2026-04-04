from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable

import numpy as np

from pmv_audio_analysis import AudioTimeline
from pmv_beat_engine import BeatTimeline
from pmv_funscript_io import FunscriptAction


FEATURE_COLUMNS = [
    "rms",
    "spectral_flux",
    "sub_bass_energy",
    "low_mid_energy",
    "mid_energy",
    "high_energy",
    "low_high_ratio",
    "spectral_centroid_hz",
    "spectral_flatness",
    "rms_mean_10s",
    "rms_std_10s",
    "flux_mean_10s",
    "bass_mean_10s",
    "energy_trend_10s",
]


@dataclass(slots=True)
class MLConfig:
    enabled: bool = True
    strength: float = 0.55
    cadence_mode: str = "auto"
    rule_fit_path: str = ""
    teaching_rule_fit_path: str = ""
    min_confidence: float = 0.12
    bidirectional_smooth: bool = True
    smooth_alpha: float = 0.15


@dataclass(slots=True)
class BeatIntelligenceResult:
    speed_mult: float
    cadence_hint: int
    energy_fullness: float
    fill_gate_pass: bool


@dataclass(slots=True)
class MappingConfig:
    pitch_range: float = 100.0
    amplitude_centering: float = 0.0
    center_offset: float = 0.0
    overflow_mode: str = "crop"
    energy_multiplier: float = 10.0
    ml_config: MLConfig = field(default_factory=MLConfig)
    min_command_delay_ms: float = 150.0
    points_per_second: int = 25
    pos_min: int = 0
    pos_max: int = 100


@dataclass(slots=True)
class PositionTimeline:
    actions: list[FunscriptAction]
    beat_actions: list[FunscriptAction]
    speed_profile: np.ndarray
    ml_results: list[BeatIntelligenceResult] | None


def _report(
    progress_callback: Callable[[str, float], None] | None,
    message: str,
    percent: float,
) -> None:
    if progress_callback is None:
        return
    progress_callback(message, float(percent))


def _resolve_model_path(config: MLConfig) -> Path:
    candidates = [
        str(config.rule_fit_path).strip(),
        str(config.teaching_rule_fit_path).strip(),
        "datasets/rule_fit.json",
        "defaults/learning/rule_fit.tranquilizer_blend.json",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError("No rule-fit model found. Checked configured path, teaching path, and datasets/rule_fit.json")


def _load_rule_fit_model(config: MLConfig) -> dict:
    path = _resolve_model_path(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid rule-fit model payload")
    if payload.get("status") != "ok":
        raise ValueError("Rule-fit model is not marked as status=ok")
    return payload


def _nearest_frame_index(frame_times_ms: np.ndarray, time_ms: float) -> int:
    if len(frame_times_ms) == 0:
        return 0
    target = float(time_ms)
    idx = int(np.searchsorted(frame_times_ms, target, side="left"))
    if idx <= 0:
        return 0
    if idx >= len(frame_times_ms):
        return len(frame_times_ms) - 1
    before = float(frame_times_ms[idx - 1])
    after = float(frame_times_ms[idx])
    return idx if abs(after - target) < abs(target - before) else idx - 1


def _frame_rms_db(timeline: AudioTimeline, frame_idx: int) -> float:
    if len(timeline.samples) == 0:
        return -120.0

    sr = max(1, int(timeline.sample_rate))
    frame_idx = int(np.clip(frame_idx, 0, max(0, len(timeline.frame_times_ms) - 1)))
    center_s = float(timeline.frame_times_ms[frame_idx]) / 1000.0
    center_sample = int(round(center_s * sr))

    if len(timeline.frame_times_ms) > 1:
        hop_ms = float(np.median(np.diff(timeline.frame_times_ms)))
    else:
        hop_ms = 20.0
    half_window = max(1, int(round((hop_ms / 1000.0) * sr)))

    lo = max(0, center_sample - half_window)
    hi = min(len(timeline.samples), center_sample + half_window)
    segment = timeline.samples[lo:hi]
    if len(segment) == 0:
        return -120.0

    rms = float(np.sqrt(np.mean(np.square(segment.astype(np.float64)))))
    if rms <= 1e-12:
        return -120.0
    return float(np.clip(20.0 * np.log10(rms), -120.0, 12.0))


def _build_feature_vector(timeline: AudioTimeline, beat_time_ms: float) -> dict[str, float]:
    idx = _nearest_frame_index(timeline.frame_times_ms, beat_time_ms)

    sub = float(timeline.band_energies_per_frame.get("sub_bass", np.zeros(0))[idx]) if len(timeline.feature_frames) else 0.0
    low = float(timeline.band_energies_per_frame.get("low_mid", np.zeros(0))[idx]) if len(timeline.feature_frames) else 0.0
    mid = float(timeline.band_energies_per_frame.get("mid", np.zeros(0))[idx]) if len(timeline.feature_frames) else 0.0
    high = float(timeline.band_energies_per_frame.get("high", np.zeros(0))[idx]) if len(timeline.feature_frames) else 0.0

    return {
        "rms": _frame_rms_db(timeline, idx),
        "spectral_flux": float(timeline.spectral_flux_per_frame[idx]) if len(timeline.spectral_flux_per_frame) else 0.0,
        "sub_bass_energy": sub,
        "low_mid_energy": low,
        "mid_energy": mid,
        "high_energy": high,
        "low_high_ratio": float((sub + low + 1e-10) / (high + 1e-10)),
        "spectral_centroid_hz": float(timeline.spectral_centroid_per_frame[idx]) if len(timeline.spectral_centroid_per_frame) else 0.0,
        "spectral_flatness": float(timeline.spectral_flatness_per_frame[idx]) if len(timeline.spectral_flatness_per_frame) else 0.0,
        "rms_mean_10s": float(timeline.rms_mean_10s[idx]) if len(timeline.rms_mean_10s) else 0.0,
        "rms_std_10s": float(timeline.rms_std_10s[idx]) if len(timeline.rms_std_10s) else 0.0,
        "flux_mean_10s": float(timeline.flux_mean_10s[idx]) if len(timeline.flux_mean_10s) else 0.0,
        "bass_mean_10s": float(timeline.bass_mean_10s[idx]) if len(timeline.bass_mean_10s) else 0.0,
        "energy_trend_10s": float(timeline.energy_trend_10s[idx]) if len(timeline.energy_trend_10s) else 0.0,
    }


def _predict_speed_mult(model: dict, features: dict[str, float]) -> float:
    cols = list(model.get("feature_columns", FEATURE_COLUMNS))
    norm = model.get("normalization", {})
    mean = norm.get("mean", {}) if isinstance(norm, dict) else {}
    std = norm.get("std", {}) if isinstance(norm, dict) else {}

    def _num(value: object, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return float(default)
        return float(default)

    raw = np.array([_num(features.get(c), _num(mean.get(c), 0.0)) for c in cols], dtype=np.float64)
    mu = np.array([_num(mean.get(c), 0.0) for c in cols], dtype=np.float64)
    sigma = np.array([max(1e-8, _num(std.get(c), 1.0)) for c in cols], dtype=np.float64)
    x = (raw - mu) / sigma

    model_spec = model.get("models", {}).get("speed_mult", {})
    intercept = float(model_spec.get("intercept", 0.0))
    coef = model_spec.get("coefficients", {})
    weights = np.array([float(coef.get(c, 0.0)) for c in cols], dtype=np.float64)

    value = intercept + float(np.dot(weights, x))
    return float(np.clip(value, 0.0, 1.0))


def _derive_cadence(speed_mult: float, model: dict) -> int:
    cadence_rule = model.get("cadence_rule", {}) if isinstance(model, dict) else {}
    quiet = float(cadence_rule.get("quiet_threshold", 0.30))
    mid = float(cadence_rule.get("mid_threshold", 0.47))
    mapping = cadence_rule.get("mapping", {}) if isinstance(cadence_rule, dict) else {}

    if speed_mult < quiet:
        return int(mapping.get("quiet", 4))
    if speed_mult < mid:
        return int(mapping.get("mid", 2))
    return int(mapping.get("loud", 1))


def _spectrum_fill_gate(timeline: AudioTimeline, beat_time_ms: float, beat_type: str) -> bool:
    idx = _nearest_frame_index(timeline.frame_times_ms, beat_time_ms)
    db = _frame_rms_db(timeline, idx)

    thresholds = {
        "downbeat": -35.0,
        "beat": -40.0,
        "syncopation": -45.0,
    }
    threshold = float(thresholds.get(str(beat_type), -40.0))
    return bool(db >= threshold)


def _energy_fullness(features: dict[str, float]) -> float:
    values = np.array(
        [
            float(features.get("sub_bass_energy", 0.0)),
            float(features.get("low_mid_energy", 0.0)),
            float(features.get("mid_energy", 0.0)),
            float(features.get("high_energy", 0.0)),
        ],
        dtype=np.float64,
    )
    return float(np.clip(np.mean(values), 0.0, 1.0))


def _ema(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []
    a = float(np.clip(alpha, 0.0, 1.0))
    out = [float(values[0])]
    for value in values[1:]:
        out.append(float(out[-1] + a * (float(value) - out[-1])))
    return out


def _interp_at(frame_times_ms: np.ndarray, values: np.ndarray, time_ms: float) -> float:
    if len(frame_times_ms) == 0 or len(values) == 0:
        return 0.0
    n = min(len(frame_times_ms), len(values))
    if n <= 0:
        return 0.0
    if n == 1:
        return float(values[0])
    return float(
        np.interp(
            float(time_ms),
            np.asarray(frame_times_ms[:n], dtype=np.float64),
            np.asarray(values[:n], dtype=np.float64),
        )
    )


def _normalize_pitch_series(pitch: np.ndarray) -> np.ndarray:
    arr = np.asarray(pitch, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span = hi - lo
    if span <= 1e-9:
        return np.full(arr.shape, 0.5, dtype=np.float64)
    return np.clip((arr - lo) / span, 0.0, 1.0)


def _compute_raw_position(
    beat_time_ms: float,
    timeline: AudioTimeline,
    ml_result: BeatIntelligenceResult | None,
    config: MappingConfig,
    is_upstroke: bool,
) -> float:
    pitch_norm_series = _normalize_pitch_series(timeline.pitch_per_frame)
    pitch_norm = _interp_at(timeline.frame_times_ms, pitch_norm_series, beat_time_ms)
    energy_norm = _interp_at(timeline.frame_times_ms, timeline.rms_per_frame, beat_time_ms)
    energy_norm = float(np.clip(energy_norm, 0.0, 1.5))

    pitch_bias = (100.0 - float(config.pitch_range)) * 0.5
    pitch_component = (pitch_norm * float(config.pitch_range)) + pitch_bias
    energy_component = energy_norm * (float(config.energy_multiplier) / 10.0) * 50.0
    centering_component = float(config.amplitude_centering) * energy_norm
    offset = pitch_component + centering_component + float(config.center_offset)

    if ml_result is not None and config.ml_config.enabled:
        ml_factor = 0.5 + 0.5 * float(ml_result.speed_mult) * float(np.clip(config.ml_config.strength, 0.0, 1.0))
        energy_component *= float(np.clip(ml_factor, 0.0, 1.0))

    if is_upstroke:
        return float(energy_component + offset)
    return float((-energy_component) + offset)


def _filter_beats_by_cadence(
    beats: BeatTimeline,
    ml_results: list[BeatIntelligenceResult],
    cadence_mode: str,
) -> list[tuple[int, bool]]:
    mode = str(cadence_mode)
    pairs: list[tuple[int, bool]] = []
    is_upstroke = True

    for idx, _beat in enumerate(beats.beats):
        if mode == "fixed_1":
            cadence = 1
        elif mode == "fixed_2":
            cadence = 2
        elif mode == "fixed_4":
            cadence = 4
        else:
            cadence = int(np.clip(ml_results[idx].cadence_hint if idx < len(ml_results) else 1, 1, 4))

        if idx % max(1, cadence) != 0:
            continue
        pairs.append((idx, is_upstroke))
        is_upstroke = not is_upstroke

    return pairs


def _fold_value(value: float, pos_min: int, pos_max: int) -> float:
    lo = float(min(pos_min, pos_max))
    hi = float(max(pos_min, pos_max))
    width = hi - lo
    if width <= 0.0:
        return lo
    shifted = float(value) - lo
    wrapped = shifted % (2.0 * width)
    if wrapped <= width:
        return lo + wrapped
    return hi - (wrapped - width)


def _apply_overflow(
    actions: list[FunscriptAction],
    mode: str,
    pos_min: int = 0,
    pos_max: int = 100,
) -> list[FunscriptAction]:
    lo = int(min(pos_min, pos_max))
    hi = int(max(pos_min, pos_max))
    if not actions:
        return []

    out: list[FunscriptAction] = []
    normalized_mode = str(mode).strip().lower()

    for action in actions:
        at = int(action.at)
        pos = float(action.pos)

        if normalized_mode == "crop":
            out.append(FunscriptAction(at=at, pos=int(np.clip(int(round(pos)), lo, hi))))
            continue

        if normalized_mode == "fold":
            out.append(FunscriptAction(at=at, pos=int(round(_fold_value(pos, lo, hi)))))
            continue

        if normalized_mode == "bounce":
            # Degenerate ranges cannot bounce; keep output in-range without looping.
            if lo == hi:
                out.append(FunscriptAction(at=at, pos=lo))
                continue

            current = pos
            if not np.isfinite(current):
                out.append(FunscriptAction(at=at, pos=lo))
                continue

            # Protect against pathological values that could otherwise iterate indefinitely.
            bounce_guard = 0
            max_bounces = 4096
            while current < lo or current > hi:
                bounce_guard += 1
                if bounce_guard > max_bounces:
                    current = _fold_value(current, lo, hi)
                    break
                if current > hi:
                    overshoot = current - hi
                    out.append(FunscriptAction(at=at, pos=hi))
                    current = hi - overshoot
                    continue
                overshoot = lo - current
                out.append(FunscriptAction(at=at, pos=lo))
                current = lo + overshoot
            out.append(FunscriptAction(at=at, pos=int(round(np.clip(current, lo, hi)))))
            continue

        out.append(FunscriptAction(at=at, pos=int(np.clip(int(round(pos)), lo, hi))))

    out.sort(key=lambda a: a.at)
    return out


def _interpolate_actions(
    actions: list[FunscriptAction],
    points_per_second: int,
    min_command_delay_ms: float,
) -> list[FunscriptAction]:
    if len(actions) <= 1:
        return list(actions)

    pps = max(1, int(points_per_second))
    min_delay = max(1, int(round(float(min_command_delay_ms))))
    step_ms = max(1.0, 1000.0 / float(pps))

    dense: list[FunscriptAction] = [FunscriptAction(at=int(actions[0].at), pos=int(actions[0].pos))]
    for prev, curr in zip(actions, actions[1:]):
        t0 = int(prev.at)
        t1 = int(curr.at)
        p0 = float(prev.pos)
        p1 = float(curr.pos)
        dt = t1 - t0
        if dt <= 0:
            continue

        steps = max(1, int(np.floor(dt / step_ms)))
        for k in range(1, steps + 1):
            ratio = float(k) / float(steps)
            at = int(round(t0 + ratio * dt))
            pos = int(round(p0 + ratio * (p1 - p0)))
            dense.append(FunscriptAction(at=at, pos=pos))

    filtered: list[FunscriptAction] = [dense[0]]
    for action in dense[1:]:
        if action.at - filtered[-1].at >= min_delay:
            filtered.append(action)

    # Keep the true endpoint without violating monotonic time.
    if dense[-1].at > filtered[-1].at:
        if dense[-1].at - filtered[-1].at >= min_delay:
            filtered.append(dense[-1])
        else:
            filtered[-1] = dense[-1]

    return filtered


def _compute_speed_profile(actions: list[FunscriptAction]) -> np.ndarray:
    n = len(actions)
    if n == 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.array([0.0], dtype=np.float32)

    speeds = np.zeros(n, dtype=np.float64)
    for i in range(n - 1):
        dt = max(1.0, float(actions[i + 1].at - actions[i].at))
        dp = abs(float(actions[i + 1].pos - actions[i].pos))
        speeds[i] = dp / dt
    speeds[-1] = speeds[-2]

    return speeds.astype(np.float32)


def generate_positions(
    timeline: AudioTimeline,
    beats: BeatTimeline,
    config: MappingConfig,
    progress_callback: Callable[[str, float], None] | None = None,
) -> PositionTimeline:
    beat_items = beats.beats
    if not beat_items:
        return PositionTimeline(
            actions=[],
            beat_actions=[],
            speed_profile=np.array([], dtype=np.float32),
            ml_results=[],
        )

    _report(progress_callback, "Running ML intelligence...", 0.0)
    ml_results = compute_beat_intelligence(timeline, beats, config.ml_config, progress_callback=None)
    _report(progress_callback, "Running ML intelligence...", 30.0)

    _report(progress_callback, "Mapping positions...", 30.0)
    selected = _filter_beats_by_cadence(beats, ml_results, config.ml_config.cadence_mode)

    raw_actions: list[FunscriptAction] = []
    for idx, is_upstroke in selected:
        beat = beat_items[idx]
        ml_result = ml_results[idx] if idx < len(ml_results) else None
        raw_pos = _compute_raw_position(beat.time_ms, timeline, ml_result, config, is_upstroke)
        raw_actions.append(FunscriptAction(at=int(round(beat.time_ms)), pos=int(round(raw_pos))))

    _report(progress_callback, "Mapping positions...", 60.0)

    _report(progress_callback, "Applying overflow...", 60.0)
    beat_actions = _apply_overflow(raw_actions, config.overflow_mode, config.pos_min, config.pos_max)
    _report(progress_callback, "Applying overflow...", 75.0)

    _report(progress_callback, "Interpolating...", 75.0)
    actions = _interpolate_actions(beat_actions, config.points_per_second, config.min_command_delay_ms)
    actions = _apply_overflow(actions, "crop", config.pos_min, config.pos_max)
    _report(progress_callback, "Interpolating...", 90.0)

    _report(progress_callback, "Computing speed profile...", 90.0)
    speed_profile = _compute_speed_profile(actions)
    _report(progress_callback, "Computing speed profile...", 100.0)

    return PositionTimeline(
        actions=actions,
        beat_actions=beat_actions,
        speed_profile=speed_profile,
        ml_results=ml_results,
    )


def compute_beat_intelligence(
    timeline: AudioTimeline,
    beats: BeatTimeline,
    config: MLConfig,
    progress_callback: Callable[[str, float], None] | None = None,
) -> list[BeatIntelligenceResult]:
    """Compute offline ML intelligence outputs for each beat."""
    beat_items = beats.beats
    if not beat_items:
        return []

    if not config.enabled:
        return [
            BeatIntelligenceResult(
                speed_mult=0.5,
                cadence_hint=1,
                energy_fullness=0.5,
                fill_gate_pass=True,
            )
            for _ in beat_items
        ]

    _report(progress_callback, "Loading rule-fit model...", 5.0)
    model = _load_rule_fit_model(config)

    raw_speeds: list[float] = []
    cadences: list[int] = []
    fullness: list[float] = []
    fill_passes: list[bool] = []

    _report(progress_callback, "Running per-beat inference...", 10.0)
    total = max(1, len(beat_items))
    for i, beat in enumerate(beat_items):
        features = _build_feature_vector(timeline, beat.time_ms)
        raw_speed = _predict_speed_mult(model, features)
        blended = 0.5 + float(np.clip(config.strength, 0.0, 1.0)) * (raw_speed - 0.5)
        speed = float(np.clip(blended, 0.0, 1.0))

        cadence = _derive_cadence(raw_speed, model)
        if str(config.cadence_mode) == "fixed_1":
            cadence = 1
        elif str(config.cadence_mode) == "fixed_2":
            cadence = 2
        elif str(config.cadence_mode) == "fixed_4":
            cadence = 4

        raw_speeds.append(speed)
        cadences.append(int(cadence))
        fullness.append(_energy_fullness(features))
        fill_passes.append(_spectrum_fill_gate(timeline, beat.time_ms, beat.beat_type))

        if i % max(1, total // 10) == 0:
            _report(progress_callback, "Running per-beat inference...", 10.0 + (60.0 * i / total))

    _report(progress_callback, "Applying smoothing...", 70.0)
    if config.bidirectional_smooth and len(raw_speeds) > 1:
        fwd = _ema(raw_speeds, config.smooth_alpha)
        bwd = list(reversed(_ema(list(reversed(raw_speeds)), config.smooth_alpha)))
        speeds = [float(np.clip((a + b) * 0.5, 0.0, 1.0)) for a, b in zip(fwd, bwd)]
    else:
        speeds = [float(np.clip(v, 0.0, 1.0)) for v in raw_speeds]

    # Re-derive cadence from smoothed speed in auto mode only.
    if str(config.cadence_mode) == "auto":
        cadences = [_derive_cadence(v, model) for v in speeds]

    _report(progress_callback, "Packaging intelligence outputs...", 92.0)
    results = [
        BeatIntelligenceResult(
            speed_mult=float(speeds[i]),
            cadence_hint=int(cadences[i]),
            energy_fullness=float(np.clip(fullness[i], 0.0, 1.0)),
            fill_gate_pass=bool(fill_passes[i]),
        )
        for i in range(len(beat_items))
    ]

    _report(progress_callback, "Beat intelligence complete", 100.0)
    return results


__all__ = [
    "BeatIntelligenceResult",
    "FEATURE_COLUMNS",
    "MLConfig",
    "MappingConfig",
    "PositionTimeline",
    "compute_beat_intelligence",
    "generate_positions",
]
