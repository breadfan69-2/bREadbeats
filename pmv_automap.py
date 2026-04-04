from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import importlib.util
from typing import Callable

import numpy as np

from pmv_beat_engine import BeatTimeline
from pmv_audio_analysis import AudioTimeline
from pmv_position_mapper import MappingConfig, generate_positions


_scipy_opt_spec = importlib.util.find_spec("scipy.optimize")
if _scipy_opt_spec is not None:  # pragma: no cover - optional dependency path
    _scipy_opt = importlib.import_module("scipy.optimize")
    _HAS_SCIPY = True
else:
    _scipy_opt = None
    _HAS_SCIPY = False


@dataclass(slots=True)
class AutomapConfig:
    enabled: bool = False
    target_y_position: float = 20.0
    target_speed: float = 250.0
    target_speed_pct: float = 65.0
    optimization_mode: str = "cmeanv2"
    optimize_ml_strength: bool = True
    max_iter: int = 120


def _report(
    progress_callback: Callable[[str, float], None] | None,
    message: str,
    percent: float,
) -> None:
    if progress_callback is None:
        return
    progress_callback(message, float(percent))


def _bounds(optimize_ml_strength: bool) -> list[tuple[float, float]]:
    base = [
        (-200.0, 200.0),  # pitch_range
        (0.0, 100.0),  # energy_multiplier
        (-200.0, 200.0),  # amplitude_centering
        (-300.0, 300.0),  # center_offset
    ]
    if optimize_ml_strength:
        base.append((0.0, 1.0))
    return base


def _clip_vector(x: np.ndarray, b: list[tuple[float, float]]) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    for i, (lo, hi) in enumerate(b):
        out[i] = float(np.clip(out[i], lo, hi))
    return out


def _vector_from_config(config: MappingConfig, optimize_ml_strength: bool) -> np.ndarray:
    values = [
        float(config.pitch_range),
        float(config.energy_multiplier),
        float(config.amplitude_centering),
        float(config.center_offset),
    ]
    if optimize_ml_strength:
        values.append(float(config.ml_config.strength))
    return np.asarray(values, dtype=np.float64)


def _apply_vector(base: MappingConfig, x: np.ndarray, optimize_ml_strength: bool) -> MappingConfig:
    arr = np.asarray(x, dtype=np.float64)
    ml_cfg = replace(base.ml_config)
    if optimize_ml_strength and len(arr) >= 5:
        ml_cfg = replace(ml_cfg, strength=float(np.clip(arr[4], 0.0, 1.0)))

    return replace(
        base,
        pitch_range=float(arr[0]),
        energy_multiplier=float(arr[1]),
        amplitude_centering=float(arr[2]),
        center_offset=float(arr[3]),
        ml_config=ml_cfg,
    )


def _quality_metrics(
    timeline: AudioTimeline,
    beats: BeatTimeline,
    config: MappingConfig,
) -> tuple[float, float, float, float]:
    result = generate_positions(timeline, beats, config)
    if not result.actions:
        return 50.0, 0.0, 0.0, 100.0

    positions = np.array([float(a.pos) for a in result.actions], dtype=np.float64)
    avg_pos = float(np.mean(positions))
    spread = float(np.max(positions) - np.min(positions))

    speeds = np.asarray(result.speed_profile, dtype=np.float64)
    if speeds.size == 0:
        speeds = np.array([0.0], dtype=np.float64)
    speed_var = float(np.var(speeds))

    return avg_pos, spread, speed_var, float(np.mean(speeds))


def _objective(
    x: np.ndarray,
    bounds: list[tuple[float, float]],
    timeline: AudioTimeline,
    beats: BeatTimeline,
    base_config: MappingConfig,
    auto_cfg: AutomapConfig,
    optimize_ml_strength: bool,
) -> float:
    clipped = _clip_vector(x, bounds)
    cfg = _apply_vector(base_config, clipped, optimize_ml_strength)

    avg_pos, spread, speed_var, _ = _quality_metrics(timeline, beats, cfg)
    y_penalty = abs(avg_pos - float(auto_cfg.target_y_position))

    result = generate_positions(timeline, beats, cfg)
    speeds = np.asarray(result.speed_profile, dtype=np.float64)
    if speeds.size == 0:
        speeds = np.array([0.0], dtype=np.float64)

    target_norm = float(np.clip(float(auto_cfg.target_speed) / 400.0, 0.0, 1.0))
    pct_above = float(np.mean(speeds > target_norm) * 100.0)
    speed_pct_penalty = abs(pct_above - float(auto_cfg.target_speed_pct))

    mode = str(auto_cfg.optimization_mode).strip().lower()
    if mode == "cmean":
        return float(speed_var + (0.25 * y_penalty))
    if mode == "clen":
        return float((-spread) + (0.50 * y_penalty))

    # cmeanv2 default
    return float(speed_pct_penalty + (0.35 * y_penalty))


def _fallback_random_search(
    timeline: AudioTimeline,
    beats: BeatTimeline,
    base_config: MappingConfig,
    auto_cfg: AutomapConfig,
    optimize_ml_strength: bool,
    progress_callback: Callable[[str, float], None] | None,
) -> MappingConfig:
    bounds = _bounds(optimize_ml_strength)
    x_best = _vector_from_config(base_config, optimize_ml_strength)
    score_best = _objective(x_best, bounds, timeline, beats, base_config, auto_cfg, optimize_ml_strength)

    rng = np.random.default_rng(1337)
    max_iter = max(20, int(auto_cfg.max_iter))

    for i in range(max_iter):
        trial = x_best.copy()
        for j, (lo, hi) in enumerate(bounds):
            span = hi - lo
            step = 0.12 * span
            trial[j] = float(np.clip(trial[j] + rng.normal(0.0, step), lo, hi))

        score = _objective(trial, bounds, timeline, beats, base_config, auto_cfg, optimize_ml_strength)
        if score < score_best:
            x_best = trial
            score_best = score

        if i % max(1, max_iter // 40) == 0:
            _report(progress_callback, "Automap optimization (fallback)...", 10.0 + (80.0 * i / max_iter))

    return _apply_vector(base_config, _clip_vector(x_best, bounds), optimize_ml_strength)


def automap_optimize(
    timeline: AudioTimeline,
    beats: BeatTimeline,
    base_config: MappingConfig,
    automap_config: AutomapConfig,
    progress_callback: Callable[[str, float], None] | None = None,
) -> MappingConfig:
    if not automap_config.enabled:
        return base_config

    _report(progress_callback, "Starting automap optimization...", 0.0)

    optimize_ml_strength = bool(automap_config.optimize_ml_strength)
    b = _bounds(optimize_ml_strength)
    x0 = _clip_vector(_vector_from_config(base_config, optimize_ml_strength), b)

    avg0, _, _, _ = _quality_metrics(timeline, beats, base_config)
    x_seed = x0.copy()
    if len(x_seed) >= 4:
        # center_offset is a strong first-order control over average output position
        # so we nudge it toward the requested target to avoid flat-start local minima.
        delta = float(automap_config.target_y_position) - avg0
        x_seed[3] = float(np.clip(x_seed[3] + (2.5 * delta), b[3][0], b[3][1]))

    def _score(x: np.ndarray) -> float:
        return _objective(
            x,
            b,
            timeline,
            beats,
            base_config,
            automap_config,
            optimize_ml_strength,
        )

    if not _HAS_SCIPY:
        seeded_base = _apply_vector(base_config, x_seed, optimize_ml_strength)
        optimized = _fallback_random_search(
            timeline,
            beats,
            seeded_base,
            automap_config,
            optimize_ml_strength,
            progress_callback,
        )
        _report(progress_callback, "Automap optimization complete", 100.0)
        return optimized

    assert _scipy_opt is not None

    max_iter = max(20, int(automap_config.max_iter))
    eval_count = 0
    progress_stride = max(1, max_iter // 40)

    def _objective_wrapped(x: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        if eval_count % progress_stride == 0 or eval_count == 1:
            pct = 10.0 + (80.0 * min(eval_count, max_iter) / max_iter)
            _report(progress_callback, "Automap optimization...", pct)
        return _objective(
            x,
            b,
            timeline,
            beats,
            base_config,
            automap_config,
            optimize_ml_strength,
        )

    _report(progress_callback, "Running Nelder-Mead optimization...", 10.0)
    x_start = x_seed

    result = _scipy_opt.minimize(
        _objective_wrapped,
        x_start,
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-3, "adaptive": True},
    )

    x_best = _clip_vector(np.asarray(result.x, dtype=np.float64), b)
    candidates = [x0, x_seed, x_best]
    scores = [_score(c) for c in candidates]
    x_final = candidates[int(np.argmin(scores))]
    optimized = _apply_vector(base_config, x_final, optimize_ml_strength)
    _report(progress_callback, "Automap optimization complete", 100.0)
    return optimized


__all__ = [
    "AutomapConfig",
    "automap_optimize",
]
