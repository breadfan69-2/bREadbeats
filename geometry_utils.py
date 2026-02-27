"""
Pure geometry helpers for motion generation.

This module intentionally has no dependencies on PyQt or audio-engine code.
All module-level functions are stateless pure math; the GeometryUtils class
wraps phase-accumulating orbit state used by the legacy display path.
"""

import math


# ── Pure-math primitives (stateless) ─────────────────────────────────


def quintic_ease(t: float) -> float:
    """Quintic smoothstep (Perlin's improved): 6t⁵ − 15t⁴ + 10t³.

    Zero first *and* second derivative at both endpoints — smoother
    than cubic, with no perceptible knee at start or end.
    """
    t = max(0.0, min(1.0, float(t)))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def exponential_approach(
    current: float, target: float, rate: float, dt: float,
) -> float:
    """Exponential glide: *current* → *target* at *rate* per second.

    ``rate ≈ 3.0`` closes ~95 % of the gap in 1 s.
    """
    return current + (target - current) * (1.0 - math.exp(-rate * dt))


def orbit_point(
    angle: float,
    radius: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> tuple[float, float]:
    """Cartesian (x, y) on a circle of *radius* centred at *(center_x, center_y)*."""
    return (
        center_x + radius * math.cos(angle),
        center_y + radius * math.sin(angle),
    )


def infer_orbit(
    alpha: float, beta: float, center_y: float = 0.0,
) -> tuple[float, float]:
    """Recover (angle, radius) from Cartesian position and orbit centre."""
    dy = beta - center_y
    return (math.atan2(dy, alpha), math.hypot(alpha, dy))


def radius_cap_for_center(center_y: float, y_offset: float = 0.0) -> float:
    """Largest radius that keeps the full orbit inside the [-1, 1] box."""
    eff = center_y + y_offset
    return max(0.0, min(1.0 - eff, 1.0 + eff))


def nearest_anchor_crossing(
    target_angle: float, anchor_angle: float, swing_rad: float,
) -> float:
    """Angle nearest *target_angle* within ±*swing_rad* of *anchor_angle* (mod 2π)."""
    two_pi = 2.0 * math.pi
    n = round((target_angle - anchor_angle) / two_pi)
    candidates = [anchor_angle + (n + k) * two_pi for k in (-1, 0, 1)]
    best = min(candidates, key=lambda c: abs(c - target_angle))
    delta = max(-swing_rad, min(swing_rad, target_angle - best))
    return best + delta


# ── Phase-accumulating orbit (legacy display path) ───────────────────


class GeometryUtils:
    """Generate normalized 2D coordinates from BPM, delta-time, and intensity.

    Usage:
        geom = GeometryUtils()
        x, y = geom.update(bpm=120.0, dt=1/60, intensity=0.8)

    Notes:
        - `bpm` controls angular speed (beats per minute).
        - `dt` is elapsed seconds since the previous update.
        - `intensity` controls radius (0.0 -> center, 1.0 -> full radius).
        - Returned `(x, y)` are normalized to approximately [-1.0, 1.0].
    """

    def __init__(
        self,
        phase: float = 0.0,
        y_offset: float = 0.5,
        sink_start_intensity: float = 0.25,
        beat_confidence_decay_per_second: float = 0.30,
        silence_threshold: float = 0.05,
        min_rotation_scale: float = 0.04,
    ) -> None:
        self._phase = float(phase) % 1.0
        self._y_offset = self._safe_finite(y_offset, default=0.5)
        self._sink_start_intensity = max(1e-6, self._safe_finite(sink_start_intensity, default=0.25))
        self._beat_confidence = 1.0
        self._beat_confidence_decay_per_second = max(
            0.0,
            self._safe_finite(beat_confidence_decay_per_second, default=0.30),
        )
        self._silence_threshold = max(0.0, min(1.0, self._safe_finite(silence_threshold, default=0.05)))
        self._min_rotation_scale = max(0.0, min(1.0, self._safe_finite(min_rotation_scale, default=0.04)))
        self._ghost_spin_intensity_threshold = 0.05
        self._ghost_spin_hold_seconds = 1.0
        self._low_intensity_elapsed = 0.0

    @staticmethod
    def _safe_finite(value: float, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        return parsed if math.isfinite(parsed) else float(default)

    def set_rest_parameters(self, y_offset: float, sink_start_intensity: float) -> None:
        self._y_offset = self._safe_finite(y_offset, default=self._y_offset)
        self._sink_start_intensity = max(
            1e-6,
            self._safe_finite(sink_start_intensity, default=self._sink_start_intensity),
        )

    def reset(self, phase: float = 0.0) -> None:
        self._phase = float(phase) % 1.0
        self._low_intensity_elapsed = 0.0

    def get_phase(self) -> float:
        return float(self._phase)

    def get_beat_confidence(self) -> float:
        return float(self._beat_confidence)

    def update(self, bpm: float, dt: float, intensity: float, beat_detected: bool = False) -> tuple[float, float]:
        bpm = max(0.0, self._safe_finite(bpm, default=0.0))
        dt = max(0.0, self._safe_finite(dt, default=0.0))
        intensity = max(0.0, min(1.0, self._safe_finite(intensity, default=0.0)))
        beat_detected = bool(beat_detected)

        if intensity < self._ghost_spin_intensity_threshold:
            self._low_intensity_elapsed += dt
        else:
            self._low_intensity_elapsed = 0.0

        if self._low_intensity_elapsed > self._ghost_spin_hold_seconds:
            bpm = 0.0

        if beat_detected:
            self._beat_confidence = 1.0
        else:
            decay = self._beat_confidence_decay_per_second * dt
            self._beat_confidence = max(0.0, self._beat_confidence - decay)

        if intensity <= self._silence_threshold:
            return 0.0, float(self._y_offset)

        speed_scale = max(self._min_rotation_scale, self._beat_confidence)
        cycles_per_second = (bpm / 60.0) * speed_scale
        self._phase = (self._phase + cycles_per_second * dt) % 1.0

        angle = self._phase * 2.0 * math.pi
        radius = intensity

        orbit_x = radius * math.cos(angle)
        orbit_y = radius * math.sin(angle)

        # Below-center rest state:
        # As intensity approaches 0, blend from orbit position to a stable
        # below-center point (0, +y_offset) using linear interpolation.
        sink_start_intensity = self._sink_start_intensity
        sink_mix = (sink_start_intensity - intensity) / sink_start_intensity
        sink_mix = max(0.0, min(1.0, sink_mix))
        confidence_sink = 1.0 - self._beat_confidence
        sink_mix = 1.0 - ((1.0 - sink_mix) * (1.0 - confidence_sink))

        x = (1.0 - sink_mix) * orbit_x
        y = ((1.0 - sink_mix) * orbit_y) + (sink_mix * self._y_offset)
        return x, y
