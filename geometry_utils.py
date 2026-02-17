"""
Pure geometry helper for motion generation.

This module intentionally has no dependencies on PyQt or audio-engine code.
"""

import math


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

    def get_phase(self) -> float:
        return float(self._phase)

    def get_beat_confidence(self) -> float:
        return float(self._beat_confidence)

    def update(self, bpm: float, dt: float, intensity: float, beat_detected: bool = False) -> tuple[float, float]:
        bpm = max(0.0, self._safe_finite(bpm, default=0.0))
        dt = max(0.0, self._safe_finite(dt, default=0.0))
        intensity = max(0.0, min(1.0, self._safe_finite(intensity, default=0.0)))
        beat_detected = bool(beat_detected)

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
