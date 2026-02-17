"""
bREadbeats - Stroke Mapper (Decision-Only Adapter)

Thin runtime adapter that delegates signal intelligence to beat_intelligence.
Legacy drawing/trajectory generation has been removed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from audio_engine import BeatEvent
from beat_intelligence import BeatDecision, BeatIntelligence
from config import Config
from network_engine import TCodeCommand


@dataclass
class StrokeState:
    alpha: float = 0.0
    beta: float = -0.70
    last_time: float = 0.0


@dataclass
class PlannedTrajectory:
    """Compatibility shim kept for import stability during migration."""


@dataclass
class PendingStrokeChange:
    """Compatibility shim kept for import stability during migration."""


class MotionMode:
    FULL_STROKE = "full_stroke"
    CREEP_MICRO = "creep_micro"


class StrokeMapper:
    """Decision-based stroke mapper that consumes BeatIntelligence."""

    def __init__(
        self,
        config: Config,
        send_callback: Optional[Callable[[TCodeCommand], None]] = None,
        get_volume: Optional[Callable[[], float]] = None,
        audio_engine=None,
    ):
        self.config = config
        self.send_callback = send_callback
        self.get_volume = get_volume if get_volume is not None else (lambda: 1.0)
        self.audio_engine = audio_engine

        self.state = StrokeState()
        self._park_y = -0.70

        # Compatibility attributes used by main.py
        self._trajectory = None
        self._micro_effects_enabled = True

        self._orbit_phase = 0.0
        self._active_interval_beats = 8
        self._last_trigger_kind = "creep"
        self._park_radius = 0.70
        self._park_angle = float(np.pi)
        self._journey_start_angle = self._park_angle
        self._journey_total_rotation = float(2.0 * np.pi)
        self._last_journey_completion = 1.0

        self._intelligence = BeatIntelligence(config=self.config, audio_engine=self.audio_engine, park_y=self._park_y)

    # ----- compatibility proxies for existing tests/introspection -----
    @property
    def _sub_bass_energy(self) -> float:
        return self._intelligence.energies.sub_bass

    @_sub_bass_energy.setter
    def _sub_bass_energy(self, value: float) -> None:
        self._intelligence.energies.sub_bass = float(value)

    @property
    def _low_mid_energy(self) -> float:
        return self._intelligence.energies.low_mid

    @_low_mid_energy.setter
    def _low_mid_energy(self, value: float) -> None:
        self._intelligence.energies.low_mid = float(value)

    @property
    def _mid_energy(self) -> float:
        return self._intelligence.energies.mid

    @_mid_energy.setter
    def _mid_energy(self, value: float) -> None:
        self._intelligence.energies.mid = float(value)

    @property
    def _high_energy(self) -> float:
        return self._intelligence.energies.high

    @_high_energy.setter
    def _high_energy(self, value: float) -> None:
        self._intelligence.energies.high = float(value)

    def configure_geometry_rest_state(self, y_offset: float, sink_start_intensity: float = 0.25) -> None:
        """Retained runtime API for compatibility; park remains fixed at -0.70."""
        self._park_y = -0.70
        self._intelligence.set_park_y(self._park_y)

    def get_current_position(self) -> tuple[float, float]:
        return self.state.alpha, self.state.beta

    def process_beat(self, event: BeatEvent) -> Optional[TCodeCommand]:
        now = event.monotonic_timestamp if getattr(event, "monotonic_timestamp", 0.0) > 0 else time.perf_counter()
        dt = max(1e-4, now - self.state.last_time) if self.state.last_time > 0 else (1.0 / 60.0)
        self.state.last_time = now

        self._intelligence.set_audio_engine(self.audio_engine)
        decision = self._intelligence.build_decision(event=event, dt=dt)

        self._active_interval_beats = decision.interval_beats
        self._last_trigger_kind = decision.trigger_kind

        if decision.silence_active:
            alpha = 0.0
            beta = self._park_y
            volume = 0.0
            self._last_journey_completion = 1.0
        else:
            progress = float(np.clip(decision.journey_completion, 0.0, 1.0))
            smooth_progress = self._s_curve(progress)

            started_new_journey = bool(progress <= 1e-9 and self._last_journey_completion > 1e-9)
            if started_new_journey:
                self._journey_start_angle = float(self._orbit_phase)
                self._journey_total_rotation = self._compute_landing_rotation(
                    start_angle=self._journey_start_angle,
                    interval_beats=decision.interval_beats,
                )

            angle = float(self._journey_start_angle + (self._journey_total_rotation * smooth_progress))
            self._orbit_phase = float(angle % (2.0 * np.pi))

            # S-curve pulse for bloom: expands then contracts back to park radius.
            bloom_delta = float(max(0.0, decision.radius_bloom - self._park_radius))
            radius = float(self._park_radius + (bloom_delta * np.sin(np.pi * smooth_progress)))

            center_offset_y = self._intelligence.compute_treble_lift(progress)

            alpha = float(radius * np.sin(angle))
            beta = float(center_offset_y + (radius * np.cos(angle)))
            volume = float(np.clip(self.get_volume(), 0.0, 1.0))

            self._last_journey_completion = progress

        alpha = float(np.clip(alpha, -1.0, 1.0))
        beta = float(np.clip(beta, -1.0, 1.0))
        self.state.alpha = alpha
        self.state.beta = beta

        return TCodeCommand(alpha=alpha, beta=beta, duration_ms=25, volume=volume)

    @staticmethod
    def _s_curve(progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        return float(p * p * (3.0 - (2.0 * p)))

    def _compute_landing_rotation(self, start_angle: float, interval_beats: int) -> float:
        # Rotation policy: 1 turn for shorter journeys, 2 turns for 4/8 beat journeys.
        turns = 2 if int(interval_beats) >= 4 else 1
        phase_to_park = float((self._park_angle - start_angle) % (2.0 * np.pi))
        return float(phase_to_park + (2.0 * np.pi * max(0, turns - 1)))

    # ----- compatibility helper wrappers (used by tests) -----
    def _build_beat_decision(self, event: BeatEvent, silence_active: bool, dt: float) -> BeatDecision:
        trigger_kind = self._intelligence.classify_trigger(event)
        interval_beats = self._intelligence.interval_beats_for_trigger(trigger_kind)
        radius_bloom = self._intelligence.compute_radius_bloom_from_sub_bass()
        journey_completion = self._intelligence.update_journey_progress(
            trigger_kind=trigger_kind,
            interval_beats=interval_beats,
            event=event,
            dt=dt,
        )
        self._intelligence.active_interval_beats = interval_beats
        self._intelligence.last_trigger_kind = trigger_kind
        return BeatDecision(
            trigger_kind=trigger_kind,
            interval_beats=interval_beats,
            radius_bloom=radius_bloom,
            silence_active=bool(silence_active),
            journey_completion=journey_completion,
        )

    def _classify_trigger(self, event: BeatEvent) -> str:
        return self._intelligence.classify_trigger(event)

    def _interval_beats_for_trigger(self, trigger_kind: str) -> int:
        return self._intelligence.interval_beats_for_trigger(trigger_kind)

    def _compute_radius_bloom_from_sub_bass(self) -> float:
        return self._intelligence.compute_radius_bloom_from_sub_bass()

    def _compute_treble_lift(self, journey_completion: float) -> float:
        return self._intelligence.compute_treble_lift(journey_completion)
