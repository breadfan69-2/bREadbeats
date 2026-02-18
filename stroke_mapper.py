"""
bREadbeats - Stroke Mapper (Decision-Only Adapter)

Thin runtime adapter that delegates signal intelligence to beat_intelligence.
Legacy drawing/trajectory generation has been removed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from audio_engine import BeatEvent
from beat_intelligence import BeatDecision, BeatIntelligence
from config import Config
from network_engine import TCodeCommand


@dataclass
class StrokeState:
    alpha: float = 0.0
    beta: float = 0.70
    last_time: float = 0.0


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
        self._park_y = 0.70

        self._orbit_phase = 0.0
        self._active_interval_beats = 8
        self._last_trigger_kind = "creep"
        self._park_radius = 0.70
        self._park_angle = 0.0
        self._journey_start_angle = self._park_angle
        self._journey_total_rotation = float(2.0 * np.pi)
        self._last_journey_completion = 1.0

        self._intelligence = BeatIntelligence(config=self.config, audio_engine=self.audio_engine, park_y=self._park_y)

        self._learning_enabled = bool(getattr(self.config.beat, "teaching_learning_enabled", False))
        self._learning_use_fitted_rules = bool(getattr(self.config.beat, "teaching_use_fitted_rules", False))
        self._learning_apply_in_circle_mode = bool(getattr(self.config.beat, "teaching_apply_in_circle_mode", False))
        self._learning_isolation_mode = bool(getattr(self.config.beat, "teaching_isolation_mode", False))
        self._learning_strength = float(getattr(self.config.beat, "teaching_learning_strength", 0.0) or 0.0)
        self._learning_min_confidence = float(getattr(self.config.beat, "teaching_min_confidence", 0.0) or 0.0)
        self._learning_no_motion_bias = float(getattr(self.config.beat, "teaching_no_motion_bias", 1.0) or 1.0)
        self._learning_rule_fit_path = str(getattr(self.config.beat, "teaching_rule_fit_path", "") or "")
        self._learning_model: Optional[dict] = None

        # Push initial learning config to intelligence
        self._sync_learning_to_intelligence()

    def configure_geometry_rest_state(self, y_offset: float, sink_start_intensity: float = 0.25) -> None:
        self._park_y = 0.70
        self._intelligence.set_park_y(self._park_y)

    def configure_learning(
        self,
        *,
        enabled: bool,
        use_fitted_rules: bool,
        apply_in_circle_mode: bool,
        isolation_mode: bool,
        learning_strength: float,
        min_confidence: float,
        no_motion_bias: float,
        rule_fit_path: str,
    ) -> None:
        self._learning_enabled = bool(enabled)
        self._learning_use_fitted_rules = bool(use_fitted_rules)
        self._learning_apply_in_circle_mode = bool(apply_in_circle_mode)
        self._learning_isolation_mode = bool(isolation_mode)
        self._learning_strength = float(learning_strength)
        self._learning_min_confidence = float(min_confidence)
        self._learning_no_motion_bias = float(no_motion_bias)
        self._learning_rule_fit_path = str(rule_fit_path or "")

        if self._learning_enabled and self._learning_use_fitted_rules:
            self._try_load_learning_model()
        else:
            self._learning_model = None

        # Forward to BeatIntelligence
        self._sync_learning_to_intelligence()

    def _sync_learning_to_intelligence(self) -> None:
        """Forward learning config and model path to BeatIntelligence."""
        self._intelligence.configure_learning(
            enabled=self._learning_enabled,
            use_fitted_rules=self._learning_use_fitted_rules,
            strength=self._learning_strength,
            min_confidence=self._learning_min_confidence,
            no_motion_bias=self._learning_no_motion_bias,
            rule_fit_path=self._learning_rule_fit_path,
        )

    def _try_load_learning_model(self) -> None:
        path_text = str(self._learning_rule_fit_path or "").strip()
        if not path_text:
            self._learning_model = None
            return

        try:
            path = Path(path_text)
            if not path.exists() or not path.is_file():
                self._learning_model = None
                return
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._learning_model = payload if isinstance(payload, dict) else None
        except Exception:
            self._learning_model = None

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
            # Apply silence fade (gradual, not binary)
            fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
            alpha = 0.0
            beta = self._park_y
            volume = float(np.clip(self.get_volume() * fade, 0.0, 1.0))
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
            # Apply post-silence ramp to volume
            ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
            volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))

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
