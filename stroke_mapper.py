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
        self._baseline_center_y = 0.70
        self._min_radius = 0.05
        self._base_center_y = self._baseline_center_y
        self._reactive_bounce_y = 0.0
        self._journey_start_total_center_y = self._baseline_center_y

        self._orbit_phase = 0.0
        self._active_interval_beats = 8
        self._last_trigger_kind = "creep"
        self._park_radius = 0.70
        self._max_radius = 1.0
        self._journey_fixed_radius = self._park_radius
        self._journey_start_radius = self._park_radius
        self._journey_learning_mult = 1.0  # frozen at journey start; never re-read mid-arc
        self._journey_center_y = self._baseline_center_y
        self._journey_park_radius = self._park_radius
        self._journey_max_radius = self._max_radius
        self._park_angle = float(np.pi / 2.0)
        self._journey_start_angle = self._park_angle
        self._journey_start_alpha = self.state.alpha
        self._journey_start_beta = self.state.beta
        self._journey_total_rotation = float(2.0 * np.pi)
        self._last_journey_completion = 1.0
        self._actual_radius = self._park_radius
        self._radius_hold_active = False
        self._radius_hold_value = self._park_radius
        self._radius_hold_start_time = 0.0
        self._angular_velocity = 0.0
        self._last_phase_for_velocity = self._orbit_phase
        self._journey_initial_speed_slope = 0.0
        self._journey_nominal_angular_speed = 0.0  # nominal speed for continuation glide
        self._orbit_phase_initialized = False  # True once orbit_phase has been actively tracked
        self._lazy_glide_active = False
        self._journey_cold_start = True
        self._journey_linked = False
        self._exit_spiral_active = False
        self._exit_spiral_progress = 0.0
        self._exit_spiral_start_angle = 0.0
        self._exit_spiral_duration_s = 0.60
        self._journey_relink_active = False
        self._journey_relink_start_radius = 0.90
        self._startup_momentum_min = 0.30
        self._startup_ramp_beats = 4.0
        self._startup_beats_seen = 0.0
        self._journey_startup_momentum = 1.0
        self._hold_start_pose_until_reactive = False
        self._idle_radius = self._min_radius
        self._silence_decay_per_beat = 0.40
        self._idle_loops_per_beat = 0.125

        # Smooth landing / settle state (exponential lerp, no oscillation)
        self._settle_active = False
        self._settle_elapsed = 0.0
        self._settle_start_angle = self._park_angle
        self._settle_decay_rate = 4.0  # higher = quicker settle, less tail linger

        # Mid-journey crossfade state: buttery blend from old trajectory
        # into new one when a beat fires before the previous journey ends.
        self._crossfade_active = False
        self._crossfade_elapsed = 0.0
        self._crossfade_duration = 0.50   # seconds – long enough to feel smooth
        self._crossfade_from_angle = 0.0  # angle at moment of transition
        self._crossfade_from_center_y = self._baseline_center_y
        self._crossfade_from_radius = self._park_radius

        # Bass-reactive jitter state (applied on creep only)
        self._bass_jitter_phase = 0.0
        self._bass_jitter_freq_ema = 0.5

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
        self._lazy_glide_active = bool(getattr(decision, "lazy_glide_active", False))

        if decision.silence_active:
            self._hold_start_pose_until_reactive = False
            # Decay-to-idle: preserve orbital motion and gradually shrink radius
            # by a beat-scaled factor until a tiny idle loop remains.
            geom = self.config.stroke.orbit_geometry.get(self._last_trigger_kind, {
                "center_y": self._baseline_center_y, "park_radius": 0.70, "max_radius": 1.0
            })
            type_center_y = float(geom["center_y"])

            bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
            if bpm <= 0.0:
                bpm = float(getattr(event, "bpm", 0.0) or 0.0)
            bpm = float(np.clip(bpm if bpm > 0.0 else 120.0, 40.0, 240.0))
            beats_elapsed = float(np.clip(dt, 1e-4, 0.25) * (bpm / 60.0))
            decay_mult = float(self._silence_decay_per_beat ** beats_elapsed)

            start_radius = float(max(self._actual_radius, self._idle_radius))
            target_radius = float(max(self._idle_radius, start_radius * decay_mult))
            self._actual_radius += 0.35 * (target_radius - self._actual_radius)
            radius = float(np.clip(self._actual_radius, self._idle_radius, 1.0))

            fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
            idle_angular_speed = float((2.0 * np.pi) * (bpm / 60.0) * self._idle_loops_per_beat)
            self._orbit_phase = float((self._orbit_phase + (idle_angular_speed * dt)) % (2.0 * np.pi))
            self._angular_velocity = float(idle_angular_speed)
            self._last_phase_for_velocity = self._orbit_phase

            self._base_center_y = self._base_center_target(
                trigger_kind=self._last_trigger_kind,
                progress=1.0,
                silence_active=True,
            )
            self._reactive_bounce_y = self._compute_reactive_bounce_y(
                event=event,
                dt=dt,
                wait_state=True,
            )
            total_center_y = float(self._base_center_y + self._reactive_bounce_y)
            orbit_radius = float(min(radius, self._radius_cap_for_center(total_center_y)))

            angle = float(self._orbit_phase)
            alpha = float(orbit_radius * np.cos(angle))
            beta = float(total_center_y + (orbit_radius * np.sin(angle)))
            volume = float(np.clip(self.get_volume() * fade, 0.0, 1.0))
            self._last_journey_completion = 1.0
        else:
            progress = float(np.clip(decision.journey_completion, 0.0, 1.0))
            creep_motion_disabled = not bool(getattr(self.config.creep, "enabled", True))

            if self._hold_start_pose_until_reactive and decision.trigger_kind == "creep":
                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))
                self._last_journey_completion = 1.0
                return TCodeCommand(
                    alpha=float(np.clip(self.state.alpha, -1.0, 1.0)),
                    beta=float(np.clip(self.state.beta, -1.0, 1.0)),
                    duration_ms=25,
                    volume=volume,
                )

            if creep_motion_disabled and decision.trigger_kind == "creep":
                # Creep disabled: use creep geometry to park
                geom = self.config.stroke.orbit_geometry.get("creep", {
                    "center_y": 0.4, "park_radius": 0.30, "max_radius": 0.60
                })
                type_park_radius = float(geom["park_radius"])
                type_max_radius = float(geom["max_radius"])
                type_center_y = float(geom["center_y"])

                self._settle_active = False
                if not self._radius_hold_active:
                    self._radius_hold_active = True
                    self._radius_hold_start_time = now
                    self._radius_hold_value = type_park_radius
                    self._actual_radius = type_park_radius  # Snap to park immediately
                self._orbit_phase = float(self._park_angle)
                self._last_phase_for_velocity = self._orbit_phase
                self._angular_velocity = 0.0

                radius = type_park_radius  # Park at type-specific radius

                self._base_center_y = self._base_center_target(
                    trigger_kind=decision.trigger_kind,
                    progress=1.0,
                    silence_active=False,
                )
                self._reactive_bounce_y = self._compute_reactive_bounce_y(
                    event=event,
                    dt=dt,
                    wait_state=True,
                )
                total_center_y = float(self._base_center_y + self._reactive_bounce_y)
                orbit_radius = float(min(radius, self._radius_cap_for_center(total_center_y)))

                alpha = float(orbit_radius * np.cos(self._park_angle))
                beta = float(total_center_y + (orbit_radius * np.sin(self._park_angle)))

                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))
                self._last_journey_completion = 1.0
            else:
                started_new_journey = bool(progress <= 1e-9 and self._last_journey_completion > 1e-9)
                if started_new_journey:
                    if decision.trigger_kind in ("beat", "downbeat", "syncopation", "start"):
                        self._hold_start_pose_until_reactive = False
                    prior_completion = float(self._last_journey_completion)
                    if self._exit_spiral_active:
                        self._journey_linked = True
                        self._journey_cold_start = False
                        self._journey_relink_active = True
                        self._journey_relink_start_radius = float(np.clip(self._actual_radius, 0.70, 1.0))
                    else:
                        self._journey_linked = bool(prior_completion < 0.999)
                        self._journey_cold_start = not self._journey_linked
                        self._journey_relink_active = False

                    self._exit_spiral_active = False
                    self._exit_spiral_progress = 0.0

                    # Crossfade disabled for trajectory stability.
                    # Hard switching to the new arc preserves circular motion
                    # and avoids blended-path squiggles at trigger boundaries.
                    self._crossfade_active = False

                    self._settle_active = False  # cancel any active settle
                    self._radius_hold_active = False

                    self._journey_start_total_center_y = float(self._base_center_y + self._reactive_bounce_y)

                    # Latch geometry at journey start so mid-journey trigger
                    # reclassification cannot reshape a running arc.
                    geom = self.config.stroke.orbit_geometry.get(decision.trigger_kind, {
                        "center_y": self._baseline_center_y, "park_radius": 0.70, "max_radius": 1.0
                    })
                    self._journey_center_y = float(geom["center_y"])
                    self._journey_park_radius = float(geom["park_radius"])
                    self._journey_max_radius = float(geom["max_radius"])

                    self._journey_start_alpha = float(np.clip(self.state.alpha, -1.0, 1.0))
                    self._journey_start_beta = float(np.clip(self.state.beta, -1.0, 1.0))
                    if self._orbit_phase_initialized:
                        # Continuous phase — avoids atan2 roundtrip jitter
                        self._journey_start_angle = float(self._orbit_phase)
                        self._journey_start_radius = float(np.clip(self._actual_radius, self._min_radius, 1.0))
                    else:
                        # First journey: infer from externally-set position
                        inherited_angle, inherited_radius = self._infer_orbit_from_position(
                            alpha=self._journey_start_alpha,
                            beta=self._journey_start_beta,
                            center_y=self._journey_start_total_center_y,
                        )
                        self._journey_start_angle = inherited_angle
                        self._journey_start_radius = float(np.clip(inherited_radius, self._min_radius, 1.0))
                        self._orbit_phase = float(inherited_angle % (2.0 * np.pi))
                        self._orbit_phase_initialized = True

                    self._journey_total_rotation = self._compute_landing_rotation(
                        start_angle=self._journey_start_angle,
                        interval_beats=decision.interval_beats,
                    )
                    self._journey_initial_speed_slope = self._compute_initial_speed_slope(
                        event=event,
                        interval_beats=decision.interval_beats,
                    )
                    # Compute nominal angular speed for smooth continuation glide
                    _bpm_nom = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
                    if _bpm_nom <= 0:
                        _bpm_nom = float(getattr(event, "bpm", 0.0) or 0.0)
                    _bpm_nom = float(np.clip(_bpm_nom if _bpm_nom > 0 else 120.0, 40.0, 240.0))
                    _journey_dur = float(decision.interval_beats) / (_bpm_nom / 60.0)
                    self._journey_nominal_angular_speed = abs(self._journey_total_rotation) / max(_journey_dur, 1e-3)

                    if self._startup_beats_seen < self._startup_ramp_beats:
                        startup_ratio = float(np.clip(
                            self._startup_beats_seen / max(self._startup_ramp_beats, 1e-6),
                            0.0,
                            1.0,
                        ))
                        self._journey_startup_momentum = float(
                            self._startup_momentum_min
                            + ((1.0 - self._startup_momentum_min) * startup_ratio)
                        )
                        self._startup_beats_seen += 1.0
                    else:
                        self._journey_startup_momentum = 1.0

                # Use latched geometry while a journey/settle is in-flight.
                # Only refresh from live trigger kind when fully parked.
                if (progress < 1.0) or (self._last_journey_completion < 1.0) or self._settle_active:
                    type_center_y = float(self._journey_center_y)
                    type_park_radius = float(self._journey_park_radius)
                    type_max_radius = float(self._journey_max_radius)
                else:
                    geom = self.config.stroke.orbit_geometry.get(decision.trigger_kind, {
                        "center_y": self._baseline_center_y, "park_radius": 0.70, "max_radius": 1.0
                    })
                    type_center_y = float(geom["center_y"])
                    type_park_radius = float(geom["park_radius"])
                    type_max_radius = float(geom["max_radius"])

# Latch learning modifiers at journey start so mid-arc
                    # predictions never cause radius/speed discontinuities.
                    if started_new_journey:
                        learning = decision.learning
                        if learning.active:
                            self._journey_learning_mult = float(np.clip(learning.radius_mult, 0.3, 2.5))
                            # Syncopation-specific size/speed scaling
                            if decision.trigger_kind == "syncopation":
                                sync_size = float(np.clip(learning.sync_size_mult, 0.5, 2.0))
                                self._journey_max_radius = float(np.clip(
                                    self._journey_max_radius * sync_size,
                                    self._journey_park_radius,
                                    1.0,
                                ))
                                sync_speed = float(np.clip(learning.sync_speed_mult, 0.3, 3.0))
                                self._journey_total_rotation = float(
                                    self._journey_total_rotation * sync_speed
                                )
                        else:
                            self._journey_learning_mult = 1.0

                learning_mult = self._journey_learning_mult

                # Map global radius_bloom (0.70→1.0) to type-specific range (park→max)
                normalized_bloom = float(np.clip((decision.radius_bloom - 0.70) / 0.30, 0.0, 1.0))
                type_bloom = float(type_park_radius + normalized_bloom * (type_max_radius - type_park_radius))
                bloom_target_radius = float(type_park_radius + ((type_bloom - type_park_radius) * learning_mult))
                bloom_target_radius = float(np.clip(bloom_target_radius, type_park_radius, type_max_radius))

                if started_new_journey:
                    self._journey_fixed_radius = bloom_target_radius
                    self._actual_radius = self._journey_start_radius

                if progress >= 1.0 and not started_new_journey:
                    continuation_expected = bool(
                        self._journey_linked
                        or self._is_upcoming_beat_expected(now=now, decision=decision)
                    )

                    if self._last_journey_completion >= 1.0:
                        self._journey_fixed_radius = bloom_target_radius

                    if continuation_expected and decision.trigger_kind != "creep":
                        self._exit_spiral_active = False
                        self._exit_spiral_progress = 0.0
                        self._settle_active = False
                        self._radius_hold_active = True
                        self._radius_hold_start_time = now
                        self._radius_hold_value = float(type_max_radius)
                        # Use nominal angular speed to prevent stalling during wait
                        cont_speed = max(abs(self._angular_velocity), self._journey_nominal_angular_speed * 0.85)
                        angle = float((self._orbit_phase + (cont_speed * dt)) % (2.0 * np.pi))
                    else:
                        self._settle_active = False
                        self._radius_hold_active = False
                        if not self._exit_spiral_active:
                            self._exit_spiral_active = True
                            self._exit_spiral_progress = 0.0
                            self._exit_spiral_start_angle = float(self._orbit_phase)
                            self._exit_spiral_duration_s = float(np.clip(
                                (2.0 * np.pi) / max(abs(self._angular_velocity), 0.35),
                                0.35,
                                1.2,
                            ))

                        self._exit_spiral_progress = float(np.clip(
                            self._exit_spiral_progress + (dt / max(self._exit_spiral_duration_s, 1e-3)),
                            0.0,
                            1.0,
                        ))
                        exit_t = self._exit_spiral_progress
                        angle = float(self._exit_spiral_start_angle + (2.0 * np.pi * exit_t))

                        if exit_t >= 1.0:
                            self._exit_spiral_active = False
                            self._journey_linked = False
                            self._journey_cold_start = True
                            self._radius_hold_active = True
                            self._radius_hold_start_time = now
                            self._radius_hold_value = 0.70
                else:
                    effective_progress = progress
                    if self._lazy_glide_active and progress > 0.70:
                        tail_t = float(np.clip((progress - 0.70) / 0.30, 0.0, 1.0))
                        stretched_tail = float(tail_t * tail_t)
                        effective_progress = float(0.70 + (0.30 * stretched_tail))

                    # Carry velocity through journey boundary when continuation expected
                    _cont_expected = bool(
                        self._journey_linked
                        or self._is_upcoming_beat_expected(now=now, decision=decision)
                    )
                    _end_slope = 1.0 if _cont_expected else 0.0
                    smooth_progress = self._s_curve_with_initial_velocity(
                        progress=effective_progress,
                        initial_slope=self._journey_initial_speed_slope,
                        end_slope=_end_slope,
                        lazy_glide=self._lazy_glide_active,
                    )
                    startup_momentum = float(np.clip(
                        self._journey_startup_momentum,
                        self._startup_momentum_min,
                        1.0,
                    ))
                    startup_progress_scale = float(
                        startup_momentum + ((1.0 - startup_momentum) * np.clip(progress, 0.0, 1.0))
                    )
                    smooth_progress_scaled = float(np.clip(
                        smooth_progress * startup_progress_scale,
                        0.0,
                        1.0,
                    ))
                    raw_angle = float(
                        self._journey_start_angle + (self._journey_total_rotation * smooth_progress_scaled)
                    )
                    angle = raw_angle

                    if self._lazy_glide_active:
                        target_angle = float(self._journey_start_angle + self._journey_total_rotation)
                        remaining = float(max(0.0, target_angle - raw_angle))
                        anchor_window = float(np.deg2rad(5.0))
                        if 0.0 < remaining < anchor_window:
                            ratio = float(np.clip(remaining / anchor_window, 0.0, 1.0))
                            slowed_remaining = float(anchor_window * (ratio ** 0.35))
                            angle = float(target_angle - slowed_remaining)

                self._orbit_phase = float(angle % (2.0 * np.pi))

                phase_delta = self._wrapped_phase_delta(self._orbit_phase, self._last_phase_for_velocity)
                self._angular_velocity = float(phase_delta / max(dt, 1e-4))
                self._last_phase_for_velocity = self._orbit_phase

                # Radius path is mathematically locked to journey angle/progress.
                # - Cold start: smoothstep from park -> max during first pass
                # - Linked beat: bypass park and lock to max immediately
                # - Continuation expected: allow controlled bloom up to 1.0
                if self._radius_hold_active:
                    radius = float(np.clip(self._radius_hold_value, type_park_radius, type_max_radius))
                elif self._exit_spiral_active:
                    exit_t = float(np.clip(self._exit_spiral_progress, 0.0, 1.0))
                    radius = float(0.90 + ((0.70 - 0.90) * self._s_curve(exit_t)))
                elif decision.trigger_kind == "start":
                    p = float(np.clip(smooth_progress_scaled, 0.0, 1.0))
                    radius = float(
                        self._journey_start_radius
                        + ((self._journey_fixed_radius - self._journey_start_radius) * (p ** 3))
                    )
                else:
                    if self._journey_cold_start:
                        first_pass_progress = float(np.clip(
                            (self._journey_total_rotation * smooth_progress_scaled) / (2.0 * np.pi),
                            0.0,
                            1.0,
                        ))
                        unhook_window = 1.0
                        unhook_t = float(np.clip(first_pass_progress / unhook_window, 0.0, 1.0))
                        radius_blend = self._s_curve(unhook_t)
                        radius = float(
                            type_park_radius
                            + ((type_max_radius - type_park_radius) * radius_blend)
                        )
                    else:
                        radius = float(type_max_radius)

                    if self._journey_relink_active:
                        first_pass_progress = float(np.clip(
                            (self._journey_total_rotation * smooth_progress_scaled) / (2.0 * np.pi),
                            0.0,
                            1.0,
                        ))
                        relink_window = 0.30
                        relink_t = float(np.clip(first_pass_progress / relink_window, 0.0, 1.0))
                        relink_blend = self._s_curve(relink_t)
                        radius = float(
                            self._journey_relink_start_radius
                            + ((type_max_radius - self._journey_relink_start_radius) * relink_blend)
                        )
                        if relink_t >= 1.0:
                            self._journey_relink_active = False

                    continuation_expected = bool(
                        self._journey_linked
                        or self._is_upcoming_beat_expected(now=now, decision=decision)
                    )
                    if continuation_expected:
                        expanded_radius = float(np.clip(decision.radius_bloom, type_max_radius, 1.0))
                        radius = float(max(radius, expanded_radius))

                min_radius_bound = self._min_radius if decision.trigger_kind == "start" else 0.70
                self._actual_radius = float(np.clip(radius, min_radius_bound, 1.0))
                radius = self._actual_radius

                base_target_center = self._base_center_target(
                    trigger_kind=decision.trigger_kind,
                    progress=progress,
                    silence_active=False,
                )
                if progress < 1.0:
                    center_blend = self._s_curve(progress)
                    self._base_center_y = float(
                        ((1.0 - center_blend) * self._journey_start_total_center_y)
                        + (center_blend * base_target_center)
                    )
                else:
                    self._base_center_y = float(base_target_center)

                wait_state = bool(decision.trigger_kind == "creep" and progress >= 1.0)
                self._reactive_bounce_y = self._compute_reactive_bounce_y(
                    event=event,
                    dt=dt,
                    wait_state=wait_state,
                )
                total_center_y = float(self._base_center_y + self._reactive_bounce_y)
                orbit_radius = float(min(radius, self._radius_cap_for_center(total_center_y)))

                alpha = float(orbit_radius * np.cos(angle))
                beta = float(total_center_y + (orbit_radius * np.sin(angle)))

                # Apply post-silence ramp to volume
                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))

                self._last_journey_completion = progress
                if decision.trigger_kind == "start" and progress >= 1.0:
                    self._hold_start_pose_until_reactive = True

        self.state.alpha = alpha
        self.state.beta = beta

        return TCodeCommand(alpha=alpha, beta=beta, duration_ms=25, volume=volume)

    def _compute_bass_jitter_offsets(self, event: BeatEvent, dt: float) -> tuple[float, float]:
        if not bool(getattr(self.config.jitter, "enabled", True)):
            return 0.0, 0.0

        base_amp = float(getattr(self.config.jitter, "amplitude", 0.0) or 0.0)
        base_speed = float(getattr(self.config.jitter, "intensity", 0.0) or 0.0)
        if base_amp <= 0.0 or base_speed <= 0.0:
            return 0.0, 0.0

        freq = float(getattr(event, "frequency", 0.0) or 0.0)
        lo_hz, hi_hz = 30.0, 220.0
        norm = float(np.clip((freq - lo_hz) / max(hi_hz - lo_hz, 1e-6), 0.0, 1.0))

        # Smooth frequency-derived jitter control to avoid frame-to-frame flicker
        self._bass_jitter_freq_ema += 0.2 * (norm - self._bass_jitter_freq_ema)
        norm_smooth = float(np.clip(self._bass_jitter_freq_ema, 0.0, 1.0))

        # Low bass -> slower/smaller, high bass -> faster/larger
        bass_mult = 0.5 + (1.5 * norm_smooth)  # 0.5..2.0

        speed_inf = float(getattr(self.config.stroke, "bass_jitter_speed_influence_percent", 100.0) or 100.0)
        size_inf = float(getattr(self.config.stroke, "bass_jitter_size_influence_percent", 100.0) or 100.0)
        speed_blend = float(np.clip(speed_inf / 100.0, 0.0, 2.0))
        size_blend = float(np.clip(size_inf / 100.0, 0.0, 2.0))

        combo_texture = float(np.clip(float(getattr(self.config.stroke, "combo_texture", 1.0) or 1.0), -2.0, 3.0))
        if combo_texture >= 1.0:
            texture_factor = 1.0 + ((combo_texture - 1.0) / 2.0)
        else:
            texture_factor = 1.0 - ((1.0 - combo_texture) / 3.0) * 0.5

        # Texture > 1 amplifies bass-driven jitter variance, < 1 damps it.
        speed_effect = (bass_mult - 1.0) * speed_blend * texture_factor
        size_effect = ((1.0 / bass_mult) - 1.0) * size_blend * texture_factor

        speed_mult = float(np.clip(1.0 + speed_effect, 0.0, 5.0))
        # Inverted size: high bass → smaller circles, low bass → bigger circles
        size_mult = float(np.clip(1.0 + size_effect, 0.0, 5.0))

        jitter_speed = max(0.0, base_speed * speed_mult)
        jitter_amp = max(0.0, base_amp * size_mult)

        self._bass_jitter_phase += float(jitter_speed * max(dt, 1e-4))
        phase = float(self._bass_jitter_phase)

        # Slight ellipse keeps movement feeling organic, not perfectly circular
        jitter_alpha = float(jitter_amp * np.sin(phase))
        jitter_beta = float((jitter_amp * 0.70) * np.cos(phase))
        return jitter_alpha, jitter_beta

    def _is_upcoming_beat_expected(self, now: float, decision: BeatDecision) -> bool:
        if decision.trigger_kind == "creep":
            return False
        if bool(getattr(decision, "lazy_glide_active", False)):
            return False
        if self.audio_engine is None:
            return False

        predicted_next = float(getattr(self.audio_engine, "predicted_next_beat_mono", 0.0) or 0.0)
        if predicted_next <= now:
            return False

        met_bpm = float(getattr(self.audio_engine, "_metronome_bpm", 0.0) or 0.0)
        bpm = met_bpm if met_bpm > 0.0 else 120.0
        beat_period_s = 60.0 / max(1e-6, bpm)
        return float(predicted_next - now) <= float(1.25 * beat_period_s)

    def _base_center_target(self, trigger_kind: str, progress: float, silence_active: bool) -> float:
        if silence_active:
            return self._baseline_center_y
        if trigger_kind == "start":
            p = float(np.clip(progress, 0.0, 1.0))
            return float(self._baseline_center_y * (1.0 - p))
        if trigger_kind in ("beat", "downbeat", "syncopation"):
            return 0.0
        return self._baseline_center_y

    def _compute_reactive_bounce_y(self, event: BeatEvent, dt: float, wait_state: bool) -> float:
        if not wait_state:
            return 0.0

        jitter_alpha, jitter_beta = self._compute_bass_jitter_offsets(event=event, dt=dt)
        _ = jitter_alpha
        treble_bump = float(self._intelligence.compute_treble_lift(0.0))
        return float(np.clip(jitter_beta + treble_bump, -0.30, 0.30))

    @staticmethod
    def _s_curve(progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        return float(p * p * (3.0 - (2.0 * p)))

    @staticmethod
    def _sine_ease_with_velocity(
        progress: float,
        initial_slope: float,
        lazy_glide: bool = False,
    ) -> float:
        """Sine-based easing with velocity continuity for buttery-smooth motion.

        Core curve: 0.5 * (1 - cos(pi * p)), providing:
        - Smooth acceleration from rest at departure
        - Peak angular speed at mid-journey
        - Gradual deceleration to near-zero velocity at arrival
        - No dead time: always moving until p=1.0

        +Y axis crossing modulation: a very minute deceleration as
        the dot crosses the +Y axis (start/end), with a minor
        re-acceleration bump on departure.  Helps keep timing tight.

        Velocity carry-over: initial_slope blends inherited angular
        momentum into the first ~25% of the journey, fading to zero
        so landing is always gentle.
        """
        p = float(np.clip(progress, 0.0, 1.0))
        m0 = float(np.clip(initial_slope, 0.0, 2.5))

        # ── Base: cosine interpolation (sine ease-in-out) ──
        eased = 0.5 * (1.0 - float(np.cos(np.pi * p)))

        # ── Velocity carry-over from prior journey / settle ──
        # Blend inherited momentum into early phase; fades by p≈0.25
        # so mid-journey and landing remain pure sine.
        if m0 > 1e-3:
            carry_window = 0.25
            carry_fade = float(np.clip(1.0 - (p / carry_window), 0.0, 1.0))
            carry_fade = carry_fade * carry_fade          # quadratic fade-out
            carry = m0 * p * carry_fade * 0.25
            eased += carry

        # ── +Y axis crossing modulation ──
        # Departure (p≈0): tiny extra slowdown, then minor re-acceleration.
        # Arrival  (p≈1): gentle extra deceleration for velvety landing.
        if p < 0.12:
            t = p / 0.12
            eased += 0.012 * float(np.sin(np.pi * t))     # post-departure bump
        elif p > 0.88:
            t = (p - 0.88) / 0.12
            eased -= 0.012 * float(np.sin(np.pi * t))     # pre-arrival cushion

        return float(np.clip(eased, 0.0, 1.02))

    @staticmethod
    def _s_curve_with_initial_velocity(
        progress: float,
        initial_slope: float,
        end_slope: float = 0.0,
        lazy_glide: bool = False,
    ) -> float:
        """Legacy cubic Hermite easing (kept for test compatibility)."""
        p = float(np.clip(progress, 0.0, 1.0))
        p_eval = p
        carrying = end_slope > 1e-3  # carrying velocity through to next journey

        if (not lazy_glide) and (not carrying) and (0.90 < p < 1.0):
            # Arrival-only micro "time stretch" before +Y crossing.
            # Skip when carrying velocity through to next journey.
            t = (p - 0.90) / 0.10
            p_eval = float(np.clip(p - (0.020 * np.sin(np.pi * t)), 0.0, 1.0))

        m0 = float(np.clip(initial_slope, 0.0, 2.5))
        m1 = float(np.clip(end_slope, 0.0, 2.5))
        h10 = (p_eval * p_eval * p_eval) - (2.0 * p_eval * p_eval) + p_eval
        h01 = (-2.0 * p_eval * p_eval * p_eval) + (3.0 * p_eval * p_eval)
        h11 = (p_eval * p_eval * p_eval) - (p_eval * p_eval)
        eased = (h10 * m0) + h01 + (h11 * m1)
        if (not lazy_glide) and (not carrying) and p > 0.92:
            # Landing overshoot - skip when carrying velocity through
            t = (p - 0.92) / 0.08
            overshoot = 0.025 * float(np.sin(t * np.pi))
            eased += overshoot

        return float(np.clip(eased, 0.0, 1.04))

    def _compute_initial_speed_slope(self, event: BeatEvent, interval_beats: int) -> float:
        """Map current angular velocity to a normalized easing start slope.

        Short journeys (1-2 beats) use a lower slope cap to prevent
        whip-like starts when inheriting velocity from a fast arc.
        """
        bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
        if bpm <= 0.0:
            bpm = float(getattr(event, "bpm", 0.0) or 0.0)
        bpm = float(np.clip(bpm if bpm > 0.0 else 120.0, 40.0, 240.0))

        beats_per_second = bpm / 60.0
        target_duration_s = float(max(1e-3, float(interval_beats) / max(1e-6, beats_per_second)))
        progress_rate = 1.0 / target_duration_s

        denom = max(1e-6, self._journey_total_rotation * progress_rate)
        slope = self._angular_velocity / denom

        # Cap slope based on journey length: short arcs must not whip-start
        max_slope = 1.2 if int(interval_beats) <= 2 else 2.0
        return float(np.clip(slope, 0.0, max_slope))

    @staticmethod
    def _wrapped_phase_delta(current: float, previous: float) -> float:
        """Return wrapped phase delta in [-pi, pi] for stable velocity estimation."""
        return float((current - previous + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _infer_orbit_from_position(alpha: float, beta: float, center_y: float) -> tuple[float, float]:
        """Infer phase/radius using current orientation: alpha=r*cos(theta), beta=center+r*sin(theta)."""
        dy = float(beta - center_y)
        radius = float(np.hypot(alpha, dy))
        angle = float(np.arctan2(dy, alpha))
        return angle, radius

    @staticmethod
    def _radius_cap_for_center(center_y: float) -> float:
        """Maximum radius that keeps a perfect circle fully inside normalized Y bounds [-1, 1]."""
        return float(max(0.0, min(1.0 - center_y, 1.0 + center_y)))

    def _compute_landing_rotation(self, start_angle: float, interval_beats: int) -> float:
        # Rotation policy: 1 journey = 1 full lap for all trigger types.
        turns = 1
        phase_to_park = float((self._park_angle - start_angle) % (2.0 * np.pi))
        rotation = float(phase_to_park + (2.0 * np.pi * max(0, turns - 1)))
        # If start is already at park and turns==1, preserve one full visible rotation.
        if rotation <= 1e-6:
            rotation = float(2.0 * np.pi)
        return rotation
