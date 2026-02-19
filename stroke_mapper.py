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
        self._max_radius = 1.0
        self._journey_fixed_radius = self._park_radius
        self._park_angle = 0.0
        self._journey_start_angle = self._park_angle
        self._journey_total_rotation = float(2.0 * np.pi)
        self._last_journey_completion = 1.0
        self._actual_radius = self._park_radius
        self._radius_hold_active = False
        self._radius_hold_value = self._park_radius
        self._radius_hold_start_time = 0.0
        self._angular_velocity = 0.0
        self._last_phase_for_velocity = self._orbit_phase
        self._journey_initial_speed_slope = 0.0
        self._lazy_glide_active = False

        # Elastic landing / settle state
        self._settle_active = False
        self._settle_start_time = 0.0
        self._settle_exit_velocity = 0.0
        self._settle_damping = 4.5     # higher = faster decay
        self._settle_frequency = 10.0  # oscillation freq (rad/s) – ~1 visible cycle
        self._settle_max_amplitude = 0.12  # max angular displacement (radians)

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
            # Apply silence fade (gradual, not binary) with radius exhale.
            # Do NOT hard-snap back to park on silence entry.
            # Use geometry from last active trigger kind
            geom = self.config.stroke.orbit_geometry.get(self._last_trigger_kind, {
                "center_y": 0.0, "park_radius": 0.70, "max_radius": 1.0
            })
            type_park_radius = float(geom["park_radius"])
            type_max_radius = float(geom["max_radius"])
            type_center_y = float(geom["center_y"])

            fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
            self._orbit_phase = float(self._park_angle)
            self._last_phase_for_velocity = self._orbit_phase
            self._angular_velocity = 0.0

            held_radius = float(np.clip(max(self._actual_radius, self._radius_hold_value), type_park_radius, type_max_radius))
            target_radius = float(type_park_radius + ((held_radius - type_park_radius) * fade))
            self._actual_radius += 0.08 * (target_radius - self._actual_radius)
            radius = float(np.clip(self._actual_radius, type_park_radius, type_max_radius))

            alpha = 0.0
            beta = float(type_center_y + (radius * np.cos(0.0)))
            volume = float(np.clip(self.get_volume() * fade, 0.0, 1.0))
            self._last_journey_completion = 1.0
        else:
            progress = float(np.clip(decision.journey_completion, 0.0, 1.0))
            creep_motion_disabled = (
                decision.trigger_kind == "creep"
                and not bool(getattr(self.config.creep, "enabled", True))
            )

            if creep_motion_disabled:
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

                alpha = 0.0
                beta = float(type_center_y + (radius * np.cos(0.0)))
                jitter_alpha, jitter_beta = self._compute_bass_jitter_offsets(event=event, dt=dt)
                alpha += jitter_alpha
                beta += jitter_beta

                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))
                self._last_journey_completion = 1.0
            else:
                started_new_journey = bool(progress <= 1e-9 and self._last_journey_completion > 1e-9)
                if started_new_journey:
                    self._settle_active = False  # cancel any active settle
                    self._radius_hold_active = False
                    self._journey_start_angle = float(self._orbit_phase)
                    self._journey_total_rotation = self._compute_landing_rotation(
                        start_angle=self._journey_start_angle,
                        interval_beats=decision.interval_beats,
                    )
                    self._journey_initial_speed_slope = self._compute_initial_speed_slope(
                        event=event,
                        interval_beats=decision.interval_beats,
                    )

                # Get beat-type-specific orbit geometry
                geom = self.config.stroke.orbit_geometry.get(decision.trigger_kind, {
                    "center_y": 0.0, "park_radius": 0.70, "max_radius": 1.0
                })
                type_park_radius = float(geom["park_radius"])
                type_max_radius = float(geom["max_radius"])
                type_center_y = float(geom["center_y"])

                learning_mult = 1.0
                if decision.learning.active:
                    learning_mult = float(np.clip(decision.learning.radius_mult, 0.3, 2.5))

                # Map global radius_bloom (0.70→1.0) to type-specific range (park→max)
                normalized_bloom = float(np.clip((decision.radius_bloom - 0.70) / 0.30, 0.0, 1.0))
                type_bloom = float(type_park_radius + normalized_bloom * (type_max_radius - type_park_radius))
                bloom_target_radius = float(type_park_radius + ((type_bloom - type_park_radius) * learning_mult))
                bloom_target_radius = float(np.clip(bloom_target_radius, type_park_radius, type_max_radius))

                if started_new_journey:
                    self._journey_fixed_radius = bloom_target_radius
                    self._actual_radius = self._journey_fixed_radius

                if progress >= 1.0 and not started_new_journey:
                    if self._last_journey_completion >= 1.0:
                        self._journey_fixed_radius = bloom_target_radius
                    if not self._radius_hold_active:
                        self._radius_hold_active = True
                        self._radius_hold_start_time = now
                        # Hold at the same radius used throughout the journey.
                        self._radius_hold_value = float(np.clip(self._journey_fixed_radius, type_park_radius, type_max_radius))
                    # ── Elastic landing: damped harmonic settle ──
                    # Momentum carry-over from exit velocity drives overshoot;
                    # damped oscillation creates one visible wobble then rest.
                    if not self._settle_active:
                        self._settle_active = True
                        self._settle_start_time = now
                        self._settle_exit_velocity = self._angular_velocity
                    t = now - self._settle_start_time
                    raw_amp = abs(self._settle_exit_velocity) * 0.06
                    amp = max(0.02, min(raw_amp, self._settle_max_amplitude))
                    disp = amp * float(np.exp(-self._settle_damping * t) * np.sin(self._settle_frequency * t))
                    if abs(disp) < 1e-5 and t > 0.15:
                        self._settle_active = False
                        disp = 0.0
                    angle = float(self._park_angle + disp)
                else:
                    effective_progress = progress
                    if self._lazy_glide_active and progress > 0.70:
                        tail_t = float(np.clip((progress - 0.70) / 0.30, 0.0, 1.0))
                        stretched_tail = float(tail_t * tail_t)
                        effective_progress = float(0.70 + (0.30 * stretched_tail))

                    smooth_progress = self._s_curve_with_initial_velocity(
                        progress=effective_progress,
                        initial_slope=self._journey_initial_speed_slope,
                        lazy_glide=self._lazy_glide_active,
                    )
                    raw_angle = float(self._journey_start_angle + (self._journey_total_rotation * smooth_progress))
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

                # Continuous "Smoooooth Arc": radius is independent of journey progress.
                # Target radius follows music/learning; actual radius lerps via EMA.
                if self._radius_hold_active:
                    target_radius = float(np.clip(self._radius_hold_value, type_park_radius, type_max_radius))
                    radius_alpha = 0.12 if target_radius > self._actual_radius else 0.04
                else:
                    target_radius = float(np.clip(self._journey_fixed_radius, type_park_radius, type_max_radius))
                    self._actual_radius = target_radius
                    radius_alpha = 0.0
                self._actual_radius += radius_alpha * (target_radius - self._actual_radius)
                radius = float(np.clip(self._actual_radius, type_park_radius, type_max_radius))

                if decision.trigger_kind == "creep":
                    treble_lift = self._intelligence.compute_treble_lift(progress)
                else:
                    treble_lift = 0.0

                # Apply beat-type-specific orbital center
                alpha = float(radius * np.sin(angle))
                beta = float(type_center_y + treble_lift + (radius * np.cos(angle)))

                if decision.trigger_kind == "creep":
                    jitter_alpha, jitter_beta = self._compute_bass_jitter_offsets(event=event, dt=dt)
                    alpha += jitter_alpha
                    beta += jitter_beta

                # Apply post-silence ramp to volume
                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))

                self._last_journey_completion = progress

        alpha = float(np.clip(alpha, -1.0, 1.0))
        beta = float(np.clip(beta, -1.0, 1.0))
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

    @staticmethod
    def _s_curve(progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        return float(p * p * (3.0 - (2.0 * p)))

    @staticmethod
    def _s_curve_with_initial_velocity(
        progress: float,
        initial_slope: float,
        end_slope: float = 0.0,
        lazy_glide: bool = False,
    ) -> float:
        """Cubic Hermite easing with configurable initial slope and elastic back-ease landing."""
        p = float(np.clip(progress, 0.0, 1.0))
        m0 = float(np.clip(initial_slope, 0.0, 2.5))
        m1 = float(np.clip(end_slope, 0.0, 2.5))

        h10 = (p * p * p) - (2.0 * p * p) + p
        h01 = (-2.0 * p * p * p) + (3.0 * p * p)
        h11 = (p * p * p) - (p * p)
        eased = (h10 * m0) + h01 + (h11 * m1)

        # Back-ease overshoot: sine bump in the final 8% of the journey.
        # Dot swings slightly past target before the post-journey settle
        # takes over → elastic, biological landing feel.
        if (not lazy_glide) and p > 0.92:
            t = (p - 0.92) / 0.08  # 0…1 within last 8%
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

    def _compute_landing_rotation(self, start_angle: float, interval_beats: int) -> float:
        # Rotation policy: 1 journey = 1 full lap for all trigger types.
        turns = 1
        phase_to_park = float((self._park_angle - start_angle) % (2.0 * np.pi))
        rotation = float(phase_to_park + (2.0 * np.pi * max(0, turns - 1)))
        # If start is already at park and turns==1, preserve one full visible rotation.
        if rotation <= 1e-6:
            rotation = float(2.0 * np.pi)
        return rotation
