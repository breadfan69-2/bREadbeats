"""
bREadbeats - Stroke Mapper (Decision-Only Adapter)

Thin runtime adapter that delegates signal intelligence to beat_intelligence.
Legacy drawing/trajectory generation has been removed.
"""

from __future__ import annotations

import json
import time
from collections import deque
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
        self._park_y = 0.20
        self._baseline_center_y = 0.20
        self._min_radius = 0.05
        self._park_idle_radius = 0.05  # tiny orbit when fully parked
        self._treble_lift_enabled = False
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
        self._journey_energy_fullness = 0.0  # latched at journey start for max_radius expansion
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
        self._radius_hold_visible_cycle_done = False
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
        self._startup_momentum_min = 0.15
        self._startup_ramp_beats = 6.0
        self._startup_beats_seen = 0.0
        self._journey_startup_momentum = 1.0
        self._post_silence_radius_ramp = 1.0   # 0→1 over first beats after silence
        self._post_silence_radius_floor = 0.12 # start radius fraction after silence
        self._hold_start_pose_until_reactive = False
        self._reactive_hold_swirl_phase = 0.0
        self._reactive_hold_swirl_center_y = float(getattr(self.config.stroke, "reactive_hold_center_y", -0.70) or -0.70)
        self._reactive_hold_swirl_radius = float(getattr(self.config.stroke, "reactive_hold_radius", 0.18) or 0.18)
        self._reactive_hold_swirl_loops_per_beat = float(getattr(self.config.stroke, "reactive_hold_loops_per_beat", 0.20) or 0.20)
        self._reactive_hold_swirl_jitter_mix = float(getattr(self.config.stroke, "reactive_hold_jitter_mix", 0.35) or 0.35)
        self._idle_radius = self._min_radius
        self._silence_decay_per_beat = 0.40
        self._idle_loops_per_beat = 0.125

        # Swirl-to-park state: tracks the spiral transition into idle
        self._swirl_progress = 0.0       # 0→1 S-curve interpolant
        self._swirl_duration_s = 1.8     # total time to spirally arrive at park
        self._swirl_start_center_y = self._baseline_center_y
        self._swirl_start_radius = self._park_radius
        self._swirl_entering = False     # True on first silence frame after motion

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

        # Gate-idle state: smooth deceleration when beat gets gated with creep disabled
        self._gate_idle_active = False
        self._gate_idle_progress = 0.0    # 0→1 S-curve deceleration
        self._gate_idle_duration_s = 1.2  # seconds to fully decelerate
        self._gate_idle_start_center_y = self._baseline_center_y
        self._gate_idle_start_radius = self._park_radius
        self._gate_idle_start_angular_vel = 0.0

        # Wait-swirl state: smooth spiral into slingshot loop while waiting for next beat
        self._wait_swirl_active = False
        self._wait_swirl_progress = 0.0    # 0→1 S-curve interpolant
        self._wait_swirl_duration_s = 0.60 # smooth spiral into visible loop
        self._wait_swirl_target_radius = 0.30  # visible slingshot loop radius
        self._wait_swirl_hook_depth = 0.05     # center_y dip for hook maneuver
        self._wait_swirl_start_center_y = self._baseline_center_y
        self._wait_swirl_start_radius = self._park_radius
        self._wait_swirl_start_angular_vel = 0.0
        self._wait_swirl_alpha = 0.0   # stored final position for post-fix
        self._wait_swirl_beta = 0.5
        self._wait_swirl_volume = 0.0
        self._wait_swirl_base_center_y = self._baseline_center_y
        self._wait_swirl_reactive_bounce_y = 0.0

        self._last_gate_fail = ""  # diagnostic: which gate blocked last beat-family event

        # ── Fixed anchor state (§1) ──
        self._anchor_sign: int = 1               # +1 = +Y anchor, -1 = -Y anchor
        self._anchor_angle: float = float(np.pi / 2.0)  # angle of anchor on orbit
        self._anchor_swing_deg: float = 10.0     # ±10° swing around y-axis
        self._anchor_phrase_locked: bool = False  # True once chosen for current phrase

        # ── Expression pause spiral (§2/§3) ──
        self._expr_pause_spiral_active: bool = False
        self._expr_pause_spiral_progress: float = 0.0
        self._expr_pause_spiral_duration_beats: float = 2.0
        self._expr_pause_spiral_start_radius: float = 0.7
        self._expr_pause_spiral_target_radius: float = 0.5
        self._expr_pause_spiral_start_angle: float = 0.0
        self._expr_pause_return_active: bool = False
        self._expr_pause_return_progress: float = 0.0
        self._expr_pause_return_start_angle: float = 0.0
        self._expr_pause_return_start_radius: float = 0.5

        # ── Entry journey gating (§8) ──
        self._post_silence_entry_done: bool = False  # True after first entry journey completes
        self._post_wait_reentry_active: bool = False
        self._post_wait_reentry_progress: float = 0.0
        self._post_wait_reentry_beats_remaining: float = 4.0

        # ── Expression layer state ──
        self._orbit_direction: int = 1           # 1=default, -1=reversed
        self._last_direction_change_time: float = 0.0
        self._center_x_offset: float = 0.0
        self._center_y_offset: float = 0.0
        self._center_wander_phase: float = 0.0
        self._tension_pause_active: bool = False
        self._tension_pause_end_time: float = 0.0
        self._tension_pause_last_time: float = 0.0
        self._tension_pause_hold_alpha: float = 0.0
        self._tension_pause_hold_beta: float = 0.0
        self._tension_pause_fade_end: float = 0.0    # crossfade-back end time
        self._energy_history: deque = deque(maxlen=120)  # ~2s at 60fps
        self._session_energy_ema: float = 0.5

        # ── Intensity timer ramp (session-level escalation) ──
        self._intensity_ramp_start_time: float = 0.0
        self._intensity_ramp_started: bool = False
        self._intensity_ramp_mult: float = 1.0
        self._intensity_ramp_floor: float = 0.25
        self._intensity_ramp_affect_size: bool = True
        self._intensity_ramp_affect_speed: bool = True

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
        self._park_y = 0.20
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
        raw_dt = (now - self.state.last_time) if self.state.last_time > 0 else (1.0 / 60.0)
        dt = float(np.clip(raw_dt, 1e-4, 0.05))
        hitch_soft_reset = bool(raw_dt > 0.25)
        self.state.last_time = now

        self._intelligence.set_audio_engine(self.audio_engine)
        decision = self._intelligence.build_decision(event=event, dt=dt)

        self._active_interval_beats = decision.interval_beats
        self._last_trigger_kind = decision.trigger_kind
        self._lazy_glide_active = bool(getattr(decision, "lazy_glide_active", False))
        self._last_gate_fail = str(getattr(decision, "gate_fail", "") or "")

        if hitch_soft_reset:
            self._angular_velocity = 0.0
            self._last_phase_for_velocity = self._orbit_phase

            ramp = float(np.clip(
                decision.silence_fade if decision.silence_active else decision.post_silence_ramp,
                0.0,
                1.0,
            ))
            volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))
            self.state.alpha = float(np.clip(self.state.alpha, -1.0, 1.0))
            self.state.beta = float(np.clip(self.state.beta, -1.0, 1.0))
            return TCodeCommand(alpha=self.state.alpha, beta=self.state.beta, duration_ms=25, volume=volume)

        # ── Expression layer: per-frame updates ──
        self._update_expression_layer(decision=decision, dt=dt, now=now)

        # ── Intensity timer ramp: session-level escalation ──
        ramp_target = str(getattr(self.config.stroke, 'intensity_ramp_target', 'both') or 'both').strip().lower()
        if ramp_target not in ('size', 'speed', 'both'):
            ramp_target = 'both'
        self._intensity_ramp_affect_size = ramp_target in ('size', 'both')
        self._intensity_ramp_affect_speed = ramp_target in ('speed', 'both')

        ramp_hours = float(getattr(self.config.stroke, 'intensity_ramp_hours', 0.0) or 0.0)
        if ramp_hours > 0.0:
            if not decision.silence_active and not self._intensity_ramp_started:
                self._intensity_ramp_started = True
                self._intensity_ramp_start_time = now
            if self._intensity_ramp_started:
                elapsed_s = now - self._intensity_ramp_start_time
                ramp_s = ramp_hours * 3600.0
                raw_t = float(np.clip(elapsed_s / max(ramp_s, 1.0), 0.0, 1.0))
                eased_t = self._quintic_ease(raw_t)
                self._intensity_ramp_mult = float(
                    self._intensity_ramp_floor + ((1.0 - self._intensity_ramp_floor) * eased_t)
                )
            else:
                self._intensity_ramp_mult = self._intensity_ramp_floor
        else:
            self._intensity_ramp_mult = 1.0

        if decision.silence_active:
            self._hold_start_pose_until_reactive = False
            # Reset cold-start momentum ramp so dot takes off slowly after silence
            self._startup_beats_seen = 0.0
            self._journey_startup_momentum = self._startup_momentum_min
            self._post_silence_radius_ramp = 0.0
            # Reset gate-idle on silence entry (swirl-to-park takes over)
            self._gate_idle_active = False
            self._gate_idle_progress = 0.0
            # Reset entry gating on silence
            self._post_silence_entry_done = False
            self._post_wait_reentry_active = False
            self._anchor_phrase_locked = False

            # ── Swirl-to-park: spiral into 0.6y with S-curve interpolation ──
            # First silence frame: latch current center/radius as start.
            if not self._swirl_entering:
                self._swirl_entering = True
                self._swirl_progress = 0.0
                self._swirl_start_center_y = float(self._base_center_y)
                self._swirl_start_radius = float(max(self._actual_radius, self._park_idle_radius))

            bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
            if bpm <= 0.0:
                bpm = float(getattr(event, "bpm", 0.0) or 0.0)
            bpm = float(np.clip(bpm if bpm > 0.0 else 120.0, 40.0, 240.0))

            # Advance swirl progress toward 1.0
            self._swirl_progress = float(np.clip(
                self._swirl_progress + (dt / max(self._swirl_duration_s, 1e-3)),
                0.0,
                1.0,
            ))
            # S-curve (3t²-2t³) ensures zero velocity at arrival
            swirl_t = self._s_curve(self._swirl_progress)

            # Interpolate center_y: current → 0.6 (park)
            swirl_center_y = float(
                self._swirl_start_center_y
                + ((self._baseline_center_y - self._swirl_start_center_y) * swirl_t)
            )
            # Interpolate radius: current → park idle radius
            swirl_radius = float(
                self._swirl_start_radius
                + ((self._park_idle_radius - self._swirl_start_radius) * swirl_t)
            )
            radius = float(np.clip(swirl_radius, self._park_idle_radius, 1.0))
            self._actual_radius = radius

            fade = float(np.clip(decision.silence_fade, 0.0, 1.0))

            # Theta keeps spinning — inherit momentum, decelerate smoothly
            # Start from current angular velocity, blend to idle speed via S-curve
            idle_angular_speed = float((2.0 * np.pi) * (bpm / 60.0) * self._idle_loops_per_beat)
            current_speed = max(abs(self._angular_velocity), idle_angular_speed)
            blended_speed = float(
                current_speed + ((idle_angular_speed - current_speed) * swirl_t)
            )
            self._orbit_phase = float((self._orbit_phase + (blended_speed * dt)) % (2.0 * np.pi))
            self._angular_velocity = float(blended_speed)
            self._last_phase_for_velocity = self._orbit_phase

            self._base_center_y = swirl_center_y

            # Bass-reactive jitter stays active at park so the tiny orbit vibrates
            jitter_alpha, jitter_beta = self._compute_bass_jitter_offsets(
                event=event, dt=dt,
            )
            treble_bump = float(self._intelligence.compute_treble_lift(0.0)) if self._treble_lift_enabled else 0.0
            self._reactive_bounce_y = float(np.clip(jitter_beta + treble_bump, -0.30, 0.30))

            total_center_y = float(self._base_center_y + self._reactive_bounce_y)
            orbit_radius = float(min(radius, self._radius_cap_for_center(total_center_y)))

            angle = float(self._orbit_phase)
            alpha = float(orbit_radius * np.cos(angle)) + float(jitter_alpha * 0.3)
            beta = float(total_center_y + (orbit_radius * np.sin(angle)))
            volume = float(np.clip(self.get_volume() * fade, 0.0, 1.0))
            self._last_journey_completion = 1.0
        else:
            progress = float(np.clip(decision.journey_completion, 0.0, 1.0))
            creep_motion_disabled = not bool(getattr(self.config.creep, "enabled", True))
            # Reset swirl state when music resumes (gate-idle persists until new journey)
            self._swirl_entering = False
            self._swirl_progress = 0.0

            if self._hold_start_pose_until_reactive and decision.trigger_kind == "creep":
                bpm_hold = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
                if bpm_hold <= 0.0:
                    bpm_hold = float(getattr(event, "bpm", 0.0) or 0.0)
                bpm_hold = float(np.clip(bpm_hold if bpm_hold > 0.0 else 120.0, 40.0, 240.0))

                # Gentle reactive swirl (no hard freeze):
                # loops_per_beat <= 1.0 guarantees never faster than one full
                # rotation per beat.  Default is intentionally slow (0.20).
                loops_per_beat = float(np.clip(self._reactive_hold_swirl_loops_per_beat, 0.02, 1.0))
                angular_speed = float((2.0 * np.pi) * (bpm_hold / 60.0) * loops_per_beat)
                self._reactive_hold_swirl_phase = float(
                    (self._reactive_hold_swirl_phase + (angular_speed * dt)) % (2.0 * np.pi)
                )

                base_center_y = float(np.clip(self._reactive_hold_swirl_center_y, -0.95, 0.95))
                total_center_y = float(base_center_y + self._center_y_offset)
                radius_target = float(np.clip(self._reactive_hold_swirl_radius, 0.05, 0.35))
                orbit_radius = float(min(radius_target, self._radius_cap_for_center(base_center_y)))

                jitter_alpha, jitter_beta = self._compute_bass_jitter_offsets(event=event, dt=dt)
                jitter_mix = float(np.clip(self._reactive_hold_swirl_jitter_mix, 0.0, 1.0))

                alpha = float(orbit_radius * np.cos(self._reactive_hold_swirl_phase)) + float(jitter_alpha * jitter_mix)
                beta = float(total_center_y + (orbit_radius * np.sin(self._reactive_hold_swirl_phase))) + float(jitter_beta * jitter_mix)

                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))
                self._last_journey_completion = 1.0
                self.state.alpha = float(np.clip(alpha, -1.0, 1.0))
                self.state.beta = float(np.clip(beta, -1.0, 1.0))
                return TCodeCommand(
                    alpha=self.state.alpha,
                    beta=self.state.beta,
                    duration_ms=25,
                    volume=volume,
                )

            if creep_motion_disabled and decision.trigger_kind == "creep":
                # Creep disabled: graceful momentum decay instead of instant park.
                # Dot keeps orbiting but smoothly decelerates to park position.
                geom = self.config.stroke.orbit_geometry.get("creep", {
                    "center_y": 0.10, "park_radius": 0.30, "max_radius": 0.60
                })
                type_park_radius = float(geom["park_radius"])
                type_center_y = float(geom["center_y"])

                # First creep frame after motion: latch current state
                if not self._gate_idle_active:
                    self._gate_idle_active = True
                    self._gate_idle_progress = 0.0
                    self._gate_idle_start_center_y = float(self._base_center_y)
                    self._gate_idle_start_radius = float(max(self._actual_radius, type_park_radius))
                    self._gate_idle_start_angular_vel = float(max(abs(self._angular_velocity), 0.5))

                self._settle_active = False
                self._radius_hold_active = False

                # Advance deceleration progress
                self._gate_idle_progress = float(np.clip(
                    self._gate_idle_progress + (dt / max(self._gate_idle_duration_s, 1e-3)),
                    0.0,
                    1.0,
                ))
                idle_t = self._s_curve(self._gate_idle_progress)

                # Interpolate center_y: current → creep center
                self._base_center_y = float(
                    self._gate_idle_start_center_y
                    + ((type_center_y - self._gate_idle_start_center_y) * idle_t)
                )

                # Interpolate radius: current → park radius
                radius = float(
                    self._gate_idle_start_radius
                    + ((type_park_radius - self._gate_idle_start_radius) * idle_t)
                )
                self._actual_radius = float(np.clip(radius, type_park_radius, 1.0))

                # Decelerate angular velocity smoothly to near-idle
                idle_angular_speed = 0.3  # gentle residual spin
                blended_speed = float(
                    self._gate_idle_start_angular_vel
                    + ((idle_angular_speed - self._gate_idle_start_angular_vel) * idle_t)
                )
                self._orbit_phase = float((self._orbit_phase + (blended_speed * dt)) % (2.0 * np.pi))
                self._angular_velocity = float(blended_speed)
                self._last_phase_for_velocity = self._orbit_phase

                # Bass jitter active so the dot bounces with bass
                jitter_alpha, jitter_beta = self._compute_bass_jitter_offsets(
                    event=event, dt=dt,
                )
                treble_bump = float(self._intelligence.compute_treble_lift(0.0)) if self._treble_lift_enabled else 0.0
                self._reactive_bounce_y = float(np.clip(jitter_beta + treble_bump, -0.30, 0.30))

                total_center_y = float(self._base_center_y + self._reactive_bounce_y)
                orbit_radius = float(min(self._actual_radius, self._radius_cap_for_center(total_center_y)))

                angle = float(self._orbit_phase)
                alpha = float(orbit_radius * np.cos(angle)) + float(jitter_alpha * 0.3)
                beta = float(total_center_y + (orbit_radius * np.sin(angle)))

                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))
                self._last_journey_completion = 1.0
            else:
                started_new_journey = bool(progress <= 1e-9 and self._last_journey_completion > 1e-9)
                if started_new_journey:
                    if decision.trigger_kind in ("beat", "downbeat", "syncopation", "start"):
                        self._hold_start_pose_until_reactive = False
                    prior_completion = float(self._last_journey_completion)
                    if self._wait_swirl_active:
                        # Slingshot exit: inherit loop momentum and phase for smooth launch
                        self._journey_linked = True
                        self._journey_cold_start = False
                        self._journey_relink_active = True
                        self._journey_relink_start_radius = float(np.clip(self._actual_radius, self._min_radius, 1.0))
                    elif self._exit_spiral_active:
                        self._journey_linked = True
                        self._journey_cold_start = False
                        self._journey_relink_active = True
                        self._journey_relink_start_radius = float(np.clip(self._actual_radius, 0.70, 1.0))
                    elif self._gate_idle_active:
                        # Coming out of gate-idle deceleration — treat as linked
                        self._journey_linked = True
                        self._journey_cold_start = False
                        self._journey_relink_active = True
                        self._journey_relink_start_radius = float(np.clip(self._actual_radius, 0.30, 1.0))
                    else:
                        self._journey_linked = bool(prior_completion < 0.999)
                        self._journey_cold_start = not self._journey_linked
                        self._journey_relink_active = False

                    # Reset wait-swirl and gate-idle state on any new journey
                    self._wait_swirl_active = False
                    self._wait_swirl_progress = 0.0
                    self._gate_idle_active = False
                    self._gate_idle_progress = 0.0

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
                    # Expand max_radius toward 1.0 based on energy fullness:
                    # quiet music stays at configured max (0.90), full music → 1.0
                    base_max = float(geom["max_radius"])
                    fullness = float(np.clip(decision.energy_fullness, 0.0, 1.0))
                    # Smooth expansion: only starts opening above 0.4 fullness
                    expand_t = float(np.clip((fullness - 0.4) / 0.6, 0.0, 1.0))
                    expanded_max = float(base_max + (1.0 - base_max) * (expand_t * expand_t))
                    # Session arc influence: long-term energy nudges max_radius
                    if getattr(self.config.stroke, 'session_arc_enabled', True):
                        arc_inf = float(getattr(self.config.stroke, 'session_arc_radius_influence', 0.10) or 0.10)
                        session_nudge = (self._session_energy_ema - 0.5) * 2.0 * arc_inf
                        expanded_max = float(np.clip(expanded_max + session_nudge, base_max, 1.0))
                    self._journey_max_radius = float(np.clip(expanded_max, base_max, 1.0))

                    # Intensity timer: scale available dynamic range toward park radius
                    if self._intensity_ramp_affect_size and self._intensity_ramp_mult < 1.0:
                        self._journey_max_radius = float(
                            self._journey_park_radius
                            + ((self._journey_max_radius - self._journey_park_radius) * self._intensity_ramp_mult)
                        )

                    self._journey_energy_fullness = fullness

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
                        # Post-silence radius ramp: quintic ease from floor to full
                        ramp_t = self._quintic_ease(startup_ratio)
                        self._post_silence_radius_ramp = float(
                            self._post_silence_radius_floor
                            + ((1.0 - self._post_silence_radius_floor) * ramp_t)
                        )
                        self._startup_beats_seen += 1.0
                    else:
                        self._journey_startup_momentum = 1.0
                        self._post_silence_radius_ramp = 1.0

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

                # Post-silence slow takeoff: scale bloom target down during ramp period
                # so the orbit starts close to park and gently expands over several beats
                radius_ramp = float(np.clip(self._post_silence_radius_ramp, 0.0, 1.0))
                if radius_ramp < 1.0:
                    ramp_park = float(self._park_idle_radius)
                    bloom_target_radius = float(
                        ramp_park + ((bloom_target_radius - ramp_park) * radius_ramp)
                    )

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

                    # allow_wait_swirl_in_creep is True when a gate (e.g. mid_trigger)
                    # is repeatedly blocking real beats and forcing the trigger to creep.
                    # In that case beats ARE incoming — treat as continuation_expected so
                    # the orbit enters wait-swirl instead of exit-spiralling to a hold.
                    allow_wait_swirl_in_creep = bool(self._last_gate_fail)
                    if (continuation_expected or allow_wait_swirl_in_creep) and (decision.trigger_kind != "creep" or allow_wait_swirl_in_creep):
                        hold_cycle_pending = bool(
                            self._radius_hold_active and not self._radius_hold_visible_cycle_done
                        )
                        if hold_cycle_pending:
                            pass
                        else:
                            # ── Wait-swirl: fast spiral into park-like idle ──
                            self._exit_spiral_active = False
                            self._exit_spiral_progress = 0.0
                            self._settle_active = False
                            self._radius_hold_active = False

                            # First wait frame: latch current state
                            if not self._wait_swirl_active:
                                self._wait_swirl_active = True
                                self._wait_swirl_progress = 0.0
                                self._wait_swirl_start_center_y = float(self._base_center_y)
                                self._wait_swirl_start_radius = float(max(self._actual_radius, self._park_idle_radius))
                                self._wait_swirl_start_angular_vel = float(
                                    max(abs(self._angular_velocity), self._journey_nominal_angular_speed * 0.85)
                                )

                            # Advance wait-swirl progress
                            self._wait_swirl_progress = float(np.clip(
                                self._wait_swirl_progress + (dt / max(self._wait_swirl_duration_s, 1e-3)),
                                0.0,
                                1.0,
                            ))
                            wait_t = self._s_curve(self._wait_swirl_progress)

                            # Interpolate radius: current → slingshot loop radius
                            wait_radius = float(
                                self._wait_swirl_start_radius
                                + ((self._wait_swirl_target_radius - self._wait_swirl_start_radius) * wait_t)
                            )
                            self._actual_radius = float(np.clip(wait_radius, self._wait_swirl_target_radius, 1.0))

                            # Center_y with hook maneuver: dip deeper at midpoint, settle at baseline.
                            # Creates an upward-hook as the dot spirals into the slingshot loop.
                            hook_bump = float(self._wait_swirl_hook_depth * np.sin(np.pi * wait_t))
                            self._base_center_y = float(
                                self._wait_swirl_start_center_y
                                + ((self._baseline_center_y - self._wait_swirl_start_center_y) * wait_t)
                                + hook_bump
                            )

                            # Decelerate to slingshot loop speed (energetic, not idle)
                            bpm_wait = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
                            if bpm_wait <= 0.0:
                                bpm_wait = float(getattr(event, "bpm", 0.0) or 0.0)
                            bpm_wait = float(np.clip(bpm_wait if bpm_wait > 0.0 else 120.0, 40.0, 240.0))
                            idle_angular_speed = float((2.0 * np.pi) * (bpm_wait / 60.0) * self._idle_loops_per_beat)
                            slingshot_speed = float(max(
                                self._journey_nominal_angular_speed * 0.80,
                                idle_angular_speed,
                            ))
                            blended_speed = float(
                                self._wait_swirl_start_angular_vel
                                + ((slingshot_speed - self._wait_swirl_start_angular_vel) * wait_t)
                            )
                            angle = float((self._orbit_phase + (blended_speed * dt)) % (2.0 * np.pi))
                            self._angular_velocity = float(blended_speed)

                            # Bass-reactive jitter at wait (same as park)
                            jitter_alpha, jitter_beta = self._compute_bass_jitter_offsets(
                                event=event, dt=dt,
                            )
                            treble_bump = float(self._intelligence.compute_treble_lift(0.0)) if self._treble_lift_enabled else 0.0
                            self._reactive_bounce_y = float(np.clip(jitter_beta + treble_bump, -0.30, 0.30))

                            # Compute final position directly — stored for post-fix
                            # (downstream code will overwrite locals; we restore at end)
                            total_center_y = float(self._base_center_y + self._reactive_bounce_y)
                            orbit_radius = float(min(self._actual_radius, self._radius_cap_for_center(total_center_y)))

                            self._wait_swirl_alpha = float(orbit_radius * np.cos(angle)) + float(jitter_alpha * 0.3)
                            self._wait_swirl_beta = float(total_center_y + (orbit_radius * np.sin(angle)))

                            ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                            self._wait_swirl_volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))
                            self._wait_swirl_base_center_y = float(self._base_center_y)
                            self._wait_swirl_reactive_bounce_y = float(self._reactive_bounce_y)
                            self._last_journey_completion = progress

                    else:
                        self._settle_active = False
                        self._radius_hold_active = False
                        self._wait_swirl_active = False
                        self._wait_swirl_progress = 0.0
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
                            self._radius_hold_visible_cycle_done = False
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
                if self._wait_swirl_active:
                    # Wait-swirl already computed alpha/beta/volume directly
                    # in the continuation-wait block above — skip radius/center logic.
                    radius = self._actual_radius
                elif self._radius_hold_active:
                    radius = float(np.clip(self._radius_hold_value, type_park_radius, type_max_radius))
                    self._radius_hold_visible_cycle_done = True
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
                        # Fast bloom: expand to full radius within first 25% of orbit
                        unhook_window = 0.25
                        unhook_t = float(np.clip(first_pass_progress / unhook_window, 0.0, 1.0))
                        # Quintic ease (6t^5-15t^4+10t^3) for velvet-smooth expansion
                        radius_blend = self._quintic_ease(unhook_t)
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
                        # Fast relink: expand from slingshot loop to full radius in ~15% of orbit
                        relink_window = 0.15
                        relink_t = float(np.clip(first_pass_progress / relink_window, 0.0, 1.0))
                        # Quintic ease for buttery-smooth slingshot release
                        relink_blend = self._quintic_ease(relink_t)
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

                if self._wait_swirl_active:
                    min_radius_bound = 0.0
                elif decision.trigger_kind == "start":
                    min_radius_bound = self._min_radius
                else:
                    min_radius_bound = 0.70
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

        # Wait-swirl computed its own final position — restore from stored values
        # (downstream code ran and overwrote locals with journey-based geometry)
        if self._wait_swirl_active:
            alpha = self._wait_swirl_alpha
            beta = self._wait_swirl_beta
            volume = self._wait_swirl_volume
            self._base_center_y = self._wait_swirl_base_center_y
            self._reactive_bounce_y = self._wait_swirl_reactive_bounce_y

        # ── Expression layer: apply center Y wander offset only ──
        beta = float(beta + self._center_y_offset)

        # ── §2/§3: Expression pause spiral override ──
        if self._expr_pause_spiral_active:
            # Logarithmic spiral inward: radius shrinks logarithmically
            t_sp = self._quintic_ease(self._expr_pause_spiral_progress)
            # Logarithmic interpolation: r = r_start * (r_target/r_start)^t
            r_start = max(self._expr_pause_spiral_start_radius, 0.1)
            r_target = max(self._expr_pause_spiral_target_radius, 0.05)
            log_radius = float(r_start * ((r_target / r_start) ** t_sp))
            # Phase keeps advancing (inherited angular velocity, slowing down)
            total_center_y = float(self._base_center_y + self._reactive_bounce_y)
            alpha = float(log_radius * np.cos(self._orbit_phase))
            beta = float(total_center_y + log_radius * np.sin(self._orbit_phase))

        if self._expr_pause_return_active:
            # §3: Spiral back out to normal orbit from pause, landing at anchor
            t_ret = self._quintic_ease(self._expr_pause_return_progress)
            r_start_ret = max(self._expr_pause_return_start_radius, 0.05)
            r_target_ret = float(max(self._actual_radius, 0.5))
            log_radius_ret = float(r_start_ret * ((r_target_ret / r_start_ret) ** t_ret))
            total_center_y = float(self._base_center_y + self._reactive_bounce_y)
            alpha = float(log_radius_ret * np.cos(self._orbit_phase))
            beta = float(total_center_y + log_radius_ret * np.sin(self._orbit_phase))

        # ── Expression layer: tension pause override (legacy fade) ──
        if self._tension_pause_active:
            if now < self._tension_pause_end_time:
                alpha = self._tension_pause_hold_alpha
                beta = self._tension_pause_hold_beta
            elif now < self._tension_pause_fade_end:
                fade_dur = self._tension_pause_fade_end - self._tension_pause_end_time
                t = float(np.clip((now - self._tension_pause_end_time) / max(fade_dur, 1e-3), 0.0, 1.0))
                t = t * t * (3.0 - 2.0 * t)  # smoothstep
                alpha = float(self._tension_pause_hold_alpha + t * (alpha - self._tension_pause_hold_alpha))
                beta = float(self._tension_pause_hold_beta + t * (beta - self._tension_pause_hold_beta))
            else:
                self._tension_pause_active = False

        # ── §8: Entry journey gating — mark entry done when first journey completes ──
        if not self._post_silence_entry_done and not decision.silence_active:
            if decision.trigger_kind == "start" and decision.journey_completion >= 1.0:
                self._post_silence_entry_done = True
            elif decision.trigger_kind != "start" and decision.trigger_kind != "creep":
                # Only allow entry journey types before unlocking; force creep otherwise
                if not self._post_silence_entry_done and self._startup_beats_seen < 8:
                    # Keep dot in entry mode by not overriding alpha/beta
                    pass

        self.state.alpha = float(np.clip(alpha, -1.0, 1.0))
        self.state.beta = float(np.clip(beta, -1.0, 1.0))

        return TCodeCommand(alpha=self.state.alpha, beta=self.state.beta, duration_ms=25, volume=volume)

    # ── Expression layer ──────────────────────────────────────────────

    def _update_expression_layer(self, decision: 'BeatDecision', dt: float, now: float) -> None:
        """Per-frame expression updates: center wander, energy tracking,
        direction changes, tension pause detection, session arc."""

        energy = float(np.clip(decision.energy_fullness, 0.0, 1.0))
        self._energy_history.append(energy)

        # Session arc EMA (mirrors beat_intelligence for local use)
        if getattr(self.config.stroke, 'session_arc_enabled', True):
            sa_alpha = float(getattr(self.config.stroke, 'session_arc_ema_alpha', 0.001) or 0.001)
            self._session_energy_ema += sa_alpha * (energy - self._session_energy_ema)

        # ── Center wandering (Y-axis only) ──
        if (getattr(self.config.stroke, 'center_wander_enabled', True)
                and not decision.silence_active
                and self._orbit_phase_initialized):
            cycle_s = float(getattr(self.config.stroke, 'center_wander_cycle_s', 25.0) or 25.0)
            max_y = float(
                getattr(
                    self.config.stroke,
                    'center_wander_max_y',
                    getattr(self.config.stroke, 'center_wander_max_x', 0.20),
                )
                or 0.20
            )
            e_scale = float(getattr(self.config.stroke, 'center_wander_energy_scale', 0.6) or 0.6)

            self._center_wander_phase += dt / max(cycle_s, 1.0)
            # Two harmonics for organic feel (golden ratio second harmonic)
            raw = float(
                0.70 * np.sin(2.0 * np.pi * self._center_wander_phase)
                + 0.30 * np.sin(2.0 * np.pi * self._center_wander_phase * 1.618)
            )
            # Amplitude scales with energy: more wander when music is fuller
            amplitude = max_y * ((1.0 - e_scale) + e_scale * energy)
            self._center_y_offset = float(np.clip(raw * amplitude, -max_y, max_y))
            self._center_x_offset = 0.0
        elif decision.silence_active:
            # Gently decay wander toward center during silence
            decay = float(max(0.0, 1.0 - 2.0 * dt))
            self._center_x_offset *= decay
            self._center_y_offset *= decay

        # ── Tension pause detection (§2: spiral-in to radius ≤0.5) ──
        if getattr(self.config.stroke, 'tension_pause_enabled', True) and not decision.silence_active:
            cooldown = float(getattr(self.config.stroke, 'tension_pause_cooldown_s', 10.0) or 10.0)
            drop_ratio = float(getattr(self.config.stroke, 'tension_pause_energy_drop', 0.40) or 0.40)

            if (not self._tension_pause_active
                    and not self._expr_pause_spiral_active
                    and len(self._energy_history) >= 30
                    and now - self._tension_pause_last_time > cooldown):
                recent = list(self._energy_history)
                recent_mean = float(np.mean(recent[-15:]))
                prior_mean = float(np.mean(recent[-30:-15]))
                # Only trigger when going from substantial energy to a real drop
                if prior_mean > 0.20 and (prior_mean - recent_mean) / prior_mean > drop_ratio:
                    # §2: Start logarithmic spiral-in instead of hard hold
                    self._expr_pause_spiral_active = True
                    self._expr_pause_spiral_progress = 0.0
                    self._expr_pause_spiral_start_radius = float(max(self._actual_radius, 0.3))
                    self._expr_pause_spiral_target_radius = 0.45  # ≤0.5
                    self._expr_pause_spiral_start_angle = float(self._orbit_phase)
                    self._expr_pause_spiral_duration_beats = 2.0
                    self._tension_pause_last_time = now

                    # After spiral-in, we'll need to spiral back out (§3)
                    self._expr_pause_return_active = False
                    self._expr_pause_return_progress = 0.0

        # ── §2/§3: Advance expression pause spiral states ──
        if self._expr_pause_spiral_active and not decision.silence_active:
            bpm_for_spiral = float(getattr(self.audio_engine, '_metronome_bpm', 120.0) if self.audio_engine else 120.0)
            bpm_for_spiral = float(np.clip(bpm_for_spiral if bpm_for_spiral > 0 else 120.0, 40.0, 240.0))
            spiral_dur_s = float(self._expr_pause_spiral_duration_beats * 60.0 / bpm_for_spiral)
            self._expr_pause_spiral_progress = float(np.clip(
                self._expr_pause_spiral_progress + (dt / max(spiral_dur_s, 0.1)),
                0.0, 1.0,
            ))
            if self._expr_pause_spiral_progress >= 1.0:
                # Spiral-in complete → start spiral return (§3)
                self._expr_pause_spiral_active = False
                self._expr_pause_return_active = True
                self._expr_pause_return_progress = 0.0
                self._expr_pause_return_start_angle = float(self._orbit_phase)
                self._expr_pause_return_start_radius = self._expr_pause_spiral_target_radius

        if self._expr_pause_return_active and not decision.silence_active:
            bpm_for_return = float(getattr(self.audio_engine, '_metronome_bpm', 120.0) if self.audio_engine else 120.0)
            bpm_for_return = float(np.clip(bpm_for_return if bpm_for_return > 0 else 120.0, 40.0, 240.0))
            return_dur_s = float(2.0 * 60.0 / bpm_for_return)  # 2 beats to return
            self._expr_pause_return_progress = float(np.clip(
                self._expr_pause_return_progress + (dt / max(return_dur_s, 0.1)),
                0.0, 1.0,
            ))
            if self._expr_pause_return_progress >= 1.0:
                self._expr_pause_return_active = False

        # ── §1: Anchor phrase management (direction change → new anchor) ──
        if getattr(self.config.stroke, 'direction_change_enabled', True) and not decision.silence_active:
            interval_s = float(getattr(self.config.stroke, 'direction_change_interval_s', 15.0) or 15.0)
            drop_needed = float(getattr(self.config.stroke, 'direction_change_energy_drop', 0.35) or 0.35)

            if now - self._last_direction_change_time > interval_s and len(self._energy_history) >= 30:
                recent = list(self._energy_history)
                recent_mean = float(np.mean(recent[-15:]))
                prior_mean = float(np.mean(recent[-30:-15]))
                # Trigger on significant energy transition (either direction)
                if prior_mean > 0.08 and abs(prior_mean - recent_mean) / max(prior_mean, 0.08) > drop_needed:
                    self._orbit_direction *= -1
                    self._last_direction_change_time = now
                    # §1: Direction change → new anchor may be chosen
                    self._anchor_phrase_locked = False
                    # Randomly choose +Y or -Y for next phrase
                    self._anchor_sign = 1 if float(np.random.random()) > 0.5 else -1

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

        # Dominant-frequency map dictates jitter speed and size with ±50% envelope
        # around base values: 0.5x..1.5x.
        centered = float((2.0 * norm_smooth) - 1.0)   # -1..1
        delta = float(0.5 * centered)                 # -0.5..0.5
        speed_mult = float(np.clip(1.0 + delta, 0.5, 1.5))
        size_mult = float(np.clip(1.0 + delta, 0.5, 1.5))

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
        treble_bump = float(self._intelligence.compute_treble_lift(0.0)) if self._treble_lift_enabled else 0.0
        return float(np.clip(jitter_beta + treble_bump, -0.30, 0.30))

    @staticmethod
    def _s_curve(progress: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        return float(p * p * (3.0 - (2.0 * p)))

    @staticmethod
    def _quintic_ease(progress: float) -> float:
        """Quintic smoothstep (6t^5 - 15t^4 + 10t^3).

        Smoother than cubic S-curve: zero first AND second derivative
        at both endpoints, giving velvet-smooth radius expansion with
        no perceptible 'knee' at start or end.
        """
        p = float(np.clip(progress, 0.0, 1.0))
        return float(p * p * p * (p * (p * 6.0 - 15.0) + 10.0))

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

    def _radius_cap_for_center(self, center_y: float) -> float:
        """Maximum radius that keeps orbit inside normalized [-1, 1] bounds in both axes."""
        # Include expression-layer Y wander in the cap math so boundary
        # protection matches prior X-wander behavior and avoids D-shape clipping.
        effective_center_y = float(center_y + self._center_y_offset)
        y_cap = float(max(0.0, min(1.0 - effective_center_y, 1.0 + effective_center_y)))
        x_cap = float(max(0.0, 1.0 - abs(self._center_x_offset)))
        return min(y_cap, x_cap)

    def _compute_landing_rotation(self, start_angle: float, interval_beats: int) -> float:
        # §7: All journeys are capped at 1x 360° rotation.
        # Exception: entry journey after silence gets 2 rotations over ≥8 beats.
        is_entry = bool(self._intelligence.is_recovering or (not self._post_silence_entry_done and self._startup_beats_seen < 2))
        max_turns = 2.0 if is_entry else 1.0

        # Base turn count: 1 full lap per journey (capped)
        turns = 1.0

        # Orbit speed variation: energy-driven turn count
        if getattr(self.config.stroke, 'orbit_speed_variation_enabled', True):
            fullness = float(np.clip(self._journey_energy_fullness, 0.0, 1.0))
            min_t = float(getattr(self.config.stroke, 'orbit_speed_min_turns', 0.75) or 0.75)
            max_t = float(getattr(self.config.stroke, 'orbit_speed_max_turns', 1.5) or 1.5)
            # Smoothstep mapping: low energy stays slow, high energy opens up
            t = fullness * fullness * (3.0 - 2.0 * fullness)
            turns = float(min_t + t * (max_t - min_t))

        # Session arc influence: slightly expand/contract turns with long-term energy
        if getattr(self.config.stroke, 'session_arc_enabled', True):
            arc_influence = float(getattr(self.config.stroke, 'session_arc_radius_influence', 0.10) or 0.10)
            # Session intensity biases turns slightly (±10%)
            session_bias = (self._session_energy_ema - 0.5) * 2.0 * arc_influence
            turns = float(np.clip(turns + session_bias, 0.5, max_turns))

        # Intensity timer: scale dynamic turn range toward minimum
        if self._intensity_ramp_affect_speed and self._intensity_ramp_mult < 1.0:
            base_turns = float(getattr(self.config.stroke, 'orbit_speed_min_turns', 0.75) or 0.75)
            turns = float(base_turns + ((turns - base_turns) * self._intensity_ramp_mult))

        # §7: Hard clamp to max_turns
        turns = float(np.clip(turns, 0.5, max_turns))

        # §1: Anchor landing – ensure the journey ends within ±10° of the
        # chosen Y-axis anchor.  Adjust total rotation so the arrival angle
        # falls inside the anchor swing window.
        anchor_angle = float(np.pi / 2.0) * self._anchor_sign  # +Y or -Y
        target_end = float(start_angle + turns * 2.0 * np.pi * self._orbit_direction)
        # Nearest anchor crossing to target_end
        swing_rad = float(np.deg2rad(self._anchor_swing_deg))
        best_end = self._nearest_anchor_crossing(target_end, anchor_angle, swing_rad)
        # Recompute turns from adjusted end
        delta = best_end - start_angle
        if abs(self._orbit_direction) > 0:
            # Ensure delta sign matches direction
            if self._orbit_direction > 0 and delta < 0:
                delta += 2.0 * np.pi
            elif self._orbit_direction < 0 and delta > 0:
                delta -= 2.0 * np.pi
        turns = abs(delta) / (2.0 * np.pi)
        turns = float(np.clip(turns, 0.3, max_turns))

        # Apply orbit direction (CW/CCW)
        rotation = float(turns * 2.0 * np.pi * self._orbit_direction)
        return rotation

    @staticmethod
    def _nearest_anchor_crossing(target_angle: float, anchor_angle: float, swing_rad: float) -> float:
        """Find the angle nearest to target_angle within ±swing_rad of anchor_angle.
        Anchor_angle repeats every 2π. Returns the adjusted angle."""
        two_pi = 2.0 * np.pi
        # Normalize to find nearest multiple of 2π offset
        base = anchor_angle
        # Number of full rotations in target
        n = round((target_angle - base) / two_pi)
        candidate = base + n * two_pi
        # Check ±1 rotation too
        candidates = [candidate - two_pi, candidate, candidate + two_pi]
        best = min(candidates, key=lambda c: abs(c - target_angle))
        # Clamp within swing window
        delta = target_angle - best
        clamped_delta = float(np.clip(delta, -swing_rad, swing_rad))
        return float(best + clamped_delta)
