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
        self._journey_latched_bloom = 0.70     # radius_bloom frozen at journey start
        self._journey_center_y = self._baseline_center_y
        self._journey_park_radius = self._park_radius
        self._journey_max_radius = self._max_radius
        self._park_angle = float(np.pi / 2.0)
        self._journey_start_angle = self._park_angle
        self._journey_start_alpha = self.state.alpha
        self._journey_start_beta = self.state.beta
        self._journey_start_time_mono = 0.0
        self._journey_timing_beats = 2
        self._journey_total_rotation = float(2.0 * np.pi)
        self._last_journey_completion = 1.0
        self._actual_radius = self._park_radius
        self._angular_velocity = 0.0
        self._last_phase_for_velocity = self._orbit_phase
        self._journey_initial_speed_slope = 0.0
        self._journey_nominal_angular_speed = 0.0  # nominal speed for continuation glide
        self._journey_target_radius = self._park_radius  # latched at journey start; never re-evaluated mid-arc
        self._orbit_phase_initialized = False  # True once orbit_phase has been actively tracked
        self._lazy_glide_active = False
        self._journey_cold_start = True
        self._journey_linked = False
        self._journey_relink_active = False
        self._journey_relink_start_radius = 0.90
        self._startup_momentum_min = 0.15
        self._startup_ramp_beats = 6.0
        self._startup_beats_seen = 0.0
        self._journey_startup_momentum = 1.0
        self._post_silence_radius_ramp = 1.0   # 0→1 over first beats after silence
        self._post_silence_radius_floor = 0.12 # start radius fraction after silence
        self._hold_start_pose_until_reactive = False
        self._idle_radius = self._min_radius
        self._silence_decay_per_beat = 0.40
        self._idle_loops_per_beat = 0.125

        # Swirl-to-park state: tracks the spiral transition into idle
        self._swirl_progress = 0.0       # 0→1 S-curve interpolant
        self._swirl_duration_s = 1.8     # total time to spirally arrive at park
        self._swirl_start_center_y = self._baseline_center_y
        self._swirl_start_radius = self._park_radius
        self._swirl_entering = False     # True on first silence frame after motion

        # Creep-disabled park transition: quintic ease to park when creep is off
        self._creep_park_active = False       # True during quintic glide to park
        self._creep_park_progress = 0.0       # 0→1 quintic interpolant
        self._creep_park_duration_beats = 1.5 # transition length in beats
        self._creep_park_start_radius = self._park_radius
        self._creep_park_start_center_y = self._baseline_center_y

        # Mode transition state: smooth spiral between park_bounce_only ↔ full arc
        self._mode_transition_active = False     # True during mode change spiral
        self._mode_transition_progress = 0.0     # 0→1 S-curve interpolant
        self._mode_transition_duration_s = 1.2   # ~2 beats at 120 BPM
        self._mode_transition_start_radius = self._park_radius
        self._mode_transition_start_center_y = self._baseline_center_y
        self._mode_transition_target_radius = self._park_radius
        self._mode_transition_target_center_y = self._baseline_center_y
        self._last_mode_was_park_bounce = False  # Track previous frame mode

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
        self._hat_bounce_phase = 0.0
        self._hat_bounce_amp = 0.0

        self._last_gate_fail = ""  # diagnostic: which gate blocked last beat-family event
        self._last_decision = None      # latest BeatDecision (for keyboard teacher snapshot)

        # ── Fixed anchor state (§1) ──
        self._anchor_sign: int = 1               # +1 = +Y anchor, -1 = -Y anchor
        self._anchor_angle: float = float(np.pi / 2.0)  # angle of anchor on orbit
        self._anchor_swing_deg: float = 10.0     # ±10° swing around y-axis
        self._anchor_phrase_locked: bool = False  # True once chosen for current phrase

        # ── Spiral-out (slingshot exit from park) ──
        self._spiral_out_active: bool = False       # True while spiralling out of park
        self._spiral_out_progress: float = 0.0      # 0→1 over spiral duration
        self._spiral_out_beats: float = 3.0          # how many beats the spiral-out takes
        self._spiral_out_start_radius: float = self._park_idle_radius
        self._spiral_out_target_radius: float = self._park_radius
        self._spiral_out_start_center_y: float = self._baseline_center_y
        self._spiral_out_target_center_y: float = self._baseline_center_y

        # ── Silence-exit position crossfade ──
        # Blends from last park position to computed trajectory over N beats
        # so the device swirls out gradually instead of teleporting.
        self._silence_exit_xfade_active: bool = False
        self._silence_exit_xfade_progress: float = 0.0
        self._silence_exit_xfade_beats: float = 2.0   # duration in beats
        self._silence_exit_latch_alpha: float = 0.0
        self._silence_exit_latch_beta: float = 0.0
        self._was_silence_active: bool = True           # tracks previous frame

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
        self._energy_history: deque = deque(maxlen=120)  # ~2s at 60fps
        self._session_energy_ema: float = 0.5

        # ── Intensity timer ramp (session-level escalation) ──
        self._intensity_ramp_start_time: float = 0.0
        self._intensity_ramp_started: bool = False
        self._intensity_ramp_mult: float = 1.0
        self._intensity_ramp_floor: float = 0.25
        self._intensity_ramp_affect_size: bool = True
        self._intensity_ramp_affect_speed: bool = True

        # ── Rate-limiter velocity state (for smoothing across ALL paths) ──
        self._smoothed_da: float = 0.0
        self._smoothed_db: float = 0.0

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

    def _rate_limited_output(
        self,
        alpha: float,
        beta: float,
        volume: float,
        dt: float,
    ) -> TCodeCommand:
        """Apply velocity-smoothed, per-frame-capped rate limiting and return TCodeCommand.

        This MUST be the single exit point for every frame so that no code
        path can produce a positional teleport.  The three-stage pipeline:
          1. Velocity EMA – smooths sudden *changes* in per-frame delta so
             the device never reverses or accelerates instantaneously.
          2. *Radial* rate cap – clamps the 2D magnitude of the delta
             vector (not each axis independently) so circular orbits
             stay round instead of developing flat edges at the quadrants.
          3. Per-frame hard cap – absolute ceiling prevents dt-spike
             frames from allowing oversized jumps.
        """
        max_delta_per_s = 3.6
        max_delta_per_frame = 0.08  # absolute safety ceiling per frame
        max_delta = float(min(max_delta_per_s * max(dt, 1e-4), max_delta_per_frame))

        prev_a = float(self.state.alpha)
        prev_b = float(self.state.beta)

        raw_da = float(alpha - prev_a)
        raw_db = float(beta - prev_b)

        # Velocity EMA: smooths sudden direction/speed changes while
        # converging quickly (~3 frames at factor=0.50) for steady motion.
        smooth_factor = 0.50
        da = float(self._smoothed_da + smooth_factor * (raw_da - self._smoothed_da))
        db = float(self._smoothed_db + smooth_factor * (raw_db - self._smoothed_db))

        # Radial rate limiter: clamp the *magnitude* of the 2D delta
        # vector so both axes scale proportionally.  This preserves the
        # direction of motion (and therefore circular shape) unlike
        # independent per-axis clamping which flattens corners.
        mag = float(np.sqrt(da * da + db * db))
        if mag > max_delta and mag > 1e-9:
            scale = max_delta / mag
            da = float(da * scale)
            db = float(db * scale)

        # Store clamped value so EMA tracks actual movement, not desired
        self._smoothed_da = da
        self._smoothed_db = db

        alpha = float(np.clip(prev_a + da, -1.0, 1.0))
        beta = float(np.clip(prev_b + db, -1.0, 1.0))

        self.state.alpha = alpha
        self.state.beta = beta

        return TCodeCommand(alpha=alpha, beta=beta, duration_ms=25, volume=volume)

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
        self._last_decision = decision

        # ── Mode transition detection: smooth spiral between park_bounce_only ↔ full arc ──
        current_mode_is_park_bounce = bool(getattr(decision, "park_bounce_only", False)) and not decision.silence_active
        mode_changed = current_mode_is_park_bounce != self._last_mode_was_park_bounce

        if mode_changed and not self._mode_transition_active:
            # Start new transition
            self._mode_transition_active = True
            self._mode_transition_progress = 0.0
            self._mode_transition_start_radius = float(self._actual_radius)
            self._mode_transition_start_center_y = float(self._base_center_y + self._reactive_bounce_y)
            
            # Reset swirl state to prevent interference with mode transition
            self._swirl_entering = False
            self._swirl_progress = 0.0
            
            # Determine target geometry based on new mode
            if current_mode_is_park_bounce:
                # Transitioning TO park_bounce_only: spiral into small idle orbit
                self._mode_transition_target_radius = float(self._park_idle_radius)
                self._mode_transition_target_center_y = float(self._baseline_center_y)
            else:
                # Transitioning TO full arc: spiral out to journey geometry
                # Use current decision's trigger geometry, or default to park_radius
                geom = self.config.stroke.orbit_geometry.get(decision.trigger_kind, {
                    "center_y": self._baseline_center_y, "park_radius": 0.70, "max_radius": 1.0
                })
                self._mode_transition_target_radius = float(geom["park_radius"])
                self._mode_transition_target_center_y = float(geom["center_y"])

        self._last_mode_was_park_bounce = current_mode_is_park_bounce

        # ── Advance mode transition if active ──
        if self._mode_transition_active:
            self._mode_transition_progress = float(np.clip(
                self._mode_transition_progress + (dt / max(self._mode_transition_duration_s, 1e-3)),
                0.0,
                1.0,
            ))
            trans_t = self._quintic_ease(self._mode_transition_progress)
            
            # Interpolate radius and center_y with quintic ease
            self._actual_radius = float(
                self._mode_transition_start_radius
                + ((self._mode_transition_target_radius - self._mode_transition_start_radius) * trans_t)
            )
            trans_center_y = float(
                self._mode_transition_start_center_y
                + ((self._mode_transition_target_center_y - self._mode_transition_start_center_y) * trans_t)
            )
            self._base_center_y = trans_center_y
            
            # Transition complete when progress reaches 1.0
            if self._mode_transition_progress >= 1.0:
                self._mode_transition_active = False

        if current_mode_is_park_bounce:
            ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
            
            # During mode transition, use interpolated geometry directly
            # to avoid conflict with _apply_park_motion_frame's internal swirl
            if self._mode_transition_active:
                # Compute position using transition-interpolated radius & center
                bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
                if bpm <= 0.0:
                    bpm = float(getattr(event, "bpm", 0.0) or 0.0)
                bpm = float(np.clip(bpm if bpm > 0.0 else 120.0, 40.0, 240.0))
                
                idle_angular_speed = float((2.0 * np.pi) * (bpm / 60.0) * self._idle_loops_per_beat)
                self._orbit_phase = float((self._orbit_phase + (idle_angular_speed * dt)) % (2.0 * np.pi))
                self._angular_velocity = float(idle_angular_speed)
                self._last_phase_for_velocity = self._orbit_phase
                
                jitter_alpha, jitter_beta = self._compute_bass_jitter_offsets(event=event, dt=dt)
                treble_bump = float(self._intelligence.compute_treble_lift(0.0)) if self._treble_lift_enabled else 0.0
                hat_bump = self._compute_hat_bounce_offset(event=event, dt=dt)
                self._reactive_bounce_y = float(np.clip(jitter_beta + treble_bump + hat_bump, -0.30, 0.30))
                
                total_center_y = float(self._base_center_y + self._reactive_bounce_y)
                orbit_radius = float(min(self._actual_radius, self._radius_cap_for_center(total_center_y)))
                
                angle = float(self._orbit_phase)
                alpha = float(orbit_radius * np.cos(angle)) + float(jitter_alpha * 0.3)
                beta = float(total_center_y + (orbit_radius * np.sin(angle)))
                volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))
            else:
                # No mode transition: use normal park_motion_frame with swirl
                alpha, beta, volume = self._apply_park_motion_frame(
                    event=event,
                    dt=dt,
                    fade=ramp,
                )
            
            self._last_journey_completion = 1.0
            return self._rate_limited_output(alpha, beta, volume, dt)

        if hitch_soft_reset:
            self._angular_velocity = 0.0
            self._last_phase_for_velocity = self._orbit_phase
            # Reset velocity EMA on hitch so stale momentum doesn't linger
            self._smoothed_da = 0.0
            self._smoothed_db = 0.0

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
            # Reset spiral-out so next exit from park triggers it
            self._spiral_out_active = False
            self._spiral_out_progress = 0.0
            # Reset silence-exit crossfade so it re-triggers on next exit
            self._silence_exit_xfade_active = False
            self._silence_exit_xfade_progress = 0.0
            # Reset entry gating on silence
            self._post_silence_entry_done = False
            self._post_wait_reentry_active = False
            self._anchor_phrase_locked = False

            creep_motion_disabled = not bool(getattr(self.config.creep, "enabled", True))
            if creep_motion_disabled and decision.trigger_kind == "creep":
                self._swirl_entering = False
                self._swirl_progress = 0.0

                # ── Quintic ease to park (instead of instant snap) ──
                # First frame: latch current geometry as transition start
                if not self._creep_park_active:
                    self._creep_park_active = True
                    self._creep_park_progress = 0.0
                    self._creep_park_start_radius = float(self._actual_radius)
                    self._creep_park_start_center_y = float(self._base_center_y)

                # Derive beat period from BPM for time→beats conversion
                bpm_cp = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
                if bpm_cp <= 0.0:
                    bpm_cp = float(getattr(event, "bpm", 0.0) or 0.0)
                bpm_cp = float(np.clip(bpm_cp if bpm_cp > 0.0 else 120.0, 40.0, 240.0))
                beat_period_s = 60.0 / bpm_cp
                duration_s = self._creep_park_duration_beats * beat_period_s

                # Advance transition progress
                self._creep_park_progress = float(np.clip(
                    self._creep_park_progress + (dt / max(duration_s, 1e-3)),
                    0.0,
                    1.0,
                ))
                cp_t = self._quintic_ease(self._creep_park_progress)

                # Interpolate radius and center_y toward park targets
                self._actual_radius = float(
                    self._creep_park_start_radius
                    + ((self._park_idle_radius - self._creep_park_start_radius) * cp_t)
                )
                self._base_center_y = float(
                    self._creep_park_start_center_y
                    + ((self._baseline_center_y - self._creep_park_start_center_y) * cp_t)
                )

                # Decelerate angular velocity smoothly toward zero
                idle_speed_cp = float((2.0 * np.pi) * (bpm_cp / 60.0) * self._idle_loops_per_beat)
                current_speed_cp = max(abs(self._angular_velocity), idle_speed_cp)
                blended_speed_cp = float(
                    current_speed_cp + ((idle_speed_cp - current_speed_cp) * cp_t)
                )
                direction_sign = 1.0 if self._angular_velocity >= 0 else -1.0
                self._angular_velocity = float(blended_speed_cp * direction_sign)
                self._orbit_phase = float(
                    (self._orbit_phase + (self._angular_velocity * dt)) % (2.0 * np.pi)
                )
                self._last_phase_for_velocity = self._orbit_phase

                self._reactive_bounce_y = 0.0
                total_center_y = float(self._base_center_y + self._reactive_bounce_y)
                orbit_radius = float(min(self._actual_radius, self._radius_cap_for_center(total_center_y)))
                alpha = float(orbit_radius * np.cos(self._orbit_phase))
                beta = float(total_center_y + (orbit_radius * np.sin(self._orbit_phase)))

                fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * fade, 0.0, 1.0))
                self._last_journey_completion = 1.0
                return self._rate_limited_output(alpha, beta, volume, dt)

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
            # Quintic ease (6t⁵-15t⁴+10t³) ensures zero velocity & acceleration at arrival
            swirl_t = self._quintic_ease(self._swirl_progress)

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
            # Reset swirl / creep-park state when music resumes
            self._swirl_entering = False
            self._swirl_progress = 0.0
            self._creep_park_active = False
            self._creep_park_progress = 0.0

            # ── Silence-exit position crossfade: latch park position on transition ──
            if self._was_silence_active and not self._silence_exit_xfade_active:
                self._silence_exit_xfade_active = True
                self._silence_exit_xfade_progress = 0.0
                self._silence_exit_latch_alpha = float(self.state.alpha)
                self._silence_exit_latch_beta = float(self.state.beta)

            # ── Spiral-out: launch on first beat/downbeat/syncopation after silence ──
            if not self._spiral_out_active and self._startup_beats_seen <= 1.0:
                if decision.trigger_kind in ("beat", "downbeat", "syncopation"):
                    geom_so = self.config.stroke.orbit_geometry.get(decision.trigger_kind, {
                        "center_y": self._baseline_center_y, "park_radius": 0.70, "max_radius": 1.0
                    })
                    self._spiral_out_active = True
                    self._spiral_out_progress = 0.0
                    self._spiral_out_start_radius = float(max(self._actual_radius, self._park_idle_radius))
                    self._spiral_out_target_radius = float(geom_so["park_radius"])
                    self._spiral_out_start_center_y = float(self._base_center_y)
                    self._spiral_out_target_center_y = float(geom_so["center_y"])

            if self._hold_start_pose_until_reactive and decision.trigger_kind == "creep":
                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                alpha, beta, volume = self._apply_park_motion_frame(
                    event=event,
                    dt=dt,
                    fade=ramp,
                )
                self._last_journey_completion = 1.0
                return self._rate_limited_output(alpha, beta, volume, dt)

            if creep_motion_disabled and decision.trigger_kind == "creep":
                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                alpha, beta, volume = self._apply_park_motion_frame(
                    event=event,
                    dt=dt,
                    fade=ramp,
                )
                self._last_journey_completion = 1.0
            else:
                started_new_journey = bool(progress <= 1e-9 and self._last_journey_completion > 1e-9)
                if started_new_journey:
                    if decision.trigger_kind in ("beat", "downbeat", "syncopation", "start"):
                        self._hold_start_pose_until_reactive = False
                    prior_completion = float(self._last_journey_completion)
                    self._journey_linked = bool(prior_completion < 0.999)
                    self._journey_cold_start = not self._journey_linked
                    self._journey_relink_active = False

                    # Crossfade disabled for trajectory stability.
                    # Hard switching to the new arc preserves circular motion
                    # and avoids blended-path squiggles at trigger boundaries.
                    self._crossfade_active = False

                    self._settle_active = False  # cancel any active settle

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
                    self._journey_start_time_mono = float(now)
                    self._journey_timing_beats = self._normalize_journey_beats(decision.interval_beats)

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

                progress = self._compute_beat_timed_progress(
                    now=now,
                    event=event,
                    fallback_progress=progress,
                )

                if progress >= 1.0 and not started_new_journey and decision.trigger_kind == "creep":
                    self._settle_active = False

                    # Preserve visible continuity when a completed journey hands
                    # off into park motion from an axis-aligned terminal pose.
                    # Without this tiny nudge, deterministic frame timing can
                    # repeatedly land on alpha≈0 and look like a hard snap.
                    if abs(float(np.cos(self._orbit_phase))) < 0.02:
                        self._orbit_phase = float((self._orbit_phase + 0.10) % (2.0 * np.pi))

                    ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                    alpha, beta, volume = self._apply_park_motion_frame(
                        event=event,
                        dt=dt,
                        fade=ramp,
                    )
                    self._last_journey_completion = 1.0
                    return self._rate_limited_output(alpha, beta, volume, dt)

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
                    self._journey_latched_bloom = float(decision.radius_bloom)

                    # ── Latch target radius at journey start ──
                    # Prevents mid-arc knee when _is_upcoming_beat_expected
                    # flips frame-to-frame after the unhook window saturates.
                    continuation_expected_at_start = bool(
                        self._journey_linked
                        or self._is_upcoming_beat_expected(now=now, decision=decision)
                    )
                    if continuation_expected_at_start:
                        self._journey_target_radius = float(np.clip(
                            self._journey_latched_bloom, type_max_radius, 1.0
                        ))
                    else:
                        self._journey_target_radius = type_max_radius

                    # During post-silence ramp, cap target radius to the
                    # ramp-scaled bloom so the orbit can't leap to full size
                    # before the slow-ramp period has expired.
                    if self._post_silence_radius_ramp < 1.0:
                        self._journey_target_radius = float(
                            min(self._journey_target_radius, bloom_target_radius)
                        )

                    # Only overwrite _actual_radius if no mode transition is active
                    # (mode transition interpolation takes priority)
                    if not self._mode_transition_active:
                        self._actual_radius = self._journey_start_radius

                angle = float(
                    self._journey_start_angle + (self._journey_total_rotation * progress)
                )
                if progress >= 1.0 and decision.trigger_kind != "creep":
                    bpm_for_terminal = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
                    if bpm_for_terminal <= 0.0:
                        bpm_for_terminal = float(getattr(event, "bpm", 0.0) or 0.0)
                    bpm_for_terminal = float(np.clip(bpm_for_terminal if bpm_for_terminal > 0.0 else 120.0, 40.0, 240.0))
                    fallback_terminal_speed = float((2.0 * np.pi) * (bpm_for_terminal / 60.0) * self._idle_loops_per_beat)
                    terminal_speed = float(max(abs(self._angular_velocity), fallback_terminal_speed, 0.8))
                    angle = float(self._orbit_phase + (terminal_speed * dt * float(self._orbit_direction)))

                self._orbit_phase = float(angle % (2.0 * np.pi))

                phase_delta = self._wrapped_phase_delta(self._orbit_phase, self._last_phase_for_velocity)
                self._angular_velocity = float(phase_delta / max(dt, 1e-4))
                self._last_phase_for_velocity = self._orbit_phase

                # Radius path is mathematically locked to journey angle/progress.
                # - Cold start: smoothstep from park -> max during first pass
                # - Linked beat: bypass park and lock to max immediately
                # - Continuation expected: allow controlled bloom up to 1.0
                # UNLESS mode transition is active: transition interpolation overrides journey radius logic
                # ── Advance spiral-out if active ──
                if self._spiral_out_active:
                    bpm_so = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
                    if bpm_so <= 0.0:
                        bpm_so = float(getattr(event, "bpm", 0.0) or 0.0)
                    bpm_so = float(np.clip(bpm_so if bpm_so > 0.0 else 120.0, 40.0, 240.0))
                    beat_duration_s = 60.0 / bpm_so
                    spiral_duration_s = float(self._spiral_out_beats * beat_duration_s)
                    self._spiral_out_progress = float(np.clip(
                        self._spiral_out_progress + (dt / max(spiral_duration_s, 1e-3)),
                        0.0,
                        1.0,
                    ))
                    if self._spiral_out_progress >= 1.0:
                        self._spiral_out_active = False

                if not self._mode_transition_active:
                    if self._spiral_out_active:
                        # Spiral-out: orbit continuously while expanding radius
                        # over 1.5 beats with quintic ease for a slingshot feel
                        so_t = self._quintic_ease(self._spiral_out_progress)
                        radius = float(
                            self._spiral_out_start_radius
                            + ((self._spiral_out_target_radius - self._spiral_out_start_radius) * so_t)
                        )
                        # Also blend center_y during spiral-out
                        self._base_center_y = float(
                            self._spiral_out_start_center_y
                            + ((self._spiral_out_target_center_y - self._spiral_out_start_center_y) * so_t)
                        )
                    elif decision.trigger_kind == "start":
                        p = float(np.clip(progress, 0.0, 1.0))
                        radius = float(
                            self._journey_start_radius
                            + ((self._journey_fixed_radius - self._journey_start_radius) * self._quintic_ease(p))
                        )
                    else:
                        # Use journey-start-latched target radius.
                        # Evaluated once at journey start and frozen so mid-arc
                        # prediction flips never cause a radius discontinuity.
                        target_radius = self._journey_target_radius

                        if self._journey_cold_start:
                            first_pass_progress = float(np.clip(
                                (self._journey_total_rotation * progress) / (2.0 * np.pi),
                                0.0,
                                1.0,
                            ))
                            # During post-silence ramp, widen the unhook window
                            # so radius expands over the full orbit instead of the
                            # first 40%.  This prevents a sudden radius pop when
                            # the first real beat fires after silence.
                            if self._post_silence_radius_ramp < 1.0:
                                unhook_window = 0.90
                            else:
                                unhook_window = 0.40
                            unhook_t = float(np.clip(first_pass_progress / unhook_window, 0.0, 1.0))
                            # Quintic ease from *current* radius to target — eliminates
                            # knee when new trigger geometry differs from the running orbit.
                            radius_blend = self._quintic_ease(unhook_t)
                            radius = float(
                                self._journey_start_radius
                                + ((target_radius - self._journey_start_radius) * radius_blend)
                            )
                        else:
                            # Linked journey: expand from latched start radius to
                            # target over first portion of orbit for seamless hand-off.
                            first_pass_progress = float(np.clip(
                                (self._journey_total_rotation * progress) / (2.0 * np.pi),
                                0.0,
                                1.0,
                            ))
                            # During post-silence ramp, use a wider window so
                            # linked journeys also expand gradually.
                            if self._post_silence_radius_ramp < 1.0:
                                link_window = 0.90
                            else:
                                link_window = 0.40
                            link_t = float(np.clip(first_pass_progress / link_window, 0.0, 1.0))
                            link_blend = self._quintic_ease(link_t)
                            radius = float(
                                self._journey_start_radius
                                + ((target_radius - self._journey_start_radius) * link_blend)
                            )

                        if self._journey_relink_active:
                            first_pass_progress = float(np.clip(
                                (self._journey_total_rotation * progress) / (2.0 * np.pi),
                                0.0,
                                1.0,
                            ))
                            # Relink: expand from slingshot loop to full radius over ~40% of orbit
                            relink_window = 0.40
                            relink_t = float(np.clip(first_pass_progress / relink_window, 0.0, 1.0))
                            # Quintic ease for buttery-smooth slingshot release
                            relink_blend = self._quintic_ease(relink_t)
                            radius = float(
                                self._journey_relink_start_radius
                                + ((target_radius - self._journey_relink_start_radius) * relink_blend)
                            )
                            if relink_t >= 1.0:
                                self._journey_relink_active = False

                    if decision.trigger_kind == "start":
                        min_radius_bound = self._min_radius
                    elif (self._spiral_out_active
                          or self._post_silence_radius_ramp < 1.0
                          or self._journey_cold_start):
                        # Transitioning out of park / silence / mid-decay:
                        # allow the orbit to start at its current small radius
                        # and expand gradually via spiral-out / cold-start ramp.
                        # Clamping to 0.70 here would teleport the device.
                        min_radius_bound = self._min_radius
                    else:
                        min_radius_bound = 0.70
                    self._actual_radius = float(np.clip(radius, min_radius_bound, 1.0))
                
                # Whether from transition or journey, finalize radius for position calc
                radius = self._actual_radius

                base_target_center = self._base_center_target(
                    trigger_kind=decision.trigger_kind,
                    progress=progress,
                    silence_active=False,
                )
                # Center interpolation: unless mode transition or spiral-out is overriding.
                # Spiral-out has its own quintic center_y blend; let it take priority
                # so park→active center_y stays on the spiral-out curve.
                if not self._mode_transition_active and not self._spiral_out_active:
                    if progress < 1.0:
                        center_blend = self._quintic_ease(progress)
                        self._base_center_y = float(
                            ((1.0 - center_blend) * self._journey_start_total_center_y)
                            + (center_blend * base_target_center)
                        )
                    else:
                        # Gently approach target center rather than hard-snapping,
                        # which prevents a jerk when journey completes and center
                        # differs from the running orbit's center_y.
                        center_gap = abs(self._base_center_y - base_target_center)
                        if center_gap > 0.01:
                            settle_rate = 3.0  # per-second exponential approach
                            settle_t = float(1.0 - np.exp(-settle_rate * dt))
                            self._base_center_y = float(
                                self._base_center_y
                                + ((base_target_center - self._base_center_y) * settle_t)
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

                if progress >= 1.0 and decision.trigger_kind != "creep" and abs(alpha) < 0.01:
                    angle = float(angle + (0.08 * float(self._orbit_direction)))
                    self._orbit_phase = float(angle % (2.0 * np.pi))
                    alpha = float(orbit_radius * np.cos(angle))
                    beta = float(total_center_y + (orbit_radius * np.sin(angle)))

                # Apply post-silence ramp to volume
                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))

                self._last_journey_completion = progress
                if decision.trigger_kind == "start" and progress >= 1.0:
                    self._hold_start_pose_until_reactive = True

        # ── Expression layer: apply center Y wander offset only ──
        beta = float(beta + self._center_y_offset)

        # ── §8: Entry journey gating — mark entry done when first journey completes ──
        if not self._post_silence_entry_done and not decision.silence_active:
            if decision.trigger_kind == "start" and decision.journey_completion >= 1.0:
                self._post_silence_entry_done = True
            elif decision.trigger_kind != "start" and decision.trigger_kind != "creep":
                # Only allow entry journey types before unlocking; force creep otherwise
                if not self._post_silence_entry_done and self._startup_beats_seen < 8:
                    # Keep dot in entry mode by not overriding alpha/beta
                    pass

        # ── Silence-exit position crossfade ──
        # Gradually blend from the latched park position to the computed
        # trajectory position over N beats so the device swirls out smoothly
        # instead of teleporting on silence→active transition.
        if self._silence_exit_xfade_active:
            bpm_xf = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
            if bpm_xf <= 0.0:
                bpm_xf = float(getattr(event, "bpm", 0.0) or 0.0)
            bpm_xf = float(np.clip(bpm_xf if bpm_xf > 0.0 else 120.0, 40.0, 240.0))
            beat_dur_xf = 60.0 / bpm_xf
            xfade_dur_s = float(self._silence_exit_xfade_beats * beat_dur_xf)
            self._silence_exit_xfade_progress = float(np.clip(
                self._silence_exit_xfade_progress + (dt / max(xfade_dur_s, 1e-3)),
                0.0,
                1.0,
            ))
            xf_t = self._quintic_ease(self._silence_exit_xfade_progress)
            alpha = float(
                self._silence_exit_latch_alpha
                + ((alpha - self._silence_exit_latch_alpha) * xf_t)
            )
            beta = float(
                self._silence_exit_latch_beta
                + ((beta - self._silence_exit_latch_beta) * xf_t)
            )
            if self._silence_exit_xfade_progress >= 1.0:
                self._silence_exit_xfade_active = False

        # Track silence state for next-frame transition detection
        self._was_silence_active = bool(decision.silence_active)

        # ── Universal per-frame rate-limited output ──
        # All paths converge here via _rate_limited_output which applies
        # velocity EMA smoothing + per-frame-capped position limiting.
        return self._rate_limited_output(alpha, beta, volume, dt)

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
                    # §1: Choose one anchor per active segment (until silence reset).
                    if not self._anchor_phrase_locked:
                        self._anchor_sign = 1 if float(np.random.random()) > 0.5 else -1
                        self._anchor_phrase_locked = True

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
        hat_bump = self._compute_hat_bounce_offset(event=event, dt=dt)
        return float(np.clip(jitter_beta + treble_bump + hat_bump, -0.30, 0.30))

    def _compute_hat_bounce_offset(self, event: BeatEvent, dt: float) -> float:
        features = getattr(event, "beat_features", None)
        if not isinstance(features, dict):
            self._hat_bounce_amp *= float(np.exp(-6.0 * max(1e-4, dt)))
            return 0.0

        hat_conf = float(np.clip(features.get("hat_like_conf", 0.0) or 0.0, 0.0, 1.0))
        kick_conf = float(np.clip(features.get("kick_like_conf", 0.0) or 0.0, 0.0, 1.0))
        bass_dom = float(np.clip(features.get("bass_dominance", 1.0) or 1.0, 0.0, 8.0))

        hat_only = bool(hat_conf >= 0.42 and kick_conf < 0.35 and bass_dom < 1.15)
        is_hat_trigger = bool(hat_only and (getattr(event, "is_beat", False) or getattr(event, "is_syncopated", False)))
        if is_hat_trigger:
            attack = float(np.clip(0.05 + (0.12 * hat_conf), 0.04, 0.20))
            self._hat_bounce_amp = max(self._hat_bounce_amp, attack)

        self._hat_bounce_amp *= float(np.exp(-6.0 * max(1e-4, dt)))

        bounce_hz = float(7.0 + (5.0 * hat_conf))
        self._hat_bounce_phase += float((2.0 * np.pi * bounce_hz) * max(1e-4, dt))
        return float(np.clip(self._hat_bounce_amp * np.sin(self._hat_bounce_phase), -0.22, 0.22))

    @staticmethod
    def _normalize_journey_beats(interval_beats: int) -> int:
        beats = int(interval_beats)
        if beats <= 1:
            return 1
        if beats <= 2:
            return 2
        return 4

    def _compute_beat_timed_progress(self, now: float, event: BeatEvent, fallback_progress: float) -> float:
        bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
        if bpm <= 0.0:
            bpm = float(getattr(event, "bpm", 0.0) or 0.0)
        bpm = float(np.clip(bpm if bpm > 0.0 else 120.0, 40.0, 240.0))

        beats = self._normalize_journey_beats(self._journey_timing_beats)
        duration_s = float(beats) * (60.0 / bpm)
        if self._journey_start_time_mono <= 0.0 or duration_s <= 1e-6:
            return float(np.clip(fallback_progress, 0.0, 1.0))

        elapsed = float(max(0.0, now - self._journey_start_time_mono))
        return float(np.clip(elapsed / duration_s, 0.0, 1.0))

    def _apply_park_motion_frame(self, event: BeatEvent, dt: float, fade: float) -> tuple[float, float, float]:
        if not self._swirl_entering:
            self._swirl_entering = True
            self._swirl_progress = 0.0
            self._swirl_start_center_y = float(self._base_center_y)
            self._swirl_start_radius = float(max(self._actual_radius, self._park_idle_radius))

        bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
        if bpm <= 0.0:
            bpm = float(getattr(event, "bpm", 0.0) or 0.0)
        bpm = float(np.clip(bpm if bpm > 0.0 else 120.0, 40.0, 240.0))

        self._swirl_progress = float(np.clip(
            self._swirl_progress + (dt / max(self._swirl_duration_s, 1e-3)),
            0.0,
            1.0,
        ))
        swirl_t = self._quintic_ease(self._swirl_progress)

        swirl_center_y = float(
            self._swirl_start_center_y
            + ((self._baseline_center_y - self._swirl_start_center_y) * swirl_t)
        )
        swirl_radius = float(
            self._swirl_start_radius
            + ((self._park_idle_radius - self._swirl_start_radius) * swirl_t)
        )
        radius = float(np.clip(swirl_radius, self._park_idle_radius, 1.0))
        self._actual_radius = radius

        idle_angular_speed = float((2.0 * np.pi) * (bpm / 60.0) * self._idle_loops_per_beat)
        current_speed = max(abs(self._angular_velocity), idle_angular_speed)
        blended_speed = float(
            current_speed + ((idle_angular_speed - current_speed) * swirl_t)
        )
        self._orbit_phase = float((self._orbit_phase + (blended_speed * dt)) % (2.0 * np.pi))
        self._angular_velocity = float(blended_speed)
        self._last_phase_for_velocity = self._orbit_phase

        self._base_center_y = swirl_center_y

        jitter_alpha, jitter_beta = self._compute_bass_jitter_offsets(event=event, dt=dt)
        treble_bump = float(self._intelligence.compute_treble_lift(0.0)) if self._treble_lift_enabled else 0.0
        hat_bump = self._compute_hat_bounce_offset(event=event, dt=dt)
        self._reactive_bounce_y = float(np.clip(jitter_beta + treble_bump + hat_bump, -0.30, 0.30))

        total_center_y = float(self._base_center_y + self._reactive_bounce_y)
        orbit_radius = float(min(radius, self._radius_cap_for_center(total_center_y)))

        angle = float(self._orbit_phase)
        alpha = float(orbit_radius * np.cos(angle)) + float(jitter_alpha * 0.3)
        beta = float(total_center_y + (orbit_radius * np.sin(angle)))
        volume = float(np.clip(self.get_volume() * float(np.clip(fade, 0.0, 1.0)), 0.0, 1.0))
        return alpha, beta, volume

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
        _ = interval_beats
        max_turns = 1.0
        turns = 1.0

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
