"""
bREadbeats - Stroke Mapper (Decision-Only Adapter)

Thin runtime adapter that delegates signal intelligence to beat_intelligence.
Legacy drawing/trajectory generation has been removed.
"""

from __future__ import annotations

import time
from collections import deque
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
    beta: float = 0.70
    last_time: float = 0.0


class StrokeMapper:
    """Decision-based stroke mapper that consumes BeatIntelligence."""

    def __init__(
        self,
        config: Config,
        get_volume: Optional[Callable[[], float]] = None,
        audio_engine=None,
    ):
        self.config = config
        self.get_volume = get_volume if get_volume is not None else (lambda: 1.0)
        self.audio_engine = audio_engine

        self.state = StrokeState()
        self._park_y = 0.20
        self._baseline_center_y = 0.20
        self._base_center_y = self._baseline_center_y
        self._journey_start_total_center_y = self._baseline_center_y

        self._orbit_phase = 0.0
        self._last_trigger_kind = "creep"
        self._park_radius = 0.70
        self._journey_start_radius = self._park_radius
        self._journey_latched_bloom = 0.70     # radius_bloom frozen at journey start
        self._journey_park_radius = self._park_radius
        self._journey_max_radius = 1.0
        self._journey_start_angle = float(np.pi / 2.0)
        self._journey_start_alpha = self.state.alpha
        self._journey_start_beta = self.state.beta
        self._journey_total_rotation = float(2.0 * np.pi)
        self._last_journey_completion = 1.0
        self._actual_radius = self._park_radius
        self._journey_target_radius = self._park_radius  # latched at journey start; never re-evaluated mid-arc
        self._orbit_phase_initialized = False  # True once orbit_phase has been actively tracked
        self._journey_linked = False
        self._idle_loops_per_beat = 0.125

        # ── Funscript idle-fill loop data (extracted from NoodleDude Megamix at 11:00.545) ──
        # 45 samples, ~33ms intervals.  Ping-pong looped.
        # Rescaled to [0.15, 0.75] so Y output maps to -0.5 → +0.7.
        self._idle_loop_alpha: tuple[float, ...] = (
            0.15, 0.24, 0.33, 0.27, 0.20, 0.30, 0.39, 0.33, 0.27, 0.36,
            0.45, 0.39, 0.32, 0.41, 0.51, 0.45, 0.39, 0.48, 0.57, 0.51,
            0.45, 0.54, 0.63, 0.57, 0.51, 0.60, 0.69, 0.63, 0.57, 0.66,
            0.75, 0.66, 0.57, 0.63, 0.69, 0.60, 0.50, 0.57, 0.63, 0.54,
            0.45, 0.51, 0.57, 0.48, 0.39,
        )
        self._idle_loop_beta: tuple[float, ...] = (
            0.53, 0.56, 0.58, 0.54, 0.51, 0.53, 0.57, 0.59, 0.56, 0.52,
            0.55, 0.58, 0.61, 0.57, 0.53, 0.56, 0.59, 0.62, 0.58, 0.54,
            0.52, 0.56, 0.60, 0.63, 0.59, 0.56, 0.53, 0.57, 0.61, 0.64,
            0.61, 0.57, 0.53, 0.56, 0.59, 0.62, 0.58, 0.54, 0.52, 0.56,
            0.53, 0.51, 0.53, 0.56, 0.53,
        )
        self._idle_loop_phase: float = 0.0       # continuous sample counter
        self._idle_loop_rate_hz: float = 30.0    # samples per second

        # ── Fill minimum dwell: stay in fill for at least 1 measure ──
        self._fill_enter_time: float = 0.0      # monotonic time fill mode started
        self._fill_min_beats: int = 4            # minimum beats before exit allowed

        # ── Fill-exit transition: nested decay from fill → orbit ──
        self._fill_exit_active: bool = False
        self._fill_exit_elapsed: float = 0.0
        self._fill_exit_duration_s: float = 0.5  # recomputed from BPM at transition start
        # Precompute fill X bias for centering
        _fc_wobble = float(np.mean(self._idle_loop_beta))   # wobble data → X axis
        self._fill_x_bias: float = float(_fc_wobble * 2.0 - 1.0)  # subtract to center fill on X=0
        # Nested decay state: micro-orbit spinning around a gliding virtual center
        self._fill_exit_vc_alpha: float = 0.0    # virtual center alpha (latched at exit)
        self._fill_exit_vc_beta: float = 0.0     # virtual center beta (latched at exit)
        self._fill_exit_target_alpha: float = 0.0  # orbit destination alpha
        self._fill_exit_target_beta: float = 0.0   # orbit destination beta
        self._fill_exit_micro_radius: float = 0.05  # initial micro-orbit radius
        self._fill_exit_micro_freq: float = 30.0    # initial micro-orbit frequency (Hz)
        self._fill_exit_micro_phase: float = 0.0    # micro-orbit phase accumulator
        self._fill_exit_creep_streak: int = 0        # consecutive creep frames during exit
        self._fill_exit_creep_cancel_threshold: int = 3  # need 3 consecutive creep frames to cancel

        self._last_gate_fail = ""  # diagnostic: which gate blocked last beat-family event
        self._last_decision = None      # latest BeatDecision (for keyboard teacher snapshot)

        # ── Fixed anchor state (§1) ──
        self._anchor_sign: int = 1               # +1 = +Y anchor, -1 = -Y anchor
        self._anchor_swing_deg: float = 10.0     # ±10° swing around y-axis
        self._anchor_phrase_locked: bool = False  # True once chosen for current phrase

        # ── Expression layer state ──
        self._orbit_direction: int = 1           # 1=default, -1=reversed
        self._last_direction_change_time: float = 0.0
        self._center_y_offset: float = 0.0
        self._center_wander_phase: float = 0.0
        self._energy_history: deque = deque(maxlen=300)  # ~5s at 60fps
        self._session_energy_ema: float = 0.5

        # ── Intensity timer ramp (session-level escalation) ──
        self._intensity_ramp_start_time: float = 0.0
        self._intensity_ramp_started: bool = False
        self._intensity_ramp_mult: float = 1.0
        self._intensity_ramp_floor: float = 0.25
        self._intensity_ramp_affect_size: bool = True

        # ── Rate-limiter velocity state (for smoothing across ALL paths) ──
        self._smoothed_da: float = 0.0
        self._smoothed_db: float = 0.0
        self._rate_limiter_clipped: bool = False  # True when rate limiter clamped last frame

        self._intelligence = BeatIntelligence(config=self.config, audio_engine=self.audio_engine, park_y=self._park_y)

        self._learning_enabled = bool(getattr(self.config.beat, "teaching_learning_enabled", False))
        self._learning_use_fitted_rules = bool(getattr(self.config.beat, "teaching_use_fitted_rules", False))
        self._learning_strength = float(getattr(self.config.beat, "teaching_learning_strength", 0.0) or 0.0)
        self._learning_min_confidence = float(getattr(self.config.beat, "teaching_min_confidence", 0.0) or 0.0)
        self._learning_no_motion_bias = float(getattr(self.config.beat, "teaching_no_motion_bias", 1.0) or 1.0)
        self._learning_rule_fit_path = str(getattr(self.config.beat, "teaching_rule_fit_path", "") or "")
        # Push initial learning config to intelligence
        self._sync_learning_to_intelligence()

    def configure_geometry_rest_state(self) -> None:
        self._park_y = 0.20
        self._intelligence.set_park_y(self._park_y)

    def configure_learning(
        self,
        *,
        enabled: bool,
        use_fitted_rules: bool,
        learning_strength: float,
        min_confidence: float,
        no_motion_bias: float,
        rule_fit_path: str,
    ) -> None:
        self._learning_enabled = bool(enabled)
        self._learning_use_fitted_rules = bool(use_fitted_rules)
        self._learning_strength = float(learning_strength)
        self._learning_min_confidence = float(min_confidence)
        self._learning_no_motion_bias = float(no_motion_bias)
        self._learning_rule_fit_path = str(rule_fit_path or "")

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
          1. Adaptive velocity EMA – fast tracking for normal orbital
             direction changes, quintic-scale slow ramp for large target
             jumps (ensures transitions ease over ~2 beats).
          2. Per-second rate cap – proportional to dt (5.0 units/s).
          3. Per-frame hard cap – absolute ceiling (0.10) prevents dt-spike
             frames from allowing oversized jumps.

        Rate limiting uses radial (magnitude) clamping so that the speed
        cap is isotropic.  Per-axis clamping would allow √2× faster
        diagonal movement, distorting circular orbits into squares.
        """
        # Beat-journey orbits require high velocity: at 180 BPM, r=1.0,
        # the orbital speed is 2π×3 ≈ 18.85 units/s.  Previous value
        # (3.5) forced the output to a tiny circle because the rate
        # limiter couldn't keep up with beat-speed angular velocity.
        max_delta_per_s = 20.0
        max_delta_per_frame = 0.50  # safety ceiling; 180 BPM r=1.0 @ 40fps ≈ 0.47
        max_delta = float(min(max_delta_per_s * max(dt, 1e-4), max_delta_per_frame))

        prev_a = float(self.state.alpha)
        prev_b = float(self.state.beta)

        raw_da = float(alpha - prev_a)
        raw_db = float(beta - prev_b)

        # Adaptive velocity EMA: two-tier smoothing keeps orbital
        # direction changes responsive while quintic-scaling large
        # target jumps over ~2 beats.
        #   cold start (smoothed ≈ 0):  factor=0.20 → gentle initial motion
        #   small delta_diff (<0.02):   factor=0.30 → orbital direction tracking
        #   large delta_diff (>0.02):   factor=0.02 → ~110 frames (~4 beats @ 130 BPM)
        delta_diff = float(np.hypot(
            raw_da - self._smoothed_da,
            raw_db - self._smoothed_db,
        ))
        smoothed_mag = float(np.hypot(self._smoothed_da, self._smoothed_db))
        if smoothed_mag < 0.001:
            smooth_factor = 0.35  # gentle ramp from rest
        elif delta_diff < 0.06:
            smooth_factor = 0.55  # orbital / fill direction tracking — high for round arcs
        else:
            smooth_factor = 0.08  # slow for large target jumps (~2 beats)
        da = float(self._smoothed_da + smooth_factor * (raw_da - self._smoothed_da))
        db = float(self._smoothed_db + smooth_factor * (raw_db - self._smoothed_db))

        # Radial position rate limiter – clamp the displacement *vector*
        # magnitude so all directions are treated equally (preserves
        # circular orbits instead of squashing them into diamonds).
        delta_mag = float(np.hypot(da, db))
        if delta_mag > max_delta:
            scale = max_delta / delta_mag
            da = float(da * scale)
            db = float(db * scale)
            self._rate_limiter_clipped = True
        else:
            self._rate_limiter_clipped = False

        # Store clamped value so EMA tracks actual movement, not desired
        self._smoothed_da = da
        self._smoothed_db = db

        alpha = float(prev_a + da)
        beta = float(prev_b + db)

        # Unit-circle clamp: ensure alpha²+beta² ≤ 1.0 so the dot
        # never draws outside the boundary circle on the display.
        mag_sq = alpha * alpha + beta * beta
        if mag_sq > 1.0:
            inv_mag = 1.0 / float(np.sqrt(mag_sq))
            alpha = float(alpha * inv_mag)
            beta = float(beta * inv_mag)

        alpha = float(np.clip(alpha, -1.0, 1.0))
        beta = float(np.clip(beta, -1.0, 1.0))

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

        prev_trigger_kind = self._last_trigger_kind
        self._last_trigger_kind = decision.trigger_kind
        self._last_gate_fail = str(getattr(decision, "gate_fail", "") or "")
        self._last_decision = decision

        # ── Resync orbit to output when rate limiter clipped last frame ──
        # Prevents the parametric orbit from diverging from the actual
        # output position, which would cause straight-line chasing.
        if self._rate_limiter_clipped:
            self._resync_orbit_to_output()

        if hitch_soft_reset:
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
            # ── Silent-still-park: hold position, fade volume, reset momentum ──
            self._anchor_phrase_locked = False
            self._fill_exit_active = False  # cancel fill exit on silence

            fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
            alpha = float(self.state.alpha)
            beta = float(self.state.beta)
            volume = float(np.clip(self.get_volume() * fade, 0.0, 1.0))
            self._last_journey_completion = 1.0
        else:
            progress = float(np.clip(decision.journey_completion, 0.0, 1.0))

            # ── Fill-exit nested decay: micro-orbit around gliding virtual center ──
            if self._fill_exit_active:
                if decision.trigger_kind == "creep":
                    self._fill_exit_creep_streak += 1
                    if self._fill_exit_creep_streak >= self._fill_exit_creep_cancel_threshold:
                        # Sustained creep — cancel exit and resume fill
                        self._fill_exit_active = False
                        self._fill_exit_creep_streak = 0
                else:
                    self._fill_exit_creep_streak = 0

                if self._fill_exit_active:
                    self._fill_exit_elapsed += dt
                    raw_t = float(np.clip(
                        self._fill_exit_elapsed / max(self._fill_exit_duration_s, 0.01),
                        0.0, 1.0,
                    ))
                    ease_t = self._quintic_ease(raw_t)
                    if raw_t >= 1.0:
                        # Nested decay complete — initialize orbit and return
                        # directly.  Do NOT fall through to orbit code because
                        # _journey_* state is stale from before fill started;
                        # using it overwrites our carefully-set phase/radius
                        # with wrong values and produces small circles.
                        self._fill_exit_active = False
                        center_y = float(self._base_center_y)
                        inf_angle, inf_radius = self._infer_orbit_from_position(
                            alpha=self._fill_exit_target_alpha,
                            beta=self._fill_exit_target_beta,
                            center_y=center_y,
                        )
                        self._orbit_phase = float(inf_angle % (2.0 * np.pi))
                        self._actual_radius = float(max(inf_radius, self._park_radius))
                        self._orbit_phase_initialized = True

                        # Re-initialize all journey state so the NEXT frame's
                        # orbit code has correct values instead of stale ones.
                        self._journey_start_angle = float(self._orbit_phase)
                        self._journey_start_radius = float(self._actual_radius)
                        self._journey_start_alpha = float(self._fill_exit_target_alpha)
                        self._journey_start_beta = float(self._fill_exit_target_beta)
                        self._journey_start_total_center_y = float(center_y)
                        self._journey_linked = False
                        self._journey_target_radius = float(self._actual_radius)
                        self._journey_total_rotation = float(2.0 * np.pi)
                        self._last_journey_completion = progress

                        # Prime rate-limiter EMA with expected orbital velocity
                        # so the output doesn't throttle on the first frames.
                        _bpm_rl = 120.0
                        if self.audio_engine is not None:
                            _met_rl = float(getattr(self.audio_engine, "_metronome_bpm", 0.0) or 0.0)
                            if _met_rl > 0:
                                _bpm_rl = _met_rl
                        _bpm_rl = float(np.clip(_bpm_rl, 40.0, 240.0))
                        _omega = float(2.0 * np.pi * (_bpm_rl / 60.0) * self._idle_loops_per_beat)
                        _dt_rl = max(dt, 1e-4)
                        _dir = float(self._orbit_direction)
                        self._smoothed_da = float(
                            -self._actual_radius * np.sin(self._orbit_phase) * _omega * _dt_rl * _dir
                        )
                        self._smoothed_db = float(
                            self._actual_radius * np.cos(self._orbit_phase) * _omega * _dt_rl * _dir
                        )

                        # Return target position directly — bypasses stale orbit code
                        alpha = float(self._fill_exit_target_alpha)
                        beta = float(self._fill_exit_target_beta)
                        ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                        volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))
                        self.state.alpha = alpha
                        self.state.beta = beta
                        return TCodeCommand(alpha=alpha, beta=beta, duration_ms=25, volume=volume)
                    else:
                        # Glide virtual center toward orbit target
                        vc_a = float(self._fill_exit_vc_alpha
                                     + (self._fill_exit_target_alpha - self._fill_exit_vc_alpha) * ease_t)
                        vc_b = float(self._fill_exit_vc_beta
                                     + (self._fill_exit_target_beta - self._fill_exit_vc_beta) * ease_t)
                        # Decay micro-orbit radius and frequency
                        micro_r = float(self._fill_exit_micro_radius * (1.0 - ease_t))
                        micro_f = float(self._fill_exit_micro_freq * (1.0 - ease_t))
                        # Advance micro-orbit phase
                        self._fill_exit_micro_phase += micro_f * dt * 2.0 * float(np.pi)
                        # Compute micro-orbit offset
                        alpha = float(vc_a + micro_r * np.cos(self._fill_exit_micro_phase))
                        beta = float(vc_b + micro_r * np.sin(self._fill_exit_micro_phase))
                        ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                        volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))
                        self._last_journey_completion = 1.0
                        self.state.alpha = alpha
                        self.state.beta = beta
                        return TCodeCommand(alpha=alpha, beta=beta, duration_ms=25, volume=volume)

            # ── Funscript fill: plays when trigger_kind == "creep" ──
            # The baked loop IS the motion — no orbit, no rate limiter,
            # no modifiers.  Raw sample values go straight to output.
            # ── Fill minimum dwell: enforce 1 measure before exit ──
            # If we just left fill but haven't stayed long enough,
            # override trigger back to creep and keep playing fill.
            _in_fill_dwell = False
            if prev_trigger_kind == "creep" and decision.trigger_kind != "creep" and not self._fill_exit_active:
                _bpm_dwell = 120.0
                if self.audio_engine is not None:
                    _met_dwell = float(getattr(self.audio_engine, "_metronome_bpm", 0.0) or 0.0)
                    if _met_dwell > 0:
                        _bpm_dwell = _met_dwell
                _bpm_dwell = float(np.clip(_bpm_dwell, 40.0, 240.0))
                _measure_s = (60.0 / _bpm_dwell) * self._fill_min_beats
                if (now - self._fill_enter_time) < _measure_s:
                    _in_fill_dwell = True

            if decision.trigger_kind == "creep" or _in_fill_dwell:
                if prev_trigger_kind != "creep" and not _in_fill_dwell:
                    self._fill_enter_time = now  # record when fill started
                if _in_fill_dwell:
                    # Override last_trigger so next frame still sees
                    # prev_trigger_kind == "creep" for dwell check
                    self._last_trigger_kind = "creep"
                self._fill_exit_active = False  # cancel any pending exit
                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                alpha, beta, volume = self._apply_park_motion_frame(dt=dt, fade=ramp)
                self._last_journey_completion = 1.0
                # Direct output — bypass rate limiter so the fill pattern
                # is not suppressed by the orbital velocity EMA.
                self.state.alpha = alpha
                self.state.beta = beta
                return TCodeCommand(alpha=alpha, beta=beta, duration_ms=25, volume=volume)

            # ── Detect fill → orbit transition: start nested decay ──
            if prev_trigger_kind == "creep" and not self._fill_exit_active:
                self._fill_exit_active = True
                self._fill_exit_elapsed = 0.0
                self._fill_exit_creep_streak = 0
                # Duration = 4 beats (clamped to reasonable BPM range)
                _bpm = 120.0
                if self.audio_engine is not None:
                    _met = float(getattr(self.audio_engine, "_metronome_bpm", 0.0) or 0.0)
                    if _met > 0:
                        _bpm = _met
                _bpm = float(np.clip(_bpm, 40.0, 240.0))
                self._fill_exit_duration_s = (60.0 / _bpm) * 4.0  # 4 beats
                # Latch virtual center to dot's current position
                self._fill_exit_vc_alpha = float(self.state.alpha)
                self._fill_exit_vc_beta = float(self.state.beta)
                self._fill_exit_micro_phase = 0.0
                self._fill_exit_micro_radius = 0.05
                self._fill_exit_micro_freq = 30.0
                # Compute orbit target: where the first beat journey wants to be.
                # Use the beat-journey minimum radius (0.80) so the exit
                # lands on a full-size orbit, not a tiny park-radius circle.
                _center_y = float(self._base_center_y)
                _target_radius = max(float(decision.radius_bloom), 0.80)
                # Use orbit phase if initialized, otherwise infer from current position
                if self._orbit_phase_initialized:
                    _target_angle = self._orbit_phase
                else:
                    _target_angle, _ = self._infer_orbit_from_position(
                        alpha=self.state.alpha, beta=self.state.beta, center_y=_center_y,
                    )
                self._fill_exit_target_alpha = float(_target_radius * np.cos(_target_angle))
                self._fill_exit_target_beta = float(_center_y + _target_radius * np.sin(_target_angle))
                # First exit frame: full micro-orbit, no decay yet
                alpha = float(self._fill_exit_vc_alpha
                              + self._fill_exit_micro_radius * np.cos(self._fill_exit_micro_phase))
                beta = float(self._fill_exit_vc_beta
                             + self._fill_exit_micro_radius * np.sin(self._fill_exit_micro_phase))
                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))
                self._last_journey_completion = 1.0
                self.state.alpha = alpha
                self.state.beta = beta
                return TCodeCommand(alpha=alpha, beta=beta, duration_ms=25, volume=volume)

            started_new_journey = bool(progress <= 1e-9 and self._last_journey_completion > 1e-9)
            if started_new_journey:
                prior_completion = float(self._last_journey_completion)
                self._journey_linked = bool(prior_completion < 0.999)

                self._journey_start_total_center_y = float(self._base_center_y)

                # Latch geometry at journey start so mid-journey trigger
                # reclassification cannot reshape a running arc.
                geom = self.config.stroke.orbit_geometry.get(decision.trigger_kind, {
                    "center_y": self._baseline_center_y, "park_radius": 0.70, "max_radius": 1.0
                })
                self._journey_park_radius = float(geom["park_radius"])
                # Expand max_radius toward 1.0 based on energy fullness:
                # quiet music stays at configured max (0.90), full music → 1.0
                base_max = float(geom["max_radius"])
                fullness = float(np.clip(decision.energy_fullness, 0.0, 1.0))
                # Smooth expansion: only starts opening above 0.55 fullness
                expand_t = float(np.clip((fullness - 0.55) / 0.45, 0.0, 1.0))
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

                self._journey_start_alpha = float(np.clip(self.state.alpha, -1.0, 1.0))
                self._journey_start_beta = float(np.clip(self.state.beta, -1.0, 1.0))
                if self._orbit_phase_initialized:
                    # Re-sync orbit from actual output to prevent
                    # straight-line chase when rate limiter has lagged
                    self._resync_orbit_to_output()
                    self._journey_start_angle = float(self._orbit_phase)
                    self._journey_start_radius = float(np.clip(
                        self._actual_radius, self._journey_park_radius, 1.0
                    ))
                else:
                    # First journey: infer from externally-set position
                    inherited_angle, inherited_radius = self._infer_orbit_from_position(
                        alpha=self._journey_start_alpha,
                        beta=self._journey_start_beta,
                        center_y=self._journey_start_total_center_y,
                    )
                    self._journey_start_angle = inherited_angle
                    self._journey_start_radius = float(np.clip(
                        inherited_radius, self._journey_park_radius, 1.0
                    ))
                    self._orbit_phase = float(inherited_angle % (2.0 * np.pi))
                    self._orbit_phase_initialized = True

                self._journey_total_rotation = self._compute_landing_rotation(
                    start_angle=self._journey_start_angle,
                    interval_beats=decision.interval_beats,
                )
            # Use latched geometry while a journey is in-flight.
            # Only refresh from live trigger kind when fully parked.
            if (progress < 1.0) or (self._last_journey_completion < 1.0):
                type_park_radius = float(self._journey_park_radius)
                type_max_radius = float(self._journey_max_radius)
            else:
                geom = self.config.stroke.orbit_geometry.get(decision.trigger_kind, {
                    "center_y": self._baseline_center_y, "park_radius": 0.70, "max_radius": 1.0
                })
                type_park_radius = float(geom["park_radius"])
                type_max_radius = float(geom["max_radius"])

            if started_new_journey:
                self._journey_latched_bloom = float(decision.radius_bloom)

                # ── Latch target radius at journey start ──
                # Prevents mid-arc knee when _is_upcoming_beat_expected
                # flips frame-to-frame after the unhook window saturates.
                continuation_expected_at_start = bool(
                    self._journey_linked
                    or self._is_upcoming_beat_expected(now=now, decision=decision)
                )
                if continuation_expected_at_start:
                    # Cap at type_max_radius — never expand beyond configured max.
                    # Previous code clipped to [max_radius, 1.0] which allowed
                    # the orbit to overshoot the boundary circle.
                    self._journey_target_radius = float(min(
                        self._journey_latched_bloom, type_max_radius
                    ))
                else:
                    self._journey_target_radius = type_max_radius

                self._actual_radius = self._journey_start_radius

            angle = float(
                self._journey_start_angle + (self._journey_total_rotation * progress)
            )
            if progress >= 1.0:
                bpm_for_terminal = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
                if bpm_for_terminal <= 0.0:
                    bpm_for_terminal = float(getattr(event, "bpm", 0.0) or 0.0)
                bpm_for_terminal = float(np.clip(bpm_for_terminal if bpm_for_terminal > 0.0 else 120.0, 40.0, 240.0))
                fallback_terminal_speed = float((2.0 * np.pi) * (bpm_for_terminal / 60.0) * self._idle_loops_per_beat)
                # Use the BPM-derived idle orbit speed — NOT the journey's
                # angular velocity.  Journey velocity can be 10-25 rad/s
                # (one turn per beat), which overwhelms the rate limiter
                # and forces the output to trace a tiny circle (~0.40 r)
                # instead of the intended 0.80+ radius orbit.
                terminal_speed = float(max(fallback_terminal_speed, 0.8))
                angle = float(self._orbit_phase + (terminal_speed * dt * float(self._orbit_direction)))

            self._orbit_phase = float(angle % (2.0 * np.pi))

            # Radius path is mathematically locked to journey angle/progress.
            # - Cold start: smoothstep from park -> max during first pass
            # - Linked beat: bypass park and lock to max immediately
            # - Continuation expected: allow controlled bloom up to 1.0
            # Use journey-start-latched target radius.
            # Evaluated once at journey start and frozen so mid-arc
            # prediction flips never cause a radius discontinuity.
            target_radius = self._journey_target_radius

            # Quintic ease from start radius to target over first 40% of orbit
            first_pass_progress = float(np.clip(
                (self._journey_total_rotation * progress) / (2.0 * np.pi),
                0.0,
                1.0,
            ))
            blend_window = 0.40
            blend_t = float(np.clip(first_pass_progress / blend_window, 0.0, 1.0))
            radius_blend = self._quintic_ease(blend_t)
            radius = float(
                self._journey_start_radius
                + ((target_radius - self._journey_start_radius) * radius_blend)
            )

            min_radius_bound = 0.80  # beat-journey minimum radius
            self._actual_radius = float(np.clip(radius, min_radius_bound, 1.0))
            
            # Whether from transition or journey, finalize radius for position calc
            radius = self._actual_radius

            base_target_center = self._base_center_target(
                trigger_kind=decision.trigger_kind,
                progress=progress,
                silence_active=False,
            )
            # Center interpolation toward target.
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

            total_center_y = float(self._base_center_y)
            orbit_radius = float(min(radius, self._radius_cap_for_center(total_center_y)))

            alpha = float(orbit_radius * np.cos(angle))
            beta = float(total_center_y + (orbit_radius * np.sin(angle)))

            if progress >= 1.0 and abs(alpha) < 0.01:
                angle = float(angle + (0.08 * float(self._orbit_direction)))
                self._orbit_phase = float(angle % (2.0 * np.pi))
                alpha = float(orbit_radius * np.cos(angle))
                beta = float(total_center_y + (orbit_radius * np.sin(angle)))

            # Apply post-silence ramp to volume
            ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
            volume = float(np.clip(self.get_volume() * ramp, 0.0, 1.0))

            self._last_journey_completion = progress

        # ── Expression layer: apply center Y wander offset only ──
        beta = float(beta + self._center_y_offset)

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
            max_y = float(getattr(self.config.stroke, 'center_wander_max_x', 0.20) or 0.20)
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
        elif decision.silence_active:
            # Gently decay wander toward center during silence
            decay = float(max(0.0, 1.0 - 2.0 * dt))
            self._center_y_offset *= decay

        # ── §1: Anchor phrase management (direction change → new anchor) ──
        if getattr(self.config.stroke, 'direction_change_enabled', True) and not decision.silence_active:
            interval_s = float(getattr(self.config.stroke, 'direction_change_interval_s', 15.0) or 15.0)
            drop_needed = float(getattr(self.config.stroke, 'direction_change_energy_drop', 0.35) or 0.35)

            if (now - self._last_direction_change_time > interval_s
                    and len(self._energy_history) >= 75
                    and self._actual_radius > 0.9):
                recent = list(self._energy_history)
                recent_mean = float(np.mean(recent[-15:]))   # last ~0.25s
                prior_mean = float(np.mean(recent[-75:-15]))  # prior ~1.0s
                # Trigger on significant energy transition (either direction)
                if prior_mean > 0.08 and abs(prior_mean - recent_mean) / max(prior_mean, 0.08) > drop_needed:
                    self._orbit_direction *= -1
                    self._last_direction_change_time = now
                    # Negate velocity EMAs so the rate limiter tracks the
                    # reversed direction immediately (zeroing causes J-hooks
                    # because delta_diff is large → slow 0.08 ramp).
                    self._smoothed_da = -self._smoothed_da
                    self._smoothed_db = -self._smoothed_db
                    # §1: Choose one anchor per active segment (until silence reset).
                    if not self._anchor_phrase_locked:
                        self._anchor_sign = 1 if float(np.random.random()) > 0.5 else -1
                        self._anchor_phrase_locked = True

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

    def _resync_orbit_to_output(self) -> None:
        """Re-sync parametric orbit (phase, radius) to actual output position.

        During transitions the rate limiter may cap per-frame movement,
        so _actual_radius / _orbit_phase can diverge from state.alpha/beta.
        Latching a transition start from stale parametric state causes a
        large target-output gap that the rate limiter chases in a
        straight line.  Re-inferring orbit parameters from the real
        output position keeps subsequent arcs circular and smooth.
        """
        if not self._orbit_phase_initialized:
            return
        # Strip expression wander (added post-orbit) and use orbit center
        center_y = float(self._base_center_y)
        effective_beta = float(self.state.beta - self._center_y_offset)
        inferred_r = float(np.hypot(float(self.state.alpha), effective_beta - center_y))
        if inferred_r > self._actual_radius + 0.05:
            # Actual output implies a larger orbit than parametric state;
            # re-infer phase via atan2 and adopt the larger radius so
            # the next transition starts from the real device position.
            dy = float(effective_beta - center_y)
            inferred_phase = float(np.arctan2(dy, float(self.state.alpha)))
            self._orbit_phase = float(inferred_phase % (2.0 * np.pi))
            self._actual_radius = float(inferred_r)

    def _apply_park_motion_frame(self, dt: float, fade: float) -> tuple[float, float, float]:
        """Funscript idle-fill: raw baked loop, no modifiers.

        Plays the 45-sample ping-pong pattern at native amplitude.
        """
        loop_alpha, loop_beta = self._sample_idle_loop(dt=dt)

        # Map normalized [0,1] loop data to [-1, 1] output range
        # Sweep (big range) → -Y axis (beta), wobble → X axis (alpha)
        # Subtract X bias so fill pattern is centered on X=0, then scale to 50%
        alpha = float((loop_beta * 2.0 - 1.0 - self._fill_x_bias) * 0.5)
        beta = float(-(loop_alpha * 2.0 - 1.0))

        volume = float(np.clip(self.get_volume() * float(np.clip(fade, 0.0, 1.0)), 0.0, 1.0))
        return alpha, beta, volume

    def _sample_idle_loop(self, dt: float) -> tuple[float, float]:
        """Advance and sample the ping-pong funscript idle loop.

        Returns (alpha, beta) in normalized [0, 1] space.
        The loop plays forward through all samples, then reverses,
        creating a seamless ping-pong cycle.
        """
        n = len(self._idle_loop_alpha)  # 45 samples
        # Ping-pong cycle length: forward (0→n-1) + reverse (n-1→0) = 2*(n-1) steps
        cycle_len = float(2 * (n - 1))  # 88

        # Advance phase by dt * playback_rate (samples per second)
        self._idle_loop_phase += dt * self._idle_loop_rate_hz
        self._idle_loop_phase = float(self._idle_loop_phase % cycle_len)

        # Convert phase to sample index with ping-pong
        phase = self._idle_loop_phase
        if phase < float(n - 1):
            # Forward leg
            idx_f = phase
        else:
            # Reverse leg: mirror around (n-1)
            idx_f = cycle_len - phase

        # Linearly interpolate between adjacent samples
        idx_lo = int(idx_f)
        idx_hi = min(idx_lo + 1, n - 1)
        frac = float(idx_f - idx_lo)
        alpha = float(self._idle_loop_alpha[idx_lo] + (self._idle_loop_alpha[idx_hi] - self._idle_loop_alpha[idx_lo]) * frac)
        beta = float(self._idle_loop_beta[idx_lo] + (self._idle_loop_beta[idx_hi] - self._idle_loop_beta[idx_lo]) * frac)
        return alpha, beta

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
    def _infer_orbit_from_position(alpha: float, beta: float, center_y: float) -> tuple[float, float]:
        """Infer phase/radius using current orientation: alpha=r*cos(theta), beta=center+r*sin(theta)."""
        dy = float(beta - center_y)
        radius = float(np.hypot(alpha, dy))
        angle = float(np.arctan2(dy, alpha))
        return angle, radius

    def _radius_cap_for_center(self, center_y: float) -> float:
        """Maximum radius that keeps orbit inside normalized [-1, 1] bounds in both axes."""
        effective_center_y = float(center_y + self._center_y_offset)
        return float(max(0.0, min(1.0 - effective_center_y, 1.0 + effective_center_y)))

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
