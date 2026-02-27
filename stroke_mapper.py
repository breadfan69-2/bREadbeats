"""
bREadbeats - Stroke Mapper (Decision-Only Adapter)

Thin runtime adapter that delegates signal intelligence to beat_intelligence.
Legacy drawing/trajectory generation has been removed.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from audio_engine import BeatEvent
from beat_intelligence import BeatDecision, BeatIntelligence
from config import Config
from geometry_utils import (
    exponential_approach,
    infer_orbit,
    nearest_anchor_crossing,
    orbit_point,
    quintic_ease,
    radius_cap_for_center,
)
from network_engine import TCodeCommand
from ramp_engine import RampEngine


@dataclass
class StrokeState:
    alpha: float = 0.0
    beta: float = 0.20       # start at park position (was 0.70 → caused visible arc on silent startup)
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
        self._journey_start_angle = float(math.pi / 2.0)
        self._journey_start_alpha = self.state.alpha
        self._journey_start_beta = self.state.beta
        self._journey_total_rotation = float(2.0 * math.pi)
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

        # ── Fill rotation phase (helicopter / spring-coil visual) ──
        self._fill_rot_phase: float = 0.0        # rotation angle accumulator (radians)
        self._fill_rot_radius: float = 0.06      # orbit radius in [-1,1] space (diameter 0.12)

        # ── Bass-frequency orbit modulation ──
        # Dominant bass freq controls fill orbit size:
        #  200 Hz  → 1.0× (highest bass = base orbit, no expansion)
        #   50 Hz  → 2.0× (lowest bass = maximum expansion)
        self._fill_bass_freq_history: deque = deque(maxlen=4)

        # ── Full-spectrum dominant freq → Y-axis position ──
        # Maps dominant freq (80–8000 Hz log scale) to center_y in [-0.5, +0.5]
        #   Low freq  → +0.5 (high Y position)
        #   High freq → -0.5 (low Y position)
        self._fill_dom_freq_history: deque = deque(maxlen=6)

        # ── Hi-hat / snare → downward Y impulse ──
        # When the 'high' (2–16 kHz) or 'mid' (500–2 kHz) z-score fires,
        # inject a quick downward kick on the Y axis that decays rapidly.
        self._fill_hh_impulse: float = 0.0        # current impulse magnitude (positive = downward)
        self._FILL_HH_IMPULSE_SIZE: float = 0.18  # initial kick magnitude in [-1,1] space
        self._FILL_HH_IMPULSE_DECAY: float = 8.0  # exponential decay rate (per second)

        # ── Fill minimum dwell: stay in fill for at least 1 measure ──
        self._fill_enter_time: float = 0.0      # monotonic time fill mode started
        self._fill_min_beats: int = 4            # minimum beats before exit allowed

        # ── Fill-exit transition: center glides to orbit, rotation decays ──
        self._fill_exit_active: bool = False
        self._fill_exit_elapsed: float = 0.0
        self._fill_exit_duration_s: float = 0.5  # recomputed from BPM at transition start
        # Precompute fill X bias for centering
        _fc_wobble = float(np.mean(self._idle_loop_beta))   # wobble data → X axis
        self._fill_x_bias: float = float(_fc_wobble * 2.0 - 1.0)  # subtract to center fill on X=0
        # Fill-exit state: rotation around a quintic-easing center
        self._fill_exit_vc_alpha: float = 0.0    # virtual center alpha (latched at exit)
        self._fill_exit_vc_beta: float = 0.0     # virtual center beta (latched at exit)
        self._fill_exit_rot_radius: float = 0.06   # initial rotation radius (diameter 0.12)
        self._fill_exit_rot_phase: float = 0.0     # rotation phase accumulator
        # Fill visual wobble freq: beta channel oscillates every ~6 samples
        # at 30 Hz → period ≈ 0.2s → ω ≈ 2π/0.2 ≈ 31 rad/s
        self._fill_exit_rot_omega: float = float(2.0 * math.pi * self._idle_loop_rate_hz / 6.0)
        self._fill_exit_creep_streak: int = 0
        self._fill_exit_creep_cancel_threshold: int = 3

        # ── Fill-exit direction lock: block reversal until 1 full rotation ──
        self._fill_exit_direction_locked: bool = False
        self._fill_exit_lock_start_phase: float = 0.0
        self._fill_exit_lock_accumulated: float = 0.0

        # ── Beat→Fill entry transition: orbit radius contracts to center ──
        self._fill_entry_active: bool = False
        self._fill_entry_elapsed: float = 0.0
        self._fill_entry_duration_s: float = 1.0   # recomputed from BPM at transition start
        self._fill_entry_start_radius: float = 0.80
        self._fill_entry_start_center_y: float = 0.0
        self._fill_entry_phase: float = 0.0
        self._fill_entry_omega_start: float = 0.0   # beat angular velocity at start
        self._fill_entry_omega_end: float = 0.0     # fill rotation angular velocity target

        # ── Fill-from-silence speed ramp ──
        # When fill starts after silence, motion displacement starts at 10%
        # and ramps to 100% over 1500 ms so the first motions are gentle.
        self._fill_silence_ramp_active: bool = False
        self._fill_silence_ramp_start: float = 0.0
        self._FILL_SILENCE_RAMP_DURATION_S: float = 1.5   # 1500 ms
        self._FILL_SILENCE_RAMP_FLOOR: float = 0.10        # 10 % speed at onset
        self._fill_was_silent: bool = True  # start True so first-ever fill ramps in

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
        self._ramp_engine = RampEngine(config)

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

    def set_scheduled_lead_ms(self, value_ms: int) -> None:
        """Forward live lead-ms update to the intelligence layer."""
        if hasattr(self, '_intelligence') and self._intelligence is not None:
            self._intelligence.set_scheduled_lead_ms(value_ms)

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

    def process_beat(self, event: BeatEvent, silence_override: bool | None = None) -> Optional[TCodeCommand]:
        now = event.monotonic_timestamp if getattr(event, "monotonic_timestamp", 0.0) > 0 else time.perf_counter()
        raw_dt = (now - self.state.last_time) if self.state.last_time > 0 else (1.0 / 60.0)
        dt = float(np.clip(raw_dt, 1e-4, 0.05))
        hitch_soft_reset = bool(raw_dt > 0.25)
        self.state.last_time = now

        self._intelligence.set_audio_engine(self.audio_engine)
        decision = self._intelligence.build_decision(event=event, dt=dt, silence_override=silence_override)

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

            fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
            ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
            volume = float(np.clip(self.get_volume() * min(fade, ramp), 0.0, 1.0))
            self.state.alpha = float(np.clip(self.state.alpha, -1.0, 1.0))
            self.state.beta = float(np.clip(self.state.beta, -1.0, 1.0))
            return TCodeCommand(alpha=self.state.alpha, beta=self.state.beta, duration_ms=25, volume=volume)

        # ── Expression layer: per-frame updates ──
        self._update_expression_layer(decision=decision, dt=dt, now=now)

        # ── Intensity timer ramp: session-level escalation ──
        self._ramp_engine.tick(now, decision.silence_active)

        if decision.silence_active:
            # ── Silence park: glide smoothly to park position, fade volume ──
            self._anchor_phrase_locked = False
            self._fill_exit_active = False  # cancel fill exit on silence
            self._fill_entry_active = False  # cancel fill entry on silence
            self._fill_exit_direction_locked = False  # release direction lock on silence
            self._fill_was_silent = True  # arm silence→fill speed ramp
            self._fill_silence_ramp_active = False  # reset any in-progress ramp

            fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
            # Glide toward park position (0, park_y) with quintic ease
            park_alpha = 0.0
            park_beta = self._park_y  # 0.20 → maps to ~0.6 in display
            glide_speed = 2.0  # full glide in ~0.5 s
            glide_t = float(np.clip(glide_speed * dt, 0.0, 1.0))
            alpha = float(self.state.alpha + (park_alpha - self.state.alpha) * glide_t)
            beta = float(self.state.beta + (park_beta - self.state.beta) * glide_t)
            volume = float(np.clip(self.get_volume() * fade, 0.0, 1.0))
            self._last_journey_completion = 1.0

            # Reset orbit internals so a brief gate flicker can't
            # produce a position jump from stale phase / radius.
            self._orbit_phase_initialized = False
            self._smoothed_da = 0.0
            self._smoothed_db = 0.0
        else:
            progress = float(np.clip(decision.journey_completion, 0.0, 1.0))
            fill_enabled = bool(getattr(self.config.jitter, "enabled", True))

            # ── Fill-exit: center eases to live orbit, wobble decays ────
            if self._fill_exit_active:
                if decision.trigger_kind == "creep":
                    self._fill_exit_creep_streak += 1
                    if self._fill_exit_creep_streak >= self._fill_exit_creep_cancel_threshold:
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
                    ease_t = quintic_ease(raw_t)

                    # Live orbit target (advances every frame)
                    _live_angle = float(
                        self._journey_start_angle
                        + (self._journey_total_rotation * progress)
                    )
                    _live_center_y = float(self._base_center_y)
                    _live_radius = float(np.clip(
                        self._actual_radius if self._actual_radius > 0.1 else self._park_radius,
                        0.80, 1.0,
                    ))
                    _live_orbit_a, _live_orbit_b = orbit_point(
                        _live_angle, _live_radius, center_y=_live_center_y,
                    )

                    # Quintic-ease virtual center toward live orbit position
                    vc_a = float(self._fill_exit_vc_alpha
                                 + (_live_orbit_a - self._fill_exit_vc_alpha) * ease_t)
                    vc_b = float(self._fill_exit_vc_beta
                                 + (_live_orbit_b - self._fill_exit_vc_beta) * ease_t)

                    # Decaying wobble: quintic radius collapse + proportional decel
                    rot_decay = quintic_ease(raw_t)
                    rot_r = float(self._fill_exit_rot_radius * (1.0 - rot_decay))
                    rot_speed = float(self._fill_exit_rot_omega * (1.0 - 0.7 * rot_decay))
                    self._fill_exit_rot_phase += rot_speed * dt
                    alpha, beta = orbit_point(
                        self._fill_exit_rot_phase, rot_r, vc_a, vc_b,
                    )

                    if raw_t >= 1.0:
                        # Transition complete — seed orbit from live position
                        self._fill_exit_active = False
                        self._orbit_phase = float(_live_angle % (2.0 * math.pi))
                        self._actual_radius = float(_live_radius)
                        self._orbit_phase_initialized = True

                        # Lock direction until one full rotation completes
                        self._fill_exit_direction_locked = True
                        self._fill_exit_lock_start_phase = float(self._orbit_phase)
                        self._fill_exit_lock_accumulated = 0.0

                        self._journey_start_angle = float(self._orbit_phase)
                        self._journey_start_radius = float(self._actual_radius)
                        self._journey_start_alpha = float(_live_orbit_a)
                        self._journey_start_beta = float(_live_orbit_b)
                        self._journey_start_total_center_y = float(_live_center_y)
                        self._journey_linked = False
                        self._journey_target_radius = float(self._actual_radius)
                        self._journey_total_rotation = float(2.0 * math.pi)
                        self._last_journey_completion = progress

                        # Prime rate-limiter EMA with expected orbital velocity
                        _bpm_rl = self._current_bpm()
                        _omega = float(2.0 * math.pi * (_bpm_rl / 60.0) * self._idle_loops_per_beat)
                        _dt_rl = max(dt, 1e-4)
                        _dir = float(self._orbit_direction)
                        self._smoothed_da = float(
                            -self._actual_radius * math.sin(self._orbit_phase) * _omega * _dt_rl * _dir
                        )
                        self._smoothed_db = float(
                            self._actual_radius * math.cos(self._orbit_phase) * _omega * _dt_rl * _dir
                        )
                        alpha, beta = float(_live_orbit_a), float(_live_orbit_b)

                    fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
                    ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                    volume = float(np.clip(self.get_volume() * min(fade, ramp), 0.0, 1.0))
                    self._last_journey_completion = progress
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
                _bpm_dwell = self._current_bpm()
                _measure_s = (60.0 / _bpm_dwell) * self._fill_min_beats
                if (now - self._fill_enter_time) < _measure_s:
                    _in_fill_dwell = True

            if fill_enabled and (decision.trigger_kind == "creep" or _in_fill_dwell):
                if prev_trigger_kind != "creep" and not _in_fill_dwell:
                    self._fill_enter_time = now  # record when fill started
                    # ── Beat→Fill entry: start radius contraction spiral ──
                    # Only activate if we have a meaningful orbit to contract from;
                    # otherwise (e.g. very first frame) just jump to fill.
                    if self._orbit_phase_initialized and self._actual_radius > 0.15:
                        self._fill_entry_active = True
                        self._fill_entry_elapsed = 0.0
                        _bpm_entry = self._current_bpm()
                        self._fill_entry_duration_s = float(2.0 * 60.0 / _bpm_entry)  # 2 beats
                        self._fill_entry_start_radius = float(self._actual_radius)
                        self._fill_entry_start_center_y = float(self._base_center_y)
                        self._fill_entry_phase = float(self._orbit_phase)
                        # Beat orbital velocity (idle cruise speed)
                        _beat_omega = float(
                            2.0 * math.pi * (_bpm_entry / 60.0)
                            * self._idle_loops_per_beat
                            * self._orbit_direction
                        )
                        self._fill_entry_omega_start = _beat_omega
                        self._fill_entry_omega_end = float(
                            self._fill_exit_rot_omega
                            * (1.0 if self._orbit_direction > 0 else -1.0)
                        )
                # Arm speed ramp on the first fill frame after any silence
                # episode — checked every frame, not just on fill-entry edge,
                # because prev_trigger_kind may already be "creep" when
                # silence lifts (silence parks in creep mode).
                if self._fill_was_silent:
                    self._fill_silence_ramp_active = True
                    self._fill_silence_ramp_start = now
                    self._fill_was_silent = False
                if _in_fill_dwell:
                    # Override last_trigger so next frame still sees
                    # prev_trigger_kind == "creep" for dwell check
                    self._last_trigger_kind = "creep"
                self._fill_exit_active = False  # cancel any pending exit

                fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))

                # ── Beat→Fill entry transition: contracting spiral ──
                if self._fill_entry_active:
                    self._fill_entry_elapsed += dt
                    raw_t = float(np.clip(
                        self._fill_entry_elapsed / max(self._fill_entry_duration_s, 0.01),
                        0.0, 1.0,
                    ))
                    ease_t = quintic_ease(raw_t)

                    # Ease angular velocity from beat cruise → fill rotation
                    omega = float(
                        self._fill_entry_omega_start
                        + (self._fill_entry_omega_end - self._fill_entry_omega_start) * ease_t
                    )
                    self._fill_entry_phase += omega * dt

                    # Contract radius from beat orbit → fill orbit
                    bass_mult = self._fill_bass_freq_orbit_mult()
                    target_radius = float(self._fill_rot_radius * bass_mult)
                    current_radius = float(
                        self._fill_entry_start_radius
                        + (target_radius - self._fill_entry_start_radius) * ease_t
                    )

                    # Ease center from beat center → fill center
                    target_center_y = self._fill_dom_freq_to_y()
                    center_y = float(
                        self._fill_entry_start_center_y
                        + (target_center_y - self._fill_entry_start_center_y) * ease_t
                    )

                    alpha, beta = orbit_point(
                        self._fill_entry_phase, current_radius, 0.0, center_y,
                    )

                    if raw_t >= 1.0:
                        self._fill_entry_active = False
                        # Seed fill rotation phase from contraction endpoint
                        self._fill_rot_phase = float(
                            self._fill_entry_phase % (2.0 * math.pi)
                        )
                        self._orbit_phase_initialized = False

                    volume = float(np.clip(
                        self.get_volume() * min(fade, ramp), 0.0, 1.0,
                    ))
                    self._last_journey_completion = 1.0
                    self.state.alpha = float(alpha)
                    self.state.beta = float(beta)
                    return TCodeCommand(
                        alpha=alpha, beta=beta, duration_ms=25, volume=volume,
                    )

                alpha, beta, volume = self._apply_park_motion_frame(dt=dt, fade=min(fade, ramp))
                self._last_journey_completion = 1.0
                # Direct output — bypass rate limiter so the fill pattern
                # is not suppressed by the orbital velocity EMA.
                self.state.alpha = alpha
                self.state.beta = beta
                return TCodeCommand(alpha=alpha, beta=beta, duration_ms=25, volume=volume)

            if not fill_enabled and decision.trigger_kind == "creep":
                self._fill_exit_active = False
                self._fill_silence_ramp_active = False
                self._fill_was_silent = False
                self._orbit_phase_initialized = False
                self._last_journey_completion = 1.0

                fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * min(fade, ramp), 0.0, 1.0))
                return self._rate_limited_output(0.0, self._park_y, volume, dt)

            # ── Detect fill → orbit transition: latch center, start wobble decay ──
            if fill_enabled and prev_trigger_kind == "creep" and not self._fill_exit_active:
                self._fill_entry_active = False  # cancel any in-progress fill-entry
                self._fill_exit_active = True
                self._fill_exit_elapsed = 0.0
                self._fill_exit_creep_streak = 0
                # Duration = 1.5 beats — enough to blend without dragging
                _bpm = self._current_bpm()
                self._fill_exit_duration_s = 1.5 * 60.0 / _bpm
                # Latch virtual center to dot's current position
                self._fill_exit_vc_alpha = float(self.state.alpha)
                self._fill_exit_vc_beta = float(self.state.beta)
                self._fill_exit_rot_phase = 0.0
                self._fill_exit_rot_radius = 0.06   # diameter 0.12, matches fill wander

                # ── Initialize journey geometry NOW so the live orbit target
                # is computed from correct values during the transition.
                _center_y = float(self._base_center_y)
                _target_radius = max(float(decision.radius_bloom), 0.80)
                if self._orbit_phase_initialized:
                    _start_angle = float(self._orbit_phase)
                else:
                    _start_angle, _ = infer_orbit(
                        self.state.alpha, self.state.beta, _center_y,
                    )
                    self._orbit_phase = float(_start_angle % (2.0 * math.pi))
                    self._orbit_phase_initialized = True
                self._journey_start_angle = float(_start_angle)
                self._journey_start_radius = float(_target_radius)
                self._journey_start_total_center_y = float(_center_y)
                self._journey_target_radius = float(_target_radius)
                self._actual_radius = float(_target_radius)
                self._journey_total_rotation = float(2.0 * math.pi * decision.interval_beats / 2.0)
                self._journey_linked = False

                # First exit frame: full wobble, no decay yet
                alpha, beta = orbit_point(
                    self._fill_exit_rot_phase, self._fill_exit_rot_radius,
                    self._fill_exit_vc_alpha, self._fill_exit_vc_beta,
                )
                fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
                ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
                volume = float(np.clip(self.get_volume() * min(fade, ramp), 0.0, 1.0))
                self._last_journey_completion = progress
                self.state.alpha = float(alpha)
                self.state.beta = float(beta)
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

                # Size ramp: expand max_radius by up to 60 % over the timer duration
                _size_exp = self._ramp_engine.size_expansion
                if _size_exp > 0.0:
                    self._journey_max_radius = float(np.clip(
                        self._journey_max_radius * (1.0 + _size_exp),
                        self._journey_park_radius,
                        1.0,
                    ))

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
                    inherited_angle, inherited_radius = infer_orbit(
                        self._journey_start_alpha,
                        self._journey_start_beta,
                        self._journey_start_total_center_y,
                    )
                    self._journey_start_angle = inherited_angle
                    self._journey_start_radius = float(np.clip(
                        inherited_radius, self._journey_park_radius, 1.0
                    ))
                    self._orbit_phase = float(inherited_angle % (2.0 * math.pi))
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
                bpm_for_terminal = float(np.clip(bpm_for_terminal if bpm_for_terminal > 0.0 else 120.0, 40.0, 200.0))
                fallback_terminal_speed = float((2.0 * math.pi) * (bpm_for_terminal / 60.0) * self._idle_loops_per_beat)
                # Use the BPM-derived idle orbit speed — NOT the journey's
                # angular velocity.  Journey velocity can be 10-25 rad/s
                # (one turn per beat), which overwhelms the rate limiter
                # and forces the output to trace a tiny circle (~0.40 r)
                # instead of the intended 0.80+ radius orbit.
                terminal_speed = float(max(fallback_terminal_speed, 0.8))
                angle = float(self._orbit_phase + (terminal_speed * dt * float(self._orbit_direction)))

            self._orbit_phase = float(angle % (2.0 * math.pi))

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
                (self._journey_total_rotation * progress) / (2.0 * math.pi),
                0.0,
                1.0,
            ))
            blend_window = 0.40
            blend_t = float(np.clip(first_pass_progress / blend_window, 0.0, 1.0))
            radius_blend = quintic_ease(blend_t)
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
                center_blend = quintic_ease(progress)
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
                    self._base_center_y = exponential_approach(
                        self._base_center_y, base_target_center, 3.0, dt,
                    )
                else:
                    self._base_center_y = float(base_target_center)

            total_center_y = float(self._base_center_y)
            orbit_radius = float(min(radius, radius_cap_for_center(total_center_y, self._center_y_offset)))

            alpha, beta = orbit_point(angle, orbit_radius, center_y=total_center_y)

            if progress >= 1.0 and abs(alpha) < 0.01:
                angle = float(angle + (0.08 * float(self._orbit_direction)))
                self._orbit_phase = float(angle % (2.0 * math.pi))
                alpha, beta = orbit_point(angle, orbit_radius, center_y=total_center_y)

            # Apply post-silence ramp AND silence-fade to volume.
            # silence_fade starts at 0 when silence lifts and ramps to
            # 1.0 over ~10 frames, preventing an instant volume spike
            # if the gate opens on noise.
            fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
            ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))
            volume = float(np.clip(self.get_volume() * min(fade, ramp), 0.0, 1.0))

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

        # ── Post-fill-exit direction lock: accumulate rotation, release after 2π ──
        if self._fill_exit_direction_locked and self._orbit_phase_initialized:
            # Track how much angle has been swept since the lock started
            prev_phase = self._fill_exit_lock_start_phase + self._fill_exit_lock_accumulated
            cur_phase = float(self._orbit_phase)
            # Compute signed delta, unwrap to [-π, π]
            delta = cur_phase - (prev_phase % (2.0 * math.pi))
            if delta > math.pi:
                delta -= 2.0 * math.pi
            elif delta < -math.pi:
                delta += 2.0 * math.pi
            self._fill_exit_lock_accumulated += abs(delta)
            if self._fill_exit_lock_accumulated >= 2.0 * math.pi:
                self._fill_exit_direction_locked = False

        # ── §1: Anchor phrase management (direction change → new anchor) ──
        if getattr(self.config.stroke, 'direction_change_enabled', True) and not decision.silence_active:
            interval_s = float(getattr(self.config.stroke, 'direction_change_interval_s', 15.0) or 15.0)
            drop_needed = float(getattr(self.config.stroke, 'direction_change_energy_drop', 0.35) or 0.35)

            # Block direction reversal during fill-exit transition or
            # until one full rotation has been completed after it.
            _direction_allowed = bool(
                not self._fill_exit_active
                and not self._fill_exit_direction_locked
            )

            if (_direction_allowed
                    and now - self._last_direction_change_time > interval_s
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
        center_y = float(self._base_center_y)
        effective_beta = float(self.state.beta - self._center_y_offset)
        inferred_phase, inferred_r = infer_orbit(self.state.alpha, effective_beta, center_y)
        if inferred_r > self._actual_radius + 0.05:
            self._orbit_phase = float(inferred_phase % (2.0 * math.pi))
            self._actual_radius = float(inferred_r)

    def _fill_silence_speed_scale(self, now: float) -> float:
        """Return 0.10→1.0 speed-scale multiplier for the silence→fill ramp."""
        if not self._fill_silence_ramp_active:
            return 1.0
        elapsed = now - self._fill_silence_ramp_start
        if elapsed >= self._FILL_SILENCE_RAMP_DURATION_S:
            self._fill_silence_ramp_active = False
            return 1.0
        t = float(np.clip(elapsed / self._FILL_SILENCE_RAMP_DURATION_S, 0.0, 1.0))
        # Smooth ease-in (quadratic) so acceleration feels natural
        eased = t * t
        return float(self._FILL_SILENCE_RAMP_FLOOR + (1.0 - self._FILL_SILENCE_RAMP_FLOOR) * eased)

    def _apply_park_motion_frame(self, dt: float, fade: float) -> tuple[float, float, float]:
        """Funscript idle-fill: circular orbit around a Y-sweeping center.

        The center oscillates vertically using the baked 45-sample sweep.
        The dot orbits that center at radius 0.06 (~31 rad/s) producing
        a tightly-wound spring / helicopter-blade trail.

        When resuming from silence the displacement from rest is scaled
        by a 10%→100% ramp over 1500 ms so the first motions are gentle.
        """
        now = time.perf_counter()
        speed_scale = self._fill_silence_speed_scale(now)

        loop_alpha, _loop_beta = self._sample_idle_loop(dt=dt)

        # Center Y driven by full-spectrum dominant frequency
        center_x = 0.0
        center_y = self._fill_dom_freq_to_y()

        # ── Hi-hat / snare impulse: quick downward kick on Y ──
        hh_hit = False
        if self.audio_engine is not None and hasattr(self.audio_engine, '_band_zscore_signals'):
            sigs = self.audio_engine._band_zscore_signals
            if sigs.get('high', 0) == 1 or sigs.get('mid', 0) == 1:
                hh_hit = True
        if hh_hit:
            # Re-trigger impulse (take the larger of current residual or fresh kick)
            self._fill_hh_impulse = max(self._fill_hh_impulse, self._FILL_HH_IMPULSE_SIZE)
        # Exponential decay
        self._fill_hh_impulse *= math.exp(-self._FILL_HH_IMPULSE_DECAY * dt)
        if self._fill_hh_impulse < 0.001:
            self._fill_hh_impulse = 0.0
        # Apply impulse downward (negative Y)
        center_y = float(np.clip(center_y - self._fill_hh_impulse, -0.5, 0.5))

        # Advance rotation phase at wobble frequency (~31 rad/s)
        self._fill_rot_phase += self._fill_exit_rot_omega * dt
        if self._fill_rot_phase > 2.0 * math.pi:
            self._fill_rot_phase -= 2.0 * math.pi

        # Bass-frequency orbit modulation: widen for sub-bass, tighten for upper bass
        bass_mult = self._fill_bass_freq_orbit_mult()
        effective_rot_radius = self._fill_rot_radius * bass_mult

        # Size ramp: expand fill orbit by up to 60 % over timer duration
        _size_exp = self._ramp_engine.size_expansion
        if _size_exp > 0.0:
            effective_rot_radius *= (1.0 + _size_exp)

        # Orbit around the moving center
        alpha = float(center_x + effective_rot_radius * math.cos(self._fill_rot_phase))
        beta  = float(center_y + effective_rot_radius * math.sin(self._fill_rot_phase))

        # ── Silence→fill speed ramp: scale displacement from rest ──
        if speed_scale < 1.0:
            rest_alpha = 0.0
            rest_beta = float(-(self._park_y * 2.0 - 1.0))  # park_y in display coords
            alpha = float(rest_alpha + (alpha - rest_alpha) * speed_scale)
            beta  = float(rest_beta  + (beta  - rest_beta)  * speed_scale)

        volume = float(np.clip(self.get_volume() * float(np.clip(fade, 0.0, 1.0)), 0.0, 1.0))
        return alpha, beta, volume

    def _fill_bass_freq_orbit_mult(self) -> float:
        """Return orbit radius multiplier based on rolling dominant bass frequency.

        Maps bass frequency to orbit size on a log scale:
         200 Hz  → 1.0× (highest bass = base orbit, no expansion)
          50 Hz  → 2.0× (lowest bass = maximum expansion)
        Uses a 4-frame rolling average for stability.
        """
        # Sample dominant frequency in the bass range (30–500 Hz)
        freq = 125.0  # neutral default
        if self.audio_engine is not None and hasattr(self.audio_engine, '_estimate_frequency'):
            spectrum = None
            if hasattr(self.audio_engine, 'get_spectrum'):
                spectrum = self.audio_engine.get_spectrum()
            if spectrum is not None:
                try:
                    f = float(self.audio_engine._estimate_frequency(spectrum, 30.0, 500.0))
                    if f > 0.0:
                        freq = f
                except Exception:
                    pass

        self._fill_bass_freq_history.append(freq)

        if len(self._fill_bass_freq_history) == 0:
            return 1.0

        avg_freq = float(sum(self._fill_bass_freq_history) / len(self._fill_bass_freq_history))

        # Log-linear mapping: lower bass → wider orbit
        #   log2(200) → 1.0×  (highest bass, no expansion)
        #   log2(50)  → 2.0×  (lowest bass, maximum expansion)
        avg_freq = float(np.clip(avg_freq, 50.0, 200.0))
        log_f = math.log2(avg_freq)
        log_50 = math.log2(50.0)    # ~5.644
        log_200 = math.log2(200.0)  # ~7.644

        # Linear interpolation in log2 space: high freq → 1.0, low freq → 2.0
        t = (log_f - log_50) / (log_200 - log_50)  # 0 at 50 Hz, 1 at 200 Hz
        mult = 2.0 - 1.0 * t                        # 2.0 at 50 Hz, 1.0 at 200 Hz

        return float(np.clip(mult, 1.0, 2.0))

    def _fill_dom_freq_to_y(self) -> float:
        """Map full-spectrum dominant frequency to fill center Y position.

        Uses log-scale mapping over 80–8000 Hz:
          Low  freq (80 Hz)   → +0.5  (high Y position)
          High freq (8000 Hz) → -0.5  (low Y position)
        Smoothed with a 6-frame rolling average for stability.
        """
        freq = 500.0  # neutral default (maps to ~0.0)
        if self.audio_engine is not None and hasattr(self.audio_engine, '_estimate_frequency'):
            spectrum = None
            if hasattr(self.audio_engine, 'get_spectrum'):
                spectrum = self.audio_engine.get_spectrum()
            if spectrum is not None:
                try:
                    f = float(self.audio_engine._estimate_frequency(spectrum))
                    if f > 0.0:
                        freq = f
                except Exception:
                    pass

        self._fill_dom_freq_history.append(freq)

        if len(self._fill_dom_freq_history) == 0:
            return 0.0

        avg_freq = float(sum(self._fill_dom_freq_history) / len(self._fill_dom_freq_history))

        # Log-linear mapping: log2(80)→+0.5, log2(8000)→-0.5
        avg_freq = float(np.clip(avg_freq, 80.0, 8000.0))
        log_f = math.log2(avg_freq)
        log_lo = math.log2(80.0)     # ~6.322
        log_hi = math.log2(8000.0)   # ~12.966

        # t goes 0→1 as freq rises from 80→8000
        t = (log_f - log_lo) / (log_hi - log_lo)
        # Map to +0.5 (low freq) → -0.5 (high freq)
        y = float(0.5 - 1.0 * t)

        return float(np.clip(y, -0.5, 0.5))

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

    def _compute_landing_rotation(self, start_angle: float, interval_beats: int) -> float:
        _ = interval_beats
        max_turns = 1.0
        turns = 1.0

        # §1: Anchor landing – ensure the journey ends within ±10° of the
        # chosen Y-axis anchor.  Adjust total rotation so the arrival angle
        # falls inside the anchor swing window.
        anchor_angle = float(math.pi / 2.0) * self._anchor_sign  # +Y or -Y
        target_end = float(start_angle + turns * 2.0 * math.pi * self._orbit_direction)
        swing_rad = float(math.radians(self._anchor_swing_deg))
        best_end = nearest_anchor_crossing(target_end, anchor_angle, swing_rad)
        # Recompute turns from adjusted end
        delta = best_end - start_angle
        if abs(self._orbit_direction) > 0:
            if self._orbit_direction > 0 and delta < 0:
                delta += 2.0 * math.pi
            elif self._orbit_direction < 0 and delta > 0:
                delta -= 2.0 * math.pi
        turns = abs(delta) / (2.0 * math.pi)
        turns = float(np.clip(turns, 0.3, max_turns))

        rotation = float(turns * 2.0 * math.pi * self._orbit_direction)
        return rotation

    def _current_bpm(self) -> float:
        """Best-effort BPM from audio engine metronome, clamped to [40, 240]."""
        if self.audio_engine is not None:
            met = float(getattr(self.audio_engine, "_metronome_bpm", 0.0) or 0.0)
            if met > 0:
                return float(np.clip(met, 40.0, 200.0))
        return 120.0
