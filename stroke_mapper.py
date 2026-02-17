"""
bREadbeats - Stroke Mapper v2
Converts beat events into alpha/beta stroke patterns.

Two behavioral modes driven by audio amplitude:
  FULL_STROKE  – high amplitude: tempo-synced full circle rotations on beats
  CREEP_MICRO  – low amplitude: slow creep around edge with micro-effects on beats

All modes use circular coordinates around (0,0) in the alpha/beta plane.
"""

import numpy as np
import time
import random
import os
import json
from collections import deque
from typing import Optional, Tuple, Callable, List
from dataclasses import dataclass, field
from pathlib import Path

from config import Config, StrokeMode
from audio_engine import BeatEvent
from network_engine import TCodeCommand
from logging_utils import log_event


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class StrokeState:
    """Current stroke position and state"""
    alpha: float = 0.0
    beta: float = 0.0
    target_alpha: float = 0.0
    target_beta: float = 0.0
    phase: float = 0.0            # 0-1 position in stroke cycle (continuous, tempo-synced)
    last_beat_time: float = 0.0
    last_stroke_time: float = 0.0
    idle_time: float = 0.0        # Time since last beat
    jitter_angle: float = 0.0     # Current jitter rotation
    creep_angle: float = 0.0      # Current creep rotation
    beat_counter: int = 0         # For beat counting within measure
    creep_reset_start_time: float = 0.0
    creep_reset_active: bool = False


@dataclass
class PlannedTrajectory:
    """Pre-computed arc trajectory for frame-by-frame playback.
    Instead of a separate thread, idle motion reads one point per frame."""
    alpha_points: np.ndarray = field(default_factory=lambda: np.array([]))
    beta_points: np.ndarray = field(default_factory=lambda: np.array([]))
    step_durations: List[int] = field(default_factory=list)
    n_points: int = 0
    current_index: int = 0
    band_volume: float = 1.0
    start_time: float = 0.0
    is_micro: bool = False  # True for noise burst micro-patterns (skip return-to-bottom)
    is_park_return: bool = False  # True when arc is explicitly returning to park anchor
    beat_target_time: float = 0.0  # Monotonic time when the dot should "land" on the beat
    original_bpm: float = 0.0  # BPM at arc creation, for mid-arc speed adjustment

    @property
    def active(self) -> bool:
        return self.current_index < self.n_points

    @property
    def finished(self) -> bool:
        return self.current_index >= self.n_points


# ---------------------------------------------------------------------------
# Behavioral mode enum
# ---------------------------------------------------------------------------

class MotionMode:
    FULL_STROKE = "full_stroke"      # High amplitude: full circle rotations on beats
    CREEP_MICRO = "creep_micro"      # Low amplitude: creep with micro-effects


# ---------------------------------------------------------------------------
# StrokeMapper v2
# ---------------------------------------------------------------------------

class StrokeMapper:
    """
    Converts beat events to alpha/beta stroke commands.

    v2 design:
      - Tempo-synced continuous rotation (one full circle per beat)
      - Amplitude-gated mode switching (FULL_STROKE vs CREEP_MICRO)
      - Micro-effect system: small jerks on beats scaled by mid/high energy
            - Park-aware launch continuity for beat arcs

    All stroke modes create circular/arc patterns in the alpha/beta plane.
    Alpha and beta range from -1 to 1, with (0,0) at center.
    """

    def __init__(self, config: Config,
                 send_callback: Callable[[TCodeCommand], None] = None,
                 get_volume: Callable[[], float] = None,
                 audio_engine=None):
        self.config = config
        self.state = StrokeState()
        self.send_callback = send_callback
        self.get_volume = get_volume if get_volume is not None else (lambda: 1.0)
        self.audio_engine = audio_engine

        # Motion intensity multiplier (0.25-2.0, default 1.0) — GUI slider
        self.motion_intensity: float = 1.0

        # ---------- Amplitude gate ----------
        # RMS envelope tracker for mode switching
        self._rms_envelope: float = 0.0
        self._rms_attack: float = 0.15     # faster attack to respond to loud passages
        self._rms_release: float = 0.008   # moderate release
        # Gate thresholds now read from config.stroke.amplitude_gate_high/low
        self._motion_mode: str = MotionMode.CREEP_MICRO  # start quiet
        self._mode_switch_time: float = 0.0

        # ---------- Tempo-synced rotation ----------
        # phase accumulator: 0.0-1.0, one full cycle = one beat
        self._beat_phase: float = 0.0
        self._phase_time: float = time.perf_counter()
        self._current_bpm: float = 0.0

        # ---------- Full-stroke planned trajectory ----------
        self._trajectory: Optional[PlannedTrajectory] = None

        # ---------- Micro-effect state ----------
        self._micro_effects_enabled: bool = True   # toggle from GUI
        self._last_micro_jerk_time: float = 0.0
        self._micro_jerk_alpha: float = 0.0
        self._micro_jerk_beta: float = 0.0
        self._micro_jerk_decay_ms: float = 120.0   # jerk decays over this many ms

        # ---------- Band energy trackers (updated from BeatEvent) ----------
        self._sub_bass_energy: float = 0.0
        self._low_mid_energy: float = 0.0
        self._mid_energy: float = 0.0
        self._high_energy: float = 0.0
        self._bass_jitter_speed_mult: float = 1.0
        self._bass_jitter_attack: float = 0.25
        self._bass_jitter_release: float = 0.06

        # ---------- Band-based scaling tables ----------
        self._band_speed_scale = {
            'sub_bass': 0.70,
            'low_mid':  0.85,
            'mid':      1.00,
            'high':     1.20,
        }

        # ---------- Spiral mode persistent state ----------
        self.spiral_beat_index = 0
        self.spiral_revolutions = 3

        # ---------- Flux tracking ----------
        self._flux_history: deque = deque()
        self._flux_rise_window_ms: float = 250.0
        self._flux_stroke_factor: float = 1.0

        # ---------- Fade / silence ----------
        self._fade_intensity: float = 1.0
        self._last_quiet_time: float = 0.0
        self._consecutive_silent_count: int = 0
        self._silence_reset_armed: bool = True

        # ---------- Creep volume fade ----------
        self._creep_sustained_start: float = 0.0
        self._creep_volume_factor: float = 1.0
        self._creep_was_active_last_frame: bool = False

        # ---------- Idle motion throttle (separate from beat stroke timing) ----------
        self._last_idle_time: float = 0.0

        # ---------- Last known BPM (persist through confidence drops) ----------
        self._last_known_bpm: float = 0.0

        # ---------- Post-arc smooth blend ----------
        # After an arc completes, smoothly blend from arc endpoint to creep orbit
        self._post_arc_blend: float = 1.0  # 1.0 = fully on creep orbit (start normal), reset to 0.0 after arc
        self._post_arc_blend_rate: float = 0.05  # per frame (at 60fps, ~20 frames = 333ms to settle)

        # ---------- Beat factoring ----------
        self.max_strokes_per_sec = 4.5
        self.beat_factor = 1

        # ---------- Beats-between-strokes counter ----------
        self._beats_since_stroke: int = 0  # counts how many beats have passed since last full stroke

        # ---------- Burst scheduler state ----------
        # Keep initialized for branches that reference scheduled burst deactivation.
        self._burst_scheduled_active: bool = False

        # ---------- Pending arc launch / anchor compatibility ----------
        self._pending_arc_event: Optional[BeatEvent] = None
        self._pending_arc_target: float = 0.0       # phase target for deferred arc fire
        self._pending_arc_is_downbeat: bool = False
        self._arc_anchor_threshold: float = 0.35     # radians (~20°) — close enough to fire
        self._single_anchor_bottom_phase: float = 0.0
        self._single_anchor_prebottom_offset: float = 0.22
        # Keep legacy single-anchor behavior only for mode3 (TEARDROP).
        self._single_anchor_enabled_modes = {StrokeMode.TEARDROP}
        self._spiral_direction: int = 1

        # ---------- Locked anchor placeholder (kept for backward compatibility) ----------
        self._locked_anchor: Optional[float] = None

        # ---------- Stroke readiness gating ----------
        # Strokes only fire when metronome + traffic light conditions are met:
        #   Option A: metronome GREEN + traffic YELLOW or GREEN
        #   Option B: metronome GREEN or YELLOW + traffic GREEN
        #   Option C: traffic YELLOW (was recently GREEN) + metronome YELLOW or GREEN
        #   Option D: metronome GREEN stable >2s + any traffic state
        # Otherwise: creep/jitter only
        # Grace period: short hold after conditions drop before returning to jitter
        self._stroke_ready: bool = False
        self._stroke_ready_lost_time: float = 0.0   # when conditions last dropped
        self._stroke_grace_ms: float = float(np.clip(
            getattr(self.config.beat, 'teaching_stroke_ready_grace_ms', 450.0) or 450.0,
            100.0,
            1300.0,
        ))
        self._stroke_gate_block_streak: int = 0
        self._stroke_finish_beats: int = int(np.clip(
            getattr(self.config.beat, 'teaching_stroke_finish_beats', 1) or 1,
            0,
            4,
        ))
        self._traffic_was_green: bool = False         # track if traffic was recently green
        self._traffic_left_green_time: float = 0.0    # when traffic left green
        self._metro_green_since: float = 0.0          # when metronome first became green
        self._prev_had_any_light: bool = False        # was at least one light yellow+ last check (track cold-start)

        # ---------- Last confirmed beat time (for no-beat timeout) ----------
        self._last_confirmed_beat_time: float = 0.0   # wall-clock of last beat with stroke_ready
        self._last_any_beat_time: float = 0.0         # monotonic time of last detected beat (ungated)
        self._tempo_reset_motion_hold_s: float = 1.8
        self._tempo_reset_motion_hold_until: float = 0.0

        # ---------- Snap timing feedback (self-checking) ----------
        # When snap-to-target fires, record the timing error so the next arc
        # can compensate by shortening/lengthening its duration.
        self._last_snap_correction_ms: float = 0.0
        self._lead_trim_ms: float = 0.0
        self._lead_trim_limit_ms: float = 40.0
        self._lead_target_error_ms: float = 8.0  # slight intentional late landing
        self._no_beat_timeout_s: float = 2.0           # seconds before returning to center+jitter
        self._timing_scale_min: float = 0.5   # up to 2x faster
        self._timing_scale_max: float = 2.0   # up to 1/2 speed

        # ---------- Post-silence volume ramp ----------
        # After silence/track-change reset, reduce volume and slowly ramp back up
        self._post_silence_ramp_active: bool = False
        self._post_silence_ramp_start: float = 0.0     # time.time() when ramp started
        self._was_silent: bool = False                  # track if we were faded out

        # ---------- Flux history / center-reset guard ----------
        self._recent_flux_values: deque = deque(maxlen=60)  # ~1s of flux history for center-reset flux guard
        self._recent_low_band_values: deque = deque(maxlen=60)  # ~1s of low-band activity for beat gating/fallback
        self._recent_high_band_values: deque = deque(maxlen=60)  # ~1s of high-band activity for beat gating
        self._recent_mid_bass_values: deque = deque(maxlen=60)  # ~1s of 200-400Hz support activity
        self._recent_high_band_beat_hits: deque = deque(maxlen=16)  # recent beat-wise upper-band context hits

        # ---------- Motion-block diagnostics (throttled) ----------
        self._motion_block_active: bool = False
        self._last_block_reason: str = ""
        self._last_block_log_time: float = 0.0
        self._block_log_interval_s: float = 0.75
        self._block_summary_interval_s: float = 10.0
        self._block_summary_window_start: float = time.time()
        self._block_reason_order: List[str] = [
            'overall_activity_gate',
            'overall_amp_fill_gate',
            'bass_gate',
            'mid_trigger_block',
            'dual_band_db_gate',
            'stroke_ready',
            'beat_divisor',
            'low_band_gate',
            'high_band_gate',
            'mode_creep_micro',
        ]
        self._block_reason_counts = {reason: 0 for reason in self._block_reason_order}
        self._motion_resumed_count: int = 0
        self._blocked_beat_events: int = 0

        # ---------- Adaptive amp-fill threshold controller ----------
        # Raises/lower required fill ratio per phase so beat fires stay selective.
        stroke_cfg = self.config.stroke
        self._auto_fill_enabled: bool = bool(getattr(stroke_cfg, 'overall_amp_fill_auto_enabled', True))
        self._auto_fill_target_pass_rate: float = float(np.clip(
            getattr(stroke_cfg, 'overall_amp_fill_auto_target_pass_rate', 0.58) or 0.58,
            0.10,
            0.95,
        ))
        self._auto_fill_ema_alpha: float = float(np.clip(
            getattr(stroke_cfg, 'overall_amp_fill_auto_ema_alpha', 0.12) or 0.12,
            0.01,
            0.60,
        ))
        self._auto_fill_deadband: float = float(np.clip(
            getattr(stroke_cfg, 'overall_amp_fill_auto_deadband', 0.06) or 0.06,
            0.0,
            0.40,
        ))
        self._auto_fill_step: float = float(np.clip(
            getattr(stroke_cfg, 'overall_amp_fill_auto_step', 0.02) or 0.02,
            0.001,
            0.15,
        ))
        self._auto_fill_max_offset: float = float(np.clip(
            getattr(stroke_cfg, 'overall_amp_fill_auto_max_offset', 0.35) or 0.35,
            0.01,
            0.80,
        ))
        self._auto_fill_min_required: float = float(np.clip(
            getattr(stroke_cfg, 'overall_amp_fill_auto_min_required', 0.05) or 0.05,
            0.0,
            0.95,
        ))
        self._auto_fill_max_required: float = float(np.clip(
            getattr(stroke_cfg, 'overall_amp_fill_auto_max_required', 0.98) or 0.98,
            0.05,
            1.0,
        ))
        if self._auto_fill_max_required < self._auto_fill_min_required:
            self._auto_fill_max_required = self._auto_fill_min_required
        self._auto_fill_state = {
            'beat': {'ema': self._auto_fill_target_pass_rate, 'offset': 0.0},
            'downbeat': {'ema': self._auto_fill_target_pass_rate, 'offset': 0.0},
            'syncopation': {'ema': self._auto_fill_target_pass_rate, 'offset': 0.0},
        }
        self._auto_fill_log_interval_s: float = 2.0
        self._auto_fill_last_log_time: float = 0.0

        # ---------- Large-jump diagnostics ----------
        self._jump_log_threshold: float = 0.85
        self._jump_log_interval_s: float = 0.25
        self._last_jump_log_time: float = 0.0
        self._trajectory_max_step_advance: int = 2

        # ---------- Anti-stop safeguards ----------
        self._anti_stop_min_delta: float = 0.010
        self._anti_stop_angle_step: float = 0.035
        self._anti_stop_edge_radius: float = 0.80

        # ---------- Center+jitter flux guard diagnostics ----------
        self._last_center_guard_log_time: float = 0.0

        # ---------- Teaching/learning adapter (runtime, bounded) ----------
        self._learning_enabled: bool = bool(getattr(self.config.beat, 'teaching_learning_enabled', True))
        self._learning_strength: float = float(np.clip(getattr(self.config.beat, 'teaching_learning_strength', 0.55), 0.0, 1.0))
        self._learning_min_confidence: float = float(np.clip(getattr(self.config.beat, 'teaching_min_confidence', 0.12), 0.0, 1.0))
        self._learning_use_fitted_rules: bool = bool(getattr(self.config.beat, 'teaching_use_fitted_rules', True))
        self._learning_rule_fit_path: str = str(getattr(self.config.beat, 'teaching_rule_fit_path', '') or '').strip()
        self._learning_no_motion_bias: float = float(np.clip(getattr(self.config.beat, 'teaching_no_motion_bias', 1.0), 0.25, 3.0))
        self._learning_apply_in_circle_mode: bool = bool(getattr(self.config.beat, 'teaching_apply_in_circle_mode', False))
        self._learning_isolation_mode: bool = bool(getattr(self.config.beat, 'teaching_isolation_mode', True))
        self._learned_divisor_hint: int = 1
        self._learned_radius_mult: float = 1.0
        self._learned_lead_ms: float = 0.0
        self._learned_sync_size_mult: float = 1.0
        self._learned_sync_speed_mult: float = 1.0
        self._edge_follow_radius: float = 0.85
        self._learning_model_loaded: bool = False
        self._learning_model_path: str = ""
        self._learning_model: dict = {}
        self._learning_norm_mean: dict[str, float] = {}
        self._learning_norm_std: dict[str, float] = {}
        self._learning_cadence_rule: dict = {}
        self._learning_feature_columns: list[str] = []
        self._try_load_learning_model()

        # ---------- Learning-first gate simplification ----------
        # When enabled, relax selected legacy hard gates because runtime
        # learning + metronome timing now carries most adaptation.
        self._learning_relax_phase1_gates: bool = bool(
            getattr(self.config.beat, 'teaching_relax_phase1_gates', True)
        )
        # Ignore traffic-light (metric settledness) for stroke readiness.
        # Keep metronome confidence as the primary readiness signal.
        self._ignore_traffic_lights: bool = bool(
            getattr(self.config.beat, 'teaching_ignore_traffic_lights', False)
        )
        self._metronome_relaxed_confidence: float = float(np.clip(
            getattr(self.config.beat, 'teaching_metronome_relaxed_confidence', 0.14) or 0.14,
            0.05,
            0.40,
        ))

        # ---------- Landing / park anchors ----------
        # Display-bottom phase is 0 in current alpha/beta orientation.
        # (PositionCanvas maps display Y from -beta after restim transforms.)
        self._landing_offset_degrees: float = 7.5
        self._park_radius_min: float = 0.65
        self._park_radius_max: float = 0.90
        self._park_radius: float = 0.70
        self._park_bottom_phase: float = 0.0
        self._park_freq_bias: float = 0.0
        self._single_anchor_bottom_phase = self._park_bottom_phase

        # ---------- Park / wait target ----------
        self._park_alpha: float = 0.0
        self._park_beta: float = 0.0
        self._update_park_anchor_from_radius(self._park_radius)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vol_floor(self, base_vol: float) -> float:
        """Minimum allowed volume given vol_reduction_limit config."""
        limit_pct = self.config.stroke.vol_reduction_limit / 100.0
        return base_vol * (1.0 - limit_pct)

    def _combo_raw(self, name: str, default: float = 1.0) -> float:
        return float(np.clip(float(getattr(self.config.stroke, name, default) or default), -2.0, 3.0))

    def _combo_scale(self, name: str, min_scale: float, max_scale: float, default: float = 1.0) -> float:
        raw = self._combo_raw(name, default)
        if raw >= 1.0:
            t = float(np.clip((raw - 1.0) / 2.0, 0.0, 1.0))
            return float(1.0 + (max_scale - 1.0) * t)
        t = float(np.clip((1.0 - raw) / 3.0, 0.0, 1.0))
        return float(1.0 - (1.0 - min_scale) * t)

    @staticmethod
    def _snap_divisor(value: float) -> int:
        candidates = (1, 2, 4, 8)
        if value <= 0:
            return 1
        target = float(value)
        return min(candidates, key=lambda c: abs(np.log2(target) - np.log2(float(c))))

    def _effective_min_interval_ms(self) -> float:
        base_ms = float(getattr(self.config.stroke, 'min_interval_ms', 260) or 260)
        speed_scale = self._combo_scale('combo_speed', 0.65, 1.55)
        effective_ms = base_ms / max(speed_scale, 1e-6)
        return float(np.clip(effective_ms, 80.0, 1400.0))

    def _update_park_anchor_from_radius(self, radius_hint: Optional[float] = None) -> None:
        """Update park/landing anchor using dynamic radius from recent stroke size.

        Landing target stays 5-10° right of display-bottom, with larger strokes
        using a slightly larger rightward offset.
        """
        if radius_hint is None:
            radius_hint = self._park_radius
        radius = float(np.clip(radius_hint, self._park_radius_min, self._park_radius_max))
        span = max(1e-6, self._park_radius_max - self._park_radius_min)
        radius_norm = float(np.clip((radius - self._park_radius_min) / span, 0.0, 1.0))
        self._landing_offset_degrees = 5.0 + (5.0 * radius_norm)
        bottom_phase = float(self._park_bottom_phase)
        self._park_radius = radius
        self._park_alpha = float(np.sin(bottom_phase) * radius)
        self._park_beta = float(np.cos(bottom_phase) * radius)

    def _get_anchor_bass_norm(self, event: Optional[BeatEvent]) -> float:
        """Return normalized low-bass activity for anchor-state motion shaping."""
        activity = float(max(0.0, (self._sub_bass_energy * 1.0) + (self._low_mid_energy * 0.65)))

        if activity <= 1e-6 and event is not None:
            peak = float(getattr(event, 'peak_energy', 0.0) or 0.0)
            freq = float(getattr(event, 'frequency', 0.0) or 0.0)
            beat_band = getattr(event, 'beat_band', '')
            if beat_band in ('sub_bass', 'low_mid') or (30.0 <= freq <= 220.0):
                activity = max(activity, peak * 0.55)

        bass_floor = float(getattr(self.config.stroke, 'anchor_bass_floor', 0.02) or 0.02)
        bass_ceil = float(getattr(self.config.stroke, 'anchor_bass_ceil', 0.22) or 0.22)
        if bass_ceil <= bass_floor:
            bass_ceil = bass_floor + 0.10
        return float(np.clip((activity - bass_floor) / (bass_ceil - bass_floor), 0.0, 1.0))

    def _update_anchor_from_bass_state(self, event: Optional[BeatEvent]) -> float:
        """Drive park/anchor radius from low-bass activity.

        Parking prefers ~0.70 on the horizontal axis. Lower bass frequencies
        nudge it farther out (toward edge), capped at 0.90.
        """
        bass_norm = self._get_anchor_bass_norm(event)
        freq_bias_target = 0.0
        if event is not None:
            freq = float(getattr(event, 'frequency', 0.0) or 0.0)
            if freq > 0.0:
                low = 70.0
                high = 220.0
                freq_bias_target = float(np.clip((high - freq) / max(1e-6, high - low), 0.0, 1.0))

        freq_smooth = 0.30 if freq_bias_target >= self._park_freq_bias else 0.10
        self._park_freq_bias = float(np.clip(
            self._park_freq_bias + ((freq_bias_target - self._park_freq_bias) * freq_smooth),
            0.0,
            1.0,
        ))

        target_radius = 0.70 + (0.18 * self._park_freq_bias) + (0.02 * bass_norm)
        target_radius = float(np.clip(target_radius, 0.65, 0.90))
        smooth = 0.28 if target_radius >= self._park_radius else 0.12
        radius = self._park_radius + ((target_radius - self._park_radius) * smooth)
        self._update_park_anchor_from_radius(radius)
        return bass_norm

    def _get_park_anchor(self) -> Tuple[float, float]:
        """Bottom-center park anchor at dynamic radius."""
        return self._park_alpha, self._park_beta

    def _get_park_phase(self) -> float:
        alpha, beta = self._get_park_anchor()
        phase = float(np.arctan2(alpha, beta))
        if phase < 0:
            phase += 2 * np.pi
        return phase

    def _get_landing_phase(self) -> float:
        return self._get_park_phase() + np.deg2rad(self._landing_offset_degrees)

    def _build_landing_arc_phases(self, current_phase: float, n_points: int, min_turns: float = 1.0) -> np.ndarray:
        """Build arc phases from current phase to landing phase with at least min_turns travel."""
        landing_phase = self._get_landing_phase() / (2 * np.pi)
        turns = float(min_turns)
        while landing_phase + turns <= current_phase:
            turns += 1.0
        return np.linspace(current_phase, landing_phase + turns, n_points, endpoint=False) % 1.0

    def _build_arc_phases_to_target(self, current_phase: float, target_phase: float, n_points: int, min_turns: float = 0.0) -> np.ndarray:
        """Build forward-only arc phases from current phase to explicit target phase."""
        target = float(target_phase % 1.0)
        turns = float(max(0.0, min_turns))
        while target + turns <= current_phase:
            turns += 1.0
        return np.linspace(current_phase, target + turns, n_points, endpoint=True) % 1.0

    def _generate_park_return_arc(self, duration_ms: int = 380) -> bool:
        """Continue current rotational path and land on landing target (off-right)."""
        if self._trajectory is None:
            return False

        n_points = max(8, int(max(120, duration_ms) / 10))
        current_a = float(self.state.alpha)
        current_b = float(self.state.beta)
        current_phase = float(np.arctan2(current_a, current_b))
        if current_phase < 0:
            current_phase += 2 * np.pi
        current_phase /= (2 * np.pi)

        landing_phase = self._get_landing_phase()
        target_phase = landing_phase / (2 * np.pi)
        arc_phases = self._build_arc_phases_to_target(current_phase, target_phase, n_points, min_turns=0.0)

        start_radius = float(np.hypot(current_a, current_b))
        start_radius = float(np.clip(start_radius, 0.12, 1.0))
        target_radius = float(np.clip(self._park_radius, self._park_radius_min, self._park_radius_max))

        alpha_pts = np.zeros(n_points)
        beta_pts = np.zeros(n_points)
        alpha_weight = self.config.alpha_weight
        beta_weight = self.config.beta_weight
        for i, phase in enumerate(arc_phases):
            t = i / max(1, n_points - 1)
            radius = start_radius + ((target_radius - start_radius) * t)
            angle = phase * 2 * np.pi
            alpha_pts[i] = np.sin(angle) * radius * alpha_weight
            beta_pts[i] = np.cos(angle) * radius * beta_weight

        alpha_pts[-1] = float(np.sin(landing_phase) * target_radius)
        beta_pts[-1] = float(np.cos(landing_phase) * target_radius)

        step_durations = self._make_landing_durations(int(max(120, duration_ms)), n_points)
        band_volume = float(self._trajectory.band_volume if self._trajectory is not None else self.get_volume())
        now = time.perf_counter()

        self._trajectory = PlannedTrajectory(
            alpha_points=alpha_pts,
            beta_points=beta_pts,
            step_durations=step_durations,
            n_points=n_points,
            current_index=0,
            band_volume=band_volume,
            start_time=now,
            is_park_return=True,
            original_bpm=0.0,
            beat_target_time=0.0,
        )
        return True

    def _note_motion_block(self, reason: str, **details) -> None:
        """Emit throttled diagnostics when beat motion is suppressed by a gate."""
        now = time.time()
        self._emit_block_summary_if_due(now)
        if reason not in self._block_reason_counts:
            self._block_reason_counts[reason] = 0
        self._block_reason_counts[reason] += 1
        self._blocked_beat_events += 1
        should_log = (
            (reason != self._last_block_reason)
            or ((now - self._last_block_log_time) >= self._block_log_interval_s)
        )
        self._motion_block_active = True
        if not should_log:
            return

        payload = {
            'reason': reason,
            'mode': self._motion_mode,
        }
        payload.update(details)
        log_event("INFO", "StrokeMapper", "Motion blocked", **payload)
        self._last_block_reason = reason
        self._last_block_log_time = now

    def _note_motion_resumed(self, context: str = "") -> None:
        """Emit one-shot diagnostic when motion resumes after being blocked."""
        now = time.time()
        self._emit_block_summary_if_due(now)
        if not self._motion_block_active:
            return
        payload = {'mode': self._motion_mode}
        if context:
            payload['context'] = context
        log_event("INFO", "StrokeMapper", "Motion resumed", **payload)
        self._motion_block_active = False
        self._last_block_reason = ""
        self._motion_resumed_count += 1

    def _emit_block_summary_if_due(self, now: Optional[float] = None) -> None:
        """Emit compact blocker summary once per time window."""
        if now is None:
            now = time.time()
        elapsed = now - self._block_summary_window_start
        if elapsed < self._block_summary_interval_s:
            return

        summary_payload = {
            'window_s': f"{elapsed:.1f}",
            'blocked_events': self._blocked_beat_events,
            'resumed_events': self._motion_resumed_count,
        }
        for reason in self._block_reason_order:
            summary_payload[reason] = self._block_reason_counts.get(reason, 0)
        for reason, count in sorted(self._block_reason_counts.items()):
            if reason not in self._block_reason_order:
                summary_payload[reason] = count

        log_event("INFO", "StrokeMapper", "Motion block summary", **summary_payload)

        self._block_summary_window_start = now
        for key in list(self._block_reason_counts.keys()):
            self._block_reason_counts[key] = 0
        self._motion_resumed_count = 0
        self._blocked_beat_events = 0

    def _log_large_motion_jump(self,
                               prev_alpha: float,
                               prev_beta: float,
                               next_alpha: float,
                               next_beta: float,
                               source: str) -> None:
        """Log potentially visible straight-line jump transitions with context."""
        delta = float(np.hypot(next_alpha - prev_alpha, next_beta - prev_beta))
        if delta < self._jump_log_threshold:
            return

        now = time.time()
        if (now - self._last_jump_log_time) < self._jump_log_interval_s:
            return
        self._last_jump_log_time = now

        traj = self._trajectory
        payload = {
            'source': source,
            'delta': f"{delta:.3f}",
            'from_a': f"{prev_alpha:.2f}",
            'from_b': f"{prev_beta:.2f}",
            'to_a': f"{next_alpha:.2f}",
            'to_b': f"{next_beta:.2f}",
            'mode': self._motion_mode,
            'stroke_ready': bool(self._stroke_ready),
            'last_block_reason': self._last_block_reason or 'none',
            'traj_active': bool(traj is not None and traj.active),
            'traj_park_return': bool(getattr(traj, 'is_park_return', False)) if traj is not None else False,
            'traj_micro': bool(getattr(traj, 'is_micro', False)) if traj is not None else False,
        }
        if traj is not None:
            payload['traj_idx'] = f"{traj.current_index}/{traj.n_points}"
        log_event("WARNING", "StrokeMapper", "Large motion jump", **payload)

    def _should_anti_stop(self, alpha: float, beta: float) -> bool:
        """Return True when motion should not be allowed to fully stall."""
        park_dist = float(np.hypot(alpha - self._park_alpha, beta - self._park_beta))
        at_bottom_park = park_dist <= 0.05
        if at_bottom_park:
            return False

        radius = float(np.hypot(alpha, beta))
        in_upper_half = beta < 0.0
        at_edge = radius >= self._anti_stop_edge_radius
        return bool(in_upper_half or at_edge)

    def _apply_anti_stop_nudge(self,
                               alpha: float,
                               beta: float,
                               reference_radius: Optional[float] = None) -> tuple[float, float]:
        """Apply a tiny forward orbital nudge so the dot never fully freezes."""
        current_radius = float(np.hypot(alpha, beta))
        if current_radius > 0.05:
            angle = float(np.arctan2(alpha, beta))
            if angle < 0:
                angle += 2 * np.pi
            self.state.creep_angle = angle
        else:
            angle = float(self.state.creep_angle)

        angle += self._anti_stop_angle_step
        if angle >= 2 * np.pi:
            angle -= 2 * np.pi
        self.state.creep_angle = angle

        radius = reference_radius if reference_radius is not None else current_radius
        radius = float(np.clip(max(radius, 0.90), 0.85, 1.0))
        nudged_alpha = float(np.sin(angle) * radius)
        nudged_beta = float(np.cos(angle) * radius)
        return nudged_alpha, nudged_beta

    def _update_flux_history(self, event: BeatEvent) -> None:
        now = event.timestamp
        self._flux_history.append((now, event.spectral_flux))
        cutoff = now - self._flux_rise_window_ms / 1000.0
        while self._flux_history and self._flux_history[0][0] < cutoff:
            self._flux_history.popleft()

    def _get_flux_rise_factor(self) -> float:
        if len(self._flux_history) < 2:
            return 0.0
        oldest_flux = self._flux_history[0][1]
        newest_flux = self._flux_history[-1][1]
        rise = max(0.0, newest_flux - oldest_flux)
        return min(1.0, rise / 0.1)

    def _is_center_reset_flux_guard_active(self) -> tuple[bool, float, float]:
        """Return whether center+jitter reset should be held due to flux activity."""
        values = list(self._recent_flux_values)
        if len(values) < 8:
            return False, 0.0, 0.0

        recent_count = min(12, len(values))
        recent = values[-recent_count:]
        recent_avg = float(np.mean(recent))
        recent_delta = float(recent[-1] - recent[0])

        beat_cfg = self.config.beat
        delta_thresh = float(getattr(beat_cfg, 'center_jitter_flux_delta_threshold', 0.20) or 0.20)
        avg_thresh = float(getattr(beat_cfg, 'center_jitter_flux_avg_threshold', 0.25) or 0.25)

        is_active = (recent_delta >= delta_thresh) or (recent_avg >= avg_thresh)
        return bool(is_active), recent_avg, recent_delta

    def _has_recent_beats(self, now: Optional[float] = None, window_s: float = 0.9) -> bool:
        current = now if now is not None else time.perf_counter()
        beat_recent = (
            self._last_any_beat_time > 0
            and (current - self._last_any_beat_time) <= window_s
        )
        reset_hold_active = current < self._tempo_reset_motion_hold_until
        return bool(beat_recent or reset_hold_active)

    def _arm_tempo_reset_motion_hold(self, now: float) -> None:
        self._last_any_beat_time = now
        self._tempo_reset_motion_hold_until = max(
            self._tempo_reset_motion_hold_until,
            now + self._tempo_reset_motion_hold_s,
        )

    def _get_band_duration_scale(self, event: BeatEvent) -> float:
        band = getattr(event, 'beat_band', 'sub_bass')
        return self._band_speed_scale.get(band, 1.0)

    def _get_anchor_phase_for_mode(self, mode: StrokeMode, fallback_phase: float, direction_override: Optional[int] = None) -> float:
        if mode != StrokeMode.TEARDROP:
            return self._get_park_phase()

        current_radius = float(np.hypot(self.state.alpha, self.state.beta))
        recent_beats_active = self._has_recent_beats(window_s=1.2)
        edge_creep_continuation = (
            self.config.creep.enabled
            and not self.state.creep_reset_active
            and self._trajectory is None
            and recent_beats_active
            and current_radius >= 0.72
        )
        if edge_creep_continuation:
            return fallback_phase

        if direction_override in (-1, 1):
            direction = int(direction_override)
        elif mode == StrokeMode.SPIRAL:
            direction = 1 if self._spiral_direction >= 0 else -1
        else:
            direction = 1

        offset = float(np.clip(self._single_anchor_prebottom_offset, 0.05, 0.6))
        base = float(self._single_anchor_bottom_phase)
        return (base - offset) if direction >= 0 else (base + offset)

    def _get_arc_launch_phase(self, mode: StrokeMode) -> float:
        current_radius = float(np.hypot(self.state.alpha, self.state.beta))
        if current_radius > 0.05:
            fallback_phase = float(np.arctan2(self.state.alpha, self.state.beta))
            if fallback_phase < 0:
                fallback_phase += 2 * np.pi
        else:
            fallback_phase = self._get_park_phase()

        parked = (
            abs(self.state.alpha - self._park_alpha) < 0.02
            and abs(self.state.beta - self._park_beta) < 0.02
        )
        recent_beats_active = self._has_recent_beats(window_s=0.9)
        if parked or self.state.creep_reset_active or not recent_beats_active:
            return self._get_park_phase()

        return self._get_anchor_phase_for_mode(mode, fallback_phase)

    def _get_adaptive_beat_divisor(self, event: BeatEvent) -> int:
        """Return beats-per-stroke divisor from tempo.

        Rules:
        - 1 beat/stroke only allowed at very slow BPM (< single_stroke_bpm_cutoff)
        - Otherwise auto-select 2 / 4 / 8 from BPM cutoffs
        - If BPM unavailable, use configured fallback (beats_between_strokes; 2/4/8)
        """
        tempo_bpm = float(getattr(event, 'metronome_bpm', 0.0) or 0.0)
        if tempo_bpm <= 0:
            tempo_bpm = float(getattr(event, 'bpm', 0.0) or 0.0)

        cfg = self.config.stroke
        try:
            fallback_divisor = int(getattr(cfg, 'beats_between_strokes', 2) or 2)
        except Exception:
            fallback_divisor = 2
        if fallback_divisor not in (2, 4, 8):
            fallback_divisor = 2

        single_cutoff = float(getattr(cfg, 'single_stroke_bpm_cutoff', 90.0) or 90.0)
        cutoff_2_to_4 = float(getattr(cfg, 'bpm_cutoff_2_to_4', 60.0) or 60.0)
        cutoff_4_to_8 = float(getattr(cfg, 'bpm_cutoff_4_to_8', 155.0) or 155.0)
        cutoff_bias = float(getattr(cfg, 'cadence_cutoff_bias_bpm', 0.0) or 0.0)
        cutoff_2_to_4 += cutoff_bias
        cutoff_4_to_8 += cutoff_bias
        if cutoff_4_to_8 <= cutoff_2_to_4:
            cutoff_4_to_8 = cutoff_2_to_4 + 1.0

        if tempo_bpm <= 0:
            base_divisor = fallback_divisor
        elif tempo_bpm < single_cutoff:
            base_divisor = 1
        elif tempo_bpm < cutoff_2_to_4:
            base_divisor = 2
        elif tempo_bpm < cutoff_4_to_8:
            base_divisor = 4
        else:
            base_divisor = 8

        if self._learning_enabled:
            hint = int(np.clip(self._learned_divisor_hint, 1, 8))
            if hint in (1, 2, 4, 8):
                if self._is_learning_isolation_active():
                    base_divisor = hint
                else:
                    # Preserve tempo safety by only increasing divisor density reduction.
                    base_divisor = max(base_divisor, hint)

        speed_density = self._combo_scale('combo_speed', 0.65, 1.55)
        sped_divisor = float(base_divisor) / max(speed_density, 1e-6)
        base_divisor = self._snap_divisor(sped_divisor)

        return base_divisor

    def _is_learning_isolation_active(self) -> bool:
        return bool(
            self._learning_enabled
            and self._learning_isolation_mode
            and self._learning_use_fitted_rules
            and self._learning_model_loaded
        )

    def _update_learning_adapter(self, event: BeatEvent) -> None:
        """Adaptive runtime 'teaching' layer from beat-window features.

        Learns in-session motion suggestions while preserving deterministic guards.
        Outputs are bounded and smoothed.
        """
        if not self._learning_enabled or not getattr(event, 'is_beat', False):
            return

        if self.config.stroke.mode == StrokeMode.SIMPLE_CIRCLE and not self._learning_apply_in_circle_mode:
            self._learned_divisor_hint = 1
            self._learned_radius_mult = 1.0
            self._learned_lead_ms = 0.0
            self._learned_sync_size_mult = 1.0
            self._learned_sync_speed_mult = 1.0
            return

        features = getattr(event, 'beat_features', None) or {}
        confidence = float(np.clip(features.get('confidence', getattr(event, 'acf_confidence', 0.0) or 0.0), 0.0, 1.0))
        if confidence < self._learning_min_confidence:
            return

        energy_norm = float(np.clip(features.get('energy_norm', getattr(event, 'intensity', 0.0) or 0.0), 0.0, 1.0))
        flux_norm = float(np.clip(features.get('flux_norm', 0.0), 0.0, 1.0))
        offbeat_score = float(np.clip(features.get('offbeat_score', 1.0 if getattr(event, 'is_syncopated', False) else 0.0), 0.0, 1.0))

        target_divisor = None
        target_radius_mult = None
        target_lead_ms = None
        target_sync_size = None
        target_sync_speed = None

        if self._learning_model_loaded:
            prediction = self._predict_learning_targets(event)
            if prediction:
                cadence_divisor = int(np.clip(prediction.get('beats_between_strokes', 2), 1, 8))
                gate_strictness = float(np.clip(prediction.get('gate_strictness', 0.5), 0.0, 1.0))
                gate_strictness = float(np.clip(gate_strictness * self._learning_no_motion_bias, 0.0, 1.0))
                arc_size = float(np.clip(prediction.get('arc_size', energy_norm), 0.0, 1.0))
                arc_duration = float(np.clip(prediction.get('arc_duration_frac', 1.0), 0.1, 4.0))
                burst_prob = float(np.clip(prediction.get('burst_prob', offbeat_score), 0.0, 1.0))
                jitter_mix = float(np.clip(prediction.get('jitter_mix', 0.0), 0.0, 1.0))

                if gate_strictness > 0.92:
                    cadence_divisor = max(cadence_divisor, 8)
                elif gate_strictness > 0.78:
                    cadence_divisor = max(cadence_divisor, 4)
                target_divisor = cadence_divisor

                target_radius_mult = float(np.clip(0.72 + 0.58 * arc_size, 0.70, 1.30))
                target_lead_ms = float(np.clip((0.35 - gate_strictness) * 10.0 + (1.0 - arc_duration) * 2.0, -8.0, 10.0))

                burst_drive = float(np.clip((0.65 * burst_prob) + (0.35 * jitter_mix), 0.0, 1.0))
                target_sync_size = float(np.clip(1.0 + 0.30 * burst_drive, 1.0, 1.35))
                target_sync_speed = float(np.clip(1.0 + 0.24 * burst_drive, 1.0, 1.25))

        if target_divisor is None:
            quietness = 1.0 - max(energy_norm, flux_norm)
            if quietness > 0.65:
                target_divisor = 4
            elif quietness > 0.45:
                target_divisor = 2
            else:
                target_divisor = 1
            if offbeat_score > 0.60:
                target_divisor = min(target_divisor, 2)

        if target_radius_mult is None:
            target_radius_mult = float(np.clip(0.82 + (0.32 * energy_norm) + (0.16 * flux_norm), 0.75, 1.25))

        mode = self.config.stroke.mode
        preserve_outer_radius = mode in (StrokeMode.SIMPLE_CIRCLE, StrokeMode.SPIRAL, StrokeMode.TEARDROP)
        if preserve_outer_radius:
            target_radius_mult = float(max(1.0, target_radius_mult))

        if target_lead_ms is None:
            target_lead_ms = float(np.clip((0.50 - energy_norm) * 10.0, -6.0, 8.0))

        if target_sync_size is None:
            target_sync_size = float(np.clip(1.0 + 0.35 * offbeat_score, 1.0, 1.35))
        if target_sync_speed is None:
            target_sync_speed = float(np.clip(1.0 + 0.25 * offbeat_score, 1.0, 1.25))

        self._learned_divisor_hint = int(np.clip(round((1.0 - self._learning_strength) * self._learned_divisor_hint + self._learning_strength * target_divisor), 1, 8))
        self._learned_radius_mult += (target_radius_mult - self._learned_radius_mult) * (0.18 * self._learning_strength + 0.05)
        if preserve_outer_radius:
            self._learned_radius_mult = float(np.clip(self._learned_radius_mult, 1.0, 1.25))
        else:
            self._learned_radius_mult = float(np.clip(self._learned_radius_mult, 0.75, 1.25))
        self._learned_lead_ms += (target_lead_ms - self._learned_lead_ms) * (0.16 * self._learning_strength + 0.04)
        self._learned_lead_ms = float(np.clip(self._learned_lead_ms, -8.0, 10.0))

        blend = 0.20 * self._learning_strength + 0.05
        self._learned_sync_size_mult += (target_sync_size - self._learned_sync_size_mult) * blend
        self._learned_sync_speed_mult += (target_sync_speed - self._learned_sync_speed_mult) * blend
        self._learned_sync_size_mult = float(np.clip(self._learned_sync_size_mult, 1.0, 1.35))
        self._learned_sync_speed_mult = float(np.clip(self._learned_sync_speed_mult, 1.0, 1.25))

    def _candidate_learning_model_paths(self) -> list[Path]:
        candidates: list[Path] = []

        if self._learning_rule_fit_path:
            candidates.append(Path(self._learning_rule_fit_path))

        env_path = os.environ.get("BREADBEATS_RULE_FIT_PATH", "").strip()
        if env_path:
            candidates.append(Path(env_path))

        repo_root = Path(__file__).resolve().parent
        candidates.extend(
            [
                Path("D:/breadbeats_datasets/rule_fit.json"),
                repo_root / "datasets" / "rule_fit.json",
                repo_root / "learning" / "rule_fit.json",
            ]
        )

        deduped: list[Path] = []
        seen: set[str] = set()
        for item in candidates:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _try_load_learning_model(self) -> None:
        if not self._learning_enabled or not self._learning_use_fitted_rules:
            return

        for path in self._candidate_learning_model_paths():
            try:
                if not path.exists() or not path.is_file():
                    continue
                payload = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(payload, dict):
                    continue
                if payload.get('status') != 'ok':
                    continue

                feature_columns = payload.get('feature_columns') or []
                normalization = payload.get('normalization') or {}
                models = payload.get('models') or {}
                cadence_rule = payload.get('cadence_rule') or {}
                if not feature_columns or not normalization or not models:
                    continue

                self._learning_feature_columns = [str(name) for name in feature_columns]
                self._learning_norm_mean = {str(k): float(v) for k, v in (normalization.get('mean') or {}).items()}
                self._learning_norm_std = {str(k): max(1e-9, float(v)) for k, v in (normalization.get('std') or {}).items()}
                self._learning_model = models
                self._learning_cadence_rule = cadence_rule
                self._learning_model_loaded = True
                self._learning_model_path = str(path)
                log_event("INFO", "StrokeMapper", "Loaded teaching rule-fit model", path=str(path))
                return
            except Exception as exc:
                log_event("WARN", "StrokeMapper", "Failed loading teaching rule-fit model", path=str(path), error=str(exc))

    def _build_runtime_feature_values(self, event: BeatEvent) -> dict[str, float]:
        features = getattr(event, 'beat_features', None) or {}
        rms = float(max(0.0, features.get('energy_mean', getattr(event, 'peak_energy', 0.0) or 0.0)))
        spectral_flux = float(max(0.0, features.get('flux_mean', getattr(event, 'spectral_flux', 0.0) or 0.0)))
        flux_peak = float(max(spectral_flux, features.get('flux_peak', spectral_flux)))
        flux_delta = float(max(0.0, flux_peak - spectral_flux))

        sub_bass = float(max(0.0, self._sub_bass_energy))
        low_mid = float(max(0.0, self._low_mid_energy))
        mid = float(max(0.0, self._mid_energy))
        high = float(max(0.0, self._high_energy))

        centroid_hz = float(max(0.0, getattr(event, 'frequency', 0.0) or 0.0))
        freq_delta = float(max(0.0, features.get('freq_delta', 0.0) or 0.0))
        bandwidth_hz = freq_delta
        rolloff_hz = float(np.clip(centroid_hz + 0.5 * bandwidth_hz, 0.0, 22050.0))

        energy_norm = float(np.clip(features.get('energy_norm', getattr(event, 'intensity', 0.0) or 0.0), 0.0, 1.0))
        flatness = float(np.clip(0.35 + 0.50 * (1.0 - energy_norm), 0.0, 1.0))

        eps = 1e-9
        return {
            'rms': rms,
            'log_energy': float(np.log10(rms + eps)),
            'spectral_flux': spectral_flux,
            'flux_delta': flux_delta,
            'sub_bass_energy': sub_bass,
            'low_mid_energy': low_mid,
            'mid_energy': mid,
            'high_energy': high,
            'low_high_ratio': float((sub_bass + low_mid + eps) / (high + eps)),
            'spectral_centroid_hz': centroid_hz,
            'spectral_bandwidth_hz': bandwidth_hz,
            'spectral_rolloff_hz': rolloff_hz,
            'spectral_flatness': flatness,
        }

    def _predict_learning_targets(self, event: BeatEvent) -> dict[str, float]:
        if not self._learning_model_loaded:
            return {}

        feature_values = self._build_runtime_feature_values(event)
        normalized: dict[str, float] = {}
        for name in self._learning_feature_columns:
            raw = float(feature_values.get(name, 0.0))
            mean = float(self._learning_norm_mean.get(name, 0.0))
            std = float(max(1e-9, self._learning_norm_std.get(name, 1.0)))
            normalized[name] = (raw - mean) / std

        output: dict[str, float] = {}
        for target_name in ('arc_size', 'arc_duration_frac', 'jitter_mix', 'creep_mix', 'gate_strictness', 'burst_prob'):
            model = self._learning_model.get(target_name) or {}
            intercept = float(model.get('intercept', 0.0) or 0.0)
            coeffs = model.get('coefficients') or {}
            value = intercept
            for feature_name in self._learning_feature_columns:
                value += float(coeffs.get(feature_name, 0.0) or 0.0) * normalized[feature_name]
            output[target_name] = float(value)

        # Clamp modeled targets to valid runtime ranges at prediction edge.
        output['arc_size'] = float(np.clip(output.get('arc_size', 0.0), 0.0, 1.0))
        output['arc_duration_frac'] = float(np.clip(output.get('arc_duration_frac', 1.0), 0.1, 4.0))
        output['jitter_mix'] = float(np.clip(output.get('jitter_mix', 0.0), 0.0, 1.0))
        output['creep_mix'] = float(np.clip(output.get('creep_mix', 0.0), 0.0, 1.0))
        output['gate_strictness'] = float(np.clip(output.get('gate_strictness', 0.5), 0.0, 1.0))
        output['burst_prob'] = float(np.clip(output.get('burst_prob', 0.0), 0.0, 1.0))

        cadence_rule = self._learning_cadence_rule or {}
        mapping = cadence_rule.get('mapping') or {}
        quiet_threshold = float(cadence_rule.get('quiet_threshold', 0.5) or 0.5)
        mid_threshold = float(cadence_rule.get('mid_threshold', 0.7) or 0.7)
        rms_norm = normalized.get('rms', 0.0)
        flux_norm = normalized.get('spectral_flux', 0.0)
        combined = (0.6 * rms_norm) + (0.4 * flux_norm)
        if combined < quiet_threshold:
            output['beats_between_strokes'] = int(mapping.get('quiet', 4) or 4)
        elif combined < mid_threshold:
            output['beats_between_strokes'] = int(mapping.get('mid', 2) or 2)
        else:
            output['beats_between_strokes'] = int(mapping.get('loud', 1) or 1)

        return output

    def _get_downbeat_span_beats(self, event: BeatEvent) -> int:
        """Return downbeat arc span in beats.

        Mode 1 (SIMPLE_CIRCLE): fixed full-measure travel (typically 4 beats).
        Other modes: at least full measure, expanded by applicable beat divisor.
        """
        beats_in_measure = int(getattr(self.config.beat, 'beats_per_measure', 4) or 4)
        beats_in_measure = max(1, beats_in_measure)

        mode = self.config.stroke.mode
        if mode == StrokeMode.SIMPLE_CIRCLE:
            return beats_in_measure

        divisor = self._get_adaptive_beat_divisor(event)
        divisor *= self._get_mode_beats_per_stroke_multiplier(mode)
        return max(beats_in_measure, int(max(1, divisor)))

    def _get_mode_beats_per_stroke_multiplier(self, mode: Optional[StrokeMode] = None) -> int:
        """Return per-mode cadence multiplier for beats-per-stroke.

        Mode 1 (SIMPLE_CIRCLE): 1x
        Mode 2 (SPIRAL): 2x
        Mode 3 (TEARDROP): 2x
        """
        mode_to_use = mode if mode is not None else self.config.stroke.mode
        if mode_to_use in (StrokeMode.SPIRAL, StrokeMode.TEARDROP):
            return 2
        return 1

    def _freq_to_factor(self, freq: float) -> float:
        """Convert frequency -> 0-1 factor.  Lower (bass) -> 0 -> deeper strokes."""
        cfg = self.config.stroke
        low, high = cfg.depth_freq_low, cfg.depth_freq_high
        if freq <= low:
            return 0.0
        elif freq >= high:
            return 1.0
        return (freq - low) / (high - low)

    def _radius_cap_from_depth(self, depth: float, max_cap: float = 1.0) -> float:
        """Compute per-stroke radius cap.

        - `stroke_fullness` sets the baseline max radius (headroom by default).
        - `freq_depth_factor` with higher depth can expand toward `max_cap`.
        """
        cfg = self.config.stroke
        base_cap = float(np.clip(cfg.stroke_fullness, 0.05, max_cap))
        depth_norm = float(np.clip(depth, 0.0, 1.0))
        freq_push = float(np.clip(cfg.freq_depth_factor, 0.0, 1.0)) * depth_norm
        cap = base_cap + (max_cap - base_cap) * freq_push
        return float(np.clip(cap, 0.05, max_cap))

    # ------------------------------------------------------------------
    # Amplitude envelope & mode gate
    # ------------------------------------------------------------------

    def _update_envelope(self, event: BeatEvent) -> None:
        """Track RMS envelope from peak_energy for mode gating."""
        energy = event.peak_energy
        if energy > self._rms_envelope:
            self._rms_envelope += (energy - self._rms_envelope) * self._rms_attack
        else:
            self._rms_envelope += (energy - self._rms_envelope) * self._rms_release

    def _update_stroke_readiness(self, event: BeatEvent) -> None:
        """Determine if strokes should fire based on metronome + traffic light.
        
                Rules (both lights = metronome + traffic):
                    - If metronome has lock confidence (yellow/green), allow strokes
                        regardless of traffic color.
                    - Traffic still boosts confidence/recovery behavior, but no longer
                        hard-blocks strokes while metrics are actively adjusting (red).
        
        Grace period: when conditions drop, strokes continue for 1300ms
        before reverting to jitter. This prevents brief dips from
        interrupting an ongoing stroke pattern.
        
        If no metrics are enabled (no traffic light), only metronome matters.
        """
        acf_conf = getattr(event, 'acf_confidence', 0.0)
        metro_bpm = getattr(event, 'metronome_bpm', 0.0)
        now = time.time()
        tempo_lock_required = bool(getattr(self.config.beat, 'tempo_lock_required', True))
        metro_relaxed_conf = float(np.clip(
            getattr(self.config.beat, 'teaching_metronome_relaxed_confidence', self._metronome_relaxed_confidence)
            or self._metronome_relaxed_confidence,
            0.05,
            0.40,
        ))
        ignore_traffic_lights = bool(getattr(self.config.beat, 'teaching_ignore_traffic_lights', self._ignore_traffic_lights))
        
        metro_green = acf_conf >= 0.25 and metro_bpm > 0
        metro_yellow = acf_conf >= 0.05 and metro_bpm > 0
        metro_relaxed = acf_conf >= metro_relaxed_conf and metro_bpm > 0
        metronome_ready = metro_green if tempo_lock_required else metro_relaxed

        if ignore_traffic_lights:
            conditions_met = metronome_ready
            self._prev_had_any_light = metro_yellow or metro_green
            if conditions_met:
                self._stroke_ready = True
                self._stroke_ready_lost_time = 0.0
            else:
                if self._stroke_ready:
                    if self._stroke_ready_lost_time == 0.0:
                        self._stroke_ready_lost_time = now
                    elapsed_ms = (now - self._stroke_ready_lost_time) * 1000.0
                    if elapsed_ms >= self._stroke_grace_ms:
                        self._stroke_ready = False
                        self._stroke_ready_lost_time = 0.0
            return
        
        # Get traffic light state from audio_engine
        traffic_green = False
        traffic_yellow = False
        traffic_has_metrics = False
        if self.audio_engine and hasattr(self.audio_engine, 'get_metric_states'):
            states = self.audio_engine.get_metric_states()
            if states:
                traffic_has_metrics = True
                all_settled = all(s == 'SETTLED' for s in states.values())
                any_settled = any(s == 'SETTLED' for s in states.values())
                traffic_green = all_settled
                traffic_yellow = any_settled and not all_settled
        
        # Track traffic-was-green state (for recovering from brief dips)
        if traffic_green:
            self._traffic_was_green = True
            self._traffic_left_green_time = 0.0
        elif self._traffic_was_green and not traffic_green:
            if self._traffic_left_green_time == 0.0:
                self._traffic_left_green_time = now
            # Expire after 3s
            if (now - self._traffic_left_green_time) > 3.0:
                self._traffic_was_green = False

        # Track metro-green stable duration
        if metro_green:
            if self._metro_green_since == 0.0:
                self._metro_green_since = now
        else:
            self._metro_green_since = 0.0
        metro_stable_2s = (self._metro_green_since > 0
                           and (now - self._metro_green_since) >= 2.0)

        # Determine current light levels
        has_any_light = metro_yellow or traffic_yellow or metro_green or traffic_green

        if traffic_has_metrics:
            # Rule: both green
            both_green = metro_green and traffic_green
            # Rule: one green + one yellow
            mixed_green_yellow = ((metro_green and traffic_yellow)
                                  or (metro_yellow and traffic_green))
            # Rule: both yellow — only if NOT cold-starting from red/off,
            # OR if beat/downbeat indicator confirms
            both_yellow = metro_yellow and (traffic_yellow or traffic_green is False)
            both_yellow_ok = False
            if metro_yellow and traffic_yellow:
                if self._prev_had_any_light:
                    # Previously had lights on → trust both-yellow
                    both_yellow_ok = True
                elif event.is_beat or getattr(event, 'is_downbeat', False):
                    # Cold start but beat/downbeat indicator confirms → allow
                    both_yellow_ok = True
            # Recovery: traffic was recently green (now yellow) + metronome yellow/green
            option_recovery = (traffic_yellow and self._traffic_was_green
                               and (metro_green or metro_yellow))
            # Fallback: metronome green stable >2s, any traffic state
            option_stable = metro_stable_2s
            # Metronome-first: if metronome is yellow/green, don't hard-block
            # on red traffic while metrics are still hunting.
            option_metronome = metronome_ready
            
            conditions_met = (both_green or mixed_green_yellow
                              or both_yellow_ok or option_recovery
                              or option_stable or option_metronome)
        else:
            conditions_met = metronome_ready

        # Update previous-light tracking for next iteration
        self._prev_had_any_light = has_any_light
        
        if conditions_met:
            # Conditions met — immediately ready, reset lost timer
            self._stroke_ready = True
            self._stroke_ready_lost_time = 0.0
            self._stroke_gate_block_streak = 0
        else:
            # Conditions dropped — start or continue grace period
            if self._stroke_ready:
                # Was ready, just lost it — start grace timer
                if self._stroke_ready_lost_time == 0.0:
                    self._stroke_ready_lost_time = now
                # Check if grace period expired
                elapsed_ms = (now - self._stroke_ready_lost_time) * 1000.0
                if elapsed_ms >= self._stroke_grace_ms:
                    self._stroke_ready = False
                    self._stroke_ready_lost_time = 0.0
                # else: still within grace period, keep _stroke_ready = True
            # else: already not ready, stay not ready

    def _update_motion_mode(self) -> None:
        """Switch between FULL_STROKE and CREEP_MICRO with hysteresis."""
        now = time.time()
        cfg = self.config.stroke
        dwell_bias = float(getattr(cfg, 'full_stroke_dwell_bias', 0.0) or 0.0)
        gate_high = float(cfg.amplitude_gate_high) - dwell_bias
        gate_low = float(cfg.amplitude_gate_low) + dwell_bias
        gate_high = float(np.clip(gate_high, 0.005, 0.95))
        gate_low = float(np.clip(gate_low, 0.001, 0.94))
        if gate_low >= gate_high:
            midpoint = (gate_low + gate_high) * 0.5
            gate_high = min(0.95, midpoint + 0.001)
            gate_low = max(0.001, midpoint - 0.001)
        # Minimum dwell time in a mode before switching (500ms)
        if now - self._mode_switch_time < 0.5:
            return
        old = self._motion_mode
        if self._motion_mode == MotionMode.CREEP_MICRO:
            if self._rms_envelope > gate_high:
                self._motion_mode = MotionMode.FULL_STROKE
                self._mode_switch_time = now
                # Sync creep angle to current position on mode switch
                self._sync_creep_angle_to_position()
        else:
            if self._rms_envelope < gate_low:
                self._motion_mode = MotionMode.CREEP_MICRO
                self._mode_switch_time = now
                self._pending_arc_event = None  # Cancel any deferred arc
                # Sync creep angle to current position on mode switch
                self._sync_creep_angle_to_position()
        if old != self._motion_mode:
            log_event("INFO", "StrokeMapper", "Mode switch",
                      mode=self._motion_mode, envelope=f"{self._rms_envelope:.4f}")

    # ------------------------------------------------------------------
    # Tempo-synced phase
    # ------------------------------------------------------------------

    def _sync_creep_angle_to_position(self) -> None:
        """Sync creep_angle to match current (alpha, beta) position.
        Called on mode transitions and after arc completion to prevent jumps."""
        r = np.sqrt(self.state.alpha**2 + self.state.beta**2)
        if r > 0.05:
            synced = np.arctan2(self.state.alpha, self.state.beta)
            if synced < 0:
                synced += 2 * np.pi
            self.state.creep_angle = synced

    def _advance_phase(self, event: BeatEvent) -> None:
        """Advance the continuous beat phase based on current BPM."""
        now = getattr(event, 'monotonic_timestamp', 0.0) or time.perf_counter()
        dt = now - self._phase_time
        self._phase_time = now

        bpm = self._get_reliable_metronome_bpm(event)
        self._current_bpm = bpm

        if bpm > 0 and dt > 0 and dt < 1.0:
            beats_per_sec = bpm / 60.0
            self._beat_phase += beats_per_sec * dt
            self._beat_phase %= 1.0

    def _get_reliable_metronome_bpm(self, event: Optional[BeatEvent], min_conf: Optional[float] = None) -> float:
        """Return metronome BPM only when confidence is reliable enough for motion timing."""
        if event is None:
            return 0.0

        metro_bpm = float(getattr(event, 'metronome_bpm', 0.0) or 0.0)
        if metro_bpm <= 0.0:
            return 0.0

        if bool(getattr(event, 'tempo_locked', False)):
            return metro_bpm

        conf = float(getattr(event, 'acf_confidence', 0.0) or 0.0)
        threshold = self._metronome_relaxed_confidence if min_conf is None else float(min_conf)
        threshold = float(np.clip(threshold, 0.05, 0.40))
        return metro_bpm if conf >= threshold else 0.0

    # ------------------------------------------------------------------
    # Band energy extraction (for micro-effects)
    # ------------------------------------------------------------------

    def _update_band_energies(self, event: BeatEvent) -> None:
        """Extract band energies from audio_engine for motion and micro-effect scaling."""
        if self.audio_engine and hasattr(self.audio_engine, '_band_energies'):
            energies = self.audio_engine._band_energies
            # Smooth tracking
            alpha = 0.2
            self._sub_bass_energy += (energies.get('sub_bass', 0.0) - self._sub_bass_energy) * alpha
            self._low_mid_energy += (energies.get('low_mid', 0.0) - self._low_mid_energy) * alpha
            self._mid_energy += (energies.get('mid', 0.0) - self._mid_energy) * alpha
            self._high_energy += (energies.get('high', 0.0) - self._high_energy) * alpha

    def _get_low_band_activity(self, event: BeatEvent) -> float:
        """Return current low-frequency activity estimate for stroke gating.

        Primary source: smoothed sub_bass + low_mid energies.
        Fallback: infer minimal low-band activity from beat context when band
        energies are temporarily unavailable.
        """
        activity = float(max(0.0, self._sub_bass_energy + self._low_mid_energy))
        if activity > 1e-6:
            return activity

        beat_band = getattr(event, 'beat_band', '')
        freq = float(getattr(event, 'frequency', 0.0) or 0.0)
        peak = float(getattr(event, 'peak_energy', 0.0) or 0.0)

        if beat_band in ('sub_bass', 'low_mid'):
            return peak * 0.5
        if 30.0 <= freq <= 500.0:
            return peak * 0.35
        return 0.0

    def _get_low_band_gate_status(self, event: BeatEvent, is_downbeat: bool = False) -> tuple[bool, float, float, float]:
        """Evaluate low-band mean + delta/variance gate for beat strokes."""
        cfg = self.config.stroke
        values = list(self._recent_low_band_values)
        if len(values) < 8:
            return False, 0.0, 0.0, 0.0

        window = int(getattr(cfg, 'low_band_window_frames', 18) or 18)
        window = int(np.clip(window, 8, len(values)))
        segment = values[-window:]

        mean_val = float(np.mean(segment))
        delta_val = float(max(segment) - min(segment))
        var_val = float(np.var(segment))

        relax = float(getattr(cfg, 'downbeat_low_band_relax', 0.85) or 0.85) if is_downbeat else 1.0
        relax = float(np.clip(relax, 0.5, 1.0))

        mean_thresh = float(getattr(cfg, 'low_band_activity_threshold', 0.20) or 0.20) * relax
        delta_thresh = float(getattr(cfg, 'low_band_delta_threshold', 0.06) or 0.06) * relax
        var_thresh = float(getattr(cfg, 'low_band_variance_threshold', 0.0015) or 0.0015) * relax

        gate_pass = (mean_val >= mean_thresh) and ((delta_val >= delta_thresh) or (var_val >= var_thresh))
        return bool(gate_pass), mean_val, delta_val, var_val

    def _is_low_band_full_enough(self, event: Optional[BeatEvent] = None, is_downbeat: bool = False) -> bool:
        """Return True when low-band bed is sufficiently present for rotational strokes.

        This helps ignore isolated mid/high peaks (e.g., ~1kHz wah) when bass
        content is not actually filled in.
        """
        cfg = self.config.stroke
        values = list(self._recent_low_band_values)
        if len(values) >= 8:
            window = int(getattr(cfg, 'low_band_window_frames', 18) or 18)
            window = int(np.clip(window, 8, len(values)))
            segment = values[-window:]
            high_values = list(self._recent_high_band_values)
            high_segment = high_values[-window:] if len(high_values) >= window else high_values
            mid_bass_values = list(self._recent_mid_bass_values)
            mid_bass_segment = mid_bass_values[-window:] if len(mid_bass_values) >= window else mid_bass_values

            relax = float(getattr(cfg, 'downbeat_low_band_relax', 0.85) or 0.85) if is_downbeat else 1.0
            relax = float(np.clip(relax, 0.5, 1.0))
            mean_thresh = float(getattr(cfg, 'low_band_activity_threshold', 0.20) or 0.20) * relax
            occ_floor = mean_thresh * 0.70
            occupancy = float(sum(1 for value in segment if value >= occ_floor) / max(1, len(segment)))
            occupancy_thresh = float(getattr(cfg, 'low_band_fullness_occupancy_threshold', 0.62) or 0.62)
            occupancy_thresh = float(np.clip(occupancy_thresh * relax, 0.40, 0.95))

            mean_low = float(np.mean(segment))
            mean_high = float(np.mean(high_segment)) if high_segment else 0.0
            low_high_ratio = mean_low / max(mean_high, 1e-6)
            ratio_min = float(getattr(cfg, 'low_band_to_high_ratio_min', 0.58) or 0.58)
            ratio_min = float(np.clip(ratio_min * relax, 0.25, 2.0))

            high_mean_thresh = float(getattr(cfg, 'high_band_mean_threshold', 0.12) or 0.12)
            high_occ_thresh = float(getattr(cfg, 'high_band_occupancy_threshold', 0.55) or 0.55)
            high_floor = float(getattr(cfg, 'high_band_floor_threshold', 0.06) or 0.06)
            high_occupancy = float(sum(1 for value in high_segment if value >= high_floor) / max(1, len(high_segment))) if high_segment else 0.0
            high_full = bool(
                (mean_high >= (high_mean_thresh * 0.90))
                and (high_occupancy >= (high_occ_thresh * 0.90))
            )

            mid_bass_support_ok = True
            if bool(getattr(cfg, 'mid_bass_support_enabled', True)) and not high_full:
                mean_mid_bass = float(np.mean(mid_bass_segment)) if mid_bass_segment else 0.0
                mid_bass_thresh = float(getattr(cfg, 'mid_bass_activity_threshold', 0.035) or 0.035)
                mid_bass_occ_floor = mid_bass_thresh * 0.80
                mid_bass_occupancy = float(sum(1 for value in mid_bass_segment if value >= mid_bass_occ_floor) / max(1, len(mid_bass_segment))) if mid_bass_segment else 0.0
                mid_bass_occ_thresh = float(getattr(cfg, 'mid_bass_occupancy_threshold', 0.45) or 0.45)
                mid_bass_support_ok = bool(
                    (mean_mid_bass >= mid_bass_thresh)
                    and (mid_bass_occupancy >= mid_bass_occ_thresh)
                )

            return bool(
                (mean_low >= mean_thresh)
                and (occupancy >= occupancy_thresh)
                and (low_high_ratio >= ratio_min)
                and mid_bass_support_ok
            )

        if event is not None:
            activity = self._get_low_band_activity(event)
            mean_thresh = float(getattr(cfg, 'low_band_activity_threshold', 0.20) or 0.20)
            return bool(activity >= mean_thresh)

        return True

    def _passes_dual_band_db_gate(self, event: Optional[BeatEvent] = None) -> bool:
        """Require both sub-bass and high-band activity in dB before firing strokes."""
        if self._learning_enabled and self._learning_relax_phase1_gates:
            return True
        cfg = self.config.stroke
        if not bool(getattr(cfg, 'dual_band_db_gate_enabled', False)):
            return True

        sub_bass_lin = float(max(1e-8, self._sub_bass_energy))
        high_lin = float(max(1e-8, self._high_energy))

        if event is not None:
            freq = float(getattr(event, 'frequency', 0.0) or 0.0)
            peak = float(getattr(event, 'peak_energy', 0.0) or 0.0)
            if sub_bass_lin <= 1e-7 and freq <= 100.0:
                sub_bass_lin = max(sub_bass_lin, peak * 0.60)
            if high_lin <= 1e-7 and freq >= 2000.0:
                high_lin = max(high_lin, peak * 0.50)

        sub_bass_db = float(20.0 * np.log10(max(sub_bass_lin, 1e-8)))
        high_db = float(20.0 * np.log10(max(high_lin, 1e-8)))

        sub_bass_min = float(getattr(cfg, 'dual_band_sub_bass_db_min', -15.0) or -15.0)
        high_min = float(getattr(cfg, 'dual_band_high_db_min', -30.0) or -30.0)

        if not (sub_bass_db >= sub_bass_min and high_db >= high_min):
            return False

        if not bool(getattr(cfg, 'high_tip_fullness_enabled', True)):
            return True

        tip_freq_low_hz = float(
            getattr(
                cfg,
                'high_tip_freq_low_hz',
                getattr(cfg, 'high_tip_freq_hz', 3500.0) or 3500.0,
            )
            or 3500.0
        )
        tip_freq_high_hz = float(getattr(cfg, 'high_tip_freq_high_hz', 16000.0) or 16000.0)
        if tip_freq_high_hz <= tip_freq_low_hz:
            tip_freq_high_hz = tip_freq_low_hz + 1000.0
        tip_db_min = float(getattr(cfg, 'high_tip_db_min', -28.0) or -28.0)
        tip_occ_thresh = float(getattr(cfg, 'high_tip_occupancy_threshold', 0.50) or 0.50)
        tip_occ_thresh = float(np.clip(tip_occ_thresh, 0.10, 0.95))
        tip_lin = float(10.0 ** (tip_db_min / 20.0))

        values = list(self._recent_high_band_values)
        window = int(getattr(cfg, 'high_band_window_frames', 18) or 18)
        window = int(np.clip(window, 1, len(values))) if values else 0
        segment = values[-window:] if window > 0 else []

        mean_high = float(np.mean(segment)) if segment else 0.0
        occupancy = float(sum(1 for value in segment if value >= tip_lin) / max(1, len(segment))) if segment else 0.0

        dominant_tip = False
        if event is not None:
            freq = float(getattr(event, 'frequency', 0.0) or 0.0)
            peak = float(getattr(event, 'peak_energy', 0.0) or 0.0)
            dominant_tip = bool((tip_freq_low_hz <= freq <= tip_freq_high_hz) and peak >= (tip_lin * 0.8))

        return bool(dominant_tip or ((mean_high >= (tip_lin * 0.85)) and (occupancy >= tip_occ_thresh)))

    def _is_mid_trigger_blocked(self, event: BeatEvent) -> bool:
        """Return True when beat/downbeat trigger lies in blocked mid range."""
        if self._learning_enabled and self._learning_relax_phase1_gates:
            return False
        cfg = self.config.stroke
        if not bool(getattr(cfg, 'block_mid_trigger_range_enabled', False)):
            return False

        low_hz = float(getattr(cfg, 'block_mid_trigger_low_hz', 100.0) or 100.0)
        high_hz = float(getattr(cfg, 'block_mid_trigger_high_hz', 2000.0) or 2000.0)
        if high_hz <= low_hz:
            high_hz = low_hz + 1.0

        freq = float(getattr(event, 'frequency', 0.0) or 0.0)
        return bool(low_hz <= freq <= high_hz)

    def _get_overall_amp_fill_required_base(self, phase: str) -> float:
        cfg = self.config.stroke
        global_scale = float(np.clip(getattr(cfg, 'overall_amp_fill_required_scale', 1.0) or 1.0, 0.05, 20.0))
        reaction_scale = self._combo_scale('combo_reaction', 0.60, 1.70)
        global_scale = float(np.clip(global_scale * reaction_scale, 0.05, 20.0))
        if phase == 'syncopation':
            required = float(getattr(cfg, 'syncopation_overall_amp_fill_required', 0.12) or 0.12)
            if required >= 0.70:
                required = 0.12
            return float(np.clip(required * global_scale, 0.0, 1.0))
        if phase == 'downbeat':
            required = float(getattr(cfg, 'downbeat_overall_amp_fill_required', 0.08) or 0.08)
            if required >= 0.60:
                required = 0.08
            return float(np.clip(required * global_scale, 0.0, 1.0))
        required = float(getattr(cfg, 'beat_overall_amp_fill_required', 0.10) or 0.10)
        if required >= 0.70:
            required = 0.10
        return float(np.clip(required * global_scale, 0.0, 1.0))

    def _update_auto_fill_required(self, phase: str, fill_pass: bool) -> None:
        if not self._auto_fill_enabled:
            return
        phase_state = self._auto_fill_state.get(phase)
        if phase_state is None:
            phase_state = {'ema': self._auto_fill_target_pass_rate, 'offset': 0.0}
            self._auto_fill_state[phase] = phase_state

        pass_value = 1.0 if fill_pass else 0.0
        ema_prev = float(phase_state.get('ema', self._auto_fill_target_pass_rate))
        ema_now = ema_prev + (pass_value - ema_prev) * self._auto_fill_ema_alpha
        phase_state['ema'] = float(np.clip(ema_now, 0.0, 1.0))

        error = phase_state['ema'] - self._auto_fill_target_pass_rate
        if abs(error) <= self._auto_fill_deadband:
            return

        normalized = abs(error) / max(self._auto_fill_deadband, 1e-6)
        step = self._auto_fill_step * min(2.0, normalized)
        offset = float(phase_state.get('offset', 0.0))
        if error > 0.0:
            offset += step
        else:
            offset -= step
        phase_state['offset'] = float(np.clip(offset, -self._auto_fill_max_offset, self._auto_fill_max_offset))

    def _maybe_log_auto_fill_status(self, phase: str, fill_ratio: float, fill_required: float, fill_pass: bool) -> None:
        if not self._auto_fill_enabled:
            return
        now = time.time()
        if (now - self._auto_fill_last_log_time) < self._auto_fill_log_interval_s:
            return

        def _phase_payload(name: str) -> dict:
            state = self._auto_fill_state.get(name) or {}
            required_now = self._get_overall_amp_fill_required(name)
            return {
                f'{name}_required': f"{required_now:.3f}",
                f'{name}_ema': f"{float(state.get('ema', self._auto_fill_target_pass_rate)):.3f}",
                f'{name}_offset': f"{float(state.get('offset', 0.0)):.3f}",
            }

        payload = {
            'phase': str(phase),
            'fill_ratio': f"{fill_ratio:.3f}",
            'fill_required_now': f"{fill_required:.3f}",
            'fill_pass': bool(fill_pass),
            'target_pass_rate': f"{self._auto_fill_target_pass_rate:.3f}",
        }
        payload.update(_phase_payload('beat'))
        payload.update(_phase_payload('downbeat'))
        payload.update(_phase_payload('syncopation'))
        log_event("INFO", "StrokeMapper", "Auto fill adaptation", **payload)
        self._auto_fill_last_log_time = now

    def _get_overall_amp_fill_required(self, phase: str) -> float:
        base_required = self._get_overall_amp_fill_required_base(phase)
        if not self._auto_fill_enabled:
            return base_required

        phase_state = self._auto_fill_state.get(phase)
        offset = float((phase_state or {}).get('offset', 0.0))
        required = base_required + offset
        return float(np.clip(required, self._auto_fill_min_required, self._auto_fill_max_required))

    def _get_spectrum_fill_ratio(self, target: float, phase: str = 'beat') -> float:
        """Return fraction of active spectrum bins above target-normalized amplitude."""
        if not self.audio_engine or not hasattr(self.audio_engine, 'get_spectrum'):
            return 0.0

        spectrum = self.audio_engine.get_spectrum()
        if spectrum is None or len(spectrum) == 0:
            return 0.0

        magnitudes = np.abs(np.asarray(spectrum, dtype=float))
        peak = float(np.max(magnitudes)) if magnitudes.size > 0 else 0.0
        if peak <= 1e-9:
            return 0.0

        n_bins = int(magnitudes.size)
        low_attr = f"{phase}_fill_bin_low"
        high_attr = f"{phase}_fill_bin_high"
        low_bin = int(float(getattr(self.config.stroke, low_attr, 0) or 0))
        high_default = max(0, n_bins - 1)
        high_bin = int(float(getattr(self.config.stroke, high_attr, high_default) or high_default))
        low_bin = int(np.clip(low_bin, 0, max(0, n_bins - 1)))
        high_bin = int(np.clip(high_bin, 0, max(0, n_bins - 1)))
        if high_bin < low_bin:
            high_bin = low_bin

        magnitudes = magnitudes[low_bin:high_bin + 1]
        if magnitudes.size == 0:
            return 0.0

        norm = magnitudes / peak
        threshold = float(np.clip(target, 0.0, 1.0))
        active_floor = float(np.clip(getattr(self.config.stroke, 'overall_amp_fill_active_floor', 0.02) or 0.02, 0.0, 1.0))
        active_bins = norm >= active_floor
        if not np.any(active_bins):
            return 0.0
        active = norm[active_bins]
        return float(np.sum(active >= threshold) / max(1, active.size))

    def _passes_overall_amp_fill_gate(self, event: BeatEvent, phase: str) -> tuple[bool, float, float, float, float]:
        """Gate beat/downbeat/syncopation strokes by overall amplitude + spectrum fill."""
        cfg = self.config.stroke
        if not bool(getattr(cfg, 'overall_amp_fill_gate_enabled', True)):
            return True, 0.0, 1.0, 0.0, 0.0

        target = float(np.clip(getattr(cfg, 'overall_amp_fill_target', 0.5) or 0.5, 0.0, 1.0))
        tolerance = float(np.clip(abs(getattr(cfg, 'overall_amp_fill_tolerance', 0.5) or 0.5), 0.0, 1.0))
        min_amp = float(np.clip(target - tolerance, 0.0, 1.0))

        overall_amp = float(np.clip(getattr(event, 'intensity', 0.0) or 0.0, 0.0, 1.0))
        amp_pass = overall_amp >= min_amp

        beat_cfg = self.config.beat
        quiet_flux_thresh = float(cfg.flux_threshold) * float(cfg.silence_flux_multiplier)
        quiet_energy_thresh = float(beat_cfg.peak_floor) * float(cfg.silence_energy_multiplier)
        near_silence_flux_thresh = quiet_flux_thresh * 1.35
        near_silence_energy_thresh = quiet_energy_thresh * 1.35
        has_silence_metrics = hasattr(event, 'spectral_flux') and hasattr(event, 'peak_energy')
        is_near_silence = bool(
            has_silence_metrics
            and float(getattr(event, 'spectral_flux', 0.0) or 0.0) < near_silence_flux_thresh
            and float(getattr(event, 'peak_energy', 0.0) or 0.0) < near_silence_energy_thresh
        )

        fill_required = self._get_overall_amp_fill_required(phase)
        if is_near_silence:
            fill_required = max(fill_required, self._get_overall_amp_fill_required_base(phase))
        fill_ratio = self._get_spectrum_fill_ratio(target, phase)
        fill_pass = fill_ratio >= fill_required
        if not is_near_silence:
            self._update_auto_fill_required(phase, fill_pass)
        self._maybe_log_auto_fill_status(phase, fill_ratio, fill_required, fill_pass)

        return bool(amp_pass and fill_pass), overall_amp, fill_ratio, min_amp, fill_required

    def _get_high_band_activity(self, event: BeatEvent) -> float:
        """Return current upper-range activity estimate (mid + high)."""
        cfg = self.config.stroke
        include_mid = bool(getattr(cfg, 'high_band_include_mid', True))
        activity = float(max(0.0, (self._mid_energy + self._high_energy) if include_mid else self._high_energy))
        if activity > 1e-6:
            return activity

        beat_band = getattr(event, 'beat_band', '')
        freq = float(getattr(event, 'frequency', 0.0) or 0.0)
        peak = float(getattr(event, 'peak_energy', 0.0) or 0.0)

        if beat_band in (('mid', 'high') if include_mid else ('high',)):
            return peak * 0.5
        if freq >= (500.0 if include_mid else 2000.0):
            return peak * 0.35
        return 0.0

    def _get_mid_bass_activity(self, event: BeatEvent) -> float:
        """Return current 200-400Hz support activity estimate."""
        cfg = self.config.stroke
        low_hz = float(getattr(cfg, 'mid_bass_freq_low_hz', 200.0) or 200.0)
        high_hz = float(getattr(cfg, 'mid_bass_freq_high_hz', 400.0) or 400.0)
        if high_hz <= low_hz:
            high_hz = low_hz + 1.0

        freq = float(getattr(event, 'frequency', 0.0) or 0.0)
        peak = float(getattr(event, 'peak_energy', 0.0) or 0.0)
        beat_band = getattr(event, 'beat_band', '')

        if low_hz <= freq <= high_hz:
            return peak * 0.60
        if beat_band == 'low_mid' and (low_hz * 0.75) <= freq <= (high_hz * 1.25):
            return peak * 0.40
        return 0.0

    def _get_high_band_presence_status(self, is_downbeat: bool = False) -> tuple[bool, float, float, float, float]:
        """Evaluate upper-range filled+active presence gate."""
        cfg = self.config.stroke
        values = list(self._recent_high_band_values)
        if len(values) < 8:
            return False, 0.0, 0.0, 0.0, 0.0

        window = int(getattr(cfg, 'high_band_window_frames', 18) or 18)
        window = int(np.clip(window, 8, len(values)))
        segment = values[-window:]

        mean_val = float(np.mean(segment))
        delta_val = float(max(segment) - min(segment))
        var_val = float(np.var(segment))

        relax = float(getattr(cfg, 'downbeat_high_band_relax', 0.90) or 0.90) if is_downbeat else 1.0
        relax = float(np.clip(relax, 0.5, 1.0))

        mean_thresh = float(getattr(cfg, 'high_band_mean_threshold', 0.12) or 0.12) * relax
        floor_thresh = float(getattr(cfg, 'high_band_floor_threshold', 0.06) or 0.06) * relax
        occ_thresh = float(getattr(cfg, 'high_band_occupancy_threshold', 0.55) or 0.55) * relax
        delta_thresh = float(getattr(cfg, 'high_band_delta_threshold', 0.05) or 0.05) * relax
        var_thresh = float(getattr(cfg, 'high_band_variance_threshold', 0.0010) or 0.0010) * relax

        occupancy = float(sum(1 for value in segment if value >= floor_thresh) / max(1, len(segment)))
        gate_pass = (
            (mean_val >= mean_thresh)
            and (occupancy >= occ_thresh)
            and ((delta_val >= delta_thresh) or (var_val >= var_thresh))
        )
        return bool(gate_pass), mean_val, occupancy, delta_val, var_val

    def _get_high_band_pattern_status(self, is_downbeat: bool = False) -> tuple[bool, int, int]:
        """Evaluate recent beat-wise upper-band hit pattern gate."""
        cfg = self.config.stroke
        hits = list(self._recent_high_band_beat_hits)
        if not hits:
            return False, 0, 0

        relax = float(getattr(cfg, 'downbeat_high_band_relax', 0.90) or 0.90) if is_downbeat else 1.0
        relax = float(np.clip(relax, 0.5, 1.0))

        window = int(getattr(cfg, 'high_band_pattern_window_beats', 5) or 5)
        window = int(np.clip(window, 1, len(hits)))
        segment = hits[-window:]
        hit_count = int(sum(1 for value in segment if value))

        min_hits = int(getattr(cfg, 'high_band_pattern_min_hits', 3) or 3)
        min_hits = int(np.clip(round(min_hits * relax), 1, window))
        return bool(hit_count >= min_hits), hit_count, window

    def _update_bass_jitter_drive(self, event: BeatEvent) -> None:
        """Update jitter speed multiplier from bass z-score context + pitch.

        Higher bass pitch -> faster jitter, lower bass pitch -> slower jitter.
        Uses sub_bass/low_mid fired bands when available, with smoothing
        to avoid twitchy frame-to-frame changes.
        """
        fired_bands = set(getattr(event, 'fired_bands', None) or [])
        beat_band = getattr(event, 'beat_band', '')

        has_bass_context = (
            'sub_bass' in fired_bands
            or 'low_mid' in fired_bands
            or beat_band in ('sub_bass', 'low_mid')
        )

        freq = float(getattr(event, 'frequency', 0.0) or 0.0)
        if (not has_bass_context
                and 30.0 <= freq <= 220.0
                and getattr(event, 'peak_energy', 0.0) > 0.001):
            has_bass_context = True

        if has_bass_context:
            bass_low_hz = 30.0
            bass_high_hz = 220.0
            bass_freq = np.clip(freq, bass_low_hz, bass_high_hz)
            pitch_norm = (bass_freq - bass_low_hz) / (bass_high_hz - bass_low_hz)
            # Bass-speed modulation depth range requested: 0.03 .. 0.075.
            # Lower bass -> slower jitter, higher bass -> faster jitter.
            depth = 0.03 + (0.045 * pitch_norm)
            centered = (pitch_norm * 2.0) - 1.0
            target_mult = 1.0 + (centered * depth)  # ~0.97..1.075
            smooth = self._bass_jitter_attack
        else:
            target_mult = 1.0
            smooth = self._bass_jitter_release

        self._bass_jitter_speed_mult += (target_mult - self._bass_jitter_speed_mult) * smooth
        self._bass_jitter_speed_mult = float(np.clip(self._bass_jitter_speed_mult, 0.92, 1.10))

    def _get_scheduled_lead_seconds(self) -> float:
        """Return configured pre-landing lead offset in seconds."""
        lead_ms = float(getattr(self.config.beat, 'scheduled_lead_ms', 0.0) or 0.0)
        lead_ms = float(np.clip(lead_ms, 0.0, 200.0))
        speed_target_scale = self._combo_scale('combo_speed', 0.75, 1.40)
        lead_ms *= speed_target_scale
        return lead_ms / 1000.0

    def _get_effective_lead_seconds(self) -> float:
        """Return bounded lead offset with adaptive trim to prevent drift buildup."""
        base_ms = float(getattr(self.config.beat, 'scheduled_lead_ms', 0.0) or 0.0)
        base_ms = float(np.clip(base_ms, 0.0, 200.0))
        speed_target_scale = self._combo_scale('combo_speed', 0.75, 1.40)
        base_ms *= speed_target_scale
        if self._is_learning_isolation_active():
            effective_ms = float(np.clip(base_ms + self._learned_lead_ms, 0.0, 220.0))
            return effective_ms / 1000.0
        min_trim = -min(30.0, base_ms)
        max_trim = self._lead_trim_limit_ms
        self._lead_trim_ms = float(np.clip(self._lead_trim_ms, min_trim, max_trim))
        effective_ms = float(np.clip(base_ms + self._lead_trim_ms + self._learned_lead_ms, 0.0, 220.0))
        self._lead_trim_ms *= 0.985
        return effective_ms / 1000.0

    def _update_lead_trim_from_landing(self, landing_error_ms: float) -> None:
        """Adaptive trim so predictive lead stays near a small, bounded early bias."""
        if not np.isfinite(landing_error_ms):
            return
        if abs(landing_error_ms) > 220.0:
            return

        control_error = landing_error_ms - self._lead_target_error_ms
        delta = float(np.clip(control_error * 0.18, -6.0, 6.0))
        self._lead_trim_ms += delta

        base_ms = float(getattr(self.config.beat, 'scheduled_lead_ms', 0.0) or 0.0)
        base_ms = float(np.clip(base_ms, 0.0, 200.0))
        min_trim = -min(30.0, base_ms)
        max_trim = self._lead_trim_limit_ms
        self._lead_trim_ms = float(np.clip(self._lead_trim_ms, min_trim, max_trim))

    def _adjust_predicted_target(self, predicted: float, now: float) -> float:
        """Shift predicted beat target earlier by configured lead time."""
        target = predicted - self._get_effective_lead_seconds()
        return target if target > now else 0.0

    def _plan_lazy_timing(self,
                          now: float,
                          nominal_duration_ms: int,
                          min_interval_ms: float,
                          predicted_target_time: float = 0.0) -> tuple[int, float, float]:
        """Plan arc timing with continuous speed scaling (no hard stop holds)."""
        nominal_ms = float(max(min_interval_ms, nominal_duration_ms))
        duration_ms = int(nominal_ms)
        start_time = now
        beat_target_time = 0.0

        if predicted_target_time > now:
            window_ms = max(1.0, (predicted_target_time - now) * 1000.0)
            scale = window_ms / max(1e-6, nominal_ms)
            scale = float(np.clip(scale, self._timing_scale_min, self._timing_scale_max))
            duration_ms = int(max(min_interval_ms, nominal_ms * scale))
            beat_target_time = now + (duration_ms / 1000.0)
        else:
            beat_target_time = now + (duration_ms / 1000.0)

        return duration_ms, start_time, beat_target_time

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_beat(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """
        Process a beat event and return a stroke command.

        Behavioral modes:
          FULL_STROKE  (high amplitude) -> tempo-synced full circle arc per beat
          CREEP_MICRO  (low amplitude)  -> creep around edge, micro-effects on beats

        Returns:
            TCodeCommand if a stroke should be sent, None otherwise.
        """
        now = getattr(event, 'monotonic_timestamp', 0.0) or time.perf_counter()
        cfg = self.config.stroke
        beat_cfg = self.config.beat
        min_interval_ms = self._effective_min_interval_ms()

        if bool(getattr(event, 'tempo_reset', False)):
            self._arm_tempo_reset_motion_hold(now)

        # ===== SPECTRUM-TUNABLE MOTION GATE =====
        # Uses a configurable frequency cutoff over a COMBINATION of sources:
        # - bands that fired this frame (fired_bands)
        # - current primary beat band (beat_band)
        # Gate applies only when strict_bass_motion_gate_enabled is True.
        # Set motion_freq_cutoff <= 0 to disable cutoff filtering while strict mode is on.
        _BAND_LOWER_HZ = {'sub_bass': 30, 'low_mid': 100, 'mid': 500, 'high': 2000}
        cutoff = float(getattr(beat_cfg, 'motion_freq_cutoff', 0.0))
        strict_gate_enabled = bool(getattr(beat_cfg, 'strict_bass_motion_gate_enabled', True))
        fired_bands = set(getattr(event, 'fired_bands', None) or [])
        primary_band = getattr(event, 'beat_band', '')
        candidate_bands = set(fired_bands)
        if primary_band:
            candidate_bands.add(primary_band)
        if not strict_gate_enabled:
            bass_motion_allowed = True
        elif cutoff <= 0:
            bass_motion_allowed = True
        else:
            bass_motion_allowed = any(_BAND_LOWER_HZ.get(b, 99999) < cutoff for b in candidate_bands)

        # Update continuous trackers
        self._update_flux_history(event)
        self._update_envelope(event)
        self._update_motion_mode()
        self._update_stroke_readiness(event)
        self._advance_phase(event)
        self._update_band_energies(event)
        self._update_bass_jitter_drive(event)
        self._update_learning_adapter(event)

        # ===== LOW-BAND DROP FALLBACK =====
        # Track recent low-band activity; if it drops sharply from a
        # high-activity state, force back to creep mode.
        self._recent_flux_values.append(event.spectral_flux)
        low_band_activity = self._get_low_band_activity(event)
        high_band_activity = self._get_high_band_activity(event)
        mid_bass_activity = self._get_mid_bass_activity(event)
        self._recent_low_band_values.append(low_band_activity)
        self._recent_high_band_values.append(high_band_activity)
        self._recent_mid_bass_values.append(mid_bass_activity)
        recent_beats_active = self._has_recent_beats(now=now, window_s=0.9)
        if len(self._recent_flux_values) >= 30:
            if bool(getattr(cfg, 'low_band_drop_guard_enabled', True)):
                recent_avg = sum(list(self._recent_low_band_values)[-15:]) / 15.0
                older_avg = sum(list(self._recent_low_band_values)[:15]) / 15.0
                flux_drop_ratio = cfg.flux_drop_ratio if hasattr(cfg, 'flux_drop_ratio') else 0.25
                min_high_band = float(getattr(cfg, 'low_band_activity_threshold', 0.20) or 0.20)
                if older_avg >= min_high_band and recent_avg < older_avg * flux_drop_ratio and not recent_beats_active:
                    if self._motion_mode == MotionMode.FULL_STROKE:
                        self._motion_mode = MotionMode.CREEP_MICRO
                        self._mode_switch_time = now
                        self._trajectory = None
                        self._pending_arc_event = None
                        self._sync_creep_angle_to_position()
                        log_event("INFO", "StrokeMapper", "Flux drop → creep fallback",
                                  recent=f"{recent_avg:.4f}", older=f"{older_avg:.4f}")

        # ===== NO-BEAT TIMEOUT =====
        # Track beat liveness from any detected beat (ungated), and
        # separately track last confirmed beat used for stroke-quality diagnostics.
        if event.is_beat:
            self._last_any_beat_time = now

        # Track last confirmed beat (stroke_ready + bass gate + is_beat)
        if event.is_beat and self._stroke_ready and bass_motion_allowed:
            self._last_confirmed_beat_time = now
        # If no confirmed beat for 2s, complete the current arc by landing at park
        # instead of abruptly canceling motion.
        if (self._last_any_beat_time > 0
                and (now - self._last_any_beat_time) > self._no_beat_timeout_s
                and self._trajectory is not None):
            hold_center_reset = False
            if bool(getattr(beat_cfg, 'center_jitter_flux_guard_enabled', False)):
                hold_center_reset, recent_avg, recent_delta = self._is_center_reset_flux_guard_active()
                if hold_center_reset:
                    if (now - self._last_center_guard_log_time) >= 1.0:
                        self._last_center_guard_log_time = now
                        log_event(
                            "INFO",
                            "StrokeMapper",
                            "No-beat timeout held by flux guard",
                            recent_avg=f"{recent_avg:.4f}",
                            recent_delta=f"{recent_delta:.4f}",
                        )

            if not hold_center_reset:
                self._pending_arc_event = None
                if not getattr(self._trajectory, 'is_park_return', False):
                    transitioned = self._generate_park_return_arc()
                    if transitioned:
                        self._locked_anchor = None
                        log_event("INFO", "StrokeMapper", "No-beat timeout → arc-to-park")
                    else:
                        self._trajectory = None
                        self._locked_anchor = None
                        if not self.state.creep_reset_active:
                            self.state.creep_reset_active = True
                            self.state.creep_reset_start_time = now
                        log_event("INFO", "StrokeMapper", "No-beat timeout → park+jitter")

        # ===== SILENCE FADE-OUT =====
        quiet_flux_thresh = cfg.flux_threshold * cfg.silence_flux_multiplier
        quiet_energy_thresh = beat_cfg.peak_floor * cfg.silence_energy_multiplier
        fade_duration = 2.0
        silence_reset_threshold = beat_cfg.silence_reset_ms / 1000.0
        consecutive_silent_required = 10

        is_truly_silent = (event.spectral_flux < quiet_flux_thresh
                           and event.peak_energy < quiet_energy_thresh)
        if is_truly_silent:
            self._consecutive_silent_count += 1
            if self._consecutive_silent_count >= consecutive_silent_required:
                if self._fade_intensity > 0.0:
                    if self._last_quiet_time == 0.0:
                        self._last_quiet_time = now
                    elapsed = now - self._last_quiet_time
                    self._fade_intensity = max(0.0, 1.0 - (elapsed / fade_duration))
                    if self.audio_engine and elapsed > silence_reset_threshold and self._silence_reset_armed:
                        self.audio_engine.reset_tempo_tracking()
                        self._locked_anchor = None  # unlock anchor for next song/section
                        self._silence_reset_armed = False
                else:
                    self._fade_intensity = 0.0
                    self._was_silent = True
        else:
            self._consecutive_silent_count = 0
            self._silence_reset_armed = True
            # Detect transition from silence → sound: trigger post-silence volume ramp
            if self._was_silent and self._fade_intensity < 0.5:
                self._post_silence_ramp_active = True
                self._post_silence_ramp_start = now
                self._was_silent = False
                log_event("INFO", "StrokeMapper", "Post-silence volume ramp started",
                          reduction=f"{cfg.post_silence_vol_reduction:.0%}",
                          duration=f"{cfg.post_silence_ramp_seconds:.1f}s")
            self._fade_intensity = min(1.0, self._fade_intensity + 0.1)
            self._last_quiet_time = 0.0

        # Track idle time
        if event.is_beat:
            self.state.idle_time = 0.0
            self.state.last_beat_time = now
        else:
            self.state.idle_time = now - self.state.last_beat_time if self.state.last_beat_time > 0 else 0.0

        # ===== FLUX FACTOR (for stroke scaling) =====
        if event.is_beat and bass_motion_allowed:
            flux_ratio = event.spectral_flux / max(cfg.flux_threshold, 0.001)
            flux_ratio = np.clip(flux_ratio, 0.2, 3.0)
            base_factor = 0.5 + (flux_ratio / 3.0)
            depth_flux_scale = self._combo_scale('combo_depth', 0.40, 1.80)
            scaling_weight = float(np.clip(cfg.flux_scaling_weight * depth_flux_scale, 0.0, 3.0))
            self._flux_stroke_factor = 1.0 + (base_factor - 1.0) * scaling_weight

        # ===== DISPATCH by behavioral mode =====
        # ===== SYNCOPATION: off-beat "and" onset detected =====
        # If metronome detects an off-beat raw onset between beats, fire a
        # 2x-speed full-circle "double" arc for the duh-DUH effect.
        is_syncopated = getattr(event, 'is_syncopated', False)
        metro_bpm = getattr(event, 'metronome_bpm', 0.0)
        if (is_syncopated and bass_motion_allowed
            and self._motion_mode == MotionMode.FULL_STROKE
            and self._is_low_band_full_enough(event)
            and self._passes_dual_band_db_gate(event)):
            # BPM limit from config
            bpm_limit = beat_cfg.syncopation_bpm_limit if hasattr(beat_cfg, 'syncopation_bpm_limit') else 160.0
            if metro_bpm > bpm_limit:
                pass
            elif self._trajectory is None or self._trajectory.finished:
                time_since_stroke = (now - self.state.last_stroke_time) * 1000
                if time_since_stroke >= min_interval_ms * 0.5:
                    sync_gate_pass, sync_amp, sync_fill, sync_min_amp, sync_fill_req = self._passes_overall_amp_fill_gate(event, 'syncopation')
                    if not sync_gate_pass:
                        self._note_motion_block(
                            "overall_amp_fill_gate",
                            phase="syncopation",
                            amp=f"{sync_amp:.3f}",
                            amp_min=f"{sync_min_amp:.3f}",
                            fill=f"{sync_fill:.3f}",
                            fill_required=f"{sync_fill_req:.3f}",
                        )
                        cmd = self._generate_idle_motion(event)
                        return self._apply_fade(cmd)
                    cmd = self._generate_syncopated_stroke(event)
                    self._note_motion_resumed("syncopation")
                    return self._apply_fade(cmd)

        # ===== NOISE-PRIMARY MODE =====
        # When enabled: noise (flux spike) fires strokes immediately,
        # and the metronome only validates timing (the reverse of default).
        if (cfg.noise_primary_mode
                and not event.is_beat
                and cfg.noise_burst_enabled
                and self._motion_mode == MotionMode.FULL_STROKE
            and self._is_low_band_full_enough(event)
            and self._passes_dual_band_db_gate(event)
                and (self._trajectory is None or self._trajectory.finished)):
            texture_burst = self._combo_scale('combo_texture', 0.60, 1.70)
            noise_mult = float(getattr(cfg, 'noise_burst_flux_multiplier', 2.0) or 2.0)
            noise_thresh = cfg.flux_threshold * (noise_mult / max(texture_burst, 1e-6))
            if event.spectral_flux >= noise_thresh:
                time_since_stroke = (now - self.state.last_stroke_time) * 1000
                if time_since_stroke >= min_interval_ms * 0.4:
                    # In noise-primary mode, fire a FULL beat stroke (not a burst)
                    # using the metronome BPM for duration if available
                    cmd = self._generate_beat_stroke(event)
                    self._note_motion_resumed("noise_primary")
                    return self._apply_fade(cmd)

        # ===== NOISE BURST (non-primary): small jitter on flux spikes =====
        # Only active when creep is engaged (CREEP_MICRO mode).
        # Produces small random jerks/swirls instead of full-circle arcs.
        if (not cfg.noise_primary_mode
                and not event.is_beat
                and cfg.noise_burst_enabled
                and self._motion_mode == MotionMode.CREEP_MICRO
            and self._is_low_band_full_enough(event)
            and self._passes_dual_band_db_gate(event)
                and (self._trajectory is None or self._trajectory.finished)):
            texture_burst = self._combo_scale('combo_texture', 0.60, 1.70)
            noise_mult = float(getattr(cfg, 'noise_burst_flux_multiplier', 2.0) or 2.0)
            noise_thresh = cfg.flux_threshold * (noise_mult / max(texture_burst, 1e-6))
            if event.spectral_flux >= noise_thresh:
                time_since_stroke = (now - self.state.last_stroke_time) * 1000
                if time_since_stroke >= min_interval_ms * 0.4:
                    cmd = self._generate_noise_burst_stroke(event)
                    self._note_motion_resumed("noise_burst")
                    return self._apply_fade(cmd)

        # ===== NOISE BURST (non-primary): FULL_STROKE transient texture =====
        # In loud/full-stroke passages, allow micro-pattern bursts on only
        # stronger transients so texture isn't completely suppressed.
        if (not cfg.noise_primary_mode
                and not event.is_beat
                and cfg.noise_burst_enabled
                and self._motion_mode == MotionMode.FULL_STROKE
            and self._is_low_band_full_enough(event)
            and self._passes_dual_band_db_gate(event)
                and (self._trajectory is None or self._trajectory.finished)):
            texture_burst = self._combo_scale('combo_texture', 0.60, 1.70)
            noise_mult = float(getattr(cfg, 'noise_burst_flux_multiplier', 2.0) or 2.0)
            noise_thresh = cfg.flux_threshold * (noise_mult / max(texture_burst, 1e-6)) * 1.5
            if event.spectral_flux >= noise_thresh:
                time_since_stroke = (now - self.state.last_stroke_time) * 1000
                if time_since_stroke >= min_interval_ms * 0.6:
                    cmd = self._generate_noise_burst_stroke(event)
                    self._note_motion_resumed("noise_burst_full")
                    return self._apply_fade(cmd)

        if event.is_beat:
            # Real beat detected — burst-scheduling yields to metronome
            if self._burst_scheduled_active:
                self._burst_scheduled_active = False
                log_event("INFO", "StrokeMapper",
                          "Burst-schedule deactivated (real beat detected)")

            use_new_gate_priority = bool(
                getattr(cfg, 'new_gate_priority_enabled', True)
                and getattr(cfg, 'overall_amp_fill_gate_enabled', True)
            )
            if (bool(getattr(cfg, 'overall_activity_guard_enabled', True))
                    and not use_new_gate_priority
                    and not (self._learning_enabled and self._learning_relax_phase1_gates)):
                low_flux = float(getattr(cfg, 'overall_low_flux_threshold', 0.06) or 0.06)
                low_energy = float(getattr(cfg, 'overall_low_energy_threshold', 0.14) or 0.14)
                reaction_scale = self._combo_scale('combo_reaction', 0.65, 1.60)
                low_flux = float(np.clip(low_flux * reaction_scale, 0.001, 1.0))
                low_energy = float(np.clip(low_energy * reaction_scale, 0.001, 1.0))
                if (event.spectral_flux < low_flux) and (event.peak_energy < low_energy):
                    self._note_motion_block(
                        "overall_activity_gate",
                        flux=f"{event.spectral_flux:.4f}",
                        flux_threshold=f"{low_flux:.4f}",
                        energy=f"{event.peak_energy:.4f}",
                        energy_threshold=f"{low_energy:.4f}",
                    )
                    cmd = self._generate_idle_motion(event)
                    return self._apply_fade(cmd)

            if not bass_motion_allowed:
                self._note_motion_block(
                    "bass_gate",
                    strict_gate=strict_gate_enabled,
                    cutoff_hz=f"{cutoff:.1f}",
                    beat_band=primary_band or "none",
                    fired_bands=','.join(sorted(fired_bands)) if fired_bands else "none",
                )
                cmd = self._generate_idle_motion(event)
                return self._apply_fade(cmd)

            if self._is_mid_trigger_blocked(event):
                self._note_motion_block(
                    "mid_trigger_block",
                    freq_hz=f"{float(getattr(event, 'frequency', 0.0) or 0.0):.1f}",
                    low_hz=f"{float(getattr(cfg, 'block_mid_trigger_low_hz', 100.0) or 100.0):.1f}",
                    high_hz=f"{float(getattr(cfg, 'block_mid_trigger_high_hz', 2000.0) or 2000.0):.1f}",
                )
                cmd = self._generate_idle_motion(event)
                return self._apply_fade(cmd)

            if not self._passes_dual_band_db_gate(event):
                sub_bass_db = float(20.0 * np.log10(max(float(max(1e-8, self._sub_bass_energy)), 1e-8)))
                high_db = float(20.0 * np.log10(max(float(max(1e-8, self._high_energy)), 1e-8)))
                self._note_motion_block(
                    "dual_band_db_gate",
                    sub_bass_db=f"{sub_bass_db:.1f}",
                    high_db=f"{high_db:.1f}",
                    sub_bass_min=f"{float(getattr(cfg, 'dual_band_sub_bass_db_min', -15.0) or -15.0):.1f}",
                    high_min=f"{float(getattr(cfg, 'dual_band_high_db_min', -30.0) or -30.0):.1f}",
                )
                cmd = self._generate_idle_motion(event)
                return self._apply_fade(cmd)

            # ===== STROKE READINESS GATE =====
            # If metronome + traffic light conditions not met,
            # fall through to idle motion (creep/jitter) instead of strokes
            if not self._stroke_ready:
                self._stroke_gate_block_streak += 1
                if (self._stroke_finish_beats > 0
                        and self._stroke_gate_block_streak <= self._stroke_finish_beats):
                    log_event(
                        "DEBUG",
                        "StrokeMapper",
                        "Stroke readiness grace beat",
                        blocked_streak=self._stroke_gate_block_streak,
                        finish_beats=self._stroke_finish_beats,
                    )
                else:
                    self._note_motion_block("stroke_ready", stroke_ready=False)
                    cmd = self._generate_idle_motion(event)
                    return self._apply_fade(cmd)
            else:
                self._stroke_gate_block_streak = 0

            is_downbeat = getattr(event, 'is_downbeat', False)
            if is_downbeat:
                self.state.beat_counter = 1
            else:
                self.state.beat_counter += 1

            effective_divisor = self._get_adaptive_beat_divisor(event)
            effective_divisor *= self._get_mode_beats_per_stroke_multiplier(cfg.mode)

            if self._motion_mode == MotionMode.FULL_STROKE:
                # High amplitude -> fire arc immediately from current position.
                # No anchor gate — the dot sweeps 360° from wherever it is.
                # Continuous rotation means it passes through top/bottom naturally.
                self._pending_arc_event = None

                is_downbeat = getattr(event, 'is_downbeat', False)
                if effective_divisor > 1 and (self.state.beat_counter % effective_divisor) != 1:
                    if is_downbeat:
                        is_high_flux = bool(event.spectral_flux >= (cfg.flux_threshold * 3.0))
                        if cfg.mode == StrokeMode.SIMPLE_CIRCLE and is_high_flux:
                            cmd = self._generate_beat_stroke(event)
                        else:
                            cmd = self._generate_downbeat_stroke(event, duration_mult=2.0)
                        self._note_motion_resumed("downbeat_fallback")
                        return self._apply_fade(cmd)
                    self._note_motion_block(
                        "beat_divisor",
                        divisor=effective_divisor,
                        mode=str(cfg.mode.name if hasattr(cfg.mode, 'name') else cfg.mode),
                        beat_counter=self.state.beat_counter,
                    )
                    cmd = self._generate_idle_motion(event)
                    return self._apply_fade(cmd)

                beat_gate_pass, beat_mean, beat_delta, beat_var = self._get_low_band_gate_status(event, is_downbeat=False)
                fired_bands = set(getattr(event, 'fired_bands', None) or [])
                beat_band = getattr(event, 'beat_band', '')
                include_mid_high_gate = bool(getattr(cfg, 'high_band_include_mid', True))
                high_beat_hit = (
                    ('high' in fired_bands)
                    or (include_mid_high_gate and ('mid' in fired_bands))
                    or (beat_band == 'high')
                    or (include_mid_high_gate and beat_band == 'mid')
                )
                self._recent_high_band_beat_hits.append(bool(high_beat_hit))

                high_gate_enabled = bool(getattr(cfg, 'high_band_gate_enabled', True))
                high_presence_pass, high_mean, high_occ, high_delta, high_var = self._get_high_band_presence_status(is_downbeat=False)
                high_pattern_pass, high_hits, high_window = self._get_high_band_pattern_status(is_downbeat=False)
                high_gate_pass = (not high_gate_enabled) or (high_presence_pass or high_pattern_pass)

                if beat_gate_pass and high_gate_pass:
                    phase_name = 'downbeat' if is_downbeat else 'beat'
                    amp_fill_pass, amp_val, fill_val, amp_min, fill_req = self._passes_overall_amp_fill_gate(event, phase_name)
                    if not amp_fill_pass:
                        self._note_motion_block(
                            "overall_amp_fill_gate",
                            phase=phase_name,
                            amp=f"{amp_val:.3f}",
                            amp_min=f"{amp_min:.3f}",
                            fill=f"{fill_val:.3f}",
                            fill_required=f"{fill_req:.3f}",
                        )
                        cmd = self._generate_idle_motion(event)
                        return self._apply_fade(cmd)
                    cmd = self._generate_beat_stroke(event)
                    self._note_motion_resumed("beat")
                    return self._apply_fade(cmd)

                if is_downbeat:
                    downbeat_gate_pass, down_mean, down_delta, down_var = self._get_low_band_gate_status(event, is_downbeat=True)
                    down_high_presence_pass, down_high_mean, down_high_occ, down_high_delta, down_high_var = self._get_high_band_presence_status(is_downbeat=True)
                    down_high_pattern_pass, down_high_hits, down_high_window = self._get_high_band_pattern_status(is_downbeat=True)
                    down_high_gate_pass = (not high_gate_enabled) or (down_high_presence_pass or down_high_pattern_pass)
                    if downbeat_gate_pass and down_high_gate_pass:
                        down_amp_pass, down_amp, down_fill, down_min_amp, down_fill_req = self._passes_overall_amp_fill_gate(event, 'downbeat')
                        if not down_amp_pass:
                            self._note_motion_block(
                                "overall_amp_fill_gate",
                                phase="downbeat",
                                amp=f"{down_amp:.3f}",
                                amp_min=f"{down_min_amp:.3f}",
                                fill=f"{down_fill:.3f}",
                                fill_required=f"{down_fill_req:.3f}",
                            )
                            cmd = self._generate_idle_motion(event)
                            return self._apply_fade(cmd)
                        cmd = self._generate_downbeat_stroke(event, duration_mult=2.0)
                        self._note_motion_resumed("downbeat_fallback")
                        return self._apply_fade(cmd)

                    if downbeat_gate_pass and not down_high_gate_pass:
                        self._note_motion_block(
                            "high_band_gate",
                            high_mean=f"{down_high_mean:.4f}",
                            high_occ=f"{down_high_occ:.3f}",
                            high_delta=f"{down_high_delta:.4f}",
                            high_var=f"{down_high_var:.4f}",
                            high_hits=f"{down_high_hits}/{down_high_window}",
                            phase="downbeat",
                        )
                        cmd = self._generate_idle_motion(event)
                        return self._apply_fade(cmd)

                if beat_gate_pass and not high_gate_pass:
                    self._note_motion_block(
                        "high_band_gate",
                        high_mean=f"{high_mean:.4f}",
                        high_occ=f"{high_occ:.3f}",
                        high_delta=f"{high_delta:.4f}",
                        high_var=f"{high_var:.4f}",
                        high_hits=f"{high_hits}/{high_window}",
                    )
                    cmd = self._generate_idle_motion(event)
                    return self._apply_fade(cmd)

                self._note_motion_block(
                    "low_band_gate",
                    low_mean=f"{beat_mean:.4f}",
                    low_delta=f"{beat_delta:.4f}",
                    low_var=f"{beat_var:.4f}",
                )
                cmd = self._generate_idle_motion(event)
                return self._apply_fade(cmd)

            else:  # CREEP_MICRO
                # Low amplitude -> micro-effects on beats, plus produce creep motion
                self._note_motion_block("mode_creep_micro", envelope=f"{self._rms_envelope:.4f}")
                if self._micro_effects_enabled:
                    self._trigger_micro_jerk(event, is_downbeat)
                # Generate creep motion on beats too (not just idle)
                cmd = self._generate_idle_motion(event)
                return self._apply_fade(cmd)

        elif self.state.idle_time > 0.05:
            # Idle motion: creep + jitter + micro-jerk decay
            if not is_truly_silent and self._fade_intensity > 0.01:
                cmd = self._generate_idle_motion(event)
                return self._apply_fade(cmd)
            return None

        if self._trajectory is not None and self._trajectory.active:
            cmd = self._generate_idle_motion(event)
            return self._apply_fade(cmd)

        return None

    # ------------------------------------------------------------------
    # Fade helper
    # ------------------------------------------------------------------

    def _apply_fade(self, cmd: Optional[TCodeCommand]) -> Optional[TCodeCommand]:
        if cmd is None:
            return None
        if hasattr(cmd, 'intensity'):
            cmd.intensity *= self._fade_intensity
        if hasattr(cmd, 'volume'):
            drop_points = int(np.clip(getattr(self.config.stroke, 'silence_fade_drop_points', 10) or 10, 0, 10))
            min_fade_mult = 1.0 - (drop_points / 100.0)
            fade_mult = max(min_fade_mult, float(self._fade_intensity))
            cmd.volume *= fade_mult
            # Post-silence volume ramp: reduce volume and slowly raise back
            if self._post_silence_ramp_active:
                cfg = self.config.stroke
                elapsed = time.perf_counter() - self._post_silence_ramp_start
                ramp_dur = max(0.5, cfg.post_silence_ramp_seconds)
                if elapsed >= ramp_dur:
                    self._post_silence_ramp_active = False
                else:
                    # Start at (1 - reduction), ramp linearly to 1.0
                    reduction = cfg.post_silence_vol_reduction
                    ramp_mult = (1.0 - reduction) + reduction * (elapsed / ramp_dur)
                    cmd.volume *= ramp_mult
        return cmd if self._fade_intensity > 0.01 else None

    # ------------------------------------------------------------------
    # FULL_STROKE generators (same proven logic from v1)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_thump_durations(total_ms: int, n_points: int) -> List[int]:
        """Create step durations that gradually accelerate over the second
        half of the arc, producing a natural 'thump' as the stroke lands
        at the beat.
        First half:  uniform pace.
        Second half:  linearly decreasing step durations (speeding up)
                      down to ~50 % of normal at the final point.
        Total time is preserved.  This also helps with beat adjustments:
        if an incoming beat is faster, the already-accelerating second
        half absorbs the timing change more gracefully."""
        if n_points <= 1:
            return [total_ms]
        first_half = n_points // 2
        second_half = n_points - first_half
        # Build ratio array: first half = 1.0, second half ramps 1.0 -> 0.5
        ratios = []
        for i in range(first_half):
            ratios.append(1.0)
        for i in range(second_half):
            t = i / max(1, second_half - 1) if second_half > 1 else 0.0
            ratios.append(1.0 - 0.15 * t)
        # Normalise so durations sum to total_ms
        total_ratio = sum(ratios)
        durations = [max(5, int(total_ms * r / total_ratio)) for r in ratios]
        # Fix rounding error on the last step
        actual_total = sum(durations)
        if actual_total != total_ms:
            durations[-1] += (total_ms - actual_total)
        return durations

    @staticmethod
    def _make_landing_durations(total_ms: int, n_points: int) -> List[int]:
        """Create step durations that produce a natural 'tap' feel:
        - Fast acceleration away from the start (leaving previous beat)
        - Cruise through the middle
        - Decelerate into the landing (approaching next beat)
        
        This mimics how a finger tap approaches the surface: slow down
        into the contact point, creating a visible 'landing' moment.
        The shape is a cosine ease-in-out curve.
        Total time is preserved."""
        if n_points <= 1:
            return [total_ms]
        # Cosine ease-in-out: fast at start, slow in middle, fast at end
        # But we want the OPPOSITE: slow at edges (landing/takeoff), fast middle
        # Use inverted cosine: ratio = 0.6 + 0.4 * cos(pi * progress)
        # This gives longer durations at start and end (slow), shorter in middle (fast)
        import math
        ratios = []
        for i in range(n_points):
            progress = i / (n_points - 1) if n_points > 1 else 0.5
            # Cosine curve: peaks at edges, valley in middle
            ratio = 0.6 + 0.4 * math.cos(2 * math.pi * progress)
            ratios.append(max(0.3, ratio))
        # Normalise so durations sum to total_ms
        total_ratio = sum(ratios)
        durations = [max(5, int(total_ms * r / total_ratio)) for r in ratios]
        actual_total = sum(durations)
        if actual_total != total_ms:
            durations[-1] += (total_ms - actual_total)
        return durations

    @staticmethod
    def _make_downbeat_tail_accel_durations(total_ms: int, n_points: int) -> List[int]:
        """Downbeat timing curve: mostly steady, then slight acceleration in last 1/8.

        This keeps long downbeat travel readable while adding a subtle push
        into the target near completion.
        """
        if n_points <= 1:
            return [total_ms]

        tail_points = max(1, int(round(n_points * 0.125)))
        base_points = max(1, n_points - tail_points)

        ratios: List[float] = [1.0] * base_points
        tail_end_ratio = 0.82  # shorter step durations near end = faster motion
        for i in range(tail_points):
            progress = (i + 1) / tail_points
            ratio = 1.0 - (1.0 - tail_end_ratio) * progress
            ratios.append(max(tail_end_ratio, ratio))

        total_ratio = sum(ratios)
        durations = [max(5, int(total_ms * r / total_ratio)) for r in ratios]
        actual_total = sum(durations)
        if actual_total != total_ms:
            durations[-1] += (total_ms - actual_total)
        return durations

    @staticmethod
    def _intensity_curve(intensity: float, power: float = 1.8) -> float:
        """Non-linear intensity-to-radius mapping for natural tap dynamics.
        Quiet taps are tiny, loud taps are dramatic.
        power=1.0 is linear. power=1.8 gives more dynamic range:
        soft sounds produce noticeably smaller motion while loud sounds
        still reach full amplitude."""
        return max(0.0, min(1.0, intensity)) ** power

    def _compute_arc_point(self,
                           phase: float,
                           radius: float,
                           stroke_len: float,
                           depth: float,
                           event: BeatEvent) -> Tuple[float, float]:
        """Compute one arc point based on current stroke mode.

        Important constraints:
        - Keep timing/trajectory generation unchanged (this is geometry only)
        - Mode 3 (TEARDROP) is rotated 90° CCW relative to legacy display
        - Mode 3 pattern traversal runs at half draw rate
        """
        mode = self.config.stroke.mode
        alpha_weight = self.config.alpha_weight
        beta_weight = self.config.beta_weight
        angle = phase * 2 * np.pi
        radius_cap = max(0.05, min(1.0, radius))

        if mode == StrokeMode.TEARDROP:
            # Trace full piriform each arc so it descends one side and
            # mirrors back up the other side.
            teardrop_phase = phase % 1.0
            t = (teardrop_phase - 0.5) * 2 * np.pi
            min_radius = 0.2
            curved_intensity = self._intensity_curve(event.intensity)
            a = min_radius + (stroke_len * depth - min_radius) * curved_intensity
            a = max(min_radius, min(1.0, a))
            if self._learning_enabled:
                teardrop_floor = float(np.clip(max(self._edge_follow_radius * 0.95, 0.82), 0.82, 0.98))
                a = max(a, teardrop_floor)

            # Piriform
            x = a * (np.sin(t) - 0.5 * np.sin(2 * t))
            y = -a * np.cos(t)

            # Legacy used +π/2. Display rotated since then; apply +90° CCW more.
            rot = np.pi
            alpha = (x * np.cos(rot) - y * np.sin(rot)) * alpha_weight
            beta = (x * np.sin(rot) + y * np.cos(rot)) * beta_weight

            # Vertical flip for current display orientation: swap top/bottom.
            beta = -beta

            # Hard arc-boundary cap: do not exceed current arc radius
            norm = np.hypot(alpha, beta)
            if norm > radius_cap and norm > 0:
                scale = radius_cap / norm
                alpha *= scale
                beta *= scale

            alpha = np.clip(alpha, -1.0, 1.0)
            beta = np.clip(beta, -1.0, 1.0)
            return alpha, beta

        if mode == StrokeMode.USER:
            flux_ref = max(0.001, self.config.stroke.flux_threshold * 3)
            flux_norm = np.clip(event.spectral_flux / flux_ref, 0, 1)
            peak_norm = np.clip(event.peak_energy, 0, 1)

            alpha_blend = alpha_weight / 2.0
            beta_blend = beta_weight / 2.0
            alpha_response = flux_norm * (1 - alpha_blend) + peak_norm * alpha_blend
            beta_response = flux_norm * (1 - beta_blend) + peak_norm * beta_blend

            min_radius = 0.2
            alpha_radius = min_radius + (stroke_len * depth - min_radius) * alpha_response
            beta_radius = min_radius + (stroke_len * depth - min_radius) * beta_response
            alpha = np.cos(angle) * alpha_radius
            beta = np.sin(angle) * beta_radius

            # Hard arc-boundary cap: do not exceed current arc radius
            norm = np.hypot(alpha, beta)
            if norm > radius_cap and norm > 0:
                scale = radius_cap / norm
                alpha *= scale
                beta *= scale

            alpha = np.clip(alpha, -1.0, 1.0)
            beta = np.clip(beta, -1.0, 1.0)
            return alpha, beta

        # SIMPLE_CIRCLE / fallback geometry
        alpha = np.sin(angle) * radius * alpha_weight
        beta = np.cos(angle) * radius * beta_weight
        # Apply hard cap here too so alpha/beta weights don't pin at edges.
        norm = np.hypot(alpha, beta)
        if norm > radius_cap and norm > 0:
            scale = radius_cap / norm
            alpha *= scale
            beta *= scale
        return alpha, beta

    def _generate_downbeat_stroke(self, event: BeatEvent, duration_mult: float = 1.0) -> Optional[TCodeCommand]:
        """Full measure-length arc on downbeat.  When tempo LOCKED -> 25% boost.
        Stores a PlannedTrajectory; idle motion reads it frame-by-frame."""
        cfg = self.config.stroke
        now = getattr(event, 'monotonic_timestamp', 0.0) or time.perf_counter()
        min_interval_ms = self._effective_min_interval_ms()

        # Beat duration — prefer metronome BPM if available
        # Downbeat arc spans configured beats for this mode.
        beats_in_measure = self._get_downbeat_span_beats(event)
        metro_bpm = getattr(event, 'metronome_bpm', 0.0)
        if metro_bpm > 0:
            beat_interval_ms = 60000.0 / metro_bpm
            beat_interval_ms = max(min_interval_ms, min(1000, beat_interval_ms))
            measure_duration_ms = int(beat_interval_ms * beats_in_measure)
        elif self.state.last_beat_time == 0.0:
            measure_duration_ms = 500 * beats_in_measure
        else:
            beat_interval_ms = (now - self.state.last_beat_time) * 1000
            beat_interval_ms = max(min_interval_ms, min(1000, beat_interval_ms))
            measure_duration_ms = int(beat_interval_ms * beats_in_measure)

        # Clamp to avoid huge sweeps at very low BPM
        measure_duration_ms = max(min_interval_ms, min(4000, measure_duration_ms))
        if cfg.mode == StrokeMode.TEARDROP:
            measure_duration_ms = int(measure_duration_ms * 1.30)
            measure_duration_ms = max(min_interval_ms, min(5000, measure_duration_ms))

        # ===== PRE-FIRE: lazy timing =====
        # Prefer delayed launch when there is extra room before beat+N.
        predicted_target_time = 0.0
        if self.audio_engine and hasattr(self.audio_engine, 'get_tempo_info'):
            tempo_info = self.audio_engine.get_tempo_info()
            predicted = tempo_info.get('predicted_next_beat_mono', 0.0) or tempo_info.get('predicted_next_beat', 0.0)
            if predicted > now:
                beat_interval_s = max(0.001, beat_interval_ms / 1000.0)
                beats_ahead = max(1, int(beats_in_measure))
                target_predicted = predicted + (beats_ahead - 1) * beat_interval_s

                target_time = self._adjust_predicted_target(target_predicted, now)
                if target_time <= 0:
                    target_time = target_predicted
                if target_time > now:
                    predicted_target_time = target_time

        duration_mult = float(max(1.0, duration_mult))
        if duration_mult > 1.0:
            measure_duration_ms = int(measure_duration_ms * duration_mult)
            measure_duration_ms = max(min_interval_ms, min(8000, measure_duration_ms))

        measure_duration_ms, trajectory_start_time, beat_target_time = self._plan_lazy_timing(
            now=now,
            nominal_duration_ms=int(measure_duration_ms),
            min_interval_ms=min_interval_ms,
            predicted_target_time=predicted_target_time,
        )

        if self._is_learning_isolation_active():
            flux_factor = self._learned_radius_mult
        else:
            flux_factor = getattr(self, '_flux_stroke_factor', 1.0) * self._learned_radius_mult
        flux_factor = float(np.clip(flux_factor, 1.0, 1.60))
        tempo_locked = getattr(event, 'tempo_locked', False)
        # Slightly stronger downbeat boost than regular beats for emphasis
        power_punch = self._combo_scale('combo_power', 0.85, 1.35)
        lock_boost = (1.35 if tempo_locked else 1.15) * power_punch
        lock_boost = float(np.clip(lock_boost, 0.90, 2.20))

        stroke_len = cfg.stroke_max * flux_factor * lock_boost * self.motion_intensity
        stroke_len = max(cfg.stroke_min, min(cfg.stroke_max * 1.25, stroke_len))

        depth = 1.0

        min_radius = 0.3
        # Non-linear intensity curve: quiet taps small, loud taps dramatic
        curved_intensity = self._intensity_curve(event.intensity)
        radius = min_radius + (1.0 - min_radius) * flux_factor * lock_boost * curved_intensity
        radius = max(min_radius, min(1.0, radius))

        n_points = max(16, int(measure_duration_ms / 20))
        alpha_arc = np.zeros(n_points)
        beta_arc = np.zeros(n_points)
        arc_radius = min_radius + (stroke_len * depth - min_radius) * curved_intensity
        arc_radius = max(min_radius, min(1.0, arc_radius))
        if cfg.mode == StrokeMode.SPIRAL:
            spiral_drive = float(np.clip((curved_intensity + np.clip(event.spectral_flux / max(cfg.flux_threshold, 0.001), 0.0, 1.5)) * 0.5, 0.0, 1.0))
            if random.random() < (0.40 + 0.20 * spiral_drive):
                self._spiral_direction *= -1
            direction = self._spiral_direction
            launch_phase = self._get_arc_launch_phase(cfg.mode)
            base_angle = self._get_anchor_phase_for_mode(cfg.mode, launch_phase, direction_override=direction)
            spiral_inner = 0.24
            spiral_outer = float(np.clip(max(self._edge_follow_radius * 0.95, 0.78 + 0.18 * spiral_drive), 0.78, 1.0))
            r_start, r_end = (spiral_inner, spiral_outer) if direction > 0 else (spiral_outer, spiral_inner)
            spiral_cap = 1.0
            max_norm_seen = 0.0
            for i in range(n_points):
                progress = i / max(1, n_points - 1)
                theta = base_angle + (progress * 2 * np.pi * direction)
                r = r_start + (r_end - r_start) * progress
                a = r * np.cos(theta) * self.config.alpha_weight
                b_ = r * np.sin(theta) * self.config.beta_weight
                norm = float(np.hypot(a, b_))
                if norm > spiral_cap and norm > 0:
                    scale = spiral_cap / norm
                    a *= scale
                    b_ *= scale
                    norm = spiral_cap
                max_norm_seen = max(max_norm_seen, norm)
                alpha_arc[i] = np.clip(a, -1.0, 1.0)
                beta_arc[i] = np.clip(b_, -1.0, 1.0)
            if self._learning_enabled and max_norm_seen > 0:
                self._edge_follow_radius = float(np.clip(max(self._edge_follow_radius, max_norm_seen), 0.82, 1.00))
            arc_radius = float(np.clip(max(self._edge_follow_radius, spiral_outer), 0.82, 1.0))
        else:
            # Arc starts from current creep angle; anchored for selected modes.
            anchor_phase = self._get_arc_launch_phase(cfg.mode)
            current_phase = anchor_phase / (2 * np.pi)
            if cfg.mode == StrokeMode.SPIRAL:
                arc_phases = np.linspace(current_phase, current_phase + 1.0, n_points, endpoint=False) % 1.0
            else:
                arc_phases = self._build_landing_arc_phases(current_phase, n_points, min_turns=1.0)
            if cfg.mode in (StrokeMode.SIMPLE_CIRCLE, StrokeMode.TEARDROP):
                arc_floor = 0.85 if cfg.mode == StrokeMode.SIMPLE_CIRCLE else 0.82
                arc_radius = float(np.clip(max(arc_radius, self._edge_follow_radius, arc_floor), arc_floor, 1.0))
            for i, phase in enumerate(arc_phases):
                alpha_arc[i], beta_arc[i] = self._compute_arc_point(
                    phase=phase,
                    radius=arc_radius,
                    stroke_len=stroke_len,
                    depth=depth,
                    event=event,
                )

        # Downbeat-specific timing: slight acceleration over the last 1/8 of travel.
        step_durations = self._make_downbeat_tail_accel_durations(measure_duration_ms, n_points)

        # Store trajectory for frame-by-frame playback (no thread)
        self._trajectory = PlannedTrajectory(
            alpha_points=alpha_arc,
            beta_points=beta_arc,
            step_durations=step_durations,
            n_points=n_points,
            current_index=0,
            band_volume=self.get_volume(),
            start_time=trajectory_start_time,
            original_bpm=metro_bpm if metro_bpm > 0 else self._last_known_bpm,
            beat_target_time=beat_target_time,
        )
        if n_points > 0:
            landing_radius = float(np.hypot(alpha_arc[-1], beta_arc[-1]))
            self._update_park_anchor_from_radius(landing_radius)
        follow_floor = 0.85 if cfg.mode == StrokeMode.SIMPLE_CIRCLE else 0.82
        self._edge_follow_radius = float(np.clip(max(self._edge_follow_radius, arc_radius), follow_floor, 1.00))

        self.state.last_stroke_time = now
        self.state.last_beat_time = now
        lock_str = "LOCKED+BOOST" if tempo_locked else "unlocked"
        log_event("INFO", "StrokeMapper", "Arc start",
                  mode=cfg.mode.name, points=n_points,
                  duration_ms=measure_duration_ms, tempo_state=lock_str,
                  delayed_start="yes" if trajectory_start_time > now else "no",
                  pre_fire="yes" if beat_target_time > 0 else "no")
        return None  # idle motion will read from trajectory

    def _generate_beat_stroke(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """Full arc stroke for a regular detected beat.
        Stores a PlannedTrajectory; idle motion reads it frame-by-frame."""
        cfg = self.config.stroke
        now = getattr(event, 'monotonic_timestamp', 0.0) or time.perf_counter()
        min_interval_ms = self._effective_min_interval_ms()

        # Prefer metronome BPM for beat timing
        metro_bpm = getattr(event, 'metronome_bpm', 0.0)
        if metro_bpm > 0:
            beat_interval_ms = 60000.0 / metro_bpm
            beat_interval_ms = max(min_interval_ms, min(1000, beat_interval_ms))
        elif self.state.last_beat_time > 0:
            beat_interval_ms = (now - self.state.last_beat_time) * 1000
            beat_interval_ms = max(min_interval_ms, min(1000, beat_interval_ms))
        else:
            beat_interval_ms = min_interval_ms
        # Use single-beat arc span; beat skipping gate has been removed.
        beat_interval_ms = int(beat_interval_ms)

        # ===== PRE-FIRE: lazy timing =====
        predicted_target_time = 0.0
        if self.audio_engine and hasattr(self.audio_engine, 'get_tempo_info'):
            tempo_info = self.audio_engine.get_tempo_info()
            predicted = tempo_info.get('predicted_next_beat_mono', 0.0) or tempo_info.get('predicted_next_beat', 0.0)
            if predicted > now:
                target_time = self._adjust_predicted_target(predicted, now)
                if target_time <= 0:
                    target_time = predicted
                if target_time > now:
                    predicted_target_time = target_time

        beat_interval_ms, trajectory_start_time, beat_target_time = self._plan_lazy_timing(
            now=now,
            nominal_duration_ms=int(beat_interval_ms),
            min_interval_ms=min_interval_ms,
            predicted_target_time=predicted_target_time,
        )

        # === SELF-CHECK: Apply snap timing correction from previous arc ===
        # If the last arc had to snap-to-target, the timing was slightly off.
        # Compensate by adjusting this arc's duration (e.g., if we were 20ms early,
        # extend the next arc by 20ms so the next landing takes that into account).
        if abs(self._last_snap_correction_ms) > 5.0:
            correction = self._last_snap_correction_ms * 0.7  # 70% correction
            beat_interval_ms = max(min_interval_ms, int(beat_interval_ms + correction))
            self._last_snap_correction_ms = 0.0  # consumed

        intensity = event.intensity
        if self._is_learning_isolation_active():
            flux_factor = self._learned_radius_mult
        else:
            flux_factor = getattr(self, '_flux_stroke_factor', 1.0) * self._learned_radius_mult
        flux_factor = float(np.clip(flux_factor, 1.0, 1.60))
        power_punch = self._combo_scale('combo_power', 0.85, 1.35)
        flux_factor = float(np.clip(flux_factor * power_punch, 1.0, 2.0))

        depth_fullness = self._combo_scale('combo_depth', 0.55, 1.35)
        effective_fullness = float(np.clip(cfg.stroke_fullness * depth_fullness, 0.05, 1.50))
        base_stroke_len = cfg.stroke_min + (cfg.stroke_max - cfg.stroke_min) * intensity * effective_fullness
        stroke_len = base_stroke_len * flux_factor * self.motion_intensity
        stroke_len = max(cfg.stroke_min, min(cfg.stroke_max, stroke_len))

        depth = 1.0

        min_radius = 0.2
        max_radius = 1.0
        # Non-linear intensity curve: quiet taps small, loud taps dramatic
        curved_intensity = self._intensity_curve(intensity)
        base_radius = min_radius + (max_radius - min_radius) * curved_intensity
        radius = base_radius * flux_factor
        radius = max(min_radius, min(1.0, radius))

        if cfg.mode == StrokeMode.SPIRAL:
            n_points = max(8, int(beat_interval_ms / 10))
            alpha_arc = np.zeros(n_points)
            beta_arc = np.zeros(n_points)
            max_norm_seen = 0.0
            alpha_weight = self.config.alpha_weight
            beta_weight = self.config.beta_weight
            spiral_drive = float(np.clip((curved_intensity + np.clip(event.spectral_flux / max(cfg.flux_threshold, 0.001), 0.0, 1.5)) * 0.5, 0.0, 1.0))
            if random.random() < (0.45 + 0.25 * spiral_drive):
                self._spiral_direction *= -1
            direction = self._spiral_direction
            if spiral_drive > 0.60 and random.random() < 0.45:
                direction = 1
                self._spiral_direction = direction

            spiral_inner = 0.24
            spiral_outer = float(np.clip(max(self._edge_follow_radius * 0.94, 0.76 + 0.20 * spiral_drive), 0.76, 1.0))
            if direction > 0:
                r_start, r_end = spiral_inner, spiral_outer
            else:
                r_start, r_end = spiral_outer, spiral_inner

            launch_phase = self._get_arc_launch_phase(cfg.mode)
            base_angle = self._get_anchor_phase_for_mode(cfg.mode, launch_phase, direction_override=direction)
            turns = 1.0
            spiral_cap = 1.0
            for i in range(n_points):
                progress = i / max(1, n_points - 1)
                theta = base_angle + (progress * 2 * np.pi * turns * direction)
                r = r_start + (r_end - r_start) * progress
                a = r * np.cos(theta) * alpha_weight
                b_ = r * np.sin(theta) * beta_weight
                norm = float(np.hypot(a, b_))
                if norm > spiral_cap and norm > 0:
                    scale = spiral_cap / norm
                    a *= scale
                    b_ *= scale
                    norm = spiral_cap
                max_norm_seen = max(max_norm_seen, norm)
                alpha_arc[i] = np.clip(a, -1.0, 1.0)
                beta_arc[i] = np.clip(b_, -1.0, 1.0)
            if self._learning_enabled and max_norm_seen > 0:
                self._edge_follow_radius = float(np.clip(max(self._edge_follow_radius, max_norm_seen), 0.82, 1.00))
        else:
            n_points = max(8, int(beat_interval_ms / 10))
            # Arc starts from current creep angle, sweeps exactly 360°.
            anchor_phase = self._get_arc_launch_phase(cfg.mode)
            current_phase = anchor_phase / (2 * np.pi)
            if cfg.mode == StrokeMode.SPIRAL:
                arc_phases = np.linspace(current_phase, current_phase + 1.0, n_points, endpoint=False) % 1.0
            else:
                arc_phases = self._build_landing_arc_phases(current_phase, n_points, min_turns=1.0)
            alpha_arc = np.zeros(n_points)
            beta_arc = np.zeros(n_points)
            arc_radius = min_radius + (max_radius - min_radius) * curved_intensity
            arc_radius = arc_radius * flux_factor
            arc_radius = max(min_radius, min(1.0, arc_radius))
            if cfg.mode in (StrokeMode.SIMPLE_CIRCLE, StrokeMode.TEARDROP):
                arc_floor = 0.85 if cfg.mode == StrokeMode.SIMPLE_CIRCLE else 0.82
                arc_radius = float(np.clip(max(arc_radius, self._edge_follow_radius, arc_floor), arc_floor, 1.0))
            for i, phase in enumerate(arc_phases):
                alpha_arc[i], beta_arc[i] = self._compute_arc_point(
                    phase=phase,
                    radius=arc_radius,
                    stroke_len=stroke_len,
                    depth=depth,
                    event=event,
                )

        # Apply timing shape: thump or landing (tap feel)
        if cfg.thump_enabled:
            step_durations = self._make_thump_durations(beat_interval_ms, n_points)
        else:
            # Landing emphasis: slow at start/end (tap feel), fast through middle
            step_durations = self._make_landing_durations(beat_interval_ms, n_points)

        # Store trajectory for frame-by-frame playback (no thread)
        self._trajectory = PlannedTrajectory(
            alpha_points=alpha_arc,
            beta_points=beta_arc,
            step_durations=step_durations,
            n_points=n_points,
            current_index=0,
            band_volume=self.get_volume(),
            start_time=trajectory_start_time,
            original_bpm=metro_bpm if metro_bpm > 0 else self._last_known_bpm,
            beat_target_time=beat_target_time,
        )
        if n_points > 0:
            landing_radius = float(np.hypot(alpha_arc[-1], beta_arc[-1]))
            self._update_park_anchor_from_radius(landing_radius)
        if cfg.mode in (StrokeMode.SIMPLE_CIRCLE, StrokeMode.TEARDROP):
            follow_floor = 0.85 if cfg.mode == StrokeMode.SIMPLE_CIRCLE else 0.82
            self._edge_follow_radius = float(np.clip(max(self._edge_follow_radius, arc_radius), follow_floor, 1.00))

        self.state.last_stroke_time = now
        self.state.last_beat_time = now

        band = getattr(event, 'beat_band', 'sub_bass')
        log_event("INFO", "StrokeMapper", "Arc start",
                  mode=cfg.mode.name, points=n_points,
                  duration_ms=beat_interval_ms, band=band,
                  motion=f"{self.motion_intensity:.2f}",
                  delayed_start="yes" if trajectory_start_time > now else "no",
                  pre_fire="yes" if beat_target_time > 0 else "no")
        return None  # idle motion will read from trajectory

    def _generate_syncopated_stroke(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """Lighter, quicker partial arc for an off-beat 'and' hit.
        Arc size and speed are configurable via syncopation_arc_size
        and syncopation_speed settings in Advanced Controls."""
        cfg = self.config.stroke
        beat_cfg = self.config.beat
        now = getattr(event, 'monotonic_timestamp', 0.0) or time.perf_counter()
        min_interval_ms = self._effective_min_interval_ms()

        # Duration is configurable fraction of beat interval
        speed_frac = getattr(beat_cfg, 'syncopation_speed', 0.5) * self._learned_sync_speed_mult
        speed_frac = float(np.clip(speed_frac, 0.10, 1.25))
        mode_multiplier = self._get_mode_beats_per_stroke_multiplier(cfg.mode)
        metro_bpm = getattr(event, 'metronome_bpm', 0.0)
        if metro_bpm > 0:
            beat_ms = 60000.0 / metro_bpm
        elif self.state.last_beat_time > 0:
            beat_ms = (now - self.state.last_beat_time) * 1000
        else:
            beat_ms = min_interval_ms * 2
        duration_ms = max(min_interval_ms * 0.4, min(1000, beat_ms * speed_frac * mode_multiplier))
        duration_ms = int(duration_ms)

        # Pre-fire: lazy timing to next beat
        predicted_target_time = 0.0
        if self.audio_engine and hasattr(self.audio_engine, 'get_tempo_info'):
            tempo_info = self.audio_engine.get_tempo_info()
            predicted = tempo_info.get('predicted_next_beat_mono', 0.0) or tempo_info.get('predicted_next_beat', 0.0)
            if predicted > now:
                target_time = self._adjust_predicted_target(predicted, now)
                if target_time <= 0:
                    target_time = predicted
                if target_time > now:
                    predicted_target_time = target_time

        duration_ms, trajectory_start_time, beat_target_time = self._plan_lazy_timing(
            now=now,
            nominal_duration_ms=int(duration_ms),
            min_interval_ms=min_interval_ms,
            predicted_target_time=predicted_target_time,
        )

        # Reduced amplitude for lighter feel (70% of normal)
        if self._is_learning_isolation_active():
            flux_factor = self._learned_radius_mult
        else:
            flux_factor = getattr(self, '_flux_stroke_factor', 1.0) * self._learned_radius_mult
        flux_factor = float(np.clip(flux_factor, 1.0, 1.60))
        intensity = event.intensity
        curved_intensity = self._intensity_curve(intensity)
        depth_fullness = self._combo_scale('combo_depth', 0.55, 1.35)
        effective_fullness = float(np.clip(cfg.stroke_fullness * depth_fullness, 0.05, 1.50))
        stroke_len = cfg.stroke_min + (cfg.stroke_max - cfg.stroke_min) * curved_intensity * effective_fullness
        stroke_len = stroke_len * flux_factor * self.motion_intensity * 0.7
        stroke_len = max(cfg.stroke_min, min(cfg.stroke_max, stroke_len))

        depth = 1.0

        # Arc size: configurable fraction of circle (0.5 = 180°)
        arc_size = getattr(beat_cfg, 'syncopation_arc_size', 0.5) * self._learned_sync_size_mult
        arc_size = float(np.clip(arc_size, 0.10, 1.0))
        n_points = max(6, int(duration_ms / 12))
        anchor_phase = self._get_arc_launch_phase(cfg.mode)
        current_phase = anchor_phase / (2 * np.pi)
        if cfg.mode == StrokeMode.SPIRAL:
            arc_phases = np.linspace(current_phase, current_phase + arc_size, n_points, endpoint=False) % 1.0
        else:
            arc_phases = self._build_landing_arc_phases(current_phase, n_points, min_turns=max(0.05, float(arc_size)))
        alpha_arc = np.zeros(n_points)
        beta_arc = np.zeros(n_points)

        min_radius = 0.15
        arc_radius = min_radius + (stroke_len * depth - min_radius) * curved_intensity * 0.7
        arc_radius = max(min_radius, min(1.0, arc_radius))
        if cfg.mode in (StrokeMode.SIMPLE_CIRCLE, StrokeMode.TEARDROP):
            follow_floor = 0.85 if cfg.mode == StrokeMode.SIMPLE_CIRCLE else 0.82
            arc_radius = float(np.clip(max(arc_radius, self._edge_follow_radius, follow_floor), follow_floor, 1.0))
        for i, phase in enumerate(arc_phases):
            alpha_arc[i], beta_arc[i] = self._compute_arc_point(
                phase=phase,
                radius=arc_radius,
                stroke_len=stroke_len,
                depth=depth,
                event=event,
            )

        # Always landing durations for tap feel
        step_durations = self._make_landing_durations(duration_ms, n_points)

        self._trajectory = PlannedTrajectory(
            alpha_points=alpha_arc,
            beta_points=beta_arc,
            step_durations=step_durations,
            n_points=n_points,
            current_index=0,
            band_volume=self.get_volume(),
            start_time=trajectory_start_time,
            beat_target_time=beat_target_time,
            original_bpm=metro_bpm if metro_bpm > 0 else self._last_known_bpm,
        )
        if n_points > 0:
            landing_radius = float(np.hypot(alpha_arc[-1], beta_arc[-1]))
            self._update_park_anchor_from_radius(landing_radius)
        if cfg.mode in (StrokeMode.SIMPLE_CIRCLE, StrokeMode.TEARDROP):
            follow_floor = 0.85 if cfg.mode == StrokeMode.SIMPLE_CIRCLE else 0.82
            self._edge_follow_radius = float(np.clip(max(self._edge_follow_radius, arc_radius), follow_floor, 1.00))

        self.state.last_stroke_time = now
        log_event("INFO", "StrokeMapper", "Syncopated arc",
                  points=n_points, duration_ms=duration_ms,
                  delayed_start="yes" if trajectory_start_time > now else "no",
                  arc_size=f"{arc_size:.2f}", speed=f"{speed_frac:.2f}")
        return None

    # ------------------------------------------------------------------
    # Noise-burst reactive arc (hybrid noise + metronome)
    # ------------------------------------------------------------------

    def _generate_noise_burst_stroke(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """Small random jitter/swirl patterns on sudden loud transients.
        Only fires in CREEP_MICRO mode when creep is active.
        Produces random tiny patterns: jerks, micro-swirls, star shapes, zigzags."""
        now = getattr(event, 'monotonic_timestamp', 0.0) or time.perf_counter()

        # Pick a random micro-pattern type
        pattern = random.choice(['jerk', 'swirl', 'star', 'zigzag'])
        texture_burst = self._combo_scale('combo_texture', 0.60, 1.70)
        magnitude_scale = float(getattr(self.config.stroke, 'noise_burst_magnitude', 1.0) or 1.0) * texture_burst
        energy_scale = 1.0 + (self._mid_energy + self._high_energy) * 2.0
        energy_scale = min(energy_scale, 2.0)
        jerk_mag = random.uniform(0.03, 0.07) * self.motion_intensity * magnitude_scale * energy_scale
        jerk_mag = float(np.clip(jerk_mag, 0.03, 0.07))
        base_angle = self.state.creep_angle
        n_points = random.randint(4, 8)
        duration_ms = random.randint(60, 120)

        # Current creep position as center of the micro-pattern.
        # Keep burst center radius independent from jitter amplitude so
        # transient bursts stay visible even when jitter is high.
        if self.config.creep.enabled:
            creep_radius = 0.30
            center_a = np.sin(base_angle) * creep_radius
            center_b = np.cos(base_angle) * creep_radius
        else:
            center_a = float(self.state.alpha)
            center_b = float(self.state.beta)
            if abs(center_a) > 1e-6 or abs(center_b) > 1e-6:
                base_angle = np.arctan2(center_a, center_b)

        alpha_pts = np.zeros(n_points)
        beta_pts = np.zeros(n_points)

        if pattern == 'jerk':
            # Single direction jerk with decay back to center
            angle = base_angle + random.uniform(-1.5, 1.5)
            for i in range(n_points):
                decay = 1.0 - (i / n_points)
                alpha_pts[i] = center_a + np.sin(angle) * jerk_mag * decay
                beta_pts[i] = center_b + np.cos(angle) * jerk_mag * decay
        elif pattern == 'swirl':
            # Tiny spiral inward (or outward)
            direction = random.choice([1, -1])
            for i in range(n_points):
                t = i / n_points
                angle = base_angle + t * np.pi * 2 * direction
                r = jerk_mag * (1.0 - t * 0.7)
                alpha_pts[i] = center_a + np.sin(angle) * r
                beta_pts[i] = center_b + np.cos(angle) * r
        elif pattern == 'star':
            # Star pattern: alternate large/small radius at different angles
            for i in range(n_points):
                angle = base_angle + (i / n_points) * np.pi * 2
                r = jerk_mag if i % 2 == 0 else jerk_mag * 0.3
                alpha_pts[i] = center_a + np.sin(angle) * r
                beta_pts[i] = center_b + np.cos(angle) * r
        else:  # zigzag
            # Zigzag perpendicular to creep direction with decay
            perp = base_angle + np.pi / 2
            for i in range(n_points):
                offset = jerk_mag * (1 if i % 2 == 0 else -1) * (1.0 - i / n_points)
                alpha_pts[i] = center_a + np.sin(perp) * offset
                beta_pts[i] = center_b + np.cos(perp) * offset

        alpha_pts = np.clip(alpha_pts, -1.0, 1.0)
        beta_pts = np.clip(beta_pts, -1.0, 1.0)

        step = max(5, duration_ms // n_points)
        step_durations = [step] * n_points
        actual = sum(step_durations)
        if actual != duration_ms and n_points > 0:
            step_durations[-1] += (duration_ms - actual)

        self._trajectory = PlannedTrajectory(
            alpha_points=alpha_pts,
            beta_points=beta_pts,
            step_durations=step_durations,
            n_points=n_points,
            current_index=0,
            band_volume=self.get_volume(),
            start_time=now,
            is_micro=True,
        )

        self.state.last_stroke_time = now
        log_event("INFO", "StrokeMapper", f"Noise jitter ({pattern})",
                  points=n_points, duration_ms=duration_ms,
                  flux=f"{event.spectral_flux:.3f}")
        return None

    # ------------------------------------------------------------------
    # Trajectory playback (called from _generate_idle_motion)
    # ------------------------------------------------------------------

    def _advance_trajectory(self) -> Optional[TCodeCommand]:
        """Read the next point from the active trajectory.
        Uses elapsed time to pick the correct point, so the arc stays in
        sync with the beat even if the frame rate fluctuates.
        If BPM changed mid-arc, rescales remaining step durations so
        the arc still finishes on time for the beat."""
        traj = self._trajectory
        if traj is None or traj.finished:
            return None

        now = time.perf_counter()

        if now < traj.start_time:
            traj.start_time = now

        # ===== MID-ARC SPEED ADJUSTMENT =====
        # If BPM changed since arc was created, rescale remaining durations
        # so the dot still lands on target at the right time.
        # Limit acceleration factor to [0.5, 2.0] to keep changes smooth.
        if (traj.original_bpm > 0
                and self._last_known_bpm > 0
                and traj.current_index < traj.n_points - 1):
            bpm_ratio = traj.original_bpm / self._last_known_bpm  # >1 = tempo sped up, need shorter steps
            if abs(bpm_ratio - 1.0) > 0.03:  # >3% change threshold
                bpm_ratio = max(0.5, min(2.0, bpm_ratio))  # limit acceleration
                for i in range(traj.current_index, traj.n_points):
                    traj.step_durations[i] = max(5, int(traj.step_durations[i] * bpm_ratio))
                traj.original_bpm = self._last_known_bpm  # update so we don't re-adjust

        # ===== BEAT-TARGET TIMING =====
        # If we have a beat_target_time, use time-to-target for index calculation
        # so the dot arrives at the target point ON the beat, not after.
        elapsed_ms = (now - traj.start_time) * 1000

        # Find the target index from cumulative step durations
        cumulative = 0
        target_idx = 0
        for i in range(traj.n_points):
            cumulative += traj.step_durations[i]
            if elapsed_ms < cumulative:
                target_idx = i
                break
        else:
            target_idx = traj.n_points - 1  # past end

        # Jump to the time-correct index, but cap per-frame advance so
        # we don't draw visible straight-line chords when timing gates block.
        target_idx = max(target_idx, traj.current_index)
        max_advance = max(1, int(self._trajectory_max_step_advance))
        if target_idx > (traj.current_index + max_advance):
            target_idx = traj.current_index + max_advance

        alpha = float(traj.alpha_points[target_idx])
        beta = float(traj.beta_points[target_idx])
        step_ms = traj.step_durations[target_idx]

        # Use short duration matching our update rate for smooth motion
        duration_ms = min(step_ms, 25)

        fade_reduction = (1.0 - self._fade_intensity) * traj.band_volume
        volume = max(self._vol_floor(traj.band_volume), traj.band_volume - fade_reduction)

        prev_alpha = float(self.state.alpha)
        prev_beta = float(self.state.beta)
        self._log_large_motion_jump(prev_alpha, prev_beta, alpha, beta, source="trajectory")
        self.state.alpha = alpha
        self.state.beta = beta
        # Keep creep_angle in sync with actual position during trajectory playback
        # so idle motion resumes smoothly after arc completes
        r = np.sqrt(alpha**2 + beta**2)
        if r > 0.05:
            self.state.creep_angle = np.arctan2(alpha, beta)
            if self.state.creep_angle < 0:
                self.state.creep_angle += 2 * np.pi
        traj.current_index = target_idx + 1

        # Intentionally do NOT snap/catch-up to target after beat time.
        # Preference: keep natural pace and allow a miss rather than rushing.

        # Check if trajectory just completed
        if traj.finished:
            if traj.beat_target_time > 0:
                landing_error_ms = (now - traj.beat_target_time) * 1000.0
                self._update_lead_trim_from_landing(landing_error_ms)
            log_event("INFO", "StrokeMapper", "Arc complete", points=traj.n_points)
            self._sync_creep_angle_to_position()

            if getattr(traj, 'is_park_return', False):
                self.state.alpha = self._park_alpha
                self.state.beta = self._park_beta
                self.state.creep_reset_active = False
                self._trajectory = None
                self._post_arc_blend = 0.0
                log_event("INFO", "StrokeMapper", "Park return complete")
                return TCodeCommand(self.state.alpha, self.state.beta, 20, self.get_volume())

            if getattr(traj, 'is_micro', False):
                # Micro patterns (noise jitter): resume creep
                # Micro trajectories are short; resume to creep quickly so
                # rapid bursts don't stack against a long blend tail.
                self._post_arc_blend = 0.7
                self._trajectory = None
            elif (self._motion_mode == MotionMode.FULL_STROKE
                    and self.config.stroke.mode == StrokeMode.SIMPLE_CIRCLE
                    and self._stroke_ready
                    and self._is_low_band_full_enough()
                    and self._last_known_bpm > 0):
                # Continuous rotation: immediately start another arc
                # so the dot never stops moving between beats.
                # Real beat events will override this when they fire.
                self._generate_continuation_arc()
            else:
                # No good BPM or not in FULL_STROKE — drop to creep
                self._post_arc_blend = 0.0
                self._trajectory = None

        return TCodeCommand(alpha, beta, duration_ms, volume)

    # ------------------------------------------------------------------
    # Continuation arc (seamless rotation between beat-driven arcs)
    # ------------------------------------------------------------------

    def _generate_continuation_arc(self) -> None:
        """Generate a new full-circle arc timed so the dot ARRIVES at the
        next beat landing point ON the beat, not starts on it.
        When metronome is locked, uses predicted_next_beat to calculate
        exactly when to land. The arc starts immediately (no gap) and
        its duration is set so it completes at beat arrival time.
        Real beat events will override this trajectory when they fire."""
        cfg = self.config.stroke
        now = time.perf_counter()
        bpm = self._last_known_bpm
        if bpm <= 0:
            self._trajectory = None
            return

        min_interval_ms = self._effective_min_interval_ms()
        beat_interval_ms = int(60000.0 / bpm)
        beat_interval_ms = max(min_interval_ms, min(4000, beat_interval_ms))

        # ===== PRE-FIRE: lazy timing =====
        predicted_target_time = 0.0
        if self.audio_engine and hasattr(self.audio_engine, 'get_tempo_info'):
            tempo_info = self.audio_engine.get_tempo_info()
            predicted = tempo_info.get('predicted_next_beat_mono', 0.0) or tempo_info.get('predicted_next_beat', 0.0)
            if predicted > now:
                target_time = self._adjust_predicted_target(predicted, now)
                if target_time <= 0:
                    target_time = predicted
                if target_time > now:
                    predicted_target_time = target_time

        beat_interval_ms, trajectory_start_time, beat_target_time = self._plan_lazy_timing(
            now=now,
            nominal_duration_ms=int(beat_interval_ms),
            min_interval_ms=min_interval_ms,
            predicted_target_time=predicted_target_time,
        )

        # Reuse the last trajectory's radius for visual continuity.
        # Fall back to a moderate default if unavailable.
        prev_traj = self._trajectory
        prev_volume = prev_traj.band_volume if prev_traj else self.get_volume()

        # Recover radius from the last arc's endpoint
        last_r = np.sqrt(self.state.alpha**2 + self.state.beta**2)
        radius = max(0.2, min(1.0, last_r)) if last_r > 0.05 else 0.5
        if cfg.mode in (StrokeMode.SIMPLE_CIRCLE, StrokeMode.SPIRAL, StrokeMode.TEARDROP):
            follow_floor = 0.85 if cfg.mode == StrokeMode.SIMPLE_CIRCLE else 0.82
            radius = float(np.clip(max(radius, self._edge_follow_radius, follow_floor), follow_floor, 1.0))

        alpha_weight = self.config.alpha_weight
        beta_weight = self.config.beta_weight

        n_points = max(8, int(beat_interval_ms / 10))
        anchor_phase = self._get_arc_launch_phase(cfg.mode)
        current_phase = anchor_phase / (2 * np.pi)
        arc_phases = self._build_landing_arc_phases(current_phase, n_points, min_turns=1.0)
        alpha_arc = np.zeros(n_points)
        beta_arc = np.zeros(n_points)
        for i, phase in enumerate(arc_phases):
            angle = phase * 2 * np.pi
            alpha_arc[i] = np.sin(angle) * radius * alpha_weight
            beta_arc[i] = np.cos(angle) * radius * beta_weight

        # Apply timing shape: thump or landing (tap feel)
        if cfg.thump_enabled:
            step_durations = self._make_thump_durations(beat_interval_ms, n_points)
        else:
            # Landing emphasis: slow at start/end (tap feel), fast through middle
            step_durations = self._make_landing_durations(beat_interval_ms, n_points)

        self._trajectory = PlannedTrajectory(
            alpha_points=alpha_arc,
            beta_points=beta_arc,
            step_durations=step_durations,
            n_points=n_points,
            current_index=0,
            band_volume=prev_volume,
            start_time=trajectory_start_time,
            beat_target_time=beat_target_time,
            original_bpm=bpm,
        )
        if n_points > 0:
            landing_radius = float(np.hypot(alpha_arc[-1], beta_arc[-1]))
            self._update_park_anchor_from_radius(landing_radius)

        log_event("INFO", "StrokeMapper", "Continuation arc",
                  bpm=f"{bpm:.1f}", points=n_points,
                  duration_ms=beat_interval_ms, radius=f"{radius:.2f}",
                  delayed_start="yes" if trajectory_start_time > now else "no",
                  pre_fire="yes" if beat_target_time > 0 else "no")

    # ------------------------------------------------------------------
    # Micro-effect: jerk on beat (CREEP_MICRO mode)
    # ------------------------------------------------------------------

    def _trigger_micro_jerk(self, event: BeatEvent, is_downbeat: bool) -> None:
        """
        Record a micro-jerk triggered by a beat while in CREEP_MICRO mode.
        The jerk is a small impulsive displacement that decays quickly.
        Size scales with mid/high band energy.
        Downbeats get a slightly larger jerk.
        """
        now = time.time()
        base_mag = 0.05 if not is_downbeat else 0.07

        # Scale by mid+high energy for musical responsiveness
        band_scale = 1.0 + (self._mid_energy + self._high_energy) * 5.0
        band_scale = min(band_scale, 3.0)

        mag = base_mag * band_scale * self.motion_intensity
        mag = float(np.clip(mag, 0.03, 0.07))

        # Direction: radially outward from current creep angle
        jerk_angle = self.state.creep_angle + random.uniform(-0.3, 0.3)
        self._micro_jerk_alpha = np.sin(jerk_angle) * mag
        self._micro_jerk_beta = np.cos(jerk_angle) * mag
        self._last_micro_jerk_time = now
        self._micro_jerk_decay_ms = 150.0 if is_downbeat else 100.0

        log_event("INFO", "StrokeMapper", "Micro-jerk",
                  downbeat=is_downbeat,
                  mag=f"{mag:.3f}",
                  band_scale=f"{band_scale:.2f}")

    def _get_micro_jerk_offset(self) -> Tuple[float, float]:
        """Get current micro-jerk offset (decaying exponential)."""
        if self._last_micro_jerk_time == 0:
            return 0.0, 0.0
        elapsed_ms = (time.time() - self._last_micro_jerk_time) * 1000
        if elapsed_ms > self._micro_jerk_decay_ms * 3:
            return 0.0, 0.0
        # Exponential decay
        decay = np.exp(-elapsed_ms / self._micro_jerk_decay_ms)
        return self._micro_jerk_alpha * decay, self._micro_jerk_beta * decay

    # ------------------------------------------------------------------
    # Idle motion (creep + jitter + micro-jerk + arc return)
    # ------------------------------------------------------------------

    def _generate_idle_motion(self, event: Optional[BeatEvent]) -> Optional[TCodeCommand]:
        """Generate motion: trajectory playback OR creep/jitter when idle."""
        now = time.perf_counter()
        jitter_cfg = self.config.jitter
        creep_cfg = self.config.creep

        # 60 fps throttle (use separate timer from beat strokes)
        time_since_last = (now - self._last_idle_time) * 1000
        if time_since_last < 17:
            return None

        # ---------- Trajectory playback (replaces arc thread) ----------
        if self._trajectory is not None and self._trajectory.active:
            self._last_idle_time = now
            return self._advance_trajectory()

        reliable_tempo_bpm = self._get_reliable_metronome_bpm(event)
        if reliable_tempo_bpm > 0:
            self._last_known_bpm = reliable_tempo_bpm

        recent_beats_active = self._has_recent_beats(now=now, window_s=0.9)

        anchor_state_active = (
            self._trajectory is None
            and (self.state.creep_reset_active or not recent_beats_active)
        )
        anchor_bass_norm = 0.0
        if anchor_state_active:
            anchor_bass_norm = self._update_anchor_from_bass_state(event)
        if (recent_beats_active
                and self._motion_mode == MotionMode.FULL_STROKE
                and reliable_tempo_bpm > 0):
            self._generate_continuation_arc()
            if self._trajectory is not None and self._trajectory.active:
                self._last_idle_time = now
                return self._advance_trajectory()

        jitter_active = jitter_cfg.enabled and jitter_cfg.amplitude > 0
        if anchor_state_active:
            jitter_active = True
        creep_active = creep_cfg.enabled and creep_cfg.speed > 0

        if not jitter_active and not creep_active and not self.state.creep_reset_active:
            # Still allow micro-jerk decay to produce motion
            jerk_a, jerk_b = self._get_micro_jerk_offset()
            parked = (abs(self.state.alpha - self._park_alpha) < 0.01 and abs(self.state.beta - self._park_beta) < 0.01)
            if abs(jerk_a) < 0.001 and abs(jerk_b) < 0.001 and parked:
                return None

        alpha, beta = self.state.alpha, self.state.beta

        creep_reset_blend = 0.0
        if self.state.creep_reset_active:
            reset_duration_ms = 400
            elapsed_ms = (now - self.state.creep_reset_start_time) * 1000
            if elapsed_ms < reset_duration_ms:
                progress = elapsed_ms / reset_duration_ms
                eased_progress = 1.0 - (1.0 - progress) ** 2
                creep_reset_blend = float(np.clip(eased_progress, 0.0, 1.0))
            else:
                self.state.creep_reset_active = False

        # ---------- Creep volume lowering ----------
        if creep_active:
            expected_beat_ms = 500.0
            if self.audio_engine and hasattr(self.audio_engine, 'get_tempo_info'):
                tempo_info = self.audio_engine.get_tempo_info()
                if tempo_info and tempo_info.get('bpm', 0) > 0:
                    expected_beat_ms = 60000.0 / tempo_info['bpm']

            if not self._creep_was_active_last_frame:
                self._creep_sustained_start = now
                self._creep_volume_factor = 1.0
            else:
                sustained_ms = (now - self._creep_sustained_start) * 1000.0
                threshold_ms = expected_beat_ms * 2.0
                if sustained_ms > threshold_ms:
                    fade_start_ms = threshold_ms
                    fade_duration_ms = 600.0
                    fade_progress = min(1.0, (sustained_ms - fade_start_ms) / fade_duration_ms)
                    self._creep_volume_factor = 1.0 - (0.03 * fade_progress)
            self._creep_was_active_last_frame = True
        else:
            self._creep_was_active_last_frame = False
            self._creep_volume_factor = 1.0

        # ---------- Creep: tempo-synced rotation ----------
        if creep_active:
            # Angle sync now happens only on mode transitions and arc completion
            # (via _sync_creep_angle_to_position), not every frame.
            # This prevents the sync from fighting the tempo-based rotation.
            
            bpm = reliable_tempo_bpm
            if bpm <= 0:
                fallback_bpm = self._last_known_bpm if self._last_known_bpm > 0 else 90.0
                bpm = float(np.clip(fallback_bpm, 45.0, 160.0))

            beats_per_sec = bpm / 60.0
            updates_per_sec = 1000.0 / 17.0
            updates_per_beat = updates_per_sec / beats_per_sec
            angle_increment = (np.pi / 2.0) / updates_per_beat * creep_cfg.speed

            if self._motion_mode == MotionMode.CREEP_MICRO:
                # In CREEP_MICRO: one full rotation per measure (4 beats -> 2pi)
                # Override speed: exactly 2pi per measure
                angle_increment = (2 * np.pi) / (updates_per_beat * self.config.beat.beats_per_measure)

            # Quiet-mode soft brake: keep moving, but cap spin speed when
            # signal is near silence so we don't whip around at high inferred BPM.
            stroke_cfg = self.config.stroke
            beat_cfg = self.config.beat
            quiet_flux_thresh = float(stroke_cfg.flux_threshold) * float(stroke_cfg.silence_flux_multiplier)
            quiet_energy_thresh = float(beat_cfg.peak_floor) * float(stroke_cfg.silence_energy_multiplier)
            near_quiet_flux = max(1e-6, quiet_flux_thresh * 2.1)
            near_quiet_energy = max(1e-6, quiet_energy_thresh * 2.1)
            event_flux = float(getattr(event, 'spectral_flux', 0.0) or 0.0)
            event_energy = float(getattr(event, 'peak_energy', 0.0) or 0.0)
            quiet_ratio = float(np.clip(max(event_flux / near_quiet_flux, event_energy / near_quiet_energy), 0.0, 1.0))
            max_quiet_increment = 0.026
            min_quiet_increment = 0.008
            quiet_cap = min_quiet_increment + ((max_quiet_increment - min_quiet_increment) * quiet_ratio)
            if angle_increment > quiet_cap:
                angle_increment = quiet_cap

            # Keep creep rotation continuous even during creep_reset so we never
            # visually stall at top/edge points.
            self.state.creep_angle += angle_increment
            if self.state.creep_angle >= 2 * np.pi:
                self.state.creep_angle -= 2 * np.pi

            if self._motion_mode == MotionMode.CREEP_MICRO:
                # CREEP_MICRO: smaller radius, drift toward center not edges
                creep_radius = 0.20
            else:
                creep_radius = 0.50

            if (self._motion_mode == MotionMode.FULL_STROKE
                    and self.config.stroke.mode in (StrokeMode.SIMPLE_CIRCLE, StrokeMode.SPIRAL, StrokeMode.TEARDROP)):
                follow_floor = 0.85 if self.config.stroke.mode == StrokeMode.SIMPLE_CIRCLE else 0.82
                creep_radius = float(np.clip(max(creep_radius, self._edge_follow_radius, follow_floor), follow_floor, 0.85))

            if self.state.creep_reset_active:
                park_radius = float(np.hypot(self._park_alpha, self._park_beta))
                creep_radius = float(creep_radius + ((park_radius - creep_radius) * creep_reset_blend))

            target_alpha = np.sin(self.state.creep_angle) * creep_radius
            target_beta = np.cos(self.state.creep_angle) * creep_radius

            # Smooth blend from arc endpoint to creep orbit
            if self._post_arc_blend < 1.0:
                self._post_arc_blend = min(1.0, self._post_arc_blend + self._post_arc_blend_rate)
                base_alpha = alpha + (target_alpha - alpha) * self._post_arc_blend
                base_beta = beta + (target_beta - beta) * self._post_arc_blend
            else:
                base_alpha = target_alpha
                base_beta = target_beta
        else:
            # Creep disabled: quickly wobble toward park so dot
            # doesn't get stuck at the edge after an arc finishes.
            blend_rate = 0.15  # per frame (~60fps → ~300ms to reach center)
            base_alpha = alpha + (self._park_alpha - alpha) * blend_rate
            base_beta = beta + (self._park_beta - beta) * blend_rate
            # Snap to park when close enough to avoid perpetual micro-drift
            if abs(base_alpha - self._park_alpha) < 0.01:
                base_alpha = self._park_alpha
            if abs(base_beta - self._park_beta) < 0.01:
                base_beta = self._park_beta

        # ---------- Jitter: sinusoidal micro-circles ----------
        if jitter_active:
            texture_jitter = self._combo_scale('combo_texture', 0.65, 1.60)
            if self._motion_mode == MotionMode.CREEP_MICRO:
                # CREEP_MICRO: slower, smaller jitter
                jitter_speed = jitter_cfg.intensity * 0.08
                jitter_r = jitter_cfg.amplitude * 0.5
            else:
                jitter_speed = jitter_cfg.intensity * 0.15
                jitter_r = jitter_cfg.amplitude

            jitter_speed *= texture_jitter
            jitter_r *= float(np.clip(0.85 + (0.15 * texture_jitter), 0.5, 1.6))

            if anchor_state_active:
                jitter_r = max(jitter_r, 0.008 + (0.028 * anchor_bass_norm))
                jitter_speed = max(jitter_speed, 0.45 + (1.15 * anchor_bass_norm))
                jitter_speed *= (0.85 + 0.40 * anchor_bass_norm)

            # Modulate jitter size by mid/high energy in CREEP_MICRO mode
            if self._motion_mode == MotionMode.CREEP_MICRO and self._micro_effects_enabled:
                energy_mod = 1.0 + (self._mid_energy + self._high_energy) * 3.0
                energy_mod = min(energy_mod, 2.5)
                jitter_r *= energy_mod
                jitter_speed *= (0.8 + energy_mod * 0.2)

            # Bass pitch mapping: higher bass pitch -> faster jitter,
            # lower bass pitch -> slower jitter.
            jitter_speed *= self._bass_jitter_speed_mult

            self.state.jitter_angle += jitter_speed
            if self.state.jitter_angle >= 2 * np.pi:
                self.state.jitter_angle -= 2 * np.pi

            alpha_target = base_alpha + np.cos(self.state.jitter_angle) * jitter_r
            beta_target = base_beta + np.sin(self.state.jitter_angle) * jitter_r
        else:
            alpha_target = base_alpha
            beta_target = base_beta

        # ---------- Add micro-jerk offset (decaying impulse) ----------
        if self._micro_effects_enabled:
            jerk_a, jerk_b = self._get_micro_jerk_offset()
            alpha_target += jerk_a
            beta_target += jerk_b

        # Clamp
        alpha_target = np.clip(alpha_target, -1.0, 1.0)
        beta_target = np.clip(beta_target, -1.0, 1.0)

        duration_ms = 25  # short duration matching update rate for smooth motion

        prev_alpha = float(self.state.alpha)
        prev_beta = float(self.state.beta)
        self._log_large_motion_jump(prev_alpha, prev_beta, alpha_target, beta_target, source="idle")
        delta = float(np.hypot(alpha_target - prev_alpha, beta_target - prev_beta))
        if delta < self._anti_stop_min_delta and self._should_anti_stop(alpha_target, beta_target):
            ref_radius = float(np.clip(max(np.hypot(alpha_target, beta_target), self._edge_follow_radius), 0.85, 1.0))
            alpha_target, beta_target = self._apply_anti_stop_nudge(alpha_target, beta_target, reference_radius=ref_radius)

        self.state.alpha = alpha_target
        self.state.beta = beta_target
        self._last_idle_time = now

        # Volume with fade + creep reduction
        base_vol = self.get_volume()
        fade = self._fade_intensity
        creep_vol = self._creep_volume_factor
        fade_reduction = (1.0 - fade) * base_vol
        creep_reduction = (1.0 - creep_vol) * base_vol
        total_reduction = fade_reduction + creep_reduction
        limit_pct = self.config.stroke.vol_reduction_limit / 100.0
        volume = max(self._vol_floor(base_vol), base_vol - min(total_reduction, base_vol * limit_pct))

        return TCodeCommand(alpha_target, beta_target, duration_ms, volume)

    # ------------------------------------------------------------------
    # Stroke target (shape generators - preserved from v1)
    # ------------------------------------------------------------------

    def _get_stroke_target(self, stroke_len: float, depth: float, event: BeatEvent) -> Tuple[float, float]:
        """Calculate target position based on stroke mode."""
        mode = self.config.stroke.mode
        alpha_weight = self.config.alpha_weight
        beta_weight = self.config.beta_weight
        phase_advance = float(self.config.stroke.phase_advance)
        phase_advance *= self._combo_scale('combo_power', 0.50, 1.80)
        phase_advance = float(np.clip(phase_advance, 0.0, 1.0))

        if mode == StrokeMode.SIMPLE_CIRCLE:
            self.state.phase = (self.state.phase + phase_advance) % 1.0
            angle = self.state.phase * 2 * np.pi
            min_radius = 0.3
            radius = min_radius + (stroke_len * depth - min_radius) * event.intensity
            radius = max(min_radius, min(1.0, radius))
            alpha = np.sin(angle) * radius * alpha_weight
            beta = np.cos(angle) * radius * beta_weight

        elif mode == StrokeMode.SPIRAL:
            self.state.phase = (self.state.phase + phase_advance) % 1.0
            revolutions = 2
            theta_max = revolutions * 2 * np.pi
            theta = (self.state.phase - 0.5) * 2 * theta_max
            min_radius = 0.3
            base_radius = min_radius + (stroke_len * depth - min_radius) * event.intensity
            base_radius = max(min_radius, min(1.0, base_radius))
            spiral_factor = abs(theta) / theta_max
            r = base_radius * spiral_factor
            alpha = r * np.cos(theta) * alpha_weight
            beta = r * np.sin(theta) * beta_weight
            alpha = np.clip(alpha, -1.0, 1.0)
            beta = np.clip(beta, -1.0, 1.0)

        elif mode == StrokeMode.TEARDROP:
            teardrop_advance = phase_advance * 0.25
            self.state.phase = (self.state.phase + teardrop_advance) % 1.0
            t = (self.state.phase - 0.5) * 2 * np.pi
            min_radius = 0.2
            a = min_radius + (stroke_len * depth - min_radius) * event.intensity
            a = max(min_radius, min(1.0, a))
            x = a * (np.sin(t) - 0.5 * np.sin(2 * t))
            y = -a * np.cos(t)
            angle = np.pi / 2
            alpha = x * np.cos(angle) - y * np.sin(angle)
            beta = x * np.sin(angle) + y * np.cos(angle)
            alpha *= alpha_weight
            beta *= beta_weight
            alpha = np.clip(alpha, -1.0, 1.0)
            beta = np.clip(beta, -1.0, 1.0)

        elif mode == StrokeMode.USER:
            self.state.phase = (self.state.phase + phase_advance) % 1.0
            angle = self.state.phase * 2 * np.pi
            flux_ref = max(0.001, self.config.stroke.flux_threshold * 3)
            flux_norm = np.clip(event.spectral_flux / flux_ref, 0, 1)
            peak_norm = np.clip(event.peak_energy, 0, 1)
            alpha_blend = alpha_weight / 2.0
            beta_blend = beta_weight / 2.0
            alpha_response = flux_norm * (1 - alpha_blend) + peak_norm * alpha_blend
            beta_response = flux_norm * (1 - beta_blend) + peak_norm * beta_blend
            min_radius = 0.2
            alpha_radius = min_radius + (stroke_len * depth - min_radius) * alpha_response
            beta_radius = min_radius + (stroke_len * depth - min_radius) * beta_response
            alpha = np.cos(angle) * alpha_radius
            beta = np.sin(angle) * beta_radius
            alpha = np.clip(alpha, -1.0, 1.0)
            beta = np.clip(beta, -1.0, 1.0)

        else:
            self.state.phase = (self.state.phase + phase_advance) % 1.0
            angle = self.state.phase * 2 * np.pi
            min_radius = 0.2
            radius = min_radius + (stroke_len - min_radius) * event.intensity
            alpha = np.sin(angle) * radius
            beta = np.cos(angle) * radius

        return alpha, beta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_position(self) -> Tuple[float, float]:
        """Get current alpha/beta position for visualization."""
        return self.state.alpha, self.state.beta

    def reset(self):
        """Reset stroke mapper state."""
        self.state = StrokeState()
        self.state.alpha = self._park_alpha
        self.state.beta = self._park_beta
        self.spiral_beat_index = 0
        self._rms_envelope = 0.0
        self._motion_mode = MotionMode.CREEP_MICRO
        self._beat_phase = 0.0
        self._micro_jerk_alpha = 0.0
        self._micro_jerk_beta = 0.0
        self._last_micro_jerk_time = 0.0
        self._bass_jitter_speed_mult = 1.0
        self._trajectory = None
        self._beats_since_stroke = 0


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import Config

    config = Config()
    mapper = StrokeMapper(config)

    for i in range(10):
        event = BeatEvent(
            timestamp=time.time(),
            intensity=random.uniform(0.3, 1.0),
            frequency=random.uniform(50, 5000),
            is_beat=(i % 2 == 0),
            spectral_flux=random.uniform(0, 1),
            peak_energy=random.uniform(0, 1)
        )
        cmd = mapper.process_beat(event)
        if cmd:
            log_event("INFO", "StrokeMapper", "Test beat", index=i, tcode=cmd.to_tcode().strip())
        time.sleep(0.2)
