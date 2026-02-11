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
import threading
from collections import deque
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

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
      - Downbeat anchored at top of circle

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
        self._phase_time: float = time.time()
        self._current_bpm: float = 0.0

        # ---------- Full-stroke arc thread ----------
        self._arc_thread: Optional[threading.Thread] = None
        self._stop_arc: bool = False
        self._arc_band_volume: float = 1.0

        # ---------- Micro-effect state ----------
        self._micro_effects_enabled: bool = True   # toggle from GUI
        self._last_micro_jerk_time: float = 0.0
        self._micro_jerk_alpha: float = 0.0
        self._micro_jerk_beta: float = 0.0
        self._micro_jerk_decay_ms: float = 120.0   # jerk decays over this many ms

        # ---------- Band energy trackers (updated from BeatEvent) ----------
        self._mid_energy: float = 0.0
        self._high_energy: float = 0.0

        # ---------- Band-based scaling tables ----------
        self._band_volume_scale = {
            'sub_bass': 1.00,
            'low_mid':  0.98,
            'mid':      0.97,
            'high':     0.93,
        }
        self._band_speed_scale = {
            'sub_bass': 0.70,
            'low_mid':  0.85,
            'mid':      1.00,
            'high':     1.20,
        }

        # ---------- Spiral mode persistent state ----------
        self.spiral_beat_index = 0
        self.spiral_revolutions = 3
        self.spiral_reset_active = False
        self.spiral_reset_start_time = 0.0
        self.spiral_reset_from = (0.0, 0.0)

        # ---------- Flux tracking ----------
        self._flux_history: deque = deque()
        self._flux_rise_window_ms: float = 250.0
        self._flux_stroke_factor: float = 1.0

        # ---------- Fade / silence ----------
        self._fade_intensity: float = 1.0
        self._last_quiet_time: float = 0.0
        self._consecutive_silent_count: int = 0

        # ---------- Creep volume fade ----------
        self._creep_sustained_start: float = 0.0
        self._creep_volume_factor: float = 1.0
        self._creep_was_active_last_frame: bool = False

        # ---------- Idle motion throttle (separate from beat stroke timing) ----------
        self._last_idle_time: float = 0.0

        # ---------- Post-arc smooth blend ----------
        # After an arc completes, smoothly blend from arc endpoint to creep orbit
        self._post_arc_blend: float = 1.0  # 1.0 = fully on creep orbit (start normal), reset to 0.0 after arc
        self._post_arc_blend_rate: float = 0.05  # per frame (at 60fps, ~20 frames = 333ms to settle)

        # ---------- Beat factoring ----------
        self.max_strokes_per_sec = 4.5
        self.beat_factor = 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vol_floor(self, base_vol: float) -> float:
        """Minimum allowed volume given vol_reduction_limit config."""
        limit_pct = self.config.stroke.vol_reduction_limit / 100.0
        return base_vol * (1.0 - limit_pct)

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

    def _get_band_volume(self, event: BeatEvent) -> float:
        band = getattr(event, 'beat_band', 'sub_bass')
        base_vol = self.get_volume()
        band_reduction = (1.0 - self._band_volume_scale.get(band, 1.0)) * base_vol
        return max(self._vol_floor(base_vol), base_vol - band_reduction)

    def _get_band_duration_scale(self, event: BeatEvent) -> float:
        band = getattr(event, 'beat_band', 'sub_bass')
        return self._band_speed_scale.get(band, 1.0)

    def _freq_to_factor(self, freq: float) -> float:
        """Convert frequency -> 0-1 factor.  Lower (bass) -> 0 -> deeper strokes."""
        cfg = self.config.stroke
        low, high = cfg.depth_freq_low, cfg.depth_freq_high
        if freq <= low:
            return 0.0
        elif freq >= high:
            return 1.0
        return (freq - low) / (high - low)

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

    def _update_motion_mode(self) -> None:
        """Switch between FULL_STROKE and CREEP_MICRO with hysteresis."""
        now = time.time()
        cfg = self.config.stroke
        gate_high = cfg.amplitude_gate_high
        gate_low = cfg.amplitude_gate_low
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
        now = time.time()
        dt = now - self._phase_time
        self._phase_time = now

        bpm = getattr(event, 'bpm', 0.0) or 0.0
        self._current_bpm = bpm

        if bpm > 0 and dt > 0 and dt < 1.0:
            beats_per_sec = bpm / 60.0
            self._beat_phase += beats_per_sec * dt
            self._beat_phase %= 1.0

    # ------------------------------------------------------------------
    # Band energy extraction (for micro-effects)
    # ------------------------------------------------------------------

    def _update_band_energies(self, event: BeatEvent) -> None:
        """Extract mid/high energy from audio_engine for micro-effect scaling."""
        if self.audio_engine and hasattr(self.audio_engine, '_band_energies'):
            energies = self.audio_engine._band_energies
            # Smooth tracking
            alpha = 0.2
            self._mid_energy += (energies.get('mid', 0.0) - self._mid_energy) * alpha
            self._high_energy += (energies.get('high', 0.0) - self._high_energy) * alpha

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
        now = time.time()
        cfg = self.config.stroke
        beat_cfg = self.config.beat

        # ===== LOW-BAND MOTION FILTER =====
        _BAND_LOWER_HZ = {'sub_bass': 30, 'low_mid': 100, 'mid': 500, 'high': 2000}
        cutoff = beat_cfg.motion_freq_cutoff
        if cutoff > 0 and event.is_beat:
            fired = getattr(event, 'fired_bands', None) or []
            if fired:
                has_low_band = any(_BAND_LOWER_HZ.get(b, 0) < cutoff for b in fired)
                if not has_low_band:
                    return None

        # Update continuous trackers
        self._update_flux_history(event)
        self._update_envelope(event)
        self._update_motion_mode()
        self._advance_phase(event)
        self._update_band_energies(event)

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
                    if self.audio_engine and elapsed > silence_reset_threshold:
                        self.audio_engine.reset_tempo_tracking()
                else:
                    self._fade_intensity = 0.0
        else:
            self._consecutive_silent_count = 0
            self._fade_intensity = min(1.0, self._fade_intensity + 0.1)
            self._last_quiet_time = 0.0

        # Track idle time
        if event.is_beat:
            self.state.idle_time = 0.0
            self.state.last_beat_time = now
        else:
            self.state.idle_time = now - self.state.last_beat_time if self.state.last_beat_time > 0 else 0.0

        # ===== FLUX FACTOR (for stroke scaling) =====
        if event.is_beat:
            flux_ratio = event.spectral_flux / max(cfg.flux_threshold, 0.001)
            flux_ratio = np.clip(flux_ratio, 0.2, 3.0)
            base_factor = 0.5 + (flux_ratio / 3.0)
            scaling_weight = cfg.flux_scaling_weight
            self._flux_stroke_factor = 1.0 + (base_factor - 1.0) * scaling_weight

        # ===== DISPATCH by behavioral mode =====
        if event.is_beat:
            time_since_stroke = (now - self.state.last_stroke_time) * 1000
            if time_since_stroke < cfg.min_interval_ms:
                return None

            is_downbeat = getattr(event, 'is_downbeat', False)

            if self._motion_mode == MotionMode.FULL_STROKE:
                # High amplitude -> full arc strokes
                if is_downbeat:
                    cmd = self._generate_downbeat_stroke(event)
                    return self._apply_fade(cmd)
                else:
                    is_high_flux = event.spectral_flux >= cfg.flux_threshold
                    if not is_high_flux:
                        return None
                    cmd = self._generate_beat_stroke(event)
                    return self._apply_fade(cmd)

            else:  # CREEP_MICRO
                # Low amplitude -> micro-effects on beats, plus produce creep motion
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
            cmd.volume *= self._fade_intensity
        return cmd if self._fade_intensity > 0.01 else None

    # ------------------------------------------------------------------
    # FULL_STROKE generators (same proven logic from v1)
    # ------------------------------------------------------------------

    def _generate_downbeat_stroke(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """Full measure-length arc on downbeat.  When tempo LOCKED -> 25% boost."""
        cfg = self.config.stroke
        now = time.time()

        # Stop any running arc
        if hasattr(self, '_arc_thread') and self._arc_thread and self._arc_thread.is_alive():
            self._stop_arc = True

        # Measure duration
        if self.state.last_beat_time == 0.0:
            measure_duration_ms = 2000
        else:
            beat_interval_ms = (now - self.state.last_beat_time) * 1000
            beat_interval_ms = max(cfg.min_interval_ms, min(1000, beat_interval_ms))
            measure_duration_ms = beat_interval_ms * 4

        band_speed = self._get_band_duration_scale(event)
        measure_duration_ms = int(measure_duration_ms * band_speed)

        flux_factor = getattr(self, '_flux_stroke_factor', 1.0)
        tempo_locked = getattr(event, 'tempo_locked', False)
        lock_boost = 1.25 if tempo_locked else 1.0

        stroke_len = cfg.stroke_max * flux_factor * lock_boost * self.motion_intensity
        stroke_len = max(cfg.stroke_min, min(cfg.stroke_max * 1.25, stroke_len))

        freq_factor = self._freq_to_factor(event.frequency)
        depth = cfg.minimum_depth + (1.0 - cfg.minimum_depth) * (1.0 - cfg.freq_depth_factor * freq_factor)
        if cfg.flux_depth_factor > 0:
            flux_rise = self._get_flux_rise_factor()
            depth = cfg.minimum_depth + (depth - cfg.minimum_depth) * max(0.0, 1.0 - cfg.flux_depth_factor * flux_rise)

        min_radius = 0.3
        radius = min_radius + (1.0 - min_radius) * flux_factor * lock_boost
        radius = max(min_radius, min(1.0, radius))

        alpha_weight = self.config.alpha_weight
        beta_weight = self.config.beta_weight

        n_points = max(16, int(measure_duration_ms / 20))
        # Start arc from current creep angle position, advance through one full circle
        current_phase = self.state.creep_angle / (2 * np.pi)
        arc_phases = np.linspace(current_phase, current_phase + 1.0, n_points, endpoint=False) % 1.0
        alpha_arc = np.zeros(n_points)
        beta_arc = np.zeros(n_points)
        for i, phase in enumerate(arc_phases):
            prev_phase = self.state.phase
            self.state.phase = phase
            a, b = self._get_stroke_target(stroke_len, depth, event)
            alpha_arc[i] = a
            beta_arc[i] = b
            self.state.phase = prev_phase

        base_step = measure_duration_ms // n_points
        remainder = measure_duration_ms % n_points
        step_durations = [base_step + 1 if i < remainder else base_step for i in range(n_points)]

        self._stop_arc = False
        self._arc_band_volume = self._get_band_volume(event)
        self._arc_thread = threading.Thread(
            target=self._send_arc_synchronous,
            args=[alpha_arc, beta_arc, step_durations, n_points],
            daemon=True
        )
        self._arc_thread.start()

        self.state.last_stroke_time = now
        self.state.last_beat_time = now
        lock_str = "LOCKED+BOOST" if tempo_locked else "unlocked"
        log_event("INFO", "StrokeMapper", "Downbeat arc start",
                  mode=cfg.mode.name, points=n_points,
                  duration_ms=measure_duration_ms, tempo_state=lock_str)
        return None  # arc thread handles sending

    def _generate_beat_stroke(self, event: BeatEvent) -> TCodeCommand:
        """Full arc stroke for a regular detected beat."""
        cfg = self.config.stroke
        now = time.time()

        if hasattr(self, '_arc_thread') and self._arc_thread and self._arc_thread.is_alive():
            self._stop_arc = True

        beat_interval_ms = (now - self.state.last_beat_time) * 1000 if self.state.last_beat_time > 0 else cfg.min_interval_ms
        beat_interval_ms = max(cfg.min_interval_ms, min(1000, beat_interval_ms))
        beat_interval_ms *= 2

        band_speed = self._get_band_duration_scale(event)
        beat_interval_ms = int(beat_interval_ms * band_speed)

        intensity = event.intensity
        flux_factor = getattr(self, '_flux_stroke_factor', 1.0)

        base_stroke_len = cfg.stroke_min + (cfg.stroke_max - cfg.stroke_min) * intensity * cfg.stroke_fullness
        stroke_len = base_stroke_len * flux_factor * self.motion_intensity
        stroke_len = max(cfg.stroke_min, min(cfg.stroke_max, stroke_len))

        freq_factor = self._freq_to_factor(event.frequency)
        depth = cfg.minimum_depth + (1.0 - cfg.minimum_depth) * (1.0 - cfg.freq_depth_factor * freq_factor)
        if cfg.flux_depth_factor > 0:
            flux_rise = self._get_flux_rise_factor()
            depth = cfg.minimum_depth + (depth - cfg.minimum_depth) * max(0.0, 1.0 - cfg.flux_depth_factor * flux_rise)

        min_radius = 0.2
        max_radius = 1.0
        base_radius = min_radius + (max_radius - min_radius) * intensity
        radius = base_radius * flux_factor
        radius = max(min_radius, min(1.0, radius))

        alpha_weight = self.config.alpha_weight
        beta_weight = self.config.beta_weight

        if cfg.mode == StrokeMode.SPIRAL:
            N = self.spiral_revolutions
            prev_index = getattr(self, 'spiral_beat_index', 0)
            next_index = prev_index + 1
            theta_prev = (prev_index / N) * (2 * np.pi * N)
            theta_next = (next_index / N) * (2 * np.pi * N)
            n_points = max(8, int(beat_interval_ms / 10))
            thetas = np.linspace(theta_prev, theta_next, n_points)
            alpha_arc = np.zeros(n_points)
            beta_arc = np.zeros(n_points)
            for i, theta in enumerate(thetas):
                margin = 0.1
                b_coeff = (1.0 - margin) / (2 * np.pi * N)
                r = b_coeff * theta * stroke_len * depth * intensity
                a = r * np.cos(theta) * alpha_weight
                b_ = r * np.sin(theta) * beta_weight
                alpha_arc[i] = np.clip(a, -1.0, 1.0)
                beta_arc[i] = np.clip(b_, -1.0, 1.0)
            self.spiral_beat_index = next_index % N
        else:
            n_points = max(8, int(beat_interval_ms / 10))
            # Start arc from current creep angle position, advance through one full circle
            current_phase = self.state.creep_angle / (2 * np.pi)
            arc_phases = np.linspace(current_phase, current_phase + 1.0, n_points, endpoint=False) % 1.0
            alpha_arc = np.zeros(n_points)
            beta_arc = np.zeros(n_points)
            for i, phase in enumerate(arc_phases):
                prev_phase = self.state.phase
                self.state.phase = phase
                a, b = self._get_stroke_target(stroke_len, depth, event)
                alpha_arc[i] = a
                beta_arc[i] = b
                self.state.phase = prev_phase

        base_step = beat_interval_ms // n_points
        remainder = beat_interval_ms % n_points
        step_durations = [base_step + 1 if i < remainder else base_step for i in range(n_points)]

        self._stop_arc = False
        self._arc_band_volume = self._get_band_volume(event)
        self._arc_thread = threading.Thread(
            target=self._send_arc_synchronous,
            args=[alpha_arc, beta_arc, step_durations, n_points],
            daemon=True
        )
        self._arc_thread.start()

        self.state.last_stroke_time = now
        self.state.last_beat_time = now

        first_alpha = float(alpha_arc[0])
        first_beta = float(beta_arc[0])
        self.state.alpha = first_alpha
        self.state.beta = first_beta

        band = getattr(event, 'beat_band', 'sub_bass')
        log_event("INFO", "StrokeMapper", "Arc start",
                  mode=cfg.mode.name, points=n_points,
                  duration_ms=beat_interval_ms, band=band,
                  motion=f"{self.motion_intensity:.2f}")
        return TCodeCommand(first_alpha, first_beta, step_durations[0], self._arc_band_volume)

    # ------------------------------------------------------------------
    # Arc sender (synchronous thread - same as v1)
    # ------------------------------------------------------------------

    def _send_arc_synchronous(self, alpha_arc, beta_arc, step_durations, n_points):
        """Send arc points synchronously with proper sleep timing."""
        for i in range(1, n_points):
            if self._stop_arc:
                log_event("INFO", "StrokeMapper", "Arc interrupted", point=i)
                return

            alpha = float(alpha_arc[i])
            beta = float(beta_arc[i])
            step_ms = step_durations[i]

            if self.send_callback:
                band_vol = getattr(self, '_arc_band_volume', self.get_volume())
                fade_reduction = (1.0 - self._fade_intensity) * band_vol
                volume = max(self._vol_floor(band_vol), band_vol - fade_reduction)
                cmd = TCodeCommand(alpha, beta, step_ms, volume)
                self.send_callback(cmd)
                self.state.alpha = alpha
                self.state.beta = beta

            time.sleep(step_ms / 1000.0)

        log_event("INFO", "StrokeMapper", "Arc complete", points=n_points)
        # Sync creep angle to where arc ended so creep continues smoothly
        self._sync_creep_angle_to_position()
        # Start smooth blend from arc endpoint to creep orbit
        self._post_arc_blend = 0.0

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
        # Base jerk magnitude: small displacement (0.02-0.10)
        base_mag = 0.04
        if is_downbeat:
            base_mag = 0.08  # stronger on downbeat

        # Scale by mid+high energy for musical responsiveness
        band_scale = 1.0 + (self._mid_energy + self._high_energy) * 5.0
        band_scale = min(band_scale, 3.0)

        mag = base_mag * band_scale * self.motion_intensity

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
        """Generate jitter/creep/micro-jerk motion when idle."""
        now = time.time()
        jitter_cfg = self.config.jitter
        creep_cfg = self.config.creep

        # 60 fps throttle (use separate timer from beat strokes)
        time_since_last = (now - self._last_idle_time) * 1000
        if time_since_last < 17:
            return None

        jitter_active = jitter_cfg.enabled and jitter_cfg.amplitude > 0
        creep_active = creep_cfg.enabled and creep_cfg.speed > 0

        if not jitter_active and not creep_active and not self.spiral_reset_active and not self.state.creep_reset_active:
            # Still allow micro-jerk decay to produce motion
            jerk_a, jerk_b = self._get_micro_jerk_offset()
            if abs(jerk_a) < 0.001 and abs(jerk_b) < 0.001:
                return None

        alpha, beta = self.state.alpha, self.state.beta

        # ---------- Spiral reset (fade to center) ----------
        if self.spiral_reset_active:
            reset_duration_ms = 400
            elapsed_ms = (now - self.spiral_reset_start_time) * 1000
            if elapsed_ms < reset_duration_ms:
                progress = elapsed_ms / reset_duration_ms
                eased = 1.0 - (1.0 - progress) ** 2
                from_a, from_b = self.spiral_reset_from
                alpha_t = from_a * (1.0 - eased)
                beta_t = from_b * (1.0 - eased)
                self.state.alpha = alpha_t
                self.state.beta = beta_t
                self._last_idle_time = now
                fade = self._fade_intensity
                volume = self.get_volume() * fade
                return TCodeCommand(alpha_t, beta_t, 200, volume)
            else:
                self.spiral_reset_active = False
                self.state.alpha = 0.0
                self.state.beta = 0.0
                self._sync_creep_angle_to_position()

        elif self.state.creep_reset_active:
            reset_duration_ms = 400
            elapsed_ms = (now - self.state.creep_reset_start_time) * 1000
            if elapsed_ms < reset_duration_ms:
                progress = elapsed_ms / reset_duration_ms
                eased_progress = 1.0 - (1.0 - progress) ** 2
                try:
                    current_angle = float(self.state.creep_angle)
                    if not np.isfinite(current_angle):
                        current_angle = 0.0
                except:
                    current_angle = 0.0
                while current_angle > np.pi:
                    current_angle -= 2 * np.pi
                while current_angle < -np.pi:
                    current_angle += 2 * np.pi
                self.state.creep_angle = current_angle * (1.0 - eased_progress)
            else:
                self.state.creep_angle = 0.0
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
            
            bpm = getattr(event, 'bpm', 0.0) if event else 0.0

            if bpm > 0:
                beats_per_sec = bpm / 60.0
                updates_per_sec = 1000.0 / 17.0
                updates_per_beat = updates_per_sec / beats_per_sec
                angle_increment = (np.pi / 2.0) / updates_per_beat * creep_cfg.speed

                if self._motion_mode == MotionMode.CREEP_MICRO:
                    # In CREEP_MICRO: one full rotation per measure (4 beats -> 2pi)
                    # Override speed: exactly 2pi per measure
                    angle_increment = (2 * np.pi) / (updates_per_beat * self.config.beat.beats_per_measure)

                if not self.state.creep_reset_active:
                    self.state.creep_angle += angle_increment
                    if self.state.creep_angle >= 2 * np.pi:
                        self.state.creep_angle -= 2 * np.pi

                jitter_r = jitter_cfg.amplitude if jitter_active else 0.0

                if self._motion_mode == MotionMode.CREEP_MICRO:
                    # CREEP_MICRO: smaller radius, keep away from edges
                    creep_radius = max(0.1, 0.5 - jitter_r)
                else:
                    creep_radius = max(0.1, 0.98 - jitter_r)

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
                if not self.state.creep_reset_active:
                    self.state.creep_angle += creep_cfg.speed * 0.02
                    if self.state.creep_angle >= 2 * np.pi:
                        self.state.creep_angle -= 2 * np.pi
                oscillation = 0.2 + 0.1 * np.sin(self.state.creep_angle)
                target_alpha = oscillation * np.sin(self.state.creep_angle * 0.5)
                target_beta = oscillation * np.cos(self.state.creep_angle * 0.5) - 0.2

                # Smooth blend for non-creep mode too
                if self._post_arc_blend < 1.0:
                    self._post_arc_blend = min(1.0, self._post_arc_blend + self._post_arc_blend_rate)
                    base_alpha = alpha + (target_alpha - alpha) * self._post_arc_blend
                    base_beta = beta + (target_beta - beta) * self._post_arc_blend
                else:
                    base_alpha = target_alpha
                    base_beta = target_beta
        else:
            base_alpha = alpha
            base_beta = beta

        # ---------- Jitter: sinusoidal micro-circles ----------
        if jitter_active:
            if self._motion_mode == MotionMode.CREEP_MICRO:
                # CREEP_MICRO: slower, smaller jitter
                jitter_speed = jitter_cfg.intensity * 0.08
                jitter_r = jitter_cfg.amplitude * 0.5
            else:
                jitter_speed = jitter_cfg.intensity * 0.15
                jitter_r = jitter_cfg.amplitude

            # Modulate jitter size by mid/high energy in CREEP_MICRO mode
            if self._motion_mode == MotionMode.CREEP_MICRO and self._micro_effects_enabled:
                energy_mod = 1.0 + (self._mid_energy + self._high_energy) * 3.0
                energy_mod = min(energy_mod, 2.5)
                jitter_r *= energy_mod
                jitter_speed *= (0.8 + energy_mod * 0.2)

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

        duration_ms = 200  # smooth continuous motion

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
        phase_advance = self.config.stroke.phase_advance

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
        self.spiral_beat_index = 0
        self._rms_envelope = 0.0
        self._motion_mode = MotionMode.CREEP_MICRO
        self._beat_phase = 0.0
        self._micro_jerk_alpha = 0.0
        self._micro_jerk_beta = 0.0
        self._last_micro_jerk_time = 0.0


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
