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
from collections import deque
from typing import Optional, Tuple, Callable, List
from dataclasses import dataclass, field

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
    beat_target_time: float = 0.0  # Wall-clock time when the dot should "land" on the beat
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

        # ---------- Full-stroke planned trajectory ----------
        self._trajectory: Optional[PlannedTrajectory] = None

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

        # ---------- Pending arc: glide to top/bottom before firing ----------
        self._pending_arc_event: Optional[BeatEvent] = None
        self._pending_arc_target: float = 0.0       # 0.0 = top, π = bottom
        self._pending_arc_is_downbeat: bool = False
        self._arc_anchor_threshold: float = 0.35     # radians (~20°) — close enough to fire

        # ---------- Locked anchor: pick top or bottom once, keep until silence ----------
        # None = unlocked (first beat picks nearest), 0.0 = locked to top, π = locked to bottom
        self._locked_anchor: Optional[float] = None

        # ---------- Stroke readiness gating ----------
        # Strokes only fire when metronome + traffic light conditions are met:
        #   Option A: metronome GREEN + traffic YELLOW or GREEN
        #   Option B: metronome GREEN or YELLOW + traffic GREEN
        #   Option C: traffic YELLOW (was recently GREEN) + metronome YELLOW or GREEN
        #   Option D: metronome GREEN stable >2s + any traffic state
        # Otherwise: creep/jitter only
        # Grace period: 1300ms after conditions drop before returning to jitter
        self._stroke_ready: bool = False
        self._stroke_ready_lost_time: float = 0.0   # when conditions last dropped
        self._stroke_grace_ms: float = 1300.0        # grace period before disabling strokes
        self._traffic_was_green: bool = False         # track if traffic was recently green
        self._traffic_left_green_time: float = 0.0    # when traffic left green
        self._metro_green_since: float = 0.0          # when metronome first became green
        self._prev_had_any_light: bool = False        # was at least one light yellow+ last check (track cold-start)

        # ---------- Last confirmed beat time (for no-beat timeout) ----------
        self._last_confirmed_beat_time: float = 0.0   # wall-clock of last beat with stroke_ready

        # ---------- Snap timing feedback (self-checking) ----------
        # When snap-to-target fires, record the timing error so the next arc
        # can compensate by shortening/lengthening its duration.
        self._last_snap_correction_ms: float = 0.0
        self._no_beat_timeout_s: float = 2.0           # seconds before returning to center+jitter

        # ---------- Post-silence volume ramp ----------
        # After silence/track-change reset, reduce volume and slowly ramp back up
        self._post_silence_ramp_active: bool = False
        self._post_silence_ramp_start: float = 0.0     # time.time() when ramp started
        self._was_silent: bool = False                  # track if we were faded out

        # ---------- Flux-drop fallback ----------
        self._recent_flux_values: deque = deque(maxlen=60)  # ~1s of flux history for drop detection

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

    def _update_stroke_readiness(self, event: BeatEvent) -> None:
        """Determine if strokes should fire based on metronome + traffic light.
        
        Rules (both lights = metronome + traffic):
          - Both GREEN → fire strokes
          - One GREEN + one YELLOW → fire strokes
          - Both YELLOW → fire strokes ONLY if previously had at least one
            light on (not coming from red/off state cold-start), OR if the
            current frame has a confirmed beat/downbeat indicator
          - Anything below both-yellow → don't fire (grace period applies)
        
        Grace period: when conditions drop, strokes continue for 1300ms
        before reverting to jitter. This prevents brief dips from
        interrupting an ongoing stroke pattern.
        
        If no metrics are enabled (no traffic light), only metronome matters.
        """
        acf_conf = getattr(event, 'acf_confidence', 0.0)
        metro_bpm = getattr(event, 'metronome_bpm', 0.0)
        now = time.time()
        
        metro_green = acf_conf >= 0.25 and metro_bpm > 0
        metro_yellow = acf_conf >= 0.05 and metro_bpm > 0
        
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
            
            conditions_met = (both_green or mixed_green_yellow
                              or both_yellow_ok or option_recovery or option_stable)
        else:
            conditions_met = metro_yellow

        # Update previous-light tracking for next iteration
        self._prev_had_any_light = has_any_light
        
        if conditions_met:
            # Conditions met — immediately ready, reset lost timer
            self._stroke_ready = True
            self._stroke_ready_lost_time = 0.0
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
        self._update_stroke_readiness(event)
        self._advance_phase(event)
        self._update_band_energies(event)

        # ===== FLUX-DROP FALLBACK =====
        # Track recent flux for drop detection — if upper spectrum flux
        # drops significantly, force back to creep mode
        self._recent_flux_values.append(event.spectral_flux)
        if len(self._recent_flux_values) >= 30:
            recent_avg = sum(list(self._recent_flux_values)[-15:]) / 15.0
            older_avg = sum(list(self._recent_flux_values)[:15]) / 15.0
            flux_drop_ratio = cfg.flux_drop_ratio if hasattr(cfg, 'flux_drop_ratio') else 0.25
            if older_avg > 0.01 and recent_avg < older_avg * flux_drop_ratio:
                # Flux dropped to <25% of recent levels — force creep
                if self._motion_mode == MotionMode.FULL_STROKE:
                    self._motion_mode = MotionMode.CREEP_MICRO
                    self._mode_switch_time = now
                    self._trajectory = None
                    self._pending_arc_event = None
                    self._sync_creep_angle_to_position()
                    log_event("INFO", "StrokeMapper", "Flux drop → creep fallback",
                              recent=f"{recent_avg:.4f}", older=f"{older_avg:.4f}")

        # ===== NO-BEAT TIMEOUT =====
        # Track last confirmed beat (stroke_ready + is_beat)
        if event.is_beat and self._stroke_ready:
            self._last_confirmed_beat_time = now
        # If no confirmed beat for 2s, cancel trajectory and return to center+jitter
        if (self._last_confirmed_beat_time > 0
                and (now - self._last_confirmed_beat_time) > self._no_beat_timeout_s
                and self._trajectory is not None):
            self._trajectory = None
            self._locked_anchor = None
            self._pending_arc_event = None
            # Start creep reset to center
            if not self.state.creep_reset_active:
                self.state.creep_reset_active = True
                self.state.creep_reset_start_time = now
            log_event("INFO", "StrokeMapper", "No-beat timeout → center+jitter")

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
                        self._locked_anchor = None  # unlock anchor for next song/section
                else:
                    self._fade_intensity = 0.0
                    self._was_silent = True
        else:
            self._consecutive_silent_count = 0
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
        if event.is_beat:
            flux_ratio = event.spectral_flux / max(cfg.flux_threshold, 0.001)
            flux_ratio = np.clip(flux_ratio, 0.2, 3.0)
            base_factor = 0.5 + (flux_ratio / 3.0)
            scaling_weight = cfg.flux_scaling_weight
            self._flux_stroke_factor = 1.0 + (base_factor - 1.0) * scaling_weight

        # ===== DISPATCH by behavioral mode =====
        # ===== SYNCOPATION: off-beat "and" onset detected =====
        # If metronome detects an off-beat raw onset between beats, fire a
        # 2x-speed full-circle "double" arc for the duh-DUH effect.
        is_syncopated = getattr(event, 'is_syncopated', False)
        metro_bpm = getattr(event, 'metronome_bpm', 0.0)
        if is_syncopated and self._motion_mode == MotionMode.FULL_STROKE:
            # BPM limit from config
            bpm_limit = beat_cfg.syncopation_bpm_limit if hasattr(beat_cfg, 'syncopation_bpm_limit') else 160.0
            if metro_bpm > bpm_limit:
                pass
            elif self._trajectory is None or self._trajectory.finished:
                time_since_stroke = (now - self.state.last_stroke_time) * 1000
                if time_since_stroke >= cfg.min_interval_ms * 0.5:
                    cmd = self._generate_syncopated_stroke(event)
                    return self._apply_fade(cmd)

        # ===== NOISE-PRIMARY MODE =====
        # When enabled: noise (flux spike) fires strokes immediately,
        # and the metronome only validates timing (the reverse of default).
        if (cfg.noise_primary_mode
                and not event.is_beat
                and cfg.noise_burst_enabled
                and self._motion_mode == MotionMode.FULL_STROKE
                and (self._trajectory is None or self._trajectory.finished)):
            noise_thresh = cfg.flux_threshold * cfg.noise_burst_flux_multiplier
            if event.spectral_flux >= noise_thresh:
                time_since_stroke = (now - self.state.last_stroke_time) * 1000
                if time_since_stroke >= cfg.min_interval_ms * 0.4:
                    # In noise-primary mode, fire a FULL beat stroke (not a burst)
                    # using the metronome BPM for duration if available
                    cmd = self._generate_beat_stroke(event)
                    return self._apply_fade(cmd)

        # ===== NOISE BURST (non-primary): small jitter on flux spikes =====
        # Only active when creep is engaged (CREEP_MICRO mode).
        # Produces small random jerks/swirls instead of full-circle arcs.
        if (not cfg.noise_primary_mode
                and not event.is_beat
                and cfg.noise_burst_enabled
                and self._motion_mode == MotionMode.CREEP_MICRO
                and self.config.creep.enabled
                and (self._trajectory is None or self._trajectory.finished)):
            noise_thresh = cfg.flux_threshold * cfg.noise_burst_flux_multiplier
            if event.spectral_flux >= noise_thresh:
                time_since_stroke = (now - self.state.last_stroke_time) * 1000
                if time_since_stroke >= cfg.min_interval_ms * 0.4:
                    cmd = self._generate_noise_burst_stroke(event)
                    return self._apply_fade(cmd)

        if event.is_beat:
            time_since_stroke = (now - self.state.last_stroke_time) * 1000
            if time_since_stroke < cfg.min_interval_ms:
                return None

            # ===== STROKE READINESS GATE =====
            # If metronome + traffic light conditions not met,
            # fall through to idle motion (creep/jitter) instead of strokes
            if not self._stroke_ready:
                cmd = self._generate_idle_motion(event)
                return self._apply_fade(cmd)

            # Beats-between-strokes gating: only fire every Nth beat
            beats_skip = cfg.beats_between_strokes if hasattr(cfg, 'beats_between_strokes') else 1
            if beats_skip > 1:
                self._beats_since_stroke += 1
                is_downbeat = getattr(event, 'is_downbeat', False)
                # Always allow downbeats through, otherwise count
                if not is_downbeat and self._beats_since_stroke < beats_skip:
                    return None
                self._beats_since_stroke = 0

            is_downbeat = getattr(event, 'is_downbeat', False)

            if self._motion_mode == MotionMode.FULL_STROKE:
                # High amplitude -> fire arc immediately from current position.
                # No anchor gate — the dot sweeps 360° from wherever it is.
                # Continuous rotation means it passes through top/bottom naturally.
                self._pending_arc_event = None

                is_downbeat = getattr(event, 'is_downbeat', False)
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
            # Post-silence volume ramp: reduce volume and slowly raise back
            if self._post_silence_ramp_active:
                cfg = self.config.stroke
                elapsed = time.time() - self._post_silence_ramp_start
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
        return alpha, beta

    def _generate_downbeat_stroke(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """Full measure-length arc on downbeat.  When tempo LOCKED -> 25% boost.
        Stores a PlannedTrajectory; idle motion reads it frame-by-frame."""
        cfg = self.config.stroke
        now = time.time()

        # Beat duration — prefer metronome BPM if available
        # Downbeat arc spans the full measure for emphasis (beats_per_measure)
        beats_in_measure = max(1, getattr(self.config.beat, 'beats_per_measure', 4))
        metro_bpm = getattr(event, 'metronome_bpm', 0.0)
        if metro_bpm > 0:
            beat_interval_ms = 60000.0 / metro_bpm
            beat_interval_ms = max(cfg.min_interval_ms, min(1000, beat_interval_ms))
            measure_duration_ms = int(beat_interval_ms * beats_in_measure)
        elif self.state.last_beat_time == 0.0:
            measure_duration_ms = 500 * beats_in_measure
        else:
            beat_interval_ms = (now - self.state.last_beat_time) * 1000
            beat_interval_ms = max(cfg.min_interval_ms, min(1000, beat_interval_ms))
            measure_duration_ms = int(beat_interval_ms * beats_in_measure)

        # Clamp to avoid huge sweeps at very low BPM
        measure_duration_ms = max(cfg.min_interval_ms, min(4000, measure_duration_ms))

        # ===== PRE-FIRE: time arc to LAND on the next beat =====
        # Use predicted_next_beat for precise timing when metronome active.
        # Account for event processing latency too.
        beat_target_time = 0.0
        if self.audio_engine and hasattr(self.audio_engine, 'get_tempo_info'):
            tempo_info = self.audio_engine.get_tempo_info()
            predicted = tempo_info.get('predicted_next_beat', 0.0)
            if predicted > now:
                time_to_beat_ms = (predicted - now) * 1000
                if measure_duration_ms * 0.4 < time_to_beat_ms < measure_duration_ms * 2.0:
                    measure_duration_ms = int(time_to_beat_ms)
                    beat_target_time = predicted
        # Also account for event age (processing latency)
        event_age_ms = (now - event.timestamp) * 1000
        if beat_target_time == 0.0 and 0 < event_age_ms < measure_duration_ms * 0.3:
            measure_duration_ms = max(cfg.min_interval_ms, int(measure_duration_ms - event_age_ms))

        flux_factor = getattr(self, '_flux_stroke_factor', 1.0)
        tempo_locked = getattr(event, 'tempo_locked', False)
        # Slightly stronger downbeat boost than regular beats for emphasis
        lock_boost = 1.35 if tempo_locked else 1.15

        stroke_len = cfg.stroke_max * flux_factor * lock_boost * self.motion_intensity
        stroke_len = max(cfg.stroke_min, min(cfg.stroke_max * 1.25, stroke_len))

        freq_factor = self._freq_to_factor(event.frequency)
        depth = cfg.minimum_depth + (1.0 - cfg.minimum_depth) * (1.0 - cfg.freq_depth_factor * freq_factor)
        if cfg.flux_depth_factor > 0:
            flux_rise = self._get_flux_rise_factor()
            depth = cfg.minimum_depth + (depth - cfg.minimum_depth) * max(0.0, 1.0 - cfg.flux_depth_factor * flux_rise)

        min_radius = 0.3
        # Non-linear intensity curve: quiet taps small, loud taps dramatic
        curved_intensity = self._intensity_curve(event.intensity)
        radius = min_radius + (1.0 - min_radius) * flux_factor * lock_boost * curved_intensity
        radius = max(min_radius, min(1.0, radius))

        n_points = max(16, int(measure_duration_ms / 20))
        # Arc starts from current creep angle, sweeps exactly 360° over
        # one beat interval.  Creep is steered to top/bottom before we get here.
        current_phase = self.state.creep_angle / (2 * np.pi)
        arc_phases = np.linspace(current_phase, current_phase + 1.0, n_points, endpoint=False) % 1.0
        alpha_arc = np.zeros(n_points)
        beta_arc = np.zeros(n_points)
        arc_radius = min_radius + (stroke_len * depth - min_radius) * curved_intensity
        arc_radius = max(min_radius, min(1.0, arc_radius))
        for i, phase in enumerate(arc_phases):
            alpha_arc[i], beta_arc[i] = self._compute_arc_point(
                phase=phase,
                radius=arc_radius,
                stroke_len=stroke_len,
                depth=depth,
                event=event,
            )

        # Apply timing shape: thump (accelerate into landing),
        # landing (ease-in-out tap feel), or uniform
        if cfg.thump_enabled:
            step_durations = self._make_thump_durations(measure_duration_ms, n_points)
        else:
            # Landing emphasis: slow at start/end (tap feel), fast through middle
            step_durations = self._make_landing_durations(measure_duration_ms, n_points)

        # Store trajectory for frame-by-frame playback (no thread)
        self._trajectory = PlannedTrajectory(
            alpha_points=alpha_arc,
            beta_points=beta_arc,
            step_durations=step_durations,
            n_points=n_points,
            current_index=0,
            band_volume=self._get_band_volume(event),
            start_time=now,
            original_bpm=metro_bpm if metro_bpm > 0 else self._last_known_bpm,
            beat_target_time=beat_target_time,
        )

        self.state.last_stroke_time = now
        self.state.last_beat_time = now
        lock_str = "LOCKED+BOOST" if tempo_locked else "unlocked"
        log_event("INFO", "StrokeMapper", "Arc start",
                  mode=cfg.mode.name, points=n_points,
                  duration_ms=measure_duration_ms, tempo_state=lock_str,
                  pre_fire="yes" if beat_target_time > 0 else "no")
        return None  # idle motion will read from trajectory

    def _generate_beat_stroke(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """Full arc stroke for a regular detected beat.
        Stores a PlannedTrajectory; idle motion reads it frame-by-frame."""
        cfg = self.config.stroke
        now = time.time()

        # Prefer metronome BPM for beat timing
        metro_bpm = getattr(event, 'metronome_bpm', 0.0)
        if metro_bpm > 0:
            beat_interval_ms = 60000.0 / metro_bpm
            beat_interval_ms = max(cfg.min_interval_ms, min(1000, beat_interval_ms))
        elif self.state.last_beat_time > 0:
            beat_interval_ms = (now - self.state.last_beat_time) * 1000
            beat_interval_ms = max(cfg.min_interval_ms, min(1000, beat_interval_ms))
        else:
            beat_interval_ms = cfg.min_interval_ms
        # 2x duration adjusted by beats_between_strokes: arc fills the full span
        beats_per_arc = max(1, cfg.beats_between_strokes if hasattr(cfg, 'beats_between_strokes') else 2)
        beat_interval_ms = int(beat_interval_ms * beats_per_arc)

        # ===== PRE-FIRE: time arc to LAND on the next beat =====
        beat_target_time = 0.0
        if self.audio_engine and hasattr(self.audio_engine, 'get_tempo_info'):
            tempo_info = self.audio_engine.get_tempo_info()
            predicted = tempo_info.get('predicted_next_beat', 0.0)
            if predicted > now:
                time_to_beat_ms = (predicted - now) * 1000
                if beat_interval_ms * 0.4 < time_to_beat_ms < beat_interval_ms * 2.0:
                    beat_interval_ms = int(time_to_beat_ms)
                    beat_target_time = predicted
        # Fallback: account for event processing latency
        event_age_ms = (now - event.timestamp) * 1000
        if beat_target_time == 0.0 and 0 < event_age_ms < beat_interval_ms * 0.3:
            beat_interval_ms = max(cfg.min_interval_ms, int(beat_interval_ms - event_age_ms))

        # === SELF-CHECK: Apply snap timing correction from previous arc ===
        # If the last arc had to snap-to-target, the timing was slightly off.
        # Compensate by adjusting this arc's duration (e.g., if we were 20ms early,
        # extend the next arc by 20ms so the next landing takes that into account).
        if abs(self._last_snap_correction_ms) > 5.0:
            correction = self._last_snap_correction_ms * 0.7  # 70% correction
            beat_interval_ms = max(cfg.min_interval_ms, int(beat_interval_ms + correction))
            self._last_snap_correction_ms = 0.0  # consumed

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
        # Non-linear intensity curve: quiet taps small, loud taps dramatic
        curved_intensity = self._intensity_curve(intensity)
        base_radius = min_radius + (max_radius - min_radius) * curved_intensity
        radius = base_radius * flux_factor
        radius = max(min_radius, min(1.0, radius))

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
            alpha_weight = self.config.alpha_weight
            beta_weight = self.config.beta_weight
            for i, theta in enumerate(thetas):
                margin = 0.1
                b_coeff = (1.0 - margin) / (2 * np.pi * N)
                r = b_coeff * theta * stroke_len * depth * curved_intensity
                a = r * np.cos(theta) * alpha_weight
                b_ = r * np.sin(theta) * beta_weight
                alpha_arc[i] = np.clip(a, -1.0, 1.0)
                beta_arc[i] = np.clip(b_, -1.0, 1.0)
            self.spiral_beat_index = next_index % N
        else:
            n_points = max(8, int(beat_interval_ms / 10))
            # Arc starts from current creep angle, sweeps exactly 360°.
            current_phase = self.state.creep_angle / (2 * np.pi)
            arc_phases = np.linspace(current_phase, current_phase + 1.0, n_points, endpoint=False) % 1.0
            alpha_arc = np.zeros(n_points)
            beta_arc = np.zeros(n_points)
            arc_radius = min_radius + (max_radius - min_radius) * curved_intensity
            arc_radius = arc_radius * flux_factor
            arc_radius = max(min_radius, min(1.0, arc_radius))
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
            band_volume=self._get_band_volume(event),
            start_time=now,
            original_bpm=metro_bpm if metro_bpm > 0 else self._last_known_bpm,
            beat_target_time=beat_target_time,
        )

        self.state.last_stroke_time = now
        self.state.last_beat_time = now

        band = getattr(event, 'beat_band', 'sub_bass')
        log_event("INFO", "StrokeMapper", "Arc start",
                  mode=cfg.mode.name, points=n_points,
                  duration_ms=beat_interval_ms, band=band,
                  motion=f"{self.motion_intensity:.2f}",
                  pre_fire="yes" if beat_target_time > 0 else "no")
        return None  # idle motion will read from trajectory

    def _generate_syncopated_stroke(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """Lighter, quicker partial arc for an off-beat 'and' hit.
        Arc size and speed are configurable via syncopation_arc_size
        and syncopation_speed settings in Advanced Controls."""
        cfg = self.config.stroke
        beat_cfg = self.config.beat
        now = time.time()

        # Duration is configurable fraction of beat interval
        speed_frac = getattr(beat_cfg, 'syncopation_speed', 0.5)
        metro_bpm = getattr(event, 'metronome_bpm', 0.0)
        if metro_bpm > 0:
            beat_ms = 60000.0 / metro_bpm
        elif self.state.last_beat_time > 0:
            beat_ms = (now - self.state.last_beat_time) * 1000
        else:
            beat_ms = cfg.min_interval_ms * 2
        duration_ms = max(cfg.min_interval_ms * 0.4, min(1000, beat_ms * speed_frac))
        duration_ms = int(duration_ms)

        # Pre-fire: if metronome predicts next beat, adjust duration so
        # the syncopated arc LANDS on the next beat
        beat_target_time = 0.0
        if self.audio_engine and hasattr(self.audio_engine, 'get_tempo_info'):
            tempo_info = self.audio_engine.get_tempo_info()
            predicted = tempo_info.get('predicted_next_beat', 0.0)
            if predicted > now:
                time_to_beat_ms = (predicted - now) * 1000
                if duration_ms * 0.5 < time_to_beat_ms < duration_ms * 3.0:
                    duration_ms = int(time_to_beat_ms)
                    beat_target_time = predicted

        # Reduced amplitude for lighter feel (70% of normal)
        flux_factor = getattr(self, '_flux_stroke_factor', 1.0)
        intensity = event.intensity
        curved_intensity = self._intensity_curve(intensity)
        stroke_len = cfg.stroke_min + (cfg.stroke_max - cfg.stroke_min) * curved_intensity * cfg.stroke_fullness
        stroke_len = stroke_len * flux_factor * self.motion_intensity * 0.7
        stroke_len = max(cfg.stroke_min, min(cfg.stroke_max, stroke_len))

        freq_factor = self._freq_to_factor(event.frequency)
        depth = cfg.minimum_depth + (1.0 - cfg.minimum_depth) * (1.0 - cfg.freq_depth_factor * freq_factor)

        # Arc size: configurable fraction of circle (0.5 = 180°)
        arc_size = getattr(beat_cfg, 'syncopation_arc_size', 0.5)
        n_points = max(6, int(duration_ms / 12))
        current_phase = self.state.creep_angle / (2 * np.pi)
        arc_phases = np.linspace(current_phase, current_phase + arc_size, n_points, endpoint=False) % 1.0
        alpha_arc = np.zeros(n_points)
        beta_arc = np.zeros(n_points)

        min_radius = 0.15
        arc_radius = min_radius + (stroke_len * depth - min_radius) * curved_intensity * 0.7
        arc_radius = max(min_radius, min(0.8, arc_radius))
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
            band_volume=self._get_band_volume(event),
            start_time=now,
            beat_target_time=beat_target_time,
            original_bpm=metro_bpm if metro_bpm > 0 else self._last_known_bpm,
        )

        self.state.last_stroke_time = now
        log_event("INFO", "StrokeMapper", "Syncopated arc",
                  points=n_points, duration_ms=duration_ms,
                  arc_size=f"{arc_size:.2f}", speed=f"{speed_frac:.2f}")
        return None

    # ------------------------------------------------------------------
    # Noise-burst reactive arc (hybrid noise + metronome)
    # ------------------------------------------------------------------

    def _generate_noise_burst_stroke(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """Small random jitter/swirl patterns on sudden loud transients.
        Only fires in CREEP_MICRO mode when creep is active.
        Produces random tiny patterns: jerks, micro-swirls, star shapes, zigzags."""
        now = time.time()

        # Pick a random micro-pattern type
        pattern = random.choice(['jerk', 'swirl', 'star', 'zigzag'])
        magnitude_scale = getattr(self.config.stroke, 'noise_burst_magnitude', 1.0)
        jerk_mag = random.uniform(0.15, 0.40) * self.motion_intensity * magnitude_scale
        base_angle = self.state.creep_angle
        n_points = random.randint(4, 8)
        duration_ms = random.randint(60, 120)

        # Current creep position as center of the micro-pattern
        jitter_r = self.config.jitter.amplitude if self.config.jitter.enabled else 0.0
        creep_radius = max(0.1, 0.5 - jitter_r)
        center_a = np.sin(base_angle) * creep_radius
        center_b = np.cos(base_angle) * creep_radius

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
            band_volume=self._get_band_volume(event),
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

        now = time.time()

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

        # Jump to the time-correct index (skip frames if needed)
        target_idx = max(target_idx, traj.current_index)

        alpha = float(traj.alpha_points[target_idx])
        beta = float(traj.beta_points[target_idx])
        step_ms = traj.step_durations[target_idx]

        # Use short duration matching our update rate for smooth motion
        duration_ms = min(step_ms, 25)

        fade_reduction = (1.0 - self._fade_intensity) * traj.band_volume
        volume = max(self._vol_floor(traj.band_volume), traj.band_volume - fade_reduction)

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

        # ===== SNAP TO TARGET (only when about to miss the beat) =====
        # Only snap if we have a beat_target_time and the beat is imminent
        # (within 50ms). This prevents snapping during normal smooth arcs
        # and only activates when the dot would visibly miss the landing.
        if (traj.beat_target_time > 0
                and traj.current_index < traj.n_points
                and 0 < (traj.beat_target_time - now) < 0.050):
            final_a = float(traj.alpha_points[-1])
            final_b = float(traj.beta_points[-1])
            dist = np.sqrt((alpha - final_a)**2 + (beta - final_b)**2)
            if dist < 0.12:
                # Beat is <50ms away and we're close — snap to landing
                alpha = final_a
                beta = final_b
                self.state.alpha = alpha
                self.state.beta = beta
                traj.current_index = traj.n_points  # mark finished
                # === SELF-CHECK: Record timing discrepancy for next arc ===
                # If we had to snap, the arc was slightly too slow/fast.
                # Store the remaining time so next arc can compensate.
                snap_error_ms = (traj.beat_target_time - now) * 1000.0
                self._last_snap_correction_ms = snap_error_ms
                log_event("INFO", "StrokeMapper", "Snap-to-target fired",
                          error_ms=f"{snap_error_ms:.1f}", dist=f"{dist:.3f}")

        # Check if trajectory just completed
        if traj.finished:
            log_event("INFO", "StrokeMapper", "Arc complete", points=traj.n_points)
            self._sync_creep_angle_to_position()

            if getattr(traj, 'is_micro', False):
                # Micro patterns (noise jitter): resume creep
                self._post_arc_blend = 0.0
                self._trajectory = None
            elif (self._motion_mode == MotionMode.FULL_STROKE
                    and self._stroke_ready
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
        now = time.time()
        bpm = self._last_known_bpm
        if bpm <= 0:
            self._trajectory = None
            return

        beats_per_arc = max(1, cfg.beats_between_strokes if hasattr(cfg, 'beats_between_strokes') else 2)
        beat_interval_ms = int(60000.0 / bpm * beats_per_arc)
        beat_interval_ms = max(cfg.min_interval_ms, min(4000, beat_interval_ms))

        # ===== PRE-FIRE: time arc to LAND on beat =====
        # If we have a predicted next beat time from the metronome,
        # adjust arc duration so it finishes exactly when the beat hits.
        beat_target_time = 0.0
        if self.audio_engine and hasattr(self.audio_engine, 'get_tempo_info'):
            tempo_info = self.audio_engine.get_tempo_info()
            predicted = tempo_info.get('predicted_next_beat', 0.0)
            if predicted > now:
                # Time until next predicted beat
                time_to_beat_ms = (predicted - now) * 1000
                # If the predicted beat is within a reasonable range (0.5x to 2x
                # of our calculated interval), use it for precise timing
                if beat_interval_ms * 0.5 < time_to_beat_ms < beat_interval_ms * 2.0:
                    beat_interval_ms = int(time_to_beat_ms)
                    beat_target_time = predicted
                    log_event("DEBUG", "StrokeMapper", "Pre-fire: arc timed to land on beat",
                              time_to_beat_ms=f"{time_to_beat_ms:.0f}")

        # Reuse the last trajectory's radius for visual continuity.
        # Fall back to a moderate default if unavailable.
        prev_traj = self._trajectory
        prev_volume = prev_traj.band_volume if prev_traj else self.get_volume()

        # Recover radius from the last arc's endpoint
        last_r = np.sqrt(self.state.alpha**2 + self.state.beta**2)
        radius = max(0.2, min(1.0, last_r)) if last_r > 0.05 else 0.5

        alpha_weight = self.config.alpha_weight
        beta_weight = self.config.beta_weight

        n_points = max(8, int(beat_interval_ms / 10))
        current_phase = self.state.creep_angle / (2 * np.pi)
        arc_phases = np.linspace(current_phase, current_phase + 1.0,
                                 n_points, endpoint=False) % 1.0
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
            start_time=now,
            beat_target_time=beat_target_time,
            original_bpm=bpm,
        )

        log_event("INFO", "StrokeMapper", "Continuation arc",
                  bpm=f"{bpm:.1f}", points=n_points,
                  duration_ms=beat_interval_ms, radius=f"{radius:.2f}",
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
        # Base jerk magnitude: small displacement (0.02-0.10)
        base_mag = 0.10
        if is_downbeat:
            base_mag = 0.20  # stronger on downbeat

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
        """Generate motion: trajectory playback OR creep/jitter when idle."""
        now = time.time()
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
                # Also drive position toward center quickly
                self.state.alpha *= (1.0 - eased_progress * 0.3)
                self.state.beta *= (1.0 - eased_progress * 0.3)
            else:
                self.state.creep_angle = 0.0
                self.state.alpha = 0.0
                self.state.beta = 0.0
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
            # Persist last known BPM so creep continues at last tempo
            # when metronome confidence drops (instead of stopping)
            if bpm > 0:
                self._last_known_bpm = bpm
            elif self._last_known_bpm > 0:
                bpm = self._last_known_bpm

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
                    # Normal creep rotation — keeps moving smoothly between arcs
                    self.state.creep_angle += angle_increment
                    if self.state.creep_angle >= 2 * np.pi:
                        self.state.creep_angle -= 2 * np.pi

                jitter_r = jitter_cfg.amplitude if jitter_active else 0.0

                if self._motion_mode == MotionMode.CREEP_MICRO:
                    # CREEP_MICRO: smaller radius, drift toward center not edges
                    creep_radius = max(0.05, 0.20 - jitter_r)
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
                # No tempo: return to center, let jitter handle micro-motion
                # No orbital oscillation — just smoothly blend position toward (0, 0)
                if self._post_arc_blend < 1.0:
                    self._post_arc_blend = min(1.0, self._post_arc_blend + self._post_arc_blend_rate)
                    base_alpha = alpha * (1.0 - self._post_arc_blend)
                    base_beta = beta * (1.0 - self._post_arc_blend)
                else:
                    base_alpha = 0.0
                    base_beta = 0.0
        else:
            # Creep disabled: quickly wobble toward center so dot
            # doesn't get stuck at the edge after an arc finishes.
            blend_rate = 0.15  # per frame (~60fps → ~300ms to reach center)
            base_alpha = alpha * (1.0 - blend_rate)
            base_beta = beta * (1.0 - blend_rate)
            # Snap to zero when close enough to avoid perpetual micro-drift
            if abs(base_alpha) < 0.01:
                base_alpha = 0.0
            if abs(base_beta) < 0.01:
                base_beta = 0.0

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

        duration_ms = 25  # short duration matching update rate for smooth motion

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
