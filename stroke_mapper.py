"""
bREadbeats - Stroke Mapper
Converts beat events into alpha/beta stroke patterns.
All modes use circular coordinates around (0,0).
"""

import numpy as np
import time
import random
import threading
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

from config import Config, StrokeMode
from audio_engine import BeatEvent
from network_engine import TCodeCommand
from logging_utils import log_event


@dataclass
class StrokeState:
    """Current stroke position and state"""
    alpha: float = 0.0
    beta: float = 0.0
    target_alpha: float = 0.0
    target_beta: float = 0.0
    phase: float = 0.0           # 0-1 position in stroke cycle
    last_beat_time: float = 0.0
    last_stroke_time: float = 0.0
    idle_time: float = 0.0       # Time since last beat
    jitter_angle: float = 0.0    # Current jitter rotation
    creep_angle: float = 0.0     # Current creep rotation
    beat_counter: int = 0        # For beat skipping on fast tempos
    creep_reset_start_time: float = 0.0  # When creep reset began
    creep_reset_active: bool = False     # Whether creep is resetting to 0
    # Smooth arc return state
    arc_return_active: bool = False      # Whether arc return is in progress
    arc_return_start_time: float = 0.0   # When arc return began  
    arc_return_from: tuple = (0.0, 0.0)  # Starting position (alpha, beta)
    arc_return_to: tuple = (0.0, 0.0)    # Target position (alpha, beta)


class StrokeMapper:
    """
    Converts beat events to alpha/beta stroke commands.
    
    All stroke modes create circular/arc patterns in the alpha/beta plane.
    Alpha and beta range from -1 to 1, with (0,0) at center.
    """
    
    def __init__(self, config: Config, send_callback: Callable[[TCodeCommand], None] = None, get_volume: Callable[[], float] = None, audio_engine=None):
        self.config = config
        self.state = StrokeState()
        self.send_callback = send_callback  # Callback to send commands directly
        self.get_volume = get_volume if get_volume is not None else (lambda: 1.0)
        self.audio_engine = audio_engine
        
        # Motion intensity multiplier (0.25-2.0, default 1.0) — scales stroke output
        self.motion_intensity: float = 1.0
        
        # Band-based scaling tables:
        # Volume: subtle scaling (0.95-1.0 range) for slight tonal shaping only
        # Speed: low bands get faster strokes (kick drum = punchy, cymbals = gentle)
        self._band_volume_scale = {
            'sub_bass': 1.00,
            'low_mid':  0.98,
            'mid':      0.97,
            'high':     0.95,
        }
        self._band_speed_scale = {
            'sub_bass': 0.70,   # fastest (shortest duration)
            'low_mid':  0.85,
            'mid':      1.00,
            'high':     1.20,   # slowest (longest duration)
        }
        
        # Mode-specific state
        self.figure8_phase = 0.0
        self.random_arc_start = 0.0
        self.random_arc_end = np.pi
        self._return_timer: Optional[threading.Timer] = None
        # Spiral mode persistent phase
        self.spiral_beat_index = 0
        self.spiral_revolutions = 3  # Number of revolutions for full spiral (configurable)
        # Spiral return smoothing state
        self.spiral_reset_active = False
        self.spiral_reset_start_time = 0.0
        self.spiral_reset_from = (0.0, 0.0)
        
        # Beat factoring for fast tempos
        self.max_strokes_per_sec = 4.5  # Maximum strokes per second
        self.beat_factor = 1  # Skip every Nth beat
        
        # Creep volume fade state (lower volume when creep sustained for >2 expected beats)
        self._creep_sustained_start: float = 0.0  # When creep started being sustained
        self._creep_volume_factor: float = 1.0  # Current creep volume reduction (1.0 = no reduction)
        self._creep_was_active_last_frame: bool = False  # Track if creep was active last frame

    def _get_band_volume(self, event: BeatEvent) -> float:
        """Return volume with band-based reduction (subtractive, never below 85%)."""
        band = getattr(event, 'beat_band', 'sub_bass')
        base_vol = self.get_volume()
        # Band reduction: 0% (sub_bass) to 5% (high)
        band_reduction = (1.0 - self._band_volume_scale.get(band, 1.0)) * base_vol
        return max(base_vol * 0.85, base_vol - band_reduction)

    def _get_band_duration_scale(self, event: BeatEvent) -> float:
        """Return duration multiplier for the current primary beat band.
        Lower values = faster (shorter) strokes."""
        band = getattr(event, 'beat_band', 'sub_bass')
        return self._band_speed_scale.get(band, 1.0)

    def process_beat(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """
        Process a beat event and return a stroke command.
        
        Spectral flux-based behavior:
        - Low flux (<threshold): Only full strokes on downbeats
        - High flux (>=threshold): Full strokes on every beat
        
        Returns:
            TCodeCommand if a stroke should be sent, None otherwise
        """
        now = time.time()
        cfg = self.config.stroke
        beat_cfg = self.config.beat
        # Fade-out state for quiet suppression
        if not hasattr(self, '_fade_intensity'):
            self._fade_intensity = 1.0
        if not hasattr(self, '_last_quiet_time'):
            self._last_quiet_time = 0.0
        if not hasattr(self, '_consecutive_silent_count'):
            self._consecutive_silent_count = 0
            
        # Thresholds for true silence
        quiet_flux_thresh = cfg.flux_threshold * cfg.silence_flux_multiplier
        quiet_energy_thresh = beat_cfg.peak_floor * cfg.silence_energy_multiplier
        fade_duration = 2.0  # seconds to fade out
        silence_reset_threshold = beat_cfg.silence_reset_ms / 1000.0  # Convert ms to seconds
        consecutive_silent_required = 10  # Require 10 consecutive silent frames before fading
        
        is_truly_silent = (event.spectral_flux < quiet_flux_thresh and event.peak_energy < quiet_energy_thresh)
        if is_truly_silent:
            self._consecutive_silent_count += 1
            # Only start fade after 10 consecutive silent frames
            if self._consecutive_silent_count >= consecutive_silent_required:
                if self._fade_intensity > 0.0:
                    # Start fade-out
                    if self._last_quiet_time == 0.0:
                        self._last_quiet_time = now
                    elapsed = now - self._last_quiet_time
                    self._fade_intensity = max(0.0, 1.0 - (elapsed / fade_duration))
                    # If silence persists for more than 250ms, reset tempo/downbeat
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
            self.state.idle_time = now - self.state.last_beat_time
        
        # Determine what to do
        if event.is_beat:
            # Check minimum interval for beats only (jitter bypasses this)
            time_since_stroke = (now - self.state.last_stroke_time) * 1000
            if time_since_stroke < cfg.min_interval_ms:
                log_event("INFO", "StrokeMapper", "Skipping stroke: min_interval_ms not met", elapsed_ms=f"{time_since_stroke:.1f}", min_ms=cfg.min_interval_ms)
                return None
            
            # Check if flux is high or low
            is_high_flux = event.spectral_flux >= cfg.flux_threshold
            is_downbeat = getattr(event, 'is_downbeat', False)
            
            # Calculate flux factor for stroke scaling (0.5 to 1.5 range)
            # Low flux = smaller strokes, high flux = larger strokes
            flux_ratio = event.spectral_flux / max(cfg.flux_threshold, 0.001)
            # Clamp ratio to reasonable range and map to 0.5-1.5
            flux_ratio = np.clip(flux_ratio, 0.2, 3.0)
            base_factor = 0.5 + (flux_ratio / 3.0)  # 0.53 to 1.5
            # Apply scaling weight (0=no effect, 1=normal, 2=strong)
            scaling_weight = cfg.flux_scaling_weight
            self._flux_stroke_factor = 1.0 + (base_factor - 1.0) * scaling_weight
            
            # DOWNBEAT: Always generate full stroke on downbeat
            if is_downbeat:
                cmd = self._generate_downbeat_stroke(event)
                if cmd is None:
                    return None
                # Apply fade-out to intensity
                if hasattr(cmd, 'intensity'):
                    cmd.intensity *= self._fade_intensity
                if hasattr(cmd, 'volume'):
                    cmd.volume *= self._fade_intensity
                log_event(
                    "INFO",
                    "StrokeMapper",
                    "Downbeat command",
                    alpha=f"{cmd.alpha:.2f}",
                    beta=f"{cmd.beta:.2f}",
                    flux_factor=f"{self._flux_stroke_factor:.2f}",
                    flux=f"{event.spectral_flux:.4f}",
                    fade=f"{self._fade_intensity:.2f}"
                )
                return cmd if self._fade_intensity > 0.01 else None
            
            # REGULAR BEAT:
            # - Low flux: skip regular beats (only downbeats get strokes)
            # - High flux: do full strokes on all beats
            if not is_high_flux:
                # Low flux: skip this beat
                log_event("INFO", "StrokeMapper", "Skipping beat (low flux)", flux=f"{event.spectral_flux:.4f}", threshold=cfg.flux_threshold)
                return None
            
            # High flux: Generate full stroke on regular beat too
            cmd = self._generate_beat_stroke(event)
            if hasattr(cmd, 'intensity'):
                cmd.intensity *= self._fade_intensity
            if hasattr(cmd, 'volume'):
                cmd.volume *= self._fade_intensity
            log_event(
                "INFO",
                "StrokeMapper",
                "Beat stroke",
                flux_factor=f"{self._flux_stroke_factor:.2f}",
                flux=f"{event.spectral_flux:.4f}",
                fade=f"{self._fade_intensity:.2f}",
                alpha=f"{cmd.alpha:.2f}",
                beta=f"{cmd.beta:.2f}"
            )
            return cmd if self._fade_intensity > 0.01 else None
            
        elif self.state.idle_time > 0.5:
            # Only allow idle motion if not truly silent and fade intensity > 0
            if not is_truly_silent and self._fade_intensity > 0.01:
                cmd = self._generate_idle_motion(event)
                if hasattr(cmd, 'intensity'):
                    cmd.intensity *= self._fade_intensity
                if hasattr(cmd, 'volume'):
                    cmd.volume *= self._fade_intensity
                if cmd is not None:
                    log_event(
                        "INFO",
                        "StrokeMapper",
                        "Idle command",
                        alpha=f"{cmd.alpha:.2f}",
                        beta=f"{cmd.beta:.2f}",
                        jitter=self.config.jitter.enabled,
                        creep=self.config.creep.enabled,
                        fade=f"{self._fade_intensity:.2f}"
                    )
                else:
                    log_event(
                        "INFO",
                        "StrokeMapper",
                        "Idle suppressed (no command)",
                        jitter=self.config.jitter.enabled,
                        creep=self.config.creep.enabled,
                        fade=f"{self._fade_intensity:.2f}"
                    )
                return cmd
            else:
                # Suppress idle motion if truly silent
                log_event("INFO", "StrokeMapper", "Idle suppressed (truly silent)", fade=f"{self._fade_intensity:.2f}")
                return None
        
        return None
    
    def _generate_downbeat_stroke(self, event: BeatEvent) -> TCodeCommand:
        """
        Generate a full measure-length stroke on downbeat.
        
        On downbeats (beat 1 of measure), create an extended full loop that takes
        approximately one full measure (4 beats) to complete. This makes downbeats
        feel more pronounced and creates a clear measure structure.
        
        When tempo is LOCKED (consecutive downbeats match predicted tempo), hit harder
        with increased stroke amplitude and radius for more emphatic feel.
        """
        cfg = self.config.stroke
        now = time.time()
        
        # Cancel any pending arc thread (non-blocking to avoid lag)
        if hasattr(self, '_arc_thread') and self._arc_thread and self._arc_thread.is_alive():
            self._stop_arc = True
            # Don't join — arc thread checks _stop_arc and exits on its own
        
        # On downbeat, use extended duration (estimate ~4 beats for measure)
        # Use last beat interval * 4 for the measure length
        if self.state.last_beat_time == 0.0:
            measure_duration_ms = 2000  # Default 2 seconds
        else:
            beat_interval_ms = (now - self.state.last_beat_time) * 1000
            beat_interval_ms = max(cfg.min_interval_ms, min(1000, beat_interval_ms))
            measure_duration_ms = beat_interval_ms * 4  # Full measure
        
        # Band-based speed scaling for downbeats too
        band_speed = self._get_band_duration_scale(event)
        measure_duration_ms = int(measure_duration_ms * band_speed)
        
        # Calculate stroke parameters
        intensity = event.intensity
        
        # Apply flux factor to scale stroke size (0.5-1.5 range)
        flux_factor = getattr(self, '_flux_stroke_factor', 1.0)
        
        # TEMPO LOCK BOOST: When tempo is locked, hit HARDER with stronger amplitude
        # This creates more confident, emphatic downbeats when we're sure of the tempo
        tempo_locked = getattr(event, 'tempo_locked', False)
        lock_boost = 1.25 if tempo_locked else 1.0  # 25% stronger when locked
        
        # On downbeat, use full stroke amplitude scaled by flux, lock boost, and motion intensity
        stroke_len = cfg.stroke_max * flux_factor * lock_boost * self.motion_intensity
        stroke_len = max(cfg.stroke_min, min(cfg.stroke_max * 1.25, stroke_len))  # Allow up to 125% of max when locked
        
        freq_factor = self._freq_to_factor(event.frequency)
        depth = cfg.minimum_depth + (1.0 - cfg.minimum_depth) * (1.0 - cfg.freq_depth_factor * freq_factor)
        
        # Radius for the arc - scale by flux factor and lock boost for more dynamic range
        min_radius = 0.3
        radius = min_radius + (1.0 - min_radius) * flux_factor * lock_boost
        radius = max(min_radius, min(1.0, radius))  # Clamp to 1.0 max
        
        # Apply axis weights
        alpha_weight = self.config.alpha_weight
        beta_weight = self.config.beta_weight
        
        # Generate arc: Use _get_stroke_target for each point in the arc
        n_points = max(16, int(measure_duration_ms / 20))  # 1 point per 20ms
        arc_phases = np.linspace(0, 1, n_points, endpoint=False)
        alpha_arc = np.zeros(n_points)
        beta_arc = np.zeros(n_points)
        for i, phase in enumerate(arc_phases):
            # Temporarily set phase for this point
            prev_phase = self.state.phase
            self.state.phase = phase
            alpha, beta = self._get_stroke_target(stroke_len, depth, event)
            alpha_arc[i] = alpha
            beta_arc[i] = beta
            self.state.phase = prev_phase  # Restore phase
        # Calculate step durations
        base_step = measure_duration_ms // n_points
        remainder = measure_duration_ms % n_points
        step_durations = [base_step + 1 if i < remainder else base_step for i in range(n_points)]
        # Start arc thread — store band volume for downbeat arc
        self._stop_arc = False
        self._arc_band_volume = self._get_band_volume(event)
        self._arc_thread = threading.Thread(
            target=self._send_arc_synchronous,
            args=[alpha_arc, beta_arc, step_durations, n_points],
            daemon=True
        )
        self._arc_thread.start()
        # Update state
        self.state.last_stroke_time = now
        self.state.last_beat_time = now
        lock_str = "LOCKED+BOOST" if tempo_locked else "unlocked"
        log_event(
            "INFO",
            "StrokeMapper",
            "Downbeat arc start",
            mode=self.config.stroke.mode.name,
            points=n_points,
            duration_ms=measure_duration_ms,
            tempo_state=lock_str
        )
        # Don't return a command here - the arc thread will send all points
        # Returning None signals that the arc is being handled asynchronously
        return None
    
    def _generate_beat_stroke(self, event: BeatEvent) -> TCodeCommand:
        """
        Generate a full arc stroke for a detected beat.
        
        Matches Breadbeats approach:
        - Full 2π circle per beat (complete loop)
        - Many points (duration/10ms each)
        - Synchronous sending with sleep between points
        """
        cfg = self.config.stroke
        now = time.time()
        
        # Cancel any pending arc thread (non-blocking to avoid lag)
        if hasattr(self, '_arc_thread') and self._arc_thread and self._arc_thread.is_alive():
            self._stop_arc = True
            # Don't join — arc thread checks _stop_arc and exits on its own
        
        # Calculate beat interval for duration (doubled for slower arc)
        beat_interval_ms = (now - self.state.last_beat_time) * 1000 if self.state.last_beat_time > 0 else cfg.min_interval_ms
        beat_interval_ms = max(cfg.min_interval_ms, min(1000, beat_interval_ms))
        beat_interval_ms *= 2  # Double the arc duration
        
        # Band-based speed scaling: low bands = faster (shorter), high = slower (longer)
        band_speed = self._get_band_duration_scale(event)
        beat_interval_ms = int(beat_interval_ms * band_speed)
        
        # Calculate stroke parameters
        intensity = event.intensity
        
        # Apply flux factor to scale stroke size (0.5-1.5 range)
        flux_factor = getattr(self, '_flux_stroke_factor', 1.0)
        
        base_stroke_len = cfg.stroke_min + (cfg.stroke_max - cfg.stroke_min) * intensity * cfg.stroke_fullness
        stroke_len = base_stroke_len * flux_factor * self.motion_intensity
        stroke_len = max(cfg.stroke_min, min(cfg.stroke_max, stroke_len))
        
        freq_factor = self._freq_to_factor(event.frequency)
        depth = cfg.minimum_depth + (1.0 - cfg.minimum_depth) * (1.0 - cfg.freq_depth_factor * freq_factor)
        
        # Radius for the arc (based on intensity and flux) - matching Breadbeats
        min_radius = 0.2
        max_radius = 1.0
        base_radius = min_radius + (max_radius - min_radius) * intensity
        radius = base_radius * flux_factor
        radius = max(min_radius, min(1.0, radius))
        
        # Apply axis weights
        alpha_weight = self.config.alpha_weight
        beta_weight = self.config.beta_weight
        
        # Spiral mode: animate from previous crest to next, full spiral in N beats
        if self.config.stroke.mode == StrokeMode.SPIRAL:
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
                # Spiral radius logic as before, but theta sweeps from prev to next crest
                margin = 0.1
                b = (1.0 - margin) / (2 * np.pi * N)
                r = b * theta * stroke_len * depth * intensity
                a = r * np.cos(theta) * alpha_weight
                b_ = r * np.sin(theta) * beta_weight
                alpha_arc[i] = np.clip(a, -1.0, 1.0)
                beta_arc[i] = np.clip(b_, -1.0, 1.0)
            # Update persistent index
            self.spiral_beat_index = next_index % N
        else:
            # Default: full arc per beat
            n_points = max(8, int(beat_interval_ms / 10))
            arc_phases = np.linspace(0, 1, n_points, endpoint=False)
            alpha_arc = np.zeros(n_points)
            beta_arc = np.zeros(n_points)
            for i, phase in enumerate(arc_phases):
                prev_phase = self.state.phase
                self.state.phase = phase
                alpha, beta = self._get_stroke_target(stroke_len, depth, event)
                alpha_arc[i] = alpha
                beta_arc[i] = beta
                self.state.phase = prev_phase
        # Calculate step durations
        base_step = beat_interval_ms // n_points
        remainder = beat_interval_ms % n_points
        step_durations = [base_step + 1 if i < remainder else base_step for i in range(n_points)]
        # Start arc thread — store band volume factor for arc thread to use
        self._stop_arc = False
        self._arc_band_volume = self._get_band_volume(event)
        self._arc_thread = threading.Thread(
            target=self._send_arc_synchronous,
            args=[alpha_arc, beta_arc, step_durations, n_points],
            daemon=True
        )
        self._arc_thread.start()
        # Update state
        self.state.last_stroke_time = now
        self.state.last_beat_time = now
        # Return first point immediately
        first_alpha = float(alpha_arc[0])
        first_beta = float(beta_arc[0])
        self.state.alpha = first_alpha
        self.state.beta = first_beta
        band = getattr(event, 'beat_band', 'sub_bass')
        log_event(
            "INFO",
            "StrokeMapper",
            "Arc start",
            mode=self.config.stroke.mode.name,
            points=n_points,
            duration_ms=beat_interval_ms,
            band=band,
            motion=f"{self.motion_intensity:.2f}"
        )
        return TCodeCommand(first_alpha, first_beta, step_durations[0], self._arc_band_volume)
    
    def _send_arc_synchronous(self, alpha_arc: np.ndarray, beta_arc: np.ndarray, step_durations: list, n_points: int):
        """Send arc points synchronously with proper sleep timing (Breadbeats approach)"""
        for i in range(1, n_points):  # Skip first point (already sent)
            if self._stop_arc:
                log_event("INFO", "StrokeMapper", "Arc interrupted", point=i)
                return
            
            alpha = float(alpha_arc[i])
            beta = float(beta_arc[i])
            step_ms = step_durations[i]  # Each step has its own duration
            
            if self.send_callback:
                # Apply fade-out and band-based volume to arc strokes (subtractive, min 85%)
                band_vol = getattr(self, '_arc_band_volume', self.get_volume())
                # Fade reduction: subtractive, not multiplicative
                fade_reduction = (1.0 - self._fade_intensity) * band_vol
                volume = max(band_vol * 0.85, band_vol - fade_reduction)
                cmd = TCodeCommand(alpha, beta, step_ms, volume)
                self.send_callback(cmd)
                self.state.alpha = alpha
                self.state.beta = beta
            
            # Sleep for this step duration (like Breadbeats time.sleep)
            time.sleep(step_ms / 1000.0)
        
        log_event("INFO", "StrokeMapper", "Arc complete", points=n_points)
        # Initiate smooth arc return after arc completes
        # Use the new arc_return mechanism for all modes - creates a curved return path
        self.state.arc_return_active = True
        self.state.arc_return_start_time = time.time()
        self.state.arc_return_from = (self.state.alpha, self.state.beta)
        if self.config.stroke.mode == StrokeMode.SPIRAL:
            # Spiral returns to center (0, 0)
            self.state.arc_return_to = (0.0, 0.0)
        else:
            # Other modes return to bottom of circle (0, -0.5) for smooth beat entry
            self.state.arc_return_to = (0.0, -0.5)
    
    def _send_return_stroke(self, duration_ms: int, alpha: float, beta: float):
        """Send the return stroke to opposite position (called by timer)"""
        if self.send_callback:
            # Apply fade-out to return strokes (subtractive, min 85%)
            base_vol = self.get_volume()
            fade_reduction = (1.0 - self._fade_intensity) * base_vol
            volume = max(base_vol * 0.85, base_vol - fade_reduction)
            cmd = TCodeCommand(alpha, beta, duration_ms, volume)
            log_event("INFO", "StrokeMapper", "Return stroke", alpha=f"{alpha:.2f}", beta=f"{beta:.2f}", duration_ms=duration_ms, fade=f"{self._fade_intensity:.2f}")
            self.send_callback(cmd)
            self.state.alpha = alpha
            self.state.beta = beta
    
    def _get_stroke_target(self, stroke_len: float, depth: float, event: BeatEvent) -> Tuple[float, float]:
        """Calculate target position based on stroke mode"""
        mode = self.config.stroke.mode
        # Debug print to confirm mode switching and parameters
        log_event(
            "INFO",
            "StrokeMapper",
            "Compute stroke target",
            mode=mode.name,
            stroke_len=f"{stroke_len:.3f}",
            depth=f"{depth:.3f}",
            intensity=f"{event.intensity:.3f}"
        )
        # Get axis weights (used differently per mode)
        alpha_weight = self.config.alpha_weight  # 0-2 range
        beta_weight = self.config.beta_weight    # 0-2 range
        
        phase_advance = self.config.stroke.phase_advance
        if mode == StrokeMode.SIMPLE_CIRCLE:
            # Standard circle
            self.state.phase = (self.state.phase + phase_advance) % 1.0
            angle = self.state.phase * 2 * np.pi
            min_radius = 0.3
            radius = min_radius + (stroke_len * depth - min_radius) * event.intensity
            radius = max(min_radius, min(1.0, radius))
            alpha = np.sin(angle) * radius * alpha_weight
            beta = np.cos(angle) * radius * beta_weight

        elif mode == StrokeMode.SPIRAL:
            # Spiral: Use stroke_min, stroke_max, fullness, min_depth, freq_depth, intensity for strong slider effect
            self.state.phase = (self.state.phase + phase_advance) % 1.0
            revolutions = 2  # Number of spiral turns (adjustable)
            theta_max = revolutions * 2 * np.pi
            theta = (self.state.phase - 0.5) * 2 * theta_max  # theta in [-theta_max, +theta_max]
            min_radius = 0.3
            # Use same radius logic as circle, but modulate with |theta/theta_max| for spiral effect
            base_radius = min_radius + (stroke_len * depth - min_radius) * event.intensity
            base_radius = max(min_radius, min(1.0, base_radius))
            spiral_factor = abs(theta) / theta_max  # 0 at center, 1 at ends
            r = base_radius * spiral_factor
            alpha = r * np.cos(theta) * alpha_weight
            beta = r * np.sin(theta) * beta_weight
            # Clamp to [-1,1]
            alpha = np.clip(alpha, -1.0, 1.0)
            beta = np.clip(beta, -1.0, 1.0)

        elif mode == StrokeMode.TEARDROP:
            # Teardrop shape (piriform):
            # x = a * (sin t - 0.5 * sin(2t)), y = -a * cos t, t in [-pi, pi] for full sweep
            # Use 1/4 of phase_advance for smoother, slower teardrop motion
            teardrop_advance = phase_advance * 0.25
            self.state.phase = (self.state.phase + teardrop_advance) % 1.0
            t = (self.state.phase - 0.5) * 2 * np.pi  # t in [-pi, pi]
            # Use minimum radius like other modes to ensure motion even at low intensity
            min_radius = 0.2
            a = min_radius + (stroke_len * depth - min_radius) * event.intensity
            a = max(min_radius, min(1.0, a))
            x = a * (np.sin(t) - 0.5 * np.sin(2 * t))
            y = -a * np.cos(t)
            # Rotate so the teardrop points up
            angle = np.pi / 2
            alpha = x * np.cos(angle) - y * np.sin(angle)
            beta = x * np.sin(angle) + y * np.cos(angle)
            # Apply axis weights
            alpha *= alpha_weight
            beta *= beta_weight
            # Clamp to [-1,1]
            alpha = np.clip(alpha, -1.0, 1.0)
            beta = np.clip(beta, -1.0, 1.0)
            
        elif mode == StrokeMode.USER:
            # USER mode: axis weights control flux vs peak response
            # alpha_weight: 0 = flux-driven, 1 = balanced, 2 = peak-driven
            # beta_weight: same behavior for beta axis
            self.state.phase = (self.state.phase + phase_advance) % 1.0
            angle = self.state.phase * 2 * np.pi
            
            # Normalize flux and peak to 0-1 range for blending
            # Flux threshold from config gives us a reference point
            flux_ref = max(0.001, self.config.stroke.flux_threshold * 3)
            flux_norm = np.clip(event.spectral_flux / flux_ref, 0, 1)
            peak_norm = np.clip(event.peak_energy, 0, 1)
            
            # Calculate blend factors based on weights (0=flux, 1=balanced, 2=peak)
            alpha_blend = alpha_weight / 2.0  # 0-1 range
            beta_blend = beta_weight / 2.0   # 0-1 range
            
            # Mix flux and peak for each axis
            alpha_response = flux_norm * (1 - alpha_blend) + peak_norm * alpha_blend
            beta_response = flux_norm * (1 - beta_blend) + peak_norm * beta_blend
            
            # Calculate radii for smooth elliptical motion
            min_radius = 0.2
            alpha_radius = min_radius + (stroke_len * depth - min_radius) * alpha_response
            beta_radius = min_radius + (stroke_len * depth - min_radius) * beta_response
            
            # Generate smooth ellipse with different axis radii
            alpha = np.cos(angle) * alpha_radius
            beta = np.sin(angle) * beta_radius
            
            # Clamp to [-1, 1]
            alpha = np.clip(alpha, -1.0, 1.0)
            beta = np.clip(beta, -1.0, 1.0)
            
        else:
            # Fallback - simple continuous circle trace
            self.state.phase = (self.state.phase + phase_advance) % 1.0
            angle = self.state.phase * 2 * np.pi
            
            # Radius based on intensity
            min_radius = 0.2
            radius = min_radius + (stroke_len - min_radius) * event.intensity
            
            alpha = np.sin(angle) * radius
            beta = np.cos(angle) * radius
        
        return alpha, beta
    
    def _generate_idle_motion(self, event: Optional[BeatEvent]) -> Optional[TCodeCommand]:
        """Generate jitter/creep motion when idle - can work independently"""
        now = time.time()
        jitter_cfg = self.config.jitter
        creep_cfg = self.config.creep
        
        # Update throttle for update frequency (17ms = ~60 updates/sec)
        time_since_last = (now - self.state.last_stroke_time) * 1000
        if time_since_last < 17:
            return None
        
        # Check if either jitter OR creep is enabled (they work independently)
        jitter_active = jitter_cfg.enabled and jitter_cfg.amplitude > 0
        creep_active = creep_cfg.enabled and creep_cfg.speed > 0
        
        # Also continue if a return/reset animation is active
        return_active = self.state.arc_return_active or self.spiral_reset_active or self.state.creep_reset_active
        
        # Skip only if nothing is enabled
        if not jitter_active and not creep_active and not return_active:
            return None
        
        alpha, beta = self.state.alpha, self.state.beta
        
        # Handle smooth arc return after beat stroke (unified for all modes)
        if self.state.arc_return_active:
            reset_duration_ms = 400  # Smooth arc return over 400ms
            step_duration_ms = 200  # Minimum 200ms per command for smooth motion
            elapsed_ms = (now - self.state.arc_return_start_time) * 1000
            if elapsed_ms < reset_duration_ms:
                progress = elapsed_ms / reset_duration_ms
                # Ease-out curve for natural deceleration
                eased_progress = 1.0 - (1.0 - progress) ** 2
                
                # Get start and end positions
                from_alpha, from_beta = self.state.arc_return_from
                to_alpha, to_beta = self.state.arc_return_to
                
                # Calculate arc path using quadratic Bezier curve
                # Control point creates a curved arc (perpendicular to line between start and end)
                mid_alpha = (from_alpha + to_alpha) / 2
                mid_beta = (from_beta + to_beta) / 2
                # Perpendicular offset for arc curvature (creates outward bulge)
                perp_alpha = -(to_beta - from_beta) * 0.3
                perp_beta = (to_alpha - from_alpha) * 0.3
                control_alpha = mid_alpha + perp_alpha
                control_beta = mid_beta + perp_beta
                
                # Quadratic Bezier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
                t = eased_progress
                t2 = t * t
                mt = 1.0 - t
                mt2 = mt * mt
                
                alpha_target = mt2 * from_alpha + 2 * mt * t * control_alpha + t2 * to_alpha
                beta_target = mt2 * from_beta + 2 * mt * t * control_beta + t2 * to_beta
                
                # Clamp to valid range
                alpha_target = np.clip(alpha_target, -1.0, 1.0)
                beta_target = np.clip(beta_target, -1.0, 1.0)
                
                self.state.alpha = alpha_target
                self.state.beta = beta_target
                self.state.last_stroke_time = now
                fade = getattr(self, '_fade_intensity', 1.0)
                volume = self.get_volume() * fade
                return TCodeCommand(alpha_target, beta_target, step_duration_ms, volume)
            else:
                # Arc return complete
                self.state.arc_return_active = False
                to_alpha, to_beta = self.state.arc_return_to
                self.state.alpha = to_alpha
                self.state.beta = to_beta
                # Also reset creep angle to match new position
                self.state.creep_angle = np.arctan2(to_alpha, to_beta)
                log_event("INFO", "StrokeMapper", "Arc return complete", alpha=f"{to_alpha:.2f}", beta=f"{to_beta:.2f}")
        
        # Legacy: Handle smooth spiral return (kept for backward compatibility)
        elif self.spiral_reset_active:
            # Convert to new arc return
            self.state.arc_return_active = True
            self.state.arc_return_start_time = self.spiral_reset_start_time
            self.state.arc_return_from = self.spiral_reset_from
            self.state.arc_return_to = (0.0, 0.0)
            self.spiral_reset_active = False
        
        # Legacy: Handle creep reset (kept for backward compatibility)
        elif self.state.creep_reset_active:
            # Reset creep angle smoothly
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
                # Normalize angle
                while current_angle > np.pi:
                    current_angle -= 2 * np.pi
                while current_angle < -np.pi:
                    current_angle += 2 * np.pi
                self.state.creep_angle = current_angle * (1.0 - eased_progress)
            else:
                self.state.creep_angle = 0.0
                self.state.creep_reset_active = False
        
        # Creep volume lowering: if creep sustained for >2 expected beats, lower volume 3% over 600ms
        if creep_active:
            # Get expected beat duration from audio_engine (if available)
            expected_beat_ms = 500.0  # Default 500ms (~120 BPM)
            if self.audio_engine and hasattr(self.audio_engine, 'get_tempo_info'):
                tempo_info = self.audio_engine.get_tempo_info()
                if tempo_info and tempo_info.get('bpm', 0) > 0:
                    expected_beat_ms = 60000.0 / tempo_info['bpm']
            
            # Track when creep became sustained (after a beat stroke ended)
            if not self._creep_was_active_last_frame:
                # Just started creep - reset timer
                self._creep_sustained_start = now
                self._creep_volume_factor = 1.0
            else:
                # Creep continues - check if sustained for >2 expected beats
                sustained_ms = (now - self._creep_sustained_start) * 1000.0
                threshold_ms = expected_beat_ms * 2.0  # 2 expected beats
                if sustained_ms > threshold_ms:
                    # Start fading volume down by 3% over 600ms
                    fade_start_ms = threshold_ms
                    fade_duration_ms = 600.0
                    fade_progress = min(1.0, (sustained_ms - fade_start_ms) / fade_duration_ms)
                    self._creep_volume_factor = 1.0 - (0.03 * fade_progress)  # 3% reduction at full fade
            self._creep_was_active_last_frame = True
        else:
            # Creep not active - reset
            self._creep_was_active_last_frame = False
            self._creep_volume_factor = 1.0
        
        # Creep: slowly rotate around outer edge of circle (works independently of jitter)
        if creep_active:
            # Tempo-synced creep: speed=1.0 moves 1/4 circle per beat
            # Lower speeds scale proportionally (e.g., 0.25 = 1/16 circle per beat)
            # At 60 updates/sec (17ms throttle), calculate increment per update
            bpm = getattr(event, 'bpm', 0.0) if event else 0.0
            
            if bpm > 0:
                # Tempo detected: rotate around circle
                # Calculate: (π/2 radians per beat) / (updates per beat)
                beats_per_sec = bpm / 60.0
                updates_per_sec = 1000.0 / 17.0  # ~60 fps at 17ms throttle
                updates_per_beat = updates_per_sec / beats_per_sec
                angle_increment = (np.pi / 2.0) / updates_per_beat * creep_cfg.speed
                
                # Only increment creep angle if no return animation is active
                if not self.state.creep_reset_active and not self.state.arc_return_active:
                    self.state.creep_angle += angle_increment
                    if self.state.creep_angle >= 2 * np.pi:
                        self.state.creep_angle -= 2 * np.pi
                
                # Position on circle - pull inward by jitter amplitude so
                # micro-circles don't get clipped at the ±1.0 boundary
                jitter_r = jitter_cfg.amplitude if jitter_active else 0.0
                creep_radius = max(0.1, 0.98 - jitter_r)
                base_alpha = np.sin(self.state.creep_angle) * creep_radius
                base_beta = np.cos(self.state.creep_angle) * creep_radius
            else:
                # No tempo detected: slowly oscillate toward center
                # Use creep_angle as oscillation phase, 0.1 base radius
                if not self.state.creep_reset_active and not self.state.arc_return_active:
                    self.state.creep_angle += creep_cfg.speed * 0.02  # Slow oscillation
                    if self.state.creep_angle >= 2 * np.pi:
                        self.state.creep_angle -= 2 * np.pi
                
                # Oscillate between center (0.1) and partial radius (0.3)
                oscillation = 0.2 + 0.1 * np.sin(self.state.creep_angle)
                base_alpha = oscillation * np.sin(self.state.creep_angle * 0.5)
                base_beta = oscillation * np.cos(self.state.creep_angle * 0.5) - 0.2  # Bias downward
        else:
            # No creep - stay at current position
            base_alpha = alpha
            base_beta = beta
        
        # Jitter: sinusoidal micro-circles around the creep position (only if enabled)
        if jitter_active:
            # Advance jitter angle smoothly based on intensity (higher = faster circles)
            # Scale by update interval for consistent speed regardless of frame rate
            jitter_speed = jitter_cfg.intensity * 0.15  # Slower, smoother circles
            self.state.jitter_angle += jitter_speed
            if self.state.jitter_angle >= 2 * np.pi:
                self.state.jitter_angle -= 2 * np.pi
            
            # Amplitude controls the size of micro-circles
            jitter_r = jitter_cfg.amplitude
            
            # Add sinusoidal micro-circle to base position
            alpha_target = base_alpha + np.cos(self.state.jitter_angle) * jitter_r
            beta_target = base_beta + np.sin(self.state.jitter_angle) * jitter_r
        else:
            # No jitter - use base position directly
            alpha_target = base_alpha
            beta_target = base_beta
        
        # Clamp to valid range
        alpha_target = np.clip(alpha_target, -1.0, 1.0)
        beta_target = np.clip(beta_target, -1.0, 1.0)
        
        # Duration: minimum 200ms per command for smooth continuous motion
        duration_ms = 200  # Minimum duration per command prevents jerky motion
        
        # Update state and timing
        self.state.alpha = alpha_target
        self.state.beta = beta_target
        self.state.last_stroke_time = now
        
        # Apply fade intensity and creep volume factor (subtractive, never below 85%)
        base_vol = self.get_volume()
        fade = getattr(self, '_fade_intensity', 1.0)
        creep_vol = getattr(self, '_creep_volume_factor', 1.0)
        # Subtractive reductions
        fade_reduction = (1.0 - fade) * base_vol
        creep_reduction = (1.0 - creep_vol) * base_vol
        total_reduction = fade_reduction + creep_reduction
        # Hard clamp: never reduce by more than 15%
        volume = max(base_vol * 0.85, base_vol - min(total_reduction, base_vol * 0.15))
        
        return TCodeCommand(alpha_target, beta_target, duration_ms, volume)
    
    def _freq_to_factor(self, freq: float) -> float:
        """Convert frequency to a 0-1 factor using config's depth frequency range.
        Lower frequencies (bass) → 0 → deeper strokes
        Higher frequencies → 1 → shallower strokes
        """
        cfg = self.config.stroke
        low = cfg.depth_freq_low
        high = cfg.depth_freq_high
        
        if freq <= low:
            return 0.0
        elif freq >= high:
            return 1.0
        else:
            return (freq - low) / (high - low)
    
    def get_current_position(self) -> Tuple[float, float]:
        """Get current alpha/beta position for visualization"""
        return self.state.alpha, self.state.beta
    
    def reset(self):
        """Reset stroke mapper state"""
        self.state = StrokeState()
        self.figure8_phase = 0.0
        self.random_arc_start = 0.0
        self.random_arc_end = np.pi


# Test
if __name__ == "__main__":
    from config import Config
    
    config = Config()
    mapper = StrokeMapper(config)
    
    # Simulate some beats
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
