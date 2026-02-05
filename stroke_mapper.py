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


class StrokeMapper:
    """
    Converts beat events to alpha/beta stroke commands.
    
    All stroke modes create circular/arc patterns in the alpha/beta plane.
    Alpha and beta range from -1 to 1, with (0,0) at center.
    """
    
    def __init__(self, config: Config, send_callback: Callable[[TCodeCommand], None] = None, get_volume: Callable[[], float] = None):
        self.config = config
        self.state = StrokeState()
        self.send_callback = send_callback  # Callback to send commands directly
        self.get_volume = get_volume if get_volume is not None else (lambda: 1.0)
        
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
        # Thresholds for true silence
        quiet_flux_thresh = cfg.flux_threshold * 0.1  # Lowered even further
        quiet_energy_thresh = beat_cfg.peak_floor * 0.7  # Lowered even further
        fade_duration = 2.0  # seconds to fade out
        # If both flux and energy are very low, treat as truly silent
        is_truly_silent = (event.spectral_flux < quiet_flux_thresh and event.peak_energy < quiet_energy_thresh)
        if is_truly_silent:
            if self._fade_intensity > 0.0:
                # Start fade-out
                if self._last_quiet_time == 0.0:
                    self._last_quiet_time = now
                elapsed = now - self._last_quiet_time
                self._fade_intensity = max(0.0, 1.0 - (elapsed / fade_duration))
            else:
                self._fade_intensity = 0.0
        else:
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
                print(f"[StrokeMapper] Skipping stroke: min_interval_ms not met ({time_since_stroke:.1f} < {cfg.min_interval_ms})")
                return None
            
            # Check if flux is high or low
            is_high_flux = event.spectral_flux >= cfg.flux_threshold
            is_downbeat = getattr(event, 'is_downbeat', False)
            
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
                print(f"[StrokeMapper] ⬇ DOWNBEAT -> cmd a={cmd.alpha:.2f} b={cmd.beta:.2f} (full loop, flux={event.spectral_flux:.4f}, fade={self._fade_intensity:.2f})")
                return cmd if self._fade_intensity > 0.01 else None
            
            # REGULAR BEAT:
            # - Low flux: skip regular beats (only downbeats get strokes)
            # - High flux: do full strokes on all beats
            if not is_high_flux:
                # Low flux: skip this beat
                print(f"[StrokeMapper] Skipping beat (low flux={event.spectral_flux:.4f} < {cfg.flux_threshold})")
                return None
            
            # High flux: Generate full stroke on regular beat too
            cmd = self._generate_beat_stroke(event)
            if hasattr(cmd, 'intensity'):
                cmd.intensity *= self._fade_intensity
            if hasattr(cmd, 'volume'):
                cmd.volume *= self._fade_intensity
            print(f"[StrokeMapper] Beat (HIGH FLUX={event.spectral_flux:.4f}, fade={self._fade_intensity:.2f}) -> cmd a={cmd.alpha:.2f} b={cmd.beta:.2f}")
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
                    print(f"[StrokeMapper] Idle -> cmd a={cmd.alpha:.2f} b={cmd.beta:.2f} jitter={self.config.jitter.enabled} creep={self.config.creep.enabled} fade={self._fade_intensity:.2f}")
                else:
                    print(f"[StrokeMapper] Idle -> cmd=None jitter={self.config.jitter.enabled} creep={self.config.creep.enabled} fade={self._fade_intensity:.2f}")
                return cmd
            else:
                # Suppress idle motion if truly silent
                print(f"[StrokeMapper] Idle suppressed (truly silent, fade={self._fade_intensity:.2f})")
                return None
        
        return None
    
    def _generate_downbeat_stroke(self, event: BeatEvent) -> TCodeCommand:
        """
        Generate a full measure-length stroke on downbeat.
        
        On downbeats (beat 1 of measure), create an extended full loop that takes
        approximately one full measure (4 beats) to complete. This makes downbeats
        feel more pronounced and creates a clear measure structure.
        """
        cfg = self.config.stroke
        now = time.time()
        
        # Cancel any pending arc thread
        if hasattr(self, '_arc_thread') and self._arc_thread and self._arc_thread.is_alive():
            self._stop_arc = True
            self._arc_thread.join(timeout=0.1)
        
        # On downbeat, use extended duration (estimate ~4 beats for measure)
        # Use last beat interval * 4 for the measure length
        if self.state.last_beat_time == 0.0:
            measure_duration_ms = 2000  # Default 2 seconds
        else:
            beat_interval_ms = (now - self.state.last_beat_time) * 1000
            beat_interval_ms = max(cfg.min_interval_ms, min(1000, beat_interval_ms))
            measure_duration_ms = beat_interval_ms * 4  # Full measure
        
        # Calculate stroke parameters
        intensity = event.intensity
        
        # On downbeat, use full stroke amplitude
        stroke_len = cfg.stroke_max  # Always full on downbeat
        
        freq_factor = self._freq_to_factor(event.frequency)
        depth = cfg.minimum_depth + (1.0 - cfg.minimum_depth) * (1.0 - cfg.freq_depth_factor * freq_factor)
        
        # Radius for the arc - full radius on downbeat for emphasis
        radius = 1.0  # Always full circle on downbeat
        
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
        # Calculate step durations with proper remainder distribution
        base_step = measure_duration_ms // n_points
        remainder = measure_duration_ms % n_points
        step_durations = [base_step + 1 if i < remainder else base_step for i in range(n_points)]
        # Start arc thread
        self._stop_arc = False
        self._arc_thread = threading.Thread(
            target=self._send_arc_synchronous,
            args=[alpha_arc, beta_arc, step_durations, n_points],
            daemon=True
        )
        self._arc_thread.start()
        # Update state
        self.state.last_stroke_time = now
        self.state.last_beat_time = now
        print(f"[StrokeMapper] ⬇ DOWNBEAT ARC start (mode={self.config.stroke.mode.name}) ({n_points} pts, {measure_duration_ms}ms total, full measure)")
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
        
        # Cancel any pending arc thread
        if hasattr(self, '_arc_thread') and self._arc_thread and self._arc_thread.is_alive():
            self._stop_arc = True
            self._arc_thread.join(timeout=0.1)
        
        # Calculate beat interval for duration (doubled for slower arc)
        beat_interval_ms = (now - self.state.last_beat_time) * 1000 if self.state.last_beat_time > 0 else cfg.min_interval_ms
        beat_interval_ms = max(cfg.min_interval_ms, min(1000, beat_interval_ms))
        beat_interval_ms *= 2  # Double the arc duration
        
        # Calculate stroke parameters
        intensity = event.intensity
        
        stroke_len = cfg.stroke_min + (cfg.stroke_max - cfg.stroke_min) * intensity * cfg.stroke_fullness
        stroke_len = max(cfg.stroke_min, min(cfg.stroke_max, stroke_len))
        
        freq_factor = self._freq_to_factor(event.frequency)
        depth = cfg.minimum_depth + (1.0 - cfg.minimum_depth) * (1.0 - cfg.freq_depth_factor * freq_factor)
        
        # Radius for the arc (based on intensity) - matching Breadbeats
        min_radius = 0.2
        max_radius = 1.0
        radius = min_radius + (max_radius - min_radius) * intensity
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
        # Calculate step durations with proper remainder distribution
        base_step = beat_interval_ms // n_points
        remainder = beat_interval_ms % n_points
        step_durations = [base_step + 1 if i < remainder else base_step for i in range(n_points)]
        # Start arc thread
        self._stop_arc = False
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
        print(f"[StrokeMapper] ARC start (mode={self.config.stroke.mode.name}) ({n_points} pts, {beat_interval_ms}ms total)")
        return TCodeCommand(first_alpha, first_beta, step_durations[0], self.get_volume())
    
    def _send_arc_synchronous(self, alpha_arc: np.ndarray, beta_arc: np.ndarray, step_durations: list, n_points: int):
        """Send arc points synchronously with proper sleep timing (Breadbeats approach)"""
        for i in range(1, n_points):  # Skip first point (already sent)
            if self._stop_arc:
                print(f"[StrokeMapper] ARC interrupted at point {i}")
                return
            
            alpha = float(alpha_arc[i])
            beta = float(beta_arc[i])
            step_ms = step_durations[i]  # Each step has its own duration
            
            if self.send_callback:
                cmd = TCodeCommand(alpha, beta, step_ms, self.get_volume())
                self.send_callback(cmd)
                self.state.alpha = alpha
                self.state.beta = beta
            
            # Sleep for this step duration (like Breadbeats time.sleep)
            time.sleep(step_ms / 1000.0)
        
        print(f"[StrokeMapper] ARC complete ({n_points} points)")
        # Initiate smooth return after arc completes
        if self.config.stroke.mode == StrokeMode.SPIRAL:
            # Spiral-specific return: smoothly interpolate from last crest to center
            self.spiral_reset_active = True
            self.spiral_reset_start_time = time.time()
            self.spiral_reset_from = (self.state.alpha, self.state.beta)
        else:
            # Default: smooth creep reset to (0, -1)
            self.state.creep_reset_active = True
            self.state.creep_reset_start_time = time.time()
    
    def _send_return_stroke(self, duration_ms: int, alpha: float, beta: float):
        """Send the return stroke to opposite position (called by timer)"""
        if self.send_callback:
            cmd = TCodeCommand(alpha, beta, duration_ms, self.get_volume())
            print(f"[StrokeMapper] RETURN stroke a={alpha:.2f} b={beta:.2f} dur={duration_ms}ms")
            self.send_callback(cmd)
            self.state.alpha = alpha
            self.state.beta = beta
    
    def _get_stroke_target(self, stroke_len: float, depth: float, event: BeatEvent) -> Tuple[float, float]:
        """Calculate target position based on stroke mode"""
        mode = self.config.stroke.mode
        # Debug print to confirm mode switching and parameters
        print(f"[StrokeMapper] _get_stroke_target: mode={mode.name} stroke_len={stroke_len:.3f} depth={depth:.3f} intensity={event.intensity:.3f}")
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
            self.state.phase = (self.state.phase + phase_advance) % 1.0
            t = (self.state.phase - 0.5) * 2 * np.pi  # t in [-pi, pi]
            a = stroke_len * depth * event.intensity
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
            # USER mode: ellipse always fits within unit circle
            self.state.phase = (self.state.phase + phase_advance) % 1.0
            angle = self.state.phase * 2 * np.pi
            freq_factor = self._freq_to_factor(event.frequency)
            aspect = 0.5 + freq_factor  # 0.5 to 1.5, as in beta
            # Compute unnormalized radii
            a = stroke_len * depth * alpha_weight
            b = stroke_len * depth * aspect * beta_weight
            # Normalize so ellipse fits in unit circle
            norm = max(abs(a), abs(b), 1e-6)
            if norm > 1.0:
                a /= norm
                b /= norm
            alpha = np.cos(angle) * a
            beta = np.sin(angle) * b
            
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
        """Generate jitter motion when idle - quick random movements to nearby targets"""
        now = time.time()
        jitter_cfg = self.config.jitter
        creep_cfg = self.config.creep
        
        # Update throttle for jitter frequency (17ms = ~60 updates/sec)
        time_since_last = (now - self.state.last_stroke_time) * 1000
        if time_since_last < 17:
            return None
        
        # Skip if jitter is disabled
        if not jitter_cfg.enabled or jitter_cfg.amplitude <= 0:
            return None
        
        alpha, beta = self.state.alpha, self.state.beta
        
        # Handle smooth spiral return after spiral arc
        if self.spiral_reset_active:
            reset_duration_ms = 500  # Smooth return over 500ms
            elapsed_ms = (now - self.spiral_reset_start_time) * 1000
            if elapsed_ms < reset_duration_ms:
                progress = elapsed_ms / reset_duration_ms
                eased_progress = 1.0 - (1.0 - progress) ** 2
                # Interpolate from last spiral crest to center (0,0)
                from_alpha, from_beta = self.spiral_reset_from
                to_alpha, to_beta = 0.0, 0.0
                alpha_target = from_alpha * (1.0 - eased_progress) + to_alpha * eased_progress
                beta_target = from_beta * (1.0 - eased_progress) + to_beta * eased_progress
                self.state.alpha = alpha_target
                self.state.beta = beta_target
                self.state.last_stroke_time = now
                return TCodeCommand(alpha_target, beta_target, 17, self.get_volume())
            else:
                self.spiral_reset_active = False
                self.state.alpha = 0.0
                self.state.beta = 0.0
                print(f"[StrokeMapper] Spiral reset complete, now at (0, 0)")
        # Handle smooth creep reset after beat stroke (non-spiral)
        elif self.state.creep_reset_active:
            reset_duration_ms = 500  # Smooth return over 500ms
            elapsed_ms = (now - self.state.creep_reset_start_time) * 1000
            # Defensive: ensure creep_angle is a valid float
            try:
                current_angle = float(self.state.creep_angle)
            except Exception as e:
                print(f"[StrokeMapper] Warning: creep_angle invalid ({self.state.creep_angle}), resetting to 0.0. Error: {e}")
                current_angle = 0.0
                self.state.creep_angle = 0.0
            if elapsed_ms < reset_duration_ms:
                progress = elapsed_ms / reset_duration_ms
                eased_progress = 1.0 - (1.0 - progress) ** 2
                # Defensive: clamp current_angle to [-2π, 2π]
                if not np.isfinite(current_angle):
                    current_angle = 0.0
                elif current_angle > np.pi:
                    current_angle = current_angle - 2 * np.pi
                elif current_angle < -np.pi:
                    current_angle = current_angle + 2 * np.pi
                target_angle = 0.0
                self.state.creep_angle = current_angle * (1.0 - eased_progress) + target_angle * eased_progress
            else:
                self.state.creep_angle = 0.0
                self.state.creep_reset_active = False
                print(f"[StrokeMapper] Creep reset complete, now at (0, -1)")
        
        # Creep: slowly rotate around outer edge of circle
        if creep_cfg.enabled and creep_cfg.speed > 0:
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
                
                # Only increment creep angle if reset is not active
                if not self.state.creep_reset_active:
                    self.state.creep_angle += angle_increment
                    if self.state.creep_angle >= 2 * np.pi:
                        self.state.creep_angle -= 2 * np.pi
                
                # Position on outer edge of circle (radius close to 1.0 for maximum extent)
                creep_radius = 0.98
                base_alpha = np.sin(self.state.creep_angle) * creep_radius
                base_beta = np.cos(self.state.creep_angle) * creep_radius
            else:
                # No tempo detected: slowly oscillate toward center
                # Use creep_angle as oscillation phase, 0.1 base radius
                if not self.state.creep_reset_active:
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
        
        # Jitter: small random movement around the creep position
        # Amplitude controls the range of movement (circle size)
        jitter_range = jitter_cfg.amplitude
        
        # Generate random target nearby creep position
        alpha_target = base_alpha + np.random.uniform(-jitter_range, jitter_range)
        beta_target = base_beta + np.random.uniform(-jitter_range, jitter_range)
        
        # Clamp to valid range
        alpha_target = np.clip(alpha_target, -1.0, 1.0)
        beta_target = np.clip(beta_target, -1.0, 1.0)
        
        # Intensity controls jitter speed (how fast to move - inversely related to duration)
        # High intensity = short duration = fast vibration
        # Low intensity = long duration = slow vibration
        # intensity: 0-3, map to duration: 200ms to 20ms (faster = higher intensity)
        base_duration = 200  # ms at intensity=0
        if jitter_cfg.intensity > 0:
            duration_ms = max(20, int(base_duration / (1.0 + jitter_cfg.intensity * 5)))
        else:
            duration_ms = base_duration
        
        # Update state and timing
        self.state.alpha = alpha_target
        self.state.beta = beta_target
        self.state.last_stroke_time = now
        
        return TCodeCommand(alpha_target, beta_target, duration_ms, self.get_volume())
    
    def _freq_to_factor(self, freq: float) -> float:
        """Convert frequency to a 0-1 factor (bass=0, treble=1)"""
        # Map roughly: 20Hz-200Hz = bass, 200Hz-2000Hz = mid, 2000Hz+ = treble
        if freq < 20:
            return 0.0
        elif freq < 200:
            return (freq - 20) / 180 * 0.33
        elif freq < 2000:
            return 0.33 + (freq - 200) / 1800 * 0.34
        else:
            return min(1.0, 0.67 + (freq - 2000) / 8000 * 0.33)
    
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
            print(f"Beat {i}: {cmd.to_tcode().strip()}")
        
        time.sleep(0.2)
