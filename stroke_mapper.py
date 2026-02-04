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
    
    def __init__(self, config: Config, send_callback: Callable[[TCodeCommand], None] = None):
        self.config = config
        self.state = StrokeState()
        self.send_callback = send_callback  # Callback to send commands directly
        
        # Mode-specific state
        self.figure8_phase = 0.0
        self.random_arc_start = 0.0
        self.random_arc_end = np.pi
        self._return_timer: Optional[threading.Timer] = None
        
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
                return None
            
            # Check if flux is high or low
            is_high_flux = event.spectral_flux >= cfg.flux_threshold
            is_downbeat = getattr(event, 'is_downbeat', False)
            
            # DOWNBEAT: Always generate full stroke on downbeat
            if is_downbeat:
                cmd = self._generate_downbeat_stroke(event)
                if cmd is None:
                    # Arc is handled asynchronously, no immediate command
                    return None
                print(f"[StrokeMapper] ⬇ DOWNBEAT -> cmd a={cmd.alpha:.2f} b={cmd.beta:.2f} (full loop, flux={event.spectral_flux:.4f})")
                return cmd
            
            # REGULAR BEAT:
            # - Low flux: skip regular beats (only downbeats get strokes)
            # - High flux: do full strokes on all beats
            if not is_high_flux:
                # Low flux: skip this beat
                print(f"[StrokeMapper] Skipping beat (low flux={event.spectral_flux:.4f} < {cfg.flux_threshold})")
                return None
            
            # High flux: Generate full stroke on regular beat too
            cmd = self._generate_beat_stroke(event)
            print(f"[StrokeMapper] Beat (HIGH FLUX={event.spectral_flux:.4f}) -> cmd a={cmd.alpha:.2f} b={cmd.beta:.2f}")
            return cmd
            
        elif self.state.idle_time > 0.5:  # 500ms of silence for idle motion
            cmd = self._generate_idle_motion(event)
            if cmd:
                print(f"[StrokeMapper] Idle -> cmd a={cmd.alpha:.2f} b={cmd.beta:.2f} jitter={self.config.jitter.enabled} creep={self.config.creep.enabled}")
            return cmd
        
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
        
        # Generate arc: FULL circle (0 to 2π) 
        # More points for smoother motion over longer duration
        n_points = max(16, int(measure_duration_ms / 20))  # 1 point per 20ms
        arc_theta = np.linspace(0, 2 * np.pi, n_points)
        
        # Alpha = cos, Beta = sin
        alpha_arc = radius * alpha_weight * np.cos(arc_theta)
        beta_arc = radius * beta_weight * np.sin(arc_theta)
        
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
        
        print(f"[StrokeMapper] ⬇ DOWNBEAT ARC start r={radius:.2f} ({n_points} pts, {measure_duration_ms}ms total, full measure)")
        
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
        
        # Calculate beat interval for duration
        beat_interval_ms = (now - self.state.last_beat_time) * 1000 if self.state.last_beat_time > 0 else cfg.min_interval_ms
        beat_interval_ms = max(cfg.min_interval_ms, min(1000, beat_interval_ms))
        
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
        
        # Generate arc: FULL circle (0 to 2π) like Breadbeats
        # n_points = duration/10ms (Breadbeats standard)
        n_points = max(8, int(beat_interval_ms / 10))
        arc_theta = np.linspace(0, 2 * np.pi, n_points)
        
        # Alpha = cos, Beta = sin (Breadbeats convention for arc)
        alpha_arc = radius * alpha_weight * np.cos(arc_theta)
        beta_arc = radius * beta_weight * np.sin(arc_theta)
        
        # Calculate step durations with proper remainder distribution (Breadbeats approach)
        # This ensures all ms are used without losing precision
        base_step = beat_interval_ms // n_points
        remainder = beat_interval_ms % n_points
        step_durations = [base_step + 1 if i < remainder else base_step for i in range(n_points)]
        
        # Start arc thread to send points synchronously with sleep (like Breadbeats)
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
        
        print(f"[StrokeMapper] ARC start r={radius:.2f} ({n_points} pts, {beat_interval_ms}ms total, full 2π)")
        return TCodeCommand(first_alpha, first_beta, step_durations[0])
    
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
                cmd = TCodeCommand(alpha, beta, step_ms)
                self.send_callback(cmd)
                self.state.alpha = alpha
                self.state.beta = beta
            
            # Sleep for this step duration (like Breadbeats time.sleep)
            time.sleep(step_ms / 1000.0)
        
        print(f"[StrokeMapper] ARC complete ({n_points} points)")
        
        # Initiate smooth creep reset to (0, -1) after arc completes
        self.state.creep_reset_active = True
        self.state.creep_reset_start_time = time.time()
    
    def _send_return_stroke(self, duration_ms: int, alpha: float, beta: float):
        """Send the return stroke to opposite position (called by timer)"""
        if self.send_callback:
            cmd = TCodeCommand(alpha, beta, duration_ms)
            print(f"[StrokeMapper] RETURN stroke a={alpha:.2f} b={beta:.2f} dur={duration_ms}ms")
            self.send_callback(cmd)
            self.state.alpha = alpha
            self.state.beta = beta
    
    def _get_stroke_target(self, stroke_len: float, depth: float, event: BeatEvent) -> Tuple[float, float]:
        """Calculate target position based on stroke mode"""
        mode = self.config.stroke.mode
        
        if mode == StrokeMode.SIMPLE_CIRCLE:
            # Trace around the circle - each beat advances position on the edge
            # Larger movements (1/4 circle) for more noticeable motion
            self.state.phase = (self.state.phase + 0.25) % 1.0  # 1/4 circle per beat = 4 beats full circle
            angle = self.state.phase * 2 * np.pi
            
            # Radius based on intensity (like Breadbeats norm_energy * impact)
            # At minimum, use some radius so there's always motion
            min_radius = 0.3
            radius = min_radius + (stroke_len * depth - min_radius) * event.intensity
            radius = max(min_radius, min(1.0, radius))
            
            # Breadbeats uses sin for alpha (horizontal), cos for beta (vertical)
            alpha = np.sin(angle) * radius
            beta = np.cos(angle) * radius
            
        elif mode == StrokeMode.FIGURE_EIGHT:
            # Figure-8 (lemniscate) pattern - smooth continuous trace
            # Based on Breadbeats oscillating mode with varied positions
            self.figure8_phase = (self.figure8_phase + 0.125) % 1.0
            t = self.figure8_phase * 2 * np.pi
            
            # Radius based on intensity
            min_radius = 0.2
            scale = min_radius + (stroke_len * depth - min_radius) * event.intensity
            
            # Figure-8 parametric equations
            alpha = np.sin(t) * scale
            beta = np.sin(2 * t) * scale * 0.5
            
        elif mode == StrokeMode.RANDOM_ARC:
            # Random arc segments - like Breadbeats "varied" mode
            # Cycles through different positions based on time
            if random.random() < 0.3:  # 30% chance to pick new arc
                self.random_arc_start = random.uniform(0, 2 * np.pi)
                arc_length = random.uniform(np.pi/4, np.pi)
                self.random_arc_end = self.random_arc_start + arc_length
            
            # Move along current arc
            self.state.phase = (self.state.phase + 0.125) % 1.0
            t = self.random_arc_start + self.state.phase * (self.random_arc_end - self.random_arc_start)
            
            # Radius based on intensity
            min_radius = 0.2
            radius = min_radius + (stroke_len * depth - min_radius) * event.intensity
            
            alpha = np.sin(t) * radius  # Match Breadbeats: sin for alpha
            beta = np.cos(t) * radius   # cos for beta
            
        elif mode == StrokeMode.USER:
            # User-controlled mode - like Breadbeats "adaptive" with frequency response
            # Shape and radius react to frequency and intensity
            self.state.phase = (self.state.phase + 0.125) % 1.0
            freq_factor = self._freq_to_factor(event.frequency)
            angle = self.state.phase * 2 * np.pi
            
            # Radius based on intensity
            min_radius = 0.2
            radius = min_radius + (stroke_len * depth - min_radius) * event.intensity
            
            # Ellipse with frequency-controlled aspect ratio (like Breadbeats alpha/beta weights)
            # Low frequency = more vertical, high frequency = more horizontal
            aspect = 0.5 + freq_factor  # 0.5 to 1.5
            
            alpha = np.sin(angle) * radius
            beta = np.cos(angle) * radius * aspect
            
        else:
            # Fallback - simple continuous circle trace
            self.state.phase = (self.state.phase + 0.125) % 1.0
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
        
        # Handle smooth creep reset after beat stroke
        if self.state.creep_reset_active:
            reset_duration_ms = 500  # Smooth return over 500ms
            elapsed_ms = (now - self.state.creep_reset_start_time) * 1000
            
            if elapsed_ms < reset_duration_ms:
                # Smoothly interpolate creep_angle from current value back to 0
                progress = elapsed_ms / reset_duration_ms
                # Ease-out: decelerate the motion smoothly
                eased_progress = 1.0 - (1.0 - progress) ** 2
                
                current_angle = self.state.creep_angle
                # Normalize to -π to π for shortest path
                if current_angle > np.pi:
                    current_angle = current_angle - 2 * np.pi
                
                target_angle = 0.0
                self.state.creep_angle = current_angle * (1.0 - eased_progress) + target_angle * eased_progress
            else:
                # Reset complete
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
        
        return TCodeCommand(alpha_target, beta_target, duration_ms)
    
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
