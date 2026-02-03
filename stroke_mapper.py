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
        
    def process_beat(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """
        Process a beat event and return a stroke command.
        
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
        
        # Check minimum interval
        time_since_stroke = (now - self.state.last_stroke_time) * 1000
        if time_since_stroke < cfg.min_interval_ms:
            return None
        
        # Determine what to do
        if event.is_beat:
            cmd = self._generate_beat_stroke(event)
            print(f"[StrokeMapper] Beat -> cmd a={cmd.alpha:.2f} b={cmd.beta:.2f} phase={self.state.phase:.2f}")
            return cmd
        elif self.state.idle_time > 0.5:  # 500ms of silence for idle motion
            cmd = self._generate_idle_motion(event)
            if cmd:
                print(f"[StrokeMapper] Idle -> cmd a={cmd.alpha:.2f} b={cmd.beta:.2f} jitter={self.config.jitter.enabled} creep={self.config.creep.enabled}")
            return cmd
        
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
    
    def _generate_idle_motion(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """Generate jitter or creep motion when idle"""
        now = time.time()
        jitter_cfg = self.config.jitter
        creep_cfg = self.config.creep
        
        # Throttle idle updates to every 500ms for smooth slow motion
        time_since_stroke = (now - self.state.last_stroke_time) * 1000
        if time_since_stroke < 500:  # Only send idle motion every 500ms
            return None
        
        alpha, beta = self.state.alpha, self.state.beta
        duration_ms = 500  # Long duration for smooth interpolation
        
        has_motion = False
        
        # Apply jitter (micro-circles) - max 10% of range (0.1 on -1 to 1 scale)
        if jitter_cfg.enabled and jitter_cfg.amplitude > 0:
            self.state.jitter_angle += jitter_cfg.intensity * 0.1  # Slower rotation
            # Max amplitude is 0.1 (10% of full range), scaled by user setting
            jitter_r = jitter_cfg.amplitude * 0.1  # 10% max
            alpha += np.cos(self.state.jitter_angle) * jitter_r
            beta += np.sin(self.state.jitter_angle) * jitter_r
            has_motion = True
        
        # Apply creep (slow drift) - also limited to 10% max
        if creep_cfg.enabled and creep_cfg.speed > 0:
            self.state.creep_angle += creep_cfg.speed * 0.05  # Very slow rotation
            creep_r = 0.1 * creep_cfg.speed  # Scale by speed setting, max 10%
            # Move toward creep target smoothly
            target_alpha = np.cos(self.state.creep_angle) * creep_r
            target_beta = np.sin(self.state.creep_angle) * creep_r
            # Blend slowly toward target
            alpha = alpha * 0.9 + target_alpha * 0.1
            beta = beta * 0.9 + target_beta * 0.1
            has_motion = True
        
        if not has_motion:
            return None
        
        # Clamp to valid range
        alpha = max(-1.0, min(1.0, alpha))
        beta = max(-1.0, min(1.0, beta))
        
        # Update state and timing
        self.state.alpha = alpha
        self.state.beta = beta
        self.state.last_stroke_time = now  # Throttle future idle updates
        
        return TCodeCommand(alpha, beta, duration_ms)
    
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
