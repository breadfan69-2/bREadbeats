"""
bREadbeats - Stroke Mapper
Converts beat events into alpha/beta stroke patterns.
All modes use circular coordinates around (0,0).
"""

import numpy as np
import time
import random
from typing import Optional, Tuple
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
    
    def __init__(self, config: Config):
        self.config = config
        self.state = StrokeState()
        
        # Mode-specific state
        self.figure8_phase = 0.0
        self.random_arc_start = 0.0
        self.random_arc_end = np.pi
        
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
            print(f"[StrokeMapper] Beat -> cmd a={cmd.alpha:.2f} b={cmd.beta:.2f}")
            return cmd
        elif self.state.idle_time > 0.5:  # 500ms of silence for idle motion
            return self._generate_idle_motion(event)
        
        return None
    
    def _generate_beat_stroke(self, event: BeatEvent) -> TCodeCommand:
        """Generate a stroke for a detected beat"""
        cfg = self.config.stroke
        now = time.time()
        
        # Calculate stroke parameters based on intensity and frequency
        intensity = event.intensity
        freq_factor = self._freq_to_factor(event.frequency)
        
        # Stroke length based on intensity and fullness
        stroke_len = cfg.stroke_min + (cfg.stroke_max - cfg.stroke_min) * intensity * cfg.stroke_fullness
        stroke_len = max(cfg.stroke_min, min(cfg.stroke_max, stroke_len))
        
        # Apply frequency depth factor
        depth = cfg.minimum_depth + (1.0 - cfg.minimum_depth) * (1.0 - cfg.freq_depth_factor * freq_factor)
        
        # Get target position based on mode
        alpha, beta = self._get_stroke_target(stroke_len, depth, event)
        
        # Apply axis weights
        alpha *= self.config.alpha_weight
        beta *= self.config.beta_weight
        
        # Calculate duration - one full up/down per beat
        # Use intensity to vary speed slightly
        duration_ms = max(50, int(cfg.min_interval_ms * (0.5 + 0.5 * (1.0 - intensity))))
        
        # Update state
        self.state.target_alpha = alpha
        self.state.target_beta = beta
        self.state.last_stroke_time = now
        self.state.phase = (self.state.phase + 0.5) % 1.0  # Half cycle per beat
        
        return TCodeCommand(alpha, beta, duration_ms)
    
    def _get_stroke_target(self, stroke_len: float, depth: float, event: BeatEvent) -> Tuple[float, float]:
        """Calculate target position based on stroke mode"""
        mode = self.config.stroke.mode
        phase = self.state.phase
        
        if mode == StrokeMode.SIMPLE_CIRCLE:
            # Simple circular motion
            angle = phase * 2 * np.pi
            alpha = np.cos(angle) * stroke_len * depth
            beta = np.sin(angle) * stroke_len * depth
            
        elif mode == StrokeMode.FIGURE_EIGHT:
            # Figure-8 (lemniscate) pattern
            self.figure8_phase = (self.figure8_phase + 0.25) % 1.0
            t = self.figure8_phase * 2 * np.pi
            scale = stroke_len * depth
            alpha = np.sin(t) * scale
            beta = np.sin(2 * t) * scale * 0.5
            
        elif mode == StrokeMode.RANDOM_ARC:
            # Random arc segments
            if random.random() < 0.3:  # 30% chance to pick new arc
                self.random_arc_start = random.uniform(0, 2 * np.pi)
                arc_length = random.uniform(np.pi/4, np.pi)
                self.random_arc_end = self.random_arc_start + arc_length
            
            # Move along current arc
            t = self.random_arc_start + phase * (self.random_arc_end - self.random_arc_start)
            alpha = np.cos(t) * stroke_len * depth
            beta = np.sin(t) * stroke_len * depth
            
        elif mode == StrokeMode.USER:
            # User-controlled mode - shape reacts to frequency and intensity
            # More freq = more elongated, more intensity = larger
            freq_factor = self._freq_to_factor(event.frequency)
            angle = phase * 2 * np.pi
            
            # Ellipse with frequency-controlled aspect ratio
            aspect = 0.5 + freq_factor  # 0.5 to 1.5
            alpha = np.cos(angle) * stroke_len * depth
            beta = np.sin(angle) * stroke_len * depth * aspect
            
        else:
            # Fallback to circle
            angle = phase * 2 * np.pi
            alpha = np.cos(angle) * stroke_len
            beta = np.sin(angle) * stroke_len
        
        return alpha, beta
    
    def _generate_idle_motion(self, event: BeatEvent) -> Optional[TCodeCommand]:
        """Generate jitter or creep motion when idle"""
        jitter_cfg = self.config.jitter
        creep_cfg = self.config.creep
        
        alpha, beta = self.state.alpha, self.state.beta
        duration_ms = 100  # Smooth idle motion
        
        has_motion = False
        
        # Apply jitter (micro-circles)
        if jitter_cfg.enabled and jitter_cfg.amplitude > 0:
            self.state.jitter_angle += jitter_cfg.intensity * 0.5
            jitter_r = jitter_cfg.amplitude * 0.1  # Scale down for micro movement
            alpha += np.cos(self.state.jitter_angle) * jitter_r
            beta += np.sin(self.state.jitter_angle) * jitter_r
            has_motion = True
        
        # Apply creep (slow drift)
        if creep_cfg.enabled and creep_cfg.speed > 0:
            self.state.creep_angle += creep_cfg.speed * 0.02
            creep_r = 0.3  # Max creep radius
            # Slowly spiral
            alpha = alpha * 0.99 + np.cos(self.state.creep_angle) * creep_r * 0.01
            beta = beta * 0.99 + np.sin(self.state.creep_angle) * creep_r * 0.01
            has_motion = True
        
        if not has_motion:
            return None
        
        # Clamp to valid range
        alpha = max(-1.0, min(1.0, alpha))
        beta = max(-1.0, min(1.0, beta))
        
        # Update state
        self.state.alpha = alpha
        self.state.beta = beta
        
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
