# bREadbeats Configuration
# All default values and constants

from dataclasses import dataclass, field
from typing import Literal
from enum import IntEnum

class StrokeMode(IntEnum):
    """Stroke mapping modes - all use alpha/beta circular coordinates"""
    SIMPLE_CIRCLE = 1      # Trace full circle on beat
    FIGURE_EIGHT = 2       # Figure-8 pattern
    RANDOM_ARC = 3         # Random arc segments
    USER = 4               # User-controlled via sliders (freq/peak reactive)

class BeatDetectionType(IntEnum):
    PEAK_ENERGY = 1
    SPECTRAL_FLUX = 2
    COMBINED = 3

@dataclass
class BeatDetectionConfig:
    """Beat detection parameters"""
    detection_type: BeatDetectionType = BeatDetectionType.COMBINED
    sensitivity: float = 0.5          # 0.0 - 1.0
    peak_floor: float = 0.1           # Minimum threshold
    peak_decay: float = 0.9           # How fast peaks decay (0.0-1.0)
    rise_sensitivity: float = 0.5     # How fast a peak must hit to register
    amplification: float = 1.0        # Audio amplification (slider 0-2)
    flux_multiplier: float = 1.0      # Weight of spectral flux
    # Frequency band selection (Hz)
    freq_low: float = 20.0            # Low cutoff frequency (Hz)
    freq_high: float = 200.0          # High cutoff frequency (Hz) - bass range default

@dataclass
class StrokeConfig:
    """Stroke generation parameters"""
    mode: StrokeMode = StrokeMode.SIMPLE_CIRCLE
    stroke_min: float = 0.2           # Minimum stroke length (0.0-1.0)
    stroke_max: float = 1.0           # Maximum stroke length (0.0-1.0)
    min_interval_ms: int = 300        # Minimum time between strokes (ms) - slider 200->1000
    stroke_fullness: float = 0.7      # How much params affect stroke length
    minimum_depth: float = 0.0        # Lower limit of stroke (absolute bottom)
    freq_depth_factor: float = 0.3    # How much frequency affects depth
    
    # Spectral flux-based stroke control
    flux_threshold: float = 0.03      # Threshold to distinguish low vs high flux
    # Low flux (<threshold): only full strokes on downbeats
    # High flux (>=threshold): full strokes on every beat

@dataclass
class JitterConfig:
    """Jitter - micro-circles when no beat detected"""
    enabled: bool = True
    intensity: float = 0.3            # Speed of jitter movement
    amplitude: float = 0.1            # Circle size (slider 0.05-0.2)

@dataclass
class CreepConfig:
    """Creep - very slow movement when idle"""
    enabled: bool = True
    speed: float = 0.25               # How fast to creep (0.0-1.0) - 0.25 = 1/4 tempo speed

@dataclass 
class ConnectionConfig:
    """TCP connection to restim"""
    host: str = "127.0.0.1"
    port: int = 12347
    auto_connect: bool = True
    reconnect_delay_ms: int = 3000

@dataclass
class AudioConfig:
    """Audio capture settings"""
    sample_rate: int = 44100
    buffer_size: int = 1024
    channels: int = 2
    # Device 1 = Stereo Mix (MME) - should capture after linking to LOGITECH SPEAKERS
    device_index: int | None = 1
    # Audio gain/amplification removed - not in original working config
    gain: float = 1.0  # Changed from 100x back to 1x (no amplification)
    # Audio gain/amplification removed - not in original working config
    gain: float = 1.0  # Changed from 100x back to 1x (no amplification)

@dataclass
class Config:
    """Master configuration"""
    beat: BeatDetectionConfig = field(default_factory=BeatDetectionConfig)
    stroke: StrokeConfig = field(default_factory=StrokeConfig)
    jitter: JitterConfig = field(default_factory=JitterConfig)
    creep: CreepConfig = field(default_factory=CreepConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    
    # Global
    alpha_weight: float = 1.0         # Per-axis mix for alpha
    beta_weight: float = 1.0          # Per-axis mix for beta


# Default config instance
DEFAULT_CONFIG = Config()
