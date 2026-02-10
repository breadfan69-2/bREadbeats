# bREadbeats Configuration
# All default values and constants

from dataclasses import dataclass, field
from typing import Dict, Literal
from enum import IntEnum

class StrokeMode(IntEnum):
    """Stroke mapping modes - all use alpha/beta circular coordinates"""
    SIMPLE_CIRCLE = 1      # Trace full circle on beat
    SPIRAL = 2             # Spiral pattern (Archimedean)
    TEARDROP = 3           # Teardrop pattern (piriform)
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
    freq_low: float = 30.0            # Low cutoff frequency (Hz)
    freq_high: float = 200.0          # High cutoff frequency (Hz) - bass range default
    silence_reset_ms: int = 400       # How long silence before resetting beat tracking (ms)
    
    # Tempo tracking parameters
    tempo_tracking_enabled: bool = True  # Enable/disable tempo & downbeat tracking
    stability_threshold: float = 0.28    # Max CV to consider tempo "stable" (lower = stricter)
    tempo_timeout_ms: int = 2000         # How long no beats before resetting tempo tracking (ms)
    beats_per_measure: int = 4           # Time signature: 4 = 4/4, 3 = 3/4, 6 = 6/8
    phase_snap_weight: float = 0.3       # How much to snap detected beats toward predicted time (0=off, 1=full)
    
    # Downbeat pattern matching (strict tempo mode)
    pattern_match_tolerance_ms: float = 100.0  # Max deviation from predicted beat (ms) to accept downbeat
    consecutive_match_threshold: int = 3       # N consecutive matching downbeats to lock tempo
    downbeat_pattern_enabled: bool = True      # Enable/disable strict downbeat pattern matching

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
    # Frequency band for stroke depth calculation (bass = deeper strokes)
    depth_freq_low: float = 30.0      # Low frequency = deepest strokes (Hz)
    depth_freq_high: float = 200.0    # High frequency = shallowest strokes (Hz)
    
    # Spectral flux-based stroke control
    flux_threshold: float = 0.03      # Threshold to distinguish low vs high flux
    # Low flux (<threshold): only full strokes on downbeats
    # High flux (>=threshold): full strokes on every beat
    flux_scaling_weight: float = 1.0  # How much flux affects stroke size (0=none, 1=normal, 2=strong)

    # Silence detection thresholds (fade-out when truly silent)
    silence_flux_multiplier: float = 0.15  # quiet_flux_thresh = flux_threshold * this (0.01-1.0)
    silence_energy_multiplier: float = 0.7  # quiet_energy_thresh = peak_floor * this (0.1-2.0)
    silence_multiplier_locked: bool = True  # Lock sliders on startup

    # Volume reduction limit: max % volume can be reduced by effects (subtractive clamp)
    vol_reduction_limit: float = 10.0  # 0-20, default 10 means max 10% reduction (floor = 0.90)

    # Flux-rise depth factor: modulates stroke depth by spectral flux rise over 250ms
    flux_depth_factor: float = 0.0     # 0-5, 0=disabled

    # Phase advance per beat (0.0 = only downbeats, 1.0 = every beat does a full circle)
    phase_advance: float = 0.25

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
    speed: float = 0.02               # Multiplier for creep rotation (0.0-0.1) - lower = slower drift

@dataclass 
class ConnectionConfig:
    """TCP connection to restim"""
    host: str = "127.0.0.1"
    port: int = 12347
    auto_connect: bool = True
    reconnect_delay_ms: int = 3000

@dataclass
class PulseFreqConfig:
    """Pulse frequency mapping settings (Other tab)"""
    monitor_freq_min: float = 30.0    # Min frequency to monitor (Hz)
    monitor_freq_max: float = 200.0   # Max frequency to monitor (Hz)
    tcode_freq_min: float = 30.0      # Min sent frequency (Hz, converted to TCode)
    tcode_freq_max: float = 105.0     # Max sent frequency (Hz, converted to TCode)
    freq_weight: float = 1.0          # How much frequency affects P0 (0=none, 1=full)

@dataclass
class CarrierFreqConfig:
    """Carrier frequency (F0 TCode) mapping settings"""
    monitor_freq_min: float = 30.0    # Min frequency to monitor (Hz)
    monitor_freq_max: float = 200.0   # Max frequency to monitor (Hz)
    tcode_freq_min: float = 500.0     # Min sent frequency (display 500-1500, converted to TCode 0-9999)
    tcode_freq_max: float = 1000.0    # Max sent frequency (display 500-1500, converted to TCode 0-9999)
    freq_weight: float = 1.0          # How much frequency affects F0 (0=none, 1=full)

@dataclass
class PulseWidthConfig:
    """Pulse Width (P1 TCode) mapping settings — higher = stronger/smoother feeling"""
    monitor_freq_min: float = 30.0    # Min frequency to monitor (Hz)
    monitor_freq_max: float = 4000.0  # Max frequency to monitor (Hz)
    tcode_min: int = 1000             # Min sent TCode value (0-9999)
    tcode_max: int = 8000             # Max sent TCode value (0-9999)
    weight: float = 1.0               # How much audio affects P1 (0=none, 1=full)

@dataclass
class RiseTimeConfig:
    """Rise Time (P3 TCode) mapping settings — higher = smoother/gentler feeling"""
    monitor_freq_min: float = 30.0    # Min frequency to monitor (Hz)
    monitor_freq_max: float = 4000.0  # Max frequency to monitor (Hz)
    tcode_min: int = 1000             # Min sent TCode value (0-9999)
    tcode_max: int = 8000             # Max sent TCode value (0-9999)
    weight: float = 1.0               # How much audio affects P3 (0=none, 1=full)

@dataclass
class AutoAdjustConfig:
    """Auto-adjust (hunting) step sizes and related settings"""
    # Step sizes for each parameter
    step_sensitivity: float = 0.008
    step_peak_floor: float = 0.004
    step_peak_decay: float = 0.002
    step_rise_sens: float = 0.008
    step_flux_mult: float = 0.015
    step_audio_amp: float = 0.040
    
    # Global settings for auto-adjust
    threshold_sec: float = 0.43       # Beat interval threshold in seconds
    cooldown_sec: float = 0.10        # Cooldown between adjustments
    consec_beats: int = 8             # Consecutive beats required to lock
    auto_range_enabled: bool = False  # Global auto-range toggle persistence
    metrics_global_enabled: bool = True  # Master toggle for all metric auto-adjust
    enabled_params: Dict[str, bool] = field(default_factory=lambda: {
        'audio_amp': False,
        'peak_floor': False,
        'peak_decay': False,
        'rise_sens': False,
        'sensitivity': False,
        'flux_mult': False,
    })

@dataclass
class AudioConfig:
    """Audio capture settings"""
    sample_rate: int = 44100
    buffer_size: int = 1024
    channels: int = 2
    # Device index - None means use system default
    device_index: int | None = None
    # Audio gain/amplification
    gain: float = 1.0
    # FFT optimization settings
    fft_size: int = 1024              # FFT size (512, 1024, 2048) - smaller = faster, less resolution
    spectrum_skip_frames: int = 2     # Skip N frames between spectrum updates (1=no skip, 2=every other)
    is_loopback: bool = True          # True for WASAPI loopback, False for regular input
    # Performance options
    visualizer_enabled: bool = True   # Enable/disable spectrum visualizer (saves CPU)
    highpass_filter_hz: int = 30      # High-pass filter cutoff (0=disabled, 30=filter sub-bass noise)
    use_butterworth: bool = True      # Use Butterworth bandpass for beat detection

@dataclass
class Config:
    """Master configuration"""
    beat: BeatDetectionConfig = field(default_factory=BeatDetectionConfig)
    stroke: StrokeConfig = field(default_factory=StrokeConfig)
    jitter: JitterConfig = field(default_factory=JitterConfig)
    creep: CreepConfig = field(default_factory=CreepConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    pulse_freq: PulseFreqConfig = field(default_factory=PulseFreqConfig)
    carrier_freq: CarrierFreqConfig = field(default_factory=CarrierFreqConfig)
    pulse_width: PulseWidthConfig = field(default_factory=PulseWidthConfig)
    rise_time: RiseTimeConfig = field(default_factory=RiseTimeConfig)
    auto_adjust: AutoAdjustConfig = field(default_factory=AutoAdjustConfig)
    
    # Global
    alpha_weight: float = 1.0         # Per-axis mix for alpha
    beta_weight: float = 1.0          # Per-axis mix for beta
    volume: float = 1.0               # Output volume (0.0-1.0)
    log_level: str = "INFO"           # Logging level (DEBUG/INFO/WARNING/ERROR)


# Default config instance
DEFAULT_CONFIG = Config()

# Centralized parameter defaults/ranges (reference only; wiring remains in main.py widgets)
BEAT_RESET_DEFAULTS = {
    'audio_amp': 0.15,
    'peak_floor': 0.08,
    'peak_decay': 0.999,
    'rise_sens': 0.02,
    'sensitivity': 0.1,
    'flux_mult': 0.2,
}

BEAT_RANGE_LIMITS = {
    'audio_amp': (0.15, 10.0),
    'peak_floor': (0.015, 2.0),
    'peak_decay': (0.230, 0.999),
    'rise_sens': (0.02, 1.0),
    'sensitivity': (0.10, 1.0),
    'flux_mult': (0.2, 10.0),
}
