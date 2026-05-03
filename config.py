# bREadbeats Configuration
# All default values and constants

from dataclasses import dataclass, field, is_dataclass
from typing import Dict
from enum import IntEnum


CURRENT_CONFIG_VERSION = 1

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
    sensitivity: float = 0.48          # 0.0 - 1.0
    peak_floor: float = 0.02          # Minimum threshold
    peak_decay: float = 0.3           # How fast peaks decay
    rise_sensitivity: float = 0.1     # How fast a peak must hit
    amplification: float = 1.0        # Audio amplification
    flux_multiplier: float = 1.0      # Weight of spectral flux
    superflux_max_filter_size: int = 3 # SuperFlux vibrato suppression (1=off, 3=default)
    freq_low: float = 100.0           
    freq_high: float = 8000.0        
    motion_freq_cutoff: float = 180.0  
    silence_reset_ms: int = 180       
    
    # --- CHANNEL-BUS ISOLATION FIELDS (NEW) ---
    trigger_bus_refractory_ms: float = 140.0
    trigger_bus_arm_threshold: float = 0.25
    trigger_bus_release_threshold: float = 0.18
    trigger_bus_sustain_frames: int = 2
    trigger_bus_weight_sub_bass: float = 0.45
    trigger_bus_weight_low_mid: float = 0.25
    trigger_bus_weight_mid: float = 0.15
    trigger_bus_weight_high: float = 0.15
    trigger_bus_mask_floor: float = 0.25
    bass_dominance_weighting_enabled: bool = True
    transient_classification_enabled: bool = True
    # ── Data-calibrated from CH-Tranquilizer (2026-02-25) ──
    transient_full_motion_min_kick_conf: float = 0.40     # lowered: more permissive kick detection
    transient_full_motion_min_bass_dom: float = 1.40      # lowered: less bass dominance required
    transient_full_motion_decisive_bass_dom: float = 2.00  # lowered: easier decisive bass threshold
    transient_full_motion_min_flux: float = 0.08           # lowered: less flux required for full motion
    transient_full_motion_min_energy_fullness: float = 0.10 # lowered: less energy fullness needed
    # ------------------------------------------

    tempo_tracking_enabled: bool = True  
    stability_threshold: float = 0.13    
    tempo_timeout_ms: int = 1100         
    beats_per_measure: int = 4           
    phase_snap_weight: float = 0.65      
    pattern_match_tolerance_ms: float = 100.0  
    consecutive_match_threshold: int = 3       
    downbeat_pattern_enabled: bool = True      
    acf_interval_ms: float = 180.0              
    metronome_bpm_alpha_slow: float = 0.08      
    metronome_bpm_alpha_fast: float = 0.40      
    metronome_pll_window: float = 0.35          
    metronome_pll_base_gain: float = 0.25       
    metronome_pll_conf_gain: float = 0.18       
    tempo_fusion_min_acf_weight: float = 0.15   
    tempo_fusion_max_acf_weight: float = 0.82   
    beat_dedup_fraction: float = 0.20            
    phase_accept_window_ms: float = 50.0         
    phase_accept_low_conf_mult: float = 2.0      
    beat_refractory_ms: float = 140.0            
    aggressive_tempo_snap_enabled: bool = True  
    aggressive_snap_confidence: float = 0.50     
    aggressive_snap_phase_error_ms: float = 55.0 
    aggressive_snap_min_matches: int = 1         
    aggressive_snap_max_bpm_jump_ratio: float = 0.12  
    octave_target_bias_confidence_max: float = 0.35  
    target_bps_lock_gate_enabled: bool = True         
    target_bps_lock_gate_acf_conf: float = 0.40      
    target_bps_lock_gate_downbeats: int = 1          
    syncopation_enabled: bool = True             
    syncopation_band: str = 'any'                
    syncopation_window: float = 0.20             
    syncopation_bpm_limit: float = 130.0         
    syncopation_arc_size: float = 0.82                
    syncopation_speed: float = 1.0                 
    scheduled_lead_ms: int = 55                    
    center_jitter_flux_guard_enabled: bool = True 
    center_jitter_flux_delta_threshold: float = 0.20  
    center_jitter_flux_avg_threshold: float = 0.25    
    teaching_learning_enabled: bool = True      
    teaching_learning_strength: float = 0.59    
    teaching_min_confidence: float = 0.39       
    teaching_use_fitted_rules: bool = True      
    teaching_rule_fit_path: str = "defaults/learning/rule_fit.tranquilizer_blend.json"  
    teaching_no_motion_bias: float = 1.5        
    teaching_apply_in_circle_mode: bool = True 
    teaching_isolation_mode: bool = True        
    teaching_relax_phase1_gates: bool = False   
    teaching_ignore_traffic_lights: bool = True  
    tempo_lock_required: bool = False            
    teaching_metronome_relaxed_confidence: float = 0.05  
    teaching_stroke_ready_grace_ms: float = 1800.0  
    teaching_stroke_finish_beats: int = 4

@dataclass
class StrokeConfig:
    """Stroke generation parameters"""
    mode: StrokeMode = StrokeMode.SIMPLE_CIRCLE
    min_interval_ms: int = 150        # Minimum time between strokes (ms) - slider 200->1000
    # Frequency band for stroke depth calculation (bass = deeper strokes)
    depth_freq_low: float = 30.0      # Low frequency = deepest strokes (Hz)
    depth_freq_high: float = 22050.0    # High frequency = shallowest strokes (Hz)
    
    # Spectral flux-based stroke control
    flux_threshold: float = 0.068      # Threshold to distinguish low vs high flux
    # Low flux (<threshold): only full strokes on downbeats
    # High flux (>=threshold): full strokes on every beat
    flux_scaling_weight: float = 1.0  # How much flux affects stroke size (0=none, 1=normal, 2=strong)

    # Session-long motion ramp tuning
    intensity_ramp_hours: float = 0.0
    intensity_ramp_target: str = "both"

    # Silence detection thresholds (fade-out when truly silent)
    silence_threshold: float = -66.0      # dBFS threshold to enter silence deadzone gate (first-run tuned)
    silence_close_threshold: float = -58.0  # dBFS threshold to exit silence deadzone gate
    combo_texture: float = 0.94   # syncopation/texture behavior

    # Energy Response: scales fill-gate thresholds based on energy_fullness.
    # 0.0 = no fullness effect on gates, 1.0 = moderate (35% reduction at
    # max fullness), 2.0 = strong (70% reduction at max fullness).
    energy_response_strength: float = 1.0

    # Beat-type-specific orbital geometry: each type blooms from a different center
    # All journey types share the same radius range; only center differs.
    orbit_geometry: dict = field(default_factory=lambda: {
        "downbeat": {"center_y": 0.3, "park_radius": 0.70, "max_radius": 0.90},
        "beat": {"center_y": 0.1, "park_radius": 0.70, "max_radius": 0.90},
        "syncopation": {"center_y": 0.0, "park_radius": 0.70, "max_radius": 0.90},
        "creep": {"center_y": 0.4, "park_radius": 0.70, "max_radius": 0.90},
    })

    overall_amp_fill_gate_enabled: bool = True
    overall_amp_fill_target: float = 0.42
    overall_amp_fill_tolerance: float = 0.50
    downbeat_overall_amp_fill_required: float = 0.04
    beat_overall_amp_fill_required: float = 0.06
    syncopation_overall_amp_fill_required: float = 0.08
    overall_amp_fill_required_scale: float = 0.55
    overall_amp_fill_auto_enabled: bool = True
    overall_amp_fill_auto_target_pass_rate: float = 0.58
    overall_amp_fill_auto_ema_alpha: float = 0.12
    overall_amp_fill_auto_deadband: float = 0.06
    overall_amp_fill_auto_step: float = 0.02
    overall_amp_fill_auto_max_offset: float = 0.35
    overall_amp_fill_auto_min_required: float = 0.05
    overall_amp_fill_auto_max_required: float = 0.98

    # dBFS-based fill gate thresholds (absolute, relative to tracked reference max).
    # Legacy relative-peak mode has been removed.
    downbeat_dbfs_threshold: float = -35.0  # Downbeat: easier threshold (dB below reference)
    beat_dbfs_threshold: float = -40.0      # Beat: moderate threshold (dB below reference)
    syncopation_dbfs_threshold: float = -45.0  # Syncopation: harder threshold (dB below reference)
    dbfs_reference_decay_rate: float = 0.9995  # Per-frame decay multiplier for reference maximum

    # FFT-bin windows used by overall fill gate for each phase.
    # Values are bin indexes (0..N/2) from current FFT size.
    downbeat_fill_bin_low: int = 0
    downbeat_fill_bin_high: int = 4
    beat_fill_bin_low: int = 3
    beat_fill_bin_high: int = 10
    syncopation_fill_bin_low: int = 333
    syncopation_fill_bin_high: int = 372

    # Fill duration gate (per phase): require sustained fullness over consecutive frames.
    # Values are frame counts (~16-23ms per frame at effective 43-60fps processing rate).
    # Set to 0 or 1 to disable duration check (instant single-frame decision) for that phase.
    downbeat_overall_amp_fill_sustain_frames: int = 5
    beat_overall_amp_fill_sustain_frames: int = 6
    syncopation_overall_amp_fill_sustain_frames: int = 10

    # High-band include mid: controls whether 'high' band visualization starts at 500 Hz or 2 kHz.
    high_band_include_mid: bool = True

    # Post-silence volume ramp: reduce volume after silence/track-change, ramp back up
    post_silence_vol_reduction: float = 0.47  # Fraction to reduce volume by (0.0-0.50, 0.15 = 15%)
    post_silence_ramp_seconds: float = 2.7    # Seconds to ramp volume back to full (1.0-8.0)
    silence_fade_drop_points: int = 10        # Max points (out of 100) fade can lower volume by in runtime fade pass

    # ── Simple Mode (intelligence disabled) ──
    intelligence_enabled: bool = True       # False = simple mode: no gating, no expression, pure circles
    experimental_simple: bool = False       # True = beat-reactive Y shuttle + half-circle arc (experimental)
    simple_mode_beats_per_rotation: int = 1 # 1, 2, or 4 beats per full 360° circle
    simple_mode_base_radius: float = 0.90   # constant orbit radius (energy can push to 1.0)

    # ── Expression Layer ──
    # Center wandering: orbit center drifts horizontally over time
    center_wander_enabled: bool = True
    center_wander_max_x: float = 0.08         # max horizontal center offset
    center_wander_cycle_s: float = 40.0       # full wander cycle period (seconds)
    center_wander_energy_scale: float = 0.3   # energy influence on wander amplitude

    # Direction changes: CW/CCW reversal at phrase boundaries
    direction_change_enabled: bool = True
    direction_change_interval_s: float = 15.0 # min seconds between direction changes
    direction_change_energy_drop: float = 0.35  # energy drop ratio to trigger reversal

    # Session arc: very slow intensity tracking for gradual session evolution
    session_arc_enabled: bool = True
    session_arc_ema_alpha: float = 0.001      # EMA alpha for session intensity
    session_arc_radius_influence: float = 0.10  # max radius modifier from session arc

@dataclass
class JitterConfig:
    """Jitter - micro-circles when no beat detected"""
    enabled: bool = True
    intensity: float = 9.5            # Speed of jitter movement
    size: float = 0.012               # Circle size

    @property
    def amplitude(self) -> float:
        return float(self.size)

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        self.size = float(value)

@dataclass 
class ConnectionConfig:
    """TCP connection to restim"""
    host: str = "127.0.0.1"
    port: int = 12347
    auto_connect: bool = True
    reconnect_delay_ms: int = 3000

@dataclass
class PulseFreqConfig:
    """Pulse frequency mapping settings (P0 TCode)"""
    monitor_freq_min: float = 30.0    # Min frequency to monitor (Hz)
    monitor_freq_max: float = 4000.0   # Max frequency to monitor (Hz)
    tcode_min: int = 2010             # Min sent TCode value (0-9999)
    tcode_max: int = 7035             # Max sent TCode value (0-9999)
    freq_weight: float = 1.0          # How much frequency affects P0 (0=none, 1=full)
    mode: int = 0                     # 0=Hz, 1=Speed, 2=Band
    invert: bool = False              # Invert mapped output
    enabled: bool = False             # Enable P0 sending

@dataclass
class CarrierFreqConfig:
    """Carrier frequency (C0 TCode) mapping settings"""
    monitor_freq_min: float = 30.0    # Min frequency to monitor (Hz)
    monitor_freq_max: float = 4000.0   # Max frequency to monitor (Hz)
    tcode_min: int = 0                # Min sent TCode value (0-9999)
    tcode_max: int = 5000             # Max sent TCode value (0-9999)
    freq_weight: float = 1.0          # How much frequency affects C0 (0=none, 1=full)
    mode: int = 0                     # 0=Hz, 1=Speed, 2=Band
    invert: bool = False              # Invert mapped output
    enabled: bool = False             # Enable C0 sending

@dataclass
class DeviceLimitsConfig:
    """User-defined device output ranges for TCode conversion display.
    When configured (non-zero), displays show converted values alongside TCode.
    P0/C0 = frequency in Hz. P1 = pulse width in carrier cycles.
    P2 = pulse interval random (0-1). P3 = rise time in carrier cycles."""
    p0_freq_min: float = 1.0          # Device P0 min frequency (Hz), 0 = not set
    p0_freq_max: float = 100.0          # Device P0 max frequency (Hz), 0 = not set
    c0_freq_min: float = 0.0          # Device C0 min frequency (Hz), 0 = not set
    c0_freq_max: float = 0.0          # Device C0 max frequency (Hz), 0 = not set
    p1_cycles_min: float = 0.0        # Device P1 min pulse width (cycles), 0 = not set
    p1_cycles_max: float = 0.0        # Device P1 max pulse width (cycles), 0 = not set
    p2_range_min: float = 0.0         # Device P2 min interval random, 0 = not set
    p2_range_max: float = 0.0         # Device P2 max interval random, 0 = not set
    p3_cycles_min: float = 0.0        # Device P3 min rise time (cycles), 0 = not set
    p3_cycles_max: float = 0.0        # Device P3 max rise time (cycles), 0 = not set
    prompted: bool = True            # Whether user has been prompted on first run
    p0_c0_sending_enabled: bool = True  # Whether to actually send P0/C0 TCode to device
    dont_show_on_startup: bool = False  # User opted out of startup device limits dialog
    dry_run: bool = False             # When True, do not send network commands (log-only)

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
    metric_response_speed: float = 1.16   # 1.0 = legacy speed, >1 faster cadence/adjustment
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
    sample_rate: int = 48000
    buffer_size: int = 1024
    channels: int = 2
    # Device index - None means use system default
    device_index: int | None = 16
    # Audio gain/amplification
    gain: float = 6.2
    # FFT optimization settings
    fft_size: int = 1024              # FFT size (512, 1024, 2048) - smaller = faster, less resolution
    spectrum_skip_frames: int = 2     # Skip N frames between spectrum updates (1=no skip, 2=every other)
    is_loopback: bool = True          # True for WASAPI loopback, False for regular input
    # Performance options
    visualizer_enabled: bool = True   # Enable/disable spectrum visualizer (saves CPU)
    highpass_filter_hz: int = 30      # High-pass filter cutoff (0=disabled, 30=filter sub-bass noise)
    use_butterworth: bool = True      # Use Butterworth bandpass for beat detection
    volume_normalize: bool = True     # Compensate for Windows volume (pycaw) so beat detection is volume-independent


@dataclass
class FunscriptConverterConfig:
    """Persisted UI state for the standalone FunScript Converter."""

    layout_model: str = "Pair At Middle"
    wiring_map: list[int] = field(default_factory=lambda: [0, 1, 2, 3])

@dataclass
class Config:
    """Master configuration"""
    version: int = 1                  # Schema version for persisted configs
    beat: BeatDetectionConfig = field(default_factory=BeatDetectionConfig)
    stroke: StrokeConfig = field(default_factory=StrokeConfig)
    jitter: JitterConfig = field(default_factory=JitterConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    pulse_freq: PulseFreqConfig = field(default_factory=PulseFreqConfig)
    carrier_freq: CarrierFreqConfig = field(default_factory=CarrierFreqConfig)
    pulse_width: PulseWidthConfig = field(default_factory=PulseWidthConfig)
    rise_time: RiseTimeConfig = field(default_factory=RiseTimeConfig)
    device_limits: DeviceLimitsConfig = field(default_factory=DeviceLimitsConfig)
    auto_adjust: AutoAdjustConfig = field(default_factory=AutoAdjustConfig)
    funscript_converter: FunscriptConverterConfig = field(default_factory=FunscriptConverterConfig)
    
    # Global
    base_radius: float = 0.30        # Global idle orbit radius (0.05-1.0)
    alpha_weight: float = 1.0         # Per-axis mix for alpha
    beta_weight: float = 1.0          # Per-axis mix for beta
    volume: float = 1.0               # Output volume (0.0-1.0)
    log_level: str = "INFO"           # Logging level (DEBUG/INFO/WARNING/ERROR)


def apply_dict_to_dataclass(target, data) -> None:
    """Recursively apply values from a dict onto a dataclass instance.
    Unknown keys are ignored; IntEnum fields are coerced when possible."""
    if not isinstance(data, dict):
        return


    for key, value in data.items():
        if not hasattr(target, key):
            continue

        current = getattr(target, key)

        if is_dataclass(current) and isinstance(value, dict):
            apply_dict_to_dataclass(current, value)
            continue

        if isinstance(current, IntEnum):
            try:
                setattr(target, key, current.__class__(value))
                continue
            except Exception:
                print(f"[Config] Warning: Could not convert {key} to {current.__class__.__name__}, keeping default")
                continue

        setattr(target, key, value)

    if hasattr(target, 'size') and hasattr(target, 'amplitude') and isinstance(data, dict):
        if 'size' in data:
            target.amplitude = target.size
        else:
            target.size = target.amplitude


def migrate_config(config: Config, loaded_version) -> None:
    """Upgrade older config structures to the current schema.
    Adds defaults for newly introduced fields and bumps version."""
    try:
        version = int(loaded_version) if loaded_version is not None else 0
    except Exception:
        version = 0

    if version < 1:
        if getattr(config.device_limits, 'p0_c0_sending_enabled', True) is None:
            config.device_limits.p0_c0_sending_enabled = True
        if getattr(config.device_limits, 'dont_show_on_startup', False) is None:
            config.device_limits.dont_show_on_startup = False
        if getattr(config.device_limits, 'prompted', False) is None:
            config.device_limits.prompted = False
        if getattr(config.device_limits, 'dry_run', False) is None:
            config.device_limits.dry_run = False

    try:
        jitter_size = float(getattr(config.jitter, 'size', getattr(config.jitter, 'amplitude', 0.024)))
    except Exception:
        jitter_size = 0.024
    config.jitter.size = max(0.0, min(0.2, jitter_size))
    config.jitter.amplitude = config.jitter.size

    try:
        layout_model = str(getattr(config.funscript_converter, 'layout_model', 'Pair At Middle') or 'Pair At Middle')
    except Exception:
        layout_model = 'Pair At Middle'
    if layout_model not in {'Pair At Top', 'Pair At Middle', 'Pair At Bottom / Rear'}:
        layout_model = 'Pair At Middle'
    config.funscript_converter.layout_model = layout_model

    raw_wiring_map = getattr(config.funscript_converter, 'wiring_map', [0, 1, 2, 3])
    try:
        wiring_map = [int(value) for value in raw_wiring_map]
    except Exception:
        wiring_map = [0, 1, 2, 3]
    if len(wiring_map) != 4 or sorted(wiring_map) != [0, 1, 2, 3]:
        wiring_map = [0, 1, 2, 3]
    config.funscript_converter.wiring_map = wiring_map

    config.version = CURRENT_CONFIG_VERSION


BEAT_RANGE_LIMITS = {
    'audio_amp': (0.15, 10.0),
    'peak_floor': (0.015, 2.0),
    'peak_decay': (0.230, 0.999),
    'rise_sens': (0.02, 1.0),
    'sensitivity': (0.10, 1.0),
    'flux_mult': (0.2, 10.0),
}

