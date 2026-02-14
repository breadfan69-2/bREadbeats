# bREadbeats Configuration
# All default values and constants

from dataclasses import dataclass, field, is_dataclass
from typing import Dict, Literal
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
    sensitivity: float = 0.5          # 0.0 - 1.0
    peak_floor: float = 0.08          # Minimum threshold (aligned with reset defaults)
    peak_decay: float = 0.9           # How fast peaks decay (0.0-1.0)
    rise_sensitivity: float = 0.5     # How fast a peak must hit to register
    amplification: float = 1.0        # Audio amplification (slider 0-2)
    flux_multiplier: float = 1.0      # Weight of spectral flux
    # Frequency band selection (Hz)
    freq_low: float = 30.0            # Low cutoff frequency (Hz)
    freq_high: float = 150.0          # High cutoff frequency (Hz) - bass range default
    motion_freq_cutoff: float = 500.0  # Only generate motion from bands below this Hz (0=disabled)
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

    # Tempo response tuning (advanced)
    acf_interval_ms: float = 250.0              # ACF update cadence in milliseconds
    metronome_bpm_alpha_slow: float = 0.03      # BPM smoothing alpha when confidence is low
    metronome_bpm_alpha_fast: float = 0.22      # BPM smoothing alpha when confidence is high
    metronome_pll_window: float = 0.35          # Phase-lock correction window (beat fraction)
    metronome_pll_base_gain: float = 0.09       # Base PLL gain
    metronome_pll_conf_gain: float = 0.08       # Extra PLL gain from confidence
    tempo_fusion_min_acf_weight: float = 0.20   # Minimum ACF weight in ACF/onset tempo fusion
    tempo_fusion_max_acf_weight: float = 0.95   # Maximum ACF weight in ACF/onset tempo fusion
    beat_dedup_fraction: float = 0.22            # Ignore second onset inside this fraction of a beat period
    phase_accept_window_ms: float = 85.0         # Base raw-onset acceptance window around expected beat (ms)
    phase_accept_low_conf_mult: float = 2.0      # Multiply phase window when metronome confidence is low
    beat_refractory_ms: float = 170.0            # Min spacing between accepted beats (ms), independent of stroke min interval
    aggressive_tempo_snap_enabled: bool = False  # Hard-snap metronome BPM when lock confidence is high
    aggressive_snap_confidence: float = 0.55     # Min ACF confidence required for aggressive snap
    aggressive_snap_phase_error_ms: float = 35.0 # Max phase error allowed for aggressive snap
    aggressive_snap_min_matches: int = 1         # Min consecutive matching downbeats for snap
    aggressive_snap_max_bpm_jump_ratio: float = 0.12  # Max relative BPM jump allowed per snap

    # Syncopation / double-stroke detection
    syncopation_enabled: bool = True             # Master on/off for syncopation detection
    syncopation_band: str = 'any'                # Which z-score band triggers syncope: 'any', 'sub_bass', 'low_mid', 'mid', 'high'
    syncopation_window: float = 0.15             # ±fraction of beat period to detect off-beat (0.05-0.30)
    syncopation_bpm_limit: float = 160.0         # Disable syncopation above this BPM
    syncopation_arc_size: float = 0.5              # Arc sweep as fraction of circle (0.25=90°, 0.5=180°, 1.0=360°)
    syncopation_speed: float = 0.5                 # Duration as fraction of beat interval (0.25=quarter, 0.5=half, 1.0=full)
    scheduled_lead_ms: int = 0                     # Land scheduled arcs this many ms before predicted beat (0-200)
    strict_bass_motion_gate_enabled: bool = False  # Require sub_bass/low_mid z-score fired bands for beat/sync stroke motion
    center_jitter_flux_guard_enabled: bool = False # Prevent no-beat center+jitter reset while flux activity is still high
    center_jitter_flux_delta_threshold: float = 0.20  # Rising-flux threshold to hold center+jitter reset
    center_jitter_flux_avg_threshold: float = 0.25    # Recent-average flux threshold to hold center+jitter reset

@dataclass
class StrokeConfig:
    """Stroke generation parameters"""
    mode: StrokeMode = StrokeMode.SIMPLE_CIRCLE
    stroke_min: float = 0.2           # Minimum stroke length (0.0-1.0)
    stroke_max: float = 1.0           # Maximum stroke length (0.0-1.0)
    min_interval_ms: int = 260        # Minimum time between strokes (ms) - slider 200->1000
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

    # Flux-rise depth factor over 250ms. Behavior is selected by flux_depth_boost_enabled.
    # boost=False: compress depth toward minimum_depth on flux rise (legacy/default)
    # boost=True:  boost depth toward 1.0 on flux rise
    flux_depth_factor: float = 0.0     # 0-5, 0=disabled
    flux_depth_boost_enabled: bool = False  # False=compressed (legacy), True=boost

    # Main Controls master combinations (1.0 = neutral)
    combo_size: float = 1.0      # stroke size/fullness/flux-scaling/intensity-curve influence
    combo_power: float = 1.0     # downbeat lock boost/jitter blend/scheduled lead
    combo_depth: float = 1.0     # minimum depth/freq depth/flux-depth behavior
    combo_speed: float = 1.0     # cadence density + min-interval behavior
    combo_texture: float = 1.0   # noise burst + syncopation texture behavior
    combo_reaction: float = 1.0  # gate/strictness/readiness aggressiveness

    # Phase advance per beat (0.0 = only downbeats, 1.0 = every beat does a full circle)
    phase_advance: float = 0.25

    # Amplitude gate thresholds for FULL_STROKE vs CREEP_MICRO mode switching
    amplitude_gate_high: float = 0.08  # RMS above this -> FULL_STROKE
    amplitude_gate_low: float = 0.04   # RMS below this -> CREEP_MICRO
    full_stroke_dwell_bias: float = 0.0  # +/- RMS hysteresis bias (0 = disabled)

    # Stroke timing cadence:
    # - 1 beat/stroke only allowed at very slow tempo (< single_stroke_bpm_cutoff)
    # - otherwise auto-select 2/4/8 beats per stroke from BPM cutoffs
    # - beats_between_strokes acts as fallback when BPM is unavailable (2/4/8)
    single_stroke_bpm_cutoff: float = 90.0   # Allow 1 beat/stroke only below this BPM
    bpm_cutoff_2_to_4: float = 60.0          # BPM at/above this moves 2 -> 4 beats/stroke
    bpm_cutoff_4_to_8: float = 180.0         # BPM at/above this moves 4 -> 8 beats/stroke
    beats_between_strokes: int = 2           # Fallback cadence when BPM unavailable (2/4/8 only)
    cadence_cutoff_bias_bpm: float = 0.0     # +/- BPM shift applied to cadence cutoffs (0 = disabled)

    # Thump: legacy setting, replaced by landing durations
    thump_enabled: bool = False             # Kept for preset compatibility, not used in UI

    # Noise-burst reactive arc (hybrid with metronome system)
    # Fires a quick partial arc on sudden loud transients between beats
    noise_burst_enabled: bool = True        # Allow transient-reactive arcs between beats
    noise_burst_flux_multiplier: float = 2.0  # Fire burst when flux > flux_threshold * this
    noise_burst_magnitude: float = 1.0      # Magnitude scaling for noise burst patterns (0.5-5.0)
    noise_burst_scale: float = 0.35         # Final burst downscale applied after magnitude/energy (0.0-0.5)
    downbeat_jitter_vector_percent: float = 50.0  # % of current jitter vector added to downbeat arc points
    bass_jitter_speed_influence_percent: float = 100.0  # % depth of bass-frequency influence on jitter speed
    bass_jitter_size_influence_percent: float = 0.0     # % depth of bass-frequency influence on jitter size
    noise_primary_mode: bool = False        # True: noise fires strokes, metronome verifies; False: metronome fires, noise supplements

    # Flux-drop detection: if recent flux drops below this fraction of older flux, force creep
    flux_drop_ratio: float = 0.25           # 0.0-1.0, lower = less sensitive (needs bigger drop)

    # Low-band activity gate for beat-based stroke generation
    # Uses sub_bass + low_mid activity window in StrokeMapper.
    # Beat strokes require:
    #   mean >= threshold AND (delta >= threshold OR variance >= threshold)
    # Downbeats use the same concept with a slightly relaxed threshold multiplier.
    low_band_window_frames: int = 18
    low_band_activity_threshold: float = 0.20
    low_band_delta_threshold: float = 0.06
    low_band_variance_threshold: float = 0.0015
    downbeat_low_band_relax: float = 0.85
    low_band_drop_guard_enabled: bool = True

    # Overall full-spectrum quiet guard for beat/downbeat stroke generation.
    # Blocks beat-based strokes only when BOTH spectral flux and peak energy
    # are below these thresholds.
    overall_activity_guard_enabled: bool = True
    overall_low_flux_threshold: float = 0.06
    overall_low_energy_threshold: float = 0.14

    # Post-silence volume ramp: reduce volume after silence/track-change, ramp back up
    post_silence_vol_reduction: float = 0.15  # Fraction to reduce volume by (0.0-0.50, 0.15 = 15%)
    post_silence_ramp_seconds: float = 3.0    # Seconds to ramp volume back to full (1.0-8.0)

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
    speed: float = 0.02               # Multiplier for creep rotation (0.0-2.0) - lower = slower drift

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
    monitor_freq_max: float = 200.0   # Max frequency to monitor (Hz)
    tcode_min: int = 2010             # Min sent TCode value (0-9999)
    tcode_max: int = 7035             # Max sent TCode value (0-9999)
    freq_weight: float = 1.0          # How much frequency affects P0 (0=none, 1=full)

@dataclass
class CarrierFreqConfig:
    """Carrier frequency (C0 TCode) mapping settings"""
    monitor_freq_min: float = 30.0    # Min frequency to monitor (Hz)
    monitor_freq_max: float = 200.0   # Max frequency to monitor (Hz)
    tcode_min: int = 0                # Min sent TCode value (0-9999)
    tcode_max: int = 5000             # Max sent TCode value (0-9999)
    freq_weight: float = 1.0          # How much frequency affects C0 (0=none, 1=full)

@dataclass
class DeviceLimitsConfig:
    """User-defined device output ranges for TCode conversion display.
    When configured (non-zero), displays show converted values alongside TCode.
    P0/C0 = frequency in Hz. P1 = pulse width in carrier cycles.
    P2 = pulse interval random (0-1). P3 = rise time in carrier cycles."""
    p0_freq_min: float = 0.0          # Device P0 min frequency (Hz), 0 = not set
    p0_freq_max: float = 0.0          # Device P0 max frequency (Hz), 0 = not set
    c0_freq_min: float = 0.0          # Device C0 min frequency (Hz), 0 = not set
    c0_freq_max: float = 0.0          # Device C0 max frequency (Hz), 0 = not set
    p1_cycles_min: float = 0.0        # Device P1 min pulse width (cycles), 0 = not set
    p1_cycles_max: float = 0.0        # Device P1 max pulse width (cycles), 0 = not set
    p2_range_min: float = 0.0         # Device P2 min interval random, 0 = not set
    p2_range_max: float = 0.0         # Device P2 max interval random, 0 = not set
    p3_cycles_min: float = 0.0        # Device P3 min rise time (cycles), 0 = not set
    p3_cycles_max: float = 0.0        # Device P3 max rise time (cycles), 0 = not set
    prompted: bool = False            # Whether user has been prompted on first run
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
    metric_response_speed: float = 1.0   # 1.0 = legacy speed, >1 faster cadence/adjustment
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
    version: int = 1                  # Schema version for persisted configs
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
    device_limits: DeviceLimitsConfig = field(default_factory=DeviceLimitsConfig)
    auto_adjust: AutoAdjustConfig = field(default_factory=AutoAdjustConfig)
    
    # Global
    alpha_weight: float = 1.0         # Per-axis mix for alpha
    beta_weight: float = 1.0          # Per-axis mix for beta
    volume: float = 1.0               # Output volume (0.0-1.0)
    log_level: str = "INFO"           # Logging level (DEBUG/INFO/WARNING/ERROR)
    report_generation_enabled: bool = True    # Master toggle for writing local reports
    privacy_notice_seen: bool = False         # First-run privacy/beta notice has been acknowledged
    app_run_count: int = 0                    # Number of app launches
    report_email_reminder_shown: bool = False # Prevent repeat reminder popup once shown
    report_email_reminder_run: int = 10       # Show reminder when run count reaches this value


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


def migrate_config(config: Config, loaded_version) -> None:
    """Upgrade older config structures to the current schema.
    Adds defaults for newly introduced fields and bumps version."""
    try:
        version = int(loaded_version) if loaded_version is not None else 0
    except Exception:
        version = 0

    if version < 1:
        if getattr(config.stroke, 'noise_burst_magnitude', 1.0) in (None, 0):
            config.stroke.noise_burst_magnitude = 1.0
        if getattr(config.stroke, 'noise_burst_scale', None) is None:
            config.stroke.noise_burst_scale = 0.35

        if getattr(config.stroke, 'downbeat_jitter_vector_percent', None) is None:
            config.stroke.downbeat_jitter_vector_percent = 50.0
        if getattr(config.stroke, 'bass_jitter_speed_influence_percent', None) is None:
            config.stroke.bass_jitter_speed_influence_percent = 100.0
        if getattr(config.stroke, 'bass_jitter_size_influence_percent', None) is None:
            config.stroke.bass_jitter_size_influence_percent = 0.0

        if getattr(config.device_limits, 'p0_c0_sending_enabled', True) is None:
            config.device_limits.p0_c0_sending_enabled = True
        if getattr(config.device_limits, 'dont_show_on_startup', False) is None:
            config.device_limits.dont_show_on_startup = False
        if getattr(config.device_limits, 'prompted', False) is None:
            config.device_limits.prompted = False
        if getattr(config.device_limits, 'dry_run', False) is None:
            config.device_limits.dry_run = False

    if getattr(config, 'report_generation_enabled', True) is None:
        config.report_generation_enabled = True
    if getattr(config, 'privacy_notice_seen', False) is None:
        config.privacy_notice_seen = False
    if getattr(config, 'app_run_count', 0) is None:
        config.app_run_count = 0
    if getattr(config, 'report_email_reminder_shown', False) is None:
        config.report_email_reminder_shown = False
    if getattr(config, 'report_email_reminder_run', 10) is None:
        config.report_email_reminder_run = 10

    # Always clamp safety range for downbeat jitter blend
    try:
        value = float(getattr(config.stroke, 'downbeat_jitter_vector_percent', 50.0))
    except Exception:
        value = 50.0
    config.stroke.downbeat_jitter_vector_percent = max(0.0, min(100.0, value))

    try:
        speed_inf = float(getattr(config.stroke, 'bass_jitter_speed_influence_percent', 100.0))
    except Exception:
        speed_inf = 100.0
    config.stroke.bass_jitter_speed_influence_percent = max(0.0, min(200.0, speed_inf))

    try:
        size_inf = float(getattr(config.stroke, 'bass_jitter_size_influence_percent', 0.0))
    except Exception:
        size_inf = 0.0
    config.stroke.bass_jitter_size_influence_percent = max(0.0, min(200.0, size_inf))

    try:
        burst_scale = float(getattr(config.stroke, 'noise_burst_scale', 0.35))
    except Exception:
        burst_scale = 0.35
    config.stroke.noise_burst_scale = max(0.0, min(0.5, burst_scale))

    config.version = CURRENT_CONFIG_VERSION


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
