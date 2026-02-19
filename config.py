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
    octave_target_bias_confidence_max: float = 0.35  # Only use target-BPM hint for octave disambiguation below this confidence
    target_bps_lock_gate_enabled: bool = True        # Disable target-BPS metric adjustments when metronome lock is confident
    target_bps_lock_gate_acf_conf: float = 0.40      # Minimum ACF confidence to consider lock-gating target-BPS metric
    target_bps_lock_gate_downbeats: int = 1          # Minimum consecutive downbeat matches to consider lock-gating target-BPS metric

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

    # Teaching/learning runtime adapter (audio -> motion-control suggestions)
    teaching_learning_enabled: bool = True      # Enable adaptive motion suggestions from recent beat-window features
    teaching_learning_strength: float = 0.55    # Blend strength for adaptive suggestions (0=off, 1=full)
    teaching_min_confidence: float = 0.12       # Minimum confidence required before adaptive suggestions apply
    teaching_use_fitted_rules: bool = True      # Enable loading fitted rule model (rule_fit.json) for runtime suggestions
    teaching_rule_fit_path: str = ""            # Optional explicit path to rule_fit.json (empty=auto-discover)
    teaching_no_motion_bias: float = 1.0        # Multiplier for hold-still behavior in quiet/unscripted moments
    teaching_apply_in_circle_mode: bool = False # Keep SIMPLE_CIRCLE behavior legacy by default unless explicitly enabled
    teaching_isolation_mode: bool = True        # Branch-only: suspend selected legacy runtime modifiers while learning drives motion
    teaching_relax_phase1_gates: bool = False   # Enforce dual-band/mid-trigger legacy gates unless explicitly relaxed
    teaching_ignore_traffic_lights: bool = False  # Re-enable traffic-light stroke readiness by default
    tempo_lock_required: bool = True             # Require metronome lock confidence for stroke readiness
    teaching_metronome_relaxed_confidence: float = 0.14  # Metronome-only fallback confidence for relaxed readiness
    teaching_stroke_ready_grace_ms: float = 450.0  # Hold readiness briefly through short confidence dips
    teaching_stroke_finish_beats: int = 4        # When readiness drops, allow this many final beat strokes before idling

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
    silence_threshold: float = 0.04      # Overall amplitude threshold for silence deadzone gate
    silence_close_threshold: float = 0.048  # Overall amplitude threshold to exit silence deadzone gate
    silence_flux_multiplier: float = 0.15  # quiet_flux_thresh = flux_threshold * this (0.01-1.0)
    silence_energy_multiplier: float = 0.7  # quiet_energy_thresh = peak_floor * this (0.1-2.0)
    silence_multiplier_locked: bool = True  # Lock sliders on startup

    # Flux-rise depth factor over 250ms.
    # Note: flux_depth_boost_enabled is legacy/internal (UI toggle removed).
    # It remains for backward compatibility with older saved configs.
    flux_depth_factor: float = 0.0     # 0-5, 0=disabled
    flux_depth_boost_enabled: bool = False

    # Main Controls master combinations (1.0 = neutral)
    combo_size: float = 1.0      # stroke size/fullness/flux-scaling/intensity-curve influence
    combo_power: float = 1.0     # downbeat lock boost/jitter blend/scheduled lead
    combo_depth: float = 1.0     # minimum depth/freq depth/flux-depth behavior
    combo_speed: float = 1.0     # cadence density + min-interval behavior
    combo_texture: float = 1.0   # noise burst + syncopation texture behavior
    combo_reaction: float = 1.0  # gate/strictness/readiness aggressiveness

    # Phase advance per beat (0.0 = only downbeats, 1.0 = every beat does a full circle)
    phase_advance: float = 0.25

    geometry_y_offset: float = 0.50  # Below-center rest Y offset used when intensity is near 0
    geometry_sink_start_intensity: float = 0.25  # Intensity where sink-to-rest lerp begins

    # Beat-type-specific orbital geometry: each type blooms from a different center
    # All park at (0, 0.70) and bloom to max (0, 1.0), but with different radius ranges
    orbit_geometry: dict = field(default_factory=lambda: {
        "downbeat": {"center_y": 0.3, "park_radius": 0.40, "max_radius": 0.70},
        "beat": {"center_y": 0.1, "park_radius": 0.60, "max_radius": 0.90},
        "syncopation": {"center_y": 0.0, "park_radius": 0.70, "max_radius": 1.0},
        "creep": {"center_y": 0.4, "park_radius": 0.30, "max_radius": 0.60},
    })

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
    bass_jitter_size_influence_percent: float = 100.0  # % depth of bass-frequency influence on jitter size (inverted: high bass = small)
    noise_primary_mode: bool = False        # True: noise fires strokes, metronome verifies; False: metronome fires, noise supplements

    # Low-band activity gate for beat-based stroke generation
    # Uses sub_bass + low_mid activity window in StrokeMapper.
    # Beat strokes require low-band mean/fullness thresholds.
    # Downbeats use the same concept with a slightly relaxed threshold multiplier.
    low_band_window_frames: int = 18
    low_band_activity_threshold: float = 0.20
    low_band_fullness_occupancy_threshold: float = 0.62
    low_band_to_high_ratio_min: float = 0.58
    mid_bass_support_enabled: bool = True
    mid_bass_freq_low_hz: float = 200.0
    mid_bass_freq_high_hz: float = 400.0
    mid_bass_activity_threshold: float = 0.035
    mid_bass_occupancy_threshold: float = 0.45
    dual_band_db_gate_enabled: bool = True
    dual_band_sub_bass_db_min: float = -15.0
    dual_band_high_db_min: float = -30.0
    high_tip_fullness_enabled: bool = True
    high_tip_freq_hz: float = 3500.0
    high_tip_freq_low_hz: float = 3500.0
    high_tip_freq_high_hz: float = 16000.0
    high_tip_db_min: float = -28.0
    high_tip_occupancy_threshold: float = 0.50
    block_mid_trigger_range_enabled: bool = True
    block_mid_trigger_low_hz: float = 100.0
    block_mid_trigger_high_hz: float = 2000.0
    overall_amp_fill_gate_enabled: bool = True
    overall_amp_fill_target: float = 0.5
    overall_amp_fill_tolerance: float = 0.5
    downbeat_overall_amp_fill_required: float = 0.75
    beat_overall_amp_fill_required: float = 0.70
    syncopation_overall_amp_fill_required: float = 0.60
    overall_amp_fill_required_scale: float = 0.5
    overall_amp_fill_auto_enabled: bool = True
    overall_amp_fill_auto_target_pass_rate: float = 0.58
    overall_amp_fill_auto_ema_alpha: float = 0.12
    overall_amp_fill_auto_deadband: float = 0.06
    overall_amp_fill_auto_step: float = 0.02
    overall_amp_fill_auto_max_offset: float = 0.35
    overall_amp_fill_auto_min_required: float = 0.05
    overall_amp_fill_auto_max_required: float = 0.98

    # dBFS-based fill gate: Use absolute dBFS instead of relative peak normalization.
    # Provides stable thresholds that don't drift with instantaneous peak changes.
    use_dbfs_fill_gate: bool = True  # True: dBFS mode (stable), False: relative peak mode (legacy)
    downbeat_dbfs_threshold: float = -25.0  # Downbeat: easier threshold (dB below reference)
    beat_dbfs_threshold: float = -30.0      # Beat: moderate threshold (dB below reference)
    syncopation_dbfs_threshold: float = -35.0  # Syncopation: harder threshold (dB below reference)
    dbfs_reference_window_ms: float = 15000.0  # Time window (ms) for tracking max signal reference
    dbfs_reference_decay_rate: float = 0.9995  # Per-frame decay multiplier for reference maximum

    # FFT-bin windows used by overall fill gate for each phase.
    # Values are bin indexes (0..N/2) from current FFT size.
    downbeat_fill_bin_low: int = 0
    downbeat_fill_bin_high: int = 512
    beat_fill_bin_low: int = 0
    beat_fill_bin_high: int = 512
    syncopation_fill_bin_low: int = 0
    syncopation_fill_bin_high: int = 512
    downbeat_low_band_relax: float = 0.85

    # Fill duration gate: require sustained fullness over consecutive frames.
    # Values are frame counts (~16-23ms per frame at effective 43-60fps processing rate).
    # Set to 0 or 1 to disable duration check (instant single-frame decision).
    overall_amp_fill_sustain_frames: int = 3  # Frames of sustained fill required (~50-70ms at 60fps)

    # High-band presence gate for beat/downbeat stroke generation.
    # Requires upper range (mid+high) to be both filled and active.
    # Presence pass:
    #   mean >= threshold AND occupancy >= threshold
    #   AND (delta >= threshold OR variance >= threshold)
    # Final upper gate: presence status
    high_band_gate_enabled: bool = True
    high_band_window_frames: int = 18
    high_band_mean_threshold: float = 0.12
    high_band_floor_threshold: float = 0.06
    high_band_occupancy_threshold: float = 0.55
    high_band_delta_threshold: float = 0.05
    high_band_variance_threshold: float = 0.0010
    high_band_include_mid: bool = True
    downbeat_high_band_relax: float = 0.90

    # Overall full-spectrum quiet guard for beat/downbeat stroke generation.
    # Blocks beat-based strokes only when BOTH spectral flux and peak energy
    # are below these thresholds.
    # Can be bypassed when new amplitude+fill gate is prioritized.
    new_gate_priority_enabled: bool = True
    overall_activity_guard_enabled: bool = True
    overall_low_flux_threshold: float = 0.06
    overall_low_energy_threshold: float = 0.14

    # Post-silence volume ramp: reduce volume after silence/track-change, ramp back up
    post_silence_vol_reduction: float = 0.15  # Fraction to reduce volume by (0.0-0.50, 0.15 = 15%)
    post_silence_ramp_seconds: float = 3.0    # Seconds to ramp volume back to full (1.0-8.0)
    silence_fade_drop_points: int = 10        # Max points (out of 100) fade can lower volume by in runtime fade pass

@dataclass
class JitterConfig:
    """Jitter - micro-circles when no beat detected"""
    enabled: bool = True
    intensity: float = 9.5            # Speed of jitter movement
    size: float = 0.024               # Circle size

    @property
    def amplitude(self) -> float:
        return float(self.size)

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        self.size = float(value)

@dataclass
class CreepConfig:
    """Creep - very slow movement when idle"""
    enabled: bool = True
    speed: float = 0.25               # Multiplier for creep rotation

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
    base_radius: float = 0.30        # Global idle orbit radius (0.05-1.0)
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
        if getattr(config.stroke, 'noise_burst_magnitude', 1.0) in (None, 0):
            config.stroke.noise_burst_magnitude = 1.0
        if getattr(config.stroke, 'noise_burst_scale', None) is None:
            config.stroke.noise_burst_scale = 0.35

        if getattr(config.stroke, 'downbeat_jitter_vector_percent', None) is None:
            config.stroke.downbeat_jitter_vector_percent = 50.0
        if getattr(config.stroke, 'bass_jitter_speed_influence_percent', None) is None:
            config.stroke.bass_jitter_speed_influence_percent = 100.0
        if getattr(config.stroke, 'bass_jitter_size_influence_percent', None) is None:
            config.stroke.bass_jitter_size_influence_percent = 100.0

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

    try:
        jitter_size = float(getattr(config.jitter, 'size', getattr(config.jitter, 'amplitude', 0.024)))
    except Exception:
        jitter_size = 0.024
    config.jitter.size = max(0.0, min(0.2, jitter_size))
    config.jitter.amplitude = config.jitter.size

    try:
        tip_low = float(getattr(config.stroke, 'high_tip_freq_low_hz', getattr(config.stroke, 'high_tip_freq_hz', 3500.0) or 3500.0))
    except Exception:
        tip_low = 3500.0
    try:
        tip_high = float(getattr(config.stroke, 'high_tip_freq_high_hz', 16000.0) or 16000.0)
    except Exception:
        tip_high = 16000.0
    tip_low = max(100.0, min(22000.0, tip_low))
    tip_high = max(100.0, min(22000.0, tip_high))
    if tip_high <= tip_low:
        tip_high = min(22000.0, tip_low + 1000.0)
    config.stroke.high_tip_freq_low_hz = tip_low
    config.stroke.high_tip_freq_high_hz = tip_high
    config.stroke.high_tip_freq_hz = tip_low

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

