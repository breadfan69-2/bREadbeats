"""
bREadbeats - Audio Engine
Captures system audio and detects beats using spectral flux / peak energy.
Uses pyaudiowpatch for WASAPI loopback capture.
"""

import numpy as np
import pyaudiowpatch as pyaudio
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import time

from logging_utils import log_event
from audio_modules.feature_extractors import (
    compute_bass_dominance,
    compute_offbeat_score,
    compute_teaching_confidence,
    compute_multiband_energies,
    estimate_dominant_frequency,
    positive_spectral_flux,
    rolling_percentile_norm,
    select_primary_band_by_fire_history,
    slice_spectrum_band,
)
from audio_modules.contracts import FeatureFrame, TempoState, TriggerDecision
from audio_modules.audioflux_adapter import AudioFluxAdapter, AudioFluxAdapterConfig
from audio_modules.event_detector import EventDetector, EventDetectorConfig
from audio_modules.tempo_tracker import (
    TempoTracker,
    TempoTrackerConfig,
    build_acf_octave_candidates,
    dedup_window_seconds,
    effective_phase_accept_window_s,
    metronome_phase_error_s,
    reference_bpm_for_onset_filters,
    estimate_onset_bpm_from_times,
    select_acf_octave_candidate,
    within_dedup_window,
)
from audio_modules.telemetry_tuning import TelemetryTuning, TriggerTelemetry

# Scipy for Butterworth bandpass filter
try:
    from scipy.signal import butter, sosfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    log_event("WARN", "AudioEngine", "scipy not found, using FFT-based frequency filtering")

from config import Config, BeatDetectionType


RMS_DB_FLOOR = -120.0


def rms_to_dbfs(rms: float, floor_db: float = RMS_DB_FLOOR) -> float:
    value = max(float(rms), 1e-12)
    return float(np.clip(20.0 * np.log10(value), floor_db, 12.0))


def silence_threshold_to_dbfs(value: float | None, default_linear: float) -> float:
    if value is None:
        return rms_to_dbfs(default_linear)
    numeric = float(value)
    if not np.isfinite(numeric):
        return rms_to_dbfs(default_linear)
    if numeric <= 0.0:
        return float(np.clip(numeric, RMS_DB_FLOOR, 12.0))
    return rms_to_dbfs(float(np.clip(numeric, 0.0, 1.0)))


class ZScorePeakDetector:
    """
    Real-time z-score peak detector for streaming data (Brakel, 2014).
    
    Processes one value at a time. A peak is detected when a value deviates
    from the rolling mean by more than `threshold` standard deviations.
    The `influence` parameter controls how much detected peaks/valleys
    affect the rolling statistics (0 = ignore peaks entirely, 1 = treat
    peaks like normal data).
    
    Used in beat detection to provide an adaptive threshold that automatically
    adjusts to the current audio level - eliminates the need for manual
    peak_floor tuning in many cases.
    """
    __slots__ = ('lag', 'threshold', 'influence', 'buffer', 'filtered',
                 'mean', 'std', 'initialized', '_buf_len')

    def __init__(self, lag: int = 30, threshold: float = 4.0, influence: float = 0.1):
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.buffer: list[float] = []
        self.filtered: list[float] = []
        self.mean: float = 0.0
        self.std: float = 0.0
        self.initialized: bool = False
        self._buf_len: int = 0  # cached len for hot path

    def update(self, value: float) -> int:
        """Feed one value. Returns +1 (peak), -1 (valley), or 0 (normal)."""
        self.buffer.append(value)
        self._buf_len += 1

        if self._buf_len < self.lag:
            self.filtered.append(value)
            return 0

        if not self.initialized:
            window = self.buffer[:self.lag]
            self.mean = float(np.mean(window))
            self.std = float(np.std(window))
            self.filtered.append(value)
            self.initialized = True
            return 0

        deviation = value - self.mean
        if self.std > 1e-10 and abs(deviation) > self.threshold * self.std:
            signal = 1 if deviation > 0 else -1
            filt = self.influence * value + (1.0 - self.influence) * self.filtered[-1]
        else:
            signal = 0
            filt = value

        self.filtered.append(filt)

        # Update rolling stats from filtered window
        window = self.filtered[-self.lag:]
        self.mean = float(np.mean(window))
        self.std = float(np.std(window))

        # Bound memory - keep ~2x lag
        max_keep = self.lag * 2
        if self._buf_len > self.lag * 3:
            self.buffer = self.buffer[-max_keep:]
            self.filtered = self.filtered[-max_keep:]
            self._buf_len = len(self.buffer)

        return signal

    def reset(self):
        """Clear all state for a fresh start."""
        self.buffer.clear()
        self.filtered.clear()
        self.mean = 0.0
        self.std = 0.0
        self.initialized = False
        self._buf_len = 0


@dataclass
class BeatEvent:
    """Represents a detected beat"""
    timestamp: float          # When the beat occurred
    intensity: float          # Strength of the beat (0.0-1.0)
    frequency: float          # Dominant frequency at beat time
    is_beat: bool            # True if this is an actual beat
    spectral_flux: float     # Current spectral flux value
    peak_energy: float       # Current peak energy value
    is_downbeat: bool = False # True if this is a downbeat (strong beat, beat 1)
    bpm: float = 0.0          # Current tempo in beats per minute
    tempo_reset: bool = False # True if tempo/beat counter was reset
    tempo_locked: bool = False  # True if consecutive downbeats match predicted pattern (locked tempo)
    phase_error_ms: float = 0.0  # How far off from predicted downbeat timing (milliseconds)
    beat_band: str = 'sub_bass'   # Which multi-band z-score sub-band is currently primary
    fired_bands: list = field(default_factory=list)      # Which z-score bands actually fired on THIS beat (per-beat, not global)
    metronome_bpm: float = 0.0    # Current internal metronome BPM (for stroke timing)
    acf_confidence: float = 0.0   # ACF peak confidence (0-1, for UI sync indicator)
    is_syncopated: bool = False   # True if an off-beat "and" onset was detected near this beat
    monotonic_timestamp: float = 0.0  # Monotonic timestamp for drift-safe timing
    beat_features: Optional[dict] = None  # Beat-window features for adaptive runtime learning
    raw_rms: float = 0.0
    raw_rms_db: float = RMS_DB_FLOOR

    def __post_init__(self) -> None:
        if (self.raw_rms_db <= RMS_DB_FLOOR) and (self.raw_rms > 0.0):
            self.raw_rms_db = rms_to_dbfs(self.raw_rms)


class AudioEngine:
    def set_zscore_threshold(self, threshold: float):
        """Update the z-score threshold on ALL multi-band detectors at runtime."""
        if hasattr(self, '_zscore_detectors'):
            for det in self._zscore_detectors.values():
                det.threshold = threshold
            log_event("INFO", "MultiBand", "Z-score threshold updated",
                      threshold=f"{threshold:.2f}")

    """
    Engine 1: The Ears
    Captures system audio and detects beats in real-time.
    """
    
    def __init__(
        self,
        config: Config,
        beat_callback: Callable[[BeatEvent], None],
    ):
        self.config = config
        self.beat_callback = beat_callback
        
        # Audio stream (PyAudio)
        self.pyaudio: Optional[Any] = None
        self.stream = None
        self.running = False
        
        # Beat detection state
        self.prev_spectrum: Optional[np.ndarray] = None
        self.peak_envelope = 0.0
        self.flux_history: list[float] = []
        self.energy_history: list[float] = []
        
        # Spectrum data for visualization
        self.spectrum_data: Optional[np.ndarray] = None
        self.spectrum_lock = threading.Lock()
        self.waveform_data: Optional[np.ndarray] = None
        self.waveform_lock = threading.Lock()
        
        # FFT settings (from config with fallback)
        self.fft_size = int(getattr(config.audio, 'fft_size', 1024) or 1024)
        self.hop_size = max(1, self.fft_size // 4)  # Typical hop = 25% of FFT size
        
        # Pre-allocated arrays for FFT optimization
        self._hanning_window: Optional[np.ndarray] = None  # FFT-size window (created on first use)
        self._fft_input_buffer = np.array([], dtype=np.float32)
        self._beat_fft_input_buffer = np.array([], dtype=np.float32)
        self._frame_counter = 0  # For spectrum skip optimization
        self._spectrum_skip_frames = getattr(config.audio, 'spectrum_skip_frames', 2)
        
        # Tempo tracking (based on madmom resonating comb filter concept)
        # Keep recent beat intervals for smooth tempo estimation
        self.beat_intervals: list[float] = []  # In seconds
        self.smoothed_tempo: float = 0.0       # In BPM
        self.last_known_tempo: float = 0.0     # Preserved tempo during silence
        self.tempo_history: list[float] = []   # For visualization
        self.last_beat_time: float = 0.0       # For calculating intervals
        self.beat_times: list[float] = []      # Last 16 beat times for stability
        self.predicted_next_beat: float = 0.0  # Predicted next beat time
        self.predicted_next_beat_mono: float = 0.0  # Predicted next beat (monotonic clock)
        self.beat_position_in_measure: int = 0 # For downbeat tracking (1, 2, 3, 4...)

        # Beat-window teaching feature buffers (runtime-only)
        self._teach_frames: deque = deque(maxlen=8192)  # (mono_time, energy, flux, freq)
        self._teach_last_beat_mono: float = 0.0
        self._teach_history = {
            'energy_mean': deque(maxlen=48),
            'energy_peak': deque(maxlen=48),
            'flux_mean': deque(maxlen=48),
            'flux_peak': deque(maxlen=48),
            'freq_mean': deque(maxlen=48),
            'freq_delta': deque(maxlen=48),
        }
        
        # These are now read from config (with fallback defaults)
        self.tempo_tracking_enabled: bool = config.beat.tempo_tracking_enabled if hasattr(config.beat, 'tempo_tracking_enabled') else True
        self.tempo_timeout_ms: float = config.beat.tempo_timeout_ms if hasattr(config.beat, 'tempo_timeout_ms') else 2000.0
        self.stability_threshold: float = config.beat.stability_threshold if hasattr(config.beat, 'stability_threshold') else 0.15
        self.beats_per_measure: int = config.beat.beats_per_measure if hasattr(config.beat, 'beats_per_measure') else 4
        self.phase_snap_weight: float = config.beat.phase_snap_weight if hasattr(config.beat, 'phase_snap_weight') else 0.3
        
        # Beat stability filtering (TISMIR PLP-inspired)
        # Only commit BPM display when recent intervals have low variance
        self.stable_tempo: float = 0.0         # Last stable BPM (only updates when CV is low)
        self.beat_stability: float = 0.0       # 0.0 = chaotic, 1.0 = perfectly stable
        
        # Downbeat detection (energy-based, StackOverflow/librosa-inspired)
        # Accumulate energy at each measure position to find the strongest = beat 1
        self.beat_energies: list[float] = []   # Track intensity of beats
        self.is_downbeat: bool = False         # True if this beat is a downbeat (strong beat)
        self.measure_energy_accum: list[float] = [0.0] * self.beats_per_measure  # Accumulated energy per position
        self.measure_beat_counts: list[float] = [0.0] * self.beats_per_measure   # How many beats at each position (decayed)
        self.downbeat_position: int = 0        # Which position (0-3) is the downbeat
        self.downbeat_confidence: float = 0.0  # How confident we are in downbeat placement
        
        # Downbeat pattern matching - strict filtering against predicted tempo
        self.pattern_match_tolerance_ms: float = getattr(config.beat, 'pattern_match_tolerance_ms', 100.0)
        self.consecutive_match_threshold: int = getattr(config.beat, 'consecutive_match_threshold', 3)
        self.downbeat_pattern_enabled: bool = getattr(config.beat, 'downbeat_pattern_enabled', True)
        self.consecutive_matching_downbeats: int = 0  # Counter for downbeats matching predicted pattern
        self.last_predicted_downbeat_time: float = 0.0  # When we predicted the downbeat should occur
        self.phase_error_ms: float = 0.0       # How far off from predicted (in ms)
        
        # Butterworth filter state (initialized in start() when sample rate is known)
        self._butter_sos = None                # Filter coefficients (second-order sections)
        self._butter_zi = None                 # Filter state for continuity between frames
        self._use_butterworth = getattr(config.audio, 'use_butterworth', True)
        self._highpass_hz = getattr(config.audio, 'highpass_filter_hz', 30)
        
        # Visualizer toggle
        self._visualizer_enabled = getattr(config.audio, 'visualizer_enabled', True)
        
        # ===== MULTI-BAND Z-SCORE ADAPTIVE PEAK DETECTION =====
        # Instead of a single z-score detector on overall band_energy, we run
        # one detector PER frequency sub-band.  Each frame, every band's energy
        # is fed to its detector.  The band that produces the strongest/most
        # consistent z-score signals wins as the "primary beat source".
        #
        # Solves the user's scenario: hi-hats fire z-score when only cymbals
        # play, but kick drum z-score takes over when bass enters.
        #
        # Band definitions: (name, low_hz, high_hz)
        self._zscore_bands = [
            ('sub_bass',  30,   100),   # kick drum, sub-bass
            ('low_mid',   100,  500),   # bass guitar, toms, low snare
            ('mid',       500,  2000),  # snare body, guitars, vocals
            ('high',      2000, 16000), # hi-hat, cymbals, clicks
        ]
        # One z-score detector per band (same params, independent rolling stats)
        self._zscore_detectors = {
            name: ZScorePeakDetector(lag=30, threshold=2.5, influence=0.05)
            for name, _, _ in self._zscore_bands
        }
        # Per-band energy values (updated every frame in audio callback)
        self._band_energies: dict[str, float] = {name: 0.0 for name, _, _ in self._zscore_bands}
        # Per-band z-score signals (updated every frame: +1, -1, or 0)
        self._band_zscore_signals: dict[str, int] = {name: 0 for name, _, _ in self._zscore_bands}
        # Band confidence: rolling count of z-score fires in last N frames
        self._band_fire_history: dict[str, list[int]] = {name: [] for name, _, _ in self._zscore_bands}
        self._band_confidence_window: int = 60  # ~1 second at 60fps
        # Which band is currently primary (best beat source)
        self._primary_beat_band: str = 'sub_bass'  # default to kick drum
        # Legacy single-detector alias (for any code that references it)
        self._zscore_detector = self._zscore_detectors['sub_bass']
        
        # ===== REAL-TIME METRIC-BASED AUTO-RANGING (NEW SYSTEM) =====
        # Tracks margins and metrics in real-time to drive parameter adjustments
        # No timer cycle - pure feedback-based optimization
        
        # Metric 1: Peak Floor Feedback (Valley-Tracking)
        # peak_floor should sit at the valley level (between beats) so only genuine
        # peaks pass the floor check.  Valley level scales naturally with amplification.
        self._metric_peak_floor_enabled: bool = False  # User toggle
        self._energy_margin_history: list[float] = []  # Last 16 margins (kept for compat)
        self._energy_margin_target_low: float = 0.02   # Fallback zone (legacy)
        self._energy_margin_target_high: float = 0.05
        self._energy_margin_adjustment_step: float = 0.002  # Step size per check
        self._valley_history: list[float] = []          # Recent energy valley values
        self._valley_max_samples: int = 16              # Rolling window size
        self._prev_energy_for_valley: float = 0.0       # Previous energy for slope detection
        self._energy_was_falling: bool = False           # True when energy was decreasing
        
        # Metric 3: Audio Amp Feedback (No Beats -> raise, Excess Beats -> lower)
        self._metric_audio_amp_enabled: bool = False
        self._audio_amp_check_interval_ms: float = 2500.0  # Check every ~2.5s (was 1.1s)
        self._audio_amp_escalate_pct: float = 0.02     # 2% of range per check
        self._last_audio_amp_check: float = 0.0         # Last time we checked
        self._audio_amp_hysteresis_count: int = 0       # Consecutive out-of-zone checks (hysteresis)
        self._metric_response_speed: float = float(getattr(config.auto_adjust, 'metric_response_speed', 1.0))
        
        # ===== PER-METRIC SETTLED STATE TRACKING =====
        # When a metric fires but no adjustment is needed (within target zone),
        # increment its settled counter.  After N consecutive settled checks,
        # the metric is considered SETTLED and stops adjusting.
        # Reset on silence / tempo reset / metric re-enable.
        self._metric_settled_threshold: int = 12     # Consecutive in-zone checks to settle (~30s at 2.5s interval)
        self._metric_hysteresis_required: int = 2    # Require 2 consecutive out-of-zone before adjusting
        self._metric_settled_counts: dict[str, int] = {
            'peak_floor': 0,
            'sensitivity': 0,
            'audio_amp': 0,
        }
        self._metric_settled_flags: dict[str, bool] = {
            'peak_floor': False,
            'sensitivity': False,
            'audio_amp': False,
        }
        
        # ===== ACF AUTO-METRONOME =====
        # Autocorrelation-based tempo estimator + internal metronome clock.
        # Replaces interval-based beat-counting with robust signal-level tempo detection.
        self._acf_metronome_enabled: bool = True         # Master toggle
        # Onset signal buffer (spectral flux values, one per audio callback)
        self._onset_buffer: list[float] = []
        self._onset_buffer_max: int = 260               # ~6 seconds at ~43 fps (44100/1024)
        self._onset_callback_count: int = 0             # For computing effective sample rate
        self._onset_first_time: float = 0.0             # Timestamp of first onset sample
        # Rolling FPS calibration (fix #4: avoid drift over long sessions)
        self._fps_calibration_times: list[float] = []   # Recent callback timestamps
        self._fps_calibration_window: int = 512          # ~12s of callbacks for rolling estimate
        # ACF estimation
        self._acf_interval_ms: float = float(getattr(config.beat, 'acf_interval_ms', 250.0))
        self._last_acf_time: float = 0.0
        self._acf_bpm: float = 0.0                      # Latest raw ACF BPM estimate
        self._acf_bpm_smoothed: float = 0.0             # Exponentially smoothed ACF BPM
        self._acf_confidence: float = 0.0               # Peak prominence (0-1)
        self._acf_onset_fps: float = 43.0               # Effective onset sample rate (calibrated)
        # Internal metronome
        self._metronome_phase: float = 0.0              # Continuous phase (integer crossings = beats)
        self._metronome_beat_count: int = 0             # Total beats since start (for downbeat)
        self._metronome_last_time: float = 0.0
        self._metronome_bpm: float = 0.0                # BPM the metronome is running at
        self._metronome_beat_fired: bool = False         # Did metronome fire a beat THIS frame?
        self._metronome_downbeat_fired: bool = False     # Did metronome fire a downbeat THIS frame?
        self._metronome_last_beat_time: float = 0.0      # When the metronome last ticked a beat
        self._metronome_conf_hold_s: float = 1.5          # Keep metronome coasting through ACF confidence dips (fix #3)
        self._metronome_conf_lost_at: float = 0.0         # Timestamp when ACF confidence dropped below threshold
        # Tempo-lock hysteresis (prevents lock flapping from brief confidence dips)
        self._tempo_lock_hysteresis_locked: bool = False
        self._tempo_lock_enter_conf_base: float = 0.20
        self._tempo_lock_enter_conf_strict: float = 0.35
        self._tempo_lock_exit_conf_base: float = 0.15
        self._tempo_lock_exit_hold_s: float = 0.90
        self._tempo_lock_drop_started_at: float = 0.0
        self._metronome_bpm_alpha_slow: float = float(getattr(config.beat, 'metronome_bpm_alpha_slow', 0.03))
        self._metronome_bpm_alpha_fast: float = float(getattr(config.beat, 'metronome_bpm_alpha_fast', 0.22))
        self._metronome_pll_window: float = float(getattr(config.beat, 'metronome_pll_window', 0.35))
        self._metronome_pll_base_gain: float = float(getattr(config.beat, 'metronome_pll_base_gain', 0.09))
        self._metronome_pll_conf_gain: float = float(getattr(config.beat, 'metronome_pll_conf_gain', 0.08))
        self._tempo_fusion_min_acf_weight: float = float(getattr(config.beat, 'tempo_fusion_min_acf_weight', 0.20))
        self._tempo_fusion_max_acf_weight: float = float(getattr(config.beat, 'tempo_fusion_max_acf_weight', 0.95))
        self._beat_dedup_fraction: float = float(getattr(config.beat, 'beat_dedup_fraction', 0.22))
        self._phase_accept_window_ms: float = float(getattr(config.beat, 'phase_accept_window_ms', 85.0))
        self._phase_accept_low_conf_mult: float = float(getattr(config.beat, 'phase_accept_low_conf_mult', 2.0))
        self._last_accepted_raw_onset_time: float = 0.0
        self._octave_target_bias_confidence_max: float = float(getattr(config.beat, 'octave_target_bias_confidence_max', 0.35))
        self._aggressive_tempo_snap_enabled: bool = bool(getattr(config.beat, 'aggressive_tempo_snap_enabled', False))
        self._aggressive_snap_confidence: float = float(getattr(config.beat, 'aggressive_snap_confidence', 0.55))
        self._aggressive_snap_phase_error_ms: float = float(getattr(config.beat, 'aggressive_snap_phase_error_ms', 35.0))
        self._aggressive_snap_min_matches: int = int(getattr(config.beat, 'aggressive_snap_min_matches', 1))
        self._aggressive_snap_max_bpm_jump_ratio: float = float(getattr(config.beat, 'aggressive_snap_max_bpm_jump_ratio', 0.12))
        # ===== Syncopation / double-stroke detection =====
        # Track raw onset times to detect off-beat ("and") hits between metronome beats
        self._raw_onset_times: list[float] = []          # Recent raw beat detection timestamps
        self._raw_onset_max: int = 16                    # Keep last 16 raw onsets
        self._syncopation_detected: bool = False         # True when an off-beat onset detected this frame
        self._syncopation_window: float = self.config.beat.syncopation_window  # from config
        self._any_band_onset: bool = False               # True if ANY z-score band fired this frame (wider detection)
        self._syncopation_streak: int = 0                # Consecutive beat periods with off-beat onsets
        self._syncopation_had_offbeat: bool = False      # Off-beat onset seen in current beat period
        self._syncopation_confirmed: bool = False        # True after confirmation
        self._syncopation_armed: bool = False            # Armed on first off-beat, fires on second in same period
        self._new_trigger_fusion_enabled: bool = bool(getattr(config.beat, 'new_trigger_fusion_enabled', False))
        self._new_trigger_telemetry_enabled: bool = bool(getattr(config.beat, 'new_trigger_telemetry_enabled', True))
        self._new_trigger_shadow_mode: bool = bool(getattr(config.beat, 'new_trigger_shadow_mode', True))
        self._shadow_telemetry = TelemetryTuning()
        self._last_acf_smoothing_tag: str = "none"
        self._event_detector = EventDetector(
            EventDetectorConfig(
                enabled=True,
                refractory_ms=float(getattr(config.beat, 'beat_refractory_ms', 170.0)),
                bus_refractory_ms=float(getattr(config.beat, 'trigger_bus_refractory_ms', 170.0)),
                bus_arm_threshold=float(getattr(config.beat, 'trigger_bus_arm_threshold', 0.58)),
                bus_release_threshold=float(getattr(config.beat, 'trigger_bus_release_threshold', 0.42)),
                bus_sustain_frames=int(getattr(config.beat, 'trigger_bus_sustain_frames', 2)),
                w_bus_sub=float(getattr(config.beat, 'trigger_bus_weight_sub_bass', 0.36)),
                w_bus_low=float(getattr(config.beat, 'trigger_bus_weight_low_mid', 0.30)),
                w_bus_mid=float(getattr(config.beat, 'trigger_bus_weight_mid', 0.20)),
                w_bus_high=float(getattr(config.beat, 'trigger_bus_weight_high', 0.14)),
                bus_mask_floor=float(getattr(config.beat, 'trigger_bus_mask_floor', 0.35)),
                bass_dominance_weighting_enabled=bool(getattr(config.beat, 'bass_dominance_weighting_enabled', False)),
                transient_classification_enabled=bool(getattr(config.beat, 'transient_classification_enabled', False)),
            )
        )
        self._audioflux_adapter = AudioFluxAdapter(
            sample_rate=self.config.audio.sample_rate,
            config=AudioFluxAdapterConfig(
                enabled=bool(getattr(config.beat, 'audioflux_enabled', False)),
                frame_stride=int(getattr(config.beat, 'audioflux_frame_stride', 2)),
                fft_size=int(getattr(config.beat, 'audioflux_fft_size', 1024)),
                emit_onset_confidence=bool(getattr(config.beat, 'audioflux_emit_onset_confidence', True)),
            ),
        )
        self._shadow_prev_band_energy: float = 0.0
        self._shadow_prev_flux: float = 0.0
        self._last_peak_ref: float = 1e-6
        self._last_flux_ref: float = 1e-6

        self._session_started_at: float = 0.0
        self._session_frame_count: int = 0
        self._session_raw_rms_db_min: float | None = None
        self._session_raw_rms_db_max: float | None = None
        self._session_band_energy_min: float | None = None
        self._session_band_energy_max: float | None = None
        self._session_flux_min: float | None = None
        self._session_flux_max: float | None = None
        self._session_raw_rms_db_sum: float = 0.0
        self._session_band_energy_sum: float = 0.0
        self._session_flux_sum: float = 0.0
        self._session_sample_times: list[float] = []
        self._session_flux_samples: list[float] = []
        self._session_peak_samples: list[float] = []
        self._session_trough_samples: list[float] = []
        self._tempo_tracker = TempoTracker(TempoTrackerConfig(enabled=True))

    def _sync_tempo_tracker_state(self, tempo_locked: bool, is_downbeat: bool) -> None:
        beat_phase = float(self._metronome_phase % 1.0) if self._metronome_bpm > 0 else 0.0
        self._tempo_tracker.sync_runtime_state(
            metronome_bpm=self._metronome_bpm,
            acf_confidence=self._acf_confidence,
            tempo_locked=tempo_locked,
            phase_error_ms=self.phase_error_ms,
            is_downbeat=is_downbeat,
            beat_phase=beat_phase,
        )

    def _compute_tempo_lock_state(self, acf_confidence: float, downbeat_matches: int, now: float) -> bool:
        """Return tempo lock with confidence hysteresis.

        Enter lock quickly when confidence is high enough; unlock only after
        sustained low confidence, so short dips do not flap the lock state.
        """
        conf = float(np.clip(acf_confidence, 0.0, 1.0))
        has_match = int(downbeat_matches) >= 1

        if not self._tempo_lock_hysteresis_locked:
            enters = bool(
                conf >= self._tempo_lock_enter_conf_base
                and (has_match or conf >= self._tempo_lock_enter_conf_strict)
            )
            if enters:
                self._tempo_lock_hysteresis_locked = True
                self._tempo_lock_drop_started_at = 0.0
            return self._tempo_lock_hysteresis_locked

        # Locked path: only unlock after confidence remains low for hold duration.
        if conf <= self._tempo_lock_exit_conf_base:
            if self._tempo_lock_drop_started_at <= 0.0:
                self._tempo_lock_drop_started_at = float(now)
            elif (float(now) - self._tempo_lock_drop_started_at) >= self._tempo_lock_exit_hold_s:
                self._tempo_lock_hysteresis_locked = False
                self._tempo_lock_drop_started_at = 0.0
        else:
            self._tempo_lock_drop_started_at = 0.0

        return self._tempo_lock_hysteresis_locked

    def _reset_session_stats(self) -> None:
        self._session_started_at = time.time()
        self._session_frame_count = 0
        self._session_raw_rms_db_min = None
        self._session_raw_rms_db_max = None
        self._session_band_energy_min = None
        self._session_band_energy_max = None
        self._session_flux_min = None
        self._session_flux_max = None
        self._session_raw_rms_db_sum = 0.0
        self._session_band_energy_sum = 0.0
        self._session_flux_sum = 0.0
        self._session_sample_times = []
        self._session_flux_samples = []
        self._session_peak_samples = []
        self._session_trough_samples = []
        self._shadow_telemetry.reset()
        self._event_detector.reset()
        self._audioflux_adapter.reset()
        self._shadow_prev_band_energy = 0.0
        self._shadow_prev_flux = 0.0
        self._last_peak_ref = 1e-6
        self._last_flux_ref = 1e-6
        self._rolling_peak_energy = deque(maxlen=430)
        self._rolling_peak_flux = deque(maxlen=430)

    def _build_shadow_feature_frame(
        self,
        band_energy: float,
        spectral_flux: float,
        sidecar_features: Optional[dict[str, float]] = None,
        silence_veto: bool = False,
    ) -> FeatureFrame:
        # --- Silence bypass: force all norms to zero instantly ---
        # When raw_rms_db is below the silence threshold the entire
        # normalization is bypassed so the downstream pipeline sees
        # true silence with no ghosting from a slowly-decaying ref.
        if silence_veto:
            # Seed refs to a conservatively high floor so the first
            # frames after silence don't over-normalise against a
            # quiet intro.  The ref will drop instantly via the
            # asymmetric slew once the rolling window fills, but the
            # 10% upward cap prevents spikes that precede full volume.
            _SILENCE_EXIT_FLOOR_PEAK = 0.30
            _SILENCE_EXIT_FLOOR_FLUX = 0.30
            self._last_peak_ref = _SILENCE_EXIT_FLOOR_PEAK
            self._last_flux_ref = _SILENCE_EXIT_FLOOR_FLUX
            self._rolling_peak_energy.clear()
            self._rolling_peak_flux.clear()
            self._shadow_prev_band_energy = 0.0
            self._shadow_prev_flux = 0.0
            energy_norm = 0.0
            flux_norm = 0.0
            energy_delta = 0.0
            flux_delta = 0.0
        else:
            # --- Rolling-percentile normalization (10-second window) ---
            # Use the 90th-percentile energy/flux over ~10 seconds instead of
            # the absolute max.  This stops rare loud peaks from squashing the
            # normalization ceiling so that regular beats still carry full weight.
            # A 0.85 headroom multiplier keeps most beats above 1.0 for
            # aggressive motion.
            self._rolling_peak_energy.append(float(band_energy))
            self._rolling_peak_flux.append(float(spectral_flux))
            _HEADROOM = 0.92
            candidate_peak = max(1e-6, float(np.percentile(list(self._rolling_peak_energy), 95)) * _HEADROOM)
            candidate_flux = max(1e-6, float(np.percentile(list(self._rolling_peak_flux), 95)) * _HEADROOM)

            # Asymmetric slew limiter:
            #  - Upward:  cap growth to 10% per frame (fast ramp on volume spikes)
            #  - Downward: drop instantly to the candidate value
            # Skip the cap when the ref is still at seed epsilon (first real
            # frame after silence/reset) so music start isn't starved.
            #
            # Post-silence grace period: while the rolling window is still
            # short (< _RAMP_WINDOW samples, ~1 s) downward drops are also
            # capped at 5% per frame.  This stops the ref from chasing a
            # single quiet sample and over-normalising a soft intro.
            _SLEW_SEED = 1e-5  # anything at or below this is "unseeded"
            _RAMP_WINDOW = 50  # symmetric slew while window is sparse

            if self._last_peak_ref <= _SLEW_SEED:
                peak_ref = candidate_peak
            elif candidate_peak > self._last_peak_ref:
                peak_ref = min(candidate_peak, self._last_peak_ref * 1.10)
            elif len(self._rolling_peak_energy) < _RAMP_WINDOW:
                peak_ref = max(candidate_peak, self._last_peak_ref * 0.95)
            else:
                peak_ref = candidate_peak  # instant drop

            if self._last_flux_ref <= _SLEW_SEED:
                flux_ref = candidate_flux
            elif candidate_flux > self._last_flux_ref:
                flux_ref = min(candidate_flux, self._last_flux_ref * 1.10)
            elif len(self._rolling_peak_flux) < _RAMP_WINDOW:
                flux_ref = max(candidate_flux, self._last_flux_ref * 0.95)
            else:
                flux_ref = candidate_flux  # instant drop

            self._last_peak_ref = max(1e-6, float(peak_ref))
            self._last_flux_ref = max(1e-6, float(flux_ref))

            energy_norm = float(np.clip(float(band_energy) / peak_ref, 0.0, 1.0))
            flux_norm = float(np.clip(float(spectral_flux) / flux_ref, 0.0, 1.0))
            energy_delta = float(np.clip((float(band_energy) - float(self._shadow_prev_band_energy)) / peak_ref, 0.0, 1.0))
            flux_delta = float(np.clip((float(spectral_flux) - float(self._shadow_prev_flux)) / flux_ref, 0.0, 1.0))

            self._shadow_prev_band_energy = float(band_energy)
            self._shadow_prev_flux = float(spectral_flux)

        sub_bass = float(self._band_energies.get('sub_bass', 0.0))
        low_mid = float(self._band_energies.get('low_mid', 0.0))
        mid = float(self._band_energies.get('mid', 0.0))
        high = float(self._band_energies.get('high', 0.0))
        total = max(1e-9, sub_bass + low_mid + mid + high)

        hfc_proxy = float(np.clip(high / total, 0.0, 1.0))
        band_sub = float(np.clip(sub_bass / total, 0.0, 1.0))
        band_low = float(np.clip(low_mid / total, 0.0, 1.0))
        band_mid = float(np.clip(mid / total, 0.0, 1.0))
        band_high = float(np.clip(high / total, 0.0, 1.0))
        bass_dominance = compute_bass_dominance(sub_bass, low_mid, mid, high)

        af_entropy = None
        af_flatness = None
        af_hfc = None
        af_novelty = None
        af_rms = None
        af_onset_conf = None
        if sidecar_features:
            entropy_value = sidecar_features.get('af_entropy')
            flatness_value = sidecar_features.get('af_flatness')
            hfc_value = sidecar_features.get('af_hfc')
            novelty_value = sidecar_features.get('af_novelty')
            rms_value = sidecar_features.get('af_rms')
            onset_conf_value = sidecar_features.get('af_onset_conf')

            af_entropy = float(entropy_value) if entropy_value is not None else None
            af_flatness = float(flatness_value) if flatness_value is not None else None
            af_hfc = float(hfc_value) if hfc_value is not None else None
            af_novelty = float(novelty_value) if novelty_value is not None else None
            af_rms = float(rms_value) if rms_value is not None else None
            af_onset_conf = float(onset_conf_value) if onset_conf_value is not None else None

        return FeatureFrame(
            flux_norm=flux_norm,
            energy_norm=energy_norm,
            energy_delta=energy_delta,
            flux_delta=flux_delta,
            hfc_proxy=hfc_proxy,
            sub_bass=band_sub,
            low_mid=band_low,
            mid=band_mid,
            high=band_high,
            bass_dominance=bass_dominance,
            af_entropy=af_entropy,
            af_flatness=af_flatness,
            af_hfc=af_hfc,
            af_novelty=af_novelty,
            af_rms=af_rms,
            af_onset_conf=af_onset_conf,
        )

    def _record_shadow_telemetry(
        self,
        *,
        legacy_fire: bool,
        current_time: float,
        decision: TriggerDecision,
        frontend_ms: float = 0.0,
        tempo_ms: float = 0.0,
        detector_ms: float = 0.0,
        sidecar_ms: float = 0.0,
    ) -> None:
        if not self._new_trigger_telemetry_enabled:
            return

        new_fire = bool(decision.is_beat_candidate)
        self._shadow_telemetry.record(
            TriggerTelemetry(
                legacy_fire=bool(legacy_fire),
                new_fire=bool(new_fire),
                beat_score=float(decision.beat_score),
                cue_flux=float(decision.c_flux),
                cue_band_spike=float(decision.c_band_spike),
                cue_energy_delta=float(decision.c_energy_delta),
                cue_phase_align=float(decision.c_phase_align),
                cue_sidecar=float(decision.c_sidecar),
                frontend_ms=float(frontend_ms),
                tempo_ms=float(tempo_ms),
                detector_ms=float(detector_ms),
                sidecar_ms=float(sidecar_ms),
                bus_raw_scores=dict(decision.bus_raw_scores),
                bus_masked_scores=dict(decision.bus_masked_scores),
                bus_pass=dict(decision.bus_pass),
                bus_reason_codes=dict(decision.bus_reason_codes),
                acf_bpm=float(self._acf_bpm_smoothed),
                acf_confidence=float(self._acf_confidence),
                phase_error_ms=float(self.phase_error_ms),
                smoothing_tag=self._last_acf_smoothing_tag,
                wall_time=float(current_time),
            )
        )

    def _reference_bpm_for_onset_filters(self) -> float:
        return reference_bpm_for_onset_filters(
            self._metronome_bpm,
            self._acf_bpm_smoothed,
            self.smoothed_tempo,
        )

    def _effective_phase_accept_window_s(self) -> float:
        return effective_phase_accept_window_s(
            self._phase_accept_window_ms,
            self._phase_accept_low_conf_mult,
            self._acf_confidence,
        )

    def _is_raw_onset_acceptable(self, now: float) -> bool:
        bpm_ref = self._reference_bpm_for_onset_filters()
        dedup_window_s = dedup_window_seconds(bpm_ref, self._beat_dedup_fraction, default_window_s=0.10)

        if within_dedup_window(self._last_accepted_raw_onset_time, now, dedup_window_s):
            return False

        if self._metronome_bpm > 0:
            phase_error_s = metronome_phase_error_s(self._metronome_phase, self._metronome_bpm)
            if phase_error_s > self._effective_phase_accept_window_s():
                return False

        return True

    def _update_session_stats(
        self,
        raw_rms_db: float,
        band_energy: float,
        spectral_flux: float,
        peak_level: float,
        sample_time: float,
    ) -> None:
        self._session_frame_count += 1
        self._session_raw_rms_db_sum += raw_rms_db
        self._session_band_energy_sum += band_energy
        self._session_flux_sum += spectral_flux
        self._session_sample_times.append(sample_time)
        self._session_flux_samples.append(spectral_flux)
        self._session_peak_samples.append(peak_level)
        self._session_trough_samples.append(band_energy)
        if self._session_raw_rms_db_min is None or raw_rms_db < self._session_raw_rms_db_min:
            self._session_raw_rms_db_min = raw_rms_db
        if self._session_raw_rms_db_max is None or raw_rms_db > self._session_raw_rms_db_max:
            self._session_raw_rms_db_max = raw_rms_db
        if self._session_band_energy_min is None or band_energy < self._session_band_energy_min:
            self._session_band_energy_min = band_energy
        if self._session_band_energy_max is None or band_energy > self._session_band_energy_max:
            self._session_band_energy_max = band_energy
        if self._session_flux_min is None or spectral_flux < self._session_flux_min:
            self._session_flux_min = spectral_flux
        if self._session_flux_max is None or spectral_flux > self._session_flux_max:
            self._session_flux_max = spectral_flux

    def _compute_persistence_stats(
        self,
        values: list[float],
        sample_times: list[float],
        threshold: float,
        is_high: bool,
    ) -> dict[str, float]:
        if len(values) < 2 or len(sample_times) < 2:
            return {
                "total_s": 0.0,
                "episode_count": 0.0,
                "episode_mean_s": 0.0,
                "episode_max_s": 0.0,
            }

        durations: list[float] = []
        current_run_s = 0.0

        for idx in range(1, min(len(values), len(sample_times))):
            dt = max(0.0, sample_times[idx] - sample_times[idx - 1])
            value = values[idx]
            in_state = value >= threshold if is_high else value <= threshold
            if in_state:
                current_run_s += dt
            elif current_run_s > 0.0:
                durations.append(current_run_s)
                current_run_s = 0.0

        if current_run_s > 0.0:
            durations.append(current_run_s)

        if not durations:
            return {
                "total_s": 0.0,
                "episode_count": 0.0,
                "episode_mean_s": 0.0,
                "episode_max_s": 0.0,
            }

        total_s = float(np.sum(durations))
        episode_count = float(len(durations))
        return {
            "total_s": total_s,
            "episode_count": episode_count,
            "episode_mean_s": total_s / episode_count,
            "episode_max_s": float(np.max(durations)),
        }

    def _session_summary_payload(self, elapsed_s: float) -> dict:
        raw_db_min = float(self._session_raw_rms_db_min or RMS_DB_FLOOR)
        raw_db_max = float(self._session_raw_rms_db_max or RMS_DB_FLOOR)
        band_min = float(self._session_band_energy_min or 0.0)
        band_max = float(self._session_band_energy_max or 0.0)
        flux_min = float(self._session_flux_min or 0.0)
        flux_max = float(self._session_flux_max or 0.0)

        frame_count = float(self._session_frame_count)
        raw_db_mean = self._session_raw_rms_db_sum / frame_count
        band_mean = self._session_band_energy_sum / frame_count
        flux_mean = self._session_flux_sum / frame_count

        flux_high_threshold = float(np.percentile(self._session_flux_samples, 90)) if self._session_flux_samples else 0.0
        peak_high_threshold = float(np.percentile(self._session_peak_samples, 90)) if self._session_peak_samples else 0.0
        trough_low_threshold = float(np.percentile(self._session_trough_samples, 10)) if self._session_trough_samples else 0.0

        flux_high = self._compute_persistence_stats(
            self._session_flux_samples,
            self._session_sample_times,
            flux_high_threshold,
            is_high=True,
        )
        peak_high = self._compute_persistence_stats(
            self._session_peak_samples,
            self._session_sample_times,
            peak_high_threshold,
            is_high=True,
        )
        trough_low = self._compute_persistence_stats(
            self._session_trough_samples,
            self._session_sample_times,
            trough_low_threshold,
            is_high=False,
        )

        ended_at = time.time()
        payload = {
            "session_started_at": self._session_started_at,
            "session_ended_at": ended_at,
            "seconds": elapsed_s,
            "frames": self._session_frame_count,
            "raw_rms_db_low": raw_db_min,
            "raw_rms_db_high": raw_db_max,
            "raw_rms_db_mean": raw_db_mean,
            "band_energy_low": band_min,
            "band_energy_high": band_max,
            "band_energy_mean": band_mean,
            "flux_low": flux_min,
            "flux_high": flux_max,
            "flux_mean": flux_mean,
            "flux_high_threshold": flux_high_threshold,
            "peak_high_threshold": peak_high_threshold,
            "trough_low_threshold": trough_low_threshold,
            "flux_high_total_s": flux_high["total_s"],
            "flux_high_episode_count": flux_high["episode_count"],
            "flux_high_episode_mean_s": flux_high["episode_mean_s"],
            "flux_high_episode_max_s": flux_high["episode_max_s"],
            "peak_high_total_s": peak_high["total_s"],
            "peak_high_episode_count": peak_high["episode_count"],
            "peak_high_episode_mean_s": peak_high["episode_mean_s"],
            "peak_high_episode_max_s": peak_high["episode_max_s"],
            "trough_low_total_s": trough_low["total_s"],
            "trough_low_episode_count": trough_low["episode_count"],
            "trough_low_episode_mean_s": trough_low["episode_mean_s"],
            "trough_low_episode_max_s": trough_low["episode_max_s"],
        }
        if self._new_trigger_telemetry_enabled:
            payload.update(self._shadow_telemetry.summary())
        return payload

    def _log_shutdown_summary(self) -> None:
        if self._session_frame_count <= 0:
            return

        elapsed_s = max(0.0, time.time() - self._session_started_at)
        payload = self._session_summary_payload(elapsed_s)

        log_event(
            "INFO",
            "Audio",
            "Shutdown levels summary",
            frames=self._session_frame_count,
            seconds=f"{elapsed_s:.1f}",
            raw_rms_db_min=f"{payload['raw_rms_db_low']:.2f}",
            raw_rms_db_max=f"{payload['raw_rms_db_high']:.2f}",
            raw_rms_db_mean=f"{payload['raw_rms_db_mean']:.2f}",
            raw_rms_db_span=f"{(payload['raw_rms_db_high'] - payload['raw_rms_db_low']):.2f}",
            band_energy_min=f"{payload['band_energy_low']:.6f}",
            band_energy_max=f"{payload['band_energy_high']:.6f}",
            band_energy_mean=f"{payload['band_energy_mean']:.6f}",
            band_energy_span=f"{(payload['band_energy_high'] - payload['band_energy_low']):.6f}",
            flux_min=f"{payload['flux_low']:.4f}",
            flux_max=f"{payload['flux_high']:.4f}",
            flux_mean=f"{payload['flux_mean']:.4f}",
            flux_span=f"{(payload['flux_high'] - payload['flux_low']):.4f}",
            flux_high_total_s=f"{payload['flux_high_total_s']:.3f}",
            peak_high_total_s=f"{payload['peak_high_total_s']:.3f}",
            trough_low_total_s=f"{payload['trough_low_total_s']:.3f}",
        )

    def _init_butterworth_filter(self):
        """Initialize Butterworth bandpass filter for bass detection"""
        if not HAS_SCIPY or not self._use_butterworth:
            return
            
        sr = self.config.audio.sample_rate
        nyquist = sr / 2
        
        # Get frequency band from beat detection config
        low_freq = max(self._highpass_hz, self.config.beat.freq_low)  # At least highpass cutoff
        high_freq = min(self.config.beat.freq_high, nyquist * 0.95)   # Stay below Nyquist
        
        # Normalize frequencies (0-1 where 1 = Nyquist)
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Clamp to valid range
        low_norm = max(0.001, min(0.99, low_norm))
        high_norm = max(low_norm + 0.01, min(0.999, high_norm))
        
        try:
            # 4th order Butterworth bandpass filter
            self._butter_sos = butter(4, [low_norm, high_norm], btype='band', output='sos')
            # Initialize filter state for smooth continuous filtering
            from scipy.signal import sosfilt_zi
            self._butter_zi = sosfilt_zi(self._butter_sos)
            log_event("INFO", "AudioEngine", "Butterworth bandpass initialized", low=f"{low_freq:.0f}", high=f"{high_freq:.0f}")
        except Exception as e:
            log_event("ERROR", "AudioEngine", "Failed to initialize Butterworth filter", error=e)
            self._butter_sos = None
        
    def start(self) -> None:
        """Start audio capture and beat detection"""
        if self.running:
            return
            
        self._reset_session_stats()
        self.running = True
        
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()
        
        # Check if we should use loopback or regular input
        use_loopback = getattr(self.config.audio, 'is_loopback', True)
        device_index = getattr(self.config.audio, 'device_index', None)
        
        try:
            if use_loopback:
                # WASAPI loopback mode (system audio capture)
                self._start_loopback_capture(device_index)
            else:
                # Regular input mode (microphone)
                self._start_input_capture(device_index)
            
            # Initialize Butterworth filter now that sample rate is known
            self._init_butterworth_filter()
                
        except Exception as e:
            log_event("ERROR", "AudioEngine", "Failed to start", error=e)
            self.running = False
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None
    
    def _start_loopback_capture(self, device_index=None):
        """Start WASAPI loopback capture (system audio)"""
        pa = self.pyaudio
        if pa is None:
            raise RuntimeError("PyAudio is not initialized")

        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        
        if device_index is not None:
            # Use specified device - find its loopback version
            device_info = pa.get_device_info_by_index(device_index)
            if not device_info.get("isLoopbackDevice", False):
                # Find the loopback version of this output device
                for loopback in pa.get_loopback_device_info_generator():
                    if device_info["name"] in loopback["name"]:
                        device_info = loopback
                        break
        else:
            # Use default output device's loopback
            device_info = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            if not device_info.get("isLoopbackDevice", False):
                for loopback in pa.get_loopback_device_info_generator():
                    if device_info["name"] in loopback["name"]:
                        device_info = loopback
                        break
        
        log_event("INFO", "AudioEngine", "Using WASAPI loopback", device=device_info['name'])
        log_event("INFO", "AudioEngine", "Loopback format", channels=device_info['maxInputChannels'], sample_rate=int(device_info['defaultSampleRate']))
        
        # Update config with actual sample rate
        self.config.audio.sample_rate = int(device_info['defaultSampleRate'])
        self.config.audio.channels = device_info['maxInputChannels']
        
        # Open stream
        self.stream = pa.open(
            format=pyaudio.paFloat32,
            channels=self.config.audio.channels,
            rate=self.config.audio.sample_rate,
            frames_per_buffer=self.config.audio.buffer_size,
            input=True,
            input_device_index=device_info["index"],
            stream_callback=self._audio_callback_pyaudio
        )
        
        self.stream.start_stream()
        log_event("INFO", "AudioEngine", "WASAPI loopback capture started")
    
    def _start_input_capture(self, device_index):
        """Start regular input capture (microphone)"""
        pa = self.pyaudio
        if pa is None:
            raise RuntimeError("PyAudio is not initialized")

        if device_index is None:
            # Find default input device
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            device_index = wasapi_info.get("defaultInputDevice", 0)
        
        device_info = pa.get_device_info_by_index(device_index)
        
        log_event("INFO", "AudioEngine", "Using input device", device=device_info['name'])
        log_event("INFO", "AudioEngine", "Input format", channels=device_info['maxInputChannels'], sample_rate=int(device_info['defaultSampleRate']))
        
        # Update config with actual sample rate
        self.config.audio.sample_rate = int(device_info['defaultSampleRate'])
        self.config.audio.channels = min(device_info['maxInputChannels'], 2)  # Use up to 2 channels
        
        # Open stream
        self.stream = pa.open(
            format=pyaudio.paFloat32,
            channels=self.config.audio.channels,
            rate=self.config.audio.sample_rate,
            frames_per_buffer=self.config.audio.buffer_size,
            input=True,
            input_device_index=device_index,
            stream_callback=self._audio_callback_pyaudio
        )
        
        self.stream.start_stream()
        log_event("INFO", "AudioEngine", "Input capture started")

        
    def stop(self) -> None:
        """Stop audio capture"""
        self.running = False
        self._log_shutdown_summary()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None
        log_event("INFO", "AudioEngine", "Stopped")
    
    def _audio_callback_pyaudio(self, in_data, frame_count, time_info, status):
        """PyAudio callback - process incoming audio data"""
        if not self.running:
            return (in_data, pyaudio.paContinue)

        callback_started = time.perf_counter()
        
        # Convert bytes to numpy array
        indata = np.frombuffer(in_data, dtype=np.float32)
        indata = indata.reshape(-1, self.config.audio.channels)
        
        # Convert to mono
        if indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata[:, 0]
        
        # Apply Butterworth bandpass filter for beat detection (if available)
        if self._butter_sos is not None and self._butter_zi is not None:
            # Filter with state preservation for continuity
            filtered_mono, self._butter_zi = sosfilt(self._butter_sos, mono, zi=self._butter_zi * mono[0])
            beat_mono = filtered_mono.astype(np.float32)
        else:
            beat_mono = mono
        
        # Frame skip optimization - only update spectrum visualization every N frames
        self._frame_counter += 1
        update_spectrum_viz = (self._frame_counter % self._spectrum_skip_frames == 0) and self._visualizer_enabled
        
        fft_size = max(16, int(self.fft_size))
        hop_size = max(1, int(self.hop_size))

        # Pre-allocate FFT window for configured FFT size
        if self._hanning_window is None or len(self._hanning_window) != fft_size:
            self._hanning_window = np.hanning(fft_size).astype(np.float32)

        mono = np.asarray(mono, dtype=np.float32)
        beat_mono = np.asarray(beat_mono, dtype=np.float32)
        self._fft_input_buffer = np.concatenate((self._fft_input_buffer, mono))
        self._beat_fft_input_buffer = np.concatenate((self._beat_fft_input_buffer, beat_mono))

        fft_scale = 1.0 / max(1e-12, (float(np.sum(self._hanning_window)) / 2.0))
        latest_spectrum: Optional[np.ndarray] = None
        latest_band_energy = 0.0
        latest_spectral_flux = 0.0

        while len(self._fft_input_buffer) >= fft_size and len(self._beat_fft_input_buffer) >= fft_size:
            frame = self._fft_input_buffer[:fft_size]
            beat_frame = self._beat_fft_input_buffer[:fft_size]

            windowed = frame * self._hanning_window
            spectrum = np.abs(np.fft.rfft(windowed)) * fft_scale
            latest_spectrum = spectrum

            if self._butter_sos is not None:
                latest_band_energy = float(np.sqrt(np.mean(beat_frame ** 2))) * self.config.audio.gain
                beat_windowed = beat_frame * self._hanning_window
                beat_spectrum = np.abs(np.fft.rfft(beat_windowed)) * fft_scale
                beat_spectrum = beat_spectrum * self.config.audio.gain
                latest_spectral_flux = self._compute_spectral_flux(beat_spectrum)
            else:
                band_spectrum = self._filter_frequency_band(spectrum)
                band_spectrum = band_spectrum * self.config.audio.gain
                latest_band_energy = float(np.sqrt(np.mean(band_spectrum ** 2))) if len(band_spectrum) > 0 else 0.0
                latest_spectral_flux = self._compute_spectral_flux(band_spectrum)

            self._fft_input_buffer = self._fft_input_buffer[hop_size:]
            self._beat_fft_input_buffer = self._beat_fft_input_buffer[hop_size:]

        if latest_spectrum is None:
            return (in_data, pyaudio.paContinue)

        frontend_ms = (time.perf_counter() - callback_started) * 1000.0

        spectrum = latest_spectrum
        band_energy = latest_band_energy
        spectral_flux = latest_spectral_flux
        
        # Store full spectrum for visualization (only on scheduled frames, if enabled)
        if update_spectrum_viz:
            with self.spectrum_lock:
                self.spectrum_data = spectrum.copy()
            with self.waveform_lock:
                self.waveform_data = mono.astype(np.float32, copy=True)

        raw_rms = np.sqrt(np.mean(mono ** 2))
        raw_rms_db = rms_to_dbfs(raw_rms)
        
        # Note: Audio gain already applied to band_spectrum above, no need to apply again
        
        # ===== MULTI-BAND ENERGY EXTRACTION =====
        # Extract energy per sub-band from the full unfiltered spectrum,
        # feed each to its z-score detector, and track which band fires.
        self._update_multiband_zscore(spectrum)

        # Wider-band onset: did ANY z-score band fire? (for syncopation detection)
        # Respects config: 'any' = any band, or a specific band name
        sync_band = self.config.beat.syncopation_band
        if sync_band == 'any':
            self._any_band_onset = any(s == 1 for s in self._band_zscore_signals.values())
        else:
            self._any_band_onset = self._band_zscore_signals.get(sync_band, 0) == 1
        
        # Debug: print every 20 frames to see levels
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 20 == 0:
            # Log raw audio level too
            full_spectrum_energy = np.sqrt(np.mean(spectrum ** 2)) if len(spectrum) > 0 else 0
            log_event(
                "INFO",
                "Audio",
                "Levels",
                raw_rms_lin=f"{raw_rms:.6f}",
                raw_rms_db=f"{raw_rms_db:.2f}",
                spectrum=f"{full_spectrum_energy:.6f}",
                band_energy=f"{band_energy:.6f}",
                flux=f"{spectral_flux:.4f}",
                peak_env=f"{self.peak_envelope:.6f}"
            )
        
        # Track peak envelope with decay (using band energy)
        decay = self.config.beat.peak_decay
        if band_energy > self.peak_envelope:
            self.peak_envelope = band_energy
        else:
            self.peak_envelope *= decay

        wall_time = time.time()
        current_time = time.perf_counter()

        self._update_session_stats(
            raw_rms_db=raw_rms_db,
            band_energy=band_energy,
            spectral_flux=spectral_flux,
            peak_level=self.peak_envelope,
            sample_time=wall_time,
        )
            
        # Check for tempo timeout.
        # Use a BPM-aware floor so low/halftime metronome periods don't
        # constantly trip resets when one beat arrives slightly late.
        time_since_last_beat = (current_time - self.last_beat_time) * 1000 if self.last_beat_time > 0 else 0
        tempo_timeout_ms = float(self.tempo_timeout_ms)
        bpm_ref = 0.0
        if self._metronome_bpm > 0:
            bpm_ref = float(self._metronome_bpm)
        elif self.smoothed_tempo > 0:
            bpm_ref = float(self.smoothed_tempo)
        elif self.last_known_tempo > 0:
            bpm_ref = float(self.last_known_tempo)
        if bpm_ref > 0:
            beat_period_ms = 60000.0 / max(1.0, bpm_ref)
            # Require nearly two beat periods before declaring timeout.
            tempo_timeout_ms = max(tempo_timeout_ms, beat_period_ms * 1.85)
        
        tempo_reset_flag = False
        if time_since_last_beat > tempo_timeout_ms and len(self.beat_intervals) > 0:
            # Timeout reached - reset tempo tracking but preserve last known tempo
            log_event(
                "INFO",
                "Tempo",
                "No beats detected, resetting tracker",
                idle_ms=f"{time_since_last_beat:.0f}",
                timeout_ms=f"{tempo_timeout_ms:.0f}",
                bpm=f"{self.smoothed_tempo:.1f}"
            )
            self.last_known_tempo = self.smoothed_tempo  # Preserve current tempo
            self.beat_intervals.clear()
            self.beat_times.clear()
            self.beat_position_in_measure = 0
            self.is_downbeat = False
            self._reset_downbeat_pattern()  # Also reset pattern matching when tempo resets
            tempo_reset_flag = True
        
        # Fixed generous threshold for beat-detection veto — only veto
        # near-digital silence.  The real adaptive silence gate lives in
        # BeatIntelligence.update_silence_deadzone_gate().
        silence_veto_active = bool(raw_rms_db < -96.0)
        if silence_veto_active:
            spectral_flux = 0.0
            self.peak_envelope = 0.0
            self._metronome_beat_fired = False
            self._metronome_downbeat_fired = False

        # Detect beat based on mode (using band energy)
        # Store last flux for flux balance metric
        self._last_spectral_flux = spectral_flux
        
        tempo_started = time.perf_counter()

        # ===== ACF ONSET BUFFERING =====
        # Fix #1: Don't feed silence-zeroed flux into the ACF buffer —
        # it poisons the autocorrelation and produces 0-BPM readings.
        if not silence_veto_active:
            self._onset_buffer.append(spectral_flux)
            if len(self._onset_buffer) > self._onset_buffer_max:
                self._onset_buffer.pop(0)
        # Fix #4: Rolling FPS calibration (avoids drift over long sessions).
        # Use a sliding window of recent callback timestamps instead of
        # all-time average which drifts with CPU load / buffer changes.
        self._onset_callback_count += 1
        self._fps_calibration_times.append(current_time)
        if len(self._fps_calibration_times) > self._fps_calibration_window:
            self._fps_calibration_times = self._fps_calibration_times[-self._fps_calibration_window:]
        if self._onset_first_time == 0.0:
            self._onset_first_time = current_time
        if len(self._fps_calibration_times) >= 60:
            fps_elapsed = self._fps_calibration_times[-1] - self._fps_calibration_times[0]
            if fps_elapsed > 0:
                self._acf_onset_fps = (len(self._fps_calibration_times) - 1) / fps_elapsed
        
        # Run ACF tempo estimation periodically
        if current_time - self._last_acf_time > self._acf_interval_ms / 1000.0:
            self._last_acf_time = current_time
            self._estimate_tempo_acf()
        
        # Raw beat detection candidate (ownership selected later)
        raw_is_beat = False if silence_veto_active else self._detect_beat(band_energy, spectral_flux)
        
        # Advance internal metronome (pass band_energy for energy-based downbeat detection)
        self._advance_metronome(current_time, band_energy)
        self._predict_next_beat(current_time, wall_time)

        tempo_ms = (time.perf_counter() - tempo_started) * 1000.0

        sidecar_started = time.perf_counter()
        self._audioflux_adapter.push_audio(mono)
        audioflux_features = self._audioflux_adapter.get_latest_features()
        sidecar_ms = (time.perf_counter() - sidecar_started) * 1000.0

        detector_started = time.perf_counter()
        shadow_features = self._build_shadow_feature_frame(
            band_energy,
            spectral_flux,
            sidecar_features=audioflux_features,
            silence_veto=silence_veto_active,
        )
        shadow_tempo = TempoState(
            metronome_bpm=float(self._metronome_bpm),
            acf_confidence=float(self._acf_confidence),
            tempo_locked=False,
            phase_error_ms=float(self.phase_error_ms),
            is_downbeat=False,
            beat_phase=float(self._metronome_phase % 1.0) if self._metronome_bpm > 0 else 0.0,
        )
        shadow_decision = self._event_detector.detect(
            shadow_features,
            shadow_tempo,
            now_mono=current_time,
        )

        raw_acceptable = self._is_raw_onset_acceptable(current_time)
        accepted_raw_is_beat_legacy = bool(raw_is_beat) and bool(raw_acceptable)
        accepted_raw_is_beat_new = bool(shadow_decision.is_beat_candidate) and bool(raw_acceptable)
        fusion_owner_active = bool(self._new_trigger_fusion_enabled and not self._new_trigger_shadow_mode)
        accepted_raw_is_beat = accepted_raw_is_beat_new if fusion_owner_active else accepted_raw_is_beat_legacy
        detector_ms = (time.perf_counter() - detector_started) * 1000.0

        if accepted_raw_is_beat:
            self._last_accepted_raw_onset_time = current_time
            self._raw_onset_times.append(current_time)
            if len(self._raw_onset_times) > self._raw_onset_max:
                self._raw_onset_times.pop(0)
        
        # Phase-lock: nudge metronome when a strong onset is detected near a beat
        if accepted_raw_is_beat and self._metronome_bpm > 0:
            onset_strength = min(1.0, band_energy / max(0.001, self.peak_envelope))
            self._nudge_metronome_phase(onset_strength)
        
        # ===== SYNCOPATION DETECTION =====
        # Detect off-beat ("and") onsets using configurable z-score band(s).
        # Fast reaction: fires on the FIRST off-beat onset if the previous beat
        # period also had one (streak >= 1). For the very first period, arms on
        # first off-beat and fires on the second off-beat in the same period.
        # Drops immediately on first beat period without any off-beat onset.
        self._syncopation_detected = False
        if (self.config.beat.syncopation_enabled
                and self._metronome_bpm > 0
                and self._any_band_onset
                and not self._metronome_beat_fired):
            bpm_limit = self.config.beat.syncopation_bpm_limit
            if self._metronome_bpm <= bpm_limit:
                phase_frac = self._metronome_phase % 1.0
                combo_texture = float(np.clip(float(getattr(self.config.stroke, 'combo_texture', 1.0) or 1.0), -2.0, 3.0))
                if combo_texture >= 1.0:
                    texture_factor = float(1.0 + ((combo_texture - 1.0) / 2.0) * (2.0 - 1.0))
                else:
                    texture_factor = float(1.0 - ((1.0 - combo_texture) / 3.0) * (1.0 - 0.5))
                window = float(np.clip(self.config.beat.syncopation_window * texture_factor, 0.05, 0.45))
                dist_to_half = abs(phase_frac - 0.5)
                if dist_to_half < window:
                    self._syncopation_had_offbeat = True
                    if self._syncopation_streak >= 1:
                        # Previous beat period had off-beats -> fire immediately
                        self._syncopation_detected = True
                        log_event("INFO", "Syncopation", "Off-beat onset detected",
                                  phase=f"{phase_frac:.2f}", bpm=f"{self._metronome_bpm:.1f}")
                    elif self._syncopation_armed:
                        # Second off-beat in same period -> fire (fast first-time reaction)
                        self._syncopation_detected = True
                        self._syncopation_streak = 1  # pre-confirm for next period
                        log_event("INFO", "Syncopation", "Armed -> firing (2nd onset)",
                                  phase=f"{phase_frac:.2f}", bpm=f"{self._metronome_bpm:.1f}")
                    else:
                        # First off-beat onset ever -> arm for second
                        self._syncopation_armed = True

        # Predictive drop-off: if we're past the off-beat window (phase > 0.65)
        # and no off-beat onset was detected this beat period, preemptively
        # reset streak so the NEXT beat won't produce a false syncopation.
        if self._metronome_bpm > 0 and not self._metronome_beat_fired:
            phase_frac = self._metronome_phase % 1.0
            combo_texture = float(np.clip(float(getattr(self.config.stroke, 'combo_texture', 1.0) or 1.0), -2.0, 3.0))
            if combo_texture >= 1.0:
                texture_factor = float(1.0 + ((combo_texture - 1.0) / 2.0) * (2.0 - 1.0))
            else:
                texture_factor = float(1.0 - ((1.0 - combo_texture) / 3.0) * (1.0 - 0.5))
            window = float(np.clip(self.config.beat.syncopation_window * texture_factor, 0.05, 0.45))
            if phase_frac > (0.5 + window) and not self._syncopation_had_offbeat:
                # Past the "and" window with no onset -> pattern broken
                if self._syncopation_streak > 0 or self._syncopation_armed:
                    self._syncopation_streak = 0
                    self._syncopation_confirmed = False
                    self._syncopation_armed = False
                    log_event("INFO", "Syncopation", "Predictive drop-off (no onset in window)")
        
        # Choose beat source: metronome (when running) or raw detection (fallback)
        metronome_owner_active = bool(self._acf_metronome_enabled and self._metronome_bpm > 0)
        if metronome_owner_active:
            legacy_is_beat = self._metronome_beat_fired
            legacy_is_downbeat_flag = self._metronome_downbeat_fired
            current_bpm = self._metronome_bpm
            tempo_is_locked = self._compute_tempo_lock_state(
                acf_confidence=self._acf_confidence,
                downbeat_matches=self.consecutive_matching_downbeats,
                now=current_time,
            )
            # Update last_beat_time for tempo timeout check
            if legacy_is_beat:
                self._metronome_last_beat_time = current_time
        else:
            legacy_is_beat = accepted_raw_is_beat_legacy
            legacy_is_downbeat_flag = self.is_downbeat if legacy_is_beat else False
            # Fix #2: Fall through ACF smoothed BPM before giving up with 0.
            current_bpm = self.smoothed_tempo if self.smoothed_tempo > 0 else (
                self._acf_bpm_smoothed if self._acf_bpm_smoothed > 0 else self.last_known_tempo
            )
            tempo_is_locked = self.consecutive_matching_downbeats >= self.consecutive_match_threshold

        is_beat = legacy_is_beat
        is_downbeat_flag = legacy_is_downbeat_flag
        if (not metronome_owner_active) and fusion_owner_active:
            is_beat = accepted_raw_is_beat_new
            is_downbeat_flag = self.is_downbeat if is_beat else False

        legacy_fire_for_telemetry = bool(legacy_is_beat)

        if silence_veto_active and is_beat:
            log_event(
                "INFO",
                "Audio",
                "[SILENCE VETO] Beat ignored",
                AmpLin=f"{raw_rms:.5f}",
                AmpDb=f"{raw_rms_db:.2f}",
            )
            is_beat = False
            is_downbeat_flag = False

        if silence_veto_active and legacy_fire_for_telemetry:
            legacy_fire_for_telemetry = False

        self._sync_tempo_tracker_state(tempo_is_locked, bool(is_downbeat_flag))
        
        # Estimate dominant frequency in the configured depth band so
        # stroke depth mapping responds directly to depth band selection.
        depth_low = float(getattr(self.config.stroke, 'depth_freq_low', 0.0))
        depth_high = float(getattr(self.config.stroke, 'depth_freq_high', self.config.audio.sample_rate / 2))
        freq = self._estimate_frequency(spectrum, depth_low, depth_high)

        self._teach_frames.append((current_time, float(band_energy), float(spectral_flux), float(freq)))

        beat_features = None
        if is_beat:
            if current_bpm > 0:
                beat_interval_s = 60.0 / max(1e-6, current_bpm)
            elif self._teach_last_beat_mono > 0:
                beat_interval_s = max(0.1, current_time - self._teach_last_beat_mono)
            else:
                beat_interval_s = 0.60
            beat_features = self._compute_teaching_features(
                now_mono=current_time,
                beat_interval_s=beat_interval_s,
                is_downbeat=bool(is_downbeat_flag),
                is_syncopated=bool(self._syncopation_detected),
            )

            transient_enabled = bool(getattr(self.config.beat, 'transient_classification_enabled', False))
            kick_hint = 0.0
            hat_hint = 0.0
            mixed_hint = 0.0
            if transient_enabled:
                fired_now = {name for name, signal in self._band_zscore_signals.items() if signal == 1}
                kick_hint = float(np.clip(shadow_decision.kick_like_conf, 0.0, 1.0))
                hat_hint = float(np.clip(shadow_decision.hat_like_conf, 0.0, 1.0))
                mixed_hint = float(np.clip(shadow_decision.mixed_conf, 0.0, 1.0))
                if any(name in fired_now for name in ("sub_bass", "low_mid")):
                    kick_hint = max(kick_hint, 0.75)
                if "high" in fired_now:
                    hat_hint = max(hat_hint, 0.75)

            beat_features.update({
                'kick_like_conf': kick_hint,
                'hat_like_conf': hat_hint,
                'mixed_conf': mixed_hint,
                'bass_dominance': float(np.clip(shadow_features.bass_dominance, 0.0, 8.0)),
                'new_beat_score': float(np.clip(shadow_decision.beat_score, 0.0, 1.0)),
                'new_raw_onset_conf': float(np.clip(shadow_decision.raw_onset_conf, 0.0, 1.0)),
                'bus_scores': dict(shadow_decision.bus_scores),
                'bus_pass': dict(shadow_decision.bus_pass),
                'bus_reason_codes': dict(shadow_decision.bus_reason_codes),
                'frontend_ms': float(max(0.0, frontend_ms)),
                'tempo_ms': float(max(0.0, tempo_ms)),
                'detector_ms': float(max(0.0, detector_ms)),
                'sidecar_ms': float(max(0.0, sidecar_ms)),
            })
            self._teach_last_beat_mono = current_time
        
        event = BeatEvent(
            timestamp=wall_time,
            intensity=min(1.0, band_energy / max(0.0001, self.peak_envelope)),
            frequency=freq,
            is_beat=is_beat,
            spectral_flux=spectral_flux,
            peak_energy=band_energy,
            is_downbeat=is_downbeat_flag,
            bpm=current_bpm,
            tempo_reset=tempo_reset_flag,
            tempo_locked=tempo_is_locked,
            phase_error_ms=self.phase_error_ms,
            beat_band=self._primary_beat_band,
            fired_bands=[n for n, s in self._band_zscore_signals.items() if s == 1] if is_beat else [],
            metronome_bpm=self._metronome_bpm,
            acf_confidence=self._acf_confidence,
            is_syncopated=self._syncopation_detected,
            monotonic_timestamp=current_time,
            beat_features=beat_features,
            raw_rms=float(raw_rms),
            raw_rms_db=float(raw_rms_db),
        )

        self._record_shadow_telemetry(
            legacy_fire=legacy_fire_for_telemetry,
            current_time=current_time,
            decision=shadow_decision,
            frontend_ms=frontend_ms,
            tempo_ms=tempo_ms,
            detector_ms=detector_ms,
            sidecar_ms=sidecar_ms,
        )
        
        # Notify callback
        self.beat_callback(event)
        
        # Clear downbeat flag after reporting so next beat must be freshly detected
        # This ensures the downbeat light only flashes once per actual downbeat
        if is_beat:
            self.is_downbeat = False
        
        return (in_data, pyaudio.paContinue)
    
    def _filter_frequency_band(self, spectrum: np.ndarray) -> np.ndarray:
        """Filter spectrum to selected frequency band"""
        cfg = self.config.beat
        sr = self.config.audio.sample_rate
        return slice_spectrum_band(
            spectrum,
            sr,
            cfg.freq_low,
            cfg.freq_high,
            fallback_full_if_invalid=True,
        )
    
    def _compute_spectral_flux(self, spectrum: np.ndarray) -> float:
        """Compute spectral flux (change in spectrum)"""
        if self.prev_spectrum is None or len(self.prev_spectrum) != len(spectrum):
            # Reset if size changed (frequency band was adjusted)
            self.prev_spectrum = spectrum.copy()
            return 0.0

        flux = positive_spectral_flux(self.prev_spectrum, spectrum)
        self.prev_spectrum = spectrum.copy()

        return flux * self.config.beat.flux_multiplier

    def _compute_teaching_features(
        self,
        now_mono: float,
        beat_interval_s: float,
        is_downbeat: bool,
        is_syncopated: bool,
    ) -> dict:
        """Build per-beat feature payload for runtime adaptive mapping."""
        if self._teach_last_beat_mono > 0:
            start = self._teach_last_beat_mono
        else:
            start = now_mono - float(np.clip(beat_interval_s, 0.25, 2.0))

        window = [row for row in self._teach_frames if row[0] >= start]
        if not window and self._teach_frames:
            window = [self._teach_frames[-1]]

        if window:
            energies = np.array([row[1] for row in window], dtype=float)
            fluxes = np.array([row[2] for row in window], dtype=float)
            freqs = np.array([row[3] for row in window], dtype=float)
            energy_mean = float(np.mean(energies))
            energy_peak = float(np.max(energies))
            flux_mean = float(np.mean(fluxes))
            flux_peak = float(np.max(fluxes))
            freq_mean = float(np.mean(freqs))
            freq_delta = float(np.max(freqs) - np.min(freqs))
        else:
            energy_mean = energy_peak = 0.0
            flux_mean = flux_peak = 0.0
            freq_mean = freq_delta = 0.0

        energy_norm = rolling_percentile_norm(self._teach_history['energy_mean'], energy_mean)
        flux_norm = rolling_percentile_norm(self._teach_history['flux_mean'], flux_mean)
        pitch_norm = rolling_percentile_norm(self._teach_history['freq_mean'], freq_mean)
        motion_delta_norm = rolling_percentile_norm(self._teach_history['freq_delta'], freq_delta)
        offbeat_score = compute_offbeat_score(is_syncopated, self._syncopation_streak)
        confidence = compute_teaching_confidence(self._acf_confidence, is_downbeat)

        return {
            'energy_mean': energy_mean,
            'energy_peak': energy_peak,
            'flux_mean': flux_mean,
            'flux_peak': flux_peak,
            'freq_mean': freq_mean,
            'freq_delta': freq_delta,
            'energy_norm': energy_norm,
            'flux_norm': flux_norm,
            'pitch_norm': pitch_norm,
            'motion_delta_norm': motion_delta_norm,
            'offbeat_score': offbeat_score,
            'confidence': confidence,
        }

    # ------------------------------------------------------------------
    # Multi-Band Z-Score
    # ------------------------------------------------------------------
    def _update_multiband_zscore(self, spectrum: np.ndarray):
        """Extract per-sub-band energy from the FFT *spectrum*, feed each
        band's z-score detector, update fire history, and select the
        primary beat band (with hysteresis to avoid rapid switching).

        Called once per audio frame from _audio_callback_pyaudio.
        """
        sr = self.config.audio.sample_rate
        n_bins = len(spectrum)
        if n_bins == 0:
            return
        gain = self.config.audio.gain
        band_energies = compute_multiband_energies(spectrum, sr, gain, self._zscore_bands)

        for name, low_hz, high_hz in self._zscore_bands:
            energy = float(band_energies.get(name, 0.0))

            self._band_energies[name] = energy

            # Feed the per-band detector
            signal = self._zscore_detectors[name].update(energy)
            self._band_zscore_signals[name] = signal

            # Append to rolling fire history (1 = fired, 0 = quiet)
            self._band_fire_history[name].append(1 if signal == 1 else 0)
            if len(self._band_fire_history[name]) > self._band_confidence_window:
                self._band_fire_history[name].pop(0)

        # ---- Select primary band (most consistent fires) ----
        best_band, best_score = select_primary_band_by_fire_history(
            self._primary_beat_band,
            self._zscore_bands,
            self._band_fire_history,
            min_samples=10,
        )

        # Hysteresis: only switch if new band is meaningfully better
        if best_band != self._primary_beat_band:
            current_score = sum(self._band_fire_history[self._primary_beat_band])
            if best_score > current_score + 2:          # 2+ extra fires required
                self._primary_beat_band = best_band
                self._zscore_detector = self._zscore_detectors[best_band]  # legacy alias
                log_event("INFO", "MultiBand", "Primary band switched",
                          band=best_band, fires=str(best_score))

    # ------------------------------------------------------------------
    # ACF Tempo Estimator + Internal Metronome
    # ------------------------------------------------------------------

    def _estimate_tempo_acf(self):
        """Estimate tempo via autocorrelation of the onset strength signal.
        Finds the dominant periodic peak in the spectral flux buffer.
        Called every ~500ms from the audio callback."""
        n = len(self._onset_buffer)
        if n < 80:  # Need at least ~1.9 seconds of data
            return

        signal = np.array(self._onset_buffer, dtype=np.float64)
        signal = signal - np.mean(signal)  # Remove DC

        # Autocorrelation via FFT (much faster than np.correlate for long signals)
        n_fft = 1
        while n_fft < 2 * n:
            n_fft *= 2
        fft_sig = np.fft.rfft(signal, n=n_fft)
        acf = np.fft.irfft(fft_sig * np.conj(fft_sig))[:n]

        if acf[0] > 0:
            acf = acf / acf[0]  # Normalize
        else:
            return

        fps = self._acf_onset_fps  # Calibrated onset sample rate

        # Lag range for 55-185 BPM
        min_lag = max(1, int(fps * 60.0 / 185.0))  # Fastest tempo
        max_lag = min(n - 1, int(fps * 60.0 / 55.0))  # Slowest tempo
        if min_lag >= max_lag:
            return

        search = acf[min_lag:max_lag + 1]
        peak_idx = int(np.argmax(search))
        peak_value = float(search[peak_idx])

        if peak_value < 0.08:  # Below noise floor - no clear tempo
            # Fix #5: Decay confidence but keep a floor so the metronome
            # doesn't die from a brief dip.  Floor = 0.05.
            self._acf_confidence = max(0.05, self._acf_confidence * 0.9)
            return

        # Parabolic interpolation for sub-sample precision
        raw_lag = min_lag + peak_idx
        if peak_idx > 0 and peak_idx < len(search) - 1:
            alpha = float(search[peak_idx - 1])
            beta = float(search[peak_idx])
            gamma = float(search[peak_idx + 1])
            denom = alpha - 2.0 * beta + gamma
            if abs(denom) > 1e-10:
                correction = 0.5 * (alpha - gamma) / denom
            else:
                correction = 0.0
            refined_lag = raw_lag + correction
        else:
            refined_lag = float(raw_lag)

        bpm = 60.0 * fps / refined_lag

        # Octave disambiguation: collect candidate tempos at 1x, 2x, and 0.5x
        # and pick the one closest to target BPM (if set), otherwise prefer
        # the faster tempo when the half-period peak is strong enough.
        candidates = build_acf_octave_candidates(
            bpm,
            peak_value,
            raw_lag,
            min_lag,
            max_lag,
            fps,
            acf,
        )

        # Fix #8: Use last stable/smoothed BPM as octave anchor so the
        # ACF doesn't freely jump to half/double tempo.
        target_bpm_hint = 0.0
        if self._acf_bpm_smoothed > 0.0:
            target_bpm_hint = self._acf_bpm_smoothed
        elif self.smoothed_tempo > 0.0:
            target_bpm_hint = self.smoothed_tempo
        bpm, peak_value, octave_mode, ranked_candidates = select_acf_octave_candidate(
            candidates,
            peak_value,
            self._acf_confidence,
            self._octave_target_bias_confidence_max,
            target_bpm_hint=target_bpm_hint,
        )
        if octave_mode == "target-guided" and ranked_candidates is not None:
            log_event("DEBUG", "ACF", "Octave disambig (target-guided)",
                      target=f"{target_bpm_hint:.0f}",
                      chosen=f"{bpm:.1f}",
                      candidates=str([(f"{c[0]:.1f}", f"{c[1]:.2f}") for c in ranked_candidates]))

        # Clamp to sane range
        if bpm < 55 or bpm > 185:
            return

        self._acf_confidence = float(peak_value)
        self._acf_bpm = bpm

        smoothing = self._tempo_tracker.smooth_acf_bpm_with_jump_gating(
            self._acf_bpm_smoothed,
            bpm,
            peak_value,
            target_bpm_hint=target_bpm_hint,
        )
        self._last_acf_smoothing_tag = smoothing.decision_tag
        self._acf_bpm_smoothed = smoothing.smoothed_bpm

        if smoothing.decision_tag == "jump-target-validated":
            log_event("INFO", "ACF", "Tempo jump (target-validated)",
                      bpm=f"{bpm:.1f}", target=f"{target_bpm_hint:.0f}",
                      confidence=f"{peak_value:.3f}")
        elif smoothing.decision_tag == "jump-target-rejected":
            log_event("INFO", "ACF", "Tempo jump REJECTED (farther from target)",
                      bpm=f"{bpm:.1f}", target=f"{target_bpm_hint:.0f}",
                      current=f"{self._acf_bpm_smoothed:.1f}")
        elif smoothing.decision_tag == "jump":
            log_event("INFO", "ACF", "Tempo jump",
                      bpm=f"{bpm:.1f}", confidence=f"{peak_value:.3f}")
        elif smoothing.decision_tag == "initial":
            log_event("INFO", "ACF", "Initial tempo lock",
                      bpm=f"{bpm:.1f}", confidence=f"{peak_value:.3f}",
                      fps=f"{fps:.1f}")

    def _estimate_onset_bpm(self) -> float:
        """Estimate BPM from recent raw onset intervals for fast fallback/fusion."""
        return estimate_onset_bpm_from_times(
            self._raw_onset_times,
            max_points=8,
            min_interval_s=0.15,
            max_interval_s=1.2,
            min_bpm=55.0,
            max_bpm=185.0,
        )

    def _advance_metronome(self, now: float, band_energy: float = 0.0):
        """Advance the internal metronome phase accumulator.
        Fires _metronome_beat_fired / _metronome_downbeat_fired when
        the phase crosses integer boundaries.
        Uses energy-based downbeat detection to identify the real beat 1.
        
        IMPORTANT: When the metronome is active, it OWNS all downbeat state
        (measure_energy_accum, beat_position_in_measure, etc.).
        The raw beat path must NOT touch downbeat state while metronome runs."""
        self._metronome_beat_fired = False
        self._metronome_downbeat_fired = False

        acf_conf = max(0.0, min(1.0, self._acf_confidence))
        onset_bpm = self._estimate_onset_bpm()
        target_bpm = self._tempo_tracker.update_from_acf_inputs(
            acf_confidence=acf_conf,
            onset_bpm=onset_bpm,
            acf_bpm_smoothed=self._acf_bpm_smoothed,
            min_acf_weight=self._tempo_fusion_min_acf_weight,
            max_acf_weight=self._tempo_fusion_max_acf_weight,
        )

        if target_bpm <= 0 or (acf_conf < 0.10 and onset_bpm <= 0):
            if self._metronome_bpm > 0:
                if self._metronome_conf_lost_at <= 0:
                    self._metronome_conf_lost_at = now
                hold_elapsed = now - self._metronome_conf_lost_at
                if hold_elapsed <= self._metronome_conf_hold_s:
                    # Fix #3: Coast at current BPM during confidence dip.
                    # Apply gentle decay so the metronome doesn't snap from
                    # full speed to zero when the hold expires.
                    decay_factor = max(0.0, 1.0 - (hold_elapsed / max(0.01, self._metronome_conf_hold_s)) * 0.15)
                    target_bpm = self._metronome_bpm * decay_factor
                else:
                    self._metronome_bpm = 0.0
                    self._metronome_conf_lost_at = 0.0
                    return
            else:
                self._metronome_bpm = 0.0
                self._metronome_conf_lost_at = 0.0
                return
        else:
            self._metronome_conf_lost_at = 0.0

        # Boot the metronome on first valid tempo
        if self._metronome_bpm <= 0:
            self._metronome_bpm = target_bpm
            self._metronome_last_time = now
            self._metronome_phase = 0.0
            self._metronome_beat_count = 0
            log_event("INFO", "Metronome", "Started",
                      bpm=f"{target_bpm:.1f}")
            return

        smoothing_conf = acf_conf if acf_conf > 0 else (0.20 if onset_bpm > 0 else 0.0)
        aggressive_ready = (
            self._aggressive_tempo_snap_enabled
            and acf_conf >= self._aggressive_snap_confidence
            and abs(self.phase_error_ms) <= self._aggressive_snap_phase_error_ms
            and self.consecutive_matching_downbeats >= self._aggressive_snap_min_matches
            and self._metronome_bpm > 0
        )
        jump_ratio = abs(target_bpm - self._metronome_bpm) / max(1e-6, self._metronome_bpm)
        if aggressive_ready and jump_ratio <= self._aggressive_snap_max_bpm_jump_ratio:
            self._metronome_bpm = target_bpm
        else:
            alpha = self._metronome_bpm_alpha_slow + (
                self._metronome_bpm_alpha_fast - self._metronome_bpm_alpha_slow
            ) * max(0.0, min(1.0, smoothing_conf))
            self._metronome_bpm = (1.0 - alpha) * self._metronome_bpm + alpha * target_bpm

        dt = now - self._metronome_last_time
        self._metronome_last_time = now
        if dt <= 0 or dt > 0.5:  # Skip huge gaps
            return

        self._metronome_phase, crossings = self._tempo_tracker.step_metronome_phase(
            self._metronome_phase,
            self._metronome_bpm,
            dt,
        )

        if crossings > 0:
            self._metronome_beat_fired = True
            self._metronome_beat_count += 1
            bpm = self.beats_per_measure

            # === Energy-based downbeat detection (metronome owns this state) ===
            # Feed metronome beats into energy accumulator to find which
            # measure position has the strongest energy (= real beat 1).
            self.beat_position_in_measure = (self.beat_position_in_measure % bpm) + 1
            pos_idx = self.beat_position_in_measure - 1  # 0-based

            decay = 0.85
            for i in range(bpm):
                self.measure_energy_accum[i] *= decay
            self.measure_energy_accum[pos_idx] += band_energy
            self.measure_beat_counts[pos_idx] += 1

            # Find which position has highest average energy
            avg_energies = []
            for i in range(bpm):
                if self.measure_beat_counts[i] > 0:
                    avg_energies.append(self.measure_energy_accum[i] / max(1.0, self.measure_beat_counts[i]))
                else:
                    avg_energies.append(0.0)

            total_beats = sum(self.measure_beat_counts)
            if total_beats >= bpm * 2:
                strongest_pos = int(np.argmax(avg_energies))
                mean_energy = np.mean(avg_energies) if np.mean(avg_energies) > 0 else 1.0
                self.downbeat_confidence = avg_energies[strongest_pos] / mean_energy
                self.downbeat_position = strongest_pos

            # Downbeat = when current position matches the energy-strongest position
            is_energy_downbeat = (pos_idx == self.downbeat_position) and total_beats >= bpm * 2

            # Apply pattern matching validation if enabled
            # Use METRONOME BPM (not raw smoothed_tempo) for measure interval
            if is_energy_downbeat and self.downbeat_pattern_enabled and self._metronome_bpm > 0:
                pattern_matches = self._validate_downbeat_against_pattern(now, use_bpm=self._metronome_bpm)
                self._metronome_downbeat_fired = pattern_matches
                self.is_downbeat = pattern_matches

                if pattern_matches:
                    self.consecutive_matching_downbeats += 1
                    log_event("INFO", "Downbeat", "Metronome+Energy accepted",
                              position=f"{pos_idx+1}/{bpm}",
                              confidence=f"{self.downbeat_confidence:.2f}",
                              consecutive=f"{self.consecutive_matching_downbeats}/{self.consecutive_match_threshold}",
                              error_ms=f"{self.phase_error_ms:.1f}",
                              energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]")
                    # === SELF-CHECK: Phase correction from downbeat timing ===
                    # If downbeat landed but with phase error, nudge metronome
                    # so next beats land more accurately
                    if abs(self.phase_error_ms) > 10.0:  # Only correct meaningful errors
                        # Convert ms error to phase fraction
                        beat_period_ms = 60000.0 / self._metronome_bpm
                        phase_correction = (self.phase_error_ms / beat_period_ms) * 0.30  # 30% correction
                        phase_correction = max(-0.15, min(0.15, phase_correction))  # Clamp
                        self._metronome_phase += phase_correction
                        log_event("INFO", "Downbeat", "Phase correction from downbeat",
                                  error_ms=f"{self.phase_error_ms:.1f}",
                                  correction=f"{phase_correction:.4f}")
                else:
                    # Don't fully reset on single mismatch - allow recovery
                    self.consecutive_matching_downbeats = max(0, self.consecutive_matching_downbeats - 1)
                    self._metronome_downbeat_fired = False
                    self.is_downbeat = False
                    log_event("INFO", "Downbeat", "Metronome+Energy rejected",
                              position=f"{pos_idx+1}/{bpm}",
                              confidence=f"{self.downbeat_confidence:.2f}",
                              consecutive=f"{self.consecutive_matching_downbeats}/{self.consecutive_match_threshold}",
                              error_ms=f"{self.phase_error_ms:.1f}",
                              energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]")
            else:
                self._metronome_downbeat_fired = is_energy_downbeat
                self.is_downbeat = is_energy_downbeat
                if is_energy_downbeat:
                    log_event("INFO", "Downbeat", "Energy downbeat (metronome)",
                              position=f"{pos_idx+1}/{bpm}",
                              confidence=f"{self.downbeat_confidence:.2f}",
                              energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]")

            # Track syncopation confirmation per beat period
            if self._syncopation_had_offbeat:
                self._syncopation_streak += 1
            else:
                self._syncopation_streak = 0
                self._syncopation_confirmed = False
                self._syncopation_armed = False
            self._syncopation_had_offbeat = False  # reset for next beat period
            if self._syncopation_streak >= 1:
                self._syncopation_confirmed = True

            src = "DB" if self._metronome_downbeat_fired else "bt"
            log_event("INFO", "Metronome", f"Tick [{src}]",
                      beat=f"{((self._metronome_beat_count - 1) % bpm) + 1}/{bpm}",
                      bpm=f"{self._metronome_bpm:.1f}",
                      acf_conf=f"{self._acf_confidence:.2f}")

    def _nudge_metronome_phase(self, onset_strength: float):
        """Phase-lock loop: nudge metronome phase toward nearest beat
        boundary when a strong onset is detected.  Keeps the metronome
        aligned with the actual music."""
        if self._metronome_bpm <= 0:
            return

        phase_frac = self._metronome_phase % 1.0

        # Distance to nearest beat boundary
        if phase_frac < 0.5:
            error = -phase_frac    # Just past last beat -> pull backward
        else:
            error = 1.0 - phase_frac  # Approaching next beat -> push forward

        if abs(error) < self._metronome_pll_window:
            conf = max(0.0, min(1.0, self._acf_confidence))
            gain = self._metronome_pll_base_gain + self._metronome_pll_conf_gain * conf
            error_scale = 0.5 + 0.5 * min(1.0, abs(error) / 0.25)
            correction = error * gain * min(1.0, onset_strength) * error_scale
            correction = max(-0.20, min(0.20, correction))
            self._metronome_phase += correction

    def _reset_acf_metronome(self):
        """Reset ACF estimator and internal metronome."""
        self._onset_buffer.clear()
        self._onset_callback_count = 0
        self._onset_first_time = 0.0
        self._fps_calibration_times.clear()
        self._acf_bpm = 0.0
        self._acf_bpm_smoothed = 0.0
        self._acf_confidence = 0.0
        self._metronome_phase = 0.0
        self._metronome_beat_count = 0
        self._metronome_conf_lost_at = 0.0
        self._tempo_lock_hysteresis_locked = False
        self._tempo_lock_drop_started_at = 0.0
        self._metronome_bpm = 0.0
        self._metronome_beat_fired = False
        self._metronome_downbeat_fired = False
        self._metronome_last_beat_time = 0.0
        log_event("INFO", "ACF", "Metronome reset")

    def _detect_beat(self, energy: float, flux: float) -> bool:
        """Detect if current frame is a beat.
        
        Uses a two-path system:
          Path 1 (classic): peak_floor + sensitivity + rise checks + threshold
          Path 2 (z-score): adaptive rolling-mean detector fires on +1 signal
        
        A beat is detected if EITHER path triggers (after refractory guard).
        Z-score adapts automatically to any audio level, so it catches beats
        that the manual peak_floor setting would miss - and vice-versa.
        """
        cfg = self.config.beat
        
        # Track valley detection (local minima) for peak_floor metric
        # A valley occurs when energy stops falling and starts rising
        if energy > self._prev_energy_for_valley and self._energy_was_falling:
            # Just turned upward -> previous value was a valley
            valley_val = self._prev_energy_for_valley
            if valley_val > 0.001:  # Ignore silence-level valleys
                self._valley_history.append(valley_val)
                if len(self._valley_history) > self._valley_max_samples:
                    self._valley_history.pop(0)
        self._energy_was_falling = energy < self._prev_energy_for_valley
        self._prev_energy_for_valley = energy
        
        # --- Multi-Band Z-Score: use the primary band's signal ---
        # (Band detectors already fed in _update_multiband_zscore during audio callback)
        primary = self._primary_beat_band
        zscore_signal = self._band_zscore_signals.get(primary, 0)
        zscore_peak = (zscore_signal == 1)  # +1 = primary band spiked
        
        # Threshold-based detection
        self.energy_history.append(energy)
        self.flux_history.append(flux)
        
        # Keep limited history
        max_history = 50
        self.energy_history = self.energy_history[-max_history:]
        self.flux_history = self.flux_history[-max_history:]
        
        if len(self.energy_history) < 5:
            return False
        
        # Refractory period - suppress re-triggers inside a short guard window.
        # Uses beat.beat_refractory_ms (tempo detector domain), not stroke.min_interval_ms
        # (stroke scheduler domain), so high-BPM metronome operation is not choked by
        # legacy stroke timing limits.
        if not hasattr(self, '_last_beat_time'):
            self._last_beat_time = 0
        
        current_time = time.perf_counter()
        beat_refractory_ms = float(getattr(self.config.beat, 'beat_refractory_ms', 170.0) or 170.0)
        beat_refractory_ms = float(np.clip(beat_refractory_ms, 80.0, 600.0))

        if self._metronome_bpm > 0:
            beat_period_ms = 60000.0 / max(1.0, float(self._metronome_bpm))
        else:
            beat_period_ms = 60000.0 / max(1.0, float(getattr(self, 'current_bpm', 120.0) or 120.0))

        # Never let refractory exceed ~70% of a beat period; this keeps fast tempos responsive.
        refractory_ms = min(beat_refractory_ms, beat_period_ms * 0.7)
        refractory_s = refractory_ms / 1000.0
        if current_time - self._last_beat_time < refractory_s:
            return False
            
        # Compute adaptive thresholds
        avg_energy = np.mean(self.energy_history)
        avg_flux = np.mean(self.flux_history)
        
        # Sensitivity now works intuitively: higher = more sensitive (lower threshold)
        # sensitivity 0.0 = need 2x average, sensitivity 1.0 = need 1.3x average
        threshold_mult = 2.0 - (cfg.sensitivity * 0.7)  # Range: 2.0 down to 1.3
        energy_threshold = avg_energy * threshold_mult
        flux_threshold = avg_flux * threshold_mult
        
        # --- PATH 1: Classic detection (peak_floor + rise + threshold) ---
        classic_beat = False
        passes_floor = (cfg.peak_floor <= 0) or (energy >= cfg.peak_floor)
        
        if passes_floor:
            # Rise sensitivity check - configurable now
            # rise_sensitivity 0 = disabled, 1.0 = must rise significantly
            passes_rise = True
            if cfg.rise_sensitivity > 0 and len(self.energy_history) >= 2:
                rise = energy - self.energy_history[-2]
                min_rise = avg_energy * cfg.rise_sensitivity * 0.5
                if rise < min_rise:
                    passes_rise = False
            
            if passes_rise:
                if cfg.detection_type == BeatDetectionType.PEAK_ENERGY:
                    classic_beat = energy > energy_threshold
                elif cfg.detection_type == BeatDetectionType.SPECTRAL_FLUX:
                    classic_beat = flux > flux_threshold
                else:  # COMBINED - need EITHER to trigger (more sensitive)
                    classic_beat = (energy > energy_threshold) or (flux > flux_threshold * 1.2)
        
        # --- PATH 2: Multi-Band Z-Score adaptive detection ---
        # The primary band's z-score already fired.  Also check if ANY band
        # fired (secondary bands can catch beats the primary misses during
        # transitions).  Sanity check: overall energy must exceed average.
        any_band_fired = any(s == 1 for s in self._band_zscore_signals.values())
        zscore_beat = (zscore_peak or any_band_fired) and (energy > avg_energy * 1.1)
        
        # --- COMBINE: either path triggers a beat ---
        is_beat = classic_beat or zscore_beat
        
        if is_beat:
            self._last_beat_time = current_time
            self._update_tempo_tracking(current_time, energy)
            src = "Z+C" if (classic_beat and zscore_beat) else ("Z" if zscore_beat else "C")
            # Identify which bands fired for diagnostic logging
            fired_bands = [n for n, s in self._band_zscore_signals.items() if s == 1]
            band_info = f"band={self._primary_beat_band}"
            if fired_bands and zscore_beat:
                band_info += f" fired={','.join(fired_bands)}"
            log_event(
                "INFO",
                "BEAT",
                f"Beat detected [{src}]",
                energy=f"{energy:.4f}",
                threshold=f"{energy_threshold:.4f}",
                flux=f"{flux:.4f}",
                bpm=f"{self.smoothed_tempo:.1f}",
                bands=band_info
            )
        
        return bool(is_beat)
    
    def _update_tempo_tracking(self, current_time: float, energy: float = 0.0):
        """Update tempo estimate with beat-based interval tracking (madmom-inspired)"""
        # Skip if tempo tracking is disabled
        if not self.tempo_tracking_enabled:
            return
            
        # Calculate interval from last beat
        prev_beat_time = self.last_beat_time
        # Always advance last_beat_time so the next interval starts fresh.
        # Without this, an out-of-range rejection causes the interval to grow
        # monotonically on every subsequent call (stuck at 1.1 BPM forever).
        self.last_beat_time = current_time

        if prev_beat_time > 0:
            interval = current_time - prev_beat_time
            
            # Strict tempo acceptance range (no octave correction)
            min_bpm = 60.0
            max_bpm = 180.0
            min_interval = 60.0 / max_bpm  # ~0.333s
            max_interval = 60.0 / min_bpm  # 1.0s
            
            # Calculate what BPM this interval would give
            if interval > 0:
                raw_bpm = 60.0 / interval
                if raw_bpm < min_bpm or raw_bpm > max_bpm:
                    log_event(
                        "INFO",
                        "Tempo",
                        "Tempo out of range",
                        bpm=f"{raw_bpm:.1f}",
                        min_bpm=f"{min_bpm:.1f}",
                        max_bpm=f"{max_bpm:.1f}",
                    )
                    return
            
            # Reject intervals outside the accepted tempo window
            if interval < min_interval or interval > max_interval:
                log_event("INFO", "Tempo", "Interval rejected", interval=f"{interval:.3f}s", bpm=f"{60.0/interval:.1f}")
                return
            if interval > 0.2:
                # Outlier rejection: if interval is way off from average, it might be a false beat
                # Fix #6: Relax rejection when we have few intervals (post-timeout
                # recovery) — allow 0.35x-2.8x so genuine tempo changes aren't blocked.
                if len(self.beat_intervals) > 0:
                    avg_interval = np.mean(self.beat_intervals)
                    if len(self.beat_intervals) <= 3:
                        lo_mult, hi_mult = 0.35, 2.8  # relaxed after timeout/restart
                    else:
                        lo_mult, hi_mult = 0.5, 2.0
                    if interval < (lo_mult * avg_interval) or interval > (hi_mult * avg_interval):
                        log_event("INFO", "Tempo", "Outlier interval rejected", interval=f"{interval:.3f}s", avg=f"{avg_interval:.3f}s")
                        return
                
                # Phase snap: if we have a stable tempo, nudge detected interval toward predicted
                # This helps lock onto tempo even with slightly off-beat detections
                if self.smoothed_tempo > 0 and self.phase_snap_weight > 0 and self.beat_stability > 0.3:
                    predicted_interval = 60.0 / self.smoothed_tempo
                    # Only snap if the detection is reasonably close (within 20% of predicted)
                    if abs(interval - predicted_interval) / predicted_interval < 0.2:
                        old_interval = interval
                        interval = interval * (1 - self.phase_snap_weight) + predicted_interval * self.phase_snap_weight
                        log_event("INFO", "Tempo", "Phase snap", old=f"{old_interval:.3f}s", new=f"{interval:.3f}s", predicted=f"{predicted_interval:.3f}s")
                
                # Add to interval history
                self.beat_intervals.append(interval)
                self.beat_times.append(current_time)
                # Keep only last 16 intervals (provides smooth averaging over ~1 minute)
                if len(self.beat_intervals) > 16:
                    self.beat_intervals.pop(0)
                    self.beat_times.pop(0)
                # Calculate smoothed tempo using weighted average
                # Recent beats get higher weight (madmom approach: prefer recent data)
                weights = np.linspace(0.5, 1.5, len(self.beat_intervals))
                weighted_avg_interval = float(np.average(self.beat_intervals, weights=weights))
                # Convert to BPM
                new_tempo = float(60.0 / weighted_avg_interval) if weighted_avg_interval > 0 else 0.0
                # Apply exponential smoothing for stability (like madmom's tempo state space)
                smoothing_factor = 0.7  # Higher = more smooth (less responsive)
                if self.smoothed_tempo > 0:
                    smoothed_tempo = (smoothing_factor * self.smoothed_tempo) + ((1 - smoothing_factor) * new_tempo)
                    self.smoothed_tempo = float(smoothed_tempo)
                else:
                    # Fix #2: Also seed from ACF if available, so we don't
                    # stay at 0.0 waiting for raw beat intervals.
                    self.smoothed_tempo = float(new_tempo)
                
                # Beat stability metric (TISMIR PLP-inspired)
                # Coefficient of variation of recent intervals: low = stable tempo
                if len(self.beat_intervals) >= 3:
                    intervals_arr = np.array(self.beat_intervals)
                    cv = np.std(intervals_arr) / np.mean(intervals_arr) if np.mean(intervals_arr) > 0 else 1.0
                    # Convert CV to a 0-1 stability score (0 = chaotic, 1 = perfect)
                    self.beat_stability = float(max(0.0, 1.0 - (cv / self.stability_threshold)))
                    
                    # Only commit to stable_tempo when stability is high enough
                    if cv < self.stability_threshold:
                        self.stable_tempo = self.smoothed_tempo
                        log_event("INFO", "Tempo", "Stable BPM committed", bpm=f"{self.stable_tempo:.1f}", cv=f"{cv:.3f}", stability=f"{self.beat_stability:.2f}")
                    else:
                        log_event("INFO", "Tempo", "BPM unstable", bpm=f"{self.smoothed_tempo:.1f}", cv=f"{cv:.3f}", stability=f"{self.beat_stability:.2f}")
                else:
                    self.beat_stability = 0.0
                
                # Update last known tempo
                self.last_known_tempo = self.smoothed_tempo
                
                # Predict next beat time
                self._predict_next_beat(current_time)
                
                # Energy-based downbeat detection (raw/fallback path)
                # ONLY runs when metronome is NOT active - metronome owns the
                # downbeat state when it's running to avoid double-counting
                metronome_active = (self._acf_metronome_enabled and self._metronome_bpm > 0)
                if not metronome_active:
                    # Accumulate energy at each measure position over multiple measures
                    # The position with highest accumulated energy is likely beat 1
                    self.beat_position_in_measure = (self.beat_position_in_measure % self.beats_per_measure) + 1
                    pos_idx = self.beat_position_in_measure - 1  # 0-based index
                    
                    # Accumulate energy with exponential decay (recent measures weighted more)
                    decay = 0.85  # Older measures fade out
                    for i in range(self.beats_per_measure):
                        self.measure_energy_accum[i] *= decay
                    self.measure_energy_accum[pos_idx] += energy
                    self.measure_beat_counts[pos_idx] += 1
                    
                    # Find which position has highest average energy
                    avg_energies = []
                    for i in range(self.beats_per_measure):
                        if self.measure_beat_counts[i] > 0:
                            avg_energies.append(self.measure_energy_accum[i] / max(1.0, self.measure_beat_counts[i]))
                        else:
                            avg_energies.append(0.0)
                    
                    # Need at least 2 full measures of data before trusting
                    total_beats = sum(self.measure_beat_counts)
                    if total_beats >= self.beats_per_measure * 2:
                        strongest_pos = int(np.argmax(avg_energies))
                        # Calculate confidence: ratio of strongest to average
                        mean_energy = np.mean(avg_energies) if np.mean(avg_energies) > 0 else 1.0
                        self.downbeat_confidence = avg_energies[strongest_pos] / mean_energy
                        self.downbeat_position = strongest_pos
                    
                    # Downbeat = when current position matches the strongest position
                    is_energy_downbeat = (pos_idx == self.downbeat_position) and total_beats >= self.beats_per_measure * 2
                    
                    # Apply pattern matching if enabled (use raw BPM)
                    if is_energy_downbeat and self.downbeat_pattern_enabled and self.smoothed_tempo > 0:
                        pattern_matches = self._validate_downbeat_against_pattern(current_time, use_bpm=self.smoothed_tempo)
                        self.is_downbeat = pattern_matches
                        
                        if pattern_matches:
                            self.consecutive_matching_downbeats += 1
                            log_event(
                                "INFO",
                                "Downbeat",
                                "Accepted (raw)",
                                position=f"{pos_idx+1}/{self.beats_per_measure}",
                                confidence=f"{self.downbeat_confidence:.2f}",
                                consecutive=f"{self.consecutive_matching_downbeats}/{self.consecutive_match_threshold}",
                                error_ms=f"{self.phase_error_ms:.1f}",
                                energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]"
                            )
                        else:
                            self.consecutive_matching_downbeats = max(0, self.consecutive_matching_downbeats - 1)
                            log_event(
                                "INFO",
                                "Downbeat",
                                "Rejected (raw)",
                                position=f"{pos_idx+1}/{self.beats_per_measure}",
                                confidence=f"{self.downbeat_confidence:.2f}",
                                error_ms=f"{self.phase_error_ms:.1f}",
                                energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]"
                            )
                    else:
                        self.is_downbeat = is_energy_downbeat
                        if self.is_downbeat:
                            log_event(
                                "INFO",
                                "Downbeat",
                                "Energy downbeat (raw)",
                                position=f"{pos_idx+1}/{self.beats_per_measure}",
                                confidence=f"{self.downbeat_confidence:.2f}",
                                energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]"
                            )
        
        # last_beat_time is now advanced at the top of _update_tempo_tracking
        # so that out-of-range early-returns don't cause stuck intervals.
    
    def _predict_next_beat(self, current_time: float, current_wall_time: float = 0.0):
        """Predict the time of the next beat using metronome when active."""
        wall_time = current_wall_time if current_wall_time > 0 else time.time()
        if self._acf_metronome_enabled and self._metronome_bpm > 0:
            phase_frac = self._metronome_phase % 1.0
            beats_to_next = 1.0 - phase_frac if phase_frac > 1e-9 else 1.0
            predicted_interval = beats_to_next * (60.0 / self._metronome_bpm)
            self.predicted_next_beat_mono = current_time + predicted_interval
            self.predicted_next_beat = wall_time + predicted_interval
            return

        if self.smoothed_tempo > 0:
            predicted_interval = 60.0 / self.smoothed_tempo
            self.predicted_next_beat_mono = current_time + predicted_interval
            self.predicted_next_beat = wall_time + predicted_interval
    
    def _validate_downbeat_against_pattern(self, current_time: float, use_bpm: float = 0.0) -> bool:
        """
        Validate that a detected downbeat matches the predicted tempo pattern within tolerance.
        
        Self-checking sequence:
        1. Metronome predicts beats at steady BPM
        2. Energy accumulator identifies strongest measure position
        3. Pattern matching verifies downbeats land at expected intervals
        4. Phase error feeds back to metronome for timing correction
        
        Args:
            current_time: Time of the detected downbeat (seconds)
            use_bpm: BPM to use for measure interval calculation.
                     When called from metronome path, pass _metronome_bpm.
                     When called from raw path, pass smoothed_tempo.
                     If 0, falls back to smoothed_tempo.
            
        Returns:
            True if downbeat matches predicted pattern, False otherwise
        """
        # Use the correct BPM source depending on which path called us
        active_bpm = use_bpm if use_bpm > 0 else self.smoothed_tempo
        if active_bpm <= 0:
            return False
        
        beat_interval = 60.0 / active_bpm  # Seconds between beats
        measure_interval = beat_interval * self.beats_per_measure  # Seconds per measure
        
        # First few downbeats: establish the predicted pattern
        if self.last_predicted_downbeat_time <= 0:
            # Set up the prediction based on this downbeat
            self.last_predicted_downbeat_time = current_time
            self.consecutive_matching_downbeats = 1
            self.phase_error_ms = 0.0
            return True
        
        # Calculate when we predicted this downbeat should occur
        # Allow matching against multiple future/past measure boundaries
        # (handles cases where a downbeat was missed)
        predicted_time = self.last_predicted_downbeat_time + measure_interval
        
        # If we've drifted far (e.g. missed a measure), find nearest expected downbeat
        time_since_last = current_time - self.last_predicted_downbeat_time
        if time_since_last > 0 and measure_interval > 0:
            measures_elapsed = round(time_since_last / measure_interval)
            if measures_elapsed >= 1:
                predicted_time = self.last_predicted_downbeat_time + measures_elapsed * measure_interval
        
        # Calculate phase error in milliseconds
        self.phase_error_ms = (current_time - predicted_time) * 1000.0
        
        # Use wider tolerance for early matches (still building confidence)
        effective_tolerance = self.pattern_match_tolerance_ms
        if self.consecutive_matching_downbeats < 2:
            effective_tolerance *= 1.5  # 50% wider tolerance for first few
        
        # Check if within tolerance
        if abs(self.phase_error_ms) <= effective_tolerance:
            # Update prediction for next downbeat
            self.last_predicted_downbeat_time = current_time
            return True
        else:
            # Error exceeds tolerance - but if it's close to a measure boundary,
            # update prediction anyway to prevent permanent lock-out
            if abs(self.phase_error_ms) <= effective_tolerance * 2.0:
                # Close but not perfect - update prediction to re-sync
                self.last_predicted_downbeat_time = current_time
            return False
        
    def _reset_downbeat_pattern(self):
        """Reset downbeat pattern matching state (call after temp lock expires or on silence)"""
        self.consecutive_matching_downbeats = 0
        self.last_predicted_downbeat_time = 0.0
        self.phase_error_ms = 0.0
        # Reset metric settled states so they re-hunt after silence/song change
        for key in self._metric_settled_counts:
            self._metric_settled_counts[key] = 0
            self._metric_settled_flags[key] = False
        
    def get_tempo_info(self) -> dict:
        """Get current tempo information for UI display"""
        tempo_state = self._tempo_tracker.get_state()
        reported_metronome_bpm = tempo_state.metronome_bpm if tempo_state.metronome_bpm > 0 else self._metronome_bpm
        reported_acf_confidence = tempo_state.acf_confidence if tempo_state.acf_confidence > 0 else self._acf_confidence
        reported_phase_error_ms = tempo_state.phase_error_ms if abs(tempo_state.phase_error_ms) > 1e-9 else self.phase_error_ms
        reported_is_downbeat = bool(tempo_state.is_downbeat) or bool(self.is_downbeat)

        # Use stable_tempo for display if available, otherwise fall back to smoothed
        display_bpm = self.stable_tempo if self.stable_tempo > 0 else self.smoothed_tempo
        # ACF metronome info (when active, these take priority)
        acf_active = self._acf_metronome_enabled and reported_metronome_bpm > 0
        if acf_active:
            display_bpm = reported_metronome_bpm
            beat_pos = ((self._metronome_beat_count - 1) % self.beats_per_measure) + 1 if self._metronome_beat_count > 0 else 0
        else:
            beat_pos = self.beat_position_in_measure

        return {
            'bpm': display_bpm,
            'raw_bpm': self.smoothed_tempo,
            'stable_bpm': self.stable_tempo,
            'beat_position': beat_pos,
            'is_downbeat': reported_is_downbeat,
            'predicted_next_beat': self.predicted_next_beat,
            'predicted_next_beat_mono': self.predicted_next_beat_mono,
            'interval_count': len(self.beat_intervals),
            'confidence': min(1.0, len(self.beat_intervals) / 4.0),
            'stability': self.beat_stability,
            'consecutive_matching_downbeats': self.consecutive_matching_downbeats,
            'phase_error_ms': reported_phase_error_ms,
            # ACF metronome fields
            'acf_bpm': self._acf_bpm_smoothed,
            'acf_confidence': reported_acf_confidence,
            'acf_active': acf_active,
            'metronome_bpm': reported_metronome_bpm,
        }
            
    def _estimate_frequency(self, spectrum: np.ndarray, low_hz: Optional[float] = None, high_hz: Optional[float] = None) -> float:
        """Estimate dominant frequency from spectrum, optionally within a frequency band."""
        return estimate_dominant_frequency(
            spectrum,
            self.config.audio.sample_rate,
            low_hz,
            high_hz,
        )
        
    def get_spectrum(self) -> Optional[np.ndarray]:
        """Get current spectrum data for visualization"""
        with self.spectrum_lock:
            return self.spectrum_data.copy() if self.spectrum_data is not None else None

    def get_waveform(self) -> Optional[np.ndarray]:
        """Get current waveform frame for visualization."""
        with self.waveform_lock:
            return self.waveform_data.copy() if self.waveform_data is not None else None
    
    # ===== REAL-TIME METRIC FEEDBACK SYSTEM =====
    
    def enable_metric_autoranging(self, metric: str, enable: bool = True):
        """Enable/disable a specific metric-based auto-ranging metric"""
        if metric == 'peak_floor':
            self._metric_peak_floor_enabled = enable
            if enable:
                self._energy_margin_history.clear()
                self._valley_history.clear()
                self._energy_was_falling = False
                self._metric_settled_counts['peak_floor'] = 0
                self._metric_settled_flags['peak_floor'] = False
                log_event("INFO", "MetricAutoRange", "Peak Floor metric enabled (valley-tracking)")
            else:
                log_event("INFO", "MetricAutoRange", "Peak Floor metric disabled")
        elif metric == 'audio_amp':
            self._metric_audio_amp_enabled = enable
            if enable:
                self._last_audio_amp_check = 0.0
                self._metric_settled_counts['audio_amp'] = 0
                self._metric_settled_flags['audio_amp'] = False
                log_event("INFO", "MetricAutoRange", "Audio Amp metric enabled (beat-driven)")
            else:
                log_event("INFO", "MetricAutoRange", "Audio Amp metric disabled")

    
    def compute_energy_margin_feedback(self, band_energy: float, callback=None):
        """
        Compute peak_floor adjustment based on valley tracking.
        
        peak_floor should sit at the average valley level (local minima between beats).
        This naturally scales with amplification since valleys scale with the signal.
        
        Valley = average of recent energy local minima (detected in _detect_beat).
        If peak_floor < valley: raise it (too much noise passes through)
        If peak_floor > valley: lower it (real peaks might be filtered out)
        
        Tolerance band: peak_floor should be within +/-20% of avg valley.
        
        Returns:
            (margin, should_adjust, adjustment_direction)
            adjustment_direction: +1 to raise floor, -1 to lower floor, 0 no change
        """
        if not self._metric_peak_floor_enabled:
            return 0.0, False, 0
        
        # If already settled, don't adjust
        if self._metric_settled_flags.get('peak_floor', False):
            margin = band_energy - self.config.beat.peak_floor
            return margin, False, 0
        
        # Need valley data to work with
        if len(self._valley_history) < 3:
            # Not enough valley data yet - fall back to simple margin check
            margin = band_energy - self.config.beat.peak_floor
            self._energy_margin_history.append(margin)
            if len(self._energy_margin_history) > 16:
                self._energy_margin_history.pop(0)
            return float(np.mean(self._energy_margin_history)) if self._energy_margin_history else margin, False, 0
        
        # Compute target: average valley level
        avg_valley = float(np.mean(self._valley_history))
        current_pf = self.config.beat.peak_floor
        
        # Amplitude proportionality: peak_floor must always be >= 10% of audio_amp
        # This prevents peak_floor from staying absurdly low when gain is cranked up
        amp_floor = self.config.audio.gain * 0.10
        if avg_valley < amp_floor:
            avg_valley = amp_floor  # Use amp-proportional floor as minimum target
        
        # How far is peak_floor from the valley level?
        # Positive = peak_floor above valley, Negative = peak_floor below valley
        error = current_pf - avg_valley
        
        # Track margin history for display
        margin = band_energy - current_pf
        self._energy_margin_history.append(margin)
        if len(self._energy_margin_history) > 16:
            self._energy_margin_history.pop(0)
        
        # Tolerance: peak_floor should be within +/-20% of valley level
        tolerance = avg_valley * 0.20
        
        should_adjust = False
        direction = 0
        
        if error > tolerance:
            # peak_floor too HIGH vs valleys -> lower it so peaks pass through
            should_adjust = True
            direction = -1
        elif error < -tolerance:
            # peak_floor too LOW vs valleys -> raise it to filter noise
            should_adjust = True
            direction = +1
        
        # Scale step size proportional to valley level for amp-agnostic adjustment
        step = max(self._energy_margin_adjustment_step, avg_valley * 0.05)
        step = self._scaled_metric_step(step)
        
        if callback and should_adjust:
            # Decay settled counter instead of hard reset (drop by 3, not to 0)
            self._metric_settled_counts['peak_floor'] = max(0, self._metric_settled_counts.get('peak_floor', 0) - 3)
            callback({
                'metric': 'peak_floor',
                'margin': float(np.mean(self._energy_margin_history)),
                'valley': avg_valley,
                'error': error,
                'adjustment': direction * step,
                'direction': 'raise' if direction > 0 else 'lower'
            })
        elif not should_adjust:
            # In zone - increment settled counter
            self._metric_settled_counts['peak_floor'] = self._metric_settled_counts.get('peak_floor', 0) + 1
            if self._metric_settled_counts['peak_floor'] >= self._effective_metric_settled_threshold():
                self._metric_settled_flags['peak_floor'] = True
                log_event("INFO", "Metric", "Peak Floor SETTLED",
                          valley=f"{avg_valley:.4f}", pf=f"{current_pf:.4f}")
        
        return float(np.mean(self._energy_margin_history)), should_adjust, direction

    def set_metric_response_speed(self, speed: float):
        """Set auto-range response speed (1.0=legacy, >1 faster, <1 slower)."""
        self._metric_response_speed = max(0.5, min(3.0, float(speed)))

    def _effective_metric_speed(self) -> float:
        return max(0.5, min(3.0, self._metric_response_speed))

    def _scaled_metric_interval_s(self, interval_ms: float) -> float:
        return (interval_ms / 1000.0) / self._effective_metric_speed()

    def _scaled_metric_step(self, base_step: float) -> float:
        return base_step * self._effective_metric_speed()

    def _effective_metric_hysteresis_required(self) -> int:
        speed = self._effective_metric_speed()
        if speed <= 1.0:
            return self._metric_hysteresis_required
        return max(1, int(round(self._metric_hysteresis_required / speed)))

    def _effective_metric_settled_threshold(self) -> int:
        speed = self._effective_metric_speed()
        return max(4, int(round(self._metric_settled_threshold / speed)))

    # ===== TIMER-DRIVEN METRIC FEEDBACK (audio_amp) =====
    # These are called from main.py's _update_display timer, NOT from _on_beat,
    # because they need to detect the ABSENCE of beats.

    def get_metric_states(self) -> dict[str, str]:
        """
        Return the current state of each enabled metric.
        States: 'ADJUSTING' (actively hunting) or 'SETTLED' (in zone, stable).
        Only returns entries for enabled metrics.
        """
        states = {}
        if self._metric_peak_floor_enabled:
            states['peak_floor'] = 'SETTLED' if self._metric_settled_flags.get('peak_floor', False) else 'ADJUSTING'

        if self._metric_audio_amp_enabled:
            states['audio_amp'] = 'SETTLED' if self._metric_settled_flags.get('audio_amp', False) else 'ADJUSTING'
        return states

    def compute_audio_amp_feedback(self, now: float, callback=None):
        """
        Timer-driven audio_amp adjustment based on beat presence.
        
        - No beats for >check_interval -> RAISE audio_amp (+2% of range)
        - Excess beats (BPS > 2x target) -> LOWER audio_amp (1% of range, half raise rate)
        - Tracks consecutive in-zone checks for SETTLED state
        - Requires 2 consecutive out-of-zone checks (hysteresis) before adjusting
        
        Called from _update_display (~30fps), but only acts every ~2.5s.
        """
        if not self._metric_audio_amp_enabled:
            return
        
        # Only check every ~2.5s
        if now - self._last_audio_amp_check < self._scaled_metric_interval_s(self._audio_amp_check_interval_ms):
            return
        self._last_audio_amp_check = now
        
        # If already settled, don't adjust
        if self._metric_settled_flags.get('audio_amp', False):
            return
        
        # Get range for percentage calculation
        from config import BEAT_RANGE_LIMITS
        amp_min, amp_max = BEAT_RANGE_LIMITS['audio_amp']
        amp_range = amp_max - amp_min
        step = amp_range * self._audio_amp_escalate_pct  # 2% of range
        step = self._scaled_metric_step(step)
        
        # Check time since last beat
        time_since_beat = now - self.last_beat_time if self.last_beat_time > 0 else float('inf')
        ref_bps = 1.5  # fixed reference; target-BPM behavior disabled
        target_interval = 1.0 / ref_bps
        
        wants_adjustment = False
        if time_since_beat > target_interval * 3.0:
            # No beats detected for 3x expected interval -> wants to RAISE audio_amp
            wants_adjustment = True
        
        # Check for excess beats: if BPS > 2x reference for consecutive checks, LOWER audio_amp
        wants_lower = False
        if self.last_beat_time > 0 and time_since_beat < target_interval:
            # Beats are coming - check if too many using beat_times history
            if len(self.beat_times) >= 2:
                window_dur = self.beat_times[-1] - self.beat_times[0]
                if window_dur > 0:
                    actual_bps = (len(self.beat_times) - 1) / window_dur
                    if actual_bps > ref_bps * 2.0:
                        wants_lower = True
        
        # Hysteresis: require 2 consecutive out-of-zone checks before adjusting
        if wants_adjustment or wants_lower:
            self._audio_amp_hysteresis_count += 1
            if self._audio_amp_hysteresis_count >= self._effective_metric_hysteresis_required():
                # Actually adjust now
                # Decay settled counter instead of hard reset (drop by 3, not to 0)
                self._metric_settled_counts['audio_amp'] = max(0, self._metric_settled_counts.get('audio_amp', 0) - 3)
                self._audio_amp_hysteresis_count = 0
                if wants_lower:
                    # De-escalate: lower at half the raise rate
                    lower_step = step * 0.5
                    if callback:
                        callback({
                            'metric': 'audio_amp',
                            'adjustment': -lower_step,
                            'direction': 'lower',
                            'reason': f'excess BPS > 2x reference (2x confirmed)',
                        })
                elif callback:
                    callback({
                        'metric': 'audio_amp',
                        'adjustment': +step,
                        'direction': 'raise',
                        'reason': f'no beats for {time_since_beat:.1f}s (2x confirmed)',
                    })
        else:
            # In zone - reset hysteresis counter and increment settled
            self._audio_amp_hysteresis_count = 0
            self._metric_settled_counts['audio_amp'] = self._metric_settled_counts.get('audio_amp', 0) + 1
            if self._metric_settled_counts['audio_amp'] >= self._effective_metric_settled_threshold():
                self._metric_settled_flags['audio_amp'] = True
                log_event("INFO", "Metric", "Audio Amp SETTLED",
                          count=f"{self._metric_settled_counts['audio_amp']}",
                          threshold=f"{self._effective_metric_settled_threshold()}")




if __name__ == "__main__":
    from config import Config
    
    def on_beat(event: BeatEvent):
        if event.is_beat:
            log_event("INFO", "BEAT", "Test beat", intensity=f"{event.intensity:.2f}", freq_hz=f"{event.frequency:.0f}")
            
    config = Config()
    engine = AudioEngine(config, on_beat)
    
    log_event("INFO", "AudioEngine", "Standalone run: skipping device enumeration helper")
        
    log_event("INFO", "AudioEngine", "Starting audio capture (Ctrl+C to stop)...")
    engine.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        engine.stop()
