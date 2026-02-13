"""
bREadbeats - Audio Engine
Captures system audio and detects beats using spectral flux / peak energy.
Uses pyaudiowpatch for WASAPI loopback capture.
"""

import numpy as np
import pyaudiowpatch as pyaudio
import threading
from dataclasses import dataclass
from typing import Callable, Optional
import time

from logging_utils import log_event

# Scipy for Butterworth bandpass filter
try:
    from scipy.signal import butter, sosfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    log_event("WARN", "AudioEngine", "scipy not found, using FFT-based frequency filtering")

from config import Config, BeatDetectionType


class ZScorePeakDetector:
    """
    Real-time z-score peak detector for streaming data (Brakel, 2014).
    
    Processes one value at a time. A peak is detected when a value deviates
    from the rolling mean by more than `threshold` standard deviations.
    The `influence` parameter controls how much detected peaks/valleys
    affect the rolling statistics (0 = ignore peaks entirely, 1 = treat
    peaks like normal data).
    
    Used in beat detection to provide an adaptive threshold that automatically
    adjusts to the current audio level — eliminates the need for manual
    peak_floor tuning in many cases.
    """
    __slots__ = ('lag', 'threshold', 'influence', 'buffer', 'filtered',
                 'mean', 'std', 'initialized', '_buf_len')

    def __init__(self, lag: int = 30, threshold: float = 3.0, influence: float = 0.1):
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

        # Bound memory — keep ~2× lag
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
    fired_bands: list = None      # Which z-score bands actually fired on THIS beat (per-beat, not global)
    metronome_bpm: float = 0.0    # Current internal metronome BPM (for stroke timing)
    acf_confidence: float = 0.0   # ACF peak confidence (0-1, for UI sync indicator)
    is_syncopated: bool = False   # True if an off-beat "and" onset was detected near this beat


class AudioEngine:
    def reset_tempo_tracking(self) -> None:
        """Public method to reset tempo and downbeat tracking immediately."""
        self.last_known_tempo = self.smoothed_tempo
        self.beat_intervals.clear()
        self.beat_times.clear()
        self.beat_position_in_measure = 0
        self.is_downbeat = False
        self.beat_stability = 0.0
        self.stable_tempo = 0.0
        # Reset energy-based downbeat accumulators
        self.measure_energy_accum = [0.0] * self.beats_per_measure
        self.measure_beat_counts = [0] * self.beats_per_measure
        self.downbeat_confidence = 0.0
        # Reset pattern matching state
        self._reset_downbeat_pattern()
        # Reset ACF metronome
        if hasattr(self, '_acf_metronome_enabled'):
            self._reset_acf_metronome()
        # Reset syncopation detection
        if hasattr(self, '_raw_onset_times'):
            self._raw_onset_times.clear()
            self._syncopation_detected = False
            self._syncopation_streak = 0
            self._syncopation_had_offbeat = False
            self._syncopation_confirmed = False
            self._any_band_onset = False
            self._syncopation_armed = False
        # Reset ALL multi-band z-score detectors so they get fresh baselines
        if hasattr(self, '_zscore_detectors'):
            for det in self._zscore_detectors.values():
                det.reset()
            # Reset fire history and confidence
            for name in self._band_fire_history:
                self._band_fire_history[name].clear()
            self._primary_beat_band = 'sub_bass'

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
    
    def __init__(self, config: Config, beat_callback: Callable[[BeatEvent], None]):
        self.config = config
        self.beat_callback = beat_callback
        
        # Audio stream (PyAudio)
        self.pyaudio = None
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
        
        # FFT settings (from config with fallback)
        self.fft_size = getattr(config.audio, 'fft_size', 1024)
        self.hop_size = self.fft_size // 4  # Typical hop = 25% of FFT size
        
        # Pre-allocated arrays for FFT optimization
        self._hanning_window: Optional[np.ndarray] = None  # Will be created on first use
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
        self.beat_position_in_measure: int = 0 # For downbeat tracking (1, 2, 3, 4...)
        
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
        
        # Metric 3: Audio Amp Feedback (No Beats → raise, Excess Beats → lower)
        self._metric_audio_amp_enabled: bool = False
        self._audio_amp_check_interval_ms: float = 2500.0  # Check every ~2.5s (was 1.1s)
        self._audio_amp_escalate_pct: float = 0.02     # 2% of range per check
        self._last_audio_amp_check: float = 0.0         # Last time we checked
        self._audio_amp_hysteresis_count: int = 0       # Consecutive out-of-zone checks (hysteresis)
        self._metric_response_speed: float = float(getattr(config.auto_adjust, 'metric_response_speed', 1.0))
        
        # Metric 5: Flux Balance (keep flux ≈ energy bars at similar height)
        self._metric_flux_balance_enabled: bool = False
        self._flux_balance_check_interval_ms: float = 1000.0  # Check every 1s (was 500ms)
        self._last_flux_balance_check: float = 0.0
        self._flux_energy_ratios: list[float] = []       # Recent flux/energy ratios
        self._flux_balance_target_low: float = 0.6       # Ratio zone: flux between 0.6x and 1.4x energy
        self._flux_balance_target_high: float = 1.4
        self._flux_balance_step_pct: float = 0.01        # 1% of range per check (fine-grained)
        self._flux_balance_hysteresis_count: int = 0     # Consecutive out-of-zone checks (hysteresis)
        
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
            'flux_balance': 0,
        }
        self._metric_settled_flags: dict[str, bool] = {
            'peak_floor': False,
            'sensitivity': False,
            'audio_amp': False,
            'flux_balance': False,
        }
        
        # ===== TARGET BPS SYSTEM (Beats Per Second) =====
        # Tracks actual beats per second and adjusts parameters to achieve target rate
        self._target_bps_enabled: bool = False          # User toggle
        self._target_bps: float = 1.5                   # Target beats per second (default 90 BPM)
        self._target_bps_tolerance: float = 0.2         # ± tolerance (0.2 = accept 1.3-1.7 BPS if target is 1.5)
        self._bps_window_seconds: float = 4.0           # Rolling window for BPS calculation
        self._bps_beat_times: list[float] = []          # Timestamps of recent beats
        self._bps_adjustment_speed: float = 1.0         # Hardcoded to max (was adjustable via slider)
        self._bps_base_step: float = 0.002              # Base step for peak_floor adjustment
        
        # ===== ACF AUTO-METRONOME =====
        # Autocorrelation-based tempo estimator + internal metronome clock.
        # Replaces interval-based beat-counting with robust signal-level tempo detection.
        self._acf_metronome_enabled: bool = True         # Master toggle
        # Onset signal buffer (spectral flux values, one per audio callback)
        self._onset_buffer: list[float] = []
        self._onset_buffer_max: int = 260               # ~6 seconds at ~43 fps (44100/1024)
        self._onset_callback_count: int = 0             # For computing effective sample rate
        self._onset_first_time: float = 0.0             # Timestamp of first onset sample
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
        self._metronome_conf_hold_s: float = 1.2          # Keep metronome running through short ACF confidence dips
        self._metronome_conf_lost_at: float = 0.0         # Timestamp when ACF confidence dropped below threshold
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

        self._session_started_at: float = 0.0
        self._session_frame_count: int = 0
        self._session_raw_rms_min: float | None = None
        self._session_raw_rms_max: float | None = None
        self._session_band_energy_min: float | None = None
        self._session_band_energy_max: float | None = None
        self._session_flux_min: float | None = None
        self._session_flux_max: float | None = None
        self._session_raw_rms_sum: float = 0.0
        self._session_band_energy_sum: float = 0.0
        self._session_flux_sum: float = 0.0

    def _reset_session_stats(self) -> None:
        self._session_started_at = time.time()
        self._session_frame_count = 0
        self._session_raw_rms_min = None
        self._session_raw_rms_max = None
        self._session_band_energy_min = None
        self._session_band_energy_max = None
        self._session_flux_min = None
        self._session_flux_max = None
        self._session_raw_rms_sum = 0.0
        self._session_band_energy_sum = 0.0
        self._session_flux_sum = 0.0

    def _reference_bpm_for_onset_filters(self) -> float:
        if self._metronome_bpm > 0:
            return self._metronome_bpm
        if self._acf_bpm_smoothed > 0:
            return self._acf_bpm_smoothed
        if self.smoothed_tempo > 0:
            return self.smoothed_tempo
        return 0.0

    def _effective_phase_accept_window_s(self) -> float:
        base_ms = float(np.clip(self._phase_accept_window_ms, 10.0, 300.0))
        low_conf_mult = float(np.clip(self._phase_accept_low_conf_mult, 1.0, 4.0))
        conf = float(np.clip(self._acf_confidence, 0.0, 1.0))
        if conf >= 0.25:
            mult = 1.0
        elif conf <= 0.05:
            mult = low_conf_mult
        else:
            t = (conf - 0.05) / 0.20
            mult = low_conf_mult + (1.0 - low_conf_mult) * t
        return (base_ms * mult) / 1000.0

    def _accept_raw_onset(self, now: float) -> bool:
        bpm_ref = self._reference_bpm_for_onset_filters()
        if bpm_ref > 0:
            beat_period_s = 60.0 / bpm_ref
            dedup_frac = float(np.clip(self._beat_dedup_fraction, 0.05, 0.45))
            dedup_window_s = dedup_frac * beat_period_s
        else:
            dedup_window_s = 0.10

        if self._last_accepted_raw_onset_time > 0 and (now - self._last_accepted_raw_onset_time) < dedup_window_s:
            return False

        if self._metronome_bpm > 0:
            beat_period_s = 60.0 / self._metronome_bpm
            phase_frac = self._metronome_phase % 1.0
            phase_dist_frac = min(phase_frac, 1.0 - phase_frac)
            phase_error_s = phase_dist_frac * beat_period_s
            if phase_error_s > self._effective_phase_accept_window_s():
                return False

        self._last_accepted_raw_onset_time = now
        return True

    def _update_session_stats(self, raw_rms: float, band_energy: float, spectral_flux: float) -> None:
        self._session_frame_count += 1
        self._session_raw_rms_sum += raw_rms
        self._session_band_energy_sum += band_energy
        self._session_flux_sum += spectral_flux
        if self._session_raw_rms_min is None or raw_rms < self._session_raw_rms_min:
            self._session_raw_rms_min = raw_rms
        if self._session_raw_rms_max is None or raw_rms > self._session_raw_rms_max:
            self._session_raw_rms_max = raw_rms
        if self._session_band_energy_min is None or band_energy < self._session_band_energy_min:
            self._session_band_energy_min = band_energy
        if self._session_band_energy_max is None or band_energy > self._session_band_energy_max:
            self._session_band_energy_max = band_energy
        if self._session_flux_min is None or spectral_flux < self._session_flux_min:
            self._session_flux_min = spectral_flux
        if self._session_flux_max is None or spectral_flux > self._session_flux_max:
            self._session_flux_max = spectral_flux

    def _log_shutdown_summary(self) -> None:
        if self._session_frame_count <= 0:
            return

        elapsed_s = max(0.0, time.time() - self._session_started_at)
        raw_min = float(self._session_raw_rms_min or 0.0)
        raw_max = float(self._session_raw_rms_max or 0.0)
        band_min = float(self._session_band_energy_min or 0.0)
        band_max = float(self._session_band_energy_max or 0.0)
        flux_min = float(self._session_flux_min or 0.0)
        flux_max = float(self._session_flux_max or 0.0)
        frame_count = float(self._session_frame_count)
        raw_mean = self._session_raw_rms_sum / frame_count
        band_mean = self._session_band_energy_sum / frame_count
        flux_mean = self._session_flux_sum / frame_count

        log_event(
            "INFO",
            "Audio",
            "Shutdown levels summary",
            frames=self._session_frame_count,
            seconds=f"{elapsed_s:.1f}",
            raw_rms_min=f"{raw_min:.6f}",
            raw_rms_max=f"{raw_max:.6f}",
            raw_rms_mean=f"{raw_mean:.6f}",
            raw_rms_span=f"{(raw_max - raw_min):.6f}",
            band_energy_min=f"{band_min:.6f}",
            band_energy_max=f"{band_max:.6f}",
            band_energy_mean=f"{band_mean:.6f}",
            band_energy_span=f"{(band_max - band_min):.6f}",
            flux_min=f"{flux_min:.4f}",
            flux_max=f"{flux_max:.4f}",
            flux_mean=f"{flux_mean:.4f}",
            flux_span=f"{(flux_max - flux_min):.4f}",
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
        wasapi_info = self.pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
        
        if device_index is not None:
            # Use specified device - find its loopback version
            device_info = self.pyaudio.get_device_info_by_index(device_index)
            if not device_info.get("isLoopbackDevice", False):
                # Find the loopback version of this output device
                for loopback in self.pyaudio.get_loopback_device_info_generator():
                    if device_info["name"] in loopback["name"]:
                        device_info = loopback
                        break
        else:
            # Use default output device's loopback
            device_info = self.pyaudio.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            if not device_info.get("isLoopbackDevice", False):
                for loopback in self.pyaudio.get_loopback_device_info_generator():
                    if device_info["name"] in loopback["name"]:
                        device_info = loopback
                        break
        
        log_event("INFO", "AudioEngine", "Using WASAPI loopback", device=device_info['name'])
        log_event("INFO", "AudioEngine", "Loopback format", channels=device_info['maxInputChannels'], sample_rate=int(device_info['defaultSampleRate']))
        
        # Update config with actual sample rate
        self.config.audio.sample_rate = int(device_info['defaultSampleRate'])
        self.config.audio.channels = device_info['maxInputChannels']
        
        # Open stream
        self.stream = self.pyaudio.open(
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
        if device_index is None:
            # Find default input device
            wasapi_info = self.pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
            device_index = wasapi_info.get("defaultInputDevice", 0)
        
        device_info = self.pyaudio.get_device_info_by_index(device_index)
        
        log_event("INFO", "AudioEngine", "Using input device", device=device_info['name'])
        log_event("INFO", "AudioEngine", "Input format", channels=device_info['maxInputChannels'], sample_rate=int(device_info['defaultSampleRate']))
        
        # Update config with actual sample rate
        self.config.audio.sample_rate = int(device_info['defaultSampleRate'])
        self.config.audio.channels = min(device_info['maxInputChannels'], 2)  # Use up to 2 channels
        
        # Open stream
        self.stream = self.pyaudio.open(
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
        
        # Pre-allocate Hanning window on first use (or if size changed)
        if self._hanning_window is None or len(self._hanning_window) != len(mono):
            self._hanning_window = np.hanning(len(mono)).astype(np.float32)
        
        # Always compute FFT for frequency estimation (needed for dominant freq detection)
        windowed = mono * self._hanning_window
        spectrum = np.abs(np.fft.rfft(windowed))
        spectrum_viz = (spectrum / max(1, len(spectrum))) * 2.0
        
        # Store full spectrum for visualization (only on scheduled frames, if enabled)
        if update_spectrum_viz:
            with self.spectrum_lock:
                self.spectrum_data = spectrum_viz.copy()
        
        # For beat detection: use Butterworth filtered signal if available, else FFT band filter
        if self._butter_sos is not None:
            # Use time-domain energy from Butterworth filtered signal
            band_energy = np.sqrt(np.mean(beat_mono ** 2))
            band_energy = band_energy * self.config.audio.gain  # Apply audio gain
            # Still compute spectral flux from filtered signal's spectrum
            beat_windowed = beat_mono * self._hanning_window
            beat_spectrum = np.abs(np.fft.rfft(beat_windowed)) * self.config.audio.gain
            spectral_flux = self._compute_spectral_flux(beat_spectrum)
        else:
            # Fallback: FFT-based frequency band filtering (spectrum already computed above)
            band_spectrum = self._filter_frequency_band(spectrum)
            band_spectrum = band_spectrum * self.config.audio.gain
            band_energy = np.sqrt(np.mean(band_spectrum ** 2)) if len(band_spectrum) > 0 else 0
            spectral_flux = self._compute_spectral_flux(band_spectrum)

        raw_rms = np.sqrt(np.mean(mono ** 2))
        self._update_session_stats(raw_rms, band_energy, spectral_flux)
        
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
                raw_rms=f"{raw_rms:.6f}",
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
            
        # Check for tempo timeout (no beats for 2000ms)
        current_time = time.time()
        time_since_last_beat = (current_time - self.last_beat_time) * 1000 if self.last_beat_time > 0 else 0
        
        tempo_reset_flag = False
        if time_since_last_beat > self.tempo_timeout_ms and len(self.beat_intervals) > 0:
            # Timeout reached - reset tempo tracking but preserve last known tempo
            log_event(
                "INFO",
                "Tempo",
                "No beats detected, resetting tracker",
                idle_ms=f"{time_since_last_beat:.0f}",
                bpm=f"{self.smoothed_tempo:.1f}"
            )
            self.last_known_tempo = self.smoothed_tempo  # Preserve current tempo
            self.beat_intervals.clear()
            self.beat_times.clear()
            self.beat_position_in_measure = 0
            self.is_downbeat = False
            self._reset_downbeat_pattern()  # Also reset pattern matching when tempo resets
            tempo_reset_flag = True
        
        # Detect beat based on mode (using band energy)
        # Store last flux for flux balance metric
        self._last_spectral_flux = spectral_flux
        
        # ===== ACF ONSET BUFFERING =====
        self._onset_buffer.append(spectral_flux)
        if len(self._onset_buffer) > self._onset_buffer_max:
            self._onset_buffer.pop(0)
        # Calibrate effective onset sample rate
        self._onset_callback_count += 1
        if self._onset_first_time == 0.0:
            self._onset_first_time = current_time
        elif self._onset_callback_count > 60:  # After ~1.5 seconds, calibrate fps
            elapsed = current_time - self._onset_first_time
            if elapsed > 0:
                self._acf_onset_fps = self._onset_callback_count / elapsed
        
        # Run ACF tempo estimation periodically
        if current_time - self._last_acf_time > self._acf_interval_ms / 1000.0:
            self._last_acf_time = current_time
            self._estimate_tempo_acf()
        
        # Raw beat detection (still runs for onset detection + phase-lock + fallback)
        raw_is_beat = self._detect_beat(band_energy, spectral_flux)
        accepted_raw_is_beat = raw_is_beat and self._accept_raw_onset(current_time)
        
        # Track raw onset times for syncopation detection
        if accepted_raw_is_beat:
            self._raw_onset_times.append(current_time)
            if len(self._raw_onset_times) > self._raw_onset_max:
                self._raw_onset_times.pop(0)
        
        # Advance internal metronome (pass band_energy for energy-based downbeat detection)
        self._advance_metronome(current_time, band_energy)
        self._predict_next_beat(current_time)
        
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
                window = self.config.beat.syncopation_window
                dist_to_half = abs(phase_frac - 0.5)
                if dist_to_half < window:
                    self._syncopation_had_offbeat = True
                    if self._syncopation_streak >= 1:
                        # Previous beat period had off-beats → fire immediately
                        self._syncopation_detected = True
                        log_event("INFO", "Syncopation", "Off-beat onset detected",
                                  phase=f"{phase_frac:.2f}", bpm=f"{self._metronome_bpm:.1f}")
                    elif self._syncopation_armed:
                        # Second off-beat in same period → fire (fast first-time reaction)
                        self._syncopation_detected = True
                        self._syncopation_streak = 1  # pre-confirm for next period
                        log_event("INFO", "Syncopation", "Armed → firing (2nd onset)",
                                  phase=f"{phase_frac:.2f}", bpm=f"{self._metronome_bpm:.1f}")
                    else:
                        # First off-beat onset ever → arm for second
                        self._syncopation_armed = True

        # Predictive drop-off: if we're past the off-beat window (phase > 0.65)
        # and no off-beat onset was detected this beat period, preemptively
        # reset streak so the NEXT beat won't produce a false syncopation.
        if self._metronome_bpm > 0 and not self._metronome_beat_fired:
            phase_frac = self._metronome_phase % 1.0
            window = self.config.beat.syncopation_window
            if phase_frac > (0.5 + window) and not self._syncopation_had_offbeat:
                # Past the "and" window with no onset → pattern broken
                if self._syncopation_streak > 0 or self._syncopation_armed:
                    self._syncopation_streak = 0
                    self._syncopation_confirmed = False
                    self._syncopation_armed = False
                    log_event("INFO", "Syncopation", "Predictive drop-off (no onset in window)")
        
        # Choose beat source: metronome (when running) or raw detection (fallback)
        if self._acf_metronome_enabled and self._metronome_bpm > 0:
            is_beat = self._metronome_beat_fired
            is_downbeat_flag = self._metronome_downbeat_fired
            current_bpm = self._metronome_bpm
            # Tempo is locked when ACF is confident enough.
            # Downbeat matching is a bonus confirmation, not a hard requirement.
            # This ensures tempo_locked=True even before downbeat pattern settles.
            tempo_is_locked = (self._acf_confidence > 0.2
                               and (self.consecutive_matching_downbeats >= 1
                                    or self._acf_confidence > 0.35))
            # Update last_beat_time for tempo timeout check
            if is_beat:
                self._metronome_last_beat_time = current_time
        else:
            is_beat = accepted_raw_is_beat
            is_downbeat_flag = self.is_downbeat if is_beat else False
            current_bpm = self.smoothed_tempo if self.smoothed_tempo > 0 else self.last_known_tempo
            tempo_is_locked = self.consecutive_matching_downbeats >= self.consecutive_match_threshold
        
        # Estimate dominant frequency
        freq = self._estimate_frequency(spectrum)
        
        event = BeatEvent(
            timestamp=time.time(),
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
        
        # Calculate bin indices for frequency range
        freq_per_bin = sr / (2 * len(spectrum))  # Nyquist / num_bins
        low_bin = max(0, int(cfg.freq_low / freq_per_bin))
        high_bin = min(len(spectrum) - 1, int(cfg.freq_high / freq_per_bin))
        
        if low_bin >= high_bin:
            return spectrum  # Return full spectrum if range is invalid
        
        return spectrum[low_bin:high_bin+1]
    
    def get_freq_band_bins(self) -> tuple:
        """Get the current frequency band as normalized positions (0-1) for visualization"""
        cfg = self.config.beat
        sr = self.config.audio.sample_rate
        max_freq = sr / 2  # Nyquist
        
        low_norm = cfg.freq_low / max_freq
        high_norm = cfg.freq_high / max_freq
        return (low_norm, high_norm)
        
    def _compute_spectral_flux(self, spectrum: np.ndarray) -> float:
        """Compute spectral flux (change in spectrum)"""
        if self.prev_spectrum is None or len(self.prev_spectrum) != len(spectrum):
            # Reset if size changed (frequency band was adjusted)
            self.prev_spectrum = spectrum.copy()
            return 0.0
            
        # Only consider positive changes (onset detection)
        diff = spectrum - self.prev_spectrum
        flux = np.sum(np.maximum(0, diff))
        
        self.prev_spectrum = spectrum.copy()
        
        # Normalize
        if len(spectrum) > 0:
            flux = flux / len(spectrum)
        return flux * self.config.beat.flux_multiplier

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
        nyquist = sr / 2
        n_bins = len(spectrum)
        if n_bins == 0:
            return
        freq_per_bin = nyquist / n_bins
        gain = self.config.audio.gain

        for name, low_hz, high_hz in self._zscore_bands:
            low_bin  = max(0, int(low_hz / freq_per_bin))
            high_bin = min(n_bins - 1, int(high_hz / freq_per_bin))
            band_slice = spectrum[low_bin : high_bin + 1]

            if len(band_slice) > 0:
                energy = float(np.sqrt(np.mean(band_slice ** 2))) * gain
            else:
                energy = 0.0

            self._band_energies[name] = energy

            # Feed the per-band detector
            signal = self._zscore_detectors[name].update(energy)
            self._band_zscore_signals[name] = signal

            # Append to rolling fire history (1 = fired, 0 = quiet)
            self._band_fire_history[name].append(1 if signal == 1 else 0)
            if len(self._band_fire_history[name]) > self._band_confidence_window:
                self._band_fire_history[name].pop(0)

        # ---- Select primary band (most consistent fires) ----
        best_band = self._primary_beat_band
        best_score = -1
        for name, _, _ in self._zscore_bands:
            hist = self._band_fire_history[name]
            score = sum(hist) if len(hist) >= 10 else 0
            if score > best_score:
                best_score = score
                best_band  = name

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
        if n < 120:  # Need at least ~3 seconds of data
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

        if peak_value < 0.08:  # Below noise floor — no clear tempo
            self._acf_confidence *= 0.9  # Fade confidence
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
        candidates = [(bpm, peak_value)]  # (bpm, confidence)

        # Half-period candidate (double BPM)
        half_lag = raw_lag // 2
        if half_lag >= min_lag:
            half_val = float(acf[half_lag])
            if half_val > peak_value * 0.60:
                bpm_half = 60.0 * fps / half_lag
                if 55 <= bpm_half <= 185:
                    candidates.append((bpm_half, half_val))

        # Double-period candidate (half BPM)
        double_lag = raw_lag * 2
        if double_lag <= max_lag:
            double_val = float(acf[double_lag])
            if double_val > peak_value * 0.60:
                bpm_double = 60.0 * fps / double_lag
                if 55 <= bpm_double <= 185:
                    candidates.append((bpm_double, double_val))

        # Pick best candidate: prefer closest to target BPM if enabled
        target_bpm_hint = self._target_bps * 60.0 if self._target_bps_enabled and self._target_bps > 0 else 0.0
        if target_bpm_hint > 0 and len(candidates) > 1:
            # Score by distance to target BPM, weighted by confidence
            def octave_score(c):
                bpm_c, conf_c = c
                # Distance as ratio (penalises being far from target)
                ratio = abs(bpm_c - target_bpm_hint) / target_bpm_hint
                # Confidence bonus (higher confidence = better)
                return ratio - conf_c * 0.3  # Lower score = better
            candidates.sort(key=octave_score)
            bpm, peak_value = candidates[0]
            log_event("DEBUG", "ACF", "Octave disambig (target-guided)",
                      target=f"{target_bpm_hint:.0f}",
                      chosen=f"{bpm:.1f}",
                      candidates=str([(f"{c[0]:.1f}", f"{c[1]:.2f}") for c in candidates]))
        else:
            # No target BPM: prefer faster tempo if half-period peak is strong
            if len(candidates) > 1:
                # Sort by BPM descending (prefer faster), filter by reasonable confidence
                fast = [c for c in candidates if c[1] > peak_value * 0.75]
                if fast:
                    fast.sort(key=lambda c: -c[0])
                    bpm, peak_value = fast[0]

        # Clamp to sane range
        if bpm < 55 or bpm > 185:
            return

        self._acf_confidence = float(peak_value)
        self._acf_bpm = bpm

        # Smooth the BPM estimate
        if self._acf_bpm_smoothed > 0:
            ratio = abs(bpm - self._acf_bpm_smoothed) / self._acf_bpm_smoothed
            if ratio < 0.15:  # Within 15% — smooth
                self._acf_bpm_smoothed = 0.85 * self._acf_bpm_smoothed + 0.15 * bpm
            elif peak_value > 0.25:  # Large jump but confident
                # Guard against octave jumps when target BPM is set
                # If the jump looks like a doubling/halving and we have a target,
                # only accept if the new BPM is closer to target than current
                if target_bpm_hint > 0 and (0.45 < ratio < 0.55 or 0.90 < ratio < 1.10):
                    old_dist = abs(self._acf_bpm_smoothed - target_bpm_hint)
                    new_dist = abs(bpm - target_bpm_hint)
                    if new_dist < old_dist:
                        self._acf_bpm_smoothed = bpm
                        log_event("INFO", "ACF", "Tempo jump (target-validated)",
                                  bpm=f"{bpm:.1f}", target=f"{target_bpm_hint:.0f}",
                                  confidence=f"{peak_value:.3f}")
                    else:
                        log_event("INFO", "ACF", "Tempo jump REJECTED (farther from target)",
                                  bpm=f"{bpm:.1f}", target=f"{target_bpm_hint:.0f}",
                                  current=f"{self._acf_bpm_smoothed:.1f}")
                else:
                    self._acf_bpm_smoothed = bpm
                    log_event("INFO", "ACF", "Tempo jump",
                              bpm=f"{bpm:.1f}", confidence=f"{peak_value:.3f}")
            # else: ignore noisy outlier
        else:
            self._acf_bpm_smoothed = bpm
            log_event("INFO", "ACF", "Initial tempo lock",
                      bpm=f"{bpm:.1f}", confidence=f"{peak_value:.3f}",
                      fps=f"{fps:.1f}")

    def _estimate_onset_bpm(self) -> float:
        """Estimate BPM from recent raw onset intervals for fast fallback/fusion."""
        if len(self._raw_onset_times) < 3:
            return 0.0

        intervals = np.diff(np.array(self._raw_onset_times[-8:], dtype=np.float64))
        if len(intervals) == 0:
            return 0.0

        valid = intervals[(intervals >= 0.15) & (intervals <= 1.2)]
        if len(valid) < 2:
            return 0.0

        median_interval = float(np.median(valid))
        bpm = 60.0 / median_interval if median_interval > 0 else 0.0
        if 55.0 <= bpm <= 185.0:
            return bpm
        return 0.0

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
        target_bpm = self._acf_bpm_smoothed
        if onset_bpm > 0:
            if target_bpm <= 0:
                target_bpm = onset_bpm
            else:
                min_w = self._tempo_fusion_min_acf_weight
                max_w = self._tempo_fusion_max_acf_weight
                if max_w < min_w:
                    min_w, max_w = max_w, min_w
                acf_weight = min_w + (max_w - min_w) * acf_conf
                acf_weight = max(min_w, min(max_w, acf_weight))
                target_bpm = acf_weight * target_bpm + (1.0 - acf_weight) * onset_bpm

        if target_bpm <= 0 or (acf_conf < 0.10 and onset_bpm <= 0):
            if self._metronome_bpm > 0:
                if self._metronome_conf_lost_at <= 0:
                    self._metronome_conf_lost_at = now
                if (now - self._metronome_conf_lost_at) <= self._metronome_conf_hold_s:
                    target_bpm = self._metronome_bpm
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

        # Advance phase
        beats_per_sec = self._metronome_bpm / 60.0
        old_phase = self._metronome_phase
        self._metronome_phase += beats_per_sec * dt

        # Detect beat boundary crossing
        old_beat = int(old_phase)
        new_beat = int(self._metronome_phase)
        if new_beat > old_beat:
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
                        phase_correction = (self.phase_error_ms / beat_period_ms) * 0.15  # 15% correction
                        phase_correction = max(-0.1, min(0.1, phase_correction))  # Clamp
                        self._metronome_phase += phase_correction
                        log_event("INFO", "Downbeat", "Phase correction from downbeat",
                                  error_ms=f"{self.phase_error_ms:.1f}",
                                  correction=f"{phase_correction:.4f}")
                else:
                    # Don't fully reset on single mismatch — allow recovery
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
            error = -phase_frac    # Just past last beat → pull backward
        else:
            error = 1.0 - phase_frac  # Approaching next beat → push forward

        if abs(error) < self._metronome_pll_window:
            conf = max(0.0, min(1.0, self._acf_confidence))
            gain = self._metronome_pll_base_gain + self._metronome_pll_conf_gain * conf
            error_scale = 0.5 + 0.5 * min(1.0, abs(error) / 0.25)
            correction = error * gain * min(1.0, onset_strength) * error_scale
            correction = max(-0.14, min(0.14, correction))
            self._metronome_phase += correction

    def _reset_acf_metronome(self):
        """Reset ACF estimator and internal metronome."""
        self._onset_buffer.clear()
        self._onset_callback_count = 0
        self._onset_first_time = 0.0
        self._acf_bpm = 0.0
        self._acf_bpm_smoothed = 0.0
        self._acf_confidence = 0.0
        self._metronome_phase = 0.0
        self._metronome_beat_count = 0
        self._metronome_conf_lost_at = 0.0
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
        that the manual peak_floor setting would miss — and vice-versa.
        """
        cfg = self.config.beat
        
        # Track valley detection (local minima) for peak_floor metric
        # A valley occurs when energy stops falling and starts rising
        if energy > self._prev_energy_for_valley and self._energy_was_falling:
            # Just turned upward → previous value was a valley
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
        
        # Refractory period — suppress re-triggers within min_interval_ms
        # This prevents burst clusters (4-6 detections within 250ms) from a single musical beat.
        # Uses the same min_interval_ms that guards strokes, so BPS metrics count only real beats.
        if not hasattr(self, '_last_beat_time'):
            self._last_beat_time = 0
        
        current_time = time.time()
        refractory_s = self.config.stroke.min_interval_ms / 1000.0  # e.g. 300ms → 0.3s
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
        
        return is_beat
    
    def _update_tempo_tracking(self, current_time: float, energy: float = 0.0):
        """Update tempo estimate with beat-based interval tracking (madmom-inspired)"""
        # Skip if tempo tracking is disabled
        if not self.tempo_tracking_enabled:
            return
            
        # Calculate interval from last beat
        if self.last_beat_time > 0:
            interval = current_time - self.last_beat_time
            
            # Auto-halve/double interval to bring BPM into 60-180 range
            min_bpm = 60.0
            max_bpm = 180.0
            min_interval = 60.0 / max_bpm  # ~0.333s
            max_interval = 60.0 / min_bpm  # 1.0s
            
            # Calculate what BPM this interval would give
            if interval > 0:
                raw_bpm = 60.0 / interval
                adjusted_interval = interval
                
                # Auto-halve: if BPM > 180, double the interval (halve BPM)
                while 60.0 / adjusted_interval > max_bpm and adjusted_interval < 2.0:
                    adjusted_interval *= 2
                    log_event("INFO", "Tempo", "Auto-halved BPM", original=f"{60.0/interval:.1f}", adjusted=f"{60.0/adjusted_interval:.1f}")
                
                # Auto-double: if BPM < 60, halve the interval (double BPM)
                while 60.0 / adjusted_interval < min_bpm and adjusted_interval > 0.1:
                    adjusted_interval /= 2
                    log_event("INFO", "Tempo", "Auto-doubled BPM", original=f"{60.0/interval:.1f}", adjusted=f"{60.0/adjusted_interval:.1f}")
                
                interval = adjusted_interval
            
            # Reject intervals still outside reasonable range after adjustment
            if interval < 0.15 or interval > 2.0:
                log_event("INFO", "Tempo", "Interval rejected", interval=f"{interval:.3f}s", bpm=f"{60.0/interval:.1f}")
                return
            if interval > 0.2:
                # Outlier rejection: if interval is way off from average, it might be a false beat
                if len(self.beat_intervals) > 0:
                    avg_interval = np.mean(self.beat_intervals)
                    # Accept if within 0.5x to 2.0x of average (allows tempo changes but rejects glitches)
                    if interval < (0.5 * avg_interval) or interval > (2.0 * avg_interval):
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
                weighted_avg_interval = np.average(self.beat_intervals, weights=weights)
                # Convert to BPM
                new_tempo = 60.0 / weighted_avg_interval if weighted_avg_interval > 0 else 0
                # Apply exponential smoothing for stability (like madmom's tempo state space)
                smoothing_factor = 0.7  # Higher = more smooth (less responsive)
                if self.smoothed_tempo > 0:
                    self.smoothed_tempo = (smoothing_factor * self.smoothed_tempo + 
                                          (1 - smoothing_factor) * new_tempo)
                else:
                    self.smoothed_tempo = new_tempo
                
                # Beat stability metric (TISMIR PLP-inspired)
                # Coefficient of variation of recent intervals: low = stable tempo
                if len(self.beat_intervals) >= 3:
                    intervals_arr = np.array(self.beat_intervals)
                    cv = np.std(intervals_arr) / np.mean(intervals_arr) if np.mean(intervals_arr) > 0 else 1.0
                    # Convert CV to a 0-1 stability score (0 = chaotic, 1 = perfect)
                    self.beat_stability = max(0.0, 1.0 - (cv / self.stability_threshold))
                    
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
                # ONLY runs when metronome is NOT active — metronome owns the
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
        
        self.last_beat_time = current_time
    
    def _predict_next_beat(self, current_time: float):
        """Predict the time of the next beat using metronome when active."""
        if self._acf_metronome_enabled and self._metronome_bpm > 0:
            phase_frac = self._metronome_phase % 1.0
            beats_to_next = 1.0 - phase_frac if phase_frac > 1e-9 else 1.0
            predicted_interval = beats_to_next * (60.0 / self._metronome_bpm)
            self.predicted_next_beat = current_time + predicted_interval
            return

        if self.smoothed_tempo > 0:
            predicted_interval = 60.0 / self.smoothed_tempo
            self.predicted_next_beat = current_time + predicted_interval
    
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
            # Error exceeds tolerance — but if it's close to a measure boundary,
            # update prediction anyway to prevent permanent lock-out
            if abs(self.phase_error_ms) <= effective_tolerance * 2.0:
                # Close but not perfect — update prediction to re-sync
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
        # Use stable_tempo for display if available, otherwise fall back to smoothed
        display_bpm = self.stable_tempo if self.stable_tempo > 0 else self.smoothed_tempo
        # ACF metronome info (when active, these take priority)
        acf_active = self._acf_metronome_enabled and self._metronome_bpm > 0
        if acf_active:
            display_bpm = self._metronome_bpm
            beat_pos = ((self._metronome_beat_count - 1) % self.beats_per_measure) + 1 if self._metronome_beat_count > 0 else 0
        else:
            beat_pos = self.beat_position_in_measure

        return {
            'bpm': display_bpm,
            'raw_bpm': self.smoothed_tempo,
            'stable_bpm': self.stable_tempo,
            'beat_position': beat_pos,
            'is_downbeat': self.is_downbeat,
            'predicted_next_beat': self.predicted_next_beat,
            'interval_count': len(self.beat_intervals),
            'confidence': min(1.0, len(self.beat_intervals) / 4.0),
            'stability': self.beat_stability,
            'consecutive_matching_downbeats': self.consecutive_matching_downbeats,
            'phase_error_ms': self.phase_error_ms,
            # ACF metronome fields
            'acf_bpm': self._acf_bpm_smoothed,
            'acf_confidence': self._acf_confidence,
            'acf_active': acf_active,
            'metronome_bpm': self._metronome_bpm,
        }
            
    def _estimate_frequency(self, spectrum: np.ndarray) -> float:
        """Estimate dominant frequency from spectrum"""
        if len(spectrum) == 0:
            return 0.0
            
        # Find peak bin
        peak_bin = np.argmax(spectrum)
        
        # Convert to frequency
        freq = peak_bin * self.config.audio.sample_rate / (2 * len(spectrum))
        return freq
        
    def get_spectrum(self) -> Optional[np.ndarray]:
        """Get current spectrum data for visualization"""
        with self.spectrum_lock:
            return self.spectrum_data.copy() if self.spectrum_data is not None else None
    
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
        elif metric == 'flux_balance':
            self._metric_flux_balance_enabled = enable
            if enable:
                self._last_flux_balance_check = 0.0
                self._flux_energy_ratios.clear()
                self._metric_settled_counts['flux_balance'] = 0
                self._metric_settled_flags['flux_balance'] = False
                log_event("INFO", "MetricAutoRange", "Flux Balance metric enabled (bar-balance)")
            else:
                log_event("INFO", "MetricAutoRange", "Flux Balance metric disabled")
        elif metric == 'target_bps':
            self._target_bps_enabled = enable
            if enable:
                self._bps_beat_times.clear()
                log_event("INFO", "MetricAutoRange", f"Target BPS enabled (target={self._target_bps:.2f})")
            else:
                log_event("INFO", "MetricAutoRange", "Target BPS disabled")
    
    def compute_energy_margin_feedback(self, band_energy: float, callback=None):
        """
        Compute peak_floor adjustment based on valley tracking.
        
        peak_floor should sit at the average valley level (local minima between beats).
        This naturally scales with amplification since valleys scale with the signal.
        
        Valley = average of recent energy local minima (detected in _detect_beat).
        If peak_floor < valley: raise it (too much noise passes through)
        If peak_floor > valley: lower it (real peaks might be filtered out)
        
        Tolerance band: peak_floor should be within ±20% of avg valley.
        
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
            # Not enough valley data yet — fall back to simple margin check
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
        
        # Tolerance: peak_floor should be within ±20% of valley level
        tolerance = avg_valley * 0.20
        
        should_adjust = False
        direction = 0
        
        if error > tolerance:
            # peak_floor too HIGH vs valleys → lower it so peaks pass through
            should_adjust = True
            direction = -1
        elif error < -tolerance:
            # peak_floor too LOW vs valleys → raise it to filter noise
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
            # In zone — increment settled counter
            self._metric_settled_counts['peak_floor'] = self._metric_settled_counts.get('peak_floor', 0) + 1
            if self._metric_settled_counts['peak_floor'] >= self._effective_metric_settled_threshold():
                self._metric_settled_flags['peak_floor'] = True
                log_event("INFO", "Metric", "Peak Floor SETTLED",
                          valley=f"{avg_valley:.4f}", pf=f"{current_pf:.4f}")
        
        return float(np.mean(self._energy_margin_history)), should_adjust, direction

    def compute_bps_feedback(self, beat_time: float, callback=None):
        """
        Compute BPS (beats per second) feedback and adjust peak_floor to hit target.
        
        Tracks beats over a rolling window and compares actual BPS to target.
        If actual < target: lower peak_floor to detect more beats
        If actual > target: raise peak_floor to detect fewer beats
        
        Args:
            beat_time: Timestamp of the detected beat
            callback: Function to call with adjustment data
            
        Returns:
            (actual_bps, should_adjust, adjustment_direction)
        """
        if not self._target_bps_enabled:
            return 0.0, False, 0
        
        now = beat_time
        
        # Add this beat
        self._bps_beat_times.append(now)
        
        # Prune beats outside the window
        window_start = now - self._bps_window_seconds
        self._bps_beat_times = [t for t in self._bps_beat_times if t >= window_start]
        
        # Need at least 2 beats to calculate BPS
        if len(self._bps_beat_times) < 2:
            return 0.0, False, 0
        
        # Calculate actual BPS
        window_duration = self._bps_beat_times[-1] - self._bps_beat_times[0]
        if window_duration <= 0:
            return 0.0, False, 0
        
        actual_bps = (len(self._bps_beat_times) - 1) / window_duration
        
        # Check if we're within tolerance
        target_low = self._target_bps - self._target_bps_tolerance
        target_high = self._target_bps + self._target_bps_tolerance
        
        should_adjust = False
        direction = 0
        
        if actual_bps < target_low:
            # Too few beats - LOWER peak_floor to detect more
            should_adjust = True
            direction = -1
        elif actual_bps > target_high:
            # Too many beats - RAISE peak_floor to detect fewer
            should_adjust = True
            direction = +1
        
        if callback and should_adjust:
            # Scale step by adjustment speed (0.5 = normal, 1.0 = 2x aggressive)
            step = self._bps_base_step * (1.0 + self._bps_adjustment_speed)
            step = self._scaled_metric_step(step)
            callback({
                'metric': 'target_bps',
                'actual_bps': actual_bps,
                'target_bps': self._target_bps,
                'tolerance': self._target_bps_tolerance,
                'adjustment': direction * step,
                'direction': 'raise' if direction > 0 else 'lower'
            })
        
        return actual_bps, should_adjust, direction

    def set_target_bps(self, target: float):
        """Set the target beats per second"""
        self._target_bps = max(0.1, min(4.0, target))
        
    def set_bps_adjustment_speed(self, speed: float):
        """Set the BPS adjustment speed (0.0=fine, 1.0=aggressive)"""
        self._bps_adjustment_speed = max(0.0, min(1.0, speed))
        
    def set_bps_tolerance(self, tolerance: float):
        """Set the BPS tolerance (how close to target before adjusting)"""
        self._target_bps_tolerance = max(0.05, min(1.0, tolerance))

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
        if self._metric_flux_balance_enabled:
            states['flux_balance'] = 'SETTLED' if self._metric_settled_flags.get('flux_balance', False) else 'ADJUSTING'
        return states

    def compute_flux_balance_feedback(self, now: float, callback=None):
        """
        Timer-driven flux_mult adjustment to keep flux ≈ energy (bar balance).
        
        Compares recent average flux to recent average energy.
        If flux >> energy: LOWER flux_mult (shrink flux bar)
        If flux << energy: RAISE flux_mult (grow flux bar)
        
        Uses current peak_envelope (energy) and last spectral_flux from BeatEvent.
        Both are already scaled by gain, so the ratio reflects display bar heights.
        
        Called from _update_display (~30fps), acts every ~500ms.
        """
        if not self._metric_flux_balance_enabled:
            return
        
        # Only check every ~500ms
        if now - self._last_flux_balance_check < self._scaled_metric_interval_s(self._flux_balance_check_interval_ms):
            return
        self._last_flux_balance_check = now
        
        # If already settled, don't adjust
        if self._metric_settled_flags.get('flux_balance', False):
            return
        
        # Get current energy and flux values
        energy = self.peak_envelope
        flux = getattr(self, '_last_spectral_flux', 0.0)
        
        # Skip if either is negligible (no audio / silence)
        if energy < 0.005 or flux < 0.001:
            return
        
        # Compute ratio: flux / energy
        ratio = flux / energy
        
        # Track rolling history (last 8 samples = ~4 seconds at 500ms interval)
        self._flux_energy_ratios.append(ratio)
        if len(self._flux_energy_ratios) > 8:
            self._flux_energy_ratios.pop(0)
        
        # Need at least 3 samples for stability
        if len(self._flux_energy_ratios) < 3:
            return
        
        avg_ratio = float(np.mean(self._flux_energy_ratios))
        
        # Get range for step calculation
        from config import BEAT_RANGE_LIMITS
        fm_min, fm_max = BEAT_RANGE_LIMITS['flux_mult']
        fm_range = fm_max - fm_min
        step = fm_range * self._flux_balance_step_pct  # 1% of range
        step = self._scaled_metric_step(step)
        
        wants_adjustment = False
        adjustment_direction = 0
        adjustment_reason = ''
        
        if avg_ratio > self._flux_balance_target_high:
            # Flux bar too tall relative to energy → wants to LOWER flux_mult
            wants_adjustment = True
            adjustment_direction = -1
            adjustment_reason = f'flux/energy ratio {avg_ratio:.2f} > {self._flux_balance_target_high:.1f}'
        elif avg_ratio < self._flux_balance_target_low:
            # Flux bar too short relative to energy → wants to RAISE flux_mult
            wants_adjustment = True
            adjustment_direction = +1
            adjustment_reason = f'flux/energy ratio {avg_ratio:.2f} < {self._flux_balance_target_low:.1f}'
        
        # Hysteresis: require 2 consecutive out-of-zone checks before adjusting
        if wants_adjustment:
            self._flux_balance_hysteresis_count += 1
            if self._flux_balance_hysteresis_count >= self._effective_metric_hysteresis_required():
                # Actually adjust now
                # Decay settled counter instead of hard reset (drop by 3, not to 0)
                self._metric_settled_counts['flux_balance'] = max(0, self._metric_settled_counts.get('flux_balance', 0) - 3)
                self._flux_balance_hysteresis_count = 0
                if callback:
                    callback({
                        'metric': 'flux_balance',
                        'adjustment': adjustment_direction * step,
                        'direction': 'lower' if adjustment_direction < 0 else 'raise',
                        'ratio': avg_ratio,
                        'reason': f'{adjustment_reason} (2x confirmed)',
                    })
        else:
            # In zone — reset hysteresis counter and increment settled
            self._flux_balance_hysteresis_count = 0
            self._metric_settled_counts['flux_balance'] = self._metric_settled_counts.get('flux_balance', 0) + 1
            if self._metric_settled_counts['flux_balance'] >= self._effective_metric_settled_threshold():
                self._metric_settled_flags['flux_balance'] = True
                log_event("INFO", "Metric", "Flux Balance SETTLED",
                          ratio=f"{avg_ratio:.2f}")

    def compute_audio_amp_feedback(self, now: float, callback=None):
        """
        Timer-driven audio_amp adjustment based on beat presence.
        
        - No beats for >check_interval → RAISE audio_amp (+2% of range)
        - Excess beats (BPS > 2× target) → LOWER audio_amp (1% of range, half raise rate)
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
        target_interval = 1.0 / self._target_bps if self._target_bps > 0 else 0.67
        
        wants_adjustment = False
        if time_since_beat > target_interval * 3.0:
            # No beats detected for 3x expected interval → wants to RAISE audio_amp
            wants_adjustment = True
        
        # Check for excess beats: if BPS > 2× target for consecutive checks, LOWER audio_amp
        wants_lower = False
        if self.last_beat_time > 0 and time_since_beat < target_interval:
            # Beats are coming — check if too many
            if len(self._bps_beat_times) >= 2:
                window_dur = self._bps_beat_times[-1] - self._bps_beat_times[0] if len(self._bps_beat_times) >= 2 else 1.0
                if window_dur > 0:
                    actual_bps = (len(self._bps_beat_times) - 1) / window_dur
                    if actual_bps > self._target_bps * 2.0:
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
                            'reason': f'excess BPS > 2x target (2x confirmed)',
                        })
                elif callback:
                    callback({
                        'metric': 'audio_amp',
                        'adjustment': +step,
                        'direction': 'raise',
                        'reason': f'no beats for {time_since_beat:.1f}s (2x confirmed)',
                    })
        else:
            # In zone — reset hysteresis counter and increment settled
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
    
    log_event("INFO", "AudioEngine", "Available devices")
    for d in engine.list_devices():
        log_event("INFO", "AudioEngine", "Device", index=d['index'], name=d['name'], inputs=d['inputs'])
        
    log_event("INFO", "AudioEngine", "Starting audio capture (Ctrl+C to stop)...")
    engine.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        engine.stop()
