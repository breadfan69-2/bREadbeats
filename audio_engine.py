"""
bREadbeats - Audio Engine
Captures system audio and detects beats using spectral flux / peak energy.
Uses pyaudiowpatch for WASAPI loopback capture.
"""

import numpy as np
try:
    import pyaudiowpatch as pyaudio
except Exception:
    class _PyAudioFallback:
        paContinue = 0

    pyaudio = _PyAudioFallback()
import threading
from collections import deque
from typing import Any, Callable, Optional
import time
import platform

from logging_utils import log_event
from audio_modules.feature_extractors import (
    compute_bass_dominance,
    compute_offbeat_score,
    compute_teaching_confidence,
    compute_multiband_energies,
    estimate_dominant_frequency,
    rolling_percentile_norm,
    select_primary_band_by_fire_history,
)
from audio_modules.contracts import (
    BeatEvent,
    FeatureFrame,
    RMS_DB_FLOOR,
    TempoState,
    TriggerDecision,
    rms_to_dbfs,
    silence_threshold_to_dbfs,
)
from audio_modules.audioflux_adapter import AudioFluxAdapter, AudioFluxAdapterConfig
from audio_modules.event_detector import EventDetector, EventDetectorConfig
from audio_modules.auto_ranging import AutoRanging
from audio_modules.beat_detector import BeatDetector
from audio_modules.syncopation import SyncopationDetector
from audio_modules.metronome import MetronomeController
from audio_modules.signal_frontend import SignalFrontend, SignalFrontendConfig
from audio_modules.audio_io import AudioIOController
from audio_modules.tempo_tracker import (
    TempoTracker,
    TempoTrackerConfig,
)
from audio_modules.session_stats import SessionStats
from audio_modules.volume_normalizer import VolumeNormalizer

# Scipy for runtime Butterworth filtering in callback
try:
    from scipy.signal import sosfilt
except ImportError:
    sosfilt = None
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
        self.peak_envelope = 0.0
        self.flux_history: list[float] = []
        self.energy_history: list[float] = []
        self._last_beat_time: float = 0.0
        
        # Spectrum data for visualization
        self.spectrum_data: Optional[np.ndarray] = None
        self.spectrum_lock = threading.Lock()
        self.waveform_data: Optional[np.ndarray] = None
        self.waveform_lock = threading.Lock()
        
        # FFT settings (from config with fallback)
        self.fft_size = int(getattr(config.audio, 'fft_size', 1024) or 1024)
        self.hop_size = max(1, self.fft_size // 4)  # Typical hop = 25% of FFT size
        
        self._frame_counter = 0  # For spectrum skip optimization
        self._spectrum_skip_frames = getattr(config.audio, 'spectrum_skip_frames', 2)
        self._signal_frontend = SignalFrontend(
            SignalFrontendConfig(
                sample_rate=int(self.config.audio.sample_rate),
                channels=int(self.config.audio.channels),
                gain=float(self.config.audio.gain),
                fft_size=int(self.fft_size),
                hop_size=int(self.hop_size),
                freq_low=float(self.config.beat.freq_low),
                freq_high=float(self.config.beat.freq_high),
                flux_multiplier=float(self.config.beat.flux_multiplier),
            )
        )
        
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
        
        # Volume normalization (compensates for Windows master volume).
        # Only meaningful for WASAPI loopback capture where the captured PCM is
        # scaled by the Windows endpoint volume.  For analog/line-in devices the
        # signal is independent of the Windows volume slider, so applying
        # compensation would incorrectly inflate the already-full-amplitude signal
        # and break silence detection.
        _is_loopback = bool(getattr(config.audio, 'is_loopback', True))
        self._volume_normalizer = VolumeNormalizer(
            enabled=(
                bool(getattr(config.audio, 'volume_normalize', True))
                and platform.system().lower() == 'windows'
                and _is_loopback
            )
        )
        
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
        
        self._auto_ranging = AutoRanging(config)
        self._beat_detector = BeatDetector(config)
        self.energy_history = self._beat_detector.energy_history
        self.flux_history = self._beat_detector.flux_history
        
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
        # Layer-2 silence gate: set by BeatIntelligence each frame.
        # When True, beat detection, metronome, ACF feeding, and
        # syncopation are all suppressed.  Starts True (guilty-until-
        # proven-innocent) so the first frames before BeatIntelligence
        # runs cannot leak phantom beats.
        self.silence_gate_active: bool = True
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
        self._metronome_bpm_alpha_slow: float = float(getattr(config.beat, 'metronome_bpm_alpha_slow', 0.08))
        self._metronome_bpm_alpha_fast: float = float(getattr(config.beat, 'metronome_bpm_alpha_fast', 0.40))
        self._metronome_pll_window: float = float(getattr(config.beat, 'metronome_pll_window', 0.35))
        self._metronome_pll_base_gain: float = float(getattr(config.beat, 'metronome_pll_base_gain', 0.25))
        self._metronome_pll_conf_gain: float = float(getattr(config.beat, 'metronome_pll_conf_gain', 0.18))
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
        self._syncopation = SyncopationDetector(config)
        self._syncopation_detected: bool = self._syncopation.detected
        self._syncopation_window: float = self._syncopation.window
        self._any_band_onset: bool = self._syncopation.any_band_onset
        self._syncopation_streak: int = self._syncopation.streak
        self._syncopation_had_offbeat: bool = self._syncopation.had_offbeat
        self._syncopation_confirmed: bool = self._syncopation.confirmed
        self._syncopation_armed: bool = self._syncopation.armed
        self._new_trigger_fusion_enabled: bool = bool(getattr(config.beat, 'new_trigger_fusion_enabled', False))
        self._new_trigger_telemetry_enabled: bool = bool(getattr(config.beat, 'new_trigger_telemetry_enabled', True))
        self._new_trigger_shadow_mode: bool = bool(getattr(config.beat, 'new_trigger_shadow_mode', True))
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

        self._session_stats = SessionStats(
            new_trigger_telemetry_enabled=self._new_trigger_telemetry_enabled,
        )
        self._tempo_tracker = TempoTracker(TempoTrackerConfig(enabled=True))
        self._metronome_controller = MetronomeController(self)
        self._audio_io_controller = AudioIOController(self)

    def _metronome_ctrl(self) -> MetronomeController:
        controller = getattr(self, '_metronome_controller', None)
        if controller is None:
            controller = MetronomeController(self)
            self._metronome_controller = controller
        return controller

    def _audio_io_ctrl(self) -> AudioIOController:
        controller = getattr(self, '_audio_io_controller', None)
        if controller is None:
            controller = AudioIOController(self)
            self._audio_io_controller = controller
        return controller

    def _sync_tempo_tracker_state(self, tempo_locked: bool, is_downbeat: bool) -> None:
        self._metronome_ctrl().sync_tempo_tracker_state(tempo_locked, is_downbeat)

    def _compute_tempo_lock_state(self, acf_confidence: float, downbeat_matches: int, now: float) -> bool:
        return self._metronome_ctrl().compute_tempo_lock_state(acf_confidence, downbeat_matches, now)

    def _reset_session_stats(self) -> None:
        self._session_stats.reset()
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
        self._session_stats.record_shadow_telemetry(
            legacy_fire=legacy_fire,
            current_time=current_time,
            decision=decision,
            acf_bpm=self._acf_bpm_smoothed,
            acf_confidence=self._acf_confidence,
            phase_error_ms=self.phase_error_ms,
            smoothing_tag=self._last_acf_smoothing_tag,
            frontend_ms=frontend_ms,
            tempo_ms=tempo_ms,
            detector_ms=detector_ms,
            sidecar_ms=sidecar_ms,
        )

    def _reference_bpm_for_onset_filters(self) -> float:
        return self._metronome_ctrl().reference_bpm_for_onset_filters()

    def _effective_phase_accept_window_s(self) -> float:
        return self._metronome_ctrl().effective_phase_accept_window_s()

    def _is_raw_onset_acceptable(self, now: float) -> bool:
        return self._metronome_ctrl().is_raw_onset_acceptable(now)

    def _update_session_stats(
        self,
        raw_rms_db: float,
        band_energy: float,
        spectral_flux: float,
        peak_level: float,
        sample_time: float,
    ) -> None:
        self._session_stats.update(
            raw_rms_db=raw_rms_db,
            band_energy=band_energy,
            spectral_flux=spectral_flux,
            peak_level=peak_level,
            sample_time=sample_time,
        )

    def _compute_persistence_stats(
        self,
        values: list[float],
        sample_times: list[float],
        threshold: float,
        is_high: bool,
    ) -> dict[str, float]:
        return self._session_stats._compute_persistence_stats(
            values=values,
            sample_times=sample_times,
            threshold=threshold,
            is_high=is_high,
        )

    def _session_summary_payload(self, elapsed_s: float) -> dict:
        return self._session_stats.summary_payload(elapsed_s)

    def _log_shutdown_summary(self) -> None:
        self._session_stats.log_shutdown_summary()

    def _init_butterworth_filter(self):
        self._audio_io_ctrl().init_butterworth_filter()
        
    def start(self) -> None:
        self._audio_io_ctrl().start()
    
    def _start_loopback_capture(self, device_index=None):
        self._audio_io_ctrl().start_loopback_capture(device_index)
    
    def _start_input_capture(self, device_index):
        self._audio_io_ctrl().start_input_capture(device_index)

        
    def stop(self) -> None:
        self._audio_io_ctrl().stop()
    
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
        
        # Compensate for Windows master volume so ALL processing (beat detection
        # AND silence detection) sees "100%-equivalent" signal levels.
        # With normalization active, absolute dBFS thresholds work reliably
        # regardless of the Windows volume slider position.
        vol_gain = self._volume_normalizer.get_compensation_gain()
        if vol_gain != 1.0:
            mono = mono * vol_gain
        
        # Compute raw RMS from the COMPENSATED mono.  With volume normalization
        # the signal is always at consistent levels, so the silence gate can use
        # simple fixed dBFS thresholds instead of complex adaptive logic.
        raw_rms = float(np.sqrt(np.mean(mono ** 2)))
        raw_rms_db = rms_to_dbfs(raw_rms)
        
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
        self._signal_frontend.configure_runtime(
            sample_rate=int(self.config.audio.sample_rate),
            channels=int(self.config.audio.channels),
            gain=float(self.config.audio.gain),
            fft_size=int(self.fft_size),
            hop_size=int(self.hop_size),
            freq_low=float(self.config.beat.freq_low),
            freq_high=float(self.config.beat.freq_high),
            flux_multiplier=float(self.config.beat.flux_multiplier),
        )
        frontend_frame = self._signal_frontend.process_dual(
            mono=np.asarray(mono, dtype=np.float32),
            beat_mono=np.asarray(beat_mono, dtype=np.float32),
            mono_time=time.perf_counter(),
            wall_time=time.time(),
            use_filtered_band=bool(self._butter_sos is not None),
        )
        if frontend_frame is None:
            return (in_data, pyaudio.paContinue)

        frontend_ms = (time.perf_counter() - callback_started) * 1000.0

        spectrum = frontend_frame.spectrum
        band_energy = frontend_frame.band_energy
        spectral_flux = frontend_frame.spectral_flux
        
        # Store full spectrum for visualization (only on scheduled frames, if enabled)
        if update_spectrum_viz:
            with self.spectrum_lock:
                self.spectrum_data = spectrum.copy()
            with self.waveform_lock:
                self.waveform_data = mono.astype(np.float32, copy=True)

        # raw_rms / raw_rms_db already computed above from the volume-compensated
        # mono signal — consistent levels regardless of Windows volume setting.
        
        # Note: Audio gain already applied to band_spectrum above, no need to apply again
        
        # ===== MULTI-BAND ENERGY EXTRACTION =====
        # Extract energy per sub-band from the full unfiltered spectrum,
        # feed each to its z-score detector, and track which band fires.
        self._update_multiband_zscore(spectrum)

        # Wider-band onset: did ANY z-score band fire? (for syncopation detection)
        # Respects config: 'any' = any band, or a specific band name
        sync_band = self.config.beat.syncopation_band
        self._any_band_onset = self._syncopation.update_any_band_onset(
            self._band_zscore_signals,
            sync_band,
        )
        
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

        # Layer 2: merge BeatIntelligence flatness gate.  If the
        # flatness-based silence gate says "silent", suppress ALL
        # beat / metronome / syncopation output just like the
        # hard -96 dBFS veto.  This prevents phantom beats from
        # noise-floor energy triggering the adaptive beat detector.
        if self.silence_gate_active:
            silence_veto_active = True

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
        # Skip during silence — feeding silence frames into ACF seeds
        # ghost tempos that free-run the metronome on noise.
        if not silence_veto_active and current_time - self._last_acf_time > self._acf_interval_ms / 1000.0:
            self._last_acf_time = current_time
            self._estimate_tempo_acf()
        
        # Raw beat detection candidate (ownership selected later)
        raw_is_beat = False if silence_veto_active else self._detect_beat(band_energy, spectral_flux)
        
        # Advance internal metronome (pass band_energy for energy-based downbeat detection)
        # Skip entirely during silence so the metronome doesn't free-run
        # and produce phantom beat ticks that drive the orbit engine.
        if not silence_veto_active:
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
        combo_texture = float(getattr(self.config.stroke, 'combo_texture', 1.0) or 1.0)
        sync_window = float(getattr(self.config.beat, 'syncopation_window', self._syncopation.window) or self._syncopation.window)
        if np.isfinite(sync_window):
            self._syncopation.window = float(np.clip(sync_window, 0.05, 0.45))
        self._syncopation_detected = self._syncopation.process_frame(
            silence_veto_active=bool(silence_veto_active),
            sync_enabled=bool(self.config.beat.syncopation_enabled),
            metronome_bpm=float(self._metronome_bpm),
            metronome_phase=float(self._metronome_phase),
            metronome_beat_fired=bool(self._metronome_beat_fired),
            bpm_limit=float(self.config.beat.syncopation_bpm_limit),
            combo_texture=combo_texture,
        )
        self._syncopation_had_offbeat = self._syncopation.had_offbeat
        self._syncopation_streak = self._syncopation.streak
        self._syncopation_confirmed = self._syncopation.confirmed
        self._syncopation_armed = self._syncopation.armed

        # Predictive drop-off: if we're past the off-beat window (phase > 0.65)
        # and no off-beat onset was detected this beat period, preemptively
        # reset streak so the NEXT beat won't produce a false syncopation.
        self._syncopation.predictive_dropoff(
            metronome_bpm=float(self._metronome_bpm),
            metronome_phase=float(self._metronome_phase),
            metronome_beat_fired=bool(self._metronome_beat_fired),
            combo_texture=combo_texture,
        )
        self._syncopation_streak = self._syncopation.streak
        self._syncopation_confirmed = self._syncopation.confirmed
        self._syncopation_armed = self._syncopation.armed
        
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
        self._metronome_ctrl().estimate_tempo_acf()

    def _estimate_onset_bpm(self) -> float:
        return self._metronome_ctrl().estimate_onset_bpm()

    def _advance_metronome(self, now: float, band_energy: float = 0.0):
        self._metronome_ctrl().advance_metronome(now, band_energy)

    def _nudge_metronome_phase(self, onset_strength: float):
        self._metronome_ctrl().nudge_metronome_phase(onset_strength)

    def _reset_acf_metronome(self):
        self._metronome_ctrl().reset_acf_metronome()

    def _detect_beat(self, energy: float, flux: float) -> bool:
        """Detect if current frame is a beat.
        
        Uses a two-path system:
          Path 1 (classic): peak_floor + sensitivity + rise checks + threshold
          Path 2 (z-score): adaptive rolling-mean detector fires on +1 signal
        
        A beat is detected if EITHER path triggers (after refractory guard).
        Z-score adapts automatically to any audio level, so it catches beats
        that the manual peak_floor setting would miss - and vice-versa.
        """
        # Layer 3: absolute energy floor.  Band energy this low is pure
        # noise — the adaptive threshold can chase it down to zero during
        # silence, so we need a hard floor that never yields.
        # 0.001 ≈ -60 dBFS band energy.  Real music beats are 10-100x higher.
        if hasattr(self, '_auto_ranging'):
            self._auto_ranging.observe_energy_for_valley(energy)
        else:
            if energy > self._prev_energy_for_valley and self._energy_was_falling:
                valley_val = self._prev_energy_for_valley
                if valley_val > 0.001:
                    self._valley_history.append(valley_val)
                    if len(self._valley_history) > self._valley_max_samples:
                        self._valley_history.pop(0)
            self._energy_was_falling = energy < self._prev_energy_for_valley
            self._prev_energy_for_valley = energy

        if hasattr(self, '_beat_detector'):
            current_time = time.perf_counter()
            result = self._beat_detector.detect(
                energy=energy,
                flux=flux,
                now=current_time,
                primary_band=self._primary_beat_band,
                band_zscore_signals=self._band_zscore_signals,
                metronome_bpm=float(self._metronome_bpm),
                fallback_bpm=float(getattr(self, 'current_bpm', 120.0) or 120.0),
            )
            self.energy_history = self._beat_detector.energy_history
            self.flux_history = self._beat_detector.flux_history
            self._last_beat_time = self._beat_detector.last_beat_time

            if result.is_beat:
                self._update_tempo_tracking(result.detected_at, energy)
                band_info = f"band={self._primary_beat_band}"
                if result.fired_bands and result.source in ('Z', 'Z+C'):
                    band_info += f" fired={','.join(result.fired_bands)}"
                log_event(
                    "INFO",
                    "BEAT",
                    f"Beat detected [{result.source}]",
                    energy=f"{energy:.4f}",
                    threshold=f"{result.energy_threshold:.4f}",
                    flux=f"{flux:.4f}",
                    bpm=f"{self.smoothed_tempo:.1f}",
                    bands=band_info
                )
            return bool(result.is_beat)

        # Legacy fallback path (for __new__ tests that bypass __init__).
        _BEAT_ENERGY_FLOOR = 0.001
        if energy < _BEAT_ENERGY_FLOOR:
            return False
        cfg = self.config.beat
        primary = self._primary_beat_band
        zscore_signal = self._band_zscore_signals.get(primary, 0)
        zscore_peak = (zscore_signal == 1)
        self.energy_history.append(energy)
        self.flux_history.append(flux)
        max_history = 50
        self.energy_history = self.energy_history[-max_history:]
        self.flux_history = self.flux_history[-max_history:]
        if len(self.energy_history) < 5:
            return False
        if not hasattr(self, '_last_beat_time'):
            self._last_beat_time = 0
        current_time = time.perf_counter()
        beat_refractory_ms = float(getattr(self.config.beat, 'beat_refractory_ms', 170.0) or 170.0)
        beat_refractory_ms = float(np.clip(beat_refractory_ms, 80.0, 600.0))
        if self._metronome_bpm > 0:
            beat_period_ms = 60000.0 / max(1.0, float(self._metronome_bpm))
        else:
            beat_period_ms = 60000.0 / max(1.0, float(getattr(self, 'current_bpm', 120.0) or 120.0))
        refractory_ms = min(beat_refractory_ms, beat_period_ms * 0.7)
        refractory_s = refractory_ms / 1000.0
        if current_time - self._last_beat_time < refractory_s:
            return False
        avg_energy = np.mean(self.energy_history)
        avg_flux = np.mean(self.flux_history)
        threshold_mult = 2.0 - (cfg.sensitivity * 0.7)
        energy_threshold = avg_energy * threshold_mult
        flux_threshold = avg_flux * threshold_mult
        classic_beat = False
        passes_floor = (cfg.peak_floor <= 0) or (energy >= cfg.peak_floor)
        if passes_floor:
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
                else:
                    classic_beat = (energy > energy_threshold) or (flux > flux_threshold * 1.2)
        any_band_fired = any(s == 1 for s in self._band_zscore_signals.values())
        zscore_beat = (zscore_peak or any_band_fired) and (energy > avg_energy * 1.1)
        is_beat = classic_beat or zscore_beat
        if is_beat:
            self._last_beat_time = current_time
            self._update_tempo_tracking(current_time, energy)
            src = "Z+C" if (classic_beat and zscore_beat) else ("Z" if zscore_beat else "C")
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
        self._metronome_ctrl().update_tempo_tracking(current_time, energy)
    
    def _predict_next_beat(self, current_time: float, current_wall_time: float = 0.0):
        self._metronome_ctrl().predict_next_beat(current_time, current_wall_time)
    
    def _validate_downbeat_against_pattern(self, current_time: float, use_bpm: float = 0.0) -> bool:
        return self._metronome_ctrl().validate_downbeat_against_pattern(current_time, use_bpm)
        
    def _reset_downbeat_pattern(self):
        self._metronome_ctrl().reset_downbeat_pattern()
        
    def get_tempo_info(self) -> dict:
        return self._metronome_ctrl().get_tempo_info()
            
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

    def get_band_energies(self) -> dict[str, float]:
        """Public accessor for current multi-band energies."""
        energies = getattr(self, '_band_energies', None)
        if not isinstance(energies, dict):
            return {}
        return {
            'sub_bass': float(energies.get('sub_bass', 0.0) or 0.0),
            'low_mid': float(energies.get('low_mid', 0.0) or 0.0),
            'mid': float(energies.get('mid', 0.0) or 0.0),
            'high': float(energies.get('high', 0.0) or 0.0),
        }

    def set_silence_gate(self, active: bool) -> None:
        """Public control signal for cross-module silence gating."""
        self.silence_gate_active = bool(active)

    def set_spectrum_skip_frames(self, value: int) -> None:
        """Set visual spectrum decimation cadence used by the callback."""
        self._spectrum_skip_frames = max(1, int(value))

    def set_aggressive_tempo_snap_enabled(self, enabled: bool) -> None:
        """Enable/disable confidence-gated aggressive metronome snapping."""
        self._aggressive_tempo_snap_enabled = bool(enabled)

    def reinitialize_butterworth_filter(self) -> None:
        """Rebuild Butterworth filter coefficients using current config."""
        self._init_butterworth_filter()

    def get_waveform(self) -> Optional[np.ndarray]:
        """Get current waveform frame for visualization."""
        with self.waveform_lock:
            return self.waveform_data.copy() if self.waveform_data is not None else None
    
    # ===== REAL-TIME METRIC FEEDBACK SYSTEM =====
    
    def enable_metric_autoranging(self, metric: str, enable: bool = True):
        """Enable/disable a specific metric-based auto-ranging metric."""
        self._auto_ranging.enable_metric_autoranging(metric, enable)

    
    def compute_energy_margin_feedback(self, band_energy: float, callback=None):
        return self._auto_ranging.compute_energy_margin_feedback(
            band_energy=band_energy,
            peak_floor=float(self.config.beat.peak_floor),
            audio_gain=float(self.config.audio.gain),
            callback=callback,
        )

    def set_metric_response_speed(self, speed: float):
        """Set auto-range response speed (1.0=legacy, >1 faster, <1 slower)."""
        self._auto_ranging.set_metric_response_speed(speed)

    def _effective_metric_speed(self) -> float:
        return self._auto_ranging._effective_metric_speed()

    def _scaled_metric_interval_s(self, interval_ms: float) -> float:
        return self._auto_ranging._scaled_metric_interval_s(interval_ms)

    def _scaled_metric_step(self, base_step: float) -> float:
        return self._auto_ranging._scaled_metric_step(base_step)

    def _effective_metric_hysteresis_required(self) -> int:
        return self._auto_ranging._effective_metric_hysteresis_required()

    def _effective_metric_settled_threshold(self) -> int:
        return self._auto_ranging._effective_metric_settled_threshold()

    # ===== TIMER-DRIVEN METRIC FEEDBACK (audio_amp) =====
    # These are called from main.py's _update_display timer, NOT from _on_beat,
    # because they need to detect the ABSENCE of beats.

    def get_metric_states(self) -> dict[str, str]:
        """Return current state of each enabled auto-ranging metric."""
        return self._auto_ranging.get_metric_states()

    def compute_audio_amp_feedback(self, now: float, callback=None):
        self._auto_ranging.compute_audio_amp_feedback(
            now=now,
            last_beat_time=float(self.last_beat_time),
            beat_times=self.beat_times,
            callback=callback,
        )




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
