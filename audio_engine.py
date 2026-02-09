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

try:
    import aubio
    HAS_AUBIO = True
except ImportError:
    HAS_AUBIO = False
    log_event("WARN", "AudioEngine", "aubio not found, using fallback beat detection")

from config import Config, BeatDetectionType


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
        
        # Aubio beat tracker (if available)
        self.tempo_detector = None
        self.beat_detector = None
        
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
        self.measure_beat_counts: list[int] = [0] * self.beats_per_measure       # How many beats at each position
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
        
        # ===== REAL-TIME METRIC-BASED AUTO-RANGING (NEW SYSTEM) =====
        # Tracks margins and metrics in real-time to drive parameter adjustments
        # No timer cycle - pure feedback-based optimization
        
        # Metric 1: Peak Floor Feedback (Energy Margin)
        self._metric_peak_floor_enabled: bool = False  # User toggle
        self._energy_margin_history: list[float] = []  # Last 16 margins
        self._energy_margin_target_low: float = 0.02   # Optimal zone: 0.02-0.05
        self._energy_margin_target_high: float = 0.05
        self._energy_margin_adjustment_step: float = 0.002  # Step size per beat
        
        # Metric 2: Sensitivity Feedback (No Downbeat → raise, Excess Downbeats → lower)
        self._metric_sensitivity_enabled: bool = False
        self._sensitivity_check_interval_ms: float = 1100.0  # Check every ~1100ms
        self._sensitivity_escalate_pct: float = 0.04   # 4% of range per check
        self._last_downbeat_time: float = 0.0           # Tracked from main.py
        self._last_sensitivity_check: float = 0.0       # Last time we checked
        self._downbeat_times: list[float] = []          # Recent downbeat timestamps
        self._downbeat_window_seconds: float = 4.0      # Rolling window for DPS calc
        
        # Metric 3: Downbeat Energy Ratio Feedback (stub)
        self._metric_downbeat_ratio_enabled: bool = False
        self._downbeat_ratio_target_low: float = 1.8    # Optimal zone: 1.8-2.2
        self._downbeat_ratio_target_high: float = 2.2
        
        # Metric 4: Audio Amp Feedback (No Beats → raise, Excess Beats → lower)
        self._metric_audio_amp_enabled: bool = False
        self._audio_amp_check_interval_ms: float = 1100.0  # Check every ~1100ms
        self._audio_amp_escalate_pct: float = 0.02     # 2% of range per check
        self._last_audio_amp_check: float = 0.0         # Last time we checked
        
        # ===== TARGET BPS SYSTEM (Beats Per Second) =====
        # Tracks actual beats per second and adjusts parameters to achieve target rate
        self._target_bps_enabled: bool = False          # User toggle
        self._target_bps: float = 1.5                   # Target beats per second (default 90 BPM)
        self._target_bps_tolerance: float = 0.2         # ± tolerance (0.2 = accept 1.3-1.7 BPS if target is 1.5)
        self._bps_window_seconds: float = 4.0           # Rolling window for BPS calculation
        self._bps_beat_times: list[float] = []          # Timestamps of recent beats
        self._bps_adjustment_speed: float = 0.5         # 0.0=fine, 1.0=aggressive (scales step size)
        self._bps_base_step: float = 0.002              # Base step for peak_floor adjustment
        
        # ===== AUTO-FREQUENCY BAND TRACKING (Consistency-Based) =====
        # Stores band energy history to find bands with consistent peaks/valleys
        self._band_energy_history: dict[int, list[float]] = {}  # center_bin -> [energy samples]
        self._band_history_max_samples: int = 32                # Keep last 32 samples per band
        self._last_spectrum_time: float = 0.0                   # For timing consistency checks
        
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
            
        self.running = True
        
        # Initialize aubio if available
        if HAS_AUBIO:
            self.tempo_detector = aubio.tempo(
                "default", 
                self.fft_size, 
                self.hop_size, 
                self.config.audio.sample_rate
            )
            self.beat_detector = aubio.onset(
                "default",
                self.fft_size,
                self.hop_size,
                self.config.audio.sample_rate
            )
            self.beat_detector.set_threshold(self.config.beat.sensitivity)
        
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
        
        # Store full spectrum for visualization (only on scheduled frames, if enabled)
        if update_spectrum_viz:
            with self.spectrum_lock:
                self.spectrum_data = spectrum.copy()
        
        # For beat detection: use Butterworth filtered signal if available, else FFT band filter
        if self._butter_sos is not None:
            # Use time-domain energy from Butterworth filtered signal
            band_energy = np.sqrt(np.mean(beat_mono ** 2))
            band_energy = band_energy * self.config.audio.gain  # Apply audio gain
            # Still compute spectral flux from filtered signal's spectrum
            beat_windowed = beat_mono * self._hanning_window
            beat_spectrum = np.abs(np.fft.rfft(beat_windowed))
            spectral_flux = self._compute_spectral_flux(beat_spectrum)
        else:
            # Fallback: FFT-based frequency band filtering (spectrum already computed above)
            band_spectrum = self._filter_frequency_band(spectrum)
            band_spectrum = band_spectrum * self.config.audio.gain
            band_energy = np.sqrt(np.mean(band_spectrum ** 2)) if len(band_spectrum) > 0 else 0
            spectral_flux = self._compute_spectral_flux(band_spectrum)
        
        # Note: Audio gain already applied to band_spectrum above, no need to apply again
        
        # Debug: print every 20 frames to see levels
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 20 == 0:
            # Log raw audio level too
            raw_rms = np.sqrt(np.mean(mono ** 2))
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
        is_beat = self._detect_beat(band_energy, spectral_flux)
        
        # Estimate dominant frequency
        freq = self._estimate_frequency(spectrum)
        
        # Create beat event using correct structure
        # Use last_known_tempo if smoothed_tempo was reset
        current_bpm = self.smoothed_tempo if self.smoothed_tempo > 0 else self.last_known_tempo
        
        # Check if tempo is locked (consecutive downbeats matching predicted pattern)
        tempo_is_locked = self.consecutive_matching_downbeats >= self.consecutive_match_threshold
        
        event = BeatEvent(
            timestamp=time.time(),
            intensity=min(1.0, band_energy / max(0.0001, self.peak_envelope)),
            frequency=freq,
            is_beat=is_beat,
            spectral_flux=spectral_flux,
            peak_energy=band_energy,
            is_downbeat=self.is_downbeat if is_beat else False,  # Only downbeat if it's an actual beat
            bpm=current_bpm,
            tempo_reset=tempo_reset_flag,
            tempo_locked=tempo_is_locked,
            phase_error_ms=self.phase_error_ms
        )
        
        # Notify callback
        self.beat_callback(event)
        
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
        
    def _detect_beat(self, energy: float, flux: float) -> bool:
        """Detect if current frame is a beat"""
        cfg = self.config.beat
        
        # Use aubio if available
        if HAS_AUBIO and self.beat_detector:
            # Note: aubio expects specific buffer, this is simplified
            pass
            
        # Fallback: threshold-based detection
        self.energy_history.append(energy)
        self.flux_history.append(flux)
        
        # Keep limited history
        max_history = 50
        self.energy_history = self.energy_history[-max_history:]
        self.flux_history = self.flux_history[-max_history:]
        
        if len(self.energy_history) < 5:
            return False
        
        # Add cooldown to prevent too many beats
        if not hasattr(self, '_last_beat_time'):
            self._last_beat_time = 0
        
        current_time = time.time()
        min_beat_interval = 0.05  # Max 20 beats per second
        if current_time - self._last_beat_time < min_beat_interval:
            return False
            
        # Compute adaptive thresholds
        avg_energy = np.mean(self.energy_history)
        avg_flux = np.mean(self.flux_history)
        
        # Sensitivity now works intuitively: higher = more sensitive (lower threshold)
        # sensitivity 0.0 = need 2x average, sensitivity 1.0 = need 1.1x average
        threshold_mult = 2.0 - (cfg.sensitivity * 0.9)  # Range: 2.0 down to 1.1
        energy_threshold = avg_energy * threshold_mult
        flux_threshold = avg_flux * threshold_mult
        
        # Peak floor - only check if set above 0
        if cfg.peak_floor > 0 and energy < cfg.peak_floor:
            return False
            
        # Rise sensitivity check - configurable now
        # rise_sensitivity 0 = disabled, 1.0 = must rise significantly
        if cfg.rise_sensitivity > 0 and len(self.energy_history) >= 2:
            rise = energy - self.energy_history[-2]
            min_rise = avg_energy * cfg.rise_sensitivity * 0.5
            if rise < min_rise:
                return False
                
        # Detect based on mode
        is_beat = False
        if cfg.detection_type == BeatDetectionType.PEAK_ENERGY:
            is_beat = energy > energy_threshold
        elif cfg.detection_type == BeatDetectionType.SPECTRAL_FLUX:
            is_beat = flux > flux_threshold
        else:  # COMBINED - need EITHER to trigger (more sensitive)
            is_beat = (energy > energy_threshold) or (flux > flux_threshold * 1.2)
        
        if is_beat:
            self._last_beat_time = current_time
            self._update_tempo_tracking(current_time, energy)
            log_event(
                "INFO",
                "BEAT",
                "Beat detected",
                energy=f"{energy:.4f}",
                threshold=f"{energy_threshold:.4f}",
                flux=f"{flux:.4f}",
                bpm=f"{self.smoothed_tempo:.1f}"
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
                
                # Energy-based downbeat detection (StackOverflow/librosa-inspired)
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
                        avg_energies.append(self.measure_energy_accum[i] / max(1, self.measure_beat_counts[i]))
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
                
                # Apply strict pattern matching if enabled
                if is_energy_downbeat and self.downbeat_pattern_enabled and self.smoothed_tempo > 0:
                    pattern_matches = self._validate_downbeat_against_pattern(current_time)
                    self.is_downbeat = pattern_matches
                    
                    if pattern_matches:
                        self.consecutive_matching_downbeats += 1
                        log_event(
                            "INFO",
                            "Downbeat",
                            "Accepted",
                            position=f"{pos_idx+1}/{self.beats_per_measure}",
                            confidence=f"{self.downbeat_confidence:.2f}",
                            consecutive=f"{self.consecutive_matching_downbeats}/{self.consecutive_match_threshold}",
                            error_ms=f"{self.phase_error_ms:.1f}",
                            energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]"
                        )
                    else:
                        self.consecutive_matching_downbeats = 0
                        log_event(
                            "INFO",
                            "Downbeat",
                            "Rejected",
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
                            "Energy downbeat",
                            position=f"{pos_idx+1}/{self.beats_per_measure}",
                            confidence=f"{self.downbeat_confidence:.2f}",
                            energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]"
                        )
        
        self.last_beat_time = current_time
    
    def _predict_next_beat(self, current_time: float):
        """Predict the time of the next beat based on smoothed tempo"""
        if self.smoothed_tempo > 0:
            predicted_interval = 60.0 / self.smoothed_tempo
            self.predicted_next_beat = current_time + predicted_interval
    
    def _validate_downbeat_against_pattern(self, current_time: float) -> bool:
        """
        Validate that a detected downbeat matches the predicted tempo pattern within tolerance.
        
        Strategy:
        1. If this is the first/second downbeat, use it to establish predicted pattern
        2. For subsequent downbeats, check if timing matches predicted downbeat time
        3. Calculate phase error (how far off from prediction)
        4. Return True only if error is within tolerance_ms
        
        Args:
            current_time: Time of the detected downbeat (seconds)
            
        Returns:
            True if downbeat matches predicted pattern, False otherwise
        """
        beat_interval = 60.0 / self.smoothed_tempo  # Seconds between beats
        measure_interval = beat_interval * self.beats_per_measure  # Seconds per measure
        
        # First few downbeats: establish the predicted pattern
        if self.last_predicted_downbeat_time <= 0:
            # Set up the prediction based on this downbeat
            self.last_predicted_downbeat_time = current_time
            self.consecutive_matching_downbeats = 1
            self.phase_error_ms = 0.0
            return True
        
        # Calculate when we predicted this downbeat should occur
        # (one measure after the last predicted downbeat)
        predicted_time = self.last_predicted_downbeat_time + measure_interval
        
        # Calculate phase error in milliseconds
        self.phase_error_ms = (current_time - predicted_time) * 1000.0
        
        # Check if within tolerance
        if abs(self.phase_error_ms) <= self.pattern_match_tolerance_ms:
            # Update prediction for next downbeat
            self.last_predicted_downbeat_time = current_time
            return True
        else:
            # Error exceeds tolerance - reject this downbeat
            return False
        
    def _reset_downbeat_pattern(self):
        """Reset downbeat pattern matching state (call after temp lock expires or on silence)"""
        self.consecutive_matching_downbeats = 0
        self.last_predicted_downbeat_time = 0.0
        self.phase_error_ms = 0.0
        
    def get_tempo_info(self) -> dict:
        """Get current tempo information for UI display"""
        # Use stable_tempo for display if available, otherwise fall back to smoothed
        display_bpm = self.stable_tempo if self.stable_tempo > 0 else self.smoothed_tempo
        return {
            'bpm': display_bpm,
            'raw_bpm': self.smoothed_tempo,
            'stable_bpm': self.stable_tempo,
            'beat_position': self.beat_position_in_measure,
            'is_downbeat': self.is_downbeat,
            'predicted_next_beat': self.predicted_next_beat,
            'interval_count': len(self.beat_intervals),
            'confidence': min(1.0, len(self.beat_intervals) / 4.0),  # Confidence grows with more beats
            'stability': self.beat_stability,  # 0.0 = chaotic, 1.0 = perfectly stable
            'consecutive_matching_downbeats': self.consecutive_matching_downbeats,  # Pattern match counter
            'phase_error_ms': self.phase_error_ms  # How far off from predicted (milliseconds)
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
    
    def find_peak_frequency_band(self, min_freq: float = 30.0, max_freq: float = 2000.0, 
                                  band_width: float = 300.0) -> tuple[float, float, float]:
        """
        Find the most powerful frequency band of specified width within the given range.
        
        Args:
            min_freq: Minimum frequency to search (Hz)
            max_freq: Maximum frequency to search (Hz)
            band_width: Width of the band to find (Hz)
            
        Returns:
            Tuple of (center_freq, low_freq, high_freq) of the most powerful band.
            Returns (0, 0, 0) if no spectrum data available.
        """
        with self.spectrum_lock:
            if self.spectrum_data is None or len(self.spectrum_data) == 0:
                return (0.0, 0.0, 0.0)
            
            spectrum = self.spectrum_data.copy()
        
        sr = self.config.audio.sample_rate
        num_bins = len(spectrum)
        freq_per_bin = sr / (2 * num_bins)
        
        # Convert Hz to bin indices
        min_bin = max(1, int(min_freq / freq_per_bin))
        max_bin = min(num_bins - 1, int(max_freq / freq_per_bin))
        band_bins = max(1, int(band_width / freq_per_bin))
        
        if max_bin - min_bin < band_bins:
            # Range too narrow for the band width
            center = (min_freq + max_freq) / 2
            return (center, min_freq, max_freq)
        
        # Sliding window: find the band with maximum total energy
        best_energy = 0.0
        best_start_bin = min_bin
        
        for start_bin in range(min_bin, max_bin - band_bins + 1):
            end_bin = start_bin + band_bins
            band_energy = np.sum(spectrum[start_bin:end_bin] ** 2)
            if band_energy > best_energy:
                best_energy = band_energy
                best_start_bin = start_bin
        
        # Convert best bin range back to Hz
        low_freq = best_start_bin * freq_per_bin
        high_freq = (best_start_bin + band_bins) * freq_per_bin
        center_freq = (low_freq + high_freq) / 2
        
        return (center_freq, low_freq, high_freq)
    
    def find_consistent_frequency_band(self, min_freq: float = 30.0, max_freq: float = 22050.0,
                                        band_width: float = 300.0) -> tuple[float, float, float, float]:
        """
        Find the frequency band with most consistent peak/valley variation.
        Looks for bands where amplitude varies regularly (like a drum beat).
        
        Args:
            min_freq: Minimum frequency to search (Hz)
            max_freq: Maximum frequency to search (Hz)
            band_width: Width of the band to evaluate (Hz)
            
        Returns:
            Tuple of (center_freq, low_freq, high_freq, consistency_score).
            Consistency score: 0.0 = chaotic, 1.0 = perfectly consistent variation.
            Returns (0, 0, 0, 0) if no data available.
        """
        import time as time_module
        
        with self.spectrum_lock:
            if self.spectrum_data is None or len(self.spectrum_data) == 0:
                return (0.0, 0.0, 0.0, 0.0)
            spectrum = self.spectrum_data.copy()
        
        now = time_module.time()
        sr = self.config.audio.sample_rate
        num_bins = len(spectrum)
        freq_per_bin = sr / (2 * num_bins)
        
        # Convert Hz to bin indices
        min_bin = max(1, int(min_freq / freq_per_bin))
        max_bin = min(num_bins - 1, int(max_freq / freq_per_bin))
        band_bins = max(1, int(band_width / freq_per_bin))
        
        if max_bin - min_bin < band_bins:
            center = (min_freq + max_freq) / 2
            return (center, min_freq, max_freq, 0.0)
        
        # Step size for scanning (50Hz steps to reduce computation)
        step_bins = max(1, int(50.0 / freq_per_bin))
        
        # Update band energy history and compute consistency scores
        best_score = 0.0
        best_center_bin = min_bin + band_bins // 2
        
        for center_bin in range(min_bin + band_bins // 2, max_bin - band_bins // 2, step_bins):
            start_bin = center_bin - band_bins // 2
            end_bin = center_bin + band_bins // 2
            
            # Current band energy
            band_energy = np.sqrt(np.mean(spectrum[start_bin:end_bin] ** 2))
            
            # Store in history
            if center_bin not in self._band_energy_history:
                self._band_energy_history[center_bin] = []
            history = self._band_energy_history[center_bin]
            history.append(band_energy)
            
            # Keep only last N samples
            if len(history) > self._band_history_max_samples:
                history.pop(0)
            
            # Need at least 8 samples to evaluate consistency
            if len(history) < 8:
                continue
            
            # Compute consistency score:
            # 1. Find peaks and valleys in the history
            # 2. Measure variance of peak-valley heights
            # 3. Lower variance = more consistent = higher score
            
            arr = np.array(history)
            mean_val = np.mean(arr)
            
            # Find local peaks and valleys (simple sign-change detection)
            peaks = []
            valleys = []
            for i in range(1, len(arr) - 1):
                if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                    peaks.append(arr[i])
                elif arr[i] < arr[i-1] and arr[i] < arr[i+1]:
                    valleys.append(arr[i])
            
            # Need at least 3 peaks and 3 valleys for meaningful consistency
            if len(peaks) < 3 or len(valleys) < 3:
                continue
            
            # Peak-valley height variations
            avg_peak = np.mean(peaks)
            avg_valley = np.mean(valleys)
            height = avg_peak - avg_valley
            
            if height < 0.001:  # Nearly flat - not useful for beat detection
                continue
            
            # Consistency = inverse of coefficient of variation of peaks and valleys
            peak_cv = np.std(peaks) / (avg_peak + 1e-6)
            valley_cv = np.std(valleys) / (avg_valley + 1e-6)
            combined_cv = (peak_cv + valley_cv) / 2
            
            # Score: 1.0 when CV is 0, decreasing as CV increases
            # Also weight by height (prefer bands with good amplitude swing)
            consistency = 1.0 / (1.0 + combined_cv * 10)
            height_factor = min(1.0, height / 0.1)  # Normalize height 0-0.1 to 0-1
            score = consistency * 0.7 + height_factor * 0.3  # 70% consistency, 30% height
            
            if score > best_score:
                best_score = score
                best_center_bin = center_bin
        
        # Clean up old band histories (bands we haven't seen recently)
        if len(self._band_energy_history) > 100:
            # Keep only bands we've updated recently
            current_keys = set(range(min_bin + band_bins // 2, max_bin - band_bins // 2, step_bins))
            old_keys = [k for k in self._band_energy_history if k not in current_keys]
            for k in old_keys[:50]:  # Remove up to 50 old keys
                del self._band_energy_history[k]
        
        # Convert best bin to Hz
        low_freq = (best_center_bin - band_bins // 2) * freq_per_bin
        high_freq = (best_center_bin + band_bins // 2) * freq_per_bin
        center_freq = (low_freq + high_freq) / 2
        
        self._last_spectrum_time = now
        return (center_freq, low_freq, high_freq, best_score)
            
    def list_devices(self) -> list[dict]:
        """List available audio devices"""
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "inputs": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
    
    # ===== REAL-TIME METRIC FEEDBACK SYSTEM =====
    
    def enable_metric_autoranging(self, metric: str, enable: bool = True):
        """Enable/disable a specific metric-based auto-ranging metric"""
        if metric == 'peak_floor':
            self._metric_peak_floor_enabled = enable
            if enable:
                self._energy_margin_history.clear()
                log_event("INFO", "MetricAutoRange", "Peak Floor metric enabled")
            else:
                log_event("INFO", "MetricAutoRange", "Peak Floor metric disabled")
        elif metric == 'sensitivity':
            self._metric_sensitivity_enabled = enable
            if enable:
                self._last_sensitivity_check = 0.0
                self._downbeat_times.clear()
                log_event("INFO", "MetricAutoRange", "Sensitivity metric enabled (downbeat-driven)")
            else:
                log_event("INFO", "MetricAutoRange", "Sensitivity metric disabled")
        elif metric == 'downbeat_ratio':
            self._metric_downbeat_ratio_enabled = enable
        elif metric == 'audio_amp':
            self._metric_audio_amp_enabled = enable
            if enable:
                self._last_audio_amp_check = 0.0
                log_event("INFO", "MetricAutoRange", "Audio Amp metric enabled (beat-driven)")
            else:
                log_event("INFO", "MetricAutoRange", "Audio Amp metric disabled")
        elif metric == 'target_bps':
            self._target_bps_enabled = enable
            if enable:
                self._bps_beat_times.clear()
                log_event("INFO", "MetricAutoRange", f"Target BPS enabled (target={self._target_bps:.2f})")
            else:
                log_event("INFO", "MetricAutoRange", "Target BPS disabled")
    
    def compute_energy_margin_feedback(self, band_energy: float, callback=None):
        """
        Compute energy margin metric and return adjustment for peak_floor.
        
        Energy margin = band_energy - peak_floor
        Optimal zone: 0.02-0.05 (peak_floor is 2-5% below current audio)
        
        Returns:
            (margin, should_adjust, adjustment_direction)
            adjustment_direction: +1 to raise floor, -1 to lower floor, 0 no change
        """
        if not self._metric_peak_floor_enabled:
            return 0.0, False, 0
        
        margin = band_energy - self.config.beat.peak_floor
        
        # Track history
        self._energy_margin_history.append(margin)
        if len(self._energy_margin_history) > 16:
            self._energy_margin_history.pop(0)
        
        avg_margin = np.mean(self._energy_margin_history) if self._energy_margin_history else margin
        
        # Determine action
        should_adjust = False
        direction = 0  # 0=no change, +1=raise floor, -1=lower floor
        
        if avg_margin < self._energy_margin_target_low:
            # Margin too tight - LOWER peak_floor to make detection more sensitive (give more headroom)
            should_adjust = True
            direction = -1
        elif avg_margin > self._energy_margin_target_high:
            # Margin too loose - RAISE peak_floor to make detection stricter (reduce headroom)
            should_adjust = True
            direction = +1
        
        if callback and should_adjust:
            callback({
                'metric': 'peak_floor',
                'margin': avg_margin,
                'target_low': self._energy_margin_target_low,
                'target_high': self._energy_margin_target_high,
                'adjustment': direction * self._energy_margin_adjustment_step,
                'direction': 'raise' if direction > 0 else 'lower'
            })
        
        return avg_margin, should_adjust, direction

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

    # ===== TIMER-DRIVEN METRIC FEEDBACK (audio_amp & sensitivity) =====
    # These are called from main.py's _update_display timer, NOT from _on_beat,
    # because they need to detect the ABSENCE of beats/downbeats.

    def record_downbeat(self, timestamp: float):
        """Record a downbeat occurrence for sensitivity metric tracking"""
        self._last_downbeat_time = timestamp
        self._downbeat_times.append(timestamp)
        # Prune old timestamps
        cutoff = timestamp - self._downbeat_window_seconds
        self._downbeat_times = [t for t in self._downbeat_times if t >= cutoff]

    def compute_audio_amp_feedback(self, now: float, callback=None):
        """
        Timer-driven audio_amp adjustment based on beat presence.
        
        - No beats for >check_interval → RAISE audio_amp (+2% of range)
        - ONLY escalates, never reverses (sensitivity handles excess detection)
        
        Called from _update_display (~30fps), but only acts every ~1100ms.
        """
        if not self._metric_audio_amp_enabled:
            return
        
        # Only check every ~1100ms
        if now - self._last_audio_amp_check < self._audio_amp_check_interval_ms / 1000.0:
            return
        self._last_audio_amp_check = now
        
        # Get range for percentage calculation
        from config import BEAT_RANGE_LIMITS
        amp_min, amp_max = BEAT_RANGE_LIMITS['audio_amp']
        amp_range = amp_max - amp_min
        step = amp_range * self._audio_amp_escalate_pct  # 2% of range
        
        # Check time since last beat
        time_since_beat = now - self.last_beat_time if self.last_beat_time > 0 else float('inf')
        target_interval = 1.0 / self._target_bps if self._target_bps > 0 else 0.67
        
        if time_since_beat > target_interval * 3.0:
            # No beats detected for 3x expected interval → RAISE audio_amp
            if callback:
                callback({
                    'metric': 'audio_amp',
                    'adjustment': +step,
                    'direction': 'raise',
                    'reason': f'no beats for {time_since_beat:.1f}s',
                })

    def compute_sensitivity_feedback(self, now: float, callback=None):
        """
        Timer-driven sensitivity adjustment based on downbeat presence.
        
        - No downbeats for >check_interval → RAISE sensitivity (+4% of range)
        - Excess downbeats (>3× expected rate) → LOWER sensitivity
        
        Called from _update_display (~30fps), but only acts every ~1100ms.
        Uses target_bps / 4 as expected downbeat rate (1 downbeat per 4 beats).
        """
        if not self._metric_sensitivity_enabled:
            return
        
        # Only check every ~1100ms
        if now - self._last_sensitivity_check < self._sensitivity_check_interval_ms / 1000.0:
            return
        self._last_sensitivity_check = now
        
        # Get range for percentage calculation
        from config import BEAT_RANGE_LIMITS
        sens_min, sens_max = BEAT_RANGE_LIMITS['sensitivity']
        sens_range = sens_max - sens_min
        step = sens_range * self._sensitivity_escalate_pct  # 4% of range
        
        # Expected downbeat rate = target_bps / 4 (1 downbeat per measure)
        expected_dps = self._target_bps / 4.0
        expected_downbeat_interval = 1.0 / expected_dps if expected_dps > 0 else 2.67
        
        # Calculate actual downbeats per second
        actual_dps = 0.0
        if len(self._downbeat_times) >= 2:
            window_dur = self._downbeat_times[-1] - self._downbeat_times[0]
            if window_dur > 0:
                actual_dps = (len(self._downbeat_times) - 1) / window_dur
        
        # Check time since last downbeat
        time_since_downbeat = now - self._last_downbeat_time if self._last_downbeat_time > 0 else float('inf')
        
        # Tolerance for excess: expected_dps * 3.0 (only reduce if way above expected)
        excess_threshold = expected_dps * 3.0
        
        if time_since_downbeat > expected_downbeat_interval * 3.0:
            # No downbeats for 3x expected interval → RAISE sensitivity
            if callback:
                callback({
                    'metric': 'sensitivity',
                    'adjustment': +step,
                    'direction': 'raise',
                    'reason': f'no downbeats for {time_since_downbeat:.1f}s',
                    'actual_dps': actual_dps,
                })
        elif actual_dps > excess_threshold:
            # Excess downbeats (>3x expected) → LOWER sensitivity
            if callback:
                callback({
                    'metric': 'sensitivity',
                    'adjustment': -step,
                    'direction': 'lower',
                    'reason': f'excess DPS {actual_dps:.2f} > {excess_threshold:.2f}',
                    'actual_dps': actual_dps,
                })


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
