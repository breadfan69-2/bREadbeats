"""
bREadbeats - Audio Engine
Captures system audio and detects beats using spectral flux / peak energy.
Uses pyaudiowpatch for WASAPI loopback capture.
"""

import numpy as np
import pyaudiowpatch as pyaudio
import threading
import queue
from dataclasses import dataclass
from typing import Callable, Optional
import time

try:
    import aubio
    HAS_AUBIO = True
except ImportError:
    HAS_AUBIO = False
    print("[AudioEngine] Warning: aubio not found, using fallback beat detection")

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


class AudioEngine:
    def reset_tempo_tracking(self):
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
        
    def start(self):
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
                
        except Exception as e:
            print(f"[AudioEngine] Failed to start: {e}")
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
        
        print(f"[AudioEngine] Using WASAPI loopback: {device_info['name']}")
        print(f"[AudioEngine] Channels: {device_info['maxInputChannels']}, SR: {int(device_info['defaultSampleRate'])}Hz")
        
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
        print("[AudioEngine] WASAPI loopback capture started successfully!")
    
    def _start_input_capture(self, device_index):
        """Start regular input capture (microphone)"""
        if device_index is None:
            # Find default input device
            wasapi_info = self.pyaudio.get_host_api_info_by_type(pyaudio.paWASAPI)
            device_index = wasapi_info.get("defaultInputDevice", 0)
        
        device_info = self.pyaudio.get_device_info_by_index(device_index)
        
        print(f"[AudioEngine] Using input device: {device_info['name']}")
        print(f"[AudioEngine] Channels: {device_info['maxInputChannels']}, SR: {int(device_info['defaultSampleRate'])}Hz")
        
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
        print("[AudioEngine] Input capture started successfully!")

        
    def stop(self):
        """Stop audio capture"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None
        print("[AudioEngine] Stopped")
    
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
        
        # Frame skip optimization - only update spectrum visualization every N frames
        self._frame_counter += 1
        update_spectrum = (self._frame_counter % self._spectrum_skip_frames == 0)
        
        # Pre-allocate Hanning window on first use (or if size changed)
        if self._hanning_window is None or len(self._hanning_window) != len(mono):
            self._hanning_window = np.hanning(len(mono)).astype(np.float32)
        
        # Compute FFT with pre-allocated window
        windowed = mono * self._hanning_window
        spectrum = np.abs(np.fft.rfft(windowed))
        
        # Store full spectrum for visualization (only on scheduled frames)
        if update_spectrum:
            with self.spectrum_lock:
                self.spectrum_data = spectrum.copy()
        
        # Filter spectrum to selected frequency band for beat detection
        band_spectrum = self._filter_frequency_band(spectrum)
        
        # Apply audio gain amplification (for weak devices like Stereo Mix)
        band_spectrum = band_spectrum * self.config.audio.gain
        
        # Compute beat detection metrics on filtered band
        band_energy = np.sqrt(np.mean(band_spectrum ** 2)) if len(band_spectrum) > 0 else 0
        spectral_flux = self._compute_spectral_flux(band_spectrum)
        
        # Debug: print every 20 frames to see levels
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 20 == 0:
            # Log raw audio level too
            raw_rms = np.sqrt(np.mean(mono ** 2))
            full_spectrum_energy = np.sqrt(np.mean(spectrum ** 2)) if len(spectrum) > 0 else 0
            print(f"[Audio] raw_rms={raw_rms:.6f} full_spectrum={full_spectrum_energy:.6f} band_energy={band_energy:.6f} flux={spectral_flux:.4f} peak_env={self.peak_envelope:.6f}")
        
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
            print(f"[Tempo] No beats for {time_since_last_beat:.0f}ms - resetting tempo tracker (keeping BPM={self.smoothed_tempo:.1f})")
            self.last_known_tempo = self.smoothed_tempo  # Preserve current tempo
            self.beat_intervals.clear()
            self.beat_times.clear()
            self.beat_position_in_measure = 0
            self.is_downbeat = False
            tempo_reset_flag = True
        
        # Detect beat based on mode (using band energy)
        is_beat = self._detect_beat(band_energy, spectral_flux)
        
        # Estimate dominant frequency
        freq = self._estimate_frequency(spectrum)
        
        # Create beat event using correct structure
        # Use last_known_tempo if smoothed_tempo was reset
        current_bpm = self.smoothed_tempo if self.smoothed_tempo > 0 else self.last_known_tempo
        
        event = BeatEvent(
            timestamp=time.time(),
            intensity=min(1.0, band_energy / max(0.0001, self.peak_envelope)),
            frequency=freq,
            is_beat=is_beat,
            spectral_flux=spectral_flux,
            peak_energy=band_energy,
            is_downbeat=self.is_downbeat if is_beat else False,  # Only downbeat if it's an actual beat
            bpm=current_bpm,
            tempo_reset=tempo_reset_flag
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
            print(f"[Beat] energy={energy:.4f} (thresh={energy_threshold:.4f}) flux={flux:.4f} bpm={self.smoothed_tempo:.1f}")
        
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
                    print(f"[Tempo] Auto-halved BPM: {60.0/interval:.1f} -> {60.0/adjusted_interval:.1f}")
                
                # Auto-double: if BPM < 60, halve the interval (double BPM)
                while 60.0 / adjusted_interval < min_bpm and adjusted_interval > 0.1:
                    adjusted_interval /= 2
                    print(f"[Tempo] Auto-doubled BPM: {60.0/interval:.1f} -> {60.0/adjusted_interval:.1f}")
                
                interval = adjusted_interval
            
            # Reject intervals still outside reasonable range after adjustment
            if interval < 0.15 or interval > 2.0:
                print(f"[Tempo] Interval rejected: {interval:.3f}s (BPM would be {60.0/interval:.1f})")
                return
            if interval > 0.2:
                # Outlier rejection: if interval is way off from average, it might be a false beat
                if len(self.beat_intervals) > 0:
                    avg_interval = np.mean(self.beat_intervals)
                    # Accept if within 0.5x to 2.0x of average (allows tempo changes but rejects glitches)
                    if interval < (0.5 * avg_interval) or interval > (2.0 * avg_interval):
                        print(f"[Tempo] Outlier interval rejected: {interval:.3f}s (avg: {avg_interval:.3f}s)")
                        return
                
                # Phase snap: if we have a stable tempo, nudge detected interval toward predicted
                # This helps lock onto tempo even with slightly off-beat detections
                if self.smoothed_tempo > 0 and self.phase_snap_weight > 0 and self.beat_stability > 0.3:
                    predicted_interval = 60.0 / self.smoothed_tempo
                    # Only snap if the detection is reasonably close (within 20% of predicted)
                    if abs(interval - predicted_interval) / predicted_interval < 0.2:
                        old_interval = interval
                        interval = interval * (1 - self.phase_snap_weight) + predicted_interval * self.phase_snap_weight
                        print(f"[Tempo] Phase snap: {old_interval:.3f}s -> {interval:.3f}s (predicted: {predicted_interval:.3f}s)")
                
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
                        print(f"[Tempo] Stable BPM committed: {self.stable_tempo:.1f} (CV={cv:.3f}, stability={self.beat_stability:.2f})")
                    else:
                        print(f"[Tempo] BPM unstable: {self.smoothed_tempo:.1f} (CV={cv:.3f}, stability={self.beat_stability:.2f})")
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
                self.is_downbeat = (pos_idx == self.downbeat_position) and total_beats >= self.beats_per_measure * 2
                
                if self.is_downbeat:
                    print(f"[Downbeat] Position {pos_idx+1}/{self.beats_per_measure} (confidence={self.downbeat_confidence:.2f}, energies={[f'{e:.2f}' for e in avg_energies]})")
        
        self.last_beat_time = current_time
    
    def _predict_next_beat(self, current_time: float):
        """Predict the time of the next beat based on smoothed tempo"""
        if self.smoothed_tempo > 0:
            predicted_interval = 60.0 / self.smoothed_tempo
            self.predicted_next_beat = current_time + predicted_interval
        
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
            'stability': self.beat_stability  # 0.0 = chaotic, 1.0 = perfectly stable
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
            
    def list_devices(self) -> list[dict]:
        """List available audio devices"""
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "inputs": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]


# Test
if __name__ == "__main__":
    from config import Config
    
    def on_beat(event: BeatEvent):
        if event.is_beat:
            print(f"BEAT! intensity={event.intensity:.2f} freq={event.frequency:.0f}Hz")
            
    config = Config()
    engine = AudioEngine(config, on_beat)
    
    print("Available devices:")
    for d in engine.list_devices():
        print(f"  [{d['index']}] {d['name']} ({d['inputs']} ch)")
        
    print("\nStarting audio capture (Ctrl+C to stop)...")
    engine.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        engine.stop()
