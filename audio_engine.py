"""
bREadbeats - Audio Engine
Captures system audio and detects beats using spectral flux / peak energy.
Uses sounddevice for low-latency capture and aubio for beat detection.
"""

import numpy as np
import sounddevice as sd
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


class AudioEngine:
    """
    Engine 1: The Ears
    Captures system audio and detects beats in real-time.
    """
    
    def __init__(self, config: Config, beat_callback: Callable[[BeatEvent], None]):
        self.config = config
        self.beat_callback = beat_callback
        
        # Audio stream
        self.stream: Optional[sd.InputStream] = None
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
        
        # FFT settings
        self.fft_size = 2048
        self.hop_size = 512
        
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
        
        # Find loopback device (system audio)
        device = self._find_loopback_device()
        
        # Start audio stream with error handling
        try:
            self.stream = sd.InputStream(
                device=device,
                channels=self.config.audio.channels,
                samplerate=self.config.audio.sample_rate,
                blocksize=self.config.audio.buffer_size,
                callback=self._audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            print(f"[AudioEngine] Started - Device: {device}, SR: {self.config.audio.sample_rate}")
        except Exception as e:
            print(f"[AudioEngine] Error opening device {device}: {e}")
            # Try with device's native sample rate
            try:
                dev_info = sd.query_devices(device)
                native_sr = int(dev_info['default_samplerate'])
                print(f"[AudioEngine] Retrying with native sample rate: {native_sr}")
                self.config.audio.sample_rate = native_sr
                self.stream = sd.InputStream(
                    device=device,
                    channels=self.config.audio.channels,
                    samplerate=native_sr,
                    blocksize=self.config.audio.buffer_size,
                    callback=self._audio_callback,
                    dtype=np.float32
                )
                self.stream.start()
                print(f"[AudioEngine] Started - Device: {device}, SR: {native_sr}")
            except Exception as e2:
                print(f"[AudioEngine] Failed to open audio device: {e2}")
                self.running = False
                return
        
    def stop(self):
        """Stop audio capture"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("[AudioEngine] Stopped")
        
    def _find_loopback_device(self) -> Optional[int]:
        """Find the system loopback device for 'What-U-Hear' capture"""
        if self.config.audio.device_index is not None:
            return self.config.audio.device_index
            
        devices = sd.query_devices()
        
        # Print all available devices for debugging
        print("[AudioEngine] Available audio devices:")
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"  [{i}] {dev['name']} (inputs={dev['max_input_channels']})")
        
        # Look for WASAPI loopback on Windows (preferred)
        for i, dev in enumerate(devices):
            name = dev['name'].lower()
            # WASAPI loopback is the most reliable on Windows
            if 'wasapi' in name and 'loopback' in name:
                print(f"[AudioEngine] Selected WASAPI loopback: {dev['name']}")
                return i
        
        # Fallback to Stereo Mix or similar
        for i, dev in enumerate(devices):
            name = dev['name'].lower()
            if 'loopback' in name or 'stereo mix' in name or 'what u hear' in name:
                print(f"[AudioEngine] Selected: {dev['name']}")
                return i
                
        # Fallback to default input
        print("[AudioEngine] No loopback found, using default input")
        return None
        
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Process incoming audio data"""
        if status:
            print(f"[AudioEngine] Status: {status}")
            
        if not self.running:
            return
            
        # Convert to mono
        if indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata[:, 0]
            
        # Compute FFT for spectrum
        spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        
        # Store full spectrum for visualization
        with self.spectrum_lock:
            self.spectrum_data = spectrum.copy()
        
        # Filter spectrum to selected frequency band for beat detection
        band_spectrum = self._filter_frequency_band(spectrum)
        
        # Compute beat detection metrics on filtered band
        band_energy = np.sqrt(np.mean(band_spectrum ** 2)) if len(band_spectrum) > 0 else 0
        spectral_flux = self._compute_spectral_flux(band_spectrum)
        
        # Debug: print every 20 frames to see levels
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 20 == 0:
            print(f"[Audio] band_energy={band_energy:.6f} flux={spectral_flux:.4f} peak_env={self.peak_envelope:.6f}")
        
        # Track peak envelope with decay (using band energy)
        decay = self.config.beat.peak_decay
        if band_energy > self.peak_envelope:
            self.peak_envelope = band_energy
        else:
            self.peak_envelope *= decay
            
        # Detect beat based on mode (using band energy)
        is_beat = self._detect_beat(band_energy, spectral_flux)
        
        # Estimate dominant frequency
        freq = self._estimate_frequency(spectrum)
        
        # Create beat event
        event = BeatEvent(
            timestamp=time.time(),
            intensity=min(1.0, band_energy / max(0.0001, self.peak_envelope)),
            frequency=freq,
            is_beat=is_beat,
            spectral_flux=spectral_flux,
            peak_energy=band_energy
        )
        
        # Notify callback
        self.beat_callback(event)
    
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
            print(f"[Beat] energy={energy:.4f} (thresh={energy_threshold:.4f}) flux={flux:.4f}")
        
        return is_beat
            
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
