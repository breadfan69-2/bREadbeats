from __future__ import annotations

from typing import Any

from logging_utils import log_event
from audio_modules.platform_audio import PlatformAudioCapture

try:
    from scipy.signal import butter, sosfilt_zi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AudioIOController:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._platform_capture = PlatformAudioCapture(engine)

    def init_butterworth_filter(self) -> None:
        eng = self.engine
        if not HAS_SCIPY or not eng._use_butterworth:
            return

        sr = eng.config.audio.sample_rate
        nyquist = sr / 2

        low_freq = max(eng._highpass_hz, eng.config.beat.freq_low)
        high_freq = min(eng.config.beat.freq_high, nyquist * 0.95)

        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist

        low_norm = max(0.001, min(0.99, low_norm))
        high_norm = max(low_norm + 0.01, min(0.999, high_norm))

        try:
            eng._butter_sos = butter(4, [low_norm, high_norm], btype='band', output='sos')
            eng._butter_zi = sosfilt_zi(eng._butter_sos)
            log_event("INFO", "AudioEngine", "Butterworth bandpass initialized", low=f"{low_freq:.0f}", high=f"{high_freq:.0f}")
        except Exception as e:
            log_event("ERROR", "AudioEngine", "Failed to initialize Butterworth filter", error=e)
            eng._butter_sos = None

    def start(self) -> None:
        eng = self.engine
        if eng.running:
            return

        eng._reset_session_stats()
        if hasattr(eng, '_signal_frontend'):
            eng._signal_frontend.reset()
        eng.running = True

        use_loopback = getattr(eng.config.audio, 'is_loopback', True)
        device_index = getattr(eng.config.audio, 'device_index', None)

        try:
            if use_loopback:
                self._platform_capture.start_loopback_capture(device_index)
            else:
                self._platform_capture.start_input_capture(device_index)

            self.init_butterworth_filter()
        except Exception as e:
            log_event("ERROR", "AudioEngine", "Failed to start", error=e)
            eng.running = False
            self.stop()

    def start_loopback_capture(self, device_index=None) -> None:
        self._platform_capture.start_loopback_capture(device_index)

    def start_input_capture(self, device_index) -> None:
        self._platform_capture.start_input_capture(device_index)

    def stop(self) -> None:
        eng = self.engine
        eng.running = False
        eng._log_shutdown_summary()
        if hasattr(eng, '_volume_normalizer'):
            eng._volume_normalizer.shutdown()
        if eng.stream:
            try:
                eng.stream.stop_stream()
            except Exception:
                try:
                    eng.stream.stop()
                except Exception:
                    pass
            try:
                eng.stream.close()
            except Exception:
                pass
            eng.stream = None
        if eng.pyaudio:
            try:
                eng.pyaudio.terminate()
            except Exception:
                pass
            eng.pyaudio = None
        log_event("INFO", "AudioEngine", "Stopped")
