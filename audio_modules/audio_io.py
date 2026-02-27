from __future__ import annotations

from typing import Any

import pyaudiowpatch as pyaudio

from logging_utils import log_event

try:
    from scipy.signal import butter, sosfilt_zi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AudioIOController:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

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
        eng.pyaudio = pyaudio.PyAudio()

        use_loopback = getattr(eng.config.audio, 'is_loopback', True)
        device_index = getattr(eng.config.audio, 'device_index', None)

        try:
            if use_loopback:
                self.start_loopback_capture(device_index)
            else:
                self.start_input_capture(device_index)

            self.init_butterworth_filter()
        except Exception as e:
            log_event("ERROR", "AudioEngine", "Failed to start", error=e)
            eng.running = False
            if eng.pyaudio:
                eng.pyaudio.terminate()
                eng.pyaudio = None

    def start_loopback_capture(self, device_index=None) -> None:
        eng = self.engine
        pa = eng.pyaudio
        if pa is None:
            raise RuntimeError("PyAudio is not initialized")

        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)

        if device_index is not None:
            device_info = pa.get_device_info_by_index(device_index)
            if not device_info.get("isLoopbackDevice", False):
                for loopback in pa.get_loopback_device_info_generator():
                    if device_info["name"] in loopback["name"]:
                        device_info = loopback
                        break
        else:
            device_info = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            if not device_info.get("isLoopbackDevice", False):
                for loopback in pa.get_loopback_device_info_generator():
                    if device_info["name"] in loopback["name"]:
                        device_info = loopback
                        break

        log_event("INFO", "AudioEngine", "Using WASAPI loopback", device=device_info['name'])
        log_event("INFO", "AudioEngine", "Loopback format", channels=device_info['maxInputChannels'], sample_rate=int(device_info['defaultSampleRate']))

        eng.config.audio.sample_rate = int(device_info['defaultSampleRate'])
        eng.config.audio.channels = device_info['maxInputChannels']

        if hasattr(eng, '_signal_frontend'):
            eng._signal_frontend.configure_runtime(
                sample_rate=int(eng.config.audio.sample_rate),
                channels=int(eng.config.audio.channels),
                gain=float(eng.config.audio.gain),
                fft_size=int(eng.fft_size),
                hop_size=int(eng.hop_size),
                freq_low=float(eng.config.beat.freq_low),
                freq_high=float(eng.config.beat.freq_high),
                flux_multiplier=float(eng.config.beat.flux_multiplier),
            )

        eng.stream = pa.open(
            format=pyaudio.paFloat32,
            channels=eng.config.audio.channels,
            rate=eng.config.audio.sample_rate,
            frames_per_buffer=eng.config.audio.buffer_size,
            input=True,
            input_device_index=device_info["index"],
            stream_callback=eng._audio_callback_pyaudio
        )

        eng.stream.start_stream()
        log_event("INFO", "AudioEngine", "WASAPI loopback capture started")

    def start_input_capture(self, device_index) -> None:
        eng = self.engine
        pa = eng.pyaudio
        if pa is None:
            raise RuntimeError("PyAudio is not initialized")

        if device_index is None:
            wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            device_index = wasapi_info.get("defaultInputDevice", 0)

        device_info = pa.get_device_info_by_index(device_index)

        log_event("INFO", "AudioEngine", "Using input device", device=device_info['name'])
        log_event("INFO", "AudioEngine", "Input format", channels=device_info['maxInputChannels'], sample_rate=int(device_info['defaultSampleRate']))

        eng.config.audio.sample_rate = int(device_info['defaultSampleRate'])
        eng.config.audio.channels = min(device_info['maxInputChannels'], 2)

        if hasattr(eng, '_signal_frontend'):
            eng._signal_frontend.configure_runtime(
                sample_rate=int(eng.config.audio.sample_rate),
                channels=int(eng.config.audio.channels),
                gain=float(eng.config.audio.gain),
                fft_size=int(eng.fft_size),
                hop_size=int(eng.hop_size),
                freq_low=float(eng.config.beat.freq_low),
                freq_high=float(eng.config.beat.freq_high),
                flux_multiplier=float(eng.config.beat.flux_multiplier),
            )

        eng.stream = pa.open(
            format=pyaudio.paFloat32,
            channels=eng.config.audio.channels,
            rate=eng.config.audio.sample_rate,
            frames_per_buffer=eng.config.audio.buffer_size,
            input=True,
            input_device_index=device_index,
            stream_callback=eng._audio_callback_pyaudio
        )

        eng.stream.start_stream()
        log_event("INFO", "AudioEngine", "Input capture started")

    def stop(self) -> None:
        eng = self.engine
        eng.running = False
        eng._log_shutdown_summary()
        if hasattr(eng, '_volume_normalizer'):
            eng._volume_normalizer.shutdown()
        if eng.stream:
            eng.stream.stop_stream()
            eng.stream.close()
            eng.stream = None
        if eng.pyaudio:
            eng.pyaudio.terminate()
            eng.pyaudio = None
        log_event("INFO", "AudioEngine", "Stopped")
