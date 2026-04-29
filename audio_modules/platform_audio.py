from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Any

import numpy as np
import sounddevice as sd

from logging_utils import log_event

try:
    import pyaudiowpatch as pyaudio  # type: ignore
except Exception:
    pyaudio = None


@dataclass(frozen=True)
class CaptureDevice:
    display_name: str
    device_index: int
    is_loopback: bool
    is_default: bool


def _find_hostapi_index(hostapis: list[dict[str, Any]], keyword: str) -> tuple[int | None, int | None, int | None]:
    needle = keyword.lower()
    for idx, api in enumerate(hostapis):
        name = str(api.get('name', '')).lower()
        if needle in name:
            return idx, api.get('default_input_device', None), api.get('default_output_device', None)
    return None, None, None


def enumerate_capture_devices() -> list[CaptureDevice]:
    """Enumerate selectable capture devices across platforms.

    Windows:
      - Include WASAPI inputs
      - Include WASAPI outputs as loopback sources

    macOS/Linux:
      - Include input-capable devices from the platform host API
      - BlackHole appears naturally as an input device on macOS
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    system_name = platform.system().lower()
    results: list[CaptureDevice] = []

    if system_name == 'windows':
        wasapi_idx, _default_input_idx, default_output_idx = _find_hostapi_index(hostapis, 'wasapi')
        if wasapi_idx is None:
            return _enumerate_generic_inputs(devices)

        seen_inputs: set[str] = set()
        for idx, dev in enumerate(devices):
            if dev.get('hostapi') != wasapi_idx or dev.get('max_input_channels', 0) <= 0:
                continue
            clean_name = str(dev.get('name', '')).strip()
            if not clean_name or clean_name in seen_inputs:
                continue
            seen_inputs.add(clean_name)
            results.append(
                CaptureDevice(
                    display_name=f"{clean_name} (Input)",
                    device_index=idx,
                    is_loopback=False,
                    is_default=False,
                )
            )

        seen_outputs: set[str] = set()
        for idx, dev in enumerate(devices):
            if dev.get('hostapi') != wasapi_idx or dev.get('max_output_channels', 0) <= 0:
                continue
            clean_name = str(dev.get('name', '')).strip()
            if not clean_name or clean_name in seen_outputs:
                continue
            seen_outputs.add(clean_name)
            is_default = idx == default_output_idx
            prefix = '★ ' if is_default else ''
            results.append(
                CaptureDevice(
                    display_name=f"{prefix}{clean_name} [WASAPI Loopback]",
                    device_index=idx,
                    is_loopback=True,
                    is_default=is_default,
                )
            )

        return results

    if system_name == 'darwin':
        coreaudio_idx, default_input_idx, _default_output_idx = _find_hostapi_index(hostapis, 'core audio')
        return _enumerate_inputs_for_hostapi(devices, coreaudio_idx, default_input_idx)

    return _enumerate_generic_inputs(devices)


def _enumerate_inputs_for_hostapi(
    devices: list[dict[str, Any]],
    hostapi_idx: int | None,
    default_input_idx: int | None,
) -> list[CaptureDevice]:
    results: list[CaptureDevice] = []
    seen_names: set[str] = set()

    for idx, dev in enumerate(devices):
        if hostapi_idx is not None and dev.get('hostapi') != hostapi_idx:
            continue
        if dev.get('max_input_channels', 0) <= 0:
            continue
        clean_name = str(dev.get('name', '')).strip()
        if not clean_name or clean_name in seen_names:
            continue
        seen_names.add(clean_name)

        is_blackhole = 'blackhole' in clean_name.lower()
        is_default = idx == default_input_idx
        label = f"{clean_name} [BlackHole System Audio]" if is_blackhole else clean_name
        if is_default:
            label = f"★ {label}"

        results.append(
            CaptureDevice(
                display_name=label,
                device_index=idx,
                is_loopback=False,
                is_default=is_default,
            )
        )

    return results


def _enumerate_generic_inputs(devices: list[dict[str, Any]]) -> list[CaptureDevice]:
    results: list[CaptureDevice] = []
    seen_names: set[str] = set()
    for idx, dev in enumerate(devices):
        if dev.get('max_input_channels', 0) <= 0:
            continue
        clean_name = str(dev.get('name', '')).strip()
        if not clean_name or clean_name in seen_names:
            continue
        seen_names.add(clean_name)
        results.append(
            CaptureDevice(
                display_name=clean_name,
                device_index=idx,
                is_loopback=False,
                is_default=False,
            )
        )
    return results


class PlatformAudioCapture:
    """Cross-platform stream startup for AudioEngine.

    - Windows loopback uses PyAudio/WASAPI via pyaudiowpatch.
    - All other modes use sounddevice InputStream.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def start_loopback_capture(self, device_index: int | None = None) -> None:
        if platform.system().lower() == 'windows' and pyaudio is not None:
            self._start_windows_loopback(device_index)
            return

        if platform.system().lower() != 'windows':
            log_event("INFO", "AudioEngine", "Loopback mode requested on non-Windows, using input capture")
        else:
            log_event("WARN", "AudioEngine", "pyaudiowpatch unavailable, falling back to input capture")
        self.start_input_capture(device_index)

    def start_input_capture(self, device_index: int | None) -> None:
        eng = self.engine
        if device_index is None:
            default_input = sd.default.device[0]
            if default_input is None:
                raise RuntimeError("No default input audio device is configured")
            device_index = int(default_input)

        device_id = int(device_index)

        device_info = sd.query_devices(device_id, 'input')
        sample_rate = int(float(device_info.get('default_samplerate', eng.config.audio.sample_rate)))
        channels = max(1, min(int(device_info.get('max_input_channels', 1)), 2))

        log_event("INFO", "AudioEngine", "Using input device", device=device_info.get('name', str(device_id)))
        log_event("INFO", "AudioEngine", "Input format", channels=channels, sample_rate=sample_rate)

        eng.config.audio.sample_rate = sample_rate
        eng.config.audio.channels = channels

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

        eng.stream = sd.InputStream(
            samplerate=eng.config.audio.sample_rate,
            blocksize=int(eng.config.audio.buffer_size),
            device=device_id,
            channels=int(eng.config.audio.channels),
            dtype='float32',
            callback=self._sounddevice_callback,
        )
        eng.stream.start()
        eng.pyaudio = None
        log_event("INFO", "AudioEngine", "Input capture started")

    def _start_windows_loopback(self, device_index: int | None = None) -> None:
        eng = self.engine
        if pyaudio is None:
            raise RuntimeError("pyaudiowpatch is required for Windows loopback capture")
        pa = pyaudio.PyAudio()

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
        log_event(
            "INFO",
            "AudioEngine",
            "Loopback format",
            channels=device_info['maxInputChannels'],
            sample_rate=int(device_info['defaultSampleRate']),
        )

        eng.config.audio.sample_rate = int(device_info['defaultSampleRate'])
        eng.config.audio.channels = int(device_info['maxInputChannels'])

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
            stream_callback=eng._audio_callback_pyaudio,
        )
        eng.stream.start_stream()
        eng.pyaudio = pa
        log_event("INFO", "AudioEngine", "WASAPI loopback capture started")

    def _sounddevice_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        eng = self.engine
        if not eng.running:
            return
        if status:
            log_event("WARN", "AudioEngine", "Input callback status", status=str(status))

        in_bytes = np.asarray(indata, dtype=np.float32, order='C').tobytes()
        eng._audio_callback_pyaudio(in_bytes, frames, time_info, 0)