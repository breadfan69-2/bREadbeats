from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .contracts import FrontendFrame
from .feature_extractors import positive_spectral_flux, slice_spectrum_band


RMS_DB_FLOOR = -120.0


def _rms_to_dbfs(rms: float, floor_db: float = RMS_DB_FLOOR) -> float:
    value = max(float(rms), 1e-12)
    return float(np.clip(20.0 * np.log10(value), floor_db, 12.0))


@dataclass(slots=True)
class SignalFrontendConfig:
    sample_rate: int = 44100
    channels: int = 2
    gain: float = 1.0
    fft_size: int = 1024
    hop_size: int = 256
    freq_low: float = 100.0
    freq_high: float = 8000.0
    flux_multiplier: float = 1.0
    superflux_max_filter_size: int = 3  # 1 = classic flux, 3 = SuperFlux (±1 bin)


class SignalFrontend:
    def __init__(self, config: SignalFrontendConfig | None = None):
        self.config = config or SignalFrontendConfig()
        self._hanning_window = np.hanning(max(16, int(self.config.fft_size))).astype(np.float32)
        self._fft_input_buffer = np.array([], dtype=np.float32)
        self._aux_fft_input_buffer = np.array([], dtype=np.float32)
        self._prev_band_spectrum: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._fft_input_buffer = np.array([], dtype=np.float32)
        self._aux_fft_input_buffer = np.array([], dtype=np.float32)
        self._prev_band_spectrum = None

    def configure_runtime(
        self,
        *,
        sample_rate: int,
        channels: int,
        gain: float,
        fft_size: int,
        hop_size: int,
        freq_low: float,
        freq_high: float,
        flux_multiplier: float,
    ) -> None:
        self.config.sample_rate = int(sample_rate)
        self.config.channels = int(channels)
        self.config.gain = float(gain)
        self.config.fft_size = int(fft_size)
        self.config.hop_size = int(hop_size)
        self.config.freq_low = float(freq_low)
        self.config.freq_high = float(freq_high)
        self.config.flux_multiplier = float(flux_multiplier)

    def configure_superflux(self, *, max_filter_size: int = 3) -> None:
        """Set SuperFlux max-filter width (1 = disabled, 3 = default)."""
        self.config.superflux_max_filter_size = max(1, int(max_filter_size))

    def _to_mono(self, indata: np.ndarray) -> np.ndarray:
        data = np.asarray(indata, dtype=np.float32)
        if data.ndim == 1:
            return data
        if data.ndim != 2:
            return data.reshape(-1)
        if data.shape[1] <= 1:
            return data[:, 0]
        return np.asarray(np.mean(data, axis=1), dtype=np.float32)

    def process(
        self,
        indata: np.ndarray,
        *,
        mono_time: float,
        wall_time: float,
    ) -> Optional[FrontendFrame]:
        fft_size = max(16, int(self.config.fft_size))
        hop_size = max(1, int(self.config.hop_size))

        if len(self._hanning_window) != fft_size:
            self._hanning_window = np.hanning(fft_size).astype(np.float32)

        mono = self._to_mono(indata)
        if mono.size == 0:
            return None

        self._fft_input_buffer = np.concatenate((self._fft_input_buffer, mono.astype(np.float32, copy=False)))

        fft_scale = 1.0 / max(1e-12, (float(np.sum(self._hanning_window)) / 2.0))
        latest_frame: Optional[FrontendFrame] = None

        while len(self._fft_input_buffer) >= fft_size:
            frame = self._fft_input_buffer[:fft_size]
            windowed = frame * self._hanning_window
            spectrum = np.abs(np.fft.rfft(windowed)) * fft_scale

            band_spectrum = slice_spectrum_band(
                spectrum,
                int(self.config.sample_rate),
                float(self.config.freq_low),
                float(self.config.freq_high),
                fallback_full_if_invalid=True,
            )
            band_spectrum = band_spectrum * float(self.config.gain)

            band_energy = float(np.sqrt(np.mean(band_spectrum ** 2))) if len(band_spectrum) > 0 else 0.0
            spectral_flux = positive_spectral_flux(
                self._prev_band_spectrum,
                band_spectrum,
                max_filter_size=int(self.config.superflux_max_filter_size),
            )
            self._prev_band_spectrum = band_spectrum.copy()

            raw_rms = float(np.sqrt(np.mean(frame ** 2))) if len(frame) > 0 else 0.0
            raw_rms_db = _rms_to_dbfs(raw_rms)

            latest_frame = FrontendFrame(
                mono_time=float(mono_time),
                wall_time=float(wall_time),
                spectrum=np.asarray(spectrum, dtype=np.float32),
                band_energy=float(band_energy),
                spectral_flux=float(spectral_flux),
                raw_rms=float(raw_rms),
                raw_rms_db=float(raw_rms_db),
            )

            self._fft_input_buffer = self._fft_input_buffer[hop_size:]

        return latest_frame

    def process_dual(
        self,
        mono: np.ndarray,
        *,
        mono_time: float,
        wall_time: float,
        beat_mono: Optional[np.ndarray] = None,
        use_filtered_band: bool = False,
    ) -> Optional[FrontendFrame]:
        fft_size = max(16, int(self.config.fft_size))
        hop_size = max(1, int(self.config.hop_size))

        if len(self._hanning_window) != fft_size:
            self._hanning_window = np.hanning(fft_size).astype(np.float32)

        mono_arr = np.asarray(mono, dtype=np.float32).reshape(-1)
        if mono_arr.size == 0:
            return None

        if beat_mono is None:
            aux_arr = mono_arr
        else:
            aux_arr = np.asarray(beat_mono, dtype=np.float32).reshape(-1)

        self._fft_input_buffer = np.concatenate((self._fft_input_buffer, mono_arr))
        self._aux_fft_input_buffer = np.concatenate((self._aux_fft_input_buffer, aux_arr))

        fft_scale = 1.0 / max(1e-12, (float(np.sum(self._hanning_window)) / 2.0))
        latest_frame: Optional[FrontendFrame] = None

        while len(self._fft_input_buffer) >= fft_size and len(self._aux_fft_input_buffer) >= fft_size:
            frame = self._fft_input_buffer[:fft_size]
            aux_frame = self._aux_fft_input_buffer[:fft_size]

            windowed = frame * self._hanning_window
            spectrum = np.abs(np.fft.rfft(windowed)) * fft_scale

            if use_filtered_band:
                band_energy = float(np.sqrt(np.mean(aux_frame ** 2))) * float(self.config.gain)
                aux_windowed = aux_frame * self._hanning_window
                band_spectrum = np.abs(np.fft.rfft(aux_windowed)) * fft_scale
                band_spectrum = band_spectrum * float(self.config.gain)
            else:
                band_spectrum = slice_spectrum_band(
                    spectrum,
                    int(self.config.sample_rate),
                    float(self.config.freq_low),
                    float(self.config.freq_high),
                    fallback_full_if_invalid=True,
                )
                band_spectrum = band_spectrum * float(self.config.gain)
                band_energy = float(np.sqrt(np.mean(band_spectrum ** 2))) if len(band_spectrum) > 0 else 0.0

            spectral_flux = positive_spectral_flux(
                self._prev_band_spectrum,
                band_spectrum,
                max_filter_size=int(self.config.superflux_max_filter_size),
            )
            spectral_flux *= float(self.config.flux_multiplier)
            self._prev_band_spectrum = band_spectrum.copy()

            raw_rms = float(np.sqrt(np.mean(frame ** 2))) if len(frame) > 0 else 0.0
            raw_rms_db = _rms_to_dbfs(raw_rms)

            latest_frame = FrontendFrame(
                mono_time=float(mono_time),
                wall_time=float(wall_time),
                spectrum=np.asarray(spectrum, dtype=np.float32),
                band_energy=float(band_energy),
                spectral_flux=float(spectral_flux),
                raw_rms=float(raw_rms),
                raw_rms_db=float(raw_rms_db),
            )

            self._fft_input_buffer = self._fft_input_buffer[hop_size:]
            self._aux_fft_input_buffer = self._aux_fft_input_buffer[hop_size:]

        return latest_frame
