from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import importlib
from typing import Optional

import numpy as np


@dataclass(slots=True)
class AudioFluxAdapterConfig:
    enabled: bool = False
    frame_stride: int = 2
    fft_size: int = 1024
    emit_onset_confidence: bool = True


class AudioFluxAdapter:
    def __init__(self, sample_rate: int, config: AudioFluxAdapterConfig | None = None):
        self.sample_rate = int(sample_rate)
        self.config = config or AudioFluxAdapterConfig()
        self._enabled = bool(self.config.enabled)
        self._available = False
        self._frame_counter = 0
        self._buffer = deque(maxlen=max(256, int(self.config.fft_size) * 4))
        self._latest_features: Optional[dict[str, float]] = None
        self._prev_mag: Optional[np.ndarray] = None
        self._audioflux_mod = None
        if self._enabled:
            try:
                self._audioflux_mod = importlib.import_module("audioflux")
                self._available = True
            except Exception:
                self._audioflux_mod = None
                self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def reset(self) -> None:
        self._frame_counter = 0
        self._latest_features = None
        self._prev_mag = None
        self._buffer.clear()

    def push_audio(self, mono_frame) -> None:
        if not self._enabled or not self._available:
            return
        if mono_frame is None:
            return

        try:
            frame = np.asarray(mono_frame, dtype=np.float32).reshape(-1)
            if frame.size == 0:
                return

            self._buffer.extend(float(v) for v in frame)
            self._frame_counter += 1

            stride = max(1, int(self.config.frame_stride))
            if self._frame_counter % stride != 0:
                return

            fft_size = max(256, int(self.config.fft_size))
            if len(self._buffer) < fft_size:
                return

            window = np.array(list(self._buffer)[-fft_size:], dtype=np.float32)
            self._latest_features = self._compute_features(window)
        except Exception:
            self._available = False
            self._latest_features = None
            self._prev_mag = None

    def get_latest_features(self) -> Optional[dict[str, float]]:
        return dict(self._latest_features) if self._latest_features is not None else None

    def _compute_features(self, window: np.ndarray) -> dict[str, float]:
        windowed = window * np.hanning(len(window)).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(windowed)).astype(np.float64)
        if spectrum.size == 0:
            return {
                "af_entropy": 0.0,
                "af_flatness": 0.0,
                "af_hfc": 0.0,
                "af_novelty": 0.0,
                "af_rms": 0.0,
                "af_onset_conf": 0.0,
            }

        power = spectrum * spectrum
        p_sum = float(np.sum(power))
        if p_sum <= 1e-12:
            power_norm = np.zeros_like(power)
        else:
            power_norm = power / p_sum

        entropy = float(-np.sum(power_norm * np.log(power_norm + 1e-12)) / np.log(len(power_norm) + 1e-12))
        am = float(np.mean(spectrum) + 1e-12)
        gm = float(np.exp(np.mean(np.log(spectrum + 1e-12))))
        flatness = float(np.clip(gm / am, 0.0, 1.0))

        idx = np.arange(1, len(spectrum) + 1, dtype=np.float64)
        hfc_raw = float(np.sum(idx * spectrum) / (np.sum(spectrum) + 1e-9))
        hfc = float(np.clip(hfc_raw / max(1.0, float(len(spectrum))), 0.0, 1.0))

        novelty = 0.0
        if self._prev_mag is not None and len(self._prev_mag) == len(spectrum):
            novelty = float(np.sum(np.maximum(0.0, spectrum - self._prev_mag)) / (np.sum(spectrum) + 1e-9))
        novelty = float(np.clip(novelty, 0.0, 1.0))
        self._prev_mag = spectrum

        rms = float(np.sqrt(np.mean(window * window)))
        onset_conf = float(np.clip((0.60 * novelty) + (0.40 * hfc), 0.0, 1.0))
        if not bool(self.config.emit_onset_confidence):
            onset_conf = 0.0

        return {
            "af_entropy": float(np.clip(entropy, 0.0, 1.0)),
            "af_flatness": flatness,
            "af_hfc": hfc,
            "af_novelty": novelty,
            "af_rms": float(np.clip(rms, 0.0, 1.0)),
            "af_onset_conf": onset_conf,
        }
