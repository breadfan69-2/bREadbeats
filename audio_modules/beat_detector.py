from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import BeatDetectionType


@dataclass(slots=True)
class BeatDetectionResult:
    is_beat: bool = False
    detected_at: float = 0.0
    source: str = ""
    fired_bands: list[str] = field(default_factory=list)
    energy_threshold: float = 0.0


class BeatDetector:
    def __init__(self, config) -> None:
        self.config = config
        self.energy_history: list[float] = []
        self.flux_history: list[float] = []
        self.last_beat_time: float = 0.0

    def detect(
        self,
        *,
        energy: float,
        flux: float,
        now: float,
        primary_band: str,
        band_zscore_signals: dict[str, int],
        metronome_bpm: float,
        fallback_bpm: float,
    ) -> BeatDetectionResult:
        _BEAT_ENERGY_FLOOR = 0.001
        if energy < _BEAT_ENERGY_FLOOR:
            return BeatDetectionResult(is_beat=False, detected_at=now)

        cfg = self.config.beat

        zscore_signal = band_zscore_signals.get(primary_band, 0)
        zscore_peak = (zscore_signal == 1)

        self.energy_history.append(float(energy))
        self.flux_history.append(float(flux))

        max_history = 50
        self.energy_history = self.energy_history[-max_history:]
        self.flux_history = self.flux_history[-max_history:]

        if len(self.energy_history) < 5:
            return BeatDetectionResult(is_beat=False, detected_at=now)

        beat_refractory_ms = float(getattr(cfg, 'beat_refractory_ms', 170.0) or 170.0)
        beat_refractory_ms = float(np.clip(beat_refractory_ms, 80.0, 600.0))

        if metronome_bpm > 0:
            beat_period_ms = 60000.0 / max(1.0, float(metronome_bpm))
        else:
            beat_period_ms = 60000.0 / max(1.0, float(fallback_bpm or 120.0))

        refractory_ms = min(beat_refractory_ms, beat_period_ms * 0.7)
        refractory_s = refractory_ms / 1000.0
        if now - self.last_beat_time < refractory_s:
            return BeatDetectionResult(is_beat=False, detected_at=now)

        avg_energy = float(np.mean(self.energy_history))
        avg_flux = float(np.mean(self.flux_history))

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

        any_band_fired = any(s == 1 for s in band_zscore_signals.values())
        zscore_beat = (zscore_peak or any_band_fired) and (energy > avg_energy * 1.1)

        is_beat = bool(classic_beat or zscore_beat)
        if not is_beat:
            return BeatDetectionResult(
                is_beat=False,
                detected_at=now,
                energy_threshold=float(energy_threshold),
            )

        self.last_beat_time = now
        source = "Z+C" if (classic_beat and zscore_beat) else ("Z" if zscore_beat else "C")
        fired_bands = [name for name, signal in band_zscore_signals.items() if signal == 1]
        return BeatDetectionResult(
            is_beat=True,
            detected_at=now,
            source=source,
            fired_bands=fired_bands,
            energy_threshold=float(energy_threshold),
        )
