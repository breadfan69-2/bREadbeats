from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .contracts import FeatureFrame, TempoState, TriggerDecision


@dataclass(slots=True)
class EventDetectorConfig:
    enabled: bool = True
    w_flux: float = 0.28
    w_band: float = 0.30
    w_delta: float = 0.17
    w_phase: float = 0.20
    w_sidecar: float = 0.05
    arm_threshold: float = 0.62
    release_threshold: float = 0.45
    refractory_ms: float = 170.0
    bass_dominance_weighting_enabled: bool = False
    transient_classification_enabled: bool = False


class EventDetector:
    def __init__(self, config: EventDetectorConfig | None = None):
        self.config = config or EventDetectorConfig()
        self._armed = False
        self._last_fire_mono = 0.0

    def reset(self) -> None:
        self._armed = False
        self._last_fire_mono = 0.0

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    def _phase_align_conf(self, tempo: TempoState) -> float:
        if float(tempo.metronome_bpm) <= 0.0:
            return 0.5
        phase = float(tempo.beat_phase) % 1.0
        distance = min(phase, 1.0 - phase)
        return self._clip01(1.0 - min(1.0, distance / 0.5))

    def _effective_weights(self, features: FeatureFrame) -> tuple[float, float, float, float, float]:
        w_flux = float(self.config.w_flux)
        w_band = float(self.config.w_band)
        w_delta = float(self.config.w_delta)
        w_phase = float(self.config.w_phase)
        w_sidecar = float(self.config.w_sidecar)

        if self.config.bass_dominance_weighting_enabled:
            dominance = float(np.clip(features.bass_dominance, 0.25, 4.0))
            if dominance > 1.0:
                shift = min(0.08, 0.08 * (dominance - 1.0))
                w_delta += shift
                w_flux = max(0.05, w_flux - shift)
            elif dominance < 1.0:
                shift = min(0.08, 0.08 * (1.0 - dominance))
                w_flux += shift
                w_delta = max(0.05, w_delta - shift)

        return w_flux, w_band, w_delta, w_phase, w_sidecar

    def _transient_confidences(self, features: FeatureFrame) -> tuple[float, float, float]:
        low = max(0.0, float(features.sub_bass) + float(features.low_mid))
        high = max(0.0, float(features.high))
        hfc = self._clip01(float(features.hfc_proxy))
        body = self._clip01(float(features.energy_norm))
        attack = self._clip01(float(features.flux_delta))

        kick_like = self._clip01((0.45 * low) + (0.30 * body) + (0.25 * (1.0 - hfc)))
        hat_like = self._clip01((0.45 * high) + (0.35 * hfc) + (0.20 * attack))
        mixed_like = self._clip01(1.0 - abs(kick_like - hat_like))
        return kick_like, hat_like, mixed_like

    def detect(self, features: FeatureFrame, tempo: TempoState, *, now_mono: Optional[float] = None) -> TriggerDecision:
        if not self.config.enabled:
            return TriggerDecision(reason_codes=["disabled"])

        c_flux = self._clip01(features.flux_norm)
        c_band_spike = self._clip01(max(features.sub_bass, features.low_mid, features.mid, features.high))
        c_energy_delta = self._clip01(features.energy_delta)
        c_phase_align = self._phase_align_conf(tempo)
        c_sidecar = self._clip01(features.af_onset_conf) if features.af_onset_conf is not None else 0.0

        w_flux, w_band, w_delta, w_phase, w_sidecar = self._effective_weights(features)
        sidecar_weight = w_sidecar if features.af_onset_conf is not None else 0.0
        total_weight = max(1e-9, w_flux + w_band + w_delta + w_phase + sidecar_weight)

        beat_score = (
            (w_flux * c_flux)
            + (w_band * c_band_spike)
            + (w_delta * c_energy_delta)
            + (w_phase * c_phase_align)
            + (sidecar_weight * c_sidecar)
        ) / total_weight

        arm_threshold = float(self.config.arm_threshold)
        release_threshold = float(self.config.release_threshold)
        threshold = release_threshold if self._armed else arm_threshold
        candidate = bool(beat_score >= threshold)
        reason_codes: list[str] = ["sustain" if self._armed else "arm"] if candidate else ["below_threshold"]

        now = float(now_mono) if now_mono is not None else 0.0
        refractory_ms = float(np.clip(self.config.refractory_ms, 80.0, 600.0))
        if float(tempo.metronome_bpm) > 0.0:
            beat_period_ms = 60000.0 / max(1.0, float(tempo.metronome_bpm))
            refractory_ms = min(refractory_ms, beat_period_ms * 0.7)
        refractory_s = refractory_ms / 1000.0

        if candidate and self._last_fire_mono > 0.0 and now > 0.0 and (now - self._last_fire_mono) < refractory_s:
            candidate = False
            reason_codes = ["refractory"]

        if candidate:
            self._armed = True
            if now > 0.0:
                self._last_fire_mono = now
        elif beat_score < release_threshold:
            self._armed = False

        kick_like, hat_like, mixed_like = self._transient_confidences(features)
        if not self.config.transient_classification_enabled:
            kick_like = 0.0
            hat_like = 0.0
            mixed_like = 0.0

        return TriggerDecision(
            beat_score=float(beat_score),
            raw_onset_conf=float(max(c_flux, c_band_spike, c_energy_delta, c_sidecar)),
            is_beat_candidate=bool(candidate),
            c_flux=float(c_flux),
            c_band_spike=float(c_band_spike),
            c_energy_delta=float(c_energy_delta),
            c_phase_align=float(c_phase_align),
            c_sidecar=float(c_sidecar),
            kick_like_conf=float(kick_like),
            hat_like_conf=float(hat_like),
            mixed_conf=float(mixed_like),
            reason_codes=reason_codes,
        )
