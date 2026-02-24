from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class TriggerTelemetry:
    legacy_fire: bool = False
    new_fire: bool = False
    beat_score: float = 0.0
    cue_flux: float = 0.0
    cue_band_spike: float = 0.0
    cue_energy_delta: float = 0.0
    cue_phase_align: float = 0.0
    cue_sidecar: float = 0.0
    acf_bpm: float = 0.0
    acf_confidence: float = 0.0
    phase_error_ms: float = 0.0
    smoothing_tag: str = ""
    wall_time: Optional[float] = None


class TelemetryTuning:
    def __init__(self, max_samples: int = 4096):
        self._max_samples = max(16, int(max_samples))
        self.reset()

    def reset(self) -> None:
        self._samples_seen = 0
        self._legacy_fire_count = 0
        self._new_fire_count = 0
        self._agreement_count = 0
        self._disagreement_count = 0
        self._last_tag = ""
        self._beat_score_sum = 0.0
        self._cue_flux_sum = 0.0
        self._cue_band_spike_sum = 0.0
        self._cue_energy_delta_sum = 0.0
        self._cue_phase_align_sum = 0.0
        self._cue_sidecar_sum = 0.0

    def summary(self) -> dict[str, float | str]:
        seen = float(self._samples_seen)
        agreement_pct = (100.0 * float(self._agreement_count) / seen) if seen > 0 else 0.0
        beat_score_mean = (self._beat_score_sum / seen) if seen > 0 else 0.0
        cue_flux_mean = (self._cue_flux_sum / seen) if seen > 0 else 0.0
        cue_band_spike_mean = (self._cue_band_spike_sum / seen) if seen > 0 else 0.0
        cue_energy_delta_mean = (self._cue_energy_delta_sum / seen) if seen > 0 else 0.0
        cue_phase_align_mean = (self._cue_phase_align_sum / seen) if seen > 0 else 0.0
        cue_sidecar_mean = (self._cue_sidecar_sum / seen) if seen > 0 else 0.0
        return {
            "shadow_samples": self._samples_seen,
            "shadow_legacy_fire_count": self._legacy_fire_count,
            "shadow_new_fire_count": self._new_fire_count,
            "shadow_agreement_count": self._agreement_count,
            "shadow_disagreement_count": self._disagreement_count,
            "shadow_agreement_pct": agreement_pct,
            "shadow_beat_score_mean": beat_score_mean,
            "shadow_cue_flux_mean": cue_flux_mean,
            "shadow_cue_band_spike_mean": cue_band_spike_mean,
            "shadow_cue_energy_delta_mean": cue_energy_delta_mean,
            "shadow_cue_phase_align_mean": cue_phase_align_mean,
            "shadow_cue_sidecar_mean": cue_sidecar_mean,
            "shadow_last_smoothing_tag": self._last_tag,
        }

    def record(self, sample: TriggerTelemetry) -> None:
        self._samples_seen += 1

        if bool(sample.legacy_fire):
            self._legacy_fire_count += 1
        if bool(sample.new_fire):
            self._new_fire_count += 1

        if bool(sample.legacy_fire) == bool(sample.new_fire):
            self._agreement_count += 1
        else:
            self._disagreement_count += 1

        self._beat_score_sum += float(sample.beat_score)
        self._cue_flux_sum += float(sample.cue_flux)
        self._cue_band_spike_sum += float(sample.cue_band_spike)
        self._cue_energy_delta_sum += float(sample.cue_energy_delta)
        self._cue_phase_align_sum += float(sample.cue_phase_align)
        self._cue_sidecar_sum += float(sample.cue_sidecar)

        if sample.smoothing_tag:
            self._last_tag = str(sample.smoothing_tag)
