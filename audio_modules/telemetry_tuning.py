from __future__ import annotations

from dataclasses import dataclass, field
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
    frontend_ms: float = 0.0
    tempo_ms: float = 0.0
    detector_ms: float = 0.0
    sidecar_ms: float = 0.0
    smoothing_tag: str = ""
    wall_time: Optional[float] = None
    bus_raw_scores: dict[str, float] = field(default_factory=dict)
    bus_masked_scores: dict[str, float] = field(default_factory=dict)
    bus_pass: dict[str, bool] = field(default_factory=dict)
    bus_reason_codes: dict[str, list[str]] = field(default_factory=dict)


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
        self._frontend_ms_sum = 0.0
        self._tempo_ms_sum = 0.0
        self._detector_ms_sum = 0.0
        self._sidecar_ms_sum = 0.0
        self._bus_raw_sums = {"sub_bass": 0.0, "low_mid": 0.0, "mid": 0.0, "high": 0.0}
        self._bus_masked_sums = {"sub_bass": 0.0, "low_mid": 0.0, "mid": 0.0, "high": 0.0}
        self._bus_pass_counts = {"sub_bass": 0, "low_mid": 0, "mid": 0, "high": 0}

    def summary(self) -> dict[str, float | str]:
        seen = float(self._samples_seen)
        agreement_pct = (100.0 * float(self._agreement_count) / seen) if seen > 0 else 0.0
        beat_score_mean = (self._beat_score_sum / seen) if seen > 0 else 0.0
        cue_flux_mean = (self._cue_flux_sum / seen) if seen > 0 else 0.0
        cue_band_spike_mean = (self._cue_band_spike_sum / seen) if seen > 0 else 0.0
        cue_energy_delta_mean = (self._cue_energy_delta_sum / seen) if seen > 0 else 0.0
        cue_phase_align_mean = (self._cue_phase_align_sum / seen) if seen > 0 else 0.0
        cue_sidecar_mean = (self._cue_sidecar_sum / seen) if seen > 0 else 0.0
        frontend_ms_mean = (self._frontend_ms_sum / seen) if seen > 0 else 0.0
        tempo_ms_mean = (self._tempo_ms_sum / seen) if seen > 0 else 0.0
        detector_ms_mean = (self._detector_ms_sum / seen) if seen > 0 else 0.0
        sidecar_ms_mean = (self._sidecar_ms_sum / seen) if seen > 0 else 0.0
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
            "shadow_frontend_ms_mean": frontend_ms_mean,
            "shadow_tempo_ms_mean": tempo_ms_mean,
            "shadow_detector_ms_mean": detector_ms_mean,
            "shadow_sidecar_ms_mean": sidecar_ms_mean,
            "shadow_bus_raw_sub_bass_mean": (self._bus_raw_sums["sub_bass"] / seen) if seen > 0 else 0.0,
            "shadow_bus_raw_low_mid_mean": (self._bus_raw_sums["low_mid"] / seen) if seen > 0 else 0.0,
            "shadow_bus_raw_mid_mean": (self._bus_raw_sums["mid"] / seen) if seen > 0 else 0.0,
            "shadow_bus_raw_high_mean": (self._bus_raw_sums["high"] / seen) if seen > 0 else 0.0,
            "shadow_bus_masked_sub_bass_mean": (self._bus_masked_sums["sub_bass"] / seen) if seen > 0 else 0.0,
            "shadow_bus_masked_low_mid_mean": (self._bus_masked_sums["low_mid"] / seen) if seen > 0 else 0.0,
            "shadow_bus_masked_mid_mean": (self._bus_masked_sums["mid"] / seen) if seen > 0 else 0.0,
            "shadow_bus_masked_high_mean": (self._bus_masked_sums["high"] / seen) if seen > 0 else 0.0,
            "shadow_bus_pass_sub_bass_count": self._bus_pass_counts["sub_bass"],
            "shadow_bus_pass_low_mid_count": self._bus_pass_counts["low_mid"],
            "shadow_bus_pass_mid_count": self._bus_pass_counts["mid"],
            "shadow_bus_pass_high_count": self._bus_pass_counts["high"],
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
        self._frontend_ms_sum += max(0.0, float(sample.frontend_ms))
        self._tempo_ms_sum += max(0.0, float(sample.tempo_ms))
        self._detector_ms_sum += max(0.0, float(sample.detector_ms))
        self._sidecar_ms_sum += max(0.0, float(sample.sidecar_ms))

        for bus in ("sub_bass", "low_mid", "mid", "high"):
            self._bus_raw_sums[bus] += float(sample.bus_raw_scores.get(bus, 0.0))
            self._bus_masked_sums[bus] += float(sample.bus_masked_scores.get(bus, 0.0))
            if bool(sample.bus_pass.get(bus, False)):
                self._bus_pass_counts[bus] += 1

        if sample.smoothing_tag:
            self._last_tag = str(sample.smoothing_tag)
