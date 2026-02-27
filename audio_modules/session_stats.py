from __future__ import annotations

import time

import numpy as np

from audio_modules.contracts import RMS_DB_FLOOR, TriggerDecision
from audio_modules.telemetry_tuning import TelemetryTuning, TriggerTelemetry
from logging_utils import log_event


class SessionStats:
    def __init__(self, new_trigger_telemetry_enabled: bool = True) -> None:
        self._new_trigger_telemetry_enabled = bool(new_trigger_telemetry_enabled)
        self._shadow_telemetry = TelemetryTuning()
        self._session_started_at: float = 0.0
        self._session_frame_count: int = 0
        self._session_raw_rms_db_min: float | None = None
        self._session_raw_rms_db_max: float | None = None
        self._session_band_energy_min: float | None = None
        self._session_band_energy_max: float | None = None
        self._session_flux_min: float | None = None
        self._session_flux_max: float | None = None
        self._session_raw_rms_db_sum: float = 0.0
        self._session_band_energy_sum: float = 0.0
        self._session_flux_sum: float = 0.0
        self._session_sample_times: list[float] = []
        self._session_flux_samples: list[float] = []
        self._session_peak_samples: list[float] = []
        self._session_trough_samples: list[float] = []

    def reset(self) -> None:
        self._session_started_at = time.time()
        self._session_frame_count = 0
        self._session_raw_rms_db_min = None
        self._session_raw_rms_db_max = None
        self._session_band_energy_min = None
        self._session_band_energy_max = None
        self._session_flux_min = None
        self._session_flux_max = None
        self._session_raw_rms_db_sum = 0.0
        self._session_band_energy_sum = 0.0
        self._session_flux_sum = 0.0
        self._session_sample_times = []
        self._session_flux_samples = []
        self._session_peak_samples = []
        self._session_trough_samples = []
        self._shadow_telemetry.reset()

    def update(
        self,
        raw_rms_db: float,
        band_energy: float,
        spectral_flux: float,
        peak_level: float,
        sample_time: float,
    ) -> None:
        self._session_frame_count += 1
        self._session_raw_rms_db_sum += raw_rms_db
        self._session_band_energy_sum += band_energy
        self._session_flux_sum += spectral_flux
        self._session_sample_times.append(sample_time)
        self._session_flux_samples.append(spectral_flux)
        self._session_peak_samples.append(peak_level)
        self._session_trough_samples.append(band_energy)
        if self._session_raw_rms_db_min is None or raw_rms_db < self._session_raw_rms_db_min:
            self._session_raw_rms_db_min = raw_rms_db
        if self._session_raw_rms_db_max is None or raw_rms_db > self._session_raw_rms_db_max:
            self._session_raw_rms_db_max = raw_rms_db
        if self._session_band_energy_min is None or band_energy < self._session_band_energy_min:
            self._session_band_energy_min = band_energy
        if self._session_band_energy_max is None or band_energy > self._session_band_energy_max:
            self._session_band_energy_max = band_energy
        if self._session_flux_min is None or spectral_flux < self._session_flux_min:
            self._session_flux_min = spectral_flux
        if self._session_flux_max is None or spectral_flux > self._session_flux_max:
            self._session_flux_max = spectral_flux

    def record_shadow_telemetry(
        self,
        *,
        legacy_fire: bool,
        current_time: float,
        decision: TriggerDecision,
        acf_bpm: float,
        acf_confidence: float,
        phase_error_ms: float,
        smoothing_tag: str,
        frontend_ms: float = 0.0,
        tempo_ms: float = 0.0,
        detector_ms: float = 0.0,
        sidecar_ms: float = 0.0,
    ) -> None:
        if not self._new_trigger_telemetry_enabled:
            return

        self._shadow_telemetry.record(
            TriggerTelemetry(
                legacy_fire=bool(legacy_fire),
                new_fire=bool(decision.is_beat_candidate),
                beat_score=float(decision.beat_score),
                cue_flux=float(decision.c_flux),
                cue_band_spike=float(decision.c_band_spike),
                cue_energy_delta=float(decision.c_energy_delta),
                cue_phase_align=float(decision.c_phase_align),
                cue_sidecar=float(decision.c_sidecar),
                frontend_ms=float(frontend_ms),
                tempo_ms=float(tempo_ms),
                detector_ms=float(detector_ms),
                sidecar_ms=float(sidecar_ms),
                bus_raw_scores=dict(decision.bus_raw_scores),
                bus_masked_scores=dict(decision.bus_masked_scores),
                bus_pass=dict(decision.bus_pass),
                bus_reason_codes=dict(decision.bus_reason_codes),
                acf_bpm=float(acf_bpm),
                acf_confidence=float(acf_confidence),
                phase_error_ms=float(phase_error_ms),
                smoothing_tag=str(smoothing_tag),
                wall_time=float(current_time),
            )
        )

    def _compute_persistence_stats(
        self,
        values: list[float],
        sample_times: list[float],
        threshold: float,
        is_high: bool,
    ) -> dict[str, float]:
        if len(values) < 2 or len(sample_times) < 2:
            return {
                "total_s": 0.0,
                "episode_count": 0.0,
                "episode_mean_s": 0.0,
                "episode_max_s": 0.0,
            }

        durations: list[float] = []
        current_run_s = 0.0

        for idx in range(1, min(len(values), len(sample_times))):
            dt = max(0.0, sample_times[idx] - sample_times[idx - 1])
            value = values[idx]
            in_state = value >= threshold if is_high else value <= threshold
            if in_state:
                current_run_s += dt
            elif current_run_s > 0.0:
                durations.append(current_run_s)
                current_run_s = 0.0

        if current_run_s > 0.0:
            durations.append(current_run_s)

        if not durations:
            return {
                "total_s": 0.0,
                "episode_count": 0.0,
                "episode_mean_s": 0.0,
                "episode_max_s": 0.0,
            }

        total_s = float(np.sum(durations))
        episode_count = float(len(durations))
        return {
            "total_s": total_s,
            "episode_count": episode_count,
            "episode_mean_s": total_s / episode_count,
            "episode_max_s": float(np.max(durations)),
        }

    def summary_payload(self, elapsed_s: float) -> dict:
        raw_db_min = float(self._session_raw_rms_db_min or RMS_DB_FLOOR)
        raw_db_max = float(self._session_raw_rms_db_max or RMS_DB_FLOOR)
        band_min = float(self._session_band_energy_min or 0.0)
        band_max = float(self._session_band_energy_max or 0.0)
        flux_min = float(self._session_flux_min or 0.0)
        flux_max = float(self._session_flux_max or 0.0)

        frame_count = float(max(1, self._session_frame_count))
        raw_db_mean = self._session_raw_rms_db_sum / frame_count
        band_mean = self._session_band_energy_sum / frame_count
        flux_mean = self._session_flux_sum / frame_count

        flux_high_threshold = float(np.percentile(self._session_flux_samples, 90)) if self._session_flux_samples else 0.0
        peak_high_threshold = float(np.percentile(self._session_peak_samples, 90)) if self._session_peak_samples else 0.0
        trough_low_threshold = float(np.percentile(self._session_trough_samples, 10)) if self._session_trough_samples else 0.0

        flux_high = self._compute_persistence_stats(
            self._session_flux_samples,
            self._session_sample_times,
            flux_high_threshold,
            is_high=True,
        )
        peak_high = self._compute_persistence_stats(
            self._session_peak_samples,
            self._session_sample_times,
            peak_high_threshold,
            is_high=True,
        )
        trough_low = self._compute_persistence_stats(
            self._session_trough_samples,
            self._session_sample_times,
            trough_low_threshold,
            is_high=False,
        )

        payload = {
            "session_started_at": self._session_started_at,
            "session_ended_at": time.time(),
            "seconds": elapsed_s,
            "frames": self._session_frame_count,
            "raw_rms_db_low": raw_db_min,
            "raw_rms_db_high": raw_db_max,
            "raw_rms_db_mean": raw_db_mean,
            "band_energy_low": band_min,
            "band_energy_high": band_max,
            "band_energy_mean": band_mean,
            "flux_low": flux_min,
            "flux_high": flux_max,
            "flux_mean": flux_mean,
            "flux_high_threshold": flux_high_threshold,
            "peak_high_threshold": peak_high_threshold,
            "trough_low_threshold": trough_low_threshold,
            "flux_high_total_s": flux_high["total_s"],
            "flux_high_episode_count": flux_high["episode_count"],
            "flux_high_episode_mean_s": flux_high["episode_mean_s"],
            "flux_high_episode_max_s": flux_high["episode_max_s"],
            "peak_high_total_s": peak_high["total_s"],
            "peak_high_episode_count": peak_high["episode_count"],
            "peak_high_episode_mean_s": peak_high["episode_mean_s"],
            "peak_high_episode_max_s": peak_high["episode_max_s"],
            "trough_low_total_s": trough_low["total_s"],
            "trough_low_episode_count": trough_low["episode_count"],
            "trough_low_episode_mean_s": trough_low["episode_mean_s"],
            "trough_low_episode_max_s": trough_low["episode_max_s"],
        }
        if self._new_trigger_telemetry_enabled:
            payload.update(self._shadow_telemetry.summary())
        return payload

    def log_shutdown_summary(self) -> None:
        if self._session_frame_count <= 0:
            return

        elapsed_s = max(0.0, time.time() - self._session_started_at)
        payload = self.summary_payload(elapsed_s)

        log_event(
            "INFO",
            "Audio",
            "Shutdown levels summary",
            frames=self._session_frame_count,
            seconds=f"{elapsed_s:.1f}",
            raw_rms_db_min=f"{payload['raw_rms_db_low']:.2f}",
            raw_rms_db_max=f"{payload['raw_rms_db_high']:.2f}",
            raw_rms_db_mean=f"{payload['raw_rms_db_mean']:.2f}",
            raw_rms_db_span=f"{(payload['raw_rms_db_high'] - payload['raw_rms_db_low']):.2f}",
            band_energy_min=f"{payload['band_energy_low']:.6f}",
            band_energy_max=f"{payload['band_energy_high']:.6f}",
            band_energy_mean=f"{payload['band_energy_mean']:.6f}",
            band_energy_span=f"{(payload['band_energy_high'] - payload['band_energy_low']):.6f}",
            flux_min=f"{payload['flux_low']:.4f}",
            flux_max=f"{payload['flux_high']:.4f}",
            flux_mean=f"{payload['flux_mean']:.4f}",
            flux_span=f"{(payload['flux_high'] - payload['flux_low']):.4f}",
            flux_high_total_s=f"{payload['flux_high_total_s']:.3f}",
            peak_high_total_s=f"{payload['peak_high_total_s']:.3f}",
            trough_low_total_s=f"{payload['trough_low_total_s']:.3f}",
        )
