from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio_engine import BeatEvent
from config import Config


@dataclass
class BandEnergies:
    sub_bass: float = 0.0
    low_mid: float = 0.0
    mid: float = 0.0
    high: float = 0.0


@dataclass
class BeatDecision:
    trigger_kind: str
    interval_beats: int
    radius_bloom: float
    silence_active: bool
    journey_completion: float


class BeatIntelligence:
    """Signal-domain decision engine for orbit control."""

    def __init__(self, config: Config, audio_engine=None, park_y: float = -0.70):
        self.config = config
        self.audio_engine = audio_engine
        self.park_y = -0.70 if park_y is None else float(park_y)

        self.band_ema_alpha = 0.2
        self.energies = BandEnergies()

        self.rms_envelope = 0.0
        self.rms_attack = 0.15
        self.rms_release = 0.05

        self.silence_deadzone_active = False
        self.silence_open_count = 0
        self.silence_close_count = 0

        self.active_interval_beats = 8
        self.last_trigger_kind = "creep"

        self.journey_duration_s = 0.0
        self.journey_elapsed_s = 0.0
        self.journey_active = False

        self.treble_lift_ema = 0.0
        self.treble_lift_attack = 0.28
        self.treble_lift_release = 0.16

    def set_audio_engine(self, audio_engine) -> None:
        self.audio_engine = audio_engine

    def set_park_y(self, park_y: float) -> None:
        self.park_y = -0.70 if park_y is None else float(park_y)

    def update_band_energies(self) -> None:
        energies = {}
        if self.audio_engine is not None and hasattr(self.audio_engine, "_band_energies"):
            maybe = getattr(self.audio_engine, "_band_energies", None)
            if isinstance(maybe, dict):
                energies = maybe

        self.energies.sub_bass += (float(energies.get("sub_bass", 0.0)) - self.energies.sub_bass) * self.band_ema_alpha
        self.energies.low_mid += (float(energies.get("low_mid", 0.0)) - self.energies.low_mid) * self.band_ema_alpha
        self.energies.mid += (float(energies.get("mid", 0.0)) - self.energies.mid) * self.band_ema_alpha
        self.energies.high += (float(energies.get("high", 0.0)) - self.energies.high) * self.band_ema_alpha

    def update_envelope(self, event: BeatEvent) -> None:
        raw_rms = float(getattr(event, "raw_rms", 0.0) or 0.0)
        target = max(0.0, raw_rms)
        alpha = self.rms_attack if target >= self.rms_envelope else self.rms_release
        self.rms_envelope += (target - self.rms_envelope) * alpha

    def get_overall_amplitude(self, event: BeatEvent) -> float:
        peak = float(getattr(event, "peak_energy", 0.0) or 0.0)
        return float(max(self.rms_envelope, peak * 0.35))

    def update_silence_deadzone_gate(self, overall_amplitude: float) -> bool:
        threshold = float(getattr(self.config.stroke, "silence_threshold", 0.04) or 0.04)
        gate_low = float(getattr(self.config.stroke, "amplitude_gate_low", 0.04) or 0.04)
        open_threshold = max(0.0, min(threshold, gate_low))
        close_threshold = max(open_threshold + 1e-6, open_threshold * 1.20)

        if overall_amplitude < open_threshold:
            self.silence_open_count += 1
            self.silence_close_count = 0
            if self.silence_open_count >= 3:
                self.silence_deadzone_active = True
        elif overall_amplitude > close_threshold:
            self.silence_close_count += 1
            self.silence_open_count = 0
            if self.silence_close_count >= 2:
                self.silence_deadzone_active = False
        else:
            self.silence_open_count = max(0, self.silence_open_count - 1)
            self.silence_close_count = max(0, self.silence_close_count - 1)

        return self.silence_deadzone_active

    def classify_trigger(self, event: BeatEvent) -> str:
        if bool(getattr(event, "is_syncopated", False)) and bool(getattr(self.config.beat, "syncopation_enabled", True)):
            return "syncopation"
        if bool(getattr(event, "is_downbeat", False)):
            return "downbeat"
        if bool(getattr(event, "is_beat", False)):
            return "beat"
        return "creep"

    def _tempo_ready_for_motion(self, event: BeatEvent) -> bool:
        if not bool(getattr(self.config.beat, "tempo_lock_required", True)):
            return True
        if bool(getattr(event, "tempo_locked", False)):
            return True
        relaxed = float(getattr(self.config.beat, "teaching_metronome_relaxed_confidence", 0.14) or 0.14)
        acf_conf = float(getattr(event, "acf_confidence", 0.0) or 0.0)
        return acf_conf >= relaxed

    def _strict_bass_motion_allowed(self, event: BeatEvent, trigger_kind: str) -> bool:
        if trigger_kind not in ("beat", "syncopation"):
            return True
        if not bool(getattr(self.config.beat, "strict_bass_motion_gate_enabled", False)):
            return True

        beat_band = str(getattr(event, "beat_band", "") or "")
        if beat_band in ("sub_bass", "low_mid"):
            return True

        fired = getattr(event, "fired_bands", None)
        if isinstance(fired, (list, tuple, set)):
            fired_set = {str(item) for item in fired}
            if "sub_bass" in fired_set or "low_mid" in fired_set:
                return True

        return False

    @staticmethod
    def interval_beats_for_trigger(trigger_kind: str) -> int:
        if trigger_kind == "syncopation":
            return 1
        if trigger_kind == "beat":
            return 2
        if trigger_kind == "downbeat":
            return 4
        return 8

    @staticmethod
    def effective_bpm(event: BeatEvent) -> float:
        bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
        if bpm <= 0.0:
            bpm = float(getattr(event, "bpm", 0.0) or 0.0)
        if bpm <= 0.0:
            bpm = 120.0
        return float(np.clip(bpm, 40.0, 240.0))

    def compute_radius_bloom_from_sub_bass(self, event: BeatEvent | None = None) -> float:
        base_radius = 0.70
        max_radius = 0.95
        max_bloom = max_radius - base_radius

        sub_bass = float(np.clip(self.energies.sub_bass, 0.0, 1.0))
        low_mid = float(np.clip(self.energies.low_mid, 0.0, 1.0))

        weighted_bass = (sub_bass * 0.70) + (low_mid * 0.30)
        bass_fill = float(np.clip(max(sub_bass, weighted_bass), 0.0, 1.0))
        bloom = max_bloom * (bass_fill ** 1.5)

        if event is not None:
            spectral_flux = float(getattr(event, "spectral_flux", 0.0) or 0.0)
            flux_threshold = float(getattr(self.config.stroke, "flux_threshold", 0.03) or 0.03)
            flux_ratio = float(np.clip(spectral_flux / max(flux_threshold, 1e-6), 0.0, 2.0))
            flux_boost = max_bloom * 0.15 * flux_ratio
            bloom += flux_boost

        return float(np.clip(base_radius + bloom, base_radius, max_radius))

    def update_journey_progress(self, trigger_kind: str, interval_beats: int, event: BeatEvent, dt: float) -> float:
        bpm = self.effective_bpm(event)
        beat_period_s = 60.0 / max(1e-6, bpm)
        target_duration = max(1e-3, beat_period_s * float(interval_beats))

        trigger_started = bool(
            (trigger_kind == "syncopation" and bool(getattr(event, "is_syncopated", False)))
            or (trigger_kind == "downbeat" and bool(getattr(event, "is_downbeat", False)))
            or (trigger_kind == "beat" and bool(getattr(event, "is_beat", False)))
            or (trigger_kind == "creep" and self.last_trigger_kind != "creep" and not self.journey_active)
        )

        if trigger_started or not self.journey_active or self.active_interval_beats != interval_beats:
            self.journey_duration_s = target_duration
            self.journey_elapsed_s = 0.0
            self.journey_active = True
            return 0.0

        step = float(np.clip(dt, 1e-4, 0.25))
        self.journey_elapsed_s = min(self.journey_duration_s, self.journey_elapsed_s + step)
        completion = float(np.clip(self.journey_elapsed_s / max(1e-6, self.journey_duration_s), 0.0, 1.0))
        if completion >= 1.0:
            self.journey_active = False
        return completion

    def compute_treble_lift(self, journey_completion: float) -> float:
        max_lift = 0.40
        treble_fill = float(np.clip((self.energies.high * 0.75) + (self.energies.mid * 0.25), 0.0, 1.0))
        lift_factor = treble_fill ** 2.0
        target_offset = max_lift * lift_factor

        alpha = self.treble_lift_attack if target_offset >= self.treble_lift_ema else self.treble_lift_release
        self.treble_lift_ema += (target_offset - self.treble_lift_ema) * alpha
        smoothed_offset = float(np.clip(self.treble_lift_ema, 0.0, max_lift))

        guard_start = 0.80
        p = float(np.clip(journey_completion, 0.0, 1.0))
        if p <= guard_start:
            guard = 1.0
        else:
            t = float(np.clip((p - guard_start) / max(1e-6, 1.0 - guard_start), 0.0, 1.0))
            smooth_t = t * t * (3.0 - 2.0 * t)
            guard = 1.0 - smooth_t

        # Returns vertical center offset (0..max_lift), not absolute Y.
        # At journey completion, this is forced to 0 by the landing guard.
        return float(smoothed_offset * guard)

    def build_decision(self, event: BeatEvent, dt: float, silence_override: bool | None = None) -> BeatDecision:
        self.update_band_energies()
        self.update_envelope(event)

        overall_amplitude = self.get_overall_amplitude(event)
        silence_active = self.update_silence_deadzone_gate(overall_amplitude)
        if silence_override is not None:
            silence_active = bool(silence_override)

        raw_trigger_kind = self.classify_trigger(event)
        trigger_kind = raw_trigger_kind

        # Preserve active beat-family journeys between discrete beat/downbeat/sync events.
        if (
            raw_trigger_kind == "creep"
            and self.journey_active
            and self.last_trigger_kind in ("syncopation", "beat", "downbeat")
        ):
            trigger_kind = self.last_trigger_kind

        # Tempo/bass gates apply to newly detected beat-family events.
        if raw_trigger_kind in ("syncopation", "beat", "downbeat") and not silence_active:
            if not self._tempo_ready_for_motion(event):
                trigger_kind = "creep"
            elif not self._strict_bass_motion_allowed(event, raw_trigger_kind):
                trigger_kind = "creep"

        interval_beats = self.interval_beats_for_trigger(trigger_kind)
        radius_bloom = self.compute_radius_bloom_from_sub_bass(event=event)
        journey_completion = self.update_journey_progress(trigger_kind, interval_beats, event, dt)

        self.active_interval_beats = interval_beats
        self.last_trigger_kind = trigger_kind

        return BeatDecision(
            trigger_kind=trigger_kind,
            interval_beats=interval_beats,
            radius_bloom=radius_bloom,
            silence_active=bool(silence_active),
            journey_completion=journey_completion,
        )
