from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import FeatureFrame, TempoState
from .event_detector import EventDetector


@dataclass(slots=True)
class ReplayFrame:
    time_mono: float
    legacy_fire: bool
    flux_norm: float
    energy_norm: float
    energy_delta: float
    flux_delta: float
    hfc_proxy: float
    sub_bass: float
    low_mid: float
    mid: float
    high: float
    bass_dominance: float
    metronome_bpm: float
    acf_confidence: float
    tempo_locked: bool
    phase_error_ms: float
    beat_phase: float
    af_onset_conf: float | None = None
    raw_rms_db: float = -120.0


@dataclass(slots=True)
class ReplaySummary:
    samples: int
    agreement_count: int
    disagreement_count: int
    legacy_fire_count: int
    new_fire_count: int
    miss_count: int
    extra_fire_count: int
    silence_false_fire_count: int
    agreement_pct: float


def run_shadow_replay(
    frames: Iterable[ReplayFrame],
    detector: EventDetector,
    *,
    silence_db_threshold: float = -58.0,
) -> ReplaySummary:
    detector.reset()

    samples = 0
    agreement_count = 0
    disagreement_count = 0
    legacy_fire_count = 0
    new_fire_count = 0
    miss_count = 0
    extra_fire_count = 0
    silence_false_fire_count = 0

    for frame in frames:
        samples += 1
        legacy_fire = bool(frame.legacy_fire)
        if legacy_fire:
            legacy_fire_count += 1

        features = FeatureFrame(
            flux_norm=float(frame.flux_norm),
            energy_norm=float(frame.energy_norm),
            energy_delta=float(frame.energy_delta),
            flux_delta=float(frame.flux_delta),
            hfc_proxy=float(frame.hfc_proxy),
            sub_bass=float(frame.sub_bass),
            low_mid=float(frame.low_mid),
            mid=float(frame.mid),
            high=float(frame.high),
            bass_dominance=float(frame.bass_dominance),
            af_onset_conf=frame.af_onset_conf,
        )
        tempo = TempoState(
            metronome_bpm=float(frame.metronome_bpm),
            acf_confidence=float(frame.acf_confidence),
            tempo_locked=bool(frame.tempo_locked),
            phase_error_ms=float(frame.phase_error_ms),
            is_downbeat=False,
            beat_phase=float(frame.beat_phase),
        )
        decision = detector.detect(features, tempo, now_mono=float(frame.time_mono))
        new_fire = bool(decision.is_beat_candidate)

        if new_fire:
            new_fire_count += 1

        if new_fire == legacy_fire:
            agreement_count += 1
        else:
            disagreement_count += 1

        if legacy_fire and not new_fire:
            miss_count += 1
        elif new_fire and not legacy_fire:
            extra_fire_count += 1
            if float(frame.raw_rms_db) <= float(silence_db_threshold):
                silence_false_fire_count += 1

    agreement_pct = (100.0 * float(agreement_count) / float(samples)) if samples > 0 else 0.0
    return ReplaySummary(
        samples=samples,
        agreement_count=agreement_count,
        disagreement_count=disagreement_count,
        legacy_fire_count=legacy_fire_count,
        new_fire_count=new_fire_count,
        miss_count=miss_count,
        extra_fire_count=extra_fire_count,
        silence_false_fire_count=silence_false_fire_count,
        agreement_pct=agreement_pct,
    )
