from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.util
from typing import Callable

import numpy as np

from audio_modules.contracts import TempoState
from audio_modules.event_detector import EventDetector, EventDetectorConfig
from pmv_audio_analysis import AudioTimeline


_librosa_spec = importlib.util.find_spec("librosa")
if _librosa_spec is not None:  # pragma: no cover - optional dependency
    librosa = importlib.import_module("librosa")
    _HAS_LIBROSA = True
else:
    librosa = None
    _HAS_LIBROSA = False


@dataclass(slots=True)
class BeatDetectionConfig:
    sensitivity: float = 0.5
    refractory_ms: float = 170.0
    use_librosa: bool = True
    use_multibus: bool = True
    use_fft_peaks: bool = True
    plp_enabled: bool = True
    peak_seek_ratio: float = 1.0
    peak_beat_threshold: float = 0.5
    multibus_config: EventDetectorConfig = field(default_factory=EventDetectorConfig)


@dataclass(slots=True)
class BeatCandidate:
    time_ms: float
    confidence: float
    source: str
    bus_scores: dict[str, float] = field(default_factory=dict)
    beat_type: str = "beat"


@dataclass(slots=True)
class BeatTimeline:
    beats: list[BeatCandidate]
    tempo_bpm: float
    tempo_confidence: float
    beat_period_ms: float
    time_signature: int = 4


def _report(
    progress_callback: Callable[[str, float], None] | None,
    message: str,
    percent: float,
) -> None:
    if progress_callback is None:
        return
    progress_callback(message, float(percent))


def _estimate_bpm_from_flux(timeline: AudioTimeline) -> tuple[float, float]:
    flux = np.asarray(timeline.spectral_flux_per_frame, dtype=np.float64)
    if len(flux) < 8:
        return 0.0, 0.0

    centered = flux - float(np.mean(flux))
    if not np.any(np.abs(centered) > 1e-9):
        return 0.0, 0.0

    acf_full = np.correlate(centered, centered, mode="full")
    acf = acf_full[len(acf_full) // 2:]
    if len(acf) < 3 or float(acf[0]) <= 1e-9:
        return 0.0, 0.0

    if len(timeline.frame_times_ms) > 1:
        hop_s = max(1e-4, float(np.median(np.diff(timeline.frame_times_ms))) / 1000.0)
    else:
        hop_s = 0.02

    min_bpm = 55.0
    max_bpm = 200.0
    min_lag = max(1, int(round(60.0 / (max_bpm * hop_s))))
    max_lag = min(len(acf) - 1, int(round(60.0 / (min_bpm * hop_s))))
    if max_lag <= min_lag:
        return 0.0, 0.0

    search = acf[min_lag:max_lag + 1]
    if len(search) == 0:
        return 0.0, 0.0

    local_index = int(np.argmax(search))
    lag = min_lag + local_index
    peak = float(search[local_index])

    for alt_lag in (lag // 2, lag * 2):
        if alt_lag < min_lag or alt_lag > max_lag:
            continue
        alt_peak = float(acf[alt_lag])
        if alt_peak > peak * 0.9:
            lag = alt_lag
            peak = alt_peak

    bpm = 60.0 / (float(lag) * hop_s)
    if not np.isfinite(bpm) or bpm <= 0.0:
        return 0.0, 0.0

    confidence = float(np.clip(peak / max(float(acf[0]), 1e-9), 0.0, 1.0))
    return float(np.clip(bpm, min_bpm, max_bpm)), confidence


def _detect_librosa(
    samples: np.ndarray,
    sr: int,
    plp_enabled: bool,
    hop_length: int,
) -> tuple[list[BeatCandidate], float]:
    if not _HAS_LIBROSA:
        return [], 0.0

    assert librosa is not None

    hop = max(1, int(hop_length))
    y = np.asarray(samples, dtype=np.float32)
    onset_env = librosa.onset.onset_strength(y=y, sr=int(sr), hop_length=hop)

    tempo, beat_frames = librosa.beat.beat_track(
        y=y,
        sr=int(sr),
        hop_length=hop,
        units="frames",
    )
    beat_frames = np.asarray(beat_frames, dtype=np.int64)

    if plp_enabled and len(onset_env) > 0:
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=int(sr), hop_length=hop)
        if len(pulse) == len(onset_env):
            onset_env = np.maximum(onset_env, pulse)

    if len(beat_frames) == 0:
        return [], float(tempo)

    p95 = float(np.percentile(onset_env, 95)) if len(onset_env) else 1.0
    p95 = max(1e-9, p95)

    beat_times = librosa.frames_to_time(beat_frames, sr=int(sr), hop_length=hop)
    candidates: list[BeatCandidate] = []
    for beat_frame, beat_time_s in zip(beat_frames, beat_times):
        if beat_frame < 0 or beat_frame >= len(onset_env):
            conf = 0.5
        else:
            conf = float(np.clip(float(onset_env[beat_frame]) / p95, 0.0, 1.0))
        candidates.append(
            BeatCandidate(
                time_ms=float(beat_time_s) * 1000.0,
                confidence=conf,
                source="librosa",
            )
        )
    return candidates, float(tempo)


def _detect_multibus(
    timeline: AudioTimeline,
    config: EventDetectorConfig,
    sensitivity: float,
    tempo_hint_bpm: float,
    tempo_hint_confidence: float,
) -> list[BeatCandidate]:
    detector = EventDetector(config)

    bpm = float(tempo_hint_bpm)
    conf = float(tempo_hint_confidence)
    if bpm <= 0.0:
        bpm, conf = _estimate_bpm_from_flux(timeline)

    beat_period_s = 60.0 / bpm if bpm > 0.0 else 0.0
    threshold = float(np.interp(float(np.clip(sensitivity, 0.0, 1.0)), [0.0, 1.0], [0.32, 0.80]))

    candidates: list[BeatCandidate] = []
    anchor_s = float(timeline.frame_times_ms[0]) / 1000.0 if len(timeline.frame_times_ms) else 0.0

    for idx, feature in enumerate(timeline.feature_frames):
        if idx >= len(timeline.frame_times_ms):
            break

        now_s = float(timeline.frame_times_ms[idx]) / 1000.0
        if beat_period_s > 0.0:
            phase = ((now_s - anchor_s) / beat_period_s) % 1.0
            phase_err_ms = min(phase, 1.0 - phase) * beat_period_s * 1000.0
            beat_count = int(max(0.0, round((now_s - anchor_s) / beat_period_s)))
            is_downbeat = bool(beat_count % 4 == 0)
        else:
            phase = 0.0
            phase_err_ms = 0.0
            is_downbeat = False

        tempo_state = TempoState(
            metronome_bpm=float(bpm),
            acf_confidence=float(conf),
            tempo_locked=bool(conf > 0.2),
            phase_error_ms=float(phase_err_ms),
            is_downbeat=is_downbeat,
            beat_phase=float(phase),
        )

        decision = detector.detect(feature, tempo_state, now_mono=now_s)
        if decision.is_beat_candidate and float(decision.beat_score) >= threshold:
            candidates.append(
                BeatCandidate(
                    time_ms=float(timeline.frame_times_ms[idx]),
                    confidence=float(np.clip(decision.beat_score, 0.0, 1.0)),
                    source="multibus",
                    bus_scores={k: float(v) for k, v in decision.bus_scores.items()},
                )
            )

    return candidates


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    w = max(1, int(width))
    if len(values) == 0 or w <= 1:
        return values.astype(np.float32, copy=True)
    kernel = np.ones(w, dtype=np.float64) / float(w)
    padded = np.pad(values.astype(np.float64), (w // 2, w - 1 - (w // 2)), mode="edge")
    out = np.convolve(padded, kernel, mode="valid")
    return out.astype(np.float32)


def _detect_fft_peaks(
    timeline: AudioTimeline,
    peak_seek_ratio: float,
    peak_beat_threshold: float,
    sensitivity: float,
) -> list[BeatCandidate]:
    flux = np.asarray(timeline.spectral_flux_per_frame, dtype=np.float32)
    times = np.asarray(timeline.frame_times_ms, dtype=np.float32)
    if len(flux) < 3 or len(times) != len(flux):
        return []

    novelty = flux * float(max(0.01, peak_seek_ratio))
    local_mean = _moving_average(novelty, 9)

    sensitivity_scale = float(np.interp(np.clip(sensitivity, 0.0, 1.0), [0.0, 1.0], [1.30, 0.70]))
    base_threshold = float(np.clip(peak_beat_threshold, 0.0, 2.0)) * sensitivity_scale

    candidates: list[BeatCandidate] = []
    for i in range(1, len(novelty) - 1):
        curr = float(novelty[i])
        if curr < base_threshold:
            continue
        if curr < float(local_mean[i]) * 1.08:
            continue
        if not (curr >= float(novelty[i - 1]) and curr > float(novelty[i + 1])):
            continue

        conf = float(np.clip((curr - base_threshold) / max(base_threshold, 1e-6), 0.0, 1.0))
        candidates.append(
            BeatCandidate(
                time_ms=float(times[i]),
                confidence=conf,
                source="fft_peak",
            )
        )

    return candidates


def _combine_cluster(cluster: list[BeatCandidate]) -> BeatCandidate:
    best = max(cluster, key=lambda c: c.confidence)
    merged_bus: dict[str, float] = {}
    for cand in cluster:
        for key, value in cand.bus_scores.items():
            merged_bus[key] = max(float(value), float(merged_bus.get(key, 0.0)))

    unique_sources = sorted({cand.source for cand in cluster})
    boosted_conf = min(1.0, float(best.confidence) + 0.15 * float(len(unique_sources) - 1))
    return BeatCandidate(
        time_ms=float(best.time_ms),
        confidence=float(boosted_conf),
        source="+".join(unique_sources),
        bus_scores=merged_bus,
        beat_type=best.beat_type,
    )


def _merge_candidates(
    candidates: list[list[BeatCandidate]],
    refractory_ms: float,
) -> list[BeatCandidate]:
    flat: list[BeatCandidate] = [item for group in candidates for item in group]
    if not flat:
        return []

    flat.sort(key=lambda c: c.time_ms)
    refractory = max(1.0, float(refractory_ms))

    grouped: list[BeatCandidate] = []
    cluster: list[BeatCandidate] = [flat[0]]

    for cand in flat[1:]:
        if float(cand.time_ms) - float(cluster[-1].time_ms) <= refractory:
            cluster.append(cand)
            continue
        grouped.append(_combine_cluster(cluster))
        cluster = [cand]

    grouped.append(_combine_cluster(cluster))

    deduped: list[BeatCandidate] = []
    for cand in grouped:
        if not deduped:
            deduped.append(cand)
            continue
        if float(cand.time_ms) - float(deduped[-1].time_ms) < refractory:
            if float(cand.confidence) > float(deduped[-1].confidence):
                deduped[-1] = cand
            continue
        deduped.append(cand)

    return deduped


def _classify_beats(
    beats: list[BeatCandidate],
    tempo_bpm: float,
    time_signature: int = 4,
) -> list[BeatCandidate]:
    if not beats:
        return []
    if tempo_bpm <= 0.0:
        return [BeatCandidate(**vars(b)) for b in beats]

    period_ms = 60000.0 / float(max(1e-9, tempo_bpm))
    anchor = float(beats[0].time_ms)

    classified: list[BeatCandidate] = []
    for beat in beats:
        grid_pos = int(round((float(beat.time_ms) - anchor) / period_ms))
        expected = anchor + (float(grid_pos) * period_ms)
        phase_error = abs(float(beat.time_ms) - expected)
        on_grid = phase_error < (0.25 * period_ms)

        if on_grid and grid_pos % max(1, int(time_signature)) == 0:
            beat_type = "downbeat"
        elif on_grid:
            beat_type = "beat"
        else:
            beat_type = "syncopation"

        classified.append(
            BeatCandidate(
                time_ms=float(beat.time_ms),
                confidence=float(beat.confidence),
                source=str(beat.source),
                bus_scores=dict(beat.bus_scores),
                beat_type=beat_type,
            )
        )

    return classified


def _estimate_tempo(
    beats: list[BeatCandidate],
    timeline: AudioTimeline,
    tempo_hint_bpm: float,
) -> tuple[float, float]:
    hint = float(tempo_hint_bpm)

    intervals = np.diff(np.array([b.time_ms for b in beats], dtype=np.float64)) if len(beats) >= 2 else np.array([], dtype=np.float64)
    valid = intervals[(intervals >= 220.0) & (intervals <= 2200.0)]

    ibi_bpm = 0.0
    ibi_conf = 0.0
    if len(valid) >= 3:
        median_interval = float(np.median(valid))
        ibi_bpm = 60000.0 / median_interval if median_interval > 0.0 else 0.0
        rel_std = float(np.std(valid) / max(1e-9, np.mean(valid)))
        ibi_conf = float(np.clip(1.0 - rel_std * 2.0, 0.0, 1.0))

    acf_bpm, acf_conf = _estimate_bpm_from_flux(timeline)

    def _is_close(a: float, b: float, rel: float = 0.15) -> bool:
        if a <= 0.0 or b <= 0.0:
            return False
        return abs(a - b) / max(a, b) <= rel

    def _is_octave_related(a: float, b: float, rel: float = 0.12) -> bool:
        if a <= 0.0 or b <= 0.0:
            return False
        ratio = max(a, b) / max(1e-9, min(a, b))
        return abs(ratio - 2.0) <= rel

    if ibi_bpm > 0.0:
        bpm = ibi_bpm
        confidence = max(0.55, ibi_conf)

        if hint > 0.0 and _is_close(ibi_bpm, hint, rel=0.18):
            bpm = (0.75 * ibi_bpm) + (0.25 * hint)
            confidence = min(1.0, confidence + 0.10)

        if acf_bpm > 0.0:
            if _is_close(ibi_bpm, acf_bpm, rel=0.18):
                bpm = (0.80 * bpm) + (0.20 * acf_bpm)
                confidence = min(1.0, confidence + 0.08 * acf_conf)
            elif _is_octave_related(ibi_bpm, acf_bpm):
                confidence = min(1.0, confidence + 0.03 * acf_conf)

        return float(np.clip(bpm, 55.0, 200.0)), float(np.clip(confidence, 0.0, 1.0))

    candidates: list[tuple[float, float]] = []
    if hint > 0.0:
        candidates.append((hint, 0.65))
    if acf_bpm > 0.0:
        candidates.append((acf_bpm, max(0.25, acf_conf)))

    if not candidates:
        return 0.0, 0.0

    weighted_sum = sum(float(bpm) * float(weight) for bpm, weight in candidates)
    weight_total = sum(float(weight) for _, weight in candidates)
    bpm = float(np.clip(weighted_sum / max(1e-9, weight_total), 55.0, 200.0))
    confidence = float(np.clip(weight_total / max(1.0, len(candidates)), 0.0, 1.0))
    return bpm, confidence


def detect_beats(
    timeline: AudioTimeline,
    config: BeatDetectionConfig,
    progress_callback: Callable[[str, float], None] | None = None,
) -> BeatTimeline:
    detector_outputs: list[list[BeatCandidate]] = []

    if len(timeline.feature_frames) == 0:
        return BeatTimeline(beats=[], tempo_bpm=0.0, tempo_confidence=0.0, beat_period_ms=0.0)

    if len(timeline.frame_times_ms) > 1:
        hop_ms = max(1.0, float(np.median(np.diff(timeline.frame_times_ms))))
    else:
        hop_ms = 20.0
    hop_samples = int(round((hop_ms / 1000.0) * float(timeline.sample_rate)))

    tempo_hint_bpm = 0.0
    tempo_hint_conf = 0.0

    if config.use_librosa:
        _report(progress_callback, "Running librosa beat detection...", 0.0)
        librosa_candidates, tempo_hint_bpm = _detect_librosa(
            timeline.samples,
            timeline.sample_rate,
            config.plp_enabled,
            max(1, hop_samples),
        )
        detector_outputs.append(librosa_candidates)
        tempo_hint_conf = 0.65 if tempo_hint_bpm > 0.0 else 0.0
    _report(progress_callback, "Running librosa beat detection...", 30.0)

    if config.use_multibus:
        _report(progress_callback, "Running multi-bus detection...", 30.0)
        multibus_candidates = _detect_multibus(
            timeline,
            config.multibus_config,
            config.sensitivity,
            tempo_hint_bpm,
            tempo_hint_conf,
        )
        detector_outputs.append(multibus_candidates)
    _report(progress_callback, "Running multi-bus detection...", 60.0)

    if config.use_fft_peaks:
        _report(progress_callback, "Running FFT peak detection...", 60.0)
        fft_candidates = _detect_fft_peaks(
            timeline,
            config.peak_seek_ratio,
            config.peak_beat_threshold,
            config.sensitivity,
        )
        detector_outputs.append(fft_candidates)
    _report(progress_callback, "Running FFT peak detection...", 80.0)

    _report(progress_callback, "Merging and classifying...", 80.0)
    merged = _merge_candidates(detector_outputs, config.refractory_ms)
    tempo_bpm, tempo_conf = _estimate_tempo(merged, timeline, tempo_hint_bpm)
    classified = _classify_beats(merged, tempo_bpm, time_signature=4)
    period_ms = (60000.0 / tempo_bpm) if tempo_bpm > 0.0 else 0.0
    _report(progress_callback, "Merging and classifying...", 100.0)

    return BeatTimeline(
        beats=classified,
        tempo_bpm=float(tempo_bpm),
        tempo_confidence=float(tempo_conf),
        beat_period_ms=float(period_ms),
        time_signature=4,
    )


__all__ = [
    "BeatCandidate",
    "BeatDetectionConfig",
    "BeatTimeline",
    "detect_beats",
]
