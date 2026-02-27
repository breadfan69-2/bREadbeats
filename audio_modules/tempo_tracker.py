from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .contracts import TempoState


def reference_bpm_for_onset_filters(
    metronome_bpm: float,
    acf_bpm_smoothed: float,
    smoothed_tempo: float,
) -> float:
    if float(metronome_bpm) > 0.0:
        return float(metronome_bpm)
    if float(acf_bpm_smoothed) > 0.0:
        return float(acf_bpm_smoothed)
    if float(smoothed_tempo) > 0.0:
        return float(smoothed_tempo)
    return 0.0


def effective_phase_accept_window_s(
    phase_accept_window_ms: float,
    phase_accept_low_conf_mult: float,
    acf_confidence: float,
) -> float:
    base_ms = float(np.clip(float(phase_accept_window_ms), 10.0, 300.0))
    low_conf_mult = float(np.clip(float(phase_accept_low_conf_mult), 1.0, 4.0))
    conf = float(np.clip(float(acf_confidence), 0.0, 1.0))
    if conf >= 0.25:
        mult = 1.0
    elif conf <= 0.05:
        mult = low_conf_mult
    else:
        t = (conf - 0.05) / 0.20
        mult = low_conf_mult + (1.0 - low_conf_mult) * t
    return (base_ms * mult) / 1000.0


def dedup_window_seconds(
    bpm_ref: float,
    beat_dedup_fraction: float,
    *,
    default_window_s: float = 0.10,
) -> float:
    if float(bpm_ref) <= 0.0:
        return float(default_window_s)
    beat_period_s = 60.0 / float(bpm_ref)
    dedup_frac = float(np.clip(float(beat_dedup_fraction), 0.05, 0.45))
    return dedup_frac * beat_period_s


def within_dedup_window(last_accepted_time: float, now: float, dedup_window_s: float) -> bool:
    return float(last_accepted_time) > 0.0 and (float(now) - float(last_accepted_time)) < float(dedup_window_s)


def metronome_phase_error_s(metronome_phase: float, metronome_bpm: float) -> float:
    if float(metronome_bpm) <= 0.0:
        return 0.0
    beat_period_s = 60.0 / float(metronome_bpm)
    phase_frac = float(metronome_phase) % 1.0
    phase_dist_frac = min(phase_frac, 1.0 - phase_frac)
    return phase_dist_frac * beat_period_s


def build_acf_octave_candidates(
    bpm: float,
    peak_value: float,
    raw_lag: int,
    min_lag: int,
    max_lag: int,
    fps: float,
    acf: np.ndarray,
) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = [(float(bpm), float(peak_value))]

    half_lag = int(raw_lag) // 2
    if half_lag >= int(min_lag):
        half_val = float(acf[half_lag])
        if half_val > float(peak_value) * 0.60:
            bpm_half = 60.0 * float(fps) / float(half_lag)
            if 55.0 <= bpm_half <= 200.0:
                candidates.append((float(bpm_half), half_val))

    double_lag = int(raw_lag) * 2
    if double_lag <= int(max_lag):
        double_val = float(acf[double_lag])
        if double_val > float(peak_value) * 0.60:
            bpm_double = 60.0 * float(fps) / float(double_lag)
            if 55.0 <= bpm_double <= 200.0:
                candidates.append((float(bpm_double), double_val))

    return candidates


def select_acf_octave_candidate(
    candidates: list[tuple[float, float]],
    peak_value: float,
    acf_confidence: float,
    octave_target_bias_confidence_max: float,
    *,
    target_bpm_hint: float = 0.0,
) -> tuple[float, float, str, Optional[list[tuple[float, float]]]]:
    if not candidates:
        return 0.0, float(peak_value), "none", None

    bpm_selected, conf_selected = candidates[0]
    use_target_guided_octave = (
        float(target_bpm_hint) > 0.0
        and len(candidates) > 1
        and float(acf_confidence) < float(octave_target_bias_confidence_max)
    )

    if use_target_guided_octave:
        def octave_score(c: tuple[float, float]) -> float:
            bpm_c, conf_c = c
            ratio = abs(float(bpm_c) - float(target_bpm_hint)) / float(target_bpm_hint)
            return ratio - float(conf_c) * 0.3

        ranked = sorted(candidates, key=octave_score)
        bpm_selected, conf_selected = ranked[0]
        return float(bpm_selected), float(conf_selected), "target-guided", ranked

    if len(candidates) > 1:
        fast = [c for c in candidates if c[1] > float(peak_value) * 0.75]
        if fast:
            fast.sort(key=lambda c: -float(c[0]))
            bpm_selected, conf_selected = fast[0]

    return float(bpm_selected), float(conf_selected), "fast-prefer", None


def estimate_onset_bpm_from_times(
    raw_onset_times: Sequence[float],
    *,
    max_points: int = 8,
    min_interval_s: float = 0.15,
    max_interval_s: float = 1.2,
    min_bpm: float = 55.0,
    max_bpm: float = 200.0,
) -> float:
    if len(raw_onset_times) < 3:
        return 0.0

    intervals = np.diff(np.array(list(raw_onset_times)[-int(max_points):], dtype=np.float64))
    if len(intervals) == 0:
        return 0.0

    valid = intervals[(intervals >= float(min_interval_s)) & (intervals <= float(max_interval_s))]
    if len(valid) < 2:
        return 0.0

    median_interval = float(np.median(valid))
    bpm = 60.0 / median_interval if median_interval > 0 else 0.0
    if float(min_bpm) <= bpm <= float(max_bpm):
        return bpm
    return 0.0


@dataclass(slots=True)
class TempoTrackerConfig:
    enabled: bool = True


@dataclass(slots=True)
class AcfSmoothingResult:
    smoothed_bpm: float
    updated: bool
    decision_tag: str


class TempoTracker:
    def __init__(self, config: TempoTrackerConfig | None = None):
        self.config = config or TempoTrackerConfig()
        self._state = TempoState()

    def reset(self) -> None:
        self._state = TempoState()

    def sync_runtime_state(
        self,
        *,
        metronome_bpm: float,
        acf_confidence: float,
        tempo_locked: bool,
        phase_error_ms: float,
        is_downbeat: bool,
        beat_phase: float,
    ) -> None:
        self._state = TempoState(
            metronome_bpm=float(metronome_bpm),
            acf_confidence=float(acf_confidence),
            tempo_locked=bool(tempo_locked),
            phase_error_ms=float(phase_error_ms),
            is_downbeat=bool(is_downbeat),
            beat_phase=float(beat_phase),
        )

    def get_state(self) -> TempoState:
        return self._state

    def update_from_acf_inputs(
        self,
        *,
        acf_confidence: float,
        onset_bpm: float,
        acf_bpm_smoothed: float,
        min_acf_weight: float,
        max_acf_weight: float,
    ) -> float:
        target_bpm = float(acf_bpm_smoothed)
        if float(onset_bpm) > 0.0:
            if target_bpm <= 0.0:
                target_bpm = float(onset_bpm)
            else:
                min_w = float(min_acf_weight)
                max_w = float(max_acf_weight)
                if max_w < min_w:
                    min_w, max_w = max_w, min_w
                acf_weight = min_w + (max_w - min_w) * float(acf_confidence)
                acf_weight = max(min_w, min(max_w, acf_weight))
                target_bpm = acf_weight * target_bpm + (1.0 - acf_weight) * float(onset_bpm)
        return float(target_bpm)

    def step_metronome_phase(self, current_phase: float, metronome_bpm: float, dt: float) -> tuple[float, int]:
        return step_metronome_phase(current_phase, metronome_bpm, dt)

    def smooth_acf_bpm_with_jump_gating(
        self,
        current_smoothed_bpm: float,
        bpm: float,
        peak_value: float,
        *,
        target_bpm_hint: float = 0.0,
        smooth_ratio_max: float = 0.15,
        jump_confidence_min: float = 0.25,
    ) -> AcfSmoothingResult:
        return smooth_acf_bpm_with_jump_gating(
            current_smoothed_bpm,
            bpm,
            peak_value,
            target_bpm_hint=target_bpm_hint,
            smooth_ratio_max=smooth_ratio_max,
            jump_confidence_min=jump_confidence_min,
        )


def step_metronome_phase(current_phase: float, metronome_bpm: float, dt: float) -> tuple[float, int]:
    if float(metronome_bpm) <= 0.0 or float(dt) <= 0.0:
        return float(current_phase), 0

    phase_step = (float(metronome_bpm) / 60.0) * float(dt)
    new_phase = float(current_phase) + phase_step
    crossings = max(0, int(new_phase) - int(float(current_phase)))
    return new_phase, int(crossings)


def smooth_acf_bpm_with_jump_gating(
    current_smoothed_bpm: float,
    bpm: float,
    peak_value: float,
    *,
    target_bpm_hint: float = 0.0,
    smooth_ratio_max: float = 0.15,
    jump_confidence_min: float = 0.25,
) -> AcfSmoothingResult:
    current = float(current_smoothed_bpm)
    candidate = float(bpm)
    conf = float(peak_value)
    target = float(target_bpm_hint)

    if current > 0.0:
        ratio = abs(candidate - current) / max(1e-9, current)
        if ratio < float(smooth_ratio_max):
            smoothed = (0.85 * current) + (0.15 * candidate)
            return AcfSmoothingResult(smoothed_bpm=float(smoothed), updated=True, decision_tag="smooth")

        if conf > float(jump_confidence_min):
            octave_like_jump = (0.45 < ratio < 0.55) or (0.90 < ratio < 1.10)
            if target > 0.0 and octave_like_jump:
                old_dist = abs(current - target)
                new_dist = abs(candidate - target)
                if new_dist < old_dist:
                    return AcfSmoothingResult(
                        smoothed_bpm=candidate,
                        updated=True,
                        decision_tag="jump-target-validated",
                    )
                return AcfSmoothingResult(
                    smoothed_bpm=current,
                    updated=False,
                    decision_tag="jump-target-rejected",
                )

            return AcfSmoothingResult(smoothed_bpm=candidate, updated=True, decision_tag="jump")

        return AcfSmoothingResult(smoothed_bpm=current, updated=False, decision_tag="ignored")

    return AcfSmoothingResult(smoothed_bpm=candidate, updated=True, decision_tag="initial")
