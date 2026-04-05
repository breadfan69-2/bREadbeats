from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .contracts import BusDecision, BusState, FeatureFrame, TempoState, TriggerDecision


BUS_NAMES: tuple[str, str, str, str] = ("sub_bass", "low_mid", "mid", "high")


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
    bus_arm_threshold: float = 0.30
    bus_release_threshold: float = 0.25
    bus_refractory_ms: float = 170.0
    bus_sustain_frames: int = 2
    bus_hist_size: int = 48
    bus_mask_floor: float = 0.25
    w_bus_sub: float = 0.36
    w_bus_low: float = 0.30
    w_bus_mid: float = 0.20
    w_bus_high: float = 0.14
    bass_dominance_weighting_enabled: bool = False
    transient_classification_enabled: bool = False


class EventDetector:
    def __init__(self, config: EventDetectorConfig | None = None):
        self.config = config or EventDetectorConfig()
        self._armed = False
        self._last_fire_mono = 0.0
        self._bus_states: dict[str, BusState] = self._init_bus_states()

    def reset(self) -> None:
        self._armed = False
        self._last_fire_mono = 0.0
        self._bus_states = self._init_bus_states()

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    def _init_bus_states(self) -> dict[str, BusState]:
        return {
            name: BusState(
                name=name,
                refractory_ms=float(np.clip(self.config.bus_refractory_ms, 0.0, 1000.0)),
                arm_threshold=float(np.clip(self.config.bus_arm_threshold, 0.20, 0.95)),
                release_threshold=float(np.clip(self.config.bus_release_threshold, 0.10, 0.90)),
                sustain_frames=max(1, int(self.config.bus_sustain_frames)),
            )
            for name in BUS_NAMES
        }

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

    def _effective_bus_weights(self, features: FeatureFrame) -> dict[str, float]:
        weights = {
            "sub_bass": max(0.01, float(self.config.w_bus_sub)),
            "low_mid": max(0.01, float(self.config.w_bus_low)),
            "mid": max(0.01, float(self.config.w_bus_mid)),
            "high": max(0.01, float(self.config.w_bus_high)),
        }

        if self.config.bass_dominance_weighting_enabled:
            dominance = float(np.clip(features.bass_dominance, 0.25, 4.0))
            if dominance > 1.0:
                shift = min(0.18, 0.09 * (dominance - 1.0))
                weights["sub_bass"] += shift * 0.55
                weights["low_mid"] += shift * 0.45
                weights["mid"] = max(0.01, weights["mid"] - shift * 0.45)
                weights["high"] = max(0.01, weights["high"] - shift * 0.55)
            elif dominance < 1.0:
                shift = min(0.18, 0.09 * (1.0 - dominance))
                weights["high"] += shift * 0.60
                weights["mid"] += shift * 0.40
                weights["sub_bass"] = max(0.01, weights["sub_bass"] - shift * 0.55)
                weights["low_mid"] = max(0.01, weights["low_mid"] - shift * 0.45)

        total = max(1e-9, sum(weights.values()))
        for key in weights:
            weights[key] = float(weights[key] / total)
        return weights

    def _bus_energy(self, features: FeatureFrame, name: str) -> float:
        return self._clip01(getattr(features, name, 0.0))

    def _score_single_bus(
        self,
        bus_name: str,
        features: FeatureFrame,
        *,
        now_mono: float,
    ) -> tuple[BusDecision, float, float, float]:
        state = self._bus_states[bus_name]
        bus_energy = self._bus_energy(features, bus_name)
        prev_env = float(state.env)

        state.env = (0.82 * prev_env) + (0.18 * bus_energy)
        if state.noise_floor <= 0.0:
            state.noise_floor = bus_energy
        elif bus_energy < state.noise_floor:
            state.noise_floor = (0.92 * state.noise_floor) + (0.08 * bus_energy)
        else:
            state.noise_floor = (0.995 * state.noise_floor) + (0.005 * bus_energy)

        hist = state.z_hist
        if len(hist) >= 4:
            h = np.array(hist, dtype=float)
            mean = float(np.mean(h))
            std = max(1e-5, float(np.std(h)))
            z_score = float((bus_energy - mean) / std)
        else:
            z_score = 0.0
        hist.append(bus_energy)
        max_hist = max(8, int(self.config.bus_hist_size))
        if len(hist) > max_hist:
            del hist[0:len(hist) - max_hist]

        local_flux = self._clip01(float(features.flux_norm) * (0.35 + (0.65 * bus_energy)))
        local_delta = self._clip01((max(0.0, bus_energy - prev_env) * 2.1) + (0.20 * float(features.energy_delta)))
        z_spike = self._clip01((max(0.0, z_score) / 3.0) + (0.45 * max(0.0, bus_energy - state.noise_floor)))

        w_flux, w_band, w_delta, _, _ = self._effective_weights(features)
        total_cue_weight = max(1e-9, w_flux + w_band + w_delta)
        raw_bus_score = (
            (w_flux * local_flux)
            + (w_band * bus_energy)
            + (w_delta * max(local_delta, z_spike))
        ) / total_cue_weight

        refractory_s = max(0.0, float(state.refractory_ms) / 1000.0)
        in_refractory = (
            now_mono > 0.0
            and state.last_onset_mono > 0.0
            and (now_mono - state.last_onset_mono) < refractory_s
        )

        passed_onset = bool(raw_bus_score >= float(state.arm_threshold)) and not in_refractory
        sustain_conf = self._clip01((0.55 * raw_bus_score) + (0.45 * local_delta))
        passed_sustain = (
            (state.active_frames >= max(1, int(state.sustain_frames)))
            and bool(sustain_conf >= float(state.release_threshold))
        )

        reason_codes: list[str] = []
        if in_refractory:
            reason_codes.append("refractory")
        if passed_onset:
            reason_codes.append("onset")
            state.last_onset_mono = now_mono if now_mono > 0.0 else state.last_onset_mono
            state.active_frames += 1
            state.inactive_frames = 0
        elif passed_sustain:
            reason_codes.append("sustain")
            state.active_frames += 1
            state.inactive_frames = 0
        else:
            reason_codes.append("below_gate")
            state.inactive_frames += 1
            if state.inactive_frames >= 2:
                state.active_frames = 0

        eligible = bool(passed_onset or passed_sustain)
        decision = BusDecision(
            name=bus_name,
            onset_conf=float(raw_bus_score),
            sustain_conf=float(sustain_conf),
            passed_onset_gate=bool(passed_onset),
            passed_sustain_gate=bool(passed_sustain),
            in_refractory=bool(in_refractory),
            eligible=bool(eligible),
            reason_codes=reason_codes,
        )
        return decision, float(raw_bus_score), float(local_flux), float(local_delta)

    def _apply_bleed_masks(
        self,
        raw_scores: dict[str, float],
        features: FeatureFrame,
    ) -> dict[str, float]:
        mask = {name: 1.0 for name in BUS_NAMES}
        mask_floor = float(np.clip(self.config.bus_mask_floor, 0.20, 0.90))

        high_dominant = (
            float(raw_scores["high"]) >= 0.68
            and float(features.high) > max(float(features.sub_bass), float(features.low_mid))
            and float(features.hfc_proxy) >= 0.55
        )
        bass_weak = max(float(raw_scores["sub_bass"]), float(raw_scores["low_mid"])) <= 0.42
        if high_dominant and bass_weak:
            mask["sub_bass"] *= 0.45
            mask["low_mid"] *= 0.58

        bass_dominant = (
            max(float(raw_scores["sub_bass"]), float(raw_scores["low_mid"])) >= 0.64
            and (float(features.sub_bass) + float(features.low_mid))
            > 1.15 * (float(features.mid) + float(features.high) + 1e-6)
        )
        high_weak = float(raw_scores["high"]) <= 0.45
        if bass_dominant and high_weak:
            mask["high"] *= 0.52

        return {
            name: float(raw_scores[name] * np.clip(mask[name], mask_floor, 1.0))
            for name in BUS_NAMES
        }

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

        now = float(now_mono) if now_mono is not None else 0.0
        bus_decisions: dict[str, BusDecision] = {}
        bus_raw_scores: dict[str, float] = {}
        bus_flux_cues: dict[str, float] = {}
        bus_delta_cues: dict[str, float] = {}

        for bus_name in BUS_NAMES:
            bus_decision, raw_score, local_flux, local_delta = self._score_single_bus(
                bus_name,
                features,
                now_mono=now,
            )
            bus_decisions[bus_name] = bus_decision
            bus_raw_scores[bus_name] = float(raw_score)
            bus_flux_cues[bus_name] = float(local_flux)
            bus_delta_cues[bus_name] = float(local_delta)

        bus_masked_scores = self._apply_bleed_masks(bus_raw_scores, features)

        c_flux = self._clip01(max(bus_flux_cues.values()))
        c_band_spike = self._clip01(max(bus_masked_scores.values()))
        c_energy_delta = self._clip01(max(bus_delta_cues.values()))
        c_phase_align = self._phase_align_conf(tempo)
        c_sidecar = self._clip01(features.af_onset_conf) if features.af_onset_conf is not None else 0.0

        bus_weights = self._effective_bus_weights(features)
        _, _, _, w_phase, w_sidecar = self._effective_weights(features)
        sidecar_weight = w_sidecar if features.af_onset_conf is not None else 0.0
        bus_weight_sum = max(1e-9, sum(bus_weights.values()))
        total_weight = max(1e-9, bus_weight_sum + w_phase + sidecar_weight)

        beat_score = (
            sum(float(bus_weights[name]) * float(bus_masked_scores[name]) for name in BUS_NAMES)
            + (w_phase * c_phase_align)
            + (sidecar_weight * c_sidecar)
        ) / total_weight

        arm_threshold = float(self.config.arm_threshold)
        release_threshold = float(self.config.release_threshold)
        threshold = release_threshold if self._armed else arm_threshold
        candidate = bool(beat_score >= threshold)
        reason_codes: list[str] = ["sustain" if self._armed else "arm"] if candidate else ["below_threshold"]
        passed_buses = [name for name, d in bus_decisions.items() if d.eligible]
        if passed_buses:
            reason_codes.append(f"bus_pass:{','.join(passed_buses)}")

        refractory_ms = float(np.clip(self.config.refractory_ms, 0.0, 600.0))
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
            bus_scores={name: float(bus_masked_scores[name]) for name in BUS_NAMES},
            bus_pass={name: bool(bus_decisions[name].eligible) for name in BUS_NAMES},
            bus_reason_codes={name: list(bus_decisions[name].reason_codes) for name in BUS_NAMES},
            bus_raw_scores={name: float(bus_raw_scores[name]) for name in BUS_NAMES},
            bus_masked_scores={name: float(bus_masked_scores[name]) for name in BUS_NAMES},
            reason_codes=reason_codes,
        )
