from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(slots=True)
class FrontendFrame:
    mono_time: float
    wall_time: float
    spectrum: np.ndarray
    band_energy: float
    spectral_flux: float
    raw_rms: float
    raw_rms_db: float


@dataclass(slots=True)
class FeatureFrame:
    flux_norm: float = 0.0
    energy_norm: float = 0.0
    energy_delta: float = 0.0
    flux_delta: float = 0.0
    hfc_proxy: float = 0.0
    sub_bass: float = 0.0
    low_mid: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    bass_dominance: float = 1.0
    af_entropy: Optional[float] = None
    af_flatness: Optional[float] = None
    af_hfc: Optional[float] = None
    af_novelty: Optional[float] = None
    af_rms: Optional[float] = None
    af_onset_conf: Optional[float] = None


@dataclass(slots=True)
class TempoState:
    metronome_bpm: float = 0.0
    acf_confidence: float = 0.0
    tempo_locked: bool = False
    phase_error_ms: float = 0.0
    is_downbeat: bool = False
    beat_phase: float = 0.0


@dataclass(slots=True)
class TriggerDecision:
    beat_score: float = 0.0
    raw_onset_conf: float = 0.0
    is_beat_candidate: bool = False
    c_flux: float = 0.0
    c_band_spike: float = 0.0
    c_energy_delta: float = 0.0
    c_phase_align: float = 0.0
    c_sidecar: float = 0.0
    kick_like_conf: float = 0.0
    hat_like_conf: float = 0.0
    mixed_conf: float = 0.0
    reason_codes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EngineDecision:
    is_beat: bool = False
    is_downbeat: bool = False
    beat_band: str = "sub_bass"
    fired_bands: list[str] = field(default_factory=list)
    beat_score: float = 0.0
    transient_class: str = "mixed"
