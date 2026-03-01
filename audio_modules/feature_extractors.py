from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import maximum_filter1d

from .contracts import FeatureFrame, FrontendFrame


def rolling_percentile_norm(
    history: list | object,
    value: float,
    *,
    min_samples: int = 4,
    low_pct: float = 10.0,
    high_pct: float = 90.0,
) -> float:
    history.append(float(value))
    values = np.array(history, dtype=float)
    if values.size < int(min_samples):
        return float(np.clip(value, 0.0, 1.0))

    lo = float(np.percentile(values, low_pct))
    hi = float(np.percentile(values, high_pct))
    if hi <= lo + 1e-9:
        return 0.5
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


def compute_offbeat_score(is_syncopated: bool, syncopation_streak: int) -> float:
    if is_syncopated:
        return 1.0
    return float(np.clip(float(syncopation_streak) / 2.0, 0.0, 1.0))


def compute_teaching_confidence(acf_confidence: float, is_downbeat: bool) -> float:
    downbeat_bonus = 0.3 if bool(is_downbeat) else 0.0
    return float(np.clip((float(acf_confidence) * 0.7) + downbeat_bonus, 0.0, 1.0))


def compute_bass_dominance(
    sub_bass: float,
    low_mid: float,
    mid: float,
    high: float,
    *,
    mid_weight: float = 0.35,
    eps: float = 1e-6,
) -> float:
    bass_energy = float(sub_bass) + float(low_mid)
    treble_energy = float(high) + (float(mid_weight) * float(mid))
    return float(bass_energy / max(float(eps), treble_energy))


def spectrum_band_bins(
    sample_rate: int,
    spectrum_len: int,
    low_hz: float,
    high_hz: float,
) -> tuple[int, int]:
    if spectrum_len <= 0:
        return 0, 0

    freq_per_bin = float(sample_rate) / (2.0 * float(spectrum_len))
    if freq_per_bin <= 0.0:
        return 0, spectrum_len - 1

    low_bin = max(0, int(float(low_hz) / freq_per_bin))
    high_bin = min(spectrum_len - 1, int(float(high_hz) / freq_per_bin))
    return low_bin, high_bin


def slice_spectrum_band(
    spectrum: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    *,
    fallback_full_if_invalid: bool = True,
) -> np.ndarray:
    n_bins = int(len(spectrum))
    if n_bins <= 0:
        return spectrum

    low_bin, high_bin = spectrum_band_bins(sample_rate, n_bins, low_hz, high_hz)
    if low_bin >= high_bin:
        return spectrum if fallback_full_if_invalid else np.array([], dtype=spectrum.dtype)

    return spectrum[low_bin:high_bin + 1]


def positive_spectral_flux(
    previous_spectrum: Optional[np.ndarray],
    spectrum: np.ndarray,
    max_filter_size: int = 1,
) -> float:
    """Compute positive spectral flux between two consecutive spectra.

    When *max_filter_size* > 1 the previous spectrum is widened with a
    1-D maximum filter (SuperFlux algorithm, Böck & Widmer DAFx-2013).
    A size of 3 covers ±1 bin and is the recommended setting for
    vibrato suppression – it reduces false-positive onset detections
    by up to 60 % without missing real onsets.
    """
    if previous_spectrum is None or len(previous_spectrum) != len(spectrum):
        return 0.0

    prev = previous_spectrum
    if max_filter_size >= 3:
        prev = maximum_filter1d(prev, size=int(max_filter_size), mode='constant')

    diff = spectrum - prev
    flux = float(np.sum(np.maximum(0.0, diff)))
    if len(spectrum) > 0:
        flux /= float(len(spectrum))
    return flux


def estimate_dominant_frequency(
    spectrum: np.ndarray,
    sample_rate: int,
    low_hz: Optional[float] = None,
    high_hz: Optional[float] = None,
) -> float:
    n_bins = int(len(spectrum))
    if n_bins <= 0:
        return 0.0

    freq_per_bin = float(sample_rate) / (2.0 * float(n_bins))
    low_bin = 0
    high_bin = n_bins - 1

    if low_hz is not None and high_hz is not None:
        low = max(0.0, float(low_hz))
        high = max(low, float(high_hz))
        low_bin = max(0, int(low / freq_per_bin))
        high_bin = min(n_bins - 1, int(high / freq_per_bin))
        if high_bin <= low_bin:
            low_bin = 0
            high_bin = n_bins - 1

    band_slice = spectrum[low_bin:high_bin + 1]
    if len(band_slice) == 0:
        return 0.0

    peak_offset = int(np.argmax(band_slice))
    peak_bin = low_bin + peak_offset
    return float(peak_bin * freq_per_bin)


def compute_multiband_energies(
    spectrum: np.ndarray,
    sample_rate: int,
    gain: float,
    bands: Sequence[tuple[str, float, float]],
) -> dict[str, float]:
    energies: dict[str, float] = {}
    n_bins = int(len(spectrum))
    if n_bins <= 0:
        for name, _, _ in bands:
            energies[name] = 0.0
        return energies

    for name, low_hz, high_hz in bands:
        band_slice = slice_spectrum_band(
            spectrum,
            sample_rate,
            low_hz,
            high_hz,
            fallback_full_if_invalid=False,
        )
        if len(band_slice) > 0:
            energies[name] = float(np.sqrt(np.mean(band_slice ** 2))) * float(gain)
        else:
            energies[name] = 0.0
    return energies


def select_primary_band_by_fire_history(
    current_band: str,
    bands: Sequence[tuple[str, float, float]],
    band_fire_history: dict[str, list[int]],
    *,
    min_samples: int = 10,
) -> tuple[str, int]:
    best_band = current_band
    best_score = -1
    for name, _, _ in bands:
        hist = band_fire_history.get(name, [])
        score = int(sum(hist)) if len(hist) >= int(min_samples) else 0
        if score > best_score:
            best_score = score
            best_band = name
    return best_band, best_score


@dataclass(slots=True)
class FeatureExtractorConfig:
    enabled: bool = True


class FeatureExtractors:
    def __init__(self, config: FeatureExtractorConfig | None = None):
        self.config = config or FeatureExtractorConfig()

    def extract(self, frame: FrontendFrame) -> FeatureFrame:
        return FeatureFrame()
