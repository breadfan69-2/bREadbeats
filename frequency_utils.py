import numpy as np


def extract_dominant_freq(
    spectrum: np.ndarray | None,
    sample_rate: int,
    freq_low: float,
    freq_high: float,
) -> float:
    """Extract dominant frequency from a specific Hz range of the spectrum."""
    if spectrum is None or len(spectrum) == 0:
        return 0.0

    freq_per_bin = sample_rate / (2 * len(spectrum))
    if freq_per_bin <= 0:
        return 0.0

    low_bin = max(0, int(freq_low / freq_per_bin))
    high_bin = min(len(spectrum) - 1, int(freq_high / freq_per_bin))
    if low_bin >= high_bin:
        return 0.0

    band = spectrum[low_bin:high_bin + 1]
    peak_bin = low_bin + int(np.argmax(band))
    return peak_bin * freq_per_bin
