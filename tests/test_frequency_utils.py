import unittest

import numpy as np

from frequency_utils import extract_dominant_freq


class TestFrequencyUtils(unittest.TestCase):
    def test_extract_dominant_freq_basic_peak(self):
        # sample_rate=1000, N=10 => freq_per_bin=50Hz
        spectrum = np.zeros(10)
        spectrum[4] = 10.0

        freq = extract_dominant_freq(spectrum, sample_rate=1000, freq_low=100.0, freq_high=300.0)
        self.assertAlmostEqual(freq, 200.0, places=6)

    def test_extract_dominant_freq_empty_or_none(self):
        self.assertEqual(extract_dominant_freq(None, 1000, 10.0, 200.0), 0.0)
        self.assertEqual(extract_dominant_freq(np.array([]), 1000, 10.0, 200.0), 0.0)

    def test_extract_dominant_freq_invalid_band(self):
        spectrum = np.ones(10)
        freq = extract_dominant_freq(spectrum, sample_rate=1000, freq_low=400.0, freq_high=200.0)
        self.assertEqual(freq, 0.0)


if __name__ == "__main__":
    unittest.main()
