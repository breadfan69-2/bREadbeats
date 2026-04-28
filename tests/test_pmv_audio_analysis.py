from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from pmv_audio_analysis import (
    AnalysisConfig,
    _HAS_LIBROSA,
    _HAS_SOUNDFILE,
    analyze_full_file,
    apply_frequency_filters,
    load_audio,
    p95_normalize,
)


class TestPmvAudioAnalysis(unittest.TestCase):
    def test_p95_normalization(self):
        values = np.random.default_rng(1337).exponential(1.0, 2000).astype(np.float32)
        normed, p95 = p95_normalize(values)
        self.assertGreater(p95, 0.0)
        self.assertAlmostEqual(float(np.percentile(normed, 95)), 1.0, places=1)

    def test_apply_frequency_filters(self):
        config = AnalysisConfig(
            lowpass_enabled=True,
            lowpass_hz=1000.0,
            freq_min_hz=0.0,
            freq_max_hz=20_000.0,
        )
        spectrum = np.ones(8, dtype=np.float32)
        freqs = np.array([0, 250, 500, 750, 1000, 1250, 1500, 1750], dtype=np.float32)
        filtered = apply_frequency_filters(spectrum, freqs, config)
        self.assertTrue(np.all(filtered[freqs > 1000.0] == 0.0))
        self.assertTrue(np.all(filtered[freqs <= 1000.0] == 1.0))

    def test_analyze_full_file_shape_synthetic(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        t = np.arange(0, cfg.sample_rate * 2, dtype=np.float32) / float(cfg.sample_rate)
        samples = (0.5 * np.sin(2.0 * math.pi * 440.0 * t)).astype(np.float32)

        timeline = analyze_full_file(samples, cfg)
        expected_frames = len(samples) // cfg.hop_size
        self.assertGreater(len(timeline.feature_frames), 0)
        self.assertLessEqual(abs(len(timeline.feature_frames) - expected_frames), 2)
        self.assertEqual(len(timeline.pitch_per_frame), len(timeline.feature_frames))
        self.assertEqual(len(timeline.rms_per_frame), len(timeline.feature_frames))

    def test_load_audio_wav(self):
        if not (_HAS_SOUNDFILE or _HAS_LIBROSA):
            self.skipTest("load_audio requires soundfile or librosa")

        cfg = AnalysisConfig(sample_rate=16000)
        tone_sr = 16000
        duration_s = 1.0
        t = np.arange(0, int(tone_sr * duration_s), dtype=np.float32) / float(tone_sr)
        tone = (0.5 * np.sin(2.0 * math.pi * 440.0 * t)).astype(np.float32)
        pcm = (tone * 32767.0).astype(np.int16)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(tone_sr)
                wf.writeframes(pcm.tobytes())

            samples = load_audio(path, cfg)
            self.assertGreater(len(samples), 0)
            self.assertEqual(samples.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
