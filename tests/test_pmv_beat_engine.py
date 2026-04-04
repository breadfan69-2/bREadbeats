from __future__ import annotations

import unittest

import numpy as np

from pmv_audio_analysis import AnalysisConfig, analyze_full_file
from pmv_beat_engine import (
    BeatCandidate,
    BeatDetectionConfig,
    _merge_candidates,
    detect_beats,
)


class TestPmvBeatEngine(unittest.TestCase):
    @staticmethod
    def _make_click_track(sr: int, duration_s: float, bpm: float) -> np.ndarray:
        total = int(sr * duration_s)
        samples = np.zeros(total, dtype=np.float32)
        period = max(1, int(round(sr * 60.0 / bpm)))
        pulse_len = max(1, int(0.012 * sr))
        pulse = np.hanning(pulse_len).astype(np.float32)

        for start in range(0, total, period):
            end = min(total, start + pulse_len)
            chunk = pulse[: end - start]
            samples[start:end] += 0.95 * chunk

        return np.clip(samples, -1.0, 1.0)

    def test_merge_candidates_refractory_dedup(self):
        clusters = [
            [
                BeatCandidate(time_ms=1000.0, confidence=0.55, source="fft_peak"),
                BeatCandidate(time_ms=1060.0, confidence=0.80, source="multibus"),
            ],
            [
                BeatCandidate(time_ms=2000.0, confidence=0.50, source="fft_peak"),
                BeatCandidate(time_ms=2015.0, confidence=0.60, source="librosa"),
            ],
        ]
        merged = _merge_candidates(clusters, refractory_ms=170.0)
        self.assertEqual(len(merged), 2)
        self.assertGreaterEqual(merged[0].confidence, 0.80)
        self.assertIn("multibus", merged[0].source)

    def test_detect_beats_reasonable_count_and_tempo(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        bpm = 120.0
        duration_s = 24.0
        samples = self._make_click_track(cfg.sample_rate, duration_s, bpm)
        timeline = analyze_full_file(samples, cfg)

        beat_cfg = BeatDetectionConfig(
            sensitivity=0.6,
            refractory_ms=170.0,
            use_librosa=False,
            use_multibus=True,
            use_fft_peaks=True,
            peak_seek_ratio=1.0,
            peak_beat_threshold=0.35,
        )
        result = detect_beats(timeline, beat_cfg)

        self.assertGreater(len(result.beats), 0)
        expected_beats = int(round((duration_s * bpm) / 60.0))
        ratio_error = abs(len(result.beats) - expected_beats) / max(1, expected_beats)
        self.assertLess(ratio_error, 0.40)

        self.assertGreater(result.tempo_bpm, 0.0)
        self.assertLess(abs(result.tempo_bpm - bpm), 12.0)

        for a, b in zip(result.beats, result.beats[1:]):
            self.assertGreaterEqual(b.time_ms - a.time_ms, 170.0)

    def test_detect_beats_classification_has_downbeats(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        samples = self._make_click_track(cfg.sample_rate, 16.0, 120.0)
        timeline = analyze_full_file(samples, cfg)

        result = detect_beats(
            timeline,
            BeatDetectionConfig(
                sensitivity=0.6,
                use_librosa=False,
                use_multibus=False,
                use_fft_peaks=True,
                peak_beat_threshold=0.3,
            ),
        )

        downbeats = [b for b in result.beats if b.beat_type == "downbeat"]
        self.assertGreater(len(downbeats), 0)


if __name__ == "__main__":
    unittest.main()
