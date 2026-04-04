from __future__ import annotations

import math
import unittest

import numpy as np

from pmv_audio_analysis import AnalysisConfig, analyze_full_file
from pmv_beat_engine import BeatCandidate, BeatTimeline
from pmv_position_mapper import (
    FEATURE_COLUMNS,
    MLConfig,
    _build_feature_vector,
    _derive_cadence,
    _load_rule_fit_model,
    compute_beat_intelligence,
)


class TestPmvMlIntelligence(unittest.TestCase):
    @staticmethod
    def _make_tone(sr: int, duration_s: float) -> np.ndarray:
        t = np.arange(0, int(sr * duration_s), dtype=np.float32) / float(sr)
        tone_a = 0.45 * np.sin(2.0 * math.pi * 120.0 * t)
        tone_b = 0.30 * np.sin(2.0 * math.pi * 440.0 * t)
        return (tone_a + tone_b).astype(np.float32)

    @staticmethod
    def _make_beats(duration_ms: float, step_ms: float = 500.0) -> BeatTimeline:
        beats: list[BeatCandidate] = []
        t = 0.0
        idx = 0
        while t <= duration_ms:
            beat_type = "downbeat" if idx % 4 == 0 else "beat"
            beats.append(BeatCandidate(t, 0.9, "test", {}, beat_type))
            t += step_ms
            idx += 1
        return BeatTimeline(
            beats=beats,
            tempo_bpm=120.0,
            tempo_confidence=1.0,
            beat_period_ms=500.0,
            time_signature=4,
        )

    def test_feature_vector_14_columns(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        timeline = analyze_full_file(self._make_tone(cfg.sample_rate, 8.0), cfg)
        vec = _build_feature_vector(timeline, beat_time_ms=1500.0)
        self.assertEqual(len(vec), 14)
        self.assertEqual(set(vec.keys()), set(FEATURE_COLUMNS))

    def test_speed_mult_range(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        timeline = analyze_full_file(self._make_tone(cfg.sample_rate, 10.0), cfg)
        beats = self._make_beats(float(timeline.duration_ms), 500.0)

        results = compute_beat_intelligence(timeline, beats, MLConfig(enabled=True))
        self.assertEqual(len(results), len(beats.beats))
        self.assertTrue(all(0.0 <= r.speed_mult <= 1.0 for r in results))
        self.assertTrue(all(r.cadence_hint in {1, 2, 4} for r in results))

    def test_cadence_derivation_from_model_thresholds(self):
        model = _load_rule_fit_model(MLConfig())
        self.assertEqual(_derive_cadence(0.20, model), 4)
        self.assertEqual(_derive_cadence(0.40, model), 2)
        self.assertEqual(_derive_cadence(0.80, model), 1)

    def test_ml_disabled_returns_defaults(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        timeline = analyze_full_file(self._make_tone(cfg.sample_rate, 6.0), cfg)
        beats = self._make_beats(float(timeline.duration_ms), 500.0)

        results = compute_beat_intelligence(timeline, beats, MLConfig(enabled=False))
        self.assertTrue(all(abs(r.speed_mult - 0.5) < 1e-6 for r in results))
        self.assertTrue(all(r.cadence_hint == 1 for r in results))


if __name__ == "__main__":
    unittest.main()
