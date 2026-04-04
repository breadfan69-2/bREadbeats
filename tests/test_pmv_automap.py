from __future__ import annotations

import math
import unittest

import numpy as np

from pmv_audio_analysis import AnalysisConfig, analyze_full_file
from pmv_automap import AutomapConfig, _HAS_SCIPY, automap_optimize
from pmv_beat_engine import BeatCandidate, BeatTimeline
from pmv_position_mapper import MLConfig, MappingConfig, generate_positions


class TestPmvAutomap(unittest.TestCase):
    @staticmethod
    def _make_tone(sr: int, duration_s: float) -> np.ndarray:
        t = np.arange(0, int(sr * duration_s), dtype=np.float32) / float(sr)
        tone_a = 0.60 * np.sin(2.0 * math.pi * 90.0 * t)
        tone_b = 0.25 * np.sin(2.0 * math.pi * 260.0 * t)
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

    def test_optimization_improves_target_y(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        timeline = analyze_full_file(self._make_tone(cfg.sample_rate, 10.0), cfg)
        beats = self._make_beats(float(timeline.duration_ms), 500.0)

        base = MappingConfig(
            pitch_range=120.0,
            energy_multiplier=8.0,
            amplitude_centering=-40.0,
            center_offset=-120.0,
            ml_config=MLConfig(enabled=False, cadence_mode="fixed_1"),
            points_per_second=6,
            min_command_delay_ms=120.0,
        )

        initial = generate_positions(timeline, beats, base)
        self.assertGreater(len(initial.actions), 0)
        initial_avg = float(np.mean([a.pos for a in initial.actions]))

        optimized = automap_optimize(
            timeline,
            beats,
            base,
            AutomapConfig(
                enabled=True,
                target_y_position=50.0,
                optimization_mode="cmeanv2",
                optimize_ml_strength=False,
                max_iter=80,
            ),
        )

        improved = generate_positions(timeline, beats, optimized)
        improved_avg = float(np.mean([a.pos for a in improved.actions]))
        self.assertLess(abs(improved_avg - 50.0), abs(initial_avg - 50.0))

    def test_optimized_values_stay_in_bounds(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        timeline = analyze_full_file(self._make_tone(cfg.sample_rate, 8.0), cfg)
        beats = self._make_beats(float(timeline.duration_ms), 500.0)

        base = MappingConfig()
        optimized = automap_optimize(
            timeline,
            beats,
            base,
            AutomapConfig(enabled=True, optimize_ml_strength=True, max_iter=50),
        )

        self.assertTrue(-200.0 <= optimized.pitch_range <= 200.0)
        self.assertTrue(0.0 <= optimized.energy_multiplier <= 100.0)
        self.assertTrue(-200.0 <= optimized.amplitude_centering <= 200.0)
        self.assertTrue(-300.0 <= optimized.center_offset <= 300.0)
        self.assertTrue(0.0 <= optimized.ml_config.strength <= 1.0)


if __name__ == "__main__":
    unittest.main()
