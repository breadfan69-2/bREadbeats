from __future__ import annotations

import math
import unittest

import numpy as np

from pmv_audio_analysis import AnalysisConfig, analyze_full_file
from pmv_beat_engine import BeatCandidate, BeatTimeline
from pmv_position_mapper import MLConfig, MappingConfig, generate_positions


class TestPmvPositionMapper(unittest.TestCase):
    @staticmethod
    def _make_tone(sr: int, duration_s: float) -> np.ndarray:
        t = np.arange(0, int(sr * duration_s), dtype=np.float32) / float(sr)
        tone_a = 0.55 * np.sin(2.0 * math.pi * 110.0 * t)
        tone_b = 0.35 * np.sin(2.0 * math.pi * 330.0 * t)
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

    def test_positions_in_range(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        timeline = analyze_full_file(self._make_tone(cfg.sample_rate, 10.0), cfg)
        beats = self._make_beats(float(timeline.duration_ms), 500.0)

        result = generate_positions(timeline, beats, MappingConfig())
        self.assertGreater(len(result.actions), 0)
        self.assertTrue(all(0 <= action.pos <= 100 for action in result.actions))

    def test_cadence_reduces_actions(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        timeline = analyze_full_file(self._make_tone(cfg.sample_rate, 12.0), cfg)
        beats = self._make_beats(float(timeline.duration_ms), 500.0)

        dense = generate_positions(
            timeline,
            beats,
            MappingConfig(ml_config=MLConfig(enabled=False, cadence_mode="fixed_1"), points_per_second=1, min_command_delay_ms=1.0),
        )
        sparse = generate_positions(
            timeline,
            beats,
            MappingConfig(ml_config=MLConfig(enabled=False, cadence_mode="fixed_4"), points_per_second=1, min_command_delay_ms=1.0),
        )

        self.assertGreater(len(dense.beat_actions), 0)
        self.assertLess(len(sparse.beat_actions), len(dense.beat_actions) * 0.5)

    def test_bounce_overflow_adds_waypoints(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        timeline = analyze_full_file(self._make_tone(cfg.sample_rate, 8.0), cfg)
        beats = self._make_beats(float(timeline.duration_ms), 500.0)

        cropped = generate_positions(
            timeline,
            beats,
            MappingConfig(
                pitch_range=200.0,
                center_offset=140.0,
                energy_multiplier=20.0,
                overflow_mode="crop",
                points_per_second=1,
                min_command_delay_ms=1.0,
            ),
        )
        bounced = generate_positions(
            timeline,
            beats,
            MappingConfig(
                pitch_range=200.0,
                center_offset=140.0,
                energy_multiplier=20.0,
                overflow_mode="bounce",
                points_per_second=1,
                min_command_delay_ms=1.0,
            ),
        )

        self.assertGreater(len(bounced.beat_actions), len(cropped.beat_actions))
        self.assertTrue(all(0 <= action.pos <= 100 for action in bounced.beat_actions))

    def test_min_delay_enforced(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        timeline = analyze_full_file(self._make_tone(cfg.sample_rate, 10.0), cfg)
        beats = self._make_beats(float(timeline.duration_ms), 250.0)

        result = generate_positions(
            timeline,
            beats,
            MappingConfig(
                ml_config=MLConfig(enabled=False, cadence_mode="fixed_1"),
                points_per_second=40,
                min_command_delay_ms=150.0,
            ),
        )

        self.assertGreater(len(result.actions), 2)
        for a, b in zip(result.actions, result.actions[1:]):
            self.assertGreaterEqual(b.at - a.at, 150)

    def test_bounce_overflow_collapsed_range_does_not_hang(self):
        cfg = AnalysisConfig(sample_rate=48000, fft_size=2048, hop_size=960, window_size=2208)
        timeline = analyze_full_file(self._make_tone(cfg.sample_rate, 6.0), cfg)
        beats = self._make_beats(float(timeline.duration_ms), 500.0)

        result = generate_positions(
            timeline,
            beats,
            MappingConfig(
                pitch_range=200.0,
                center_offset=200.0,
                energy_multiplier=20.0,
                overflow_mode="bounce",
                points_per_second=1,
                min_command_delay_ms=1.0,
                pos_min=50,
                pos_max=50,
            ),
        )

        self.assertGreater(len(result.beat_actions), 0)
        self.assertTrue(all(action.pos == 50 for action in result.beat_actions))
        self.assertTrue(all(action.pos == 50 for action in result.actions))


if __name__ == "__main__":
    unittest.main()
