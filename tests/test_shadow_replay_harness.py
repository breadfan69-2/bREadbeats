import unittest

from audio_modules.event_detector import EventDetector
from audio_modules.replay_harness import ReplayFrame, run_shadow_replay


class TestShadowReplayHarness(unittest.TestCase):
    def test_replay_summary_fields(self):
        detector = EventDetector()
        frames = [
            ReplayFrame(
                time_mono=1.00,
                legacy_fire=True,
                flux_norm=0.95,
                energy_norm=0.85,
                energy_delta=0.75,
                flux_delta=0.60,
                hfc_proxy=0.20,
                sub_bass=0.80,
                low_mid=0.70,
                mid=0.25,
                high=0.20,
                bass_dominance=2.5,
                metronome_bpm=120.0,
                acf_confidence=0.65,
                tempo_locked=True,
                phase_error_ms=8.0,
                beat_phase=0.01,
                raw_rms_db=-22.0,
            ),
            ReplayFrame(
                time_mono=1.12,
                legacy_fire=False,
                flux_norm=0.20,
                energy_norm=0.15,
                energy_delta=0.05,
                flux_delta=0.03,
                hfc_proxy=0.40,
                sub_bass=0.10,
                low_mid=0.08,
                mid=0.20,
                high=0.30,
                bass_dominance=0.6,
                metronome_bpm=120.0,
                acf_confidence=0.6,
                tempo_locked=True,
                phase_error_ms=20.0,
                beat_phase=0.55,
                raw_rms_db=-70.0,
            ),
            ReplayFrame(
                time_mono=1.50,
                legacy_fire=False,
                flux_norm=0.92,
                energy_norm=0.70,
                energy_delta=0.66,
                flux_delta=0.61,
                hfc_proxy=0.12,
                sub_bass=0.78,
                low_mid=0.65,
                mid=0.20,
                high=0.12,
                bass_dominance=2.8,
                metronome_bpm=120.0,
                acf_confidence=0.7,
                tempo_locked=True,
                phase_error_ms=6.0,
                beat_phase=0.01,
                raw_rms_db=-72.0,
            ),
        ]

        summary = run_shadow_replay(frames, detector, silence_db_threshold=-58.0)

        self.assertEqual(summary.samples, 3)
        self.assertEqual(summary.legacy_fire_count, 1)
        self.assertEqual(summary.new_fire_count, 2)
        self.assertEqual(summary.agreement_count, 2)
        self.assertEqual(summary.disagreement_count, 1)
        self.assertEqual(summary.miss_count, 0)
        self.assertEqual(summary.extra_fire_count, 1)
        self.assertEqual(summary.silence_false_fire_count, 1)
        self.assertAlmostEqual(summary.agreement_pct, 66.6666, places=3)

    def test_empty_replay(self):
        summary = run_shadow_replay([], EventDetector())
        self.assertEqual(summary.samples, 0)
        self.assertAlmostEqual(summary.agreement_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
