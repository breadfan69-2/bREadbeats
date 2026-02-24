import unittest

from audio_modules.contracts import FeatureFrame, TempoState
from audio_modules.event_detector import EventDetector, EventDetectorConfig


class TestEventDetectorFusion(unittest.TestCase):
    def test_detects_candidate_with_strong_cues(self):
        detector = EventDetector()
        features = FeatureFrame(
            flux_norm=0.92,
            energy_delta=0.70,
            sub_bass=0.85,
            low_mid=0.70,
            mid=0.30,
            high=0.25,
            af_onset_conf=0.6,
        )
        tempo = TempoState(metronome_bpm=128.0, beat_phase=0.02)

        decision = detector.detect(features, tempo, now_mono=1.00)
        self.assertGreaterEqual(decision.beat_score, 0.62)
        self.assertTrue(decision.is_beat_candidate)
        self.assertGreater(decision.c_phase_align, 0.9)

    def test_hysteresis_sustain_after_arm(self):
        detector = EventDetector()
        tempo = TempoState(metronome_bpm=120.0, beat_phase=0.02)

        first = detector.detect(
            FeatureFrame(flux_norm=0.95, energy_delta=0.75, sub_bass=0.8, low_mid=0.6),
            tempo,
            now_mono=1.00,
        )
        self.assertTrue(first.is_beat_candidate)

        second = detector.detect(
            FeatureFrame(flux_norm=0.48, energy_delta=0.30, sub_bass=0.45, low_mid=0.30),
            tempo,
            now_mono=1.40,
        )
        self.assertTrue(second.is_beat_candidate)
        self.assertGreaterEqual(second.beat_score, 0.45)
        self.assertLess(second.beat_score, 0.62)

    def test_refractory_blocks_close_retrigger(self):
        detector = EventDetector(EventDetectorConfig(refractory_ms=200.0))
        tempo = TempoState(metronome_bpm=120.0, beat_phase=0.0)
        features = FeatureFrame(flux_norm=0.95, energy_delta=0.80, sub_bass=0.9, low_mid=0.8)

        first = detector.detect(features, tempo, now_mono=1.00)
        second = detector.detect(features, tempo, now_mono=1.05)

        self.assertTrue(first.is_beat_candidate)
        self.assertFalse(second.is_beat_candidate)
        self.assertIn("refractory", second.reason_codes)

    def test_transient_classifier_can_be_enabled(self):
        detector = EventDetector(EventDetectorConfig(transient_classification_enabled=True))
        tempo = TempoState(metronome_bpm=0.0, beat_phase=0.0)
        decision = detector.detect(
            FeatureFrame(
                energy_norm=0.8,
                flux_delta=0.7,
                hfc_proxy=0.85,
                high=0.9,
                mid=0.2,
                low_mid=0.1,
                sub_bass=0.05,
            ),
            tempo,
            now_mono=1.0,
        )
        self.assertGreater(decision.hat_like_conf, 0.4)


if __name__ == "__main__":
    unittest.main()