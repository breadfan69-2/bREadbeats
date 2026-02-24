import unittest

from audio_modules.contracts import FeatureFrame, TempoState
from audio_modules.event_detector import EventDetector, EventDetectorConfig


class TestTransientClassifier(unittest.TestCase):
    def test_kick_like_frame_scores_kick_higher_than_hat(self):
        detector = EventDetector(EventDetectorConfig(transient_classification_enabled=True))
        tempo = TempoState(metronome_bpm=0.0, beat_phase=0.0)
        decision = detector.detect(
            FeatureFrame(
                energy_norm=0.85,
                flux_delta=0.35,
                hfc_proxy=0.10,
                sub_bass=0.90,
                low_mid=0.65,
                mid=0.20,
                high=0.08,
            ),
            tempo,
            now_mono=1.0,
        )
        self.assertGreater(decision.kick_like_conf, decision.hat_like_conf)

    def test_hat_like_frame_scores_hat_higher_than_kick(self):
        detector = EventDetector(EventDetectorConfig(transient_classification_enabled=True))
        tempo = TempoState(metronome_bpm=0.0, beat_phase=0.0)
        decision = detector.detect(
            FeatureFrame(
                energy_norm=0.45,
                flux_delta=0.90,
                hfc_proxy=0.88,
                sub_bass=0.05,
                low_mid=0.10,
                mid=0.22,
                high=0.95,
            ),
            tempo,
            now_mono=1.2,
        )
        self.assertGreater(decision.hat_like_conf, decision.kick_like_conf)


if __name__ == "__main__":
    unittest.main()
