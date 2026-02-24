import unittest

from audio_modules.contracts import FeatureFrame, TempoState
from audio_modules.event_detector import EventDetector, EventDetectorConfig


class TestBassDominanceWeighting(unittest.TestCase):
    def test_bass_dominance_shifts_weight_toward_delta(self):
        tempo = TempoState(metronome_bpm=120.0, beat_phase=0.05)
        features = FeatureFrame(
            flux_norm=0.60,
            energy_delta=0.95,
            sub_bass=0.80,
            low_mid=0.70,
            mid=0.20,
            high=0.15,
            bass_dominance=2.8,
        )

        plain = EventDetector(EventDetectorConfig(bass_dominance_weighting_enabled=False))
        weighted = EventDetector(EventDetectorConfig(bass_dominance_weighting_enabled=True))

        plain_decision = plain.detect(features, tempo, now_mono=1.0)
        weighted_decision = weighted.detect(features, tempo, now_mono=1.0)

        self.assertGreater(weighted_decision.beat_score, plain_decision.beat_score)


if __name__ == "__main__":
    unittest.main()
