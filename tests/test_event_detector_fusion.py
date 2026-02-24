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
        self.assertIn("sub_bass", decision.bus_scores)
        self.assertIn("high", decision.bus_scores)

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
            FeatureFrame(flux_norm=0.58, energy_delta=0.46, sub_bass=0.56, low_mid=0.42, mid=0.18, high=0.14),
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

    def test_high_hat_spike_does_not_promote_bass_bus(self):
        detector = EventDetector()
        tempo = TempoState(metronome_bpm=120.0, beat_phase=0.20)

        decision = detector.detect(
            FeatureFrame(
                flux_norm=0.90,
                energy_delta=0.72,
                hfc_proxy=0.92,
                high=0.95,
                mid=0.52,
                sub_bass=0.08,
                low_mid=0.12,
            ),
            tempo,
            now_mono=1.0,
        )

        self.assertLess(decision.bus_scores["sub_bass"], 0.45)
        self.assertLess(decision.bus_scores["low_mid"], 0.50)

    def test_kick_does_not_force_high_bus(self):
        detector = EventDetector()
        tempo = TempoState(metronome_bpm=120.0, beat_phase=0.01)

        decision = detector.detect(
            FeatureFrame(
                flux_norm=0.82,
                energy_delta=0.74,
                hfc_proxy=0.10,
                sub_bass=0.95,
                low_mid=0.80,
                mid=0.20,
                high=0.10,
            ),
            tempo,
            now_mono=1.0,
        )

        self.assertGreater(decision.bus_scores["sub_bass"], decision.bus_scores["high"])
        self.assertLess(decision.bus_scores["high"], 0.45)

    def test_mixed_transient_preserves_both_buses(self):
        detector = EventDetector()
        tempo = TempoState(metronome_bpm=124.0, beat_phase=0.01)

        decision = detector.detect(
            FeatureFrame(
                flux_norm=0.88,
                energy_delta=0.70,
                hfc_proxy=0.52,
                sub_bass=0.86,
                low_mid=0.72,
                mid=0.58,
                high=0.78,
            ),
            tempo,
            now_mono=1.0,
        )

        self.assertGreater(decision.bus_scores["sub_bass"], 0.50)
        self.assertGreater(decision.bus_scores["high"], 0.45)

    def test_refractory_is_per_bus_other_bus_can_pass(self):
        detector = EventDetector(
            EventDetectorConfig(
                refractory_ms=0.0,
                bus_refractory_ms=280.0,
                arm_threshold=0.35,
                release_threshold=0.25,
            )
        )
        tempo = TempoState(metronome_bpm=0.0, beat_phase=0.0)

        first = detector.detect(
            FeatureFrame(flux_norm=0.90, energy_delta=0.80, sub_bass=0.92, low_mid=0.78, high=0.08),
            tempo,
            now_mono=1.00,
        )
        second = detector.detect(
            FeatureFrame(flux_norm=0.90, energy_delta=0.75, sub_bass=0.90, low_mid=0.76, high=0.10),
            tempo,
            now_mono=1.10,
        )
        third = detector.detect(
            FeatureFrame(flux_norm=0.92, energy_delta=0.82, sub_bass=0.09, low_mid=0.10, mid=0.40, high=0.95, hfc_proxy=0.9),
            tempo,
            now_mono=1.10,
        )

        self.assertTrue(first.bus_pass["sub_bass"])
        self.assertIn("refractory", second.bus_reason_codes["sub_bass"])
        self.assertTrue(third.bus_pass["high"])

    def test_global_score_derives_from_bus_fusion_when_optional_cues_disabled(self):
        cfg = EventDetectorConfig(
            w_phase=0.0,
            w_sidecar=0.0,
            w_bus_sub=0.40,
            w_bus_low=0.30,
            w_bus_mid=0.20,
            w_bus_high=0.10,
        )
        detector = EventDetector(cfg)
        tempo = TempoState(metronome_bpm=0.0, beat_phase=0.0)
        decision = detector.detect(
            FeatureFrame(
                flux_norm=0.86,
                energy_delta=0.72,
                sub_bass=0.82,
                low_mid=0.65,
                mid=0.35,
                high=0.20,
            ),
            tempo,
            now_mono=1.0,
        )

        expected = (
            (0.40 * decision.bus_scores["sub_bass"]) +
            (0.30 * decision.bus_scores["low_mid"]) +
            (0.20 * decision.bus_scores["mid"]) +
            (0.10 * decision.bus_scores["high"])
        )
        self.assertAlmostEqual(decision.beat_score, expected, places=6)

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