import time
import unittest
from typing import Any

from audio_engine import BeatEvent
from config import Config
from stroke_mapper import BeatDecision, StrokeMapper


class TestStrokeMapperContract(unittest.TestCase):
    def _event(self, **overrides) -> BeatEvent:
        payload: dict[str, Any] = dict(
            timestamp=time.time(),
            intensity=0.5,
            frequency=100.0,
            is_beat=False,
            spectral_flux=0.1,
            peak_energy=0.3,
            is_downbeat=False,
            bpm=120.0,
            tempo_locked=False,
            metronome_bpm=120.0,
            is_syncopated=False,
            monotonic_timestamp=time.perf_counter(),
            raw_rms=0.08,
        )
        payload.update(overrides)
        return BeatEvent(**payload)

    def test_decision_dataclass_exists(self):
        decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.8,
            silence_active=False,
            journey_completion=0.25,
        )
        self.assertEqual(decision.trigger_kind, "beat")
        self.assertEqual(decision.interval_beats, 2)
        self.assertAlmostEqual(decision.radius_bloom, 0.8, places=6)
        self.assertFalse(decision.silence_active)
        self.assertAlmostEqual(decision.journey_completion, 0.25, places=6)

    def test_trigger_interval_mapping(self):
        mapper = StrokeMapper(Config())

        self.assertEqual(mapper._interval_beats_for_trigger("syncopation"), 1)
        self.assertEqual(mapper._interval_beats_for_trigger("beat"), 2)
        self.assertEqual(mapper._interval_beats_for_trigger("downbeat"), 4)
        self.assertEqual(mapper._interval_beats_for_trigger("creep"), 8)

    def test_trigger_classifier_maps_to_creep_when_no_beat_family(self):
        mapper = StrokeMapper(Config())
        event = self._event(is_beat=False, is_downbeat=False, is_syncopated=False)
        self.assertEqual(mapper._classify_trigger(event), "creep")

    def test_trigger_classifier_priority_syncopation_then_downbeat_then_beat(self):
        mapper = StrokeMapper(Config())

        sync_event = self._event(is_syncopated=True, is_downbeat=True, is_beat=True)
        self.assertEqual(mapper._classify_trigger(sync_event), "syncopation")

        downbeat_event = self._event(is_syncopated=False, is_downbeat=True, is_beat=True)
        self.assertEqual(mapper._classify_trigger(downbeat_event), "downbeat")

        beat_event = self._event(is_syncopated=False, is_downbeat=False, is_beat=True)
        self.assertEqual(mapper._classify_trigger(beat_event), "beat")

    def test_sub_bass_maps_to_radius_bloom_range(self):
        mapper = StrokeMapper(Config())

        mapper._sub_bass_energy = 0.0
        self.assertAlmostEqual(mapper._compute_radius_bloom_from_sub_bass(), 0.70, places=6)

        mapper._sub_bass_energy = 1.0
        self.assertAlmostEqual(mapper._compute_radius_bloom_from_sub_bass(), 0.95, places=6)

        mapper._sub_bass_energy = 0.5
        self.assertAlmostEqual(mapper._compute_radius_bloom_from_sub_bass(), 0.825, places=6)

    def test_beat_decision_bridge_uses_trigger_and_radius_mapping(self):
        mapper = StrokeMapper(Config())
        mapper._sub_bass_energy = 0.8
        event = self._event(is_syncopated=False, is_downbeat=False, is_beat=True)

        decision = mapper._build_beat_decision(event, silence_active=False, dt=1.0 / 60.0)

        self.assertEqual(decision.trigger_kind, "beat")
        self.assertEqual(decision.interval_beats, 2)
        self.assertAlmostEqual(decision.radius_bloom, 0.90, places=6)
        self.assertFalse(decision.silence_active)
        self.assertGreaterEqual(decision.journey_completion, 0.0)
        self.assertLessEqual(decision.journey_completion, 1.0)

    def test_treble_elevator_uses_negative_park_geometry(self):
        mapper = StrokeMapper(Config())
        self.assertAlmostEqual(mapper._park_y, -0.70, places=6)

    def test_treble_elevator_landing_guard_returns_center_to_park(self):
        mapper = StrokeMapper(Config())
        mapper._high_energy = 1.0
        mapper._mid_energy = 1.0

        early_offset = mapper._compute_treble_lift(journey_completion=0.0)
        late_offset = mapper._compute_treble_lift(journey_completion=0.95)
        landed_offset = mapper._compute_treble_lift(journey_completion=1.0)

        self.assertGreater(early_offset, 0.0)
        self.assertLess(late_offset, early_offset)
        self.assertAlmostEqual(landed_offset, 0.0, places=6)

    def test_s_curve_hook_is_cubic(self):
        mapper = StrokeMapper(Config())
        p = 0.5
        sp = mapper._s_curve(p)
        self.assertAlmostEqual(sp, p * p * (3.0 - 2.0 * p), places=6)

    def test_radius_bloom_uses_s_curve_pulse_shape(self):
        mapper = StrokeMapper(Config())
        decision_bloom = 0.95
        bloom_delta = decision_bloom - mapper._park_radius

        radius_start = mapper._park_radius + bloom_delta * 0.0
        radius_mid = mapper._park_radius + bloom_delta * 1.0
        radius_end = mapper._park_radius + bloom_delta * 0.0

        self.assertAlmostEqual(radius_start, mapper._park_radius, places=6)
        self.assertGreater(radius_mid, radius_start)
        self.assertAlmostEqual(radius_end, mapper._park_radius, places=6)

    def test_process_beat_returns_tcode_command(self):
        mapper = StrokeMapper(Config())
        event = self._event(is_beat=True, raw_rms=0.15, peak_energy=0.5)

        cmd = mapper.process_beat(event)

        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.duration_ms, 25)

    def test_terminal_pose_lands_at_park_with_high_treble_and_max_bloom(self):
        mapper = StrokeMapper(Config())
        mapper._high_energy = 1.0
        mapper._mid_energy = 1.0

        mapper._intelligence.build_decision = lambda event, dt: BeatDecision(
            trigger_kind="downbeat",
            interval_beats=4,
            radius_bloom=0.95,
            silence_active=False,
            journey_completion=1.0,
        )

        cmd = mapper.process_beat(self._event(is_beat=True, raw_rms=0.2, peak_energy=0.8))

        self.assertIsNotNone(cmd)
        assert cmd is not None
        epsilon = 1e-6
        self.assertLessEqual(abs(cmd.alpha - 0.0), epsilon)
        self.assertLessEqual(abs(cmd.beta - (-0.70)), epsilon)


if __name__ == "__main__":
    unittest.main()
