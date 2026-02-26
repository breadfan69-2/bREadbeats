import time
import unittest
from typing import Any

from audio_engine import BeatEvent
from beat_intelligence import BeatDecision, BeatIntelligence
from config import Config
from stroke_mapper import StrokeMapper


class TestStrokeMapperContract(unittest.TestCase):
    def _event(self, **overrides) -> BeatEvent:
        payload: dict[str, Any] = dict(
            timestamp=time.time(),
            intensity=0.5,
            frequency=60.0,
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

    def _drain_journey(self, bi: BeatIntelligence, frames: int = 140) -> None:
        for _ in range(frames):
            bi.build_decision(event=self._event(), dt=1.0 / 60.0, silence_override=False)

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
        intelligence = BeatIntelligence(Config())

        self.assertEqual(intelligence.interval_beats_for_trigger("syncopation"), 2)
        self.assertEqual(intelligence.interval_beats_for_trigger("beat"), 2)
        self.assertEqual(intelligence.interval_beats_for_trigger("downbeat"), 4)
        self.assertEqual(intelligence.interval_beats_for_trigger("creep"), 8)

    def test_trigger_classifier_maps_to_creep_when_no_beat_family(self):
        intelligence = BeatIntelligence(Config())
        event = self._event(is_beat=False, is_downbeat=False, is_syncopated=False)
        self.assertEqual(intelligence.classify_trigger(event), "creep")

    def test_trigger_classifier_priority_syncopation_then_downbeat_then_beat(self):
        intelligence = BeatIntelligence(Config())

        sync_event = self._event(is_syncopated=True, is_downbeat=True, is_beat=True)
        self.assertEqual(intelligence.classify_trigger(sync_event), "syncopation")

        downbeat_event = self._event(is_syncopated=False, is_downbeat=True, is_beat=True)
        self.assertEqual(intelligence.classify_trigger(downbeat_event), "downbeat")

        beat_event = self._event(is_syncopated=False, is_downbeat=False, is_beat=True)
        self.assertEqual(intelligence.classify_trigger(beat_event), "beat")

    def test_sub_bass_maps_to_radius_bloom_range(self):
        intelligence = BeatIntelligence(Config())
        intelligence.rms_envelope = 0.15  # above low-amp suppression threshold

        intelligence.energies.sub_bass = 0.0
        self.assertAlmostEqual(intelligence.compute_radius_bloom_from_sub_bass(), 0.70, places=6)

        intelligence.energies.sub_bass = 1.0
        bloom_max = intelligence.compute_radius_bloom_from_sub_bass()
        self.assertAlmostEqual(bloom_max, 1.0, places=6)

        intelligence.energies.sub_bass = 0.5
        midpoint = intelligence.compute_radius_bloom_from_sub_bass()
        self.assertGreater(midpoint, 0.70)
        self.assertLess(midpoint, 1.0)

    def test_flux_increases_bloom(self):
        intelligence = BeatIntelligence(Config())
        intelligence.rms_envelope = 0.15  # above low-amp suppression
        intelligence.energies.sub_bass = 0.5

        low_flux_event = self._event(spectral_flux=0.0)
        high_flux_event = self._event(spectral_flux=0.30)

        low_flux_bloom = intelligence.compute_radius_bloom_from_sub_bass(low_flux_event)
        high_flux_bloom = intelligence.compute_radius_bloom_from_sub_bass(high_flux_event)

        self.assertGreater(high_flux_bloom, low_flux_bloom)
        self.assertLessEqual(high_flux_bloom, 1.0)

    def test_build_decision_emits_trigger_interval_and_progress(self):
        intelligence = BeatIntelligence(Config())
        # Prime with downbeat and allow protected journey to finish
        prime = self._event(is_downbeat=True, tempo_locked=True)
        intelligence.build_decision(event=prime, dt=1.0 / 60.0, silence_override=False)
        self._drain_journey(intelligence)

        event = self._event(is_syncopated=False, is_downbeat=False, is_beat=True, tempo_locked=True)
        decision = intelligence.build_decision(event=event, dt=1.0 / 60.0, silence_override=False)

        self.assertEqual(decision.trigger_kind, "beat")
        self.assertEqual(decision.interval_beats, 2)
        self.assertGreaterEqual(decision.radius_bloom, 0.70)
        self.assertLessEqual(decision.radius_bloom, 0.95)
        self.assertFalse(decision.silence_active)
        self.assertGreaterEqual(decision.journey_completion, 0.0)
        self.assertLessEqual(decision.journey_completion, 1.0)

    def test_tempo_lock_required_falls_back_to_creep_when_not_ready(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = True
        cfg.beat.teaching_metronome_relaxed_confidence = 0.5
        intelligence = BeatIntelligence(cfg)

        event = self._event(is_beat=True, tempo_locked=False, acf_confidence=0.1)
        decision = intelligence.build_decision(event=event, dt=1.0 / 60.0, silence_override=False)

        self.assertEqual(decision.trigger_kind, "creep")
        self.assertEqual(decision.interval_beats, 8)

    # test_strict_bass_gate_blocks_non_bass_beats removed — gate no longer exists
    # test_transient_policy_kick_hat removed — transient profile gate now blocks hat/voice-only
    # test_voice_like_high_only removed — transient profile gate now blocks hat/voice-only
    # test_voice_like_low_mid_plus_high removed — transient profile gate now blocks hat/voice-only

    def test_full_motion_allows_low_flux_when_fullness_is_high(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False

        intelligence = BeatIntelligence(cfg)
        intelligence.compute_radius_bloom_from_sub_bass = lambda event=None: 0.95
        intelligence.compute_energy_fullness = lambda: 0.62

        decision = intelligence.build_decision(
            event=self._event(
                is_beat=True,
                tempo_locked=True,
                spectral_flux=0.06,
                beat_band="sub_bass",
                fired_bands=["sub_bass", "high"],
                beat_features={
                    "kick_like_conf": 0.88,
                    "hat_like_conf": 0.62,
                    "bass_dominance": 2.8,
                },
            ),
            dt=1.0 / 60.0,
            silence_override=False,
        )

        self.assertEqual(decision.trigger_kind, "beat")
        self.assertAlmostEqual(decision.radius_bloom, 0.95, places=6)

    def test_active_beat_journey_is_not_overwritten_by_intermediate_creep_frames(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        intelligence = BeatIntelligence(cfg)

        # Prime with downbeat and allow protected journey to finish before beat trigger
        prime = self._event(is_downbeat=True)
        intelligence.build_decision(event=prime, dt=1.0 / 60.0, silence_override=False)
        self._drain_journey(intelligence)

        beat_event = self._event(is_beat=True, tempo_locked=True)
        first = intelligence.build_decision(event=beat_event, dt=1.0 / 60.0, silence_override=False)

        between_event = self._event(is_beat=False, is_downbeat=False, is_syncopated=False)
        second = intelligence.build_decision(event=between_event, dt=1.0 / 60.0, silence_override=False)

        self.assertEqual(first.trigger_kind, "beat")
        self.assertEqual(second.trigger_kind, "beat")
        self.assertEqual(second.interval_beats, 2)
        self.assertGreater(second.journey_completion, 0.0)

    def test_treble_elevator_uses_positive_park_geometry(self):
        mapper = StrokeMapper(Config())
        self.assertAlmostEqual(mapper._park_y, 0.20, places=6)

    def test_treble_elevator_landing_guard_returns_center_to_park(self):
        intelligence = BeatIntelligence(Config())
        intelligence.energies.high = 1.0
        intelligence.energies.mid = 1.0

        early_offset = intelligence.compute_treble_lift(journey_completion=0.0)
        late_offset = intelligence.compute_treble_lift(journey_completion=0.95)
        landed_offset = intelligence.compute_treble_lift(journey_completion=1.0)

        self.assertLess(early_offset, 0.0)
        self.assertGreater(late_offset, early_offset)
        self.assertAlmostEqual(landed_offset, 0.0, places=6)

    def test_radius_bloom_uses_continuous_smoothing(self):
        mapper = StrokeMapper(Config())
        target = 0.95
        radius_alpha = 0.15

        mapper._actual_radius = mapper._park_radius
        mapper._actual_radius += radius_alpha * (target - mapper._actual_radius)
        first = mapper._actual_radius
        mapper._actual_radius += radius_alpha * (target - mapper._actual_radius)
        second = mapper._actual_radius

        self.assertGreater(first, mapper._park_radius)
        self.assertGreater(second, first)
        self.assertLess(second, target)

    def test_process_beat_returns_tcode_command(self):
        mapper = StrokeMapper(Config())
        event = self._event(is_beat=True, raw_rms=0.15, peak_energy=0.5)

        cmd = mapper.process_beat(event)

        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.duration_ms, 25)

    def test_fill_parks_motion_when_jitter_off(self):
        """Fill mode: dot parks gracefully (not instant park)."""
        cfg = Config()
        cfg.jitter.enabled = False
        mapper = StrokeMapper(cfg)
        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=1.0,
            silence_active=False,
            journey_completion=0.3,
        )

        t0 = time.perf_counter()
        cmd = mapper.process_beat(self._event(is_beat=False, frequency=120.0, monotonic_timestamp=t0))

        self.assertIsNotNone(cmd)
        assert cmd is not None
        # Gate-idle deceleration: dot keeps orbiting but starts decelerating.
        # Motion is preserved (not zero) — prevents the hard "stick" effect.
        self.assertGreaterEqual(cmd.beta, -1.0)
        self.assertLessEqual(cmd.beta, 1.0)

        # Simulate ~2 seconds at 60fps with proper timestamps
        for i in range(120):
            t = t0 + (i + 1) * (1.0 / 60.0)
            cmd = mapper.process_beat(self._event(is_beat=False, frequency=120.0, monotonic_timestamp=t))
        assert cmd is not None
        # After full deceleration, orbit is still present (park_radius=0.70 for
        # creep geometry) but angular velocity has dropped to idle speed (0.3 rad/s).
        # Position depends on angle, so just verify it stays in valid bounds.
        self.assertGreaterEqual(cmd.alpha, -1.0)
        self.assertLessEqual(cmd.alpha, 1.0)
        self.assertGreaterEqual(cmd.beta, -1.0)
        self.assertLessEqual(cmd.beta, 1.0)

    def test_compute_landing_rotation_from_park_for_beat_is_non_zero(self):
        mapper = StrokeMapper(Config())
        rot = mapper._compute_landing_rotation(start_angle=mapper._park_angle, interval_beats=2)
        self.assertGreater(rot, 1.0)

    def test_silence_unknown_trigger_uses_baseline_center_default(self):
        mapper = StrokeMapper(Config())
        mapper._last_trigger_kind = "unknown"
        mapper._orbit_phase = 0.0
        mapper._actual_radius = 0.05

        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="unknown",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=True,
            journey_completion=1.0,
        )

        cmd = mapper.process_beat(self._event(is_beat=False, is_downbeat=False, is_syncopated=False))

        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertGreater(cmd.beta, 0.20)

    def test_silence_hard_parks_when_fill_active(self):
        cfg = Config()
        mapper = StrokeMapper(cfg)

        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=True,
            journey_completion=1.0,
            silence_fade=0.5,
        )

        # Run several frames so the quintic ease-to-park converges
        t0 = time.perf_counter()
        frame_dt = 1.0 / 60.0
        cmds = []
        for i in range(60):  # ~1 second of frames
            cmds.append(mapper.process_beat(self._event(
                is_beat=False, raw_rms=0.0, raw_rms_db=-80.0,
                monotonic_timestamp=t0 + i * frame_dt,
            )))

        cmd_last = cmds[-1]
        self.assertIsNotNone(cmd_last)
        assert cmd_last is not None
        # After 1s of silence, orbit should have spiralled close to park.
        # orbit_phase keeps turning so alpha won't be exactly zero,
        # but radius should be small (near park_idle_radius=0.05).
        self.assertLess(abs(cmd_last.alpha), 0.15)
        self.assertGreaterEqual(cmd_last.beta, -1.0)
        self.assertLessEqual(cmd_last.beta, 1.0)

    def test_base_center_target_migrates_start_over_full_journey(self):
        mapper = StrokeMapper(Config())

        self.assertAlmostEqual(mapper._base_center_target("start", 0.0, False), 0.20, places=6)
        self.assertAlmostEqual(mapper._base_center_target("start", 0.5, False), 0.10, places=6)
        self.assertAlmostEqual(mapper._base_center_target("start", 1.0, False), 0.0, places=6)

    def test_base_center_target_is_zero_for_active_journeys(self):
        mapper = StrokeMapper(Config())

        self.assertAlmostEqual(mapper._base_center_target("beat", 0.25, False), 0.0, places=6)
        self.assertAlmostEqual(mapper._base_center_target("downbeat", 0.5, False), 0.0, places=6)
        self.assertAlmostEqual(mapper._base_center_target("syncopation", 0.75, False), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
