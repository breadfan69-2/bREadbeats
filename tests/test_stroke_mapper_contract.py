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

        self.assertEqual(intelligence.interval_beats_for_trigger("syncopation"), 1)
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

    def test_strict_bass_gate_blocks_non_bass_beats(self):
        cfg = Config()
        cfg.beat.strict_bass_motion_gate_enabled = True
        cfg.beat.tempo_lock_required = False
        intelligence = BeatIntelligence(cfg)

        event = self._event(is_beat=True, beat_band="high", fired_bands=["high"])
        decision = intelligence.build_decision(event=event, dt=1.0 / 60.0, silence_override=False)

        self.assertEqual(decision.trigger_kind, "creep")
        self.assertEqual(decision.interval_beats, 8)

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
        self.assertAlmostEqual(mapper._park_y, 0.70, places=6)

    def test_treble_elevator_landing_guard_returns_center_to_park(self):
        intelligence = BeatIntelligence(Config())
        intelligence.energies.high = 1.0
        intelligence.energies.mid = 1.0

        early_offset = intelligence.compute_treble_lift(journey_completion=0.0)
        late_offset = intelligence.compute_treble_lift(journey_completion=0.95)
        landed_offset = intelligence.compute_treble_lift(journey_completion=1.0)

        self.assertGreater(early_offset, 0.0)
        self.assertLess(late_offset, early_offset)
        self.assertAlmostEqual(landed_offset, 0.0, places=6)

    def test_s_curve_hook_is_cubic(self):
        mapper = StrokeMapper(Config())
        p = 0.5
        sp = mapper._s_curve(p)
        self.assertAlmostEqual(sp, p * p * (3.0 - 2.0 * p), places=6)

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

    def test_bass_jitter_applies_only_on_creep(self):
        mapper = StrokeMapper(Config())
        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.70,
            silence_active=False,
            journey_completion=0.5,
        )

        phase_before = mapper._bass_jitter_phase
        mapper.process_beat(self._event(is_beat=True, frequency=220.0))
        self.assertAlmostEqual(mapper._bass_jitter_phase, phase_before, places=7)

        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=False,
            journey_completion=0.5,
        )
        mapper.process_beat(self._event(is_beat=False, frequency=220.0))
        self.assertGreater(mapper._bass_jitter_phase, phase_before)

    def test_bass_jitter_requires_jitter_enabled(self):
        cfg = Config()
        cfg.jitter.enabled = False
        mapper = StrokeMapper(cfg)
        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=False,
            journey_completion=0.5,
        )

        phase_before = mapper._bass_jitter_phase
        mapper.process_beat(self._event(is_beat=False, frequency=220.0))
        self.assertAlmostEqual(mapper._bass_jitter_phase, phase_before, places=7)

    def test_bass_jitter_speed_influence_changes_phase_rate(self):
        cfg_low = Config()
        cfg_low.stroke.bass_jitter_speed_influence_percent = 0.0
        mapper_low = StrokeMapper(cfg_low)
        mapper_low._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=False,
            journey_completion=0.5,
        )

        cfg_high = Config()
        cfg_high.stroke.bass_jitter_speed_influence_percent = 200.0
        mapper_high = StrokeMapper(cfg_high)
        mapper_high._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=False,
            journey_completion=0.5,
        )

        event = self._event(is_beat=False, frequency=220.0)
        mapper_low.process_beat(event)
        mapper_high.process_beat(event)

        self.assertGreater(mapper_high._bass_jitter_phase, mapper_low._bass_jitter_phase)

    def test_bass_jitter_size_influence_changes_offset_magnitude(self):
        cfg_low = Config()
        cfg_low.stroke.bass_jitter_size_influence_percent = 0.0
        mapper_low = StrokeMapper(cfg_low)

        cfg_high = Config()
        cfg_high.stroke.bass_jitter_size_influence_percent = 200.0
        mapper_high = StrokeMapper(cfg_high)

        event = self._event(is_beat=False, frequency=220.0)
        alpha_low, beta_low = mapper_low._compute_bass_jitter_offsets(event=event, dt=1.0 / 60.0)
        alpha_high, beta_high = mapper_high._compute_bass_jitter_offsets(event=event, dt=1.0 / 60.0)

        mag_low = abs(alpha_low) + abs(beta_low)
        mag_high = abs(alpha_high) + abs(beta_high)
        self.assertGreater(mag_high, mag_low)

    def test_creep_disabled_parks_motion_when_jitter_off(self):
        cfg = Config()
        cfg.creep.enabled = False
        cfg.jitter.enabled = False
        mapper = StrokeMapper(cfg)
        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=1.0,
            silence_active=False,
            journey_completion=0.3,
        )

        cmd = mapper.process_beat(self._event(is_beat=False, frequency=120.0))

        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertAlmostEqual(cmd.alpha, 0.0, places=6)
        # Creep parks at center_y (0.4) + park_radius (0.30) = 0.70
        self.assertAlmostEqual(cmd.beta, 0.70, places=6)

    def test_compute_landing_rotation_from_park_for_beat_is_non_zero(self):
        mapper = StrokeMapper(Config())
        rot = mapper._compute_landing_rotation(start_angle=mapper._park_angle, interval_beats=2)
        self.assertGreater(rot, 1.0)

    def test_terminal_pose_lands_at_park_angle_with_smoothed_radius_downbeat(self):
        mapper = StrokeMapper(Config())
        mapper._intelligence.energies.high = 1.0
        mapper._intelligence.energies.mid = 1.0

        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
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
        self.assertGreater(cmd.beta, 0.70)
        self.assertLessEqual(cmd.beta, 1.0)

    def test_terminal_pose_lands_at_park_angle_with_smoothed_radius_syncopation(self):
        mapper = StrokeMapper(Config())
        mapper._intelligence.energies.high = 1.0
        mapper._intelligence.energies.mid = 1.0

        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="syncopation",
            interval_beats=1,
            radius_bloom=0.95,
            silence_active=False,
            journey_completion=1.0,
        )

        cmd = mapper.process_beat(self._event(is_syncopated=True, raw_rms=0.2, peak_energy=0.8))

        self.assertIsNotNone(cmd)
        assert cmd is not None
        epsilon = 1e-6
        self.assertLessEqual(abs(cmd.alpha - 0.0), epsilon)
        self.assertGreater(cmd.beta, 0.70)
        self.assertLessEqual(cmd.beta, 1.0)

    def test_terminal_pose_lands_at_park_angle_from_non_park_start_angle(self):
        mapper = StrokeMapper(Config())
        mapper._intelligence.energies.high = 1.0
        mapper._intelligence.energies.mid = 1.0
        mapper._orbit_phase = 1.0471975512
        mapper._journey_start_angle = mapper._orbit_phase
        mapper._last_journey_completion = 1.0

        # Journey start then many settle frames to let smooth lerp converge
        n_settle = 120
        completions = iter([0.0] + [1.0] * n_settle)

        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="downbeat",
            interval_beats=4,
            radius_bloom=0.95,
            silence_active=False,
            journey_completion=next(completions),
        )

        # Use properly-spaced monotonic timestamps so the settle’s
        # dt accumulation converges (simulating real 60 fps frames).
        t0 = time.perf_counter()
        frame_dt = 1.0 / 60.0

        # Start journey
        mapper.process_beat(self._event(is_downbeat=True, raw_rms=0.2, peak_energy=0.8,
                                        monotonic_timestamp=t0))

        # Settle frames (smooth exponential lerp toward park)
        for i in range(1, n_settle + 1):
            cmd = mapper.process_beat(self._event(
                is_downbeat=True, raw_rms=0.2, peak_energy=0.8,
                monotonic_timestamp=t0 + i * frame_dt,
            ))

        self.assertIsNotNone(cmd)
        assert cmd is not None
        epsilon = 0.01  # smooth lerp converges gradually
        self.assertLessEqual(abs(cmd.alpha - 0.0), epsilon)
        self.assertGreater(cmd.beta, 0.70)
        self.assertLessEqual(cmd.beta, 1.0)


if __name__ == "__main__":
    unittest.main()
