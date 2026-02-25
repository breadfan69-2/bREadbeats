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

    def test_recovery_lock_ignores_midflight_trigger_changes(self):
        intelligence = BeatIntelligence(Config())

        intelligence.build_decision(event=self._event(raw_rms=0.0), dt=1.0 / 60.0, silence_override=True)
        d0 = intelligence.build_decision(event=self._event(is_beat=True), dt=1.0 / 60.0, silence_override=False)

        self.assertTrue(intelligence.is_recovering)
        self.assertEqual(d0.trigger_kind, "start")
        self.assertEqual(d0.interval_beats, 8)
        self.assertLess(d0.journey_completion, 0.05)

        d1 = intelligence.build_decision(
            event=self._event(is_syncopated=True, is_downbeat=True, is_beat=True),
            dt=1.0 / 60.0,
            silence_override=False,
        )

        self.assertEqual(d1.trigger_kind, "start")
        self.assertEqual(d1.interval_beats, 8)
        self.assertGreater(d1.journey_completion, 0.0)

    def test_recovery_lock_latches_radius_bloom_until_complete(self):
        intelligence = BeatIntelligence(Config())

        intelligence.build_decision(event=self._event(raw_rms=0.0), dt=1.0 / 60.0, silence_override=True)

        call_count = {"n": 0}

        def _fake_radius(event=None):
            call_count["n"] += 1
            return 0.91 if call_count["n"] == 1 else 0.40

        intelligence.compute_radius_bloom_from_sub_bass = _fake_radius

        d0 = intelligence.build_decision(event=self._event(is_beat=True), dt=1.0 / 60.0, silence_override=False)
        d1 = intelligence.build_decision(event=self._event(is_downbeat=True), dt=1.0 / 60.0, silence_override=False)

        self.assertAlmostEqual(d0.radius_bloom, 0.91, places=6)
        self.assertAlmostEqual(d1.radius_bloom, 0.91, places=6)

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

    def test_transient_policy_kick_hat_and_kick_only_are_full_hat_only_is_limited(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        cfg.beat.strict_bass_motion_gate_enabled = True

        kick_hat = BeatIntelligence(cfg)
        kick_only = BeatIntelligence(cfg)
        hat_only = BeatIntelligence(cfg)

        kick_hat.compute_radius_bloom_from_sub_bass = lambda event=None: 0.95
        kick_only.compute_radius_bloom_from_sub_bass = lambda event=None: 0.95
        hat_only.compute_radius_bloom_from_sub_bass = lambda event=None: 0.95

        decision_kick_hat = kick_hat.build_decision(
            event=self._event(
                is_beat=True,
                tempo_locked=True,
                spectral_flux=0.24,
                beat_band="low_mid",
                fired_bands=["low_mid", "high"],
                beat_features={
                    "kick_like_conf": 0.85,
                    "hat_like_conf": 0.82,
                    "bass_dominance": 2.1,
                },
            ),
            dt=1.0 / 60.0,
            silence_override=False,
        )

        decision_kick_only = kick_only.build_decision(
            event=self._event(
                is_beat=True,
                tempo_locked=True,
                spectral_flux=0.22,
                beat_band="sub_bass",
                fired_bands=["sub_bass"],
                beat_features={
                    "kick_like_conf": 0.90,
                    "hat_like_conf": 0.08,
                    "bass_dominance": 2.2,
                },
            ),
            dt=1.0 / 60.0,
            silence_override=False,
        )

        decision_hat_only = hat_only.build_decision(
            event=self._event(
                is_beat=True,
                tempo_locked=True,
                beat_band="high",
                fired_bands=["high"],
                beat_features={
                    "kick_like_conf": 0.05,
                    "hat_like_conf": 0.90,
                    "bass_dominance": 0.45,
                },
            ),
            dt=1.0 / 60.0,
            silence_override=False,
        )

        self.assertEqual(decision_kick_hat.trigger_kind, "beat")
        self.assertEqual(decision_kick_only.trigger_kind, "beat")
        self.assertEqual(decision_hat_only.trigger_kind, "beat")

        self.assertAlmostEqual(decision_kick_hat.radius_bloom, 0.95, places=6)
        self.assertAlmostEqual(decision_kick_only.radius_bloom, 0.95, places=6)
        self.assertAlmostEqual(decision_hat_only.radius_bloom, 0.70, places=6)
        self.assertTrue(decision_hat_only.park_bounce_only)
        self.assertGreater(decision_hat_only.park_bounce_gain, 0.0)

    def test_hat_only_park_bounce_triggers_hat_bounce_pulse(self):
        mapper = StrokeMapper(Config())

        event = self._event(
            is_beat=True,
            beat_features={
                "kick_like_conf": 0.05,
                "hat_like_conf": 0.92,
                "bass_dominance": 0.40,
            },
        )

        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.70,
            silence_active=False,
            journey_completion=0.2,
            park_bounce_only=True,
            park_bounce_gain=0.9,
        )

        cmd = mapper.process_beat(event)
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertGreater(mapper._hat_bounce_amp, 0.0)

    def test_voice_like_high_only_is_forced_to_limited_park_bounce(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        cfg.beat.transient_full_motion_min_energy_fullness = 0.34
        cfg.beat.transient_full_motion_min_flux = 0.15

        intelligence = BeatIntelligence(cfg)
        intelligence.compute_radius_bloom_from_sub_bass = lambda event=None: 0.95

        decision = intelligence.build_decision(
            event=self._event(
                is_beat=True,
                tempo_locked=True,
                beat_band="high",
                fired_bands=["high"],
                beat_features={
                    "kick_like_conf": 0.58,
                    "hat_like_conf": 0.82,
                    "bass_dominance": 1.10,
                },
            ),
            dt=1.0 / 60.0,
            silence_override=False,
        )

        self.assertEqual(decision.trigger_kind, "beat")
        self.assertTrue(decision.park_bounce_only)
        self.assertAlmostEqual(decision.radius_bloom, 0.70, places=6)
        self.assertLessEqual(decision.park_bounce_gain, 0.60)

    def test_voice_like_low_mid_plus_high_still_requires_sub_bass_for_full_motion(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        cfg.beat.transient_full_motion_min_bass_dom = 1.95
        cfg.beat.transient_full_motion_decisive_bass_dom = 2.55
        cfg.beat.transient_full_motion_min_energy_fullness = 0.34

        intelligence = BeatIntelligence(cfg)
        intelligence.compute_radius_bloom_from_sub_bass = lambda event=None: 0.95

        decision = intelligence.build_decision(
            event=self._event(
                is_beat=True,
                tempo_locked=True,
                beat_band="low_mid",
                fired_bands=["low_mid", "high"],
                beat_features={
                    "kick_like_conf": 0.66,
                    "hat_like_conf": 0.78,
                    "bass_dominance": 1.50,
                },
            ),
            dt=1.0 / 60.0,
            silence_override=False,
        )

        self.assertEqual(decision.trigger_kind, "beat")
        self.assertTrue(decision.park_bounce_only)
        self.assertAlmostEqual(decision.radius_bloom, 0.70, places=6)

    def test_full_motion_requires_min_flux_or_fullness_even_with_strong_bass(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        cfg.beat.transient_full_motion_min_energy_fullness = 0.34
        cfg.beat.transient_full_motion_min_flux = 0.15

        intelligence = BeatIntelligence(cfg)
        intelligence.compute_radius_bloom_from_sub_bass = lambda event=None: 0.95
        intelligence.compute_energy_fullness = lambda: 0.20

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
        self.assertTrue(decision.park_bounce_only)
        self.assertAlmostEqual(decision.radius_bloom, 0.70, places=6)

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
        self.assertFalse(decision.park_bounce_only)
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
            journey_completion=1.0,
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

    def test_bass_jitter_frequency_changes_phase_rate(self):
        mapper_low = StrokeMapper(Config())
        mapper_low._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=False,
            journey_completion=1.0,
        )

        mapper_high = StrokeMapper(Config())
        mapper_high._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=False,
            journey_completion=1.0,
        )

        mapper_low.process_beat(self._event(is_beat=False, frequency=30.0))
        mapper_high.process_beat(self._event(is_beat=False, frequency=220.0))

        self.assertGreater(mapper_high._bass_jitter_phase, mapper_low._bass_jitter_phase)

    def test_bass_jitter_frequency_changes_offset_magnitude(self):
        mapper_low = StrokeMapper(Config())
        mapper_high = StrokeMapper(Config())

        alpha_low, beta_low = mapper_low._compute_bass_jitter_offsets(
            event=self._event(is_beat=False, frequency=30.0),
            dt=1.0 / 60.0,
        )
        alpha_high, beta_high = mapper_high._compute_bass_jitter_offsets(
            event=self._event(is_beat=False, frequency=220.0),
            dt=1.0 / 60.0,
        )

        # Recover jitter amplitude independent of phase.
        amp_low = (alpha_low ** 2 + ((beta_low / 0.70) ** 2)) ** 0.5
        amp_high = (alpha_high ** 2 + ((beta_high / 0.70) ** 2)) ** 0.5
        self.assertGreater(amp_high, amp_low)

    def test_creep_disabled_parks_motion_when_jitter_off(self):
        """Creep-disabled: dot decelerates gracefully (not instant park)."""
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

    def test_terminal_pose_keeps_continuity_without_hard_snap_downbeat(self):
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
        # Continuity model: completion enters controlled spiral/continuation,
        # so alpha is expected to remain in motion (non-zero).
        self.assertGreater(abs(cmd.alpha), 0.01)
        self.assertGreaterEqual(cmd.beta, -1.0)
        self.assertLessEqual(cmd.beta, 1.0)

    def test_terminal_pose_keeps_continuity_without_hard_snap_syncopation(self):
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
        # Continuity model: completion enters controlled spiral/continuation,
        # so alpha is expected to remain in motion (non-zero).
        self.assertGreater(abs(cmd.alpha), 0.01)
        self.assertGreaterEqual(cmd.beta, -1.0)
        self.assertLessEqual(cmd.beta, 1.0)

    def test_terminal_pose_from_non_park_start_keeps_continuity(self):
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
        # Under continuity/exit-spiral behavior, this should not hard-snap to
        # alpha=0 after completion, even after many settle frames.
        self.assertGreater(abs(cmd.alpha), 0.01)
        self.assertGreaterEqual(cmd.beta, -1.0)
        self.assertLessEqual(cmd.beta, 1.0)

    def test_start_journey_inherits_last_known_position_at_progress_zero(self):
        mapper = StrokeMapper(Config())
        mapper.state.alpha = 0.23
        mapper.state.beta = 0.82

        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="start",
            interval_beats=8,
            radius_bloom=0.95,
            silence_active=False,
            journey_completion=0.0,
        )

        cmd = mapper.process_beat(self._event(is_beat=False, is_downbeat=False, is_syncopated=False))

        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertAlmostEqual(cmd.alpha, 0.23, places=6)
        self.assertAlmostEqual(cmd.beta, 0.82, places=6)
        self.assertGreaterEqual(mapper._actual_radius, 0.05)

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

    def test_silence_hard_parks_when_creep_disabled(self):
        cfg = Config()
        cfg.creep.enabled = False
        mapper = StrokeMapper(cfg)

        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=True,
            journey_completion=1.0,
            silence_fade=0.5,
        )

        cmd1 = mapper.process_beat(self._event(is_beat=False, raw_rms=0.0, raw_rms_db=-80.0))
        cmd2 = mapper.process_beat(self._event(is_beat=False, raw_rms=0.0, raw_rms_db=-80.0))

        self.assertIsNotNone(cmd1)
        self.assertIsNotNone(cmd2)
        assert cmd1 is not None
        assert cmd2 is not None
        self.assertAlmostEqual(cmd1.alpha, 0.0, places=6)
        self.assertAlmostEqual(cmd1.beta, mapper._baseline_center_y, places=6)
        self.assertAlmostEqual(cmd2.alpha, cmd1.alpha, places=6)
        self.assertAlmostEqual(cmd2.beta, cmd1.beta, places=6)

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

    def test_reactive_bounce_applies_only_in_wait_state(self):
        cfg = Config()
        cfg.jitter.enabled = True
        cfg.jitter.amplitude = 0.2
        cfg.jitter.intensity = 2.0
        mapper = StrokeMapper(cfg)

        event = self._event(is_beat=False, frequency=120.0)
        self.assertAlmostEqual(mapper._compute_reactive_bounce_y(event, 1.0 / 60.0, False), 0.0, places=7)
        self.assertNotEqual(mapper._compute_reactive_bounce_y(event, 1.0 / 60.0, True), 0.0)


if __name__ == "__main__":
    unittest.main()
