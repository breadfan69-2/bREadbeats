"""
Phase 2 checkpoint tests — ReadinessState (#17), SilenceDecayState (#19),
post-silence volume ramp (#14).
"""

import time
import unittest
from typing import Any

from audio_engine import BeatEvent
from beat_intelligence import BeatDecision, BeatIntelligence
from config import Config
from stroke_mapper import StrokeMapper


class Phase2Mixin:
    """Shared event factory for Phase 2 tests."""

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
            tempo_locked=True,
            metronome_bpm=120.0,
            is_syncopated=False,
            monotonic_timestamp=time.perf_counter(),
            raw_rms=0.08,
            beat_band="sub_bass",
            fired_bands=["sub_bass"],
        )
        payload.update(overrides)
        return BeatEvent(**payload)


# ── §17 ReadinessState machine ─────────────────────────────────────────


class TestReadinessState(Phase2Mixin, unittest.TestCase):
    def test_cold_start_not_ready(self):
        bi = BeatIntelligence(Config())
        self.assertFalse(bi._stroke_ready)

    def test_one_green_frame_makes_ready(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        event = self._event(tempo_locked=True, monotonic_timestamp=now)
        bi._update_stroke_readiness(event, now)
        self.assertTrue(bi._stroke_ready)
        self.assertEqual(bi._stroke_ready_reason, "green")

    def test_brief_dip_holds_readiness_via_grace(self):
        cfg = Config()
        cfg.beat.teaching_stroke_ready_grace_ms = 500.0
        bi = BeatIntelligence(cfg)
        now = time.perf_counter()

        # Establish readiness
        bi._update_stroke_readiness(self._event(tempo_locked=True, monotonic_timestamp=now), now)
        self.assertTrue(bi._stroke_ready)

        # Dip: confidence too low, but within grace window
        dip_now = now + 0.1  # 100ms later, well within 500ms grace
        bi._update_stroke_readiness(
            self._event(tempo_locked=False, acf_confidence=0.0, monotonic_timestamp=dip_now),
            dip_now,
        )
        self.assertTrue(bi._stroke_ready)
        self.assertIn(bi._stroke_ready_reason, ("grace", "finishing"))

    def test_sustained_loss_kills_readiness(self):
        cfg = Config()
        cfg.beat.teaching_stroke_ready_grace_ms = 50.0  # very short grace
        cfg.beat.teaching_stroke_finish_beats = 0
        bi = BeatIntelligence(cfg)
        now = time.perf_counter()

        # Establish
        bi._update_stroke_readiness(self._event(tempo_locked=True, monotonic_timestamp=now), now)
        self.assertTrue(bi._stroke_ready)

        # Sustained loss: well past grace, 0 finish beats
        for i in range(10):
            late = now + 1.0 + (i * 0.02)
            bi._update_stroke_readiness(
                self._event(tempo_locked=False, acf_confidence=0.0, monotonic_timestamp=late),
                late,
            )
        self.assertFalse(bi._stroke_ready)
        self.assertEqual(bi._stroke_ready_reason, "blocked")

    def test_readiness_integrates_into_build_decision(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = True
        cfg.beat.teaching_metronome_relaxed_confidence = 0.5
        bi = BeatIntelligence(cfg)

        # Locked event → readiness kicks in → downbeat fires
        decision = bi.build_decision(
            self._event(is_downbeat=True, tempo_locked=True), dt=1 / 60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "downbeat")

    def test_readiness_blocks_with_tempo_lock_required_and_no_confidence(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = True
        cfg.beat.teaching_metronome_relaxed_confidence = 0.5
        bi = BeatIntelligence(cfg)

        decision = bi.build_decision(
            self._event(is_downbeat=True, tempo_locked=False, acf_confidence=0.0),
            dt=1 / 60, silence_override=False,
        )
        self.assertEqual(decision.trigger_kind, "creep")


# ── §19 SilenceDecayState ──────────────────────────────────────────────


class TestSilenceDecayState(Phase2Mixin, unittest.TestCase):
    def test_silence_fade_decreases_during_silence(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        bi._silence_fade = 1.0

        for _ in range(10):
            bi._update_silence_fade(silence_active=True, now=now)
        self.assertLess(bi._silence_fade, 1.0)

    def test_silence_fade_recovers_after_silence(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        bi._silence_fade = 0.3

        for _ in range(20):
            bi._update_silence_fade(silence_active=False, now=now)
        self.assertGreater(bi._silence_fade, 0.3)

    def test_prolonged_silence_triggers_tempo_reset(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()

        any_reset = False
        for _ in range(200):
            _, reset = bi._update_silence_fade(silence_active=True, now=now)
            if reset:
                any_reset = True
        self.assertTrue(any_reset)

    def test_tempo_reset_fires_only_once_per_silence(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()

        reset_count = 0
        for _ in range(300):
            _, reset = bi._update_silence_fade(silence_active=True, now=now)
            if reset:
                reset_count += 1
        self.assertEqual(reset_count, 1)

    def test_silence_fade_in_decision(self):
        bi = BeatIntelligence(Config())
        # During silence, fade should be < 1
        for _ in range(20):
            decision = bi.build_decision(self._event(raw_rms=0.0), dt=1 / 60, silence_override=True)
        self.assertLess(decision.silence_fade, 1.0)

    def test_decision_has_silence_fade_field(self):
        bi = BeatIntelligence(Config())
        decision = bi.build_decision(self._event(), dt=1 / 60)
        self.assertTrue(hasattr(decision, "silence_fade"))
        self.assertTrue(hasattr(decision, "post_silence_ramp"))
        self.assertTrue(hasattr(decision, "request_tempo_reset"))


# ── §14 Post-silence volume ramp ───────────────────────────────────────


class TestPostSilenceRamp(Phase2Mixin, unittest.TestCase):
    def test_ramp_starts_reduced_after_silence(self):
        cfg = Config()
        cfg.stroke.post_silence_vol_reduction = 0.20
        cfg.stroke.post_silence_ramp_seconds = 2.0
        bi = BeatIntelligence(cfg)

        now = time.perf_counter()
        # Enter silence
        bi._was_silent = False
        bi._update_post_silence_ramp(silence_active=True, now=now)
        # Exit silence
        ramp = bi._update_post_silence_ramp(silence_active=False, now=now)
        # Should start at (1 - reduction)
        self.assertAlmostEqual(ramp, 0.80, delta=0.01)

    def test_ramp_reaches_full_after_duration(self):
        cfg = Config()
        cfg.stroke.post_silence_vol_reduction = 0.20
        cfg.stroke.post_silence_ramp_seconds = 2.0
        bi = BeatIntelligence(cfg)

        now = time.perf_counter()
        bi._was_silent = True
        bi._update_post_silence_ramp(silence_active=False, now=now)

        # After full ramp duration
        ramp = bi._update_post_silence_ramp(silence_active=False, now=now + 3.0)
        self.assertAlmostEqual(ramp, 1.0, places=4)

    def test_no_ramp_when_no_prior_silence(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        ramp = bi._update_post_silence_ramp(silence_active=False, now=now)
        self.assertAlmostEqual(ramp, 1.0, places=4)

    def test_ramp_midpoint_is_between_reduced_and_full(self):
        cfg = Config()
        cfg.stroke.post_silence_vol_reduction = 0.20
        cfg.stroke.post_silence_ramp_seconds = 2.0
        bi = BeatIntelligence(cfg)

        now = time.perf_counter()
        bi._was_silent = True
        bi._update_post_silence_ramp(silence_active=False, now=now)

        ramp = bi._update_post_silence_ramp(silence_active=False, now=now + 1.0)
        self.assertGreater(ramp, 0.80)
        self.assertLess(ramp, 1.0)

    def test_decision_post_silence_ramp_field(self):
        bi = BeatIntelligence(Config())
        decision = bi.build_decision(self._event(), dt=1 / 60)
        self.assertAlmostEqual(decision.post_silence_ramp, 1.0, places=4)


# ── StrokeMapper integration ───────────────────────────────────────────


class TestStrokeMapperPhase2Integration(Phase2Mixin, unittest.TestCase):
    def test_silence_uses_fade_not_binary_zero(self):
        """During silence with partial fade, volume should be > 0 but < full."""
        mapper = StrokeMapper(Config(), get_volume=lambda: 1.0)
        # Partially faded silence
        mapper._intelligence._silence_fade = 0.5
        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=True,
            journey_completion=1.0,
            silence_fade=0.5,
        )
        cmd = mapper.process_beat(self._event(raw_rms=0.0))
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertGreater(cmd.volume, 0.0)
        self.assertLess(cmd.volume, 1.0)

    def test_post_silence_ramp_reduces_volume(self):
        """After silence, volume should be reduced by ramp multiplier."""
        mapper = StrokeMapper(Config(), get_volume=lambda: 1.0)
        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="downbeat",
            interval_beats=4,
            radius_bloom=0.85,
            silence_active=False,
            journey_completion=0.5,
            post_silence_ramp=0.85,
        )
        cmd = mapper.process_beat(self._event(is_downbeat=True, raw_rms=0.1))
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertAlmostEqual(cmd.volume, 0.85, delta=0.01)


if __name__ == "__main__":
    unittest.main()
