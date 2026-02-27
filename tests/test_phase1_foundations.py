"""
Phase 1 checkpoint tests — rolling deques, FluxTracker, beat-family admission,
no-beat timeout, mid-trigger block, activity helpers.
"""

import time
import unittest
from typing import Any

from audio_engine import BeatEvent
from beat_intelligence import BeatIntelligence
from config import Config


class Phase1Mixin:
    """Shared event factory for Phase 1 tests."""

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

    def _drain_journey(self, bi: BeatIntelligence, frames: int = 140) -> None:
        for _ in range(frames):
            bi.build_decision(self._event(), dt=1 / 60, silence_override=False)


# ── §1 Rolling history deques ──────────────────────────────────────────────


class TestRollingDeques(Phase1Mixin, unittest.TestCase):
    def test_deques_populate_on_build_decision(self):
        bi = BeatIntelligence(Config())
        self.assertEqual(len(bi._recent_flux_values), 0)
        bi.build_decision(self._event(), dt=1 / 60)
        self.assertEqual(len(bi._recent_flux_values), 1)
        self.assertEqual(len(bi._recent_low_band_values), 1)
        self.assertEqual(len(bi._recent_high_band_values), 1)
        self.assertEqual(len(bi._recent_mid_bass_values), 1)

    def test_deques_respect_maxlen(self):
        bi = BeatIntelligence(Config())
        for _ in range(80):
            bi.build_decision(self._event(), dt=1 / 60)
        # _recent_flux_values holds ~10 s of history (600 frames @ 60 fps)
        self.assertEqual(len(bi._recent_flux_values), 80)
        self.assertEqual(len(bi._recent_low_band_values), 60)


# ── §2 FluxTracker ─────────────────────────────────────────────────────────


class TestFluxTracker(Phase1Mixin, unittest.TestCase):
    def test_flux_history_populated(self):
        bi = BeatIntelligence(Config())
        bi.build_decision(self._event(spectral_flux=0.05), dt=1 / 60)
        self.assertGreaterEqual(len(bi._flux_history), 1)


# ── §3 Beat-family admission ───────────────────────────────────────────────


class TestBeatHierarchyGuards(Phase1Mixin, unittest.TestCase):
    def test_beat_without_prior_downbeat_falls_to_creep(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        bi = BeatIntelligence(cfg)

        decision = bi.build_decision(self._event(is_beat=True), dt=1 / 60, silence_override=False)
        self.assertEqual(decision.trigger_kind, "beat")

    def test_beat_after_downbeat_fires(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        bi = BeatIntelligence(cfg)

        bi.build_decision(self._event(is_downbeat=True), dt=1 / 60, silence_override=False)
        self._drain_journey(bi)
        decision = bi.build_decision(self._event(is_beat=True), dt=1 / 60, silence_override=False)
        self.assertEqual(decision.trigger_kind, "beat")

    def test_syncopation_without_downbeat_falls_to_creep(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        bi = BeatIntelligence(cfg)

        decision = bi.build_decision(
            self._event(is_syncopated=True), dt=1 / 60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "syncopation")

    def test_syncopation_after_downbeat_and_beat_fires(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        bi = BeatIntelligence(cfg)

        bi.build_decision(self._event(is_downbeat=True), dt=1 / 60, silence_override=False)
        self._drain_journey(bi)
        bi.build_decision(self._event(is_beat=True), dt=1 / 60, silence_override=False)
        self._drain_journey(bi)
        decision = bi.build_decision(
            self._event(is_syncopated=True), dt=1 / 60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "syncopation")

    def test_downbeat_always_allowed_even_cold_start(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        bi = BeatIntelligence(cfg)

        decision = bi.build_decision(self._event(is_downbeat=True), dt=1 / 60, silence_override=False)
        self.assertEqual(decision.trigger_kind, "downbeat")

    def test_tempo_reset_arms_motion_hold(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        bi = BeatIntelligence(cfg)

        now = time.perf_counter()
        event = self._event(is_downbeat=True, monotonic_timestamp=now)
        event.tempo_reset = True
        bi.build_decision(event, dt=1 / 60, silence_override=False)

        self.assertGreaterEqual(bi._last_any_beat_time, now)


# ── §4 No-beat timeout ─────────────────────────────────────────────────────


class TestNoBeatTimeout(Phase1Mixin, unittest.TestCase):
    def test_no_timeout_on_cold_start(self):
        bi = BeatIntelligence(Config())
        self.assertFalse(bi._check_no_beat_timeout(time.perf_counter()))

    def test_timeout_triggers_after_gap(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        bi._last_any_beat_time = now - 3.1  # 3.1s ago (well past 3.0s timeout)
        self.assertTrue(bi._check_no_beat_timeout(now))

    def test_timeout_does_not_trigger_when_recent(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        bi._last_any_beat_time = now - 0.5  # 0.5s ago
        self.assertFalse(bi._check_no_beat_timeout(now))

    def test_timeout_forces_journey_inactive_in_build_decision(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        bi = BeatIntelligence(cfg)

        # Start a journey
        bi.build_decision(self._event(is_downbeat=True), dt=1 / 60, silence_override=False)
        self.assertTrue(bi.journey_active)

        # Simulate 3s gap by backdating last beat time
        bi._last_any_beat_time = time.perf_counter() - 3.0

        decision = bi.build_decision(self._event(), dt=1 / 60, silence_override=False)
        self.assertFalse(bi.journey_active)
        self.assertEqual(decision.trigger_kind, "creep")


# ── §21 Activity helpers ───────────────────────────────────────────────────


class TestActivityHelpers(Phase1Mixin, unittest.TestCase):
    def test_low_band_activity_from_sub_bass(self):
        bi = BeatIntelligence(Config())
        bi.energies.sub_bass = 0.8
        bi.energies.low_mid = 0.0
        activity = bi._get_low_band_activity(self._event())
        self.assertGreater(activity, 0.0)

    def test_high_band_activity_from_high(self):
        bi = BeatIntelligence(Config())
        bi.energies.high = 0.9
        bi.energies.mid = 0.0
        activity = bi._get_high_band_activity(self._event())
        self.assertGreater(activity, 0.0)

    def test_mid_bass_activity_from_low_mid(self):
        bi = BeatIntelligence(Config())
        bi.energies.low_mid = 0.5
        activity = max(0.0, min(1.0, float(bi.energies.low_mid)))
        self.assertAlmostEqual(activity, 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
