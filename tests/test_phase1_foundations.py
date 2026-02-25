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
        self.assertEqual(len(bi._recent_flux_values), 60)
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

        self.assertGreater(bi._tempo_reset_motion_hold_until, now)

    def test_has_recent_beats_true_after_beat(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        event = self._event(is_downbeat=True, monotonic_timestamp=now)
        bi.build_decision(event, dt=1 / 60)
        self.assertTrue(bi._has_recent_beats(now + 0.1))

    def test_has_recent_beats_false_after_long_gap(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        event = self._event(is_downbeat=True, monotonic_timestamp=now)
        bi.build_decision(event, dt=1 / 60)
        self.assertFalse(bi._has_recent_beats(now + 5.0))


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


# ── §7 Mid-trigger block ───────────────────────────────────────────────────


class TestMidTriggerBlock(Phase1Mixin, unittest.TestCase):
    def test_mid_freq_beat_not_runtime_blocked(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        cfg.stroke.block_mid_trigger_range_enabled = True
        bi = BeatIntelligence(cfg)
        bi.energies.sub_bass = 0.01
        bi.energies.low_mid = 0.01
        bi.energies.mid = 0.9

        # Prime with downbeat and allow protected journey to finish
        bi.build_decision(self._event(is_downbeat=True, frequency=60.0), dt=1 / 60, silence_override=False)
        self._drain_journey(bi)
        bi._recent_low_band_values.clear()
        bi._recent_mid_band_values.clear()
        for _ in range(20):
            bi._recent_low_band_values.append(0.01)
            bi._recent_mid_band_values.append(0.9)

        # Mid-trigger helper exists, but runtime chain no longer blocks on it.
        decision = bi.build_decision(
            self._event(is_beat=True, frequency=1500.0), dt=1 / 60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "beat")

    def test_mid_freq_with_strong_bass_not_blocked(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        cfg.stroke.block_mid_trigger_range_enabled = True
        bi = BeatIntelligence(cfg)
        bi.energies.sub_bass = 0.8
        bi.energies.low_mid = 0.7
        bi.energies.mid = 0.9

        bi.build_decision(self._event(is_downbeat=True, frequency=60.0), dt=1 / 60, silence_override=False)
        self._drain_journey(bi)
        bi._recent_low_band_values.clear()
        for _ in range(20):
            bi._recent_low_band_values.append(0.8)

        decision = bi.build_decision(
            self._event(is_beat=True, frequency=1500.0), dt=1 / 60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "beat")

    def test_bass_freq_beat_not_blocked(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        cfg.stroke.block_mid_trigger_range_enabled = True
        bi = BeatIntelligence(cfg)

        bi.build_decision(self._event(is_downbeat=True, frequency=60.0), dt=1 / 60, silence_override=False)
        self._drain_journey(bi)

        # Beat at 60 Hz (bass) should pass
        decision = bi.build_decision(
            self._event(is_beat=True, frequency=60.0), dt=1 / 60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "beat")

    def test_mid_block_disabled_allows_mid_freq(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        cfg.stroke.block_mid_trigger_range_enabled = False
        bi = BeatIntelligence(cfg)
        bi.energies.sub_bass = 0.01
        bi.energies.low_mid = 0.01
        bi.energies.mid = 0.9

        bi.build_decision(self._event(is_downbeat=True), dt=1 / 60, silence_override=False)
        self._drain_journey(bi)

        decision = bi.build_decision(
            self._event(is_beat=True, frequency=1500.0), dt=1 / 60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "beat")

    def test_learning_relax_bypasses_mid_block(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        cfg.stroke.block_mid_trigger_range_enabled = True
        cfg.beat.teaching_learning_enabled = True
        cfg.beat.teaching_relax_phase1_gates = True
        bi = BeatIntelligence(cfg)
        bi.energies.sub_bass = 0.01
        bi.energies.low_mid = 0.01
        bi.energies.mid = 0.9

        bi.build_decision(self._event(is_downbeat=True), dt=1 / 60, silence_override=False)
        self._drain_journey(bi)

        decision = bi.build_decision(
            self._event(is_beat=True, frequency=1500.0), dt=1 / 60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "beat")

    def test_mid_block_window_is_adjustable_via_mid_block_window_frames(self):
        cfg = Config()
        cfg.stroke.block_mid_trigger_range_enabled = True
        cfg.stroke.block_mid_trigger_window_frames = 2
        bi = BeatIntelligence(cfg)

        # Older history indicates low bass / high mid, but recent 2 frames are bass-strong.
        for _ in range(14):
            bi._recent_mid_band_values.append(0.9)
            bi._recent_low_band_values.append(0.01)
        for _ in range(2):
            bi._recent_mid_band_values.append(0.9)
            bi._recent_low_band_values.append(0.9)

        self.assertFalse(bi._is_mid_trigger_blocked(self._event(frequency=1500.0)))

        # Larger window includes older low-bass history, so it blocks.
        cfg.stroke.block_mid_trigger_window_frames = 16
        self.assertTrue(bi._is_mid_trigger_blocked(self._event(frequency=1500.0)))

    def test_mid_block_uses_dedicated_bass_to_mid_ratio(self):
        cfg = Config()
        cfg.stroke.block_mid_trigger_range_enabled = True
        cfg.stroke.low_band_activity_threshold = 0.80
        cfg.stroke.block_mid_trigger_bass_to_mid_max_ratio = 0.20
        bi = BeatIntelligence(cfg)

        for _ in range(20):
            bi._recent_mid_band_values.append(0.90)
            bi._recent_low_band_values.append(0.17)

        # Should block using dedicated mid-block bass-to-mid ratio (0.20),
        # even though generic low-band threshold is much higher.
        self.assertTrue(bi._is_mid_trigger_blocked(self._event(frequency=1500.0)))


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
        activity = bi._get_mid_bass_activity(self._event())
        self.assertAlmostEqual(activity, 0.5, places=4)

    def test_high_band_presence_status_passes_when_enough_data(self):
        bi = BeatIntelligence(Config())
        # Fill with above-threshold values
        for _ in range(20):
            bi._recent_high_band_values.append(0.30)
        self.assertTrue(bi._get_high_band_presence_status())

    def test_high_band_presence_status_fails_when_low(self):
        cfg = Config()
        cfg.stroke.high_band_window_frames = 18
        bi = BeatIntelligence(cfg)
        for _ in range(20):
            bi._recent_high_band_values.append(0.01)
        self.assertFalse(bi._get_high_band_presence_status())

    def test_high_band_presence_status_passes_with_insufficient_data(self):
        bi = BeatIntelligence(Config())
        # < 8 frames → don't block
        for _ in range(3):
            bi._recent_high_band_values.append(0.01)
        self.assertTrue(bi._get_high_band_presence_status())


if __name__ == "__main__":
    unittest.main()
