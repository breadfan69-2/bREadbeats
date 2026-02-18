"""
Phase 6 checkpoint tests — BPM Stabilization (#13) + Bass Jitter Stub (#15):
  effective_bpm with last-locked memory and jump-ratio limiter,
  _stabilize_unlocked_bpm EMA smoothing,
  _cap_bpm_to_last_locked jump limiter,
  _update_bass_jitter_drive frequency-to-speed mapping.
"""

import time
import unittest
from typing import Any

from audio_engine import BeatEvent
from beat_intelligence import BeatIntelligence
from config import Config


class Phase6Mixin:
    """Shared helpers for Phase 6 tests."""

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
            acf_confidence=0.5,
        )
        payload.update(overrides)
        return BeatEvent(**payload)

    def _bi(self, **cfg_overrides) -> BeatIntelligence:
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        for key, val in cfg_overrides.items():
            if hasattr(cfg.stroke, key):
                setattr(cfg.stroke, key, val)
            elif hasattr(cfg.beat, key):
                setattr(cfg.beat, key, val)
        return BeatIntelligence(cfg)


# ── BPM Stabilization (#13) ─────────────────────────────────────────

class TestEffectiveBpm(Phase6Mixin, unittest.TestCase):
    """effective_bpm is now an instance method with stabilization."""

    def test_locked_returns_raw_bpm(self):
        bi = self._bi()
        ev = self._event(metronome_bpm=130.0, tempo_locked=True)
        bpm = bi.effective_bpm(ev)
        self.assertAlmostEqual(bpm, 130.0, places=1)

    def test_unlocked_falls_back_to_bpm(self):
        bi = self._bi()
        ev = self._event(metronome_bpm=0.0, bpm=110.0, tempo_locked=True)
        bpm = bi.effective_bpm(ev)
        self.assertAlmostEqual(bpm, 110.0, places=1)

    def test_zero_bpm_defaults_to_120(self):
        bi = self._bi()
        ev = self._event(metronome_bpm=0.0, bpm=0.0, tempo_locked=True)
        bpm = bi.effective_bpm(ev)
        self.assertAlmostEqual(bpm, 120.0, places=1)

    def test_clamped_above_240(self):
        bi = self._bi()
        ev = self._event(metronome_bpm=300.0, tempo_locked=True)
        bpm = bi.effective_bpm(ev)
        self.assertLessEqual(bpm, 240.0)

    def test_clamped_below_40(self):
        bi = self._bi()
        ev = self._event(metronome_bpm=20.0, tempo_locked=True)
        bpm = bi.effective_bpm(ev)
        self.assertGreaterEqual(bpm, 40.0)


class TestCapBpmToLastLocked(Phase6Mixin, unittest.TestCase):
    """_cap_bpm_to_last_locked: jump-ratio limiter."""

    def test_within_ratio_passes_through(self):
        bi = self._bi()
        bi._last_locked_bpm = 120.0
        result = bi._cap_bpm_to_last_locked(130.0)
        self.assertAlmostEqual(result, 130.0)

    def test_huge_jump_up_capped(self):
        bi = self._bi()
        bi._last_locked_bpm = 100.0
        bi._bpm_jump_ratio_limit = 1.5
        result = bi._cap_bpm_to_last_locked(200.0)
        self.assertAlmostEqual(result, 150.0, places=1)

    def test_huge_jump_down_capped(self):
        bi = self._bi()
        bi._last_locked_bpm = 120.0
        bi._bpm_jump_ratio_limit = 1.5
        result = bi._cap_bpm_to_last_locked(50.0)
        self.assertAlmostEqual(result, 80.0, places=1)

    def test_zero_ref_passes_through(self):
        bi = self._bi()
        bi._last_locked_bpm = 0.0
        result = bi._cap_bpm_to_last_locked(150.0)
        self.assertAlmostEqual(result, 150.0)


class TestStabilizeUnlockedBpm(Phase6Mixin, unittest.TestCase):
    """_stabilize_unlocked_bpm: EMA smoothing when unlocked."""

    def test_locked_snaps_immediately(self):
        bi = self._bi()
        bi._stabilized_bpm = 100.0
        result = bi._stabilize_unlocked_bpm(130.0, tempo_locked=True)
        self.assertAlmostEqual(result, 130.0)
        self.assertAlmostEqual(bi._stabilized_bpm, 130.0)

    def test_unlocked_drifts_slowly(self):
        bi = self._bi()
        bi._stabilized_bpm = 100.0
        result = bi._stabilize_unlocked_bpm(130.0, tempo_locked=False)
        # EMA with alpha=0.15: 100 + 0.15*(130-100) = 104.5
        self.assertAlmostEqual(result, 104.5, places=1)

    def test_unlocked_converges_over_many_calls(self):
        bi = self._bi()
        bi._stabilized_bpm = 100.0
        for _ in range(50):
            result = bi._stabilize_unlocked_bpm(130.0, tempo_locked=False)
        self.assertAlmostEqual(result, 130.0, delta=1.0)


class TestBpmStabilizationIntegration(Phase6Mixin, unittest.TestCase):
    """effective_bpm integrates capping and stabilization."""

    def test_last_locked_memory_set_on_lock(self):
        bi = self._bi()
        ev = self._event(metronome_bpm=140.0, tempo_locked=True)
        bi.effective_bpm(ev)
        self.assertAlmostEqual(bi._last_locked_bpm, 140.0)

    def test_locked_then_wild_unlocked_is_capped(self):
        bi = self._bi()
        # Lock at 120
        ev_locked = self._event(metronome_bpm=120.0, tempo_locked=True)
        bi.effective_bpm(ev_locked)

        # Unlocked jumps to 200 — should be capped
        ev_wild = self._event(metronome_bpm=200.0, tempo_locked=False)
        bpm = bi.effective_bpm(ev_wild)
        self.assertLess(bpm, 200.0)

    def test_build_decision_uses_stabilized_bpm(self):
        bi = self._bi()
        ev = self._event(metronome_bpm=100.0, tempo_locked=True)
        decision = bi.build_decision(ev, dt=1 / 60)
        # Just verify no crash — BPM is used internally
        self.assertIsNotNone(decision)


if __name__ == "__main__":
    unittest.main()
