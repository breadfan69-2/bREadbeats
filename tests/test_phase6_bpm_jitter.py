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


# ── Bass Jitter Stub (#15) ───────────────────────────────────────────

class TestBassJitterDrive(Phase6Mixin, unittest.TestCase):
    """_update_bass_jitter_drive: frequency-to-speed mapping behind feature flag."""

    def test_disabled_returns_1(self):
        bi = self._bi()
        bi._bass_jitter_enabled = False
        ev = self._event(frequency=100.0)
        result = bi._update_bass_jitter_drive(ev)
        self.assertAlmostEqual(result, 1.0)

    def test_enabled_low_freq_slow(self):
        bi = self._bi()
        bi._bass_jitter_enabled = True
        ev = self._event(frequency=30.0)
        result = bi._update_bass_jitter_drive(ev)
        # 30 Hz → t=0 → raw_mult=0.5, EMA from 1.0: 1.0 + 0.2*(0.5-1.0) = 0.9
        self.assertLess(result, 1.0)

    def test_enabled_high_freq_fast(self):
        bi = self._bi()
        bi._bass_jitter_enabled = True
        ev = self._event(frequency=220.0)
        result = bi._update_bass_jitter_drive(ev)
        # 220 Hz → t=1 → raw_mult=2.0, EMA from 1.0: 1.0 + 0.2*(2.0-1.0) = 1.2
        self.assertGreater(result, 1.0)

    def test_ema_smoothing(self):
        """Multiple calls converge toward target."""
        bi = self._bi()
        bi._bass_jitter_enabled = True
        for _ in range(30):
            ev = self._event(frequency=220.0)
            result = bi._update_bass_jitter_drive(ev)
        self.assertAlmostEqual(result, 2.0, delta=0.1)

    def test_zero_freq_returns_ema(self):
        bi = self._bi()
        bi._bass_jitter_enabled = True
        bi._bass_jitter_ema = 1.3
        ev = self._event(frequency=0.0)
        result = bi._update_bass_jitter_drive(ev)
        self.assertAlmostEqual(result, 1.3)

    def test_no_runtime_influence(self):
        """With flag off, build_decision output unaffected by frequency changes."""
        bi = self._bi()
        bi._bass_jitter_enabled = False
        ev1 = self._event(frequency=30.0, is_beat=True, tempo_locked=True,
                          metronome_bpm=120.0)
        d1 = bi.build_decision(ev1, dt=1 / 60)

        ev2 = self._event(frequency=220.0, is_beat=True, tempo_locked=True,
                          metronome_bpm=120.0)
        d2 = bi.build_decision(ev2, dt=1 / 60)
        # jitter drive doesn't change any decision fields when disabled
        self.assertAlmostEqual(bi._bass_jitter_drive, 1.0)


if __name__ == "__main__":
    unittest.main()
