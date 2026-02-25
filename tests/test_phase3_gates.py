"""
Phase 3 checkpoint tests — gate family:
  #8  Spectrum fill gate (+ auto-fill adaptation #20)
  #16 Flux-drop guard
"""

import time
import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from audio_engine import BeatEvent
from beat_intelligence import BeatIntelligence
from config import Config


class Phase3Mixin:
    """Shared event factory for Phase 3 tests."""

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

    def _bi(self, **cfg_overrides) -> BeatIntelligence:
        """Create a BeatIntelligence with tempo_lock_required=False for easy testing."""
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        for key, val in cfg_overrides.items():
            if hasattr(cfg.stroke, key):
                setattr(cfg.stroke, key, val)
            elif hasattr(cfg.beat, key):
                setattr(cfg.beat, key, val)
        return BeatIntelligence(cfg)

    def _drain_journey(self, bi: BeatIntelligence, frames: int = 140) -> None:
        for _ in range(frames):
            bi.build_decision(self._event(), dt=1 / 60, silence_override=False)



# ── §8 Spectrum fill gate ──────────────────────────────────────────────────


class TestSpectrumFillGate(Phase3Mixin, unittest.TestCase):
    def test_no_audio_engine_passes(self):
        bi = self._bi()
        self.assertAlmostEqual(bi._get_spectrum_fill_ratio("beat"), 1.0)
        self.assertTrue(bi._passes_overall_amp_fill_gate(self._event(), "beat"))

    def test_none_spectrum_passes(self):
        bi = self._bi()
        engine = MagicMock()
        engine.get_spectrum.return_value = None
        bi.audio_engine = engine
        self.assertAlmostEqual(bi._get_spectrum_fill_ratio("beat"), 1.0)

    def test_flat_spectrum_high_fill(self):
        """Uniform spectrum → high fill ratio."""
        bi = self._bi()
        engine = MagicMock()
        engine.get_spectrum.return_value = np.ones(1024) * 0.5
        bi.audio_engine = engine
        ratio = bi._get_spectrum_fill_ratio("beat")
        self.assertGreater(ratio, 0.8)

    def test_sparse_spectrum_low_fill(self):
        """Many active bins below fill threshold → low fill ratio."""
        bi = self._bi()
        engine = MagicMock()
        # Peak at bin 0, many bins with small values (above active floor but below fill threshold)
        spectrum = np.full(1024, 0.03)  # all bins just above active floor (0.02)
        spectrum[0] = 1.0               # peak
        engine.get_spectrum.return_value = spectrum
        bi.audio_engine = engine
        ratio = bi._get_spectrum_fill_ratio("beat")
        self.assertLess(ratio, 0.1)  # most active bins are 0.03/1.0 = 0.03 < 0.5 target

    def test_zero_spectrum_returns_zero(self):
        bi = self._bi()
        engine = MagicMock()
        engine.get_spectrum.return_value = np.zeros(1024)
        bi.audio_engine = engine
        self.assertAlmostEqual(bi._get_spectrum_fill_ratio("beat"), 0.0)

    def test_overall_fill_gate_disabled_passes(self):
        bi = self._bi(overall_amp_fill_gate_enabled=False)
        engine = MagicMock()
        engine.get_spectrum.return_value = np.zeros(1024)
        bi.audio_engine = engine
        self.assertTrue(bi._passes_overall_amp_fill_gate(self._event(), "beat"))

    def test_overall_fill_gate_blocks_when_low_fill(self):
        bi = self._bi(beat_overall_amp_fill_sustain_frames=1)  # Test instant decision
        engine = MagicMock()
        # Many bins above active floor but below fill threshold
        spectrum = np.full(1024, 0.03)
        spectrum[0] = 1.0
        engine.get_spectrum.return_value = spectrum
        engine._band_energies = {}
        bi.audio_engine = engine
        result = bi._passes_overall_amp_fill_gate(self._event(intensity=0.5), "beat")
        self.assertFalse(result)

    def test_overall_fill_gate_passes_when_rich_spectrum(self):
        bi = self._bi(beat_overall_amp_fill_sustain_frames=1)  # Test instant decision
        engine = MagicMock()
        engine.get_spectrum.return_value = np.ones(1024) * 0.5
        engine._band_energies = {}
        bi.audio_engine = engine
        result = bi._passes_overall_amp_fill_gate(self._event(intensity=0.5), "beat")
        self.assertTrue(result)

    def test_low_intensity_blocks_regardless_of_spectrum(self):
        """Intensity below (target - tolerance) → blocked."""
        bi = self._bi(
            overall_amp_fill_target=0.5,
            overall_amp_fill_tolerance=0.1,
            beat_overall_amp_fill_sustain_frames=1  # Test instant decision
        )
        engine = MagicMock()
        engine.get_spectrum.return_value = np.ones(1024) * 0.5
        engine._band_energies = {}
        bi.audio_engine = engine
        # intensity 0.1 < (0.5 - 0.1 = 0.4) → fail
        result = bi._passes_overall_amp_fill_gate(
            self._event(intensity=0.1), "beat"
        )
        self.assertFalse(result)

    def test_per_phase_bin_windows(self):
        """Different trigger kinds use different FFT bin windows."""
        bi = self._bi(
            downbeat_fill_bin_low=0, downbeat_fill_bin_high=100,
            beat_fill_bin_low=0, beat_fill_bin_high=200,
            syncopation_fill_bin_low=0, syncopation_fill_bin_high=50,
        )
        engine = MagicMock()
        # Bins 0-100: value 1.0, bins 101-200: small value (above floor, below fill threshold)
        spectrum = np.full(1024, 0.005)  # below active floor
        spectrum[:101] = 1.0
        spectrum[101:201] = 0.03  # above active floor (0.02) but below fill threshold (0.5)
        engine.get_spectrum.return_value = spectrum
        bi.audio_engine = engine

        # Downbeat window 0-100: all active bins are 1.0
        ratio_db = bi._get_spectrum_fill_ratio("downbeat")
        # Beat window 0-200: 101 bins at 1.0, 100 bins at 0.03 → lower fill ratio
        ratio_bt = bi._get_spectrum_fill_ratio("beat")
        # Sync window 0-50: all active bins are 1.0
        ratio_sync = bi._get_spectrum_fill_ratio("syncopation")

        self.assertGreater(ratio_db, ratio_bt)
        self.assertGreater(ratio_sync, ratio_bt)

    def test_sustained_duration_requires_consecutive_frames(self):
        """Sustained duration gate requires fill to be maintained over consecutive frames."""
        bi = self._bi(beat_overall_amp_fill_sustain_frames=3)  # Require 3 consecutive frames
        engine = MagicMock()
        engine.get_spectrum.return_value = np.ones(1024) * 0.5  # Rich spectrum
        engine._band_energies = {}
        bi.audio_engine = engine
        
        evt = self._event(intensity=0.5)
        
        # First frame: passes instant check but not sustained (count=1)
        result = bi._passes_overall_amp_fill_gate(evt, "beat")
        self.assertFalse(result)
        self.assertEqual(bi._fill_pass_consecutive["beat"], 1)
        
        # Second frame: still not sustained (count=2)
        result = bi._passes_overall_amp_fill_gate(evt, "beat")
        self.assertFalse(result)
        self.assertEqual(bi._fill_pass_consecutive["beat"], 2)
        
        # Third frame: NOW sustained (count=3) → passes
        result = bi._passes_overall_amp_fill_gate(evt, "beat")
        self.assertTrue(result)
        self.assertEqual(bi._fill_pass_consecutive["beat"], 3)
        
        # Fourth frame: still passes (count=4)
        result = bi._passes_overall_amp_fill_gate(evt, "beat")
        self.assertTrue(result)
        self.assertEqual(bi._fill_pass_consecutive["beat"], 4)
        
        # Change to sparse spectrum → instant fail, counter reset
        engine.get_spectrum.return_value = np.full(1024, 0.03)  # Sparse
        engine.get_spectrum.return_value[0] = 1.0  # Peak only
        result = bi._passes_overall_amp_fill_gate(evt, "beat")
        self.assertFalse(result)
        self.assertEqual(bi._fill_pass_consecutive["beat"], 0)


# ── §20 Auto-fill adaptation ───────────────────────────────────────────────


class TestAutoFillAdaptation(Phase3Mixin, unittest.TestCase):
    def test_initial_offsets_zero(self):
        bi = self._bi()
        self.assertAlmostEqual(bi._auto_fill_offsets["beat"], 0.0)
        self.assertAlmostEqual(bi._auto_fill_offsets["downbeat"], 0.0)
        self.assertAlmostEqual(bi._auto_fill_offsets["syncopation"], 0.0)

    def test_initial_ema_at_half(self):
        bi = self._bi()
        self.assertAlmostEqual(bi._auto_fill_ema["beat"], 0.5)

    def test_repeated_passes_tighten_offset(self):
        """Consistently passing → offset increases (tighten)."""
        bi = self._bi()
        for _ in range(50):
            bi._update_auto_fill_required("beat", gate_passed=True)
        self.assertGreater(bi._auto_fill_offsets["beat"], 0.0)

    def test_repeated_failures_relax_offset(self):
        """Consistently failing → offset decreases (relax)."""
        bi = self._bi()
        for _ in range(50):
            bi._update_auto_fill_required("beat", gate_passed=False)
        self.assertLess(bi._auto_fill_offsets["beat"], 0.0)

    def test_offset_bounded_by_max(self):
        """Offset never exceeds max_offset."""
        bi = self._bi(overall_amp_fill_auto_max_offset=0.35)
        for _ in range(500):
            bi._update_auto_fill_required("beat", gate_passed=True)
        self.assertLessEqual(bi._auto_fill_offsets["beat"], 0.35)

    def test_offset_bounded_negative(self):
        """Offset never goes below -max_offset."""
        bi = self._bi(overall_amp_fill_auto_max_offset=0.35)
        for _ in range(500):
            bi._update_auto_fill_required("beat", gate_passed=False)
        self.assertGreaterEqual(bi._auto_fill_offsets["beat"], -0.35)

    def test_disabled_does_nothing(self):
        bi = self._bi(overall_amp_fill_auto_enabled=False)
        bi._update_auto_fill_required("beat", gate_passed=True)
        self.assertAlmostEqual(bi._auto_fill_offsets["beat"], 0.0)

    def test_fill_required_respects_offset(self):
        """_get_overall_amp_fill_required includes auto-fill offset."""
        bi = self._bi(beat_overall_amp_fill_required=0.70,
                      overall_amp_fill_required_scale=1.0)
        base = bi._get_overall_amp_fill_required("beat")
        bi._auto_fill_offsets["beat"] = -0.10
        relaxed = bi._get_overall_amp_fill_required("beat")
        self.assertLess(relaxed, base)

    def test_fill_required_clamped(self):
        """Required value is clamped to [min, max]."""
        bi = self._bi(
            overall_amp_fill_auto_min_required=0.05,
            overall_amp_fill_auto_max_required=0.98,
        )
        bi._auto_fill_offsets["beat"] = -5.0
        req = bi._get_overall_amp_fill_required("beat")
        self.assertGreaterEqual(req, 0.05)

        bi._auto_fill_offsets["beat"] = 5.0
        req = bi._get_overall_amp_fill_required("beat")
        self.assertLessEqual(req, 0.98)

    def test_per_phase_independence(self):
        """Each phase has its own offset/EMA state."""
        bi = self._bi()
        for _ in range(30):
            bi._update_auto_fill_required("beat", gate_passed=True)
            bi._update_auto_fill_required("downbeat", gate_passed=False)
        self.assertGreater(bi._auto_fill_offsets["beat"], 0.0)
        self.assertLess(bi._auto_fill_offsets["downbeat"], 0.0)


# ── Integration: gate cascade ordering ──────────────────────────────────────


class TestGateCascadeIntegration(Phase3Mixin, unittest.TestCase):
    def test_all_gates_pass_when_no_audio_engine(self):
        """Without an audio engine, all Phase 3 gates pass."""
        bi = self._bi()
        # Prime with downbeat and allow protected journey to finish
        bi.build_decision(self._event(is_downbeat=True), dt=1/60, silence_override=False)
        self._drain_journey(bi)
        decision = bi.build_decision(
            self._event(is_beat=True), dt=1/60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "beat")

    def test_downbeat_always_passes_basic_gates(self):
        """Downbeat with relaxation should pass even marginal conditions."""
        bi = self._bi()
        decision = bi.build_decision(
            self._event(is_downbeat=True), dt=1/60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "downbeat")

    def test_gates_do_not_affect_creep_events(self):
        """Creep-classified events bypass gate cascade entirely."""
        bi = self._bi()
        # Fill deques with bad data
        bi.energies.sub_bass = 0.001
        bi.energies.high = 0.001
        for _ in range(20):
            bi._populate_rolling_deques(self._event())
        decision = bi.build_decision(
            self._event(is_beat=False, is_downbeat=False, is_syncopated=False),
            dt=1/60, silence_override=False,
        )
        self.assertEqual(decision.trigger_kind, "creep")
        # No crash, no unexpected type change

    def test_journey_continuity_preserved_through_gates(self):
        """An active beat journey is not interrupted by subsequent creep frames."""
        bi = self._bi()
        # Prime with downbeat and allow protected journey to finish before beat trigger
        bi.build_decision(self._event(is_downbeat=True), dt=1/60, silence_override=False)
        self._drain_journey(bi)
        d1 = bi.build_decision(self._event(is_beat=True), dt=1/60, silence_override=False)
        self.assertEqual(d1.trigger_kind, "beat")

        # Intermediate frame with no beat flags → journey continues
        d2 = bi.build_decision(
            self._event(is_beat=False, is_downbeat=False), dt=1/60, silence_override=False
        )
        self.assertEqual(d2.trigger_kind, "beat")
        self.assertGreater(d2.journey_completion, 0.0)


if __name__ == "__main__":
    unittest.main()
