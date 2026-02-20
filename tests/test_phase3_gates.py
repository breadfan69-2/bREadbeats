"""
Phase 3 checkpoint tests — gate family:
  #5  Low-band fullness gate
  #6  Dual-band dB gate
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


# ── §5 Low-band fullness gate ──────────────────────────────────────────────


class TestLowBandFullnessGate(Phase3Mixin, unittest.TestCase):
    def test_insufficient_data_passes(self):
        """< 8 frames of data → gate passes (don't block)."""
        bi = self._bi()
        for _ in range(5):
            bi._recent_low_band_values.append(0.01)
        result = bi._is_low_band_full_enough(self._event(), "beat", 120.0)
        self.assertTrue(result)

    def test_no_signal_data_passes(self):
        """Deque with 10+ zero values → gate passes (no audio engine)."""
        bi = self._bi()
        for _ in range(20):
            bi._recent_low_band_values.append(0.0)
            bi._recent_high_band_values.append(0.0)
        result = bi._is_low_band_full_enough(self._event(), "beat", 120.0)
        self.assertTrue(result)

    def test_sufficient_low_band_passes(self):
        """Good low-band activity → gate passes."""
        bi = self._bi()
        for _ in range(20):
            bi._recent_low_band_values.append(0.35)
            bi._recent_high_band_values.append(0.20)
        result = bi._is_low_band_full_enough(self._event(), "beat", 120.0)
        self.assertTrue(result)

    def test_low_mean_blocks(self):
        """Mean below threshold without mid-bass support → blocks."""
        bi = self._bi(mid_bass_support_enabled=False)
        for _ in range(20):
            bi._recent_low_band_values.append(0.05)
            bi._recent_high_band_values.append(0.10)
        result = bi._is_low_band_full_enough(self._event(), "beat", 120.0)
        self.assertFalse(result)

    def test_low_occupancy_blocks(self):
        """Some frames above threshold but too few → blocks."""
        bi = self._bi(mid_bass_support_enabled=False)
        # Mix: 3 above threshold, 15 below → occupancy ~0.17
        values = [0.30] * 3 + [0.01] * 15
        for v in values:
            bi._recent_low_band_values.append(v)
            bi._recent_high_band_values.append(0.10)
        # Mean may be above 0.20 * 0.70 floor but occupancy is low
        # Adjust so mean is above threshold but occupancy below
        # Mean ≈ (3*0.30 + 15*0.01) / 18 ≈ 0.058 → below threshold, blocked on mean
        result = bi._is_low_band_full_enough(self._event(), "beat", 120.0)
        self.assertFalse(result)

    def test_downbeat_gets_relaxed_threshold(self):
        """Downbeat uses downbeat_low_band_relax multiplier."""
        bi = self._bi(mid_bass_support_enabled=False, downbeat_low_band_relax=0.85)
        # Value that fails normal threshold (0.20) but passes relaxed (0.17)
        for _ in range(20):
            bi._recent_low_band_values.append(0.18)
            bi._recent_high_band_values.append(0.10)
        self.assertFalse(bi._is_low_band_full_enough(self._event(), "beat", 120.0))
        self.assertTrue(bi._is_low_band_full_enough(self._event(), "downbeat", 120.0))

    def test_mid_bass_support_fallback(self):
        """When low-band fails, mid-bass support can still pass."""
        bi = self._bi(mid_bass_support_enabled=True)
        # Low-band is weak
        for _ in range(20):
            bi._recent_low_band_values.append(0.05)
            bi._recent_high_band_values.append(0.10)
            bi._recent_mid_bass_values.append(0.15)  # above mid-bass threshold (0.035)
        result = bi._is_low_band_full_enough(self._event(), "beat", 120.0)
        self.assertTrue(result)

    def test_low_high_ratio_blocks(self):
        """When high dominates low, gate blocks (treble-only content)."""
        bi = self._bi(mid_bass_support_enabled=False, low_band_to_high_ratio_min=0.58)
        for _ in range(20):
            bi._recent_low_band_values.append(0.25)    # passes mean + occupancy
            bi._recent_high_band_values.append(0.80)   # ratio = 0.25/0.80 = 0.31 < 0.58
        result = bi._is_low_band_full_enough(self._event(), "beat", 120.0)
        self.assertFalse(result)

    def test_wired_into_build_decision(self):
        """In build_decision, low-band gate blocks beat when conditions fail."""
        bi = self._bi(mid_bass_support_enabled=False)
        # Prime with downbeat and allow protected journey to finish
        bi.build_decision(self._event(is_downbeat=True), dt=1/60, silence_override=False)
        self._drain_journey(bi)
        # Fill deques with low activity to trigger gate
        bi.energies.sub_bass = 0.05
        bi.energies.low_mid = 0.02
        for _ in range(20):
            bi._populate_rolling_deques(self._event())
        # Try a beat — should be blocked by low-band gate
        decision = bi.build_decision(
            self._event(is_beat=True), dt=1/60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "creep")


# ── §5 Mid-bass support ────────────────────────────────────────────────────


class TestMidBassSupport(Phase3Mixin, unittest.TestCase):
    def test_insufficient_data_passes(self):
        bi = self._bi()
        for _ in range(5):
            bi._recent_mid_bass_values.append(0.01)
        self.assertTrue(bi._mid_bass_support_passes("beat"))

    def test_no_signal_passes(self):
        bi = self._bi()
        for _ in range(20):
            bi._recent_mid_bass_values.append(0.0)
        self.assertTrue(bi._mid_bass_support_passes("beat"))

    def test_good_mid_bass_passes(self):
        bi = self._bi()
        for _ in range(20):
            bi._recent_mid_bass_values.append(0.10)
        self.assertTrue(bi._mid_bass_support_passes("beat"))

    def test_low_mid_bass_blocks(self):
        bi = self._bi()
        for _ in range(20):
            bi._recent_mid_bass_values.append(0.01)
        self.assertFalse(bi._mid_bass_support_passes("beat"))


# ── §6 Dual-band dB gate ───────────────────────────────────────────────────


class TestDualBandDbGate(Phase3Mixin, unittest.TestCase):
    def test_disabled_passes(self):
        bi = self._bi(dual_band_db_gate_enabled=False)
        bi.energies.sub_bass = 0.0
        bi.energies.high = 0.0
        self.assertTrue(bi._passes_dual_band_db_gate(self._event()))

    def test_no_energy_data_passes(self):
        """When all energies are zero (no audio engine), gate passes."""
        bi = self._bi()
        self.assertTrue(bi._passes_dual_band_db_gate(self._event()))

    def test_both_bands_above_minimum_passes(self):
        bi = self._bi()
        bi.energies.sub_bass = 0.30   # -10.5 dB > -15 dB
        bi.energies.high = 0.10       # -20 dB > -30 dB
        # Fill high-band deque so high-tip sub-gate doesn't block
        for _ in range(20):
            bi._recent_high_band_values.append(0.20)
        self.assertTrue(bi._passes_dual_band_db_gate(self._event()))

    def test_low_sub_bass_blocks(self):
        """Sub-bass below -15 dB threshold blocks."""
        bi = self._bi(high_tip_fullness_enabled=False)
        bi.energies.sub_bass = 0.10   # -20 dB < -15 dB
        bi.energies.high = 0.10       # -20 dB > -30 dB
        # Use frequency outside bass fallback range and low peak_energy
        self.assertFalse(bi._passes_dual_band_db_gate(
            self._event(frequency=500.0, peak_energy=0.001)
        ))

    def test_low_high_band_blocks(self):
        """High-band below -30 dB threshold blocks."""
        bi = self._bi(high_tip_fullness_enabled=False)
        bi.energies.sub_bass = 0.30   # -10.5 dB > -15 dB
        bi.energies.high = 0.001      # -60 dB < -30 dB
        self.assertFalse(bi._passes_dual_band_db_gate(self._event()))

    def test_event_frequency_fallback_sub_bass(self):
        """When sub-bass dB is low but event freq is bass, infer and pass."""
        bi = self._bi(high_tip_fullness_enabled=False)
        bi.energies.sub_bass = 0.10   # -20 dB < -15 dB → would fail
        bi.energies.high = 0.10       # -20 dB > -30 dB
        # Event at 60 Hz with peak_energy — fallback should rescue sub-bass
        result = bi._passes_dual_band_db_gate(
            self._event(frequency=60.0, peak_energy=0.5)
        )
        self.assertTrue(result)

    def test_learning_relax_bypasses(self):
        bi = self._bi(
            teaching_learning_enabled=True,
            teaching_relax_phase1_gates=True,
        )
        bi.energies.sub_bass = 0.001
        bi.energies.high = 0.001
        self.assertTrue(bi._passes_dual_band_db_gate(self._event()))

    def test_high_tip_fullness_sub_gate(self):
        """High-tip fullness blocks when high-band occupancy is low."""
        bi = self._bi(high_tip_fullness_enabled=True)
        bi.energies.sub_bass = 0.30
        bi.energies.high = 0.10
        # Fill high-band deque with near-zero values → low occupancy
        for _ in range(20):
            bi._recent_high_band_values.append(0.001)
        self.assertFalse(bi._passes_dual_band_db_gate(self._event()))

    def test_wired_into_build_decision(self):
        """In build_decision, dual-band gate blocks when sub-bass too low."""
        bi = self._bi(high_tip_fullness_enabled=False)
        # Prime with downbeat and allow protected journey to finish
        bi.build_decision(self._event(is_downbeat=True), dt=1/60, silence_override=False)
        self._drain_journey(bi)
        # Set energy so dual-band fails (sub_bass too low)
        bi.energies.sub_bass = 0.10   # -20 dB < -15 dB min
        bi.energies.high = 0.10       # -20 dB > -30 dB min
        # Fill low-band deques so low-band gate passes
        for _ in range(20):
            bi._recent_low_band_values.append(0.40)
            bi._recent_high_band_values.append(0.20)
            bi._recent_mid_bass_values.append(0.10)
        # Beat with non-bass frequency (no fallback rescue)
        decision = bi.build_decision(
            self._event(is_beat=True, frequency=600.0, peak_energy=0.001),
            dt=1/60, silence_override=False,
        )
        self.assertEqual(decision.trigger_kind, "creep")


# ── §6 High-tip fullness sub-gate ──────────────────────────────────────────


class TestHighTipFullness(Phase3Mixin, unittest.TestCase):
    def test_insufficient_data_passes(self):
        bi = self._bi()
        for _ in range(5):
            bi._recent_high_band_values.append(0.001)
        self.assertTrue(bi._high_tip_fullness_passes())

    def test_no_signal_passes(self):
        bi = self._bi()
        for _ in range(20):
            bi._recent_high_band_values.append(0.0)
        self.assertTrue(bi._high_tip_fullness_passes())

    def test_good_high_band_passes(self):
        bi = self._bi()
        for _ in range(20):
            bi._recent_high_band_values.append(0.20)
        self.assertTrue(bi._high_tip_fullness_passes())

    def test_low_occupancy_blocks(self):
        bi = self._bi()
        # -28 dB linear ≈ 0.0398, most values below that
        for _ in range(20):
            bi._recent_high_band_values.append(0.01)
        self.assertFalse(bi._high_tip_fullness_passes())


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
        bi = self._bi()
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
