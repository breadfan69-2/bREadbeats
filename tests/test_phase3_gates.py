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
        """Sparse spectrum with only one strong bin → low fill ratio in dBFS mode."""
        bi = self._bi()
        engine = MagicMock()
        # Peak at bin 0 is 1.0, rest are 0.001 (far below any dBFS threshold)
        spectrum = np.full(1024, 0.001)
        spectrum[0] = 1.0
        engine.get_spectrum.return_value = spectrum
        bi.audio_engine = engine
        ratio = bi._get_spectrum_fill_ratio("beat")
        self.assertLess(ratio, 0.1)  # only 1 bin above dBFS threshold out of the window

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
        bi = self._bi(beat_overall_amp_fill_sustain_frames=1)
        engine = MagicMock()
        # Sparse spectrum: only 1 bin strong, rest far below dBFS threshold
        spectrum = np.full(1024, 0.001)
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
        # Bins 0-100: value 1.0, bins 101-200: very small (below any dBFS threshold)
        spectrum = np.full(1024, 0.001)  # far below threshold
        spectrum[:101] = 1.0
        engine.get_spectrum.return_value = spectrum
        bi.audio_engine = engine

        # Downbeat window 0-100: all bins are 1.0 → fill ratio ≈ 1.0
        ratio_db = bi._get_spectrum_fill_ratio("downbeat")
        # Beat window 0-200: 101 bins at 1.0, 100 bins at 0.001 → lower fill ratio
        ratio_bt = bi._get_spectrum_fill_ratio("beat")
        # Sync window 0-50: all bins are 1.0 → fill ratio ≈ 1.0
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
        sparse = np.full(1024, 0.001)  # Far below dBFS threshold
        sparse[0] = 1.0  # Peak only
        engine.get_spectrum.return_value = sparse
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


# ── Phrase commitment coverage ─────────────────────────────────────────────


class TestPhraseCommitment(Phase3Mixin, unittest.TestCase):
    def test_fill_to_beat_starts_eight_beat_lock(self):
        bi = self._bi()
        bi._phrase_renew_ratio = 2.0  # disable renewal so lock can naturally end for assertion
        bi.last_trigger_kind = "creep"
        bi._recent_flux_values.clear()
        bi._recent_flux_values.extend([1.0, 1.0, 1.0, 1.0])

        trigger_kind = bi._update_phrase_commitment(
            trigger_kind="beat",
            silence_active=False,
            gate_fail_reason="",
            is_beat_event=True,
        )
        self.assertEqual(trigger_kind, "beat")
        self.assertTrue(bi._phrase_committed)
        self.assertEqual(bi._phrase_beats_remaining, 7)

        for beat_index in range(6):
            trigger_kind = bi._update_phrase_commitment(
                trigger_kind="creep",
                silence_active=False,
                gate_fail_reason="",
                is_beat_event=True,
            )
            self.assertEqual(trigger_kind, "beat")
            self.assertTrue(bi._phrase_committed)
            self.assertEqual(bi._phrase_beats_remaining, 6 - beat_index)

        trigger_kind = bi._update_phrase_commitment(
            trigger_kind="creep",
            silence_active=False,
            gate_fail_reason="",
            is_beat_event=True,
        )
        self.assertEqual(trigger_kind, "beat")
        self.assertFalse(bi._phrase_committed)
        self.assertEqual(bi._phrase_beats_remaining, 0)

    def test_flux_crash_cancels_commitment_early(self):
        bi = self._bi()
        bi.last_trigger_kind = "creep"
        bi._recent_flux_values.clear()
        bi._recent_flux_values.extend([1.0, 1.0, 1.0, 1.0])

        bi._update_phrase_commitment(
            trigger_kind="beat",
            silence_active=False,
            gate_fail_reason="",
            is_beat_event=False,
        )
        self.assertTrue(bi._phrase_committed)

        bi._recent_flux_values.clear()
        bi._recent_flux_values.extend([0.10, 0.10, 0.10, 0.10])
        bi._update_phrase_commitment(
            trigger_kind="beat",
            silence_active=False,
            gate_fail_reason="",
            is_beat_event=True,
        )
        self.assertFalse(bi._phrase_committed)
        self.assertEqual(bi._phrase_beats_remaining, 0)

    def test_measure_end_renews_on_sustained_flux(self):
        bi = self._bi()
        bi.last_trigger_kind = "creep"
        bi._recent_flux_values.clear()
        bi._recent_flux_values.extend([1.0, 1.0, 1.0, 1.0])

        bi._update_phrase_commitment(
            trigger_kind="beat",
            silence_active=False,
            gate_fail_reason="",
            is_beat_event=False,
        )
        self.assertTrue(bi._phrase_committed)

        bi._phrase_beats_remaining = 1
        bi._recent_flux_values.clear()
        bi._recent_flux_values.extend([0.80, 0.80, 0.80, 0.80])

        bi._update_phrase_commitment(
            trigger_kind="beat",
            silence_active=False,
            gate_fail_reason="",
            is_beat_event=True,
        )
        self.assertTrue(bi._phrase_committed)
        self.assertEqual(bi._phrase_beats_remaining, bi._phrase_measure_beats)
        self.assertAlmostEqual(bi._phrase_flux_baseline, 0.8, places=3)


# ── Gate-state snapshot contract ───────────────────────────────────────────


class TestSnapshotGateState(Phase3Mixin, unittest.TestCase):
    def test_snapshot_gate_state_has_expected_keys_and_types(self):
        bi = self._bi()
        bi.build_decision(self._event(is_downbeat=True), dt=1 / 60, silence_override=False)
        gs = bi.snapshot_gate_state()

        expected_types = {
            "gs_sub_bass": float,
            "gs_low_mid": float,
            "gs_mid": float,
            "gs_high": float,
            "gs_flux_mean": float,
            "gs_flux_std": float,
            "gs_flux_delta": float,
            "gs_low_band_mean": float,
            "gs_mid_band_mean": float,
            "gs_high_band_mean": float,
            "gs_mid_bass_mean": float,
            "gs_rms_envelope_db": float,
            "gs_energy_fullness": float,
            "gs_silence_active": int,
            "gs_silence_fade": float,
            "gs_consecutive_silent": int,
            "gs_stroke_ready": int,
            "gs_stroke_ready_reason": str,
            "gs_phrase_committed": int,
            "gs_phrase_beats_remaining": int,
            "gs_journey_active": int,
            "gs_journey_elapsed_s": float,
            "gs_journey_duration_s": float,
            "gs_last_trigger_kind": str,
            "gs_active_interval_beats": int,
            "gs_stabilized_bpm": float,
            "gs_tempo_unlock_hold": int,
            "gs_time_since_last_beat_s": float,
            "gs_fill_ema_downbeat": float,
            "gs_fill_ema_beat": float,
            "gs_fill_ema_syncopation": float,
            "gs_fill_offset_downbeat": float,
            "gs_fill_offset_beat": float,
            "gs_fill_offset_syncopation": float,
        }

        self.assertEqual(set(gs.keys()), set(expected_types.keys()))
        for key, expected_type in expected_types.items():
            self.assertIn(key, gs)
            self.assertIsInstance(gs[key], expected_type)


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
