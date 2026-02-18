"""
Phase 4 checkpoint tests — Motion Mode Resolver (#9):
  Advisory FULL_STROKE / CREEP_MICRO mode based on RMS envelope
  with 500ms dwell hysteresis.
"""

import time
import unittest
from typing import Any

from audio_engine import BeatEvent
from beat_intelligence import BeatIntelligence
from config import Config


class Phase4Mixin:
    """Shared event factory for Phase 4 tests."""

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
        """Create a BeatIntelligence with tempo_lock_required=False."""
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        for key, val in cfg_overrides.items():
            if hasattr(cfg.stroke, key):
                setattr(cfg.stroke, key, val)
            elif hasattr(cfg.beat, key):
                setattr(cfg.beat, key, val)
        return BeatIntelligence(cfg)


class TestMotionModeInitState(Phase4Mixin, unittest.TestCase):
    """Verify initial state of the motion mode resolver."""

    def test_starts_in_creep_micro(self):
        bi = self._bi()
        self.assertEqual(bi._motion_mode, "creep_micro")

    def test_switch_time_starts_zero(self):
        bi = self._bi()
        self.assertEqual(bi._mode_switch_time, 0.0)

    def test_decision_includes_motion_mode(self):
        bi = self._bi()
        ev = self._event(is_downbeat=True)
        decision = bi.build_decision(ev, dt=1 / 60)
        self.assertIn(decision.motion_mode, ("creep_micro", "full_stroke"))

    def test_default_decision_mode_is_creep(self):
        bi = self._bi()
        ev = self._event()
        decision = bi.build_decision(ev, dt=1 / 60)
        self.assertEqual(decision.motion_mode, "creep_micro")


class TestModeTransitionToFullStroke(Phase4Mixin, unittest.TestCase):
    """Test transition from creep_micro to full_stroke."""

    def test_no_switch_below_threshold(self):
        """RMS below amplitude_gate_high should stay creep_micro."""
        bi = self._bi(amplitude_gate_high=0.08, amplitude_gate_low=0.04)
        bi.rms_envelope = 0.06  # between low and high
        mode = bi._update_motion_mode(now=100.0)
        self.assertEqual(mode, "creep_micro")

    def test_dwell_required_for_switch_up(self):
        """Reaching threshold starts dwell timer, doesn't switch immediately."""
        bi = self._bi(amplitude_gate_high=0.08)
        bi.rms_envelope = 0.10  # above threshold

        # First call: starts dwell timer
        mode = bi._update_motion_mode(now=100.0)
        self.assertEqual(mode, "creep_micro")
        self.assertGreater(bi._mode_switch_time, 0.0)

        # Just under 500ms: still creep
        mode = bi._update_motion_mode(now=100.4)
        self.assertEqual(mode, "creep_micro")

    def test_switch_up_after_dwell(self):
        """After 500ms dwell with RMS above threshold, switch to full_stroke."""
        bi = self._bi(amplitude_gate_high=0.08)
        bi.rms_envelope = 0.10

        bi._update_motion_mode(now=100.0)  # start dwell
        mode = bi._update_motion_mode(now=100.6)  # 600ms later
        self.assertEqual(mode, "full_stroke")
        self.assertEqual(bi._mode_switch_time, 0.0)  # timer reset

    def test_dwell_reset_on_drop(self):
        """If RMS drops below threshold during dwell, timer resets."""
        bi = self._bi(amplitude_gate_high=0.08)
        bi.rms_envelope = 0.10

        bi._update_motion_mode(now=100.0)  # start dwell
        self.assertGreater(bi._mode_switch_time, 0.0)

        # RMS drops
        bi.rms_envelope = 0.05
        bi._update_motion_mode(now=100.3)
        self.assertEqual(bi._mode_switch_time, 0.0)  # timer cleared
        self.assertEqual(bi._motion_mode, "creep_micro")


class TestModeTransitionToCreepMicro(Phase4Mixin, unittest.TestCase):
    """Test transition from full_stroke back to creep_micro."""

    def _set_full_stroke(self, bi: BeatIntelligence):
        bi._motion_mode = "full_stroke"
        bi._mode_switch_time = 0.0

    def test_no_switch_above_low_threshold(self):
        """RMS above amplitude_gate_low should stay full_stroke."""
        bi = self._bi(amplitude_gate_low=0.04)
        self._set_full_stroke(bi)
        bi.rms_envelope = 0.06

        mode = bi._update_motion_mode(now=200.0)
        self.assertEqual(mode, "full_stroke")

    def test_dwell_required_for_switch_down(self):
        """Dropping below threshold starts dwell, doesn't switch immediately."""
        bi = self._bi(amplitude_gate_low=0.04)
        self._set_full_stroke(bi)
        bi.rms_envelope = 0.02  # below low threshold

        mode = bi._update_motion_mode(now=200.0)
        self.assertEqual(mode, "full_stroke")  # still in dwell
        self.assertGreater(bi._mode_switch_time, 0.0)

    def test_switch_down_after_dwell(self):
        """After 500ms dwell below threshold, switch to creep_micro."""
        bi = self._bi(amplitude_gate_low=0.04)
        self._set_full_stroke(bi)
        bi.rms_envelope = 0.02

        bi._update_motion_mode(now=200.0)  # start dwell
        mode = bi._update_motion_mode(now=200.6)  # 600ms later
        self.assertEqual(mode, "creep_micro")

    def test_dwell_reset_on_recovery(self):
        """If RMS recovers above low threshold during dwell, timer resets."""
        bi = self._bi(amplitude_gate_low=0.04)
        self._set_full_stroke(bi)
        bi.rms_envelope = 0.02

        bi._update_motion_mode(now=200.0)  # start dwell
        self.assertGreater(bi._mode_switch_time, 0.0)

        bi.rms_envelope = 0.06  # recover
        bi._update_motion_mode(now=200.3)
        self.assertEqual(bi._mode_switch_time, 0.0)
        self.assertEqual(bi._motion_mode, "full_stroke")


class TestDwellBias(Phase4Mixin, unittest.TestCase):
    """Test full_stroke_dwell_bias shifts thresholds."""

    def test_positive_bias_raises_up_threshold(self):
        """Positive bias makes it harder to switch to full_stroke."""
        bi = self._bi(amplitude_gate_high=0.08, full_stroke_dwell_bias=0.02)
        bi.rms_envelope = 0.09  # above 0.08 but below 0.08+0.02=0.10

        bi._update_motion_mode(now=300.0)
        # Should NOT start dwell — threshold is 0.10 with bias
        self.assertEqual(bi._mode_switch_time, 0.0)
        self.assertEqual(bi._motion_mode, "creep_micro")

    def test_positive_bias_lowers_down_threshold(self):
        """Positive bias makes it harder to switch back to creep_micro."""
        bi = self._bi(amplitude_gate_low=0.04, full_stroke_dwell_bias=0.02)
        bi._motion_mode = "full_stroke"
        bi.rms_envelope = 0.03  # below 0.04 but above 0.04-0.02=0.02

        bi._update_motion_mode(now=300.0)
        # Should NOT start dwell — threshold is 0.02 with bias
        self.assertEqual(bi._mode_switch_time, 0.0)
        self.assertEqual(bi._motion_mode, "full_stroke")


class TestModeInBuildDecision(Phase4Mixin, unittest.TestCase):
    """Verify motion mode is wired through build_decision to BeatDecision."""

    def test_full_stroke_propagates_to_decision(self):
        """Once mode is full_stroke, decisions report it."""
        bi = self._bi(amplitude_gate_high=0.08)
        bi._motion_mode = "full_stroke"
        bi._mode_switch_time = 0.0
        bi.rms_envelope = 0.10  # keep above low threshold

        ev = self._event(raw_rms=0.10)
        decision = bi.build_decision(ev, dt=1 / 60)
        self.assertEqual(decision.motion_mode, "full_stroke")

    def test_mode_advisory_does_not_affect_trigger(self):
        """Motion mode is advisory only — creep trigger still fires normally."""
        bi = self._bi()
        bi._motion_mode = "full_stroke"

        ev = self._event(is_beat=False, is_downbeat=False, raw_rms=0.10)
        decision = bi.build_decision(ev, dt=1 / 60)
        # No beat event → creep trigger, regardless of motion mode
        self.assertEqual(decision.trigger_kind, "creep")
        self.assertEqual(decision.motion_mode, "full_stroke")

    def test_rms_envelope_updates_affect_mode(self):
        """RMS envelope updates from event drive mode transitions over time."""
        bi = self._bi(amplitude_gate_high=0.08, amplitude_gate_low=0.04)
        t = 1000.0

        # Feed high-RMS events for >500ms
        for i in range(40):  # ~667ms at 60fps
            ev = self._event(raw_rms=0.15, monotonic_timestamp=t)
            bi.build_decision(ev, dt=1 / 60)
            t += 1 / 60

        # After sustained high RMS, should switch to full_stroke
        self.assertEqual(bi._motion_mode, "full_stroke")


class TestModeEdgeCases(Phase4Mixin, unittest.TestCase):
    """Edge cases for motion mode resolver."""

    def test_zero_rms_stays_creep(self):
        bi = self._bi()
        bi.rms_envelope = 0.0
        mode = bi._update_motion_mode(now=0.0)
        self.assertEqual(mode, "creep_micro")

    def test_exact_threshold_no_switch(self):
        """RMS exactly at gate_high should NOT trigger switch (needs >= not >)."""
        bi = self._bi(amplitude_gate_high=0.08, full_stroke_dwell_bias=0.0)
        bi.rms_envelope = 0.08
        bi._update_motion_mode(now=100.0)
        # At exact threshold: starts dwell (>=)
        self.assertGreater(bi._mode_switch_time, 0.0)

    def test_rapid_oscillation_no_switch(self):
        """Rapidly alternating RMS above/below should never complete dwell."""
        bi = self._bi(amplitude_gate_high=0.08)
        for i in range(20):
            bi.rms_envelope = 0.10 if i % 2 == 0 else 0.05
            bi._update_motion_mode(now=100.0 + i * 0.1)
        # Should never switch because dwell keeps resetting
        self.assertEqual(bi._motion_mode, "creep_micro")

    def test_round_trip_full_then_back(self):
        """Full round trip: creep → full_stroke → creep_micro."""
        bi = self._bi(amplitude_gate_high=0.08, amplitude_gate_low=0.04)

        # Switch up
        bi.rms_envelope = 0.10
        bi._update_motion_mode(now=100.0)
        bi._update_motion_mode(now=100.6)
        self.assertEqual(bi._motion_mode, "full_stroke")

        # Switch down
        bi.rms_envelope = 0.02
        bi._update_motion_mode(now=200.0)
        bi._update_motion_mode(now=200.6)
        self.assertEqual(bi._motion_mode, "creep_micro")


if __name__ == "__main__":
    unittest.main()
