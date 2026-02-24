import unittest

from audio_modules.tempo_tracker import (
    TempoTracker,
    smooth_acf_bpm_with_jump_gating,
    step_metronome_phase,
)


class TestStepMetronomePhase(unittest.TestCase):
    def test_nonpositive_inputs_keep_phase(self):
        phase, crossings = step_metronome_phase(3.2, 0.0, 0.1)
        self.assertAlmostEqual(phase, 3.2)
        self.assertEqual(crossings, 0)

        phase, crossings = step_metronome_phase(1.0, 120.0, 0.0)
        self.assertAlmostEqual(phase, 1.0)
        self.assertEqual(crossings, 0)

    def test_advances_without_crossing(self):
        phase, crossings = step_metronome_phase(2.10, 120.0, 0.20)
        self.assertAlmostEqual(phase, 2.50, places=6)
        self.assertEqual(crossings, 0)

    def test_counts_integer_crossings(self):
        phase, crossings = step_metronome_phase(2.80, 120.0, 0.20)
        self.assertAlmostEqual(phase, 3.20, places=6)
        self.assertEqual(crossings, 1)

        phase, crossings = step_metronome_phase(2.10, 180.0, 0.70)
        self.assertAlmostEqual(phase, 4.20, places=6)
        self.assertEqual(crossings, 2)


class TestSmoothAcfBpmWithJumpGating(unittest.TestCase):
    def test_initial_lock(self):
        result = smooth_acf_bpm_with_jump_gating(0.0, 120.0, 0.40)
        self.assertAlmostEqual(result.smoothed_bpm, 120.0)
        self.assertTrue(result.updated)
        self.assertEqual(result.decision_tag, "initial")

    def test_small_ratio_smooths(self):
        result = smooth_acf_bpm_with_jump_gating(120.0, 126.0, 0.20)
        self.assertAlmostEqual(result.smoothed_bpm, 120.9, places=6)
        self.assertTrue(result.updated)
        self.assertEqual(result.decision_tag, "smooth")

    def test_confident_jump_accepted_without_target(self):
        result = smooth_acf_bpm_with_jump_gating(120.0, 160.0, 0.40)
        self.assertAlmostEqual(result.smoothed_bpm, 160.0)
        self.assertTrue(result.updated)
        self.assertEqual(result.decision_tag, "jump")

    def test_octave_like_jump_rejected_when_farther_from_target(self):
        result = smooth_acf_bpm_with_jump_gating(
            120.0,
            180.0,
            0.35,
            target_bpm_hint=110.0,
        )
        self.assertAlmostEqual(result.smoothed_bpm, 120.0)
        self.assertFalse(result.updated)
        self.assertEqual(result.decision_tag, "jump-target-rejected")

    def test_octave_like_jump_accepted_when_closer_to_target(self):
        result = smooth_acf_bpm_with_jump_gating(
            120.0,
            180.0,
            0.35,
            target_bpm_hint=190.0,
        )
        self.assertAlmostEqual(result.smoothed_bpm, 180.0)
        self.assertTrue(result.updated)
        self.assertEqual(result.decision_tag, "jump-target-validated")

    def test_low_confidence_outlier_ignored(self):
        result = smooth_acf_bpm_with_jump_gating(120.0, 170.0, 0.10)
        self.assertAlmostEqual(result.smoothed_bpm, 120.0)
        self.assertFalse(result.updated)
        self.assertEqual(result.decision_tag, "ignored")


class TestTempoTrackerAdapters(unittest.TestCase):
    def test_phase_step_adapter(self):
        tracker = TempoTracker()
        phase, crossings = tracker.step_metronome_phase(2.80, 120.0, 0.20)
        self.assertAlmostEqual(phase, 3.20, places=6)
        self.assertEqual(crossings, 1)

    def test_smoothing_adapter(self):
        tracker = TempoTracker()
        result = tracker.smooth_acf_bpm_with_jump_gating(120.0, 126.0, 0.20)
        self.assertAlmostEqual(result.smoothed_bpm, 120.9, places=6)
        self.assertEqual(result.decision_tag, "smooth")


if __name__ == "__main__":
    unittest.main()