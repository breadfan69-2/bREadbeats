from __future__ import annotations

import unittest

from pmv_axis_converter import AxisConfig, convert_to_2d
from pmv_funscript_io import FunscriptAction


class TestPmvAxisConverter(unittest.TestCase):
    def test_circular_center_at_rest(self):
        actions = [FunscriptAction(0, 50)]
        result = convert_to_2d(actions, AxisConfig(enabled_axes={"alpha", "beta"}, direction_flip_probability=0.0), 1000)

        self.assertIn("alpha", result.axes)
        self.assertIn("beta", result.axes)

        alpha = result.axes["alpha"][0].pos
        beta = result.axes["beta"][0].pos
        self.assertTrue(45 <= alpha <= 55)
        self.assertTrue(45 <= beta <= 55)

    def test_circular_spread_depends_on_stroke_span_not_timing(self):
        cfg = AxisConfig(enabled_axes={"alpha", "beta"}, direction_flip_probability=0.0)

        fast = [
            FunscriptAction(0, 0),
            FunscriptAction(100, 100),
            FunscriptAction(200, 0),
            FunscriptAction(300, 100),
        ]
        slow = [
            FunscriptAction(0, 0),
            FunscriptAction(1000, 100),
            FunscriptAction(2000, 0),
            FunscriptAction(3000, 100),
        ]

        fast_r = convert_to_2d(fast, cfg, 4000)
        slow_r = convert_to_2d(slow, cfg, 4000)

        fast_alpha = [a.pos for a in fast_r.axes["alpha"]]
        slow_alpha = [a.pos for a in slow_r.axes["alpha"]]

        fast_spread = max(fast_alpha) - min(fast_alpha)
        slow_spread = max(slow_alpha) - min(slow_alpha)
        self.assertLessEqual(abs(fast_spread - slow_spread), 5)

    def test_e_curve_bell_peaks_midway(self):
        actions = [FunscriptAction(i * 100, int(i / 10 * 100)) for i in range(11)]
        cfg = AxisConfig(enabled_axes={"e1"}, e1_curve="bell")
        result = convert_to_2d(actions, cfg, 1000)

        e1 = result.axes["e1"]
        midpoint = [a for a in e1 if a.at == 500][0]
        self.assertGreater(midpoint.pos, 80)

    def test_enabled_axes_selection(self):
        actions = [FunscriptAction(0, 0), FunscriptAction(500, 100)]
        cfg = AxisConfig(enabled_axes={"main", "alpha", "volume"})
        result = convert_to_2d(actions, cfg, 500)

        self.assertIn("main", result.axes)
        self.assertIn("alpha", result.axes)
        self.assertIn("volume", result.axes)
        self.assertNotIn("beta", result.axes)


if __name__ == "__main__":
    unittest.main()
