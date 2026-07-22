import unittest

from PyQt6.QtWidgets import QApplication

from command_wiring import compute_live_fourphase_levels
from widgets import PositionCanvas


class TestPositionCanvas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_defaults_to_threephase_circle_view(self):
        canvas = PositionCanvas(get_rotation=lambda: 0)

        self.assertEqual(canvas._live_tcode_mode, "threephase")
        self.assertTrue(canvas._circle_outline.isVisible())
        self.assertFalse(canvas._fourphase_fill_bars.isVisible())

    def test_fourphase_mode_shows_bars_and_matches_live_math(self):
        canvas = PositionCanvas(get_rotation=lambda: 0)
        canvas.set_live_tcode_mode("fourphase")
        canvas.update_position(0.25, -0.5)
        expected = compute_live_fourphase_levels(0.25, -0.5, 0.0)

        self.assertEqual(canvas._live_tcode_mode, "fourphase")
        self.assertFalse(canvas._circle_outline.isVisible())
        self.assertTrue(canvas._fourphase_fill_bars.isVisible())
        self.assertEqual(canvas._last_fourphase_levels, expected)

    def test_fourphase_mode_respects_selected_motion_model(self):
        canvas = PositionCanvas(get_rotation=lambda: 0)
        canvas.set_live_tcode_mode("fourphase")
        canvas.set_live_fourphase_model("classic")
        canvas.update_position(0.25, -0.5)
        expected = compute_live_fourphase_levels(0.25, -0.5, 0.0, model="classic")

        self.assertEqual(canvas._last_fourphase_levels, expected)

    def test_fourphase_mode_respects_selected_layout_model(self):
        canvas = PositionCanvas(get_rotation=lambda: 0)
        canvas.set_live_tcode_mode("fourphase")
        canvas.set_live_fourphase_layout_model("Pair At Top")
        canvas.update_position(0.25, -0.5)
        expected = compute_live_fourphase_levels(
            0.25,
            -0.5,
            0.0,
            model="tetra3d",
            layout_model="Pair At Top",
        )

        self.assertEqual(canvas._last_fourphase_levels, expected)


if __name__ == "__main__":
    unittest.main()