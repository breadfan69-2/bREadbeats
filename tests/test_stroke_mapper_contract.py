import inspect
import unittest

from config import Config, StrokeMode
from stroke_mapper import StrokeMapper


class TestStrokeMapperContract(unittest.TestCase):
    def test_constructor_and_entrypoint_contract(self):
        sig = inspect.signature(StrokeMapper.__init__)
        params = list(sig.parameters.keys())

        self.assertIn("config", params)
        self.assertIn("send_callback", params)
        self.assertIn("get_volume", params)
        self.assertIn("audio_engine", params)

        mapper = StrokeMapper(Config())
        self.assertTrue(hasattr(mapper, "process_beat"))
        self.assertTrue(callable(getattr(mapper, "process_beat")))

    def test_non_mode3_anchor_phase_uses_bottom_park(self):
        mapper = StrokeMapper(Config())
        park_phase = mapper._get_park_phase()

        simple_anchor = mapper._get_anchor_phase_for_mode(StrokeMode.SIMPLE_CIRCLE, fallback_phase=0.123)
        spiral_anchor = mapper._get_anchor_phase_for_mode(StrokeMode.SPIRAL, fallback_phase=0.456)

        self.assertAlmostEqual(simple_anchor, park_phase, places=6)
        self.assertAlmostEqual(spiral_anchor, park_phase, places=6)


if __name__ == "__main__":
    unittest.main()
