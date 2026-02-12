import inspect
import unittest

from config import Config
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


if __name__ == "__main__":
    unittest.main()
