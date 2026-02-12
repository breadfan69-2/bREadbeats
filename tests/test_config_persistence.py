import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from config import Config
import config_persistence


class TestConfigPersistence(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = Path(tmpdir) / "config.json"
            cfg = Config()
            cfg.stroke.phase_advance = 0.42

            with mock.patch.object(config_persistence, "get_config_file", return_value=cfg_file):
                self.assertTrue(config_persistence.save_config(cfg))
                loaded = config_persistence.load_config()

            self.assertAlmostEqual(loaded.stroke.phase_advance, 0.42, places=6)

    def test_load_default_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = Path(tmpdir) / "config.json"
            with mock.patch.object(config_persistence, "get_config_file", return_value=cfg_file):
                loaded = config_persistence.load_config()
            self.assertIsInstance(loaded, Config)

    def test_load_migrates_and_autosaves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = Path(tmpdir) / "config.json"
            legacy = Config()
            legacy.version = 0
            legacy_data = asdict(legacy)
            legacy_data["stroke"]["noise_burst_magnitude"] = None
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump(legacy_data, f)

            with mock.patch.object(config_persistence, "get_config_file", return_value=cfg_file):
                loaded = config_persistence.load_config()

            self.assertEqual(loaded.version, 1)
            self.assertEqual(loaded.stroke.noise_burst_magnitude, 1.0)

    def test_load_invalid_json_returns_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = Path(tmpdir) / "config.json"
            with open(cfg_file, "w", encoding="utf-8") as f:
                f.write("{invalid json")

            with mock.patch.object(config_persistence, "get_config_file", return_value=cfg_file):
                loaded = config_persistence.load_config()

            self.assertIsInstance(loaded, Config)

    def test_save_failure_returns_false(self):
        cfg = Config()

        with mock.patch.object(config_persistence, "get_config_file", side_effect=OSError("boom")):
            self.assertFalse(config_persistence.save_config(cfg))


if __name__ == "__main__":
    unittest.main()
