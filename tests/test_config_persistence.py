import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from config import Config
import config_persistence


class TestConfigPersistence(unittest.TestCase):
    def test_get_config_dir_prefers_env_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict("os.environ", {"BREADBEATS_CONFIG_DIR": tmpdir}):
                path = config_persistence.get_config_dir()
        self.assertEqual(path, Path(tmpdir).resolve())

    def test_get_config_dir_uses_project_dir_when_not_frozen(self):
        fake_module_path = Path("C:/tmp/breadbeats_src/config_persistence.py")
        with mock.patch.object(config_persistence, "__file__", str(fake_module_path)):
            with mock.patch.object(config_persistence.sys, "frozen", False, create=True):
                with mock.patch.dict("os.environ", {}, clear=True):
                    with mock.patch("builtins.open", mock.mock_open()):
                        path = config_persistence.get_config_dir()
        self.assertEqual(path, fake_module_path.resolve().parent)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = Path(tmpdir) / "config.json"
            cfg = Config()
            cfg.stroke.flux_threshold = 0.42

            with mock.patch.object(config_persistence, "get_config_file", return_value=cfg_file):
                self.assertTrue(config_persistence.save_config(cfg))
                loaded = config_persistence.load_config()

            self.assertAlmostEqual(loaded.stroke.flux_threshold, 0.42, places=6)

    def test_save_after_legacy_load_omits_deprecated_noise_burst_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = Path(tmpdir) / "config.json"
            legacy_data = {
                "version": 0,
                "stroke": {
                    "noise_burst_enabled": True,
                    "noise_burst_flux_multiplier": 9.0,
                    "noise_burst_magnitude": 3.0,
                    "noise_burst_scale": 0.2,
                },
            }
            cfg_file.write_text(json.dumps(legacy_data), encoding="utf-8")

            with mock.patch.object(config_persistence, "get_config_file", return_value=cfg_file):
                cfg = config_persistence.load_config()
                self.assertTrue(config_persistence.save_config(cfg))

            persisted = json.loads(cfg_file.read_text(encoding="utf-8"))
            stroke = persisted.get("stroke", {})
            self.assertNotIn("noise_burst_enabled", stroke)
            self.assertNotIn("noise_burst_flux_multiplier", stroke)
            self.assertNotIn("noise_burst_magnitude", stroke)
            self.assertNotIn("noise_burst_scale", stroke)

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
            self.assertFalse(hasattr(loaded.stroke, "noise_burst_magnitude"))

    def test_load_legacy_noise_burst_keys_still_works(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = Path(tmpdir) / "config.json"
            legacy_data = {
                "version": 0,
                "stroke": {
                    "noise_burst_enabled": True,
                    "noise_burst_flux_multiplier": 4.5,
                    "noise_burst_magnitude": 2.75,
                    "noise_burst_scale": 0.21,
                },
            }
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump(legacy_data, f)

            with mock.patch.object(config_persistence, "get_config_file", return_value=cfg_file):
                loaded = config_persistence.load_config()

            self.assertFalse(hasattr(loaded.stroke, "noise_burst_enabled"))
            self.assertFalse(hasattr(loaded.stroke, "noise_burst_flux_multiplier"))
            self.assertFalse(hasattr(loaded.stroke, "noise_burst_magnitude"))
            self.assertFalse(hasattr(loaded.stroke, "noise_burst_scale"))

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
