import json
import tempfile
from dataclasses import asdict
from pathlib import Path
import unittest

from config import (
    Config,
    CURRENT_CONFIG_VERSION,
    apply_dict_to_dataclass,
    migrate_config,
)
import config_persistence as config_persistence_module
from config_facade import (
    load_config,
)


class TestConfigMigration(unittest.TestCase):
    def test_missing_version_sets_defaults_and_bumps(self):
        cfg = Config()
        data = {
            # version intentionally omitted to simulate legacy file
            "stroke": {},
            "device_limits": {},
        }

        apply_dict_to_dataclass(cfg, data)
        migrate_config(cfg, data.get("version"))

        self.assertEqual(cfg.version, CURRENT_CONFIG_VERSION)
        self.assertEqual(cfg.stroke.noise_burst_magnitude, 1.0)
        self.assertTrue(cfg.device_limits.p0_c0_sending_enabled)
        self.assertFalse(cfg.device_limits.dont_show_on_startup)
        self.assertFalse(cfg.device_limits.dry_run)

    def test_none_values_are_sanitized(self):
        cfg = Config()
        data = {
            "version": 0,
            "stroke": {"noise_burst_magnitude": None},
            "device_limits": {
                "p0_c0_sending_enabled": None,
                "dont_show_on_startup": None,
                "prompted": None,
            },
        }

        apply_dict_to_dataclass(cfg, data)
        migrate_config(cfg, data.get("version"))

        self.assertEqual(cfg.version, CURRENT_CONFIG_VERSION)
        self.assertEqual(cfg.stroke.noise_burst_magnitude, 1.0)
        self.assertTrue(cfg.device_limits.p0_c0_sending_enabled)
        self.assertFalse(cfg.device_limits.dont_show_on_startup)
        self.assertFalse(cfg.device_limits.prompted)
        self.assertFalse(cfg.device_limits.dry_run)

    def test_preserves_custom_values(self):
        cfg = Config()
        data = {
            "version": 1,
            "stroke": {"noise_burst_magnitude": 2.5},
            "device_limits": {
                "p0_c0_sending_enabled": False,
                "dont_show_on_startup": True,
                "prompted": True,
                "dry_run": True,
            },
        }

        apply_dict_to_dataclass(cfg, data)
        migrate_config(cfg, data.get("version"))

        self.assertEqual(cfg.version, CURRENT_CONFIG_VERSION)
        self.assertEqual(cfg.stroke.noise_burst_magnitude, 2.5)
        self.assertFalse(cfg.device_limits.p0_c0_sending_enabled)
        self.assertTrue(cfg.device_limits.dont_show_on_startup)
        self.assertTrue(cfg.device_limits.prompted)
        self.assertTrue(cfg.device_limits.dry_run)

    def test_load_config_auto_saves_bumped_version(self):
        # Set up a temp config file with an old version
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        legacy_cfg = Config()
        legacy_cfg.version = 0
        legacy_data = asdict(legacy_cfg)
        legacy_data["stroke"]["noise_burst_magnitude"] = None  # force migration path
        with open(tmp.name, "w") as f:
            json.dump(legacy_data, f)

        # Monkeypatch config_persistence get_config_file/save_config to use temp file
        orig_get_config_file = config_persistence_module.get_config_file
        orig_save_config = config_persistence_module.save_config
        calls = {}

        def fake_get_config_file():
            return Path(tmp.name)

        def fake_save_config(cfg):
            calls["saved"] = True
            with open(tmp.name, "w") as f:
                json.dump(asdict(cfg), f)
            return True

        try:
            config_persistence_module.get_config_file = fake_get_config_file  # type: ignore
            config_persistence_module.save_config = fake_save_config  # type: ignore

            cfg = load_config()

            self.assertEqual(cfg.version, CURRENT_CONFIG_VERSION)
            self.assertTrue(calls.get("saved"))

            with open(tmp.name, "r") as f:
                persisted = json.load(f)
            self.assertEqual(persisted.get("version"), CURRENT_CONFIG_VERSION)
        finally:
            # Restore originals
            config_persistence_module.get_config_file = orig_get_config_file  # type: ignore
            config_persistence_module.save_config = orig_save_config  # type: ignore


if __name__ == "__main__":
    unittest.main()
