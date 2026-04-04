from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from pmv_generator import PMVGeneratorWindow


class TestPmvPresets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_default_preset_files_exist(self):
        win = PMVGeneratorWindow()
        defaults_dir, _ = win._preset_dirs()
        expected = {
            "balanced.json",
            "high_energy.json",
            "chill.json",
            "beat_focused.json",
            "ml_driven.json",
        }
        existing = {p.name for p in defaults_dir.glob("*.json")}
        self.assertTrue(expected.issubset(existing))

    def test_catalog_populates_combo(self):
        win = PMVGeneratorWindow()
        self.assertGreaterEqual(win.preset_combo.count(), 5)
        labels = [win.preset_combo.itemText(i) for i in range(win.preset_combo.count())]
        self.assertTrue(any("balanced" in label.lower() for label in labels))

    def test_save_and_load_preset_helpers(self):
        win = PMVGeneratorWindow()
        win.controls.sample_rate_spin.setValue(44100)

        with tempfile.TemporaryDirectory() as tmp:
            preset_path = Path(tmp) / "custom_roundtrip.json"
            self.assertTrue(win._save_preset_to_path(str(preset_path), preset_name="Custom Roundtrip", show_errors=False))
            self.assertTrue(preset_path.exists())

            payload = json.loads(preset_path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("name"), "Custom Roundtrip")
            self.assertEqual(payload.get("pmv_preset_version"), 1)

            win.controls.sample_rate_spin.setValue(48000)
            self.assertTrue(win._load_preset_from_path(str(preset_path), show_errors=False))
            self.assertEqual(win.controls.sample_rate_spin.value(), 44100)


if __name__ == "__main__":
    unittest.main()
