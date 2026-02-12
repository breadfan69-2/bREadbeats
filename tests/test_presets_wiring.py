import json
import tempfile
import unittest
from pathlib import Path

from presets_wiring import get_presets_file_path, load_presets_data, save_presets_data


class TestPresetsWiring(unittest.TestCase):
    def test_get_presets_file_path_source(self):
        path = get_presets_file_path(
            frozen=False,
            executable_path="C:/x/app.exe",
            source_file="C:/work/main.py",
        )
        self.assertEqual(path.as_posix(), "C:/work/presets.json")

    def test_get_presets_file_path_frozen(self):
        path = get_presets_file_path(
            frozen=True,
            executable_path="C:/apps/breadbeats/app.exe",
            source_file="C:/work/main.py",
        )
        self.assertEqual(path.as_posix(), "C:/apps/breadbeats/presets.json")

    def test_save_and_load_presets_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            presets_file = Path(tmpdir) / "presets.json"
            payload = {"0": {"preset_name": "MyPreset", "x": 1}}

            save_presets_data(presets_file, payload)
            loaded = load_presets_data(presets_file, frozen=False, meipass=None)

            self.assertEqual(loaded, payload)

    def test_load_presets_data_copies_factory_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            presets_file = base / "user" / "presets.json"
            presets_file.parent.mkdir(parents=True, exist_ok=True)

            meipass = base / "bundle"
            meipass.mkdir(parents=True, exist_ok=True)
            factory_file = meipass / "presets.json"
            factory_payload = {"1": {"preset_name": "Factory"}}
            with open(factory_file, "w", encoding="utf-8") as f:
                json.dump(factory_payload, f)

            loaded = load_presets_data(presets_file, frozen=True, meipass=str(meipass))
            self.assertEqual(loaded, factory_payload)
            self.assertTrue(presets_file.exists())


if __name__ == "__main__":
    unittest.main()
