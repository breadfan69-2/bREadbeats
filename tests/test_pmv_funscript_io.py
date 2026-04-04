from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pmv_funscript_io import (
    FunscriptAction,
    FunscriptMetadata,
    actions_to_dict_list,
    dict_list_to_actions,
    read_csv,
    read_funscript,
    write_csv,
    write_funscript,
)


class TestPmvFunscriptIO(unittest.TestCase):
    def test_action_dict_conversion_round_trip(self):
        original = [FunscriptAction(0, 50), FunscriptAction(500, 95), FunscriptAction(1000, 5)]
        as_dicts = actions_to_dict_list(original)
        recovered = dict_list_to_actions(as_dicts)
        self.assertEqual([(a.at, a.pos) for a in recovered], [(0, 50), (500, 95), (1000, 5)])

    def test_funscript_round_trip(self):
        actions = [FunscriptAction(0, 50), FunscriptAction(500, 95), FunscriptAction(1000, 5)]
        metadata = FunscriptMetadata(
            title="Test",
            duration=1000,
            parameters={"beat_sensitivity": 0.5, "axis_algorithm": "circular"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.funscript"
            write_funscript(path, actions, metadata)

            read_actions, read_meta = read_funscript(path)
            self.assertEqual(len(read_actions), 3)
            self.assertEqual(read_actions[0].at, 0)
            self.assertEqual(read_actions[0].pos, 50)
            self.assertEqual(read_meta.title, "Test")
            self.assertEqual(read_meta.parameters.get("axis_algorithm"), "circular")

            # Deterministic writer check for our own format.
            first_payload = json.loads(path.read_text(encoding="utf-8"))
            write_funscript(path, read_actions, read_meta)
            second_payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(first_payload, second_payload)

    def test_csv_round_trip(self):
        actions = [FunscriptAction(0, 50), FunscriptAction(250, 75), FunscriptAction(500, 25)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            write_csv(path, actions)
            read_back = read_csv(path)
            self.assertEqual([(a.at, a.pos) for a in read_back], [(0, 50), (250, 75), (500, 25)])

    def test_read_existing_script(self):
        script_path = Path("scripts") / "CH-Tranquilizer.beta.funscript"
        actions, _ = read_funscript(script_path)
        self.assertGreater(len(actions), 0)
        self.assertTrue(all(0 <= a.pos <= 100 for a in actions))


if __name__ == "__main__":
    unittest.main()
