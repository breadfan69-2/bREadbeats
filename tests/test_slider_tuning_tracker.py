import json
import tempfile
import unittest
from pathlib import Path

from slider_tuning_tracker import SliderTuningTracker


class TestSliderTuningTracker(unittest.TestCase):
    def test_record_and_save_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            tracker = SliderTuningTracker(report_dir)

            tracker.record_value("Sensitivity", 0.2)
            tracker.record_value("Sensitivity", 0.3)
            tracker.record_value("Peak Floor [low]", 20.0)
            tracker.save_reports()

            json_path = report_dir / "slider_tuning_report.json"
            csv_path = report_dir / "slider_tuning_report.csv"

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertIn("ranked", payload)
            ranked = payload["ranked"]
            self.assertGreaterEqual(len(ranked), 2)

            by_name = {row["name"]: row for row in ranked}
            self.assertIn("Sensitivity", by_name)
            self.assertEqual(by_name["Sensitivity"]["count"], 2)
            self.assertAlmostEqual(by_name["Sensitivity"]["min_value"], 0.2, places=6)
            self.assertAlmostEqual(by_name["Sensitivity"]["max_value"], 0.3, places=6)

    def test_load_existing_accumulates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            tracker = SliderTuningTracker(report_dir)
            tracker.record_value("Flux", 1.0)
            tracker.save_reports()

            tracker2 = SliderTuningTracker(report_dir)
            tracker2.record_value("Flux", 1.5)
            tracker2.save_reports()

            with open(report_dir / "slider_tuning_report.json", "r", encoding="utf-8") as f:
                payload = json.load(f)

            by_name = {row["name"]: row for row in payload["ranked"]}
            self.assertEqual(by_name["Flux"]["count"], 2)


if __name__ == "__main__":
    unittest.main()
