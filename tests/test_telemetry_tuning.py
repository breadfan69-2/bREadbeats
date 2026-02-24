import unittest

from audio_modules.telemetry_tuning import TelemetryTuning, TriggerTelemetry


class TestTelemetryTuning(unittest.TestCase):
    def test_empty_summary_defaults(self):
        telemetry = TelemetryTuning()
        summary = telemetry.summary()
        self.assertEqual(summary["shadow_samples"], 0)
        self.assertEqual(summary["shadow_agreement_count"], 0)
        self.assertEqual(summary["shadow_disagreement_count"], 0)
        self.assertAlmostEqual(summary["shadow_agreement_pct"], 0.0)

    def test_records_and_counts_agreement(self):
        telemetry = TelemetryTuning()
        telemetry.record(TriggerTelemetry(legacy_fire=True, new_fire=False, smoothing_tag="jump"))
        telemetry.record(TriggerTelemetry(legacy_fire=False, new_fire=False, smoothing_tag="smooth"))

        summary = telemetry.summary()
        self.assertEqual(summary["shadow_samples"], 2)
        self.assertEqual(summary["shadow_legacy_fire_count"], 1)
        self.assertEqual(summary["shadow_new_fire_count"], 0)
        self.assertEqual(summary["shadow_agreement_count"], 1)
        self.assertEqual(summary["shadow_disagreement_count"], 1)
        self.assertAlmostEqual(summary["shadow_agreement_pct"], 50.0)
        self.assertEqual(summary["shadow_last_smoothing_tag"], "smooth")

    def test_reset_clears_counts(self):
        telemetry = TelemetryTuning()
        telemetry.record(TriggerTelemetry(legacy_fire=True, new_fire=True, smoothing_tag="initial"))
        telemetry.reset()

        summary = telemetry.summary()
        self.assertEqual(summary["shadow_samples"], 0)
        self.assertEqual(summary["shadow_last_smoothing_tag"], "")


if __name__ == "__main__":
    unittest.main()