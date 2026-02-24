import unittest

from audio_modules.telemetry_tuning import TelemetryTuning, TriggerTelemetry


class TestTelemetryTuning(unittest.TestCase):
    def test_empty_summary_defaults(self):
        telemetry = TelemetryTuning()
        summary = telemetry.summary()
        self.assertEqual(summary["shadow_samples"], 0)
        self.assertEqual(summary["shadow_agreement_count"], 0)
        self.assertEqual(summary["shadow_disagreement_count"], 0)
        self.assertAlmostEqual(float(summary["shadow_agreement_pct"]), 0.0)

    def test_records_and_counts_agreement(self):
        telemetry = TelemetryTuning()
        telemetry.record(
            TriggerTelemetry(
                legacy_fire=True,
                new_fire=False,
                frontend_ms=1.5,
                tempo_ms=0.8,
                detector_ms=1.1,
                sidecar_ms=0.3,
                smoothing_tag="jump",
            )
        )
        telemetry.record(
            TriggerTelemetry(
                legacy_fire=False,
                new_fire=False,
                frontend_ms=2.5,
                tempo_ms=1.2,
                detector_ms=1.9,
                sidecar_ms=0.7,
                smoothing_tag="smooth",
            )
        )

        summary = telemetry.summary()
        self.assertEqual(summary["shadow_samples"], 2)
        self.assertEqual(summary["shadow_legacy_fire_count"], 1)
        self.assertEqual(summary["shadow_new_fire_count"], 0)
        self.assertEqual(summary["shadow_agreement_count"], 1)
        self.assertEqual(summary["shadow_disagreement_count"], 1)
        self.assertAlmostEqual(float(summary["shadow_agreement_pct"]), 50.0)
        self.assertEqual(summary["shadow_last_smoothing_tag"], "smooth")
        self.assertAlmostEqual(float(summary["shadow_frontend_ms_mean"]), 2.0)
        self.assertAlmostEqual(float(summary["shadow_tempo_ms_mean"]), 1.0)
        self.assertAlmostEqual(float(summary["shadow_detector_ms_mean"]), 1.5)
        self.assertAlmostEqual(float(summary["shadow_sidecar_ms_mean"]), 0.5)

    def test_reset_clears_counts(self):
        telemetry = TelemetryTuning()
        telemetry.record(TriggerTelemetry(legacy_fire=True, new_fire=True, smoothing_tag="initial"))
        telemetry.reset()

        summary = telemetry.summary()
        self.assertEqual(summary["shadow_samples"], 0)
        self.assertEqual(summary["shadow_last_smoothing_tag"], "")


if __name__ == "__main__":
    unittest.main()