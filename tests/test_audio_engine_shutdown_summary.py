import unittest
from unittest import mock

from audio_engine import AudioEngine
from config import Config


class TestAudioEngineShutdownSummary(unittest.TestCase):
    def test_shutdown_summary_logs_ranges(self):
        engine = AudioEngine(Config(), lambda event: None)
        engine._reset_session_stats()
        engine._update_session_stats(raw_rms=0.10, band_energy=0.20, spectral_flux=0.30)
        engine._update_session_stats(raw_rms=0.40, band_energy=0.70, spectral_flux=1.10)

        with mock.patch("audio_engine.log_event") as log_event_mock:
            engine._log_shutdown_summary()

        self.assertTrue(log_event_mock.called)
        _, kwargs = log_event_mock.call_args
        self.assertEqual(kwargs["raw_rms_min"], "0.100000")
        self.assertEqual(kwargs["raw_rms_max"], "0.400000")
        self.assertEqual(kwargs["raw_rms_mean"], "0.250000")
        self.assertEqual(kwargs["raw_rms_span"], "0.300000")
        self.assertEqual(kwargs["band_energy_min"], "0.200000")
        self.assertEqual(kwargs["band_energy_max"], "0.700000")
        self.assertEqual(kwargs["band_energy_mean"], "0.450000")
        self.assertEqual(kwargs["band_energy_span"], "0.500000")
        self.assertEqual(kwargs["flux_min"], "0.3000")
        self.assertEqual(kwargs["flux_max"], "1.1000")
        self.assertEqual(kwargs["flux_mean"], "0.7000")
        self.assertEqual(kwargs["flux_span"], "0.8000")

    def test_shutdown_summary_no_frames_no_log(self):
        engine = AudioEngine(Config(), lambda event: None)
        engine._reset_session_stats()

        with mock.patch("audio_engine.log_event") as log_event_mock:
            engine._log_shutdown_summary()

        log_event_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
