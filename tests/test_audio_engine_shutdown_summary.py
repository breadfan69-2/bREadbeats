import unittest
import json
import tempfile
from unittest import mock
from pathlib import Path

from audio_engine import AudioEngine
from config import Config


class TestAudioEngineShutdownSummary(unittest.TestCase):
    def test_shutdown_summary_logs_ranges(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AudioEngine(Config(), lambda event: None, report_dir=Path(tmpdir))
            engine._reset_session_stats()
            engine._update_session_stats(raw_rms=0.10, band_energy=0.20, spectral_flux=0.30, peak_level=0.20, sample_time=1.0)
            engine._update_session_stats(raw_rms=0.40, band_energy=0.70, spectral_flux=1.10, peak_level=0.60, sample_time=1.1)

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
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = AudioEngine(Config(), lambda event: None, report_dir=Path(tmpdir))
            engine._reset_session_stats()

            with mock.patch("audio_engine.log_event") as log_event_mock:
                engine._log_shutdown_summary()

            log_event_mock.assert_not_called()

    def test_shutdown_summary_writes_audio_report_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            engine = AudioEngine(Config(), lambda event: None, report_dir=report_dir)
            engine._reset_session_stats()
            engine._session_started_at = 10.0

            engine._update_session_stats(raw_rms=0.10, band_energy=0.20, spectral_flux=0.20, peak_level=0.20, sample_time=10.0)
            engine._update_session_stats(raw_rms=0.20, band_energy=0.15, spectral_flux=0.30, peak_level=0.40, sample_time=10.1)
            engine._update_session_stats(raw_rms=0.30, band_energy=0.10, spectral_flux=0.95, peak_level=0.90, sample_time=10.2)
            engine._update_session_stats(raw_rms=0.40, band_energy=0.12, spectral_flux=0.92, peak_level=0.88, sample_time=10.3)

            with mock.patch("audio_engine.time.time", return_value=10.5):
                engine._log_shutdown_summary()

            json_path = report_dir / "audio_session_report.json"
            csv_path = report_dir / "audio_session_report.csv"
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

            with open(json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            latest = payload["latest"]

            self.assertGreater(latest["flux_high_total_s"], 0.0)
            self.assertGreater(latest["peak_high_total_s"], 0.0)
            self.assertGreater(latest["trough_low_total_s"], 0.0)
            self.assertGreater(latest["flux_high_threshold"], 0.0)
            self.assertGreater(latest["peak_high_threshold"], 0.0)
            self.assertGreaterEqual(latest["trough_low_threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
