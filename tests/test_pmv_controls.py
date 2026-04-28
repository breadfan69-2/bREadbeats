from __future__ import annotations

import unittest

from PyQt6.QtWidgets import QApplication

from pmv_audio_analysis import AnalysisConfig
from pmv_automap import AutomapConfig
from pmv_axis_converter import AxisConfig
from pmv_beat_engine import BeatDetectionConfig
from pmv_controls import PMVControlsPanel
from pmv_position_mapper import MappingConfig


class TestPmvControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.panel = PMVControlsPanel()

    def test_getters_return_expected_config_types(self):
        self.assertIsInstance(self.panel.get_analysis_config(), AnalysisConfig)
        self.assertIsInstance(self.panel.get_beat_config(), BeatDetectionConfig)
        self.assertIsInstance(self.panel.get_mapping_config(), MappingConfig)
        self.assertIsInstance(self.panel.get_axis_config(), AxisConfig)
        self.assertIsInstance(self.panel.get_automap_config(), AutomapConfig)

    def test_preset_round_trip_applies_values(self):
        preset = {
            "analysis": {"sample_rate": 44100, "fft_size": 4096, "lowpass_enabled": True, "lowpass_hz": 1200.0},
            "beat_detection": {
                "sensitivity": 0.75,
                "refractory_ms": 190.0,
                "use_librosa": False,
                "multibus_config": {"w_flux": 0.25, "w_band": 0.50, "w_delta": 0.75, "w_phase": 0.40},
            },
            "mapping": {
                "pitch_range": 80.0,
                "energy_multiplier": 18.0,
                "center_offset": 20.0,
                "overflow_mode": "bounce",
            },
            "ml": {"enabled": True, "strength": 0.42, "cadence_mode": "fixed_2"},
            "axis": {
                "speed_threshold_pct": 66.0,
                "preview_tcode_mode": "fourphase",
                "pulse_freq_center": 57.0,
                "pulse_freq_min": 24.0,
                "pulse_freq_max": 78.0,
                "carrier_freq_center": 52.0,
                "carrier_freq_min": 42.0,
                "carrier_freq_max": 58.0,
                "enabled_axes": ["main", "alpha", "beta", "e1"],
            },
            "automap": {"enabled": True, "target_y_position": 48.0, "optimization_mode": "cmean"},
            "output": {"format": "csv"},
        }

        self.panel.set_from_preset(preset)
        actual = self.panel.to_preset()

        self.assertEqual(actual["analysis"]["sample_rate"], 44100)
        self.assertEqual(actual["analysis"]["fft_size"], 4096)
        self.assertAlmostEqual(actual["beat_detection"]["sensitivity"], 0.75, places=2)
        self.assertAlmostEqual(actual["mapping"]["pitch_range"], 80.0, places=1)
        self.assertEqual(actual["mapping"]["overflow_mode"], "bounce")
        self.assertEqual(actual["ml"]["cadence_mode"], "fixed_2")
        self.assertAlmostEqual(actual["axis"]["pulse_freq_center"], 57.0, places=1)
        self.assertAlmostEqual(actual["axis"]["pulse_freq_min"], 24.0, places=1)
        self.assertAlmostEqual(actual["axis"]["pulse_freq_max"], 78.0, places=1)
        for legacy_suffix in ("start", "end"):
            self.assertNotIn(f"pulse_freq_range_{legacy_suffix}", actual["axis"])
        self.assertAlmostEqual(actual["axis"]["carrier_freq_center"], 52.0, places=1)
        self.assertAlmostEqual(actual["axis"]["carrier_freq_min"], 42.0, places=1)
        self.assertAlmostEqual(actual["axis"]["carrier_freq_max"], 58.0, places=1)
        self.assertIn("alpha", actual["axis"]["enabled_axes"])
        self.assertEqual(actual["axis"]["preview_tcode_mode"], "fourphase")
        self.assertTrue(actual["automap"]["enabled"])
        self.assertEqual(actual["output"]["format"], "csv")


if __name__ == "__main__":
    unittest.main()
