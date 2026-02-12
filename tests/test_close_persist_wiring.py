import unittest

from close_persist_wiring import persist_runtime_ui_to_config
from config import Config


class _Value:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _Range:
    def __init__(self, low, high):
        self._low = low
        self._high = high

    def low(self):
        return self._low

    def high(self):
        return self._high


class _Check:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _Combo:
    def __init__(self, index):
        self._index = index

    def currentIndex(self):
        return self._index


class _WindowStub:
    def __init__(self):
        self.phase_advance_slider = _Value(0.42)
        self.pulse_freq_range_slider = _Range(33.0, 222.0)
        self.tcode_freq_range_slider = _Range(2001, 7007)
        self.freq_weight_slider = _Value(0.75)

        self.f0_freq_range_slider = _Range(44.0, 333.0)
        self.f0_tcode_range_slider = _Range(111, 4444)
        self.f0_weight_slider = _Value(0.5)

        self.volume_slider = _Value(84)
        self.alpha_weight_slider = _Value(1.1)
        self.beta_weight_slider = _Value(0.9)

        self.tempo_tracking_checkbox = _Check(True)
        self.time_sig_combo = _Combo(2)
        self.stability_threshold_slider = _Value(0.31)
        self.tempo_timeout_slider = _Value(2500)
        self.phase_snap_slider = _Value(0.27)

        self.metrics_global_cb = _Check(False)


class TestClosePersistWiring(unittest.TestCase):
    def test_persist_runtime_ui_to_config(self):
        cfg = Config()
        window = _WindowStub()

        persist_runtime_ui_to_config(window, cfg)

        self.assertAlmostEqual(cfg.stroke.phase_advance, 0.42, places=6)

        self.assertAlmostEqual(cfg.pulse_freq.monitor_freq_min, 33.0, places=6)
        self.assertAlmostEqual(cfg.pulse_freq.monitor_freq_max, 222.0, places=6)
        self.assertEqual(cfg.pulse_freq.tcode_min, 2001)
        self.assertEqual(cfg.pulse_freq.tcode_max, 7007)
        self.assertAlmostEqual(cfg.pulse_freq.freq_weight, 0.75, places=6)

        self.assertAlmostEqual(cfg.carrier_freq.monitor_freq_min, 44.0, places=6)
        self.assertAlmostEqual(cfg.carrier_freq.monitor_freq_max, 333.0, places=6)
        self.assertEqual(cfg.carrier_freq.tcode_min, 111)
        self.assertEqual(cfg.carrier_freq.tcode_max, 4444)
        self.assertAlmostEqual(cfg.carrier_freq.freq_weight, 0.5, places=6)

        self.assertAlmostEqual(cfg.volume, 0.84, places=6)
        self.assertAlmostEqual(cfg.alpha_weight, 1.1, places=6)
        self.assertAlmostEqual(cfg.beta_weight, 0.9, places=6)

        self.assertTrue(cfg.beat.tempo_tracking_enabled)
        self.assertEqual(cfg.beat.beats_per_measure, 6)
        self.assertAlmostEqual(cfg.beat.stability_threshold, 0.31, places=6)
        self.assertEqual(cfg.beat.tempo_timeout_ms, 2500)
        self.assertAlmostEqual(cfg.beat.phase_snap_weight, 0.27, places=6)

        self.assertFalse(cfg.auto_adjust.metrics_global_enabled)


if __name__ == "__main__":
    unittest.main()
