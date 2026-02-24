import unittest
from unittest.mock import patch

import numpy as np

from audio_modules.audioflux_adapter import AudioFluxAdapter, AudioFluxAdapterConfig


class TestAudioFluxAdapterFailOpen(unittest.TestCase):
    def test_disabled_adapter_is_inert(self):
        adapter = AudioFluxAdapter(48000, AudioFluxAdapterConfig(enabled=False))
        self.assertFalse(adapter.available)
        adapter.push_audio(np.ones(256, dtype=np.float32))
        self.assertIsNone(adapter.get_latest_features())

    def test_import_failure_is_fail_open(self):
        with patch("audio_modules.audioflux_adapter.importlib.import_module", side_effect=ImportError("no audioflux")):
            adapter = AudioFluxAdapter(48000, AudioFluxAdapterConfig(enabled=True))

        self.assertFalse(adapter.available)
        adapter.push_audio(np.ones(512, dtype=np.float32))
        self.assertIsNone(adapter.get_latest_features())

    def test_enabled_adapter_computes_features_when_available(self):
        with patch("audio_modules.audioflux_adapter.importlib.import_module", return_value=object()):
            adapter = AudioFluxAdapter(
                48000,
                AudioFluxAdapterConfig(enabled=True, frame_stride=1, fft_size=512, emit_onset_confidence=True),
            )

        self.assertTrue(adapter.available)
        t = np.linspace(0.0, 1.0, 1024, endpoint=False)
        wave = (0.4 * np.sin(2.0 * np.pi * 120.0 * t)).astype(np.float32)
        adapter.push_audio(wave)
        features = adapter.get_latest_features()

        self.assertIsNotNone(features)
        assert features is not None
        self.assertIn("af_entropy", features)
        self.assertIn("af_flatness", features)
        self.assertIn("af_hfc", features)
        self.assertIn("af_novelty", features)
        self.assertIn("af_rms", features)
        self.assertIn("af_onset_conf", features)


if __name__ == "__main__":
    unittest.main()
