import unittest

import numpy as np

from audio_modules.signal_frontend import SignalFrontend, SignalFrontendConfig


class TestSignalFrontend(unittest.TestCase):
    def test_process_requires_enough_samples(self):
        frontend = SignalFrontend(
            SignalFrontendConfig(sample_rate=48000, channels=2, fft_size=1024, hop_size=256)
        )
        tiny = np.zeros((128, 2), dtype=np.float32)
        frame = frontend.process(tiny, mono_time=1.0, wall_time=1.0)
        self.assertIsNone(frame)

    def test_process_emits_frame_with_expected_fields(self):
        frontend = SignalFrontend(
            SignalFrontendConfig(sample_rate=48000, channels=2, gain=1.0, fft_size=1024, hop_size=256)
        )

        t = np.linspace(0.0, 1024 / 48000.0, 1024, endpoint=False)
        wave = (0.5 * np.sin(2.0 * np.pi * 120.0 * t)).astype(np.float32)
        stereo = np.stack((wave, wave), axis=1)

        frame = frontend.process(stereo, mono_time=2.0, wall_time=3.0)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.mono_time, 2.0)
        self.assertEqual(frame.wall_time, 3.0)
        self.assertGreater(len(frame.spectrum), 0)
        self.assertGreater(frame.raw_rms, 0.0)
        self.assertGreater(frame.raw_rms_db, -120.0)

    def test_spectral_flux_rises_on_change(self):
        frontend = SignalFrontend(
            SignalFrontendConfig(sample_rate=48000, channels=1, gain=1.0, fft_size=512, hop_size=256)
        )

        t = np.linspace(0.0, 512 / 48000.0, 512, endpoint=False)
        a = (0.3 * np.sin(2.0 * np.pi * 90.0 * t)).astype(np.float32)
        b = (0.3 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)

        first = frontend.process(a, mono_time=1.0, wall_time=1.0)
        second = frontend.process(b, mono_time=1.1, wall_time=1.1)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertAlmostEqual(first.spectral_flux, 0.0, places=6)
        self.assertGreater(second.spectral_flux, 0.0)


if __name__ == "__main__":
    unittest.main()
