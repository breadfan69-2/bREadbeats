import unittest

from command_wiring import apply_volume_ramp, attach_cached_tcode_values
from network_engine import TCodeCommand


class TestCommandWiring(unittest.TestCase):
    def test_attach_cached_tcode_values(self):
        cmd = TCodeCommand(alpha=0.0, beta=0.0, duration_ms=100, volume=1.0)

        attach_cached_tcode_values(
            cmd,
            p0c0_enabled=True,
            cached_p0_enabled=True,
            cached_p0_val=3333,
            cached_f0_enabled=True,
            cached_f0_val=2222,
            cached_p1_enabled=True,
            cached_p1_val=4444,
            cached_p3_enabled=True,
            cached_p3_val=5555,
            freq_window_ms=250,
        )

        self.assertEqual(cmd.pulse_freq, 3333)
        self.assertEqual(cmd.tcode_tags['C0'], 2222)
        self.assertEqual(cmd.tcode_tags['P1'], 4444)
        self.assertEqual(cmd.tcode_tags['P1_duration'], 250)
        self.assertEqual(cmd.tcode_tags['P3'], 5555)
        self.assertEqual(cmd.tcode_tags['P3_duration'], 250)

    def test_attach_respects_p0c0_disable(self):
        cmd = TCodeCommand(alpha=0.0, beta=0.0, duration_ms=100, volume=1.0)

        attach_cached_tcode_values(
            cmd,
            p0c0_enabled=False,
            cached_p0_enabled=True,
            cached_p0_val=3333,
            cached_f0_enabled=True,
            cached_f0_val=2222,
            cached_p1_enabled=False,
            cached_p1_val=None,
            cached_p3_enabled=False,
            cached_p3_val=None,
            freq_window_ms=250,
        )

        self.assertIsNone(cmd.pulse_freq)
        self.assertNotIn('C0', cmd.tcode_tags)

    def test_apply_volume_ramp(self):
        cmd = TCodeCommand(alpha=0.0, beta=0.0, duration_ms=100, volume=1.0)

        apply_volume_ramp(
            cmd,
            volume_ramp_active=True,
            volume_ramp_start_time=10.0,
            volume_ramp_duration=2.0,
            volume_ramp_from=0.0,
            volume_ramp_to=1.0,
            now=11.0,
        )

        self.assertAlmostEqual(cmd.volume, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
