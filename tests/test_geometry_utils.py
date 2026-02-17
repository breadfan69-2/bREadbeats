import unittest

from geometry_utils import GeometryUtils


class TestGeometryUtils(unittest.TestCase):
    def test_silence_threshold_locks_to_rest_without_phase_advance(self):
        geom = GeometryUtils(phase=0.25, y_offset=0.5, silence_threshold=0.05)
        before = geom.get_phase()

        x, y = geom.update(bpm=140.0, dt=1.0, intensity=0.01, beat_detected=False)

        self.assertAlmostEqual(geom.get_phase(), before, places=7)
        self.assertAlmostEqual(x, 0.0, places=7)
        self.assertAlmostEqual(y, 0.5, places=7)

    def test_beat_confidence_decays_when_no_beats(self):
        geom = GeometryUtils(beat_confidence_decay_per_second=0.5)

        geom.update(bpm=120.0, dt=0.1, intensity=0.5, beat_detected=True)
        self.assertAlmostEqual(geom.get_beat_confidence(), 1.0, places=7)

        geom.update(bpm=120.0, dt=1.0, intensity=0.5, beat_detected=False)
        self.assertAlmostEqual(geom.get_beat_confidence(), 0.5, places=7)

    def test_low_confidence_increases_rest_pull(self):
        geom = GeometryUtils(
            phase=0.0,
            y_offset=0.5,
            sink_start_intensity=0.25,
            beat_confidence_decay_per_second=1.0,
        )

        x1, y1 = geom.update(bpm=0.0, dt=0.0, intensity=0.8, beat_detected=True)
        x2, y2 = geom.update(bpm=0.0, dt=1.0, intensity=0.8, beat_detected=False)

        self.assertGreater(y2, y1)
        self.assertLess(abs(x2), abs(x1))

    def test_sustained_low_intensity_forces_bpm_zero(self):
        geom = GeometryUtils(
            phase=0.0,
            silence_threshold=0.0,
            beat_confidence_decay_per_second=0.0,
            min_rotation_scale=1.0,
        )

        geom.update(bpm=120.0, dt=0.60, intensity=0.04, beat_detected=True)
        phase_after_first = geom.get_phase()
        self.assertGreater(phase_after_first, 0.0)

        geom.update(bpm=120.0, dt=0.60, intensity=0.04, beat_detected=False)
        phase_after_second = geom.get_phase()

        self.assertAlmostEqual(phase_after_second, phase_after_first, places=7)


if __name__ == "__main__":
    unittest.main()
