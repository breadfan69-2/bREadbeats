import math
import unittest

from geometry_utils import (
    GeometryUtils,
    exponential_approach,
    infer_orbit,
    nearest_anchor_crossing,
    orbit_point,
    quintic_ease,
    radius_cap_for_center,
)


# ── Pure-math primitives ─────────────────────────────────────────────


class TestQuinticEase(unittest.TestCase):
    def test_endpoints(self):
        self.assertAlmostEqual(quintic_ease(0.0), 0.0, places=10)
        self.assertAlmostEqual(quintic_ease(1.0), 1.0, places=10)

    def test_midpoint_is_half(self):
        self.assertAlmostEqual(quintic_ease(0.5), 0.5, places=10)

    def test_monotonic_increasing(self):
        prev = 0.0
        for i in range(1, 101):
            t = i / 100.0
            val = quintic_ease(t)
            self.assertGreaterEqual(val, prev)
            prev = val

    def test_clamps_out_of_range(self):
        self.assertAlmostEqual(quintic_ease(-0.5), 0.0, places=10)
        self.assertAlmostEqual(quintic_ease(1.5), 1.0, places=10)


class TestExponentialApproach(unittest.TestCase):
    def test_zero_dt_no_change(self):
        self.assertAlmostEqual(exponential_approach(0.0, 1.0, 3.0, 0.0), 0.0, places=10)

    def test_converges_toward_target(self):
        val = exponential_approach(0.0, 1.0, 3.0, 1.0)
        self.assertGreater(val, 0.9)
        self.assertLess(val, 1.0)

    def test_already_at_target(self):
        self.assertAlmostEqual(exponential_approach(5.0, 5.0, 10.0, 0.1), 5.0, places=10)


class TestOrbitPoint(unittest.TestCase):
    def test_angle_zero_on_x_axis(self):
        x, y = orbit_point(0.0, 1.0)
        self.assertAlmostEqual(x, 1.0, places=10)
        self.assertAlmostEqual(y, 0.0, places=10)

    def test_quarter_turn(self):
        x, y = orbit_point(math.pi / 2.0, 1.0)
        self.assertAlmostEqual(x, 0.0, places=7)
        self.assertAlmostEqual(y, 1.0, places=7)

    def test_center_offset(self):
        x, y = orbit_point(0.0, 0.5, center_x=1.0, center_y=2.0)
        self.assertAlmostEqual(x, 1.5, places=10)
        self.assertAlmostEqual(y, 2.0, places=10)


class TestInferOrbit(unittest.TestCase):
    def test_roundtrip_with_orbit_point(self):
        angle_in, radius_in = 1.23, 0.75
        x, y = orbit_point(angle_in, radius_in, center_y=0.2)
        angle_out, radius_out = infer_orbit(x, y, center_y=0.2)
        self.assertAlmostEqual(angle_out, angle_in, places=7)
        self.assertAlmostEqual(radius_out, radius_in, places=7)

    def test_at_origin(self):
        angle, radius = infer_orbit(0.0, 0.0, center_y=0.0)
        self.assertAlmostEqual(radius, 0.0, places=10)


class TestRadiusCapForCenter(unittest.TestCase):
    def test_centered(self):
        self.assertAlmostEqual(radius_cap_for_center(0.0), 1.0, places=10)

    def test_offset_reduces_cap(self):
        self.assertAlmostEqual(radius_cap_for_center(0.3), 0.7, places=10)
        self.assertAlmostEqual(radius_cap_for_center(-0.3), 0.7, places=10)

    def test_with_y_offset(self):
        # center_y=0.2, y_offset=0.1 → eff=0.3 → cap=0.7
        self.assertAlmostEqual(radius_cap_for_center(0.2, 0.1), 0.7, places=10)


class TestNearestAnchorCrossing(unittest.TestCase):
    def test_target_at_anchor(self):
        result = nearest_anchor_crossing(math.pi / 2, math.pi / 2, 0.2)
        self.assertAlmostEqual(result, math.pi / 2, places=7)

    def test_clamped_within_swing(self):
        # Target far from anchor: should clamp within swing
        result = nearest_anchor_crossing(math.pi, math.pi / 2, 0.1)
        self.assertLessEqual(abs(result - math.pi / 2), 0.1 + 1e-9)


# ── Legacy GeometryUtils class ───────────────────────────────────────


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
