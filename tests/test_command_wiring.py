import unittest
from collections import deque
from types import SimpleNamespace
import time
import math

from command_wiring import (
    apply_volume_ramp,
    apply_live_output_mode,
    attach_cached_tcode_values,
    compute_live_fourphase_bandrouter_pulse_interval_random_normalized,
    compute_live_fourphase_levels,
)
from audio_engine import BeatEvent
from network_engine import TCodeCommand
from tcode_wiring import compute_and_attach_tcode


class _TCodeWindowStub:
    def __init__(self, live_tcode_mode: str):
        self.config = SimpleNamespace(
            live_tcode_mode=live_tcode_mode,
            live_fourphase_model="tetra3d",
            live_fourphase_bandrouter_pulse_interval_random_percent=10.0,
            audio=SimpleNamespace(sample_rate=48_000),
            device_limits=SimpleNamespace(
                p0_freq_min=1.0,
                p0_freq_max=100.0,
                c0_freq_min=500.0,
                c0_freq_max=1500.0,
                p1_cycles_min=0.0,
                p1_cycles_max=20.0,
                p2_range_min=0.0,
                p2_range_max=1.0,
                p3_cycles_min=0.0,
                p3_cycles_max=20.0,
            ),
        )
        self._last_dot_alpha = 0.0
        self._last_dot_beta = 0.0
        self._last_dot_time = 0.0
        self._cached_p0_enabled = False
        self._cached_f0_enabled = False
        self._cached_p1_enabled = False
        self._cached_p3_enabled = False
        self._p0_freq_window = deque()
        self._f0_freq_window = deque()
        self._p1_window = deque()
        self._p3_window = deque()
        self._freq_window_ms = 80.0
        self._f0_last_sent_tcode = None
        self._f0_max_change_per_send = 1500
        self._f0_duration_base_ms = 220.0
        self._f0_duration_variance_ms = 40.0
        self.stroke_mapper = None


class TestCommandWiring(unittest.TestCase):
    @staticmethod
    def _excursion_from_center(levels: tuple[float, float, float, float]) -> float:
        return math.sqrt(sum((1.0 - float(value)) ** 2 for value in levels))

    @staticmethod
    def _event() -> BeatEvent:
        return BeatEvent(
            timestamp=time.time(),
            intensity=0.5,
            frequency=60.0,
            is_beat=False,
            spectral_flux=0.1,
            peak_energy=0.3,
            monotonic_timestamp=time.perf_counter(),
            raw_rms=0.08,
        )

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

    def test_compute_live_fourphase_levels_classic_and_tetra3d_differ(self):
        classic = compute_live_fourphase_levels(0.25, -0.5, model="classic", sub_bass=0.8)
        tetra3d = compute_live_fourphase_levels(0.25, -0.5, model="tetra3d", sub_bass=0.8)

        self.assertNotEqual(classic, tetra3d)

    def test_compute_live_fourphase_levels_classic_response_curves_change_output(self):
        linear = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="classic",
            beat_response_curves=("linear", "linear", "linear", "linear"),
        )
        bell = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="classic",
            beat_response_curves=("bell", "linear", "linear", "linear"),
        )

        self.assertNotEqual(linear, bell)

    def test_compute_live_fourphase_levels_classic_radius_contrast_changes_output(self):
        low_contrast = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="classic",
            beat_radius_contrast_strength=0.0,
        )
        high_contrast = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="classic",
            beat_radius_contrast_strength=1.0,
        )

        self.assertNotEqual(low_contrast, high_contrast)

    def test_compute_live_fourphase_levels_classic_speed_spread_changes_output(self):
        low_spread = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="classic",
            beat_speed_threshold_spread_strength=0.0,
            orbit_angular_speed=8.0,
        )
        high_spread = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="classic",
            beat_speed_threshold_spread_strength=1.0,
            orbit_angular_speed=8.0,
        )

        self.assertNotEqual(low_spread, high_spread)

    def test_compute_live_fourphase_levels_tetra3d_uses_sub_bass_vertical_push(self):
        without_bass = compute_live_fourphase_levels(0.25, -0.5, model="tetra3d", sub_bass=0.0)
        with_bass = compute_live_fourphase_levels(0.25, -0.5, model="tetra3d", sub_bass=1.0)

        self.assertNotEqual(without_bass, with_bass)

    def test_compute_live_fourphase_levels_layout_model_changes_tetra3d_output(self):
        straight = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="tetra3d",
            sub_bass=0.8,
            layout_model="Straight Line",
        )
        pair_top = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="tetra3d",
            sub_bass=0.8,
            layout_model="Pair At Top",
        )

        self.assertNotEqual(straight, pair_top)

    def test_compute_live_fourphase_levels_tetra3d_post_projection_expansion_increases_excursion(self):
        base = compute_live_fourphase_levels(
            0.18,
            -0.22,
            model="tetra3d",
            sub_bass=0.35,
            tetra_post_projection_expansion=1.0,
        )
        expanded = compute_live_fourphase_levels(
            0.18,
            -0.22,
            model="tetra3d",
            sub_bass=0.35,
            tetra_post_projection_expansion=1.6,
        )

        self.assertNotEqual(base, expanded)
        self.assertGreater(self._excursion_from_center(expanded), self._excursion_from_center(base))

    def test_compute_live_fourphase_levels_bandrouter_uses_mapping(self):
        band_levels = {
            "sub_bass": 0.9,
            "bass": 0.7,
            "low_mid": 0.4,
            "mid": 0.1,
            "upper_mid": 0.0,
            "presence": 0.0,
            "brilliance": 0.0,
        }
        default_mapping = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="bandrouter",
            band_levels=band_levels,
            band_mapping=(("mid", "upper_mid", "presence"), ("low_mid", "mid"), ("bass", "low_mid"), ("sub_bass", "bass")),
            fill_angle=0.0,
            base=0.92,
            silence_fade=1.0,
            orbit_radius=0.92,
        )
        remapped = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="bandrouter",
            band_levels=band_levels,
            band_mapping=(("brilliance",), ("presence",), ("mid",), ("sub_bass",)),
            fill_angle=0.0,
            base=0.92,
            silence_fade=1.0,
            orbit_radius=0.92,
        )

        self.assertNotEqual(default_mapping, remapped)

    def test_compute_live_fourphase_levels_bandrouter_supports_seven_band_mapping(self):
        band_levels = {
            "sub_bass": 0.10,
            "bass": 0.20,
            "low_mid": 0.30,
            "mid": 0.40,
            "upper_mid": 0.50,
            "presence": 0.60,
            "brilliance": 0.70,
        }
        presence_heavy = compute_live_fourphase_levels(
            0.12,
            -0.18,
            model="bandrouter",
            band_levels=band_levels,
            band_mapping=(("presence",), ("upper_mid",), ("bass",), ("sub_bass",)),
            fill_angle=0.0,
            base=0.72,
            silence_fade=1.0,
            orbit_radius=0.72,
        )
        brilliance_heavy = compute_live_fourphase_levels(
            0.12,
            -0.18,
            model="bandrouter",
            band_levels=band_levels,
            band_mapping=(("brilliance",), ("upper_mid",), ("bass",), ("sub_bass",)),
            fill_angle=0.0,
            base=0.72,
            silence_fade=1.0,
            orbit_radius=0.72,
        )

        self.assertNotEqual(presence_heavy, brilliance_heavy)

    def test_compute_live_fourphase_bandrouter_pulse_interval_random_uses_brilliance(self):
        normalized = compute_live_fourphase_bandrouter_pulse_interval_random_normalized(
            band_levels={"brilliance": 1.0},
            pulse_interval_random_percent=10.0,
        )

        self.assertAlmostEqual(normalized, 0.50, places=6)

    def test_compute_live_fourphase_levels_bandrouter_fill_mix_changes_output(self):
        band_levels = {
            "sub_bass": 0.8,
            "low_mid": 0.5,
            "mid": 0.2,
            "high": 0.1,
        }
        low_fill = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="bandrouter",
            band_levels=band_levels,
            fill_angle=0.0,
            base=0.92,
            silence_fade=1.0,
            orbit_radius=0.92,
            bandrouter_fill_mix=0.0,
        )
        high_fill = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="bandrouter",
            band_levels=band_levels,
            fill_angle=0.0,
            base=0.92,
            silence_fade=1.0,
            orbit_radius=0.92,
            bandrouter_fill_mix=0.45,
        )

        self.assertNotEqual(low_fill, high_fill)

    def test_compute_live_fourphase_levels_bandrouter_post_projection_expansion_increases_excursion(self):
        band_levels = {
            "sub_bass": 0.55,
            "low_mid": 0.30,
            "mid": 0.15,
            "high": 0.05,
        }
        base = compute_live_fourphase_levels(
            0.18,
            -0.22,
            model="bandrouter",
            band_levels=band_levels,
            fill_angle=0.0,
            base=0.55,
            silence_fade=1.0,
            orbit_radius=0.55,
            bandrouter_post_projection_expansion=1.0,
        )
        expanded = compute_live_fourphase_levels(
            0.18,
            -0.22,
            model="bandrouter",
            band_levels=band_levels,
            fill_angle=0.0,
            base=0.55,
            silence_fade=1.0,
            orbit_radius=0.55,
            bandrouter_post_projection_expansion=1.6,
        )

        self.assertNotEqual(base, expanded)
        self.assertGreater(self._excursion_from_center(expanded), self._excursion_from_center(base))

    def test_compute_live_fourphase_levels_bandrouter_idle_floor_changes_output(self):
        band_levels = {
            "sub_bass": 0.8,
            "low_mid": 0.3,
            "mid": 0.1,
            "high": 0.0,
        }
        low_idle = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="bandrouter",
            band_levels=band_levels,
            fill_angle=0.0,
            base=0.72,
            silence_fade=1.0,
            orbit_radius=0.72,
            bandrouter_idle_floor=0.0,
        )
        high_idle = compute_live_fourphase_levels(
            0.25,
            -0.5,
            model="bandrouter",
            band_levels=band_levels,
            fill_angle=0.0,
            base=0.72,
            silence_fade=1.0,
            orbit_radius=0.72,
            bandrouter_idle_floor=0.25,
        )

        self.assertNotEqual(low_idle, high_idle)

    def test_apply_live_output_mode_preserves_existing_direct_fourphase_tags(self):
        cmd = TCodeCommand(
            alpha=0.25,
            beta=-0.5,
            duration_ms=25,
            volume=1.0,
            tcode_tags={"E1": 1111, "E2": 2222, "E3": 3333, "E4": 4444},
        )

        apply_live_output_mode(
            cmd,
            live_tcode_mode="fourphase",
            live_fourphase_model="tetra3d",
            sub_bass=1.0,
        )

        self.assertFalse(cmd.include_linear_axes)
        self.assertEqual(cmd.tcode_tags["E1"], 1111)
        self.assertEqual(cmd.tcode_tags["E2"], 2222)
        self.assertEqual(cmd.tcode_tags["E3"], 3333)
        self.assertEqual(cmd.tcode_tags["E4"], 4444)
        self.assertEqual(cmd.tcode_tags["E1_duration"], 25)
        self.assertEqual(cmd.tcode_tags["E4_duration"], 25)

    def test_compute_and_attach_tcode_keeps_threephase_linear_axes(self):
        win = _TCodeWindowStub(live_tcode_mode="threephase")
        cmd = TCodeCommand(alpha=0.25, beta=-0.5, duration_ms=25, volume=1.0)
        event = self._event()

        compute_and_attach_tcode(win, cmd, event, spectrum=None)

        self.assertTrue(cmd.include_linear_axes)
        self.assertNotIn("E1", cmd.tcode_tags)
        tcode = cmd.to_tcode()
        self.assertIn("L0", tcode)
        self.assertIn("L1", tcode)

    def test_compute_and_attach_tcode_switches_to_fourphase_electrode_tags(self):
        win = _TCodeWindowStub(live_tcode_mode="fourphase")
        cmd = TCodeCommand(alpha=0.25, beta=-0.5, duration_ms=25, volume=1.0)
        event = self._event()

        compute_and_attach_tcode(win, cmd, event, spectrum=None)

        self.assertFalse(cmd.include_linear_axes)
        for tag in ("E1", "E2", "E3", "E4"):
            self.assertIn(tag, cmd.tcode_tags)
            self.assertIn(f"{tag}_duration", cmd.tcode_tags)
            self.assertGreaterEqual(cmd.tcode_tags[tag], 0)
            self.assertLessEqual(cmd.tcode_tags[tag], 9999)
            self.assertEqual(cmd.tcode_tags[f"{tag}_duration"], 25)
        tcode = cmd.to_tcode()
        self.assertNotIn("L0", tcode)
        self.assertNotIn("L1", tcode)
        self.assertIn("E1", tcode)

    def test_compute_and_attach_tcode_bandrouter_adds_p2_from_brilliance(self):
        win = _TCodeWindowStub(live_tcode_mode="fourphase")
        win.config.live_fourphase_model = "bandrouter"
        win.stroke_mapper = SimpleNamespace(
            _actual_radius=0.72,
            _orbit_phase=0.0,
            _last_decision=SimpleNamespace(silence_active=False),
            _current_band_levels=lambda: {
                "sub_bass": 0.10,
                "bass": 0.20,
                "low_mid": 0.30,
                "mid": 0.40,
                "upper_mid": 0.50,
                "presence": 0.60,
                "brilliance": 1.00,
            },
            _current_vertical_lift_signal=lambda: 0.10,
            _current_classic_orbit_angular_speed=lambda: 0.0,
            _intelligence=SimpleNamespace(
                energies=SimpleNamespace(sub_bass=0.10, low_mid=0.30, mid=0.40, high=0.0)
            ),
        )
        cmd = TCodeCommand(alpha=0.25, beta=-0.5, duration_ms=25, volume=1.0)
        event = self._event()

        compute_and_attach_tcode(win, cmd, event, spectrum=None)

        self.assertEqual(cmd.tcode_tags["P2"], 5000)
        self.assertEqual(cmd.tcode_tags["P2_duration"], 80)


if __name__ == "__main__":
    unittest.main()
