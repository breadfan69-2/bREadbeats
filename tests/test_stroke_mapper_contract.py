import inspect
import time
import unittest
from types import SimpleNamespace

from config import Config, StrokeMode
from stroke_mapper import StrokeMapper


class _StubAudioEngine:
    def __init__(self, spectrum):
        self._spectrum = spectrum

    def get_spectrum(self):
        return self._spectrum


class TestStrokeMapperContract(unittest.TestCase):
    def test_constructor_and_entrypoint_contract(self):
        sig = inspect.signature(StrokeMapper.__init__)
        params = list(sig.parameters.keys())

        self.assertIn("config", params)
        self.assertIn("send_callback", params)
        self.assertIn("get_volume", params)
        self.assertIn("audio_engine", params)

        mapper = StrokeMapper(Config())
        self.assertTrue(hasattr(mapper, "process_beat"))
        self.assertTrue(callable(getattr(mapper, "process_beat")))

    def test_non_mode3_anchor_phase_uses_bottom_park(self):
        mapper = StrokeMapper(Config())
        park_phase = mapper._get_park_phase()

        simple_anchor = mapper._get_anchor_phase_for_mode(StrokeMode.SIMPLE_CIRCLE, fallback_phase=0.123)
        spiral_anchor = mapper._get_anchor_phase_for_mode(StrokeMode.SPIRAL, fallback_phase=0.456)

        self.assertAlmostEqual(simple_anchor, park_phase, places=6)
        self.assertAlmostEqual(spiral_anchor, park_phase, places=6)

    def test_tempo_reset_hold_marks_recent_beats(self):
        mapper = StrokeMapper(Config())
        now = time.perf_counter()
        mapper._arm_tempo_reset_motion_hold(now)

        self.assertTrue(mapper._has_recent_beats(now=now + 1.0, window_s=0.1))
        self.assertFalse(mapper._has_recent_beats(now=now + 3.0, window_s=0.1))

    def test_arc_launch_phase_uses_geometry_phase(self):
        mapper = StrokeMapper(Config())
        mapper.state.alpha = 0.6
        mapper.state.beta = 0.2
        mapper.state.creep_reset_active = False
        mapper._geometry.reset(0.375)

        launch_phase = mapper._get_arc_launch_phase(StrokeMode.SIMPLE_CIRCLE)
        expected = 0.375 * 2.0 * 3.141592653589793
        self.assertAlmostEqual(launch_phase, expected, places=6)

    def test_beat_rejected_by_downbeat_gate_still_returns_idle_motion(self):
        cfg = Config()
        mapper = StrokeMapper(cfg)
        mapper._last_idle_time = time.perf_counter()

        event = SimpleNamespace(
            monotonic_timestamp=time.perf_counter(),
            timestamp=time.perf_counter(),
            intensity=0.9,
            spectral_flux=0.4,
            peak_energy=0.6,
            metronome_bpm=120.0,
            tempo_locked=True,
            acf_confidence=1.0,
            is_beat=True,
            is_downbeat=False,
            is_syncopated=False,
            beat_band='low_mid',
            fired_bands=['low_mid', 'high'],
            frequency=120.0,
        )

        cmd = mapper.process_beat(event)
        self.assertIsNotNone(cmd)

    def test_auto_fill_required_increases_when_fill_always_passes(self):
        cfg = Config()
        cfg.stroke.overall_amp_fill_target = 0.5
        cfg.stroke.overall_amp_fill_tolerance = 0.5
        cfg.stroke.overall_amp_fill_gate_enabled = True
        cfg.stroke.overall_amp_fill_auto_enabled = True
        cfg.stroke.overall_amp_fill_auto_target_pass_rate = 0.55
        cfg.stroke.overall_amp_fill_auto_ema_alpha = 0.30
        cfg.stroke.overall_amp_fill_auto_step = 0.03
        cfg.stroke.overall_amp_fill_auto_deadband = 0.02

        mapper = StrokeMapper(cfg, audio_engine=_StubAudioEngine([1.0] * 64))
        event = SimpleNamespace(intensity=1.0)

        initial_required = mapper._get_overall_amp_fill_required('beat')
        for _ in range(16):
            mapper._passes_overall_amp_fill_gate(event, 'beat')
        raised_required = mapper._get_overall_amp_fill_required('beat')

        self.assertGreater(raised_required, initial_required)

    def test_auto_fill_required_decreases_when_fill_always_fails(self):
        cfg = Config()
        cfg.stroke.overall_amp_fill_target = 0.5
        cfg.stroke.overall_amp_fill_tolerance = 0.5
        cfg.stroke.overall_amp_fill_gate_enabled = True
        cfg.stroke.overall_amp_fill_auto_enabled = True
        cfg.stroke.overall_amp_fill_auto_target_pass_rate = 0.70
        cfg.stroke.overall_amp_fill_auto_ema_alpha = 0.30
        cfg.stroke.overall_amp_fill_auto_step = 0.03
        cfg.stroke.overall_amp_fill_auto_deadband = 0.02

        mapper = StrokeMapper(cfg, audio_engine=_StubAudioEngine([0.0] * 64))
        event = SimpleNamespace(intensity=1.0)

        mapper._auto_fill_state['beat']['offset'] = 0.20
        initial_required = mapper._get_overall_amp_fill_required('beat')
        for _ in range(16):
            mapper._passes_overall_amp_fill_gate(event, 'beat')
        lowered_required = mapper._get_overall_amp_fill_required('beat')

        self.assertLess(lowered_required, initial_required)


if __name__ == "__main__":
    unittest.main()
