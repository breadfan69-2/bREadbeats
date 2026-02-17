import inspect
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from config import Config, StrokeMode
from stroke_mapper import StrokeMapper, PendingStrokeChange, PlannedTrajectory, MotionMode


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

    def test_hard_clamp_forces_overall_amplitude_to_zero_below_silence_threshold(self):
        cfg = Config()
        cfg.stroke.silence_threshold = 0.04
        mapper = StrokeMapper(cfg)

        loud_event = SimpleNamespace(peak_energy=0.7, raw_rms=0.3)
        mapper._update_envelope(loud_event)
        self.assertGreater(mapper._get_overall_amplitude(loud_event), 0.0)

        near_threshold_event = SimpleNamespace(peak_energy=0.7, raw_rms=0.039)
        mapper._update_envelope(near_threshold_event)
        self.assertEqual(mapper._get_overall_amplitude(near_threshold_event), 0.0)

        silent_event = SimpleNamespace(peak_energy=0.7, raw_rms=0.0003)
        mapper._update_envelope(silent_event)
        self.assertEqual(mapper._rms_envelope, 0.0)
        self.assertEqual(mapper._get_overall_amplitude(silent_event), 0.0)

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

    def test_phase_crossed_gate_detects_wrap(self):
        crossed = StrokeMapper._phase_crossed_gate(0.49, 0.52, (0.5,))
        self.assertTrue(crossed)

        wrapped_crossed = StrokeMapper._phase_crossed_gate(0.98, 0.04, (0.5, 1.0))
        self.assertTrue(wrapped_crossed)

        not_crossed = StrokeMapper._phase_crossed_gate(0.10, 0.40, (0.5,))
        self.assertFalse(not_crossed)

    def test_generate_idle_motion_commits_pending_change_on_gate_cross(self):
        cfg = Config()
        mapper = StrokeMapper(cfg)
        mapper._arc_commit_gate_points = (0.5,)
        mapper._last_geometry_phase = 0.49
        mapper._trajectory = None

        event = SimpleNamespace(
            monotonic_timestamp=time.perf_counter(),
            spectral_flux=0.2,
            peak_energy=0.2,
            beat_band='low_mid',
            fired_bands=['low_mid'],
            frequency=120.0,
            is_beat=False,
            is_downbeat=False,
        )
        mapper._pending_stroke_change = PendingStrokeChange(
            kind='beat',
            event=event,
            duration_mult=1.0,
            queued_at=time.perf_counter(),
        )

        committed_cmd = object()
        with patch.object(mapper._geometry, 'update', return_value=(0.1, 0.9)), \
             patch.object(mapper._geometry, 'get_phase', return_value=0.52), \
             patch.object(mapper, '_generate_beat_stroke', return_value=committed_cmd):
            cmd = mapper._generate_idle_motion(event, force_update=True)

        self.assertIs(cmd, committed_cmd)
        self.assertIsNone(mapper._pending_stroke_change)

    def test_forward_orbit_mode_detection(self):
        mapper = StrokeMapper(Config())
        self.assertTrue(mapper._is_forward_orbit_mode(StrokeMode.SIMPLE_CIRCLE))
        self.assertTrue(mapper._is_forward_orbit_mode(StrokeMode.SPIRAL))
        self.assertFalse(mapper._is_forward_orbit_mode(StrokeMode.TEARDROP))

    def test_spiral_beat_keeps_forward_direction(self):
        cfg = Config()
        cfg.stroke.mode = StrokeMode.SPIRAL
        mapper = StrokeMapper(cfg)
        mapper._spiral_direction = -1

        event = SimpleNamespace(
            monotonic_timestamp=time.perf_counter(),
            intensity=0.8,
            spectral_flux=0.5,
            flux_threshold=0.1,
            metronome_bpm=120.0,
            tempo_locked=True,
        )

        mapper._generate_beat_stroke(event)
        self.assertEqual(mapper._spiral_direction, 1)

    def test_align_trajectory_launch_handoff_overwrites_start_and_blends_ingress(self):
        mapper = StrokeMapper(Config())
        mapper.state.alpha = 0.42
        mapper.state.beta = 0.18

        alpha = np.array([-0.70, -0.62, -0.55, -0.40], dtype=float)
        beta = np.array([-0.76, -0.70, -0.64, -0.45], dtype=float)
        original_gap = float(np.hypot(alpha[1] - mapper.state.alpha, beta[1] - mapper.state.beta))

        mapper._align_trajectory_launch_handoff(alpha, beta, StrokeMode.SIMPLE_CIRCLE)

        self.assertAlmostEqual(alpha[0], mapper.state.alpha, places=6)
        self.assertAlmostEqual(beta[0], mapper.state.beta, places=6)

        blended_gap = float(np.hypot(alpha[1] - mapper.state.alpha, beta[1] - mapper.state.beta))
        self.assertLess(blended_gap, original_gap)

        expected_phase = float(np.arctan2(mapper.state.alpha, mapper.state.beta))
        if expected_phase < 0:
            expected_phase += 2 * np.pi
        self.assertAlmostEqual(mapper._last_geometry_phase, expected_phase / (2 * np.pi), places=6)

    def test_trajectory_playback_uses_forward_geometry_phase_overlay(self):
        cfg = Config()
        cfg.stroke.mode = StrokeMode.SIMPLE_CIRCLE
        mapper = StrokeMapper(cfg)
        mapper._last_known_bpm = 120.0
        mapper._geometry.reset(0.55)
        mapper._last_geometry_phase = mapper._geometry.get_phase()

        mapper._trajectory = PlannedTrajectory(
            alpha_points=np.array([0.80, 0.80], dtype=float),
            beta_points=np.array([0.00, 0.00], dtype=float),
            step_durations=[120, 120],
            n_points=2,
            current_index=0,
            band_volume=1.0,
            start_time=time.perf_counter() - 0.05,
            original_bpm=120.0,
        )

        now = time.perf_counter()
        cmd = mapper._advance_trajectory(now=now, dt_s=0.05)

        self.assertIsNotNone(cmd)
        self.assertGreater(mapper._last_geometry_phase, 0.55)
        # 0.55 -> ~0.65 phase at 120 BPM for 50ms, so sin() should be negative.
        self.assertLess(cmd.alpha, 0.0)

    def test_trajectory_overlay_uses_shared_radius_for_alpha_beta(self):
        cfg = Config()
        cfg.stroke.mode = StrokeMode.SPIRAL
        cfg.alpha_weight = 0.2
        cfg.beta_weight = 1.0
        mapper = StrokeMapper(cfg)
        mapper._trajectory_radius_value = 0.9
        mapper._trajectory_radius_target = 0.9

        alpha, beta = mapper._trajectory_overlay_from_phase(
            traj_alpha=0.9,
            traj_beta=0.0,
            phase=0.25,
            mode=StrokeMode.SPIRAL,
        )

        # At phase=0.25 => angle=pi/2, so alpha~radius and beta~0 on a true circle.
        self.assertAlmostEqual(alpha, 0.9, places=3)
        self.assertAlmostEqual(beta, 0.0, places=3)

    def test_idle_motion_forces_forward_spiral_direction(self):
        mapper = StrokeMapper(Config())
        mapper._spiral_direction = -1
        mapper._last_idle_time = 0.0

        mapper._generate_idle_motion(event=None, force_update=True)

        self.assertEqual(mapper._spiral_direction, 1)

    def test_slew_toward_limits_step_size(self):
        self.assertAlmostEqual(StrokeMapper._slew_toward(0.20, 0.90, 0.05), 0.25, places=6)
        self.assertAlmostEqual(StrokeMapper._slew_toward(0.90, 0.20, 0.05), 0.85, places=6)

    def test_gate_amplitude_changes_max_five_hundredths_per_frame(self):
        mapper = StrokeMapper(Config())
        next_val = mapper._step_gate_modulation_value(1.0, 0.4, max_step=0.05)
        self.assertAlmostEqual(next_val, 0.95, places=6)

    def test_trajectory_overlay_radius_is_slewed(self):
        mapper = StrokeMapper(Config())
        mapper._trajectory_radius_value = 0.20

        alpha, beta = mapper._trajectory_overlay_from_phase(
            traj_alpha=0.90,
            traj_beta=0.00,
            phase=0.0,
            mode=StrokeMode.SIMPLE_CIRCLE,
        )

        self.assertAlmostEqual(float(np.hypot(alpha, beta)), 0.40, places=6)

    def test_radius_slew_uses_fast_attack_and_slow_decay(self):
        mapper = StrokeMapper(Config())

        attack = mapper._slew_radius_toward(0.20, 0.90)
        decay = mapper._slew_radius_toward(0.90, 0.20)

        self.assertAlmostEqual(attack, 0.40, places=6)
        self.assertAlmostEqual(decay, 0.87, places=6)

    def test_silence_deadzone_gate_uses_hysteresis(self):
        mapper = StrokeMapper(Config())

        self.assertTrue(mapper._update_silence_deadzone_gate(0.01))
        self.assertTrue(mapper._update_silence_deadzone_gate(0.045))
        self.assertTrue(mapper._update_silence_deadzone_gate(0.055))
        self.assertTrue(mapper._update_silence_deadzone_gate(0.055))
        self.assertTrue(mapper._update_silence_deadzone_gate(0.055))
        self.assertFalse(mapper._update_silence_deadzone_gate(0.055))

    def test_idle_motion_noise_floor_runs_landing_sequence_to_bottom(self):
        mapper = StrokeMapper(Config())
        mapper._last_idle_time = 0.0
        mapper._silence_deadzone_active = False
        mapper.state.alpha = 0.42
        mapper.state.beta = -0.35

        event = SimpleNamespace(
            monotonic_timestamp=time.perf_counter(),
            intensity=0.0,
            spectral_flux=0.0,
            peak_energy=0.0,
            metronome_bpm=120.0,
            beat_band='',
            fired_bands=[],
            frequency=0.0,
            is_beat=False,
            is_downbeat=False,
        )

        with patch.object(mapper._geometry, 'update', wraps=mapper._geometry.update) as update_mock:
            cmd = None
            for _ in range(120):
                event.monotonic_timestamp = time.perf_counter()
                cmd = mapper._generate_idle_motion(event, force_update=True)

        self.assertTrue(mapper._silence_deadzone_active)
        self.assertGreaterEqual(update_mock.call_count, 1)
        expected_alpha = 0.0
        expected_beta = float(mapper._silence_park_radius)
        self.assertLessEqual(abs(mapper.state.alpha - expected_alpha), 0.05)
        self.assertLessEqual(abs(mapper.state.beta - expected_beta), 0.05)
        self.assertTrue(mapper._silence_parked_lock)
        self.assertAlmostEqual(mapper._idle_radius_value, 0.70, places=6)
        self.assertAlmostEqual(cmd.volume, 0.0, places=6)

    def test_idle_motion_reopens_with_fast_attack_from_park(self):
        mapper = StrokeMapper(Config())
        mapper._last_idle_time = 0.0
        mapper._motion_mode = MotionMode.FULL_STROKE
        mapper._silence_deadzone_active = True
        mapper._idle_radius_value = 0.10
        mapper._idle_radius_target = 0.10

        event = SimpleNamespace(
            monotonic_timestamp=time.perf_counter(),
            intensity=0.8,
            spectral_flux=0.4,
            peak_energy=0.8,
            metronome_bpm=120.0,
            beat_band='low_mid',
            fired_bands=['low_mid'],
            frequency=120.0,
            is_beat=False,
            is_downbeat=False,
        )

        captured = {}

        def _capture_update(*, bpm, dt, intensity, beat_detected):
            captured['intensity'] = intensity
            return mapper._park_alpha, mapper._park_beta

        with patch.object(mapper._geometry, 'update', side_effect=_capture_update), \
             patch.object(mapper._geometry, 'get_phase', return_value=mapper._last_geometry_phase):
            for _ in range(5):
                mapper._generate_idle_motion(event, force_update=True)

        self.assertFalse(mapper._silence_deadzone_active)
        self.assertAlmostEqual(captured['intensity'], 0.30, places=6)
        self.assertAlmostEqual(mapper._idle_radius_value, 0.30, places=6)

    def test_apply_micro_jerk_is_bypassed_for_circular_modes(self):
        cfg = Config()
        cfg.stroke.mode = StrokeMode.SIMPLE_CIRCLE
        mapper = StrokeMapper(cfg)
        mapper._micro_effects_enabled = True
        mapper._micro_jerk_alpha = 0.2
        mapper._micro_jerk_beta = -0.2
        mapper._last_micro_jerk_time = time.time()

        alpha, beta = mapper._apply_micro_jerk(
            alpha=0.3,
            beta=0.4,
            overall_amplitude=0.9,
            silence_deadzone_active=False,
        )

        self.assertAlmostEqual(alpha, 0.3, places=6)
        self.assertAlmostEqual(beta, 0.4, places=6)

    def test_apply_anti_stop_is_bypassed_at_low_amplitude(self):
        mapper = StrokeMapper(Config())

        alpha, beta = mapper._apply_anti_stop(
            alpha=0.8,
            beta=-0.2,
            delta=0.001,
            orbit_replacement_active=False,
            overall_amplitude=0.01,
            silence_deadzone_active=False,
        )

        self.assertAlmostEqual(alpha, 0.8, places=6)
        self.assertAlmostEqual(beta, -0.2, places=6)

    def test_jump_debug_filters_silence_and_small_delta(self):
        mapper = StrokeMapper(Config())
        mapper.state.alpha = 0.0
        mapper.state.beta = 0.0

        mapper._silence_deadzone_active = True
        with patch('stroke_mapper.log_event') as mocked_log:
            mapper._log_jump_debug(source="silence", alpha=0.8, beta=0.8)
            mocked_log.assert_not_called()

        mapper._silence_deadzone_active = False
        with patch('stroke_mapper.log_event') as mocked_log:
            mapper._log_jump_debug(source="small", alpha=0.05, beta=0.00)
            mocked_log.assert_not_called()

        with patch('stroke_mapper.log_event') as mocked_log:
            mapper._log_jump_debug(source="large", alpha=0.2, beta=0.0)
            mocked_log.assert_called_once()

    def test_rhythmic_phrase_interval_selection_uses_trigger_type_mapping(self):
        mapper = StrokeMapper(Config())
        now = time.perf_counter()

        beat_event = SimpleNamespace(is_syncopated=False, is_downbeat=False, is_beat=True)
        downbeat_event = SimpleNamespace(is_syncopated=False, is_downbeat=True, is_beat=True)
        sync_event = SimpleNamespace(is_syncopated=True, is_downbeat=False, is_beat=False)

        mapper._last_any_beat_time = now
        self.assertEqual(mapper._rhythmic_phrase_interval_for_trigger(mapper._classify_rhythmic_phrase_trigger(sync_event, now=now)), 1)
        self.assertEqual(mapper._rhythmic_phrase_interval_for_trigger(mapper._classify_rhythmic_phrase_trigger(beat_event, now=now)), 2)
        self.assertEqual(mapper._rhythmic_phrase_interval_for_trigger(mapper._classify_rhythmic_phrase_trigger(downbeat_event, now=now)), 4)

        mapper._last_any_beat_time = 0.0
        self.assertEqual(mapper._rhythmic_phrase_interval_for_trigger(mapper._classify_rhythmic_phrase_trigger(beat_event, now=now)), 8)

    def test_rhythmic_phrase_queued_launch_uses_next_beat_and_duration_formula(self):
        cfg = Config()
        mapper = StrokeMapper(cfg)
        base_now = time.perf_counter()
        mapper._last_any_beat_time = base_now

        queue_event = SimpleNamespace(
            monotonic_timestamp=base_now,
            intensity=0.95,
            peak_energy=0.95,
            spectral_flux=0.30,
            metronome_bpm=120.0,
            is_beat=True,
            is_downbeat=False,
            is_syncopated=False,
        )
        mapper._rhythmic_phrase_beat_counter = 10
        mapper._trigger_rhythmic_phrase_explode(queue_event, now=base_now)

        self.assertTrue(mapper._rhythmic_phrase.pending_movement)
        self.assertFalse(mapper._rhythmic_phrase.active)

        mapper._rhythmic_phrase_beat_counter = 11
        launch_cmd = mapper._try_start_queued_rhythmic_phrase(queue_event, now=base_now + 0.1)

        self.assertIsNotNone(launch_cmd)
        self.assertTrue(mapper._rhythmic_phrase.active)
        self.assertEqual(mapper._rhythmic_phrase.interval_beats, 2)
        self.assertAlmostEqual(mapper._rhythmic_phrase.duration_s, (60.0 / 120.0) * 2.0, places=6)

    def test_rhythmic_phrase_curve_is_deterministic_bloom_arc(self):
        mapper = StrokeMapper(Config())
        phrase = mapper._rhythmic_phrase
        phrase.start_angle = 0.0
        phrase.total_rotation = float(2.0 * np.pi)
        phrase.park_radius = 0.70
        phrase.bloom_amount = 0.20

        a0, b0 = mapper._sample_rhythmic_phrase_curve(phrase, 0.0)
        am, bm = mapper._sample_rhythmic_phrase_curve(phrase, 0.5)
        a1, b1 = mapper._sample_rhythmic_phrase_curve(phrase, 1.0)

        self.assertAlmostEqual(a0, 0.0, places=6)
        self.assertAlmostEqual(b0, 0.70, places=6)
        self.assertAlmostEqual(am, 0.0, places=3)
        self.assertLess(bm, -0.89)
        self.assertAlmostEqual(a1, 0.0, places=6)
        self.assertAlmostEqual(b1, 0.70, places=6)

    def test_pending_rhythmic_phrase_keeps_wait_state_rock_solid_at_park(self):
        mapper = StrokeMapper(Config())
        mapper._last_idle_time = 0.0
        mapper._silence_deadzone_active = False
        mapper.state.alpha = -0.45
        mapper.state.beta = 0.12
        mapper._rhythmic_phrase.pending_movement = True
        mapper._rhythmic_phrase.active = False
        mapper._rhythmic_phrase.state = "pending"

        event = SimpleNamespace(
            monotonic_timestamp=time.perf_counter(),
            intensity=0.2,
            spectral_flux=0.0,
            peak_energy=0.0,
            metronome_bpm=120.0,
            beat_band='',
            fired_bands=[],
            frequency=0.0,
            is_beat=False,
            is_downbeat=False,
        )

        cmd = mapper._generate_idle_motion(event, force_update=True)

        self.assertIsNotNone(cmd)
        self.assertAlmostEqual(mapper.state.alpha, 0.0, places=6)
        self.assertAlmostEqual(mapper.state.beta, 0.70, places=6)
        self.assertAlmostEqual(cmd.alpha, 0.0, places=6)
        self.assertAlmostEqual(cmd.beta, 0.70, places=6)


if __name__ == "__main__":
    unittest.main()
