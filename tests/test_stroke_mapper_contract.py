import math
import time
import unittest
from typing import Any

from audio_engine import BeatEvent
from beat_intelligence import BeatDecision, BeatIntelligence
from config import Config
from stroke_mapper import StrokeMapper


class TestStrokeMapperContract(unittest.TestCase):
    @staticmethod
    def _excursion_from_center(levels: tuple[float, float, float, float]) -> float:
        return math.sqrt(sum((1.0 - float(value)) ** 2 for value in levels))

    def _event(self, **overrides) -> BeatEvent:
        payload: dict[str, Any] = dict(
            timestamp=time.time(),
            intensity=0.5,
            frequency=60.0,
            is_beat=False,
            spectral_flux=0.1,
            peak_energy=0.3,
            is_downbeat=False,
            bpm=120.0,
            tempo_locked=False,
            metronome_bpm=120.0,
            is_syncopated=False,
            monotonic_timestamp=time.perf_counter(),
            raw_rms=0.08,
        )
        payload.update(overrides)
        return BeatEvent(**payload)

    def _drain_journey(self, bi: BeatIntelligence, frames: int = 140) -> None:
        for _ in range(frames):
            bi.build_decision(event=self._event(), dt=1.0 / 60.0, silence_override=False)

    def test_decision_dataclass_exists(self):
        decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.8,
            silence_active=False,
            journey_completion=0.25,
        )
        self.assertEqual(decision.trigger_kind, "beat")
        self.assertEqual(decision.interval_beats, 2)
        self.assertAlmostEqual(decision.radius_bloom, 0.8, places=6)
        self.assertFalse(decision.silence_active)
        self.assertAlmostEqual(decision.journey_completion, 0.25, places=6)

    def test_trigger_interval_mapping(self):
        intelligence = BeatIntelligence(Config())

        self.assertEqual(intelligence.interval_beats_for_trigger("syncopation"), 1)
        self.assertEqual(intelligence.interval_beats_for_trigger("beat"), 2)
        self.assertEqual(intelligence.interval_beats_for_trigger("downbeat"), 4)
        self.assertEqual(intelligence.interval_beats_for_trigger("creep"), 4)

    def test_trigger_classifier_maps_to_creep_when_no_beat_family(self):
        intelligence = BeatIntelligence(Config())
        event = self._event(is_beat=False, is_downbeat=False, is_syncopated=False)
        self.assertEqual(intelligence.classify_trigger(event), "creep")

    def test_trigger_classifier_priority_syncopation_then_downbeat_then_beat(self):
        intelligence = BeatIntelligence(Config())

        sync_event = self._event(is_syncopated=True, is_downbeat=True, is_beat=True)
        self.assertEqual(intelligence.classify_trigger(sync_event), "syncopation")

        downbeat_event = self._event(is_syncopated=False, is_downbeat=True, is_beat=True)
        self.assertEqual(intelligence.classify_trigger(downbeat_event), "downbeat")

        beat_event = self._event(is_syncopated=False, is_downbeat=False, is_beat=True)
        self.assertEqual(intelligence.classify_trigger(beat_event), "beat")

    def test_sub_bass_maps_to_radius_bloom_range(self):
        intelligence = BeatIntelligence(Config())
        intelligence.rms_envelope = 0.15  # above low-amp suppression threshold

        intelligence.energies.sub_bass = 0.0
        self.assertAlmostEqual(intelligence.compute_radius_bloom_from_sub_bass(), 0.70, places=6)

        intelligence.energies.sub_bass = 1.0
        bloom_max = intelligence.compute_radius_bloom_from_sub_bass()
        self.assertAlmostEqual(bloom_max, 1.0, places=6)

        intelligence.energies.sub_bass = 0.5
        midpoint = intelligence.compute_radius_bloom_from_sub_bass()
        self.assertGreater(midpoint, 0.70)
        self.assertLess(midpoint, 1.0)

    def test_flux_increases_bloom(self):
        intelligence = BeatIntelligence(Config())
        intelligence.rms_envelope = 0.15  # above low-amp suppression
        intelligence.energies.sub_bass = 0.5

        low_flux_event = self._event(spectral_flux=0.0)
        high_flux_event = self._event(spectral_flux=0.30)

        low_flux_bloom = intelligence.compute_radius_bloom_from_sub_bass(low_flux_event)
        high_flux_bloom = intelligence.compute_radius_bloom_from_sub_bass(high_flux_event)

        self.assertGreater(high_flux_bloom, low_flux_bloom)
        self.assertLessEqual(high_flux_bloom, 1.0)

    def test_build_decision_emits_trigger_interval_and_progress(self):
        intelligence = BeatIntelligence(Config())
        # Prime with downbeat and allow protected journey to finish
        prime = self._event(is_downbeat=True, tempo_locked=True)
        intelligence.build_decision(event=prime, dt=1.0 / 60.0, silence_override=False)
        self._drain_journey(intelligence)

        event = self._event(is_syncopated=False, is_downbeat=False, is_beat=True, tempo_locked=True)
        decision = intelligence.build_decision(event=event, dt=1.0 / 60.0, silence_override=False)

        self.assertEqual(decision.trigger_kind, "beat")
        self.assertEqual(decision.interval_beats, 2)
        self.assertGreaterEqual(decision.radius_bloom, 0.70)
        self.assertLessEqual(decision.radius_bloom, 0.95)
        self.assertFalse(decision.silence_active)
        self.assertGreaterEqual(decision.journey_completion, 0.0)
        self.assertLessEqual(decision.journey_completion, 1.0)

    def test_tempo_lock_required_falls_back_to_creep_when_not_ready(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = True
        cfg.beat.teaching_metronome_relaxed_confidence = 0.5
        intelligence = BeatIntelligence(cfg)

        event = self._event(is_beat=True, tempo_locked=False, acf_confidence=0.1)
        decision = intelligence.build_decision(event=event, dt=1.0 / 60.0, silence_override=False)

        self.assertEqual(decision.trigger_kind, "creep")
        self.assertEqual(decision.interval_beats, 4)

    # test_strict_bass_gate_blocks_non_bass_beats removed — gate no longer exists
    # test_transient_policy_kick_hat removed — transient profile gate now blocks hat/voice-only
    # test_voice_like_high_only removed — transient profile gate now blocks hat/voice-only
    # test_voice_like_low_mid_plus_high removed — transient profile gate now blocks hat/voice-only

    def test_full_motion_allows_low_flux_when_fullness_is_high(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False

        intelligence = BeatIntelligence(cfg)
        intelligence.compute_radius_bloom_from_sub_bass = lambda event=None: 0.95
        intelligence.compute_energy_fullness = lambda: 0.62

        decision = intelligence.build_decision(
            event=self._event(
                is_beat=True,
                tempo_locked=True,
                spectral_flux=0.06,
                beat_band="sub_bass",
                fired_bands=["sub_bass", "high"],
                beat_features={
                    "kick_like_conf": 0.88,
                    "hat_like_conf": 0.62,
                    "bass_dominance": 2.8,
                },
            ),
            dt=1.0 / 60.0,
            silence_override=False,
        )

        self.assertEqual(decision.trigger_kind, "beat")
        self.assertAlmostEqual(decision.radius_bloom, 0.95, places=6)

    def test_active_beat_journey_is_not_overwritten_by_intermediate_creep_frames(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        intelligence = BeatIntelligence(cfg)

        # Prime with downbeat and allow protected journey to finish before beat trigger
        prime = self._event(is_downbeat=True)
        intelligence.build_decision(event=prime, dt=1.0 / 60.0, silence_override=False)
        self._drain_journey(intelligence)

        beat_event = self._event(is_beat=True, tempo_locked=True)
        first = intelligence.build_decision(event=beat_event, dt=1.0 / 60.0, silence_override=False)

        between_event = self._event(is_beat=False, is_downbeat=False, is_syncopated=False)
        second = intelligence.build_decision(event=between_event, dt=1.0 / 60.0, silence_override=False)

        self.assertEqual(first.trigger_kind, "beat")
        self.assertEqual(second.trigger_kind, "beat")
        self.assertEqual(second.interval_beats, 2)
        self.assertGreater(second.journey_completion, 0.0)

    def test_treble_elevator_uses_positive_park_geometry(self):
        mapper = StrokeMapper(Config())
        self.assertAlmostEqual(mapper._park_y, 0.20, places=6)

    def test_band_levels_prefer_audio_engine_seven_band_getter(self):
        class _SevenBandAudioStub:
            @staticmethod
            def get_live_fourphase_band_energies() -> dict[str, float]:
                return {
                    "sub_bass": 0.10,
                    "bass": 0.20,
                    "low_mid": 0.30,
                    "mid": 0.40,
                    "upper_mid": 0.50,
                    "presence": 0.60,
                    "brilliance": 0.70,
                }

        mapper = StrokeMapper(Config())
        mapper._intelligence.audio_engine = _SevenBandAudioStub()

        band_levels = mapper._current_band_levels()

        self.assertAlmostEqual(band_levels["presence"], 0.60, places=6)
        self.assertAlmostEqual(band_levels["brilliance"], 0.70, places=6)

    def test_radius_bloom_uses_continuous_smoothing(self):
        mapper = StrokeMapper(Config())
        target = 0.95
        radius_alpha = 0.15

        mapper._actual_radius = mapper._park_radius
        mapper._actual_radius += radius_alpha * (target - mapper._actual_radius)
        first = mapper._actual_radius
        mapper._actual_radius += radius_alpha * (target - mapper._actual_radius)
        second = mapper._actual_radius

        self.assertGreater(first, mapper._park_radius)
        self.assertGreater(second, first)
        self.assertLess(second, target)

    def test_process_beat_returns_tcode_command(self):
        mapper = StrokeMapper(Config())
        event = self._event(is_beat=True, raw_rms=0.15, peak_energy=0.5)

        cmd = mapper.process_beat(event)

        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.duration_ms, 25)

    def test_process_beat_fourphase_attaches_direct_electrode_tags(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "classic"
        mapper = StrokeMapper(cfg)

        cmd = mapper.process_beat(self._event(is_beat=True, raw_rms=0.15, peak_energy=0.5))

        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertFalse(cmd.include_linear_axes)
        self.assertIsNotNone(mapper._last_fourphase_levels)
        assert mapper._last_fourphase_levels is not None
        for index, level in enumerate(mapper._last_fourphase_levels, start=1):
            tag = f"E{index}"
            self.assertIn(tag, cmd.tcode_tags)
            self.assertIn(f"{tag}_duration", cmd.tcode_tags)
            self.assertEqual(cmd.tcode_tags[f"{tag}_duration"], 25)
            self.assertEqual(cmd.tcode_tags[tag], int(max(0.0, min(1.0, level)) * 9999))

    def test_tetra3d_fourphase_levels_follow_mapper_orbit_state(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "tetra3d"
        cfg.live_fourphase_vertical_lift_mix = 0.9
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.92
        mapper._base_center_y = 0.30
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.8

        mapper._orbit_phase = 0.0
        phase_a = mapper.get_current_fourphase_levels()
        mapper._orbit_phase = float(math.pi / 2.0)
        phase_b = mapper.get_current_fourphase_levels()

        self.assertNotEqual(phase_a, phase_b)

    def test_tetra3d_vertical_lift_mix_is_user_tunable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "tetra3d"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.92
        mapper._base_center_y = 0.30
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.8

        cfg.live_fourphase_vertical_lift_mix = 0.0
        low_mix = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_vertical_lift_mix = 1.6
        high_mix = mapper.get_current_fourphase_levels()

        self.assertNotEqual(low_mix, high_mix)

    def test_tetra3d_center_drift_mix_is_user_tunable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "tetra3d"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.92
        mapper._base_center_y = 0.35
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.0

        cfg.live_fourphase_center_drift_mix = 0.0
        low_mix = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_center_drift_mix = 1.2
        high_mix = mapper.get_current_fourphase_levels()

        self.assertNotEqual(low_mix, high_mix)

    def test_tetra3d_post_projection_expansion_is_user_tunable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "tetra3d"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.18
        mapper.state.beta = -0.22
        mapper._actual_radius = 0.58
        mapper._base_center_y = 0.30
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.35

        cfg.live_fourphase_tetra_post_projection_expansion = 1.0
        base = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_tetra_post_projection_expansion = 1.6
        expanded = mapper.get_current_fourphase_levels()

        self.assertNotEqual(base, expanded)
        self.assertGreater(self._excursion_from_center(expanded), self._excursion_from_center(base))

    def test_tetra3d_trigger_bias_mix_is_user_tunable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "tetra3d"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.92
        mapper._base_center_y = mapper._baseline_center_y
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="downbeat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.0

        cfg.live_fourphase_trigger_bias_mix = 0.0
        low_mix = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_trigger_bias_mix = 2.0
        high_mix = mapper.get_current_fourphase_levels()

        self.assertNotEqual(low_mix, high_mix)

    def test_tetra3d_layout_model_is_user_assignable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "tetra3d"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.92
        mapper._base_center_y = 0.30
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.8

        cfg.live_fourphase_layout_model = "Straight Line"
        straight_layout = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_layout_model = "Pair At Top"
        pair_top_layout = mapper.get_current_fourphase_levels()

        self.assertNotEqual(straight_layout, pair_top_layout)

    def test_classic_radius_contrast_is_user_tunable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "classic"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5

        cfg.live_fourphase_beat_radius_contrast_strength = 0.0
        low_contrast = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_beat_radius_contrast_strength = 1.0
        high_contrast = mapper.get_current_fourphase_levels()

        self.assertNotEqual(low_contrast, high_contrast)

    def test_classic_speed_spread_is_user_tunable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "classic"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )

        cfg.live_fourphase_beat_speed_spread_strength = 0.0
        low_spread = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_beat_speed_spread_strength = 1.0
        high_spread = mapper.get_current_fourphase_levels()

        self.assertNotEqual(low_spread, high_spread)

    def test_classic_response_curves_are_user_assignable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "classic"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5

        cfg.live_fourphase_beat_response_curves = ["linear", "linear", "linear", "linear"]
        linear = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_beat_response_curves = ["bell", "linear", "linear", "linear"]
        bell = mapper.get_current_fourphase_levels()

        self.assertNotEqual(linear, bell)

    def test_bandrouter_mapping_is_user_assignable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "bandrouter"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.92
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.8
        mapper._intelligence.energies.bass = 0.6
        mapper._intelligence.energies.low_mid = 0.4
        mapper._intelligence.energies.mid = 0.1
        mapper._intelligence.energies.upper_mid = 0.0
        mapper._intelligence.energies.presence = 0.0
        mapper._intelligence.energies.brilliance = 0.0

        cfg.live_fourphase_band_mapping = [["mid", "upper_mid", "presence"], ["low_mid", "mid"], ["bass", "low_mid"], ["sub_bass", "bass"]]
        default_mapping = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_band_mapping = [["sub_bass"], ["bass"], ["mid"], ["brilliance"]]
        remapped = mapper.get_current_fourphase_levels()

        self.assertNotEqual(default_mapping, remapped)

    def test_tetra3d_vertical_lift_band_supports_seven_band_selection(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "tetra3d"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.92
        mapper._base_center_y = 0.30
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.1
        mapper._intelligence.energies.presence = 0.8

        cfg.live_fourphase_vertical_lift_band = "sub_bass"
        low_band = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_vertical_lift_band = "presence"
        high_band = mapper.get_current_fourphase_levels()

        self.assertNotEqual(low_band, high_band)

    def test_bandrouter_fill_mix_is_user_tunable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "bandrouter"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.92
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.8
        mapper._intelligence.energies.low_mid = 0.4
        mapper._intelligence.energies.mid = 0.1
        mapper._intelligence.energies.high = 0.0

        cfg.live_fourphase_bandrouter_fill_mix = 0.0
        low_fill = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_bandrouter_fill_mix = 0.45
        high_fill = mapper.get_current_fourphase_levels()

        self.assertNotEqual(low_fill, high_fill)

    def test_bandrouter_idle_floor_is_user_tunable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "bandrouter"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.72
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.8
        mapper._intelligence.energies.low_mid = 0.3
        mapper._intelligence.energies.mid = 0.1
        mapper._intelligence.energies.high = 0.0

        cfg.live_fourphase_bandrouter_idle_floor = 0.0
        low_idle = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_bandrouter_idle_floor = 0.25
        high_idle = mapper.get_current_fourphase_levels()

        self.assertNotEqual(low_idle, high_idle)

    def test_bandrouter_post_projection_expansion_is_user_tunable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "bandrouter"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.18
        mapper.state.beta = -0.22
        mapper._actual_radius = 0.55
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.55
        mapper._intelligence.energies.low_mid = 0.30
        mapper._intelligence.energies.mid = 0.15
        mapper._intelligence.energies.high = 0.05

        cfg.live_fourphase_bandrouter_post_projection_expansion = 1.0
        base = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_bandrouter_post_projection_expansion = 1.6
        expanded = mapper.get_current_fourphase_levels()

        self.assertNotEqual(base, expanded)
        self.assertGreater(self._excursion_from_center(expanded), self._excursion_from_center(base))

    def test_tetra3d_vertical_lift_curve_is_user_tunable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "tetra3d"
        cfg.live_fourphase_vertical_lift_band = "sub_bass"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.92
        mapper._base_center_y = 0.30
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.5

        cfg.live_fourphase_vertical_lift_curve = 0.5
        low_curve = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_vertical_lift_curve = 2.0
        high_curve = mapper.get_current_fourphase_levels()

        self.assertNotEqual(low_curve, high_curve)

    def test_tetra3d_vertical_lift_band_is_user_assignable(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "tetra3d"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.92
        mapper._base_center_y = 0.30
        mapper._orbit_phase = 0.0
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.2
        mapper._intelligence.energies.high = 0.8

        cfg.live_fourphase_vertical_lift_band = "sub_bass"
        sub_bass_levels = mapper.get_current_fourphase_levels()
        cfg.live_fourphase_vertical_lift_band = "high"
        high_levels = mapper.get_current_fourphase_levels()

        self.assertNotEqual(sub_bass_levels, high_levels)

    def test_classic_fourphase_levels_ignore_mapper_orbit_state(self):
        cfg = Config()
        cfg.live_tcode_mode = "fourphase"
        cfg.live_fourphase_model = "classic"
        mapper = StrokeMapper(cfg)
        mapper.state.alpha = 0.25
        mapper.state.beta = -0.5
        mapper._actual_radius = 0.92
        mapper._base_center_y = 0.30
        mapper._orbit_phase_initialized = True
        mapper._last_decision = BeatDecision(
            trigger_kind="beat",
            interval_beats=2,
            radius_bloom=0.9,
            silence_active=False,
            journey_completion=0.25,
        )
        mapper._intelligence.energies.sub_bass = 0.8

        mapper._orbit_phase = 0.0
        phase_a = mapper.get_current_fourphase_levels()
        mapper._orbit_phase = float(math.pi / 2.0)
        phase_b = mapper.get_current_fourphase_levels()

        self.assertEqual(phase_a, phase_b)

    def test_fill_parks_motion_when_jitter_off(self):
        """Creep mode parks when Jitter/Fill is disabled."""
        cfg = Config()
        cfg.jitter.enabled = False
        mapper = StrokeMapper(cfg)
        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=1.0,
            silence_active=False,
            journey_completion=0.3,
        )

        t0 = time.perf_counter()
        cmd = mapper.process_beat(self._event(is_beat=False, frequency=120.0, monotonic_timestamp=t0))

        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertAlmostEqual(cmd.alpha, 0.0, places=3)
        self.assertAlmostEqual(cmd.beta, mapper._park_y, places=3)

        # Simulate ~2 seconds at 60fps with proper timestamps
        for i in range(120):
            t = t0 + (i + 1) * (1.0 / 60.0)
            cmd = mapper.process_beat(self._event(is_beat=False, frequency=120.0, monotonic_timestamp=t))
        assert cmd is not None
        self.assertAlmostEqual(cmd.alpha, 0.0, places=3)
        self.assertAlmostEqual(cmd.beta, mapper._park_y, places=3)

    def test_compute_landing_rotation_from_park_for_beat_is_non_zero(self):
        mapper = StrokeMapper(Config())
        rot = mapper._compute_landing_rotation(start_angle=float(math.pi / 2.0), interval_beats=2)
        self.assertGreater(rot, 1.0)

    def test_silence_unknown_trigger_uses_baseline_center_default(self):
        mapper = StrokeMapper(Config())
        mapper._last_trigger_kind = "unknown"
        mapper._orbit_phase = 0.0
        mapper._actual_radius = 0.05

        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="unknown",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=True,
            journey_completion=1.0,
        )

        cmd = mapper.process_beat(self._event(is_beat=False, is_downbeat=False, is_syncopated=False))

        self.assertIsNotNone(cmd)
        assert cmd is not None
        # beta starts at park (0.20) and silence glides toward park, so
        # it should be at or very near 0.20 on the first silent frame.
        self.assertAlmostEqual(cmd.beta, 0.20, places=2)

    def test_silence_hard_parks_when_fill_active(self):
        cfg = Config()
        mapper = StrokeMapper(cfg)

        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=True,
            journey_completion=1.0,
            silence_fade=0.5,
        )

        # Run several frames so the quintic ease-to-park converges
        t0 = time.perf_counter()
        frame_dt = 1.0 / 60.0
        cmds = []
        for i in range(60):  # ~1 second of frames
            cmds.append(mapper.process_beat(self._event(
                is_beat=False, raw_rms=0.0, raw_rms_db=-80.0,
                monotonic_timestamp=t0 + i * frame_dt,
            )))

        cmd_last = cmds[-1]
        self.assertIsNotNone(cmd_last)
        assert cmd_last is not None
        # After 1s of silence, orbit should have spiralled close to park.
        # orbit_phase keeps turning so alpha won't be exactly zero,
        # but radius should be small (near park_idle_radius=0.05).
        self.assertLess(abs(cmd_last.alpha), 0.15)
        self.assertGreaterEqual(cmd_last.beta, -1.0)
        self.assertLessEqual(cmd_last.beta, 1.0)

    def test_base_center_target_migrates_start_over_full_journey(self):
        mapper = StrokeMapper(Config())

        self.assertAlmostEqual(mapper._base_center_target("start", 0.0, False), 0.20, places=6)
        self.assertAlmostEqual(mapper._base_center_target("start", 0.5, False), 0.10, places=6)
        self.assertAlmostEqual(mapper._base_center_target("start", 1.0, False), 0.0, places=6)

    def test_base_center_target_is_zero_for_active_journeys(self):
        mapper = StrokeMapper(Config())

        self.assertAlmostEqual(mapper._base_center_target("beat", 0.25, False), 0.0, places=6)
        self.assertAlmostEqual(mapper._base_center_target("downbeat", 0.5, False), 0.0, places=6)
        self.assertAlmostEqual(mapper._base_center_target("syncopation", 0.75, False), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
