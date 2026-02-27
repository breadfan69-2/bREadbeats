"""
Phase 2 checkpoint tests — ReadinessState (#17), SilenceDecayState (#19),
post-silence volume ramp (#14).
"""

import time
import unittest
from typing import Any

from audio_engine import BeatEvent
from beat_intelligence import BeatDecision, BeatIntelligence
from config import Config
from stroke_mapper import StrokeMapper


class Phase2Mixin:
    """Shared event factory for Phase 2 tests."""

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
            tempo_locked=True,
            metronome_bpm=120.0,
            is_syncopated=False,
            monotonic_timestamp=time.perf_counter(),
            raw_rms=0.08,
            beat_band="sub_bass",
            fired_bands=["sub_bass"],
        )
        payload.update(overrides)
        return BeatEvent(**payload)


# ── §17 ReadinessState machine ─────────────────────────────────────────


class TestReadinessState(Phase2Mixin, unittest.TestCase):
    def test_cold_start_not_ready(self):
        bi = BeatIntelligence(Config())
        self.assertFalse(bi._stroke_ready)

    def test_one_green_frame_makes_ready(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        event = self._event(tempo_locked=True, monotonic_timestamp=now)
        bi._update_stroke_readiness(event, now)
        self.assertTrue(bi._stroke_ready)
        self.assertEqual(bi._stroke_ready_reason, "green")

    def test_brief_dip_holds_readiness_via_grace(self):
        cfg = Config()
        cfg.beat.teaching_stroke_ready_grace_ms = 500.0
        bi = BeatIntelligence(cfg)
        now = time.perf_counter()

        # Establish readiness
        bi._update_stroke_readiness(self._event(tempo_locked=True, monotonic_timestamp=now), now)
        self.assertTrue(bi._stroke_ready)

        # Dip: confidence too low, but within grace window
        dip_now = now + 0.1  # 100ms later, well within 500ms grace
        bi._update_stroke_readiness(
            self._event(tempo_locked=False, acf_confidence=0.0, monotonic_timestamp=dip_now),
            dip_now,
        )
        self.assertTrue(bi._stroke_ready)
        self.assertIn(bi._stroke_ready_reason, ("grace", "finishing"))

    def test_sustained_loss_kills_readiness(self):
        cfg = Config()
        cfg.beat.teaching_stroke_ready_grace_ms = 50.0  # very short grace
        cfg.beat.teaching_stroke_finish_beats = 0
        bi = BeatIntelligence(cfg)
        now = time.perf_counter()

        # Establish
        bi._update_stroke_readiness(self._event(tempo_locked=True, monotonic_timestamp=now), now)
        self.assertTrue(bi._stroke_ready)

        # Sustained loss: well past grace, 0 finish beats
        for i in range(10):
            late = now + 1.0 + (i * 0.02)
            bi._update_stroke_readiness(
                self._event(tempo_locked=False, acf_confidence=0.0, monotonic_timestamp=late),
                late,
            )
        self.assertFalse(bi._stroke_ready)
        self.assertEqual(bi._stroke_ready_reason, "blocked")

    def test_readiness_integrates_into_build_decision(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = True
        cfg.beat.teaching_metronome_relaxed_confidence = 0.5
        bi = BeatIntelligence(cfg)
        bi._post_silence_entry_complete = True

        # Locked event → readiness kicks in → downbeat fires
        decision = bi.build_decision(
            self._event(is_downbeat=True, tempo_locked=True), dt=1 / 60, silence_override=False
        )
        self.assertEqual(decision.trigger_kind, "downbeat")

    def test_readiness_blocks_with_tempo_lock_required_and_no_confidence(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = True
        cfg.beat.teaching_metronome_relaxed_confidence = 0.5
        bi = BeatIntelligence(cfg)

        decision = bi.build_decision(
            self._event(is_downbeat=True, tempo_locked=False, acf_confidence=0.0),
            dt=1 / 60, silence_override=False,
        )
        self.assertEqual(decision.trigger_kind, "creep")

    def test_legacy_metronome_only_mode_allows_motion_with_relaxed_confidence(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = True
        cfg.beat.teaching_ignore_traffic_lights = True
        cfg.beat.teaching_metronome_relaxed_confidence = 0.14
        bi = BeatIntelligence(cfg)
        bi._post_silence_entry_complete = True

        decision = bi.build_decision(
            self._event(
                is_downbeat=True,
                tempo_locked=False,
                acf_confidence=0.20,
                metronome_bpm=120.0,
            ),
            dt=1 / 60,
            silence_override=False,
        )
        self.assertEqual(decision.trigger_kind, "downbeat")

    def test_legacy_metronome_only_mode_still_requires_metronome_bpm(self):
        cfg = Config()
        cfg.beat.tempo_lock_required = True
        cfg.beat.teaching_ignore_traffic_lights = True
        cfg.beat.teaching_metronome_relaxed_confidence = 0.14
        bi = BeatIntelligence(cfg)

        decision = bi.build_decision(
            self._event(
                is_downbeat=True,
                tempo_locked=False,
                acf_confidence=0.80,
                metronome_bpm=0.0,
            ),
            dt=1 / 60,
            silence_override=False,
        )
        self.assertEqual(decision.trigger_kind, "creep")


# ── Silence gate: simple fixed-threshold ───────────────────────────────


class TestSilenceGateThreshold(Phase2Mixin, unittest.TestCase):
    """Fixed-threshold silence gate with hysteresis."""

    def _feed(self, bi, level_db: float, frames: int):
        """Feed constant dBFS for N frames."""
        for _ in range(frames):
            bi.update_silence_deadzone_gate(level_db)

    def test_starts_silent(self):
        """Gate starts in silence (guilty-until-proven)."""
        bi = BeatIntelligence(Config())
        self.assertTrue(bi.silence_deadzone_active)

    def test_loud_signal_exits_silence(self):
        """Sustained signal above exit threshold opens the gate."""
        bi = BeatIntelligence(Config())
        self.assertTrue(bi.silence_deadzone_active)
        # Feed well above exit threshold (-48 dB) for enough frames
        self._feed(bi, -30.0, frames=20)
        self.assertFalse(bi.silence_deadzone_active)

    def test_quiet_signal_enters_silence(self):
        """Signal dropping below enter threshold closes the gate."""
        bi = BeatIntelligence(Config())
        # First exit silence
        self._feed(bi, -30.0, frames=20)
        self.assertFalse(bi.silence_deadzone_active)
        # Now drop to silence well below enter threshold
        self._feed(bi, -80.0, frames=20)
        self.assertTrue(bi.silence_deadzone_active)

    def test_hysteresis_band_holds_state(self):
        """Signal in the hysteresis band does not change state."""
        bi = BeatIntelligence(Config())
        # Exit silence first
        self._feed(bi, -30.0, frames=20)
        self.assertFalse(bi.silence_deadzone_active)
        # Feed midpoint between enter/exit thresholds — state should hold.
        mid_db = (bi._silence_enter_db + bi._silence_exit_db) * 0.5
        self._feed(bi, mid_db, frames=60)
        self.assertFalse(bi.silence_deadzone_active)

    def test_near_floor_stays_silent(self):
        """Very quiet signal (-80 dB) stays silent even after many frames."""
        bi = BeatIntelligence(Config())
        self._feed(bi, -80.0, frames=200)
        self.assertTrue(bi.silence_deadzone_active)

    def test_deep_silence_fast_path_enters_quickly(self):
        """Near-digital silence should re-enter silence quickly (few frames)."""
        bi = BeatIntelligence(Config())
        # Exit silence first so we can validate re-entry timing.
        self._feed(bi, -20.0, frames=20)
        self.assertFalse(bi.silence_deadzone_active)

        # Standard gate needs more frames; deep-silence path should trigger in ~3.
        self._feed(bi, -100.0, frames=3)
        self.assertTrue(bi.silence_deadzone_active)


# ── §19 SilenceDecayState ──────────────────────────────────────────────


class TestSilenceDecayState(Phase2Mixin, unittest.TestCase):
    def test_silence_fade_decreases_during_silence(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        bi._silence_fade = 1.0

        for _ in range(10):
            bi._update_silence_fade(silence_active=True, now=now)
        self.assertLess(bi._silence_fade, 1.0)

    def test_silence_fade_recovers_after_silence(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        bi._silence_fade = 0.3

        for _ in range(20):
            bi._update_silence_fade(silence_active=False, now=now)
        self.assertGreater(bi._silence_fade, 0.3)

    def test_prolonged_silence_triggers_tempo_reset(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()

        any_reset = False
        for _ in range(200):
            _, reset = bi._update_silence_fade(silence_active=True, now=now)
            if reset:
                any_reset = True
        self.assertTrue(any_reset)

    def test_tempo_reset_fires_only_once_per_silence(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()

        reset_count = 0
        for _ in range(300):
            _, reset = bi._update_silence_fade(silence_active=True, now=now)
            if reset:
                reset_count += 1
        self.assertEqual(reset_count, 1)

    def test_silence_reset_ms_live_update_takes_effect(self):
        cfg = Config()
        cfg.beat.silence_reset_ms = 180
        bi = BeatIntelligence(cfg)
        now = time.perf_counter()

        cfg.beat.silence_reset_ms = 1000
        bi._update_silence_fade(silence_active=True, now=now)

        expected = max(1, int(round((1000 / 1000.0) * 60.0)))
        self.assertEqual(bi._silence_reset_threshold_frames, expected)

    def test_silence_fade_in_decision(self):
        bi = BeatIntelligence(Config())
        # During silence, fade should be < 1
        for _ in range(20):
            decision = bi.build_decision(self._event(raw_rms=0.0), dt=1 / 60, silence_override=True)
        self.assertLess(decision.silence_fade, 1.0)

    def test_decision_has_silence_fade_field(self):
        bi = BeatIntelligence(Config())
        decision = bi.build_decision(self._event(), dt=1 / 60)
        self.assertTrue(hasattr(decision, "silence_fade"))
        self.assertTrue(hasattr(decision, "post_silence_ramp"))


# ── §14 Post-silence volume ramp ───────────────────────────────────────


class TestPostSilenceRamp(Phase2Mixin, unittest.TestCase):
    def test_ramp_starts_reduced_after_silence(self):
        cfg = Config()
        cfg.stroke.post_silence_vol_reduction = 0.20
        cfg.stroke.post_silence_ramp_seconds = 2.0
        cfg.stroke.silence_threshold = 0.002
        bi = BeatIntelligence(cfg)

        now = time.perf_counter()
        # Enter true silence-open band (below threshold) to arm ramp
        bi._was_silent = False
        bi._update_silence_fade(silence_active=True, now=now, overall_amplitude=0.001)
        # Exit silence
        ramp = bi._update_post_silence_ramp(silence_active=False, now=now)
        # Should start at (1 - reduction)
        self.assertAlmostEqual(ramp, 0.80, delta=0.01)

    def test_ramp_reaches_full_after_duration(self):
        cfg = Config()
        cfg.stroke.post_silence_vol_reduction = 0.20
        cfg.stroke.post_silence_ramp_seconds = 2.0
        bi = BeatIntelligence(cfg)

        now = time.perf_counter()
        bi._was_silent = True
        bi._update_post_silence_ramp(silence_active=False, now=now)

        # After full ramp duration
        ramp = bi._update_post_silence_ramp(silence_active=False, now=now + 3.0)
        self.assertAlmostEqual(ramp, 1.0, places=4)

    def test_no_ramp_when_no_prior_silence(self):
        bi = BeatIntelligence(Config())
        now = time.perf_counter()
        ramp = bi._update_post_silence_ramp(silence_active=False, now=now)
        self.assertAlmostEqual(ramp, 1.0, places=4)

    def test_ramp_midpoint_is_between_reduced_and_full(self):
        cfg = Config()
        cfg.stroke.post_silence_vol_reduction = 0.20
        cfg.stroke.post_silence_ramp_seconds = 2.0
        bi = BeatIntelligence(cfg)

        now = time.perf_counter()
        bi._was_silent = True
        bi._update_post_silence_ramp(silence_active=False, now=now)

        ramp = bi._update_post_silence_ramp(silence_active=False, now=now + 1.0)
        self.assertGreater(ramp, 0.80)
        self.assertLess(ramp, 1.0)

    def test_decision_post_silence_ramp_field(self):
        bi = BeatIntelligence(Config())
        decision = bi.build_decision(self._event(), dt=1 / 60)
        self.assertAlmostEqual(decision.post_silence_ramp, 1.0, places=4)

    def test_silence_active_arms_ramp(self):
        """Flatness model: silence_active=True always arms _was_silent."""
        cfg = Config()
        bi = BeatIntelligence(cfg)

        now = time.perf_counter()
        bi._update_silence_fade(silence_active=True, now=now, overall_amplitude=0.004)
        self.assertTrue(bi._was_silent)

    def test_open_threshold_arms_ramp(self):
        cfg = Config()
        cfg.stroke.silence_threshold = 0.002
        bi = BeatIntelligence(cfg)

        now = time.perf_counter()
        bi._update_silence_fade(silence_active=True, now=now, overall_amplitude=0.001)
        self.assertTrue(bi._was_silent)

    def test_silence_reset_ms_arms_ramp(self):
        cfg = Config()
        cfg.stroke.silence_threshold = 0.002
        cfg.beat.silence_reset_ms = 180
        bi = BeatIntelligence(cfg)

        now = time.perf_counter()
        bi._was_silent = False
        frames = max(1, bi._silence_reset_threshold_frames)
        for _ in range(frames):
            bi._update_silence_fade(silence_active=True, now=now, overall_amplitude=0.01)
        self.assertTrue(bi._was_silent)


# ── StrokeMapper integration ───────────────────────────────────────────


class TestStrokeMapperPhase2Integration(Phase2Mixin, unittest.TestCase):
    def test_silence_uses_fade_not_binary_zero(self):
        """During silence with partial fade, volume should be > 0 but < full."""
        mapper = StrokeMapper(Config(), get_volume=lambda: 1.0)
        # Partially faded silence
        mapper._intelligence._silence_fade = 0.5
        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="creep",
            interval_beats=8,
            radius_bloom=0.70,
            silence_active=True,
            journey_completion=1.0,
            silence_fade=0.5,
        )
        cmd = mapper.process_beat(self._event(raw_rms=0.0))
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertGreater(cmd.volume, 0.0)
        self.assertLess(cmd.volume, 1.0)

    def test_post_silence_ramp_reduces_volume(self):
        """After silence, volume should be reduced by ramp multiplier."""
        mapper = StrokeMapper(Config(), get_volume=lambda: 1.0)
        mapper._intelligence.build_decision = lambda event, dt, silence_override=None: BeatDecision(
            trigger_kind="downbeat",
            interval_beats=4,
            radius_bloom=0.85,
            silence_active=False,
            journey_completion=0.5,
            post_silence_ramp=0.85,
        )
        cmd = mapper.process_beat(self._event(is_downbeat=True, raw_rms=0.1))
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertAlmostEqual(cmd.volume, 0.85, delta=0.01)


# ── Layer-2 / Layer-3 silence suppression tests ────────────────────────


class TestSilenceGatePropagation(Phase2Mixin, unittest.TestCase):
    """Layer 2: BeatIntelligence writes silence_gate_active back to AudioEngine."""

    def _make_bi_with_engine(self):
        """Create BeatIntelligence with a lightweight AudioEngine stub."""
        cfg = Config()
        bi = BeatIntelligence(cfg)
        # Minimal stub with the silence_gate_active attribute
        class _EngineStub:
            silence_gate_active: bool = True
        engine = _EngineStub()
        bi.audio_engine = engine
        return bi, engine

    def test_silence_propagates_to_engine(self):
        """When BeatIntelligence says silence, engine.silence_gate_active is True."""
        bi, engine = self._make_bi_with_engine()
        # Feed flat noise to stay in silence
        for _ in range(30):
            bi.build_decision(self._event(raw_rms=0.0001), dt=1/60)
        self.assertTrue(engine.silence_gate_active)

    def test_music_clears_engine_silence_flag(self):
        """When BeatIntelligence exits silence, engine.silence_gate_active becomes False."""
        bi, engine = self._make_bi_with_engine()
        # Prime with stable low noise
        for _ in range(30):
            bi.build_decision(self._event(raw_rms=0.0001), dt=1/60)
        self.assertTrue(engine.silence_gate_active)

        # Feed loud music dynamics (high raw_rms → high dBFS with variation)
        import random
        random.seed(99)
        for _ in range(120):
            rms = random.choice([0.05, 0.15, 0.08, 0.20, 0.03, 0.18])
            bi.build_decision(self._event(raw_rms=rms), dt=1/60)
        self.assertFalse(engine.silence_gate_active)


class TestBeatEnergyFloor(unittest.TestCase):
    """Layer 3: Absolute energy floor in _detect_beat."""

    def _make_engine(self):
        """Create minimal AudioEngine for _detect_beat testing."""
        from audio_engine import AudioEngine
        from config import Config
        engine = AudioEngine.__new__(AudioEngine)
        cfg = Config()
        engine.config = cfg
        engine.energy_history = []
        engine.flux_history = []
        engine._last_beat_time = 0
        engine._prev_energy_for_valley = 0.0
        engine._energy_was_falling = False
        engine._valley_history = []
        engine._valley_max_samples = 50
        engine._primary_beat_band = "sub_bass"
        engine._band_zscore_signals = {"sub_bass": 0, "low_mid": 0, "mid": 0, "high": 0}
        engine._metronome_bpm = 0.0
        return engine

    def test_noise_level_energy_rejected(self):
        """Band energy < 0.001 should never count as a beat."""
        engine = self._make_engine()
        # Even if we have history that would make 0.0005 pass the adaptive
        # threshold, the absolute floor should block it.
        engine.energy_history = [0.0002] * 20
        engine.flux_history = [0.0001] * 20
        result = engine._detect_beat(0.0005, 0.0001)
        self.assertFalse(result)

    def test_real_energy_can_pass(self):
        """Band energy above floor should be evaluated normally."""
        engine = self._make_engine()
        # Prime with low history so a big spike exceeds adaptive threshold
        engine.energy_history = [0.002] * 20
        engine.flux_history = [0.002] * 20
        engine.tempo_tracking_enabled = False
        engine.smoothed_tempo = 0.0
        engine.last_known_tempo = 120.0
        engine.beat_intervals = []
        engine.beat_times = []
        # A spike well above average should trigger
        result = engine._detect_beat(0.05, 0.05)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
