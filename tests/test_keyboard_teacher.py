"""Tests for keyboard_teacher.py – the human-in-the-loop teaching recorder."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pytest

from keyboard_teacher import (
    KeyboardTeacher,
    ConditionTracker,
    _snapshot_from_event,
    _snapshot_from_decision,
)


# ── Minimal stubs ──────────────────────────────────────────────────

@dataclass
class FakeBeatEvent:
    intensity: float = 0.5
    frequency: float = 120.0
    is_beat: bool = True
    spectral_flux: float = 0.3
    peak_energy: float = 0.4
    is_downbeat: bool = False
    bpm: float = 128.0
    tempo_locked: bool = True
    beat_band: str = "sub_bass"
    metronome_bpm: float = 128.0
    acf_confidence: float = 0.8
    is_syncopated: bool = False
    raw_rms: float = 0.01
    raw_rms_db: float = -40.0
    fired_bands: list = field(default_factory=lambda: ["sub_bass", "low_mid"])
    beat_features: Optional[dict] = None
    monotonic_timestamp: float = 0.0


@dataclass
class FakeBeatDecision:
    trigger_kind: str = "beat"
    interval_beats: int = 4
    radius_bloom: float = 0.85
    silence_active: bool = False
    journey_completion: float = 0.5
    silence_fade: float = 1.0
    post_silence_ramp: float = 1.0
    lazy_glide_active: bool = False
    gate_fail: str = ""
    energy_fullness: float = 0.6
    session_intensity: float = 0.5
    park_bounce_only: bool = False
    park_bounce_gain: float = 0.0


def _make_gate_state(**overrides) -> dict:
    """Return a minimal gate-state dict with sensible defaults."""
    gs = {
        "gs_sub_bass": 0.1,
        "gs_low_mid": 0.05,
        "gs_mid": 0.03,
        "gs_high": 0.02,
        "gs_silence_active": 0,
        "gs_last_trigger_kind": "creep",
    }
    gs.update(overrides)
    return gs


# ── Tests ──────────────────────────────────────────────────────────

class TestSnapshotHelpers:
    def test_snapshot_from_event_extracts_fields(self):
        ev = FakeBeatEvent(bpm=140.0, fired_bands=["sub_bass"])
        snap = _snapshot_from_event(ev)
        assert snap["bpm"] == 140.0
        assert snap["fired_bands"] == "sub_bass"
        assert snap["is_beat"] is True

    def test_snapshot_from_event_flattens_beat_features(self):
        ev = FakeBeatEvent(beat_features={"rms": 0.02, "bus_scores": {"kick": 0.9, "hat": 0.1}})
        snap = _snapshot_from_event(ev)
        assert snap["bf_rms"] == 0.02
        assert snap["bf_bus_scores_kick"] == 0.9
        assert snap["bf_bus_scores_hat"] == 0.1

    def test_snapshot_from_decision(self):
        dec = FakeBeatDecision(trigger_kind="syncopation", energy_fullness=0.8)
        snap = _snapshot_from_decision(dec)
        assert snap["dec_trigger_kind"] == "syncopation"
        assert snap["dec_energy_fullness"] == 0.8


class TestKeyboardTeacherLifecycle:
    def test_start_creates_directory(self, tmp_path: Path):
        teacher = KeyboardTeacher(base_dir=tmp_path)
        session = teacher.start_session()
        assert session.exists()
        assert teacher.active

    def test_stop_flushes_csv(self, tmp_path: Path):
        teacher = KeyboardTeacher(base_dir=tmp_path)
        teacher.start_session()
        ev = FakeBeatEvent()
        teacher.on_frame(ev, dt=1 / 60)
        teacher.on_frame(ev, dt=1 / 60)
        saved = teacher.stop_session()
        assert saved is not None
        csv_path = saved / "directives.csv"
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["directive"] == "park"

    def test_stop_when_inactive_returns_none(self, tmp_path: Path):
        teacher = KeyboardTeacher(base_dir=tmp_path)
        assert teacher.stop_session() is None

    def test_double_start_resets(self, tmp_path: Path):
        teacher = KeyboardTeacher(base_dir=tmp_path)
        s1 = teacher.start_session()
        teacher.on_frame(FakeBeatEvent(), dt=1 / 60)
        # Second start should reset and remain active (same-second path is fine)
        s2 = teacher.start_session()
        assert teacher.active
        assert teacher._frame_count == 0  # frame counter was reset


class TestArrowKeyTracking:
    def test_held_keys_update(self, tmp_path: Path):
        """Discrete model: keydown latches state, keyup is a no-op."""
        teacher = KeyboardTeacher(base_dir=tmp_path)
        teacher.start_session()
        # Start parked
        assert teacher.is_parked
        # → = unpark at 1x
        teacher.on_arrow_down("right")
        assert not teacher.is_parked
        assert teacher.speed_step == 0
        # ↓ = park again
        teacher.on_arrow_down("down")
        assert teacher.is_parked
        # ↑ = unpark (no speed change)
        teacher.on_arrow_down("up")
        assert not teacher.is_parked
        # on_arrow_up is a no-op
        teacher.on_arrow_up("right")
        assert not teacher.is_parked

    def test_intensity_axis_ramps(self, tmp_path: Path):
        """Discrete: multiple right-presses increase speed_step (not a smooth axis)."""
        teacher = KeyboardTeacher(base_dir=tmp_path)
        teacher.start_session()
        teacher.on_arrow_down("right")  # leave park at 1x
        assert teacher.speed_step == 0
        teacher.on_arrow_down("right")  # 2x
        assert teacher.speed_step == 1
        teacher.on_arrow_down("right")  # 4x
        assert teacher.speed_step == 2
        # speed_scale should match 2**step
        assert teacher.speed_scale == pytest.approx(4.0)

    def test_speed_axis_ramps(self, tmp_path: Path):
        """Discrete: right from park = step 0 (1x); right again = step 1 (2x); left = halve."""
        teacher = KeyboardTeacher(base_dir=tmp_path)
        teacher.start_session()
        teacher.on_arrow_down("right")  # leave park at step 0
        assert teacher.speed_step == 0
        teacher.on_arrow_down("right")  # step 1
        assert teacher.speed_scale == pytest.approx(2.0)
        teacher.on_arrow_down("left")   # back to step 0
        assert teacher.speed_scale == pytest.approx(1.0)
        teacher.on_arrow_down("left")   # step -1 = 0.5x
        assert teacher.speed_scale == pytest.approx(0.5)

    def test_opposing_keys_cancel(self, tmp_path: Path):
        """Discrete: left while already parked goes to step -1; right later un-parks at 1x."""
        teacher = KeyboardTeacher(base_dir=tmp_path)
        teacher.start_session()
        # pre-set step to -1 while still parked
        teacher.on_arrow_down("left")
        assert teacher.is_parked
        assert teacher.speed_step == -1
        # right from park: leave park at step 0 (ignores pre-set step per design)
        teacher.on_arrow_down("right")
        assert not teacher.is_parked
        assert teacher.speed_step == 0

    def test_directive_label_changes(self, tmp_path: Path):
        teacher = KeyboardTeacher(base_dir=tmp_path)
        teacher.start_session()
        assert teacher.current_directive == "park"
        teacher.on_arrow_down("right")    # leave park at 1x
        assert teacher.current_directive == "1x"
        teacher.on_arrow_down("right")    # 2x
        assert teacher.current_directive == "2x"
        teacher.on_arrow_down("down")     # back to park
        assert teacher.current_directive == "park"
        teacher.on_arrow_down("up")       # unpark (speed stays 2x from before)
        assert teacher.current_directive == "2x"


class TestCSVOutput:
    def test_csv_contains_audio_fields(self, tmp_path: Path):
        teacher = KeyboardTeacher(base_dir=tmp_path)
        teacher.start_session()
        ev = FakeBeatEvent(bpm=140.0, raw_rms_db=-35.0)
        dec = FakeBeatDecision(trigger_kind="downbeat", radius_bloom=0.9)
        teacher.on_frame(ev, dec, dt=1 / 60)
        saved = teacher.stop_session()
        csv_path = saved / "directives.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        row = rows[0]
        assert row["bpm"] == "140.0"
        assert row["dec_trigger_kind"] == "downbeat"
        assert "is_parked" in row
        assert "speed_step" in row
        assert "session_time_s" in row

    def test_csv_contains_condition_tracker_fields(self, tmp_path: Path):
        teacher = KeyboardTeacher(base_dir=tmp_path)
        teacher.start_session()
        ev = FakeBeatEvent(spectral_flux=0.3, metronome_bpm=120.0)
        dec = FakeBeatDecision(gate_fail="", trigger_kind="beat")
        gs = _make_gate_state(gs_sub_bass=0.15, gs_low_mid=0.1)
        teacher.on_frame(ev, dec, gs, dt=1 / 60)
        saved = teacher.stop_session()
        csv_path = saved / "directives.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        row = rows[0]
        # Condition tracker cols should exist
        assert "ct_flux_ema" in row
        assert "ct_flux_delta" in row
        assert "ct_since_flux_rise_s" in row
        assert "ct_since_flux_fall_s" in row
        assert "ct_since_sub_bass_arrive_s" in row
        assert "ct_since_silence_exit_s" in row
        assert "ct_beats_since_flux_rise" in row
        assert "ct_current_gate_fail" in row
        assert "ct_since_trigger_change_s" in row
        assert "ct_beat_count" in row

    def test_csv_contains_gate_state_fields(self, tmp_path: Path):
        teacher = KeyboardTeacher(base_dir=tmp_path)
        teacher.start_session()
        ev = FakeBeatEvent()
        gs = _make_gate_state(gs_sub_bass=0.2, gs_stroke_ready=1)
        teacher.on_frame(ev, None, gs, dt=1 / 60)
        saved = teacher.stop_session()
        csv_path = saved / "directives.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        row = rows[0]
        assert row["gs_sub_bass"] == "0.2"
        assert row["gs_stroke_ready"] == "1"

    def test_auto_flush_preserves_data(self, tmp_path: Path):
        teacher = KeyboardTeacher(base_dir=tmp_path)
        teacher.start_session()
        ev = FakeBeatEvent()
        # Push past the auto-flush threshold
        for _ in range(3010):
            teacher.on_frame(ev, dt=1 / 60)
        # Some rows should already be flushed (auto-flush at 3000)
        csv_path = teacher.session_dir / "directives.csv"
        assert csv_path.exists()
        saved = teacher.stop_session()
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3010


class TestConditionTracker:
    """Focused tests for the floating temporal feature engine."""

    def test_flux_ema_tracks_spectral_flux(self):
        ct = ConditionTracker()
        ev = FakeBeatEvent(spectral_flux=1.0)
        for _ in range(50):
            out = ct.update(ev)
        # EMA should converge toward 1.0
        assert out["ct_flux_ema"] > 0.8

    def test_flux_rise_detected(self):
        ct = ConditionTracker()
        # Warm up EMA with low flux
        low_ev = FakeBeatEvent(spectral_flux=0.1)
        for _ in range(30):
            ct.update(low_ev)
        # Spike flux high
        high_ev = FakeBeatEvent(spectral_flux=0.5)
        out = ct.update(high_ev)
        # since_flux_rise should be very small (just happened)
        assert out["ct_since_flux_rise_s"] < 0.1

    def test_band_arrival_tracked(self):
        ct = ConditionTracker()
        gs_low = _make_gate_state(gs_sub_bass=0.01)
        gs_high = _make_gate_state(gs_sub_bass=0.2)
        ev = FakeBeatEvent()
        # Start with low energy — not present
        for _ in range(30):
            ct.update(ev, gate_state=gs_low)
        # Raise energy above threshold
        for _ in range(20):
            out = ct.update(ev, gate_state=gs_high)
        # sub_bass should be "present" now (EMA rises above 0.08)
        assert out["ct_sub_bass_present"] == 1
        assert out["ct_since_sub_bass_arrive_s"] < 1.0

    def test_silence_transition_tracking(self):
        ct = ConditionTracker()
        ev = FakeBeatEvent()
        active_dec = FakeBeatDecision(silence_active=True)
        inactive_dec = FakeBeatDecision(silence_active=False)

        # Start non-silent
        ct.update(ev, inactive_dec)
        # Enter silence
        out = ct.update(ev, active_dec)
        assert out["ct_since_silence_enter_s"] < 0.1
        # Exit silence
        out = ct.update(ev, inactive_dec)
        assert out["ct_since_silence_exit_s"] < 0.1

    def test_beat_counting(self):
        ct = ConditionTracker()
        beat_ev = FakeBeatEvent(is_beat=True)
        no_beat_ev = FakeBeatEvent(is_beat=False)

        ct.update(beat_ev)
        ct.update(beat_ev)
        out = ct.update(no_beat_ev)
        assert out["ct_beat_count"] == 2

    def test_beats_since_uses_bpm(self):
        ct = ConditionTracker()
        ev = FakeBeatEvent(metronome_bpm=120.0, spectral_flux=0.1)
        # Run for a moment to establish EMA
        for _ in range(20):
            ct.update(ev)
        out = ct.update(ev)
        # With bpm=120, beat_period=0.5s. beats_since values should be positive
        assert out["ct_beats_since_flux_rise"] >= 0
        assert out["ct_beats_since_silence_exit"] >= 0
        assert out["ct_beats_since_gate_open"] >= 0

    def test_gate_fail_transitions(self):
        ct = ConditionTracker()
        ev = FakeBeatEvent()

        # Initially no gate fail
        ct.update(ev, FakeBeatDecision(gate_fail=""))
        # Gate fails
        out = ct.update(ev, FakeBeatDecision(gate_fail="strict_bass"))
        assert out["ct_current_gate_fail"] == "strict_bass"
        assert out["ct_since_gate_close_s"] < 0.1
        # Gate opens again
        out = ct.update(ev, FakeBeatDecision(gate_fail=""))
        assert out["ct_current_gate_fail"] == ""
        assert out["ct_since_gate_open_s"] < 0.1

    def test_trigger_kind_transition(self):
        ct = ConditionTracker()
        ev = FakeBeatEvent()

        ct.update(ev, FakeBeatDecision(trigger_kind="creep"))
        out = ct.update(ev, FakeBeatDecision(trigger_kind="beat"))
        assert out["ct_since_trigger_change_s"] < 0.1

    def test_no_bpm_returns_sentinel(self):
        ct = ConditionTracker()
        ev = FakeBeatEvent(metronome_bpm=0.0, bpm=0.0)
        out = ct.update(ev)
        assert out["ct_beats_since_flux_rise"] == -1.0

    def test_reset_clears_state(self):
        ct = ConditionTracker()
        ev = FakeBeatEvent(spectral_flux=1.0)
        for _ in range(50):
            ct.update(ev)
        ct.reset()
        ev2 = FakeBeatEvent(spectral_flux=0.01)
        out = ct.update(ev2)
        # After reset, EMA should be near 0 again
        assert out["ct_flux_ema"] < 0.01


class TestNoRecordingWhenInactive:
    def test_frames_ignored_when_not_active(self, tmp_path: Path):
        teacher = KeyboardTeacher(base_dir=tmp_path)
        # Don't start session
        teacher.on_frame(FakeBeatEvent(), dt=1 / 60)
        teacher.on_arrow_down("up")
        teacher.on_arrow_up("up")
        assert not teacher.active
        assert teacher._rows == []
