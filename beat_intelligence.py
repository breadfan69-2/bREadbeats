from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from audio_engine import BeatEvent
from config import Config


@dataclass
class BandEnergies:
    sub_bass: float = 0.0
    low_mid: float = 0.0
    mid: float = 0.0
    high: float = 0.0


@dataclass
class BeatDecision:
    trigger_kind: str
    interval_beats: int
    radius_bloom: float
    silence_active: bool
    journey_completion: float


class BeatIntelligence:
    """Signal-domain decision engine for orbit control."""

    def __init__(self, config: Config, audio_engine=None, park_y: float = 0.70):
        self.config = config
        self.audio_engine = audio_engine
        self.park_y = 0.70 if park_y is None else float(park_y)

        self.band_ema_alpha = 0.2
        self.energies = BandEnergies()

        self.rms_envelope = 0.0
        self.rms_attack = 0.15
        self.rms_release = 0.05

        self.silence_deadzone_active = False
        self.silence_open_count = 0
        self.silence_close_count = 0

        self.active_interval_beats = 8
        self.last_trigger_kind = "creep"

        self.journey_duration_s = 0.0
        self.journey_elapsed_s = 0.0
        self.journey_active = False

        self.treble_lift_ema = 0.0
        self.treble_lift_attack = 0.28
        self.treble_lift_release = 0.16

        # ── Phase 1: Rolling history deques (#1) ──
        self._recent_flux_values: deque = deque(maxlen=60)
        self._recent_low_band_values: deque = deque(maxlen=60)
        self._recent_high_band_values: deque = deque(maxlen=60)
        self._recent_mid_bass_values: deque = deque(maxlen=60)
        self._recent_high_band_beat_hits: deque = deque(maxlen=16)

        # ── Phase 1: FluxTracker (#2) ──
        self._flux_history: deque = deque()  # (timestamp, flux) tuples
        self._flux_rise_window_ms: float = 250.0
        self._flux_stroke_factor: float = 1.0

        # ── Phase 1: Beat hierarchy guards (#3) ──
        self._last_any_beat_time: float = 0.0
        self._last_confirmed_beat_time: float = 0.0
        self._last_downbeat_call_time: float = 0.0
        self._last_beat_or_downbeat_call_time: float = 0.0
        self._last_downbeat_stroke_time: float = 0.0
        self._downbeat_chain_active: bool = False
        self._downbeat_chain_last_time: float = 0.0
        self._tempo_reset_motion_hold_s: float = 1.8
        self._tempo_reset_motion_hold_until: float = 0.0

        # ── Phase 1: No-beat timeout (#4) ──
        self._no_beat_timeout_s: float = 2.0

    def set_audio_engine(self, audio_engine) -> None:
        self.audio_engine = audio_engine

    def set_park_y(self, park_y: float) -> None:
        self.park_y = 0.70 if park_y is None else float(park_y)

    # ── Phase 1 §2: FluxTracker ──────────────────────────────────────

    def _update_flux_history(self, event: BeatEvent) -> None:
        now = float(getattr(event, "monotonic_timestamp", 0.0) or 0.0)
        if now <= 0.0:
            now = time.perf_counter()
        flux = float(getattr(event, "spectral_flux", 0.0) or 0.0)
        self._flux_history.append((now, flux))
        window_s = self._flux_rise_window_ms / 1000.0
        while self._flux_history and (now - self._flux_history[0][0]) > window_s:
            self._flux_history.popleft()

    def _get_flux_rise_factor(self) -> float:
        if len(self._flux_history) < 2:
            return 0.0
        oldest_flux = self._flux_history[0][1]
        newest_flux = self._flux_history[-1][1]
        rise = max(0.0, newest_flux - oldest_flux)
        return float(min(1.0, rise / 0.1))

    def _update_flux_stroke_factor(self, event: BeatEvent) -> None:
        flux_threshold = float(getattr(self.config.stroke, "flux_threshold", 0.03) or 0.03)
        scaling_weight = float(getattr(self.config.stroke, "flux_scaling_weight", 1.0) or 1.0)
        flux = float(getattr(event, "spectral_flux", 0.0) or 0.0)
        flux_ratio = float(np.clip(flux / max(flux_threshold, 0.001), 0.2, 3.0))
        base_factor = 0.5 + (flux_ratio / 3.0)
        self._flux_stroke_factor = 1.0 + (base_factor - 1.0) * scaling_weight

    # ── Phase 1 §21: Activity helpers ────────────────────────────────

    def _get_low_band_activity(self, event: BeatEvent) -> float:
        """Estimate low-band (sub_bass + low_mid) activity for this frame."""
        sub = float(np.clip(self.energies.sub_bass, 0.0, 1.0))
        low = float(np.clip(self.energies.low_mid, 0.0, 1.0))
        return float(max(sub, (sub * 0.6 + low * 0.4)))

    def _get_high_band_activity(self, event: BeatEvent) -> float:
        """Estimate upper-range (mid + high) activity for this frame."""
        mid = float(np.clip(self.energies.mid, 0.0, 1.0))
        high = float(np.clip(self.energies.high, 0.0, 1.0))
        return float(max(high, (mid * 0.3 + high * 0.7)))

    def _get_mid_bass_activity(self, event: BeatEvent) -> float:
        """Estimate 200-400 Hz mid-bass activity from low_mid band."""
        return float(np.clip(self.energies.low_mid, 0.0, 1.0))

    def _get_high_band_presence_status(self) -> bool:
        """Check if high-band has consistent presence over recent window."""
        window = int(getattr(self.config.stroke, "high_band_window_frames", 18))
        mean_thresh = float(getattr(self.config.stroke, "high_band_mean_threshold", 0.12))
        occ_thresh = float(getattr(self.config.stroke, "high_band_occupancy_threshold", 0.55))
        floor_thresh = float(getattr(self.config.stroke, "high_band_floor_threshold", 0.06))

        recent = list(self._recent_high_band_values)[-window:]
        if len(recent) < 8:
            return True  # insufficient data, don't block

        mean_val = float(np.mean(recent))
        if mean_val < mean_thresh:
            return False
        above_floor = sum(1 for v in recent if v >= floor_thresh)
        occupancy = above_floor / max(1, len(recent))
        return occupancy >= occ_thresh

    def _get_high_band_pattern_status(self) -> bool:
        """Check if recent beats had upper-band context (mid/high fired)."""
        window = int(getattr(self.config.stroke, "high_band_pattern_window_beats", 5))
        min_hits = int(getattr(self.config.stroke, "high_band_pattern_min_hits", 3))
        recent = list(self._recent_high_band_beat_hits)[-window:]
        if len(recent) < min_hits:
            return True  # insufficient data, don't block
        return sum(recent) >= min_hits

    # ── Phase 1 §3: Beat hierarchy guards ────────────────────────────

    def _has_recent_beats(self, now: float, window_s: float = 0.9) -> bool:
        """True if any beat/downbeat happened within window, or tempo-reset hold is active."""
        beat_recent = (
            self._last_any_beat_time > 0.0
            and (now - self._last_any_beat_time) <= window_s
        )
        reset_hold_active = now < self._tempo_reset_motion_hold_until
        return bool(beat_recent or reset_hold_active)

    def _arm_tempo_reset_motion_hold(self, now: float) -> None:
        """After tempo_reset, grant a grace window before requiring beats."""
        self._last_any_beat_time = now
        self._tempo_reset_motion_hold_until = now + self._tempo_reset_motion_hold_s

    def _record_beat_times(self, event: BeatEvent, trigger_kind: str, now: float) -> None:
        """Track timestamp of the most recent beat/downbeat/syncopation."""
        if bool(getattr(event, "tempo_reset", False)):
            self._arm_tempo_reset_motion_hold(now)

        is_beat = bool(getattr(event, "is_beat", False))
        is_downbeat = bool(getattr(event, "is_downbeat", False))
        is_syncopated = bool(getattr(event, "is_syncopated", False))

        if is_beat or is_downbeat or is_syncopated:
            self._last_any_beat_time = now

        if is_beat:
            self._last_confirmed_beat_time = now
            self._last_beat_or_downbeat_call_time = now

        if is_downbeat:
            self._last_downbeat_call_time = now
            self._last_beat_or_downbeat_call_time = now
            self._last_downbeat_stroke_time = now
            self._downbeat_chain_active = True
            self._downbeat_chain_last_time = now

    def _beat_hierarchy_allows(self, event: BeatEvent, trigger_kind: str, now: float, bpm: float) -> bool:
        """Enforce beat hierarchy: syncopation needs recent beat/downbeat, beat needs recent downbeat."""
        if trigger_kind not in ("syncopation", "beat"):
            return True

        beat_period_s = 60.0 / max(1e-6, bpm)
        prereq_window_s = max(2.0, beat_period_s * 2.5)

        if trigger_kind == "syncopation":
            # Syncopation requires recent beat or downbeat call
            recent_bd = (
                self._last_beat_or_downbeat_call_time > 0.0
                and (now - self._last_beat_or_downbeat_call_time) <= prereq_window_s
            )
            if not recent_bd:
                return False
            # Syncopation also requires recent downbeat stroke
            recent_ds = (
                self._last_downbeat_stroke_time > 0.0
                and (now - self._last_downbeat_stroke_time) <= prereq_window_s
            )
            return bool(recent_ds)

        if trigger_kind == "beat":
            # Beat requires recent downbeat stroke
            recent_ds = (
                self._last_downbeat_stroke_time > 0.0
                and (now - self._last_downbeat_stroke_time) <= prereq_window_s
            )
            return bool(recent_ds)

        return True

    # ── Phase 1 §7: Mid-trigger block ────────────────────────────────

    def _is_mid_trigger_blocked(self, event: BeatEvent) -> bool:
        """Block beats in vocal/guitar frequency range."""
        if not bool(getattr(self.config.stroke, "block_mid_trigger_range_enabled", True)):
            return False
        # Learning relax bypass
        if (bool(getattr(self.config.beat, "teaching_learning_enabled", False))
                and bool(getattr(self.config.beat, "teaching_relax_phase1_gates", False))):
            return False
        freq = float(getattr(event, "frequency", 0.0) or 0.0)
        if freq <= 0.0:
            return False
        low_hz = float(getattr(self.config.stroke, "block_mid_trigger_low_hz", 100.0))
        high_hz = float(getattr(self.config.stroke, "block_mid_trigger_high_hz", 2000.0))
        return bool(low_hz <= freq <= high_hz)

    # ── Phase 1 §4: No-beat timeout ──────────────────────────────────

    def _check_no_beat_timeout(self, now: float) -> bool:
        """Return True if we've had no beats for > timeout and should decay to park."""
        if self._last_any_beat_time <= 0.0:
            return False  # never had a beat, don't force timeout
        return (now - self._last_any_beat_time) > self._no_beat_timeout_s

    # ── Phase 1: deque population ────────────────────────────────────

    def _populate_rolling_deques(self, event: BeatEvent) -> None:
        """Append current-frame activity values to rolling history deques."""
        self._recent_flux_values.append(float(getattr(event, "spectral_flux", 0.0) or 0.0))
        self._recent_low_band_values.append(self._get_low_band_activity(event))
        self._recent_high_band_values.append(self._get_high_band_activity(event))
        self._recent_mid_bass_values.append(self._get_mid_bass_activity(event))

    def _record_high_band_beat_hit(self, event: BeatEvent, trigger_kind: str) -> None:
        """Record whether this beat event had high-band context."""
        if trigger_kind in ("beat", "downbeat", "syncopation"):
            fired = getattr(event, "fired_bands", None) or []
            fired_set = {str(b) for b in fired} if isinstance(fired, (list, tuple, set)) else set()
            beat_band = str(getattr(event, "beat_band", "") or "")
            include_mid = bool(getattr(self.config.stroke, "high_band_include_mid", True))
            hit = (
                "high" in fired_set
                or beat_band == "high"
                or (include_mid and ("mid" in fired_set or beat_band == "mid"))
            )
            self._recent_high_band_beat_hits.append(bool(hit))

    def update_band_energies(self) -> None:
        energies = {}
        if self.audio_engine is not None and hasattr(self.audio_engine, "_band_energies"):
            maybe = getattr(self.audio_engine, "_band_energies", None)
            if isinstance(maybe, dict):
                energies = maybe

        self.energies.sub_bass += (float(energies.get("sub_bass", 0.0)) - self.energies.sub_bass) * self.band_ema_alpha
        self.energies.low_mid += (float(energies.get("low_mid", 0.0)) - self.energies.low_mid) * self.band_ema_alpha
        self.energies.mid += (float(energies.get("mid", 0.0)) - self.energies.mid) * self.band_ema_alpha
        self.energies.high += (float(energies.get("high", 0.0)) - self.energies.high) * self.band_ema_alpha

    def update_envelope(self, event: BeatEvent) -> None:
        raw_rms = float(getattr(event, "raw_rms", 0.0) or 0.0)
        target = max(0.0, raw_rms)
        alpha = self.rms_attack if target >= self.rms_envelope else self.rms_release
        self.rms_envelope += (target - self.rms_envelope) * alpha

    def get_overall_amplitude(self, event: BeatEvent) -> float:
        # Keep silence-gate units aligned with console [Audio] raw_rms values.
        raw_rms = float(getattr(event, "raw_rms", 0.0) or 0.0)
        if raw_rms > 0.0:
            return float(max(0.0, raw_rms))
        return float(max(0.0, self.rms_envelope))

    def update_silence_deadzone_gate(self, overall_amplitude: float) -> bool:
        open_threshold = float(getattr(self.config.stroke, "silence_threshold", 0.04) or 0.04)
        open_threshold = max(0.0, open_threshold)
        close_raw = float(getattr(self.config.stroke, "silence_close_threshold", open_threshold * 1.20) or (open_threshold * 1.20))
        close_threshold = max(open_threshold + 1e-6, close_raw)

        if overall_amplitude < open_threshold:
            self.silence_open_count += 1
            self.silence_close_count = 0
            if self.silence_open_count >= 3:
                self.silence_deadzone_active = True
        elif overall_amplitude > close_threshold:
            self.silence_close_count += 1
            self.silence_open_count = 0
            if self.silence_close_count >= 2:
                self.silence_deadzone_active = False
        else:
            self.silence_open_count = max(0, self.silence_open_count - 1)
            self.silence_close_count = max(0, self.silence_close_count - 1)

        return self.silence_deadzone_active

    def classify_trigger(self, event: BeatEvent) -> str:
        if bool(getattr(event, "is_syncopated", False)) and bool(getattr(self.config.beat, "syncopation_enabled", True)):
            return "syncopation"
        if bool(getattr(event, "is_downbeat", False)):
            return "downbeat"
        if bool(getattr(event, "is_beat", False)):
            return "beat"
        return "creep"

    def _tempo_ready_for_motion(self, event: BeatEvent) -> bool:
        if not bool(getattr(self.config.beat, "tempo_lock_required", True)):
            return True
        if bool(getattr(event, "tempo_locked", False)):
            return True
        relaxed = float(getattr(self.config.beat, "teaching_metronome_relaxed_confidence", 0.14) or 0.14)
        acf_conf = float(getattr(event, "acf_confidence", 0.0) or 0.0)
        return acf_conf >= relaxed

    def _strict_bass_motion_allowed(self, event: BeatEvent, trigger_kind: str) -> bool:
        if trigger_kind not in ("beat", "syncopation"):
            return True
        if not bool(getattr(self.config.beat, "strict_bass_motion_gate_enabled", False)):
            return True

        beat_band = str(getattr(event, "beat_band", "") or "")
        if beat_band in ("sub_bass", "low_mid"):
            return True

        fired = getattr(event, "fired_bands", None)
        if isinstance(fired, (list, tuple, set)):
            fired_set = {str(item) for item in fired}
            if "sub_bass" in fired_set or "low_mid" in fired_set:
                return True

        return False

    @staticmethod
    def interval_beats_for_trigger(trigger_kind: str) -> int:
        if trigger_kind == "syncopation":
            return 1
        if trigger_kind == "beat":
            return 2
        if trigger_kind == "downbeat":
            return 4
        return 8

    @staticmethod
    def effective_bpm(event: BeatEvent) -> float:
        bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
        if bpm <= 0.0:
            bpm = float(getattr(event, "bpm", 0.0) or 0.0)
        if bpm <= 0.0:
            bpm = 120.0
        return float(np.clip(bpm, 40.0, 240.0))

    def compute_radius_bloom_from_sub_bass(self, event: BeatEvent | None = None) -> float:
        base_radius = 0.70
        max_radius = 0.95
        max_bloom = max_radius - base_radius

        sub_bass = float(np.clip(self.energies.sub_bass, 0.0, 1.0))
        low_mid = float(np.clip(self.energies.low_mid, 0.0, 1.0))

        weighted_bass = (sub_bass * 0.70) + (low_mid * 0.30)
        bass_fill = float(np.clip(max(sub_bass, weighted_bass), 0.0, 1.0))
        bloom = max_bloom * (bass_fill ** 1.5)

        if event is not None:
            spectral_flux = float(getattr(event, "spectral_flux", 0.0) or 0.0)
            flux_threshold = float(getattr(self.config.stroke, "flux_threshold", 0.03) or 0.03)
            flux_ratio = float(np.clip(spectral_flux / max(flux_threshold, 1e-6), 0.0, 2.0))
            flux_boost = max_bloom * 0.15 * flux_ratio
            bloom += flux_boost

        return float(np.clip(base_radius + bloom, base_radius, max_radius))

    def update_journey_progress(self, trigger_kind: str, interval_beats: int, event: BeatEvent, dt: float) -> float:
        bpm = self.effective_bpm(event)
        beat_period_s = 60.0 / max(1e-6, bpm)
        target_duration = max(1e-3, beat_period_s * float(interval_beats))

        trigger_started = bool(
            (trigger_kind == "syncopation" and bool(getattr(event, "is_syncopated", False)))
            or (trigger_kind == "downbeat" and bool(getattr(event, "is_downbeat", False)))
            or (trigger_kind == "beat" and bool(getattr(event, "is_beat", False)))
            or (trigger_kind == "creep" and self.last_trigger_kind != "creep" and not self.journey_active)
        )

        if trigger_started or not self.journey_active or self.active_interval_beats != interval_beats:
            self.journey_duration_s = target_duration
            self.journey_elapsed_s = 0.0
            self.journey_active = True
            return 0.0

        step = float(np.clip(dt, 1e-4, 0.25))
        self.journey_elapsed_s = min(self.journey_duration_s, self.journey_elapsed_s + step)
        completion = float(np.clip(self.journey_elapsed_s / max(1e-6, self.journey_duration_s), 0.0, 1.0))
        if completion >= 1.0:
            self.journey_active = False
        return completion

    def compute_treble_lift(self, journey_completion: float) -> float:
        max_lift = 0.40
        treble_fill = float(np.clip((self.energies.high * 0.75) + (self.energies.mid * 0.25), 0.0, 1.0))
        lift_factor = treble_fill ** 2.0
        target_offset = max_lift * lift_factor

        alpha = self.treble_lift_attack if target_offset >= self.treble_lift_ema else self.treble_lift_release
        self.treble_lift_ema += (target_offset - self.treble_lift_ema) * alpha
        smoothed_offset = float(np.clip(self.treble_lift_ema, 0.0, max_lift))

        guard_start = 0.80
        p = float(np.clip(journey_completion, 0.0, 1.0))
        if p <= guard_start:
            guard = 1.0
        else:
            t = float(np.clip((p - guard_start) / max(1e-6, 1.0 - guard_start), 0.0, 1.0))
            smooth_t = t * t * (3.0 - 2.0 * t)
            guard = 1.0 - smooth_t

        # Returns vertical center offset (0..max_lift), not absolute Y.
        # At journey completion, this is forced to 0 by the landing guard.
        return float(smoothed_offset * guard)

    def build_decision(self, event: BeatEvent, dt: float, silence_override: bool | None = None) -> BeatDecision:
        self.update_band_energies()
        self.update_envelope(event)

        # Phase 1: flux + deque tracking (every frame)
        self._update_flux_history(event)
        self._update_flux_stroke_factor(event)
        self._populate_rolling_deques(event)

        now = float(getattr(event, "monotonic_timestamp", 0.0) or 0.0)
        if now <= 0.0:
            now = time.perf_counter()

        overall_amplitude = self.get_overall_amplitude(event)
        silence_active = self.update_silence_deadzone_gate(overall_amplitude)
        if silence_override is not None:
            silence_active = bool(silence_override)

        raw_trigger_kind = self.classify_trigger(event)
        trigger_kind = raw_trigger_kind

        # Record beat times for hierarchy tracking
        self._record_beat_times(event, raw_trigger_kind, now)

        # Preserve active beat-family journeys between discrete beat/downbeat/sync events.
        if (
            raw_trigger_kind == "creep"
            and self.journey_active
            and self.last_trigger_kind in ("syncopation", "beat", "downbeat")
        ):
            trigger_kind = self.last_trigger_kind

        bpm = self.effective_bpm(event)

        # Tempo/bass/hierarchy gates apply to newly detected beat-family events.
        if raw_trigger_kind in ("syncopation", "beat", "downbeat") and not silence_active:
            if not self._tempo_ready_for_motion(event):
                trigger_kind = "creep"
            elif not self._strict_bass_motion_allowed(event, raw_trigger_kind):
                trigger_kind = "creep"
            elif self._is_mid_trigger_blocked(event):
                trigger_kind = "creep"
            elif not self._beat_hierarchy_allows(event, raw_trigger_kind, now, bpm):
                trigger_kind = "creep"

        # Record high-band beat hit for pattern gate
        self._record_high_band_beat_hit(event, trigger_kind)

        # No-beat timeout: force decay to park
        no_beat_timed_out = False
        if self._check_no_beat_timeout(now) and self.journey_active:
            trigger_kind = "creep"
            self.journey_active = False
            self.last_trigger_kind = "creep"
            self.active_interval_beats = 8
            no_beat_timed_out = True

        interval_beats = self.interval_beats_for_trigger(trigger_kind)
        radius_bloom = self.compute_radius_bloom_from_sub_bass(event=event)

        if no_beat_timed_out:
            journey_completion = 1.0  # fully parked
        else:
            journey_completion = self.update_journey_progress(trigger_kind, interval_beats, event, dt)

        self.active_interval_beats = interval_beats
        self.last_trigger_kind = trigger_kind

        return BeatDecision(
            trigger_kind=trigger_kind,
            interval_beats=interval_beats,
            radius_bloom=radius_bloom,
            silence_active=bool(silence_active),
            journey_completion=journey_completion,
        )
