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
    silence_fade: float = 1.0          # 1.0 = full volume, 0.0 = fully faded
    post_silence_ramp: float = 1.0     # 1.0 = full volume, <1 = ramping back in
    request_tempo_reset: bool = False  # edge-triggered: silence threshold crossed


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

        # ── Phase 2: ReadinessState (#17) ──
        self._stroke_ready: bool = False
        self._stroke_ready_reason: str = "cold_start"
        self._readiness_green_count: int = 0
        self._readiness_yellow_count: int = 0
        self._readiness_grace_until: float = 0.0
        self._readiness_block_streak: int = 0
        self._readiness_finish_beats_remaining: int = 0

        # ── Phase 2: SilenceDecayState (#19) ──
        self._silence_fade: float = 1.0            # 1.0 = full volume, 0.0 = muted
        self._consecutive_silent_count: int = 0
        self._silence_reset_armed: bool = False
        self._silence_fade_rate: float = 0.02       # fade per frame (~60fps → ~0.8s full fade)
        self._silence_reset_threshold_frames: int = 180  # ~3s at 60fps

        # ── Phase 2: Post-silence ramp (#14) ──
        self._post_silence_ramp_active: bool = False
        self._post_silence_ramp_start: float = 0.0
        self._was_silent: bool = False

        # ── Phase 3: Auto-fill adaptation (#20) ──
        self._auto_fill_offsets: dict[str, float] = {"downbeat": 0.0, "beat": 0.0, "syncopation": 0.0}
        self._auto_fill_ema: dict[str, float] = {"downbeat": 0.5, "beat": 0.5, "syncopation": 0.5}

    def set_audio_engine(self, audio_engine) -> None:
        self.audio_engine = audio_engine

    def set_park_y(self, park_y: float) -> None:
        self.park_y = 0.70 if park_y is None else float(park_y)

    # ── Phase 2 §17: Readiness state machine ─────────────────────────

    def _update_stroke_readiness(self, event: BeatEvent, now: float) -> bool:
        """Evaluate stroke readiness with hysteresis and grace period.

        Returns True if motion should be allowed.
        Brief confidence dips don't kill motion; sustained loss does.
        """
        grace_ms = float(getattr(self.config.beat, "teaching_stroke_ready_grace_ms", 450.0) or 450.0)
        _raw_finish = getattr(self.config.beat, "teaching_stroke_finish_beats", 4)
        finish_beats = int(_raw_finish) if _raw_finish is not None else 4

        # Raw readiness check (same logic as _tempo_ready_for_motion)
        raw_ready = self._tempo_ready_for_motion(event)

        if raw_ready:
            self._readiness_green_count += 1
            self._readiness_yellow_count = 0
            self._readiness_block_streak = 0
            self._readiness_grace_until = now + (grace_ms / 1000.0)
            self._readiness_finish_beats_remaining = finish_beats

            if self._readiness_green_count >= 1:
                self._stroke_ready = True
                self._stroke_ready_reason = "green"
        else:
            self._readiness_yellow_count += 1
            self._readiness_green_count = 0

            # Grace period: hold readiness through brief dips
            if now < self._readiness_grace_until:
                self._stroke_ready_reason = "grace"
                # Still ready, but tick down finish beats on beat events
                is_beat_event = bool(
                    getattr(event, "is_beat", False)
                    or getattr(event, "is_downbeat", False)
                    or getattr(event, "is_syncopated", False)
                )
                if is_beat_event and self._readiness_finish_beats_remaining > 0:
                    self._readiness_finish_beats_remaining -= 1
            elif self._readiness_finish_beats_remaining > 0:
                # Past grace but allow N more beat strokes
                self._stroke_ready_reason = "finishing"
                is_beat_event = bool(
                    getattr(event, "is_beat", False)
                    or getattr(event, "is_downbeat", False)
                    or getattr(event, "is_syncopated", False)
                )
                if is_beat_event:
                    self._readiness_finish_beats_remaining -= 1
            else:
                self._readiness_block_streak += 1
                if self._readiness_block_streak >= 3:
                    self._stroke_ready = False
                    self._stroke_ready_reason = "blocked"

        return self._stroke_ready

    # ── Phase 2 §19: Silence fade-out tracker ────────────────────────

    def _update_silence_fade(self, silence_active: bool, now: float) -> tuple[float, bool]:
        """Track prolonged silence: emit fade scalar and tempo-reset request.

        Returns (fade_scalar 0..1, request_tempo_reset bool).
        """
        request_reset = False

        if silence_active:
            self._consecutive_silent_count += 1
            # Gradual fade
            self._silence_fade = max(0.0, self._silence_fade - self._silence_fade_rate)
            # Tempo reset after prolonged silence (once per silence episode)
            if (self._consecutive_silent_count >= self._silence_reset_threshold_frames
                    and not self._silence_reset_armed):
                self._silence_reset_armed = True
                request_reset = True
        else:
            if self._consecutive_silent_count > 0:
                # Transition from silent → active
                self._was_silent = True
            self._consecutive_silent_count = 0
            self._silence_reset_armed = False
            # Restore fade (quick recovery)
            self._silence_fade = min(1.0, self._silence_fade + self._silence_fade_rate * 3.0)

        return float(self._silence_fade), request_reset

    # ── Phase 2 §14: Post-silence volume ramp ────────────────────────

    def _update_post_silence_ramp(self, silence_active: bool, now: float) -> float:
        """Return volume multiplier for post-silence ramp-in.

        When audio resumes after silence, start volume reduced and linearly
        ramp back to 1.0 over configured duration.
        """
        reduction = float(getattr(self.config.stroke, "post_silence_vol_reduction", 0.15) or 0.15)
        ramp_seconds = float(getattr(self.config.stroke, "post_silence_ramp_seconds", 3.0) or 3.0)
        ramp_seconds = max(0.1, ramp_seconds)

        if not silence_active and self._was_silent:
            # Kick off ramp
            self._post_silence_ramp_active = True
            self._post_silence_ramp_start = now
            self._was_silent = False

        if self._post_silence_ramp_active:
            elapsed = now - self._post_silence_ramp_start
            if elapsed >= ramp_seconds:
                self._post_silence_ramp_active = False
                return 1.0
            t = float(np.clip(elapsed / ramp_seconds, 0.0, 1.0))
            return float((1.0 - reduction) + (reduction * t))

        if silence_active:
            self._was_silent = True
            return 1.0  # fade tracker handles volume during silence

        return 1.0

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
    # ── Phase 3 §5: Low-band fullness gate ────────────────────────────────

    def _is_low_band_full_enough(self, event: BeatEvent, trigger_kind: str, bpm: float) -> bool:
        """Low-band fullness gate (#5): require sustained low-band activity.

        Uses rolling deque history to enforce minimum mean, occupancy, and
        low/high ratio.  Downbeats get relaxed thresholds.  Falls back to
        mid-bass support when low-band alone is insufficient.
        """
        cfg = self.config.stroke
        window = int(getattr(cfg, 'low_band_window_frames', 18))
        threshold = float(getattr(cfg, 'low_band_activity_threshold', 0.20))
        occ_threshold = float(getattr(cfg, 'low_band_fullness_occupancy_threshold', 0.62))
        ratio_min = float(getattr(cfg, 'low_band_to_high_ratio_min', 0.58))

        if trigger_kind == "downbeat":
            relax = float(getattr(cfg, 'downbeat_low_band_relax', 0.85))
            threshold *= relax
            occ_threshold *= relax
            ratio_min *= relax

        recent_low = list(self._recent_low_band_values)[-window:]

        # Insufficient data: don't block
        if len(recent_low) < 8:
            return True

        # No meaningful signal data (e.g. no audio engine): don't block
        max_val = max(recent_low) if recent_low else 0.0
        if max_val < 1e-6:
            return True

        mean_low = float(np.mean(recent_low))
        if mean_low < threshold:
            if bool(getattr(cfg, 'mid_bass_support_enabled', True)):
                return self._mid_bass_support_passes(trigger_kind)
            return False

        # Occupancy: fraction of frames above floor
        floor_val = 0.70 * threshold
        above_floor = sum(1 for v in recent_low if v >= floor_val)
        occupancy = above_floor / max(1, len(recent_low))
        if occupancy < occ_threshold:
            if bool(getattr(cfg, 'mid_bass_support_enabled', True)):
                return self._mid_bass_support_passes(trigger_kind)
            return False

        # Low/high ratio: prevent treble-only content from passing
        recent_high = list(self._recent_high_band_values)[-window:]
        if len(recent_high) >= 8:
            mean_high = float(np.mean(recent_high))
            if mean_high > 1e-6 and (mean_low / mean_high) < ratio_min:
                if bool(getattr(cfg, 'mid_bass_support_enabled', True)):
                    return self._mid_bass_support_passes(trigger_kind)
                return False

        return True

    def _mid_bass_support_passes(self, trigger_kind: str) -> bool:
        """Mid-bass (200-400 Hz) fallback when low-band gate fails."""
        cfg = self.config.stroke
        mb_threshold = float(getattr(cfg, 'mid_bass_activity_threshold', 0.035))
        mb_occ_threshold = float(getattr(cfg, 'mid_bass_occupancy_threshold', 0.45))
        window = int(getattr(cfg, 'low_band_window_frames', 18))

        recent_mb = list(self._recent_mid_bass_values)[-window:]
        if len(recent_mb) < 8:
            return True  # insufficient data, don't block

        max_val = max(recent_mb) if recent_mb else 0.0
        if max_val < 1e-6:
            return True  # no signal data

        mean_mb = float(np.mean(recent_mb))
        if mean_mb < mb_threshold:
            return False

        floor_val = 0.70 * mb_threshold
        above_floor = sum(1 for v in recent_mb if v >= floor_val)
        occupancy = above_floor / max(1, len(recent_mb))
        return occupancy >= mb_occ_threshold

    # ── Phase 3 §6: Dual-band dB gate ────────────────────────────────────

    def _passes_dual_band_db_gate(self, event: BeatEvent) -> bool:
        """Dual-band dB gate (#6): require sub-bass AND high-band energy.

        Both bands must exceed their dB minimum.  Has event-frequency
        fallback and high-tip fullness sub-gate.
        """
        cfg = self.config.stroke
        if not bool(getattr(cfg, 'dual_band_db_gate_enabled', True)):
            return True

        # Learning relax bypass
        if (bool(getattr(self.config.beat, 'teaching_learning_enabled', False))
                and bool(getattr(self.config.beat, 'teaching_relax_phase1_gates', False))):
            return True

        # No meaningful energy data: don't block
        total_energy = (abs(self.energies.sub_bass) + abs(self.energies.low_mid)
                        + abs(self.energies.mid) + abs(self.energies.high))
        if total_energy < 1e-6:
            return True

        sub_bass_db_min = float(getattr(cfg, 'dual_band_sub_bass_db_min', -15.0))
        high_db_min = float(getattr(cfg, 'dual_band_high_db_min', -30.0))

        sub_bass_energy = max(1e-10, float(self.energies.sub_bass))
        high_energy = max(1e-10, float(self.energies.high))

        sub_bass_db = float(20.0 * np.log10(sub_bass_energy))
        high_db = float(20.0 * np.log10(high_energy))

        passes = (sub_bass_db >= sub_bass_db_min and high_db >= high_db_min)

        if not passes:
            # Event frequency fallback: infer band from event frequency
            freq = float(getattr(event, 'frequency', 0.0) or 0.0)
            peak_energy = float(getattr(event, 'peak_energy', 0.0) or 0.0)
            if peak_energy > 0.01:
                if 20.0 <= freq <= 120.0 and sub_bass_db < sub_bass_db_min:
                    passes = (high_db >= high_db_min)
                elif freq > 3500.0 and high_db < high_db_min:
                    passes = (sub_bass_db >= sub_bass_db_min)

        # High-tip fullness sub-gate
        if passes and bool(getattr(cfg, 'high_tip_fullness_enabled', True)):
            passes = self._high_tip_fullness_passes()

        return passes

    def _high_tip_fullness_passes(self) -> bool:
        """High-tip fullness sub-gate for dual-band dB gate."""
        cfg = self.config.stroke
        occ_threshold = float(getattr(cfg, 'high_tip_occupancy_threshold', 0.50))
        db_min = float(getattr(cfg, 'high_tip_db_min', -28.0))
        window = int(getattr(cfg, 'low_band_window_frames', 18))

        recent = list(self._recent_high_band_values)[-window:]
        if len(recent) < 8:
            return True  # insufficient data

        max_val = max(recent) if recent else 0.0
        if max_val < 1e-6:
            return True  # no signal data

        linear_min = 10.0 ** (db_min / 20.0)  # -28 dB → ~0.0398
        above = sum(1 for v in recent if v >= linear_min)
        occupancy = above / max(1, len(recent))
        return occupancy >= occ_threshold

    # ── Phase 3 §8: Spectrum fill gate ───────────────────────────────────

    def _get_spectrum_fill_ratio(self, trigger_kind: str) -> float:
        """Compute spectrum fill ratio from live FFT for given phase (#8)."""
        if self.audio_engine is None:
            return 1.0  # no engine, don't block

        spectrum = None
        if hasattr(self.audio_engine, 'get_spectrum'):
            spectrum = self.audio_engine.get_spectrum()

        if spectrum is None:
            return 1.0

        magnitudes = np.abs(np.asarray(spectrum, dtype=float))
        if magnitudes.size == 0:
            return 1.0

        peak = float(np.max(magnitudes))
        if peak < 1e-10:
            return 0.0

        cfg = self.config.stroke
        phase_map = {
            "downbeat": ("downbeat_fill_bin_low", "downbeat_fill_bin_high"),
            "beat": ("beat_fill_bin_low", "beat_fill_bin_high"),
            "syncopation": ("syncopation_fill_bin_low", "syncopation_fill_bin_high"),
        }
        low_key, high_key = phase_map.get(trigger_kind, ("beat_fill_bin_low", "beat_fill_bin_high"))
        low_bin = int(getattr(cfg, low_key, 0))
        high_bin = int(getattr(cfg, high_key, 512))

        high_bin = min(high_bin + 1, magnitudes.size)
        low_bin = max(0, min(low_bin, high_bin - 1))
        band = magnitudes[low_bin:high_bin]

        if band.size == 0:
            return 0.0

        norm = band / peak
        active_floor = 0.02
        active_mask = norm >= active_floor
        active = norm[active_mask]

        if active.size == 0:
            return 0.0

        threshold = float(getattr(cfg, 'overall_amp_fill_target', 0.5))
        filled = float(np.sum(active >= threshold))
        return float(filled / max(1, active.size))

    def _passes_overall_amp_fill_gate(self, event: BeatEvent, trigger_kind: str) -> bool:
        """Overall amplitude fill gate (#8): require spectral fill for phase."""
        cfg = self.config.stroke
        if not bool(getattr(cfg, 'overall_amp_fill_gate_enabled', True)):
            return True

        # No audio engine: don't block
        if self.audio_engine is None:
            return True

        intensity = float(getattr(event, 'intensity', 0.0) or 0.0)
        target = float(getattr(cfg, 'overall_amp_fill_target', 0.5))
        tolerance = float(getattr(cfg, 'overall_amp_fill_tolerance', 0.5))

        if intensity < (target - tolerance):
            return False

        fill_ratio = self._get_spectrum_fill_ratio(trigger_kind)
        required = self._get_overall_amp_fill_required(trigger_kind)

        passed = fill_ratio >= required
        self._update_auto_fill_required(trigger_kind, passed)

        return passed

    def _get_overall_amp_fill_required(self, trigger_kind: str) -> float:
        """Get fill required for phase, including auto-adapt offset (#20)."""
        cfg = self.config.stroke
        base_map = {
            "downbeat": float(getattr(cfg, 'downbeat_overall_amp_fill_required', 0.75)),
            "beat": float(getattr(cfg, 'beat_overall_amp_fill_required', 0.90)),
            "syncopation": float(getattr(cfg, 'syncopation_overall_amp_fill_required', 1.00)),
        }
        base = base_map.get(trigger_kind, base_map.get("beat", 0.90))
        offset = self._auto_fill_offsets.get(trigger_kind, 0.0)

        min_req = float(getattr(cfg, 'overall_amp_fill_auto_min_required', 0.05))
        max_req = float(getattr(cfg, 'overall_amp_fill_auto_max_required', 0.98))

        return float(np.clip(base + offset, min_req, max_req))

    def _update_auto_fill_required(self, trigger_kind: str, gate_passed: bool) -> None:
        """Auto-fill adaptation (#20): adjust offset per phase targeting pass rate."""
        cfg = self.config.stroke
        if not bool(getattr(cfg, 'overall_amp_fill_auto_enabled', True)):
            return

        alpha = float(getattr(cfg, 'overall_amp_fill_auto_ema_alpha', 0.12))
        target_rate = float(getattr(cfg, 'overall_amp_fill_auto_target_pass_rate', 0.58))
        deadband = float(getattr(cfg, 'overall_amp_fill_auto_deadband', 0.06))
        step = float(getattr(cfg, 'overall_amp_fill_auto_step', 0.02))
        max_offset = float(getattr(cfg, 'overall_amp_fill_auto_max_offset', 0.35))

        current_ema = self._auto_fill_ema.get(trigger_kind, 0.5)
        sample = 1.0 if gate_passed else 0.0
        new_ema = current_ema + alpha * (sample - current_ema)
        self._auto_fill_ema[trigger_kind] = new_ema

        error = new_ema - target_rate
        current_offset = self._auto_fill_offsets.get(trigger_kind, 0.0)

        if abs(error) > deadband:
            if error > 0:
                # Passing too often → tighten (increase required)
                current_offset += step
            else:
                # Passing too rarely → relax (decrease required)
                current_offset -= step
            current_offset = float(np.clip(current_offset, -max_offset, max_offset))
            self._auto_fill_offsets[trigger_kind] = current_offset

    # ── Phase 3 §16: Flux-drop guard ─────────────────────────────────────

    def _check_flux_drop_guard(self, now: float) -> bool:
        """Flux-drop guard (#16): block if low-band energy dropped sharply.

        When recent tail drops below drop_ratio of full window average,
        and no confirmed beats were recent, downgrade to creep.
        """
        cfg = self.config.stroke
        if not bool(getattr(cfg, 'low_band_drop_guard_enabled', True)):
            return False

        drop_ratio = float(getattr(cfg, 'flux_drop_ratio', 0.25))

        recent = list(self._recent_low_band_values)
        if len(recent) < 30:
            return False

        full_mean = float(np.mean(recent))
        tail_mean = float(np.mean(recent[-8:]))

        if full_mean < 1e-6:
            return False

        ratio = tail_mean / full_mean
        if ratio < drop_ratio:
            # Only downgrade when no recent confirmed beats
            if self._has_recent_beats(now, window_s=0.9):
                return False
            return True

        return False
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

        # Phase 2: silence fade + post-silence ramp
        silence_fade, request_tempo_reset = self._update_silence_fade(silence_active, now)
        post_silence_ramp = self._update_post_silence_ramp(silence_active, now)

        # Phase 2: readiness state machine (replaces raw _tempo_ready_for_motion)
        stroke_ready = self._update_stroke_readiness(event, now)

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
            if not stroke_ready:
                trigger_kind = "creep"
            elif not self._strict_bass_motion_allowed(event, raw_trigger_kind):
                trigger_kind = "creep"
            elif self._is_mid_trigger_blocked(event):
                trigger_kind = "creep"
            elif not self._beat_hierarchy_allows(event, raw_trigger_kind, now, bpm):
                trigger_kind = "creep"
            # Phase 3 gates: low-band → dual-band → spectrum fill → flux-drop
            elif not self._is_low_band_full_enough(event, raw_trigger_kind, bpm):
                trigger_kind = "creep"
            elif not self._passes_dual_band_db_gate(event):
                trigger_kind = "creep"
            elif not self._passes_overall_amp_fill_gate(event, raw_trigger_kind):
                trigger_kind = "creep"
            elif self._check_flux_drop_guard(now):
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
            silence_fade=float(silence_fade),
            post_silence_ramp=float(post_silence_ramp),
            request_tempo_reset=bool(request_tempo_reset),
        )
