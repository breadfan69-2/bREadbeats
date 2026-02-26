from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from audio_engine import BeatEvent, RMS_DB_FLOOR, rms_to_dbfs, silence_threshold_to_dbfs
from config import Config
from logging_utils import log_event


@dataclass
class BandEnergies:
    sub_bass: float = 0.0
    low_mid: float = 0.0
    mid: float = 0.0
    high: float = 0.0


@dataclass
class LearningOutputs:
    """Cue-based speed prediction from the learning adapter.

    The model answers one question: given recent audio, how fast should
    motion be?  ``speed_mult`` (0 = still, 1 = full speed) is the sole
    continuous output.  ``cadence_hint`` selects beats-between-strokes.
    """
    speed_mult: float = 0.5            # 0..1 target motion speed
    cadence_hint: int = 1              # beats-between-strokes (1, 2, or 4)
    active: bool = False               # whether learning produced valid output


@dataclass
class BeatDecision:
    trigger_kind: str
    interval_beats: int
    radius_bloom: float
    silence_active: bool
    journey_completion: float
    silence_fade: float = 1.0          # 1.0 = full volume, 0.0 = fully faded
    post_silence_ramp: float = 1.0     # 1.0 = full volume, <1 = ramping back in
    lazy_glide_active: bool = False
    gate_fail: str = ""                # which gate rejected a beat-family event (empty = passed or N/A)
    energy_fullness: float = 0.0       # 0..1 how "full" the music is (drives max_radius expansion)
    session_intensity: float = 0.5     # long-term session energy envelope (0..1)

    learning: LearningOutputs = field(default_factory=LearningOutputs)


class BeatIntelligence:
    """Signal-domain decision engine for orbit control."""

    def __init__(self, config: Config, audio_engine=None, park_y: float = 0.70):
        self.config = config
        self.audio_engine = audio_engine
        self.park_y = 0.70 if park_y is None else float(park_y)

        self.band_ema_alpha = 0.2
        self.energies = BandEnergies()

        self.rms_envelope = RMS_DB_FLOOR
        self.rms_attack = 0.15
        self.rms_release = 0.05

        self.silence_deadzone_active = False
        self.silence_open_count = 0
        self.silence_close_count = 0
        self._silence_default_enter_db = -66.0
        self._silence_default_exit_db = -58.0

        self.active_interval_beats = 8
        self.last_trigger_kind = "creep"

        # Journey preservation safety: count consecutive beat-family events
        # that failed a gate but were preserved by the "let it finish" path.
        # After N consecutive failures, let the journey expire to fill so
        # a permanently stuck gate doesn't loop the orbit forever.
        self._gate_fail_preserve_count: int = 0
        self._gate_fail_preserve_limit: int = 2  # max consecutive beat-fail preservations

        self.journey_duration_s = 0.0
        self.journey_elapsed_s = 0.0
        self.journey_active = False
        self.is_recovering: bool = False
        self._was_silence_active: bool = False
        self._recovery_radius_bloom: float = 0.70
        self._journey_duration_target_s = 0.0
        self._journey_duration_blend_frames_remaining = 0
        self._journey_duration_blend_alpha = 0.35
        self._journey_start_intensity = 0.0  # intensity of current journey's trigger
        self._lazy_glide_active: bool = False

        # ── Phase 1: Rolling history deques (#1) ──
        self._recent_flux_values: deque = deque(maxlen=600)  # ~10 s for lookback features
        self._recent_low_band_values: deque = deque(maxlen=60)
        self._recent_mid_band_values: deque = deque(maxlen=60)
        self._recent_high_band_values: deque = deque(maxlen=60)
        self._recent_mid_bass_values: deque = deque(maxlen=60)

        # Rolling RMS history for dynamic amp-gate (adapts to current volume)
        self._recent_rms_db: deque = deque(maxlen=600)  # ~10 s at 60 fps

        # Rolling band energy history for volume-adaptive normalization.
        # Raw band energies scale with OS volume; these deques let us normalize
        # each band against its own recent P95 so music "intensity" is volume-independent.
        self._band_energy_history: dict[str, deque] = {
            "sub_bass": deque(maxlen=600),
            "low_mid": deque(maxlen=600),
            "mid": deque(maxlen=600),
            "high": deque(maxlen=600),
        }

        # ── Phase 1: FluxTracker (#2) ──
        self._flux_history: deque = deque()  # (timestamp, flux) tuples
        self._flux_rise_window_ms: float = 250.0

        # ── Phase 1: Beat timing (#3) ──
        self._last_any_beat_time: float = 0.0

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
        self._silence_fade_rate: float = 0.035      # fade per frame (~60fps → ~0.5s full fade)
        silence_reset_ms = int(getattr(getattr(self.config, "beat", None), "silence_reset_ms", 180) or 180)
        self._silence_reset_threshold_frames: int = max(1, int(round((silence_reset_ms / 1000.0) * 60.0)))

        # ── Phase 2: Post-silence ramp (#14) ──
        self._post_silence_ramp_active: bool = False
        self._post_silence_ramp_start: float = 0.0
        self._was_silent: bool = False

        # ── Phase 3: Auto-fill adaptation (#20) ──
        self._auto_fill_offsets: dict[str, float] = {"downbeat": 0.0, "beat": 0.0, "syncopation": 0.0}
        self._auto_fill_ema: dict[str, float] = {"downbeat": 0.5, "beat": 0.5, "syncopation": 0.5}
        self._fill_pass_consecutive: dict[str, int] = {"downbeat": 0, "beat": 0, "syncopation": 0}  # Sustained fill tracking

        # ── dBFS-based fill gate: Absolute signal reference tracking ──
        self._dbfs_reference_max: float = 1e-10  # Recent maximum signal magnitude (for dBFS calculation)
        self._dbfs_reference_last_update: float = 0.0  # Timestamp of last reference update

        # ── Phase 5: Learning Adapter (#10-12, #18) ──
        self._learning_enabled: bool = False
        self._learning_use_fitted_rules: bool = False
        self._learning_strength: float = 0.55
        self._learning_min_confidence: float = 0.12
        self._learning_no_motion_bias: float = 1.0
        self._learning_model: Optional[dict] = None
        self._learning_model_loaded: bool = False
        self._learning_norm_mean: dict[str, float] = {}
        self._learning_norm_std: dict[str, float] = {}
        self._learning_feature_columns: list[str] = []
        self._learning_cadence_rule: dict = {}
        self._learning_outputs: LearningOutputs = LearningOutputs()
        # Blended output fields (EMA-smoothed)
        self._learned_speed_mult: float = 0.5
        self._learned_cadence_hint: int = 1
        self._committed_divisor_hint: int = 1   # only applied at journey start

        # ── Phrase Commitment: measure-locked high-gear hold ──
        self._phrase_committed: bool = False
        self._phrase_beats_remaining: int = 0
        self._phrase_gear: str = ""             # "beat" or "syncopation"
        self._phrase_flux_baseline: float = 0.0  # mean flux when phrase entered
        self._phrase_flux_drop_ratio: float = 0.35  # cancel if flux drops below 35% of baseline
        self._phrase_renew_ratio: float = 0.55       # renew if flux still >= 55% at measure end
        self._phrase_measure_beats: int = 8          # beats per phrase commitment (two measures for stability)

        # ── Phase 6: BPM Stabilization (#13) ──
        self._last_locked_bpm: float = 120.0       # last BPM when tempo_locked was True
        self._stabilized_bpm: float = 120.0         # EMA-smoothed BPM output
        self._bpm_jump_ratio_limit: float = 1.5     # max allowed jump ratio per update

        # ── §5: Tempo-unlock hold ──
        # When we had mild confidence but metronome is not green,
        # hold the last cadence until cancelled by flux spike/drop or silence.
        self._tempo_unlock_hold_active: bool = False
        self._tempo_unlock_hold_bpm: float = 120.0
        self._tempo_unlock_hold_flux_baseline: float = 0.0
        self._tempo_unlock_hold_flux_spike_ratio: float = 2.0   # cancel if flux > 2x baseline
        self._tempo_unlock_hold_flux_drop_ratio: float = 0.25   # cancel if flux < 25% of baseline

        # ── Session arc: very slow energy tracking ──
        self._session_intensity_ema: float = 0.5

    @staticmethod
    def _linear_to_dbfs(value: float) -> float:
        return rms_to_dbfs(float(value), floor_db=RMS_DB_FLOOR)

    @staticmethod
    def _dbfs_to_unit(value_db: float, floor_db: float = RMS_DB_FLOOR) -> float:
        clipped = float(np.clip(value_db, floor_db, 0.0))
        return float(np.clip((clipped - floor_db) / max(1e-9, -floor_db), 0.0, 1.0))

    def _event_rms_db(self, event: BeatEvent) -> float:
        raw_rms_db = float(getattr(event, "raw_rms_db", RMS_DB_FLOOR) or RMS_DB_FLOOR)
        if np.isfinite(raw_rms_db) and raw_rms_db > RMS_DB_FLOOR:
            return float(np.clip(raw_rms_db, RMS_DB_FLOOR, 12.0))
        raw_rms = float(getattr(event, "raw_rms", 0.0) or 0.0)
        if raw_rms > 0.0:
            return self._linear_to_dbfs(raw_rms)
        return RMS_DB_FLOOR

    def _coerce_amplitude_db(self, amplitude: float | None) -> float:
        if amplitude is None:
            return RMS_DB_FLOOR
        value = float(amplitude)
        if not np.isfinite(value):
            return RMS_DB_FLOOR
        if 0.0 < value <= 1.0:
            return self._linear_to_dbfs(value)
        return float(np.clip(value, RMS_DB_FLOOR, 12.0))


    def set_audio_engine(self, audio_engine) -> None:
        self.audio_engine = audio_engine

    def set_park_y(self, park_y: float) -> None:
        self.park_y = 0.70 if park_y is None else float(park_y)

    # ── Phase 5: Learning Adapter ────────────────────────────────────

    def configure_learning(
        self,
        *,
        enabled: bool,
        use_fitted_rules: bool,
        strength: float,
        min_confidence: float,
        no_motion_bias: float,
        rule_fit_path: str = "",
    ) -> None:
        """Push learning config from StrokeMapper / GUI into BeatIntelligence."""
        self._learning_enabled = bool(enabled)
        self._learning_use_fitted_rules = bool(use_fitted_rules)
        self._learning_strength = float(np.clip(strength, 0.0, 1.0))
        self._learning_min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self._learning_no_motion_bias = float(np.clip(no_motion_bias, 0.25, 3.0))

        if self._learning_enabled and self._learning_use_fitted_rules:
            self._try_load_learning_model(rule_fit_path)
        else:
            self._learning_model = None
            self._learning_model_loaded = False

    def _try_load_learning_model(self, path_text: str = "") -> None:
        """Load rule_fit JSON with schema validation (#12)."""
        path_text = str(path_text or "").strip()
        if not path_text:
            self._learning_model = None
            self._learning_model_loaded = False
            return

        try:
            path = Path(path_text)
            if not path.exists() or not path.is_file():
                self._learning_model = None
                self._learning_model_loaded = False
                return

            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("status") != "ok":
                self._learning_model = None
                self._learning_model_loaded = False
                return

            # Validate required schema fields
            feature_cols = payload.get("feature_columns", [])
            norm = payload.get("normalization", {})
            models = payload.get("models", {})
            if not feature_cols or not isinstance(norm, dict) or not isinstance(models, dict):
                self._learning_model = None
                self._learning_model_loaded = False
                return

            norm_mean = norm.get("mean", {})
            norm_std = norm.get("std", {})
            if not isinstance(norm_mean, dict) or not isinstance(norm_std, dict):
                self._learning_model = None
                self._learning_model_loaded = False
                return

            self._learning_model = payload
            self._learning_feature_columns = list(feature_cols)
            self._learning_norm_mean = {str(k): float(v) for k, v in norm_mean.items()}
            self._learning_norm_std = {str(k): float(v) for k, v in norm_std.items()}
            self._learning_cadence_rule = payload.get("cadence_rule", {}) or {}
            self._learning_model_loaded = True

            targets = payload.get("target_columns", list(models.keys()))
            print(f"[Learning] Loaded rule_fit: {len(feature_cols)} features -> {len(targets)} targets")
        except Exception as exc:
            print(f"[Learning] Failed to load rule_fit: {exc}")
            self._learning_model = None
            self._learning_model_loaded = False

    def _build_runtime_feature_values(self, event: BeatEvent) -> dict[str, float]:
        """Build the 14-feature vector that matches the cue-based training
        pipeline.  Instantaneous features come from the current event /
        BandEnergies; lookback aggregates are computed from rolling deques.

        Band energies are already P95-normalised by update_band_energies,
        and flux is P95-normalised here against the rolling deque, so all
        volume-dependent columns are inherently volume-independent.
        """
        rms_db = self._event_rms_db(event)
        raw_flux = float(getattr(event, "spectral_flux", 0.0) or 0.0)

        # P95-normalize flux against rolling history (matches training pipeline)
        flux_history = list(self._recent_flux_values)
        if len(flux_history) >= 10:
            p95 = float(np.percentile(flux_history, 95))
            flux_norm = float(np.clip(raw_flux / max(p95, 1e-9), 0.0, 1.0))
        else:
            flux_norm = 0.0  # not enough history yet

        # Band energies are already P95-normalised (0..1) by update_band_energies
        sub = float(self.energies.sub_bass)
        low = float(self.energies.low_mid)
        mid = float(self.energies.mid)
        high = float(self.energies.high)

        # Derived & spectral features
        eps = 1e-10
        low_high_ratio = float((sub + low + eps) / (high + eps))

        centroid = float(getattr(event, "spectral_centroid_hz", 0.0) or 0.0)
        flatness = float(getattr(event, "spectral_flatness", 0.0) or 0.0)

        # Fallback estimates when spectral features not available from audio engine
        if centroid <= 0.0:
            centroid = 80.0 + 3000.0 * float(np.clip(high / max(sub + low + eps, eps), 0.0, 1.0))
        if flatness <= 0.0:
            energy_norm = self._dbfs_to_unit(rms_db)
            flatness = 0.35 + 0.50 * (1.0 - energy_norm)

        # ── Lookback aggregate features (10-second window) ──
        rms_list = list(self._recent_rms_db)
        flux_list = flux_history  # already computed above
        bass_list = list(self._band_energy_history.get("sub_bass", deque()))

        if len(rms_list) >= 3:
            rms_arr = np.asarray(rms_list, dtype=np.float64)
            rms_mean_10s = float(np.mean(rms_arr))
            rms_std_10s = float(np.std(rms_arr))
            # Linear trend: slope of RMS over lookback window
            x = np.arange(len(rms_arr), dtype=np.float64)
            x_mean = np.mean(x)
            y_mean = np.mean(rms_arr)
            denom = float(np.sum(np.square(x - x_mean)))
            energy_trend_10s = float(np.sum((x - x_mean) * (rms_arr - y_mean)) / max(denom, 1e-12))
        else:
            rms_mean_10s = rms_db
            rms_std_10s = 0.0
            energy_trend_10s = 0.0

        flux_mean_10s = float(np.mean(flux_list)) if len(flux_list) >= 3 else flux_norm
        bass_mean_10s = float(np.mean(bass_list)) if len(bass_list) >= 3 else sub

        return {
            "rms": rms_db,
            "spectral_flux": flux_norm,
            "sub_bass_energy": sub,
            "low_mid_energy": low,
            "mid_energy": mid,
            "high_energy": high,
            "low_high_ratio": low_high_ratio,
            "spectral_centroid_hz": centroid,
            "spectral_flatness": flatness,
            "rms_mean_10s": rms_mean_10s,
            "rms_std_10s": rms_std_10s,
            "flux_mean_10s": flux_mean_10s,
            "bass_mean_10s": bass_mean_10s,
            "energy_trend_10s": energy_trend_10s,
        }

    def _predict_learning_targets(self, features: dict[str, float]) -> dict[str, float]:
        """Run one z-score normalized linear inference pass (#11).

        Returns dict of target_name → raw predicted value, or {} on failure.
        """
        model = self._learning_model
        if model is None or not self._learning_model_loaded:
            return {}

        try:
            cols = self._learning_feature_columns
            mean = self._learning_norm_mean
            std = self._learning_norm_std
            models = model.get("models", {})

            # Z-score normalize features
            x_norm = np.array([
                (features.get(c, mean.get(c, 0.0)) - mean.get(c, 0.0))
                / max(std.get(c, 1e-8), 1e-8)
                for c in cols
            ], dtype=float)

            result: dict[str, float] = {}
            for target_name, spec in models.items():
                intercept = float(spec.get("intercept", 0.0))
                coefs = spec.get("coefficients", {})
                coef_vec = np.array([float(coefs.get(c, 0.0)) for c in cols], dtype=float)
                result[target_name] = float(intercept + np.dot(coef_vec, x_norm))

            return result
        except Exception:
            return {}

    def _derive_cadence_beats(self, predicted_speed: float) -> int:
        """Derive beats-between-strokes from cadence_rule thresholds applied
        to the model's predicted speed_mult value."""
        rule = self._learning_cadence_rule
        if not rule:
            return 1

        quiet_thresh = float(rule.get("quiet_threshold", 0.15))
        mid_thresh = float(rule.get("mid_threshold", 0.45))
        mapping = rule.get("mapping", {})

        if predicted_speed < quiet_thresh:
            return int(mapping.get("quiet", 4))
        elif predicted_speed < mid_thresh:
            return int(mapping.get("mid", 2))
        else:
            return int(mapping.get("loud", 1))

    def _update_learning_adapter(self, event: BeatEvent) -> LearningOutputs:
        """Cue-based speed prediction: extract lookback features, predict
        speed_mult, derive cadence.  Only fires on beat events when learning
        is enabled.  The single output is ``speed_mult`` (0 = still, 1 = full).
        """
        outputs = LearningOutputs()

        if not self._learning_enabled or not self._learning_model_loaded:
            self._learning_outputs = outputs
            return outputs

        is_beat = bool(
            getattr(event, "is_beat", False)
            or getattr(event, "is_downbeat", False)
            or getattr(event, "is_syncopated", False)
        )
        if not is_beat:
            return self._learning_outputs  # return last valid outputs

        # Check confidence: need acf_confidence above threshold
        acf_conf = float(getattr(event, "acf_confidence", 0.0) or 0.0)
        if acf_conf < self._learning_min_confidence:
            return self._learning_outputs

        features = self._build_runtime_feature_values(event)
        predictions = self._predict_learning_targets(features)
        if not predictions:
            return self._learning_outputs

        strength = self._learning_strength

        # Clamp raw predicted speed_mult to 0..1
        raw_speed = float(np.clip(predictions.get("speed_mult", 0.5), 0.0, 1.0))

        # Blend toward neutral (0.5) at low strength
        blended_speed = 0.5 + strength * (raw_speed - 0.5)
        blended_speed = float(np.clip(blended_speed, 0.0, 1.0))

        # Derive cadence from predicted speed
        cadence_beats = self._derive_cadence_beats(raw_speed)

        # EMA smooth speed_mult (slow smoothing to avoid mid-journey jumps)
        alpha = 0.15
        self._learned_speed_mult += alpha * (blended_speed - self._learned_speed_mult)
        self._learned_cadence_hint = cadence_beats  # discrete, no smoothing

        outputs = LearningOutputs(
            speed_mult=float(np.clip(self._learned_speed_mult, 0.0, 1.0)),
            cadence_hint=self._learned_cadence_hint,
            active=True,
        )
        self._learning_outputs = outputs
        return outputs

    # ── Phase 2 §17: Readiness state machine ─────────────────────────

    def _update_stroke_readiness(self, event: BeatEvent, now: float) -> bool:
        """Evaluate stroke readiness with hysteresis and grace period.

        Returns True if motion should be allowed.
        Brief confidence dips don't kill motion; sustained loss does.

        §5/§6: Tempo-unlock hold integration.
        When hold is active, stroke_ready stays True even if tempo is not green.
        Hold releases immediately on flux spike/drop or silence.
        """
        grace_ms = float(getattr(self.config.beat, "teaching_stroke_ready_grace_ms", 450.0) or 450.0)
        _raw_finish = getattr(self.config.beat, "teaching_stroke_finish_beats", 4)
        finish_beats = int(_raw_finish) if _raw_finish is not None else 4

        # §5: Update tempo-unlock hold state
        self._update_tempo_unlock_hold(event, now)

        # Raw readiness check (same logic as _tempo_ready_for_motion)
        raw_ready = self._tempo_ready_for_motion(event)

        # §6: If unlock-hold is active, override raw_ready to True
        if self._tempo_unlock_hold_active and not raw_ready:
            raw_ready = True

        if raw_ready:
            self._readiness_green_count += 1
            self._readiness_yellow_count = 0
            self._readiness_block_streak = 0
            self._readiness_grace_until = now + (grace_ms / 1000.0)
            self._readiness_finish_beats_remaining = finish_beats

            if self._readiness_green_count >= 1:
                self._stroke_ready = True
                self._stroke_ready_reason = "green" if not self._tempo_unlock_hold_active else "hold_active"
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

    def _update_tempo_unlock_hold(self, event: BeatEvent, now: float) -> None:
        """§5: Tempo-unlock hold management.

        Activates when we have mild beat confidence but metronome is not green.
        Holds the last locked cadence until cancelled by flux spike/drop or silence.
        Releases immediately on flux event or silence for responsive relaxation.
        """
        tempo_locked = bool(getattr(event, "tempo_locked", False))
        acf_conf = float(getattr(event, "acf_confidence", 0.0) or 0.0)
        if not math.isfinite(acf_conf):
            acf_conf = 0.0
        metro_bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)

        # Cancel on silence
        if self.silence_deadzone_active:
            self._tempo_unlock_hold_active = False
            return

        # Cancel on flux spike or drop
        if self._tempo_unlock_hold_active:
            recent_flux = list(self._recent_flux_values)
            if len(recent_flux) >= 4:
                current_flux = float(np.mean(recent_flux[-4:]))
                baseline = max(self._tempo_unlock_hold_flux_baseline, 1e-6)
                if current_flux > baseline * self._tempo_unlock_hold_flux_spike_ratio:
                    # Flux spike — release hold
                    self._tempo_unlock_hold_active = False
                    return
                if current_flux < baseline * self._tempo_unlock_hold_flux_drop_ratio:
                    # Flux drop — release hold
                    self._tempo_unlock_hold_active = False
                    return

        # If tempo is truly locked, deactivate hold (not needed)
        if tempo_locked:
            self._tempo_unlock_hold_active = False
            self._tempo_unlock_hold_bpm = float(self._last_locked_bpm)
            return

        # Activate hold only when confidence is at least the configured relaxed
        # threshold; this prevents hold from masking explicit "not ready" states.
        relaxed_conf = float(getattr(self.config.beat, "teaching_metronome_relaxed_confidence", 0.14) or 0.14)
        relaxed_conf = float(np.clip(relaxed_conf, 0.0, 1.0))
        mild_confidence = acf_conf >= max(0.08, relaxed_conf)

        # In legacy metronome-only mode, require a live metronome BPM.
        # Otherwise allow last-locked BPM memory as fallback.
        if bool(getattr(self.config.beat, "teaching_ignore_traffic_lights", False)):
            has_bpm = metro_bpm > 0.0
        else:
            has_bpm = metro_bpm > 0.0 or self._last_locked_bpm > 0.0

        if mild_confidence and has_bpm and not self._tempo_unlock_hold_active:
            self._tempo_unlock_hold_active = True
            self._tempo_unlock_hold_bpm = float(self._last_locked_bpm if self._last_locked_bpm > 0 else metro_bpm)
            recent_flux = list(self._recent_flux_values)
            self._tempo_unlock_hold_flux_baseline = float(np.mean(recent_flux)) if len(recent_flux) >= 4 else 0.1

    # ── Phase 2 §19: Silence fade-out tracker ────────────────────────

    def _update_silence_fade(self, silence_active: bool, now: float, overall_amplitude: float | None = None) -> tuple[float, bool]:
        """Track prolonged silence: emit fade scalar and tempo-reset request.

        Returns (fade_scalar 0..1, request_tempo_reset bool).
        """
        request_reset = False
        open_threshold_raw = getattr(self.config.stroke, "silence_threshold", -66.0)
        open_threshold_db = silence_threshold_to_dbfs(open_threshold_raw, default_linear=0.001)
        amp = self._coerce_amplitude_db(overall_amplitude)

        if silence_active:
            self._consecutive_silent_count += 1
            # Gradual fade
            self._silence_fade = max(0.0, self._silence_fade - self._silence_fade_rate)

            # Only arm post-silence ramp reset from true silence-open amplitude.
            if amp < open_threshold_db:
                self._was_silent = True

            # Tempo reset after prolonged silence (once per silence episode)
            if (self._consecutive_silent_count >= self._silence_reset_threshold_frames
                    and not self._silence_reset_armed):
                self._silence_reset_armed = True
                request_reset = True
                self._was_silent = True
        else:
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



    # ── Phase 1 §21: Activity helpers ────────────────────────────────

    def _get_low_band_activity(self, event: BeatEvent) -> float:
        """Estimate low-band (sub_bass + low_mid) activity for this frame."""
        sub = float(np.clip(self.energies.sub_bass, 0.0, 1.0))
        low = float(np.clip(self.energies.low_mid, 0.0, 1.0))
        return float(max(sub, (sub * 0.6 + low * 0.4)))

    def _get_mid_band_activity(self, event: BeatEvent) -> float:
        """Estimate vocal/guitar mid-band activity for this frame."""
        return float(np.clip(self.energies.mid, 0.0, 1.0))

    def _get_high_band_activity(self, event: BeatEvent) -> float:
        """Estimate upper-range (mid + high) activity for this frame."""
        mid = float(np.clip(self.energies.mid, 0.0, 1.0))
        high = float(np.clip(self.energies.high, 0.0, 1.0))
        return float(max(high, (mid * 0.3 + high * 0.7)))

    def _get_mid_bass_activity(self, event: BeatEvent) -> float:
        """Estimate 200-400 Hz mid-bass activity from low_mid band."""
        return float(np.clip(self.energies.low_mid, 0.0, 1.0))

    # ── Phase 1 §3: Beat timing ─────────────────────────────────────

    def _arm_tempo_reset_motion_hold(self, now: float) -> None:
        """After tempo_reset, record a beat time so no-beat timeout doesn't fire."""
        self._last_any_beat_time = now

    def _record_beat_times(self, event: BeatEvent, trigger_kind: str, now: float) -> None:
        """Track timestamp of the most recent beat/downbeat/syncopation."""
        if bool(getattr(event, "tempo_reset", False)):
            self._arm_tempo_reset_motion_hold(now)

        is_beat = bool(getattr(event, "is_beat", False))
        is_downbeat = bool(getattr(event, "is_downbeat", False))
        is_syncopated = bool(getattr(event, "is_syncopated", False))

        if is_beat or is_downbeat or is_syncopated:
            self._last_any_beat_time = now

    # ── Phase 3 §5: Low-band fullness gate (removed) ───────────────────

    # ── Phase 3 §6: Dual-band dB gate (removed) ──────────────────────

    # ── Phase 3 §8: Spectrum fill gate ───────────────────────────────────

    def _update_dbfs_reference(self, magnitudes: np.ndarray, now: float) -> None:
        """Update the dBFS reference maximum with decay over time."""
        cfg = self.config.stroke
        decay_rate = float(getattr(cfg, 'dbfs_reference_decay_rate', 0.9995))
        
        # Apply time-based decay
        if self._dbfs_reference_last_update > 0.0:
            dt = now - self._dbfs_reference_last_update
            # Decay per frame (~60fps typical)
            frames_elapsed = max(1, int(dt * 60.0))
            self._dbfs_reference_max *= (decay_rate ** frames_elapsed)
        
        # Update with current peak if higher
        current_peak = float(np.max(magnitudes))
        if current_peak > self._dbfs_reference_max:
            self._dbfs_reference_max = current_peak
        
        self._dbfs_reference_last_update = now

    def _get_spectrum_fill_ratio(self, trigger_kind: str) -> float:
        """Compute spectrum fill ratio from live FFT for given phase (#8).

        Uses absolute dBFS thresholds relative to recent max signal.
        """
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
        
        # Get bin range for this trigger type
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

        now = time.perf_counter()
        self._update_dbfs_reference(magnitudes, now)

        # dBFS threshold per trigger type
        dbfs_map = {
            "downbeat": float(getattr(cfg, 'downbeat_dbfs_threshold', -25.0)),
            "beat": float(getattr(cfg, 'beat_dbfs_threshold', -30.0)),
            "syncopation": float(getattr(cfg, 'syncopation_dbfs_threshold', -35.0)),
        }
        dbfs_threshold = dbfs_map.get(trigger_kind, -30.0)

        # Convert dBFS threshold to linear scale relative to reference max
        # dB = 20 * log10(magnitude / reference)
        # magnitude_threshold = reference * 10^(dB / 20)
        reference_max = max(self._dbfs_reference_max, 1e-10)
        linear_threshold = reference_max * (10.0 ** (dbfs_threshold / 20.0))

        # Count bins above absolute threshold
        filled = float(np.sum(band >= linear_threshold))
        return float(filled / max(1, band.size))

    def _passes_overall_amp_fill_gate(self, event: BeatEvent, trigger_kind: str) -> bool:
        """Overall amplitude fill gate (#8): require spectral fill for phase."""
        cfg = self.config.stroke
        if not bool(getattr(cfg, 'overall_amp_fill_gate_enabled', True)):
            # Gate disabled: reset counters and pass
            self._fill_pass_consecutive[trigger_kind] = 0
            return True

        # No audio engine: don't block
        if self.audio_engine is None:
            self._fill_pass_consecutive[trigger_kind] = 0
            return True

        intensity = float(getattr(event, 'intensity', 0.0) or 0.0)
        target = float(getattr(cfg, 'overall_amp_fill_target', 0.5))
        tolerance = float(getattr(cfg, 'overall_amp_fill_tolerance', 0.5))

        if intensity < (target - tolerance):
            self._fill_pass_consecutive[trigger_kind] = 0
            return False

        fill_ratio = self._get_spectrum_fill_ratio(trigger_kind)
        required = self._get_overall_amp_fill_required(trigger_kind)

        # Check instant fill pass/fail
        instant_passed = fill_ratio >= required
        self._update_auto_fill_required(trigger_kind, instant_passed)

        # Update consecutive frame counter
        if instant_passed:
            self._fill_pass_consecutive[trigger_kind] = self._fill_pass_consecutive.get(trigger_kind, 0) + 1
        else:
            self._fill_pass_consecutive[trigger_kind] = 0

        # Check sustain duration requirement.
        # Prefer per-trigger settings, then fall back to global.
        sustain_key_map = {
            "downbeat": "downbeat_overall_amp_fill_sustain_frames",
            "beat": "beat_overall_amp_fill_sustain_frames",
            "syncopation": "syncopation_overall_amp_fill_sustain_frames",
        }
        sustain_key = sustain_key_map.get(trigger_kind, "overall_amp_fill_sustain_frames")
        sustain_frames = max(
            0,
            int(
                getattr(
                    cfg,
                    sustain_key,
                    getattr(cfg, 'overall_amp_fill_sustain_frames', 3),
                )
                or 3
            ),
        )
        if sustain_frames <= 1:
            # Duration check disabled (0 or 1 = instant decision)
            return instant_passed

        # Require sustained fullness over consecutive frames
        consecutive = self._fill_pass_consecutive.get(trigger_kind, 0)
        return consecutive >= sustain_frames

    def _get_overall_amp_fill_required(self, trigger_kind: str) -> float:
        """Get fill required for phase, including auto-adapt offset (#20)."""
        cfg = self.config.stroke
        base_map = {
            "downbeat": float(getattr(cfg, 'downbeat_overall_amp_fill_required', 0.75)),
            "beat": float(getattr(cfg, 'beat_overall_amp_fill_required', 0.90)),
            "syncopation": float(getattr(cfg, 'syncopation_overall_amp_fill_required', 1.00)),
        }
        base = base_map.get(trigger_kind, base_map.get("beat", 0.90))
        scale = float(np.clip(getattr(cfg, 'overall_amp_fill_required_scale', 1.0) or 1.0, 0.05, 20.0))
        offset = self._auto_fill_offsets.get(trigger_kind, 0.0)

        min_req = float(getattr(cfg, 'overall_amp_fill_auto_min_required', 0.05))
        max_req = float(getattr(cfg, 'overall_amp_fill_auto_max_required', 0.98))

        return float(np.clip((base * scale) + offset, min_req, max_req))

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
        self._recent_mid_band_values.append(self._get_mid_band_activity(event))
        self._recent_high_band_values.append(self._get_high_band_activity(event))
        self._recent_mid_bass_values.append(self._get_mid_bass_activity(event))

    def update_band_energies(self) -> None:
        raw_energies = {}
        if self.audio_engine is not None and hasattr(self.audio_engine, "_band_energies"):
            maybe = getattr(self.audio_engine, "_band_energies", None)
            if isinstance(maybe, dict):
                raw_energies = maybe

        for band_name in ("sub_bass", "low_mid", "mid", "high"):
            raw = float(raw_energies.get(band_name, 0.0))
            hist = self._band_energy_history.get(band_name)
            if hist is not None:
                hist.append(raw)
            # Volume-adaptive normalization: map raw energy into 0..1
            # using the rolling 95th percentile as the ceiling.
            # This makes band energies represent *relative* activity
            # regardless of OS volume level.
            if hist is not None and len(hist) >= 30:
                p95 = float(np.percentile(list(hist), 95))
                ref = max(p95, 1e-9)
                normed = float(np.clip(raw / ref, 0.0, 1.0))
            else:
                # Not enough history — pass through raw (clamped)
                normed = float(np.clip(raw, 0.0, 1.0))
            old_val = getattr(self.energies, band_name, 0.0)
            setattr(
                self.energies,
                band_name,
                old_val + (normed - old_val) * self.band_ema_alpha,
            )

    def update_envelope(self, event: BeatEvent) -> None:
        target = self._event_rms_db(event)
        alpha = self.rms_attack if target >= self.rms_envelope else self.rms_release
        self.rms_envelope += (target - self.rms_envelope) * alpha

    def get_overall_amplitude(self, event: BeatEvent) -> float:
        raw_rms_db = self._event_rms_db(event)
        if np.isfinite(raw_rms_db):
            return raw_rms_db
        return float(np.clip(self.rms_envelope, RMS_DB_FLOOR, 12.0))

    def update_silence_deadzone_gate(self, overall_amplitude: float, now: float | None = None) -> bool:
        open_threshold_raw = getattr(self.config.stroke, "silence_threshold", -66.0)
        close_threshold_raw = getattr(self.config.stroke, "silence_close_threshold", -58.0)
        close_frames_required = 6
        open_threshold = silence_threshold_to_dbfs(open_threshold_raw, default_linear=0.001)
        close_threshold = silence_threshold_to_dbfs(close_threshold_raw, default_linear=0.003)
        if close_threshold <= open_threshold:
            close_threshold = float(min(12.0, open_threshold + 1.5))
        level_db = self._coerce_amplitude_db(overall_amplitude)

        if level_db < open_threshold:
            self.silence_open_count += 1
            self.silence_close_count = 0
            if self.silence_open_count >= 1:
                self.silence_deadzone_active = True
        elif level_db > close_threshold:
            self.silence_close_count += 1
            self.silence_open_count = 0
            if self.silence_close_count >= close_frames_required:
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
        # Legacy permissive mode: ignore traffic-light style strictness and
        # use metronome presence + relaxed confidence as readiness gate.
        if bool(getattr(self.config.beat, "teaching_ignore_traffic_lights", False)):
            relaxed = float(getattr(self.config.beat, "teaching_metronome_relaxed_confidence", 0.14) or 0.14)
            relaxed = float(np.clip(relaxed, 0.0, 1.0))
            acf_conf = float(getattr(event, "acf_confidence", 0.0) or 0.0)
            metro_bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
            if not np.isfinite(acf_conf):
                acf_conf = 0.0
            if not np.isfinite(metro_bpm):
                metro_bpm = 0.0
            if metro_bpm <= 0.0:
                return False
            if bool(getattr(event, "tempo_locked", False)):
                return True
            return acf_conf >= relaxed

        if not bool(getattr(self.config.beat, "tempo_lock_required", True)):
            return True
        if bool(getattr(event, "tempo_locked", False)):
            return True

        # Metronome-presence fallback: when the metronome is ticking at a
        # musically valid BPM the tempo is known even if acf_confidence
        # dips below the normal threshold.  Use a very low floor (0.05)
        # so only truly random phases are rejected.
        metro_bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
        if not np.isfinite(metro_bpm):
            metro_bpm = 0.0
        acf_conf = float(getattr(event, "acf_confidence", 0.0) or 0.0)
        if not np.isfinite(acf_conf):
            acf_conf = 0.0
        if 50.0 < metro_bpm < 200.0 and acf_conf >= 0.05:
            return True

        relaxed = float(getattr(self.config.beat, "teaching_metronome_relaxed_confidence", 0.14) or 0.14)
        return acf_conf >= relaxed

    # _strict_bass_motion_allowed removed — gate was disabled by default
    # and is no longer part of the gate chain.

    def _transient_motion_profile(self, event: BeatEvent, energy_fullness: float) -> tuple[str, float, bool, float]:
        """Return (profile_kind, radius_mult, _reserved, _reserved).

        profile_kind:
        - "kick_hat": full motion (kick + hi-hat detected)
        - "kick_only": full motion (kick detected)
        - "neutral": no kick/bass evidence (features present but no bass)
        - "no_features": beat_features unavailable — cannot judge
        """
        features = getattr(event, "beat_features", None)
        if not isinstance(features, dict):
            return "no_features", 1.0, False, 0.0

        kick_conf = float(np.clip(features.get("kick_like_conf", 0.0) or 0.0, 0.0, 1.0))
        hat_conf = float(np.clip(features.get("hat_like_conf", 0.0) or 0.0, 0.0, 1.0))
        bass_dom = float(np.clip(features.get("bass_dominance", 1.0) or 1.0, 0.0, 8.0))

        beat_band = str(getattr(event, "beat_band", "") or "")
        fired = getattr(event, "fired_bands", None)
        fired_set = {str(item) for item in fired} if isinstance(fired, (list, tuple, set)) else set()
        bass_band_hit = bool(
            beat_band in ("sub_bass", "low_mid")
            or "sub_bass" in fired_set
            or "low_mid" in fired_set
        )
        sub_bass_hit = bool(beat_band == "sub_bass" or "sub_bass" in fired_set)
        low_mid_hit = bool(beat_band == "low_mid" or "low_mid" in fired_set)

        has_hat = bool(hat_conf >= 0.50)

        # Tightened kick evidence:
        # - high-tone/voice-like frames should not unlock full motion unless
        #   there is real bass support.
        # Defaults calibrated from training data (CH-Tranquilizer):
        #   kick_conf: still-frames rarely exceed 0.5, active frames ~0.6+
        #   bass_dom (low_high_ratio): slow=1.91 median, fast=2.15, P75=5.05
        #   flux (P95-normed): still=0.14, moving=0.25, fast=0.35
        #   energy_fullness: composite 0-1, slow sub_bass=0.02
        min_kick_conf = float(np.clip(getattr(self.config.beat, "transient_full_motion_min_kick_conf", 0.60) or 0.60, 0.0, 1.0))
        min_bass_dom = float(np.clip(getattr(self.config.beat, "transient_full_motion_min_bass_dom", 1.95) or 1.95, 0.0, 8.0))
        decisive_bass_dom_threshold = float(np.clip(getattr(self.config.beat, "transient_full_motion_decisive_bass_dom", 2.55) or 2.55, 0.0, 8.0))
        min_flux_for_full = float(np.clip(getattr(self.config.beat, "transient_full_motion_min_flux", 0.15) or 0.15, 0.0, 4.0))
        min_fullness_for_full = float(np.clip(getattr(self.config.beat, "transient_full_motion_min_energy_fullness", 0.18) or 0.18, 0.0, 1.0))

        strong_kick_conf = bool(kick_conf >= min_kick_conf)
        strong_bass_dom = bool(bass_dom >= min_bass_dom)
        decisive_bass_dom = bool(bass_dom >= decisive_bass_dom_threshold)
        raw_flux = float(np.clip(getattr(event, "spectral_flux", 0.0) or 0.0, 0.0, 8.0))
        # Normalize flux against rolling P95 so the threshold works at any volume
        flux_history = list(self._recent_flux_values)
        if len(flux_history) >= 10:
            flux_p95 = float(np.percentile(flux_history, 95))
            flux_now = float(np.clip(raw_flux / max(flux_p95, 1e-9), 0.0, 1.0))
        else:
            flux_now = raw_flux
        full_spectrum_active = bool(flux_now >= min_flux_for_full or float(np.clip(energy_fullness, 0.0, 1.0)) >= min_fullness_for_full)
        has_kick = bool(
            (strong_kick_conf and (sub_bass_hit or (low_mid_hit and strong_bass_dom)))
            or (decisive_bass_dom and kick_conf >= 0.55)
        )
        has_kick = bool(has_kick and full_spectrum_active)

        if has_kick and has_hat:
            return "kick_hat", 1.0, False, 0.0
        if has_kick:
            return "kick_only", 1.0, False, 0.0
        return "neutral", 1.0, False, 0.0

    @staticmethod
    def interval_beats_for_trigger(trigger_kind: str) -> int:
        if trigger_kind == "syncopation":
            return 2          # was 1; doubled to prevent half-beat jerking
        if trigger_kind == "beat":
            return 2
        if trigger_kind == "downbeat":
            return 4
        return 8

    def effective_bpm(self, event: BeatEvent) -> float:
        """Return stabilized BPM (#13): last-locked memory + jump-ratio limiter."""
        bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
        if bpm <= 0.0:
            bpm = float(getattr(event, "bpm", 0.0) or 0.0)
        if bpm <= 0.0:
            bpm = 120.0
        bpm = float(np.clip(bpm, 40.0, 240.0))

        tempo_locked = bool(getattr(event, "tempo_locked", False))
        acf_conf = float(getattr(event, "acf_confidence", 0.0) or 0.0)
        # Fix #7: Only latch last-locked BPM when confidence is meaningful
        # so a garbage low-confidence value can't poison future clamping.
        if tempo_locked and acf_conf >= 0.15 and 50.0 <= bpm <= 200.0:
            self._last_locked_bpm = bpm

        # Stabilize: cap jump ratio relative to last locked BPM
        bpm = self._cap_bpm_to_last_locked(bpm)
        bpm = self._stabilize_unlocked_bpm(bpm, tempo_locked)
        return bpm

    def _cap_bpm_to_last_locked(self, raw_bpm: float) -> float:
        """Limit BPM jumps to within jump_ratio_limit of last locked BPM (#13)."""
        ref = self._last_locked_bpm
        if ref <= 0.0:
            return raw_bpm
        ratio = raw_bpm / max(ref, 1e-6)
        limit = self._bpm_jump_ratio_limit
        if ratio > limit:
            return float(ref * limit)
        if ratio < 1.0 / limit:
            return float(ref / limit)
        return raw_bpm

    def _stabilize_unlocked_bpm(self, raw_bpm: float, tempo_locked: bool) -> float:
        """EMA-smooth BPM when tempo is unlocked (#13).

        When locked, snap to raw BPM immediately.
        When unlocked, slowly drift toward raw via EMA to avoid jitter.
        """
        if tempo_locked:
            self._stabilized_bpm = raw_bpm
            return raw_bpm
        alpha = 0.15  # slow smoothing when unlocked
        self._stabilized_bpm += alpha * (raw_bpm - self._stabilized_bpm)
        return float(np.clip(self._stabilized_bpm, 40.0, 240.0))

    def compute_radius_bloom_from_sub_bass(self, event: BeatEvent | None = None) -> float:
        base_radius = 0.70
        max_radius = 1.0   # cap at canvas edge
        max_bloom = max_radius - base_radius

        sub_bass = float(np.clip(self.energies.sub_bass, 0.0, 1.0))
        low_mid = float(np.clip(self.energies.low_mid, 0.0, 1.0))

        # Exponential bass scaling: power function for dramatic dynamic range
        # Low-amplitude bass stays near park; high bass balloons out quickly
        weighted_bass = (sub_bass * 0.70) + (low_mid * 0.30)
        bass_fill = float(np.clip(max(sub_bass, weighted_bass), 0.0, 1.0))
        bass_power = bass_fill ** 2.0

        # Spectral flux influence: flux adds extra bloom proportional to config weight
        # Normalize flux against its rolling P95 so bloom contribution is
        # volume-independent.
        flux = 0.0
        flux_weight = 1.0
        if event is not None:
            raw_flux = float(getattr(event, 'spectral_flux', 0.0) or 0.0)
            flux_weight = float(getattr(self.config.stroke, 'flux_scaling_weight', 1.0) or 1.0)
            flux_history = list(self._recent_flux_values)
            if len(flux_history) >= 10:
                p95 = float(np.percentile(flux_history, 95))
                flux = float(np.clip(raw_flux / max(p95, 1e-9), 0.0, 1.0))
            else:
                flux = raw_flux
        flux_boost = float(np.clip(flux * flux_weight * 0.15, 0.0, 0.15))

        # Dynamic amp-gate: derive quiet/full thresholds from the rolling
        # RMS history so the gate adapts to whatever OS volume the user is
        # listening at.  Uses 5th percentile as "quiet" and 90th as "full".
        # Falls back to conservative fixed thresholds until enough history.
        rms_db = self.rms_envelope
        self._recent_rms_db.append(rms_db)
        if len(self._recent_rms_db) >= 30:
            rms_list = list(self._recent_rms_db)
            quiet_db = float(np.percentile(rms_list, 5))
            full_db = float(np.percentile(rms_list, 90))
            # Ensure a minimum spread so the gate isn't binary
            if full_db - quiet_db < 6.0:
                full_db = quiet_db + 6.0
        else:
            # Not enough history yet — use permissive fixed fallback
            quiet_db = self._linear_to_dbfs(0.005)
            full_db = self._linear_to_dbfs(0.03)
        if rms_db < quiet_db:
            amp_gate = 0.0
        elif rms_db < full_db:
            amp_gate = float((rms_db - quiet_db) / max(1e-6, (full_db - quiet_db)))
        else:
            amp_gate = 1.0

        bloom = (max_bloom * bass_power + flux_boost) * amp_gate

        return float(np.clip(base_radius + bloom, base_radius, max_radius))

    def compute_energy_fullness(self) -> float:
        """0..1 measure of how 'full' the music is right now.

        Combines RMS envelope with bass energy to produce a smooth
        scalar that stroke_mapper latches at journey start to decide
        whether max_radius should expand toward 1.0.
        """
        # Use rolling-relative RMS so fullness adapts to OS volume.
        # When enough history exists, map current RMS into the recent
        # dynamic range (P5..P95) instead of the absolute -120..0 dBFS scale.
        if len(self._recent_rms_db) >= 30:
            rms_list = list(self._recent_rms_db)
            lo_db = float(np.percentile(rms_list, 5))
            hi_db = float(np.percentile(rms_list, 95))
            spread = max(hi_db - lo_db, 6.0)
            rms = float(np.clip((self.rms_envelope - lo_db) / spread, 0.0, 1.0))
        else:
            rms = self._dbfs_to_unit(self.rms_envelope)
        sub = float(np.clip(self.energies.sub_bass, 0.0, 1.0))
        low = float(np.clip(self.energies.low_mid, 0.0, 1.0))
        mid = float(np.clip(self.energies.mid, 0.0, 1.0))

        # Weighted composite: RMS is the main loudness signal,
        # bass adds body, mid adds presence.
        composite = (rms * 0.45) + (sub * 0.25) + (low * 0.15) + (mid * 0.15)

        # Soft-knee curve: gentle at low levels, opens up at high levels.
        # fullness = composite^0.6 gives a slightly compressed feel.
        fullness = float(composite ** 0.6)

        return float(np.clip(fullness, 0.0, 1.0))

    # Priority rank: lower number = faster arc = higher priority for interrupts.
    _TRIGGER_PRIORITY: dict[str, int] = {
        "syncopation": 0,
        "beat": 1,
        "downbeat": 2,
        "creep": 3,
    }

    def update_journey_progress(
        self,
        trigger_kind: str,
        interval_beats: int,
        event: BeatEvent,
        dt: float,
        force_start: bool = False,
    ) -> float:
        bpm = self.effective_bpm(event)
        beat_period_s = 60.0 / max(1e-6, bpm)
        if self.is_recovering:
            trigger_kind = "start"
            interval_beats = 8
        target_duration = max(1e-3, beat_period_s * float(interval_beats))

        if self.is_recovering and self.journey_active and not force_start:
            step = float(np.clip(dt, 1e-4, 0.25))
            self.journey_elapsed_s = min(self.journey_duration_s, self.journey_elapsed_s + step)
            completion = float(np.clip(self.journey_elapsed_s / max(1e-6, self.journey_duration_s), 0.0, 1.0))
            if completion >= 1.0:
                self.journey_active = False
                self._lazy_glide_active = False
                self.is_recovering = False
            return completion

        is_new_beat = bool(
            (trigger_kind == "syncopation" and bool(getattr(event, "is_syncopated", False)))
            or (trigger_kind == "downbeat" and bool(getattr(event, "is_downbeat", False)))
            or (trigger_kind == "beat" and bool(getattr(event, "is_beat", False)))
        )

        # ── Priority interrupt logic ──
        # KEY: Interrupts only happen ON actual beat events (is_new_beat=True)
        # Never interrupt mid-frame based on classifier changes alone.
        incoming_pri = self._TRIGGER_PRIORITY.get(trigger_kind, 3)
        active_pri = self._TRIGGER_PRIORITY.get(self.last_trigger_kind, 3)
        should_start = False
        is_interrupt = False   # True when restarting over an active journey

        if self.is_recovering:
            if force_start and not self.journey_active:
                should_start = True
            elif not self.journey_active:
                should_start = True
            else:
                should_start = False
            is_interrupt = False
        elif force_start:
            should_start = True
            is_interrupt = bool(self.journey_active)

        elif not self.journey_active:
            # Always start if no journey is running
            should_start = True
        elif is_new_beat:
            # Only consider interrupts on real beat events
            if incoming_pri < active_pri:
                # Higher priority always interrupts (synco > beat > downbeat)
                should_start = True
                is_interrupt = True
            elif self.last_trigger_kind in ("creep", "syncopation") and trigger_kind in ("beat", "downbeat", "syncopation"):
                # Syncopation freely interruptible by any beat event.
                # (Creep no longer starts journeys, but guard remains
                #  in case last_trigger_kind is still "creep" from a
                #  recent fill→beat transition.)
                should_start = True
                is_interrupt = True
            elif self.last_trigger_kind == "downbeat" and trigger_kind in ("beat", "syncopation"):
                # Downbeat can turn into beat/synco once past halfway
                completion = float(np.clip(
                    self.journey_elapsed_s / max(1e-6, self.journey_duration_s), 0.0, 1.0
                ))
                if completion >= 0.50:
                    should_start = True
                    is_interrupt = True
            elif self._lazy_glide_active and incoming_pri <= active_pri and trigger_kind != "creep":
                # During lazy glide, snap back to high-energy immediately
                should_start = True
                is_interrupt = True
        # NOTE: creep (fill) never enters update_journey_progress.
        # It is handled upstream by setting journey_completion = 1.0 directly.

        if should_start:
            self._journey_start_intensity = float(getattr(event, 'intensity', 0.0) or 0.0)

            # When interrupting an active arc, time the new journey so
            # it arrives at the next beat rather than using the full
            # interval duration.  Clamped to [80%..110%] of normal so
            # arcs never feel rushed or overly slow.  (§4: was 40-100%)
            if is_interrupt:
                now_ts = float(getattr(event, 'monotonic_timestamp', 0.0) or 0.0)
                if now_ts <= 0.0:
                    now_ts = time.perf_counter()
                next_beat_s = self._seconds_until_next_beat(event=event, bpm=bpm, now=now_ts)
                clamped = float(np.clip(next_beat_s, target_duration * 0.80, target_duration * 1.10))
                self.journey_duration_s = clamped
            else:
                self.journey_duration_s = target_duration

            # Apply scheduled pipeline-latency lead (config.beat.scheduled_lead_ms).
            # The audio callback fires AFTER the audio buffer is captured (buffer
            # latency) plus WASAPI loopback delay, so beat timestamps arrive late.
            # Shortening journey_duration_s by this amount makes the orbit reach
            # the anchor at the actual musical beat rather than lagging behind.
            # Tune this to: (WASAPI input latency) + (buffer_size / sample_rate / 2).
            # Typical value on Windows: 40-80 ms.
            _sched_lead_s = float(np.clip(
                float(getattr(self.config.beat, 'scheduled_lead_ms', 0) or 0) / 1000.0,
                0.0, 0.25,
            ))
            if _sched_lead_s > 0.0:
                self.journey_duration_s = float(max(
                    target_duration * 0.60,
                    self.journey_duration_s - _sched_lead_s,
                ))

            self._journey_duration_target_s = self.journey_duration_s
            self._journey_duration_blend_frames_remaining = 0
            self.journey_elapsed_s = 0.0
            self.journey_active = True
            self._lazy_glide_active = False
            # Latch pending learning cadence: cadence changes only take
            # effect at journey boundaries, never mid-arc.
            self._committed_divisor_hint = self._learned_cadence_hint
            return 0.0

        if self._journey_duration_blend_frames_remaining > 0:
            self.journey_duration_s += self._journey_duration_blend_alpha * (
                self._journey_duration_target_s - self.journey_duration_s
            )
            self._journey_duration_blend_frames_remaining -= 1
            if self._journey_duration_blend_frames_remaining <= 0:
                self.journey_duration_s = self._journey_duration_target_s

        now = float(getattr(event, "monotonic_timestamp", 0.0) or 0.0)
        if now <= 0.0:
            now = time.perf_counter()

        completion_before = float(np.clip(self.journey_elapsed_s / max(1e-6, self.journey_duration_s), 0.0, 1.0))
        seconds_until_next = self._seconds_until_next_beat(event=event, bpm=bpm, now=now)
        quarter_measure_window_s = beat_period_s  # 4/4 default: a quarter-measure is one beat

        lazy_glide = (
            trigger_kind != "creep"
            and completion_before > 0.70
            and seconds_until_next > quarter_measure_window_s
        )

        if is_new_beat and self._lazy_glide_active:
            lazy_glide = False

        if lazy_glide:
            tail_t = float(np.clip((completion_before - 0.70) / 0.30, 0.0, 1.0))
            lazy_scale = float(np.clip(np.exp(-1.8 * tail_t), 0.16, 1.0))
        else:
            lazy_scale = 1.0

        step = float(np.clip(dt, 1e-4, 0.25)) * lazy_scale
        self.journey_elapsed_s = min(self.journey_duration_s, self.journey_elapsed_s + step)
        completion = float(np.clip(self.journey_elapsed_s / max(1e-6, self.journey_duration_s), 0.0, 1.0))
        if completion >= 1.0:
            self.journey_active = False
            lazy_glide = False
            if self.is_recovering:
                self.is_recovering = False
        self._lazy_glide_active = bool(lazy_glide)
        return completion

    def _seconds_until_next_beat(self, event: BeatEvent, bpm: float, now: float) -> float:
        beat_period_s = 60.0 / max(1e-6, bpm)

        predicted_mono = 0.0
        if self.audio_engine is not None:
            predicted_mono = float(getattr(self.audio_engine, "predicted_next_beat_mono", 0.0) or 0.0)
        if predicted_mono > now:
            return float(max(0.0, predicted_mono - now))

        met_bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
        if self.audio_engine is not None and met_bpm <= 0.0:
            met_bpm = float(getattr(self.audio_engine, "_metronome_bpm", 0.0) or 0.0)
        if met_bpm > 0.0 and self.audio_engine is not None:
            met_phase = getattr(self.audio_engine, "_metronome_phase", None)
            if met_phase is not None:
                phase_frac = float(met_phase) % 1.0
                until_frac = 1.0 - phase_frac
                if until_frac <= 1e-6:
                    until_frac = 1.0
                return float(until_frac * (60.0 / max(1e-6, met_bpm)))

        if met_bpm > 0.0:
            return float(0.5 * (60.0 / max(1e-6, met_bpm)))

        return float(10.0 * beat_period_s)

    def build_decision(self, event: BeatEvent, dt: float, silence_override: bool | None = None) -> BeatDecision:
        self.update_band_energies()
        self.update_envelope(event)

        # Phase 1: flux + deque tracking (every frame)
        self._update_flux_history(event)
        self._populate_rolling_deques(event)

        energy_fullness_now = self.compute_energy_fullness()

        # Session arc: very slow EMA of energy fullness for long-term modulation
        if getattr(self.config.stroke, 'session_arc_enabled', True):
            session_alpha = float(getattr(self.config.stroke, 'session_arc_ema_alpha', 0.001) or 0.001)
            self._session_intensity_ema += session_alpha * (energy_fullness_now - self._session_intensity_ema)
        session_intensity = float(np.clip(self._session_intensity_ema, 0.0, 1.0))

        now = float(getattr(event, "monotonic_timestamp", 0.0) or 0.0)
        if now <= 0.0:
            now = time.perf_counter()

        overall_amplitude = self.get_overall_amplitude(event)
        silence_active = self.update_silence_deadzone_gate(overall_amplitude, now=now)
        if silence_override is not None:
            silence_active = bool(silence_override)

        recovery_start = False
        if silence_active:
            self._was_silence_active = True
            self.is_recovering = False
            # §5: Cancel unlock hold on silence
            self._tempo_unlock_hold_active = False
        elif self._was_silence_active:
            self._was_silence_active = False
            self.is_recovering = True
            recovery_start = True
            self._recovery_radius_bloom = self.compute_radius_bloom_from_sub_bass(event=event)

        # Reset fill duration tracking during silence
        if silence_active:
            for kind in ("downbeat", "beat", "syncopation"):
                self._fill_pass_consecutive[kind] = 0

        # Phase 2: silence fade + post-silence ramp
        silence_fade, request_tempo_reset = self._update_silence_fade(
            silence_active,
            now,
            overall_amplitude=overall_amplitude,
        )
        post_silence_ramp = self._update_post_silence_ramp(silence_active, now)

        # On silence reset: zero fill-gate EMA offsets so gate re-learns fresh for new section
        if request_tempo_reset:
            self._auto_fill_offsets = {"downbeat": 0.0, "beat": 0.0, "syncopation": 0.0}
            self._auto_fill_ema = {"downbeat": 0.5, "beat": 0.5, "syncopation": 0.5}

        # DISABLED FOR TESTING: "start" recovery path bypassed —
        # fall through to normal gate chain instead.
        if self.is_recovering and not silence_active:
            self.is_recovering = False

        # Phase 5: learning adapter
        learning = self._update_learning_adapter(event)

        raw_trigger_kind = self.classify_trigger(event)
        trigger_kind = raw_trigger_kind
        gate_fail_reason = ""  # tracks which gate blocked a beat-family event
        motion_profile, motion_radius_mult, _, _ = self._transient_motion_profile(
            event,
            energy_fullness_now,
        )

        # Record beat times for hierarchy tracking
        self._record_beat_times(event, raw_trigger_kind, now)

        bpm = self.effective_bpm(event)

        # §6: Readiness check runs AFTER effective_bpm so the unlock-hold
        # can protect stroke_ready using stabilized BPM state.
        stroke_ready = self._update_stroke_readiness(event, now)

        # ── Priority interrupt: beat-family events run the gate chain.
        # Non-beat frames during an active beat-family journey keep the current kind.
        # When a beat-family event FAILS gates during an active journey, the
        # active journey is preserved (not killed to fill) — it finishes
        # naturally on its own timing.
        #
        if raw_trigger_kind == "creep" and self.journey_active and self.last_trigger_kind in ("syncopation", "beat", "downbeat"):
            trigger_kind = self.last_trigger_kind
        elif raw_trigger_kind in ("syncopation", "beat", "downbeat") and not silence_active:
            gate_passed = True
            gate_fail_reason = ""
            if not stroke_ready:
                gate_passed = False
                gate_fail_reason = "stroke_ready"
            # Transient profile gate: require kick/bass evidence.
            # motion_profile is "neutral" when no sub-bass or kick
            # transient is detected — vocals / synths alone must not
            # start orbit journeys; they should play funscript fill.
            elif motion_profile == "neutral":
                gate_passed = False
                gate_fail_reason = "no_bass"
            # Phase 3 gates: spectrum fill
            elif not self._passes_overall_amp_fill_gate(event, raw_trigger_kind):
                gate_passed = False
                gate_fail_reason = "amp_fill"

            if gate_passed:
                self._gate_fail_preserve_count = 0
            elif (self.journey_active
                  and self.last_trigger_kind in ("syncopation", "beat", "downbeat")
                  and self._gate_fail_preserve_count < self._gate_fail_preserve_limit):
                # Gate failed but a beat-family journey is running — let it
                # finish.  But only preserve for a limited number of
                # consecutive failures to prevent infinite orbit loops.
                trigger_kind = self.last_trigger_kind
                self._gate_fail_preserve_count += 1
            else:
                trigger_kind = "creep"
                self._gate_fail_preserve_count = 0

        # ── Phrase Commitment: musical phrase locking ──
        # When switching from fill to beat, 
        # commit for a period to prevent sporadic jumps. But allow
        # natural musical transitions (downbeat ↔ beat ↔ syncopation).
        is_beat_event = bool(
            getattr(event, "is_beat", False)
            or getattr(event, "is_downbeat", False)
            or getattr(event, "is_syncopated", False)
        )

        # Detect new phrase entry: fill→beat transition only
        if (not self._phrase_committed
                and trigger_kind == "beat"
                and not silence_active
                and self.last_trigger_kind == "creep"):
            self._phrase_committed = True
            self._phrase_beats_remaining = self._phrase_measure_beats
            self._phrase_gear = trigger_kind
            recent_flux = list(self._recent_flux_values)
            self._phrase_flux_baseline = float(np.mean(recent_flux)) if len(recent_flux) >= 4 else 0.0

        # Enforce phrase commitment
        if self._phrase_committed and self._phrase_beats_remaining > 0:
            # Count beats toward measure completion
            if is_beat_event:
                self._phrase_beats_remaining -= 1

            # Flux-drop cancellation: if energy crashes, release early
            recent_flux = list(self._recent_flux_values)
            current_flux_mean = float(np.mean(recent_flux)) if len(recent_flux) >= 4 else 0.0
            if (self._phrase_flux_baseline > 1e-6
                    and current_flux_mean < (self._phrase_flux_baseline * self._phrase_flux_drop_ratio)):
                # Hard flux drop — cancel phrase commitment immediately
                self._phrase_committed = False
                self._phrase_beats_remaining = 0
            elif trigger_kind == "creep" and not silence_active and not gate_fail_reason:
                # Override: don't allow fill during committed phrase.
                # Gate failures always win — do NOT re-promote a gated-out beat
                # back to "beat", or the orbit loops at full radius against the
                # perimeter instead of pulling in to wait-swirl / jitter.
                trigger_kind = "beat"

        # Intensity-Lock: at measure end, renew if still pumping
        if self._phrase_committed and self._phrase_beats_remaining <= 0:
            recent_flux = list(self._recent_flux_values)
            current_flux_mean = float(np.mean(recent_flux)) if len(recent_flux) >= 4 else 0.0
            if (self._phrase_flux_baseline > 1e-6
                    and current_flux_mean >= (self._phrase_flux_baseline * self._phrase_renew_ratio)):
                # Still pumping — renew for another measure
                self._phrase_beats_remaining = self._phrase_measure_beats
                self._phrase_flux_baseline = current_flux_mean  # update baseline
            else:
                # Release commitment — graceful gear-down on the '1'
                self._phrase_committed = False

        # Cancel phrase commitment during silence
        if silence_active and self._phrase_committed:
            self._phrase_committed = False
            self._phrase_beats_remaining = 0

        # No-beat timeout: force decay to park
        no_beat_timed_out = False
        if self._check_no_beat_timeout(now) and self.journey_active:
            trigger_kind = "creep"
            self.journey_active = False
            self.last_trigger_kind = "creep"
            self.active_interval_beats = 8
            self.is_recovering = False
            self._phrase_committed = False
            self._phrase_beats_remaining = 0
            no_beat_timed_out = True

        if self.is_recovering:
            interval_beats = 8
        else:
            interval_beats = self.interval_beats_for_trigger(trigger_kind)

        # Apply learning cadence hint only at journey boundaries.
        # _committed_divisor_hint is latched from _learned_cadence_hint
        # inside update_journey_progress when a new journey actually starts,
        # so cadence changes never cause mid-arc discontinuities.
        if learning.active and self._committed_divisor_hint > 1:
            interval_beats = max(interval_beats, self._committed_divisor_hint)

        radius_bloom = self.compute_radius_bloom_from_sub_bass(event=event)

        # Learning speed_mult is applied as a snapshot inside StrokeMapper
        # at journey start, not per-frame here, to avoid mid-arc stepping.

        if no_beat_timed_out:
            journey_completion = 1.0  # fully parked
        elif trigger_kind == "creep":
            # Creep = fill motion (funscript ping-pong pattern).
            # No journey runs — report fully parked so StrokeMapper
            # enters _apply_park_motion_frame immediately.
            journey_completion = 1.0
            self.journey_active = False
        else:
            journey_completion = self.update_journey_progress(
                trigger_kind,
                interval_beats,
                event,
                dt,
                force_start=recovery_start,
            )

        # Only update the active trigger kind / interval when a journey
        # actually (re)started, or during fill (creep).  This keeps the
        # running arc's priority identity locked so a same-or-lower-priority
        # event can't poison it on a non-restarting frame.
        if journey_completion <= 1e-9 or no_beat_timed_out or trigger_kind == "creep":
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
            lazy_glide_active=bool(self._lazy_glide_active),
            gate_fail=gate_fail_reason,
            energy_fullness=energy_fullness_now,
            session_intensity=session_intensity,
            learning=learning,
        )

    # ── Gate-state snapshot (for keyboard teacher) ───────────────────

    def snapshot_gate_state(self) -> dict:
        """Return a flat dict of internal gate/condition state for teaching capture.

        Every value is a plain float/int/bool/str safe for CSV serialisation.
        Prefixed ``gs_`` (gate-state) so columns don't collide with other snapshots.
        """
        now = time.perf_counter()

        # Rolling band means (last N frames)
        def _mean(dq: deque) -> float:
            vals = list(dq)
            return float(np.mean(vals)) if vals else 0.0

        flux_vals = list(self._recent_flux_values)
        flux_mean = float(np.mean(flux_vals)) if flux_vals else 0.0
        flux_std = float(np.std(flux_vals)) if len(flux_vals) >= 2 else 0.0

        # Flux delta: mean of last 4 vs mean of preceding 4
        flux_recent4 = float(np.mean(flux_vals[-4:])) if len(flux_vals) >= 4 else flux_mean
        flux_prev4 = float(np.mean(flux_vals[-8:-4])) if len(flux_vals) >= 8 else flux_recent4
        flux_delta = flux_recent4 - flux_prev4

        return {
            # ── Band energies (EMA-smoothed) ──
            "gs_sub_bass": round(self.energies.sub_bass, 5),
            "gs_low_mid": round(self.energies.low_mid, 5),
            "gs_mid": round(self.energies.mid, 5),
            "gs_high": round(self.energies.high, 5),

            # ── Rolling means ──
            "gs_flux_mean": round(flux_mean, 5),
            "gs_flux_std": round(flux_std, 5),
            "gs_flux_delta": round(flux_delta, 5),
            "gs_low_band_mean": round(_mean(self._recent_low_band_values), 5),
            "gs_mid_band_mean": round(_mean(self._recent_mid_band_values), 5),
            "gs_high_band_mean": round(_mean(self._recent_high_band_values), 5),
            "gs_mid_bass_mean": round(_mean(self._recent_mid_bass_values), 5),

            # ── Envelope / loudness ──
            "gs_rms_envelope_db": round(self.rms_envelope, 2),
            "gs_energy_fullness": round(self.compute_energy_fullness(), 4),

            # ── Silence state ──
            "gs_silence_active": int(self.silence_deadzone_active),
            "gs_silence_fade": round(self._silence_fade, 4),
            "gs_consecutive_silent": self._consecutive_silent_count,

            # ── Readiness / gate state ──
            "gs_stroke_ready": int(self._stroke_ready),
            "gs_stroke_ready_reason": self._stroke_ready_reason,
            "gs_phrase_committed": int(self._phrase_committed),
            "gs_phrase_beats_remaining": self._phrase_beats_remaining,

            # ── Journey state ──
            "gs_journey_active": int(self.journey_active),
            "gs_journey_elapsed_s": round(self.journey_elapsed_s, 4),
            "gs_journey_duration_s": round(self.journey_duration_s, 4),
            "gs_is_recovering": int(self.is_recovering),

            # ── Trigger / kind ──
            "gs_last_trigger_kind": self.last_trigger_kind,
            "gs_active_interval_beats": self.active_interval_beats,

            # ── Tempo state ──
            "gs_stabilized_bpm": round(self._stabilized_bpm, 2),
            "gs_tempo_unlock_hold": int(self._tempo_unlock_hold_active),

            # ── Beat timing ──
            "gs_time_since_last_beat_s": round(now - self._last_any_beat_time, 4) if self._last_any_beat_time > 0 else -1.0,

            # ── Session arc ──
            "gs_session_intensity": round(self._session_intensity_ema, 4),

            # ── Auto-fill gate state ──
            "gs_fill_ema_downbeat": round(self._auto_fill_ema.get("downbeat", 0.5), 4),
            "gs_fill_ema_beat": round(self._auto_fill_ema.get("beat", 0.5), 4),
            "gs_fill_ema_syncopation": round(self._auto_fill_ema.get("syncopation", 0.5), 4),
            "gs_fill_offset_downbeat": round(self._auto_fill_offsets.get("downbeat", 0.0), 4),
            "gs_fill_offset_beat": round(self._auto_fill_offsets.get("beat", 0.0), 4),
            "gs_fill_offset_syncopation": round(self._auto_fill_offsets.get("syncopation", 0.0), 4),
        }
