"""
bREadbeats – Keyboard Teaching Mode  (dev-only)

Allows a human to listen to music and dictate motion behaviour in real time
using the arrow keys.  Every keypress records:
  • the current audio / beat-intelligence conditions (a "snapshot")
  • the directive the human gave (more / less / faster / slower)
  • wall-clock and monotonic timestamps
  • time elapsed since the last *audio condition* change  ← floating features
  • time elapsed since the last directive change

The CSV captures the *temporal gap* between when a musical condition changed
(flux rose, bass arrived, silence ended …) and when the human reacted.  That
gap IS the gating-timing parameter we want to learn.

Captured session → teaching_captures/keyboard/session_YYYYMMDD_HHMMSS/directives.csv

Arrow-key mapping
─────────────────
  ↑  = more motion   (bigger radius, more energy)
  ↓  = less motion   (smaller radius, park / creep)
  ←  = slower         (larger interval, gentler cadence)
  →  = faster         (tighter interval, more aggressive cadence)

The four axes are independent: the human can hold ↑+→ for "big and fast".
Internally we track two signed floats that drift toward 0 when no key is
held — *intensity_axis* (↑/↓) and *speed_axis* (←/→) — each in [-1, +1].
"""

from __future__ import annotations

import csv
import math
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional


# ── Snapshot helpers ────────────────────────────────────────────────

def _safe(value: Any) -> Any:
    """Coerce to a JSON/CSV-safe native type."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _snapshot_from_event(event: Any) -> dict[str, Any]:
    """Pull the interesting audio-condition fields out of a BeatEvent."""
    snap: dict[str, Any] = {}
    for attr in (
        "intensity", "frequency", "is_beat", "spectral_flux", "peak_energy",
        "is_downbeat", "bpm", "tempo_locked", "beat_band", "metronome_bpm",
        "acf_confidence", "is_syncopated", "raw_rms", "raw_rms_db",
    ):
        snap[attr] = _safe(getattr(event, attr, None))

    # Flatten fired_bands into a comma-separated string
    fired = getattr(event, "fired_bands", None)
    if isinstance(fired, (list, tuple)):
        snap["fired_bands"] = ",".join(str(b) for b in fired)
    else:
        snap["fired_bands"] = ""

    # Flatten beat_features dict if present
    features = getattr(event, "beat_features", None) or {}
    for key, val in features.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                snap[f"bf_{key}_{sub_key}"] = _safe(sub_val)
        else:
            snap[f"bf_{key}"] = _safe(val)
    return snap


def _snapshot_from_decision(decision: Any) -> dict[str, Any]:
    """Pull decision-side fields from a BeatDecision."""
    snap: dict[str, Any] = {}
    for attr in (
        "trigger_kind", "interval_beats", "radius_bloom",
        "silence_active", "journey_completion", "silence_fade",
        "post_silence_ramp", "lazy_glide_active", "gate_fail",
        "energy_fullness", "session_intensity",
        "park_bounce_only", "park_bounce_gain",
    ):
        snap[f"dec_{attr}"] = _safe(getattr(decision, attr, None))
    return snap


# ── Condition Tracker: floating temporal features ───────────────────
#
# Watches audio signals frame-by-frame and records *when* they last
# changed significantly.  The per-frame output is a dict of
# ``ct_*`` columns that give "seconds since <condition changed>",
# plus rolling deltas and beat-relative timings.
#
# These "floating" features capture exactly what the user needs:
#   flux starts rising at T=5.0,  human presses → at T=7.0
#   → ct_since_flux_rise = 2.0 seconds
#   → at 120 BPM that's ~4 beats  ← the gating timing parameter

class ConditionTracker:
    """Lightweight per-session tracker of audio condition transitions."""

    # Thresholds for detecting "significant change" in each signal.
    # These are intentionally simple — the point is to find *when*
    # something happened, not *exactly how much* it moved.
    _FLUX_RISE_RATIO = 1.40       # flux > ema * ratio → "flux rose"
    _FLUX_FALL_RATIO = 0.60       # flux < ema * ratio → "flux fell"
    _BAND_ARRIVE_THRESH = 0.08    # band mean crosses above → "arrived"
    _BAND_DEPART_THRESH = 0.03    # band mean crosses below → "departed"
    _EMA_ALPHA = 0.08             # for flux/band tracking EMAs

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        now = time.perf_counter()
        # Flux tracking
        self._flux_ema: float = 0.0
        self._flux_prev: float = 0.0
        self._last_flux_rise_mono: float = now
        self._last_flux_fall_mono: float = now

        # Band arrival / departure
        self._band_emas: dict[str, float] = {
            "sub_bass": 0.0, "low_mid": 0.0, "mid": 0.0, "high": 0.0,
        }
        self._band_present: dict[str, bool] = {k: False for k in self._band_emas}
        self._band_arrive_mono: dict[str, float] = {k: now for k in self._band_emas}
        self._band_depart_mono: dict[str, float] = {k: now for k in self._band_emas}

        # Silence transitions
        self._last_silence_enter_mono: float = now
        self._last_silence_exit_mono: float = now
        self._was_silent: bool = False

        # Beat events
        self._last_beat_mono: float = now
        self._last_downbeat_mono: float = now
        self._beat_count_session: int = 0

        # Gate-fail transitions
        self._last_gate_fail: str = ""
        self._last_gate_fail_change_mono: float = now
        self._last_gate_open_mono: float = now      # last time gate_fail went from blocked → ""
        self._last_gate_close_mono: float = now     # last time gate_fail went from "" → blocked

        # Trigger kind transitions
        self._last_trigger_kind: str = "creep"
        self._last_trigger_change_mono: float = now

    def update(self, event: Any, decision: Any = None, gate_state: dict | None = None) -> dict[str, Any]:
        """Feed one frame, return floating-feature dict (``ct_*`` prefix)."""
        now = time.perf_counter()
        out: dict[str, Any] = {}

        # ── Flux tracking ──
        flux = float(getattr(event, "spectral_flux", 0.0) or 0.0)
        self._flux_ema += self._EMA_ALPHA * (flux - self._flux_ema)
        flux_delta = flux - self._flux_prev
        self._flux_prev = flux

        if self._flux_ema > 1e-6 and flux > self._flux_ema * self._FLUX_RISE_RATIO:
            self._last_flux_rise_mono = now
        if self._flux_ema > 1e-6 and flux < self._flux_ema * self._FLUX_FALL_RATIO:
            self._last_flux_fall_mono = now

        out["ct_flux_ema"] = round(self._flux_ema, 5)
        out["ct_flux_delta"] = round(flux_delta, 5)
        out["ct_since_flux_rise_s"] = round(now - self._last_flux_rise_mono, 4)
        out["ct_since_flux_fall_s"] = round(now - self._last_flux_fall_mono, 4)

        # ── Band arrival / departure ──
        if gate_state:
            band_vals = {
                "sub_bass": float(gate_state.get("gs_sub_bass", 0.0)),
                "low_mid": float(gate_state.get("gs_low_mid", 0.0)),
                "mid": float(gate_state.get("gs_mid", 0.0)),
                "high": float(gate_state.get("gs_high", 0.0)),
            }
        else:
            band_vals = {k: 0.0 for k in self._band_emas}

        for band, val in band_vals.items():
            self._band_emas[band] += self._EMA_ALPHA * (val - self._band_emas[band])
            was_present = self._band_present[band]
            is_present = self._band_emas[band] >= self._BAND_ARRIVE_THRESH

            if is_present and not was_present:
                self._band_arrive_mono[band] = now
                self._band_present[band] = True
            elif not is_present and was_present and self._band_emas[band] < self._BAND_DEPART_THRESH:
                self._band_depart_mono[band] = now
                self._band_present[band] = False

            out[f"ct_since_{band}_arrive_s"] = round(now - self._band_arrive_mono[band], 4)
            out[f"ct_since_{band}_depart_s"] = round(now - self._band_depart_mono[band], 4)
            out[f"ct_{band}_present"] = int(self._band_present[band])

        # ── Silence transitions ──
        silence_active = False
        if decision is not None:
            silence_active = bool(getattr(decision, "silence_active", False))
        elif gate_state:
            silence_active = bool(gate_state.get("gs_silence_active", 0))

        if silence_active and not self._was_silent:
            self._last_silence_enter_mono = now
        elif not silence_active and self._was_silent:
            self._last_silence_exit_mono = now
        self._was_silent = silence_active

        out["ct_since_silence_enter_s"] = round(now - self._last_silence_enter_mono, 4)
        out["ct_since_silence_exit_s"] = round(now - self._last_silence_exit_mono, 4)

        # ── Beat events ──
        is_beat = bool(getattr(event, "is_beat", False))
        is_downbeat = bool(getattr(event, "is_downbeat", False))
        if is_beat or is_downbeat:
            self._last_beat_mono = now
            self._beat_count_session += 1
        if is_downbeat:
            self._last_downbeat_mono = now

        out["ct_since_last_beat_s"] = round(now - self._last_beat_mono, 4)
        out["ct_since_last_downbeat_s"] = round(now - self._last_downbeat_mono, 4)
        out["ct_beat_count"] = self._beat_count_session

        # ── Beats-since estimates (using current BPM) ──
        bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
        if bpm <= 0:
            bpm = float(getattr(event, "bpm", 0.0) or 0.0)
        if bpm > 0:
            beat_period = 60.0 / bpm
            out["ct_beats_since_flux_rise"] = round((now - self._last_flux_rise_mono) / beat_period, 2)
            out["ct_beats_since_flux_fall"] = round((now - self._last_flux_fall_mono) / beat_period, 2)
            out["ct_beats_since_silence_exit"] = round((now - self._last_silence_exit_mono) / beat_period, 2)
            out["ct_beats_since_bass_arrive"] = round(
                (now - self._band_arrive_mono.get("sub_bass", now)) / beat_period, 2)
            out["ct_beats_since_gate_open"] = round((now - self._last_gate_open_mono) / beat_period, 2)
            out["ct_beats_since_gate_close"] = round((now - self._last_gate_close_mono) / beat_period, 2)
            out["ct_beats_since_trigger_change"] = round((now - self._last_trigger_change_mono) / beat_period, 2)
        else:
            out["ct_beats_since_flux_rise"] = -1.0
            out["ct_beats_since_flux_fall"] = -1.0
            out["ct_beats_since_silence_exit"] = -1.0
            out["ct_beats_since_bass_arrive"] = -1.0
            out["ct_beats_since_gate_open"] = -1.0
            out["ct_beats_since_gate_close"] = -1.0
            out["ct_beats_since_trigger_change"] = -1.0

        # ── Gate-fail transitions ──
        gate_fail = ""
        if decision is not None:
            gate_fail = str(getattr(decision, "gate_fail", "") or "")
        elif gate_state:
            # gate_fail isn't in gate_state directly; it's on the decision
            pass

        if gate_fail != self._last_gate_fail:
            self._last_gate_fail_change_mono = now
            if gate_fail == "" and self._last_gate_fail != "":
                self._last_gate_open_mono = now   # gate just opened
            elif gate_fail != "" and self._last_gate_fail == "":
                self._last_gate_close_mono = now  # gate just closed
            self._last_gate_fail = gate_fail

        out["ct_current_gate_fail"] = gate_fail
        out["ct_since_gate_fail_change_s"] = round(now - self._last_gate_fail_change_mono, 4)
        out["ct_since_gate_open_s"] = round(now - self._last_gate_open_mono, 4)
        out["ct_since_gate_close_s"] = round(now - self._last_gate_close_mono, 4)

        # ── Trigger-kind transitions ──
        trigger_kind = "creep"
        if decision is not None:
            trigger_kind = str(getattr(decision, "trigger_kind", "creep") or "creep")
        elif gate_state:
            trigger_kind = str(gate_state.get("gs_last_trigger_kind", "creep"))

        if trigger_kind != self._last_trigger_kind:
            self._last_trigger_change_mono = now
            self._last_trigger_kind = trigger_kind

        out["ct_since_trigger_change_s"] = round(now - self._last_trigger_change_mono, 4)

        return out


# ── Core recorder ───────────────────────────────────────────────────

# Directive constants — kept as plain strings for CSV friendliness
DIR_MORE   = "more"     # ↑
DIR_LESS   = "less"     # ↓
DIR_SLOWER = "slower"   # ←
DIR_FASTER = "faster"   # →
DIR_NONE   = "none"     # idle frame (no key held)


class KeyboardTeacher:
    """Records human motion directives alongside live audio conditions.

    Key scheme (discrete latching):
        ↓         park (latch on — bounce at park position)
        →         if parked: leave park at 1x speed
                  if moving: multiply speed ×2
        ←         divide speed ×0.5 (works while parked to pre-set step)
        ↑         unpark at current speed step (no speed change)

    Base 1x speed = one full geometry rotation per measure (4 beats).
    Speed steps: …, 1/8x, 1/4x, 1/2x, 1x, 2x, 4x, 8x, …
    """

    _SPEED_STEP_MIN: int = -4   # 1/16x
    _SPEED_STEP_MAX: int = 4    # 16x

    def __init__(self, base_dir: str | Path = "."):
        self._base = Path(base_dir)
        self._lock = Lock()
        self.active = False
        self._session_dir: Optional[Path] = None
        self._rows: list[dict[str, Any]] = []

        # Discrete latch state
        self._is_parked: bool = True
        self._speed_step: int = 0            # 0 = 1x; +1 = 2x; -1 = 0.5x

        # Timing
        self._session_start_mono: float = 0.0
        self._last_directive_change_mono: float = 0.0
        self._last_directive: str = "park"
        self._frame_count: int = 0

        # Last known BPM (updated by on_frame; read by canvas preview)
        self._last_bpm: float = 120.0

        # Floating condition tracker
        self._condition_tracker = ConditionTracker()

    # ── Session lifecycle ───────────────────────────────────────────

    def start_session(self) -> Path:
        with self._lock:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._session_dir = self._base / "teaching_captures" / "keyboard" / f"session_{ts}"
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._rows.clear()
            self._is_parked = True
            self._speed_step = 0
            self._last_bpm = 120.0
            self._session_start_mono = time.perf_counter()
            self._last_directive_change_mono = self._session_start_mono
            self._last_directive = "park"
            self._frame_count = 0
            self._condition_tracker.reset()
            self.active = True
            return self._session_dir

    def stop_session(self) -> Optional[Path]:
        with self._lock:
            if not self.active:
                return None
            self._flush_locked()
            self.active = False
            return self._session_dir

    # ── Key events (call from Qt keyPressEvent / keyReleaseEvent) ───

    def on_arrow_down(self, direction: str) -> None:
        """Discrete press handler — all state changes happen on keydown only.

        direction: 'up' | 'down' | 'left' | 'right'
        """
        with self._lock:
            if not self.active:
                return
            if direction == "down":
                self._is_parked = True
            elif direction == "right":
                if self._is_parked:
                    # Leave park at 1x (step 0)
                    self._is_parked = False
                    self._speed_step = 0
                else:
                    self._speed_step = min(self._speed_step + 1, self._SPEED_STEP_MAX)
            elif direction == "left":
                # Halve speed (works while parked: pre-sets step for when you unpark)
                self._speed_step = max(self._speed_step - 1, self._SPEED_STEP_MIN)
            elif direction == "up":
                # Unpark at current speed, no step change
                self._is_parked = False
            self._update_directive_label()

    def on_arrow_up(self, direction: str) -> None:  # noqa: ARG002
        """No-op — all state is latched on keydown."""
        pass

    # ── Per-frame recording (called from audio callback path) ───────

    def on_frame(
        self,
        event: Any,
        decision: Any = None,
        gate_state: dict | None = None,
        dt: float = 1 / 60,
    ) -> None:
        """Record one row per audio frame with current axes + audio snapshot.

        Args:
            event:      BeatEvent from audio engine.
            decision:   BeatDecision from beat_intelligence (may be None on first frames).
            gate_state: dict from BeatIntelligence.snapshot_gate_state() (may be None).
            dt:         Frame delta time in seconds.
        """
        with self._lock:
            if not self.active:
                return
            self._frame_count += 1

            # Track live BPM from audio event
            bpm = float(getattr(event, "metronome_bpm", 0.0) or 0.0)
            if bpm <= 0:
                bpm = float(getattr(event, "bpm", 0.0) or 0.0)
            if bpm > 0:
                self._last_bpm = bpm

            now_mono = time.perf_counter()
            elapsed = now_mono - self._session_start_mono
            since_change = now_mono - self._last_directive_change_mono

            row: dict[str, Any] = {
                "frame": self._frame_count,
                "session_time_s": round(elapsed, 4),
                "since_directive_change_s": round(since_change, 4),
                "directive": self._last_directive,
                "is_parked": int(self._is_parked),
                "speed_step": self._speed_step,
                "speed_scale": round(2.0 ** self._speed_step, 5),
                "bpm_at_frame": round(self._last_bpm, 2),
            }

            # Flatten audio conditions
            row.update(_snapshot_from_event(event))
            if decision is not None:
                row.update(_snapshot_from_decision(decision))

            # Gate-state snapshot from BeatIntelligence internals
            if gate_state:
                row.update(gate_state)

            # Floating temporal features (condition tracker)
            ct = self._condition_tracker.update(event, decision, gate_state)
            row.update(ct)

            self._rows.append(row)

            # Auto-flush every 3000 frames (~50 s at 60 fps) to avoid data loss
            if len(self._rows) >= 3000:
                self._flush_locked()

    # ── Public queries ──────────────────────────────────────────────

    @property
    def is_parked(self) -> bool:
        return self._is_parked

    @property
    def speed_scale(self) -> float:
        """Current speed multiplier (1x = one rotation per measure)."""
        return 2.0 ** self._speed_step

    @property
    def speed_step(self) -> int:
        return self._speed_step

    # Kept for backward compat with any callers that use the old axis names
    @property
    def intensity(self) -> float:
        return -1.0 if self._is_parked else 0.0

    @property
    def speed(self) -> float:
        return float(self.speed_scale)

    @property
    def current_directive(self) -> str:
        return self._last_directive

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir

    @property
    def last_gate_fail(self) -> str:
        """Most recent gate_fail value seen by the condition tracker."""
        return self._condition_tracker._last_gate_fail

    # ── Internals ───────────────────────────────────────────────────

    def _update_directive_label(self) -> None:
        """Build a human-readable directive from the current discrete state."""
        if self._is_parked:
            new_dir = "park"
        else:
            scale = 2.0 ** self._speed_step
            if scale >= 1.0:
                new_dir = f"{scale:.0f}x"
            else:
                new_dir = f"1/{round(1.0/scale):.0f}x"
        if new_dir != self._last_directive:
            self._last_directive = new_dir
            self._last_directive_change_mono = time.perf_counter()

    def _flush_locked(self) -> None:
        if not self._rows or self._session_dir is None:
            return
        csv_path = self._session_dir / "directives.csv"
        _append_csv(csv_path, self._rows)
        self._rows.clear()

    def __del__(self) -> None:
        try:
            self.stop_session()
        except Exception:
            pass


# ── Utilities ───────────────────────────────────────────────────────

def _move_toward(current: float, target: float, max_delta: float) -> float:
    diff = target - current
    if abs(diff) <= max_delta:
        return target
    return current + (max_delta if diff > 0 else -max_delta)


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a CSV, writing a header only if the file is new/empty."""
    if not rows:
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
