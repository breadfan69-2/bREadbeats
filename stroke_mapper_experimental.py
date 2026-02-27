"""
bREadbeats – Experimental Simple-Mode Stroke Mapper
====================================================

Beat-reactive Y-axis shuttle with half-circle arc projection.

Concept
-------
Instead of continuous circular orbit, this mode:

1.  **Shuttles on the Y axis** in sync with detected beats.
    - One full travel (bottom → top **or** top → bottom) = **1 beat**.
    - A complete round-trip (bottom → top → bottom) = **2 beats**.

2.  **On each beat fire**, the direction reverses instantly so the
    stroke always starts a fresh leg at the exact moment the detector
    fires, eliminating timing drift that accumulates in free-running
    orbit modes.

3.  The 1-D shuttle positions are then projected onto a **half-circle
    arc** in the alpha/beta plane so the device traces a smooth curved
    path rather than a harsh linear ramp.

Coordinate contract
-------------------
* ``alpha`` / ``beta`` ∈ [-1, 1] — same as the rest of the stroke mapper.
* ``volume`` ∈ [0, 1].

Integration
-----------
Call ``process()`` every audio frame (~25 Hz / 40 ms) with the current
``BeatEvent`` and ``BeatDecision``.  It returns a ``TCodeCommand``
ready to send on the wire.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from audio_engine import BeatEvent
from beat_intelligence import BeatDecision, BeatIntelligence
from config import Config
from geometry_utils import exponential_approach, quintic_ease
from network_engine import TCodeCommand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _half_circle_arc_y(position: float, radius: float, direction: float,
                       center_alpha: float = 0.0,
                       center_beta: float = 0.0) -> tuple[float, float]:
    """Map a linear shuttle position to a semicircular arc.

    **Y (beta) is the travel axis** — the shuttle sweeps bottom ↔ top.
    **X (alpha) bulges outward** to one side, determined by *direction*.

    direction = +1  →  arc bulges to +X  (right semicircle, bottom→top)
    direction = -1  →  arc bulges to -X  (left semicircle, top→bottom)

    Over two consecutive beats (one up, one down) the pair of
    semicircles covers a **full 360° circle**.

    position ∈ [0, 1]:  0 = bottom (-R), 0.5 = midpoint, 1 = top (+R).
    """
    theta = position * math.pi                      # 0 → π
    beta  = center_beta  - radius * math.cos(theta) # Y travel: -R → +R
    alpha = center_alpha + direction * radius * math.sin(theta)  # X arc
    return float(alpha), float(beta)


# ---------------------------------------------------------------------------
# Shuttle state
# ---------------------------------------------------------------------------

@dataclass
class ShuttleState:
    """Mutable state for the beat-reactive Y shuttle."""

    # Current normalised position along the leg  0.0 → 1.0
    leg_progress: float = 0.0

    # +1 = travelling "up" (pos 0→1),  -1 = travelling "down" (pos 1→0)
    direction: int = 1

    # Latched speed: how fast to traverse the leg (units per second).
    # Recomputed on each beat fire from current BPM.
    speed: float = 2.0          # default ~120 BPM (2 legs/sec)

    # Most recent BPM used for speed calc
    last_bpm: float = 120.0

    # Monotonic timestamp of the last beat fire
    last_beat_time: float = 0.0

    # Whether we are actively shuttling (vs parked/silent)
    active: bool = False

    # Arc tuning
    radius: float = 0.90
    center_alpha: float = 0.0
    center_beta: float = 0.0

    # Output position (for state tracking / smoothing)
    alpha: float = 0.0
    beta: float = 0.20


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class ExperimentalSimpleMapper:
    """Beat-reactive shuttle with half-circle arc projection.

    Usage::

        mapper = ExperimentalSimpleMapper(config, get_volume, audio_engine)
        mapper.set_intelligence(intelligence)  # share the BeatIntelligence

        # Every audio frame:
        cmd = mapper.process(event, decision, dt, now)
    """

    def __init__(
        self,
        config: Config,
        get_volume: Optional[Callable[[], float]] = None,
        audio_engine=None,
    ):
        self.config = config
        self.get_volume = get_volume if get_volume is not None else (lambda: 1.0)
        self.audio_engine = audio_engine
        self._intelligence: Optional[BeatIntelligence] = None

        self.state = ShuttleState()

        # Park position (silence)
        self._park_alpha = 0.0
        self._park_beta = 0.20

        # Easing curve for the shuttle traverse
        # 'linear', 'sine', 'quintic' — quintic has zero velocity at endpoints (smoothest)
        self._ease_mode = 'quintic'

        # Beat fire tracking: avoid double-firing in the same frame
        self._last_fire_mono: float = 0.0
        self._min_fire_interval_s: float = 0.05   # 50 ms debounce

        # Arc-direction crossfade: blends the X-bulge side smoothly at turnarounds.
        # +1.0 = fully right arc, -1.0 = fully left arc, 0.0 = straight line.
        # Exponentially chases self.state.direction after each beat fire.
        self._arc_dir_blend: float = 1.0
        # Rate at which blend chases target (per second). ~16 → ~95% done in 180ms.
        self._arc_crossfade_rate: float = 16.0

        # Silence / ramp state
        self._silence_glide_rate: float = 2.0      # park glide speed

        # Sub-bass bounce while parked (same as original simple mode)
        self._bounce_freq_hz: float = 1.5
        self._bounce_radius: float = 0.075

    # -- wiring ----------------------------------------------------------

    def set_intelligence(self, intelligence: BeatIntelligence) -> None:
        self._intelligence = intelligence

    # -- easing helpers --------------------------------------------------

    def _ease(self, t: float) -> float:
        """Apply easing curve to linear progress *t* ∈ [0, 1]."""
        t = float(np.clip(t, 0.0, 1.0))
        if self._ease_mode == 'quintic':
            return quintic_ease(t)
        if self._ease_mode == 'sine':
            # Sine ease-in-out: slow start, fast middle, slow end
            return float(0.5 * (1.0 - math.cos(t * math.pi)))
        return t  # linear fallback

    # -- beat fire logic -------------------------------------------------

    def _fire_beat(self, bpm: float, now: float) -> None:
        """Reverse shuttle direction and latch new speed from BPM.

        Called on each confirmed beat the direction reverses and speed
        re-latches from the current BPM.

        Speed is set to ~92% of a full leg per beat period so the shuttle
        is always *approaching* the pole when the beat fires — never sitting
        and waiting.  The beat IS the bounce event.  The end-of-travel
        fallback only fires when beats drop out entirely.
        """
        if bpm <= 0:
            return

        beat_period_s = 60.0 / bpm   # seconds per beat

        # 0.92× means the shuttle reaches ~92% by the time the next beat
        # arrives, so the beat always drives the reversal without a stall.
        _SPEED_FACTOR = 0.92
        self.state.speed = _SPEED_FACTOR / beat_period_s

        # Reverse direction — don't snap progress; we're near the pole
        # already and avoiding the snap prevents a beta jump.
        self.state.direction *= -1

        self.state.last_beat_time = now
        self.state.last_bpm = bpm
        self.state.active = True

    def _should_fire(self, event: BeatEvent, decision: BeatDecision,
                     now: float) -> bool:
        """Decide whether to trigger a direction reversal this frame.

        Fires on:
        1. Actual beat detected by the audio engine  (primary path).
        2. End-of-travel safety: shuttle overshot past 98%/2% and no beat
           has arrived yet — keeps motion going during beat dropout.
        """
        # Debounce
        if (now - self._last_fire_mono) < self._min_fire_interval_s:
            return False

        # Beat detected by audio engine
        if event.is_beat and decision.trigger_kind in ("beat", "downbeat"):
            self._last_fire_mono = now
            return True

        # End-of-travel safety fallback (only fires on beat dropout)
        if self.state.active:
            if self.state.direction == 1 and self.state.leg_progress >= 0.98:
                self._last_fire_mono = now
                return True
            if self.state.direction == -1 and self.state.leg_progress <= 0.02:
                self._last_fire_mono = now
                return True

        return False

    # -- frame update ----------------------------------------------------

    def process(
        self,
        event: BeatEvent,
        decision: BeatDecision,
        dt: float,
        now: float,
    ) -> TCodeCommand:
        """Run one frame of the experimental simple-mode mapper.

        Parameters
        ----------
        event : BeatEvent
            Raw audio event from the engine.
        decision : BeatDecision
            Pre-built decision from BeatIntelligence.
        dt : float
            Seconds since last frame (clamped upstream).
        now : float
            Monotonic timestamp of this frame.

        Returns
        -------
        TCodeCommand
        """
        fade = float(np.clip(decision.silence_fade, 0.0, 1.0))
        ramp = float(np.clip(decision.post_silence_ramp, 0.0, 1.0))

        # ── Silence / park ──────────────────────────────────────────
        if decision.silence_active:
            return self._park_frame(event, decision, dt, now, fade)

        # ── Creep (no tempo yet) ────────────────────────────────────
        if decision.trigger_kind == "creep":
            return self._park_frame(event, decision, dt, now,
                                    min(fade, ramp))

        # ── Active shuttling ────────────────────────────────────────
        bpm = event.bpm if event.bpm > 0 else event.metronome_bpm

        # Check for beat fire (real beat or end-of-travel)
        if self._should_fire(event, decision, now):
            self._fire_beat(bpm, now)
        elif not self.state.active and bpm > 0:
            # First activation after park
            self._fire_beat(bpm, now)

        # Advance shuttle position
        if self.state.active and self.state.speed > 0:
            step = self.state.speed * dt
            if self.state.direction == 1:
                self.state.leg_progress = min(1.0, self.state.leg_progress + step)
            else:
                self.state.leg_progress = max(0.0, self.state.leg_progress - step)

        # Apply easing
        eased = self._ease(self.state.leg_progress)

        # Project onto half-circle arc — use blended direction for smooth X-morph
        # Advance crossfade blend toward current integer direction each frame
        self._arc_dir_blend = float(
            exponential_approach(
                self._arc_dir_blend,
                float(self.state.direction),
                self._arc_crossfade_rate,
                dt,
            )
        )
        radius = float(np.clip(decision.radius_bloom, 0.80, 1.0))
        alpha, beta = _half_circle_arc_y(
            eased, radius,
            direction=self._arc_dir_blend,
            center_alpha=self.state.center_alpha,
            center_beta=self.state.center_beta,
        )

        # Commit
        volume = float(np.clip(self.get_volume() * min(fade, ramp), 0.0, 1.0))
        self.state.alpha = float(np.clip(alpha, -1.0, 1.0))
        self.state.beta = float(np.clip(beta, -1.0, 1.0))

        return TCodeCommand(
            alpha=self.state.alpha,
            beta=self.state.beta,
            duration_ms=25,
            volume=volume,
        )

    # -- park / silence helper -------------------------------------------

    def _park_frame(
        self,
        event: BeatEvent,
        decision: BeatDecision,
        dt: float,
        now: float,
        vol_scale: float,
    ) -> TCodeCommand:
        """Glide to park position with optional sub-bass bounce."""
        self.state.active = False
        self._arc_dir_blend = float(self.state.direction)  # reset blend on park

        glide_t = float(np.clip(self._silence_glide_rate * dt, 0.0, 1.0))
        alpha = float(self.state.alpha
                       + (self._park_alpha - self.state.alpha) * glide_t)
        beta = float(self.state.beta
                      + (self._park_beta - self.state.beta) * glide_t)

        # Sub-bass bounce
        if self._intelligence is not None:
            sub_bass = float(np.clip(
                self._intelligence.energies.sub_bass, 0.0, 1.0))
            if sub_bass > 0.05:
                br = self._bounce_radius * (1.0 + 0.80 * sub_bass)
                bp = float(now * 2.0 * math.pi * self._bounce_freq_hz)
                alpha += float(br * math.cos(bp) * sub_bass)
                beta += float(br * math.sin(bp) * sub_bass * 0.5)

        volume = float(np.clip(self.get_volume() * vol_scale, 0.0, 1.0))
        self.state.alpha = float(np.clip(alpha, -1.0, 1.0))
        self.state.beta = float(np.clip(beta, -1.0, 1.0))

        return TCodeCommand(
            alpha=self.state.alpha,
            beta=self.state.beta,
            duration_ms=25,
            volume=volume,
        )

    # -- diagnostics -----------------------------------------------------

    def get_current_position(self) -> tuple[float, float]:
        return self.state.alpha, self.state.beta

    def get_debug_info(self) -> dict:
        """Snapshot for keyboard teacher / debug overlay."""
        return {
            "leg_progress": self.state.leg_progress,
            "direction": self.state.direction,
            "speed": self.state.speed,
            "last_bpm": self.state.last_bpm,
            "active": self.state.active,
            "ease_mode": self._ease_mode,
        }
