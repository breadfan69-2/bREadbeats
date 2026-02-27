"""Adaptive pipeline-lead estimator.

Auto-adjusts the ``scheduled_lead_ms`` compensation per track so that
beat-journey arcs land on the musical beat rather than consistently
early or late.

Usage::

    lead = AdaptiveLead(base_lead_ms=55.0)

    # Called once per journey start (in BeatIntelligence):
    lead_s = lead.get_lead_s()

    # Called every time a downbeat with phase_error_ms is observed:
    lead.observe(phase_error_ms)

    # Called on track change / silence gate enter:
    lead.reset()

Design constraints:
    * Pure data – no imports from audio_engine or beat_intelligence.
    * Clamps output to [0, max_lead_ms] to prevent runaway.
    * Converges in ~6-8 beats via EMA on observed error.
    * Resets to base_lead_ms on track boundaries.
"""

from __future__ import annotations

__all__ = ["AdaptiveLead"]


class AdaptiveLead:
    """Tracks per-track phase error and adapts the pipeline lead.

    Positive ``phase_error_ms`` means the journey arrived **late**
    (musical beat happened before the orbit reached the anchor).
    Negative means **early**.

    The adaptive lead increases when late, decreases when early,
    converging on the value that zeros out the average error.

    Parameters
    ----------
    base_lead_ms : float
        Starting lead (the config default, e.g. 55 ms).
    ema_alpha : float
        Smoothing factor for the error EMA (0–1).
        Higher = faster adaptation, noisier.  Default 0.25.
    correction_gain : float
        Fraction of observed error applied per observation.
        Default 0.35 (35% of the EMA error added to lead).
    max_lead_ms : float
        Hard ceiling to prevent runaway.  Default 120 ms.
    min_lead_ms : float
        Hard floor.  Default 0 ms.
    min_observations : int
        Require this many observations before adapting.
        Default 3 (ignore first few noisy beats).
    """

    def __init__(
        self,
        base_lead_ms: float = 55.0,
        ema_alpha: float = 0.25,
        correction_gain: float = 0.35,
        max_lead_ms: float = 120.0,
        min_lead_ms: float = 0.0,
        min_observations: int = 3,
    ) -> None:
        self._base_lead_ms = float(base_lead_ms)
        self._ema_alpha = float(ema_alpha)
        self._correction_gain = float(correction_gain)
        self._max_lead_ms = float(max_lead_ms)
        self._min_lead_ms = float(min_lead_ms)
        self._min_observations = int(min_observations)

        # Mutable state – reset per track
        self._current_lead_ms: float = self._base_lead_ms
        self._error_ema: float = 0.0
        self._observation_count: int = 0

    # ── Public API ──

    def get_lead_s(self) -> float:
        """Return the current adaptive lead in **seconds** (for journey shortening)."""
        return max(0.0, self._current_lead_ms / 1000.0)

    def get_lead_ms(self) -> float:
        """Return the current adaptive lead in milliseconds."""
        return self._current_lead_ms

    def observe(self, phase_error_ms: float) -> None:
        """Feed an observed phase error (positive = late, negative = early).

        Call this once per downbeat (or per beat if you prefer faster convergence).
        """
        self._observation_count += 1

        # Update EMA of error
        if self._observation_count == 1:
            self._error_ema = float(phase_error_ms)
        else:
            self._error_ema = (
                (1.0 - self._ema_alpha) * self._error_ema
                + self._ema_alpha * float(phase_error_ms)
            )

        # Only start adapting after min_observations
        if self._observation_count < self._min_observations:
            return

        # Adjust lead: if we're late (positive error), increase lead;
        # if early (negative error), decrease lead.
        adjustment = self._error_ema * self._correction_gain
        self._current_lead_ms = max(
            self._min_lead_ms,
            min(self._max_lead_ms, self._current_lead_ms + adjustment),
        )

    def reset(self) -> None:
        """Reset to base lead.  Call on track change / silence boundary."""
        self._current_lead_ms = self._base_lead_ms
        self._error_ema = 0.0
        self._observation_count = 0

    # ── Diagnostics ──

    @property
    def observation_count(self) -> int:
        return self._observation_count

    @property
    def error_ema_ms(self) -> float:
        return self._error_ema

    def __repr__(self) -> str:
        return (
            f"AdaptiveLead(lead={self._current_lead_ms:.1f}ms, "
            f"ema={self._error_ema:.1f}ms, n={self._observation_count})"
        )
