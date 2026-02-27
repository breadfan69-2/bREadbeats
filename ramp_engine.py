"""Session-length intensity ramp engine.

Two independent ramp channels controlled by the Duration (hrs) slider
and the Size / Speed / Both selector on the main GUI:

- **speed**  – Raises ``config.stroke.energy_response_strength`` from its
  current value to 2.0 (maximum) over the timer duration.  This makes the
  fill gate progressively easier to pass as energy builds, so more beats
  produce orbit motion.

- **size**   – Expands both the fill-orbit radius and the beat-orbit
  ``max_radius`` by up to 60 % over the timer duration, making the overall
  motion pattern physically larger.

- **both**   – Both channels run simultaneously.

If the user manually moves the Energy Response slider while the speed
ramp is running, the ramp detects the change, adopts the new value as
its starting point, and restarts the timer so it ramps from the user's
value to 2.0 over the full remaining duration.
"""

from __future__ import annotations

import numpy as np

from config import Config
from geometry_utils import quintic_ease


class RampEngine:
    """Stateful session ramp driven once per audio frame."""

    # Maximum expansion ratio for the size channel (60 %).
    SIZE_MAX_EXPANSION: float = 0.60

    def __init__(self, config: Config) -> None:
        self.config = config

        # ── Speed channel ──
        self._speed_started: bool = False
        self._speed_complete: bool = False
        self._speed_start_time: float = 0.0
        self._speed_start_value: float = 1.0
        self._speed_last_written: float | None = None

        # ── Size channel ──
        self._size_started: bool = False
        self._size_complete: bool = False
        self._size_start_time: float = 0.0

        # ── Public outputs (read by stroke_mapper) ──
        self.size_expansion: float = 0.0  # 0.0 → SIZE_MAX_EXPANSION

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @property
    def _ramp_hours(self) -> float:
        return float(getattr(self.config.stroke, 'intensity_ramp_hours', 0.0) or 0.0)

    @property
    def _ramp_target(self) -> str:
        t = str(getattr(self.config.stroke, 'intensity_ramp_target', 'both') or 'both').strip().lower()
        return t if t in ('size', 'speed', 'both') else 'both'

    @property
    def speed_active(self) -> bool:
        return self._ramp_target in ('speed', 'both')

    @property
    def size_active(self) -> bool:
        return self._ramp_target in ('size', 'both')

    # ------------------------------------------------------------------
    # Per-frame tick (called from stroke_mapper.process_beat)
    # ------------------------------------------------------------------

    def tick(self, now: float, silence_active: bool) -> None:
        """Advance both ramp channels by one frame."""
        ramp_hours = self._ramp_hours
        if ramp_hours <= 0.0:
            self._reset()
            return

        ramp_s = ramp_hours * 3600.0

        # Speed channel
        if self.speed_active:
            self._tick_speed(now, silence_active, ramp_s)
        else:
            self._speed_started = False
            self._speed_complete = False
            self._speed_last_written = None

        # Size channel
        if self.size_active:
            self._tick_size(now, silence_active, ramp_s)
        else:
            self._size_started = False
            self._size_complete = False
            self.size_expansion = 0.0

    # ------------------------------------------------------------------
    # Speed channel
    # ------------------------------------------------------------------

    def _tick_speed(self, now: float, silence_active: bool, ramp_s: float) -> None:
        if self._speed_complete:
            return

        _v = getattr(self.config.stroke, 'energy_response_strength', 1.0)
        current_cfg = float(_v if _v is not None else 1.0)

        # First non-silence frame → arm the ramp
        if not silence_active and not self._speed_started:
            self._speed_started = True
            self._speed_start_time = now
            self._speed_start_value = current_cfg
            self._speed_last_written = None

        if not self._speed_started:
            return

        # Detect manual slider move: config value differs from what we
        # last wrote → user dragged the slider.  Re-base and restart.
        if (
            self._speed_last_written is not None
            and abs(current_cfg - self._speed_last_written) > 0.005
        ):
            self._speed_start_value = current_cfg
            self._speed_start_time = now

        elapsed = now - self._speed_start_time
        raw_t = float(np.clip(elapsed / max(ramp_s, 1.0), 0.0, 1.0))
        eased_t = quintic_ease(raw_t)

        target = 2.0
        new_value = float(
            self._speed_start_value
            + (target - self._speed_start_value) * eased_t
        )
        new_value = float(np.clip(new_value, 0.0, 2.0))

        setattr(self.config.stroke, 'energy_response_strength', new_value)
        self._speed_last_written = new_value

        if raw_t >= 1.0:
            self._speed_complete = True

    # ------------------------------------------------------------------
    # Size channel
    # ------------------------------------------------------------------

    def _tick_size(self, now: float, silence_active: bool, ramp_s: float) -> None:
        if self._size_complete:
            self.size_expansion = self.SIZE_MAX_EXPANSION
            return

        if not silence_active and not self._size_started:
            self._size_started = True
            self._size_start_time = now

        if not self._size_started:
            self.size_expansion = 0.0
            return

        elapsed = now - self._size_start_time
        raw_t = float(np.clip(elapsed / max(ramp_s, 1.0), 0.0, 1.0))
        eased_t = quintic_ease(raw_t)

        self.size_expansion = float(self.SIZE_MAX_EXPANSION * eased_t)

        if raw_t >= 1.0:
            self._size_complete = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Reset both channels (ramp disabled or hours == 0)."""
        self._speed_started = False
        self._speed_complete = False
        self._speed_last_written = None
        _v = getattr(self.config.stroke, 'energy_response_strength', 1.0)
        self._speed_start_value = float(_v if _v is not None else 1.0)
        self._size_started = False
        self._size_complete = False
        self.size_expansion = 0.0

    @property
    def speed_display_value(self) -> float | None:
        """Return current energy_response_strength if speed ramp is actively
        driving it, else ``None``.  Used by the GUI to sync the slider."""
        if self._speed_started and self._speed_last_written is not None:
            return self._speed_last_written
        return None
