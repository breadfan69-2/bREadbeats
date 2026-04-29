"""
Volume Normalizer - Compensates for Windows system volume in WASAPI loopback capture.

Problem: WASAPI loopback captures audio *after* Windows applies the master endpoint
volume.  If the user has Windows volume at 30%, the captured PCM is ~30% amplitude,
which makes beat detection settings volume-dependent.  Turning Windows volume up
later means previously-tuned sensitivity is now way too aggressive.

Solution: Read the current Windows endpoint master volume via pycaw/COM and compute
a compensation gain = 1 / master_volume.  Applied to captured samples, this undoes
the system attenuation so the beat engine always sees "100%-equivalent" signal levels
regardless of what the Windows volume slider is set to.

The user's actual audio output is NOT changed — we only *read* the volume, never set it.

Usage:
    normalizer = VolumeNormalizer()
    # In the audio callback:
    gain = normalizer.get_compensation_gain()
    mono *= gain
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from logging_utils import log_event

# Minimum volume level we'll compensate for.  Below this, the user probably
# *wants* near-silence, so we cap the gain to avoid amplifying noise to insane
# levels.  At 5% volume the max compensation gain is 1/0.05 = 20×.
_MIN_VOLUME_FLOOR = 0.05

# How often (seconds) we re-read the Windows volume.  Polling every 0.5s is
# plenty responsive and negligible CPU cost vs the ~86 Hz audio callback rate.
_POLL_INTERVAL_S = 0.5


def _try_import_pycaw():
    """Lazy-import pycaw so the rest of the app works fine if it's missing."""
    try:
        from pycaw.pycaw import AudioUtilities
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import IAudioEndpointVolume
        return AudioUtilities, IAudioEndpointVolume, CLSCTX_ALL
    except ImportError:
        return None, None, None


class VolumeNormalizer:
    """Reads Windows master endpoint volume and provides a compensation gain.

    Thread-safe.  Designed to be polled from the audio callback hot path
    (returns a cached float — no COM calls on the audio thread).

    Parameters
    ----------
    enabled : bool
        If False, ``get_compensation_gain()`` always returns 1.0 (passthrough).
    poll_interval : float
        Seconds between Windows volume re-reads (default 0.5).
    min_volume_floor : float
        Minimum volume we compensate for (default 0.05 = 5%).  Volumes below
        this are clamped to avoid extreme gain.
    """

    def __init__(
        self,
        enabled: bool = True,
        poll_interval: float = _POLL_INTERVAL_S,
        min_volume_floor: float = _MIN_VOLUME_FLOOR,
    ):
        self._enabled = enabled
        self._poll_interval = poll_interval
        self._min_volume_floor = max(min_volume_floor, 0.01)

        # Cached compensation gain (read by audio thread, written by poller)
        self._gain: float = 1.0
        self._raw_volume: float = 1.0   # Last read Windows volume (for UI/debug)
        self._lock = threading.Lock()

        # COM / pycaw objects (created once on poller thread)
        self._endpoint_volume: Optional[Any] = None  # IAudioEndpointVolume interface
        self._com_initialized = False

        # Poller thread
        self._poller_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self._enabled:
            self._start_poller()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_compensation_gain(self) -> float:
        """Return the current compensation multiplier (fast, lock-free read)."""
        return self._gain  # atomic float read on CPython

    def get_raw_volume(self) -> float:
        """Return the last-read Windows master volume (0.0–1.0) for UI display."""
        return self._raw_volume

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        if value == self._enabled:
            return
        self._enabled = value
        if value:
            self._start_poller()
        else:
            self._stop_poller()
            self._gain = 1.0
            self._raw_volume = 1.0

    def shutdown(self):
        """Stop the poller thread.  Call on app exit."""
        self._stop_poller()

    # ------------------------------------------------------------------
    # Poller internals
    # ------------------------------------------------------------------

    def _start_poller(self):
        if self._poller_thread is not None and self._poller_thread.is_alive():
            return
        self._stop_event.clear()
        self._poller_thread = threading.Thread(
            target=self._poll_loop, name="VolumeNormalizer", daemon=True
        )
        self._poller_thread.start()
        log_event("INFO", "VolumeNorm", "Poller started",
                  interval=f"{self._poll_interval:.1f}s")

    def _stop_poller(self):
        self._stop_event.set()
        if self._poller_thread is not None:
            self._poller_thread.join(timeout=2.0)
            self._poller_thread = None
        # Release COM
        self._endpoint_volume = None
        self._com_initialized = False

    def _init_com(self) -> bool:
        """Initialize COM and acquire the endpoint volume interface (poller thread only)."""
        import comtypes
        try:
            comtypes.CoInitialize()
        except OSError:
            pass  # Already initialized on this thread

        AudioUtilities, _IAudioEndpointVolume, _CLSCTX_ALL = _try_import_pycaw()
        if AudioUtilities is None or _IAudioEndpointVolume is None or _CLSCTX_ALL is None:
            log_event("WARN", "VolumeNorm",
                      "pycaw not installed — volume normalization disabled. "
                      "Install with: pip install pycaw")
            return False

        try:
            device = AudioUtilities.GetSpeakers()
            if device is None:
                log_event("ERROR", "VolumeNorm", "No default speakers endpoint found")
                return False

            # pycaw >= 2024 exposes a high-level .EndpointVolume property;
            # older versions require the low-level .Activate() COM call.
            endpoint_volume = getattr(device, 'EndpointVolume', None)
            if endpoint_volume is not None:
                self._endpoint_volume = endpoint_volume
            else:
                activate = getattr(device, 'Activate', None)
                endpoint_iid = getattr(_IAudioEndpointVolume, '_iid_', None)
                if not callable(activate) or endpoint_iid is None:
                    log_event("ERROR", "VolumeNorm", "Endpoint volume interface is unavailable")
                    return False
                interface: Any = activate(endpoint_iid, _CLSCTX_ALL, None)
                self._endpoint_volume = interface.QueryInterface(_IAudioEndpointVolume)

            if self._endpoint_volume is None:
                log_event("ERROR", "VolumeNorm", "Failed to acquire endpoint volume interface")
                return False

            self._com_initialized = True
            # Report initial volume reading
            vol = self._endpoint_volume.GetMasterVolumeLevelScalar()
            log_event("INFO", "VolumeNorm", "COM endpoint acquired",
                      windows_volume=f"{vol:.0%}")
            return True
        except Exception as exc:
            log_event("ERROR", "VolumeNorm", "Failed to acquire endpoint volume",
                      error=str(exc))
            return False

    def _poll_loop(self):
        """Background loop: read Windows volume, compute gain, sleep."""
        if not self._init_com():
            # pycaw missing or COM failure — fall back to passthrough
            self._gain = 1.0
            return

        while not self._stop_event.is_set():
            try:
                endpoint = self._endpoint_volume
                if endpoint is None:
                    break
                vol = endpoint.GetMasterVolumeLevelScalar()
                vol = float(vol)

                # Clamp to floor
                safe_vol = max(vol, self._min_volume_floor)
                gain = 1.0 / safe_vol

                self._raw_volume = vol
                self._gain = gain

            except Exception as exc:
                # COM can occasionally hiccup (device switch, etc.)
                log_event("WARN", "VolumeNorm", "Read failed, using last gain",
                          error=str(exc))

            self._stop_event.wait(self._poll_interval)

        # Cleanup COM on thread exit
        try:
            import comtypes
            comtypes.CoUninitialize()
        except Exception:
            pass

    def __del__(self):
        self._stop_poller()
