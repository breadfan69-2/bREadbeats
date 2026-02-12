from network_engine import TCodeCommand
from dataclasses import dataclass


@dataclass(frozen=True)
class VolumeRampState:
    active: bool
    start_time: float
    from_volume: float
    to_volume: float


@dataclass(frozen=True)
class StartStopUiState:
    start_text: str
    play_enabled: bool
    play_reset_checked: bool
    play_text: str | None
    is_sending: bool | None


def send_zero_volume_immediate(network_engine, duration_ms: int) -> None:
    """Send immediate zero-volume center command when engine exists."""
    if not network_engine:
        return
    zero_cmd = TCodeCommand(alpha=0.5, beta=0.5, volume=0.0, duration_ms=duration_ms)
    network_engine.send_immediate(zero_cmd)


def set_transport_sending(network_engine, enabled: bool) -> None:
    """Set network sending flag when engine exists."""
    if not network_engine:
        return
    network_engine.set_sending_enabled(enabled)


def trigger_network_test(network_engine) -> tuple[bool, bool]:
    """Run network test pattern when connected.
    Returns (did_trigger, should_restore_sending_state)."""
    if not network_engine or not network_engine.connected:
        return False, False

    was_sending = network_engine.sending_enabled
    network_engine.set_sending_enabled(True)
    network_engine.send_test()
    return True, not was_sending


def begin_volume_ramp(now: float) -> VolumeRampState:
    """Return canonical start state for play volume ramp."""
    return VolumeRampState(
        active=True,
        start_time=now,
        from_volume=0.0,
        to_volume=1.0,
    )


def start_stop_ui_state(is_running: bool) -> StartStopUiState:
    """Return UI state for Start/Play controls based on running state."""
    if is_running:
        return StartStopUiState(
            start_text="■ Stop",
            play_enabled=True,
            play_reset_checked=False,
            play_text=None,
            is_sending=None,
        )

    return StartStopUiState(
        start_text="▶ Start",
        play_enabled=False,
        play_reset_checked=True,
        play_text="▶ Play",
        is_sending=False,
    )


def play_button_text(is_playing: bool) -> str:
    """Return Play button text for playing/paused state."""
    return "⏸ Pause" if is_playing else "▶ Play"


def shutdown_runtime(stop_engines_callback, network_engine) -> None:
    """Stop local engines first, then stop network engine if present."""
    stop_engines_callback()
    if network_engine:
        network_engine.stop()
