from network_engine import TCodeCommand


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


def begin_volume_ramp(now: float) -> dict:
    """Return canonical start state for play volume ramp."""
    return {
        'active': True,
        'start_time': now,
        'from': 0.0,
        'to': 1.0,
    }


def start_stop_ui_state(is_running: bool) -> dict:
    """Return UI state for Start/Play controls based on running state."""
    if is_running:
        return {
            'start_text': "■ Stop",
            'play_enabled': True,
            'play_reset_checked': False,
            'play_text': None,
            'is_sending': None,
        }

    return {
        'start_text': "▶ Start",
        'play_enabled': False,
        'play_reset_checked': True,
        'play_text': "▶ Play",
        'is_sending': False,
    }


def play_button_text(is_playing: bool) -> str:
    """Return Play button text for playing/paused state."""
    return "⏸ Pause" if is_playing else "▶ Play"


def shutdown_runtime(stop_engines_callback, network_engine) -> None:
    """Stop local engines first, then stop network engine if present."""
    stop_engines_callback()
    if network_engine:
        network_engine.stop()
