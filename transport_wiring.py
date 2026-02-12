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
