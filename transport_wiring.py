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


def begin_volume_ramp(now: float) -> dict:
    """Return canonical start state for play volume ramp."""
    return {
        'active': True,
        'start_time': now,
        'from': 0.0,
        'to': 1.0,
    }
