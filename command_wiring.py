import time

from network_engine import TCodeCommand


def attach_cached_tcode_values(
    cmd: TCodeCommand,
    *,
    p0c0_enabled: bool,
    cached_p0_enabled: bool,
    cached_p0_val,
    cached_f0_enabled: bool,
    cached_f0_val,
    cached_p1_enabled: bool,
    cached_p1_val,
    cached_p3_enabled: bool,
    cached_p3_val,
    freq_window_ms: int,
) -> None:
    """Attach cached P0/C0/P1/P3 values to a command using existing send rules."""
    if p0c0_enabled and cached_p0_enabled and cached_p0_val is not None:
        cmd.pulse_freq = cached_p0_val

    if p0c0_enabled and cached_f0_enabled and cached_f0_val is not None:
        if cmd.tcode_tags is None:
            cmd.tcode_tags = {}
        cmd.tcode_tags['C0'] = cached_f0_val

    if cached_p1_enabled and cached_p1_val is not None:
        if cmd.tcode_tags is None:
            cmd.tcode_tags = {}
        cmd.tcode_tags['P1'] = cached_p1_val
        cmd.tcode_tags['P1_duration'] = int(freq_window_ms)

    if cached_p3_enabled and cached_p3_val is not None:
        if cmd.tcode_tags is None:
            cmd.tcode_tags = {}
        cmd.tcode_tags['P3'] = cached_p3_val
        cmd.tcode_tags['P3_duration'] = int(freq_window_ms)


def apply_volume_ramp(
    cmd: TCodeCommand,
    *,
    volume_ramp_active: bool,
    volume_ramp_start_time: float,
    volume_ramp_duration: float,
    volume_ramp_from: float,
    volume_ramp_to: float,
    now: float | None = None,
) -> None:
    """Apply in-place volume ramp multiplier using existing linear ramp behavior."""
    if not volume_ramp_active:
        return

    current_time = time.time() if now is None else now
    elapsed = current_time - volume_ramp_start_time
    progress = min(1.0, elapsed / volume_ramp_duration)
    ramp_mult = volume_ramp_from + (volume_ramp_to - volume_ramp_from) * progress
    cmd.volume = cmd.volume * ramp_mult
