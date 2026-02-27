# tcode_wiring.py -- called from the audio thread.
# Do NOT import PyQt6.QtWidgets or call any Qt GUI API here.
"""
bREadbeats - TCode frequency computation (P0/F0/P1/P3).
Extracted from main.py for modularization.
"""

import time
import random
import numpy as np
from typing import Optional
from frequency_utils import extract_dominant_freq
from network_engine import TCodeCommand
from audio_engine import BeatEvent


def compute_and_attach_tcode(win, cmd: TCodeCommand, event: BeatEvent, spectrum: Optional[np.ndarray] = None):
    """Compute P0/F0 TCode values and attach to command. Thread-safe (no widget access)."""
    now = time.time()

    def _effective_output_limits(raw_min: float, raw_max: float, default_min: float, default_max: float) -> tuple[float, float]:
        lo = float(raw_min)
        hi = float(raw_max)
        if hi <= lo or lo <= 0.0 or hi <= 0.0:
            lo = float(default_min)
            hi = float(default_max)
        return lo, hi
    
    # Extract dominant frequencies independently for P0 and F0 monitor ranges
    dom_freq = event.frequency if hasattr(event, 'frequency') else 0.0
    p0_dom_freq = dom_freq  # fallback
    f0_dom_freq = dom_freq  # fallback
    if spectrum is not None:
        sr = win.config.audio.sample_rate
        p0_dom_freq = extract_dominant_freq(spectrum, sr,
            win.config.pulse_freq.monitor_freq_min,
            win.config.pulse_freq.monitor_freq_max)
        f0_dom_freq = extract_dominant_freq(spectrum, sr,
            win.config.carrier_freq.monitor_freq_min,
            win.config.carrier_freq.monitor_freq_max)
    
    # Calculate dot speed for Speed mode
    dt = max(0.001, now - win._last_dot_time)
    delta_alpha = cmd.alpha - win._last_dot_alpha
    delta_beta = cmd.beta - win._last_dot_beta
    dot_speed = np.sqrt(delta_alpha**2 + delta_beta**2) / dt
    win._last_dot_alpha = cmd.alpha
    win._last_dot_beta = cmd.beta
    win._last_dot_time = now
    
    # --- P0 (Pulse Frequency) with short sliding window averaging ---
    p0_enabled = win._cached_p0_enabled
    if p0_enabled:
        pulse_mode = win._cached_pulse_mode
        pulse_invert = win._cached_pulse_invert
        freq_weight = win.config.pulse_freq.freq_weight
        
        if pulse_mode == 0:  # Hz mode
            in_low = win.config.pulse_freq.monitor_freq_min
            in_high = win.config.pulse_freq.monitor_freq_max
            norm = (p0_dom_freq - in_low) / max(1.0, in_high - in_low)
        elif pulse_mode == 2:  # Band (sub_bass) mode
            # Use sub_bass band energy directly — long booming bass = "feeling" the pulse
            sub_bass_energy = 0.0
            if win.audio_engine:
                if hasattr(win.audio_engine, 'get_band_energies'):
                    sub_bass_energy = win.audio_engine.get_band_energies().get('sub_bass', 0.0)
                elif hasattr(win.audio_engine, '_band_energies'):
                    sub_bass_energy = win.audio_engine._band_energies.get('sub_bass', 0.0)
            # Normalize: typical sub_bass energy 0-0.3 after gain
            norm = min(1.0, sub_bass_energy * 4.0)
        else:  # Speed mode
            norm = min(1.0, dot_speed / 10.0)
        
        norm = max(0.0, min(1.0, norm))
        norm_weighted = 0.5 + (norm - 0.5) * freq_weight
        norm_weighted = max(0.0, min(1.0, norm_weighted))
        
        if pulse_invert:
            norm_weighted = 1.0 - norm_weighted
        
        # Add sample to sliding window
        win._p0_freq_window.append((now, norm_weighted))
        
        # Remove samples older than window size
        window_cutoff = now - (win._freq_window_ms / 1000.0)
        while win._p0_freq_window and win._p0_freq_window[0][0] < window_cutoff:
            win._p0_freq_window.popleft()
        
        # Calculate average over window
        if win._p0_freq_window:
            avg_norm = sum(s[1] for s in win._p0_freq_window) / len(win._p0_freq_window)
        else:
            avg_norm = norm_weighted
        
        # Map averaged frequency to TCode output range (direct TCode, 0-9999)
        tcode_min_val = win._cached_tcode_freq_min
        tcode_max_val = win._cached_tcode_freq_max
        tcode_min_val = max(0, min(9999, tcode_min_val))
        tcode_max_val = max(0, min(9999, tcode_max_val))
        p0_val = int(tcode_min_val + avg_norm * (tcode_max_val - tcode_min_val))
        p0_val = max(0, min(9999, p0_val))
        
        # Send P0 using current low-latency window duration
        cmd.pulse_freq = p0_val
        cmd.pulse_freq_duration = int(win._freq_window_ms)
        # Display converted real output (with safe fallback defaults when limits are unset).
        dl = win.config.device_limits
        p0_lo, p0_hi = _effective_output_limits(dl.p0_freq_min, dl.p0_freq_max, 1.0, 100.0)
        hz = p0_lo + (p0_val / 9999.0) * (p0_hi - p0_lo)
        win._cached_pulse_display = f"Pulse Freq: {hz:.0f}Hz"
    else:
        cmd.pulse_freq = None
        win._cached_pulse_display = "Pulse Freq: off"
        win._p0_freq_window.clear()  # Clear window when disabled
    
    # --- F0 (Carrier Frequency) with short sliding window averaging ---
    f0_enabled = win._cached_f0_enabled
    if f0_enabled:
        f0_mode = win._cached_f0_mode
        f0_invert = win._cached_f0_invert
        f0_weight = win.config.carrier_freq.freq_weight
        
        if f0_mode == 0:  # Hz mode
            f0_in_low = win.config.carrier_freq.monitor_freq_min
            f0_in_high = win.config.carrier_freq.monitor_freq_max
            f0_norm = (f0_dom_freq - f0_in_low) / max(1.0, f0_in_high - f0_in_low)
        elif f0_mode == 2:  # Band (mid) mode — voice, brass, dominant strings (500-2000 Hz)
            # Use mid band energy directly — strict rate limit below
            mid_energy = 0.0
            if win.audio_engine:
                if hasattr(win.audio_engine, 'get_band_energies'):
                    mid_energy = win.audio_engine.get_band_energies().get('mid', 0.0)
                elif hasattr(win.audio_engine, '_band_energies'):
                    mid_energy = win.audio_engine._band_energies.get('mid', 0.0)
            # Normalize: typical mid energy 0-0.2 after gain
            f0_norm = min(1.0, mid_energy * 5.0)
        else:  # Speed mode
            f0_norm = min(1.0, dot_speed / 10.0)
        
        f0_norm = max(0.0, min(1.0, f0_norm))
        f0_norm_weighted = 0.5 + (f0_norm - 0.5) * f0_weight
        f0_norm_weighted = max(0.0, min(1.0, f0_norm_weighted))
        
        if f0_invert:
            f0_norm_weighted = 1.0 - f0_norm_weighted
        
        # Add sample to sliding window
        win._f0_freq_window.append((now, f0_norm_weighted))
        
        # Remove samples older than window size
        f0_window_cutoff = now - (win._freq_window_ms / 1000.0)
        while win._f0_freq_window and win._f0_freq_window[0][0] < f0_window_cutoff:
            win._f0_freq_window.popleft()
        
        # Calculate average over window
        if win._f0_freq_window:
            f0_avg_norm = sum(s[1] for s in win._f0_freq_window) / len(win._f0_freq_window)
        else:
            f0_avg_norm = f0_norm_weighted
        
        # Map averaged frequency to TCode output range (direct TCode, 0-9999)
        f0_tcode_min = win._cached_f0_tcode_min
        f0_tcode_max = win._cached_f0_tcode_max
        f0_tcode_min = max(0, min(9999, f0_tcode_min))
        f0_tcode_max = max(0, min(9999, f0_tcode_max))
        f0_val_raw = int(f0_tcode_min + f0_avg_norm * (f0_tcode_max - f0_tcode_min))
        f0_val_raw = max(0, min(9999, f0_val_raw))
        
        # Smooth F0: limit change rate for smoother transitions
        if f0_mode == 2:
            # Band (mid) mode: strict rate limiter — ±500 tcode per 2 seconds
            # Must finish traveling to current target before accepting new one
            if win._c0_band_current is None:
                win._c0_band_current = f0_val_raw
                win._c0_band_target = f0_val_raw

            # Check if we've arrived at current target
            at_target = (win._c0_band_target is not None
                         and abs(win._c0_band_current - win._c0_band_target) < 5)

            if at_target:
                # Accept new target only if different enough (>50 tcode)
                current_target = win._c0_band_target
                if current_target is not None and abs(f0_val_raw - current_target) > 50:
                    # Clamp new target to bounded jump from current position
                    delta_from_current = f0_val_raw - win._c0_band_current
                    delta_from_current = max(-win._c0_band_max_target_delta, min(win._c0_band_max_target_delta, delta_from_current))
                    win._c0_band_target = win._c0_band_current + delta_from_current
                    win._c0_band_target = max(0, min(9999, win._c0_band_target))

            # Travel toward target at _c0_band_travel_rate tcode/sec (=250/s → 500 per 2s)
            if win._c0_band_target is not None and win._c0_band_current != win._c0_band_target:
                max_step = max(1, int(win._c0_band_travel_rate * dt))
                diff = win._c0_band_target - win._c0_band_current
                step = max(-max_step, min(max_step, diff))
                win._c0_band_current += step
                win._c0_band_current = max(0, min(9999, win._c0_band_current))

            f0_val = int(win._c0_band_current)
        elif win._f0_last_sent_tcode is not None:
            delta = f0_val_raw - win._f0_last_sent_tcode
            if abs(delta) > win._f0_max_change_per_send:
                if delta > 0:
                    f0_val = win._f0_last_sent_tcode + win._f0_max_change_per_send
                else:
                    f0_val = win._f0_last_sent_tcode - win._f0_max_change_per_send
            else:
                f0_val = f0_val_raw
        else:
            f0_val = f0_val_raw
        f0_val = max(0, min(9999, f0_val))
        win._f0_last_sent_tcode = f0_val
        
        # Generate short random duration for live response
        f0_duration = int(win._f0_duration_base_ms + random.uniform(-win._f0_duration_variance_ms, win._f0_duration_variance_ms))
        f0_duration = max(100, f0_duration)  # Minimum 100ms
        
        if cmd.tcode_tags is None:
            cmd.tcode_tags = {}
        cmd.tcode_tags['C0'] = f0_val  # restim uses C0 for carrier frequency, not F0
        cmd.tcode_tags['C0_duration'] = f0_duration
        # Display converted real output (with safe fallback defaults when limits are unset).
        dl = win.config.device_limits
        c0_lo, c0_hi = _effective_output_limits(dl.c0_freq_min, dl.c0_freq_max, 500.0, 1500.0)
        hz = c0_lo + (f0_val / 9999.0) * (c0_hi - c0_lo)
        win._cached_carrier_display = f"Carrier Freq: {hz:.0f}Hz"
    else:
        win._cached_carrier_display = "Carrier Freq: off"
        win._f0_freq_window.clear()  # Clear window when disabled
        win._f0_last_sent_tcode = None  # Reset smoothing state when disabled
    
    # --- P1 (Pulse Width) with short sliding window averaging ---
    p1_enabled = win._cached_p1_enabled
    if p1_enabled:
        p1_mode = win._cached_p1_mode
        p1_invert = win._cached_p1_invert
        p1_weight = win.config.pulse_width.weight
        
        if p1_mode == 0:  # Volume (RMS energy) mode
            # Use spectrum RMS as volume proxy (0-1 normalized)
            if spectrum is not None and len(spectrum) > 0:
                spec_rms = float(np.sqrt(np.mean(spectrum ** 2)))
                # Normalize: typical spec_rms range ~0.0001-0.05, map with log scale
                p1_norm = max(0.0, min(1.0, (np.log10(max(spec_rms, 1e-8)) + 4) / 3.0))
            else:
                p1_norm = 0.5
        elif p1_mode == 1:  # Hz (dominant freq) mode
            p1_dom_freq = extract_dominant_freq(spectrum, win.config.audio.sample_rate,
                win.config.pulse_width.monitor_freq_min, win.config.pulse_width.monitor_freq_max) if spectrum is not None else 0.0
            p1_in_low = win.config.pulse_width.monitor_freq_min
            p1_in_high = win.config.pulse_width.monitor_freq_max
            p1_norm = (p1_dom_freq - p1_in_low) / max(1.0, p1_in_high - p1_in_low)
        else:  # Speed (dot movement) mode
            p1_norm = min(1.0, dot_speed / 10.0)
        
        p1_norm = max(0.0, min(1.0, p1_norm))
        p1_norm_weighted = 0.5 + (p1_norm - 0.5) * p1_weight
        p1_norm_weighted = max(0.0, min(1.0, p1_norm_weighted))
        
        if p1_invert:
            p1_norm_weighted = 1.0 - p1_norm_weighted
        
        # Sliding window average
        win._p1_window.append((now, p1_norm_weighted))
        p1_window_cutoff = now - (win._freq_window_ms / 1000.0)
        while win._p1_window and win._p1_window[0][0] < p1_window_cutoff:
            win._p1_window.popleft()
        p1_avg = sum(s[1] for s in win._p1_window) / len(win._p1_window) if win._p1_window else p1_norm_weighted
        
        # Map to TCode range
        p1_tcode_min = win._cached_p1_tcode_min
        p1_tcode_max = win._cached_p1_tcode_max
        p1_val = int(p1_tcode_min + p1_avg * (p1_tcode_max - p1_tcode_min))
        p1_val = max(0, min(9999, p1_val))
        
        if cmd.tcode_tags is None:
            cmd.tcode_tags = {}
        cmd.tcode_tags['P1'] = p1_val
        cmd.tcode_tags['P1_duration'] = int(win._freq_window_ms)
        # Display converted real output (with safe fallback defaults when limits are unset).
        dl = win.config.device_limits
        p1_lo, p1_hi = _effective_output_limits(dl.p1_cycles_min, dl.p1_cycles_max, 0.0, 20.0)
        p1_cyc = p1_lo + (p1_val / 9999.0) * (p1_hi - p1_lo)
        win._cached_p1_display = f"Pulse Width: {p1_cyc:.1f}cyc"
    else:
        win._cached_p1_display = "Pulse Width: off"
        win._p1_window.clear()
    
    # --- P3 (Rise Time) with short sliding window averaging ---
    p3_enabled = win._cached_p3_enabled
    if p3_enabled:
        p3_mode = win._cached_p3_mode
        p3_invert = win._cached_p3_invert
        p3_weight = win.config.rise_time.weight
        
        if p3_mode == 0:  # Brightness (spectral centroid) mode
            if spectrum is not None and len(spectrum) > 0:
                sr = win.config.audio.sample_rate
                freqs = np.linspace(0, sr / 2, len(spectrum))
                total_energy = float(np.sum(spectrum))
                if total_energy > 1e-10:
                    centroid = float(np.sum(freqs * spectrum) / total_energy)
                else:
                    centroid = sr / 4  # midpoint fallback
                # Normalize centroid: typical range 200-8000 Hz
                p3_norm = max(0.0, min(1.0, (centroid - 200) / 7800))
                # INVERT inherently: bright audio → LOW rise time (exciting)
                # So high centroid → low p3_norm (before user invert)
                p3_norm = 1.0 - p3_norm
            else:
                p3_norm = 0.5
        elif p3_mode == 1:  # Hz (dominant freq) mode
            p3_dom_freq = extract_dominant_freq(spectrum, win.config.audio.sample_rate,
                win.config.rise_time.monitor_freq_min, win.config.rise_time.monitor_freq_max) if spectrum is not None else 0.0
            p3_in_low = win.config.rise_time.monitor_freq_min
            p3_in_high = win.config.rise_time.monitor_freq_max
            p3_norm = (p3_dom_freq - p3_in_low) / max(1.0, p3_in_high - p3_in_low)
        else:  # Speed (dot movement) mode
            p3_norm = min(1.0, dot_speed / 10.0)
        
        p3_norm = max(0.0, min(1.0, p3_norm))
        p3_norm_weighted = 0.5 + (p3_norm - 0.5) * p3_weight
        p3_norm_weighted = max(0.0, min(1.0, p3_norm_weighted))
        
        if p3_invert:
            p3_norm_weighted = 1.0 - p3_norm_weighted
        
        # Sliding window average
        win._p3_window.append((now, p3_norm_weighted))
        p3_window_cutoff = now - (win._freq_window_ms / 1000.0)
        while win._p3_window and win._p3_window[0][0] < p3_window_cutoff:
            win._p3_window.popleft()
        p3_avg = sum(s[1] for s in win._p3_window) / len(win._p3_window) if win._p3_window else p3_norm_weighted
        
        # Map to TCode range
        p3_tcode_min = win._cached_p3_tcode_min
        p3_tcode_max = win._cached_p3_tcode_max
        p3_val = int(p3_tcode_min + p3_avg * (p3_tcode_max - p3_tcode_min))
        p3_val = max(0, min(9999, p3_val))
        
        if cmd.tcode_tags is None:
            cmd.tcode_tags = {}
        cmd.tcode_tags['P3'] = p3_val
        cmd.tcode_tags['P3_duration'] = int(win._freq_window_ms)
        # Display converted real output (with safe fallback defaults when limits are unset).
        dl = win.config.device_limits
        p3_lo, p3_hi = _effective_output_limits(dl.p3_cycles_min, dl.p3_cycles_max, 0.0, 20.0)
        p3_cyc = p3_lo + (p3_val / 9999.0) * (p3_hi - p3_lo)
        win._cached_p3_display = f"Rise Time: {p3_cyc:.1f}cyc"
    else:
        win._cached_p3_display = "Rise Time: off"
        win._p3_window.clear()
    
    # Log
    p0_str = f"P0={cmd.pulse_freq:04d}" if cmd.pulse_freq is not None else "P0=off"
    c0_tag = cmd.tcode_tags.get('C0', None) if cmd.tcode_tags else None
    c0_str = f"C0={c0_tag:04d}" if c0_tag is not None else "C0=off"
    p1_tag = cmd.tcode_tags.get('P1', None) if cmd.tcode_tags else None
    p1_str = f"P1={p1_tag:04d}" if p1_tag is not None else "P1=off"
    p3_tag = cmd.tcode_tags.get('P3', None) if cmd.tcode_tags else None
    p3_str = f"P3={p3_tag:04d}" if p3_tag is not None else "P3=off"
    gate_str = ""
    mapper = getattr(win, 'stroke_mapper', None)
    gf = getattr(mapper, '_last_gate_fail', None) if mapper is not None else None
    if gf:
        gate_str = f" GATE_FAIL={gf}"
    print(f"[Main] Cmd: a={cmd.alpha:.2f} b={cmd.beta:.2f} v={cmd.volume:.2f} {p0_str} {c0_str} {p1_str} {p3_str}{gate_str}")

