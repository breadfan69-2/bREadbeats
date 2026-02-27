"""
event_handlers.py – extracted event-handler functions for BREadbeatsWindow.

Every function here used to be a method on BREadbeatsWindow.  The first
parameter is *win* (the window instance) instead of *self*.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
from typing import Optional
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer
from config import BeatDetectionType, StrokeMode, BEAT_RANGE_LIMITS
from logging_utils import get_log_level, log_event, set_log_level
from audio_engine import AudioEngine, BeatEvent
from network_engine import TCodeCommand
from network_lifecycle import ensure_network_engine
from command_wiring import apply_volume_ramp
from transport_wiring import (
    begin_volume_ramp, play_button_text, send_zero_volume_immediate,
    set_transport_sending, shutdown_runtime, start_stop_ui_state,
)
from stroke_mapper import StrokeMapper
from config_facade import load_config, save_config
from frequency_utils import extract_dominant_freq

def apply_config_to_ui(win):
    """Apply loaded config values to UI sliders"""
    try:
        win.config.stroke.mode = StrokeMode.SIMPLE_CIRCLE
        win._enforce_fixed_effect_axis_values()
        beats_to_index = {4: 0, 3: 1, 6: 2}
        with win._signals_blocked(
            getattr(win, 'detection_type_combo', None),
            getattr(win, 'sensitivity_slider', None),
            getattr(win, 'peak_floor_slider', None),
            getattr(win, 'peak_decay_slider', None),
            getattr(win, 'rise_sens_slider', None),
            getattr(win, 'flux_mult_slider', None),
            getattr(win, 'audio_gain_slider', None),
            getattr(win, 'silence_reset_slider', None),
            getattr(win, 'freq_range_slider', None),
            getattr(win, 'metrics_global_cb', None),
            getattr(win, 'tempo_tracking_checkbox', None),
            getattr(win, 'time_sig_combo', None),
            getattr(win, 'stability_threshold_slider', None),
            getattr(win, 'tempo_timeout_slider', None),
            getattr(win, 'phase_snap_slider', None),
            getattr(win, 'intensity_ramp_spin', None),
            getattr(win, 'intensity_ramp_target_combo', None),
            getattr(win, 'fill_gate_scale_spin', None),
            getattr(win, 'main_silence_close_slider', None),
            getattr(win, 'jitter_effect_action', None),
            getattr(win, 'metronome_lock_required_action', None),
            getattr(win, 'host_edit', None),
            getattr(win, 'port_spin', None),
            getattr(win, 'pulse_freq_range_slider', None),
            getattr(win, 'tcode_freq_range_slider', None),
            getattr(win, 'freq_weight_slider', None),
            getattr(win, 'f0_freq_range_slider', None),
            getattr(win, 'f0_tcode_range_slider', None),
            getattr(win, 'f0_weight_slider', None),
            getattr(win, 'volume_slider', None),
        ):
            # Beat detection tab
            if all(hasattr(win, name) for name in (
                'detection_type_combo', 'sensitivity_slider', 'peak_floor_slider',
                'peak_decay_slider', 'rise_sens_slider', 'flux_mult_slider',
                'audio_gain_slider', 'silence_reset_slider', 'freq_range_slider'
            )):
                win.detection_type_combo.setCurrentIndex(win.config.beat.detection_type - 1)
                win.sensitivity_slider.setValue(win.config.beat.sensitivity)
                win.peak_floor_slider.setValue(win.config.beat.peak_floor)
                win.peak_decay_slider.setValue(win.config.beat.peak_decay)
                win.rise_sens_slider.setValue(win.config.beat.rise_sensitivity)
                win.flux_mult_slider.setValue(win.config.beat.flux_multiplier)
                win.audio_gain_slider.setValue(win.config.audio.gain)
                win.silence_reset_slider.setValue(win.config.beat.silence_reset_ms)
                win.freq_range_slider.setLow(win.config.beat.freq_low)
                win.freq_range_slider.setHigh(win.config.beat.freq_high)

            # Auto-adjust global toggle
            if hasattr(win, 'metrics_global_cb'):
                win.metrics_global_cb.setChecked(win.config.auto_adjust.metrics_global_enabled)

            # Tempo tracking settings
            if hasattr(win, 'tempo_tracking_checkbox'):
                win.tempo_tracking_checkbox.setChecked(win.config.beat.tempo_tracking_enabled)
            if hasattr(win, 'time_sig_combo'):
                win.time_sig_combo.setCurrentIndex(beats_to_index.get(win.config.beat.beats_per_measure, 0))
            if hasattr(win, 'stability_threshold_slider'):
                win.stability_threshold_slider.setValue(win.config.beat.stability_threshold)
            if hasattr(win, 'tempo_timeout_slider'):
                win.tempo_timeout_slider.setValue(win.config.beat.tempo_timeout_ms)
            if hasattr(win, 'phase_snap_slider'):
                win.phase_snap_slider.setValue(win.config.beat.phase_snap_weight)
            win.config.stroke.min_interval_ms = 150
            metronome_lock_required_action = getattr(win, 'metronome_lock_required_action', None)
            if metronome_lock_required_action is not None:
                metronome_lock_required_action.setChecked(bool(getattr(win.config.beat, 'tempo_lock_required', False)))
            if hasattr(win, 'intensity_ramp_spin'):
                win.intensity_ramp_spin.setValue(
                    float(getattr(win.config.stroke, 'intensity_ramp_hours', 0.0) or 0.0)
                )
            if hasattr(win, 'intensity_ramp_target_combo'):
                target_control: Any = win.intensity_ramp_target_combo
                target = str(getattr(win.config.stroke, 'intensity_ramp_target', 'both') or 'both').strip().lower()
                if target not in ('size', 'speed', 'both'):
                    target = 'both'
                set_current_text = getattr(target_control, 'setCurrentText', None)
                set_value = getattr(target_control, 'setValue', None)
                if callable(set_current_text):
                    set_current_text(target)
                elif callable(set_value):
                    set_value({'size': 0, 'speed': 1, 'both': 2}.get(target, 2))
            win._refresh_motion_ramp_visual_state()
            if hasattr(win, 'fill_gate_scale_spin'):
                win.fill_gate_scale_spin.setValue(
                    win._fill_gate_scale_to_percent(
                        float(getattr(win.config.stroke, 'overall_amp_fill_required_scale', 1.0) or 1.0)
                    )
                )
            if hasattr(win, 'main_silence_close_slider'):
                win.main_silence_close_slider.setValue(
                    float(getattr(win.config.stroke, 'energy_response_strength', 1.0))
                )
            advanced_flux_slider = getattr(win, '_advanced_flux_threshold_slider', None)
            if advanced_flux_slider is not None:
                try:
                    advanced_flux_slider.setValue(win.config.stroke.flux_threshold)
                except RuntimeError:
                    win._advanced_flux_threshold_slider = None
            advanced_flux_scaling_slider = getattr(win, '_advanced_flux_scaling_slider', None)
            if advanced_flux_scaling_slider is not None:
                try:
                    advanced_flux_scaling_slider.setValue(win.config.stroke.flux_scaling_weight)
                except RuntimeError:
                    win._advanced_flux_scaling_slider = None
            auto_fill_widgets = getattr(win, '_auto_fill_controls_widgets', {}) or {}
            auto_fill_enabled = auto_fill_widgets.get('enabled')
            if auto_fill_enabled is not None:
                auto_fill_enabled.setChecked(bool(getattr(win.config.stroke, 'overall_amp_fill_auto_enabled', True)))
            auto_fill_target = auto_fill_widgets.get('target_pass_rate')
            if auto_fill_target is not None:
                auto_fill_target.setValue(float(getattr(win.config.stroke, 'overall_amp_fill_auto_target_pass_rate', 0.58) or 0.58))
            auto_fill_alpha = auto_fill_widgets.get('ema_alpha')
            if auto_fill_alpha is not None:
                auto_fill_alpha.setValue(float(getattr(win.config.stroke, 'overall_amp_fill_auto_ema_alpha', 0.12) or 0.12))
            auto_fill_deadband = auto_fill_widgets.get('deadband')
            if auto_fill_deadband is not None:
                auto_fill_deadband.setValue(float(getattr(win.config.stroke, 'overall_amp_fill_auto_deadband', 0.06) or 0.06))
            auto_fill_step = auto_fill_widgets.get('step')
            if auto_fill_step is not None:
                auto_fill_step.setValue(float(getattr(win.config.stroke, 'overall_amp_fill_auto_step', 0.02) or 0.02))
            auto_fill_max_offset = auto_fill_widgets.get('max_offset')
            if auto_fill_max_offset is not None:
                auto_fill_max_offset.setValue(float(getattr(win.config.stroke, 'overall_amp_fill_auto_max_offset', 0.35) or 0.35))
            auto_fill_min_req = auto_fill_widgets.get('min_required')
            if auto_fill_min_req is not None:
                auto_fill_min_req.setValue(float(getattr(win.config.stroke, 'overall_amp_fill_auto_min_required', 0.05) or 0.05))
            auto_fill_max_req = auto_fill_widgets.get('max_required')
            if auto_fill_max_req is not None:
                auto_fill_max_req.setValue(float(getattr(win.config.stroke, 'overall_amp_fill_auto_max_required', 0.98) or 0.98))

            # Effects menu toggles
            jitter_action = getattr(win, 'jitter_effect_action', None)
            if jitter_action is not None:
                jitter_action.setChecked(bool(getattr(win.config.jitter, 'enabled', True)))

            # Connection settings
            if hasattr(win, 'host_edit'):
                win.host_edit.setText(win.config.connection.host)
            if hasattr(win, 'port_spin'):
                win.port_spin.setValue(win.config.connection.port)

            # Other tab (pulse freq settings)
            if all(hasattr(win, name) for name in ('pulse_freq_range_slider', 'tcode_freq_range_slider', 'freq_weight_slider')):
                win.pulse_freq_range_slider.setLow(win.config.pulse_freq.monitor_freq_min)
                win.pulse_freq_range_slider.setHigh(win.config.pulse_freq.monitor_freq_max)
                win.tcode_freq_range_slider.setLow(win.config.pulse_freq.tcode_min)
                win.tcode_freq_range_slider.setHigh(win.config.pulse_freq.tcode_max)
                win.freq_weight_slider.setValue(win.config.pulse_freq.freq_weight)

            # Carrier freq (F0) settings
            if all(hasattr(win, name) for name in ('f0_freq_range_slider', 'f0_tcode_range_slider', 'f0_weight_slider')):
                win.f0_freq_range_slider.setLow(win.config.carrier_freq.monitor_freq_min)
                win.f0_freq_range_slider.setHigh(win.config.carrier_freq.monitor_freq_max)
                win.f0_tcode_range_slider.setLow(win.config.carrier_freq.tcode_min)
                win.f0_tcode_range_slider.setHigh(win.config.carrier_freq.tcode_max)
                win.f0_weight_slider.setValue(win.config.carrier_freq.freq_weight)

            # Volume (config stores 0-1, slider shows 0-100)
            if hasattr(win, 'volume_slider'):
                win.volume_slider.setValue(int(win.config.volume * 100))

        # Set active visualizer sample rates and update frequency bands
        if hasattr(win, 'freq_range_slider'):
            win._on_freq_band_change()  # Update beat detection band (red)

        # Apply mode-dependent limits after sliders are set
        win._on_mode_change(0)  # Mode temporarily pinned to circle
        win._on_depth_band_change()  # Update stroke depth band (green)
        if hasattr(win, 'pulse_freq_range_slider'):
            win._on_p0_band_change()  # Update P0 TCode band (blue)
        if hasattr(win, 'f0_freq_range_slider'):
            win._on_f0_band_change()  # Update F0 TCode band (cyan)

        # Apply tempo tracking side effects after values are in place
        if hasattr(win, 'tempo_tracking_checkbox'):
            win._on_tempo_tracking_toggle(2 if win.config.beat.tempo_tracking_enabled else 0)

        # Log level menu (persisted)
        win._sync_log_level_menu(getattr(win.config, 'log_level', get_log_level()))

        print("[UI] Loaded all settings from config")
    except AttributeError as e:
        print(f"[UI] Warning: Could not apply all config values: {e}")

def populate_audio_devices(win):
    """Populate audio device dropdown - WASAPI devices only (deduplicated)"""
    import sounddevice as sd
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    # Find WASAPI host API index and default output device
    wasapi_idx = None
    default_output_idx = None
    for idx, api in enumerate(hostapis):
        if 'WASAPI' in api['name']:
            wasapi_idx = idx
            default_output_idx = api.get('default_output_device', None)
            break

    win.device_combo.clear()
    win.audio_device_map = {}  # Map combo index to device index
    win.audio_device_is_loopback = {}  # Track which devices should use WASAPI loopback

    loopback_keywords = ['stereo mix', 'what u hear', 'loopback', 'wave out mix', 'system audio']
    loopback_idx = None
    default_output_combo_idx = None  # Track where default output appears
    combo_idx = 0
    seen_names = set()  # For deduplication

    if wasapi_idx is not None:
        # Add WASAPI input devices (microphones) - deduplicated by name
        for i, dev in enumerate(devices):
            if dev['hostapi'] == wasapi_idx and dev['max_input_channels'] > 0:
                # Normalize name for dedup
                clean_name = dev['name'].strip()
                if clean_name in seen_names:
                    continue
                seen_names.add(clean_name)

                name = f"{clean_name} (Input)"
                win.device_combo.addItem(name)
                win.audio_device_map[combo_idx] = i
                win.audio_device_is_loopback[combo_idx] = False

                # Find loopback device for default selection
                if loopback_idx is None and any(keyword in dev['name'].lower() for keyword in loopback_keywords):
                    loopback_idx = combo_idx

                combo_idx += 1

        # Add WASAPI output devices as loopback sources - deduplicated by name
        seen_output_names = set()
        for i, dev in enumerate(devices):
            if dev['hostapi'] == wasapi_idx and dev['max_output_channels'] > 0:
                clean_name = dev['name'].strip()
                if clean_name in seen_output_names:
                    continue
                seen_output_names.add(clean_name)

                # Mark if this is the system default output device
                is_default = (i == default_output_idx)
                prefix = "★ " if is_default else ""
                name = f"{prefix}{clean_name} [WASAPI Loopback]"
                win.device_combo.addItem(name)
                win.audio_device_map[combo_idx] = i
                win.audio_device_is_loopback[combo_idx] = True

                # Track default output device's combo index
                if is_default:
                    default_output_combo_idx = combo_idx

                # Fallback: first WASAPI loopback if no default found
                if loopback_idx is None:
                    loopback_idx = combo_idx

                combo_idx += 1
    else:
        # Fallback: no WASAPI found, show all input devices deduplicated
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                clean_name = dev['name'].strip()
                if clean_name in seen_names:
                    continue
                seen_names.add(clean_name)

                name = f"{clean_name}"
                win.device_combo.addItem(name)
                win.audio_device_map[combo_idx] = i
                win.audio_device_is_loopback[combo_idx] = False
                combo_idx += 1

    # Pre-select: prefer system default output loopback > stereo mix/loopback > first device
    if default_output_combo_idx is not None:
        win.device_combo.setCurrentIndex(default_output_combo_idx)
        print(f"[Main] Auto-selected system default output device for loopback")
    elif loopback_idx is not None:
        win.device_combo.setCurrentIndex(loopback_idx)
    elif combo_idx > 0:
        win.device_combo.setCurrentIndex(0)

def apply_release_learning_defaults(win) -> None:
    import sys
    defaults_dir: Path | None = None
    meipass = getattr(sys, '_MEIPASS', None)
    frozen = bool(getattr(sys, 'frozen', False))
    exe_root = Path(sys.executable).parent if frozen else None

    def _is_allowed_frozen_candidate(path: Path) -> bool:
        if not frozen:
            return True
        try:
            resolved = path.resolve()
        except Exception:
            return False
        if exe_root is None:
            return False
        try:
            resolved_parent = resolved.parent.resolve()
            exe_resolved = exe_root.resolve()
            defaults_resolved = (exe_root / "defaults" / "learning").resolve()
            if resolved_parent == exe_resolved:
                return True
            if resolved_parent == defaults_resolved:
                return True
        except Exception:
            return False
        return False

    if getattr(sys, 'frozen', False) and meipass:
        try:
            bundle_root = Path(str(meipass))
            exe_root = Path(sys.executable).parent

            bundle_defaults = bundle_root / "defaults" / "learning"
            if bundle_defaults.exists():
                target_defaults = exe_root / "defaults" / "learning"
                target_defaults.mkdir(parents=True, exist_ok=True)
                for source in bundle_defaults.glob("*.json"):
                    target = target_defaults / source.name
                    if not target.exists():
                        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

            bundle_rule_fit = bundle_root / "datasets" / "rule_fit.json"
            if bundle_rule_fit.exists():
                target_datasets = exe_root / "datasets"
                target_datasets.mkdir(parents=True, exist_ok=True)
                target_rule_fit = target_datasets / "rule_fit.json"
                if not target_rule_fit.exists():
                    target_rule_fit.write_text(bundle_rule_fit.read_text(encoding="utf-8"), encoding="utf-8")

            bundle_slots = bundle_root / "learned_profile_slots.json"
            if bundle_slots.exists():
                target_slots = exe_root / "learned_profile_slots.json"
                if not target_slots.exists():
                    target_slots.write_text(bundle_slots.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception as exc:
            print(f"[Learning] Failed to materialize bundled learning files: {exc}")

    # Discover profile/rule_fit candidates from two roots only:
    # base dir (EXE dir for frozen, repo dir for source) and base/defaults/learning.
    search_roots: list[Path] = []
    base_dir = Path(sys.executable).parent if frozen else Path(__file__).resolve().parent
    search_roots.append(base_dir)
    search_roots.append(base_dir / "defaults" / "learning")

    profile_candidates: list[Path] = []
    rule_fit_candidates: list[Path] = []
    seen_profiles: set[Path] = set()
    seen_rule_fits: set[Path] = set()

    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for candidate in sorted(root.glob("profile*.json")):
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved in seen_profiles:
                continue
            seen_profiles.add(resolved)
            profile_candidates.append(candidate)

        for candidate in sorted(root.glob("rule_fit*.json")):
            try:
                resolved = candidate.resolve()
            except Exception:
                resolved = candidate
            if resolved in seen_rule_fits:
                continue
            seen_rule_fits.add(resolved)
            rule_fit_candidates.append(candidate)

    if not profile_candidates and not rule_fit_candidates:
        print("[Learning] No release learning profile/rule_fit files found (exe dir or defaults/learning) — skipping.")
        return

    selected_profile = profile_candidates[0] if profile_candidates else None
    selected_rule_fit: Path | None = None

    if selected_profile is not None:
        try:
            payload = json.loads(selected_profile.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[Learning] Failed reading release profile {selected_profile}: {exc}")
            payload = {}

        if isinstance(payload, dict):
            learning_cfg = payload.get("learning", {})
            model_cfg = payload.get("model", {})
            if not isinstance(learning_cfg, dict):
                learning_cfg = {}
            if not isinstance(model_cfg, dict):
                model_cfg = {}

            bool_keys = {
                "teaching_learning_enabled",
                "teaching_use_fitted_rules",
                "teaching_apply_in_circle_mode",
                "teaching_isolation_mode",
            }
            float_keys = {
                "teaching_learning_strength",
                "teaching_min_confidence",
                "teaching_no_motion_bias",
            }

            for key in bool_keys:
                if key in learning_cfg:
                    setattr(win.config.beat, key, bool(learning_cfg.get(key)))
            for key in float_keys:
                if key in learning_cfg:
                    try:
                        raw_value = learning_cfg.get(key)
                        if isinstance(raw_value, (int, float, str)):
                            setattr(win.config.beat, key, float(raw_value))
                    except Exception:
                        pass

            # Profile may embed rule_fit path — resolve relative to profile location
            raw_rule_fit = model_cfg.get("rule_fit") or learning_cfg.get("teaching_rule_fit_path") or payload.get("rule_fit")
            if isinstance(raw_rule_fit, str) and raw_rule_fit.strip():
                candidate = Path(raw_rule_fit.strip())
                if not candidate.is_absolute():
                    candidate = selected_profile.parent / candidate
                if candidate.exists() and _is_allowed_frozen_candidate(candidate):
                    selected_rule_fit = candidate

    # Fallback: discovered rule_fit candidates
    if selected_rule_fit is None:
        selected_rule_fit = rule_fit_candidates[0] if rule_fit_candidates else None

    if selected_profile is not None:
        setattr(win.config.beat, 'teaching_profile_path', str(selected_profile))
    if selected_rule_fit is not None:
        win.config.beat.teaching_rule_fit_path = str(selected_rule_fit)

    win.config.beat.teaching_learning_enabled = True
    win.config.beat.teaching_use_fitted_rules = True

    source = "frozen" if frozen else "bundled"
    profile_label = selected_profile.name if selected_profile is not None else "(none)"
    rule_fit_label = str(selected_rule_fit) if selected_rule_fit is not None else "(none)"
    print(f"[Learning] Release defaults applied — source={source} profile={profile_label}, rule_fit={rule_fit_label}")

def on_start_stop(win, checked: bool | None = None):
    """Start/stop audio capture and TCode pipeline.
    Start enables TCode sending (V0=0 until Play). Stop kills everything."""
    if checked is None:
        checked = not win.is_running

    if not win._transport_ready:
        if checked and not win.is_running:
            win._transport_pending_start = True
        win._sync_transport_buttons()
        return

    if win._transport_transition:
        if checked and not win.is_running:
            win._transport_pending_start = True
        elif not checked:
            # Stop should never be dropped; prioritize it over pending start.
            win._transport_pending_stop = True
            win._transport_pending_start = False
            win._transport_pending_play = None
        win._sync_transport_buttons()
        return

    win._transport_transition = True
    pending_transport_action: str | None = None
    try:
        if checked:
            try:
                # Reflect start intent immediately so a quick follow-up click
                # during startup is interpreted as Stop, not another Start.
                win.is_running = True
                win._sync_transport_buttons()

                win._start_engines()
                win._pause_park_active = False
                ui_state = start_stop_ui_state(True)
                win.start_btn.setText(ui_state.start_text)
                win.play_btn.setEnabled(ui_state.play_enabled)
                # Enable TCode sending immediately on Start (V0=0 until Play is pressed)
                set_transport_sending(win.network_engine, True)
                send_zero_volume_immediate(win.network_engine, duration_ms=160)
            except Exception as e:
                print(f"[Main] Start failed: {e}")
                win._stop_engines()
                win.is_sending = False
                set_transport_sending(win.network_engine, False)
        else:
            # Stop should immediately clear play/sending state before shutdown work.
            win._volume_ramp_active = False
            win._play_warmup_active = False
            win._play_warmup_seen_beat = False
            win.is_sending = False
            win._pause_park_active = False
            win._transport_pending_play = None

            # Make stop visually immediate and prevent second-click feel.
            win.is_running = False
            win._sync_transport_buttons()

            # Send zero-volume command before stopping (always, not just when is_sending)
            send_zero_volume_immediate(win.network_engine, duration_ms=160)
            set_transport_sending(win.network_engine, False)
            win._stop_engines()
            # Note: Auto-range state is preserved across stop/start - no reset here
    finally:
        win._transport_transition = False
        win._sync_transport_buttons()

        # Clear stale pending flags that no longer match runtime state.
        if win._transport_pending_start and win.is_running:
            win._transport_pending_start = False
        if win._transport_pending_stop and not win.is_running:
            win._transport_pending_stop = False

        # Stop wins over start when both were requested during transition.
        if win._transport_pending_stop and win.is_running:
            win._transport_pending_stop = False
            pending_transport_action = 'stop'
        elif win._transport_pending_start and not win.is_running:
            win._transport_pending_start = False
            pending_transport_action = 'start'
        elif win._transport_pending_play is not None and win.is_running:
            pending_transport_action = 'play'

    if pending_transport_action == 'stop':
        QTimer.singleShot(0, win._apply_pending_stop)
    elif pending_transport_action == 'start':
        QTimer.singleShot(0, win._apply_pending_start)
    elif pending_transport_action == 'play':
            QTimer.singleShot(0, win._apply_pending_play)

def on_play_pause(win, checked: bool | None = None):
    """Play/pause motion generation. Pause sends V0=0 but keeps TCode pipeline active."""
    if checked is None:
        checked = not win.is_sending

    if win._transport_transition:
        win._transport_pending_play = bool(checked)
        win._sync_transport_buttons()
        return

    if not win.is_running:
        win.is_sending = False
        win._transport_pending_play = None
        win._sync_transport_buttons()
        return
    win.is_sending = checked
    if checked:
        win._pause_park_active = False
        # Re-instantiate StrokeMapper with current config (for live mode switching)
        win.stroke_mapper = StrokeMapper(win.config, get_volume=lambda: win.volume_slider.value() / 100.0, audio_engine=win.audio_engine)
        win._apply_geometry_rest_to_mapper()
        win._apply_learning_config_to_mapper()
        # Warmup gate: allow audio analysis to settle and beat pickup before motion
        win._play_warmup_active = True
        win._play_warmup_started_at = time.time()
        win._play_warmup_seen_beat = False
        send_zero_volume_immediate(win.network_engine, duration_ms=1750)
        # Start volume ramp from 0 to set value over 1.3s
        ramp_state = begin_volume_ramp(time.time())
        win._volume_ramp_active = ramp_state.active
        win._volume_ramp_start_time = ramp_state.start_time
        win._volume_ramp_from = ramp_state.from_volume
        win._volume_ramp_to = ramp_state.to_volume
        # sending_enabled already True from Start — no need to set again
    else:
        win._pause_park_active = True
        # Send V0=0 immediately with fade, but keep TCode pipeline active
        win._play_warmup_active = False
        win._play_warmup_seen_beat = False
        win._volume_ramp_active = False
        win._last_sent_volume_pct = 0.0
        send_zero_volume_immediate(win.network_engine, duration_ms=500)
        # DON'T disable sending_enabled — connection stays active until Stop
    win._transport_pending_play = None
    win._sync_transport_buttons()

def audio_callback(win, event: BeatEvent):
    """Called from audio thread on each frame - NO direct Qt widget access for thread safety"""
    # Emit signal for thread-safe GUI update
    win.signals.beat_detected.emit(event)

    # ── Keyboard Teaching: record frame (runs on audio thread; lock-guarded inside) ──
    teacher = getattr(win, '_keyboard_teacher', None)
    if teacher is not None and teacher.active:
        decision = None
        gate_state = None
        mapper = getattr(win, 'stroke_mapper', None)
        if mapper is not None:
            decision = getattr(mapper, '_last_decision', None)
            intelligence = getattr(mapper, '_intelligence', None)
            if intelligence is not None:
                try:
                    gate_state = intelligence.snapshot_gate_state()
                except Exception:
                    pass
        teacher.on_frame(event, decision, gate_state)

    # Get spectrum for visualization
    spectrum = None
    if win.audio_engine:
        spectrum = win.audio_engine.get_spectrum()
        if spectrum is not None:
            waveform = win.audio_engine.get_waveform()
            sample_rate = int(getattr(win.config.audio, 'sample_rate', 44100))
            spectrum_with_stats = {
                'spectrum': spectrum,
                'peak_energy': event.peak_energy,
                'spectral_flux': event.spectral_flux,
                'waveform': waveform,
                'sample_rate': sample_rate,
            }
            win.signals.spectrum_ready.emit(spectrum_with_stats)

    # Process through stroke mapper
    pause_park_active = bool(getattr(win, '_pause_park_active', False))
    if win.stroke_mapper and (win.is_sending or pause_park_active):
        if win._play_warmup_active:
            if event.is_beat:
                win._play_warmup_seen_beat = True

            now = event.timestamp if event and event.timestamp > 0 else time.time()
            elapsed = max(0.0, now - win._play_warmup_started_at)
            warmup_ready = elapsed >= win._play_warmup_min_seconds and win._play_warmup_seen_beat
            warmup_timeout = elapsed >= win._play_warmup_max_seconds

            if not warmup_ready and not warmup_timeout:
                return

            win._play_warmup_active = False

        cmd = win.stroke_mapper.process_beat(
            event,
            silence_override=(True if pause_park_active and not win.is_sending else None),
        )

        # Sync Energy Response slider when the ramp engine is driving it
        _ramp = getattr(win.stroke_mapper, '_ramp_engine', None)
        if _ramp is not None:
            _sv = _ramp.speed_display_value
            _slider = getattr(win, 'main_silence_close_slider', None)
            if _sv is not None and _slider is not None:
                _slider.blockSignals(True)
                _slider.setValue(_sv)
                _slider.blockSignals(False)
                if hasattr(win, '_set_energy_response_display'):
                    win._set_energy_response_display(_sv)

        if cmd and win.network_engine:
            # Compute P0/F0 and attach to command (thread-safe, no widget access)
            win._compute_and_attach_tcode(cmd, event, spectrum)
            if pause_park_active and not win.is_sending:
                cmd.volume = 0.0
            else:
                apply_volume_ramp(
                    cmd,
                    volume_ramp_active=win._volume_ramp_active,
                    volume_ramp_start_time=win._volume_ramp_start_time,
                    volume_ramp_duration=win._volume_ramp_duration,
                    volume_ramp_from=win._volume_ramp_from,
                    volume_ramp_to=win._volume_ramp_to,
                )
            win._last_sent_volume_pct = float(cmd.volume) * 100.0
            win.network_engine.send_command(cmd)
    elif event.is_beat and not win.is_sending:
        print("[Main] Beat detected but Play not enabled")

def on_beat(win, event: BeatEvent):
    """Handle beat event in GUI thread"""
    if getattr(win, '_is_shutting_down', False):
        return

    # ===== METRONOME SYNC INDICATOR (updates every frame, not just on beat) =====
    acf_conf_raw = getattr(event, 'acf_confidence', 0.0)
    metro_bpm_raw = getattr(event, 'metronome_bpm', 0.0)
    try:
        acf_conf = float(acf_conf_raw)
    except (TypeError, ValueError):
        acf_conf = 0.0
    try:
        metro_bpm = float(metro_bpm_raw)
    except (TypeError, ValueError):
        metro_bpm = 0.0

    if not np.isfinite(acf_conf):
        acf_conf = 0.0
    if not np.isfinite(metro_bpm):
        metro_bpm = 0.0

    is_event_beat = bool(getattr(event, 'is_beat', False))
    tempo_locked = bool(getattr(event, 'tempo_locked', False))
    tempo_lock_required = bool(getattr(getattr(win.config, 'beat', None), 'tempo_lock_required', False))
    relaxed_conf = float(getattr(getattr(win.config, 'beat', None), 'teaching_metronome_relaxed_confidence', 0.14) or 0.14)
    if not np.isfinite(relaxed_conf):
        relaxed_conf = 0.14
    relaxed_conf = float(np.clip(relaxed_conf, 0.0, 1.0))

    # Display readiness should never be stricter than the metronome-sync lamp.
    # If sync lamp is green (acf_conf >= 0.25), beat/downbeat indicators should blink.
    metronome_ready_for_display = bool(
        metro_bpm > 0.0 and (tempo_locked or acf_conf >= relaxed_conf or acf_conf >= 0.25)
    )
    beat_passes_display_gate = bool(
        is_event_beat and (
            metronome_ready_for_display
            or (not tempo_lock_required and metro_bpm <= 0.0)
        )
    )

    if hasattr(win, 'metronome_sync_indicator') and win.metronome_sync_indicator is not None:
        try:
            if metro_bpm <= 0 or acf_conf < 0.05:
                win.metronome_sync_indicator.setStyleSheet("color: #333; font-size: 20px;")  # Off
            elif acf_conf < 0.25:
                win.metronome_sync_indicator.setStyleSheet("color: #cc0; font-size: 20px;")  # Yellow: locking
            else:
                win.metronome_sync_indicator.setStyleSheet("color: #0f0; font-size: 20px;")  # Green: locked
        except RuntimeError:
            pass

    if beat_passes_display_gate:
        # Track beat time for auto-adjustment feature
        win._last_beat_time_for_auto = time.time()

        # ===== REAL-TIME METRIC FEEDBACK =====
        # Compute energy margin and apply metric-based adjustments
        if hasattr(win, 'audio_engine') and win.audio_engine is not None:
            # Get energy margin metric and apply feedback if enabled
            margin, should_adjust, direction = win.audio_engine.compute_energy_margin_feedback(
                event.peak_energy, 
                callback=win._on_metric_feedback
            )

            # Target-BPM behavior disabled

        # Light up the beat indicator (green for any beat)
        if hasattr(win, 'beat_indicator') and win.beat_indicator is not None:
            win.beat_indicator.setStyleSheet("color: #0f0; font-size: 24px;")
        # Reset timer to keep it lit for minimum duration
        if hasattr(win, 'beat_timer') and win.beat_timer is not None:
            win.beat_timer.stop()
            win.beat_timer.start(win.beat_indicator_min_duration)
        # Get tempo from audio engine (now includes smoothing, beat prediction, downbeat detection)
        if hasattr(win, 'audio_engine') and win.audio_engine is not None:
            tempo_info = win.audio_engine.get_tempo_info()
            if tempo_info['bpm'] > 0:
                # Use event.is_downbeat (frozen at construction time) instead of
                # polling get_tempo_info() which races with audio thread clearing the flag
                is_downbeat = bool(getattr(event, 'is_downbeat', False)) and metronome_ready_for_display

                # Light up downbeat indicator (cyan/blue for downbeat)
                if is_downbeat:
                    if hasattr(win, 'downbeat_indicator') and win.downbeat_indicator is not None:
                        win.downbeat_indicator.setStyleSheet("color: #0ff; font-size: 24px;")
                    if hasattr(win, 'downbeat_timer') and win.downbeat_timer is not None:
                        win.downbeat_timer.stop()
                        win.downbeat_timer.start(win.beat_indicator_min_duration)
                    # Record downbeat for sensitivity metric
                    if hasattr(win, 'audio_engine') and win.audio_engine is not None:
                        pass  # downbeat recording removed
    # Show reset in GUI and console if tempo was reset
    if hasattr(event, 'tempo_reset') and event.tempo_reset:
        print("[GUI] Beat counter/tempo reset due to silence.")

def on_spectrum(win, spectrum: np.ndarray):
    """Queue spectrum for throttled update"""
    if getattr(win, '_is_shutting_down', False):
        return
    win._pending_spectrum = spectrum

def do_spectrum_update(win):
    """Actually update spectrum at throttled rate - only update visible visualizer"""
    if getattr(win, '_is_shutting_down', False):
        win._pending_spectrum = None
        return
    if win._pending_spectrum is not None:
        # Handle both old format (numpy array) and new format (dict with stats)
        if isinstance(win._pending_spectrum, dict):
            spectrum = win._pending_spectrum['spectrum']
            peak, flux = win._compute_visual_metrics(spectrum)
            waveform = win._pending_spectrum.get('waveform')
            sample_rate = int(win._pending_spectrum.get('sample_rate', getattr(win.config.audio, 'sample_rate', 44100)))
            # Only update the currently visible in-window visualizer for performance
            if hasattr(win, 'waveform_canvas') and win.waveform_canvas is not None and win.waveform_canvas.isVisible():
                win.waveform_canvas.update_from_audio(waveform, sample_rate)
            elif hasattr(win, 'freqdb_canvas') and win.freqdb_canvas is not None and win.freqdb_canvas.isVisible():
                win.freqdb_canvas.update_from_spectrum(spectrum, sample_rate)
            elif hasattr(win, 'fft_bin_canvas') and win.fft_bin_canvas is not None and win.fft_bin_canvas.isVisible():
                win.fft_bin_canvas.update_from_spectrum(spectrum, sample_rate)
        else:
            # Legacy format - only update visible visualizer
            peak, flux = win._compute_visual_metrics(win._pending_spectrum)
            if hasattr(win, 'waveform_canvas') and win.waveform_canvas is not None and win.waveform_canvas.isVisible() and win.audio_engine is not None:
                win.waveform_canvas.update_from_audio(win.audio_engine.get_waveform(), int(getattr(win.config.audio, 'sample_rate', 44100)))
            elif hasattr(win, 'freqdb_canvas') and win.freqdb_canvas is not None and win.freqdb_canvas.isVisible():
                win.freqdb_canvas.update_from_spectrum(win._pending_spectrum, int(getattr(win.config.audio, 'sample_rate', 44100)))
            elif hasattr(win, 'fft_bin_canvas') and win.fft_bin_canvas is not None and win.fft_bin_canvas.isVisible():
                win.fft_bin_canvas.update_from_spectrum(win._pending_spectrum, int(getattr(win.config.audio, 'sample_rate', 44100)))
        win._pending_spectrum = None

def update_display(win):
    """Periodic display update + sync cached widget states for thread-safe audio access"""
    if getattr(win, '_is_shutting_down', False):
        return

    def _is_live_widget_attr(name: str) -> bool:
        widget = getattr(win, name, None)
        if widget is None:
            return False
        try:
            widget.parent()
        except RuntimeError:
            return False
        return True

    if win.stroke_mapper:
        alpha, beta = win.stroke_mapper.get_current_position()
        win.position_canvas.update_position(alpha, beta)

    # Refresh keyboard teaching overlay at display rate
    teacher = getattr(win, '_keyboard_teacher', None)
    if teacher is not None and teacher.active:
        win.position_canvas.update_teacher_preview(teacher.is_parked, teacher.speed_scale, teacher._last_bpm)
        win._update_keyboard_teacher_label()
    else:
        win.position_canvas.clear_teacher_preview()

    # Sync widget states to cached values for thread-safe reading by audio thread.
    # Some controls may not exist yet (e.g., optional dialogs/tabs not instantiated),
    # so fall back to cached state instead of raising per-frame AttributeError.
    control_toggle_names = (
        'pulse_enabled_checkbox',
        'f0_enabled_checkbox',
        'p1_enabled_checkbox',
        'p3_enabled_checkbox',
    )
    controls_toggle_ready = all(_is_live_widget_attr(name) for name in control_toggle_names)
    if controls_toggle_ready:
        # P0/F0/P1/P3 enable state MUST be synced immediately (every frame) for instant response
        new_p0_enabled = win.pulse_enabled_checkbox.isChecked()
        new_f0_enabled = win.f0_enabled_checkbox.isChecked()
        new_p1_enabled = win.p1_enabled_checkbox.isChecked()
        new_p3_enabled = win.p3_enabled_checkbox.isChecked()
    else:
        new_p0_enabled = bool(getattr(win, '_cached_p0_enabled', False))
        new_f0_enabled = bool(getattr(win, '_cached_f0_enabled', False))
        new_p1_enabled = bool(getattr(win, '_cached_p1_enabled', False))
        new_p3_enabled = bool(getattr(win, '_cached_p3_enabled', False))

    # Handle P0/C0 checkboxes being unchecked (enabled→disabled transition)
    # Simply stop sending the axis — do NOT send 0 value, which still affects device
    if win._prev_p0_enabled and not new_p0_enabled:
        win._cached_pulse_display = "Pulse Freq: off"
        win._p0_freq_window.clear()
        print("[Main] P0 disabled — stopped sending")
    if win._prev_f0_enabled and not new_f0_enabled:
        win._cached_carrier_display = "Carrier Freq: off"
        win._f0_freq_window.clear()
        win._f0_last_sent_tcode = None
        print("[Main] C0 disabled — stopped sending")
    if win._prev_p1_enabled and not new_p1_enabled:
        win._cached_p1_display = "Pulse Width: off"
        win._p1_window.clear()
        print("[Main] P1 disabled — stopped sending")
    if win._prev_p3_enabled and not new_p3_enabled:
        win._cached_p3_display = "Rise Time: off"
        win._p3_window.clear()
        print("[Main] P3 disabled — stopped sending")

    win._prev_p0_enabled = new_p0_enabled
    win._prev_f0_enabled = new_f0_enabled
    win._prev_p1_enabled = new_p1_enabled
    win._prev_p3_enabled = new_p3_enabled
    win._cached_p0_enabled = new_p0_enabled
    win._cached_f0_enabled = new_f0_enabled
    win._cached_p1_enabled = new_p1_enabled
    win._cached_p3_enabled = new_p3_enabled

    # Update freq display labels — throttled to 100ms
    now = time.time()
    if now - win._last_freq_display_time > 0.1:
        win._last_freq_display_time = now
        # Update freq display labels from cached strings (written by audio thread)
        win.pulse_freq_label.setText(win._cached_pulse_display)
        win.carrier_freq_label.setText(win._cached_carrier_display)
        win.p1_display_label.setText(win._cached_p1_display)
        win.p3_display_label.setText(win._cached_p3_display)
        win.pulse_freq_label.setStyleSheet(f"color: {'#fff' if new_p0_enabled else '#0af'}; font-size: 10px;")
        win.carrier_freq_label.setStyleSheet(f"color: {'#fff' if new_f0_enabled else '#0af'}; font-size: 10px;")
        win.p1_display_label.setStyleSheet(f"color: {'#fff' if new_p1_enabled else '#0af'}; font-size: 10px;")
        win.p3_display_label.setStyleSheet(f"color: {'#fff' if new_p3_enabled else '#0af'}; font-size: 10px;")
        # Show target volume when stopped, actual sent tcode volume when running.
        # Pause-park sends live commands at V0, so treat it as live-output display too.
        show_live_output = bool(win.is_sending or getattr(win, '_pause_park_active', False))
        if show_live_output:
            display_pct = win._last_sent_volume_pct
            win.volume_slider.value_label.setStyleSheet("color: #fff;")
            win.volume_slider.label.setStyleSheet("color: #fff;")
        else:
            display_pct = float(win.volume_slider.value())
            win.volume_slider.value_label.setStyleSheet("color: #0af;")
            win.volume_slider.label.setStyleSheet("color: #0af;")
        win.volume_slider.value_label.setText(f"{display_pct:.0f}")
        control_sync_names = (
            'pulse_mode_combo', 'pulse_invert_checkbox',
            'f0_mode_combo', 'f0_invert_checkbox',
            'tcode_freq_range_slider', 'f0_tcode_range_slider',
            'p1_mode_combo', 'p1_invert_checkbox',
            'p1_tcode_range_slider', 'p1_monitor_range_slider', 'p1_weight_slider',
            'p3_mode_combo', 'p3_invert_checkbox',
            'p3_tcode_range_slider', 'p3_monitor_range_slider', 'p3_weight_slider',
        )
        if all(_is_live_widget_attr(name) for name in control_sync_names):
            # Sync other combo/checkbox states for audio thread (throttled is fine)
            win._cached_pulse_mode = win.pulse_mode_combo.currentIndex()
            win._cached_pulse_invert = win.pulse_invert_checkbox.isChecked()
            win._cached_f0_mode = win.f0_mode_combo.currentIndex()
            win._cached_f0_invert = win.f0_invert_checkbox.isChecked()
            # Sync TCode Sent slider values for thread-safe access
            win._cached_tcode_freq_min = int(win.tcode_freq_range_slider.low())
            win._cached_tcode_freq_max = int(win.tcode_freq_range_slider.high())
            win._cached_f0_tcode_min = int(win.f0_tcode_range_slider.low())
            win._cached_f0_tcode_max = int(win.f0_tcode_range_slider.high())
            # Sync P1 (Pulse Width) widget states
            win._cached_p1_mode = win.p1_mode_combo.currentIndex()
            win._cached_p1_invert = win.p1_invert_checkbox.isChecked()
            win._cached_p1_tcode_min = int(win.p1_tcode_range_slider.low())
            win._cached_p1_tcode_max = int(win.p1_tcode_range_slider.high())
            win.config.pulse_width.monitor_freq_min = win.p1_monitor_range_slider.low()
            win.config.pulse_width.monitor_freq_max = win.p1_monitor_range_slider.high()
            win.config.pulse_width.weight = win.p1_weight_slider.value()
            # Sync P3 (Rise Time) widget states
            win._cached_p3_mode = win.p3_mode_combo.currentIndex()
            win._cached_p3_invert = win.p3_invert_checkbox.isChecked()
            win._cached_p3_tcode_min = int(win.p3_tcode_range_slider.low())
            win._cached_p3_tcode_max = int(win.p3_tcode_range_slider.high())
            win.config.rise_time.monitor_freq_min = win.p3_monitor_range_slider.low()
            win.config.rise_time.monitor_freq_max = win.p3_monitor_range_slider.high()
            win.config.rise_time.weight = win.p3_weight_slider.value()

        # Update peak floor bars on all visualizers
        peak_floor = win.config.beat.peak_floor
        for canvas_name in ['waveform_canvas', 'freqdb_canvas', 'fft_bin_canvas']:
            if hasattr(win, canvas_name):
                canvas = getattr(win, canvas_name)
                if hasattr(canvas, 'set_peak_floor'):
                    canvas.set_peak_floor(peak_floor)

    # Handle volume ramp completion
    if win._volume_ramp_active:
        elapsed = time.time() - win._volume_ramp_start_time
        if elapsed >= win._volume_ramp_duration:
            win._volume_ramp_active = False

    # ===== TIMER-DRIVEN METRIC FEEDBACK: Audio Amp =====
    # These fire from the display timer (not from _on_beat) so they can
    # detect the ABSENCE of beats and escalate accordingly.
    if (not getattr(win, '_is_shutting_down', False)
            and hasattr(win, 'audio_engine') and win.audio_engine is not None):
        now = time.perf_counter()
        if _is_live_widget_attr('audio_gain_slider'):
            win.audio_engine.compute_audio_amp_feedback(now, callback=win._on_metric_feedback)

        # Target-BPM auto-align behavior disabled

        # Keep metric state polling active even though traffic-light UI is removed.
        win.audio_engine.get_metric_states()

# ── Keyboard Teaching Mode: key events ─────────────────────────

_ARROW_MAP = {
    Qt.Key.Key_Up: "up",
    Qt.Key.Key_Down: "down",
    Qt.Key.Key_Left: "left",
    Qt.Key.Key_Right: "right",
}

