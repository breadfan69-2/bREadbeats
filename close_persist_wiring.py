from config import Config


def _require_window_attr(window, attr_name: str):
    try:
        return getattr(window, attr_name)
    except AttributeError as exc:
        raise AttributeError(
            f"persist_runtime_ui_to_config missing required control: {attr_name}"
        ) from exc


def _optional_window_attr(window, attr_name: str):
    return getattr(window, attr_name, None)


def _commit_pending_spinbox_text(window) -> None:
    """Force-edit commit for spinboxes so close-save captures typed values."""
    spinbox_names = (
        "p0_monitor_min_spin", "p0_monitor_max_spin",
        "p0_sent_min_spin", "p0_sent_max_spin",
        "f0_monitor_min_spin", "f0_monitor_max_spin",
        "f0_sent_min_spin", "f0_sent_max_spin",
    )
    for name in spinbox_names:
        control = getattr(window, name, None)
        interpret_text = getattr(control, "interpretText", None)
        if callable(interpret_text):
            interpret_text()


def persist_runtime_ui_to_config(window, config: Config) -> None:
    """Copy selected runtime UI control values into config on shutdown."""
    _commit_pending_spinbox_text(window)

    pulse_freq_range_slider = _optional_window_attr(window, "pulse_freq_range_slider")
    tcode_freq_range_slider = _optional_window_attr(window, "tcode_freq_range_slider")
    freq_weight_slider = _optional_window_attr(window, "freq_weight_slider")
    pulse_mode_combo = _optional_window_attr(window, "pulse_mode_combo")
    pulse_invert_checkbox = _optional_window_attr(window, "pulse_invert_checkbox")
    pulse_enabled_checkbox = _optional_window_attr(window, "pulse_enabled_checkbox")
    f0_freq_range_slider = _optional_window_attr(window, "f0_freq_range_slider")
    f0_tcode_range_slider = _optional_window_attr(window, "f0_tcode_range_slider")
    f0_weight_slider = _optional_window_attr(window, "f0_weight_slider")
    f0_mode_combo = _optional_window_attr(window, "f0_mode_combo")
    f0_invert_checkbox = _optional_window_attr(window, "f0_invert_checkbox")
    f0_enabled_checkbox = _optional_window_attr(window, "f0_enabled_checkbox")
    volume_slider = _optional_window_attr(window, "volume_slider")
    tempo_tracking_checkbox = _optional_window_attr(window, "tempo_tracking_checkbox")
    time_sig_combo = _optional_window_attr(window, "time_sig_combo")
    stability_threshold_slider = _optional_window_attr(window, "stability_threshold_slider")
    tempo_timeout_slider = _optional_window_attr(window, "tempo_timeout_slider")
    phase_snap_slider = _optional_window_attr(window, "phase_snap_slider")
    metrics_global_cb = _optional_window_attr(window, "metrics_global_cb")

    if pulse_freq_range_slider is not None and hasattr(pulse_freq_range_slider, "low") and hasattr(pulse_freq_range_slider, "high"):
        config.pulse_freq.monitor_freq_min = pulse_freq_range_slider.low()
        config.pulse_freq.monitor_freq_max = pulse_freq_range_slider.high()
    if tcode_freq_range_slider is not None and hasattr(tcode_freq_range_slider, "low") and hasattr(tcode_freq_range_slider, "high"):
        config.pulse_freq.tcode_min = int(tcode_freq_range_slider.low())
        config.pulse_freq.tcode_max = int(tcode_freq_range_slider.high())
    if freq_weight_slider is not None and hasattr(freq_weight_slider, "value"):
        config.pulse_freq.freq_weight = freq_weight_slider.value()
    if pulse_mode_combo is not None and hasattr(pulse_mode_combo, "currentIndex"):
        config.pulse_freq.mode = int(pulse_mode_combo.currentIndex())
    if pulse_invert_checkbox is not None and hasattr(pulse_invert_checkbox, "isChecked"):
        config.pulse_freq.invert = bool(pulse_invert_checkbox.isChecked())
    if pulse_enabled_checkbox is not None and hasattr(pulse_enabled_checkbox, "isChecked"):
        config.pulse_freq.enabled = bool(pulse_enabled_checkbox.isChecked())

    if f0_freq_range_slider is not None and hasattr(f0_freq_range_slider, "low") and hasattr(f0_freq_range_slider, "high"):
        config.carrier_freq.monitor_freq_min = f0_freq_range_slider.low()
        config.carrier_freq.monitor_freq_max = f0_freq_range_slider.high()
    if f0_tcode_range_slider is not None and hasattr(f0_tcode_range_slider, "low") and hasattr(f0_tcode_range_slider, "high"):
        config.carrier_freq.tcode_min = int(f0_tcode_range_slider.low())
        config.carrier_freq.tcode_max = int(f0_tcode_range_slider.high())
    if f0_weight_slider is not None and hasattr(f0_weight_slider, "value"):
        config.carrier_freq.freq_weight = f0_weight_slider.value()
    if f0_mode_combo is not None and hasattr(f0_mode_combo, "currentIndex"):
        config.carrier_freq.mode = int(f0_mode_combo.currentIndex())
    if f0_invert_checkbox is not None and hasattr(f0_invert_checkbox, "isChecked"):
        config.carrier_freq.invert = bool(f0_invert_checkbox.isChecked())
    if f0_enabled_checkbox is not None and hasattr(f0_enabled_checkbox, "isChecked"):
        config.carrier_freq.enabled = bool(f0_enabled_checkbox.isChecked())

    if volume_slider is not None and hasattr(volume_slider, "value"):
        config.volume = volume_slider.value() / 100.0
    config.alpha_weight = 1.0
    config.beta_weight = 1.0

    if tempo_tracking_checkbox is not None and hasattr(tempo_tracking_checkbox, "isChecked"):
        config.beat.tempo_tracking_enabled = tempo_tracking_checkbox.isChecked()
    if time_sig_combo is not None and hasattr(time_sig_combo, "currentIndex"):
        beats_map = {0: 4, 1: 3, 2: 6}
        config.beat.beats_per_measure = beats_map.get(time_sig_combo.currentIndex(), 4)
    if stability_threshold_slider is not None and hasattr(stability_threshold_slider, "value"):
        config.beat.stability_threshold = stability_threshold_slider.value()
    if tempo_timeout_slider is not None and hasattr(tempo_timeout_slider, "value"):
        config.beat.tempo_timeout_ms = int(tempo_timeout_slider.value())
    if phase_snap_slider is not None and hasattr(phase_snap_slider, "value"):
        config.beat.phase_snap_weight = phase_snap_slider.value()

    if metrics_global_cb is not None and hasattr(metrics_global_cb, "isChecked"):
        config.auto_adjust.metrics_global_enabled = metrics_global_cb.isChecked()

    intensity_ramp_spin = _optional_window_attr(window, "intensity_ramp_spin")
    if intensity_ramp_spin is not None and hasattr(intensity_ramp_spin, "value"):
        config.stroke.intensity_ramp_hours = float(intensity_ramp_spin.value())

    intensity_ramp_target_combo = _optional_window_attr(window, "intensity_ramp_target_combo")
    if intensity_ramp_target_combo is not None and hasattr(intensity_ramp_target_combo, "currentText"):
        target = str(intensity_ramp_target_combo.currentText() or "both").strip().lower()
        if target not in {"size", "speed", "both"}:
            target = "both"
        config.stroke.intensity_ramp_target = target
    
        # Always persist trigger bus and related fields from config.beat
        # These fields are not controlled by UI, so ensure they are saved
        bus_fields = [
            "trigger_bus_refractory_ms",
            "trigger_bus_arm_threshold",
            "trigger_bus_release_threshold",
            "trigger_bus_sustain_frames",
            "trigger_bus_weight_sub_bass",
            "trigger_bus_weight_low_mid",
            "trigger_bus_weight_mid",
            "trigger_bus_weight_high",
            "trigger_bus_mask_floor",
            "bass_dominance_weighting_enabled",
            "transient_classification_enabled",
            "transient_full_motion_min_kick_conf",
            "transient_full_motion_min_bass_dom",
            "transient_full_motion_decisive_bass_dom",
            "transient_full_motion_min_flux",
            "transient_full_motion_min_energy_fullness",
        ]
        for field in bus_fields:
            # No UI controls, so just ensure config.beat.<field> is present for persistence
            setattr(config.beat, field, getattr(config.beat, field, None))
