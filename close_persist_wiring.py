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


def persist_runtime_ui_to_config(window, config: Config) -> None:
    """Copy selected runtime UI control values into config on shutdown."""
    pulse_freq_range_slider = _optional_window_attr(window, "pulse_freq_range_slider")
    tcode_freq_range_slider = _optional_window_attr(window, "tcode_freq_range_slider")
    freq_weight_slider = _optional_window_attr(window, "freq_weight_slider")
    f0_freq_range_slider = _optional_window_attr(window, "f0_freq_range_slider")
    f0_tcode_range_slider = _optional_window_attr(window, "f0_tcode_range_slider")
    f0_weight_slider = _optional_window_attr(window, "f0_weight_slider")
    volume_slider = _optional_window_attr(window, "volume_slider")
    tempo_tracking_checkbox = _optional_window_attr(window, "tempo_tracking_checkbox")
    time_sig_combo = _optional_window_attr(window, "time_sig_combo")
    stability_threshold_slider = _optional_window_attr(window, "stability_threshold_slider")
    tempo_timeout_slider = _optional_window_attr(window, "tempo_timeout_slider")
    phase_snap_slider = _optional_window_attr(window, "phase_snap_slider")
    metrics_global_cb = _optional_window_attr(window, "metrics_global_cb")

    if hasattr(pulse_freq_range_slider, "low") and hasattr(pulse_freq_range_slider, "high"):
        config.pulse_freq.monitor_freq_min = pulse_freq_range_slider.low()
        config.pulse_freq.monitor_freq_max = pulse_freq_range_slider.high()
    if hasattr(tcode_freq_range_slider, "low") and hasattr(tcode_freq_range_slider, "high"):
        config.pulse_freq.tcode_min = int(tcode_freq_range_slider.low())
        config.pulse_freq.tcode_max = int(tcode_freq_range_slider.high())
    if hasattr(freq_weight_slider, "value"):
        config.pulse_freq.freq_weight = freq_weight_slider.value()

    if hasattr(f0_freq_range_slider, "low") and hasattr(f0_freq_range_slider, "high"):
        config.carrier_freq.monitor_freq_min = f0_freq_range_slider.low()
        config.carrier_freq.monitor_freq_max = f0_freq_range_slider.high()
    if hasattr(f0_tcode_range_slider, "low") and hasattr(f0_tcode_range_slider, "high"):
        config.carrier_freq.tcode_min = int(f0_tcode_range_slider.low())
        config.carrier_freq.tcode_max = int(f0_tcode_range_slider.high())
    if hasattr(f0_weight_slider, "value"):
        config.carrier_freq.freq_weight = f0_weight_slider.value()

    if hasattr(volume_slider, "value"):
        config.volume = volume_slider.value() / 100.0
    config.alpha_weight = 1.0
    config.beta_weight = 1.0

    if hasattr(tempo_tracking_checkbox, "isChecked"):
        config.beat.tempo_tracking_enabled = tempo_tracking_checkbox.isChecked()
    if hasattr(time_sig_combo, "currentIndex"):
        beats_map = {0: 4, 1: 3, 2: 6}
        config.beat.beats_per_measure = beats_map.get(time_sig_combo.currentIndex(), 4)
    if hasattr(stability_threshold_slider, "value"):
        config.beat.stability_threshold = stability_threshold_slider.value()
    if hasattr(tempo_timeout_slider, "value"):
        config.beat.tempo_timeout_ms = int(tempo_timeout_slider.value())
    if hasattr(phase_snap_slider, "value"):
        config.beat.phase_snap_weight = phase_snap_slider.value()

    if hasattr(metrics_global_cb, "isChecked"):
        config.auto_adjust.metrics_global_enabled = metrics_global_cb.isChecked()

    intensity_ramp_spin = _optional_window_attr(window, "intensity_ramp_spin")
    if hasattr(intensity_ramp_spin, "value"):
        config.stroke.intensity_ramp_hours = float(intensity_ramp_spin.value())

    intensity_ramp_target_combo = _optional_window_attr(window, "intensity_ramp_target_combo")
    if hasattr(intensity_ramp_target_combo, "currentText"):
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
