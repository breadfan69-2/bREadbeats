from config import Config


def _require_window_attr(window, attr_name: str):
    try:
        return getattr(window, attr_name)
    except AttributeError as exc:
        raise AttributeError(
            f"persist_runtime_ui_to_config missing required control: {attr_name}"
        ) from exc


def persist_runtime_ui_to_config(window, config: Config) -> None:
    """Copy selected runtime UI control values into config on shutdown."""
    phase_advance_slider = _require_window_attr(window, "phase_advance_slider")
    pulse_freq_range_slider = _require_window_attr(window, "pulse_freq_range_slider")
    tcode_freq_range_slider = _require_window_attr(window, "tcode_freq_range_slider")
    freq_weight_slider = _require_window_attr(window, "freq_weight_slider")
    f0_freq_range_slider = _require_window_attr(window, "f0_freq_range_slider")
    f0_tcode_range_slider = _require_window_attr(window, "f0_tcode_range_slider")
    f0_weight_slider = _require_window_attr(window, "f0_weight_slider")
    volume_slider = _require_window_attr(window, "volume_slider")
    alpha_weight_slider = _require_window_attr(window, "alpha_weight_slider")
    beta_weight_slider = _require_window_attr(window, "beta_weight_slider")
    tempo_tracking_checkbox = _require_window_attr(window, "tempo_tracking_checkbox")
    time_sig_combo = _require_window_attr(window, "time_sig_combo")
    stability_threshold_slider = _require_window_attr(window, "stability_threshold_slider")
    tempo_timeout_slider = _require_window_attr(window, "tempo_timeout_slider")
    phase_snap_slider = _require_window_attr(window, "phase_snap_slider")
    metrics_global_cb = _require_window_attr(window, "metrics_global_cb")

    config.stroke.phase_advance = phase_advance_slider.value()

    config.pulse_freq.monitor_freq_min = pulse_freq_range_slider.low()
    config.pulse_freq.monitor_freq_max = pulse_freq_range_slider.high()
    config.pulse_freq.tcode_min = int(tcode_freq_range_slider.low())
    config.pulse_freq.tcode_max = int(tcode_freq_range_slider.high())
    config.pulse_freq.freq_weight = freq_weight_slider.value()

    config.carrier_freq.monitor_freq_min = f0_freq_range_slider.low()
    config.carrier_freq.monitor_freq_max = f0_freq_range_slider.high()
    config.carrier_freq.tcode_min = int(f0_tcode_range_slider.low())
    config.carrier_freq.tcode_max = int(f0_tcode_range_slider.high())
    config.carrier_freq.freq_weight = f0_weight_slider.value()

    config.volume = volume_slider.value() / 100.0
    config.alpha_weight = alpha_weight_slider.value()
    config.beta_weight = beta_weight_slider.value()

    config.beat.tempo_tracking_enabled = tempo_tracking_checkbox.isChecked()
    beats_map = {0: 4, 1: 3, 2: 6}
    config.beat.beats_per_measure = beats_map.get(time_sig_combo.currentIndex(), 4)
    config.beat.stability_threshold = stability_threshold_slider.value()
    config.beat.tempo_timeout_ms = int(tempo_timeout_slider.value())
    config.beat.phase_snap_weight = phase_snap_slider.value()

    config.auto_adjust.metrics_global_enabled = metrics_global_cb.isChecked()
