from config import Config


def persist_runtime_ui_to_config(window, config: Config) -> None:
    """Copy selected runtime UI control values into config on shutdown."""
    config.stroke.phase_advance = window.phase_advance_slider.value()

    config.pulse_freq.monitor_freq_min = window.pulse_freq_range_slider.low()
    config.pulse_freq.monitor_freq_max = window.pulse_freq_range_slider.high()
    config.pulse_freq.tcode_min = int(window.tcode_freq_range_slider.low())
    config.pulse_freq.tcode_max = int(window.tcode_freq_range_slider.high())
    config.pulse_freq.freq_weight = window.freq_weight_slider.value()

    config.carrier_freq.monitor_freq_min = window.f0_freq_range_slider.low()
    config.carrier_freq.monitor_freq_max = window.f0_freq_range_slider.high()
    config.carrier_freq.tcode_min = int(window.f0_tcode_range_slider.low())
    config.carrier_freq.tcode_max = int(window.f0_tcode_range_slider.high())
    config.carrier_freq.freq_weight = window.f0_weight_slider.value()

    config.volume = window.volume_slider.value() / 100.0
    config.alpha_weight = window.alpha_weight_slider.value()
    config.beta_weight = window.beta_weight_slider.value()

    config.beat.tempo_tracking_enabled = window.tempo_tracking_checkbox.isChecked()
    beats_map = {0: 4, 1: 3, 2: 6}
    config.beat.beats_per_measure = beats_map.get(window.time_sig_combo.currentIndex(), 4)
    config.beat.stability_threshold = window.stability_threshold_slider.value()
    config.beat.tempo_timeout_ms = int(window.tempo_timeout_slider.value())
    config.beat.phase_snap_weight = window.phase_snap_slider.value()

    config.auto_adjust.metrics_global_enabled = window.metrics_global_cb.isChecked()
