from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from audio_modules.event_detector import EventDetectorConfig
from pmv_audio_analysis import AnalysisConfig
from pmv_automap import AutomapConfig
from curve_editor import CurveEditorDialog
from pmv_axis_converter import PRESET_CURVES, AxisConfig
from pmv_beat_engine import BeatDetectionConfig
from pmv_position_mapper import MLConfig, MappingConfig
from widgets import CollapsibleGroupBox, SliderWithLabel


class StepButtonBar(QWidget):
    """Top-level PMV pipeline step controls."""

    step_requested = pyqtSignal(int)

    _STEP_LABELS = {
        1: "1. Load Audio",
        2: "2. Analyze",
        3: "3. Detect Beats",
        4: "4. Generate",
        5: "5. Export",
    }

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._buttons: dict[int, QPushButton] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        for step in range(1, 6):
            btn = QPushButton(self._STEP_LABELS[step])
            btn.setEnabled(step == 1)
            btn.clicked.connect(lambda _checked=False, s=step: self.step_requested.emit(s))
            layout.addWidget(btn)
            self._buttons[step] = btn

    def set_step_enabled(self, step: int, enabled: bool) -> None:
        button = self._buttons.get(int(step))
        if button is not None:
            button.setEnabled(bool(enabled))

    def set_step_status(self, step: int, status: str) -> None:
        button = self._buttons.get(int(step))
        if button is None:
            return

        base = self._STEP_LABELS.get(int(step), f"Step {step}")
        badge = {
            "ready": "",
            "running": " - running",
            "done": " - done",
            "error": " - error",
        }.get(str(status), "")
        button.setText(f"{base}{badge}")


class PMVControlsPanel(QWidget):
    """PMV controls panel with full config sections."""

    config_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(6)

        self.step_bar = StepButtonBar(self)
        outer_layout.addWidget(self.step_bar)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(10)

        self._build_analysis_section()
        self._build_beat_section()
        self._build_mapping_section()
        self._build_ml_section()
        self._build_automap_section()
        self._build_axis_section()

        # Step -> sections mapping for auto-collapse/expand
        self._step_sections: dict[int, list[CollapsibleGroupBox]] = {
            2: [self._analysis_group],
            3: [self._beat_group],
            4: [self._mapping_group, self._ml_group, self._automap_group, self._axis_group],
        }

        self._layout.addStretch(1)
        self._scroll.setWidget(self._container)
        outer_layout.addWidget(self._scroll, 1)

    def _emit_changed(self, *_args: Any) -> None:
        self.config_changed.emit()

    def on_step_completed(self, step: int) -> None:
        """Auto-collapse the completed step's sections and expand the next."""
        # Collapse sections belonging to the completed step
        for group in self._step_sections.get(step, []):
            group.setCollapsed(True)
        # Expand sections belonging to the next step
        next_step = step + 1
        for group in self._step_sections.get(next_step, []):
            group.setCollapsed(False)

    @staticmethod
    def _curve_row(text: str, combo: QComboBox, edit_btn: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(text)
        label.setMinimumWidth(150)
        layout.addWidget(label)
        layout.addWidget(combo, 1)
        layout.addWidget(edit_btn)
        return row

    @staticmethod
    def _labeled_widget(text: str, widget: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(text)
        label.setMinimumWidth(150)
        layout.addWidget(label)
        layout.addWidget(widget, 1)
        return row

    def _wire(self, *controls: QWidget) -> None:
        for control in controls:
            for sig_name in (
                "valueChanged",
                "toggled",
                "currentTextChanged",
                "currentIndexChanged",
                "textChanged",
            ):
                sig = getattr(control, sig_name, None)
                if sig is None:
                    continue
                try:
                    sig.connect(self._emit_changed)
                    break
                except Exception:
                    continue

    @staticmethod
    def _set_combo_text(combo: QComboBox, value: str) -> None:
        index = combo.findText(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _as_float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return float(fallback)

    @staticmethod
    def _as_int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except Exception:
            return int(fallback)

    def _build_analysis_section(self) -> None:
        defaults = AnalysisConfig()
        group = CollapsibleGroupBox("Audio Analysis", collapsed=True)
        self._analysis_group = group
        form = QVBoxLayout(group)
        form.setContentsMargins(8, 10, 8, 8)
        form.setSpacing(8)

        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 192000)
        self.sample_rate_spin.setSingleStep(1000)
        self.sample_rate_spin.setValue(defaults.sample_rate)

        self.fft_size_combo = QComboBox()
        for value in (1024, 2048, 4096):
            self.fft_size_combo.addItem(str(value), value)
        self._set_combo_text(self.fft_size_combo, str(defaults.fft_size))

        self.hop_size_spin = QSpinBox()
        self.hop_size_spin.setRange(128, 8192)
        self.hop_size_spin.setSingleStep(64)
        self.hop_size_spin.setValue(defaults.hop_size)

        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(256, 8192)
        self.window_size_spin.setSingleStep(64)
        self.window_size_spin.setValue(defaults.window_size)

        self.lowpass_enabled_chk = QCheckBox("Enable lowpass")
        self.lowpass_enabled_chk.setChecked(defaults.lowpass_enabled)

        self.lowpass_hz_spin = QDoubleSpinBox()
        self.lowpass_hz_spin.setRange(20.0, 24000.0)
        self.lowpass_hz_spin.setDecimals(1)
        self.lowpass_hz_spin.setSingleStep(10.0)
        self.lowpass_hz_spin.setValue(defaults.lowpass_hz)

        self.highpass_enabled_chk = QCheckBox("Enable highpass")
        self.highpass_enabled_chk.setChecked(defaults.highpass_enabled)

        self.highpass_hz_spin = QDoubleSpinBox()
        self.highpass_hz_spin.setRange(20.0, 24000.0)
        self.highpass_hz_spin.setDecimals(1)
        self.highpass_hz_spin.setSingleStep(10.0)
        self.highpass_hz_spin.setValue(defaults.highpass_hz)

        self.freq_min_hz_spin = QDoubleSpinBox()
        self.freq_min_hz_spin.setRange(20.0, 20000.0)
        self.freq_min_hz_spin.setDecimals(1)
        self.freq_min_hz_spin.setSingleStep(10.0)
        self.freq_min_hz_spin.setValue(defaults.freq_min_hz)

        self.freq_max_hz_spin = QDoubleSpinBox()
        self.freq_max_hz_spin.setRange(20.0, 24000.0)
        self.freq_max_hz_spin.setDecimals(1)
        self.freq_max_hz_spin.setSingleStep(10.0)
        self.freq_max_hz_spin.setValue(defaults.freq_max_hz)

        self.gain_slider = SliderWithLabel("Gain", 0.0, 12.0, defaults.gain, decimals=2, step=0.1)

        for widget in (
            self._labeled_widget("Sample Rate", self.sample_rate_spin),
            self._labeled_widget("FFT Size", self.fft_size_combo),
            self._labeled_widget("Hop Size", self.hop_size_spin),
            self._labeled_widget("Window Size", self.window_size_spin),
            self.lowpass_enabled_chk,
            self._labeled_widget("Lowpass Hz", self.lowpass_hz_spin),
            self.highpass_enabled_chk,
            self._labeled_widget("Highpass Hz", self.highpass_hz_spin),
            self._labeled_widget("Freq Min Hz", self.freq_min_hz_spin),
            self._labeled_widget("Freq Max Hz", self.freq_max_hz_spin),
            self.gain_slider,
        ):
            form.addWidget(widget)

        self._wire(
            self.sample_rate_spin,
            self.fft_size_combo,
            self.hop_size_spin,
            self.window_size_spin,
            self.lowpass_enabled_chk,
            self.lowpass_hz_spin,
            self.highpass_enabled_chk,
            self.highpass_hz_spin,
            self.freq_min_hz_spin,
            self.freq_max_hz_spin,
            self.gain_slider,
        )
        self._layout.addWidget(group)

    def _build_beat_section(self) -> None:
        defaults = BeatDetectionConfig()
        detector_defaults = defaults.multibus_config

        group = CollapsibleGroupBox("Beat Detection", collapsed=False)
        self._beat_group = group
        form = QVBoxLayout(group)
        form.setContentsMargins(8, 10, 8, 8)
        form.setSpacing(8)

        self.sensitivity_slider = SliderWithLabel("Sensitivity", 0.0, 1.0, defaults.sensitivity, decimals=2, step=0.01)

        self.refractory_spin = QDoubleSpinBox()
        self.refractory_spin.setRange(50.0, 500.0)
        self.refractory_spin.setDecimals(1)
        self.refractory_spin.setSingleStep(5.0)
        self.refractory_spin.setValue(defaults.refractory_ms)

        self.use_librosa_chk = QCheckBox("Use librosa detector")
        self.use_librosa_chk.setChecked(defaults.use_librosa)
        self.use_multibus_chk = QCheckBox("Use multi-bus detector")
        self.use_multibus_chk.setChecked(defaults.use_multibus)
        self.use_fft_chk = QCheckBox("Use FFT peak detector")
        self.use_fft_chk.setChecked(defaults.use_fft_peaks)
        self.plp_chk = QCheckBox("Enable PLP enhancement")
        self.plp_chk.setChecked(defaults.plp_enabled)

        self.peak_seek_slider = SliderWithLabel("Peak/Seek Ratio", 0.10, 5.00, defaults.peak_seek_ratio, decimals=2, step=0.01)
        self.peak_threshold_slider = SliderWithLabel("Peak Beat Threshold", 0.00, 1.00, defaults.peak_beat_threshold, decimals=2, step=0.01)

        self.w_flux_slider = SliderWithLabel("Bus Weight Flux", 0.00, 1.00, detector_defaults.w_flux, decimals=2, step=0.01)
        self.w_band_slider = SliderWithLabel("Bus Weight Band", 0.00, 1.00, detector_defaults.w_band, decimals=2, step=0.01)
        self.w_delta_slider = SliderWithLabel("Bus Weight Delta", 0.00, 1.00, detector_defaults.w_delta, decimals=2, step=0.01)
        self.w_phase_slider = SliderWithLabel("Bus Weight Phase", 0.00, 1.00, detector_defaults.w_phase, decimals=2, step=0.01)

        self.bus_arm_slider = SliderWithLabel("Bus Arm Threshold", 0.00, 1.00, detector_defaults.bus_arm_threshold, decimals=2, step=0.01)
        self.bus_release_slider = SliderWithLabel("Bus Release Threshold", 0.00, 1.00, detector_defaults.bus_release_threshold, decimals=2, step=0.01)

        self.bus_refractory_spin = QDoubleSpinBox()
        self.bus_refractory_spin.setRange(50.0, 500.0)
        self.bus_refractory_spin.setDecimals(1)
        self.bus_refractory_spin.setSingleStep(5.0)
        self.bus_refractory_spin.setValue(detector_defaults.bus_refractory_ms)

        self.transient_chk = QCheckBox("Enable transient classification")
        self.transient_chk.setChecked(detector_defaults.transient_classification_enabled)
        self.bass_weighting_chk = QCheckBox("Enable bass-dominance weighting")
        self.bass_weighting_chk.setChecked(detector_defaults.bass_dominance_weighting_enabled)

        for widget in (
            self.sensitivity_slider,
            self._labeled_widget("Refractory (ms)", self.refractory_spin),
            self.use_librosa_chk,
            self.use_multibus_chk,
            self.use_fft_chk,
            self.plp_chk,
            self.peak_seek_slider,
            self.peak_threshold_slider,
            self.w_flux_slider,
            self.w_band_slider,
            self.w_delta_slider,
            self.w_phase_slider,
            self.bus_arm_slider,
            self.bus_release_slider,
            self._labeled_widget("Bus Refractory (ms)", self.bus_refractory_spin),
            self.transient_chk,
            self.bass_weighting_chk,
        ):
            form.addWidget(widget)

        self._wire(
            self.sensitivity_slider,
            self.refractory_spin,
            self.use_librosa_chk,
            self.use_multibus_chk,
            self.use_fft_chk,
            self.plp_chk,
            self.peak_seek_slider,
            self.peak_threshold_slider,
            self.w_flux_slider,
            self.w_band_slider,
            self.w_delta_slider,
            self.w_phase_slider,
            self.bus_arm_slider,
            self.bus_release_slider,
            self.bus_refractory_spin,
            self.transient_chk,
            self.bass_weighting_chk,
        )
        self._layout.addWidget(group)

    def _build_mapping_section(self) -> None:
        defaults = MappingConfig()
        group = CollapsibleGroupBox("Pitch and Energy Mapping", collapsed=True)
        self._mapping_group = group
        form = QVBoxLayout(group)
        form.setContentsMargins(8, 10, 8, 8)
        form.setSpacing(8)

        self.pitch_range_slider = SliderWithLabel("Pitch Range", -200.0, 200.0, defaults.pitch_range, decimals=1, step=1.0)
        self.amplitude_centering_slider = SliderWithLabel("Amplitude Centering", -200.0, 200.0, defaults.amplitude_centering, decimals=1, step=1.0)
        self.center_offset_slider = SliderWithLabel("Center Offset", -300.0, 300.0, defaults.center_offset, decimals=1, step=1.0)

        self.overflow_mode_combo = QComboBox()
        self.overflow_mode_combo.addItems(["crop", "bounce", "fold"])
        self._set_combo_text(self.overflow_mode_combo, defaults.overflow_mode)

        self.energy_multiplier_slider = SliderWithLabel("Energy Multiplier", 0.0, 100.0, defaults.energy_multiplier, decimals=1, step=1.0)

        self.min_command_delay_spin = QDoubleSpinBox()
        self.min_command_delay_spin.setRange(50.0, 500.0)
        self.min_command_delay_spin.setDecimals(1)
        self.min_command_delay_spin.setSingleStep(5.0)
        self.min_command_delay_spin.setValue(defaults.min_command_delay_ms)

        self.mapping_points_per_second_spin = QSpinBox()
        self.mapping_points_per_second_spin.setRange(1, 100)
        self.mapping_points_per_second_spin.setValue(defaults.points_per_second)

        self.pos_min_spin = QSpinBox()
        self.pos_min_spin.setRange(0, 100)
        self.pos_min_spin.setValue(defaults.pos_min)

        self.pos_max_spin = QSpinBox()
        self.pos_max_spin.setRange(0, 100)
        self.pos_max_spin.setValue(defaults.pos_max)

        self.range_normalization_slider = SliderWithLabel("Range Normalization", 0.0, 1.0, defaults.range_normalization, decimals=2, step=0.05)

        self.blank_threshold_spin = QSpinBox()
        self.blank_threshold_spin.setRange(1, 30)
        self.blank_threshold_spin.setValue(5)
        self.blank_threshold_spin.setToolTip(
            "Position range (0-100) below which a region is considered 'blank' / no motion.\n"
            "Used by the Fill Blanks button."
        )

        for widget in (
            self.pitch_range_slider,
            self.amplitude_centering_slider,
            self.center_offset_slider,
            self._labeled_widget("Overflow Mode", self.overflow_mode_combo),
            self.energy_multiplier_slider,
            self._labeled_widget("Min Command Delay (ms)", self.min_command_delay_spin),
            self._labeled_widget("Points per Second", self.mapping_points_per_second_spin),
            self._labeled_widget("Position Min", self.pos_min_spin),
            self._labeled_widget("Position Max", self.pos_max_spin),
            self.range_normalization_slider,
            self._labeled_widget("Blank Threshold", self.blank_threshold_spin),
        ):
            form.addWidget(widget)

        self._wire(
            self.pitch_range_slider,
            self.amplitude_centering_slider,
            self.center_offset_slider,
            self.overflow_mode_combo,
            self.energy_multiplier_slider,
            self.min_command_delay_spin,
            self.mapping_points_per_second_spin,
            self.pos_min_spin,
            self.pos_max_spin,
            self.range_normalization_slider,
        )
        self._layout.addWidget(group)

    def _build_ml_section(self) -> None:
        defaults = MLConfig()
        group = CollapsibleGroupBox("ML Intelligence", collapsed=True)
        self._ml_group = group
        form = QVBoxLayout(group)
        form.setContentsMargins(8, 10, 8, 8)
        form.setSpacing(8)

        self.ml_enabled_chk = QCheckBox("Enable ML modulation")
        self.ml_enabled_chk.setChecked(defaults.enabled)

        self.ml_strength_slider = SliderWithLabel("ML Strength", 0.0, 1.0, defaults.strength, decimals=2, step=0.01)

        self.ml_cadence_mode_combo = QComboBox()
        self.ml_cadence_mode_combo.addItems(["auto", "fixed_1", "fixed_2", "fixed_4"])
        self._set_combo_text(self.ml_cadence_mode_combo, defaults.cadence_mode)

        self.ml_rule_fit_path_edit = QLineEdit(defaults.rule_fit_path)
        self.ml_teaching_rule_fit_path_edit = QLineEdit(defaults.teaching_rule_fit_path)

        self.ml_min_confidence_slider = SliderWithLabel("Min Confidence", 0.0, 1.0, defaults.min_confidence, decimals=2, step=0.01)

        self.ml_bidirectional_smooth_chk = QCheckBox("Enable bidirectional smoothing")
        self.ml_bidirectional_smooth_chk.setChecked(defaults.bidirectional_smooth)

        self.ml_smooth_alpha_slider = SliderWithLabel("Smooth Alpha", 0.0, 1.0, defaults.smooth_alpha, decimals=2, step=0.01)

        for widget in (
            self.ml_enabled_chk,
            self.ml_strength_slider,
            self._labeled_widget("Cadence Mode", self.ml_cadence_mode_combo),
            self._labeled_widget("Rule Fit Path", self.ml_rule_fit_path_edit),
            self._labeled_widget("Teaching Rule Fit Path", self.ml_teaching_rule_fit_path_edit),
            self.ml_min_confidence_slider,
            self.ml_bidirectional_smooth_chk,
            self.ml_smooth_alpha_slider,
        ):
            form.addWidget(widget)

        self._wire(
            self.ml_enabled_chk,
            self.ml_strength_slider,
            self.ml_cadence_mode_combo,
            self.ml_rule_fit_path_edit,
            self.ml_teaching_rule_fit_path_edit,
            self.ml_min_confidence_slider,
            self.ml_bidirectional_smooth_chk,
            self.ml_smooth_alpha_slider,
        )
        self._layout.addWidget(group)

    def _build_automap_section(self) -> None:
        defaults = AutomapConfig()
        group = CollapsibleGroupBox("Automap", collapsed=True)
        self._automap_group = group
        form = QVBoxLayout(group)
        form.setContentsMargins(8, 10, 8, 8)
        form.setSpacing(8)

        self.automap_enabled_chk = QCheckBox("Enable automap optimization")
        self.automap_enabled_chk.setChecked(defaults.enabled)

        self.automap_target_y_spin = QDoubleSpinBox()
        self.automap_target_y_spin.setRange(0.0, 100.0)
        self.automap_target_y_spin.setDecimals(1)
        self.automap_target_y_spin.setValue(defaults.target_y_position)

        self.automap_target_speed_spin = QDoubleSpinBox()
        self.automap_target_speed_spin.setRange(0.0, 400.0)
        self.automap_target_speed_spin.setDecimals(1)
        self.automap_target_speed_spin.setValue(defaults.target_speed)

        self.automap_target_speed_pct_spin = QDoubleSpinBox()
        self.automap_target_speed_pct_spin.setRange(0.0, 100.0)
        self.automap_target_speed_pct_spin.setDecimals(1)
        self.automap_target_speed_pct_spin.setValue(defaults.target_speed_pct)

        self.automap_mode_combo = QComboBox()
        self.automap_mode_combo.addItems(["cmean", "cmeanv2", "clen"])
        self._set_combo_text(self.automap_mode_combo, defaults.optimization_mode)

        self.automap_optimize_ml_chk = QCheckBox("Optimize ML strength")
        self.automap_optimize_ml_chk.setChecked(defaults.optimize_ml_strength)

        self.automap_max_iter_spin = QSpinBox()
        self.automap_max_iter_spin.setRange(20, 1000)
        self.automap_max_iter_spin.setValue(defaults.max_iter)

        for widget in (
            self.automap_enabled_chk,
            self._labeled_widget("Target Y Position", self.automap_target_y_spin),
            self._labeled_widget("Target Speed", self.automap_target_speed_spin),
            self._labeled_widget("Target Speed Percent", self.automap_target_speed_pct_spin),
            self._labeled_widget("Optimization Mode", self.automap_mode_combo),
            self.automap_optimize_ml_chk,
            self._labeled_widget("Max Iterations", self.automap_max_iter_spin),
        ):
            form.addWidget(widget)

        self._wire(
            self.automap_enabled_chk,
            self.automap_target_y_spin,
            self.automap_target_speed_spin,
            self.automap_target_speed_pct_spin,
            self.automap_mode_combo,
            self.automap_optimize_ml_chk,
            self.automap_max_iter_spin,
        )
        self._layout.addWidget(group)

    def _build_axis_section(self) -> None:
        defaults = AxisConfig()
        group = CollapsibleGroupBox("Multi Axis and Output", collapsed=True)
        self._axis_group = group
        form = QVBoxLayout(group)
        form.setContentsMargins(8, 10, 8, 8)
        form.setSpacing(8)

        self.axis_direction_flip_slider = SliderWithLabel(
            "Direction Flip Probability",
            0.0,
            1.0,
            defaults.direction_flip_probability,
            decimals=2,
            step=0.01,
        )

        self.axis_min_distance_slider = SliderWithLabel("Min Distance", 0.10, 0.50, defaults.min_distance, decimals=2, step=0.01)
        self.axis_speed_threshold_slider = SliderWithLabel("Speed Threshold Percent", 0.0, 100.0, defaults.speed_threshold_pct, decimals=1, step=1.0)

        self.axis_prostate_algo_combo = QComboBox()
        self.axis_prostate_algo_combo.addItems(["standard", "tear_shaped"])
        self._set_combo_text(self.axis_prostate_algo_combo, defaults.prostate_algorithm)

        self.axis_prostate_volume_slider = SliderWithLabel("Prostate Volume Mult", 1.0, 3.0, defaults.prostate_volume_mult, decimals=2, step=0.01)

        self._e_custom_points: dict[str, list[tuple[float, float]] | None] = {
            "e1": None, "e2": None, "e3": None, "e4": None,
        }
        _curve_items = list(PRESET_CURVES.keys()) + ["custom"]

        self.e1_curve_combo = QComboBox()
        self.e2_curve_combo = QComboBox()
        self.e3_curve_combo = QComboBox()
        self.e4_curve_combo = QComboBox()
        for combo, value in (
            (self.e1_curve_combo, defaults.e1_curve),
            (self.e2_curve_combo, defaults.e2_curve),
            (self.e3_curve_combo, defaults.e3_curve),
            (self.e4_curve_combo, defaults.e4_curve),
        ):
            combo.addItems(_curve_items)
            self._set_combo_text(combo, value)

        self._e_edit_buttons: dict[str, QPushButton] = {}
        for ename, combo in (("e1", self.e1_curve_combo), ("e2", self.e2_curve_combo),
                             ("e3", self.e3_curve_combo), ("e4", self.e4_curve_combo)):
            btn = QPushButton("Edit")
            btn.setFixedWidth(48)
            btn.clicked.connect(lambda _=False, e=ename, c=combo: self._open_curve_editor(e, c))
            self._e_edit_buttons[ename] = btn

        self.e1_phase_slider = SliderWithLabel("E1 Phase Shift", 0.0, 100.0, defaults.e_phase_shift.get("e1", 0.0), decimals=1, step=1.0)
        self.e2_phase_slider = SliderWithLabel("E2 Phase Shift", 0.0, 100.0, defaults.e_phase_shift.get("e2", 0.0), decimals=1, step=1.0)
        self.e3_phase_slider = SliderWithLabel("E3 Phase Shift", 0.0, 100.0, defaults.e_phase_shift.get("e3", 0.0), decimals=1, step=1.0)
        self.e4_phase_slider = SliderWithLabel("E4 Phase Shift", 0.0, 100.0, defaults.e_phase_shift.get("e4", 0.0), decimals=1, step=1.0)

        self.e_min_segment_spin = QDoubleSpinBox()
        self.e_min_segment_spin.setRange(0.1, 5.0)
        self.e_min_segment_spin.setDecimals(2)
        self.e_min_segment_spin.setSingleStep(0.1)
        self.e_min_segment_spin.setValue(defaults.e_min_segment_sec)

        self.frequency_ratio_slider = SliderWithLabel("Frequency Ramp Ratio", 1.0, 10.0, defaults.frequency_ramp_ratio, decimals=1, step=0.1)
        self.pulse_frequency_ratio_slider = SliderWithLabel("Pulse Frequency Ratio", 1.0, 10.0, defaults.pulse_frequency_ratio, decimals=1, step=0.1)

        _FREQ_MODES = ["Ratio", "Hz (Spectral)", "Speed", "Band Energy", "Hybrid", "Motion Envelope"]
        _BAND_NAMES = ["sub_bass", "low_mid", "mid", "high"]

        self._pulse_freq_mode_combo = QComboBox()
        self._pulse_freq_mode_combo.addItems(_FREQ_MODES)
        self._pulse_freq_mode_combo.setCurrentIndex(defaults.pulse_freq_mode)
        self._pulse_freq_mode_combo.setToolTip(
            "Ratio: legacy speed+alpha mix\n"
            "Hz: spectral centroid (audio-reactive)\n"
            "Speed: position derivative\n"
            "Band Energy: selected frequency band envelope\n"
            "Hybrid: ratio mix modulated by spectral centroid\n"
            "Motion Envelope: legacy frequency logic (ramp+speed blend)"
        )
        self._pulse_freq_band_combo = QComboBox()
        self._pulse_freq_band_combo.addItems(_BAND_NAMES)
        self._pulse_freq_band_combo.setCurrentText(defaults.pulse_freq_band)
        self._pulse_freq_weight_slider = SliderWithLabel("Pulse Freq Weight", 0.0, 1.0, defaults.pulse_freq_weight, decimals=2, step=0.01)

        self._carrier_freq_ratio_slider = SliderWithLabel("Carrier Frequency Ratio", 1.0, 10.0, defaults.carrier_frequency_ratio, decimals=1, step=0.1)
        self._carrier_freq_mode_combo = QComboBox()
        self._carrier_freq_mode_combo.addItems(_FREQ_MODES)
        self._carrier_freq_mode_combo.setCurrentIndex(defaults.carrier_freq_mode)
        self._carrier_freq_mode_combo.setToolTip(
            "Ratio: speed+alpha mix\n"
            "Hz: spectral centroid (audio-reactive)\n"
            "Speed: position derivative\n"
            "Band Energy: selected frequency band envelope\n"
            "Hybrid: ratio mix modulated by spectral centroid\n"
            "Motion Envelope: legacy frequency logic (ramp+speed blend)"
        )
        self._carrier_freq_band_combo = QComboBox()
        self._carrier_freq_band_combo.addItems(_BAND_NAMES)
        self._carrier_freq_band_combo.setCurrentText(defaults.carrier_freq_band)
        self._carrier_freq_weight_slider = SliderWithLabel("Carrier Freq Weight", 0.0, 1.0, defaults.carrier_freq_weight, decimals=2, step=0.01)

        # Visibility toggling: band combo visible only in BandEnergy mode
        def _update_pulse_freq_visibility(idx: int) -> None:
            self._pulse_freq_band_combo.setVisible(idx == 3)
            self.pulse_frequency_ratio_slider.setVisible(idx in (0, 4, 5))
        _update_pulse_freq_visibility(defaults.pulse_freq_mode)
        self._pulse_freq_mode_combo.currentIndexChanged.connect(_update_pulse_freq_visibility)

        def _update_carrier_freq_visibility(idx: int) -> None:
            self._carrier_freq_band_combo.setVisible(idx == 3)
            self._carrier_freq_ratio_slider.setVisible(idx in (0, 4, 5))
        _update_carrier_freq_visibility(defaults.carrier_freq_mode)
        self._carrier_freq_mode_combo.currentIndexChanged.connect(_update_carrier_freq_visibility)

        self.volume_ratio_slider = SliderWithLabel("Volume Ramp Ratio", 10.0, 40.0, defaults.volume_ramp_ratio, decimals=1, step=0.1)
        self.ramp_pct_per_hour_spin = QDoubleSpinBox()
        self.ramp_pct_per_hour_spin.setRange(0.0, 100.0)
        self.ramp_pct_per_hour_spin.setDecimals(1)
        self.ramp_pct_per_hour_spin.setSingleStep(1.0)
        self.ramp_pct_per_hour_spin.setValue(defaults.ramp_percent_per_hour)
        self.pulse_rise_ratio_slider = SliderWithLabel("Pulse Rise Ratio", 1.0, 10.0, defaults.pulse_rise_ratio, decimals=1, step=0.1)
        self.pulse_width_ratio_slider = SliderWithLabel("Pulse Width Ratio", 1.0, 10.0, defaults.pulse_width_ratio, decimals=1, step=0.1)

        self.rest_level_slider = SliderWithLabel("Rest Level", 0.0, 1.0, defaults.rest_level, decimals=2, step=0.01)

        self.ramp_up_spin = QDoubleSpinBox()
        self.ramp_up_spin.setRange(0.0, 10.0)
        self.ramp_up_spin.setDecimals(2)
        self.ramp_up_spin.setSingleStep(0.1)
        self.ramp_up_spin.setValue(defaults.ramp_up_duration_sec)

        self.axis_speed_window_spin = QDoubleSpinBox()
        self.axis_speed_window_spin.setRange(1.0, 20.0)
        self.axis_speed_window_spin.setDecimals(1)
        self.axis_speed_window_spin.setSingleStep(0.5)
        self.axis_speed_window_spin.setValue(defaults.speed_window_sec)

        self.axis_points_per_second_spin = QSpinBox()
        self.axis_points_per_second_spin.setRange(1, 100)
        self.axis_points_per_second_spin.setValue(defaults.points_per_second)

        self.axis_checkboxes: dict[str, QCheckBox] = {}
        # Individual checkboxes kept in dict for preset round-trip, but hidden
        for axis_name in (
            "main",
            "alpha",
            "beta",
            "alpha_prostate",
            "beta_prostate",
            "e1",
            "e2",
            "e3",
            "e4",
            "frequency",
            "pulse_frequency",
            "carrier_frequency",
            "volume",
            "pulse_rise",
            "pulse_width",
        ):
            chk = QCheckBox(f"Enable {axis_name}")
            chk.setChecked(axis_name in defaults.enabled_axes)
            self.axis_checkboxes[axis_name] = chk

        # -- Group toggles that drive multiple individual checkboxes --
        self._alpha_beta_toggle = QCheckBox("Enable Alpha/Beta")
        self._alpha_beta_toggle.setChecked(
            self.axis_checkboxes["alpha"].isChecked() or self.axis_checkboxes["beta"].isChecked()
        )
        self._alpha_beta_toggle.toggled.connect(self._on_alpha_beta_toggled)

        self._alpha_beta_mode_combo = QComboBox()
        self._alpha_beta_mode_combo.addItems(["restim", "orbital"])
        self._alpha_beta_mode_combo.setToolTip(
            "restim: semicircle arcs from main axis (fast)\n"
            "orbital: replay live StrokeMapper engine offline (slower, matches live motion)"
        )

        self._orbital_blend_slider = SliderWithLabel(
            "Orbital Blend", 0.0, 1.0, defaults.orbital_blend,
            decimals=2, step=0.01,
        )
        self._orbital_blend_slider.setToolTip(
            "0 = pure restim arcs, 1 = pure orbital motion, 0.5 = 50/50 blend"
        )

        self._e1234_toggle = QCheckBox("Enable E1\u2013E4")
        self._e1234_toggle.setChecked(
            any(self.axis_checkboxes[k].isChecked() for k in ("e1", "e2", "e3", "e4"))
        )
        self._e1234_toggle.toggled.connect(self._on_e1234_toggled)

        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["funscript", "csv"])

        for widget in (
            self.axis_direction_flip_slider,
            self.axis_min_distance_slider,
            self.axis_speed_threshold_slider,
            self._labeled_widget("Prostate Algorithm", self.axis_prostate_algo_combo),
            self.axis_prostate_volume_slider,
            self._curve_row("E1 Curve", self.e1_curve_combo, self._e_edit_buttons["e1"]),
            self._curve_row("E2 Curve", self.e2_curve_combo, self._e_edit_buttons["e2"]),
            self._curve_row("E3 Curve", self.e3_curve_combo, self._e_edit_buttons["e3"]),
            self._curve_row("E4 Curve", self.e4_curve_combo, self._e_edit_buttons["e4"]),
            self.e1_phase_slider,
            self.e2_phase_slider,
            self.e3_phase_slider,
            self.e4_phase_slider,
            self._labeled_widget("E Min Segment (s)", self.e_min_segment_spin),
            self.frequency_ratio_slider,
            self._labeled_widget("Pulse Freq Mode", self._pulse_freq_mode_combo),
            self.pulse_frequency_ratio_slider,
            self._labeled_widget("Pulse Freq Band", self._pulse_freq_band_combo),
            self._pulse_freq_weight_slider,
            self._labeled_widget("Carrier Freq Mode", self._carrier_freq_mode_combo),
            self._carrier_freq_ratio_slider,
            self._labeled_widget("Carrier Freq Band", self._carrier_freq_band_combo),
            self._carrier_freq_weight_slider,
            self.volume_ratio_slider,
            self._labeled_widget("Ramp %/Hour", self.ramp_pct_per_hour_spin),
            self.pulse_rise_ratio_slider,
            self.pulse_width_ratio_slider,
            self.rest_level_slider,
            self._labeled_widget("Ramp Up Duration (s)", self.ramp_up_spin),
            self._labeled_widget("Speed Window (s)", self.axis_speed_window_spin),
            self._labeled_widget("Axis Points per Second", self.axis_points_per_second_spin),
            self._labeled_widget("Output Format", self.output_format_combo),
        ):
            form.addWidget(widget)

        for axis_name in self.axis_checkboxes:
            if axis_name in ("alpha", "beta", "e1", "e2", "e3", "e4"):
                continue  # driven by group toggles
            form.addWidget(self.axis_checkboxes[axis_name])
        form.addWidget(self._alpha_beta_toggle)
        form.addWidget(self._labeled_widget("Alpha/Beta Mode", self._alpha_beta_mode_combo))
        form.addWidget(self._orbital_blend_slider)
        form.addWidget(self._e1234_toggle)

        self._wire(
            self.axis_direction_flip_slider,
            self.axis_min_distance_slider,
            self.axis_speed_threshold_slider,
            self.axis_prostate_algo_combo,
            self.axis_prostate_volume_slider,
            self.e1_curve_combo,
            self.e2_curve_combo,
            self.e3_curve_combo,
            self.e4_curve_combo,
            self.e1_phase_slider,
            self.e2_phase_slider,
            self.e3_phase_slider,
            self.e4_phase_slider,
            self.e_min_segment_spin,
            self.frequency_ratio_slider,
            self.pulse_frequency_ratio_slider,
            self._pulse_freq_mode_combo,
            self._pulse_freq_band_combo,
            self._pulse_freq_weight_slider,
            self._carrier_freq_ratio_slider,
            self._carrier_freq_mode_combo,
            self._carrier_freq_band_combo,
            self._carrier_freq_weight_slider,
            self.volume_ratio_slider,
            self.ramp_pct_per_hour_spin,
            self.pulse_rise_ratio_slider,
            self.pulse_width_ratio_slider,
            self.rest_level_slider,
            self.ramp_up_spin,
            self.axis_speed_window_spin,
            self.axis_points_per_second_spin,
            self.output_format_combo,
            *tuple(self.axis_checkboxes.values()),
            self._alpha_beta_toggle,
            self._alpha_beta_mode_combo,
            self._orbital_blend_slider,
            self._e1234_toggle,
        )
        self._layout.addWidget(group)

    def get_analysis_config(self) -> AnalysisConfig:
        return AnalysisConfig(
            sample_rate=int(self.sample_rate_spin.value()),
            fft_size=int(self.fft_size_combo.currentText()),
            hop_size=int(self.hop_size_spin.value()),
            window_size=int(self.window_size_spin.value()),
            lowpass_enabled=bool(self.lowpass_enabled_chk.isChecked()),
            lowpass_hz=float(self.lowpass_hz_spin.value()),
            highpass_enabled=bool(self.highpass_enabled_chk.isChecked()),
            highpass_hz=float(self.highpass_hz_spin.value()),
            freq_min_hz=float(self.freq_min_hz_spin.value()),
            freq_max_hz=float(self.freq_max_hz_spin.value()),
            gain=float(self.gain_slider.value()),
        )

    def get_beat_config(self) -> BeatDetectionConfig:
        detector_cfg = EventDetectorConfig(
            w_flux=float(self.w_flux_slider.value()),
            w_band=float(self.w_band_slider.value()),
            w_delta=float(self.w_delta_slider.value()),
            w_phase=float(self.w_phase_slider.value()),
            bus_arm_threshold=float(self.bus_arm_slider.value()),
            bus_release_threshold=float(self.bus_release_slider.value()),
            bus_refractory_ms=float(self.bus_refractory_spin.value()),
            transient_classification_enabled=bool(self.transient_chk.isChecked()),
            bass_dominance_weighting_enabled=bool(self.bass_weighting_chk.isChecked()),
        )
        return BeatDetectionConfig(
            sensitivity=float(self.sensitivity_slider.value()),
            refractory_ms=float(self.refractory_spin.value()),
            use_librosa=bool(self.use_librosa_chk.isChecked()),
            use_multibus=bool(self.use_multibus_chk.isChecked()),
            use_fft_peaks=bool(self.use_fft_chk.isChecked()),
            plp_enabled=bool(self.plp_chk.isChecked()),
            peak_seek_ratio=float(self.peak_seek_slider.value()),
            peak_beat_threshold=float(self.peak_threshold_slider.value()),
            multibus_config=detector_cfg,
        )

    def get_ml_config(self) -> MLConfig:
        return MLConfig(
            enabled=bool(self.ml_enabled_chk.isChecked()),
            strength=float(self.ml_strength_slider.value()),
            cadence_mode=str(self.ml_cadence_mode_combo.currentText()),
            rule_fit_path=str(self.ml_rule_fit_path_edit.text()).strip(),
            teaching_rule_fit_path=str(self.ml_teaching_rule_fit_path_edit.text()).strip(),
            min_confidence=float(self.ml_min_confidence_slider.value()),
            bidirectional_smooth=bool(self.ml_bidirectional_smooth_chk.isChecked()),
            smooth_alpha=float(self.ml_smooth_alpha_slider.value()),
        )

    def get_mapping_config(self) -> MappingConfig:
        pos_min = int(self.pos_min_spin.value())
        pos_max = int(self.pos_max_spin.value())
        if pos_min > pos_max:
            pos_min, pos_max = pos_max, pos_min
        return MappingConfig(
            pitch_range=float(self.pitch_range_slider.value()),
            amplitude_centering=float(self.amplitude_centering_slider.value()),
            center_offset=float(self.center_offset_slider.value()),
            overflow_mode=str(self.overflow_mode_combo.currentText()),
            energy_multiplier=float(self.energy_multiplier_slider.value()),
            ml_config=self.get_ml_config(),
            min_command_delay_ms=float(self.min_command_delay_spin.value()),
            points_per_second=int(self.mapping_points_per_second_spin.value()),
            pos_min=pos_min,
            pos_max=pos_max,
            range_normalization=float(self.range_normalization_slider.value()),
        )

    def get_axis_config(self) -> AxisConfig:
        enabled_axes = {name for name, chk in self.axis_checkboxes.items() if chk.isChecked()}
        if not enabled_axes:
            enabled_axes = {"main"}

        return AxisConfig(
            direction_flip_probability=float(self.axis_direction_flip_slider.value()),
            min_distance=float(self.axis_min_distance_slider.value()),
            speed_threshold_pct=float(self.axis_speed_threshold_slider.value()),
            prostate_algorithm=str(self.axis_prostate_algo_combo.currentText()),
            prostate_volume_mult=float(self.axis_prostate_volume_slider.value()),
            e1_curve=str(self.e1_curve_combo.currentText()),
            e2_curve=str(self.e2_curve_combo.currentText()),
            e3_curve=str(self.e3_curve_combo.currentText()),
            e4_curve=str(self.e4_curve_combo.currentText()),
            e_custom_points={
                k: list(v) for k, v in self._e_custom_points.items() if v is not None
            },
            e_phase_shift={
                "e1": float(self.e1_phase_slider.value()),
                "e2": float(self.e2_phase_slider.value()),
                "e3": float(self.e3_phase_slider.value()),
                "e4": float(self.e4_phase_slider.value()),
            },
            e_min_segment_sec=float(self.e_min_segment_spin.value()),
            frequency_ramp_ratio=float(self.frequency_ratio_slider.value()),
            pulse_frequency_ratio=float(self.pulse_frequency_ratio_slider.value()),
            pulse_freq_mode=int(self._pulse_freq_mode_combo.currentIndex()),
            pulse_freq_band=str(self._pulse_freq_band_combo.currentText()),
            pulse_freq_weight=float(self._pulse_freq_weight_slider.value()),
            carrier_frequency_ratio=float(self._carrier_freq_ratio_slider.value()),
            carrier_freq_mode=int(self._carrier_freq_mode_combo.currentIndex()),
            carrier_freq_band=str(self._carrier_freq_band_combo.currentText()),
            carrier_freq_weight=float(self._carrier_freq_weight_slider.value()),
            volume_ramp_ratio=float(self.volume_ratio_slider.value()),
            ramp_percent_per_hour=float(self.ramp_pct_per_hour_spin.value()),
            pulse_rise_ratio=float(self.pulse_rise_ratio_slider.value()),
            pulse_width_ratio=float(self.pulse_width_ratio_slider.value()),
            rest_level=float(self.rest_level_slider.value()),
            ramp_up_duration_sec=float(self.ramp_up_spin.value()),
            speed_window_sec=float(self.axis_speed_window_spin.value()),
            points_per_second=int(self.axis_points_per_second_spin.value()),
            enabled_axes=enabled_axes,
            alpha_beta_mode=str(self._alpha_beta_mode_combo.currentText()),
            orbital_blend=float(self._orbital_blend_slider.value()),
        )

    # -- group toggle handlers -------------------------------------------------

    def _open_curve_editor(self, electrode: str, combo: QComboBox) -> None:
        current = combo.currentText()
        custom = self._e_custom_points.get(electrode)
        dlg = CurveEditorDialog(electrode, current, custom, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            curve = dlg.result_curve()
            points = dlg.result_points()
            if curve is not None:
                self._e_custom_points[electrode] = points
                self._set_combo_text(combo, curve)
                self._emit_changed()

    def _on_alpha_beta_toggled(self, checked: bool) -> None:
        for k in ("alpha", "beta"):
            self.axis_checkboxes[k].setChecked(checked)

    def _on_e1234_toggled(self, checked: bool) -> None:
        for k in ("e1", "e2", "e3", "e4"):
            self.axis_checkboxes[k].setChecked(checked)

    def get_automap_config(self) -> AutomapConfig:
        return AutomapConfig(
            enabled=bool(self.automap_enabled_chk.isChecked()),
            target_y_position=float(self.automap_target_y_spin.value()),
            target_speed=float(self.automap_target_speed_spin.value()),
            target_speed_pct=float(self.automap_target_speed_pct_spin.value()),
            optimization_mode=str(self.automap_mode_combo.currentText()),
            optimize_ml_strength=bool(self.automap_optimize_ml_chk.isChecked()),
            max_iter=int(self.automap_max_iter_spin.value()),
        )

    def to_preset(self) -> dict[str, Any]:
        analysis_cfg = self.get_analysis_config()
        beat_cfg = self.get_beat_config()
        mapping_cfg = self.get_mapping_config()
        ml_cfg = mapping_cfg.ml_config
        axis_cfg = self.get_axis_config()
        automap_cfg = self.get_automap_config()

        mapping_dict = asdict(mapping_cfg)
        mapping_dict.pop("ml_config", None)

        return {
            "analysis": asdict(analysis_cfg),
            "beat_detection": {
                **asdict(beat_cfg),
                "multibus_config": asdict(beat_cfg.multibus_config),
            },
            "mapping": mapping_dict,
            "ml": asdict(ml_cfg),
            "automap": asdict(automap_cfg),
            "axis": {
                **asdict(axis_cfg),
                "enabled_axes": sorted(axis_cfg.enabled_axes),
            },
            "output": {
                "enabled_axes": sorted(axis_cfg.enabled_axes),
                "format": str(self.output_format_combo.currentText()),
            },
        }

    def set_from_preset(self, preset: dict[str, Any]) -> None:
        if not isinstance(preset, dict):
            return

        analysis = preset.get("analysis", {})
        if isinstance(analysis, dict):
            self.sample_rate_spin.setValue(self._as_int(analysis.get("sample_rate"), self.sample_rate_spin.value()))
            self._set_combo_text(self.fft_size_combo, str(analysis.get("fft_size", self.fft_size_combo.currentText())))
            self.hop_size_spin.setValue(self._as_int(analysis.get("hop_size"), self.hop_size_spin.value()))
            self.window_size_spin.setValue(self._as_int(analysis.get("window_size"), self.window_size_spin.value()))
            self.lowpass_enabled_chk.setChecked(bool(analysis.get("lowpass_enabled", self.lowpass_enabled_chk.isChecked())))
            self.lowpass_hz_spin.setValue(self._as_float(analysis.get("lowpass_hz"), self.lowpass_hz_spin.value()))
            self.highpass_enabled_chk.setChecked(bool(analysis.get("highpass_enabled", self.highpass_enabled_chk.isChecked())))
            self.highpass_hz_spin.setValue(self._as_float(analysis.get("highpass_hz"), self.highpass_hz_spin.value()))
            self.freq_min_hz_spin.setValue(self._as_float(analysis.get("freq_min_hz"), self.freq_min_hz_spin.value()))
            self.freq_max_hz_spin.setValue(self._as_float(analysis.get("freq_max_hz"), self.freq_max_hz_spin.value()))
            self.gain_slider.setValue(self._as_float(analysis.get("gain"), self.gain_slider.value()))

        beat = preset.get("beat_detection", {})
        if isinstance(beat, dict):
            self.sensitivity_slider.setValue(self._as_float(beat.get("sensitivity"), self.sensitivity_slider.value()))
            self.refractory_spin.setValue(self._as_float(beat.get("refractory_ms"), self.refractory_spin.value()))
            self.use_librosa_chk.setChecked(bool(beat.get("use_librosa", self.use_librosa_chk.isChecked())))
            self.use_multibus_chk.setChecked(bool(beat.get("use_multibus", self.use_multibus_chk.isChecked())))
            self.use_fft_chk.setChecked(bool(beat.get("use_fft_peaks", self.use_fft_chk.isChecked())))
            self.plp_chk.setChecked(bool(beat.get("plp_enabled", self.plp_chk.isChecked())))
            self.peak_seek_slider.setValue(self._as_float(beat.get("peak_seek_ratio"), self.peak_seek_slider.value()))
            self.peak_threshold_slider.setValue(self._as_float(beat.get("peak_beat_threshold"), self.peak_threshold_slider.value()))

            multibus = beat.get("multibus_config", {})
            if isinstance(multibus, dict):
                self.w_flux_slider.setValue(self._as_float(multibus.get("w_flux"), self.w_flux_slider.value()))
                self.w_band_slider.setValue(self._as_float(multibus.get("w_band"), self.w_band_slider.value()))
                self.w_delta_slider.setValue(self._as_float(multibus.get("w_delta"), self.w_delta_slider.value()))
                self.w_phase_slider.setValue(self._as_float(multibus.get("w_phase"), self.w_phase_slider.value()))
                self.bus_arm_slider.setValue(self._as_float(multibus.get("bus_arm_threshold"), self.bus_arm_slider.value()))
                self.bus_release_slider.setValue(self._as_float(multibus.get("bus_release_threshold"), self.bus_release_slider.value()))
                self.bus_refractory_spin.setValue(self._as_float(multibus.get("bus_refractory_ms"), self.bus_refractory_spin.value()))
                self.transient_chk.setChecked(bool(multibus.get("transient_classification_enabled", self.transient_chk.isChecked())))
                self.bass_weighting_chk.setChecked(bool(multibus.get("bass_dominance_weighting_enabled", self.bass_weighting_chk.isChecked())))

        mapping = preset.get("mapping", {})
        if isinstance(mapping, dict):
            self.pitch_range_slider.setValue(self._as_float(mapping.get("pitch_range"), self.pitch_range_slider.value()))
            self.amplitude_centering_slider.setValue(self._as_float(mapping.get("amplitude_centering"), self.amplitude_centering_slider.value()))
            self.center_offset_slider.setValue(self._as_float(mapping.get("center_offset"), self.center_offset_slider.value()))
            self._set_combo_text(self.overflow_mode_combo, str(mapping.get("overflow_mode", self.overflow_mode_combo.currentText())))
            self.energy_multiplier_slider.setValue(self._as_float(mapping.get("energy_multiplier"), self.energy_multiplier_slider.value()))
            self.min_command_delay_spin.setValue(self._as_float(mapping.get("min_command_delay_ms"), self.min_command_delay_spin.value()))
            self.mapping_points_per_second_spin.setValue(self._as_int(mapping.get("points_per_second"), self.mapping_points_per_second_spin.value()))
            self.pos_min_spin.setValue(self._as_int(mapping.get("pos_min"), self.pos_min_spin.value()))
            self.pos_max_spin.setValue(self._as_int(mapping.get("pos_max"), self.pos_max_spin.value()))
            self.range_normalization_slider.setValue(self._as_float(mapping.get("range_normalization"), self.range_normalization_slider.value()))

        ml = preset.get("ml", {})
        if isinstance(ml, dict):
            self.ml_enabled_chk.setChecked(bool(ml.get("enabled", self.ml_enabled_chk.isChecked())))
            self.ml_strength_slider.setValue(self._as_float(ml.get("strength"), self.ml_strength_slider.value()))
            self._set_combo_text(self.ml_cadence_mode_combo, str(ml.get("cadence_mode", self.ml_cadence_mode_combo.currentText())))
            self.ml_rule_fit_path_edit.setText(str(ml.get("rule_fit_path", self.ml_rule_fit_path_edit.text())))
            self.ml_teaching_rule_fit_path_edit.setText(str(ml.get("teaching_rule_fit_path", self.ml_teaching_rule_fit_path_edit.text())))
            self.ml_min_confidence_slider.setValue(self._as_float(ml.get("min_confidence"), self.ml_min_confidence_slider.value()))
            self.ml_bidirectional_smooth_chk.setChecked(bool(ml.get("bidirectional_smooth", self.ml_bidirectional_smooth_chk.isChecked())))
            self.ml_smooth_alpha_slider.setValue(self._as_float(ml.get("smooth_alpha"), self.ml_smooth_alpha_slider.value()))

        automap = preset.get("automap", {})
        if isinstance(automap, dict):
            self.automap_enabled_chk.setChecked(bool(automap.get("enabled", self.automap_enabled_chk.isChecked())))
            self.automap_target_y_spin.setValue(self._as_float(automap.get("target_y_position"), self.automap_target_y_spin.value()))
            self.automap_target_speed_spin.setValue(self._as_float(automap.get("target_speed"), self.automap_target_speed_spin.value()))
            self.automap_target_speed_pct_spin.setValue(self._as_float(automap.get("target_speed_pct"), self.automap_target_speed_pct_spin.value()))
            self._set_combo_text(self.automap_mode_combo, str(automap.get("optimization_mode", self.automap_mode_combo.currentText())))
            self.automap_optimize_ml_chk.setChecked(bool(automap.get("optimize_ml_strength", self.automap_optimize_ml_chk.isChecked())))
            self.automap_max_iter_spin.setValue(self._as_int(automap.get("max_iter"), self.automap_max_iter_spin.value()))

        axis = preset.get("axis", {})
        if isinstance(axis, dict):
            self.axis_direction_flip_slider.setValue(self._as_float(axis.get("direction_flip_probability"), self.axis_direction_flip_slider.value()))
            self.axis_min_distance_slider.setValue(self._as_float(axis.get("min_distance"), self.axis_min_distance_slider.value()))
            self.axis_speed_threshold_slider.setValue(self._as_float(axis.get("speed_threshold_pct"), self.axis_speed_threshold_slider.value()))
            self._set_combo_text(self.axis_prostate_algo_combo, str(axis.get("prostate_algorithm", self.axis_prostate_algo_combo.currentText())))
            self.axis_prostate_volume_slider.setValue(self._as_float(axis.get("prostate_volume_mult"), self.axis_prostate_volume_slider.value()))
            self._set_combo_text(self.e1_curve_combo, str(axis.get("e1_curve", self.e1_curve_combo.currentText())))
            self._set_combo_text(self.e2_curve_combo, str(axis.get("e2_curve", self.e2_curve_combo.currentText())))
            self._set_combo_text(self.e3_curve_combo, str(axis.get("e3_curve", self.e3_curve_combo.currentText())))
            self._set_combo_text(self.e4_curve_combo, str(axis.get("e4_curve", self.e4_curve_combo.currentText())))

            saved_cp = axis.get("e_custom_points", {})
            if isinstance(saved_cp, dict):
                for ename in ("e1", "e2", "e3", "e4"):
                    pts = saved_cp.get(ename)
                    if isinstance(pts, list) and len(pts) >= 2:
                        self._e_custom_points[ename] = [(float(p[0]), float(p[1])) for p in pts]
                    else:
                        self._e_custom_points[ename] = None

            phase = axis.get("e_phase_shift", {})
            if isinstance(phase, dict):
                self.e1_phase_slider.setValue(self._as_float(phase.get("e1"), self.e1_phase_slider.value()))
                self.e2_phase_slider.setValue(self._as_float(phase.get("e2"), self.e2_phase_slider.value()))
                self.e3_phase_slider.setValue(self._as_float(phase.get("e3"), self.e3_phase_slider.value()))
                self.e4_phase_slider.setValue(self._as_float(phase.get("e4"), self.e4_phase_slider.value()))

            self.e_min_segment_spin.setValue(self._as_float(axis.get("e_min_segment_sec"), self.e_min_segment_spin.value()))
            self.frequency_ratio_slider.setValue(self._as_float(axis.get("frequency_ramp_ratio"), self.frequency_ratio_slider.value()))
            self.pulse_frequency_ratio_slider.setValue(self._as_float(axis.get("pulse_frequency_ratio"), self.pulse_frequency_ratio_slider.value()))
            self._pulse_freq_mode_combo.setCurrentIndex(self._as_int(axis.get("pulse_freq_mode"), self._pulse_freq_mode_combo.currentIndex()))
            self._set_combo_text(self._pulse_freq_band_combo, str(axis.get("pulse_freq_band", self._pulse_freq_band_combo.currentText())))
            self._pulse_freq_weight_slider.setValue(self._as_float(axis.get("pulse_freq_weight"), self._pulse_freq_weight_slider.value()))
            self._carrier_freq_ratio_slider.setValue(self._as_float(axis.get("carrier_frequency_ratio"), self._carrier_freq_ratio_slider.value()))
            self._carrier_freq_mode_combo.setCurrentIndex(self._as_int(axis.get("carrier_freq_mode"), self._carrier_freq_mode_combo.currentIndex()))
            self._set_combo_text(self._carrier_freq_band_combo, str(axis.get("carrier_freq_band", self._carrier_freq_band_combo.currentText())))
            self._carrier_freq_weight_slider.setValue(self._as_float(axis.get("carrier_freq_weight"), self._carrier_freq_weight_slider.value()))
            self.volume_ratio_slider.setValue(self._as_float(axis.get("volume_ramp_ratio"), self.volume_ratio_slider.value()))
            self.ramp_pct_per_hour_spin.setValue(self._as_float(axis.get("ramp_percent_per_hour"), self.ramp_pct_per_hour_spin.value()))
            self.pulse_rise_ratio_slider.setValue(self._as_float(axis.get("pulse_rise_ratio"), self.pulse_rise_ratio_slider.value()))
            self.pulse_width_ratio_slider.setValue(self._as_float(axis.get("pulse_width_ratio"), self.pulse_width_ratio_slider.value()))
            self.rest_level_slider.setValue(self._as_float(axis.get("rest_level"), self.rest_level_slider.value()))
            self.ramp_up_spin.setValue(self._as_float(axis.get("ramp_up_duration_sec"), self.ramp_up_spin.value()))
            self.axis_speed_window_spin.setValue(self._as_float(axis.get("speed_window_sec"), self.axis_speed_window_spin.value()))
            self.axis_points_per_second_spin.setValue(self._as_int(axis.get("points_per_second"), self.axis_points_per_second_spin.value()))

            ab_mode = axis.get("alpha_beta_mode", self._alpha_beta_mode_combo.currentText())
            self._set_combo_text(self._alpha_beta_mode_combo, str(ab_mode))
            self._orbital_blend_slider.setValue(self._as_float(axis.get("orbital_blend"), self._orbital_blend_slider.value()))

            enabled_axes = axis.get("enabled_axes", [])
            if isinstance(enabled_axes, (list, tuple, set)):
                enabled_set = {str(v) for v in enabled_axes}
                for axis_name, chk in self.axis_checkboxes.items():
                    chk.setChecked(axis_name in enabled_set)
                self._alpha_beta_toggle.setChecked(
                    self.axis_checkboxes["alpha"].isChecked() or self.axis_checkboxes["beta"].isChecked()
                )
                self._e1234_toggle.setChecked(
                    any(self.axis_checkboxes[k].isChecked() for k in ("e1", "e2", "e3", "e4"))
                )

        output = preset.get("output", {})
        if isinstance(output, dict):
            self._set_combo_text(self.output_format_combo, str(output.get("format", self.output_format_combo.currentText())))

        self.config_changed.emit()

    def import_funscript_tools_config(self, path: str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Invalid funscript-tools config payload")

        mapped = {
            "min_distance": payload.get("min_distance_from_center", self.axis_min_distance_slider.value()),
            "speed_threshold_pct": payload.get("speed_threshold_percent", self.axis_speed_threshold_slider.value()),
            "prostate_algorithm": payload.get("prostate_algorithm", self.axis_prostate_algo_combo.currentText()),
            "prostate_volume_mult": payload.get("prostate_volume_multiplier", self.axis_prostate_volume_slider.value()),
            "frequency_ramp_ratio": payload.get("frequency_ramp_combine_ratio", self.frequency_ratio_slider.value()),
            "pulse_frequency_ratio": payload.get("pulse_frequency_combine_ratio", self.pulse_frequency_ratio_slider.value()),
            "volume_ramp_ratio": payload.get("volume_ramp_combine_ratio", self.volume_ratio_slider.value()),
            "rest_level": payload.get("rest_level", self.rest_level_slider.value()),
            "ramp_up_duration_sec": payload.get("ramp_up_duration_after_rest", self.ramp_up_spin.value()),
        }
        self.set_from_preset({"axis": mapped})

    def export_funscript_tools_config(self, path: str) -> None:
        axis = self.get_axis_config()
        payload = {
            "min_distance_from_center": axis.min_distance,
            "speed_threshold_percent": axis.speed_threshold_pct,
            "prostate_algorithm": axis.prostate_algorithm,
            "prostate_volume_multiplier": axis.prostate_volume_mult,
            "frequency_ramp_combine_ratio": axis.frequency_ramp_ratio,
            "pulse_frequency_combine_ratio": axis.pulse_frequency_ratio,
            "volume_ramp_combine_ratio": axis.volume_ramp_ratio,
            "rest_level": axis.rest_level,
            "ramp_up_duration_after_rest": axis.ramp_up_duration_sec,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
