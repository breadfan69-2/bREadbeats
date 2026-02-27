"""
dialog_builders  –  Dialog / pop-out builder functions extracted from BREadbeatsWindow.

Every public function here takes *win* (the BREadbeatsWindow instance)
as its first parameter, replacing the former *self*.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from functools import partial

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget,
    QGroupBox, QLabel, QSlider, QComboBox, QPushButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QTabWidget, QScrollArea,
    QSizePolicy, QMessageBox, QInputDialog,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

from config import BEAT_RANGE_LIMITS, BeatDetectionType, StrokeMode
from widgets import SliderWithLabel, RangeSliderWithLabel, CollapsibleGroupBox, NoWheelScrollArea
from stylesheet import get_thin_scrollbar_style
from logging_utils import get_log_level, log_event, set_log_level
from version import __version__

def on_options_audio_device(win):
    """Show Audio Device selection dialog"""
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout

    dialog = QDialog(win)
    dialog.setWindowTitle("Audio Device")
    dialog.setMinimumWidth(400)
    layout = QVBoxLayout(dialog)

    layout.addWidget(QLabel("Select Audio Device:"))

    # Create a combo box mirroring the main device_combo
    device_combo = QComboBox()
    device_combo.setMinimumWidth(350)

    # Copy items from main combo
    for i in range(win.device_combo.count()):
        device_combo.addItem(win.device_combo.itemText(i))
    device_combo.setCurrentIndex(win.device_combo.currentIndex())
    layout.addWidget(device_combo)

    # Quick preset buttons
    preset_row = QHBoxLayout()
    mic_btn = QPushButton("🎤 Mic (Reactive)")
    mic_btn.clicked.connect(lambda: win._dialog_set_device_mic(device_combo))
    preset_row.addWidget(mic_btn)

    loopback_btn = QPushButton("🔊 System Audio")
    loopback_btn.clicked.connect(lambda: win._dialog_set_device_loopback(device_combo))
    preset_row.addWidget(loopback_btn)
    preset_row.addStretch()
    layout.addLayout(preset_row)

    # OK/Cancel buttons
    btn_row = QHBoxLayout()
    ok_btn = QPushButton("OK")
    ok_btn.clicked.connect(dialog.accept)
    cancel_btn = QPushButton("Cancel")
    cancel_btn.clicked.connect(dialog.reject)
    btn_row.addStretch()
    btn_row.addWidget(ok_btn)
    btn_row.addWidget(cancel_btn)
    layout.addLayout(btn_row)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        # Apply the selected device
        win.device_combo.setCurrentIndex(device_combo.currentIndex())


def dialog_set_device_mic(win, combo: QComboBox):
    """Set mic device in dialog combo"""
    for i in range(combo.count()):
        text = combo.itemText(i).lower()
        if 'microphone' in text or 'mic' in text or 'input' in text:
            if 'loopback' not in text and 'stereo mix' not in text:
                combo.setCurrentIndex(i)
                return


def dialog_set_device_loopback(win, combo: QComboBox):
    """Set loopback/system audio device in dialog combo"""
    for i in range(combo.count()):
        text = combo.itemText(i).lower()
        if 'blackhole' in text or 'loopback' in text or 'stereo mix' in text or 'wasapi' in text:
            combo.setCurrentIndex(i)
            return
    # Fallback to speakers
    for i in range(combo.count()):
        text = combo.itemText(i).lower()
        if 'speakers' in text or 'headphone' in text:
            combo.setCurrentIndex(i)
            return


def on_options_connection(win):
    """Show Connection settings dialog"""
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QSpinBox, QPushButton, QHBoxLayout, QGridLayout

    dialog = QDialog(win)
    dialog.setWindowTitle("TCP Connection")
    dialog.setMinimumWidth(300)
    layout = QVBoxLayout(dialog)

    # Host/Port grid
    grid = QGridLayout()
    grid.addWidget(QLabel("Host:"), 0, 0)
    host_edit = QLineEdit(win.host_edit.text())
    grid.addWidget(host_edit, 0, 1)

    grid.addWidget(QLabel("Port:"), 1, 0)
    port_spin = QSpinBox()
    port_spin.setRange(1, 65535)
    port_spin.setValue(win.port_spin.value())
    port_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
    grid.addWidget(port_spin, 1, 1)
    layout.addLayout(grid)

    # OK/Cancel buttons
    btn_row = QHBoxLayout()
    ok_btn = QPushButton("OK")
    ok_btn.clicked.connect(dialog.accept)
    cancel_btn = QPushButton("Cancel")
    cancel_btn.clicked.connect(dialog.reject)
    btn_row.addStretch()
    btn_row.addWidget(ok_btn)
    btn_row.addWidget(cancel_btn)
    layout.addLayout(btn_row)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        # Apply the settings
        win.host_edit.setText(host_edit.text())
        win.port_spin.setValue(port_spin.value())
        # Reconnect if already connected
        if hasattr(win, 'network_engine') and win.network_engine:
            win._on_connect()


def open_developer_controls_window(win, tab_index: int = 0, scroll_to_flux: bool = False) -> None:
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTabWidget

    dialog = getattr(win, '_developer_controls_dialog', None)
    if dialog is not None:
        try:
            dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            tab_widget = getattr(win, '_developer_controls_tab_widget', None)
            if tab_widget is not None:
                tab_widget.setCurrentIndex(max(0, min(int(tab_index), tab_widget.count() - 1)))
            if scroll_to_flux and int(tab_index) == 1:
                win._scroll_advanced_controls_to_flux()
            if not bool(getattr(win, '_developer_controls_unlocked', False)):
                dialog.setEnabled(False)
                win._show_developer_controls_unlock_popup()
            return
        except RuntimeError:
            win._developer_controls_dialog = None
            win._developer_controls_tab_widget = None

    dialog = QDialog(win)
    dialog.setWindowTitle("Developer Controls")
    dialog.setMinimumWidth(620)
    dialog.setMinimumHeight(560)
    dialog.setModal(False)
    dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    _developer_dialog_cleared = False

    def _clear_developer_dialog_state() -> None:
        nonlocal _developer_dialog_cleared
        if _developer_dialog_cleared:
            return
        _developer_dialog_cleared = True
        win._developer_controls_dialog = None
        win._developer_controls_tab_widget = None
        win._tempo_tracking_popout_content = None
        win._trigger_settings_tab_content = None
        win._auto_fill_tab_content = None
        win._advanced_flux_threshold_slider = None
        win._advanced_flux_scaling_slider = None
        win._auto_fill_controls_widgets = {}
        unlock_dialog_ref = getattr(win, '_developer_unlock_dialog', None)
        if unlock_dialog_ref is not None:
            try:
                unlock_dialog_ref.close()
            except RuntimeError:
                pass
            win._developer_unlock_dialog = None

    dialog.rejected.connect(_clear_developer_dialog_state)
    dialog.finished.connect(lambda _result: _clear_developer_dialog_state())
    dialog.destroyed.connect(lambda *_: _clear_developer_dialog_state())

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(8, 8, 8, 8)

    tab_widget = QTabWidget()
    win._developer_controls_tab_widget = tab_widget

    tempo_content = getattr(win, '_tempo_tracking_popout_content', None)
    if tempo_content is None:
        tempo_content = win._create_tempo_tracking_tab(include_advanced_controls=True, advanced_locked=True)
        win._tempo_tracking_popout_content = tempo_content
        win._apply_config_to_ui()

    trigger_content = getattr(win, '_trigger_settings_tab_content', None)
    if trigger_content is None:
        trigger_content = win._on_advanced_controls(as_tab=True)
        win._trigger_settings_tab_content = trigger_content

    auto_fill_content = getattr(win, '_auto_fill_tab_content', None)
    if auto_fill_content is None:
        auto_fill_content = win._on_options_auto_fill_adaptation(as_tab=True)
        win._auto_fill_tab_content = auto_fill_content

    tab_widget.addTab(tempo_content, "Tempo Tracking")
    tab_widget.addTab(trigger_content, "Trigger Settings")
    tab_widget.addTab(auto_fill_content, "Auto Fill %")
    tab_widget.setCurrentIndex(max(0, min(int(tab_index), tab_widget.count() - 1)))

    layout.addWidget(tab_widget)

    win._developer_controls_dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

    if scroll_to_flux and int(tab_index) == 1:
        win._scroll_advanced_controls_to_flux()

    if not bool(getattr(win, '_developer_controls_unlocked', False)):
        dialog.setEnabled(False)
        win._show_developer_controls_unlock_popup()


def show_developer_controls_unlock_popup(win) -> None:
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

    existing = getattr(win, '_developer_unlock_dialog', None)
    if existing is not None:
        try:
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return
        except RuntimeError:
            win._developer_unlock_dialog = None

    unlock_dialog = QDialog(win)
    unlock_dialog.setWindowTitle("Developer Controls Warning")
    unlock_dialog.setMinimumWidth(420)
    unlock_dialog.setModal(True)
    unlock_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    unlock_dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    unlock_dialog.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
    unlock_dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    unlock_layout = QVBoxLayout(unlock_dialog)
    warning_label = QLabel("⚠️ DON'T BORK YOUR BEATS ⚠️")
    warning_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffaa00;")
    unlock_layout.addWidget(warning_label)

    warning_text = QLabel(
        "These controls are for advanced users. Incorrect settings\n"
        "may cause erratic behavior or break beat detection."
    )
    warning_text.setStyleSheet("color: #ccaa66;")
    unlock_layout.addWidget(warning_text)

    button_row = QHBoxLayout()
    cancel_btn = QPushButton("Cancel")
    unlock_btn = QPushButton("Unlock Advanced Controls")
    button_row.addWidget(cancel_btn)
    button_row.addWidget(unlock_btn)
    unlock_layout.addLayout(button_row)

    def _cancel() -> None:
        developer_dialog = getattr(win, '_developer_controls_dialog', None)
        win._developer_controls_unlocked = False
        win._developer_controls_dialog = None
        win._developer_controls_tab_widget = None
        if developer_dialog is not None:
            try:
                developer_dialog.close()
            except RuntimeError:
                pass
        unlock_dialog.close()

    def _unlock() -> None:
        win._developer_controls_unlocked = True
        developer_dialog = getattr(win, '_developer_controls_dialog', None)
        if developer_dialog is not None:
            try:
                developer_dialog.setEnabled(True)
                developer_dialog.show()
                developer_dialog.raise_()
                developer_dialog.activateWindow()
            except RuntimeError:
                pass
        unlock_dialog.accept()

    cancel_btn.clicked.connect(_cancel)
    unlock_btn.clicked.connect(_unlock)
    unlock_dialog.destroyed.connect(lambda *_: setattr(win, '_developer_unlock_dialog', None))

    win._developer_unlock_dialog = unlock_dialog
    unlock_dialog.show()
    unlock_dialog.raise_()
    unlock_dialog.activateWindow()


def on_options_beat_detection(win):
    """Show Beat Detection controls popout."""
    from PyQt6.QtWidgets import QDialog, QVBoxLayout

    dialog = getattr(win, '_beat_detection_dialog', None)
    if dialog is not None:
        try:
            dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            return
        except RuntimeError:
            win._beat_detection_dialog = None
            dialog = None

    dialog = QDialog(win)
    dialog.setWindowTitle("Beat Detection")
    dialog.setMinimumWidth(520)
    dialog.setMinimumHeight(640)
    dialog.setModal(False)
    dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    def _on_beat_detection_dialog_destroyed() -> None:
        win._beat_detection_dialog = None
        win._beat_detection_popout_content = None

    dialog.destroyed.connect(_on_beat_detection_dialog_destroyed)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(8, 8, 8, 8)

    content = getattr(win, '_beat_detection_popout_content', None)
    if content is not None:
        try:
            _ = content.parent()
        except RuntimeError:
            content = None
            win._beat_detection_popout_content = None
    if content is None:
        content = win._create_beat_detection_tab()
        win._beat_detection_popout_content = content
        win._apply_config_to_ui()
    layout.addWidget(content)

    win._beat_detection_dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def on_options_auto_fill_adaptation(win, as_tab: bool = False):
    """Show or build adaptive amp-fill gate tuning controls."""
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox

    if not as_tab:
        win._open_developer_controls_window(tab_index=2)
        return

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)

    info = QLabel(
        "Adaptive per-phase fill gating.\n"
        "If fill passes too often, required % rises; if too strict, it falls."
    )
    info.setStyleSheet("color: #bbb; font-size: 11px;")
    layout.addWidget(info)

    auto_enabled_cb = QCheckBox("Enable adaptive fill requirement")
    auto_enabled_cb.setChecked(bool(getattr(win.config.stroke, 'overall_amp_fill_auto_enabled', True)))
    auto_enabled_cb.stateChanged.connect(
        lambda state: setattr(win.config.stroke, 'overall_amp_fill_auto_enabled', state == 2)
    )
    layout.addWidget(auto_enabled_cb)

    target_rate_slider = SliderWithLabel(
        "Target fill pass rate",
        0.10,
        0.95,
        float(getattr(win.config.stroke, 'overall_amp_fill_auto_target_pass_rate', 0.58) or 0.58),
        2,
    )
    target_rate_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'overall_amp_fill_auto_target_pass_rate', float(v))
    )
    layout.addWidget(target_rate_slider)

    ema_alpha_slider = SliderWithLabel(
        "Pass-rate EMA alpha",
        0.01,
        0.60,
        float(getattr(win.config.stroke, 'overall_amp_fill_auto_ema_alpha', 0.12) or 0.12),
        3,
    )
    ema_alpha_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'overall_amp_fill_auto_ema_alpha', float(v))
    )
    layout.addWidget(ema_alpha_slider)

    deadband_slider = SliderWithLabel(
        "Deadband",
        0.00,
        0.40,
        float(getattr(win.config.stroke, 'overall_amp_fill_auto_deadband', 0.06) or 0.06),
        3,
    )
    deadband_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'overall_amp_fill_auto_deadband', float(v))
    )
    layout.addWidget(deadband_slider)

    step_slider = SliderWithLabel(
        "Step size",
        0.001,
        0.15,
        float(getattr(win.config.stroke, 'overall_amp_fill_auto_step', 0.02) or 0.02),
        3,
    )
    step_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'overall_amp_fill_auto_step', float(v))
    )
    layout.addWidget(step_slider)

    max_offset_slider = SliderWithLabel(
        "Max offset from base requirement",
        0.01,
        0.80,
        float(getattr(win.config.stroke, 'overall_amp_fill_auto_max_offset', 0.35) or 0.35),
        3,
    )
    max_offset_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'overall_amp_fill_auto_max_offset', float(v))
    )
    layout.addWidget(max_offset_slider)

    min_required_slider = SliderWithLabel(
        "Minimum required fill",
        0.00,
        0.95,
        float(getattr(win.config.stroke, 'overall_amp_fill_auto_min_required', 0.05) or 0.05),
        3,
    )
    max_required_slider = SliderWithLabel(
        "Maximum required fill",
        0.05,
        1.00,
        float(getattr(win.config.stroke, 'overall_amp_fill_auto_max_required', 0.98) or 0.98),
        3,
    )

    def _sync_required_bounds() -> None:
        min_val = float(min_required_slider.value())
        max_val = float(max_required_slider.value())
        if max_val < min_val:
            max_val = min_val
            max_required_slider.setValue(max_val)
        win.config.stroke.overall_amp_fill_auto_min_required = min_val
        win.config.stroke.overall_amp_fill_auto_max_required = max_val

    min_required_slider.valueChanged.connect(lambda _: _sync_required_bounds())
    max_required_slider.valueChanged.connect(lambda _: _sync_required_bounds())

    layout.addWidget(min_required_slider)
    layout.addWidget(max_required_slider)
    layout.addStretch()

    win._auto_fill_controls_widgets = {
        'enabled': auto_enabled_cb,
        'target_pass_rate': target_rate_slider,
        'ema_alpha': ema_alpha_slider,
        'deadband': deadband_slider,
        'step': step_slider,
        'max_offset': max_offset_slider,
        'min_required': min_required_slider,
        'max_required': max_required_slider,
    }
    return content


def on_options_motion_settings(win, tab_index: int = 0):
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTabWidget

    dialog = getattr(win, '_motion_settings_dialog', None)
    if dialog is not None:
        try:
            dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            tab_widget = getattr(win, '_motion_settings_tab_widget', None)
            if tab_widget is not None:
                tab_widget.setCurrentIndex(max(0, min(int(tab_index), tab_widget.count() - 1)))
            return
        except RuntimeError:
            win._motion_settings_dialog = None
            win._motion_settings_tab_widget = None

    dialog = QDialog(win)
    dialog.setWindowTitle("Motion Settings")
    dialog.setMinimumWidth(620)
    dialog.setMinimumHeight(560)
    dialog.setModal(False)
    dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    def _clear_motion_settings_state() -> None:
        win._motion_settings_dialog = None
        win._motion_settings_tab_widget = None

    dialog.rejected.connect(_clear_motion_settings_state)
    dialog.finished.connect(lambda _result: _clear_motion_settings_state())
    dialog.destroyed.connect(lambda *_: _clear_motion_settings_state())

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(8, 8, 8, 8)

    tab_widget = QTabWidget()
    win._motion_settings_tab_widget = tab_widget
    tab_widget.addTab(win._build_motion_options_tab(), "Motion Options")
    tab_widget.addTab(win._on_options_motion_readiness(as_tab=True), "Motion Readiness")
    tab_widget.setCurrentIndex(max(0, min(int(tab_index), tab_widget.count() - 1)))
    layout.addWidget(tab_widget)

    win._motion_settings_dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def on_options_motion_readiness(win, as_tab: bool = False):
    """Show or build readiness gating controls."""
    from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QCheckBox, QGroupBox, QPushButton

    if not as_tab:
        win._on_options_motion_settings(tab_index=1)
        return

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(10)

    info = QLabel(
        "Tune readiness fallback gating for beat-family motion.\n"
        "Lower confidence threshold + longer grace/finish = more fluid motion."
    )
    info.setStyleSheet("color: #bbb; font-size: 11px;")
    layout.addWidget(info)

    readiness_group = QGroupBox("Readiness")
    readiness_layout = QVBoxLayout(readiness_group)

    relaxed_conf_slider = SliderWithLabel(
        "Metronome Relaxed Confidence",
        0.00,
        1.00,
        float(getattr(win.config.beat, 'teaching_metronome_relaxed_confidence', 0.14) or 0.14),
        2,
    )
    relaxed_conf_slider.valueChanged.connect(
        lambda v: setattr(win.config.beat, 'teaching_metronome_relaxed_confidence', float(v))
    )
    readiness_layout.addWidget(relaxed_conf_slider)

    grace_ms_slider = SliderWithLabel(
        "Stroke Ready Grace (ms)",
        0.0,
        3000.0,
        float(getattr(win.config.beat, 'teaching_stroke_ready_grace_ms', 450.0) or 450.0),
        0,
    )
    grace_ms_slider.valueChanged.connect(
        lambda v: setattr(win.config.beat, 'teaching_stroke_ready_grace_ms', float(v))
    )
    readiness_layout.addWidget(grace_ms_slider)

    finish_row = QHBoxLayout()
    finish_label = QLabel("Stroke Finish Beats")
    finish_label.setStyleSheet("color: #ddd;")
    finish_spin = QSpinBox()
    finish_spin.setRange(0, 64)
    finish_spin.setValue(int(getattr(win.config.beat, 'teaching_stroke_finish_beats', 4) or 4))
    finish_spin.valueChanged.connect(
        lambda v: setattr(win.config.beat, 'teaching_stroke_finish_beats', int(v))
    )
    finish_row.addWidget(finish_label)
    finish_row.addStretch()
    finish_row.addWidget(finish_spin)
    readiness_layout.addLayout(finish_row)

    ignore_traffic_cb = QCheckBox("Use metronome-only readiness (legacy permissive)")
    ignore_traffic_cb.setChecked(bool(getattr(win.config.beat, 'teaching_ignore_traffic_lights', False)))
    ignore_traffic_cb.setToolTip(
        "When enabled, readiness uses metronome BPM + relaxed confidence only, "
        "ignoring stricter lock-style gating."
    )
    ignore_traffic_cb.stateChanged.connect(
        lambda state: setattr(win.config.beat, 'teaching_ignore_traffic_lights', state == 2)
    )
    readiness_layout.addWidget(ignore_traffic_cb)

    layout.addWidget(readiness_group)

    tuning_group = QGroupBox("Tuning")
    tuning_layout = QVBoxLayout(tuning_group)

    strength_slider = SliderWithLabel(
        "Advance", 0.0, 1.0,
        float(getattr(win.config.beat, 'teaching_learning_strength', 0.55) or 0.55), 2
    )
    tuning_layout.addWidget(strength_slider)

    holdback_slider = SliderWithLabel(
        "Restraint", 0.0, 1.0,
        float(getattr(win.config.beat, 'teaching_min_confidence', 0.12) or 0.12), 2
    )
    tuning_layout.addWidget(holdback_slider)

    no_motion_bias_slider = SliderWithLabel(
        "Quiet Bias", 0.25, 3.0,
        float(getattr(win.config.beat, 'teaching_no_motion_bias', 1.0) or 1.0), 2
    )
    tuning_layout.addWidget(no_motion_bias_slider)

    direction_hint = QLabel("⬅️ less         more ➡️")
    direction_hint.setStyleSheet("color: #d0d0d0; font-size: 14px; font-weight: 500;")
    direction_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
    tuning_layout.addWidget(direction_hint)

    settle_hint = QLabel("Move one, wait for adjust")
    settle_hint.setStyleSheet("color: #c7c7c7; font-size: 12px;")
    settle_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
    tuning_layout.addWidget(settle_hint)

    tuning_apply_btn = QPushButton("Apply Tuning")
    tuning_apply_btn.setStyleSheet("font-weight: 500;")

    def _apply_tuning_settings() -> None:
        win.config.beat.teaching_learning_strength = float(strength_slider.value())
        win.config.beat.teaching_min_confidence = float(holdback_slider.value())
        win.config.beat.teaching_no_motion_bias = float(no_motion_bias_slider.value())
        win._apply_learning_config_to_mapper()
        save_config(win.config)

    tuning_apply_btn.clicked.connect(_apply_tuning_settings)
    tuning_layout.addWidget(tuning_apply_btn)

    layout.addWidget(tuning_group)

    layout.addStretch()

    return content


def on_device_limits(win, first_run: bool = False):
    """Show Device Limits dialog for value-to-real-units conversion.
    Pulse Freq/Carrier Freq (Hz) are always shown. Pulse Width/Interval Random/Rise Time are optional.
    Called from Options menu or on first startup if not yet prompted."""
    if getattr(win, '_is_shutting_down', False):
        return

    from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QDoubleSpinBox,
                                  QPushButton, QHBoxLayout, QGridLayout, QGroupBox, QCheckBox)

    dialog = QDialog(win)
    dialog.setWindowFlags(
        dialog.windowFlags() | Qt.WindowType.WindowStaysOnTopHint
    )
    dialog.setWindowTitle("Device Output Limits")
    dialog.setMinimumWidth(400)
    layout = QVBoxLayout(dialog)

    info = QLabel(
        "If min/max limits in Restim have been changed, please adjust here."
    )
    info.setWordWrap(True)
    layout.addWidget(info)

    dl = win.config.device_limits

    def _first_run_default(value: float, default: float) -> float:
        return float(default) if first_run and float(value) <= 0.0 else float(value)

    def _default_if_unset(value: float, default: float) -> float:
        return float(default) if float(value) <= 0.0 else float(value)

    # --- Pulse Freq / Carrier Freq group (always visible) ---
    main_group = QGroupBox("Pulse Freq / Carrier Freq  —  Hz")
    grid = QGridLayout(main_group)
    grid.addWidget(QLabel("Pulse Freq Min Hz:"), 0, 0)
    p0_min = QDoubleSpinBox()
    p0_min.setRange(0, 99999)
    p0_min.setDecimals(1)
    p0_min.setValue(_first_run_default(dl.p0_freq_min, 1.0))
    p0_min.setSpecialValueText("not set")
    grid.addWidget(p0_min, 0, 1)

    grid.addWidget(QLabel("Pulse Freq Max Hz:"), 0, 2)
    p0_max = QDoubleSpinBox()
    p0_max.setRange(0, 99999)
    p0_max.setDecimals(1)
    p0_max.setValue(_first_run_default(dl.p0_freq_max, 100.0))
    p0_max.setSpecialValueText("not set")
    grid.addWidget(p0_max, 0, 3)

    grid.addWidget(QLabel("Carrier Freq Min Hz:"), 1, 0)
    c0_min = QDoubleSpinBox()
    c0_min.setRange(0, 99999)
    c0_min.setDecimals(1)
    c0_min.setValue(_default_if_unset(dl.c0_freq_min, 500.0))
    c0_min.setSpecialValueText("not set")
    grid.addWidget(c0_min, 1, 1)

    grid.addWidget(QLabel("Carrier Freq Max Hz:"), 1, 2)
    c0_max = QDoubleSpinBox()
    c0_max.setRange(0, 99999)
    c0_max.setDecimals(1)
    c0_max.setValue(_default_if_unset(dl.c0_freq_max, 1500.0))
    c0_max.setSpecialValueText("not set")
    grid.addWidget(c0_max, 1, 3)
    layout.addWidget(main_group)

    # --- Pulse Width / Interval Random / Rise Time group (optional, collapsed by default) ---
    has_extra = (dl.p1_cycles_max > 0 or dl.p2_range_max > 0 or dl.p3_cycles_max > 0)
    show_extra = QCheckBox("Show Pulse Width / Interval Random / Rise Time limits")
    show_extra.setChecked(has_extra)
    layout.addWidget(show_extra)

    extra_group = QGroupBox("Pulse Width / Interval Random / Rise Time")
    extra_grid = QGridLayout(extra_group)

    # Pulse Width in carrier cycles
    extra_grid.addWidget(QLabel("Pulse Width Min (cycles):"), 0, 0)
    p1_min = QDoubleSpinBox()
    p1_min.setRange(0, 99999)
    p1_min.setDecimals(1)
    p1_min.setValue(dl.p1_cycles_min)
    p1_min.setSpecialValueText("not set")
    extra_grid.addWidget(p1_min, 0, 1)

    extra_grid.addWidget(QLabel("Pulse Width Max (cycles):"), 0, 2)
    p1_max = QDoubleSpinBox()
    p1_max.setRange(0, 99999)
    p1_max.setDecimals(1)
    p1_max.setValue(dl.p1_cycles_max)
    p1_max.setSpecialValueText("not set")
    extra_grid.addWidget(p1_max, 0, 3)

    # Interval Random (0-1 range typically)
    extra_grid.addWidget(QLabel("Interval Random Min:"), 1, 0)
    p2_min = QDoubleSpinBox()
    p2_min.setRange(0, 99999)
    p2_min.setDecimals(2)
    p2_min.setValue(dl.p2_range_min)
    p2_min.setSpecialValueText("not set")
    p2_min.setToolTip("Pulse interval randomization — 0 to 1 on most devices")
    extra_grid.addWidget(p2_min, 1, 1)

    extra_grid.addWidget(QLabel("Interval Random Max:"), 1, 2)
    p2_max = QDoubleSpinBox()
    p2_max.setRange(0, 99999)
    p2_max.setDecimals(2)
    p2_max.setValue(dl.p2_range_max)
    p2_max.setSpecialValueText("not set")
    p2_max.setToolTip("Pulse interval randomization — 0 to 1 on most devices")
    extra_grid.addWidget(p2_max, 1, 3)

    # Rise Time in carrier cycles
    extra_grid.addWidget(QLabel("Rise Time Min (cycles):"), 2, 0)
    p3_min = QDoubleSpinBox()
    p3_min.setRange(0, 99999)
    p3_min.setDecimals(1)
    p3_min.setValue(dl.p3_cycles_min)
    p3_min.setSpecialValueText("not set")
    extra_grid.addWidget(p3_min, 2, 1)

    extra_grid.addWidget(QLabel("Rise Time Max (cycles):"), 2, 2)
    p3_max = QDoubleSpinBox()
    p3_max.setRange(0, 99999)
    p3_max.setDecimals(1)
    p3_max.setValue(dl.p3_cycles_max)
    p3_max.setSpecialValueText("not set")
    extra_grid.addWidget(p3_max, 2, 3)

    extra_group.setVisible(has_extra)
    show_extra.toggled.connect(extra_group.setVisible)
    layout.addWidget(extra_group)

    # --- P0/C0 sending toggle ---
    p0c0_cb = QCheckBox("Enable P0/C0 TCode sending to device")
    p0c0_cb.setChecked(dl.p0_c0_sending_enabled)
    p0c0_cb.setToolTip("Uncheck to disable sending Pulse Freq (P0) and Carrier Freq (C0) TCode commands.\n"
                       "Useful if your device doesn't support these axes.")
    layout.addWidget(p0c0_cb)

    # --- Don't show on startup checkbox ---
    dont_show_cb = QCheckBox("Don't show this dialog on startup")
    dont_show_cb.setChecked(dl.dont_show_on_startup)
    dont_show_cb.setToolTip("You can always open this later from Options → Device Limits")
    layout.addWidget(dont_show_cb)

    # OK / Cancel / Clear
    btn_row = QHBoxLayout()
    clear_btn = QPushButton("Clear All")
    def _clear_all():
        for spin in [p0_min, p0_max, c0_min, c0_max, p1_min, p1_max, p2_min, p2_max, p3_min, p3_max]:
            spin.setValue(0)
    clear_btn.clicked.connect(_clear_all)
    btn_row.addWidget(clear_btn)
    btn_row.addStretch()
    ok_btn = QPushButton("OK")
    ok_btn.clicked.connect(dialog.accept)
    cancel_btn = QPushButton("Cancel")
    cancel_btn.clicked.connect(dialog.reject)
    if first_run:
        skip_btn = QPushButton("Skip")
        skip_btn.setToolTip("Skip for now — you can set this later in Options → Device Limits")
        skip_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(skip_btn)
    btn_row.addWidget(ok_btn)
    btn_row.addWidget(cancel_btn)
    layout.addLayout(btn_row)

    dialog.raise_()
    dialog.activateWindow()
    if dialog.exec() == QDialog.DialogCode.Accepted:
        win.config.device_limits.p0_freq_min = p0_min.value()
        win.config.device_limits.p0_freq_max = p0_max.value()
        win.config.device_limits.c0_freq_min = c0_min.value()
        win.config.device_limits.c0_freq_max = c0_max.value()
        win.config.device_limits.p1_cycles_min = p1_min.value()
        win.config.device_limits.p1_cycles_max = p1_max.value()
        win.config.device_limits.p2_range_min = p2_min.value()
        win.config.device_limits.p2_range_max = p2_max.value()
        win.config.device_limits.p3_cycles_min = p3_min.value()
        win.config.device_limits.p3_cycles_max = p3_max.value()
        win.config.device_limits.p0_c0_sending_enabled = p0c0_cb.isChecked()
        win.config.device_limits.dont_show_on_startup = dont_show_cb.isChecked()
        win.config.device_limits.prompted = True
        win._sync_pulse_sent_spin_limits_from_device_limits()
        print(f"[Config] Device limits updated: P0={p0_min.value()}-{p0_max.value()}Hz, "
              f"C0={c0_min.value()}-{c0_max.value()}Hz, "
              f"P1={p1_min.value()}-{p1_max.value()}cyc, "
              f"P2={p2_min.value()}-{p2_max.value()}, "
              f"P3={p3_min.value()}-{p3_max.value()}cyc, "
              f"P0/C0 sending={'ON' if p0c0_cb.isChecked() else 'OFF'}")
    else:
        # Mark as prompted even if skipped/cancelled so we don't ask again
        win.config.device_limits.prompted = True
        win.config.device_limits.dont_show_on_startup = dont_show_cb.isChecked()


def sync_pulse_sent_spin_limits_from_device_limits(win) -> None:
    """Clamp Pulse Settings sent spinboxes to current Device Limits ranges."""
    dl = win.config.device_limits

    def _effective_limits(raw_min: float, raw_max: float, default_min: float, default_max: float) -> tuple[float, float]:
        lo = float(raw_min)
        hi = float(raw_max)
        if hi <= lo:
            lo = float(default_min)
            hi = float(default_max)
        return lo, hi

    def _apply_pair(min_attr: str, max_attr: str, raw_min: float, raw_max: float, default_min: float, default_max: float) -> None:
        min_spin = getattr(win, min_attr, None)
        max_spin = getattr(win, max_attr, None)
        if min_spin is None or max_spin is None:
            return
        lo, hi = _effective_limits(raw_min, raw_max, default_min, default_max)
        min_spin.setRange(lo, hi)
        max_spin.setRange(lo, hi)
        min_spin.setValue(max(lo, min(hi, float(min_spin.value()))))
        max_spin.setValue(max(lo, min(hi, float(max_spin.value()))))

    _apply_pair('p0_sent_min_spin', 'p0_sent_max_spin', dl.p0_freq_min, dl.p0_freq_max, 1.0, 100.0)
    _apply_pair('f0_sent_min_spin', 'f0_sent_max_spin', dl.c0_freq_min, dl.c0_freq_max, 500.0, 1500.0)
    _apply_pair('p1_sent_min_spin', 'p1_sent_max_spin', dl.p1_cycles_min, dl.p1_cycles_max, 0.0, 20.0)
    _apply_pair('p3_sent_min_spin', 'p3_sent_max_spin', dl.p3_cycles_min, dl.p3_cycles_max, 0.0, 20.0)


def scroll_advanced_controls_to_flux(win):
    """Scroll open Advanced Controls dialog near the Flux Sensitivity group."""
    scroll = getattr(win, '_advanced_controls_scroll', None)
    flux_group = getattr(win, '_advanced_flux_group', None)
    if scroll is None or flux_group is None:
        return

    def _apply_scroll():
        bar = scroll.verticalScrollBar()
        if bar is None:
            return
        target = max(0, int(flux_group.y()) - 12)
        bar.setValue(min(target, bar.maximum()))

    QTimer.singleShot(0, _apply_scroll)


def on_advanced_controls(win, scroll_to_flux: bool = False, as_tab: bool = False):
    """Show Advanced Controls dialog with experimental/expert settings"""
    from PyQt6.QtWidgets import QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QScrollArea, QGroupBox, QSpinBox

    if not as_tab:
        win._open_developer_controls_window(tab_index=1, scroll_to_flux=scroll_to_flux)
        return

    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setSpacing(10)

    units_box = QGroupBox("Normalized Units Reference")
    units_box.setStyleSheet("QGroupBox { background-color: #1f2630; border: 1px solid #4b5b70; border-radius: 4px; padding: 8px; }")
    units_layout = QVBoxLayout(units_box)
    units_label = QLabel(
        "Most Trigger Settings sliders use normalized units (0.0–1.0).\n"
        "• Amp (norm): 0 = no energy, 1 = near current peak envelope in active band.\n"
        "• Silence Gate controls use dBFS thresholds (negative dB values).\n"
        "• Fill/Occupancy (norm): fraction of active FFT bins that pass threshold in selected bin range.\n"
        "• Mean/Δ/Var thresholds: unitless activity metrics over recent windows (not dB/Hz).\n"
        "• dB, Hz, BPM, ms, and % controls are absolute units.\n"
        "Tip: change one normalized slider by ±0.02 to ±0.05 at a time."
    )
    units_label.setWordWrap(True)
    units_label.setStyleSheet("color: #b7c7d9; font-size: 11px;")
    units_layout.addWidget(units_label)
    layout.addWidget(units_box)

    # Scroll area for future controls
    scroll = NoWheelScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    win._advanced_controls_scroll = scroll
    scroll_content = QWidget()
    scroll_layout = QVBoxLayout(scroll_content)
    scroll_layout.setSpacing(10)

    # ===== Silence Gate Controls =====
    silence_group = QGroupBox("Silence Gate (dBFS)")
    silence_layout = QVBoxLayout(silence_group)

    silence_info = QLabel(
        "These thresholds control silence deadzone hysteresis in dBFS.\n"
        "More negative = quieter/more permissive. Open = enter silence, Close = exit silence."
    )
    silence_info.setStyleSheet("color: #aaa; font-size: 11px;")
    silence_layout.addWidget(silence_info)

    silence_open_db = win._silence_threshold_to_db(
        getattr(win.config.stroke, 'silence_threshold', -40.0),
        default_linear=0.001,
    )
    silence_close_db = win._silence_threshold_to_db(
        getattr(win.config.stroke, 'silence_close_threshold', -26.0),
        default_linear=0.01,
    )
    if silence_close_db <= silence_open_db:
        silence_close_db = float(min(0.0, silence_open_db + 1.5))

    silence_open_slider = SliderWithLabel(
        "No Motion Under (dBFS)",
        -90.0,
        -6.0,
        silence_open_db,
        1,
    )
    silence_open_slider.setToolTip("Audio level below this dBFS value enters silence mode (motion stops, dot parks)")

    silence_close_slider = SliderWithLabel(
        "Limited Motion Under (dBFS)",
        -90.0,
        -3.0,
        silence_close_db,
        1,
    )
    silence_close_slider.setToolTip("Audio level must exceed this dBFS value to exit silence mode (motion resumes)")

    def _set_silence_open(v: float) -> None:
        open_v = float(v)
        setattr(win.config.stroke, 'silence_threshold', open_v)
        close_v = win._silence_threshold_to_db(
            getattr(win.config.stroke, 'silence_close_threshold', -26.0),
            default_linear=0.01,
        )
        if close_v <= open_v:
            close_v = float(min(0.0, open_v + 1.5))
            setattr(win.config.stroke, 'silence_close_threshold', close_v)
            silence_close_slider.blockSignals(True)
            silence_close_slider.setValue(close_v)
            silence_close_slider.blockSignals(False)

    def _set_silence_close(v: float) -> None:
        close_v = float(v)
        open_v = win._silence_threshold_to_db(
            getattr(win.config.stroke, 'silence_threshold', -40.0),
            default_linear=0.001,
        )
        if close_v <= open_v:
            close_v = float(min(0.0, open_v + 1.5))
            silence_close_slider.blockSignals(True)
            silence_close_slider.setValue(close_v)
            silence_close_slider.blockSignals(False)
        setattr(win.config.stroke, 'silence_close_threshold', close_v)

    silence_open_slider.valueChanged.connect(_set_silence_open)
    silence_close_slider.valueChanged.connect(_set_silence_close)
    silence_layout.addWidget(silence_open_slider)
    silence_layout.addWidget(silence_close_slider)
    scroll_layout.addWidget(silence_group)

    # ===== Expression Layer Controls =====
    expression_group = QGroupBox("Expression Layer")
    expression_layout = QVBoxLayout(expression_group)

    expr_info = QLabel(
        "Artistic expression: center wandering,\n"
        "direction changes, tension pauses, and session arc."
    )
    expr_info.setStyleSheet("color: #aaa; font-size: 11px;")
    expression_layout.addWidget(expr_info)

    wander_cb = QCheckBox("Center wandering (orbit drifts horizontally)")
    wander_cb.setChecked(bool(getattr(win.config.stroke, 'center_wander_enabled', True)))
    wander_cb.stateChanged.connect(
        lambda state: setattr(win.config.stroke, 'center_wander_enabled', state == 2)
    )
    expression_layout.addWidget(wander_cb)

    wander_max_slider = SliderWithLabel(
        "Wander max offset",
        0.0, 0.50,
        float(getattr(win.config.stroke, 'center_wander_max_x', 0.20) or 0.20),
        2,
    )
    wander_max_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'center_wander_max_x', float(v))
    )
    expression_layout.addWidget(wander_max_slider)

    wander_cycle_slider = SliderWithLabel(
        "Wander cycle (seconds)",
        5.0, 60.0,
        float(getattr(win.config.stroke, 'center_wander_cycle_s', 25.0) or 25.0),
        1,
    )
    wander_cycle_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'center_wander_cycle_s', float(v))
    )
    expression_layout.addWidget(wander_cycle_slider)

    wander_energy_slider = SliderWithLabel(
        "Wander energy influence",
        0.0, 1.0,
        float(getattr(win.config.stroke, 'center_wander_energy_scale', 0.6) or 0.6),
        2,
    )
    wander_energy_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'center_wander_energy_scale', float(v))
    )
    expression_layout.addWidget(wander_energy_slider)

    direction_cb = QCheckBox("Direction changes at phrase boundaries")
    direction_cb.setChecked(bool(getattr(win.config.stroke, 'direction_change_enabled', True)))
    direction_cb.stateChanged.connect(
        lambda state: setattr(win.config.stroke, 'direction_change_enabled', state == 2)
    )
    expression_layout.addWidget(direction_cb)

    direction_interval_slider = SliderWithLabel(
        "Min interval between reversals (s)",
        5.0, 60.0,
        float(getattr(win.config.stroke, 'direction_change_interval_s', 15.0) or 15.0),
        1,
    )
    direction_interval_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'direction_change_interval_s', float(v))
    )
    expression_layout.addWidget(direction_interval_slider)

    direction_drop_slider = SliderWithLabel(
        "Energy change to trigger reversal",
        0.10, 0.80,
        float(getattr(win.config.stroke, 'direction_change_energy_drop', 0.35) or 0.35),
        2,
    )
    direction_drop_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'direction_change_energy_drop', float(v))
    )
    expression_layout.addWidget(direction_drop_slider)

    session_cb = QCheckBox("Session arc (gradual long-term intensity evolution)")
    session_cb.setChecked(bool(getattr(win.config.stroke, 'session_arc_enabled', True)))
    session_cb.stateChanged.connect(
        lambda state: setattr(win.config.stroke, 'session_arc_enabled', state == 2)
    )
    expression_layout.addWidget(session_cb)

    session_influence_slider = SliderWithLabel(
        "Session arc radius influence",
        0.0, 0.30,
        float(getattr(win.config.stroke, 'session_arc_radius_influence', 0.10) or 0.10),
        2,
    )
    session_influence_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'session_arc_radius_influence', float(v))
    )
    expression_layout.addWidget(session_influence_slider)

    scroll_layout.addWidget(expression_group)

    # ===== Post-Silence Volume Ramp =====
    silence_ramp_group = QGroupBox("Post-Silence Volume Ramp")
    silence_ramp_layout = QVBoxLayout(silence_ramp_group)

    silence_ramp_info = QLabel("After silence (track change), reduce volume and slowly\nraise it back over a configurable duration.")
    silence_ramp_info.setStyleSheet("color: #aaa; font-size: 11px;")
    silence_ramp_layout.addWidget(silence_ramp_info)

    vol_reduction_slider = SliderWithLabel(
        "Volume reduction (%)",
        0.0,
        0.50,
        float(getattr(win.config.stroke, 'post_silence_vol_reduction', 0.15) or 0.15),
        2,
    )
    vol_reduction_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'post_silence_vol_reduction', v)
    )
    silence_ramp_layout.addWidget(vol_reduction_slider)

    ramp_dur_slider = SliderWithLabel(
        "Ramp duration (seconds)",
        1.0,
        8.0,
        float(getattr(win.config.stroke, 'post_silence_ramp_seconds', 4.0) or 4.0),
        1,
    )
    ramp_dur_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'post_silence_ramp_seconds', v)
    )
    silence_ramp_layout.addWidget(ramp_dur_slider)

    fade_drop_row = QHBoxLayout()
    fade_drop_label = QLabel("Fade max drop points (out of 100):")
    fade_drop_label.setStyleSheet("color: #ccc;")
    fade_drop_row.addWidget(fade_drop_label)
    fade_drop_spin = QSpinBox()
    fade_drop_spin.setRange(0, 10)
    fade_drop_spin.setSingleStep(1)
    fade_drop_spin.setValue(int(np.clip(getattr(win.config.stroke, 'silence_fade_drop_points', 10) or 10, 0, 10)))
    fade_drop_spin.setToolTip("Caps runtime fade reduction to this many volume points (0-10)")
    fade_drop_spin.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'silence_fade_drop_points', int(v))
    )
    fade_drop_row.addWidget(fade_drop_spin)
    fade_drop_row.addStretch()
    silence_ramp_layout.addLayout(fade_drop_row)

    scroll_layout.addWidget(silence_ramp_group)

    # ===== Syncopation Controls =====
    syncope_group = QGroupBox("Syncopation / Double-Stroke")
    syncope_layout = QVBoxLayout(syncope_group)

    # On/Off checkbox
    syncope_enabled_cb = QCheckBox("Allow Off-Beat Strokes")
    syncope_enabled_cb.setChecked(bool(getattr(win.config.beat, 'syncopation_enabled', True)))
    syncope_enabled_cb.setToolTip("When enabled, system detects syncopation and fires rapid 1-beat strokes on off-beats")
    syncope_enabled_cb.stateChanged.connect(
        lambda state: setattr(win.config.beat, 'syncopation_enabled', state == 2)
    )
    syncope_layout.addWidget(syncope_enabled_cb)

    # Band selector
    from PyQt6.QtWidgets import QComboBox, QHBoxLayout as QHBox
    band_row = QHBox()
    band_label = QLabel("Off-Beat Detection Band:")
    band_label.setStyleSheet("color: #ccc;")
    band_label.setToolTip("Which frequency range to scan for off-beat onsets: 'any' = all bands, or specific range (sub_bass/low_mid/mid/high)")
    band_row.addWidget(band_label)
    band_combo = QComboBox()
    band_options = ['any', 'sub_bass', 'low_mid', 'mid', 'high']
    band_combo.addItems(band_options)
    current_band = str(getattr(win.config.beat, 'syncopation_band', 'any'))
    if current_band in band_options:
        band_combo.setCurrentIndex(band_options.index(current_band))
    band_combo.currentTextChanged.connect(
        lambda text: setattr(win.config.beat, 'syncopation_band', text)
    )
    band_row.addWidget(band_combo)
    syncope_layout.addLayout(band_row)

    # Syncopation window slider
    syncope_window_slider = SliderWithLabel(
        "Off-Beat Timing Window",
        0.05,
        0.30,
        float(getattr(win.config.beat, 'syncopation_window', 0.16)),
        2,
    )
    syncope_window_slider.setToolTip("Search window (as fraction of beat period) around expected off-beat position. Wider window = more permissive but slower")
    syncope_window_slider.valueChanged.connect(
        lambda v: setattr(win.config.beat, 'syncopation_window', v)
    )
    syncope_layout.addWidget(syncope_window_slider)

    # BPM limit slider
    syncope_bpm_slider = SliderWithLabel(
        "Max BPM for Off-Beats",
        80.0,
        200.0,
        float(getattr(win.config.beat, 'syncopation_bpm_limit', 130.0)),
        0,
    )
    syncope_bpm_slider.setToolTip("Disable off-beat detection above this BPM (to prevent false positives in very fast music)")
    syncope_bpm_slider.valueChanged.connect(
        lambda v: setattr(win.config.beat, 'syncopation_bpm_limit', v)
    )
    syncope_layout.addWidget(syncope_bpm_slider)

    scroll_layout.addWidget(syncope_group)

    # ===== Amplitude Gate Controls =====
    gate_group = QGroupBox("Amplitude Gate")
    gate_layout = QVBoxLayout(gate_group)

    gate_info = QLabel(
        "Controls stroke gating behavior and overall amplitude/fill requirements.\n"
        "Overall amp target/tolerance define the full-spectrum amplitude zone used by the amp+fill gate (target ± tolerance)."
    )
    gate_info.setStyleSheet("color: #aaa; font-size: 11px;")
    gate_layout.addWidget(gate_info)
    def _show_freqdb_ghost_ref(
        key: str,
        value: float,
        label: str,
        color: str = '#FF66AA',
        dashed: bool = False,
        band: str = 'full',
        range_box: bool = False,
        mode: str = 'threshold',
        hz_max: float | None = None,
    ) -> None:
        if hasattr(win, 'freqdb_canvas') and hasattr(win.freqdb_canvas, 'show_flux_ghost'):
            win.freqdb_canvas.show_flux_ghost(
                key,
                float(value),
                label,
                color=color,
                duration_s=15.0,
                dashed=dashed,
                band=band,
                range_box=range_box,
                mode=mode,
                hz_max=hz_max,
            )

    def _update_overall_amp_fill_refs():
        target = float(getattr(win.config.stroke, 'overall_amp_fill_target', 0.5) or 0.5)
        tol = float(abs(getattr(win.config.stroke, 'overall_amp_fill_tolerance', 0.5) or 0.5))
        min_amp = max(0.0, target - tol)
        _show_freqdb_ghost_ref('overall_amp_target', target, 'Amp target', '#66CCFF', dashed=False, band='full')
        _show_freqdb_ghost_ref('overall_amp_min', min_amp, 'Amp min', '#FFAA66', dashed=True, band='full')

    def _update_fill_requirement_refs() -> None:
        win._preview_fill_requirement_ghosts()

    def _show_fft_bin_fill_ref(key: str, ratio: float, label: str, color: str = '#66E0FF', dashed: bool = True) -> None:
        canvas = getattr(win, 'fft_bin_canvas', None)
        if canvas is not None and hasattr(canvas, 'show_fill_ratio_ghost'):
            canvas.show_fill_ratio_ghost(key, float(ratio), label, color=color, duration_s=5.0, dashed=dashed)

    def _show_fft_bin_range_ref(key: str, low_bin: int, high_bin: int, label: str, color: str = '#FFFFFF', dashed: bool = False) -> None:
        canvas = getattr(win, 'fft_bin_canvas', None)
        if canvas is not None and hasattr(canvas, 'show_bin_range_ghost'):
            if hasattr(canvas, '_bar_count') and int(getattr(canvas, '_bar_count', 0) or 0) <= 0 and hasattr(canvas, '_ensure_bars'):
                fft_size = int(getattr(win.config.audio, 'fft_size', 1024) or 1024)
                bin_count = max(2, (fft_size // 2) + 1)
                canvas._ensure_bars(bin_count)
            canvas.show_bin_range_ghost(key, int(low_bin), int(high_bin), label, color=color, duration_s=5.0, dashed=dashed)


    motion_cutoff_row = QHBoxLayout()
    motion_cutoff_label = QLabel("Motion Bass Cutoff Frequency:")
    motion_cutoff_label.setToolTip("If Bass Energy check is on: only allow strokes from sounds whose lowest frequency is below this cutoff (Hz)")
    motion_cutoff_row.addWidget(motion_cutoff_label)

    win.motion_freq_cutoff_spin = QSpinBox()
    win.motion_freq_cutoff_spin.setRange(0, 2000)
    win.motion_freq_cutoff_spin.setSingleStep(20)
    win.motion_freq_cutoff_spin.setValue(int(getattr(win.config.beat, 'motion_freq_cutoff', 500)))
    win.motion_freq_cutoff_spin.setSuffix(" Hz")
    win.motion_freq_cutoff_spin.setFixedWidth(90)
    win.motion_freq_cutoff_spin.setToolTip("0 disables cutoff filtering (while Bass Gating is enabled)")
    win.motion_freq_cutoff_spin.valueChanged.connect(
        lambda v: (
            win._on_motion_freq_cutoff_change(v),
            _show_freqdb_ghost_ref('motion_freq_cutoff_hz', float(v), 'Motion cutoff', '#FFD166', dashed=True, mode='hz_line', range_box=False)
        )
    )
    motion_cutoff_row.addWidget(win.motion_freq_cutoff_spin)
    motion_cutoff_row.addStretch()
    gate_layout.addLayout(motion_cutoff_row)

    amp_fill_gate_cb = QCheckBox("Enable Spectral Fullness Gate")
    amp_fill_gate_cb.setChecked(bool(getattr(win.config.stroke, 'overall_amp_fill_gate_enabled', True)))
    amp_fill_gate_cb.setToolTip("Require both overall amplitude AND spectrum 'fullness' before strokes fire (prevents sparse/thin sections from triggering)")
    amp_fill_gate_cb.stateChanged.connect(
        lambda state: setattr(win.config.stroke, 'overall_amp_fill_gate_enabled', state == 2)
    )
    gate_layout.addWidget(amp_fill_gate_cb)

    amp_fill_target_slider = SliderWithLabel(
        "Required Spectral Fullness",
        0.0,
        1.0,
        float(getattr(win.config.stroke, 'overall_amp_fill_target', 0.5) or 0.5),
        2,
    )
    amp_fill_target_slider.valueChanged.connect(
        lambda v: (setattr(win.config.stroke, 'overall_amp_fill_target', float(v)), _update_overall_amp_fill_refs())
    )
    amp_fill_target_slider.setToolTip("Normalized target amplitude (0-1). Gates won't fire below this intensity range")
    gate_layout.addWidget(amp_fill_target_slider)

    amp_fill_tol_slider = SliderWithLabel(
        "Amplitude Range Width",
        0.0,
        1.0,
        float(getattr(win.config.stroke, 'overall_amp_fill_tolerance', 0.5) or 0.5),
        2,
    )
    amp_fill_tol_slider.valueChanged.connect(
        lambda v: (setattr(win.config.stroke, 'overall_amp_fill_tolerance', float(v)), _update_overall_amp_fill_refs())
    )
    amp_fill_tol_slider.setToolTip("Normalized tolerance band around the target (±). Strokes fire when amplitude stays in this zone")
    gate_layout.addWidget(amp_fill_tol_slider)

    downbeat_fill_slider = SliderWithLabel(
        "Downbeat Spectral Fullness Requirement",
        0.0,
        1.0,
        float(getattr(win.config.stroke, 'downbeat_overall_amp_fill_required', 0.08) or 0.08),
        2,
    )
    downbeat_fill_slider.setToolTip("How 'full' the spectrum must be (0-1) for downbeat to fire. Higher = requires more filled/dense spectrum")
    downbeat_fill_slider.valueChanged.connect(
        lambda v: (setattr(win.config.stroke, 'downbeat_overall_amp_fill_required', float(v)), _update_fill_requirement_refs())
    )
    gate_layout.addWidget(downbeat_fill_slider)

    beat_fill_slider = SliderWithLabel(
        "Beat Spectral Fullness Requirement",
        0.0,
        1.0,
        float(getattr(win.config.stroke, 'beat_overall_amp_fill_required', 0.10) or 0.10),
        2,
    )
    beat_fill_slider.setToolTip("How 'full' the spectrum must be (0-1) for beat strokes to fire. Higher = requires fuller, richer spectrum")
    beat_fill_slider.valueChanged.connect(
        lambda v: (setattr(win.config.stroke, 'beat_overall_amp_fill_required', float(v)), _update_fill_requirement_refs())
    )
    gate_layout.addWidget(beat_fill_slider)

    sync_fill_slider = SliderWithLabel(
        "Off-Beat Spectral Fullness Requirement",
        0.0,
        1.0,
        float(getattr(win.config.stroke, 'syncopation_overall_amp_fill_required', 0.12) or 0.12),
        2,
    )
    sync_fill_slider.setToolTip("How 'full' the spectrum must be (0-1) for off-beat strokes to fire. Usually highest since off-beats are stricter")
    sync_fill_slider.valueChanged.connect(
        lambda v: (setattr(win.config.stroke, 'syncopation_overall_amp_fill_required', float(v)), _update_fill_requirement_refs())
    )
    gate_layout.addWidget(sync_fill_slider)

    fill_bin_info = QLabel("Fill gate FFT-bin windows (tight range control per phase)")
    fill_bin_info.setStyleSheet("color: #999; font-size: 10px;")
    gate_layout.addWidget(fill_bin_info)

    fft_size = int(getattr(win.config.audio, 'fft_size', 1024) or 1024)
    max_bin = max(1, fft_size // 2)

    def _bin_to_hz(bin_value: int) -> float:
        sample_rate = float(getattr(win.config.audio, 'sample_rate', 44100) or 44100)
        return float(np.clip(bin_value, 0, max_bin) * (sample_rate / max(1, fft_size)))

    def _add_fill_bin_range_row(title: str, low_attr: str, high_attr: str, ghost_key: str) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel(title))

        low_spin = QSpinBox()
        low_spin.setRange(0, max_bin)
        low_spin.setSingleStep(1)
        low_spin.setValue(int(np.clip(int(getattr(win.config.stroke, low_attr, 0) or 0), 0, max_bin)))
        low_spin.setPrefix("low ")
        row.addWidget(low_spin)

        high_spin = QSpinBox()
        high_spin.setRange(0, max_bin)
        high_spin.setSingleStep(1)
        high_spin.setValue(int(np.clip(int(getattr(win.config.stroke, high_attr, max_bin) or max_bin), 0, max_bin)))
        high_spin.setPrefix("high ")
        row.addWidget(high_spin)
        row.addStretch()

        def _emit_ghost() -> None:
            low_bin = int(low_spin.value())
            high_bin = int(high_spin.value())
            low_hz = _bin_to_hz(min(low_bin, high_bin))
            high_hz = _bin_to_hz(max(low_bin, high_bin))
            _show_freqdb_ghost_ref(
                ghost_key,
                low_hz,
                f"{title} range",
                color='#FFFFFF',
                dashed=False,
                mode='hz_line',
                range_box=True,
                hz_max=high_hz,
            )
            _show_fft_bin_range_ref(
                ghost_key,
                min(low_bin, high_bin),
                max(low_bin, high_bin),
                title,
                color='#FFFFFF',
                dashed=False,
            )

        def _on_low_change(v: int) -> None:
            low_val = int(v)
            high_val = int(high_spin.value())
            if high_val < low_val:
                high_spin.setValue(low_val)
                high_val = low_val
            setattr(win.config.stroke, low_attr, int(low_val))
            setattr(win.config.stroke, high_attr, int(high_val))
            _emit_ghost()

        def _on_high_change(v: int) -> None:
            high_val = int(v)
            low_val = int(low_spin.value())
            if high_val < low_val:
                low_spin.setValue(high_val)
                low_val = high_val
            setattr(win.config.stroke, low_attr, int(low_val))
            setattr(win.config.stroke, high_attr, int(high_val))
            _emit_ghost()

        low_spin.valueChanged.connect(_on_low_change)
        high_spin.valueChanged.connect(_on_high_change)
        gate_layout.addLayout(row)
        _emit_ghost()

    def _add_fill_sustain_slider(title: str, sustain_attr: str) -> None:
        sustain_slider = SliderWithLabel(
            f"{title} Sustained Fullness Duration",
            0,
            15,
            int(getattr(win.config.stroke, sustain_attr, 3) or 3),
            0,
        )
        sustain_slider.setToolTip(
            "Consecutive frames (~20ms each) that spectrum must stay full before stroke fires.\n"
            "0-1 = instant, 3 = ~60ms, 5 = ~100ms, 10 = ~200ms.\n"
            "Prevents single-frame spikes from triggering."
        )
        sustain_slider.valueChanged.connect(
            lambda v: setattr(win.config.stroke, sustain_attr, int(v))
        )
        gate_layout.addWidget(sustain_slider)

    _add_fill_bin_range_row("Downbeat fill bins", 'downbeat_fill_bin_low', 'downbeat_fill_bin_high', 'downbeat_fill_bin_range')
    _add_fill_sustain_slider("Downbeat", 'downbeat_overall_amp_fill_sustain_frames')
    _add_fill_bin_range_row("Beat fill bins", 'beat_fill_bin_low', 'beat_fill_bin_high', 'beat_fill_bin_range')
    _add_fill_sustain_slider("Beat", 'beat_overall_amp_fill_sustain_frames')
    _add_fill_bin_range_row("Sync fill bins", 'syncopation_fill_bin_low', 'syncopation_fill_bin_high', 'sync_fill_bin_range')
    _add_fill_sustain_slider("Syncopation", 'syncopation_overall_amp_fill_sustain_frames')

    scroll_layout.addWidget(gate_group)

    scheduling_group = QGroupBox("Beat Scheduling")
    scheduling_layout = QVBoxLayout(scheduling_group)

    lead_row = QHBoxLayout()
    lead_label = QLabel("Pipeline lead (ms):")
    lead_label.setStyleSheet("color: #ccc;")
    lead_row.addWidget(lead_label)
    lead_spin = QSpinBox()
    lead_spin.setMinimum(0)
    lead_spin.setMaximum(200)
    lead_spin.setSingleStep(1)
    lead_spin.setValue(int(getattr(win.config.beat, 'scheduled_lead_ms', 0)))
    lead_spin.setToolTip(
        "Pipeline latency compensation (ms).\n"
        "Shortens each orbit journey by this amount so the motion peak\n"
        "lands on the actual musical beat instead of lagging behind.\n"
        "Compensates for WASAPI loopback delay + audio buffer latency.\n"
        "Typical range: 40-80 ms. Increase if motion arrives late; decrease if early."
    )
    def _on_lead_ms_changed(v):
        val = int(v)
        win.config.beat.scheduled_lead_ms = val
        mapper = getattr(win, 'stroke_mapper', None)
        if mapper is not None:
            mapper.set_scheduled_lead_ms(val)
    lead_spin.valueChanged.connect(_on_lead_ms_changed)
    lead_row.addWidget(lead_spin)
    scheduling_layout.addLayout(lead_row)

    scroll_layout.addWidget(scheduling_group)

    # ===== Flux Controls =====
    flux_group = QGroupBox("Flux Sensitivity")
    win._advanced_flux_group = flux_group
    flux_layout = QVBoxLayout(flux_group)

    flux_info = QLabel(
        "Controls flux- and activity-based guards.\n"
        "Beat/downbeat stroke admission now uses low-band activity (sub-bass + low-mid).\n"
        "Overall activity guard blocks beat strokes when full-spectrum flux+energy are both low."
    )
    flux_info.setStyleSheet("color: #aaa; font-size: 11px;")
    flux_layout.addWidget(flux_info)

    # Flux threshold slider
    flux_thresh_slider = SliderWithLabel(
        "Flux threshold",
        0.005,
        0.20,
        float(getattr(win.config.stroke, 'flux_threshold', 0.05) or 0.05),
        3,
    )
    win._advanced_flux_threshold_slider = flux_thresh_slider
    flux_thresh_slider.valueChanged.connect(
        lambda v: (
            setattr(win.config.stroke, 'flux_threshold', v),
            _set_stroke_attr_with_ref('flux_threshold', 'flux_threshold', 'Flux threshold', '#66D9FF', ghost_band='full')(float(v))
        )
    )
    flux_layout.addWidget(flux_thresh_slider)

    flux_scaling_slider = SliderWithLabel(
        "Flux Scaling (size)",
        0.0,
        2.0,
        float(getattr(win.config.stroke, 'flux_scaling_weight', 1.0) or 1.0),
        2,
    )
    win._advanced_flux_scaling_slider = flux_scaling_slider
    flux_scaling_slider.valueChanged.connect(
        lambda v: setattr(win.config.stroke, 'flux_scaling_weight', float(v))
    )
    flux_layout.addWidget(flux_scaling_slider)

    def _set_stroke_attr_with_ref(
        attr_name: str,
        ref_key: str,
        ref_label: str,
        ref_color: str,
        dashed: bool = False,
        ghost_band: str = 'full',
        ghost_range: bool = False,
        ghost_mode: str = 'threshold',
        ghost_value_resolver=None,
    ):
        def _handler(v: float):
            value = float(v)
            setattr(win.config.stroke, attr_name, value)
            resolved = ghost_value_resolver(value) if callable(ghost_value_resolver) else value
            ghost_value = win.freqdb_canvas._as_float(resolved, value) if hasattr(win, 'freqdb_canvas') and hasattr(win.freqdb_canvas, '_as_float') else float(value)
            if hasattr(win, 'freqdb_canvas') and hasattr(win.freqdb_canvas, 'show_flux_ghost'):
                win.freqdb_canvas.show_flux_ghost(
                    ref_key,
                    ghost_value,
                    ref_label,
                    color=ref_color,
                    duration_s=15.0,
                    dashed=dashed,
                    band=ghost_band,
                    range_box=ghost_range,
                    mode=ghost_mode,
                )
        return _handler

    high_include_mid_cb = QCheckBox("Include mid band in upper-range visualization")
    high_include_mid_cb.setChecked(bool(getattr(win.config.stroke, 'high_band_include_mid', True)))
    high_include_mid_cb.stateChanged.connect(
        lambda state: setattr(win.config.stroke, 'high_band_include_mid', state == 2)
    )
    flux_layout.addWidget(high_include_mid_cb)

    scroll_layout.addWidget(flux_group)

    scroll_layout.addStretch()
    scroll.setWidget(scroll_content)

    layout.addWidget(scroll)

    if scroll_to_flux:
        win._scroll_advanced_controls_to_flux()

    return content


def on_help(win):
    """Show Help/Troubleshooting dialog with reset buttons (non-modal)"""
    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea, QGroupBox, QPushButton, QHBoxLayout

    dialog = QDialog(win)
    dialog.setWindowTitle("Help - Troubleshooting")
    dialog.setMinimumWidth(420)
    dialog.setMinimumHeight(500)
    # Make non-modal so user can interact with main window
    dialog.setModal(False)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    layout = QVBoxLayout(dialog)
    layout.setSpacing(8)

    # Scroll area for content
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll_content = QWidget()
    scroll_layout = QVBoxLayout(scroll_content)
    scroll_layout.setSpacing(10)

    # === No motion? ===
    group1 = QGroupBox("No motion?")
    g1_layout = QVBoxLayout(group1)
    g1_layout.setSpacing(4)

    # Audio device check with button
    audio_box = QGroupBox()
    audio_box.setStyleSheet("QGroupBox { border: 1px solid #555; padding: 4px; margin-top: 2px; }")
    ab_layout = QVBoxLayout(audio_box)
    ab_layout.setSpacing(2)
    ab_layout.addWidget(QLabel("Check [Options]→[Audio Device] is your\ncurrent speakers or input with signal"))
    audio_btn = QPushButton("Open Audio Device")
    audio_btn.clicked.connect(lambda: win._on_options_audio_device())
    ab_layout.addWidget(audio_btn)
    g1_layout.addWidget(audio_box)

    g1_layout.addWidget(QLabel("• Check [Start] and [Play] are pressed"))
    g1_layout.addWidget(QLabel("• Both BPM lights should blink with stable count"))
    g1_layout.addWidget(QLabel("• Raise sensitivity/amplification until beats detected"))

    scroll_layout.addWidget(group1)

    # === Still no motion? ===
    group2 = QGroupBox("Still no motion?")
    g2_layout = QVBoxLayout(group2)
    g2_layout.setSpacing(4)

    stroke_box = QGroupBox()
    stroke_box.setStyleSheet("QGroupBox { border: 1px solid #555; padding: 4px; margin-top: 2px; }")
    sb_layout = QVBoxLayout(stroke_box)
    sb_layout.setSpacing(2)
    sb_layout.addWidget(QLabel("Stroke Settings tab has been removed in orbital mode."))
    sb_layout.addWidget(QLabel("Use Main Controls + Advanced Controls for motion tuning."))
    g2_layout.addWidget(stroke_box)

    # Peak floor reset
    floor_box = QGroupBox()
    floor_box.setStyleSheet("QGroupBox { border: 1px solid #555; padding: 4px; margin-top: 2px; }")
    flb_layout = QVBoxLayout(floor_box)
    flb_layout.setSpacing(2)
    flb_layout.addWidget(QLabel("[Beat Detection] Check depth:"))
    floor_reset_btn = QPushButton("Reset Depth to 0")
    floor_reset_btn.clicked.connect(lambda: win.peak_floor_slider.setValue(0.0))
    flb_layout.addWidget(floor_reset_btn)
    g2_layout.addWidget(floor_box)

    g2_layout.addWidget(QLabel("If using stroke mode 1, 2, or 3:"))

    # Axis weights reset
    axis_box = QGroupBox()
    axis_box.setStyleSheet("QGroupBox { border: 1px solid #555; padding: 4px; margin-top: 2px; }")
    axb_layout = QVBoxLayout(axis_box)
    axb_layout.setSpacing(2)
    axb_layout.addWidget(QLabel("[Options→Effects] Check Jitter toggle:"))
    axis_reset_btn = QPushButton("Enable Jitter")
    axis_reset_btn.clicked.connect(lambda: (
        setattr(win.config.jitter, 'enabled', True),
        win._sync_effects_menu_actions()
    ))
    axb_layout.addWidget(axis_reset_btn)
    g2_layout.addWidget(axis_box)

    scroll_layout.addWidget(group2)

    # === Too much motion? ===
    group3 = QGroupBox("Too much motion?")
    g3_layout = QVBoxLayout(group3)
    g3_layout.setSpacing(4)
    g3_layout.addWidget(QLabel("• [Beat Detection] Lower audio amplification,\n  sensitivity, flux multiplier"))
    g3_layout.addWidget(QLabel("• [Options→Effects] Disable Jitter"))
    g3_layout.addWidget(QLabel("• [Tempo Tracking] Check spectral flux control"))
    scroll_layout.addWidget(group3)

    scroll_layout.addStretch()
    scroll.setWidget(scroll_content)
    layout.addWidget(scroll)

    # Close button
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.close)
    layout.addWidget(close_btn)

    dialog.show()  # Use show() instead of exec() for non-modal


def on_fft_bin_diagnostics(win):
    """Show FFT bin resolution details and nearest-bin mapping for a target frequency."""
    sample_rate = int(getattr(win.config.audio, 'sample_rate', 44100) or 44100)
    fft_size = int(getattr(win.config.audio, 'fft_size', 1024) or 1024)
    if hasattr(win, 'audio_engine') and win.audio_engine is not None:
        try:
            sample_rate = int(getattr(win.audio_engine.config.audio, 'sample_rate', sample_rate) or sample_rate)
            fft_size = int(getattr(win.audio_engine, 'fft_size', fft_size) or fft_size)
        except Exception:
            pass

    fft_size = max(16, fft_size)
    nyquist = sample_rate / 2.0
    hz_per_bin = sample_rate / float(fft_size)
    max_bin = (fft_size // 2)

    target_hz, ok = QInputDialog.getDouble(
        win,
        "FFT Bin Diagnostics",
        "Frequency to inspect (Hz):",
        1000.0,
        0.0,
        nyquist,
        3,
    )
    if not ok:
        return

    bin_float = target_hz / hz_per_bin if hz_per_bin > 0 else 0.0
    nearest_bin = int(np.clip(round(bin_float), 0, max_bin))
    nearest_hz = nearest_bin * hz_per_bin
    offset_hz = target_hz - nearest_hz
    bin_offset = bin_float - nearest_bin
    centered_bin_tol = 0.01
    centered_hz_tol = centered_bin_tol * hz_per_bin
    is_centered = abs(bin_offset) <= centered_bin_tol

    QMessageBox.information(
        win,
        "FFT Bin Diagnostics",
        (
            f"Sample rate: {sample_rate} Hz\n"
            f"FFT size: {fft_size}\n"
            f"Nyquist: {nyquist:.3f} Hz\n"
            f"Resolution: {hz_per_bin:.6f} Hz/bin\n\n"
            f"Target frequency: {target_hz:.3f} Hz\n"
            f"Nearest bin: {nearest_bin} (center {nearest_hz:.3f} Hz)\n"
            f"Offset from bin center: {bin_offset:+.6f} bins\n"
            f"Offset from bin center: {offset_hz:+.6f} Hz\n"
            f"Bin-centered (±{centered_bin_tol:.2f} bins, ≈ ±{centered_hz_tol:.6f} Hz): {'yes' if is_centered else 'no'}"
        ),
    )


def on_about(win):
    """Show About dialog"""
    display_version = __version__
    if display_version.startswith("v"):
        display_version = display_version[1:]
    if "-" in display_version:
        display_version = display_version.split("-", 1)[0]

    about_html = f"""
<b>bREadbeats {display_version}</b><br>
Live Audio to Restim<br><br>
Inspired by:<br>
&nbsp;&nbsp;&nbsp;&nbsp;digitalparkinglot's creations<br>
&nbsp;&nbsp;&nbsp;&nbsp;edger477 (ideas from generator tooling)<br>
&nbsp;&nbsp;&nbsp;&nbsp;diglet48 (wouldn't be here without restim!)<br>
&nbsp;&nbsp;&nbsp;&nbsp;shadlock0133 (music-vibes)<br><br>
Bug reports:<br>
bREadfan_69@hotmail.com<br><br>
Like the app?<br>
<a href="https://ko-fi.com/breadbeats">https://ko-fi.com/breadbeats</a>
"""
    msg = QMessageBox(win)
    msg.setWindowTitle("About bREadbeats")
    msg.setIcon(QMessageBox.Icon.Information)
    msg.setTextFormat(Qt.TextFormat.RichText)
    msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    msg.setText(about_html)
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    for label in msg.findChildren(QLabel):
        label.setOpenExternalLinks(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    msg.exec()


