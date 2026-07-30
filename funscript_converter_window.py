"""FunScript Converter window — 6-axis to 4-phase electrode tool."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from funscript_converter import (
    CONVERTER_INPUT_AXES,
    DEFAULT_LAYOUT_MODEL,
    FreqConfig,
    IDENTITY_WIRING_MAP,
    LAYOUT_MODEL_DISPLAY_NAMES,
    MixWeights,
    convert,
)
from config import Config
from funscript_utils import AXIS_SUFFIXES, load_folder, load_script_axes, strip_axis_suffix
from pmv_colors import FOURPHASE_AXIS_COLORS, FOURPHASE_AXIS_ORDER
from pmv_funscript_io import FunscriptAction, FunscriptMetadata, read_funscript, write_funscript
from widgets import SliderWithLabel

# Electrode channel colors (match FOC-Stim output colors)
_E_COLORS = [FOURPHASE_AXIS_COLORS[name] for name in FOURPHASE_AXIS_ORDER]
_E_NAMES = [name.upper() for name in FOURPHASE_AXIS_ORDER]
_PREVIEW_SPECS = [
    ("e1", _E_NAMES[0], _E_COLORS[0]),
    ("e2", _E_NAMES[1], _E_COLORS[1]),
    ("e3", _E_NAMES[2], _E_COLORS[2]),
    ("e4", _E_NAMES[3], _E_COLORS[3]),
    ("pulse_frequency", "Pulse", "#ff9f43"),
    ("carrier_frequency", "Carrier", "#a29bfe"),
]

_LAYOUT_DIAGRAMS = {
    "Pair At Top": (
        "      E1   E2\n"
        "        E3\n"
        "        E4"
    ),
    "Pair At Middle": (
        "        E1\n"
        "      E2   E3\n"
        "        E4"
    ),
    "Pair At Bottom / Rear": (
        "        E1\n"
        "        E2\n"
        "      E3   E4"
    ),
}

# Performance options for pyqtgraph curves
_DS = dict(clipToView=True, autoDownsample=True, downsampleMethod="peak")


def _style_plot(widget: pg.PlotWidget) -> None:
    widget.setBackground("#2f2f2f")
    widget.showGrid(x=True, y=True, alpha=0.12)
    widget.getAxis("bottom").setTextPen(pg.mkPen("#c8c8c8"))
    widget.getAxis("left").setTextPen(pg.mkPen("#c8c8c8"))
    widget.setMouseEnabled(x=True, y=False)
    widget.getViewBox().setLimits(yMin=0.0, yMax=100.0, minYRange=100.0, maxYRange=100.0)


class FunscriptConverterWindow(QMainWindow):
    def __init__(
        self,
        parent: QWidget | None = None,
        preview_callback: Callable[[str, dict[str, list[FunscriptAction]], Path | None], None] | None = None,
        config: Config | None = None,
        save_settings: Callable[[Config], bool] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("FunScript Converter — 6-Axis to 4-Phase")
        self.resize(1100, 700)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._preview_callback = preview_callback
        self._config = config
        self._save_settings = save_settings

        # State
        self._loaded_axes: dict[str, list[FunscriptAction]] = {}
        self._source_folder: Path | None = None
        self._base_stem: str = ""
        self._result: dict[str, list[FunscriptAction]] = {}
        self._wiring_map: tuple[int, int, int, int] = IDENTITY_WIRING_MAP
        self._reconvert_timer = QTimer(self)
        self._reconvert_timer.setSingleShot(True)
        self._reconvert_timer.setInterval(150)
        self._reconvert_timer.timeout.connect(self._run_conversion)

        self._build_ui()
        self._restore_persisted_state()
        self._update_axes_status()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # Top bar — load / export buttons
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)

        btn_load_file = QPushButton("Load Script...")
        btn_load_file.clicked.connect(self._on_load_file)
        top_row.addWidget(btn_load_file)

        btn_load_folder = QPushButton("Load Folder...")
        btn_load_folder.clicked.connect(self._on_load_folder)
        top_row.addWidget(btn_load_folder)

        top_row.addStretch(1)

        self._preview_btn = QPushButton("Preview In Generator...")
        self._preview_btn.setEnabled(False)
        self._preview_btn.clicked.connect(self._on_preview_in_generator)
        if self._preview_callback is None:
            self._preview_btn.setToolTip("Open the converter from bREadbeats to preview in the PMV Generator.")
        top_row.addWidget(self._preview_btn)

        self._export_btn = QPushButton("Export e1–e4...")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        top_row.addWidget(self._export_btn)

        root_layout.addLayout(top_row)

        # Main splitter — settings on left, preview on right
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        root_layout.addWidget(splitter, 1)

        # ---- Left panel (scrollable settings) ----
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(260)
        left_scroll.setMaximumWidth(360)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)

        # Loaded axes group
        axes_group = QGroupBox("Loaded Axes")
        axes_layout = QVBoxLayout(axes_group)
        self._axis_labels: dict[str, QLabel] = {}
        for axis in ["main", "surge", "sway", "twist", "roll", "pitch"]:
            lbl = QLabel(f"  {axis}: —")
            lbl.setStyleSheet("color: #999; font-size: 11px;")
            axes_layout.addWidget(lbl)
            self._axis_labels[axis] = lbl
        left_layout.addWidget(axes_group)

        # Layout model group
        layout_group = QGroupBox("Layout Model")
        layout_gl = QVBoxLayout(layout_group)

        layout_row = QHBoxLayout()
        layout_row.addWidget(QLabel("Layout:"))
        self._layout_combo = QComboBox()
        for model_name, display_name in LAYOUT_MODEL_DISPLAY_NAMES.items():
            self._layout_combo.addItem(display_name, model_name)
        default_index = self._layout_combo.findData(DEFAULT_LAYOUT_MODEL)
        if default_index >= 0:
            self._layout_combo.setCurrentIndex(default_index)
        self._layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        layout_row.addWidget(self._layout_combo, 1)
        layout_gl.addLayout(layout_row)

        # Diagram label
        self._diagram_label = QLabel()
        self._diagram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._diagram_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px; color: #ccc; "
            "background: #2a2a2a; padding: 8px; border-radius: 4px;"
        )
        layout_gl.addWidget(self._diagram_label)
        self._update_diagram()

        left_layout.addWidget(layout_group)

        # Mixing weights group
        mix_group = QGroupBox("Mixing Weights")
        mix_layout = QVBoxLayout(mix_group)

        self._w_primary = self._add_spin(mix_layout, "Primary (linear):", 0.0, 1.0, 0.8)
        self._w_secondary = self._add_spin(mix_layout, "Secondary (rotational):", 0.0, 1.0, 0.2)
        self._w_twist = self._add_spin(mix_layout, "Twist weight:", 0.0, 1.0, 0.3)
        self._twist_phase = self._add_spin(mix_layout, "Twist phase (rad):", 0.0, 6.28, 0.7854, decimals=4)

        left_layout.addWidget(mix_group)

        # Frequency output group
        freq_group = QGroupBox("Frequency Outputs")
        freq_layout = QVBoxLayout(freq_group)
        self._freq_enabled = QCheckBox("Export frequency axes")
        self._freq_enabled.setChecked(True)
        self._freq_enabled.stateChanged.connect(self._schedule_reconvert)
        freq_layout.addWidget(self._freq_enabled)
        self._freq_scale = self._add_spin(freq_layout, "Pulse freq scale:", 0.0, 5.0, 1.0)
        self._pulse_surge_influence = self._add_slider(freq_layout, "Pulse surge influence:", 0.0, 2.0, 1.0)
        self._pulse_speed_influence = self._add_slider(freq_layout, "Pulse speed influence:", 0.0, 1.0, 0.15)
        self._pulse_center = self._add_spin(freq_layout, "Pulse freq center:", 0.0, 100.0, 55.0, decimals=1)
        self._pulse_min = self._add_spin(freq_layout, "Pulse freq min:", 0.0, 100.0, 20.0, decimals=1)
        self._pulse_max = self._add_spin(freq_layout, "Pulse freq max:", 0.0, 100.0, 80.0, decimals=1)
        self._carrier_scale = self._add_spin(freq_layout, "Carrier freq scale:", 0.0, 5.0, 1.0)
        self._carrier_surge_influence = self._add_slider(freq_layout, "Carrier surge influence:", 0.0, 2.0, 1.0)
        self._carrier_speed_influence = self._add_slider(freq_layout, "Carrier speed influence:", 0.0, 1.0, 0.10)
        self._carrier_center = self._add_spin(freq_layout, "Carrier freq center:", 0.0, 100.0, 50.0, decimals=1)
        self._carrier_min = self._add_spin(freq_layout, "Carrier freq min:", 0.0, 100.0, 40.0, decimals=1)
        self._carrier_max = self._add_spin(freq_layout, "Carrier freq max:", 0.0, 100.0, 60.0, decimals=1)
        left_layout.addWidget(freq_group)

        left_layout.addStretch(1)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # ---- Right panel (preview plots) ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)

        self._plots: list[pg.PlotWidget] = []
        self._curves: list[pg.PlotDataItem] = []

        for i, (_axis_name, axis_label, color) in enumerate(_PREVIEW_SPECS):
            pw = pg.PlotWidget()
            _style_plot(pw)
            pw.setLabel("left", axis_label)
            pw.setYRange(0, 100)
            if i < len(_PREVIEW_SPECS) - 1:
                pw.getAxis("bottom").setStyle(showValues=False)
            else:
                pw.setLabel("bottom", "Time (s)")
            curve = pw.plot([], [], pen=pg.mkPen(color, width=1.4), **_DS)
            self._plots.append(pw)
            self._curves.append(curve)
            right_layout.addWidget(pw)

        # Link X axes so they zoom/pan together
        for i in range(1, len(self._plots)):
            self._plots[i].setXLink(self._plots[0])

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(root)

    def _add_spin(
        self,
        layout: QVBoxLayout,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        decimals: int = 2,
    ) -> QDoubleSpinBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.05)
        spin.setValue(default)
        spin.valueChanged.connect(self._schedule_reconvert)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def _add_slider(
        self,
        layout: QVBoxLayout,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        decimals: int = 2,
        step: float = 0.01,
    ) -> SliderWithLabel:
        slider = SliderWithLabel(label, min_val, max_val, default, decimals=decimals, step=step)
        slider.valueChanged.connect(self._schedule_reconvert)
        layout.addWidget(slider)
        return slider

    # ------------------------------------------------------------------
    # Diagram
    # ------------------------------------------------------------------

    def _get_current_layout_model(self) -> str:
        model_name = self._layout_combo.currentData()
        return model_name if isinstance(model_name, str) else DEFAULT_LAYOUT_MODEL

    def _get_current_wiring_map(self) -> tuple[int, int, int, int]:
        return getattr(self, "_wiring_map", IDENTITY_WIRING_MAP)

    def _update_diagram(self) -> None:
        layout_model = self._get_current_layout_model()
        diagram = _LAYOUT_DIAGRAMS.get(layout_model, _LAYOUT_DIAGRAMS[DEFAULT_LAYOUT_MODEL])
        self._diagram_label.setText(diagram)

    def _normalize_wiring_map(self, raw_wiring_map: object) -> tuple[int, int, int, int]:
        try:
            wiring_map = [int(value) for value in raw_wiring_map]  # type: ignore[arg-type]
        except Exception:
            return IDENTITY_WIRING_MAP
        if len(wiring_map) != 4 or sorted(wiring_map) != [0, 1, 2, 3]:
            return IDENTITY_WIRING_MAP
        return (wiring_map[0], wiring_map[1], wiring_map[2], wiring_map[3])

    def _restore_persisted_state(self) -> None:
        if self._config is None:
            self._update_diagram()
            return
        state = getattr(self._config, "funscript_converter", None)
        if state is None:
            self._update_diagram()
            return

        layout_model = str(getattr(state, "layout_model", DEFAULT_LAYOUT_MODEL) or DEFAULT_LAYOUT_MODEL)
        if layout_model not in LAYOUT_MODEL_DISPLAY_NAMES:
            layout_model = DEFAULT_LAYOUT_MODEL
        self._wiring_map = self._normalize_wiring_map(getattr(state, "wiring_map", IDENTITY_WIRING_MAP))

        layout_index = self._layout_combo.findData(layout_model)
        if layout_index < 0:
            layout_index = self._layout_combo.findData(DEFAULT_LAYOUT_MODEL)
        self._layout_combo.blockSignals(True)
        if layout_index >= 0:
            self._layout_combo.setCurrentIndex(layout_index)
        self._layout_combo.blockSignals(False)
        self._w_primary.setValue(
            max(0.0, min(1.0, float(getattr(state, "w_primary", 0.8))))
        )
        self._w_secondary.setValue(
            max(0.0, min(1.0, float(getattr(state, "w_secondary", 0.2))))
        )
        self._w_twist.setValue(
            max(0.0, min(1.0, float(getattr(state, "w_twist", 0.3))))
        )
        self._twist_phase.setValue(
            max(0.0, min(6.28, float(getattr(state, "twist_phase", 0.7854))))
        )
        self._freq_enabled.setChecked(bool(getattr(state, "freq_enabled", True)))
        self._freq_scale.setValue(
            max(0.0, min(5.0, float(getattr(state, "freq_scale", 1.0))))
        )
        self._pulse_surge_influence.setValue(
            max(0.0, min(2.0, float(getattr(state, "pulse_surge_influence", 1.0))))
        )
        self._pulse_speed_influence.setValue(
            max(0.0, min(1.0, float(getattr(state, "pulse_speed_influence", 0.15))))
        )
        self._pulse_center.setValue(
            max(0.0, min(100.0, float(getattr(state, "pulse_center", 55.0))))
        )
        self._pulse_min.setValue(
            max(0.0, min(100.0, float(getattr(state, "pulse_min", 20.0))))
        )
        self._pulse_max.setValue(
            max(0.0, min(100.0, float(getattr(state, "pulse_max", 80.0))))
        )
        self._carrier_scale.setValue(
            max(0.0, min(5.0, float(getattr(state, "carrier_scale", 1.0))))
        )
        self._carrier_surge_influence.setValue(
            max(0.0, min(2.0, float(getattr(state, "carrier_surge_influence", 1.0))))
        )
        self._carrier_speed_influence.setValue(
            max(0.0, min(1.0, float(getattr(state, "carrier_speed_influence", 0.10))))
        )
        self._carrier_center.setValue(
            max(0.0, min(100.0, float(getattr(state, "carrier_center", 50.0))))
        )
        self._carrier_min.setValue(
            max(0.0, min(100.0, float(getattr(state, "carrier_min", 40.0))))
        )
        self._carrier_max.setValue(
            max(0.0, min(100.0, float(getattr(state, "carrier_max", 60.0))))
        )
        self._update_diagram()

    def _persist_state(self, *, save_now: bool) -> None:
        if self._config is None:
            return
        state = getattr(self._config, "funscript_converter", None)
        if state is None:
            return
        state.layout_model = self._get_current_layout_model()
        state.wiring_map = list(self._get_current_wiring_map())
        state.w_primary = self._w_primary.value()
        state.w_secondary = self._w_secondary.value()
        state.w_twist = self._w_twist.value()
        state.twist_phase = self._twist_phase.value()
        state.freq_enabled = self._freq_enabled.isChecked()
        state.freq_scale = self._freq_scale.value()
        state.pulse_surge_influence = self._pulse_surge_influence.value()
        state.pulse_speed_influence = self._pulse_speed_influence.value()
        state.pulse_center = self._pulse_center.value()
        state.pulse_min = self._pulse_min.value()
        state.pulse_max = self._pulse_max.value()
        state.carrier_scale = self._carrier_scale.value()
        state.carrier_surge_influence = self._carrier_surge_influence.value()
        state.carrier_speed_influence = self._carrier_speed_influence.value()
        state.carrier_center = self._carrier_center.value()
        state.carrier_min = self._carrier_min.value()
        state.carrier_max = self._carrier_max.value()
        if save_now and self._save_settings is not None:
            self._save_settings(self._config)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _on_load_file(self) -> None:
        path_list, _ = QFileDialog.getOpenFileNames(
            self, "Open Funscript", "", "Funscript Files (*.funscript);;All Files (*)"
        )
        selected_paths = [Path(path_str) for path_str in path_list if path_str]
        if not selected_paths:
            return

        if len(selected_paths) > 1:
            self._load_selected_files(selected_paths)
            return

        script_path = selected_paths[0]
        self._source_folder = script_path.parent

        base_stem, suffix = strip_axis_suffix(script_path.stem)
        self._base_stem = base_stem

        try:
            loaded_axes = load_script_axes(script_path, CONVERTER_INPUT_AXES | {"main"})
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", str(exc))
            return

        if not loaded_axes:
            QMessageBox.warning(self, "Load Error", "No supported axes found in file.")
            return

        self._loaded_axes = loaded_axes

        # Discover siblings
        from funscript_utils import discover_sibling_axes
        siblings = discover_sibling_axes(script_path, CONVERTER_INPUT_AXES | {"main"})
        self._loaded_axes.update(siblings)

        self._update_axes_status()
        self._run_conversion()

    def _load_selected_files(self, script_paths: list[Path]) -> None:
        loaded_axes: dict[str, list[FunscriptAction]] = {}
        base_script: Path | None = None

        for script_path in script_paths:
            try:
                script_axes = load_script_axes(script_path, CONVERTER_INPUT_AXES | {"main"})
            except Exception as exc:
                QMessageBox.warning(self, "Load Error", f"{script_path.name}: {exc}")
                return

            if not script_axes:
                continue

            base_stem, suffix = strip_axis_suffix(script_path.stem)
            loaded_axes.update(script_axes)

            if base_script is None or "main" in script_axes or suffix is None:
                base_script = script_path

        if not loaded_axes:
            return

        assert base_script is not None
        self._source_folder = base_script.parent
        self._base_stem, _ = strip_axis_suffix(base_script.stem)
        self._loaded_axes = loaded_axes
        self._update_axes_status()
        self._run_conversion()

    def _on_load_folder(self) -> None:
        folder_str = QFileDialog.getExistingDirectory(self, "Select Funscript Folder")
        if not folder_str:
            return
        folder = Path(folder_str)
        self._source_folder = folder

        loaded = load_folder(folder, CONVERTER_INPUT_AXES | {"main"})
        if not loaded:
            QMessageBox.information(self, "No Files", "No funscript files found in folder.")
            return

        # Infer base stem from first loaded file
        candidates = sorted(folder.glob("*.funscript"))
        if candidates:
            base, _ = strip_axis_suffix(candidates[0].stem)
            self._base_stem = base

        self._loaded_axes = loaded
        self._update_axes_status()
        self._run_conversion()

    def _update_axes_status(self) -> None:
        for axis, lbl in self._axis_labels.items():
            if axis in self._loaded_axes:
                count = len(self._loaded_axes[axis])
                lbl.setText(f"  {axis}: ✓ ({count} pts)")
                lbl.setStyleSheet("color: #6bcb77; font-size: 11px;")
            else:
                lbl.setText(f"  {axis}: —")
                lbl.setStyleSheet("color: #999; font-size: 11px;")

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _schedule_reconvert(self, *_args) -> None:
        if self._loaded_axes:
            self._reconvert_timer.start()

    def _get_mix_weights(self) -> MixWeights:
        return MixWeights(
            w_primary=self._w_primary.value(),
            w_secondary=self._w_secondary.value(),
            w_twist=self._w_twist.value(),
            twist_phase=self._twist_phase.value(),
        )

    def _get_freq_config(self) -> FreqConfig:
        pulse_min = min(self._pulse_min.value(), self._pulse_max.value())
        pulse_max = max(self._pulse_min.value(), self._pulse_max.value())
        pulse_center = min(max(self._pulse_center.value(), pulse_min), pulse_max)
        carrier_min = min(self._carrier_min.value(), self._carrier_max.value())
        carrier_max = max(self._carrier_min.value(), self._carrier_max.value())
        carrier_center = min(max(self._carrier_center.value(), carrier_min), carrier_max)

        return FreqConfig(
            enabled=self._freq_enabled.isChecked(),
            freq_scale=self._freq_scale.value(),
            pulse_surge_influence=self._pulse_surge_influence.value(),
            pulse_speed_influence=self._pulse_speed_influence.value(),
            pulse_center=pulse_center,
            pulse_min=pulse_min,
            pulse_max=pulse_max,
            carrier_scale=self._carrier_scale.value(),
            carrier_surge_influence=self._carrier_surge_influence.value(),
            carrier_speed_influence=self._carrier_speed_influence.value(),
            carrier_center=carrier_center,
            carrier_min=carrier_min,
            carrier_max=carrier_max,
        )

    def _run_conversion(self) -> None:
        if not self._loaded_axes:
            return

        layout_model = self._get_current_layout_model()
        wiring_map = self._get_current_wiring_map()
        weights = self._get_mix_weights()
        freq_cfg = self._get_freq_config()

        self._result = convert(
            self._loaded_axes,
            placement=wiring_map,
            weights=weights,
            freq_config=freq_cfg,
            layout_model=layout_model,
        )
        self._update_preview()
        self._preview_btn.setEnabled(bool(self._result) and self._preview_callback is not None)
        self._export_btn.setEnabled(bool(self._result))

    def _on_preview_in_generator(self) -> None:
        if self._reconvert_timer.isActive():
            self._reconvert_timer.stop()
            self._run_conversion()

        if not self._result:
            return
        if self._preview_callback is None:
            QMessageBox.information(
                self,
                "Preview Unavailable",
                "Open the converter from bREadbeats to preview in the PMV Generator.",
            )
            return

        preview_axes = {
            axis_name: [FunscriptAction(a.at, a.pos) for a in actions]
            for axis_name, actions in self._result.items()
            if actions
        }
        try:
            self._preview_callback(self._base_stem or "converted", preview_axes, self._source_folder)
        except Exception as exc:
            QMessageBox.warning(self, "Preview Error", str(exc))

    def _update_preview(self) -> None:
        for i, (name, _axis_label, _color) in enumerate(_PREVIEW_SPECS):
            actions = self._result.get(name, [])
            if actions:
                t = np.array([a.at / 1000.0 for a in actions])
                p = np.array([a.pos for a in actions])
                self._curves[i].setData(t, p)
            else:
                self._curves[i].setData([], [])

    # ------------------------------------------------------------------
    # Electrode swapping
    # ------------------------------------------------------------------

    def _swap_electrodes(self, a: int, b: int) -> None:
        wiring_map = list(self._get_current_wiring_map())
        wiring_map[a], wiring_map[b] = wiring_map[b], wiring_map[a]
        self._wiring_map = (wiring_map[0], wiring_map[1], wiring_map[2], wiring_map[3])

        self._update_diagram()
        self._persist_state(save_now=True)
        self._schedule_reconvert()

    def _reset_wiring_map(self) -> None:
        self._wiring_map = IDENTITY_WIRING_MAP
        self._update_diagram()
        self._persist_state(save_now=True)
        self._schedule_reconvert()

    def _on_layout_changed(self, _index: int) -> None:
        self._update_diagram()
        self._persist_state(save_now=True)
        self._schedule_reconvert()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._persist_state(save_now=True)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        if self._reconvert_timer.isActive():
            self._reconvert_timer.stop()
            self._run_conversion()

        if not self._result:
            return

        if self._source_folder and self._base_stem:
            default_dir = str(self._source_folder)
        else:
            default_dir = ""

        folder_str = QFileDialog.getExistingDirectory(
            self, "Export to Folder", default_dir
        )
        if not folder_str:
            return
        folder = Path(folder_str)

        base = self._base_stem or "converted"
        meta = FunscriptMetadata(creator="bREadbeats FunScript Converter")
        exported = []

        for axis_name, actions in self._result.items():
            if axis_name == "main":
                fname = f"{base}.funscript"
            else:
                fname = f"{base}.{axis_name}.funscript"
            out_path = folder / fname
            write_funscript(out_path, actions, meta)
            exported.append(fname)

        QMessageBox.information(
            self,
            "Export Complete",
            f"Exported {len(exported)} file(s) to:\n{folder}\n\n" + "\n".join(exported),
        )
