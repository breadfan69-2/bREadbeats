"""FunScript Converter window — 6-axis to 4-phase electrode tool."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
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
    DEFAULT_PRESET,
    PRESETS,
    FreqConfig,
    MixWeights,
    convert,
)
from funscript_utils import AXIS_SUFFIXES, load_folder, strip_axis_suffix
from pmv_funscript_io import FunscriptAction, FunscriptMetadata, read_funscript, write_funscript

# Electrode channel colors (match restim convention)
_E_COLORS = ["#ff6b6b", "#6bcb77", "#4d96ff", "#ffd93d"]
_E_NAMES = ["E1", "E2", "E3", "E4"]

# Performance options for pyqtgraph curves
_DS = dict(clipToView=True, autoDownsample=True, downsampleMethod="peak")


def _style_plot(widget: pg.PlotWidget) -> None:
    widget.setBackground("#2f2f2f")
    widget.showGrid(x=True, y=True, alpha=0.12)
    widget.getAxis("bottom").setTextPen(pg.mkPen("#c8c8c8"))
    widget.getAxis("left").setTextPen(pg.mkPen("#c8c8c8"))


class FunscriptConverterWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("FunScript Converter — 6-Axis to 4-Phase")
        self.resize(1100, 700)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # State
        self._loaded_axes: dict[str, list[FunscriptAction]] = {}
        self._source_folder: Path | None = None
        self._base_stem: str = ""
        self._result: dict[str, list[FunscriptAction]] = {}
        self._reconvert_timer = QTimer(self)
        self._reconvert_timer.setSingleShot(True)
        self._reconvert_timer.setInterval(150)
        self._reconvert_timer.timeout.connect(self._run_conversion)

        self._build_ui()
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

        # Electrode Layout group
        layout_group = QGroupBox("Electrode Layout")
        layout_gl = QVBoxLayout(layout_group)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        for name in PRESETS:
            self._preset_combo.addItem(name)
        self._preset_combo.addItem("Custom (modified)")
        self._preset_combo.setCurrentText(DEFAULT_PRESET)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self._preset_combo, 1)
        layout_gl.addLayout(preset_row)

        # Diagram label
        self._diagram_label = QLabel()
        self._diagram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._diagram_label.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px; color: #ccc; "
            "background: #2a2a2a; padding: 8px; border-radius: 4px;"
        )
        layout_gl.addWidget(self._diagram_label)
        self._update_diagram()

        # Swap buttons
        swap_row1 = QHBoxLayout()
        btn_swap_12 = QPushButton("Swap E1↔E2")
        btn_swap_12.clicked.connect(lambda: self._swap_electrodes(0, 1))
        swap_row1.addWidget(btn_swap_12)
        btn_swap_34 = QPushButton("Swap E3↔E4")
        btn_swap_34.clicked.connect(lambda: self._swap_electrodes(2, 3))
        swap_row1.addWidget(btn_swap_34)
        layout_gl.addLayout(swap_row1)

        swap_row2 = QHBoxLayout()
        btn_swap_tb = QPushButton("Swap Top↔Bottom")
        btn_swap_tb.clicked.connect(lambda: self._swap_electrodes(0, 3))
        swap_row2.addWidget(btn_swap_tb)
        btn_swap_lr = QPushButton("Swap Left↔Right")
        btn_swap_lr.clicked.connect(lambda: self._swap_electrodes(1, 2))
        swap_row2.addWidget(btn_swap_lr)
        layout_gl.addLayout(swap_row2)

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
        self._freq_enabled.stateChanged.connect(self._schedule_reconvert)
        freq_layout.addWidget(self._freq_enabled)
        self._freq_scale = self._add_spin(freq_layout, "Pulse freq scale:", 0.0, 5.0, 1.0)
        self._carrier_scale = self._add_spin(freq_layout, "Carrier freq scale:", 0.0, 5.0, 1.0)
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

        for i in range(4):
            pw = pg.PlotWidget()
            _style_plot(pw)
            pw.setLabel("left", _E_NAMES[i])
            pw.setYRange(0, 100)
            if i < 3:
                pw.getAxis("bottom").setStyle(showValues=False)
            else:
                pw.setLabel("bottom", "Time (s)")
            curve = pw.plot([], [], pen=pg.mkPen(_E_COLORS[i], width=1.4), **_DS)
            self._plots.append(pw)
            self._curves.append(curve)
            right_layout.addWidget(pw)

        # Link X axes so they zoom/pan together
        for i in range(1, 4):
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

    # ------------------------------------------------------------------
    # Diagram
    # ------------------------------------------------------------------

    def _get_current_placement(self) -> tuple[int, int, int, int]:
        name = self._preset_combo.currentText()
        if name in PRESETS:
            return PRESETS[name]
        return getattr(self, "_custom_placement", PRESETS[DEFAULT_PRESET])

    def _update_diagram(self) -> None:
        p = self._get_current_placement()
        # Map vertex indices to labels
        labels = [f"v{p[i]}" for i in range(4)]
        text = (
            f"      {_E_NAMES[0]} ({labels[0]})\n"
            f"     /        \\\n"
            f"  {_E_NAMES[1]} ({labels[1]})   {_E_NAMES[2]} ({labels[2]})\n"
            f"     \\        /\n"
            f"      {_E_NAMES[3]} ({labels[3]})"
        )
        self._diagram_label.setText(text)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _on_load_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open Funscript", "", "Funscript Files (*.funscript);;All Files (*)"
        )
        if not path_str:
            return
        script_path = Path(path_str)
        self._source_folder = script_path.parent

        base_stem, suffix = strip_axis_suffix(script_path.stem)
        self._base_stem = base_stem

        # Load selected file
        axis_name = suffix if suffix else "main"
        try:
            actions, _ = read_funscript(script_path)
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", str(exc))
            return

        self._loaded_axes = {axis_name: actions}

        # Discover siblings
        from funscript_utils import discover_sibling_axes
        siblings = discover_sibling_axes(script_path, CONVERTER_INPUT_AXES | {"main"})
        self._loaded_axes.update(siblings)

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
        return FreqConfig(
            enabled=self._freq_enabled.isChecked(),
            freq_scale=self._freq_scale.value(),
            carrier_scale=self._carrier_scale.value(),
        )

    def _run_conversion(self) -> None:
        if not self._loaded_axes:
            return

        placement = self._get_current_placement()
        weights = self._get_mix_weights()
        freq_cfg = self._get_freq_config()

        self._result = convert(self._loaded_axes, placement, weights, freq_cfg)
        self._update_preview()
        self._export_btn.setEnabled(bool(self._result))

    def _update_preview(self) -> None:
        for i, name in enumerate(["e1", "e2", "e3", "e4"]):
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
        placement = list(self._get_current_placement())
        placement[a], placement[b] = placement[b], placement[a]
        perm = tuple(placement)

        # Check if it matches a preset
        matched = False
        for name, preset in PRESETS.items():
            if preset == perm:
                self._preset_combo.setCurrentText(name)
                matched = True
                break

        if not matched:
            self._custom_placement = perm  # type: ignore[attr-defined]
            self._preset_combo.setCurrentText("Custom (modified)")

        self._update_diagram()
        self._schedule_reconvert()

    def _on_preset_changed(self, name: str) -> None:
        if name in PRESETS:
            self._update_diagram()
            self._schedule_reconvert()

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
