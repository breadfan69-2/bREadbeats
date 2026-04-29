from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pmv_axis_converter import PRESET_CURVES

_CURVE_NAMES: dict[str, str] = {
    "linear": "Linear",
    "ease_in": "Ease In",
    "ease_out": "Ease Out",
    "bell": "Bell Curve",
    "inverted": "Inverted",
    "s_curve": "S-Curve",
    "sharp_peak": "Sharp Peak",
    "gentle_wave": "Gentle Wave",
}


def _interp_curve(points: list[tuple[float, float]], n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    ordered = sorted(points, key=lambda p: p[0])
    xs = np.array([p[0] for p in ordered], dtype=np.float64)
    ys = np.array([p[1] for p in ordered], dtype=np.float64)
    x_fine = np.linspace(0.0, 1.0, n)
    y_fine = np.interp(x_fine, xs, ys)
    return x_fine, y_fine


class CurveEditorWidget(pg.PlotWidget):
    """Interactive curve editor backed by pyqtgraph.

    Control points are displayed as draggable scatter dots.
    Left-click on empty area to add a point.
    Right-click on a point to remove it (minimum 2 kept).
    """

    points_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setBackground("#2d2d2d")
        self.setMouseEnabled(x=False, y=False)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setXRange(0, 100)
        self.setYRange(0, 100)
        self.setLabel("bottom", "Input Position (0–100)")
        self.setLabel("left", "Output Position (0–100)")

        self._curve_pen = QPen(QColor("#00cccc"), 2)
        self._curve_item = self.plot([], [], pen=self._curve_pen)

        self._scatter = pg.ScatterPlotItem(
            size=14,
            pen=pg.mkPen("#ffffff", width=1.5),
            brush=pg.mkBrush("#ff4444"),
            hoverable=True,
            hoverBrush=pg.mkBrush("#ffaa00"),
        )
        self.addItem(self._scatter)

        self._control_points: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
        self._dragging_index: int | None = None

        self._scatter.sigClicked.connect(self._on_scatter_clicked)
        scene = self.scene()
        if scene is not None:
            sig_mouse_moved = getattr(scene, "sigMouseMoved", None)
            if sig_mouse_moved is not None:
                sig_mouse_moved.connect(self._on_mouse_moved)
            sig_mouse_clicked = getattr(scene, "sigMouseClicked", None)
            if sig_mouse_clicked is not None:
                sig_mouse_clicked.connect(self._on_scene_clicked)

    # -- public API --

    def get_points(self) -> list[tuple[float, float]]:
        return list(self._control_points)

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._control_points = sorted(
            [(float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))) for x, y in points],
            key=lambda p: p[0],
        )
        if len(self._control_points) < 2:
            self._control_points = [(0.0, 0.0), (1.0, 1.0)]
        self._refresh()
        self.points_changed.emit()

    # -- internal drawing --

    def _refresh(self) -> None:
        pts = self._control_points
        x_fine, y_fine = _interp_curve(pts)
        self._curve_item.setData(x_fine * 100, y_fine * 100)

        self._scatter.setData(
            [p[0] * 100 for p in pts],
            [p[1] * 100 for p in pts],
        )

    # -- interaction --

    def _scene_to_data(self, scene_pos) -> tuple[float, float] | None:
        plot_item = getattr(self, "plotItem", None)
        if plot_item is None:
            return None
        vb = getattr(plot_item, "vb", None)
        if vb is None:
            return None
        mapped = vb.mapSceneToView(scene_pos)
        x, y = mapped.x() / 100.0, mapped.y() / 100.0
        if -0.05 <= x <= 1.05 and -0.05 <= y <= 1.05:
            return float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))
        return None

    def _on_scatter_clicked(self, _scatter, points, ev):
        if not points:
            return
        idx = points[0].index()
        if ev.button() == Qt.MouseButton.RightButton:
            if len(self._control_points) > 2:
                self._control_points.pop(idx)
                self._refresh()
                self.points_changed.emit()
        elif ev.button() == Qt.MouseButton.LeftButton:
            self._dragging_index = idx

    def _on_scene_clicked(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            if self._dragging_index is not None:
                self._dragging_index = None
                return
            coord = self._scene_to_data(ev.scenePos())
            if coord is None:
                return
            # check proximity to existing points
            for px, py in self._control_points:
                if abs(px - coord[0]) < 0.03 and abs(py - coord[1]) < 0.03:
                    return
            self._control_points.append(coord)
            self._control_points.sort(key=lambda p: p[0])
            self._refresh()
            self.points_changed.emit()
        elif ev.button() == Qt.MouseButton.RightButton:
            self._dragging_index = None

    def _on_mouse_moved(self, scene_pos):
        if self._dragging_index is None:
            return
        coord = self._scene_to_data(scene_pos)
        if coord is None:
            return
        self._control_points[self._dragging_index] = coord
        self._control_points.sort(key=lambda p: p[0])
        # find new index after sort
        for i, p in enumerate(self._control_points):
            if p is coord or (abs(p[0] - coord[0]) < 1e-9 and abs(p[1] - coord[1]) < 1e-9):
                self._dragging_index = i
                break
        self._refresh()
        self.points_changed.emit()

    def mouseReleaseEvent(self, ev):
        self._dragging_index = None
        super().mouseReleaseEvent(ev)


class CurvePointsList(QGroupBox):
    """Shows the list of control points with manual add/remove."""

    point_added = pyqtSignal(float, float)
    point_removed = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Control Points", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 10, 6, 6)

        self._list = QListWidget()
        self._list.setMaximumWidth(200)
        layout.addWidget(self._list)

        entry_row = QHBoxLayout()
        entry_row.setSpacing(4)
        entry_row.addWidget(QLabel("X:"))
        self._x_spin = QDoubleSpinBox()
        self._x_spin.setRange(0.0, 100.0)
        self._x_spin.setDecimals(1)
        self._x_spin.setSingleStep(1.0)
        entry_row.addWidget(self._x_spin)
        entry_row.addWidget(QLabel("Y:"))
        self._y_spin = QDoubleSpinBox()
        self._y_spin.setRange(0.0, 100.0)
        self._y_spin.setDecimals(1)
        self._y_spin.setSingleStep(1.0)
        entry_row.addWidget(self._y_spin)
        layout.addLayout(entry_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add)
        btn_row.addWidget(add_btn)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(rm_btn)
        layout.addLayout(btn_row)

    def update_points(self, points: list[tuple[float, float]]) -> None:
        self._list.clear()
        for x, y in points:
            self._list.addItem(f"({x * 100:.1f}, {y * 100:.1f})")

    def _on_add(self) -> None:
        x = self._x_spin.value() / 100.0
        y = self._y_spin.value() / 100.0
        self.point_added.emit(x, y)

    def _on_remove(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self.point_removed.emit(row)


class CurveEditorDialog(QDialog):
    """Modal dialog for editing a single electrode's response curve.

    Returns the selected preset name (str) or ``"custom"`` and the
    control points list.
    """

    def __init__(
        self,
        electrode_name: str,
        current_curve: str,
        custom_points: list[tuple[float, float]] | None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Curve Editor — {electrode_name.upper()}")
        self.setMinimumSize(780, 480)
        self.resize(860, 520)

        self._electrode = electrode_name
        self._original_curve = current_curve
        self._original_points = copy.deepcopy(custom_points) if custom_points else None

        self._result_curve: str | None = None
        self._result_points: list[tuple[float, float]] | None = None

        self._build_ui()
        self._load_initial(current_curve, custom_points)

    # -- build UI --------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # Left: preset list
        preset_panel = QWidget()
        preset_layout = QVBoxLayout(preset_panel)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.addWidget(QLabel("Presets"))
        self._preset_list = QListWidget()
        for key in PRESET_CURVES:
            self._preset_list.addItem(_CURVE_NAMES.get(key, key))
        self._preset_list.itemClicked.connect(self._on_preset_clicked)
        preset_layout.addWidget(self._preset_list)
        splitter.addWidget(preset_panel)

        # Center: interactive plot
        self._editor = CurveEditorWidget()
        splitter.addWidget(self._editor)

        # Right: points list
        self._points_panel = CurvePointsList()
        splitter.addWidget(self._points_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([160, 480, 180])

        # Instructions
        info = QLabel(
            "Left-click: add point  |  Drag: move point  |  Right-click: remove point  |  Min 2 points"
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(info)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(reset_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        save_btn = QPushButton("Save Curve")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)
        root.addLayout(btn_row)

        # Wire signals
        self._editor.points_changed.connect(self._sync_points_list)
        self._points_panel.point_added.connect(self._on_manual_add)
        self._points_panel.point_removed.connect(self._on_manual_remove)

    # -- preset / data loading -------------------------------------------

    def _load_initial(self, curve: str, custom_points: list[tuple[float, float]] | None) -> None:
        if custom_points:
            self._editor.set_points(custom_points)
            self._select_preset_in_list(None)
        elif curve in PRESET_CURVES:
            self._editor.set_points(PRESET_CURVES[curve])
            self._select_preset_in_list(curve)
        else:
            self._editor.set_points(PRESET_CURVES["linear"])
            self._select_preset_in_list("linear")
        self._sync_points_list()

    def _select_preset_in_list(self, key: str | None) -> None:
        self._preset_list.blockSignals(True)
        if key is None:
            self._preset_list.clearSelection()
        else:
            keys = list(PRESET_CURVES.keys())
            if key in keys:
                self._preset_list.setCurrentRow(keys.index(key))
        self._preset_list.blockSignals(False)

    def _on_preset_clicked(self, item) -> None:
        idx = self._preset_list.row(item)
        keys = list(PRESET_CURVES.keys())
        if 0 <= idx < len(keys):
            self._editor.set_points(PRESET_CURVES[keys[idx]])

    # -- points list sync ------------------------------------------------

    def _sync_points_list(self) -> None:
        self._points_panel.update_points(self._editor.get_points())

    def _on_manual_add(self, x: float, y: float) -> None:
        pts = self._editor.get_points()
        pts.append((x, y))
        self._editor.set_points(pts)

    def _on_manual_remove(self, idx: int) -> None:
        pts = self._editor.get_points()
        if len(pts) > 2 and 0 <= idx < len(pts):
            pts.pop(idx)
            self._editor.set_points(pts)

    # -- buttons ---------------------------------------------------------

    def _on_reset(self) -> None:
        self._load_initial(self._original_curve, self._original_points)

    def _on_save(self) -> None:
        pts = self._editor.get_points()
        if len(pts) < 2:
            QMessageBox.warning(self, "Invalid", "Need at least 2 control points.")
            return
        # check if points match a known preset
        matched_preset: str | None = None
        for name, preset_pts in PRESET_CURVES.items():
            if len(pts) == len(preset_pts):
                if all(
                    abs(a[0] - b[0]) < 0.005 and abs(a[1] - b[1]) < 0.005
                    for a, b in zip(pts, preset_pts)
                ):
                    matched_preset = name
                    break
        self._result_curve = matched_preset if matched_preset else "custom"
        self._result_points = pts if matched_preset is None else None
        self.accept()

    # -- public result ---------------------------------------------------

    def result_curve(self) -> str | None:
        return self._result_curve

    def result_points(self) -> list[tuple[float, float]] | None:
        return self._result_points
