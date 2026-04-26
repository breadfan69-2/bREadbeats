from __future__ import annotations

import bisect
import importlib
import importlib.util
import time
from typing import Callable, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from funscript_edit_state import FunscriptEditState

from pmv_audio_analysis import AudioTimeline
from pmv_axis_converter import MultiAxisResult
from pmv_beat_engine import BeatCandidate, BeatTimeline
from pmv_position_mapper import PositionTimeline


_soundfile_spec = importlib.util.find_spec("soundfile")
if _soundfile_spec is not None:  # pragma: no cover - optional dependency
    _sf = importlib.import_module("soundfile")
    _HAS_SOUNDFILE = True
else:
    _sf = None
    _HAS_SOUNDFILE = False

_sounddevice_spec = importlib.util.find_spec("sounddevice")
if _sounddevice_spec is not None:  # pragma: no cover - optional dependency
    _sd = importlib.import_module("sounddevice")
    _HAS_SOUNDDEVICE = True
else:
    _sd = None
    _HAS_SOUNDDEVICE = False


def _style_plot(widget: pg.PlotWidget) -> None:
    widget.setBackground("#2f2f2f")
    widget.showGrid(x=True, y=True, alpha=0.12)
    widget.getAxis("bottom").setTextPen(pg.mkPen("#c8c8c8"))
    widget.getAxis("left").setTextPen(pg.mkPen("#c8c8c8"))


class TimeAxisSync:
    """Synchronize X range across all registered plot widgets."""

    def __init__(self):
        self._plots: list[pg.PlotWidget] = []
        self._syncing = False

    def register_plot(self, plot: pg.PlotWidget) -> None:
        if plot in self._plots:
            return
        self._plots.append(plot)
        plot.getViewBox().sigXRangeChanged.connect(lambda _vb, rng, src=plot: self._on_range_changed(src, rng))

    def _on_range_changed(self, source: pg.PlotWidget, x_range: tuple[float, float]) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            lo, hi = float(x_range[0]), float(x_range[1])
            for plot in self._plots:
                if plot is source:
                    continue
                plot.getViewBox().setXRange(lo, hi, padding=0.0)
        finally:
            self._syncing = False

    def set_view_range(self, start_ms: float, end_ms: float) -> None:
        lo = float(min(start_ms, end_ms))
        hi = float(max(start_ms, end_ms))
        if hi <= lo:
            hi = lo + 1.0

        self._syncing = True
        try:
            for plot in self._plots:
                plot.getViewBox().setXRange(lo, hi, padding=0.0)
        finally:
            self._syncing = False

    def zoom(self, center_ms: float, factor: float) -> None:
        if not self._plots:
            return
        ref = self._plots[0].viewRange()[0]
        span = max(1.0, float(ref[1] - ref[0]))
        new_span = max(1.0, span / max(1e-3, float(factor)))
        lo = float(center_ms) - (new_span * 0.5)
        hi = float(center_ms) + (new_span * 0.5)
        self.set_view_range(lo, hi)

    def scroll(self, delta_ms: float) -> None:
        if not self._plots:
            return
        ref = self._plots[0].viewRange()[0]
        self.set_view_range(ref[0] + float(delta_ms), ref[1] + float(delta_ms))


class WaveformPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.plot = pg.PlotWidget()
        _style_plot(self.plot)
        self.plot.setLabel("left", "Amplitude")
        self.plot.setLabel("bottom", "Time (ms)")

        self.wave_curve = self.plot.plot([], [], pen=pg.mkPen("#8ad4ff", width=1.2),
                                          clipToView=True, autoDownsample=True, downsampleMethod="peak")
        self.beat_scatter = pg.ScatterPlotItem(size=6)
        self.plot.addItem(self.beat_scatter)

        self.playhead_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#ffe082", width=1))
        self.plot.addItem(self.playhead_line)

        self._duration_ms = 0.0
        layout.addWidget(self.plot)

    def set_waveform(self, samples: np.ndarray, sr: int) -> None:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.size == 0 or sr <= 0:
            self.wave_curve.setData([], [])
            self._duration_ms = 0.0
            return

        max_points = 3000
        step = max(1, int(np.ceil(arr.size / max_points)))
        y = arr[::step]
        x = (np.arange(y.size, dtype=np.float64) * step / float(sr)) * 1000.0

        self._duration_ms = float(arr.size / float(sr) * 1000.0)
        self.wave_curve.setData(x, y)
        self.plot.getViewBox().setXRange(0.0, max(1000.0, self._duration_ms), padding=0.0)
        self.plot.getViewBox().setYRange(-1.05, 1.05, padding=0.0)

    def set_beats(self, beats: list[BeatCandidate]) -> None:
        if not beats:
            self.beat_scatter.setData([], [])
            return

        brush_map = {
            "downbeat": pg.mkBrush("#ff6e6e"),
            "beat": pg.mkBrush("#6ec6ff"),
            "syncopation": pg.mkBrush("#8df58d"),
        }

        spots = []
        for beat in beats:
            spots.append(
                {
                    "pos": (float(beat.time_ms), 0.0),
                    "brush": brush_map.get(str(beat.beat_type), pg.mkBrush("#d0d0d0")),
                    "size": 7,
                }
            )
        self.beat_scatter.setData(spots)

    def set_cursor(self, time_ms: float) -> None:
        self.playhead_line.setPos(float(time_ms))


class SpectralFluxPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.plot = pg.PlotWidget()
        _style_plot(self.plot)
        self.plot.setLabel("left", "Flux")
        self.plot.setLabel("bottom", "Time (ms)")

        _ds = dict(clipToView=True, autoDownsample=True, downsampleMethod="peak")
        self.flux_curve = self.plot.plot([], [], pen=pg.mkPen("#74c0fc", width=1.4), **_ds)
        self.threshold_curve = self.plot.plot([], [], pen=pg.mkPen("#ffb74d", width=1.0, style=Qt.PenStyle.DashLine), **_ds)
        self.peak_scatter = pg.ScatterPlotItem(size=6, brush=pg.mkBrush("#fdd835"))
        self.plot.addItem(self.peak_scatter)

        self.playhead_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#ffe082", width=1))
        self.plot.addItem(self.playhead_line)

        layout.addWidget(self.plot)

    def set_features(self, timeline: AudioTimeline) -> None:
        t = np.asarray(timeline.frame_times_ms, dtype=np.float64)
        flux = np.asarray(timeline.spectral_flux_per_frame, dtype=np.float64)
        n = min(len(t), len(flux))
        if n == 0:
            self.flux_curve.setData([], [])
            self.threshold_curve.setData([], [])
            self.peak_scatter.setData([], [])
            return

        t = t[:n]
        flux = flux[:n]
        self.flux_curve.setData(t, flux)

        width = 9
        kernel = np.ones(width, dtype=np.float64) / float(width)
        padded = np.pad(flux, (width // 2, width - 1 - width // 2), mode="edge")
        local_mean = np.convolve(padded, kernel, mode="valid")
        threshold = local_mean * 1.08
        self.threshold_curve.setData(t, threshold)

        peak_idx: list[int] = []
        for i in range(1, n - 1):
            if flux[i] > threshold[i] and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
                peak_idx.append(i)
        self.peak_scatter.setData(t[peak_idx], flux[peak_idx])

    def set_cursor(self, time_ms: float) -> None:
        self.playhead_line.setPos(float(time_ms))


class PositionTimelinePanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.plot = pg.PlotWidget()
        _style_plot(self.plot)
        self.plot.setLabel("left", "Position")
        self.plot.setLabel("bottom", "Time (ms)")
        self.plot.getViewBox().setYRange(0.0, 100.0, padding=0.0)

        self.main_curve = self.plot.plot([], [], pen=pg.mkPen("#fff176", width=1.6),
                                          clipToView=True, autoDownsample=True, downsampleMethod="peak")
        self.extra_curves: dict[str, pg.PlotDataItem] = {}

        self.playhead_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#ffe082", width=1))
        self.plot.addItem(self.playhead_line)

        layout.addWidget(self.plot)

    def set_positions(self, positions: PositionTimeline) -> None:
        if not positions.actions:
            self.main_curve.setData([], [])
            return
        t = np.array([float(a.at) for a in positions.actions], dtype=np.float64)
        y = np.array([float(a.pos) for a in positions.actions], dtype=np.float64)
        self.main_curve.setData(t, y)

    def set_multi_axis(self, result: MultiAxisResult) -> None:
        palette = {
            "alpha": "#4dd0e1",
            "beta": "#ff80ab",
            "e1": "#90caf9",
            "e2": "#a5d6a7",
            "e3": "#ffcc80",
            "e4": "#ce93d8",
        }

        keep = set()
        for axis_name, actions in result.axes.items():
            if axis_name in {"main"}:
                continue
            if axis_name not in palette:
                continue
            keep.add(axis_name)
            if axis_name not in self.extra_curves:
                curve = self.plot.plot([], [], pen=pg.mkPen(palette[axis_name], width=1.0))
                self.extra_curves[axis_name] = curve
            if actions:
                t = np.array([float(a.at) for a in actions], dtype=np.float64)
                y = np.array([float(a.pos) for a in actions], dtype=np.float64)
                self.extra_curves[axis_name].setData(t, y)
            else:
                self.extra_curves[axis_name].setData([], [])

        for axis_name in list(self.extra_curves.keys()):
            if axis_name in keep:
                continue
            curve = self.extra_curves.pop(axis_name)
            self.plot.removeItem(curve)

    def set_cursor(self, time_ms: float) -> None:
        self.playhead_line.setPos(float(time_ms))


class SpeedHeatmapPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.plot = pg.PlotWidget()
        _style_plot(self.plot)
        self.plot.setLabel("left", "Speed")
        self.plot.setLabel("bottom", "Time (ms)")

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        self.playhead_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#ffe082", width=1))
        self.plot.addItem(self.playhead_line)

        layout.addWidget(self.plot)

    def set_speed_profile(self, speed: np.ndarray, times_ms: np.ndarray) -> None:
        s = np.asarray(speed, dtype=np.float64)
        t = np.asarray(times_ms, dtype=np.float64)
        n = min(len(s), len(t))
        if n == 0:
            self.image_item.setImage(np.zeros((1, 1), dtype=np.float32), autoLevels=False)
            return

        s = np.clip(s[:n], 0.0, np.percentile(s[:n], 99) if n > 1 else 1.0)
        max_s = float(np.max(s)) if np.max(s) > 1e-9 else 1.0
        norm = (s / max_s).astype(np.float32)
        image = np.expand_dims(norm, axis=0)

        self.image_item.setImage(image, autoLevels=False, levels=(0.0, 1.0))
        cmap = pg.colormap.get("CET-L9") or pg.colormap.get("viridis")
        if cmap is not None:
            lut = np.asarray(cmap.getLookupTable(), dtype=np.float64)
            self.image_item.setLookupTable(lut)

        t0 = float(t[0])
        t1 = float(t[n - 1])
        width = max(1.0, t1 - t0)
        self.image_item.setRect(pg.QtCore.QRectF(t0, 0.0, width, 1.0))
        self.plot.getViewBox().setYRange(0.0, 1.0, padding=0.0)

    def set_cursor(self, time_ms: float) -> None:
        self.playhead_line.setPos(float(time_ms))


class PlaybackPanel(QWidget):
    """Lightweight transport scaffold that drives cursor sync in visualization panels."""

    position_changed = pyqtSignal(float)
    transport_changed = pyqtSignal(str, float)  # ("play"|"pause"|"stop"|"seek", position_ms)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._duration_ms = 0.0
        self._position_ms = 0.0
        self._playing = False
        self._audio_samples = np.array([], dtype=np.float32)
        self._audio_sr = 0
        self._playback_t0 = 0.0
        self._playback_ref_pos_ms = 0.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.time_label = QLabel("00:00 / 00:00")

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setValue(0)
        self._slider_held = False

        layout.addWidget(self.play_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.seek_slider, 1)
        layout.addWidget(self.time_label)

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60fps for smooth playback
        self._timer.timeout.connect(self._tick)

        self.play_btn.clicked.connect(self.play)
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.valueChanged.connect(self._on_seek_slider)

    @staticmethod
    def _fmt(ms: float) -> str:
        secs = max(0, int(round(ms / 1000.0)))
        m = secs // 60
        s = secs % 60
        return f"{m:02d}:{s:02d}"

    def _update_label(self) -> None:
        self.time_label.setText(f"{self._fmt(self._position_ms)} / {self._fmt(self._duration_ms)}")

    def _emit_position(self) -> None:
        self.position_changed.emit(float(self._position_ms))

    def _refresh_position_from_clock(self) -> None:
        if not self._playing:
            return
        elapsed_ms = (time.perf_counter() - self._playback_t0) * 1000.0
        self._position_ms = min(self._duration_ms, self._playback_ref_pos_ms + elapsed_ms)

    def _start_audio_from_position(self) -> bool:
        if not _HAS_SOUNDDEVICE or _sd is None:
            return False
        if self._audio_sr <= 0 or len(self._audio_samples) <= 0:
            return False

        start_idx = int(round((self._position_ms / 1000.0) * float(self._audio_sr)))
        start_idx = int(np.clip(start_idx, 0, len(self._audio_samples)))
        chunk = self._audio_samples[start_idx:]
        if len(chunk) <= 0:
            return False

        try:
            _sd.stop()
            _sd.play(chunk, self._audio_sr, blocking=False)
            return True
        except Exception:
            return False

    def _tick(self) -> None:
        if not self._playing:
            return
        self._refresh_position_from_clock()
        self._sync_slider_from_position()
        self._update_label()
        self._emit_position()
        if self._position_ms >= self._duration_ms:
            self._playing = False
            self._timer.stop()

    def _on_slider_pressed(self) -> None:
        self._slider_held = True

    def _on_slider_released(self) -> None:
        self._slider_held = False
        self._on_seek_slider(self.seek_slider.value())

    def _sync_slider_from_position(self) -> None:
        if self._slider_held:
            return
        if self._duration_ms <= 0:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(0)
            self.seek_slider.blockSignals(False)
            return

        value = int(round((self._position_ms / self._duration_ms) * 1000.0))
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(int(np.clip(value, 0, 1000)))
        self.seek_slider.blockSignals(False)

    def _on_seek_slider(self, value: int) -> None:
        if self._duration_ms <= 0:
            return
        self._position_ms = float(value) / 1000.0 * self._duration_ms
        if self._playing:
            if self._start_audio_from_position():
                self._playback_ref_pos_ms = self._position_ms
                self._playback_t0 = time.perf_counter()
            else:
                self._playing = False
                self._timer.stop()
        self._update_label()
        self._emit_position()
        self.transport_changed.emit("seek", self._position_ms)

    def set_duration_ms(self, duration_ms: float) -> None:
        if self._playing:
            self.pause()
        self._duration_ms = max(0.0, float(duration_ms))
        self._position_ms = 0.0
        self._sync_slider_from_position()
        self._update_label()
        self._emit_position()

    def set_audio_buffer(self, samples: np.ndarray, sr: int) -> None:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.mean(arr, axis=1).astype(np.float32, copy=False)
        elif arr.ndim > 2:
            arr = np.ravel(arr).astype(np.float32, copy=False)

        self._audio_samples = arr.copy()
        self._audio_sr = int(max(0, sr))
        duration_ms = (len(self._audio_samples) / float(max(1, self._audio_sr))) * 1000.0
        self.set_duration_ms(duration_ms)

    def load_audio(self, file_path: str) -> None:
        if _HAS_SOUNDFILE and _sf is not None:
            try:
                info = _sf.info(file_path)
                duration_ms = float(info.duration) * 1000.0
                self.set_duration_ms(duration_ms)
                return
            except Exception:
                return

    def play(self) -> None:
        if self._duration_ms <= 0.0:
            return

        if self._position_ms >= self._duration_ms:
            self._position_ms = 0.0

        if not self._start_audio_from_position():
            self.time_label.setText(f"{self._fmt(self._position_ms)} / {self._fmt(self._duration_ms)} (audio unavailable)")
            return

        self._playing = True
        self._playback_ref_pos_ms = self._position_ms
        self._playback_t0 = time.perf_counter()
        if not self._timer.isActive():
            self._timer.start()
        self.transport_changed.emit("play", self._position_ms)

    def pause(self) -> None:
        self._refresh_position_from_clock()
        self._playing = False
        if _HAS_SOUNDDEVICE and _sd is not None:
            try:
                _sd.stop()
            except Exception:
                pass
        self._sync_slider_from_position()
        self._update_label()
        self._emit_position()
        self._timer.stop()
        self.transport_changed.emit("pause", self._position_ms)

    def stop(self) -> None:
        self._playing = False
        if _HAS_SOUNDDEVICE and _sd is not None:
            try:
                _sd.stop()
            except Exception:
                pass
        self._position_ms = 0.0
        self._sync_slider_from_position()
        self._update_label()
        self._emit_position()
        self._timer.stop()
        self.transport_changed.emit("stop", 0.0)

    def seek(self, time_ms: float) -> None:
        self._position_ms = float(np.clip(time_ms, 0.0, max(0.0, self._duration_ms)))
        if self._playing:
            if self._start_audio_from_position():
                self._playback_ref_pos_ms = self._position_ms
                self._playback_t0 = time.perf_counter()
            else:
                self._playing = False
                self._timer.stop()
        self._sync_slider_from_position()
        self._update_label()
        self._emit_position()


# ---------------------------------------------------------------------------
# LOD helper: stores full-resolution (x, y) numpy arrays and returns a
# downsampled slice for the current viewport.  Keeps any curve to ~MAX_PTS
# visible points so pyqtgraph never has to paint more than a few thousand.
# ---------------------------------------------------------------------------
_LOD_MAX_PTS = 4000


def _lod_slice(
    x_full: np.ndarray,
    y_full: np.ndarray,
    view_lo: float,
    view_hi: float,
    max_pts: int = _LOD_MAX_PTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) downsampled to *max_pts* within [view_lo, view_hi]."""
    if x_full.size == 0:
        return x_full, y_full
    i0 = int(np.searchsorted(x_full, view_lo, side="left"))
    i1 = int(np.searchsorted(x_full, view_hi, side="right"))
    # Include one extra sample each side so lines connect at viewport edge
    i0 = max(0, i0 - 1)
    i1 = min(len(x_full), i1 + 1)
    xv = x_full[i0:i1]
    yv = y_full[i0:i1]
    if len(xv) <= max_pts:
        return xv, yv
    # Peak-preserving downsample: keep min & max per bucket
    step = max(1, len(xv) // (max_pts // 2))
    n_buckets = len(xv) // step
    trunc = n_buckets * step
    yr = yv[:trunc].reshape(n_buckets, step)
    idx_min = yr.argmin(axis=1)
    idx_max = yr.argmax(axis=1)
    bucket_idx = np.arange(n_buckets, dtype=np.intp) * step
    # Interleave min/max keeping time order (vectorised)
    i_min = bucket_idx + idx_min
    i_max = bucket_idx + idx_max
    # Ensure earlier index comes first in each pair
    first = np.minimum(i_min, i_max)
    second = np.maximum(i_min, i_max)
    indices = np.empty(2 * n_buckets, dtype=np.intp)
    indices[0::2] = first
    indices[1::2] = second
    return xv[indices], yv[indices]


class VisualizationArea(QWidget):
    """Container for a single overlaid PMV timeline plot with trace toggles."""

    position_changed = pyqtSignal(float)
    edit_mode_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._duration_ms = 0.0
        self._view_span_ms = 0.0
        self._nav_syncing = False
        self._auto_follow_playhead = True
        self._lod_refreshing = False  # re-entrancy guard
        self._last_follow_scroll_t = 0.0

        # Edit-mode state
        self._edit_mode = False
        self._edit_state: FunscriptEditState | None = None
        self._drag_idx: int | None = None
        self._drag_active = False
        self._rect_selecting = False
        self._rect_start_pos = None  # (time_ms, pos) of rect select start
        self._lock_overlays: list[pg.LinearRegionItem] = []

        # Non-edit-mode navigation drag state
        self._nav_dragging = False
        self._nav_press_time_ms: float | None = None
        self._nav_last_x: float | None = None

        # Full-resolution data stores for LOD rendering
        self._lod_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Beat stores: {beat_type: x_array}
        self._beat_data: dict[str, np.ndarray] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)
        layout.addLayout(toolbar)

        self.overlay_plot = pg.PlotWidget(self)
        _style_plot(self.overlay_plot)
        self.overlay_plot.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.overlay_plot.installEventFilter(self)
        self.overlay_plot.viewport().installEventFilter(self)
        self.overlay_plot.setLabel("left", "Normalized / Position")
        self.overlay_plot.getAxis("bottom").setTicks([])
        self.overlay_plot.getAxis("bottom").setStyle(showValues=False)
        self.overlay_plot.setLabel("bottom", "")
        vb = self.overlay_plot.getViewBox()
        vb.setYRange(0.0, 100.0, padding=0.0)
        vb.setMouseEnabled(x=True, y=False)
        vb.setLimits(yMin=0.0, yMax=100.0)
        vb.disableAutoRange()  # LOD manages the X range; prevent setData from auto-ranging

        self.wave_curve = self.overlay_plot.plot([], [], pen=pg.mkPen("#8ad4ff", width=1.0), name="Waveform")
        self.flux_curve = self.overlay_plot.plot([], [], pen=pg.mkPen("#74c0fc", width=1.4), name="Flux")
        self.position_curve = self.overlay_plot.plot([], [], pen=pg.mkPen("#fff176", width=1.7), name="Main Position")
        self.speed_curve = self.overlay_plot.plot([], [], pen=pg.mkPen("#ffb74d", width=1.2), name="Speed")

        # Per-type beat marker curves (no symbols — we use plain curves with LOD)
        self._beat_curves: dict[str, pg.PlotDataItem] = {}
        for btype, color in [("downbeat", "#ff6e6e"), ("beat", "#6ec6ff"), ("syncopation", "#8df58d")]:
            c = self.overlay_plot.plot([], [], pen=None,
                                       symbol="o", symbolSize=7, symbolBrush=pg.mkBrush(color),
                                       symbolPen=None)
            self._beat_curves[btype] = c
        self.beat_scatter = None

        self.extra_curves: dict[str, pg.PlotDataItem] = {}
        self._extra_curves_visible = True

        self.playhead_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#ffe082", width=1))
        self.overlay_plot.addItem(self.playhead_line)

        self._playhead_label = pg.TextItem(text="0.00s", color="#ffe082", anchor=(0.5, 1.0))
        self._playhead_label.setFont(QFont("Consolas", 9))
        self._playhead_label.setZValue(20)
        self.overlay_plot.addItem(self._playhead_label)
        self._playhead_label.setPos(0.0, 0.0)

        # Edit-mode scatter overlays (all points + selected points)
        self._edit_scatter = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen("#fff176", width=1),
            brush=pg.mkBrush("#fff176"),
            hoverable=True,
            hoverSize=12,
            hoverBrush=pg.mkBrush("#ffffff"),
        )
        self._edit_scatter.setVisible(False)
        self._edit_scatter.setZValue(10)
        self.overlay_plot.addItem(self._edit_scatter)

        self._selection_scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen("#42a5f5", width=2),
            brush=pg.mkBrush("#42a5f580"),
        )
        self._selection_scatter.setVisible(False)
        self._selection_scatter.setZValue(11)
        self.overlay_plot.addItem(self._selection_scatter)

        # Rectangle-selection overlay
        self._rect_roi = pg.LinearRegionItem(
            values=(0, 0), orientation="vertical",
            brush=pg.mkBrush(66, 165, 245, 30),
            pen=pg.mkPen("#42a5f5", width=1, style=Qt.PenStyle.DashLine),
            movable=False,
        )
        self._rect_roi.setVisible(False)
        self._rect_roi.setZValue(5)
        self.overlay_plot.addItem(self._rect_roi)

        self.playback_panel = PlaybackPanel(self)

        self._trace_names = ("Waveform", "Flux", "Main", "Speed", "Beats", "Aux")

        self._toggle_buttons: dict[str, QPushButton] = {}
        for name in self._trace_names:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.toggled.connect(lambda checked, trace=name: self._set_trace_visible(trace, checked))
            toolbar.addWidget(btn)
            self._toggle_buttons[name] = btn

        self.zoom_in_btn = QPushButton("+")
        self.zoom_out_btn = QPushButton("-")
        self.fit_btn = QPushButton("Fit")
        self.follow_btn = QPushButton("Follow")
        self.follow_btn.setCheckable(True)
        self.follow_btn.setChecked(True)
        self.follow_btn.toggled.connect(self._on_follow_toggled)
        self.zoom_in_btn.clicked.connect(lambda: self._zoom(1.6))
        self.zoom_out_btn.clicked.connect(lambda: self._zoom(1.0 / 1.6))
        self.fit_btn.clicked.connect(self._fit_to_duration)
        toolbar.addWidget(self.zoom_in_btn)
        toolbar.addWidget(self.zoom_out_btn)
        toolbar.addWidget(self.fit_btn)
        toolbar.addWidget(self.follow_btn)

        # ── Edit mode toggle ──
        self._edit_mode_btn = QPushButton("Edit")
        self._edit_mode_btn.setCheckable(True)
        self._edit_mode_btn.setToolTip("Toggle interactive point editing (select, move, add, delete)")
        self._edit_mode_btn.toggled.connect(self._on_edit_mode_toggled)
        toolbar.addWidget(self._edit_mode_btn)

        toolbar.addStretch(1)

        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(6)
        nav_row.addWidget(QLabel("Scroll"))

        self.nav_slider = QSlider(Qt.Orientation.Horizontal)
        self.nav_slider.setRange(0, 1000)
        self.nav_slider.setValue(0)
        self.nav_slider.valueChanged.connect(self._on_nav_slider_changed)
        nav_row.addWidget(self.nav_slider, 1)

        self.nav_label = QLabel("00:00-00:00")
        nav_row.addWidget(self.nav_label)

        layout.addWidget(self.overlay_plot, 1)
        layout.addLayout(nav_row)
        layout.addWidget(self.playback_panel, 0)

        self.playback_panel.position_changed.connect(self._on_playback_position)
        self.overlay_plot.getViewBox().sigXRangeChanged.connect(self._on_x_range_changed)

    def _set_trace_visible(self, trace_name: str, visible: bool) -> None:
        if trace_name == "Aux":
            self._extra_curves_visible = bool(visible)
            for curve in self.extra_curves.values():
                curve.setVisible(self._extra_curves_visible)
            return
        if trace_name == "Waveform":
            self.wave_curve.setVisible(bool(visible))
        elif trace_name == "Flux":
            self.flux_curve.setVisible(bool(visible))
        elif trace_name == "Main":
            self.position_curve.setVisible(bool(visible))
        elif trace_name == "Speed":
            self.speed_curve.setVisible(bool(visible))
        elif trace_name == "Beats":
            for c in self._beat_curves.values():
                c.setVisible(bool(visible))

    def _on_playback_position(self, time_ms: float) -> None:
        self.set_playback_position(time_ms)
        self._auto_scroll_to_playhead(float(time_ms))
        self.position_changed.emit(float(time_ms))

    def _on_follow_toggled(self, checked: bool) -> None:
        self._auto_follow_playhead = bool(checked)

    def _auto_scroll_to_playhead(self, time_ms: float) -> None:
        if not self._auto_follow_playhead:
            return
        x_range = self.overlay_plot.viewRange()[0]
        lo, hi = float(x_range[0]), float(x_range[1])
        force_scroll = (time_ms < lo) or (time_ms > hi)

        now = time.perf_counter()
        if (not force_scroll) and ((now - self._last_follow_scroll_t) < 0.05):  # ~20 FPS follow updates
            return
        self._last_follow_scroll_t = now

        span = max(1.0, float(x_range[1]) - float(x_range[0]))

        # Keep playhead near center while avoiding costly per-frame relayout.
        new_start = time_ms - span * 0.5
        current_start = float(x_range[0])
        if abs(new_start - current_start) < 1.0:
            return
        self._apply_view_range(new_start, new_start + span)
        self._sync_nav_from_view()

    @staticmethod
    def _fmt_ms(ms: float) -> str:
        secs = max(0, int(round(ms / 1000.0)))
        m = secs // 60
        s = secs % 60
        return f"{m:02d}:{s:02d}"

    def _apply_view_range(self, start_ms: float, end_ms: float) -> None:
        if self._duration_ms > 0.0:
            span = max(1.0, float(end_ms) - float(start_ms))
            span = min(span, self._duration_ms)
            max_start = max(0.0, self._duration_ms - span)
            start = float(np.clip(float(start_ms), 0.0, max_start))
            end = start + span
        else:
            start = float(min(start_ms, end_ms))
            end = float(max(start_ms, end_ms))
            if end <= start:
                end = start + 1.0

        self.overlay_plot.getViewBox().setXRange(start, end, padding=0.0)

    def _sync_nav_from_view(self) -> None:
        if self._nav_syncing:
            return

        x_range = self.overlay_plot.viewRange()[0]
        start = float(x_range[0])
        end = float(x_range[1])
        span = max(1.0, end - start)
        self._view_span_ms = span
        self.nav_label.setText(f"{self._fmt_ms(start)}-{self._fmt_ms(end)}")

        self._nav_syncing = True
        try:
            if self._duration_ms <= span + 1e-6:
                self.nav_slider.setValue(0)
                return
            max_start = max(1e-6, self._duration_ms - span)
            value = int(round(np.clip(start / max_start, 0.0, 1.0) * 1000.0))
            self.nav_slider.setValue(value)
        finally:
            self._nav_syncing = False

    def _on_nav_slider_changed(self, value: int) -> None:
        if self._nav_syncing:
            return
        if self._duration_ms <= 0.0:
            return

        span = max(1.0, min(self._view_span_ms, self._duration_ms))
        max_start = max(0.0, self._duration_ms - span)
        start = (float(value) / 1000.0) * max_start
        self._apply_view_range(start, start + span)

    def _on_x_range_changed(self, _vb: object, _rng: object) -> None:
        self._sync_nav_from_view()
        self._refresh_lod()

    def _refresh_lod(self) -> None:
        """Downsample all stored data to the current viewport and push to curves."""
        if self._lod_refreshing:
            return
        self._lod_refreshing = True
        try:
            self._do_refresh_lod()
        finally:
            self._lod_refreshing = False

    def _do_refresh_lod(self) -> None:
        x_range = self.overlay_plot.viewRange()[0]
        lo, hi = float(x_range[0]), float(x_range[1])

        _curve_map: dict[str, pg.PlotDataItem] = {
            "wave": self.wave_curve,
            "flux": self.flux_curve,
            "position": self.position_curve,
            "speed": self.speed_curve,
        }

        for key, curve in _curve_map.items():
            if not curve.isVisible():
                continue
            full = self._lod_data.get(key)
            if full is None:
                continue
            x_full, y_full = full
            if x_full.size == 0:
                continue
            xs, ys = _lod_slice(x_full, y_full, lo, hi)
            curve.setData(xs, ys)

        # Extra axis curves
        for axis_name, curve in self.extra_curves.items():
            if not curve.isVisible():
                continue
            full = self._lod_data.get(f"extra_{axis_name}")
            if full is None:
                continue
            x_full, y_full = full
            if x_full.size == 0:
                continue
            xs, ys = _lod_slice(x_full, y_full, lo, hi)
            curve.setData(xs, ys)

        # Beat marker curves — simple viewport clip (beats are sparse enough)
        for btype, curve in self._beat_curves.items():
            if not curve.isVisible():
                continue
            x_full = self._beat_data.get(btype)
            if x_full is None or x_full.size == 0:
                continue
            i0 = max(0, int(np.searchsorted(x_full, lo, side="left")) - 1)
            i1 = min(len(x_full), int(np.searchsorted(x_full, hi, side="right")) + 1)
            xv = x_full[i0:i1]
            if xv.size == 0:
                curve.setData([], [])
            else:
                curve.setData(xv, np.full(len(xv), 98.0, dtype=np.float64))

    def _zoom(self, factor: float) -> None:
        x_range = self.overlay_plot.viewRange()[0]
        center = float(x_range[0] + x_range[1]) * 0.5
        span = max(1.0, float(x_range[1] - x_range[0]))
        if factor > 1.0:
            new_span = span / factor
        else:
            new_span = span / max(1e-6, factor)

        if self._duration_ms > 0.0:
            new_span = float(np.clip(new_span, 500.0, self._duration_ms))
        lo = center - (new_span * 0.5)
        hi = center + (new_span * 0.5)
        self._apply_view_range(lo, hi)

    def _fit_to_duration(self) -> None:
        if self._duration_ms <= 0.0:
            self._apply_view_range(0.0, 1000.0)
            return
        self._apply_view_range(0.0, self._duration_ms)

    def set_audio_data(self, samples: np.ndarray, sr: int) -> None:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.size == 0 or sr <= 0:
            self._lod_data.pop("wave", None)
            self.wave_curve.setData([], [])
            self._duration_ms = 0.0
            self.playback_panel.set_duration_ms(0.0)
            self._sync_nav_from_view()
            return

        max_points = 50000  # store denser; LOD will downsample on render
        step = max(1, int(np.ceil(arr.size / max_points)))
        y = arr[::step]
        x = (np.arange(y.size, dtype=np.float64) * step / float(sr)) * 1000.0
        y_norm = np.clip(50.0 + (45.0 * y.astype(np.float64)), 0.0, 100.0)
        self._lod_data["wave"] = (x, y_norm)

        self.playback_panel.set_audio_buffer(arr, int(sr))
        duration_ms = (len(arr) / float(max(1, sr))) * 1000.0 if len(arr) else 0.0
        self._duration_ms = max(0.0, float(duration_ms))
        self._view_span_ms = max(1000.0, self._duration_ms) if self._duration_ms > 0.0 else 1000.0
        self._apply_view_range(0.0, self._view_span_ms)
        self._sync_nav_from_view()

    def set_duration_hint(self, duration_ms: float) -> None:
        duration = max(0.0, float(duration_ms))
        self._duration_ms = duration
        self._view_span_ms = max(1000.0, duration) if duration > 0.0 else 1000.0
        self.playback_panel.set_duration_ms(duration)
        if duration > 0.0:
            self._apply_view_range(0.0, self._view_span_ms)
        self._sync_nav_from_view()

    def set_features(self, timeline: AudioTimeline) -> None:
        t = np.asarray(timeline.frame_times_ms, dtype=np.float64)
        flux = np.asarray(timeline.spectral_flux_per_frame, dtype=np.float64)
        n = min(len(t), len(flux))
        if n == 0:
            self._lod_data.pop("flux", None)
            self.flux_curve.setData([], [])
            return
        t = t[:n]
        flux = flux[:n]
        max_flux = float(np.max(flux)) if np.max(flux) > 1e-9 else 1.0
        flux_norm = np.clip((flux / max_flux) * 100.0, 0.0, 100.0)
        self._lod_data["flux"] = (t, flux_norm)
        self._refresh_lod()

    def set_beats(self, beats: BeatTimeline) -> None:
        if not beats.beats:
            self._beat_data.clear()
            for c in self._beat_curves.values():
                c.setData([], [])
            return

        groups: dict[str, list[float]] = {"downbeat": [], "beat": [], "syncopation": []}
        for b in beats.beats:
            groups.setdefault(str(b.beat_type), []).append(float(b.time_ms))
        for btype in list(self._beat_curves.keys()):
            times = groups.get(btype, [])
            if times:
                self._beat_data[btype] = np.array(sorted(times), dtype=np.float64)
            else:
                self._beat_data.pop(btype, None)
        self._refresh_lod()

    def set_positions(self, positions: PositionTimeline) -> None:
        if not positions.actions:
            self._lod_data.pop("position", None)
            self._lod_data.pop("speed", None)
            self.position_curve.setData([], [])
            self.speed_curve.setData([], [])
            return

        times = np.array([float(a.at) for a in positions.actions], dtype=np.float64)
        pos = np.array([float(a.pos) for a in positions.actions], dtype=np.float64)
        self._lod_data["position"] = (times, np.clip(pos, 0.0, 100.0))

        speed = np.asarray(positions.speed_profile, dtype=np.float64)
        n = min(len(speed), len(times))
        if n > 0:
            speed = speed[:n]
            speed_max = float(np.max(speed)) if np.max(speed) > 1e-9 else 1.0
            speed_norm = np.clip((speed / speed_max) * 100.0, 0.0, 100.0)
            self._lod_data["speed"] = (times[:n], speed_norm)
        else:
            self._lod_data.pop("speed", None)
        self._refresh_lod()

    def set_multi_axis(self, result: MultiAxisResult) -> None:
        palette = {
            "alpha": "#4dd0e1",
            "beta": "#ff80ab",
            "e1": "#90caf9",
            "e2": "#a5d6a7",
            "e3": "#ffcc80",
            "e4": "#ce93d8",
        }

        keep = set()
        for axis_name, actions in result.axes.items():
            if axis_name in {"main"}:
                continue
            if axis_name not in palette:
                continue
            keep.add(axis_name)
            if axis_name not in self.extra_curves:
                curve = self.overlay_plot.plot([], [], pen=pg.mkPen(palette[axis_name], width=1.0))
                curve.setVisible(self._extra_curves_visible)
                self.extra_curves[axis_name] = curve
            if actions:
                t = np.array([float(a.at) for a in actions], dtype=np.float64)
                y = np.array([float(a.pos) for a in actions], dtype=np.float64)
                self._lod_data[f"extra_{axis_name}"] = (t, np.clip(y, 0.0, 100.0))
            else:
                self._lod_data.pop(f"extra_{axis_name}", None)

        for axis_name in list(self.extra_curves.keys()):
            if axis_name in keep:
                continue
            curve = self.extra_curves.pop(axis_name)
            self.overlay_plot.removeItem(curve)
            self._lod_data.pop(f"extra_{axis_name}", None)

        self._refresh_lod()

    def set_playback_position(self, time_ms: float) -> None:
        self.playhead_line.setPos(float(time_ms))
        secs = max(0.0, time_ms / 1000.0)
        self._playhead_label.setText(f"{secs:.2f}s")
        self._playhead_label.setPos(float(time_ms), 0.0)

    def zoom_to_range(self, start_ms: float, end_ms: float) -> None:
        self._apply_view_range(start_ms, end_ms)
        self._sync_nav_from_view()

    # ── Edit mode ───────────────────────────────────────────

    def set_edit_state(self, state: FunscriptEditState | None) -> None:
        """Bind an edit state. When set, edit mode can be toggled."""
        if self._edit_state is not None:
            try:
                self._edit_state.changed.disconnect(self._on_edit_state_changed)
            except (TypeError, RuntimeError):
                pass
        self._edit_state = state
        if state is not None:
            state.changed.connect(self._on_edit_state_changed)
        self._edit_mode_btn.setEnabled(state is not None)
        if state is None and self._edit_mode:
            self._edit_mode_btn.setChecked(False)

    def _on_edit_mode_toggled(self, active: bool) -> None:
        self._edit_mode = active
        self._edit_scatter.setVisible(active)
        self._selection_scatter.setVisible(active)
        if active:
            self.overlay_plot.getViewBox().setMouseEnabled(x=True, y=False)
            self.overlay_plot.setFocus()
            self._refresh_edit_overlay()
            self._rebuild_lock_overlays()
        else:
            if self._edit_state:
                self._edit_state.clear_selection()
            self._edit_scatter.setData([], [])
            self._selection_scatter.setData([], [])
            self._clear_lock_overlays()
        self.edit_mode_changed.emit(active)

    def _on_edit_state_changed(self) -> None:
        if self._edit_mode:
            self._refresh_edit_overlay()
            self._rebuild_lock_overlays()

    def _refresh_edit_overlay(self) -> None:
        """Refresh the scatter points from the edit state."""
        if not self._edit_mode or self._edit_state is None:
            return
        actions = self._edit_state.actions
        if not actions:
            self._edit_scatter.setData([], [])
            self._selection_scatter.setData([], [])
            return

        # Apply LOD for rendering scatter
        x_range = self.overlay_plot.viewRange()[0]
        lo, hi = float(x_range[0]), float(x_range[1])

        x_all = np.array([float(a.at) for a in actions], dtype=np.float64)
        y_all = np.array([float(a.pos) for a in actions], dtype=np.float64)

        # Viewport clip for scatter
        i0 = max(0, int(np.searchsorted(x_all, lo, side="left")) - 1)
        i1 = min(len(x_all), int(np.searchsorted(x_all, hi, side="right")) + 1)
        xv = x_all[i0:i1]
        yv = y_all[i0:i1]

        # LOD downsample if too many visible
        sel = self._edit_state.selection_indices
        if len(xv) > 2000:
            step = max(1, len(xv) // 2000)
            lod_indices = set(range(0, len(xv), step))
            # Always include selected points so they remain visible
            if sel:
                for si in sel:
                    local = si - i0
                    if 0 <= local < len(xv):
                        lod_indices.add(local)
            indices = np.array(sorted(lod_indices))
            xv = xv[indices]
            yv = yv[indices]
            vis_global_indices = indices + i0
        else:
            vis_global_indices = np.arange(i0, i1)

        self._edit_scatter.setData(x=xv, y=yv)

        # Selection scatter
        if sel and len(vis_global_indices) > 0:
            sel_mask = np.isin(vis_global_indices, list(sel))
            if np.any(sel_mask):
                self._selection_scatter.setData(x=xv[sel_mask], y=yv[sel_mask])
            else:
                self._selection_scatter.setData([], [])
        else:
            self._selection_scatter.setData([], [])

        # Also update the position line curve from edit state
        self._lod_data["position"] = (x_all, np.clip(y_all, 0.0, 100.0))
        self._refresh_lod()

    def _rebuild_lock_overlays(self) -> None:
        self._clear_lock_overlays()
        if self._edit_state is None:
            return
        for region in self._edit_state.locked_regions:
            lr = pg.LinearRegionItem(
                values=(region.start_ms, region.end_ms),
                orientation="vertical",
                brush=pg.mkBrush(100, 100, 255, 40),
                pen=pg.mkPen("#6666ff", width=1, style=Qt.PenStyle.DashLine),
                movable=False,
            )
            lr.setZValue(-10)
            self.overlay_plot.addItem(lr)
            self._lock_overlays.append(lr)

    def _clear_lock_overlays(self) -> None:
        for item in self._lock_overlays:
            self.overlay_plot.removeItem(item)
        self._lock_overlays.clear()

    # ── Hit-testing (full action list, not LOD) ─────────────

    def _find_nearest_action(self, time_ms: float, pos: float, max_dist_ms: float = 0) -> int | None:
        """Search full action list (binary search by time) for nearest point to click."""
        if self._edit_state is None:
            return None
        actions = self._edit_state.actions
        if not actions:
            return None

        # Scale max_dist_ms based on viewport width for usability
        if max_dist_ms <= 0:
            x_range = self.overlay_plot.viewRange()[0]
            max_dist_ms = max(100, (x_range[1] - x_range[0]) * 0.005)

        lo = bisect.bisect_left(actions, time_ms - max_dist_ms, key=lambda a: a.at)
        hi = bisect.bisect_right(actions, time_ms + max_dist_ms, key=lambda a: a.at)
        best_idx, best_dist = None, float('inf')
        for i in range(lo, min(hi, len(actions))):
            a = actions[i]
            dt = abs(a.at - time_ms) / max(1, max_dist_ms)
            dp = abs(a.pos - pos) / 100.0
            dist = (dt ** 2 + dp ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx if best_dist < 1.0 else None

    # ── Mouse & keyboard event handling ─────────────────────

    def _scene_to_data(self, scene_pos) -> tuple[float, float]:
        """Convert a scene position to data coordinates (time_ms, pos)."""
        vb = self.overlay_plot.getViewBox()
        point = vb.mapSceneToView(scene_pos)
        return float(point.x()), float(point.y())

    def _plot_event_to_data(self, ev) -> tuple[float, float]:
        scene_pos = self.overlay_plot.mapToScene(ev.position().toPoint())
        return self._scene_to_data(scene_pos)

    def _handle_plot_viewport_mouse_press(self, ev) -> bool:
        if not self._edit_mode or self._edit_state is None:
            # Non-edit mode: left-click starts potential seek or drag-to-pan
            if ev.button() == Qt.MouseButton.LeftButton:
                time_ms, _pos = self._plot_event_to_data(ev)
                self._nav_press_time_ms = time_ms
                self._nav_last_x = ev.position().x()
                self._nav_dragging = False
                ev.accept()
                return True
            return False

        time_ms, pos = self._plot_event_to_data(ev)

        if ev.button() == Qt.MouseButton.LeftButton:
            shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)

            # Shift+click always starts rectangle selection, even on a point
            if shift:
                self._rect_selecting = True
                self._rect_start_pos = (time_ms, pos)
                self._rect_roi.setRegion((time_ms, time_ms))
                self._rect_roi.setVisible(True)
                ev.accept()
                return True

            idx = self._find_nearest_action(time_ms, pos)
            if idx is not None:
                ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                if ctrl:
                    self._edit_state.select_index(idx, toggle=True)
                else:
                    if idx not in self._edit_state.selection_indices:
                        self._edit_state.clear_selection()
                        self._edit_state.select_index(idx)
                self._drag_idx = idx
                self._drag_active = False
                self._edit_state.begin_drag()
                ev.accept()
                return True

            self._edit_state.clear_selection()
            ev.accept()
            return True

        if ev.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(ev.globalPosition().toPoint(), time_ms, pos)
            ev.accept()
            return True

        return False

    def _handle_plot_viewport_mouse_move(self, ev) -> bool:
        if not self._edit_mode or self._edit_state is None:
            # Non-edit mode: drag to pan the view horizontally
            if self._nav_last_x is not None:
                dx_pixels = ev.position().x() - self._nav_last_x
                if not self._nav_dragging and abs(dx_pixels) > 4:
                    self._nav_dragging = True
                    # User is panning — suspend auto-follow so it doesn't fight
                    if self._auto_follow_playhead:
                        self._auto_follow_playhead = False
                        self.follow_btn.setChecked(False)
                if self._nav_dragging:
                    vb = self.overlay_plot.getViewBox()
                    # Convert pixel delta to data-coordinate delta
                    view_rect = vb.viewRect()
                    pixel_width = vb.width()
                    if pixel_width > 0:
                        data_dx = (dx_pixels / pixel_width) * view_rect.width()
                        new_start = view_rect.left() - data_dx
                        new_end = new_start + view_rect.width()
                        self._apply_view_range(new_start, new_end)
                        # Keep playhead visually fixed — seek by the same delta
                        cur = self.playhead_line.value()
                        new_pos_ms = max(0.0, cur - data_dx)
                        self.playback_panel.seek(new_pos_ms)
                    self._nav_last_x = ev.position().x()
                ev.accept()
                return True
            return False

        time_ms, pos = self._plot_event_to_data(ev)

        if self._drag_idx is not None:
            self._drag_active = True
            new_at = int(time_ms)
            new_pos = int(max(0, min(100, pos)))
            if self._drag_idx < len(self._edit_state.actions):
                current_action = self._edit_state.actions[self._drag_idx]
                if not self._edit_state.is_locked(current_action.at):
                    new_idx = self._edit_state.move_action(self._drag_idx, new_at, new_pos)
                    if new_idx >= 0:
                        self._drag_idx = new_idx
            ev.accept()
            return True

        if self._rect_selecting and self._rect_start_pos is not None:
            t_start, _ = self._rect_start_pos
            self._rect_roi.setRegion((min(t_start, time_ms), max(t_start, time_ms)))
            ev.accept()
            return True

        return False

    def _handle_plot_viewport_mouse_release(self, ev) -> bool:
        if not self._edit_mode or self._edit_state is None:
            if self._nav_press_time_ms is not None:
                if not self._nav_dragging:
                    # No drag — treat as click-to-seek
                    self.playback_panel.seek(max(0.0, self._nav_press_time_ms))
                self._nav_press_time_ms = None
                self._nav_last_x = None
                self._nav_dragging = False
                ev.accept()
                return True
            return False

        time_ms, pos = self._plot_event_to_data(ev)

        if self._rect_selecting and self._rect_start_pos is not None:
            t_start, _p_start = self._rect_start_pos
            ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if not ctrl:
                self._edit_state.clear_selection()
            # Use time-range-only selection — the visual overlay only
            # shows the time span and users drag horizontally, so
            # filtering by Y position would discard most points.
            self._edit_state.select_range(int(t_start), int(time_ms))
            self._rect_roi.setVisible(False)
            self._rect_selecting = False
            self._rect_start_pos = None
            ev.accept()
            return True

        if self._drag_idx is not None:
            self._drag_idx = None
            self._drag_active = False
            ev.accept()
            return True

        return False

    def _handle_plot_viewport_mouse_double_click(self, ev) -> bool:
        if not self._edit_mode or self._edit_state is None:
            return False
        if ev.button() != Qt.MouseButton.LeftButton:
            return False

        time_ms, pos = self._plot_event_to_data(ev)
        from pmv_funscript_io import FunscriptAction

        self._edit_state.add_action(FunscriptAction(int(time_ms), int(max(0, min(100, pos)))))
        ev.accept()
        return True

    def _handle_plot_key_press(self, ev: QKeyEvent) -> bool:
        if not self._edit_mode or self._edit_state is None:
            return False

        key = ev.key()
        mod = ev.modifiers()

        if key == Qt.Key.Key_Delete:
            self._edit_state.remove_selected()
            ev.accept()
            return True

        if key == Qt.Key.Key_Z and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.undo()
            ev.accept()
            return True

        if key == Qt.Key.Key_Y and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.redo()
            ev.accept()
            return True

        if key == Qt.Key.Key_A and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.select_all()
            ev.accept()
            return True

        if key == Qt.Key.Key_D and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.clear_selection()
            ev.accept()
            return True

        if key == Qt.Key.Key_C and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.copy_selection()
            ev.accept()
            return True

        if key == Qt.Key.Key_X and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.cut_selection()
            ev.accept()
            return True

        if key == Qt.Key.Key_V and mod & Qt.KeyboardModifier.ControlModifier:
            playhead_ms = int(self.playhead_line.value())
            if mod & Qt.KeyboardModifier.ShiftModifier:
                self._edit_state.paste_exact()
            else:
                self._edit_state.paste_at(playhead_ms)
            ev.accept()
            return True

        if key == Qt.Key.Key_I and mod & Qt.KeyboardModifier.ShiftModifier:
            self._edit_state.invert_all()
            ev.accept()
            return True

        if key == Qt.Key.Key_I and not mod:
            self._edit_state.invert_selection()
            ev.accept()
            return True

        if key == Qt.Key.Key_E and not mod:
            self._edit_state.equalize_selection()
            ev.accept()
            return True

        if key == Qt.Key.Key_L and not mod:
            self._edit_state.lock_selection_region()
            ev.accept()
            return True

        if key == Qt.Key.Key_L and mod & Qt.KeyboardModifier.ControlModifier and mod & Qt.KeyboardModifier.ShiftModifier:
            self._edit_state.lock_all_except_selection()
            ev.accept()
            return True

        if key == Qt.Key.Key_L and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.unlock_at(int(self.playhead_line.value()))
            ev.accept()
            return True

        if key == Qt.Key.Key_Up and mod & Qt.KeyboardModifier.ShiftModifier:
            self._edit_state.move_selection_position(5)
            ev.accept()
            return True

        if key == Qt.Key.Key_Down and mod & Qt.KeyboardModifier.ShiftModifier:
            self._edit_state.move_selection_position(-5)
            ev.accept()
            return True

        if key == Qt.Key.Key_Left and mod & Qt.KeyboardModifier.ShiftModifier:
            self._edit_state.move_selection_time(-100)
            ev.accept()
            return True

        if key == Qt.Key.Key_Right and mod & Qt.KeyboardModifier.ShiftModifier:
            self._edit_state.move_selection_time(100)
            ev.accept()
            return True

        return False

    def eventFilter(self, watched, event) -> bool:
        if watched is self.overlay_plot.viewport():
            if event.type() == QEvent.Type.MouseButtonPress and self._handle_plot_viewport_mouse_press(event):
                return True
            if event.type() == QEvent.Type.MouseMove and self._handle_plot_viewport_mouse_move(event):
                return True
            if event.type() == QEvent.Type.MouseButtonRelease and self._handle_plot_viewport_mouse_release(event):
                return True
            if event.type() == QEvent.Type.MouseButtonDblClick and self._handle_plot_viewport_mouse_double_click(event):
                return True
        if watched is self.overlay_plot and event.type() == QEvent.Type.KeyPress and self._handle_plot_key_press(event):
            return True
        return super().eventFilter(watched, event)

    def _show_context_menu(self, global_pos, time_ms: float, pos: float) -> None:
        if self._edit_state is None:
            return
        menu = QMenu(self)

        add_act = menu.addAction("Add Point Here")
        add_act.triggered.connect(lambda: self._ctx_add_point(time_ms, pos))

        menu.addSeparator()

        lock_act = menu.addAction("Lock Selection Region")
        lock_act.setEnabled(self._edit_state.has_selection)
        lock_act.triggered.connect(self._edit_state.lock_selection_region)

        unlock_act = menu.addAction("Unlock This Region")
        unlock_act.setEnabled(self._edit_state.is_locked(int(time_ms)))
        unlock_act.triggered.connect(lambda: self._edit_state.unlock_at(int(time_ms)))

        clear_locks_act = menu.addAction("Clear All Locks")
        clear_locks_act.setEnabled(len(self._edit_state.locked_regions) > 0)
        clear_locks_act.triggered.connect(self._edit_state.clear_all_locks)

        menu.addSeparator()

        cut_act = menu.addAction("Cut\tCtrl+X")
        cut_act.setEnabled(self._edit_state.has_selection)
        cut_act.triggered.connect(self._edit_state.cut_selection)

        copy_act = menu.addAction("Copy\tCtrl+C")
        copy_act.setEnabled(self._edit_state.has_selection)
        copy_act.triggered.connect(self._edit_state.copy_selection)

        paste_act = menu.addAction("Paste Here\tCtrl+V")
        paste_act.setEnabled(not self._edit_state.clipboard_empty)
        paste_act.triggered.connect(lambda: self._edit_state.paste_at(int(time_ms)))

        menu.addSeparator()

        invert_act = menu.addAction("Invert Selected\tI")
        invert_act.setEnabled(self._edit_state.has_selection)
        invert_act.triggered.connect(self._edit_state.invert_selection)

        invert_all_act = menu.addAction("Invert All\tShift+I")
        invert_all_act.setEnabled(len(self._edit_state.actions) > 0)
        invert_all_act.triggered.connect(self._edit_state.invert_all)

        eq_act = menu.addAction("Equalize\tE")
        eq_act.setEnabled(len(self._edit_state.selection_indices) >= 3)
        eq_act.triggered.connect(self._edit_state.equalize_selection)

        center_sel_act = menu.addAction("Center Selected At...")
        center_sel_act.setEnabled(self._edit_state.has_selection)
        center_sel_act.triggered.connect(lambda: self._prompt_center(selected_only=True))

        center_all_act = menu.addAction("Center All At...")
        center_all_act.setEnabled(len(self._edit_state.actions) > 0)
        center_all_act.triggered.connect(lambda: self._prompt_center(selected_only=False))

        sa_act = menu.addAction("Select All\tCtrl+A")
        sa_act.triggered.connect(self._edit_state.select_all)

        menu.addSeparator()

        del_act = menu.addAction("Delete\tDel")
        del_act.setEnabled(self._edit_state.has_selection)
        del_act.triggered.connect(self._edit_state.remove_selected)

        menu.exec(global_pos)

    def _prompt_center(self, selected_only: bool) -> None:
        if self._edit_state is None:
            return
        val, ok = QInputDialog.getInt(
            self, "Center At Position",
            "Target mean position (0–100):",
            value=50, min=0, max=100,
        )
        if ok:
            self._edit_state.center_at(float(val), selected_only=selected_only)

    def _ctx_add_point(self, time_ms: float, pos: float) -> None:
        from pmv_funscript_io import FunscriptAction
        self._edit_state.add_action(FunscriptAction(int(time_ms), int(max(0, min(100, pos)))))


# ---------------------------------------------------------------------------
# Axis grouping for AuxAxisPanel
# ---------------------------------------------------------------------------

_AXIS_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Alpha / Beta", [("alpha", "#4dd0e1"), ("beta", "#ff80ab")]),
    ("Electrodes", [("e1", "#90caf9"), ("e2", "#a5d6a7"), ("e3", "#ffcc80"), ("e4", "#ce93d8")]),
    ("Alpha Prostate / Beta Prostate", [("alpha_prostate", "#26a69a"), ("beta_prostate", "#ec407a")]),
    ("Frequency", [("frequency", "#66bb6a")]),
    ("Pulse Frequency", [("pulse_frequency", "#42a5f5")]),
    ("Carrier Frequency", [("carrier_frequency", "#26c6da")]),
    ("Volume", [("volume", "#ab47bc")]),
    ("Pulse Rise", [("pulse_rise", "#ffa726")]),
    ("Pulse Width", [("pulse_width", "#ef5350")]),
]

# Flat lookup: axis_name -> (group_title, color)
_AXIS_META: dict[str, tuple[str, str]] = {}
for _grp_title, _members in _AXIS_GROUPS:
    for _ax_name, _ax_color in _members:
        _AXIS_META[_ax_name] = (_grp_title, _ax_color)


class AuxAxisPanel(QWidget):
    """Vertically-stacked mini-plots for auxiliary funscript axes.

    Groups related axes onto the same plot (alpha+beta share one,
    e1-e4 share another) while other axes get individual rows.
    Only groups that have data are shown.  Supports point editing on
    a selected axis.
    """

    edit_axis_changed = pyqtSignal(str)  # axis_name

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        # ── Edit toolbar ──
        edit_bar = QHBoxLayout()
        edit_bar.setContentsMargins(0, 0, 0, 0)
        edit_bar.setSpacing(6)
        edit_bar.addWidget(QLabel("Aux Edit"))
        self._edit_axis_combo = QComboBox()
        self._edit_axis_combo.addItem("(none)")
        self._edit_axis_combo.setToolTip("Select an auxiliary axis to edit")
        self._edit_axis_combo.currentTextChanged.connect(self._on_combo_axis_changed)
        edit_bar.addWidget(self._edit_axis_combo, 1)
        self._edit_btn = QPushButton("Edit")
        self._edit_btn.setCheckable(True)
        self._edit_btn.setEnabled(False)
        self._edit_btn.toggled.connect(self._on_edit_toggled)
        edit_bar.addWidget(self._edit_btn)
        self._layout.addLayout(edit_bar)

        # group_title -> (PlotWidget, {axis_name: PlotDataItem}, InfiniteLine)
        self._group_plots: dict[str, tuple[pg.PlotWidget, dict[str, pg.PlotDataItem], pg.InfiniteLine]] = {}

        # Per-group TCode send toggles: group_title -> QCheckBox
        self._send_toggles: dict[str, QCheckBox] = {}

        # Full-resolution data for LOD: axis_name -> (x, y)
        self._lod_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Reference to main overlay_plot for X-axis linking
        self._main_plot: pg.PlotWidget | None = None

        # ── Edit mode state ──
        self._edit_state: FunscriptEditState | None = None
        self._edit_mode = False
        self._edit_axis_name: str | None = None
        self._edit_plot: pg.PlotWidget | None = None  # currently-editing plot widget
        self._edit_plot_default_height = 80
        self._edit_plot_edit_height = 200

        self._edit_scatter = pg.ScatterPlotItem(
            size=8, pen=pg.mkPen("#fff176", width=1), brush=pg.mkBrush("#fff176"),
            hoverable=True, hoverSize=12, hoverBrush=pg.mkBrush("#ffffff"),
        )
        self._edit_scatter.setVisible(False)
        self._edit_scatter.setZValue(10)

        self._selection_scatter = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen("#42a5f5", width=2), brush=pg.mkBrush("#42a5f580"),
        )
        self._selection_scatter.setVisible(False)
        self._selection_scatter.setZValue(11)

        self._rect_roi = pg.LinearRegionItem(
            values=(0, 0), orientation="vertical",
            brush=pg.mkBrush(66, 165, 245, 30),
            pen=pg.mkPen("#42a5f5", width=1, style=Qt.PenStyle.DashLine),
            movable=False,
        )
        self._rect_roi.setVisible(False)
        self._rect_roi.setZValue(5)

        self._lock_overlays: list[pg.LinearRegionItem] = []
        self._drag_idx: int | None = None
        self._drag_active = False
        self._rect_selecting = False
        self._rect_start_pos: tuple[float, float] | None = None

    def link_x_axis(self, main_plot: pg.PlotWidget) -> None:
        """Link all mini-plot X-axes to the main visualization plot."""
        self._main_plot = main_plot
        for _title, (plot, _curves, _ph) in self._group_plots.items():
            plot.setXLink(main_plot)
        # Refresh LOD when main view range changes
        main_plot.getViewBox().sigXRangeChanged.connect(lambda _vb, _rng: self._refresh_lod())

    def set_multi_axis(self, result: MultiAxisResult) -> None:
        """Update mini-plots from a MultiAxisResult. Creates/removes groups as needed."""
        # Determine which groups have data
        active_groups: dict[str, list[tuple[str, str, list]]] = {}
        for axis_name, actions in result.axes.items():
            if axis_name == "main":
                continue
            meta = _AXIS_META.get(axis_name)
            if meta is None:
                continue
            grp_title, color = meta
            active_groups.setdefault(grp_title, []).append((axis_name, color, actions))

        # Remove groups no longer present
        for grp_title in list(self._group_plots.keys()):
            if grp_title not in active_groups:
                plot, _curves, _ph = self._group_plots.pop(grp_title)
                self._layout.removeWidget(plot)
                plot.deleteLater()
                cb = self._send_toggles.pop(grp_title, None)
                if cb is not None:
                    cb.deleteLater()

        # Create or update groups in canonical order
        for grp_title, members in _AXIS_GROUPS:
            if grp_title not in active_groups:
                continue

            if grp_title not in self._group_plots:
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(4)

                cb = QCheckBox()
                cb.setChecked(True)
                cb.setToolTip(f"Send {grp_title} via TCode during preview")
                cb.setFixedWidth(18)
                row.addWidget(cb)
                self._send_toggles[grp_title] = cb

                plot = pg.PlotWidget()
                _style_plot(plot)
                plot.setFixedHeight(80)
                plot.setLabel("left", grp_title)
                plot.hideAxis("bottom")
                vb = plot.getViewBox()
                vb.setYRange(0.0, 100.0, padding=0.0)
                vb.setMouseEnabled(x=True, y=False)
                vb.setLimits(yMin=0.0, yMax=100.0)
                playhead = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("#ffe082", width=1))
                plot.addItem(playhead)
                if self._main_plot is not None:
                    plot.setXLink(self._main_plot)
                row.addWidget(plot, 1)
                self._layout.addLayout(row)
                self._group_plots[grp_title] = (plot, {}, playhead)

            plot, curves, _ph = self._group_plots[grp_title]

            # Track which curves are still active
            active_in_group = set()
            for axis_name, color, actions in active_groups[grp_title]:
                active_in_group.add(axis_name)
                if axis_name not in curves:
                    curve = plot.plot([], [], pen=pg.mkPen(color, width=1.2), name=axis_name)
                    curves[axis_name] = curve
                if actions:
                    t = np.array([float(a.at) for a in actions], dtype=np.float64)
                    y = np.array([float(a.pos) for a in actions], dtype=np.float64)
                    self._lod_data[axis_name] = (t, np.clip(y, 0.0, 100.0))
                else:
                    self._lod_data.pop(axis_name, None)

            # Remove curves no longer in this group
            for axis_name in list(curves.keys()):
                if axis_name not in active_in_group:
                    curve = curves.pop(axis_name)
                    plot.removeItem(curve)
                    self._lod_data.pop(axis_name, None)

        self._refresh_lod()

    def _refresh_lod(self) -> None:
        """Downsample all aux axis data to the current viewport."""
        if self._main_plot is None:
            return
        x_range = self._main_plot.viewRange()[0]
        lo, hi = float(x_range[0]), float(x_range[1])

        for _title, (_plot, curves, _ph) in self._group_plots.items():
            for axis_name, curve in curves.items():
                full = self._lod_data.get(axis_name)
                if full is None:
                    continue
                x_full, y_full = full
                if x_full.size == 0:
                    continue
                xs, ys = _lod_slice(x_full, y_full, lo, hi)
                curve.setData(xs, ys)

    def set_playhead(self, time_ms: float) -> None:
        """Move playhead across all mini-plots."""
        for _title, (_plot, _curves, playhead) in self._group_plots.items():
            playhead.setPos(float(time_ms))

    def get_send_axes(self) -> set[str]:
        """Return axis names whose group has the send-toggle checked."""
        result: set[str] = set()
        for grp_title, cb in self._send_toggles.items():
            if cb.isChecked():
                plot_info = self._group_plots.get(grp_title)
                if plot_info is not None:
                    result.update(plot_info[1].keys())
        return result

    # ------------------------------------------------------------------
    # Axis editing
    # ------------------------------------------------------------------

    def update_edit_axis_list(self, axis_names: list[str]) -> None:
        """Populate the axis combo with available aux axes."""
        prev = self._edit_axis_combo.currentText()
        self._edit_axis_combo.blockSignals(True)
        self._edit_axis_combo.clear()
        self._edit_axis_combo.addItem("(none)")
        for n in axis_names:
            if n != "main":
                self._edit_axis_combo.addItem(n)
        if prev in axis_names:
            self._edit_axis_combo.setCurrentText(prev)
        self._edit_axis_combo.blockSignals(False)

    def select_edit_axis(self, axis_name: str) -> None:
        target = axis_name if axis_name and self._edit_axis_combo.findText(axis_name) >= 0 else "(none)"
        self._edit_axis_combo.setCurrentText(target)

    def set_edit_state(self, state: FunscriptEditState | None) -> None:
        if self._edit_state is not None:
            try:
                self._edit_state.changed.disconnect(self._on_edit_state_changed)
            except (TypeError, RuntimeError):
                pass
        self._edit_state = state
        if state is not None:
            state.changed.connect(self._on_edit_state_changed)
        if self._edit_mode:
            self._refresh_edit_overlay()
            self._rebuild_lock_overlays()

    def _on_combo_axis_changed(self, text: str) -> None:
        self._edit_btn.setEnabled(text != "(none)")
        if text == "(none)":
            if self._edit_mode:
                self._edit_btn.setChecked(False)
        else:
            self.edit_axis_changed.emit(text)

    def _on_edit_toggled(self, active: bool) -> None:
        axis_name = self._edit_axis_combo.currentText()
        if axis_name == "(none)":
            self._edit_btn.setChecked(False)
            return
        if active:
            self._activate_edit(axis_name)
        else:
            self._deactivate_edit()

    def _find_plot_for_axis(self, axis_name: str) -> pg.PlotWidget | None:
        meta = _AXIS_META.get(axis_name)
        if meta is None:
            return None
        grp_title, _color = meta
        entry = self._group_plots.get(grp_title)
        if entry is None:
            return None
        return entry[0]

    def _activate_edit(self, axis_name: str) -> None:
        self._deactivate_edit()
        plot = self._find_plot_for_axis(axis_name)
        if plot is None:
            return
        self._edit_mode = True
        self._edit_axis_name = axis_name
        self._edit_plot = plot

        plot.addItem(self._edit_scatter)
        plot.addItem(self._selection_scatter)
        plot.addItem(self._rect_roi)
        self._edit_scatter.setVisible(True)
        self._selection_scatter.setVisible(True)

        plot.setFixedHeight(self._edit_plot_edit_height)
        plot.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        plot.setFocus()
        plot.installEventFilter(self)
        plot.viewport().installEventFilter(self)

        self._refresh_edit_overlay()
        self._rebuild_lock_overlays()

    def _deactivate_edit(self) -> None:
        if not self._edit_mode:
            return
        if self._edit_state:
            self._edit_state.clear_selection()
        self._edit_scatter.setData([], [])
        self._selection_scatter.setData([], [])
        self._edit_scatter.setVisible(False)
        self._selection_scatter.setVisible(False)
        self._rect_roi.setVisible(False)
        self._clear_lock_overlays()

        if self._edit_plot is not None:
            self._edit_plot.removeItem(self._edit_scatter)
            self._edit_plot.removeItem(self._selection_scatter)
            self._edit_plot.removeItem(self._rect_roi)
            self._edit_plot.removeEventFilter(self)
            self._edit_plot.viewport().removeEventFilter(self)
            self._edit_plot.setFixedHeight(self._edit_plot_default_height)
            self._edit_plot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._edit_plot = None

        self._edit_mode = False
        self._edit_axis_name = None
        self._drag_idx = None
        self._drag_active = False
        self._rect_selecting = False
        self._rect_start_pos = None

    def _on_edit_state_changed(self) -> None:
        if self._edit_mode:
            self._refresh_edit_overlay()
            self._rebuild_lock_overlays()

    def _refresh_edit_overlay(self) -> None:
        if not self._edit_mode or self._edit_state is None or self._edit_plot is None:
            return
        actions = self._edit_state.actions
        if not actions:
            self._edit_scatter.setData([], [])
            self._selection_scatter.setData([], [])
            return

        x_range = self._edit_plot.viewRange()[0]
        lo, hi = float(x_range[0]), float(x_range[1])

        x_all = np.array([float(a.at) for a in actions], dtype=np.float64)
        y_all = np.array([float(a.pos) for a in actions], dtype=np.float64)

        i0 = max(0, int(np.searchsorted(x_all, lo, side="left")) - 1)
        i1 = min(len(x_all), int(np.searchsorted(x_all, hi, side="right")) + 1)
        xv = x_all[i0:i1]
        yv = y_all[i0:i1]

        sel = self._edit_state.selection_indices
        if len(xv) > 2000:
            step = max(1, len(xv) // 2000)
            lod_indices = set(range(0, len(xv), step))
            if sel:
                for si in sel:
                    local = si - i0
                    if 0 <= local < len(xv):
                        lod_indices.add(local)
            indices = np.array(sorted(lod_indices))
            xv = xv[indices]
            yv = yv[indices]
            vis_global_indices = indices + i0
        else:
            vis_global_indices = np.arange(i0, i1)

        self._edit_scatter.setData(x=xv, y=yv)

        if sel and len(vis_global_indices) > 0:
            sel_mask = np.isin(vis_global_indices, list(sel))
            if np.any(sel_mask):
                self._selection_scatter.setData(x=xv[sel_mask], y=yv[sel_mask])
            else:
                self._selection_scatter.setData([], [])
        else:
            self._selection_scatter.setData([], [])

        # Update the LOD curve for this axis so the line redraws from edit state
        if self._edit_axis_name:
            self._lod_data[self._edit_axis_name] = (x_all, np.clip(y_all, 0.0, 100.0))
            self._refresh_lod()

    def _rebuild_lock_overlays(self) -> None:
        self._clear_lock_overlays()
        if self._edit_state is None or self._edit_plot is None:
            return
        for region in self._edit_state.locked_regions:
            lr = pg.LinearRegionItem(
                values=(region.start_ms, region.end_ms), orientation="vertical",
                brush=pg.mkBrush(100, 100, 255, 40),
                pen=pg.mkPen("#6666ff", width=1, style=Qt.PenStyle.DashLine),
                movable=False,
            )
            lr.setZValue(-10)
            self._edit_plot.addItem(lr)
            self._lock_overlays.append(lr)

    def _clear_lock_overlays(self) -> None:
        for item in self._lock_overlays:
            if self._edit_plot is not None:
                self._edit_plot.removeItem(item)
        self._lock_overlays.clear()

    def _find_nearest_action(self, time_ms: float, pos: float) -> int | None:
        if self._edit_state is None or self._edit_plot is None:
            return None
        actions = self._edit_state.actions
        if not actions:
            return None
        x_range = self._edit_plot.viewRange()[0]
        max_dist_ms = max(100, (x_range[1] - x_range[0]) * 0.005)
        lo = bisect.bisect_left(actions, time_ms - max_dist_ms, key=lambda a: a.at)
        hi = bisect.bisect_right(actions, time_ms + max_dist_ms, key=lambda a: a.at)
        best_idx, best_dist = None, float('inf')
        for i in range(lo, min(hi, len(actions))):
            a = actions[i]
            dt = abs(a.at - time_ms) / max(1, max_dist_ms)
            dp = abs(a.pos - pos) / 100.0
            dist = (dt ** 2 + dp ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx if best_dist < 1.0 else None

    def _plot_event_to_data(self, ev) -> tuple[float, float]:
        if self._edit_plot is None:
            return 0.0, 0.0
        scene_pos = self._edit_plot.mapToScene(ev.position().toPoint())
        vb = self._edit_plot.getViewBox()
        point = vb.mapSceneToView(scene_pos)
        return float(point.x()), float(point.y())

    # ── Event filter ──

    def eventFilter(self, watched, event) -> bool:
        if self._edit_plot is not None and watched is self._edit_plot.viewport():
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress and self._handle_mouse_press(event):
                return True
            if etype == QEvent.Type.MouseMove and self._handle_mouse_move(event):
                return True
            if etype == QEvent.Type.MouseButtonRelease and self._handle_mouse_release(event):
                return True
            if etype == QEvent.Type.MouseButtonDblClick and self._handle_mouse_dbl(event):
                return True
        if self._edit_plot is not None and watched is self._edit_plot and event.type() == QEvent.Type.KeyPress:
            if self._handle_key_press(event):
                return True
        return super().eventFilter(watched, event)

    # ── Mouse handlers ──

    def _handle_mouse_press(self, ev) -> bool:
        if not self._edit_mode or self._edit_state is None:
            return False
        time_ms, pos = self._plot_event_to_data(ev)
        if ev.button() == Qt.MouseButton.LeftButton:
            shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            if shift:
                self._rect_selecting = True
                self._rect_start_pos = (time_ms, pos)
                self._rect_roi.setRegion((time_ms, time_ms))
                self._rect_roi.setVisible(True)
                ev.accept()
                return True
            idx = self._find_nearest_action(time_ms, pos)
            if idx is not None:
                ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
                if ctrl:
                    self._edit_state.select_index(idx, toggle=True)
                else:
                    if idx not in self._edit_state.selection_indices:
                        self._edit_state.clear_selection()
                        self._edit_state.select_index(idx)
                self._drag_idx = idx
                self._drag_active = False
                self._edit_state.begin_drag()
                ev.accept()
                return True
            self._edit_state.clear_selection()
            ev.accept()
            return True
        if ev.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(ev.globalPosition().toPoint(), time_ms, pos)
            ev.accept()
            return True
        return False

    def _handle_mouse_move(self, ev) -> bool:
        if not self._edit_mode or self._edit_state is None:
            return False
        time_ms, pos = self._plot_event_to_data(ev)
        if self._drag_idx is not None:
            self._drag_active = True
            new_at = int(time_ms)
            new_pos = int(max(0, min(100, pos)))
            if self._drag_idx < len(self._edit_state.actions):
                current_action = self._edit_state.actions[self._drag_idx]
                if not self._edit_state.is_locked(current_action.at):
                    new_idx = self._edit_state.move_action(self._drag_idx, new_at, new_pos)
                    if new_idx >= 0:
                        self._drag_idx = new_idx
            ev.accept()
            return True
        if self._rect_selecting and self._rect_start_pos is not None:
            t_start, _ = self._rect_start_pos
            self._rect_roi.setRegion((min(t_start, time_ms), max(t_start, time_ms)))
            ev.accept()
            return True
        return False

    def _handle_mouse_release(self, ev) -> bool:
        if not self._edit_mode or self._edit_state is None:
            return False
        time_ms, _pos = self._plot_event_to_data(ev)
        if self._rect_selecting and self._rect_start_pos is not None:
            t_start, _ = self._rect_start_pos
            ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if not ctrl:
                self._edit_state.clear_selection()
            self._edit_state.select_range(int(t_start), int(time_ms))
            self._rect_roi.setVisible(False)
            self._rect_selecting = False
            self._rect_start_pos = None
            ev.accept()
            return True
        if self._drag_idx is not None:
            self._drag_idx = None
            self._drag_active = False
            ev.accept()
            return True
        return False

    def _handle_mouse_dbl(self, ev) -> bool:
        if not self._edit_mode or self._edit_state is None:
            return False
        if ev.button() != Qt.MouseButton.LeftButton:
            return False
        time_ms, pos = self._plot_event_to_data(ev)
        from pmv_funscript_io import FunscriptAction
        self._edit_state.add_action(FunscriptAction(int(time_ms), int(max(0, min(100, pos)))))
        ev.accept()
        return True

    # ── Keyboard handler ──

    def _handle_key_press(self, ev: QKeyEvent) -> bool:
        if not self._edit_mode or self._edit_state is None:
            return False
        key = ev.key()
        mod = ev.modifiers()

        if key == Qt.Key.Key_Delete:
            self._edit_state.remove_selected()
            return True
        if key == Qt.Key.Key_Z and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.undo()
            return True
        if key == Qt.Key.Key_Y and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.redo()
            return True
        if key == Qt.Key.Key_A and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.select_all()
            return True
        if key == Qt.Key.Key_D and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.clear_selection()
            return True
        if key == Qt.Key.Key_C and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.copy_selection()
            return True
        if key == Qt.Key.Key_X and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.cut_selection()
            return True
        if key == Qt.Key.Key_V and mod & Qt.KeyboardModifier.ControlModifier:
            self._edit_state.paste_at(0)
            return True
        if key == Qt.Key.Key_I and mod & Qt.KeyboardModifier.ShiftModifier:
            self._edit_state.invert_all()
            return True
        if key == Qt.Key.Key_I and not mod:
            self._edit_state.invert_selection()
            return True
        if key == Qt.Key.Key_E and not mod:
            self._edit_state.equalize_selection()
            return True
        if key == Qt.Key.Key_L and not mod:
            self._edit_state.lock_selection_region()
            return True
        if key == Qt.Key.Key_Up and mod & Qt.KeyboardModifier.ShiftModifier:
            self._edit_state.move_selection_position(5)
            return True
        if key == Qt.Key.Key_Down and mod & Qt.KeyboardModifier.ShiftModifier:
            self._edit_state.move_selection_position(-5)
            return True
        if key == Qt.Key.Key_Left and mod & Qt.KeyboardModifier.ShiftModifier:
            self._edit_state.move_selection_time(-100)
            return True
        if key == Qt.Key.Key_Right and mod & Qt.KeyboardModifier.ShiftModifier:
            self._edit_state.move_selection_time(100)
            return True
        return False

    # ── Context menu ──

    def _show_context_menu(self, global_pos, time_ms: float, pos: float) -> None:
        if self._edit_state is None:
            return
        menu = QMenu(self)
        add_act = menu.addAction("Add Point Here")
        add_act.triggered.connect(lambda: self._edit_state.add_action(
            __import__("pmv_funscript_io", fromlist=["FunscriptAction"]).FunscriptAction(
                int(time_ms), int(max(0, min(100, pos))))))

        menu.addSeparator()

        cut_act = menu.addAction("Cut\tCtrl+X")
        cut_act.setEnabled(self._edit_state.has_selection)
        cut_act.triggered.connect(self._edit_state.cut_selection)

        copy_act = menu.addAction("Copy\tCtrl+C")
        copy_act.setEnabled(self._edit_state.has_selection)
        copy_act.triggered.connect(self._edit_state.copy_selection)

        paste_act = menu.addAction("Paste Here\tCtrl+V")
        paste_act.setEnabled(not self._edit_state.clipboard_empty)
        paste_act.triggered.connect(lambda: self._edit_state.paste_at(int(time_ms)))

        menu.addSeparator()

        invert_act = menu.addAction("Invert Selected\tI")
        invert_act.setEnabled(self._edit_state.has_selection)
        invert_act.triggered.connect(self._edit_state.invert_selection)

        invert_all_act = menu.addAction("Invert All\tShift+I")
        invert_all_act.setEnabled(len(self._edit_state.actions) > 0)
        invert_all_act.triggered.connect(self._edit_state.invert_all)

        eq_act = menu.addAction("Equalize\tE")
        eq_act.setEnabled(len(self._edit_state.selection_indices) >= 3)
        eq_act.triggered.connect(self._edit_state.equalize_selection)

        center_sel_act = menu.addAction("Center Selected At...")
        center_sel_act.setEnabled(self._edit_state.has_selection)
        center_sel_act.triggered.connect(lambda: self._prompt_center(selected_only=True))

        center_all_act = menu.addAction("Center All At...")
        center_all_act.setEnabled(len(self._edit_state.actions) > 0)
        center_all_act.triggered.connect(lambda: self._prompt_center(selected_only=False))

        sa_act = menu.addAction("Select All\tCtrl+A")
        sa_act.triggered.connect(self._edit_state.select_all)

        menu.addSeparator()

        del_act = menu.addAction("Delete\tDel")
        del_act.setEnabled(self._edit_state.has_selection)
        del_act.triggered.connect(self._edit_state.remove_selected)

        menu.addSeparator()
        lock_act = menu.addAction("Lock Selection Region")
        lock_act.setEnabled(self._edit_state.has_selection)
        lock_act.triggered.connect(self._edit_state.lock_selection_region)

        unlock_act = menu.addAction("Unlock This Region")
        unlock_act.setEnabled(self._edit_state.is_locked(int(time_ms)))
        unlock_act.triggered.connect(lambda: self._edit_state.unlock_at(int(time_ms)))

        clear_locks_act = menu.addAction("Clear All Locks")
        clear_locks_act.setEnabled(len(self._edit_state.locked_regions) > 0)
        clear_locks_act.triggered.connect(self._edit_state.clear_all_locks)

        menu.exec(global_pos)

    def _prompt_center(self, selected_only: bool) -> None:
        if self._edit_state is None:
            return
        val, ok = QInputDialog.getInt(
            self, "Center At Position",
            "Target mean position (0\u2013100):",
            value=50, min=0, max=100,
        )
        if ok:
            self._edit_state.center_at(float(val), selected_only=selected_only)


# ---------------------------------------------------------------------------
# Video preview widget — syncs to PlaybackPanel transport signals
# ---------------------------------------------------------------------------
_qt_multimedia_available = False
try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput as _QAudioOutput
    from PyQt6.QtMultimediaWidgets import QVideoWidget as _QVideoWidget
    _qt_multimedia_available = True
except ImportError:
    pass


class VideoPreviewWidget(QWidget):
    """Popout video player window that stays in sync with PlaybackPanel transport.

    Uses QMediaPlayer + QVideoWidget from PyQt6.QtMultimedia.  Falls back to
    a plain "no video" label if the Qt Multimedia module is not installed.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Window)  # separate top-level window
        self.setWindowTitle("Video Preview")
        self.resize(640, 480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._media_path: str | None = None
        self._player: QMediaPlayer | None = None
        self._video_widget = None
        self._audio_output = None
        self._syncing = False  # guard against recursive position updates
        self._was_playing = False

        if _qt_multimedia_available:
            self._video_widget = _QVideoWidget(self)
            self._audio_output = _QAudioOutput(self)
            self._audio_output.setVolume(0.0)  # muted — audio comes from sounddevice

            self._player = QMediaPlayer(self)
            self._player.setVideoOutput(self._video_widget)
            self._player.setAudioOutput(self._audio_output)

            layout.addWidget(self._video_widget, 1)
        else:
            lbl = QLabel("Video preview unavailable\n(install PyQt6-Qt6 multimedia)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #888; font-size: 11px;")
            layout.addWidget(lbl, 1)

    # ── public API ──

    def load_media(self, file_path: str | None) -> None:
        """Load a video file for preview.  Pass None to clear."""
        self._media_path = file_path
        if self._player is None:
            return
        from PyQt6.QtCore import QUrl
        if file_path is None:
            self._player.stop()
            self._player.setSource(QUrl())
            return
        url = QUrl.fromLocalFile(file_path)
        self._player.setSource(url)
        # Seek to beginning and pause so a frame is visible
        self._player.pause()

    def on_transport(self, action: str, position_ms: float) -> None:
        """Slot for PlaybackPanel.transport_changed(str, float)."""
        if self._player is None:
            return
        pos_int = max(0, int(round(position_ms)))
        if action == "play":
            self._player.setPosition(pos_int)
            self._player.play()
            self._was_playing = True
        elif action == "pause":
            self._player.pause()
            self._player.setPosition(pos_int)
            self._was_playing = False
        elif action == "stop":
            self._player.stop()
            self._player.setPosition(0)
            self._was_playing = False
        elif action == "seek":
            self._player.setPosition(pos_int)

    def set_muted(self, muted: bool) -> None:
        """Mute / un-mute the video audio track."""
        if self._audio_output is not None:
            self._audio_output.setVolume(0.0 if muted else 1.0)


__all__ = [
    "AuxAxisPanel",
    "PlaybackPanel",
    "PositionTimelinePanel",
    "SpectralFluxPanel",
    "SpeedHeatmapPanel",
    "TimeAxisSync",
    "VideoPreviewWidget",
    "VisualizationArea",
    "WaveformPanel",
]
