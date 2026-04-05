from __future__ import annotations

import importlib
import importlib.util
import time
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

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
        self._timer.setInterval(33)
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
        self._update_label()
        self._emit_position()

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

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._duration_ms = 0.0
        self._view_span_ms = 0.0
        self._nav_syncing = False
        self._auto_follow_playhead = True
        self._lod_refreshing = False  # re-entrancy guard

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
        self.overlay_plot.setLabel("left", "Normalized / Position")
        self.overlay_plot.setLabel("bottom", "Time (ms)")
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
        start = float(x_range[0])
        end = float(x_range[1])
        span = max(1.0, end - start)
        lead_margin = span * 0.10

        if time_ms > (end - lead_margin):
            new_start = time_ms - (span * 0.80)
            self._apply_view_range(new_start, new_start + span)
            self._sync_nav_from_view()
            return

        if time_ms < (start + lead_margin):
            new_start = time_ms - (span * 0.20)
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

    def zoom_to_range(self, start_ms: float, end_ms: float) -> None:
        self._apply_view_range(start_ms, end_ms)
        self._sync_nav_from_view()


# ---------------------------------------------------------------------------
# Axis grouping for AuxAxisPanel
# ---------------------------------------------------------------------------

_AXIS_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Alpha / Beta", [("alpha", "#4dd0e1"), ("beta", "#ff80ab")]),
    ("Electrodes", [("e1", "#90caf9"), ("e2", "#a5d6a7"), ("e3", "#ffcc80"), ("e4", "#ce93d8")]),
    ("Alpha Prostate / Beta Prostate", [("alpha_prostate", "#26a69a"), ("beta_prostate", "#ec407a")]),
    ("Frequency", [("frequency", "#66bb6a")]),
    ("Pulse Frequency", [("pulse_frequency", "#42a5f5")]),
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
    Only groups that have data are shown.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        # group_title -> (PlotWidget, {axis_name: PlotDataItem}, InfiniteLine)
        self._group_plots: dict[str, tuple[pg.PlotWidget, dict[str, pg.PlotDataItem], pg.InfiniteLine]] = {}

        # Full-resolution data for LOD: axis_name -> (x, y)
        self._lod_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        # Reference to main overlay_plot for X-axis linking
        self._main_plot: pg.PlotWidget | None = None

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

        # Create or update groups in canonical order
        for grp_title, members in _AXIS_GROUPS:
            if grp_title not in active_groups:
                continue

            if grp_title not in self._group_plots:
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
                self._layout.addWidget(plot)
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


__all__ = [
    "AuxAxisPanel",
    "PlaybackPanel",
    "PositionTimelinePanel",
    "SpectralFluxPanel",
    "SpeedHeatmapPanel",
    "TimeAxisSync",
    "VisualizationArea",
    "WaveformPanel",
]
