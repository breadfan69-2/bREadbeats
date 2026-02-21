"""
bREadbeats - Main Application
Qt GUI with beat detection, stroke mapping, and spectrum visualization.
"""

# Heavy imports - these are the slow ones, but splash is already showing by this point
import sys
import atexit
import socket
from contextlib import contextmanager
import time
import json
from datetime import datetime
import subprocess
import os

_DEBUG_STDIO_ENABLED = os.environ.get("BREADBEATS_DEBUG_STDIO", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
if bool(getattr(sys, "frozen", False)) and not _DEBUG_STDIO_ENABLED:
    _null_stream = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = _null_stream
    sys.stderr = _null_stream

_import_t0 = time.perf_counter()
print("\n[Startup] main.py loading heavy modules...", flush=True)

import numpy as np
import queue
import threading
import random
from collections import deque
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSlider, QComboBox, QPushButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QFrame,
    QGridLayout, QMenuBar, QMenu, QMessageBox,
    QSplashScreen, QScrollArea, QInputDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRectF
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QPixmap
from typing import Any, Optional

# PyQtGraph for high-performance real-time plotting
import pyqtgraph as pg
pg.setConfigOptions(antialias=False, useOpenGL=False)  # Disable for compatibility

from config import (
    BEAT_RANGE_LIMITS,
    BEAT_RESET_DEFAULTS,
    BeatDetectionType,
    Config,
    CURRENT_CONFIG_VERSION,
    DeviceLimitsConfig,
    StrokeMode,
    apply_dict_to_dataclass,
    migrate_config,
)
from logging_utils import get_log_level, log_event, set_log_level
from audio_engine import AudioEngine, BeatEvent
from network_engine import NetworkEngine, TCodeCommand
from network_lifecycle import ensure_network_engine, toggle_user_connection
from command_wiring import attach_cached_tcode_values, apply_volume_ramp
from close_persist_wiring import persist_runtime_ui_to_config
from config_facade import (
    get_config_dir,
    get_config_file,
    load_config,
    save_config,
)
from frequency_utils import extract_dominant_freq
from transport_wiring import (
    begin_volume_ramp,
    play_button_text,
    send_zero_volume_immediate,
    set_transport_sending,
    shutdown_runtime,
    start_stop_ui_state,
    trigger_network_test,
)
from stroke_mapper import StrokeMapper

print(f"[Startup] main.py imports ready (+{(time.perf_counter()-_import_t0)*1000:.0f} ms)", flush=True)


def _track_slider_value(name: str, value: float) -> None:
    return


_apply_dict_to_dataclass = apply_dict_to_dataclass
_migrate_config = migrate_config


_CHILD_PROCESSES: set[subprocess.Popen] = set()
_CHILD_PROCESSES_LOCK = threading.Lock()


def _cleanup_child_processes() -> None:
    with _CHILD_PROCESSES_LOCK:
        procs = list(_CHILD_PROCESSES)
        _CHILD_PROCESSES.clear()

    for proc in procs:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


def _spawn_background_process(args: list[str]) -> subprocess.Popen | None:
    popen_kwargs: dict = {"shell": False}
    if os.name == 'nt':
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.Popen(args, **popen_kwargs)
        with _CHILD_PROCESSES_LOCK:
            _CHILD_PROCESSES.add(proc)
        return proc
    except Exception:
        return None


class _SingleInstanceLock:
    def __init__(self, port: int = 48173):
        self._port = int(port)
        self._sock: socket.socket | None = None

    def acquire(self) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self._port))
            sock.listen(1)
            self._sock = sock
            return True
        except OSError:
            try:
                sock.close()
            except Exception:
                pass
            return False

    def release(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


atexit.register(_cleanup_child_processes)


class SignalBridge(QObject):
    """Bridge for thread-safe signal emission"""
    beat_detected = pyqtSignal(object)
    spectrum_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str, bool)


class FFTBinBarGraphCanvas(pg.PlotWidget):
    """Exact FFT-bin bar graph visualizer (one bar per incoming FFT bin)."""

    def __init__(self, parent=None, width=8, height=3):
        super().__init__(parent)

        self.setBackground('#0a0a12')
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.showGrid(x=False, y=True, alpha=0.08)
        self.showAxis('left')
        self.showAxis('bottom')
        self.getAxis('left').setTextPen(pg.mkPen('#888888'))
        self.getAxis('left').setTickPen(pg.mkPen('#666666'))
        self.getAxis('bottom').setTextPen(pg.mkPen('#888888'))
        self.getAxis('bottom').setTickPen(pg.mkPen('#666666'))
        self.getAxis('left').setLabel('dBFS')
        self.getAxis('bottom').setLabel('FFT Bin')

        self._display_floor_db = -90.0
        self._display_ceil_db = 0.0
        self._bar_item: Optional[pg.BarGraphItem] = None
        self._bar_count = 0
        self._bar_x = np.array([], dtype=np.float32)
        self._bar_floor = np.array([], dtype=np.float32)
        self._latest_peak_db = self._display_floor_db
        self._ghost_overlays: dict[str, dict] = {}
        self._ghost_timer = QTimer(self)
        self._ghost_timer.setInterval(80)
        self._ghost_timer.timeout.connect(self._tick_ghosts)

        self.setXRange(0, 1)
        self.setYRange(self._display_floor_db, self._display_ceil_db)

    def _ensure_bars(self, count: int):
        """Rebuild bars when FFT-bin count changes."""
        count = max(1, int(count))
        if count == self._bar_count and self._bar_item is not None:
            return

        if self._bar_item is not None:
            self.removeItem(self._bar_item)

        self._bar_count = count
        self._bar_x = np.arange(count, dtype=np.float32)
        self._bar_floor = np.full(count, self._display_floor_db, dtype=np.float32)
        heights = np.zeros(count, dtype=np.float32)

        self._bar_item = pg.BarGraphItem(
            x=self._bar_x,
            y0=self._bar_floor,
            height=heights,
            width=0.9,
            brush=pg.mkBrush(90, 200, 255, 180),
            pen=pg.mkPen(120, 230, 255, 180),
        )
        self.addItem(self._bar_item)
        self.setXRange(-0.5, count - 0.5)

    def set_peak_and_flux(self, peak_value: float, flux_value: float):
        """Compatibility no-op for shared visualizer interfaces."""
        pass

    def set_peak_indicators_visible(self, visible: bool):
        """Compatibility no-op for shared visualizer interfaces."""
        pass

    def set_range_indicators_visible(self, visible: bool):
        """Compatibility no-op for shared visualizer interfaces."""
        pass

    def update_from_spectrum(self, spectrum: np.ndarray, sample_rate: int):
        """Render exact incoming FFT bins without bin interpolation/merging."""
        if spectrum is None:
            return

        arr = np.asarray(spectrum, dtype=np.float32)
        if arr.size == 0:
            return

        self._ensure_bars(arr.size)
        bin_width_hz = float(sample_rate) / float(max(1, (arr.size - 1) * 2))
        self.getAxis('bottom').setLabel('FFT Bin', units=f'Δf={bin_width_hz:.2f}Hz')

        db_values = 20.0 * np.log10(np.maximum(arr, 1e-12))
        self._latest_peak_db = float(np.max(db_values)) if db_values.size > 0 else self._display_floor_db
        db_values = np.clip(db_values, self._display_floor_db, self._display_ceil_db)
        heights = db_values - self._display_floor_db

        if self._bar_item is not None:
            self._bar_item.setOpts(height=heights, y0=self._bar_floor)

    def update_spectrum(self, spectrum: np.ndarray, peak_energy: Optional[float] = None, spectral_flux: Optional[float] = None):
        """Compatibility wrapper for callers that use update_spectrum."""
        self.update_from_spectrum(spectrum, 44100)

    def show_fill_ratio_ghost(self, key: str, ratio: float, label: str, color: str = '#66E0FF', duration_s: float = 5.0, dashed: bool = True) -> None:
        """Show temporary dB-threshold line from fill ratio using live FFT peak reference."""
        ratio_clamped = float(np.clip(ratio, 0.0, 1.0))
        peak_db = float(np.clip(self._latest_peak_db, self._display_floor_db, self._display_ceil_db))
        threshold_db = peak_db + (20.0 * np.log10(max(ratio_clamped, 1e-6)))
        threshold_db = float(np.clip(threshold_db, self._display_floor_db, self._display_ceil_db))
        self._show_ghost(
            key=key,
            label=f"{label}: {ratio_clamped:.3f} (~{threshold_db:.1f} dB)",
            color=color,
            duration_s=duration_s,
            dashed=dashed,
            mode='line',
            y=threshold_db,
        )

    def show_bin_range_ghost(self, key: str, low_bin: int, high_bin: int, label: str, color: str = '#FFFFFF', duration_s: float = 5.0, dashed: bool = False) -> None:
        """Show temporary FFT-bin range box aligned to real bin indices."""
        if self._bar_count <= 0:
            return
        lo = int(np.clip(min(low_bin, high_bin), 0, self._bar_count - 1))
        hi = int(np.clip(max(low_bin, high_bin), 0, self._bar_count - 1))
        self._show_ghost(
            key=key,
            label=f"{label}: bins {lo}-{hi}",
            color=color,
            duration_s=duration_s,
            dashed=dashed,
            mode='box',
            x0=float(lo - 0.5),
            x1=float(hi + 0.5),
            y0=self._display_floor_db,
            y1=self._display_ceil_db,
        )

    def _show_ghost(
        self,
        *,
        key: str,
        label: str,
        color: str,
        duration_s: float,
        dashed: bool,
        mode: str,
        y: float | None = None,
        x0: float | None = None,
        x1: float | None = None,
        y0: float | None = None,
        y1: float | None = None,
    ) -> None:
        now = time.monotonic()
        overlay = self._ghost_overlays.get(key)
        if overlay is None:
            qcolor = QColor(color)
            line = pg.InfiniteLine(pos=0.0, angle=0, movable=False, pen=pg.mkPen(qcolor, width=1, style=(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)))
            line.setZValue(30)
            self.addItem(line)

            text = pg.TextItem('', color=qcolor, anchor=(0.0, 1.0))
            text.setZValue(31)
            self.addItem(text)

            box = pg.QtWidgets.QGraphicsRectItem()
            box.setZValue(29)
            self.addItem(box)
            box.hide()

            overlay = {
                'line': line,
                'text': text,
                'box': box,
                'color': qcolor,
                'started_at': now,
                'duration_s': float(max(0.5, duration_s)),
            }
            self._ghost_overlays[key] = overlay

        overlay['started_at'] = now
        overlay['duration_s'] = float(max(0.5, duration_s))
        overlay['dashed'] = bool(dashed)
        overlay['mode'] = mode

        text_item = overlay['text']
        line_item = overlay['line']
        box_item = overlay['box']
        text_item.setText(label)

        if mode == 'line' and y is not None:
            line_item.show()
            line_item.setPos(float(y))
            text_item.setPos(1.0, min(self._display_ceil_db - 0.5, float(y) + 0.8))
            box_item.hide()
            overlay['base_rect'] = None
        elif mode == 'box' and None not in (x0, x1, y0, y1):
            assert x0 is not None and x1 is not None and y0 is not None and y1 is not None
            x0f = float(x0)
            x1f = float(x1)
            y0f = float(y0)
            y1f = float(y1)
            line_item.hide()
            text_item.setPos(x0f + 0.25, self._display_ceil_db - 0.5)
            overlay['base_rect'] = QRectF(
                min(x0f, x1f),
                min(y0f, y1f),
                max(0.001, abs(x1f - x0f)),
                max(0.2, abs(y1f - y0f)),
            )
            box_item.show()

        self._apply_ghost_style(overlay, 0.0)
        if not self._ghost_timer.isActive():
            self._ghost_timer.start()

    def _apply_ghost_style(self, overlay: dict, progress: float) -> None:
        eased = float(np.clip(progress, 0.0, 1.0))
        alpha = max(0, min(230, int(230 * (1.0 - eased))))

        color = QColor(overlay['color'])
        color.setAlpha(alpha)
        line = overlay['line']
        line.setPen(pg.mkPen(color, width=1, style=(Qt.PenStyle.DashLine if overlay.get('dashed', False) else Qt.PenStyle.SolidLine)))
        overlay['text'].setColor(color)

        base_rect = overlay.get('base_rect')
        box = overlay.get('box')
        if base_rect is not None and box is not None and box.isVisible():
            box.setRect(base_rect)
            pen = QPen(color)
            pen.setWidthF(0.9)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine if overlay.get('dashed', False) else Qt.PenStyle.SolidLine)
            box.setPen(pen)
            box.setBrush(QBrush(Qt.BrushStyle.NoBrush))

    def _tick_ghosts(self) -> None:
        if not self._ghost_overlays:
            self._ghost_timer.stop()
            return

        now = time.monotonic()
        expired: list[str] = []
        for key, overlay in list(self._ghost_overlays.items()):
            elapsed = max(0.0, now - float(overlay.get('started_at', now)))
            duration = max(0.5, float(overlay.get('duration_s', 5.0)))
            progress = elapsed / duration
            if progress >= 1.0:
                expired.append(key)
                continue
            self._apply_ghost_style(overlay, progress)

        for key in expired:
            overlay = self._ghost_overlays.pop(key, None)
            if overlay is None:
                continue
            for item_key in ('line', 'text', 'box'):
                item = overlay.get(item_key)
                if item is not None:
                    try:
                        item.hide()
                    except Exception:
                        pass
                    try:
                        self.removeItem(item)
                    except Exception:
                        pass
                    try:
                        scene = item.scene()
                        if scene is not None:
                            scene.removeItem(item)
                    except Exception:
                        pass

        if not self._ghost_overlays:
            self._ghost_timer.stop()


def launch_projectm():
    """Attempt to launch projectM standalone application"""
    import shutil
    
    # Common projectM executable paths
    possible_paths = [
        # Steam installation
        r"C:\Program Files (x86)\Steam\steamapps\common\projectM Music Visualizer\projectM.exe",
        r"C:\Program Files\Steam\steamapps\common\projectM Music Visualizer\projectM.exe",
        # Standalone installation
        r"C:\Program Files\projectM\projectM.exe",
        r"C:\Program Files (x86)\projectM\projectM.exe",
        # Check PATH
        "projectM",
        "projectm",
    ]
    
    for path in possible_paths:
        if path in ["projectM", "projectm"]:
            # Try to find in PATH
            if shutil.which(path):
                try:
                    proc = _spawn_background_process([path])
                    if proc is None:
                        raise RuntimeError("launch failed")
                    print(f"[Visualizer] Launched projectM from PATH")
                    return True
                except:
                    continue
        else:
            if os.path.exists(path):
                try:
                    proc = _spawn_background_process([path])
                    if proc is None:
                        raise RuntimeError("launch failed")
                    print(f"[Visualizer] Launched projectM from {path}")
                    return True
                except Exception as e:
                    print(f"[Visualizer] Failed to launch {path}: {e}")
                    continue
    
    print("[Visualizer] projectM not found. Install from Steam or https://github.com/projectM-visualizer/projectm")
    return False


class PositionCanvas(pg.PlotWidget):
    """Alpha/Beta position visualizer using PyQtGraph - circular display"""
    
    def __init__(self, parent=None, size=2, get_rotation=None):
        super().__init__(parent)
        
        # Match window background for ghost effect
        self.setBackground('#3d3d3d')
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.setAspectLocked(True)
        
        # Fixed axis ranges
        self.setXRange(-1.2, 1.2)
        self.setYRange(-1.2, 1.2)
        self.hideAxis('left')
        self.hideAxis('bottom')
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        self.addItem(pg.PlotCurveItem(circle_x, circle_y, pen=pg.mkPen('#555555', width=1)))
        
        # Draw crosshairs
        self.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#4a4a4a', width=0.5)))
        self.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#4a4a4a', width=0.5)))
        
        # Trail storage
        self.trail_x = []
        self.trail_y = []
        self.max_trail = 50
        self.trail_curve = pg.PlotCurveItem(pen=pg.mkPen('#00aaff', width=1))
        self.addItem(self.trail_curve)
        
        # Current position marker
        self.position_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush('#00ffff'), pen=None)
        self.addItem(self.position_scatter)
        
        self.get_rotation = get_rotation

    def update_position(self, alpha: float, beta: float):
        # Alpha = vertical (y-axis): 1.0 = top, -1.0 = bottom
        # Beta = horizontal (x-axis): 1.0 = LEFT, -1.0 = right (matches restim orientation)
        angle_deg = self.get_rotation() if self.get_rotation else 0.0
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        x_base = -beta  # Negated to match restim
        y_base = alpha
        x_rot = x_base * cos_a - y_base * sin_a
        y_rot = x_base * sin_a + y_base * cos_a
        
        # 90° CCW rotation so our display matches restim orientation
        x_display = -y_rot
        y_display = x_rot
        
        self.trail_x.append(x_display)
        self.trail_y.append(y_display)
        if len(self.trail_x) > self.max_trail:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
        
        # Update trail curve
        if len(self.trail_x) > 1:
            self.trail_curve.setData(self.trail_x, self.trail_y)
        
        # Update position marker
        self.position_scatter.setData([x_display], [y_display])


class RangeSlider(QWidget):
    """A slider with two handles for selecting a range - can grab middle to slide entire range"""
    
    rangeChanged = pyqtSignal(float, float)  # low, high
    
    def __init__(self, min_val: float, max_val: float, low_default: float, 
                 high_default: float, decimals: int = 0, log_scale: bool = False, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
        self.log_scale = log_scale
        self._low = low_default
        self._high = high_default
        self._dragging = None  # 'low', 'high', 'range', or None
        self._drag_offset = 0  # Offset from click position to range start when dragging range
        self._handle_width = 12
        self.setMinimumHeight(24)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
    def low(self) -> float:
        return self._low
    
    def high(self) -> float:
        return self._high
    
    def setLow(self, value: float):
        self._low = max(self.min_val, min(value, self._high - 1))
        self.update()
        
    def setHigh(self, value: float):
        self._high = min(self.max_val, max(value, self._low + 1))
        self.update()
    
    def _val_to_pos(self, value: float) -> int:
        """Convert value to pixel position"""
        if self.log_scale and self.min_val > 0 and self.max_val > 0:
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            ratio = (np.log10(max(value, self.min_val)) - log_min) / (log_max - log_min)
        else:
            ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return int(self._handle_width/2 + ratio * (self.width() - self._handle_width))
    
    def _pos_to_val(self, pos: float) -> float:
        """Convert pixel position to value"""
        ratio = (pos - self._handle_width/2) / (self.width() - self._handle_width)
        ratio = max(0, min(1, ratio))
        if self.log_scale and self.min_val > 0 and self.max_val > 0:
            log_min = np.log10(self.min_val)
            log_max = np.log10(self.max_val)
            return 10 ** (log_min + ratio * (log_max - log_min))
        else:
            return self.min_val + ratio * (self.max_val - self.min_val)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        h = self.height()
        w = self.width()
        track_y = h // 2 - 4
        track_h = 8
        
        # Draw track background - matches QSlider groove
        painter.setBrush(QBrush(QColor(0x5d, 0x5d, 0x5d)))  # #5d5d5d
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, track_y, w, track_h, 4, 4)
        
        # Draw selected range - dark turquoise accent
        low_pos = self._val_to_pos(self._low)
        high_pos = self._val_to_pos(self._high)
        painter.setBrush(QBrush(QColor(0x00, 0x8b, 0x8b)))  # #008b8b
        painter.drawRoundedRect(low_pos, track_y, high_pos - low_pos, track_h, 4, 4)
        
        # Draw handles - matches QSlider handle
        handle_w = 18
        handle_h = 18
        handle_y = h // 2 - handle_h // 2
        
        # Low handle
        painter.setBrush(QBrush(QColor(0x00, 0x8b, 0x8b)))  # #008b8b
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(low_pos - handle_w//2, handle_y, handle_w, handle_h)
        
        # High handle
        painter.drawEllipse(high_pos - handle_w//2, handle_y, handle_w, handle_h)
        
    def mousePressEvent(self, event):
        pos = event.position().x()
        low_pos = self._val_to_pos(self._low)
        high_pos = self._val_to_pos(self._high)
        
        # Determine which handle is closer
        dist_to_low = abs(pos - low_pos)
        dist_to_high = abs(pos - high_pos)
        handle_threshold = self._handle_width * 1.5  # Generous touch area for handles
        
        # Check if clicking on or very near a handle first
        if dist_to_low < handle_threshold and dist_to_low <= dist_to_high:
            self._dragging = 'low'
        elif dist_to_high < handle_threshold:
            self._dragging = 'high'
        # Check if clicking in the middle area (between handles) - grab entire range
        elif low_pos < pos < high_pos:
            self._dragging = 'range'
            # Store offset from click to low value for smooth dragging
            self._drag_offset = self._pos_to_val(pos) - self._low
        else:
            # Click outside range - move closest handle to that position
            if dist_to_low < dist_to_high:
                self._dragging = 'low'
            else:
                self._dragging = 'high'
            self._update_from_pos(pos)
        
    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_from_pos(event.position().x())
    
    def mouseReleaseEvent(self, event):
        self._dragging = None
    
    def _update_from_pos(self, pos: float):
        value = self._pos_to_val(pos)
        if self.decimals == 0:
            value = round(value)
        else:
            value = round(value, self.decimals)
            
        if self._dragging == 'low':
            if value < self._high:
                self._low = max(self.min_val, value)
        elif self._dragging == 'high':
            if value > self._low:
                self._high = min(self.max_val, value)
        elif self._dragging == 'range':
            # Move entire range while maintaining width
            range_width = self._high - self._low
            new_low = value - self._drag_offset
            
            # Round the new values
            if self.decimals == 0:
                new_low = round(new_low)
            else:
                new_low = round(new_low, self.decimals)
            
            new_high = new_low + range_width
            
            # Clamp to bounds
            if new_low < self.min_val:
                new_low = self.min_val
                new_high = new_low + range_width
            if new_high > self.max_val:
                new_high = self.max_val
                new_low = new_high - range_width
            
            self._low = new_low
            self._high = new_high
        
        self.update()
        self.rangeChanged.emit(self._low, self._high)


class RangeSliderWithLabel(QWidget):
    """Range slider with label showing current range values"""
    
    rangeChanged = pyqtSignal(float, float)
    
    def __init__(self, name: str, min_val: float, max_val: float,
                 low_default: float, high_default: float, decimals: int = 0,
                 log_scale: bool = False, parent=None):
        super().__init__(parent)
        
        self.decimals = decimals
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(name)
        self.label.setFixedWidth(120)
        self.label.setStyleSheet("color: #aaa;")
        
        self.slider = RangeSlider(min_val, max_val, low_default, high_default, decimals, log_scale)
        self.slider.rangeChanged.connect(self._on_change)
        
        self.value_label = QLabel(f"{low_default:.{decimals}f}-{high_default:.{decimals}f}")
        self.value_label.setFixedWidth(80)
        self.value_label.setStyleSheet("color: #0af;")
        
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)
    
    def _on_change(self, low: float, high: float):
        self.value_label.setText(f"{low:.{self.decimals}f}-{high:.{self.decimals}f}")
        base_name = self.label.text()
        _track_slider_value(f"{base_name} [low]", low)
        _track_slider_value(f"{base_name} [high]", high)
        self.rangeChanged.emit(low, high)
    
    def low(self) -> float:
        return self.slider.low()
    
    def high(self) -> float:
        return self.slider.high()
    
    def setLow(self, value: float):
        self.slider.setLow(value)
        self._on_change(self.slider.low(), self.slider.high())
    
    def setHigh(self, value: float):
        self.slider.setHigh(value)
        self._on_change(self.slider.low(), self.slider.high())


class SliderWithLabel(QWidget):
    """Slider with label showing current value"""
    
    valueChanged = pyqtSignal(float)
    
    def __init__(self, name: str, min_val: float, max_val: float,
                 default: float, decimals: int = 2, step: Optional[float] = None, parent=None):
        super().__init__(parent)
        
        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
        self.multiplier = 10 ** decimals
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        
        self.label = QLabel(name)
        self.label.setWordWrap(True)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.label.setStyleSheet("color: #aaa;")
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * self.multiplier))
        self.slider.setMaximum(int(max_val * self.multiplier))
        self.slider.setValue(int(default * self.multiplier))
        if step is not None:
            step_units = max(1, int(round(step * self.multiplier)))
            self.slider.setSingleStep(step_units)
            self.slider.setPageStep(step_units)
        self.slider.valueChanged.connect(self._on_change)
        
        self.value_label = QLabel(f"{default:.{decimals}f}")
        self.value_label.setFixedWidth(64)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setStyleSheet("color: #0af;")
        
        header.addWidget(self.label)
        header.addWidget(self.value_label)
        layout.addLayout(header)
        layout.addWidget(self.slider)
        
    def _on_change(self, value: int):
        real_value = value / self.multiplier
        self.value_label.setText(f"{real_value:.{self.decimals}f}")
        _track_slider_value(self.label.text(), real_value)
        self.valueChanged.emit(real_value)
        
    def value(self) -> float:
        return self.slider.value() / self.multiplier
    
    def setValue(self, value: float):
        self.slider.setValue(int(value * self.multiplier))


class TrafficLightWidget(QWidget):
    """
    Horizontal traffic light indicator for metric auto-range state:
    - Red = any metric actively ADJUSTING (hunting for good values)
    - Yellow = metrics SETTLED (some stable, some adjusting)
    - Green = all active metrics LOCKED (stable for N consecutive checks)
    All lights off when no metrics are enabled.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(54, 18)  # 3 circles of 14px diameter + spacing
        self._green_on = False
        self._yellow_on = False
        self._red_on = False
        
    def set_state(self, green: bool, yellow: bool, red: bool):
        """Set which lights are on"""
        self._green_on = green
        self._yellow_on = yellow
        self._red_on = red
        self.update()
        
    def all_off(self):
        """Turn all lights off"""
        self.set_state(False, False, False)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw 3 circles: Green, Yellow, Red (left to right)
        colors = [
            (self._green_on, QColor(0, 200, 0), QColor(0, 60, 0)),      # Green
            (self._yellow_on, QColor(255, 200, 0), QColor(80, 60, 0)),  # Yellow
            (self._red_on, QColor(255, 50, 50), QColor(80, 20, 20)),    # Red
        ]
        
        for i, (is_on, on_color, off_color) in enumerate(colors):
            x = 2 + i * 18  # 18px spacing between circles
            y = 2
            diameter = 14
            
            # Draw circle
            painter.setPen(QPen(QColor(60, 60, 60), 1))
            if is_on:
                painter.setBrush(QBrush(on_color))
            else:
                painter.setBrush(QBrush(off_color))
            painter.drawEllipse(x, y, diameter, diameter)
        
        painter.end()


class CollapsibleGroupBox(QGroupBox):
    """
    A QGroupBox that can be collapsed/expanded by clicking the title.
    When collapsed, only the title bar is visible (windowshade effect).
    Uses mousePressEvent instead of setCheckable to avoid Qt's built-in
    child-disable behavior that prevents widget interaction after expand.
    """

    def __init__(self, title: str = "", parent=None, collapsed: bool = False):
        super().__init__(title, parent)
        self._collapsed = collapsed
        self._base_title_text = title
        self._first_show = True
        self._update_title()

    def _update_title(self):
        arrow = "▶" if self._collapsed else "▼"
        self.setTitle(f"{arrow} {self._base_title_text}")
        # Large clickable title with prominent arrow
        self.setStyleSheet(self.styleSheet() + """
            CollapsibleGroupBox::title {
                font-size: 16px;
                font-weight: bold;
                padding: 6px 10px;
            }
        """)

    def mousePressEvent(self, event):
        # Toggle collapse only when clicking in the title-bar area (top ~40px)
        if event is None:
            return
        if event.position().y() <= 40:
            self.setCollapsed(not self._collapsed)
            event.accept()
        else:
            super().mousePressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        # On first show, apply collapsed state so children added after __init__
        # are properly hidden when starting collapsed
        if self._first_show:
            self._first_show = False
            if self._collapsed:
                self._apply_visibility(False)

    def _apply_visibility(self, visible: bool):
        layout = self.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item is None:
                    continue
                widget = item.widget()
                if widget:
                    widget.setVisible(visible)
                inner_layout = item.layout()
                if inner_layout:
                    self._set_layout_visible(inner_layout, visible)

    def _set_layout_visible(self, layout, visible: bool):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item is None:
                continue
            widget = item.widget()
            if widget:
                widget.setVisible(visible)
            inner = item.layout()
            if inner:
                self._set_layout_visible(inner, visible)

    def setCollapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self._apply_visibility(not collapsed)
        self._update_title()

    def isCollapsed(self) -> bool:
        return self._collapsed


class NoWheelScrollArea(QScrollArea):
    """
    Custom QScrollArea that ignores mouse wheel events.
    Prevents scroll interference when adjusting parameter sliders.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def wheelEvent(self, event):
        # Ignore wheel events - do not scroll the container
        event.ignore()


class WaveformLiveCanvas(pg.PlotWidget):
    """Simple in-window waveform visualizer (no calibration overlays)."""

    def __init__(self, parent=None, width=8, height=3):
        super().__init__(parent)
        self.setBackground('#0d0d0d')
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True, alpha=0.08)
        self.showAxis('left')
        self.showAxis('bottom')
        self.getAxis('left').setTextPen(pg.mkPen('#ffffff'))
        self.getAxis('left').setTickPen(pg.mkPen('#ffffff'))
        self.getAxis('bottom').setTextPen(pg.mkPen('#ffffff'))
        self.getAxis('bottom').setTickPen(pg.mkPen('#ffffff'))
        self.setLabel('left', 'Amplitude', units='A.U.')
        self.setLabel('bottom', 'Time', units='ms')
        self.enableAutoRange(x=False, y=False)
        self.setYRange(-1.05, 1.05)
        self.setXRange(0.0, 25.0)

        self.waveform_curve = self.plot(pen=pg.mkPen(120, 230, 255, 220, width=2))
        self.zero_line = pg.InfiniteLine(pos=0.0, angle=0, movable=False, pen=pg.mkPen('#7f7f7f', width=1))
        self.addItem(self.zero_line)

        self._sample_rate = 44100
        self._x_max_ms = 25.0
        self._latest_peak = 0.0
        self._reference_overlays: dict[str, dict] = {}
        self._fill_ratio_overlays: dict[str, dict] = {}
        self._reference_fade_timer = QTimer(self)
        self._reference_fade_timer.setInterval(80)
        self._reference_fade_timer.timeout.connect(self._tick_reference_overlays)

    def show_reference_line(self, key: str, value: float, label: str, color: str = '#FF66AA', duration_s: float = 15.0, dashed: bool = False) -> None:
        """Show or refresh a temporary symmetric +/- amplitude guide that fades out."""
        amp = float(np.clip(abs(value), 0.0, 1.0))
        now = time.monotonic()

        overlay = self._reference_overlays.get(key)
        if overlay is None:
            qcolor = QColor(color)
            pen = pg.mkPen(qcolor, width=1, style=(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
            line_pos = pg.InfiniteLine(pos=amp, angle=0, movable=False, pen=pen)
            line_neg = pg.InfiniteLine(pos=-amp, angle=0, movable=False, pen=pen)
            line_pos.setZValue(15)
            line_neg.setZValue(15)
            self.addItem(line_pos)
            self.addItem(line_neg)

            text = pg.TextItem("", color=qcolor, anchor=(1.0, 0.0))
            text.setZValue(16)
            self.addItem(text)

            overlay = {
                'line_pos': line_pos,
                'line_neg': line_neg,
                'text': text,
                'color': qcolor,
                'dashed': bool(dashed),
                'started_at': now,
                'duration_s': float(max(0.5, duration_s)),
            }
            self._reference_overlays[key] = overlay

        overlay['started_at'] = now
        overlay['duration_s'] = float(max(0.5, duration_s))

        line_pos = overlay['line_pos']
        line_neg = overlay['line_neg']
        text = overlay['text']
        line_pos.setPos(amp)
        line_neg.setPos(-amp)
        text.setPos(self._x_max_ms - 0.2, min(1.0, amp + 0.03))
        text.setText(f"{label}: ±{amp:.3f}")

        full_color = QColor(overlay['color'])
        full_color.setAlpha(210)
        pen = pg.mkPen(full_color, width=1, style=(Qt.PenStyle.DashLine if overlay.get('dashed', False) else Qt.PenStyle.SolidLine))
        line_pos.setPen(pen)
        line_neg.setPen(pen)
        text.setColor(full_color)

        if not self._reference_fade_timer.isActive():
            self._reference_fade_timer.start()

    def show_fill_ratio_ghost(self, key: str, ratio: float, label: str, color: str = '#FFFFFF', duration_s: float = 15.0, dashed: bool = False) -> None:
        """Show or refresh a temporary symmetric static fill-ratio band around zero."""
        ratio_clamped = float(np.clip(ratio, 0.0, 1.0))
        amp = ratio_clamped
        now = time.monotonic()

        overlay = self._fill_ratio_overlays.get(key)
        if overlay is None:
            qcolor = QColor(color)
            pen = pg.mkPen(qcolor, width=1, style=(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine))
            line_pos = pg.InfiniteLine(pos=amp, angle=0, movable=False, pen=pen)
            line_neg = pg.InfiniteLine(pos=-amp, angle=0, movable=False, pen=pen)
            line_pos.setZValue(15)
            line_neg.setZValue(15)
            self.addItem(line_pos)
            self.addItem(line_neg)

            text = pg.TextItem("", color=qcolor, anchor=(1.0, 0.0))
            text.setZValue(16)
            self.addItem(text)

            overlay = {
                'line_pos': line_pos,
                'line_neg': line_neg,
                'text': text,
                'color': qcolor,
                'dashed': bool(dashed),
                'started_at': now,
                'duration_s': float(max(0.5, duration_s)),
            }
            self._fill_ratio_overlays[key] = overlay

        overlay['started_at'] = now
        overlay['duration_s'] = float(max(0.5, duration_s))
        overlay['line_pos'].setPos(amp)
        overlay['line_neg'].setPos(-amp)
        overlay['text'].setPos(self._x_max_ms - 0.2, min(1.0, amp + 0.03))
        overlay['text'].setText(f"{label}: {ratio_clamped * 100.0:.0f}%")

        full_color = QColor(overlay['color'])
        full_color.setAlpha(210)
        pen = pg.mkPen(full_color, width=1, style=(Qt.PenStyle.DashLine if overlay.get('dashed', False) else Qt.PenStyle.SolidLine))
        overlay['line_pos'].setPen(pen)
        overlay['line_neg'].setPen(pen)
        overlay['text'].setColor(full_color)

        if not self._reference_fade_timer.isActive():
            self._reference_fade_timer.start()

    def _tick_reference_overlays(self) -> None:
        """Fade and remove expired amplitude reference overlays."""
        if not self._reference_overlays and not self._fill_ratio_overlays:
            self._reference_fade_timer.stop()
            return

        now = time.monotonic()
        expired: list[str] = []
        for key, overlay in self._reference_overlays.items():
            age = now - float(overlay['started_at'])
            duration_s = float(overlay['duration_s'])
            if age >= duration_s:
                self.removeItem(overlay['line_pos'])
                self.removeItem(overlay['line_neg'])
                self.removeItem(overlay['text'])
                expired.append(key)
                continue

            fade = max(0.0, 1.0 - (age / duration_s))
            alpha = int(210 * fade)
            qcolor = QColor(overlay['color'])
            qcolor.setAlpha(alpha)
            pen = pg.mkPen(qcolor, width=1, style=(Qt.PenStyle.DashLine if overlay.get('dashed', False) else Qt.PenStyle.SolidLine))
            overlay['line_pos'].setPen(pen)
            overlay['line_neg'].setPen(pen)
            overlay['text'].setColor(qcolor)

        for key in expired:
            self._reference_overlays.pop(key, None)

        expired_ratio: list[str] = []
        for key, overlay in self._fill_ratio_overlays.items():
            age = now - float(overlay['started_at'])
            duration_s = float(overlay['duration_s'])
            if age >= duration_s:
                self.removeItem(overlay['line_pos'])
                self.removeItem(overlay['line_neg'])
                self.removeItem(overlay['text'])
                expired_ratio.append(key)
                continue

            fade = max(0.0, 1.0 - (age / duration_s))
            alpha = int(210 * fade)
            qcolor = QColor(overlay['color'])
            qcolor.setAlpha(alpha)
            pen = pg.mkPen(qcolor, width=1, style=(Qt.PenStyle.DashLine if overlay.get('dashed', False) else Qt.PenStyle.SolidLine))
            overlay['line_pos'].setPen(pen)
            overlay['line_neg'].setPen(pen)
            overlay['text'].setColor(qcolor)

        for key in expired_ratio:
            self._fill_ratio_overlays.pop(key, None)

        if not self._reference_overlays and not self._fill_ratio_overlays:
            self._reference_fade_timer.stop()

    def update_from_audio(self, waveform: Optional[np.ndarray], sample_rate: int) -> None:
        if waveform is None:
            return
        arr = np.asarray(waveform, dtype=np.float32)
        if arr.size == 0:
            return

        self._sample_rate = int(max(1, sample_rate))
        peak_abs = float(np.max(np.abs(arr)))
        if peak_abs > 1.0:
            arr = arr / peak_abs
        self._latest_peak = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0

        x_ms = (np.arange(arr.size, dtype=np.float32) / float(self._sample_rate)) * 1000.0
        x_end = float(x_ms[-1]) if x_ms.size > 0 else 25.0
        x_max = max(2.0, x_end)
        self._x_max_ms = x_max

        self.waveform_curve.setData(x=x_ms, y=arr)
        self.setXRange(0.0, x_max)

    def set_peak_indicators_visible(self, visible: bool):
        return

    def set_range_indicators_visible(self, visible: bool):
        return


class FrequencyDbLiveCanvas(pg.PlotWidget):
    """Simple in-window frequency/dB visualizer."""

    def __init__(self, parent=None, width=8, height=3):
        super().__init__(parent)
        self.setBackground('#0d0d0d')
        self.setMenuEnabled(False)
        self.showGrid(x=True, y=True, alpha=0.08)
        self.showAxis('left')
        self.showAxis('bottom')
        self.getAxis('left').setTextPen(pg.mkPen('#ffffff'))
        self.getAxis('left').setTickPen(pg.mkPen('#ffffff'))
        self.getAxis('bottom').setTextPen(pg.mkPen('#ffffff'))
        self.getAxis('bottom').setTickPen(pg.mkPen('#ffffff'))
        self.setLabel('left', 'Level', units='dB')
        self.setLabel('bottom', 'Frequency', units='kHz')
        self.enableAutoRange(x=False, y=False)
        self.setLogMode(x=True, y=False)
        self.setXRange(np.log10(0.04), np.log10(19.9))
        self.setYRange(-120.0, 6.0)

        self.db_curve = self.plot(pen=pg.mkPen(120, 230, 255, 220, width=2))
        self.zero_db_line = pg.InfiniteLine(pos=0.0, angle=0, movable=False, pen=pg.mkPen('#aaaaaa', width=1, style=Qt.PenStyle.DashLine))
        self.addItem(self.zero_db_line)
        self.beat_band = pg.LinearRegionItem(values=(np.log10(0.04), np.log10(0.50)), orientation='vertical', brush=pg.mkBrush(255, 80, 80, 22), pen=pg.mkPen('#ff6666', width=1), movable=False)
        self.depth_band = pg.LinearRegionItem(values=(np.log10(0.04), np.log10(0.50)), orientation='vertical', brush=pg.mkBrush(80, 255, 120, 18), pen=pg.mkPen('#44cc66', width=1), movable=False)
        self.p0_band = pg.LinearRegionItem(values=(np.log10(0.04), np.log10(0.50)), orientation='vertical', brush=pg.mkBrush(80, 140, 255, 18), pen=pg.mkPen('#5599ff', width=1), movable=False)
        self.f0_band = pg.LinearRegionItem(values=(np.log10(0.08), np.log10(1.50)), orientation='vertical', brush=pg.mkBrush(80, 240, 255, 16), pen=pg.mkPen('#55ddff', width=1), movable=False)
        for region in (self.beat_band, self.depth_band, self.p0_band, self.f0_band):
            region.setZValue(-2)
            self.addItem(region)

        self._sample_rate = 44100
        self._flux_ghost_overlays: dict[str, dict] = {}
        self._flux_ghost_timer = QTimer(self)
        self._flux_ghost_timer.setInterval(80)
        self._flux_ghost_timer.timeout.connect(self._tick_flux_ghosts)

    @staticmethod
    def _hz_to_log_khz(hz: float) -> float:
        khz = max(0.04, min(19.9, float(hz) / 1000.0))
        return float(np.log10(khz))

    def _set_region_hz(self, region: pg.LinearRegionItem, low_hz: float, high_hz: float) -> None:
        low = self._hz_to_log_khz(min(low_hz, high_hz))
        high = self._hz_to_log_khz(max(low_hz, high_hz))
        if high <= low:
            high = low + 1e-4
        region.setRegion((low, high))

    def set_frequency_band(self, low_ratio: float, high_ratio: float) -> None:
        nyquist = max(1.0, float(self._sample_rate) / 2.0)
        self._set_region_hz(self.beat_band, float(low_ratio) * nyquist, float(high_ratio) * nyquist)

    def set_depth_band(self, low_hz: float, high_hz: float) -> None:
        self._set_region_hz(self.depth_band, low_hz, high_hz)

    def set_p0_band(self, low_hz: float, high_hz: float) -> None:
        self._set_region_hz(self.p0_band, low_hz, high_hz)

    def set_f0_band(self, low_hz: float, high_hz: float) -> None:
        self._set_region_hz(self.f0_band, low_hz, high_hz)

    def _band_x_range(self, band: str) -> tuple[float, float]:
        nyquist_khz = min(19.9, max(0.04, float(self._sample_rate) / 2000.0))
        if band == 'low':
            return (np.log10(0.04), np.log10(0.50))
        if band == 'high':
            include_mid = True
            try:
                parent = self.parent()
                cfg = getattr(parent, 'config', None)
                stroke_cfg = getattr(cfg, 'stroke', None)
                include_mid = bool(getattr(stroke_cfg, 'high_band_include_mid', True))
            except Exception:
                include_mid = True
            high_start_khz = 0.50 if include_mid else 2.0
            high_end_khz = min(19.9, max(high_start_khz + 0.001, nyquist_khz))
            return (np.log10(high_start_khz), np.log10(high_end_khz))
        return (np.log10(0.04), np.log10(nyquist_khz))

    def show_flux_ghost(
        self,
        key: str,
        value: float,
        label: str,
        color: str = '#FF66AA',
        duration_s: float = 15.0,
        dashed: bool = False,
        band: str = 'full',
        range_box: bool = False,
        mode: str = 'threshold',
        hz_max: float | None = None,
    ) -> None:
        now = time.monotonic()
        numeric_value = float(value)
        mode_kind = str(mode or 'threshold')
        if mode_kind == 'db_line':
            y_db = float(np.clip(numeric_value, -120.0, 12.0))
        elif mode_kind in ('occupancy', 'hz_line'):
            y_db = -60.0
        else:
            y_db = float(self._to_db(numeric_value))

        overlay = self._flux_ghost_overlays.get(key)
        if overlay is None:
            qcolor = QColor(color)
            line_angle = 90 if mode_kind == 'hz_line' else 0
            line_pos = self._hz_to_log_khz(numeric_value) if mode_kind == 'hz_line' else y_db
            line = pg.InfiniteLine(pos=line_pos, angle=line_angle, movable=False, pen=pg.mkPen(qcolor, width=1, style=(Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)))
            line.setZValue(18)
            self.addItem(line)

            text = pg.TextItem('', color=qcolor, anchor=(0.0, 1.0))
            text.setZValue(19)
            self.addItem(text)

            box = pg.QtWidgets.QGraphicsRectItem()
            box.setZValue(17)
            self.addItem(box)
            box.hide()

            overlay = {
                'line': line,
                'text': text,
                'box': box,
                'color': qcolor,
                'dashed': bool(dashed),
                'started_at': now,
                'duration_s': float(max(0.5, duration_s)),
                'mode': mode,
                'base_rect': None,
            }
            self._flux_ghost_overlays[key] = overlay

        overlay['started_at'] = now
        overlay['duration_s'] = float(max(0.5, duration_s))
        overlay['mode'] = mode
        overlay['dashed'] = bool(dashed)
        overlay['hz_max'] = hz_max

        x_left, x_right = self._band_x_range(band)
        if mode_kind == 'occupancy':
            occ = float(np.clip(numeric_value, 0.0, 1.0))
            span = max(0.001, x_right - x_left)
            x_right = x_left + (span * occ)
            y_low = -120.0
            y_high = 6.0
            overlay['line'].hide()
            overlay['base_rect'] = QRectF(min(x_left, x_right), min(y_low, y_high), max(0.001, abs(x_right - x_left)), max(0.2, abs(y_high - y_low)))
            overlay['box'].show()
            overlay['text'].setText(f"{label}: {occ:.3f}")
            overlay['text'].setPos(x_left, min(5.5, y_high - 0.2))
        elif mode_kind == 'hz_line':
            x_hz = self._hz_to_log_khz(numeric_value)
            overlay['line'].show()
            overlay['line'].setPos(x_hz)
            if hz_max is not None:
                hz_hi = float(max(numeric_value, float(hz_max)))
                hz_lo = float(min(numeric_value, float(hz_max)))
                overlay['text'].setText(f"{label}: {hz_lo:.0f}-{hz_hi:.0f} Hz")
            else:
                overlay['text'].setText(f"{label}: {numeric_value:.0f} Hz")
            overlay['text'].setPos(x_hz, 5.2)
            if range_box:
                if hz_max is not None:
                    x_lo = self._hz_to_log_khz(float(min(numeric_value, float(hz_max))))
                    x_hi = self._hz_to_log_khz(float(max(numeric_value, float(hz_max))))
                    overlay['base_rect'] = QRectF(
                        min(x_lo, x_hi),
                        -120.0,
                        max(0.001, abs(x_hi - x_lo)),
                        126.0,
                    )
                else:
                    overlay['base_rect'] = QRectF(
                        x_hz - 0.002,
                        -120.0,
                        0.004,
                        126.0,
                    )
                overlay['box'].show()
            else:
                overlay['base_rect'] = None
                overlay['box'].hide()
        else:
            overlay['line'].show()
            overlay['line'].setPos(y_db)
            overlay['text'].setText(f"{label}: {numeric_value:.4f}")
            overlay['text'].setPos(x_left, min(5.5, y_db + 1.2))
            if range_box:
                overlay['base_rect'] = QRectF(
                    min(x_left, x_right),
                    y_db - 0.2,
                    max(0.001, abs(x_right - x_left)),
                    0.4,
                )
                overlay['box'].show()
            else:
                overlay['base_rect'] = None
                overlay['box'].hide()

        self._apply_flux_ghost_style(overlay, 0.0)
        if not self._flux_ghost_timer.isActive():
            self._flux_ghost_timer.start()

    def _apply_flux_ghost_style(self, overlay: dict, progress: float) -> None:
        eased = float(np.clip(progress, 0.0, 1.0))
        alpha = max(0, min(230, int(230 * (1.0 - eased))))
        color = QColor(overlay['color'])
        color.setAlpha(alpha)

        if overlay['line'].isVisible():
            overlay['line'].setPen(pg.mkPen(color, width=1, style=(Qt.PenStyle.DashLine if overlay.get('dashed', False) else Qt.PenStyle.SolidLine)))
        overlay['text'].setColor(color)

        base_rect = overlay.get('base_rect', None)
        box = overlay.get('box', None)
        if box is not None and base_rect is not None and box.isVisible():
            mode_kind = str(overlay.get('mode') or '')
            hz_range_box = mode_kind == 'hz_line' and overlay.get('hz_max') is not None
            if hz_range_box:
                box.setRect(base_rect)
            else:
                cx = base_rect.x() + (base_rect.width() / 2.0)
                cy = base_rect.y() + (base_rect.height() / 2.0)
                x_expand = (base_rect.width() * 0.08 * eased) + 0.0005
                y_expand = (6.0 * eased) if mode_kind != 'occupancy' else (2.0 * eased)
                width = max(0.001, base_rect.width() + (2.0 * x_expand))
                height = max(0.2, base_rect.height() + (2.0 * y_expand))
                rect = QRectF(cx - (width / 2.0), cy - (height / 2.0), width, height)
                box.setRect(rect)

            box_color = QColor('#FFFFFF')
            box_color.setAlpha(alpha)
            pen = QPen(box_color)
            pen.setWidthF(0.7)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.SolidLine)
            box.setPen(pen)
            box.setBrush(QBrush(Qt.BrushStyle.NoBrush))

    def _tick_flux_ghosts(self) -> None:
        if not self._flux_ghost_overlays:
            self._flux_ghost_timer.stop()
            return

        now = time.monotonic()
        expired: list[str] = []
        for key, overlay in list(self._flux_ghost_overlays.items()):
            elapsed = max(0.0, now - float(overlay.get('started_at', now)))
            duration = max(0.5, float(overlay.get('duration_s', 15.0)))
            progress = elapsed / duration
            if progress >= 1.0:
                expired.append(key)
                continue
            self._apply_flux_ghost_style(overlay, progress)

        for key in expired:
            overlay = self._flux_ghost_overlays.pop(key, None)
            if overlay is None:
                continue
            for item_key in ('line', 'text', 'box'):
                item = overlay.get(item_key)
                if item is not None:
                    try:
                        item.hide()
                    except Exception:
                        pass
                    try:
                        self.removeItem(item)
                    except Exception:
                        pass
                    try:
                        scene = item.scene()
                        if scene is not None:
                            scene.removeItem(item)
                    except Exception:
                        pass

        if not self._flux_ghost_overlays:
            self._flux_ghost_timer.stop()

    @staticmethod
    def _to_db(value: float, floor_db: float = -120.0) -> float:
        v = max(float(value), 1e-12)
        return float(np.clip(20.0 * np.log10(v), floor_db, 12.0))

    @staticmethod
    def _as_float(value, default: float = 0.0) -> float:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        try:
            parsed = np.asarray(value, dtype=np.float64).reshape(-1)
            if parsed.size > 0:
                return float(parsed[0])
        except Exception:
            pass
        return float(default)

    def update_from_spectrum(self, spectrum: Optional[np.ndarray], sample_rate: int) -> None:
        if spectrum is None:
            return
        arr = np.asarray(spectrum, dtype=np.float32)
        if arr.size == 0:
            return

        sr = int(max(1, sample_rate))
        self._sample_rate = sr
        nyquist = sr / 2.0
        freqs_khz = np.linspace(0.0, nyquist, arr.size, dtype=np.float32) / 1000.0
        db = 20.0 * np.log10(np.maximum(arr, 1e-12))
        db = np.clip(db, -120.0, 12.0)
        start_idx = 1 if freqs_khz.size > 1 else 0
        valid = (freqs_khz[start_idx:] >= 0.04) & (freqs_khz[start_idx:] <= 19.9)
        self.db_curve.setData(x=freqs_khz[start_idx:][valid], y=db[start_idx:][valid])
        self.setXRange(np.log10(0.04), np.log10(min(19.9, max(0.04, nyquist / 1000.0))))

    def set_peak_indicators_visible(self, visible: bool):
        return

    def set_range_indicators_visible(self, visible: bool):
        for region in (self.beat_band, self.depth_band, self.p0_band, self.f0_band):
            region.setVisible(visible)


class BREadbeatsWindow(QMainWindow):
    """Main application window"""
    FIXED_JITTER_AMPLITUDE = 0.012
    FIXED_JITTER_INTENSITY = 9.5
    FIXED_CREEP_SPEED = 0.25
    FIXED_AXIS_WEIGHT = 1.0
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("bREadbeats")
        self.setMinimumSize(400, 300)
        self.resize(825, 475)
        self.setStyleSheet(self._get_stylesheet())
        # Set window icon (appears in taskbar and title bar)
        try:
            from pathlib import Path
            from PyQt6.QtGui import QIcon
            
            # Handle both development and packaged (PyInstaller) modes
            if getattr(sys, 'frozen', False):
                # Running as packaged exe
                meipass = getattr(sys, '_MEIPASS', None)
                if meipass:
                    icon_path = Path(meipass) / 'bREadbeats.ico'
                else:
                    icon_path = Path(__file__).parent / 'bREadbeats.ico'
            else:
                # Running from source
                icon_path = Path(__file__).parent / 'bREadbeats.ico'
            
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
            else:
                print(f"[UI] Icon not found at: {icon_path}")
        except Exception as e:
            print(f"[UI] Could not load icon: {e}")
        
        # Initialize config from saved file (or defaults)
        self.config = load_config()
        self.config.creep.enabled = False
        self._enforce_fixed_effect_axis_values()
        
        # Initialize engines to None early (required before learning-config apply)
        self.audio_engine = None
        self.network_engine = None
        self.stroke_mapper = None
        
        self.config.stroke.mode = StrokeMode.SIMPLE_CIRCLE
        self._apply_release_learning_defaults()
        self._apply_learning_config_to_mapper()
        # Apply persisted log level early so downstream modules inherit
        set_log_level(getattr(self.config, 'log_level', 'INFO'))
        self.signals = SignalBridge()
        
        # Command queue
        self.cmd_queue = queue.Queue()
        
        # Initialize optional UI state
        self._dry_run_enabled = bool(getattr(self.config.device_limits, 'dry_run', False))
        self._advanced_controls_dialog = None
        self._advanced_flux_threshold_slider = None
        self._advanced_flux_scaling_slider = None
        self._advanced_controls_scroll = None
        self._advanced_flux_group = None
        self._beat_detection_dialog = None
        self._beat_detection_popout_content = None
        self._pulse_settings_dialog = None
        self._pulse_settings_popout_content = None
        self._tempo_tracking_dialog = None
        self._tempo_tracking_popout_content = None
        self._auto_fill_controls_dialog = None
        self._auto_fill_controls_widgets = {}
        self._motion_readiness_dialog = None
        self._motion_readiness_controls_widgets = {}
        self._developer_controls_dialog = None
        self._developer_controls_tab_widget = None
        self._developer_unlock_dialog = None
        self._developer_controls_unlocked = False
        self._trigger_settings_tab_content = None
        self._auto_fill_tab_content = None
        self._motion_readiness_tab_content = None
        self.jitter_effect_action = None
        self.connection_toggle_action = None
        self.connection_test_action = None
        self.revert_btn = None
        
        # Setup UI
        self._setup_ui()
        
        # Load config values into UI sliders
        self._apply_config_to_ui()
        
        # Initialize indicator visibility: peak visible, range bands hidden by default
        self._on_show_peak_indicators_toggle(True)
        self._on_toggle_beat_band(False)
        self._on_toggle_depth_band(False)
        
        self._schedule_startup_notices()

        # Connect signals
        self.signals.beat_detected.connect(self._on_beat)
        self.signals.spectrum_ready.connect(self._on_spectrum)
        self.signals.status_changed.connect(self._on_status_change)
        
        # Update timer for position display (30 FPS)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(33)  # ~30 FPS
        
        # Spectrum update throttling
        self._pending_spectrum = None
        self._spectrum_timer = QTimer()
        self._spectrum_timer.timeout.connect(self._do_spectrum_update)
        self._spectrum_timer.start(33)  # ~30 FPS max
        # Visual-only metric state (decoupled from beat detector thresholds)
        self._viz_prev_spectrum: Optional[np.ndarray] = None
        self._viz_peak_ref: float = 0.10
        self._viz_flux_ref: float = 0.02
        
        # Cached P0/F0 values for thread-safe access (written by audio thread, read by GUI + send_direct)
        self._cached_p0_val: Optional[int] = None  # Last computed P0 TCode value
        self._cached_f0_val: Optional[int] = None  # Last computed F0 TCode value
        self._cached_p0_enabled: bool = False
        self._cached_f0_enabled: bool = False
        self._cached_pulse_mode: int = 0  # 0=Hz, 1=Speed
        self._cached_pulse_invert: bool = False
        self._cached_f0_mode: int = 0
        self._cached_f0_invert: bool = False
        self._cached_pulse_display: str = "Pulse Freq: off"
        self._cached_carrier_display: str = "Carrier Freq: off"
        # Cached TCode Sent slider values (0-9999) for thread-safe P0/C0 computation
        self._cached_tcode_freq_min: int = 2010
        self._cached_tcode_freq_max: int = 7035
        self._cached_f0_tcode_min: int = 0
        self._cached_f0_tcode_max: int = 5000
        # Track previous enabled state for send-zero-once logic
        self._prev_p0_enabled: bool = False
        self._prev_f0_enabled: bool = False
        
        # P1 (Pulse Width) cached state
        self._cached_p1_enabled: bool = False
        self._cached_p1_val: Optional[int] = None
        self._cached_p1_mode: int = 0  # 0=Volume(RMS), 1=Hz, 2=Speed
        self._cached_p1_invert: bool = False
        self._cached_p1_display: str = "Pulse Width: off"
        self._cached_p1_tcode_min: int = 1000
        self._cached_p1_tcode_max: int = 8000
        self._prev_p1_enabled: bool = False
        
        # P3 (Rise Time) cached state
        self._cached_p3_enabled: bool = False
        self._cached_p3_val: Optional[int] = None
        self._cached_p3_mode: int = 0  # 0=Brightness(centroid), 1=Hz, 2=Speed
        self._cached_p3_invert: bool = False
        self._cached_p3_display: str = "Rise Time: off"
        self._cached_p3_tcode_min: int = 1000
        self._cached_p3_tcode_max: int = 8000
        self._prev_p3_enabled: bool = False
        
        # P0/F0 sliding window averaging (short window for low-latency response)
        import random
        self._p0_freq_window: deque = deque()  # (timestamp, norm_weighted) tuples
        self._f0_freq_window: deque = deque()  # (timestamp, norm_weighted) tuples
        self._p1_window: deque = deque()       # (timestamp, norm_weighted) tuples for Pulse Width
        self._p3_window: deque = deque()       # (timestamp, norm_weighted) tuples for Rise Time
        self._freq_window_ms: float = 80.0  # Window size in milliseconds
        self._p0_last_send_time: float = 0.0  # For throttling P0 sends
        self._f0_last_send_time: float = 0.0  # For throttling F0 sends
        self._f0_last_sent_tcode: Optional[int] = None  # Last F0 tcode value sent (for smoothing)
        self._f0_duration_base_ms: float = 220.0  # Base F0 duration (ms)
        # C0 Band mode rate limiter: fast travel for low-latency response
        self._c0_band_target: Optional[int] = None   # Current target tcode for band mode
        self._c0_band_current: Optional[int] = None   # Current sent tcode value (traveling)
        self._c0_band_last_target_time: float = 0.0   # When last target was set
        self._c0_band_travel_rate: float = 1200.0      # Max tcode change per second
        self._c0_band_max_target_delta: int = 1800     # Max target jump accepted per retarget
        self._f0_duration_variance_ms: float = 40.0   # ±variance for random duration
        self._f0_max_change_per_send: int = 1500  # Max tcode change per send
        self._last_freq_display_time: float = 0.0  # Throttle freq display updates to 100ms
        self._last_dot_alpha: float = 0.0
        self._last_dot_beta: float = 0.0
        self._last_dot_time: float = 0.0

        # Volume ramping state for play/stop
        self._volume_ramp_active: bool = False
        self._volume_ramp_start_time: float = 0.0
        self._volume_ramp_from: float = 0.0
        self._volume_ramp_to: float = 1.0
        self._volume_ramp_duration: float = 1.3  # 1.3s ramp
        # Actual tcode volume last sent (0-100), updated in audio thread, read in GUI thread
        self._last_sent_volume_pct: float = 0.0
        
        # Advanced controls dialog singleton reference
        self._advanced_controls_dialog = None
        self._advanced_flux_threshold_slider = None
        self._advanced_flux_scaling_slider = None
        self._tempo_tracking_dialog = None
        self._auto_fill_controls_dialog = None
        self._auto_fill_controls_widgets = {}
        self._motion_readiness_dialog = None
        self._motion_readiness_controls_widgets = {}
        self._developer_controls_dialog = None
        self._developer_controls_tab_widget = None
        self._developer_unlock_dialog = None
        self._developer_controls_unlocked = False
        self._trigger_settings_tab_content = None
        self._auto_fill_tab_content = None
        self._motion_readiness_tab_content = None
        
        # Auto-align target BPM tracking (wall-clock time-based)
        self._auto_align_target_enabled: bool = True  # Auto-align target BPM to metronome when stable
        self._auto_align_stable_since: float = 0.0      # time.time() when stability started
        self._auto_align_is_stable: bool = False         # currently in stable state
        self._auto_align_required_seconds: float = 0.2   # seconds of stability before first alignment
        self._auto_align_last_adjust_time: float = 0.0   # time.time() of last ±1 BPM adjustment
        self._auto_align_cooldown: float = 0.3            # seconds between each ±1 BPM step
        self._last_sensed_bpm: float = 0.0
        
        # State
        self.is_running = False
        self.is_sending = False
        self._transport_ready = False
        self._transport_transition = False
        self._transport_pending_start = False
        self._transport_pending_stop = False
        self._transport_pending_play: bool | None = None
        self._play_warmup_active: bool = False
        self._play_warmup_started_at: float = 0.0
        self._play_warmup_seen_beat: bool = False
        self._play_warmup_min_seconds: float = 1.0
        self._play_warmup_max_seconds: float = 3.0
        
        # Auto-connect TCP on startup
        self._auto_connect_tcp()

        # Mark transport UI as ready on the first event-loop turn.
        # This guarantees an early single Start click is queued then applied,
        # instead of feeling like it was dropped during startup warm-up.
        QTimer.singleShot(0, self._mark_transport_ready)

    def _enforce_fixed_effect_axis_values(self):
        self.config.jitter.amplitude = float(self.FIXED_JITTER_AMPLITUDE)
        self.config.jitter.intensity = float(self.FIXED_JITTER_INTENSITY)
        self.config.creep.speed = float(self.FIXED_CREEP_SPEED)
        self.config.alpha_weight = float(self.FIXED_AXIS_WEIGHT)
        self.config.beta_weight = float(self.FIXED_AXIS_WEIGHT)

    def _mark_transport_ready(self) -> None:
        """Enable transport input after startup and apply queued Start, if any."""
        self._transport_ready = True
        self._sync_transport_buttons()
        if self._transport_pending_start and not self.is_running:
            self._transport_pending_start = False
            QTimer.singleShot(0, self._apply_pending_start)
        
    def _get_stylesheet(self) -> str:
        """Restim-Coyote3 darkmode theme with #3d3d3d background"""
        return """
            /* Main Window and Widgets */
            QMainWindow, QWidget {
                background-color: #3d3d3d;
                color: #e0e0e0;
            }

            QFrame {
                background-color: #3d3d3d;
                color: #e0e0e0;
            }

            /* Menu Bar */
            QMenuBar {
                background-color: #4d4d4d;
                color: #e0e0e0;
                border-bottom: 1px solid #5d5d5d;
            }

            QMenuBar::item:selected {
                background-color: #5d5d5d;
            }

            /* Menus */
            QMenu {
                background-color: #4d4d4d;
                color: #e0e0e0;
                border: 1px solid #5d5d5d;
            }

            QMenu::item:selected {
                background-color: #008b8b;
                color: #ffffff;
            }

            /* Buttons */
            QPushButton {
                background-color: #565d7f;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 5px 15px;
            }

            QPushButton:hover {
                background-color: #6d6d8f;
            }

            QPushButton:checked {
                background-color: #008b8b;
                color: #ffffff;
            }

            QPushButton:checked:hover {
                background-color: #109b9b;
            }

            QPushButton:checked:pressed {
                background-color: #006f6f;
            }

            QPushButton:pressed {
                background-color: #4a4d6f;
            }

            QPushButton:disabled {
                background-color: #424242;
                color: #757575;
            }

            /* Labels */
            QLabel {
                color: #e0e0e0;
            }

            /* Line Edit */
            QLineEdit {
                background-color: #4d4d4d;
                color: #e0e0e0;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
                padding: 5px;
            }

            QLineEdit:focus {
                border: 1px solid #565d7f;
            }

            /* Spin Box */
            QSpinBox, QDoubleSpinBox {
                background-color: #4d4d4d;
                color: #e0e0e0;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
                padding: 5px;
            }

            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background-color: #3d3d3d;
                border: 1px solid #2d2d2d;
                width: 20px;
            }

            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #4d4d4d;
            }

            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #565d7f;
            }

            /* Sliders */
            QSlider::groove:horizontal {
                background-color: #5d5d5d;
                height: 8px;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background-color: #565d7f;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }

            QSlider::handle:horizontal:hover {
                background-color: #6d6d8f;
            }

            /* ComboBox */
            QComboBox {
                background-color: #4d4d4d;
                color: #e0e0e0;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
                padding: 5px;
            }

            QComboBox:focus {
                border: 1px solid #565d7f;
            }

            QComboBox::drop-down {
                border: none;
                width: 20px;
            }

            /* CheckBox and RadioButton */
            QCheckBox, QRadioButton {
                color: #e0e0e0;
            }

            QCheckBox::indicator, QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }

            QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked {
                background-color: #4d4d4d;
                border: 1px solid #5d5d5d;
                border-radius: 3px;
            }

            QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                background-color: #008b8b;
                border: 1px solid #008b8b;
                border-radius: 3px;
            }

            /* GroupBox */
            QGroupBox {
                color: #e0e0e0;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }

            QGroupBox::indicator {
                width: 0px;
                height: 0px;
            }

            /* Tabs */
            QTabBar::tab {
                background-color: #4d4d4d;
                color: #e0e0e0;
                border: 1px solid #5d5d5d;
                padding: 8px 20px;
            }

            QTabBar::tab:selected {
                background-color: #008b8b;
                color: #ffffff;
            }

            QTabWidget::pane {
                border: 1px solid #5d5d5d;
            }

            /* ScrollBar */
            QScrollBar:vertical {
                background-color: #3d3d3d;
                width: 12px;
                border: none;
            }

            QScrollBar::handle:vertical {
                background-color: #626262;
                border-radius: 6px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #727272;
            }

            QScrollBar:horizontal {
                background-color: #3d3d3d;
                height: 12px;
                border: none;
            }

            QScrollBar::handle:horizontal {
                background-color: #626262;
                border-radius: 6px;
                min-width: 20px;
            }

            QScrollBar::handle:horizontal:hover {
                background-color: #727272;
            }

            /* ProgressBar */
            QProgressBar {
                background-color: #4d4d4d;
                color: #e0e0e0;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: #565d7f;
                border-radius: 3px;
            }

            /* Text Edit */
            QTextEdit, QPlainTextEdit {
                background-color: #4d4d4d;
                color: #e0e0e0;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
            }

            /* List View and Table View */
            QListView, QTableView, QTreeView {
                background-color: #4d4d4d;
                color: #e0e0e0;
                border: 1px solid #5d5d5d;
                gridline-color: #5d5d5d;
            }

            QListView::item:selected, QTableView::item:selected, QTreeView::item:selected {
                background-color: #008b8b;
            }

            /* Dialogs */
            QDialog {
                background-color: #3d3d3d;
                color: #e0e0e0;
            }
        """
        
    def _setup_ui(self):
        """Build the user interface"""
        # Create menu bar
        self._create_menu_bar()
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        
        # Top: Connection and controls
        top_group = QWidget()
        top_layout = QHBoxLayout(top_group)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(12)
        top_layout.addWidget(self._create_connection_panel(), stretch=1)
        top_layout.addWidget(self._create_control_panel(), stretch=5)
        main_layout.addWidget(top_group)
        
        # Middle: Visualizers + fixed main controls row
        top_pane = QWidget()
        top_pane_layout = QVBoxLayout(top_pane)
        top_pane_layout.setContentsMargins(0, 0, 0, 0)
        top_pane_layout.setSpacing(8)

        viz_widget = QWidget()
        viz_layout = QHBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.addWidget(self._create_spectrum_panel(), stretch=3)
        viz_layout.addWidget(self._create_position_panel(), stretch=1)
        viz_widget.setMinimumHeight(220)
        top_pane_layout.addWidget(viz_widget, stretch=1)

        # Main controls row (auto-height from content)
        main_controls_widget = QWidget()
        main_controls_layout = QHBoxLayout(main_controls_widget)
        main_controls_layout.setContentsMargins(0, 0, 0, 0)
        main_controls_layout.addWidget(self._create_main_controls_panel())
        top_pane_layout.addWidget(main_controls_widget, stretch=0)
        main_layout.addWidget(top_pane, stretch=1)
    
    def _create_menu_bar(self):
        """Create menu bar with top-level menus for app controls, options, and help."""
        menubar = self.menuBar()
        assert menubar is not None
        
        # Menu (main menu with preset load and About)
        main_menu = menubar.addMenu("Menu")
        assert main_menu is not None
        
        # Connection submenu
        connection_menu = main_menu.addMenu("Connection")
        assert connection_menu is not None

        connection_settings_action = connection_menu.addAction("Settings...")
        assert connection_settings_action is not None
        connection_settings_action.triggered.connect(self._on_options_connection)

        self.connection_toggle_action = connection_menu.addAction("Connect")
        assert self.connection_toggle_action is not None
        self.connection_toggle_action.triggered.connect(self._on_connect)

        self.connection_test_action = connection_menu.addAction("Test")
        assert self.connection_test_action is not None
        self.connection_test_action.triggered.connect(self._on_test)
        self.connection_test_action.setEnabled(False)

        # Audio Device option
        audio_device_action = main_menu.addAction("Audio Device...")
        assert audio_device_action is not None
        audio_device_action.triggered.connect(self._on_options_audio_device)

        # Device Limits option
        device_limits_action = main_menu.addAction("Device Limits...")
        assert device_limits_action is not None
        device_limits_action.triggered.connect(self._on_device_limits)
        
        # Nerds menu (advanced perf + diagnostics) - inserted near Help later
        nerds_menu = QMenu("Nerds", menubar)
        assert nerds_menu is not None
        
        # FFT Size submenu
        fft_menu = nerds_menu.addMenu("FFT Size (requires restart)")
        assert fft_menu is not None
        fft_sizes = [512, 1024, 2048, 4096, 8192]
        fft_labels = [
            "512 (fast, ~86Hz/bin)",
            "1024 (balanced, ~43Hz/bin)",
            "2048 (good bass, ~21Hz/bin)",
            "4096 (great bass, ~11Hz/bin)",
            "8192 (best bass, ~5Hz/bin)"
        ]
        current_fft = getattr(self.config.audio, 'fft_size', 1024)
        for i, (size, label) in enumerate(zip(fft_sizes, fft_labels)):
            action = fft_menu.addAction(label)
            assert action is not None
            action.triggered.connect(lambda checked, idx=i: self._on_menu_fft_change(idx))
            if size == current_fft:
                action.setCheckable(True)
                action.setChecked(True)
        
        # Spectrum Updates submenu
        spec_menu = nerds_menu.addMenu("Spectrum Updates")
        assert spec_menu is not None
        spec_options = ["Every frame (smooth)", "Every 2 frames (fast)", "Every 4 frames (faster)"]
        spec_values = [1, 2, 4]
        current_skip = getattr(self.config.audio, 'spectrum_skip_frames', 2)
        for i, (label, value) in enumerate(zip(spec_options, spec_values)):
            action = spec_menu.addAction(label)
            assert action is not None
            action.triggered.connect(lambda checked, idx=i: self._on_menu_spectrum_change(idx))
            if value == current_skip:
                action.setCheckable(True)
                action.setChecked(True)

        fft_diag_action = nerds_menu.addAction("FFT Bin Diagnostics...")
        assert fft_diag_action is not None
        fft_diag_action.triggered.connect(self._on_fft_bin_diagnostics)
        
        # Options menu (separate top-level menu)
        options_menu = menubar.addMenu("Options")
        assert options_menu is not None
        
        beat_detection_action = options_menu.addAction("Beat Detection...")
        assert beat_detection_action is not None
        beat_detection_action.triggered.connect(self._on_options_beat_detection)

        # Spectrum visualizer type submenu
        viz_menu = options_menu.addMenu("Spectrum Type")
        assert viz_menu is not None
        viz_names = ["Waveform", "Freq dB", "Digital FFT"]
        default_viz_index = 0  # Waveform
        self.visualizer_type_combo = QComboBox()  # Hidden combo for state tracking
        self.visualizer_type_combo.addItems(viz_names)
        self.visualizer_type_combo.setCurrentIndex(default_viz_index)
        self._viz_type_actions = []
        for i, name in enumerate(viz_names):
            action = viz_menu.addAction(name)
            assert action is not None
            action.setCheckable(True)
            action.setChecked(i == default_viz_index)
            action.triggered.connect(lambda checked, idx=i: self._on_viz_menu_change(idx))
            self._viz_type_actions.append(action)

        effects_menu = options_menu.addMenu("Effects")
        assert effects_menu is not None

        self.jitter_effect_action = effects_menu.addAction("Jitter")
        assert self.jitter_effect_action is not None
        self.jitter_effect_action.setCheckable(True)
        self.jitter_effect_action.setChecked(bool(getattr(self.config.jitter, 'enabled', True)))
        self.jitter_effect_action.triggered.connect(self._on_effects_jitter_toggle)

        self.metronome_lock_required_action = options_menu.addAction("Metronome Lock Req'd")
        assert self.metronome_lock_required_action is not None
        self.metronome_lock_required_action.setCheckable(True)
        self.metronome_lock_required_action.setChecked(bool(getattr(self.config.beat, 'tempo_lock_required', False)))
        self.metronome_lock_required_action.toggled.connect(self._on_tempo_lock_required_toggle)

        developer_controls_action = options_menu.addAction("Developer Controls")
        assert developer_controls_action is not None
        developer_controls_action.triggered.connect(self._open_developer_controls_window)

        nerds_menu.addSeparator()

        # Log level submenu
        log_menu = nerds_menu.addMenu("Log Level")
        assert log_menu is not None
        self._log_level_actions = []
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            action = log_menu.addAction(level)
            assert action is not None
            action.setCheckable(True)
            action.triggered.connect(lambda checked, lvl=level: self._on_log_level_change(lvl))
            self._log_level_actions.append(action)
        self._sync_log_level_menu(getattr(self.config, 'log_level', 'INFO'))
        
        # Nerds menu should be second-to-last (right before Help)
        menubar.addMenu(nerds_menu)
        
        # Help menu (separate top-level menu)
        help_menu = menubar.addMenu("Help")
        assert help_menu is not None
        
        help_action = help_menu.addAction("Troubleshooting...")
        assert help_action is not None
        help_action.triggered.connect(self._on_help)

        about_action = help_menu.addAction("About")
        assert about_action is not None
        about_action.triggered.connect(self._on_about)
    
    def _on_options_audio_device(self):
        """Show Audio Device selection dialog"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Audio Device")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("Select Audio Device:"))
        
        # Create a combo box mirroring the main device_combo
        device_combo = QComboBox()
        device_combo.setMinimumWidth(350)
        
        # Copy items from main combo
        for i in range(self.device_combo.count()):
            device_combo.addItem(self.device_combo.itemText(i))
        device_combo.setCurrentIndex(self.device_combo.currentIndex())
        layout.addWidget(device_combo)
        
        # Quick preset buttons
        preset_row = QHBoxLayout()
        mic_btn = QPushButton("🎤 Mic (Reactive)")
        mic_btn.clicked.connect(lambda: self._dialog_set_device_mic(device_combo))
        preset_row.addWidget(mic_btn)
        
        loopback_btn = QPushButton("🔊 System Audio")
        loopback_btn.clicked.connect(lambda: self._dialog_set_device_loopback(device_combo))
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
            self.device_combo.setCurrentIndex(device_combo.currentIndex())
    
    def _dialog_set_device_mic(self, combo: QComboBox):
        """Set mic device in dialog combo"""
        for i in range(combo.count()):
            text = combo.itemText(i).lower()
            if 'microphone' in text or 'mic' in text or 'input' in text:
                if 'loopback' not in text and 'stereo mix' not in text:
                    combo.setCurrentIndex(i)
                    return
    
    def _dialog_set_device_loopback(self, combo: QComboBox):
        """Set loopback/system audio device in dialog combo"""
        for i in range(combo.count()):
            text = combo.itemText(i).lower()
            if 'loopback' in text or 'stereo mix' in text or 'wasapi' in text:
                combo.setCurrentIndex(i)
                return
        # Fallback to speakers
        for i in range(combo.count()):
            text = combo.itemText(i).lower()
            if 'speakers' in text or 'headphone' in text:
                combo.setCurrentIndex(i)
                return
    
    def _on_options_connection(self):
        """Show Connection settings dialog"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QSpinBox, QPushButton, QHBoxLayout, QGridLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle("TCP Connection")
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)
        
        # Host/Port grid
        grid = QGridLayout()
        grid.addWidget(QLabel("Host:"), 0, 0)
        host_edit = QLineEdit(self.host_edit.text())
        grid.addWidget(host_edit, 0, 1)
        
        grid.addWidget(QLabel("Port:"), 1, 0)
        port_spin = QSpinBox()
        port_spin.setRange(1, 65535)
        port_spin.setValue(self.port_spin.value())
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
            self.host_edit.setText(host_edit.text())
            self.port_spin.setValue(port_spin.value())
            # Reconnect if already connected
            if hasattr(self, 'network_engine') and self.network_engine:
                self._on_connect()

    def _on_options_tempo_tracking(self):
        self._open_developer_controls_window(tab_index=0)

    def _open_developer_controls_window(self, tab_index: int = 0, scroll_to_flux: bool = False) -> None:
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTabWidget

        dialog = getattr(self, '_developer_controls_dialog', None)
        if dialog is not None:
            try:
                dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()
                tab_widget = getattr(self, '_developer_controls_tab_widget', None)
                if tab_widget is not None:
                    tab_widget.setCurrentIndex(max(0, min(int(tab_index), tab_widget.count() - 1)))
                if scroll_to_flux and int(tab_index) == 1:
                    self._scroll_advanced_controls_to_flux()
                if not bool(getattr(self, '_developer_controls_unlocked', False)):
                    dialog.setEnabled(False)
                    self._show_developer_controls_unlock_popup()
                return
            except RuntimeError:
                self._developer_controls_dialog = None
                self._developer_controls_tab_widget = None

        dialog = QDialog(self)
        dialog.setWindowTitle("Developer Controls")
        dialog.setMinimumWidth(620)
        dialog.setMinimumHeight(560)
        dialog.setModal(False)
        dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        def _on_developer_dialog_destroyed() -> None:
            self._developer_controls_dialog = None
            self._developer_controls_tab_widget = None
            self._developer_controls_unlocked = False
            self._trigger_settings_tab_content = None
            self._auto_fill_tab_content = None
            self._motion_readiness_tab_content = None
            unlock_dialog_ref = getattr(self, '_developer_unlock_dialog', None)
            if unlock_dialog_ref is not None:
                try:
                    unlock_dialog_ref.close()
                except RuntimeError:
                    pass
                self._developer_unlock_dialog = None

        dialog.destroyed.connect(_on_developer_dialog_destroyed)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)

        tab_widget = QTabWidget()
        self._developer_controls_tab_widget = tab_widget

        tempo_content = getattr(self, '_tempo_tracking_popout_content', None)
        if tempo_content is None:
            tempo_content = self._create_tempo_tracking_tab(include_advanced_controls=True, advanced_locked=True)
            self._tempo_tracking_popout_content = tempo_content
            self._apply_config_to_ui()

        trigger_content = getattr(self, '_trigger_settings_tab_content', None)
        if trigger_content is None:
            trigger_content = self._on_advanced_controls(as_tab=True)
            self._trigger_settings_tab_content = trigger_content

        auto_fill_content = getattr(self, '_auto_fill_tab_content', None)
        if auto_fill_content is None:
            auto_fill_content = self._on_options_auto_fill_adaptation(as_tab=True)
            self._auto_fill_tab_content = auto_fill_content

        motion_content = getattr(self, '_motion_readiness_tab_content', None)
        if motion_content is None:
            motion_content = self._on_options_motion_readiness(as_tab=True)
            self._motion_readiness_tab_content = motion_content

        tab_widget.addTab(tempo_content, "Tempo Tracking")
        tab_widget.addTab(trigger_content, "Trigger Settings")
        tab_widget.addTab(auto_fill_content, "Auto Fill %")
        tab_widget.addTab(motion_content, "Motion Readiness")
        tab_widget.setCurrentIndex(max(0, min(int(tab_index), tab_widget.count() - 1)))

        layout.addWidget(tab_widget)

        self._developer_controls_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

        if scroll_to_flux and int(tab_index) == 1:
            self._scroll_advanced_controls_to_flux()

        if not bool(getattr(self, '_developer_controls_unlocked', False)):
            dialog.setEnabled(False)
            self._show_developer_controls_unlock_popup()

    def _show_developer_controls_unlock_popup(self) -> None:
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

        existing = getattr(self, '_developer_unlock_dialog', None)
        if existing is not None:
            try:
                existing.show()
                existing.raise_()
                existing.activateWindow()
                return
            except RuntimeError:
                self._developer_unlock_dialog = None

        unlock_dialog = QDialog(self)
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
            developer_dialog = getattr(self, '_developer_controls_dialog', None)
            if developer_dialog is not None:
                try:
                    developer_dialog.close()
                except RuntimeError:
                    pass
            unlock_dialog.close()

        def _unlock() -> None:
            self._developer_controls_unlocked = True
            developer_dialog = getattr(self, '_developer_controls_dialog', None)
            if developer_dialog is not None:
                try:
                    developer_dialog.setEnabled(True)
                    developer_dialog.show()
                    developer_dialog.raise_()
                    developer_dialog.activateWindow()
                except RuntimeError:
                    pass
            unlock_dialog.close()

        cancel_btn.clicked.connect(_cancel)
        unlock_btn.clicked.connect(_unlock)
        unlock_dialog.rejected.connect(_cancel)
        unlock_dialog.destroyed.connect(lambda *_: setattr(self, '_developer_unlock_dialog', None))

        self._developer_unlock_dialog = unlock_dialog
        unlock_dialog.show()
        unlock_dialog.raise_()
        unlock_dialog.activateWindow()

    def _on_options_tempo_tracking_legacy(self):
        """Show Tempo Tracking controls popout (tempo tab + advanced tempo controls)."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        dialog = getattr(self, '_tempo_tracking_dialog', None)
        if dialog is not None:
            try:
                dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()
                return
            except RuntimeError:
                self._tempo_tracking_dialog = None
                dialog = None

        dialog = QDialog(self)
        dialog.setWindowTitle("Tempo Tracking")
        dialog.setMinimumWidth(520)
        dialog.setMinimumHeight(560)
        dialog.setModal(False)
        dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)

        content = getattr(self, '_tempo_tracking_popout_content', None)
        if content is None:
            content = self._create_tempo_tracking_tab(include_advanced_controls=True, advanced_locked=True)
            self._tempo_tracking_popout_content = content
            self._apply_config_to_ui()
        layout.addWidget(content)

        self._tempo_tracking_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_options_beat_detection(self):
        """Show Beat Detection controls popout."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        dialog = getattr(self, '_beat_detection_dialog', None)
        if dialog is not None:
            try:
                dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()
                return
            except RuntimeError:
                self._beat_detection_dialog = None
                dialog = None

        dialog = QDialog(self)
        dialog.setWindowTitle("Beat Detection")
        dialog.setMinimumWidth(520)
        dialog.setMinimumHeight(640)
        dialog.setModal(False)
        dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        def _on_beat_detection_dialog_destroyed() -> None:
            self._beat_detection_dialog = None
            self._beat_detection_popout_content = None

        dialog.destroyed.connect(_on_beat_detection_dialog_destroyed)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)

        content = getattr(self, '_beat_detection_popout_content', None)
        if content is not None:
            try:
                _ = content.parent()
            except RuntimeError:
                content = None
                self._beat_detection_popout_content = None
        if content is None:
            content = self._create_beat_detection_tab()
            self._beat_detection_popout_content = content
            self._apply_config_to_ui()
        layout.addWidget(content)

        self._beat_detection_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_pulse_settings_popup(self):
        """Show Pulse settings popout."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout

        def _focus_pulse_dialog(target_dialog) -> None:
            try:
                target_dialog.activateWindow()
                target_dialog.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
                if hasattr(self, 'pulse_enabled_checkbox'):
                    self.pulse_enabled_checkbox.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            except RuntimeError:
                pass

        dialog = getattr(self, '_pulse_settings_dialog', None)
        if dialog is not None:
            try:
                dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
                dialog.show()
                dialog.raise_()
                _focus_pulse_dialog(dialog)
                QTimer.singleShot(0, lambda d=dialog: _focus_pulse_dialog(d))
                return
            except RuntimeError:
                self._pulse_settings_dialog = None
                dialog = None

        dialog = QDialog(self)
        dialog.setWindowTitle("Pulse Settings")
        dialog.setMinimumWidth(520)
        dialog.setMinimumHeight(620)
        dialog.setModal(False)
        dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        def _on_pulse_settings_dialog_destroyed() -> None:
            self._pulse_settings_dialog = None
            self._pulse_settings_popout_content = None

        dialog.destroyed.connect(_on_pulse_settings_dialog_destroyed)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)

        content = getattr(self, '_pulse_settings_popout_content', None)
        if content is not None:
            try:
                _ = content.parent()
            except RuntimeError:
                content = None
                self._pulse_settings_popout_content = None
        if content is None:
            content = self._create_tcode_freq_tab()
            self._pulse_settings_popout_content = content
            self._apply_config_to_ui()
        layout.addWidget(content)

        self._pulse_settings_dialog = dialog
        dialog.show()
        dialog.raise_()
        _focus_pulse_dialog(dialog)
        QTimer.singleShot(0, lambda d=dialog: _focus_pulse_dialog(d))

    def _on_options_geometry_rest_state(self):
        """Show Geometry Rest State controls."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel

        dialog = getattr(self, '_geometry_rest_dialog', None)
        if dialog is not None:
            try:
                dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
                dialog.show()
                dialog.raise_()
                dialog.activateWindow()
                return
            except RuntimeError:
                self._geometry_rest_dialog = None
                dialog = None

        dialog = QDialog(self)
        dialog.setWindowTitle("Geometry Rest State")
        dialog.setMinimumWidth(460)
        dialog.setMinimumHeight(220)
        dialog.setModal(False)
        dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        def _on_geometry_rest_dialog_destroyed() -> None:
            self._geometry_rest_dialog = None

        dialog.destroyed.connect(_on_geometry_rest_dialog_destroyed)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        info = QLabel(
            "Tune rest parking geometry behavior.\n"
            "Higher y-offset = deeper rest point."
        )
        info.setStyleSheet("color: #bbb; font-size: 11px;")
        layout.addWidget(info)

        y_offset_slider = SliderWithLabel(
            "Rest Y Offset",
            0.00,
            1.00,
            float(getattr(self.config.stroke, 'geometry_y_offset', 0.50) or 0.50),
            2,
        )

        def _apply_geometry_rest_from_sliders() -> None:
            self.config.stroke.geometry_y_offset = float(y_offset_slider.value())
            self._apply_geometry_rest_to_mapper()

        y_offset_slider.valueChanged.connect(lambda _v: _apply_geometry_rest_from_sliders())

        layout.addWidget(y_offset_slider)

        dialog.finished.connect(lambda _r: save_config(self.config))

        self._geometry_rest_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _apply_geometry_rest_to_mapper(self) -> None:
        if not self.stroke_mapper:
            return
        y_offset = float(getattr(self.config.stroke, 'geometry_y_offset', 0.50) or 0.50)
        if hasattr(self.stroke_mapper, 'configure_geometry_rest_state'):
            self.stroke_mapper.configure_geometry_rest_state(y_offset=y_offset)

    def _on_options_auto_fill_adaptation(self, as_tab: bool = False):
        """Show or build adaptive amp-fill gate tuning controls."""
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox

        if not as_tab:
            self._open_developer_controls_window(tab_index=2)
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
        auto_enabled_cb.setChecked(bool(getattr(self.config.stroke, 'overall_amp_fill_auto_enabled', True)))
        auto_enabled_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'overall_amp_fill_auto_enabled', state == 2)
        )
        layout.addWidget(auto_enabled_cb)

        target_rate_slider = SliderWithLabel(
            "Target fill pass rate",
            0.10,
            0.95,
            float(getattr(self.config.stroke, 'overall_amp_fill_auto_target_pass_rate', 0.58) or 0.58),
            2,
        )
        target_rate_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'overall_amp_fill_auto_target_pass_rate', float(v))
        )
        layout.addWidget(target_rate_slider)

        ema_alpha_slider = SliderWithLabel(
            "Pass-rate EMA alpha",
            0.01,
            0.60,
            float(getattr(self.config.stroke, 'overall_amp_fill_auto_ema_alpha', 0.12) or 0.12),
            3,
        )
        ema_alpha_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'overall_amp_fill_auto_ema_alpha', float(v))
        )
        layout.addWidget(ema_alpha_slider)

        deadband_slider = SliderWithLabel(
            "Deadband",
            0.00,
            0.40,
            float(getattr(self.config.stroke, 'overall_amp_fill_auto_deadband', 0.06) or 0.06),
            3,
        )
        deadband_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'overall_amp_fill_auto_deadband', float(v))
        )
        layout.addWidget(deadband_slider)

        step_slider = SliderWithLabel(
            "Step size",
            0.001,
            0.15,
            float(getattr(self.config.stroke, 'overall_amp_fill_auto_step', 0.02) or 0.02),
            3,
        )
        step_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'overall_amp_fill_auto_step', float(v))
        )
        layout.addWidget(step_slider)

        max_offset_slider = SliderWithLabel(
            "Max offset from base requirement",
            0.01,
            0.80,
            float(getattr(self.config.stroke, 'overall_amp_fill_auto_max_offset', 0.35) or 0.35),
            3,
        )
        max_offset_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'overall_amp_fill_auto_max_offset', float(v))
        )
        layout.addWidget(max_offset_slider)

        min_required_slider = SliderWithLabel(
            "Minimum required fill",
            0.00,
            0.95,
            float(getattr(self.config.stroke, 'overall_amp_fill_auto_min_required', 0.05) or 0.05),
            3,
        )
        max_required_slider = SliderWithLabel(
            "Maximum required fill",
            0.05,
            1.00,
            float(getattr(self.config.stroke, 'overall_amp_fill_auto_max_required', 0.98) or 0.98),
            3,
        )

        def _sync_required_bounds() -> None:
            min_val = float(min_required_slider.value())
            max_val = float(max_required_slider.value())
            if max_val < min_val:
                max_val = min_val
                max_required_slider.setValue(max_val)
            self.config.stroke.overall_amp_fill_auto_min_required = min_val
            self.config.stroke.overall_amp_fill_auto_max_required = max_val

        min_required_slider.valueChanged.connect(lambda _: _sync_required_bounds())
        max_required_slider.valueChanged.connect(lambda _: _sync_required_bounds())

        layout.addWidget(min_required_slider)
        layout.addWidget(max_required_slider)
        layout.addStretch()

        self._auto_fill_controls_widgets = {
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

    def _on_options_motion_readiness(self, as_tab: bool = False):
        """Show or build readiness gating controls."""
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QCheckBox, QGroupBox, QPushButton

        if not as_tab:
            self._open_developer_controls_window(tab_index=3)
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
            float(getattr(self.config.beat, 'teaching_metronome_relaxed_confidence', 0.14) or 0.14),
            2,
        )
        relaxed_conf_slider.valueChanged.connect(
            lambda v: setattr(self.config.beat, 'teaching_metronome_relaxed_confidence', float(v))
        )
        readiness_layout.addWidget(relaxed_conf_slider)

        grace_ms_slider = SliderWithLabel(
            "Stroke Ready Grace (ms)",
            0.0,
            3000.0,
            float(getattr(self.config.beat, 'teaching_stroke_ready_grace_ms', 450.0) or 450.0),
            0,
        )
        grace_ms_slider.valueChanged.connect(
            lambda v: setattr(self.config.beat, 'teaching_stroke_ready_grace_ms', float(v))
        )
        readiness_layout.addWidget(grace_ms_slider)

        finish_row = QHBoxLayout()
        finish_label = QLabel("Stroke Finish Beats")
        finish_label.setStyleSheet("color: #ddd;")
        finish_spin = QSpinBox()
        finish_spin.setRange(0, 64)
        finish_spin.setValue(int(getattr(self.config.beat, 'teaching_stroke_finish_beats', 4) or 4))
        finish_spin.valueChanged.connect(
            lambda v: setattr(self.config.beat, 'teaching_stroke_finish_beats', int(v))
        )
        finish_row.addWidget(finish_label)
        finish_row.addStretch()
        finish_row.addWidget(finish_spin)
        readiness_layout.addLayout(finish_row)

        relax_phase1_cb = QCheckBox("Relax Phase-1 gates during learning")
        relax_phase1_cb.setChecked(bool(getattr(self.config.beat, 'teaching_relax_phase1_gates', False)))
        relax_phase1_cb.setToolTip(
            "When enabled, learning mode can bypass mid-trigger and dual-band gate strictness."
        )
        relax_phase1_cb.stateChanged.connect(
            lambda state: setattr(self.config.beat, 'teaching_relax_phase1_gates', state == 2)
        )
        readiness_layout.addWidget(relax_phase1_cb)

        ignore_traffic_cb = QCheckBox("Use metronome-only readiness (legacy permissive)")
        ignore_traffic_cb.setChecked(bool(getattr(self.config.beat, 'teaching_ignore_traffic_lights', False)))
        ignore_traffic_cb.setToolTip(
            "When enabled, readiness uses metronome BPM + relaxed confidence only, "
            "ignoring stricter lock-style gating."
        )
        ignore_traffic_cb.stateChanged.connect(
            lambda state: setattr(self.config.beat, 'teaching_ignore_traffic_lights', state == 2)
        )
        readiness_layout.addWidget(ignore_traffic_cb)

        layout.addWidget(readiness_group)

        tuning_group = QGroupBox("Tuning")
        tuning_layout = QVBoxLayout(tuning_group)

        strength_slider = SliderWithLabel(
            "Advance", 0.0, 1.0,
            float(getattr(self.config.beat, 'teaching_learning_strength', 0.55) or 0.55), 2
        )
        tuning_layout.addWidget(strength_slider)

        holdback_slider = SliderWithLabel(
            "Restraint", 0.0, 1.0,
            float(getattr(self.config.beat, 'teaching_min_confidence', 0.12) or 0.12), 2
        )
        tuning_layout.addWidget(holdback_slider)

        no_motion_bias_slider = SliderWithLabel(
            "Quiet Bias", 0.25, 3.0,
            float(getattr(self.config.beat, 'teaching_no_motion_bias', 1.0) or 1.0), 2
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
            self.config.beat.teaching_learning_strength = float(strength_slider.value())
            self.config.beat.teaching_min_confidence = float(holdback_slider.value())
            self.config.beat.teaching_no_motion_bias = float(no_motion_bias_slider.value())
            self._apply_learning_config_to_mapper()
            save_config(self.config)

        tuning_apply_btn.clicked.connect(_apply_tuning_settings)
        tuning_layout.addWidget(tuning_apply_btn)

        layout.addWidget(tuning_group)

        layout.addStretch()

        self._motion_readiness_controls_widgets = {
            'relaxed_confidence': relaxed_conf_slider,
            'stroke_ready_grace_ms': grace_ms_slider,
            'stroke_finish_beats': finish_spin,
            'relax_phase1_gates': relax_phase1_cb,
            'tuning_strength': strength_slider,
            'tuning_holdback': holdback_slider,
            'tuning_quiet_bias': no_motion_bias_slider,
            'tuning_apply': tuning_apply_btn,
        }
        return content
    
    def _on_device_limits(self, first_run: bool = False):
        """Show Device Limits dialog for value-to-real-units conversion.
        Pulse Freq/Carrier Freq (Hz) are always shown. Pulse Width/Interval Random/Rise Time are optional.
        Called from Options menu or on first startup if not yet prompted."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QDoubleSpinBox,
                                      QPushButton, QHBoxLayout, QGridLayout, QGroupBox, QCheckBox)
        
        dialog = QDialog(self)
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
        
        dl = self.config.device_limits

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
            self.config.device_limits.p0_freq_min = p0_min.value()
            self.config.device_limits.p0_freq_max = p0_max.value()
            self.config.device_limits.c0_freq_min = c0_min.value()
            self.config.device_limits.c0_freq_max = c0_max.value()
            self.config.device_limits.p1_cycles_min = p1_min.value()
            self.config.device_limits.p1_cycles_max = p1_max.value()
            self.config.device_limits.p2_range_min = p2_min.value()
            self.config.device_limits.p2_range_max = p2_max.value()
            self.config.device_limits.p3_cycles_min = p3_min.value()
            self.config.device_limits.p3_cycles_max = p3_max.value()
            self.config.device_limits.p0_c0_sending_enabled = p0c0_cb.isChecked()
            self.config.device_limits.dont_show_on_startup = dont_show_cb.isChecked()
            self.config.device_limits.prompted = True
            self._sync_pulse_sent_spin_limits_from_device_limits()
            print(f"[Config] Device limits updated: P0={p0_min.value()}-{p0_max.value()}Hz, "
                  f"C0={c0_min.value()}-{c0_max.value()}Hz, "
                  f"P1={p1_min.value()}-{p1_max.value()}cyc, "
                  f"P2={p2_min.value()}-{p2_max.value()}, "
                  f"P3={p3_min.value()}-{p3_max.value()}cyc, "
                  f"P0/C0 sending={'ON' if p0c0_cb.isChecked() else 'OFF'}")
        else:
            # Mark as prompted even if skipped/cancelled so we don't ask again
            self.config.device_limits.prompted = True
            self.config.device_limits.dont_show_on_startup = dont_show_cb.isChecked()

    def _sync_pulse_sent_spin_limits_from_device_limits(self) -> None:
        """Clamp Pulse Settings sent spinboxes to current Device Limits ranges."""
        dl = self.config.device_limits

        def _effective_limits(raw_min: float, raw_max: float, default_min: float, default_max: float) -> tuple[float, float]:
            lo = float(raw_min)
            hi = float(raw_max)
            if hi <= lo:
                lo = float(default_min)
                hi = float(default_max)
            return lo, hi

        def _apply_pair(min_attr: str, max_attr: str, raw_min: float, raw_max: float, default_min: float, default_max: float) -> None:
            min_spin = getattr(self, min_attr, None)
            max_spin = getattr(self, max_attr, None)
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

    def _scroll_advanced_controls_to_flux(self):
        """Scroll open Advanced Controls dialog near the Flux Sensitivity group."""
        scroll = getattr(self, '_advanced_controls_scroll', None)
        flux_group = getattr(self, '_advanced_flux_group', None)
        if scroll is None or flux_group is None:
            return

        def _apply_scroll():
            bar = scroll.verticalScrollBar()
            if bar is None:
                return
            target = max(0, int(flux_group.y()) - 12)
            bar.setValue(min(target, bar.maximum()))

        QTimer.singleShot(0, _apply_scroll)

    def _on_advanced_controls(self, scroll_to_flux: bool = False, as_tab: bool = False):
        """Show Advanced Controls dialog with experimental/expert settings"""
        from PyQt6.QtWidgets import QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QScrollArea, QGroupBox, QSpinBox

        if not as_tab:
            self._open_developer_controls_window(tab_index=1, scroll_to_flux=scroll_to_flux)
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
            "• RMS-labeled controls use the same raw_rms units shown in console [Audio] logs.\n"
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
        self._advanced_controls_scroll = scroll
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)

        # ===== Silence Gate Controls (Top Priority) =====
        silence_group = QGroupBox("Silence Gate (RMS)")
        silence_layout = QVBoxLayout(silence_group)

        silence_info = QLabel(
            "These thresholds control the silence deadzone hysteresis using RMS amplitude units.\n"
            "Open = enter silence (park), Close = exit silence (resume motion)."
        )
        silence_info.setStyleSheet("color: #aaa; font-size: 11px;")
        silence_layout.addWidget(silence_info)

        silence_open_slider = SliderWithLabel(
            "Silence Threshold Open",
            0.001,
            0.250,
            float(getattr(self.config.stroke, 'silence_threshold', 0.010825) or 0.010825),
            3,
        )
        silence_open_slider.setToolTip("Audio amplitude below this RMS level enters silence mode (motion stops, dot parks)")

        silence_close_slider = SliderWithLabel(
            "Silence Threshold Close",
            0.001,
            0.300,
            float(getattr(self.config.stroke, 'silence_close_threshold', 0.0433) or 0.0433),
            3,
        )
        silence_close_slider.setToolTip("Audio amplitude must exceed this RMS level to exit silence mode (motion resumes)")

        def _set_silence_open(v: float) -> None:
            open_v = float(v)
            setattr(self.config.stroke, 'silence_threshold', open_v)
            close_v = float(getattr(self.config.stroke, 'silence_close_threshold', 0.0433) or 0.0433)
            if close_v <= open_v:
                close_v = open_v + 0.001
                setattr(self.config.stroke, 'silence_close_threshold', close_v)
                silence_close_slider.blockSignals(True)
                silence_close_slider.setValue(close_v)
                silence_close_slider.blockSignals(False)

        def _set_silence_close(v: float) -> None:
            close_v = float(v)
            open_v = float(getattr(self.config.stroke, 'silence_threshold', 0.010825) or 0.010825)
            if close_v <= open_v:
                close_v = open_v + 0.001
                silence_close_slider.blockSignals(True)
                silence_close_slider.setValue(close_v)
                silence_close_slider.blockSignals(False)
            setattr(self.config.stroke, 'silence_close_threshold', close_v)

        silence_open_slider.valueChanged.connect(_set_silence_open)
        silence_close_slider.valueChanged.connect(_set_silence_close)
        silence_layout.addWidget(silence_open_slider)
        silence_layout.addWidget(silence_close_slider)
        scroll_layout.addWidget(silence_group)
        
        # ===== Syncopation Controls =====
        syncope_group = QGroupBox("Syncopation / Double-Stroke")
        syncope_layout = QVBoxLayout(syncope_group)

        # On/Off checkbox
        syncope_enabled_cb = QCheckBox("Allow Off-Beat Strokes")
        syncope_enabled_cb.setChecked(bool(getattr(self.config.beat, 'syncopation_enabled', True)))
        syncope_enabled_cb.setToolTip("When enabled, system detects syncopation and fires rapid 1-beat strokes on off-beats")
        syncope_enabled_cb.stateChanged.connect(
            lambda state: setattr(self.config.beat, 'syncopation_enabled', state == 2)
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
        current_band = str(getattr(self.config.beat, 'syncopation_band', 'any'))
        if current_band in band_options:
            band_combo.setCurrentIndex(band_options.index(current_band))
        band_combo.currentTextChanged.connect(
            lambda text: setattr(self.config.beat, 'syncopation_band', text)
        )
        band_row.addWidget(band_combo)
        syncope_layout.addLayout(band_row)

        # Syncopation window slider
        syncope_window_slider = SliderWithLabel(
            "Off-Beat Timing Window",
            0.05,
            0.30,
            float(getattr(self.config.beat, 'syncopation_window', 0.16)),
            2,
        )
        syncope_window_slider.setToolTip("Search window (as fraction of beat period) around expected off-beat position. Wider window = more permissive but slower")
        syncope_window_slider.valueChanged.connect(
            lambda v: setattr(self.config.beat, 'syncopation_window', v)
        )
        syncope_layout.addWidget(syncope_window_slider)

        # BPM limit slider
        syncope_bpm_slider = SliderWithLabel(
            "Max BPM for Off-Beats",
            80.0,
            200.0,
            float(getattr(self.config.beat, 'syncopation_bpm_limit', 130.0)),
            0,
        )
        syncope_bpm_slider.setToolTip("Disable off-beat detection above this BPM (to prevent false positives in very fast music)")
        syncope_bpm_slider.valueChanged.connect(
            lambda v: setattr(self.config.beat, 'syncopation_bpm_limit', v)
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
            if hasattr(self, 'freqdb_canvas') and hasattr(self.freqdb_canvas, 'show_flux_ghost'):
                self.freqdb_canvas.show_flux_ghost(
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
            target = float(getattr(self.config.stroke, 'overall_amp_fill_target', 0.5) or 0.5)
            tol = float(abs(getattr(self.config.stroke, 'overall_amp_fill_tolerance', 0.5) or 0.5))
            min_amp = max(0.0, target - tol)
            _show_freqdb_ghost_ref('overall_amp_target', target, 'Amp target', '#66CCFF', dashed=False, band='full')
            _show_freqdb_ghost_ref('overall_amp_min', min_amp, 'Amp min', '#FFAA66', dashed=True, band='full')

        def _update_fill_requirement_refs() -> None:
            self._preview_fill_requirement_ghosts()

        def _show_fft_bin_fill_ref(key: str, ratio: float, label: str, color: str = '#66E0FF', dashed: bool = True) -> None:
            canvas = getattr(self, 'fft_bin_canvas', None)
            if canvas is not None and hasattr(canvas, 'show_fill_ratio_ghost'):
                canvas.show_fill_ratio_ghost(key, float(ratio), label, color=color, duration_s=5.0, dashed=dashed)

        def _show_fft_bin_range_ref(key: str, low_bin: int, high_bin: int, label: str, color: str = '#FFFFFF', dashed: bool = False) -> None:
            canvas = getattr(self, 'fft_bin_canvas', None)
            if canvas is not None and hasattr(canvas, 'show_bin_range_ghost'):
                if hasattr(canvas, '_bar_count') and int(getattr(canvas, '_bar_count', 0) or 0) <= 0 and hasattr(canvas, '_ensure_bars'):
                    fft_size = int(getattr(self.config.audio, 'fft_size', 1024) or 1024)
                    bin_count = max(2, (fft_size // 2) + 1)
                    canvas._ensure_bars(bin_count)
                canvas.show_bin_range_ghost(key, int(low_bin), int(high_bin), label, color=color, duration_s=5.0, dashed=dashed)


        bass_gate_cb = QCheckBox("Require Bass Energy for Motion")
        bass_gate_cb.setChecked(getattr(self.config.beat, 'strict_bass_motion_gate_enabled', False))
        bass_gate_cb.setToolTip("When enabled: motion only fires if sub-bass or low-mid frequency band has strong energy")
        bass_gate_cb.stateChanged.connect(
            lambda state: setattr(self.config.beat, 'strict_bass_motion_gate_enabled', state == 2)
        )
        gate_layout.addWidget(bass_gate_cb)

        motion_cutoff_row = QHBoxLayout()
        motion_cutoff_label = QLabel("Motion Bass Cutoff Frequency:")
        motion_cutoff_label.setToolTip("If Bass Energy check is on: only allow strokes from sounds whose lowest frequency is below this cutoff (Hz)")
        motion_cutoff_row.addWidget(motion_cutoff_label)

        self.motion_freq_cutoff_spin = QSpinBox()
        self.motion_freq_cutoff_spin.setRange(0, 2000)
        self.motion_freq_cutoff_spin.setSingleStep(20)
        self.motion_freq_cutoff_spin.setValue(int(getattr(self.config.beat, 'motion_freq_cutoff', 500)))
        self.motion_freq_cutoff_spin.setSuffix(" Hz")
        self.motion_freq_cutoff_spin.setFixedWidth(90)
        self.motion_freq_cutoff_spin.setToolTip("0 disables cutoff filtering (while Bass Gating is enabled)")
        self.motion_freq_cutoff_spin.valueChanged.connect(
            lambda v: (
                self._on_motion_freq_cutoff_change(v),
                _show_freqdb_ghost_ref('motion_freq_cutoff_hz', float(v), 'Motion cutoff', '#FFD166', dashed=True, mode='hz_line', range_box=False)
            )
        )
        motion_cutoff_row.addWidget(self.motion_freq_cutoff_spin)
        motion_cutoff_row.addStretch()
        gate_layout.addLayout(motion_cutoff_row)

        amp_fill_gate_cb = QCheckBox("Enable Spectral Fullness Gate")
        amp_fill_gate_cb.setChecked(bool(getattr(self.config.stroke, 'overall_amp_fill_gate_enabled', True)))
        amp_fill_gate_cb.setToolTip("Require both overall amplitude AND spectrum 'fullness' before strokes fire (prevents sparse/thin sections from triggering)")
        amp_fill_gate_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'overall_amp_fill_gate_enabled', state == 2)
        )
        gate_layout.addWidget(amp_fill_gate_cb)

        amp_fill_target_slider = SliderWithLabel(
            "Required Spectral Fullness",
            0.0,
            1.0,
            float(getattr(self.config.stroke, 'overall_amp_fill_target', 0.5) or 0.5),
            2,
        )
        amp_fill_target_slider.valueChanged.connect(
            lambda v: (setattr(self.config.stroke, 'overall_amp_fill_target', float(v)), _update_overall_amp_fill_refs())
        )
        amp_fill_target_slider.setToolTip("Normalized target amplitude (0-1). Gates won't fire below this intensity range")
        gate_layout.addWidget(amp_fill_target_slider)

        amp_fill_tol_slider = SliderWithLabel(
            "Amplitude Range Width",
            0.0,
            1.0,
            float(getattr(self.config.stroke, 'overall_amp_fill_tolerance', 0.5) or 0.5),
            2,
        )
        amp_fill_tol_slider.valueChanged.connect(
            lambda v: (setattr(self.config.stroke, 'overall_amp_fill_tolerance', float(v)), _update_overall_amp_fill_refs())
        )
        amp_fill_tol_slider.setToolTip("Normalized tolerance band around the target (±). Strokes fire when amplitude stays in this zone")
        gate_layout.addWidget(amp_fill_tol_slider)

        downbeat_fill_slider = SliderWithLabel(
            "Downbeat Spectral Fullness Requirement",
            0.0,
            1.0,
            float(getattr(self.config.stroke, 'downbeat_overall_amp_fill_required', 0.08) or 0.08),
            2,
        )
        downbeat_fill_slider.setToolTip("How 'full' the spectrum must be (0-1) for downbeat to fire. Higher = requires more filled/dense spectrum")
        downbeat_fill_slider.valueChanged.connect(
            lambda v: (setattr(self.config.stroke, 'downbeat_overall_amp_fill_required', float(v)), _update_fill_requirement_refs())
        )
        gate_layout.addWidget(downbeat_fill_slider)

        beat_fill_slider = SliderWithLabel(
            "Beat Spectral Fullness Requirement",
            0.0,
            1.0,
            float(getattr(self.config.stroke, 'beat_overall_amp_fill_required', 0.10) or 0.10),
            2,
        )
        beat_fill_slider.setToolTip("How 'full' the spectrum must be (0-1) for beat strokes to fire. Higher = requires fuller, richer spectrum")
        beat_fill_slider.valueChanged.connect(
            lambda v: (setattr(self.config.stroke, 'beat_overall_amp_fill_required', float(v)), _update_fill_requirement_refs())
        )
        gate_layout.addWidget(beat_fill_slider)

        sync_fill_slider = SliderWithLabel(
            "Off-Beat Spectral Fullness Requirement",
            0.0,
            1.0,
            float(getattr(self.config.stroke, 'syncopation_overall_amp_fill_required', 0.12) or 0.12),
            2,
        )
        sync_fill_slider.setToolTip("How 'full' the spectrum must be (0-1) for off-beat strokes to fire. Usually highest since off-beats are stricter")
        sync_fill_slider.valueChanged.connect(
            lambda v: (setattr(self.config.stroke, 'syncopation_overall_amp_fill_required', float(v)), _update_fill_requirement_refs())
        )
        gate_layout.addWidget(sync_fill_slider)

        fill_bin_info = QLabel("Fill gate FFT-bin windows (tight range control per phase)")
        fill_bin_info.setStyleSheet("color: #999; font-size: 10px;")
        gate_layout.addWidget(fill_bin_info)

        fft_size = int(getattr(self.config.audio, 'fft_size', 1024) or 1024)
        max_bin = max(1, fft_size // 2)

        def _bin_to_hz(bin_value: int) -> float:
            sample_rate = float(getattr(self.config.audio, 'sample_rate', 44100) or 44100)
            return float(np.clip(bin_value, 0, max_bin) * (sample_rate / max(1, fft_size)))

        def _add_fill_bin_range_row(title: str, low_attr: str, high_attr: str, ghost_key: str) -> None:
            row = QHBoxLayout()
            row.addWidget(QLabel(title))

            low_spin = QSpinBox()
            low_spin.setRange(0, max_bin)
            low_spin.setSingleStep(1)
            low_spin.setValue(int(np.clip(int(getattr(self.config.stroke, low_attr, 0) or 0), 0, max_bin)))
            low_spin.setPrefix("low ")
            row.addWidget(low_spin)

            high_spin = QSpinBox()
            high_spin.setRange(0, max_bin)
            high_spin.setSingleStep(1)
            high_spin.setValue(int(np.clip(int(getattr(self.config.stroke, high_attr, max_bin) or max_bin), 0, max_bin)))
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
                setattr(self.config.stroke, low_attr, int(low_val))
                setattr(self.config.stroke, high_attr, int(high_val))
                _emit_ghost()

            def _on_high_change(v: int) -> None:
                high_val = int(v)
                low_val = int(low_spin.value())
                if high_val < low_val:
                    low_spin.setValue(high_val)
                    low_val = high_val
                setattr(self.config.stroke, low_attr, int(low_val))
                setattr(self.config.stroke, high_attr, int(high_val))
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
                int(getattr(self.config.stroke, sustain_attr, 3) or 3),
                0,
            )
            sustain_slider.setToolTip(
                "Consecutive frames (~20ms each) that spectrum must stay full before stroke fires.\n"
                "0-1 = instant, 3 = ~60ms, 5 = ~100ms, 10 = ~200ms.\n"
                "Prevents single-frame spikes from triggering."
            )
            sustain_slider.valueChanged.connect(
                lambda v: setattr(self.config.stroke, sustain_attr, int(v))
            )
            gate_layout.addWidget(sustain_slider)

        _add_fill_bin_range_row("Downbeat fill bins", 'downbeat_fill_bin_low', 'downbeat_fill_bin_high', 'downbeat_fill_bin_range')
        _add_fill_sustain_slider("Downbeat", 'downbeat_overall_amp_fill_sustain_frames')
        _add_fill_bin_range_row("Beat fill bins", 'beat_fill_bin_low', 'beat_fill_bin_high', 'beat_fill_bin_range')
        _add_fill_sustain_slider("Beat", 'beat_overall_amp_fill_sustain_frames')
        _add_fill_bin_range_row("Sync fill bins", 'syncopation_fill_bin_low', 'syncopation_fill_bin_high', 'sync_fill_bin_range')
        _add_fill_sustain_slider("Syncopation", 'syncopation_overall_amp_fill_sustain_frames')

        dual_band_gate_cb = QCheckBox("Enable Sub-Bass + Treble Lock")
        dual_band_gate_cb.setChecked(bool(getattr(self.config.stroke, 'dual_band_db_gate_enabled', False)))
        dual_band_gate_cb.setToolTip("Require BOTH sub-bass energy AND high-frequency presence before strokes fire (prevents bass-only or treble-only false positives)")
        dual_band_gate_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'dual_band_db_gate_enabled', state == 2)
        )
        gate_layout.addWidget(dual_band_gate_cb)

        dual_sub_bass_db_slider = SliderWithLabel(
            "Sub-Bass Minimum Energy (dB)",
            -100.0,
            -50.0,
            float(getattr(self.config.stroke, 'dual_band_sub_bass_db_min', -80.0) or -80.0),
            1,
        )
        dual_sub_bass_db_slider.valueChanged.connect(
            lambda v: (
                setattr(self.config.stroke, 'dual_band_sub_bass_db_min', float(v)),
                _show_freqdb_ghost_ref('dual_sub_bass_db_min', float(v), 'Dual sub-bass dB', '#5CFF9A', band='low', mode='db_line')
            )
        )
        dual_sub_bass_db_slider.setToolTip("Sub-bass must be this strong (in dB) for the gate to pass. Lower (more negative) = more permissive; higher (closer to 0) = stricter")
        gate_layout.addWidget(dual_sub_bass_db_slider)

        dual_high_db_slider = SliderWithLabel(
            "Treble Minimum Energy (dB)",
            -100.0,
            -50.0,
            float(getattr(self.config.stroke, 'dual_band_high_db_min', -80.0) or -80.0),
            1,
        )
        dual_high_db_slider.valueChanged.connect(
            lambda v: (
                setattr(self.config.stroke, 'dual_band_high_db_min', float(v)),
                _show_freqdb_ghost_ref('dual_high_db_min', float(v), 'Dual high dB', '#FF9AD9', band='high', mode='db_line')
            )
        )
        dual_high_db_slider.setToolTip("Treble/high-band must be this strong (in dB) for gate to pass. Lower = more permissive; higher = stricter")
        gate_layout.addWidget(dual_high_db_slider)

        high_tip_gate_cb = QCheckBox("Enable high-tip fullness gate")
        high_tip_gate_cb.setChecked(bool(getattr(self.config.stroke, 'high_tip_fullness_enabled', False)))
        high_tip_gate_cb.setToolTip("Require high-tip presence (frequency range + dB floor + occupancy) after dual-band dB gate")
        high_tip_gate_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'high_tip_fullness_enabled', state == 2)
        )
        gate_layout.addWidget(high_tip_gate_cb)

        high_tip_range_row = QHBoxLayout()
        high_tip_range_row.addWidget(QLabel("High-tip range (Hz):"))

        high_tip_low_spin = QSpinBox()
        high_tip_low_spin.setRange(100, 22000)
        high_tip_low_spin.setSingleStep(50)
        high_tip_low_spin.setValue(int(float(getattr(self.config.stroke, 'high_tip_freq_low_hz', getattr(self.config.stroke, 'high_tip_freq_hz', 3500.0) or 3500.0) or 3500.0)))
        high_tip_low_spin.setPrefix("low ")
        high_tip_low_spin.setSuffix(" Hz")
        high_tip_range_row.addWidget(high_tip_low_spin)

        high_tip_high_spin = QSpinBox()
        high_tip_high_spin.setRange(100, 22050)
        high_tip_high_spin.setSingleStep(50)
        high_tip_high_spin.setValue(int(float(getattr(self.config.stroke, 'high_tip_freq_high_hz', 16000.0) or 16000.0)))
        high_tip_high_spin.setPrefix("high ")
        high_tip_high_spin.setSuffix(" Hz")
        high_tip_range_row.addWidget(high_tip_high_spin)
        high_tip_range_row.addStretch()

        def _emit_high_tip_range_ghost() -> None:
            low_hz = float(min(high_tip_low_spin.value(), high_tip_high_spin.value()))
            high_hz = float(max(high_tip_low_spin.value(), high_tip_high_spin.value()))
            _show_freqdb_ghost_ref(
                'high_tip_freq_range',
                low_hz,
                'High-tip range',
                color='#FFB3F0',
                dashed=False,
                mode='hz_line',
                range_box=True,
                hz_max=high_hz,
            )

        def _on_high_tip_low_change(v: int) -> None:
            low_val = int(v)
            high_val = int(high_tip_high_spin.value())
            if high_val <= low_val:
                high_val = min(22050, low_val + 50)
                high_tip_high_spin.setValue(high_val)
            self.config.stroke.high_tip_freq_low_hz = float(low_val)
            self.config.stroke.high_tip_freq_high_hz = float(high_val)
            _emit_high_tip_range_ghost()

        def _on_high_tip_high_change(v: int) -> None:
            high_val = int(v)
            low_val = int(high_tip_low_spin.value())
            if high_val <= low_val:
                low_val = max(100, high_val - 50)
                high_tip_low_spin.setValue(low_val)
            self.config.stroke.high_tip_freq_low_hz = float(low_val)
            self.config.stroke.high_tip_freq_high_hz = float(high_val)
            _emit_high_tip_range_ghost()

        high_tip_low_spin.valueChanged.connect(_on_high_tip_low_change)
        high_tip_high_spin.valueChanged.connect(_on_high_tip_high_change)
        gate_layout.addLayout(high_tip_range_row)
        _emit_high_tip_range_ghost()

        high_tip_db_slider = SliderWithLabel(
            "High-tip dB min",
            -100.0,
            -40.0,
            float(getattr(self.config.stroke, 'high_tip_db_min', -90.0) or -90.0),
            1,
        )
        high_tip_db_slider.valueChanged.connect(
            lambda v: (
                setattr(self.config.stroke, 'high_tip_db_min', float(v)),
                _show_freqdb_ghost_ref('high_tip_db_min', float(v), 'High-tip dB', '#FFB3F0', dashed=True, band='high', mode='db_line')
            )
        )
        high_tip_db_slider.setToolTip("Minimum dB floor for high-tip occupancy gate")
        gate_layout.addWidget(high_tip_db_slider)

        high_tip_occ_slider = SliderWithLabel(
            "High-tip occupancy",
            0.0,
            1.0,
            float(getattr(self.config.stroke, 'high_tip_occupancy_threshold', 0.20) or 0.20),
            3,
            step=0.001,
        )
        high_tip_occ_slider.valueChanged.connect(
            lambda v: (
                setattr(self.config.stroke, 'high_tip_occupancy_threshold', float(v)),
                _show_freqdb_ghost_ref('high_tip_occ', float(v), 'High-tip occ', '#FFC8F4', dashed=True, band='high', range_box=True, mode='occupancy')
            )
        )
        high_tip_occ_slider.setToolTip("Required occupancy in the high-tip gate window")
        gate_layout.addWidget(high_tip_occ_slider)

        mid_block_cb = QCheckBox("Block triggers when mid is high and bass is too low")
        mid_block_cb.setChecked(bool(getattr(self.config.stroke, 'block_mid_trigger_range_enabled', False)))
        mid_block_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'block_mid_trigger_range_enabled', state == 2)
        )
        mid_block_cb.setToolTip(
            "When enabled, beat/downbeat triggers are blocked if detected frequency is in the configured mid range and\n"
            "rolling bass activity is below threshold."
        )
        gate_layout.addWidget(mid_block_cb)

        mid_block_row = QHBoxLayout()
        mid_block_low_spin = QSpinBox()
        mid_block_low_spin.setRange(0, 10000)
        mid_block_low_spin.setSingleStep(10)
        mid_block_low_spin.setValue(int(float(getattr(self.config.stroke, 'block_mid_trigger_low_hz', 1000.0) or 1000.0)))
        mid_block_low_spin.setSuffix(" Hz")
        mid_block_row.addWidget(QLabel("Mid block low:"))
        mid_block_row.addWidget(mid_block_low_spin)

        mid_block_high_spin = QSpinBox()
        mid_block_high_spin.setRange(1, 12000)
        mid_block_high_spin.setSingleStep(10)
        mid_block_high_spin.setValue(int(float(getattr(self.config.stroke, 'block_mid_trigger_high_hz', 2200.0) or 2200.0)))
        mid_block_high_spin.setSuffix(" Hz")
        mid_block_row.addWidget(QLabel("high:"))
        mid_block_row.addWidget(mid_block_high_spin)
        mid_block_row.addStretch()

        def _on_mid_block_low_change(v: int) -> None:
            low = int(v)
            high = int(mid_block_high_spin.value())
            if high <= low:
                high = low + 1
                mid_block_high_spin.setValue(high)
            self.config.stroke.block_mid_trigger_low_hz = float(low)
            self.config.stroke.block_mid_trigger_high_hz = float(high)
            _show_freqdb_ghost_ref('mid_block_low_hz', float(low), 'Mid block low', '#FF9A66', dashed=True, mode='hz_line', range_box=False)
            _show_freqdb_ghost_ref('mid_block_high_hz', float(high), 'Mid block high', '#FFB366', dashed=True, mode='hz_line', range_box=False)

        def _on_mid_block_high_change(v: int) -> None:
            high = int(v)
            low = int(mid_block_low_spin.value())
            if high <= low:
                low = max(0, high - 1)
                mid_block_low_spin.setValue(low)
            self.config.stroke.block_mid_trigger_low_hz = float(low)
            self.config.stroke.block_mid_trigger_high_hz = float(high)
            _show_freqdb_ghost_ref('mid_block_low_hz', float(low), 'Mid block low', '#FF9A66', dashed=True, mode='hz_line', range_box=False)
            _show_freqdb_ghost_ref('mid_block_high_hz', float(high), 'Mid block high', '#FFB366', dashed=True, mode='hz_line', range_box=False)

        mid_block_low_spin.valueChanged.connect(_on_mid_block_low_change)
        mid_block_high_spin.valueChanged.connect(_on_mid_block_high_change)
        gate_layout.addLayout(mid_block_row)

        mid_block_window_row = QHBoxLayout()
        mid_block_window_row.addWidget(QLabel("Mid block avg window:"))
        mid_block_window_spin = QSpinBox()
        mid_block_window_spin.setRange(1, 60)
        mid_block_window_spin.setSingleStep(1)
        mid_block_window_spin.setValue(
            int(getattr(self.config.stroke, 'block_mid_trigger_window_frames', 8) or 8)
        )
        mid_block_window_spin.setSuffix(" frames")
        mid_block_window_spin.setToolTip(
            "Rolling average window used by the mid-block gate (larger = smoother, slower response)."
        )
        mid_block_window_spin.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'block_mid_trigger_window_frames', int(v))
        )
        mid_block_window_row.addWidget(mid_block_window_spin)
        mid_block_window_row.addStretch()
        gate_layout.addLayout(mid_block_window_row)

        mid_block_bass_ratio_slider = SliderWithLabel(
            "Mid block bass/mid max ratio",
            0.0,
            2.0,
            float(
                getattr(
                    self.config.stroke,
                    'block_mid_trigger_bass_to_mid_max_ratio',
                    0.5,
                ) or 0.5
            ),
            3,
            step=0.01,
        )
        mid_block_bass_ratio_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'block_mid_trigger_bass_to_mid_max_ratio', float(v))
        )
        mid_block_bass_ratio_slider.setToolTip(
            "Mid-trigger block closes when rolling bass activity is at or below (rolling mid activity × this ratio)."
        )
        gate_layout.addWidget(mid_block_bass_ratio_slider)

        scroll_layout.addWidget(gate_group)

        scheduling_group = QGroupBox("Beat Scheduling")
        scheduling_layout = QVBoxLayout(scheduling_group)

        lead_row = QHBoxLayout()
        lead_label = QLabel("Scheduled lead (ms):")
        lead_label.setStyleSheet("color: #ccc;")
        lead_row.addWidget(lead_label)
        lead_spin = QSpinBox()
        lead_spin.setMinimum(0)
        lead_spin.setMaximum(200)
        lead_spin.setSingleStep(1)
        lead_spin.setValue(int(getattr(self.config.beat, 'scheduled_lead_ms', 0)))
        lead_spin.setToolTip("Land scheduled arcs this many milliseconds before predicted beat")
        lead_spin.valueChanged.connect(
            lambda v: setattr(self.config.beat, 'scheduled_lead_ms', int(v))
        )
        lead_row.addWidget(lead_spin)
        scheduling_layout.addLayout(lead_row)

        scroll_layout.addWidget(scheduling_group)

        # ===== Post-Silence Volume Ramp =====
        silence_ramp_group = QGroupBox("Post-Silence Volume Ramp")
        silence_ramp_layout = QVBoxLayout(silence_ramp_group)

        silence_ramp_info = QLabel("After silence (track change), reduce volume and slowly\nraise it back over a configurable duration.")
        silence_ramp_info.setStyleSheet("color: #aaa; font-size: 11px;")
        silence_ramp_layout.addWidget(silence_ramp_info)

        # Volume reduction slider (0% - 50%)
        vol_reduction_slider = SliderWithLabel(
            "Volume reduction (%)",
            0.0,
            0.50,
            float(getattr(self.config.stroke, 'post_silence_vol_reduction', 0.15) or 0.15),
            2,
        )
        vol_reduction_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'post_silence_vol_reduction', v)
        )
        silence_ramp_layout.addWidget(vol_reduction_slider)

        # Ramp duration slider (1.0 - 8.0 seconds)
        ramp_dur_slider = SliderWithLabel(
            "Ramp duration (seconds)",
            1.0,
            8.0,
            float(getattr(self.config.stroke, 'post_silence_ramp_seconds', 4.0) or 4.0),
            1,
        )
        ramp_dur_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'post_silence_ramp_seconds', v)
        )
        silence_ramp_layout.addWidget(ramp_dur_slider)

        fade_drop_row = QHBoxLayout()
        fade_drop_label = QLabel("Fade max drop points (out of 100):")
        fade_drop_label.setStyleSheet("color: #ccc;")
        fade_drop_row.addWidget(fade_drop_label)
        fade_drop_spin = QSpinBox()
        fade_drop_spin.setRange(0, 10)
        fade_drop_spin.setSingleStep(1)
        fade_drop_spin.setValue(int(np.clip(getattr(self.config.stroke, 'silence_fade_drop_points', 10) or 10, 0, 10)))
        fade_drop_spin.setToolTip("Caps runtime fade reduction to this many volume points (0-10)")
        fade_drop_spin.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'silence_fade_drop_points', int(v))
        )
        fade_drop_row.addWidget(fade_drop_spin)
        fade_drop_row.addStretch()
        silence_ramp_layout.addLayout(fade_drop_row)

        scroll_layout.addWidget(silence_ramp_group)

        # ===== Flux Controls =====
        flux_group = QGroupBox("Flux Sensitivity")
        self._advanced_flux_group = flux_group
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
            float(getattr(self.config.stroke, 'flux_threshold', 0.05) or 0.05),
            3,
        )
        self._advanced_flux_threshold_slider = flux_thresh_slider
        flux_thresh_slider.valueChanged.connect(
            lambda v: (
                setattr(self.config.stroke, 'flux_threshold', v),
                _set_stroke_attr_with_ref('flux_threshold', 'flux_threshold', 'Flux threshold', '#66D9FF', ghost_band='full')(float(v))
            )
        )
        flux_layout.addWidget(flux_thresh_slider)

        flux_scaling_slider = SliderWithLabel(
            "Flux Scaling (size)",
            0.0,
            2.0,
            float(getattr(self.config.stroke, 'flux_scaling_weight', 1.0) or 1.0),
            2,
        )
        self._advanced_flux_scaling_slider = flux_scaling_slider
        flux_scaling_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'flux_scaling_weight', float(v))
        )
        flux_layout.addWidget(flux_scaling_slider)

        low_band_info = QLabel("Beat gate = low-band mean/fullness/ratio checks.")
        low_band_info.setStyleSheet("color: #999; font-size: 10px;")
        flux_layout.addWidget(low_band_info)

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
                setattr(self.config.stroke, attr_name, value)
                resolved = ghost_value_resolver(value) if callable(ghost_value_resolver) else value
                ghost_value = self.freqdb_canvas._as_float(resolved, value) if hasattr(self, 'freqdb_canvas') and hasattr(self.freqdb_canvas, '_as_float') else float(value)
                if hasattr(self, 'freqdb_canvas') and hasattr(self.freqdb_canvas, 'show_flux_ghost'):
                    self.freqdb_canvas.show_flux_ghost(
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

        def _style_ref_slider(widget: SliderWithLabel, ref_color: str, ref_label: str):
            hint = f"Adjusts '{ref_label}' spectral gate reference on the dB/Hz visualizer"
            widget.label.setStyleSheet(f"color: {ref_color};")
            widget.setToolTip(hint)
            widget.label.setToolTip(hint)
            widget.slider.setToolTip(hint)
            widget.value_label.setToolTip(hint)

        def _set_beat_attr_with_ref(
            attr_name: str,
            ref_key: str,
            ref_label: str,
            ref_color: str,
            dashed: bool = False,
            ghost_band: str = 'full',
            ghost_range: bool = False,
            ghost_mode: str = 'threshold',
        ):
            def _handler(v: float):
                value = float(v)
                setattr(self.config.beat, attr_name, value)
                if hasattr(self, 'freqdb_canvas') and hasattr(self.freqdb_canvas, 'show_flux_ghost'):
                    self.freqdb_canvas.show_flux_ghost(
                        ref_key,
                        value,
                        ref_label,
                        color=ref_color,
                        duration_s=15.0,
                        dashed=dashed,
                        band=ghost_band,
                        range_box=ghost_range,
                        mode=ghost_mode,
                    )
            return _handler

        low_band_window_row = QHBoxLayout()
        low_band_window_label = QLabel("Low-band gate window (frames):")
        low_band_window_label.setStyleSheet("color: #ccc;")
        low_band_window_row.addWidget(low_band_window_label)
        low_band_window_spin = QSpinBox()
        low_band_window_spin.setMinimum(8)
        low_band_window_spin.setMaximum(60)
        low_band_window_spin.setValue(int(getattr(self.config.stroke, 'low_band_window_frames', 18) or 18))
        low_band_window_spin.setToolTip("Low-band gate history window (frames)")
        low_band_window_label.setToolTip("Low-band gate history window (frames)")
        low_band_window_spin.valueChanged.connect(
            lambda v: (
                setattr(self.config.stroke, 'low_band_window_frames', int(v)),
            )
        )
        low_band_window_row.addWidget(low_band_window_spin)
        flux_layout.addLayout(low_band_window_row)

        low_band_mean_slider = SliderWithLabel(
            "Low-band mean threshold (norm)",
            0.001,
            2.00,
            float(getattr(self.config.stroke, 'low_band_activity_threshold', 0.20) or 0.20),
            3,
            step=0.001,
        )
        low_band_mean_slider.valueChanged.connect(_set_stroke_attr_with_ref('low_band_activity_threshold', 'low_band_mean', 'Low mean', '#32FF32', ghost_band='low'))
        _style_ref_slider(low_band_mean_slider, '#32FF32', 'Low mean')
        flux_layout.addWidget(low_band_mean_slider)

        low_band_occ_slider = SliderWithLabel(
            "Low-band fullness occupancy (0-1)",
            0.00,
            1.00,
            float(getattr(self.config.stroke, 'low_band_fullness_occupancy_threshold', 0.62) or 0.62),
            3,
            step=0.001,
        )
        low_band_occ_slider.valueChanged.connect(_set_stroke_attr_with_ref(
            'low_band_fullness_occupancy_threshold',
            'low_band_fullness_occ',
            'Low fullness occ',
            '#66FFAA',
            dashed=True,
            ghost_band='low',
            ghost_range=True,
            ghost_mode='occupancy',
        ))
        _style_ref_slider(low_band_occ_slider, '#66FFAA', 'Low fullness occ')
        flux_layout.addWidget(low_band_occ_slider)

        low_high_ratio_slider = SliderWithLabel(
            "Low:high mean ratio min (unitless)",
            0.10,
            3.00,
            float(getattr(self.config.stroke, 'low_band_to_high_ratio_min', 0.58) or 0.58),
            3,
            step=0.001,
        )
        low_high_ratio_slider.valueChanged.connect(_set_stroke_attr_with_ref(
            'low_band_to_high_ratio_min',
            'low_high_ratio_min',
            'Low:high ratio',
            '#99FFCC',
            dashed=True,
            ghost_band='full',
            ghost_range=False,
            ghost_mode='threshold',
        ))
        _style_ref_slider(low_high_ratio_slider, '#99FFCC', 'Low:high ratio')
        flux_layout.addWidget(low_high_ratio_slider)

        mid_bass_support_cb = QCheckBox("Enable mid-bass support fallback")
        mid_bass_support_cb.setChecked(bool(getattr(self.config.stroke, 'mid_bass_support_enabled', True)))
        mid_bass_support_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'mid_bass_support_enabled', state == 2)
        )
        flux_layout.addWidget(mid_bass_support_cb)

        mid_bass_range_row = QHBoxLayout()
        mid_bass_range_row.addWidget(QLabel("Mid-bass support range (Hz):"))
        mid_bass_low_spin = QSpinBox()
        mid_bass_low_spin.setRange(20, 2000)
        mid_bass_low_spin.setSingleStep(10)
        mid_bass_low_spin.setValue(int(float(getattr(self.config.stroke, 'mid_bass_freq_low_hz', 200.0) or 200.0)))
        mid_bass_low_spin.setPrefix("low ")
        mid_bass_low_spin.setSuffix(" Hz")
        mid_bass_range_row.addWidget(mid_bass_low_spin)

        mid_bass_high_spin = QSpinBox()
        mid_bass_high_spin.setRange(30, 3000)
        mid_bass_high_spin.setSingleStep(10)
        mid_bass_high_spin.setValue(int(float(getattr(self.config.stroke, 'mid_bass_freq_high_hz', 400.0) or 400.0)))
        mid_bass_high_spin.setPrefix("high ")
        mid_bass_high_spin.setSuffix(" Hz")
        mid_bass_range_row.addWidget(mid_bass_high_spin)
        mid_bass_range_row.addStretch()

        def _emit_mid_bass_range_ghost() -> None:
            low_hz = float(min(mid_bass_low_spin.value(), mid_bass_high_spin.value()))
            high_hz = float(max(mid_bass_low_spin.value(), mid_bass_high_spin.value()))
            _show_freqdb_ghost_ref(
                'mid_bass_range',
                low_hz,
                'Mid-bass range',
                color='#8BFFB6',
                dashed=False,
                mode='hz_line',
                range_box=True,
                hz_max=high_hz,
            )

        def _on_mid_bass_low_change(v: int) -> None:
            low_val = int(v)
            high_val = int(mid_bass_high_spin.value())
            if high_val <= low_val:
                high_val = min(3000, low_val + 10)
                mid_bass_high_spin.setValue(high_val)
            self.config.stroke.mid_bass_freq_low_hz = float(low_val)
            self.config.stroke.mid_bass_freq_high_hz = float(high_val)
            _emit_mid_bass_range_ghost()

        def _on_mid_bass_high_change(v: int) -> None:
            high_val = int(v)
            low_val = int(mid_bass_low_spin.value())
            if high_val <= low_val:
                low_val = max(20, high_val - 10)
                mid_bass_low_spin.setValue(low_val)
            self.config.stroke.mid_bass_freq_low_hz = float(low_val)
            self.config.stroke.mid_bass_freq_high_hz = float(high_val)
            _emit_mid_bass_range_ghost()

        mid_bass_low_spin.valueChanged.connect(_on_mid_bass_low_change)
        mid_bass_high_spin.valueChanged.connect(_on_mid_bass_high_change)
        flux_layout.addLayout(mid_bass_range_row)
        _emit_mid_bass_range_ghost()

        mid_bass_activity_slider = SliderWithLabel(
            "Mid-bass activity min (norm)",
            0.0005,
            1.00,
            float(getattr(self.config.stroke, 'mid_bass_activity_threshold', 0.035) or 0.035),
            3,
            step=0.001,
        )
        mid_bass_activity_slider.valueChanged.connect(_set_stroke_attr_with_ref(
            'mid_bass_activity_threshold',
            'mid_bass_activity_min',
            'Mid-bass activity',
            '#7DFFB0',
            dashed=False,
            ghost_band='low',
            ghost_range=False,
            ghost_mode='threshold',
        ))
        _style_ref_slider(mid_bass_activity_slider, '#7DFFB0', 'Mid-bass activity')
        flux_layout.addWidget(mid_bass_activity_slider)

        mid_bass_occ_slider = SliderWithLabel(
            "Mid-bass occupancy (0-1)",
            0.00,
            1.00,
            float(getattr(self.config.stroke, 'mid_bass_occupancy_threshold', 0.45) or 0.45),
            3,
            step=0.001,
        )
        mid_bass_occ_slider.valueChanged.connect(_set_stroke_attr_with_ref(
            'mid_bass_occupancy_threshold',
            'mid_bass_occ',
            'Mid-bass occ',
            '#A2FFC8',
            dashed=True,
            ghost_band='low',
            ghost_range=True,
            ghost_mode='occupancy',
        ))
        _style_ref_slider(mid_bass_occ_slider, '#A2FFC8', 'Mid-bass occ')
        flux_layout.addWidget(mid_bass_occ_slider)

        downbeat_relax_slider = SliderWithLabel(
            "Downbeat gate relax",
            0.50,
            1.00,
            float(getattr(self.config.stroke, 'downbeat_low_band_relax', 0.85) or 0.85),
            3,
            step=0.001,
        )
        downbeat_relax_slider.valueChanged.connect(_set_stroke_attr_with_ref(
            'downbeat_low_band_relax',
            'downbeat_low_relax',
            'Low relax eff mean',
            '#66CC88',
            dashed=True,
            ghost_band='low',
            ghost_range=True,
            ghost_mode='threshold',
            ghost_value_resolver=lambda relax: float(getattr(self.config.stroke, 'low_band_activity_threshold', 0.20) or 0.20) * float(relax),
        ))
        flux_layout.addWidget(downbeat_relax_slider)

        high_gate_cb = QCheckBox("Require upper-band presence/pattern for beat strokes")
        high_gate_cb.setChecked(bool(getattr(self.config.stroke, 'high_band_gate_enabled', True)))
        high_gate_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'high_band_gate_enabled', state == 2)
        )
        flux_layout.addWidget(high_gate_cb)

        high_include_mid_cb = QCheckBox("Include mid band in upper gate")
        high_include_mid_cb.setChecked(bool(getattr(self.config.stroke, 'high_band_include_mid', True)))
        high_include_mid_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'high_band_include_mid', state == 2)
        )
        flux_layout.addWidget(high_include_mid_cb)

        high_info = QLabel("Upper gate pass = presence (mean+occupancy+variation) OR recent upper-band beat pattern.")
        high_info.setStyleSheet("color: #999; font-size: 10px;")
        flux_layout.addWidget(high_info)

        high_band_window_row = QHBoxLayout()
        high_band_window_label = QLabel("Treble Presence Window (frames):")
        high_band_window_label.setStyleSheet("color: #ccc;")
        high_band_window_row.addWidget(high_band_window_label)
        high_band_window_spin = QSpinBox()
        high_band_window_spin.setMinimum(8)
        high_band_window_spin.setMaximum(60)
        high_band_window_spin.setValue(int(getattr(self.config.stroke, 'high_band_window_frames', 18) or 18))
        high_band_window_spin.setToolTip("How many recent frames to check for sustained high-band activity. Larger = more stable confirmation")
        high_band_window_label.setToolTip("How many recent frames to check for sustained high-band activity. Larger = more stable confirmation")
        high_band_window_spin.valueChanged.connect(
            lambda v: (
                setattr(self.config.stroke, 'high_band_window_frames', int(v)),
            )
        )
        high_band_window_row.addWidget(high_band_window_spin)
        flux_layout.addLayout(high_band_window_row)

        high_mean_slider = SliderWithLabel(
            "Treble Activity Floor",
            0.001,
            2.00,
            float(getattr(self.config.stroke, 'high_band_mean_threshold', 0.12) or 0.12),
            3,
            step=0.001,
        )
        high_mean_slider.setToolTip("Average treble energy (normalized 0-1) required across the window. Filters out brief spikes")
        high_mean_slider.valueChanged.connect(_set_stroke_attr_with_ref('high_band_mean_threshold', 'high_band_mean', 'High mean', '#FF66CC', ghost_band='high'))
        _style_ref_slider(high_mean_slider, '#FF66CC', 'High mean')
        flux_layout.addWidget(high_mean_slider)

        high_floor_slider = SliderWithLabel(
            "High-band fill floor (norm)",
            0.0005,
            1.00,
            float(getattr(self.config.stroke, 'high_band_floor_threshold', 0.06) or 0.06),
            3,
            step=0.001,
        )
        high_floor_slider.valueChanged.connect(_set_stroke_attr_with_ref('high_band_floor_threshold', 'high_band_floor', 'High floor', '#FF88DD', ghost_band='high'))
        _style_ref_slider(high_floor_slider, '#FF88DD', 'High floor')
        flux_layout.addWidget(high_floor_slider)

        high_occ_slider = SliderWithLabel(
            "Treble Occupancy Requirement",
            0.00,
            1.00,
            float(getattr(self.config.stroke, 'high_band_occupancy_threshold', 0.55) or 0.55),
            3,
            step=0.001,
        )
        high_occ_slider.setToolTip("Fraction of frames (0-1) that must exceed threshold for confirmation. Higher = stricter presence check")
        high_occ_slider.valueChanged.connect(_set_stroke_attr_with_ref('high_band_occupancy_threshold', 'high_band_occ', 'High occ', '#FFAAEE', dashed=True, ghost_band='high', ghost_range=True, ghost_mode='occupancy'))
        _style_ref_slider(high_occ_slider, '#FFAAEE', 'High occ')
        flux_layout.addWidget(high_occ_slider)

        high_delta_slider = SliderWithLabel(
            "High-band Δ threshold (norm)",
            0.0005,
            1.00,
            float(getattr(self.config.stroke, 'high_band_delta_threshold', 0.05) or 0.05),
            3,
            step=0.001,
        )
        high_delta_slider.valueChanged.connect(_set_stroke_attr_with_ref('high_band_delta_threshold', 'high_band_delta', 'High Δ', '#FF99DD', ghost_band='high', ghost_range=True))
        _style_ref_slider(high_delta_slider, '#FF99DD', 'High Δ')
        flux_layout.addWidget(high_delta_slider)

        high_var_slider = SliderWithLabel(
            "High-band variance threshold (norm)",
            0.00001,
            0.2000,
            float(getattr(self.config.stroke, 'high_band_variance_threshold', 0.0010) or 0.0010),
            4,
            step=0.001,
        )
        high_var_slider.valueChanged.connect(_set_stroke_attr_with_ref('high_band_variance_threshold', 'high_band_var', 'High var', '#FFBBEE', ghost_band='high', ghost_range=True))
        _style_ref_slider(high_var_slider, '#FFBBEE', 'High var')
        flux_layout.addWidget(high_var_slider)

        center_guard_cb = QCheckBox("Block center+jitter reset while flux activity is high")
        center_guard_cb.setChecked(bool(getattr(self.config.beat, 'center_jitter_flux_guard_enabled', False)))
        center_guard_cb.stateChanged.connect(
            lambda state: setattr(self.config.beat, 'center_jitter_flux_guard_enabled', state == 2)
        )
        flux_layout.addWidget(center_guard_cb)

        center_guard_delta_slider = SliderWithLabel(
            "Center reset guard Δflux",
            0.01,
            3.00,
            float(getattr(self.config.beat, 'center_jitter_flux_delta_threshold', 0.20) or 0.20),
            2,
        )
        center_guard_delta_slider.valueChanged.connect(_set_beat_attr_with_ref(
            'center_jitter_flux_delta_threshold',
            'center_reset_guard_delta',
            'Center reset Δflux',
            '#F7D774',
            dashed=True,
            ghost_band='full',
            ghost_range=True,
        ))
        _style_ref_slider(center_guard_delta_slider, '#F7D774', 'Center reset Δflux')
        flux_layout.addWidget(center_guard_delta_slider)

        center_guard_avg_slider = SliderWithLabel(
            "Center reset guard avg",
            0.01,
            3.00,
            float(getattr(self.config.beat, 'center_jitter_flux_avg_threshold', 0.25) or 0.25),
            2,
        )
        center_guard_avg_slider.valueChanged.connect(_set_beat_attr_with_ref(
            'center_jitter_flux_avg_threshold',
            'center_reset_guard_avg',
            'Center reset avg',
            '#E8C96A',
            dashed=True,
            ghost_band='full',
            ghost_range=True,
        ))
        _style_ref_slider(center_guard_avg_slider, '#E8C96A', 'Center reset avg')
        flux_layout.addWidget(center_guard_avg_slider)

        scroll_layout.addWidget(flux_group)

        # ===== Expression Layer Controls =====
        expression_group = QGroupBox("Expression Layer")
        expression_layout = QVBoxLayout(expression_group)

        expr_info = QLabel(
            "Artistic expression: orbit speed variation, center wandering,\n"
            "direction changes, tension pauses, and session arc."
        )
        expr_info.setStyleSheet("color: #aaa; font-size: 11px;")
        expression_layout.addWidget(expr_info)

        # ── Orbit Speed Variation ──
        orbit_speed_cb = QCheckBox("Orbit speed variation (energy-driven turns per journey)")
        orbit_speed_cb.setChecked(bool(getattr(self.config.stroke, 'orbit_speed_variation_enabled', True)))
        orbit_speed_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'orbit_speed_variation_enabled', state == 2)
        )
        expression_layout.addWidget(orbit_speed_cb)

        orbit_min_turns_slider = SliderWithLabel(
            "Min turns (low energy)",
            0.25, 1.50,
            float(getattr(self.config.stroke, 'orbit_speed_min_turns', 0.75) or 0.75),
            2,
        )
        orbit_min_turns_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'orbit_speed_min_turns', float(v))
        )
        expression_layout.addWidget(orbit_min_turns_slider)

        orbit_max_turns_slider = SliderWithLabel(
            "Max turns (high energy)",
            0.50, 2.00,
            float(getattr(self.config.stroke, 'orbit_speed_max_turns', 1.5) or 1.5),
            2,
        )
        orbit_max_turns_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'orbit_speed_max_turns', float(v))
        )
        expression_layout.addWidget(orbit_max_turns_slider)

        # ── Center Wandering ──
        wander_cb = QCheckBox("Center wandering (orbit drifts horizontally)")
        wander_cb.setChecked(bool(getattr(self.config.stroke, 'center_wander_enabled', True)))
        wander_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'center_wander_enabled', state == 2)
        )
        expression_layout.addWidget(wander_cb)

        wander_max_slider = SliderWithLabel(
            "Wander max offset",
            0.0, 0.50,
            float(getattr(self.config.stroke, 'center_wander_max_x', 0.20) or 0.20),
            2,
        )
        wander_max_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'center_wander_max_x', float(v))
        )
        expression_layout.addWidget(wander_max_slider)

        wander_cycle_slider = SliderWithLabel(
            "Wander cycle (seconds)",
            5.0, 60.0,
            float(getattr(self.config.stroke, 'center_wander_cycle_s', 25.0) or 25.0),
            1,
        )
        wander_cycle_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'center_wander_cycle_s', float(v))
        )
        expression_layout.addWidget(wander_cycle_slider)

        wander_energy_slider = SliderWithLabel(
            "Wander energy influence",
            0.0, 1.0,
            float(getattr(self.config.stroke, 'center_wander_energy_scale', 0.6) or 0.6),
            2,
        )
        wander_energy_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'center_wander_energy_scale', float(v))
        )
        expression_layout.addWidget(wander_energy_slider)

        # ── Direction Changes ──
        direction_cb = QCheckBox("Direction changes at phrase boundaries")
        direction_cb.setChecked(bool(getattr(self.config.stroke, 'direction_change_enabled', True)))
        direction_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'direction_change_enabled', state == 2)
        )
        expression_layout.addWidget(direction_cb)

        direction_interval_slider = SliderWithLabel(
            "Min interval between reversals (s)",
            5.0, 60.0,
            float(getattr(self.config.stroke, 'direction_change_interval_s', 15.0) or 15.0),
            1,
        )
        direction_interval_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'direction_change_interval_s', float(v))
        )
        expression_layout.addWidget(direction_interval_slider)

        direction_drop_slider = SliderWithLabel(
            "Energy change to trigger reversal",
            0.10, 0.80,
            float(getattr(self.config.stroke, 'direction_change_energy_drop', 0.35) or 0.35),
            2,
        )
        direction_drop_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'direction_change_energy_drop', float(v))
        )
        expression_layout.addWidget(direction_drop_slider)

        # ── Tension Pauses ──
        tension_cb = QCheckBox("Tension pauses (dramatic freeze on energy drops)")
        tension_cb.setChecked(bool(getattr(self.config.stroke, 'tension_pause_enabled', True)))
        tension_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'tension_pause_enabled', state == 2)
        )
        expression_layout.addWidget(tension_cb)

        tension_drop_slider = SliderWithLabel(
            "Energy drop to trigger pause",
            0.15, 0.80,
            float(getattr(self.config.stroke, 'tension_pause_energy_drop', 0.40) or 0.40),
            2,
        )
        tension_drop_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'tension_pause_energy_drop', float(v))
        )
        expression_layout.addWidget(tension_drop_slider)

        tension_hold_slider = SliderWithLabel(
            "Pause hold duration (s)",
            0.10, 1.50,
            float(getattr(self.config.stroke, 'tension_pause_hold_s', 0.45) or 0.45),
            2,
        )
        tension_hold_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'tension_pause_hold_s', float(v))
        )
        expression_layout.addWidget(tension_hold_slider)

        tension_cooldown_slider = SliderWithLabel(
            "Pause cooldown (s)",
            2.0, 30.0,
            float(getattr(self.config.stroke, 'tension_pause_cooldown_s', 10.0) or 10.0),
            1,
        )
        tension_cooldown_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'tension_pause_cooldown_s', float(v))
        )
        expression_layout.addWidget(tension_cooldown_slider)

        # ── Session Arc ──
        session_cb = QCheckBox("Session arc (gradual long-term intensity evolution)")
        session_cb.setChecked(bool(getattr(self.config.stroke, 'session_arc_enabled', True)))
        session_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'session_arc_enabled', state == 2)
        )
        expression_layout.addWidget(session_cb)

        session_influence_slider = SliderWithLabel(
            "Session arc radius influence",
            0.0, 0.30,
            float(getattr(self.config.stroke, 'session_arc_radius_influence', 0.10) or 0.10),
            2,
        )
        session_influence_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'session_arc_radius_influence', float(v))
        )
        expression_layout.addWidget(session_influence_slider)

        scroll_layout.addWidget(expression_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        
        layout.addWidget(scroll)

        if scroll_to_flux:
            self._scroll_advanced_controls_to_flux()

        return content

    def _on_help(self):
        """Show Help/Troubleshooting dialog with reset buttons (non-modal)"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QScrollArea, QGroupBox, QPushButton, QHBoxLayout
        
        dialog = QDialog(self)
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
        audio_btn.clicked.connect(lambda: self._on_options_audio_device())
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
        floor_reset_btn.clicked.connect(lambda: self.peak_floor_slider.setValue(0.0))
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
            setattr(self.config.jitter, 'enabled', True),
            self._sync_effects_menu_actions()
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
    
    def _get_external_data_start_dir(self) -> str:
        """Pick a sensible starting directory for user-provided files, preferring D: if available."""
        candidates = [
            Path("D:/breadbeats_datasets"),
            Path("D:/"),
            Path(str(get_config_dir())),
            Path.cwd(),
        ]
        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_dir():
                    return str(candidate)
            except Exception:
                continue
        return str(Path.cwd())
    
    def _apply_preset_data(self, data: dict):
        """Apply preset data dictionary to config and UI"""
        # Helper to safely set attributes
        def safe_set(obj, key, value):
            if hasattr(obj, key):
                setattr(obj, key, value)
        
        # Beat detection
        if 'beat' in data:
            for k, v in data['beat'].items():
                safe_set(self.config.beat, k, v)
        
        # Stroke settings
        if 'stroke' in data:
            for k, v in data['stroke'].items():
                safe_set(self.config.stroke, k, v)
        
        # Jitter
        if 'jitter' in data:
            for k, v in data['jitter'].items():
                safe_set(self.config.jitter, k, v)
        
        # Creep
        if 'creep' in data:
            for k, v in data['creep'].items():
                safe_set(self.config.creep, k, v)
        self.config.creep.enabled = False
        
        # Audio
        if 'audio' in data:
            for k, v in data['audio'].items():
                safe_set(self.config.audio, k, v)
        
        # Pulse frequency
        if 'pulse_freq' in data:
            for k, v in data['pulse_freq'].items():
                safe_set(self.config.pulse_freq, k, v)
        
        # Carrier frequency
        if 'carrier_freq' in data:
            for k, v in data['carrier_freq'].items():
                safe_set(self.config.carrier_freq, k, v)
        
        # Apply to UI
        self._apply_config_to_ui()
    
    def _on_menu_fft_change(self, index: int):
        """Handle FFT size change from menu"""
        self._on_fft_size_change(index)
    
    def _on_menu_spectrum_change(self, index: int):
        """Handle spectrum update rate change from menu"""
        self._on_spectrum_skip_change(index)

    def _on_fft_bin_diagnostics(self):
        """Show FFT bin resolution details and nearest-bin mapping for a target frequency."""
        sample_rate = int(getattr(self.config.audio, 'sample_rate', 44100) or 44100)
        fft_size = int(getattr(self.config.audio, 'fft_size', 1024) or 1024)
        if hasattr(self, 'audio_engine') and self.audio_engine is not None:
            try:
                sample_rate = int(getattr(self.audio_engine.config.audio, 'sample_rate', sample_rate) or sample_rate)
                fft_size = int(getattr(self.audio_engine, 'fft_size', fft_size) or fft_size)
            except Exception:
                pass

        fft_size = max(16, fft_size)
        nyquist = sample_rate / 2.0
        hz_per_bin = sample_rate / float(fft_size)
        max_bin = (fft_size // 2)

        target_hz, ok = QInputDialog.getDouble(
            self,
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
            self,
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

    def _on_log_level_change(self, level: str):
        """Set global log level and persist selection."""
        set_log_level(level)
        self.config.log_level = level.upper()
        self._sync_log_level_menu(self.config.log_level)

    def _sync_log_level_menu(self, active_level: str):
        """Update log level menu checkmarks."""
        if not hasattr(self, '_log_level_actions'):
            return
        lvl_upper = (active_level or "INFO").upper()
        for action in self._log_level_actions:
            action.blockSignals(True)
            action.setChecked(action.text().upper() == lvl_upper)
            action.blockSignals(False)

    @contextmanager
    def _signals_blocked(self, *widgets):
        """Temporarily block signals on provided widgets."""
        blocked = []
        for w in widgets:
            if w is not None and hasattr(w, 'blockSignals'):
                w.blockSignals(True)
                blocked.append(w)
        try:
            yield
        finally:
            for w in blocked:
                w.blockSignals(False)
    
    def _on_about(self):
        """Show About dialog"""
        about_html = """
<b>bREadbeats v2.0</b><br>
Live Audio to Restim<br><br>
Inspired by:<br>
&nbsp;&nbsp;&nbsp;&nbsp;digitalparkinglot's creations<br>
&nbsp;&nbsp;&nbsp;&nbsp;edger477 (ideas from generator tooling)<br>
&nbsp;&nbsp;&nbsp;&nbsp;diglet48 (wouldn't be here without restim!)<br>
&nbsp;&nbsp;&nbsp;&nbsp;shadlock0133 (music-vibes)<br><br>
Bug reports/share your presets:<br>
bREadfan_69@hotmail.com<br><br>
Like the app?<br>
<a href="https://ko-fi.com/breadbeats">https://ko-fi.com/breadbeats</a>
"""
        msg = QMessageBox(self)
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

    def _schedule_startup_notices(self):
        # First-run device limits prompt eligibility
        dl = self.config.device_limits
        has_values = (dl.p0_freq_max > 0 or dl.c0_freq_max > 0)
        show_device_limits = (not dl.prompted and not dl.dont_show_on_startup and not has_values)

        if show_device_limits:
            QTimer.singleShot(500, lambda: self._on_device_limits(first_run=True))
    
    def _apply_config_to_ui(self):
        """Apply loaded config values to UI sliders"""
        try:
            self.config.stroke.mode = StrokeMode.SIMPLE_CIRCLE
            self._enforce_fixed_effect_axis_values()
            beats_to_index = {4: 0, 3: 1, 6: 2}
            with self._signals_blocked(
                getattr(self, 'detection_type_combo', None),
                getattr(self, 'sensitivity_slider', None),
                getattr(self, 'peak_floor_slider', None),
                getattr(self, 'peak_decay_slider', None),
                getattr(self, 'rise_sens_slider', None),
                getattr(self, 'flux_mult_slider', None),
                getattr(self, 'audio_gain_slider', None),
                getattr(self, 'silence_reset_slider', None),
                getattr(self, 'freq_range_slider', None),
                getattr(self, 'metrics_global_cb', None),
                getattr(self, 'tempo_tracking_checkbox', None),
                getattr(self, 'time_sig_combo', None),
                getattr(self, 'stability_threshold_slider', None),
                getattr(self, 'tempo_timeout_slider', None),
                getattr(self, 'phase_snap_slider', None),
                getattr(self, 'mode_combo', None),
                getattr(self, 'tempo_lock_required_cb', None),
                getattr(self, 'intensity_ramp_spin', None),
                getattr(self, 'intensity_ramp_target_combo', None),
                getattr(self, 'fill_gate_scale_spin', None),
                getattr(self, 'main_silence_close_slider', None),
                getattr(self, 'jitter_effect_action', None),
                getattr(self, 'metronome_lock_required_action', None),
                getattr(self, 'host_edit', None),
                getattr(self, 'port_spin', None),
                getattr(self, 'pulse_freq_range_slider', None),
                getattr(self, 'tcode_freq_range_slider', None),
                getattr(self, 'freq_weight_slider', None),
                getattr(self, 'f0_freq_range_slider', None),
                getattr(self, 'f0_tcode_range_slider', None),
                getattr(self, 'f0_weight_slider', None),
                getattr(self, 'volume_slider', None),
            ):
                # Beat detection tab
                if all(hasattr(self, name) for name in (
                    'detection_type_combo', 'sensitivity_slider', 'peak_floor_slider',
                    'peak_decay_slider', 'rise_sens_slider', 'flux_mult_slider',
                    'audio_gain_slider', 'silence_reset_slider', 'freq_range_slider'
                )):
                    self.detection_type_combo.setCurrentIndex(self.config.beat.detection_type - 1)
                    self.sensitivity_slider.setValue(self.config.beat.sensitivity)
                    self.peak_floor_slider.setValue(self.config.beat.peak_floor)
                    self.peak_decay_slider.setValue(self.config.beat.peak_decay)
                    self.rise_sens_slider.setValue(self.config.beat.rise_sensitivity)
                    self.flux_mult_slider.setValue(self.config.beat.flux_multiplier)
                    self.audio_gain_slider.setValue(self.config.audio.gain)
                    self.silence_reset_slider.setValue(self.config.beat.silence_reset_ms)
                    self.freq_range_slider.setLow(self.config.beat.freq_low)
                    self.freq_range_slider.setHigh(self.config.beat.freq_high)

                # Auto-adjust global toggle
                if hasattr(self, 'metrics_global_cb'):
                    self.metrics_global_cb.setChecked(self.config.auto_adjust.metrics_global_enabled)

                # Tempo tracking settings
                if hasattr(self, 'tempo_tracking_checkbox'):
                    self.tempo_tracking_checkbox.setChecked(self.config.beat.tempo_tracking_enabled)
                if hasattr(self, 'time_sig_combo'):
                    self.time_sig_combo.setCurrentIndex(beats_to_index.get(self.config.beat.beats_per_measure, 0))
                if hasattr(self, 'stability_threshold_slider'):
                    self.stability_threshold_slider.setValue(self.config.beat.stability_threshold)
                if hasattr(self, 'tempo_timeout_slider'):
                    self.tempo_timeout_slider.setValue(self.config.beat.tempo_timeout_ms)
                if hasattr(self, 'phase_snap_slider'):
                    self.phase_snap_slider.setValue(self.config.beat.phase_snap_weight)
                if hasattr(self, 'mode_combo'):
                    self.mode_combo.setCurrentIndex(0)
                self.config.stroke.min_interval_ms = 150
                if hasattr(self, 'tempo_lock_required_cb'):
                    self.tempo_lock_required_cb.setChecked(bool(getattr(self.config.beat, 'tempo_lock_required', False)))
                if hasattr(self, 'metronome_lock_required_action'):
                    self.metronome_lock_required_action.setChecked(bool(getattr(self.config.beat, 'tempo_lock_required', False)))
                if hasattr(self, 'intensity_ramp_spin'):
                    self.intensity_ramp_spin.setValue(
                        float(getattr(self.config.stroke, 'intensity_ramp_hours', 0.0) or 0.0)
                    )
                if hasattr(self, 'intensity_ramp_target_combo'):
                    target_control: Any = self.intensity_ramp_target_combo
                    target = str(getattr(self.config.stroke, 'intensity_ramp_target', 'both') or 'both').strip().lower()
                    if target not in ('size', 'speed', 'both'):
                        target = 'both'
                    set_current_text = getattr(target_control, 'setCurrentText', None)
                    set_value = getattr(target_control, 'setValue', None)
                    if callable(set_current_text):
                        set_current_text(target)
                    elif callable(set_value):
                        set_value({'size': 0, 'speed': 1, 'both': 2}.get(target, 2))
                self._refresh_motion_ramp_visual_state()
                if hasattr(self, 'fill_gate_scale_spin'):
                    self.fill_gate_scale_spin.setValue(
                        self._fill_gate_scale_to_percent(
                            float(getattr(self.config.stroke, 'overall_amp_fill_required_scale', 1.0) or 1.0)
                        )
                    )
                if hasattr(self, 'main_silence_close_slider'):
                    self.main_silence_close_slider.setValue(
                        self._silence_close_to_normalized(
                            float(getattr(self.config.stroke, 'silence_close_threshold', 0.0433) or 0.0433)
                        )
                    )
                advanced_flux_slider = getattr(self, '_advanced_flux_threshold_slider', None)
                if advanced_flux_slider is not None:
                    advanced_flux_slider.setValue(self.config.stroke.flux_threshold)
                advanced_flux_scaling_slider = getattr(self, '_advanced_flux_scaling_slider', None)
                if advanced_flux_scaling_slider is not None:
                    advanced_flux_scaling_slider.setValue(self.config.stroke.flux_scaling_weight)
                auto_fill_widgets = getattr(self, '_auto_fill_controls_widgets', {}) or {}
                auto_fill_enabled = auto_fill_widgets.get('enabled')
                if auto_fill_enabled is not None:
                    auto_fill_enabled.setChecked(bool(getattr(self.config.stroke, 'overall_amp_fill_auto_enabled', True)))
                auto_fill_target = auto_fill_widgets.get('target_pass_rate')
                if auto_fill_target is not None:
                    auto_fill_target.setValue(float(getattr(self.config.stroke, 'overall_amp_fill_auto_target_pass_rate', 0.58) or 0.58))
                auto_fill_alpha = auto_fill_widgets.get('ema_alpha')
                if auto_fill_alpha is not None:
                    auto_fill_alpha.setValue(float(getattr(self.config.stroke, 'overall_amp_fill_auto_ema_alpha', 0.12) or 0.12))
                auto_fill_deadband = auto_fill_widgets.get('deadband')
                if auto_fill_deadband is not None:
                    auto_fill_deadband.setValue(float(getattr(self.config.stroke, 'overall_amp_fill_auto_deadband', 0.06) or 0.06))
                auto_fill_step = auto_fill_widgets.get('step')
                if auto_fill_step is not None:
                    auto_fill_step.setValue(float(getattr(self.config.stroke, 'overall_amp_fill_auto_step', 0.02) or 0.02))
                auto_fill_max_offset = auto_fill_widgets.get('max_offset')
                if auto_fill_max_offset is not None:
                    auto_fill_max_offset.setValue(float(getattr(self.config.stroke, 'overall_amp_fill_auto_max_offset', 0.35) or 0.35))
                auto_fill_min_req = auto_fill_widgets.get('min_required')
                if auto_fill_min_req is not None:
                    auto_fill_min_req.setValue(float(getattr(self.config.stroke, 'overall_amp_fill_auto_min_required', 0.05) or 0.05))
                auto_fill_max_req = auto_fill_widgets.get('max_required')
                if auto_fill_max_req is not None:
                    auto_fill_max_req.setValue(float(getattr(self.config.stroke, 'overall_amp_fill_auto_max_required', 0.98) or 0.98))

                # Effects menu toggles
                jitter_action = getattr(self, 'jitter_effect_action', None)
                if jitter_action is not None:
                    jitter_action.setChecked(bool(getattr(self.config.jitter, 'enabled', True)))

                # Connection settings
                if hasattr(self, 'host_edit'):
                    self.host_edit.setText(self.config.connection.host)
                if hasattr(self, 'port_spin'):
                    self.port_spin.setValue(self.config.connection.port)

                # Other tab (pulse freq settings)
                if all(hasattr(self, name) for name in ('pulse_freq_range_slider', 'tcode_freq_range_slider', 'freq_weight_slider')):
                    self.pulse_freq_range_slider.setLow(self.config.pulse_freq.monitor_freq_min)
                    self.pulse_freq_range_slider.setHigh(self.config.pulse_freq.monitor_freq_max)
                    self.tcode_freq_range_slider.setLow(self.config.pulse_freq.tcode_min)
                    self.tcode_freq_range_slider.setHigh(self.config.pulse_freq.tcode_max)
                    self.freq_weight_slider.setValue(self.config.pulse_freq.freq_weight)

                # Carrier freq (F0) settings
                if all(hasattr(self, name) for name in ('f0_freq_range_slider', 'f0_tcode_range_slider', 'f0_weight_slider')):
                    self.f0_freq_range_slider.setLow(self.config.carrier_freq.monitor_freq_min)
                    self.f0_freq_range_slider.setHigh(self.config.carrier_freq.monitor_freq_max)
                    self.f0_tcode_range_slider.setLow(self.config.carrier_freq.tcode_min)
                    self.f0_tcode_range_slider.setHigh(self.config.carrier_freq.tcode_max)
                    self.f0_weight_slider.setValue(self.config.carrier_freq.freq_weight)

                # Volume (config stores 0-1, slider shows 0-100)
                if hasattr(self, 'volume_slider'):
                    self.volume_slider.setValue(int(self.config.volume * 100))

            # Set active visualizer sample rates and update frequency bands
            if hasattr(self, 'freq_range_slider'):
                self._on_freq_band_change()  # Update beat detection band (red)
            
            # Apply mode-dependent limits after sliders are set
            if hasattr(self, 'mode_combo'):
                self._on_mode_change(0)  # Mode temporarily pinned to circle
            self._on_depth_band_change()  # Update stroke depth band (green)
            if hasattr(self, 'pulse_freq_range_slider'):
                self._on_p0_band_change()  # Update P0 TCode band (blue)
            if hasattr(self, 'f0_freq_range_slider'):
                self._on_f0_band_change()  # Update F0 TCode band (cyan)

            # Apply tempo tracking side effects after values are in place
            if hasattr(self, 'tempo_tracking_checkbox'):
                self._on_tempo_tracking_toggle(2 if self.config.beat.tempo_tracking_enabled else 0)

            # Log level menu (persisted)
            self._sync_log_level_menu(getattr(self.config, 'log_level', get_log_level()))
            
            print("[UI] Loaded all settings from config")
        except AttributeError as e:
            print(f"[UI] Warning: Could not apply all config values: {e}")
        
    def _create_connection_panel(self) -> QWidget:
        """Connection settings panel - simplified, host/port in Options menu (no visible groupbox)"""
        group = QWidget()
        layout = QHBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Hidden Host/Port widgets (needed for functionality but now in Options menu)
        self.host_edit = QLineEdit(self.config.connection.host)
        self.host_edit.setVisible(False)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(self.config.connection.port)
        self.port_spin.setVisible(False)
        
        # Connection status / refresh button
        self.status_label = QPushButton("Connect")
        self.status_label.setFixedSize(100, 40)
        self.status_label.setStyleSheet("color: #fff;")
        self.status_label.setToolTip("Reconnect / refresh TCP connection")
        self.status_label.clicked.connect(self._on_connection_refresh)
        layout.addWidget(self.status_label)
        
        return group
    
    def _create_control_panel(self) -> QWidget:
        """Main control buttons - audio device selection moved to Options menu (no visible groupbox)"""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Hidden audio device widgets (needed for functionality but now in Options menu)
        self.device_combo = QComboBox()
        self._populate_audio_devices()
        self.device_combo.setVisible(False)
        
        # Controls row - all on one row now
        btn_layout = QGridLayout()
        btn_layout.setSpacing(8)
        
        # Start/Stop audio capture
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.clicked.connect(lambda _checked=False: self._on_start_stop())
        self.start_btn.setFixedSize(100, 40)
        self.start_btn.setStyleSheet("color: #0af;")
        start_stack = QVBoxLayout()
        start_stack.setSpacing(2)
        start_stack.addWidget(self.start_btn)
        btn_layout.addLayout(start_stack, 0, 0)
        
        # Play/Pause sending
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(lambda _checked=False: self._on_play_pause())
        self.play_btn.setEnabled(False)
        self.play_btn.setFixedSize(100, 40)
        self.play_btn.setStyleSheet("color: #0af;")
        play_stack = QVBoxLayout()
        play_stack.setSpacing(2)
        play_stack.addWidget(self.play_btn)
        btn_layout.addLayout(play_stack, 0, 1)

        # Volume slider (0 - 100)
        self.volume_slider = SliderWithLabel("Volume", 0, 100, 100, decimals=0)
        self.volume_slider.label.setFixedWidth(60)
        self.volume_slider.setMinimumWidth(180)
        self.volume_slider.setContentsMargins(0, 0, 0, 0)
        btn_layout.addWidget(self.volume_slider, 0, 2, 1, 1, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        # Frequency displays - stacked vertically
        freq_display_layout = QVBoxLayout()
        freq_display_layout.setSpacing(0)
        
        # Carrier Freq display
        self.carrier_freq_label = QLabel("Carrier Freq: off")
        self.carrier_freq_label.setStyleSheet("color: #0af; font-size: 10px;")
        self.carrier_freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_display_layout.addWidget(self.carrier_freq_label)
        
        # Pulse Freq display
        self.pulse_freq_label = QLabel("Pulse Freq: off")
        self.pulse_freq_label.setStyleSheet("color: #0af; font-size: 10px;")
        self.pulse_freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_display_layout.addWidget(self.pulse_freq_label)
        
        # Pulse Width display
        self.p1_display_label = QLabel("Pulse Width: off")
        self.p1_display_label.setStyleSheet("color: #0af; font-size: 10px;")
        self.p1_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_display_layout.addWidget(self.p1_display_label)
        
        # Rise Time display
        self.p3_display_label = QLabel("Rise Time: off")
        self.p3_display_label.setStyleSheet("color: #0af; font-size: 10px;")
        self.p3_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_display_layout.addWidget(self.p3_display_label)
        
        freq_display_widget = QWidget()
        freq_display_widget.setLayout(freq_display_layout)
        btn_layout.addWidget(freq_display_widget, 0, 3, 1, 1, Qt.AlignmentFlag.AlignHCenter)

        # Right-side stack: beat indicators
        right_stack = QVBoxLayout()
        right_stack.setSpacing(2)

        # Beat & Downbeat & Metronome Sync indicators (bottom of right stack)
        beat_row = QHBoxLayout()
        beat_row.setSpacing(4)
        self.beat_indicator = QLabel("●")
        self.beat_indicator.setStyleSheet("color: #333; font-size: 20px;")
        self.beat_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.beat_indicator.setFixedWidth(24)
        self.beat_indicator.setToolTip("Beat")
        beat_row.addWidget(self.beat_indicator)
        self.downbeat_indicator = QLabel("●")
        self.downbeat_indicator.setStyleSheet("color: #333; font-size: 20px;")
        self.downbeat_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.downbeat_indicator.setFixedWidth(24)
        self.downbeat_indicator.setToolTip("Downbeat")
        beat_row.addWidget(self.downbeat_indicator)
        self.metronome_sync_indicator = QLabel("●")
        self.metronome_sync_indicator.setStyleSheet("color: #333; font-size: 20px;")
        self.metronome_sync_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metronome_sync_indicator.setFixedWidth(24)
        self.metronome_sync_indicator.setToolTip("Metronome sync (gray=off, yellow=locking, green=locked)")
        beat_row.addWidget(self.metronome_sync_indicator)
        right_stack.addLayout(beat_row)

        right_stack_widget = QWidget()
        right_stack_widget.setLayout(right_stack)
        btn_layout.addWidget(right_stack_widget, 0, 4, 1, 1, Qt.AlignmentFlag.AlignHCenter)

        for col in range(5):
            btn_layout.setColumnStretch(col, 1)
        layout.addLayout(btn_layout)

        # Beat indicator timer for visual feedback duration
        self.beat_timer = QTimer()
        self.beat_timer.setSingleShot(True)
        self.beat_timer.timeout.connect(self._turn_off_beat_indicator)
        self.beat_indicator_min_duration = 100  # ms

        # Downbeat indicator timer
        self.downbeat_timer = QTimer()
        self.downbeat_timer.setSingleShot(True)
        self.downbeat_timer.timeout.connect(self._turn_off_downbeat_indicator)

        return group
    
    def _populate_audio_devices(self):
        """Populate audio device dropdown - WASAPI devices only (deduplicated)"""
        import sounddevice as sd
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        
        # Find WASAPI host API index and default output device
        wasapi_idx = None
        default_output_idx = None
        for idx, api in enumerate(hostapis):
            if 'WASAPI' in api['name']:
                wasapi_idx = idx
                default_output_idx = api.get('default_output_device', None)
                break
        
        self.device_combo.clear()
        self.audio_device_map = {}  # Map combo index to device index
        self.audio_device_is_loopback = {}  # Track which devices should use WASAPI loopback
        
        loopback_keywords = ['stereo mix', 'what u hear', 'loopback', 'wave out mix', 'system audio']
        loopback_idx = None
        default_output_combo_idx = None  # Track where default output appears
        combo_idx = 0
        seen_names = set()  # For deduplication
        
        if wasapi_idx is not None:
            # Add WASAPI input devices (microphones) - deduplicated by name
            for i, dev in enumerate(devices):
                if dev['hostapi'] == wasapi_idx and dev['max_input_channels'] > 0:
                    # Normalize name for dedup
                    clean_name = dev['name'].strip()
                    if clean_name in seen_names:
                        continue
                    seen_names.add(clean_name)
                    
                    name = f"{clean_name} (Input)"
                    self.device_combo.addItem(name)
                    self.audio_device_map[combo_idx] = i
                    self.audio_device_is_loopback[combo_idx] = False
                    
                    # Find loopback device for default selection
                    if loopback_idx is None and any(keyword in dev['name'].lower() for keyword in loopback_keywords):
                        loopback_idx = combo_idx
                    
                    combo_idx += 1
            
            # Add WASAPI output devices as loopback sources - deduplicated by name
            seen_output_names = set()
            for i, dev in enumerate(devices):
                if dev['hostapi'] == wasapi_idx and dev['max_output_channels'] > 0:
                    clean_name = dev['name'].strip()
                    if clean_name in seen_output_names:
                        continue
                    seen_output_names.add(clean_name)
                    
                    # Mark if this is the system default output device
                    is_default = (i == default_output_idx)
                    prefix = "★ " if is_default else ""
                    name = f"{prefix}{clean_name} [WASAPI Loopback]"
                    self.device_combo.addItem(name)
                    self.audio_device_map[combo_idx] = i
                    self.audio_device_is_loopback[combo_idx] = True
                    
                    # Track default output device's combo index
                    if is_default:
                        default_output_combo_idx = combo_idx
                    
                    # Fallback: first WASAPI loopback if no default found
                    if loopback_idx is None:
                        loopback_idx = combo_idx
                    
                    combo_idx += 1
        else:
            # Fallback: no WASAPI found, show all input devices deduplicated
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    clean_name = dev['name'].strip()
                    if clean_name in seen_names:
                        continue
                    seen_names.add(clean_name)
                    
                    name = f"{clean_name}"
                    self.device_combo.addItem(name)
                    self.audio_device_map[combo_idx] = i
                    self.audio_device_is_loopback[combo_idx] = False
                    combo_idx += 1
        
        # Pre-select: prefer system default output loopback > stereo mix/loopback > first device
        if default_output_combo_idx is not None:
            self.device_combo.setCurrentIndex(default_output_combo_idx)
            print(f"[Main] Auto-selected system default output device for loopback")
        elif loopback_idx is not None:
            self.device_combo.setCurrentIndex(loopback_idx)
        elif combo_idx > 0:
            self.device_combo.setCurrentIndex(0)
    
    def _create_spectrum_panel(self) -> QWidget:
        """Spectrum visualizer panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create active visualizers (only one visible at a time)
        self.waveform_canvas = WaveformLiveCanvas(self, width=8, height=3)
        self.freqdb_canvas = FrequencyDbLiveCanvas(self, width=8, height=3)
        self.fft_bin_canvas = FFTBinBarGraphCanvas(self, width=8, height=3)
        self.waveform_canvas.setVisible(True)  # Start with waveform
        self.freqdb_canvas.setVisible(False)
        self.fft_bin_canvas.setVisible(False)

        layout.addWidget(self.waveform_canvas)
        layout.addWidget(self.freqdb_canvas)
        layout.addWidget(self.fft_bin_canvas)
        
        return widget
    
    def _on_launch_projectm(self):
        """Launch projectM standalone application"""
        if not launch_projectm():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "projectM Not Found",
                "projectM is not installed.\n\n"
                "Install via Steam or download from:\n"
                "https://github.com/projectM-visualizer/projectm"
            )
    
    def _on_visualizer_type_change(self, index: int):
        """Switch visualizer types: 0=Waveform, 1=Freq dB, 2=FFT Bins."""
        self.waveform_canvas.setVisible(index == 0)
        self.freqdb_canvas.setVisible(index == 1)
        self.fft_bin_canvas.setVisible(index == 2)
        
        # Sync the frequency bands to the newly visible visualizer
        if hasattr(self, 'freq_range_slider'):
            self._on_freq_band_change()
        self._on_depth_band_change()
        if hasattr(self, 'pulse_freq_range_slider'):
            self._on_p0_band_change()
    
    def _on_show_peak_indicators_toggle(self, checked: bool):
        """Toggle visibility of peak indicator bars on all visualizers"""
        for canvas in [self.waveform_canvas, self.freqdb_canvas, self.fft_bin_canvas]:
            if hasattr(canvas, 'set_peak_indicators_visible'):
                canvas.set_peak_indicators_visible(checked)

    def _on_show_range_indicators_toggle(self, checked: bool):
        """Toggle visibility of range indicator bands on all visualizers"""
        for canvas in [self.waveform_canvas, self.freqdb_canvas, self.fft_bin_canvas]:
            if hasattr(canvas, 'set_range_indicators_visible'):
                canvas.set_range_indicators_visible(checked)

    def _apply_release_learning_defaults(self) -> None:
        import sys
        defaults_dir: Path | None = None
        meipass = getattr(sys, '_MEIPASS', None)
        frozen = bool(getattr(sys, 'frozen', False))
        exe_root = Path(sys.executable).parent if frozen else None

        def _is_allowed_frozen_candidate(path: Path) -> bool:
            if not frozen:
                return True
            try:
                resolved = path.resolve()
            except Exception:
                return False
            if exe_root is None:
                return False
            try:
                resolved_parent = resolved.parent.resolve()
                exe_resolved = exe_root.resolve()
                defaults_resolved = (exe_root / "defaults" / "learning").resolve()
                if resolved_parent == exe_resolved:
                    return True
                if resolved_parent == defaults_resolved:
                    return True
            except Exception:
                return False
            return False

        if getattr(sys, 'frozen', False) and meipass:
            try:
                bundle_root = Path(str(meipass))
                exe_root = Path(sys.executable).parent

                bundle_defaults = bundle_root / "defaults" / "learning"
                if bundle_defaults.exists():
                    target_defaults = exe_root / "defaults" / "learning"
                    target_defaults.mkdir(parents=True, exist_ok=True)
                    for source in bundle_defaults.glob("*.json"):
                        target = target_defaults / source.name
                        if not target.exists():
                            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

                bundle_rule_fit = bundle_root / "datasets" / "rule_fit.json"
                if bundle_rule_fit.exists():
                    target_datasets = exe_root / "datasets"
                    target_datasets.mkdir(parents=True, exist_ok=True)
                    target_rule_fit = target_datasets / "rule_fit.json"
                    if not target_rule_fit.exists():
                        target_rule_fit.write_text(bundle_rule_fit.read_text(encoding="utf-8"), encoding="utf-8")

                bundle_slots = bundle_root / "learned_profile_slots.json"
                if bundle_slots.exists():
                    target_slots = exe_root / "learned_profile_slots.json"
                    if not target_slots.exists():
                        target_slots.write_text(bundle_slots.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception as exc:
                print(f"[Learning] Failed to materialize bundled learning files: {exc}")

        # Discover profile/rule_fit candidates from two roots only:
        # base dir (EXE dir for frozen, repo dir for source) and base/defaults/learning.
        search_roots: list[Path] = []
        base_dir = Path(sys.executable).parent if frozen else Path(__file__).resolve().parent
        search_roots.append(base_dir)
        search_roots.append(base_dir / "defaults" / "learning")

        profile_candidates: list[Path] = []
        rule_fit_candidates: list[Path] = []
        seen_profiles: set[Path] = set()
        seen_rule_fits: set[Path] = set()

        for root in search_roots:
            if not root.exists() or not root.is_dir():
                continue
            for candidate in sorted(root.glob("profile*.json")):
                try:
                    resolved = candidate.resolve()
                except Exception:
                    resolved = candidate
                if resolved in seen_profiles:
                    continue
                seen_profiles.add(resolved)
                profile_candidates.append(candidate)

            for candidate in sorted(root.glob("rule_fit*.json")):
                try:
                    resolved = candidate.resolve()
                except Exception:
                    resolved = candidate
                if resolved in seen_rule_fits:
                    continue
                seen_rule_fits.add(resolved)
                rule_fit_candidates.append(candidate)

        if not profile_candidates and not rule_fit_candidates:
            print("[Learning] No release learning profile/rule_fit files found (exe dir or defaults/learning) — skipping.")
            return

        selected_profile = profile_candidates[0] if profile_candidates else None
        selected_rule_fit: Path | None = None

        if selected_profile is not None:
            try:
                payload = json.loads(selected_profile.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[Learning] Failed reading release profile {selected_profile}: {exc}")
                payload = {}

            if isinstance(payload, dict):
                learning_cfg = payload.get("learning", {})
                model_cfg = payload.get("model", {})
                if not isinstance(learning_cfg, dict):
                    learning_cfg = {}
                if not isinstance(model_cfg, dict):
                    model_cfg = {}

                bool_keys = {
                    "teaching_learning_enabled",
                    "teaching_use_fitted_rules",
                    "teaching_apply_in_circle_mode",
                    "teaching_isolation_mode",
                }
                float_keys = {
                    "teaching_learning_strength",
                    "teaching_min_confidence",
                    "teaching_no_motion_bias",
                }

                for key in bool_keys:
                    if key in learning_cfg:
                        setattr(self.config.beat, key, bool(learning_cfg.get(key)))
                for key in float_keys:
                    if key in learning_cfg:
                        try:
                            raw_value = learning_cfg.get(key)
                            if isinstance(raw_value, (int, float, str)):
                                setattr(self.config.beat, key, float(raw_value))
                        except Exception:
                            pass

                # Profile may embed rule_fit path — resolve relative to profile location
                raw_rule_fit = model_cfg.get("rule_fit") or learning_cfg.get("teaching_rule_fit_path") or payload.get("rule_fit")
                if isinstance(raw_rule_fit, str) and raw_rule_fit.strip():
                    candidate = Path(raw_rule_fit.strip())
                    if not candidate.is_absolute():
                        candidate = selected_profile.parent / candidate
                    if candidate.exists() and _is_allowed_frozen_candidate(candidate):
                        selected_rule_fit = candidate

        # Fallback: discovered rule_fit candidates
        if selected_rule_fit is None:
            selected_rule_fit = rule_fit_candidates[0] if rule_fit_candidates else None

        if selected_profile is not None:
            setattr(self.config.beat, 'teaching_profile_path', str(selected_profile))
        if selected_rule_fit is not None:
            self.config.beat.teaching_rule_fit_path = str(selected_rule_fit)

        self.config.beat.teaching_learning_enabled = True
        self.config.beat.teaching_use_fitted_rules = True

        source = "frozen" if frozen else "bundled"
        profile_label = selected_profile.name if selected_profile is not None else "(none)"
        rule_fit_label = str(selected_rule_fit) if selected_rule_fit is not None else "(none)"
        print(f"[Learning] Release defaults applied — source={source} profile={profile_label}, rule_fit={rule_fit_label}")

    def _apply_learning_config_to_mapper(self) -> None:
        mapper_live = self.stroke_mapper
        if mapper_live is None:
            return
        if hasattr(mapper_live, 'configure_learning'):
            mapper_live.configure_learning(
                enabled=bool(self.config.beat.teaching_learning_enabled),
                use_fitted_rules=bool(self.config.beat.teaching_use_fitted_rules),
                apply_in_circle_mode=bool(self.config.beat.teaching_apply_in_circle_mode),
                isolation_mode=bool(self.config.beat.teaching_isolation_mode),
                learning_strength=float(self.config.beat.teaching_learning_strength),
                min_confidence=float(self.config.beat.teaching_min_confidence),
                no_motion_bias=float(self.config.beat.teaching_no_motion_bias),
                rule_fit_path=str(self.config.beat.teaching_rule_fit_path),
            )

    def _on_learning_tune_controls(self) -> None:
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout, QPushButton

        dialog = QDialog(self)
        dialog.setWindowTitle("Tuning")
        dialog.setMinimumWidth(560)
        layout = QVBoxLayout(dialog)

        strength_slider = SliderWithLabel(
            "Advance", 0.0, 1.0,
            float(getattr(self.config.beat, 'teaching_learning_strength', 0.55) or 0.55), 2
        )
        layout.addWidget(strength_slider)

        holdback_slider = SliderWithLabel(
            "Restraint", 0.0, 1.0,
            float(getattr(self.config.beat, 'teaching_min_confidence', 0.12) or 0.12), 2
        )
        layout.addWidget(holdback_slider)

        no_motion_bias_slider = SliderWithLabel(
            "Quiet Bias", 0.25, 3.0,
            float(getattr(self.config.beat, 'teaching_no_motion_bias', 1.0) or 1.0), 2
        )
        layout.addWidget(no_motion_bias_slider)

        direction_hint = QLabel("⬅️ less         more ➡️")
        direction_hint.setStyleSheet("color: #d0d0d0; font-size: 18px; font-weight: 500;")
        direction_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(direction_hint)

        settle_hint = QLabel("Move one, wait for adjust")
        settle_hint.setStyleSheet("color: #c7c7c7; font-size: 14px;")
        settle_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(settle_hint)

        button_row = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        close_btn = QPushButton("Close")
        apply_btn.setStyleSheet("font-weight: 500;")
        close_btn.setStyleSheet("font-weight: 500;")
        button_row.addStretch()
        button_row.addWidget(apply_btn)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

        def _apply_learning_settings() -> None:
            self.config.beat.teaching_learning_strength = float(strength_slider.value())
            self.config.beat.teaching_min_confidence = float(holdback_slider.value())
            self.config.beat.teaching_no_motion_bias = float(no_motion_bias_slider.value())
            self._apply_learning_config_to_mapper()
            save_config(self.config)

        apply_btn.clicked.connect(_apply_learning_settings)
        close_btn.clicked.connect(dialog.close)
        dialog.exec()

    def _on_show_peak_indicators_menu_toggle(self, checked: bool):
        """Handle Show Peak Indicators toggle from Options menu"""
        self._on_show_peak_indicators_toggle(checked)

    def _on_effects_jitter_toggle(self, checked: bool):
        self.config.jitter.enabled = bool(checked)

    def _sync_effects_menu_actions(self):
        jitter_action = getattr(self, 'jitter_effect_action', None)
        if jitter_action is not None:
            with self._signals_blocked(jitter_action):
                jitter_action.setChecked(bool(getattr(self.config.jitter, 'enabled', True)))

    def _on_toggle_beat_band(self, checked: bool):
        """Toggle visibility of beat detection band (red) on all visualizers"""
        for canvas in [self.waveform_canvas, self.freqdb_canvas, self.fft_bin_canvas]:
            if hasattr(canvas, 'beat_band'):
                canvas.beat_band.setVisible(checked)
            if hasattr(canvas, 'beat_label'):
                canvas.beat_label.setVisible(checked)

    def _on_toggle_depth_band(self, checked: bool):
        """Toggle visibility of stroke depth band (green) on all visualizers"""
        for canvas in [self.waveform_canvas, self.freqdb_canvas, self.fft_bin_canvas]:
            if hasattr(canvas, 'depth_band'):
                canvas.depth_band.setVisible(checked)
            if hasattr(canvas, 'depth_label'):
                canvas.depth_label.setVisible(checked)

    def _on_toggle_p0_band(self, checked: bool):
        """Toggle visibility of pulse frequency band (blue) on all visualizers"""
        for canvas in [self.waveform_canvas, self.freqdb_canvas, self.fft_bin_canvas]:
            if hasattr(canvas, 'p0_band'):
                canvas.p0_band.setVisible(checked)
            if hasattr(canvas, 'pulse_label'):
                canvas.pulse_label.setVisible(checked)

    def _on_toggle_f0_band(self, checked: bool):
        """Toggle visibility of carrier frequency band (cyan) on all visualizers"""
        for canvas in [self.waveform_canvas, self.freqdb_canvas, self.fft_bin_canvas]:
            if hasattr(canvas, 'f0_band'):
                canvas.f0_band.setVisible(checked)
            if hasattr(canvas, 'carrier_label'):
                canvas.carrier_label.setVisible(checked)

    def _on_viz_menu_change(self, index: int):
        """Handle spectrum type change from Options menu"""
        # Update checkmarks
        for i, action in enumerate(self._viz_type_actions):
            action.setChecked(i == index)
        # Sync hidden combo (for preset save/load compatibility)
        self.visualizer_type_combo.setCurrentIndex(index)
        self._on_visualizer_type_change(index)
    
    def _create_position_panel(self) -> QWidget:
        """Alpha/Beta position display"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Position canvas (no rotation - fixed at 0)
        self.position_canvas = PositionCanvas(self, size=2, get_rotation=lambda: 0)
        layout.addWidget(self.position_canvas)

        # Position labels (hidden but still tracked internally)
        self.alpha_label = QLabel("α: 0.00")
        self.alpha_label.setVisible(False)
        self.beta_label = QLabel("β: 0.00")
        self.beta_label.setVisible(False)

        return widget
    
    def _create_main_controls_panel(self) -> QWidget:
        """Main stroke controls panel."""
        group = QWidget()
        layout = QHBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Temporary pin: stroke mode selector hidden during main development.
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["1: Circle"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.hide()

        tempo_widget = QWidget()
        tempo_layout = QVBoxLayout(tempo_widget)
        tempo_layout.setContentsMargins(0, 0, 0, 0)
        tempo_layout.setSpacing(4)
        self.tempo_lock_required_cb = QCheckBox()
        self.tempo_lock_required_cb.setChecked(bool(getattr(self.config.beat, 'tempo_lock_required', False)))
        self.tempo_lock_required_cb.toggled.connect(self._on_tempo_lock_required_toggle)
        self.tempo_lock_required_cb.setVisible(False)

        intensity_label = QLabel("Motion Ramp")
        intensity_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        intensity_label.setStyleSheet("color: #aaa; margin-top: 8px;")
        tempo_layout.addWidget(intensity_label)
        self.intensity_ramp_spin = SliderWithLabel(
            "Duration (hrs)",
            0.0,
            8.0,
            float(getattr(self.config.stroke, 'intensity_ramp_hours', 0.0) or 0.0),
            2,
            step=0.25,
        )
        self.intensity_ramp_spin.setMinimumWidth(180)
        self.intensity_ramp_spin.setToolTip(
            "Session intensity ramp duration (0 = disabled).\n"
            "Reactions gradually climb from gentle to full power\n"
            "over this many hours."
        )
        self.intensity_ramp_spin.valueChanged.connect(
            lambda v: (
                setattr(self.config.stroke, 'intensity_ramp_hours', float(v)),
                self._refresh_motion_ramp_visual_state(),
            )
        )

        self.intensity_ramp_target_combo = QSlider(Qt.Orientation.Horizontal)
        self.intensity_ramp_target_combo.setRange(0, 2)
        self.intensity_ramp_target_combo.setSingleStep(1)
        self.intensity_ramp_target_combo.setPageStep(1)
        self.intensity_ramp_target_combo.setTickInterval(1)
        self.intensity_ramp_target_combo.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.intensity_ramp_target_combo.setMinimumWidth(180)
        current_target = str(getattr(self.config.stroke, 'intensity_ramp_target', 'both') or 'both').strip().lower()
        if current_target not in ('size', 'speed', 'both'):
            current_target = 'both'
        target_to_idx = {'size': 0, 'speed': 1, 'both': 2}
        idx_to_target = {0: 'size', 1: 'speed', 2: 'both'}
        self.intensity_ramp_target_combo.setValue(target_to_idx.get(current_target, 2))
        self.intensity_ramp_target_combo.setToolTip(
            "Choose what the intensity timer affects:\n"
            "size = orbit size only, speed = orbit speed only, both = size + speed."
        )
        self.intensity_ramp_target_combo.valueChanged.connect(
            lambda idx: (
                setattr(self.config.stroke, 'intensity_ramp_target', idx_to_target.get(int(idx), 'both')),
                self._refresh_motion_ramp_visual_state(),
            )
        )

        ramp_target_widget = QWidget()
        ramp_target_layout = QVBoxLayout(ramp_target_widget)
        ramp_target_layout.setContentsMargins(0, 0, 0, 0)
        ramp_target_layout.setSpacing(0)
        ramp_target_layout.addWidget(self.intensity_ramp_target_combo)
        ramp_target_labels = QHBoxLayout()
        ramp_target_labels.setContentsMargins(0, 0, 0, 0)
        self.intensity_ramp_target_size_label = QLabel("Size")
        self.intensity_ramp_target_speed_label = QLabel("Speed")
        self.intensity_ramp_target_both_label = QLabel("Both")
        ramp_target_labels.addWidget(self.intensity_ramp_target_size_label, 0, Qt.AlignmentFlag.AlignLeft)
        ramp_target_labels.addWidget(self.intensity_ramp_target_speed_label, 1, Qt.AlignmentFlag.AlignHCenter)
        ramp_target_labels.addWidget(self.intensity_ramp_target_both_label, 0, Qt.AlignmentFlag.AlignRight)
        ramp_target_layout.addLayout(ramp_target_labels)
        self._intensity_ramp_target_labels = {
            'size': self.intensity_ramp_target_size_label,
            'speed': self.intensity_ramp_target_speed_label,
            'both': self.intensity_ramp_target_both_label,
        }

        tempo_layout.addWidget(self.intensity_ramp_spin)
        tempo_layout.addWidget(ramp_target_widget)
        self._refresh_motion_ramp_visual_state()
        tempo_layout.addStretch(1)

        self.fill_gate_scale_spin = SliderWithLabel(
            "Sensitivity",
            1.0,
            100.0,
            self._fill_gate_scale_to_percent(
                float(getattr(self.config.stroke, 'overall_amp_fill_required_scale', 0.5) or 0.5)
            ),
            1,
            step=1.0,
        )
        self.fill_gate_scale_spin.setMinimumWidth(260)
        self.fill_gate_scale_spin.setToolTip(
            "Sensitivity scale (1-100). "
            "Maps to the same internal fill-gate effect range as before."
        )
        self.fill_gate_scale_spin.valueChanged.connect(self._on_fill_gate_scale_change)

        self.pulse_settings_btn = QPushButton("Pulse\nSettings")
        self.pulse_settings_btn.setToolTip("Open Pulse settings popout window")
        self.pulse_settings_btn.clicked.connect(self._on_pulse_settings_popup)
        self.pulse_settings_btn.setFixedSize(76, 76)
        self.pulse_settings_btn.setStyleSheet("text-align: center;")

        self.main_silence_close_slider = SliderWithLabel(
            "Volume/Motion Threshold",
            0.0,
            2.0,
            self._silence_close_to_normalized(
                float(getattr(self.config.stroke, 'silence_close_threshold', 0.0433) or 0.0433)
            ),
            2,
            step=0.01,
        )
        self.main_silence_close_slider.setToolTip(
            "Normalized silence threshold factor (1.00 = baseline).\n"
            "Adjusts both gate close and gate open together at a fixed 4:1 close/open ratio."
        )
        self.main_silence_close_slider.setMinimumWidth(260)
        self.main_silence_close_slider.valueChanged.connect(self._on_main_silence_close_change)

        self._refresh_main_controls_value_label_colors()

        sensitivity_volume_widget = QWidget()
        sensitivity_volume_layout = QVBoxLayout(sensitivity_volume_widget)
        sensitivity_volume_layout.setContentsMargins(0, 0, 0, 0)
        sensitivity_volume_layout.setSpacing(4)
        sensitivity_volume_layout.addWidget(self.fill_gate_scale_spin)
        sensitivity_volume_layout.addWidget(self.main_silence_close_slider)

        for widget in (
            tempo_widget,
            self.pulse_settings_btn,
            sensitivity_volume_widget,
        ):
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            layout.addWidget(widget, stretch=1)

        self._refresh_motion_ramp_visual_state()
        return group

    def _fill_gate_scale_to_percent(self, scale: float) -> float:
        # Internal percent domain is still [-300, +300] for unchanged effect range.
        safe_scale = max(1e-6, float(scale))
        percent = float(-np.log2(safe_scale) * 100.0)
        ui_value = 1.0 + ((percent + 300.0) * (99.0 / 600.0))
        return float(np.clip(ui_value, 1.0, 100.0))

    def _fill_gate_percent_to_scale(self, percent: float) -> float:
        ui_value = float(np.clip(percent, 1.0, 100.0))
        percent_internal = -300.0 + ((ui_value - 1.0) * (600.0 / 99.0))
        scale = float(np.power(2.0, -percent_internal / 100.0))
        return float(np.clip(scale, 0.05, 20.0))

    def _silence_close_to_normalized(self, close_threshold: float) -> float:
        min_close = 0.001
        base_close = 0.048
        max_close = 0.300
        close_v = float(np.clip(float(close_threshold), min_close, max_close))
        if close_v <= base_close:
            return float((close_v - min_close) / max(1e-9, (base_close - min_close)))
        return float(1.0 + ((close_v - base_close) / max(1e-9, (max_close - base_close))))

    def _silence_normalized_to_close(self, normalized_value: float) -> float:
        min_close = 0.001
        base_close = 0.048
        max_close = 0.300
        norm_v = float(np.clip(float(normalized_value), 0.0, 2.0))
        if norm_v <= 1.0:
            return float(min_close + ((base_close - min_close) * norm_v))
        return float(base_close + ((max_close - base_close) * (norm_v - 1.0)))

    def _on_fill_gate_scale_change(self, pct: float) -> None:
        setattr(
            self.config.stroke,
            'overall_amp_fill_required_scale',
            self._fill_gate_percent_to_scale(float(pct)),
        )
        self._preview_fill_requirement_ghosts()

    def _on_tempo_lock_required_toggle(self, checked: bool) -> None:
        is_required = bool(checked)
        setattr(self.config.beat, 'tempo_lock_required', is_required)

        action = getattr(self, 'metronome_lock_required_action', None)
        if action is not None and action.isChecked() != is_required:
            action.blockSignals(True)
            action.setChecked(is_required)
            action.blockSignals(False)

        checkbox = getattr(self, 'tempo_lock_required_cb', None)
        if checkbox is not None and checkbox.isChecked() != is_required:
            checkbox.blockSignals(True)
            checkbox.setChecked(is_required)
            checkbox.blockSignals(False)

    def _refresh_main_controls_value_label_colors(self) -> None:
        always_white = '#fff'
        for attr_name in ('fill_gate_scale_spin', 'main_silence_close_slider'):
            slider_widget = getattr(self, attr_name, None)
            if slider_widget is not None and hasattr(slider_widget, 'value_label'):
                slider_widget.value_label.setStyleSheet(f"color: {always_white};")

    def _refresh_motion_ramp_visual_state(self) -> None:
        active_color = '#fff'
        inactive_color = '#0af'

        duration_slider = getattr(self, 'intensity_ramp_spin', None)
        if duration_slider is not None and hasattr(duration_slider, 'value_label'):
            try:
                is_running = float(duration_slider.value()) > 0.0
            except Exception:
                is_running = False
            duration_slider.value_label.setStyleSheet(
                f"color: {active_color if is_running else inactive_color};"
            )
            if hasattr(duration_slider, 'slider'):
                duration_slider.slider.setStyleSheet("")

        self._refresh_main_controls_value_label_colors()

        target_slider = getattr(self, 'intensity_ramp_target_combo', None)
        target_labels = getattr(self, '_intensity_ramp_target_labels', {}) or {}
        if target_slider is None or not target_labels:
            return

        try:
            active_idx = int(target_slider.value())
        except Exception:
            active_idx = 2

        for idx, key in enumerate(('size', 'speed', 'both')):
            label_widget = target_labels.get(key)
            if label_widget is None:
                continue
            label_widget.setStyleSheet(
                f"color: {active_color if idx == active_idx else inactive_color};"
            )

    def _on_main_silence_close_change(self, value: float) -> None:
        ratio = 4.0  # Maintain fixed hysteresis shape: close/open
        min_open = 0.001
        max_open = 0.250
        min_close = 0.001
        max_close = 0.300

        close_v = self._silence_normalized_to_close(float(value))
        open_v = float(np.clip(close_v / ratio, min_open, max_open))
        close_v = float(np.clip(open_v * ratio, min_close, max_close))

        if close_v <= open_v:
            close_v = min(max_close, open_v + 0.001)

        setattr(self.config.stroke, 'silence_threshold', open_v)
        setattr(self.config.stroke, 'silence_close_threshold', close_v)

        normalized_close = self._silence_close_to_normalized(close_v)
        if hasattr(self, 'main_silence_close_slider') and abs(normalized_close - float(value)) > 1e-9:
            self.main_silence_close_slider.blockSignals(True)
            self.main_silence_close_slider.setValue(normalized_close)
            self.main_silence_close_slider.blockSignals(False)

    def _capture_current_settings(self) -> dict:
        """Capture all current UI settings for revert functionality"""
        return {
            # Beat Detection Tab
            'freq_low': self.freq_range_slider.low(),
            'freq_high': self.freq_range_slider.high(),
            'sensitivity': self.sensitivity_slider.value(),
            'peak_floor': self.peak_floor_slider.value(),
            'peak_decay': self.peak_decay_slider.value(),
            'rise_sensitivity': self.rise_sens_slider.value(),
            'flux_multiplier': self.flux_mult_slider.value(),
            'audio_gain': self.audio_gain_slider.value(),
            'zscore_threshold': self.zscore_threshold_slider.value(),
            'silence_reset_ms': int(self.silence_reset_slider.value()),
            'detection_type': self.detection_type_combo.currentIndex(),
            
            # Tempo Tracking
            'tempo_tracking_enabled': self.tempo_tracking_checkbox.isChecked(),
            'tempo_lock_required': self.tempo_lock_required_cb.isChecked(),
            'time_sig_index': self.time_sig_combo.currentIndex(),
            'stability_threshold': self.stability_threshold_slider.value(),
            'tempo_timeout_ms': int(self.tempo_timeout_slider.value()),
            'phase_snap_weight': self.phase_snap_slider.value(),
            'acf_interval_ms': getattr(self.config.beat, 'acf_interval_ms', 250.0),
            'metronome_bpm_alpha_slow': getattr(self.config.beat, 'metronome_bpm_alpha_slow', 0.03),
            'metronome_bpm_alpha_fast': getattr(self.config.beat, 'metronome_bpm_alpha_fast', 0.22),
            'metronome_pll_window': getattr(self.config.beat, 'metronome_pll_window', 0.35),
            'metronome_pll_base_gain': getattr(self.config.beat, 'metronome_pll_base_gain', 0.09),
            'metronome_pll_conf_gain': getattr(self.config.beat, 'metronome_pll_conf_gain', 0.08),
            'tempo_fusion_min_acf_weight': getattr(self.config.beat, 'tempo_fusion_min_acf_weight', 0.20),
            'tempo_fusion_max_acf_weight': getattr(self.config.beat, 'tempo_fusion_max_acf_weight', 0.95),
            'beat_dedup_fraction': getattr(self.config.beat, 'beat_dedup_fraction', 0.22),
            'phase_accept_window_ms': getattr(self.config.beat, 'phase_accept_window_ms', 85.0),
            'phase_accept_low_conf_mult': getattr(self.config.beat, 'phase_accept_low_conf_mult', 2.0),
            'octave_target_bias_confidence_max': getattr(self.config.beat, 'octave_target_bias_confidence_max', 0.35),
            'target_bps_lock_gate_enabled': getattr(self.config.beat, 'target_bps_lock_gate_enabled', True),
            'target_bps_lock_gate_acf_conf': getattr(self.config.beat, 'target_bps_lock_gate_acf_conf', 0.40),
            'target_bps_lock_gate_downbeats': getattr(self.config.beat, 'target_bps_lock_gate_downbeats', 1),
            'aggressive_tempo_snap_enabled': getattr(self.config.beat, 'aggressive_tempo_snap_enabled', False),
            'aggressive_snap_confidence': getattr(self.config.beat, 'aggressive_snap_confidence', 0.55),
            'aggressive_snap_phase_error_ms': getattr(self.config.beat, 'aggressive_snap_phase_error_ms', 35.0),
            'aggressive_snap_min_matches': getattr(self.config.beat, 'aggressive_snap_min_matches', 1),
            'aggressive_snap_max_bpm_jump_ratio': getattr(self.config.beat, 'aggressive_snap_max_bpm_jump_ratio', 0.12),
            'metric_response_speed': getattr(self.config.auto_adjust, 'metric_response_speed', 1.0),

            # Stroke Settings Tab
            'stroke_mode': 0,
            'min_interval_ms': 150,
            'flux_depth_boost_enabled': bool(getattr(self.config.stroke, 'flux_depth_boost_enabled', False)),
            'combo_size': float(getattr(self.config.stroke, 'combo_size', 1.0)),
            'combo_depth': float(getattr(self.config.stroke, 'combo_depth', 1.0)),
            'combo_texture': float(getattr(self.config.stroke, 'combo_texture', 1.0)),
            'combo_reaction': float(getattr(self.config.stroke, 'combo_reaction', 1.0)),
            'overall_amp_fill_required_scale': float(getattr(self.config.stroke, 'overall_amp_fill_required_scale', 1.0) or 1.0),
            'flux_threshold': float(getattr(self.config.stroke, 'flux_threshold', 0.02)),
            'flux_scaling_weight': float(getattr(self.config.stroke, 'flux_scaling_weight', 1.0) or 1.0),

            # Jitter / Creep Tab
            'jitter_enabled': bool(getattr(self.config.jitter, 'enabled', True)),
            'jitter_amplitude': float(self.FIXED_JITTER_AMPLITUDE),
            'jitter_intensity': float(self.FIXED_JITTER_INTENSITY),
            'creep_enabled': bool(getattr(self.config.creep, 'enabled', True)),
            'creep_speed': float(self.FIXED_CREEP_SPEED),

            # Axis Weights Tab
            'alpha_weight': float(self.FIXED_AXIS_WEIGHT),
            'beta_weight': float(self.FIXED_AXIS_WEIGHT),

            # Pulse Freq Tab
            'pulse_freq_low': self.pulse_freq_range_slider.low(),
            'pulse_freq_high': self.pulse_freq_range_slider.high(),
            'tcode_min': int(self.tcode_freq_range_slider.low()),
            'tcode_max': int(self.tcode_freq_range_slider.high()),
            'freq_weight': self.freq_weight_slider.value(),
        }
    
    def _revert_preset(self):
        """Revert to settings from before the last preset was loaded"""
        if self._revert_settings is None:
            return
        
        # Restore all settings from _revert_settings (same logic as _load_freq_preset)
        preset_data = self._revert_settings
        
        # Beat Detection Tab
        self.freq_range_slider.setLow(preset_data['freq_low'])
        self.freq_range_slider.setHigh(preset_data['freq_high'])
        self.sensitivity_slider.setValue(preset_data['sensitivity'])
        self.peak_floor_slider.setValue(preset_data['peak_floor'])
        self.peak_decay_slider.setValue(preset_data['peak_decay'])
        self.rise_sens_slider.setValue(preset_data['rise_sensitivity'])
        self.flux_mult_slider.setValue(preset_data['flux_multiplier'])
        self.audio_gain_slider.setValue(preset_data['audio_gain'])
        if 'zscore_threshold' in preset_data:
            self.zscore_threshold_slider.setValue(preset_data['zscore_threshold'])
            self._on_zscore_threshold_change(preset_data['zscore_threshold'])
        self.silence_reset_slider.setValue(preset_data['silence_reset_ms'])
        self.detection_type_combo.setCurrentIndex(preset_data['detection_type'])
        
        # Tempo Tracking
        self.tempo_tracking_checkbox.setChecked(preset_data['tempo_tracking_enabled'])
        self._on_tempo_tracking_toggle(2 if preset_data['tempo_tracking_enabled'] else 0)
        if 'tempo_lock_required' in preset_data:
            self.tempo_lock_required_cb.setChecked(bool(preset_data['tempo_lock_required']))
        self.time_sig_combo.setCurrentIndex(preset_data['time_sig_index'])
        self._on_time_sig_change(preset_data['time_sig_index'])
        self.stability_threshold_slider.setValue(preset_data['stability_threshold'])
        self._on_stability_threshold_change(preset_data['stability_threshold'])
        self.tempo_timeout_slider.setValue(preset_data['tempo_timeout_ms'])
        self._on_tempo_timeout_change(preset_data['tempo_timeout_ms'])
        self.phase_snap_slider.setValue(preset_data['phase_snap_weight'])
        self._on_phase_snap_change(preset_data['phase_snap_weight'])
        if 'acf_interval_ms' in preset_data:
            self._on_acf_interval_change(int(preset_data['acf_interval_ms']))
        if 'metronome_bpm_alpha_slow' in preset_data:
            self._on_metronome_bpm_alpha_slow_change(preset_data['metronome_bpm_alpha_slow'])
        if 'metronome_bpm_alpha_fast' in preset_data:
            self._on_metronome_bpm_alpha_fast_change(preset_data['metronome_bpm_alpha_fast'])
        if 'metronome_pll_window' in preset_data:
            self._on_metronome_pll_window_change(preset_data['metronome_pll_window'])
        if 'metronome_pll_base_gain' in preset_data:
            self._on_metronome_pll_base_gain_change(preset_data['metronome_pll_base_gain'])
        if 'metronome_pll_conf_gain' in preset_data:
            self._on_metronome_pll_conf_gain_change(preset_data['metronome_pll_conf_gain'])
        if 'tempo_fusion_min_acf_weight' in preset_data:
            self._on_tempo_fusion_min_acf_weight_change(preset_data['tempo_fusion_min_acf_weight'])
        if 'tempo_fusion_max_acf_weight' in preset_data:
            self._on_tempo_fusion_max_acf_weight_change(preset_data['tempo_fusion_max_acf_weight'])
        if 'beat_dedup_fraction' in preset_data:
            self._on_beat_dedup_fraction_change(preset_data['beat_dedup_fraction'])
        if 'phase_accept_window_ms' in preset_data:
            self._on_phase_accept_window_ms_change(preset_data['phase_accept_window_ms'])
        if 'phase_accept_low_conf_mult' in preset_data:
            self._on_phase_accept_low_conf_mult_change(preset_data['phase_accept_low_conf_mult'])
        if 'octave_target_bias_confidence_max' in preset_data:
            self._on_octave_target_bias_confidence_max_change(preset_data['octave_target_bias_confidence_max'])
        if 'target_bps_lock_gate_enabled' in preset_data:
            self._on_target_bps_lock_gate_toggle(bool(preset_data['target_bps_lock_gate_enabled']))
        if 'target_bps_lock_gate_acf_conf' in preset_data:
            self._on_target_bps_lock_gate_acf_conf_change(preset_data['target_bps_lock_gate_acf_conf'])
        if 'target_bps_lock_gate_downbeats' in preset_data:
            self._on_target_bps_lock_gate_downbeats_change(int(preset_data['target_bps_lock_gate_downbeats']))
        if 'aggressive_tempo_snap_enabled' in preset_data:
            self._on_aggressive_tempo_snap_toggle(bool(preset_data['aggressive_tempo_snap_enabled']))
        if 'aggressive_snap_confidence' in preset_data:
            self._on_aggressive_snap_confidence_change(preset_data['aggressive_snap_confidence'])
        if 'aggressive_snap_phase_error_ms' in preset_data:
            self._on_aggressive_snap_phase_error_ms_change(preset_data['aggressive_snap_phase_error_ms'])
        if 'aggressive_snap_min_matches' in preset_data:
            self._on_aggressive_snap_min_matches_change(int(preset_data['aggressive_snap_min_matches']))
        if 'aggressive_snap_max_bpm_jump_ratio' in preset_data:
            self._on_aggressive_snap_max_jump_change(preset_data['aggressive_snap_max_bpm_jump_ratio'])
        if 'metric_response_speed' in preset_data:
            self._on_metric_response_speed_change(preset_data['metric_response_speed'])
        
        # Stroke Settings Tab
        self.mode_combo.setCurrentIndex(0)
        self._on_mode_change(0)
        self.config.stroke.min_interval_ms = 150
        if 'overall_amp_fill_required_scale' in preset_data:
            self.fill_gate_scale_spin.setValue(
                self._fill_gate_scale_to_percent(float(preset_data['overall_amp_fill_required_scale']))
            )
        self.config.stroke.flux_threshold = float(preset_data['flux_threshold'])
        advanced_flux_slider = getattr(self, '_advanced_flux_threshold_slider', None)
        if advanced_flux_slider is not None:
            advanced_flux_slider.setValue(self.config.stroke.flux_threshold)
        self.config.stroke.flux_scaling_weight = float(preset_data['flux_scaling_weight'])
        advanced_flux_scaling_slider = getattr(self, '_advanced_flux_scaling_slider', None)
        if advanced_flux_scaling_slider is not None:
            advanced_flux_scaling_slider.setValue(self.config.stroke.flux_scaling_weight)
        
        # Jitter / Creep Tab
        self.config.jitter.enabled = bool(preset_data.get('jitter_enabled', getattr(self.config.jitter, 'enabled', True)))
        self.config.creep.enabled = False
        self._enforce_fixed_effect_axis_values()
        self._sync_effects_menu_actions()
        
        # Pulse Freq Tab
        self.pulse_freq_range_slider.setLow(preset_data['pulse_freq_low'])
        self.pulse_freq_range_slider.setHigh(preset_data['pulse_freq_high'])
        # Support both new (tcode_min) and old (tcode_freq_min) preset keys
        p0_tcode_min = preset_data.get('tcode_min', preset_data.get('tcode_freq_min', 2010))
        p0_tcode_max = preset_data.get('tcode_max', preset_data.get('tcode_freq_max', 7035))
        # Backward compat: old presets stored Hz values (typically < 200), convert to TCode
        if p0_tcode_min < 200:
            p0_tcode_min = int(p0_tcode_min * 67)
        if p0_tcode_max < 200:
            p0_tcode_max = int(p0_tcode_max * 67)
        self.tcode_freq_range_slider.setLow(p0_tcode_min)
        self.tcode_freq_range_slider.setHigh(p0_tcode_max)
        self.freq_weight_slider.setValue(preset_data['freq_weight'])
        
        # Sync config
        self.config.stroke.mode = StrokeMode.SIMPLE_CIRCLE
        
        # Deactivate all preset buttons
        for btn in getattr(self, 'preset_buttons', []):
            btn.set_active(False)
        
        # Disable revert button (already reverted)
        revert_btn = getattr(self, 'revert_btn', None)
        if revert_btn is not None:
            revert_btn.setEnabled(False)
        self._revert_settings = None
        
        print("[Config] Reverted to previous settings")

    def _get_thin_scrollbar_style(self) -> str:
        """Return thin minimal scrollbar CSS for NoWheelScrollArea tabs"""
        return """
            QScrollBar:vertical {
                background-color: transparent;
                width: 4px;
                border: none;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(100, 100, 100, 0.5);
                border-radius: 2px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: rgba(150, 150, 150, 0.7);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """

    def _create_tcode_freq_tab(self) -> QWidget:
        """Combined Pulse (P0) and Carrier (F0) frequency controls"""
        def _effective_limits(raw_min: float, raw_max: float, default_min: float, default_max: float) -> tuple[float, float]:
            lo = float(raw_min)
            hi = float(raw_max)
            if hi <= lo:
                lo = float(default_min)
                hi = float(default_max)
            return lo, hi

        def _unit_to_tcode(value: float, unit_min: float, unit_max: float) -> int:
            span = max(1e-9, float(unit_max) - float(unit_min))
            norm = (float(value) - float(unit_min)) / span
            return int(max(0, min(9999, round(norm * 9999.0))))

        def _tcode_to_unit(value: int, unit_min: float, unit_max: float) -> float:
            tcode = max(0, min(9999, int(value)))
            return float(unit_min) + (float(tcode) / 9999.0) * (float(unit_max) - float(unit_min))

        class _RangeProxy:
            def __init__(self, low_get, high_get, low_set, high_set, parent_get, blockers):
                self._low_get = low_get
                self._high_get = high_get
                self._low_set = low_set
                self._high_set = high_set
                self._parent_get = parent_get
                self._blockers = list(blockers)

            def low(self):
                return self._low_get()

            def high(self):
                return self._high_get()

            def setLow(self, value):
                self._low_set(value)

            def setHigh(self, value):
                self._high_set(value)

            def parent(self):
                return self._parent_get()

            def blockSignals(self, block: bool):
                for widget in self._blockers:
                    if widget is not None:
                        widget.blockSignals(block)

        scroll_area = NoWheelScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(self._get_thin_scrollbar_style())

        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ===== PULSE FREQUENCY =====
        pulse_group = CollapsibleGroupBox("Pulse Frequency - blue overlay on spectrum", collapsed=False)
        pulse_layout = QVBoxLayout(pulse_group)

        dl = self.config.device_limits

        p0_monitor_row = QHBoxLayout()
        p0_monitor_row.addWidget(QLabel("Monitor bass min:"))
        self.p0_monitor_min_spin = QSpinBox()
        self.p0_monitor_min_spin.setRange(20, 500)
        self.p0_monitor_min_spin.setSingleStep(5)
        self.p0_monitor_min_spin.setValue(int(max(20.0, min(500.0, float(self.config.pulse_freq.monitor_freq_min)))))
        self.p0_monitor_min_spin.setSuffix(" Hz")
        p0_monitor_row.addWidget(self.p0_monitor_min_spin)

        p0_monitor_row.addWidget(QLabel("max:"))
        self.p0_monitor_max_spin = QSpinBox()
        self.p0_monitor_max_spin.setRange(20, 500)
        self.p0_monitor_max_spin.setSingleStep(5)
        self.p0_monitor_max_spin.setValue(int(max(20.0, min(500.0, float(self.config.pulse_freq.monitor_freq_max)))))
        self.p0_monitor_max_spin.setSuffix(" Hz")
        p0_monitor_row.addWidget(self.p0_monitor_max_spin)
        p0_monitor_row.addStretch()
        pulse_layout.addLayout(p0_monitor_row)

        p0_out_min_hz, p0_out_max_hz = _effective_limits(dl.p0_freq_min, dl.p0_freq_max, 1.0, 100.0)
        p0_sent_row = QHBoxLayout()
        p0_sent_row.addWidget(QLabel("Sent min:"))
        self.p0_sent_min_spin = QDoubleSpinBox()
        self.p0_sent_min_spin.setRange(float(p0_out_min_hz), float(p0_out_max_hz))
        self.p0_sent_min_spin.setSingleStep(1.0)
        self.p0_sent_min_spin.setDecimals(1)
        self.p0_sent_min_spin.setValue(float(p0_out_min_hz))
        self.p0_sent_min_spin.setSuffix(" Hz")
        p0_sent_row.addWidget(self.p0_sent_min_spin)

        p0_sent_row.addWidget(QLabel("max:"))
        self.p0_sent_max_spin = QDoubleSpinBox()
        self.p0_sent_max_spin.setRange(float(p0_out_min_hz), float(p0_out_max_hz))
        self.p0_sent_max_spin.setSingleStep(1.0)
        self.p0_sent_max_spin.setDecimals(1)
        self.p0_sent_max_spin.setValue(float(p0_out_max_hz))
        self.p0_sent_max_spin.setSuffix(" Hz")
        p0_sent_row.addWidget(self.p0_sent_max_spin)
        p0_sent_row.addStretch()
        pulse_layout.addLayout(p0_sent_row)

        self.freq_weight_slider = SliderWithLabel("Frequency Weight", 0.0, 5.0, 1.0, 2)
        pulse_layout.addWidget(self.freq_weight_slider)

        pulse_mode_layout = QHBoxLayout()
        pulse_mode_layout.addWidget(QLabel("Mode:"))
        self.pulse_mode_combo = QComboBox()
        self.pulse_mode_combo.addItems(["Hz (dominant freq)", "Speed (dot movement)", "Band (sub_bass)"])
        self.pulse_mode_combo.setCurrentIndex(0)
        pulse_mode_layout.addWidget(self.pulse_mode_combo)
        self.pulse_invert_checkbox = QCheckBox("Invert")
        self.pulse_invert_checkbox.setChecked(False)
        pulse_mode_layout.addWidget(self.pulse_invert_checkbox)
        self.pulse_enabled_checkbox = QCheckBox("Enable")
        self.pulse_enabled_checkbox.setChecked(bool(getattr(self, '_cached_p0_enabled', False)))
        pulse_mode_layout.addWidget(self.pulse_enabled_checkbox)
        pulse_mode_layout.addStretch()
        pulse_layout.addLayout(pulse_mode_layout)

        layout.addWidget(pulse_group)

        # ===== CARRIER FREQUENCY =====
        carrier_group = CollapsibleGroupBox("Carrier Frequency", collapsed=False)
        carrier_layout = QVBoxLayout(carrier_group)

        f0_monitor_row = QHBoxLayout()
        f0_monitor_row.addWidget(QLabel("Monitor bass min:"))
        self.f0_monitor_min_spin = QSpinBox()
        self.f0_monitor_min_spin.setRange(20, 500)
        self.f0_monitor_min_spin.setSingleStep(5)
        self.f0_monitor_min_spin.setValue(int(max(20.0, min(500.0, float(self.config.carrier_freq.monitor_freq_min)))))
        self.f0_monitor_min_spin.setSuffix(" Hz")
        f0_monitor_row.addWidget(self.f0_monitor_min_spin)

        f0_monitor_row.addWidget(QLabel("max:"))
        self.f0_monitor_max_spin = QSpinBox()
        self.f0_monitor_max_spin.setRange(20, 500)
        self.f0_monitor_max_spin.setSingleStep(5)
        self.f0_monitor_max_spin.setValue(int(max(20.0, min(500.0, float(self.config.carrier_freq.monitor_freq_max)))))
        self.f0_monitor_max_spin.setSuffix(" Hz")
        f0_monitor_row.addWidget(self.f0_monitor_max_spin)
        f0_monitor_row.addStretch()
        carrier_layout.addLayout(f0_monitor_row)

        c0_out_min_hz, c0_out_max_hz = _effective_limits(dl.c0_freq_min, dl.c0_freq_max, 500.0, 1500.0)
        f0_sent_row = QHBoxLayout()
        f0_sent_row.addWidget(QLabel("Sent min:"))
        self.f0_sent_min_spin = QDoubleSpinBox()
        self.f0_sent_min_spin.setRange(float(c0_out_min_hz), float(c0_out_max_hz))
        self.f0_sent_min_spin.setSingleStep(1.0)
        self.f0_sent_min_spin.setDecimals(1)
        self.f0_sent_min_spin.setValue(float(c0_out_min_hz))
        self.f0_sent_min_spin.setSuffix(" Hz")
        f0_sent_row.addWidget(self.f0_sent_min_spin)

        f0_sent_row.addWidget(QLabel("max:"))
        self.f0_sent_max_spin = QDoubleSpinBox()
        self.f0_sent_max_spin.setRange(float(c0_out_min_hz), float(c0_out_max_hz))
        self.f0_sent_max_spin.setSingleStep(1.0)
        self.f0_sent_max_spin.setDecimals(1)
        self.f0_sent_max_spin.setValue(float(c0_out_max_hz))
        self.f0_sent_max_spin.setSuffix(" Hz")
        f0_sent_row.addWidget(self.f0_sent_max_spin)
        f0_sent_row.addStretch()
        carrier_layout.addLayout(f0_sent_row)

        self.f0_weight_slider = SliderWithLabel("Frequency Weight", 0.0, 5.0, 1.0, 2)
        carrier_layout.addWidget(self.f0_weight_slider)

        f0_mode_layout = QHBoxLayout()
        f0_mode_layout.addWidget(QLabel("Mode:"))
        self.f0_mode_combo = QComboBox()
        self.f0_mode_combo.addItems(["Hz (dominant freq)", "Speed (dot movement)", "Band (mid)"])
        self.f0_mode_combo.setCurrentIndex(0)
        f0_mode_layout.addWidget(self.f0_mode_combo)
        self.f0_invert_checkbox = QCheckBox("Invert")
        self.f0_invert_checkbox.setChecked(False)
        f0_mode_layout.addWidget(self.f0_invert_checkbox)
        self.f0_enabled_checkbox = QCheckBox("Enable")
        self.f0_enabled_checkbox.setChecked(bool(getattr(self, '_cached_f0_enabled', False)))
        f0_mode_layout.addWidget(self.f0_enabled_checkbox)
        f0_mode_layout.addStretch()
        carrier_layout.addLayout(f0_mode_layout)

        layout.addWidget(carrier_group)

        # ===== PULSE WIDTH =====
        p1_group = CollapsibleGroupBox("Pulse Width — higher = stronger, smoother", collapsed=True)
        p1_layout = QVBoxLayout(p1_group)

        p1_monitor_row = QHBoxLayout()
        p1_monitor_row.addWidget(QLabel("Monitor bass min:"))
        self.p1_monitor_min_spin = QSpinBox()
        self.p1_monitor_min_spin.setRange(20, 500)
        self.p1_monitor_min_spin.setSingleStep(5)
        self.p1_monitor_min_spin.setValue(int(max(20.0, min(500.0, float(self.config.pulse_width.monitor_freq_min)))))
        self.p1_monitor_min_spin.setSuffix(" Hz")
        p1_monitor_row.addWidget(self.p1_monitor_min_spin)

        p1_monitor_row.addWidget(QLabel("max:"))
        self.p1_monitor_max_spin = QSpinBox()
        self.p1_monitor_max_spin.setRange(20, 500)
        self.p1_monitor_max_spin.setSingleStep(5)
        self.p1_monitor_max_spin.setValue(int(max(20.0, min(500.0, float(self.config.pulse_width.monitor_freq_max)))))
        self.p1_monitor_max_spin.setSuffix(" Hz")
        p1_monitor_row.addWidget(self.p1_monitor_max_spin)
        p1_monitor_row.addStretch()
        p1_layout.addLayout(p1_monitor_row)

        p1_out_min, p1_out_max = _effective_limits(dl.p1_cycles_min, dl.p1_cycles_max, 0.0, 20.0)
        p1_sent_row = QHBoxLayout()
        p1_sent_row.addWidget(QLabel("Sent min:"))
        self.p1_sent_min_spin = QDoubleSpinBox()
        self.p1_sent_min_spin.setRange(float(p1_out_min), float(p1_out_max))
        self.p1_sent_min_spin.setSingleStep(0.1)
        self.p1_sent_min_spin.setDecimals(1)
        self.p1_sent_min_spin.setValue(float(p1_out_min))
        self.p1_sent_min_spin.setSuffix(" cyc")
        p1_sent_row.addWidget(self.p1_sent_min_spin)

        p1_sent_row.addWidget(QLabel("max:"))
        self.p1_sent_max_spin = QDoubleSpinBox()
        self.p1_sent_max_spin.setRange(float(p1_out_min), float(p1_out_max))
        self.p1_sent_max_spin.setSingleStep(0.1)
        self.p1_sent_max_spin.setDecimals(1)
        self.p1_sent_max_spin.setValue(float(p1_out_max))
        self.p1_sent_max_spin.setSuffix(" cyc")
        p1_sent_row.addWidget(self.p1_sent_max_spin)
        p1_sent_row.addStretch()
        p1_layout.addLayout(p1_sent_row)

        self.p1_weight_slider = SliderWithLabel("Weight", 0.0, 5.0, 1.0, 2)
        p1_layout.addWidget(self.p1_weight_slider)

        p1_mode_layout = QHBoxLayout()
        p1_mode_layout.addWidget(QLabel("Mode:"))
        self.p1_mode_combo = QComboBox()
        self.p1_mode_combo.addItems(["Volume (RMS energy)", "Hz (dominant freq)", "Speed (dot movement)"])
        self.p1_mode_combo.setCurrentIndex(0)
        p1_mode_layout.addWidget(self.p1_mode_combo)
        self.p1_invert_checkbox = QCheckBox("Invert")
        self.p1_invert_checkbox.setChecked(False)
        p1_mode_layout.addWidget(self.p1_invert_checkbox)
        self.p1_enabled_checkbox = QCheckBox("Enable")
        self.p1_enabled_checkbox.setChecked(bool(getattr(self, '_cached_p1_enabled', False)))
        p1_mode_layout.addWidget(self.p1_enabled_checkbox)
        p1_mode_layout.addStretch()
        p1_layout.addLayout(p1_mode_layout)

        layout.addWidget(p1_group)

        # ===== RISE TIME =====
        p3_group = CollapsibleGroupBox("Rise Time — higher = smoother, gentler", collapsed=True)
        p3_layout = QVBoxLayout(p3_group)

        p3_monitor_row = QHBoxLayout()
        p3_monitor_row.addWidget(QLabel("Monitor bass min:"))
        self.p3_monitor_min_spin = QSpinBox()
        self.p3_monitor_min_spin.setRange(20, 500)
        self.p3_monitor_min_spin.setSingleStep(5)
        self.p3_monitor_min_spin.setValue(int(max(20.0, min(500.0, float(self.config.rise_time.monitor_freq_min)))))
        self.p3_monitor_min_spin.setSuffix(" Hz")
        p3_monitor_row.addWidget(self.p3_monitor_min_spin)

        p3_monitor_row.addWidget(QLabel("max:"))
        self.p3_monitor_max_spin = QSpinBox()
        self.p3_monitor_max_spin.setRange(20, 500)
        self.p3_monitor_max_spin.setSingleStep(5)
        self.p3_monitor_max_spin.setValue(int(max(20.0, min(500.0, float(self.config.rise_time.monitor_freq_max)))))
        self.p3_monitor_max_spin.setSuffix(" Hz")
        p3_monitor_row.addWidget(self.p3_monitor_max_spin)
        p3_monitor_row.addStretch()
        p3_layout.addLayout(p3_monitor_row)

        p3_out_min, p3_out_max = _effective_limits(dl.p3_cycles_min, dl.p3_cycles_max, 0.0, 20.0)
        p3_sent_row = QHBoxLayout()
        p3_sent_row.addWidget(QLabel("Sent min:"))
        self.p3_sent_min_spin = QDoubleSpinBox()
        self.p3_sent_min_spin.setRange(float(p3_out_min), float(p3_out_max))
        self.p3_sent_min_spin.setSingleStep(0.1)
        self.p3_sent_min_spin.setDecimals(1)
        self.p3_sent_min_spin.setValue(float(p3_out_min))
        self.p3_sent_min_spin.setSuffix(" cyc")
        p3_sent_row.addWidget(self.p3_sent_min_spin)

        p3_sent_row.addWidget(QLabel("max:"))
        self.p3_sent_max_spin = QDoubleSpinBox()
        self.p3_sent_max_spin.setRange(float(p3_out_min), float(p3_out_max))
        self.p3_sent_max_spin.setSingleStep(0.1)
        self.p3_sent_max_spin.setDecimals(1)
        self.p3_sent_max_spin.setValue(float(p3_out_max))
        self.p3_sent_max_spin.setSuffix(" cyc")
        p3_sent_row.addWidget(self.p3_sent_max_spin)
        p3_sent_row.addStretch()
        p3_layout.addLayout(p3_sent_row)

        self.p3_weight_slider = SliderWithLabel("Weight", 0.0, 5.0, 1.0, 2)
        p3_layout.addWidget(self.p3_weight_slider)

        p3_mode_layout = QHBoxLayout()
        p3_mode_layout.addWidget(QLabel("Mode:"))
        self.p3_mode_combo = QComboBox()
        self.p3_mode_combo.addItems(["Brightness (spectral centroid)", "Hz (dominant freq)", "Speed (dot movement)"])
        self.p3_mode_combo.setCurrentIndex(0)
        p3_mode_layout.addWidget(self.p3_mode_combo)
        self.p3_invert_checkbox = QCheckBox("Invert")
        self.p3_invert_checkbox.setChecked(False)
        p3_mode_layout.addWidget(self.p3_invert_checkbox)
        self.p3_enabled_checkbox = QCheckBox("Enable")
        self.p3_enabled_checkbox.setChecked(bool(getattr(self, '_cached_p3_enabled', False)))
        p3_mode_layout.addWidget(self.p3_enabled_checkbox)
        p3_mode_layout.addStretch()
        p3_layout.addLayout(p3_mode_layout)

        layout.addWidget(p3_group)

        # Compatibility proxies: preserve existing low()/high()/setLow()/setHigh() call sites.
        self.pulse_freq_range_slider = _RangeProxy(
            low_get=lambda: min(float(self.p0_monitor_min_spin.value()), float(self.p0_monitor_max_spin.value())),
            high_get=lambda: max(float(self.p0_monitor_min_spin.value()), float(self.p0_monitor_max_spin.value())),
            low_set=lambda v: self.p0_monitor_min_spin.setValue(int(v)),
            high_set=lambda v: self.p0_monitor_max_spin.setValue(int(v)),
            parent_get=lambda: self.p0_monitor_min_spin.parent(),
            blockers=(self.p0_monitor_min_spin, self.p0_monitor_max_spin),
        )
        self.f0_freq_range_slider = _RangeProxy(
            low_get=lambda: min(float(self.f0_monitor_min_spin.value()), float(self.f0_monitor_max_spin.value())),
            high_get=lambda: max(float(self.f0_monitor_min_spin.value()), float(self.f0_monitor_max_spin.value())),
            low_set=lambda v: self.f0_monitor_min_spin.setValue(int(v)),
            high_set=lambda v: self.f0_monitor_max_spin.setValue(int(v)),
            parent_get=lambda: self.f0_monitor_min_spin.parent(),
            blockers=(self.f0_monitor_min_spin, self.f0_monitor_max_spin),
        )
        self.p1_monitor_range_slider = _RangeProxy(
            low_get=lambda: min(float(self.p1_monitor_min_spin.value()), float(self.p1_monitor_max_spin.value())),
            high_get=lambda: max(float(self.p1_monitor_min_spin.value()), float(self.p1_monitor_max_spin.value())),
            low_set=lambda v: self.p1_monitor_min_spin.setValue(int(v)),
            high_set=lambda v: self.p1_monitor_max_spin.setValue(int(v)),
            parent_get=lambda: self.p1_monitor_min_spin.parent(),
            blockers=(self.p1_monitor_min_spin, self.p1_monitor_max_spin),
        )
        self.p3_monitor_range_slider = _RangeProxy(
            low_get=lambda: min(float(self.p3_monitor_min_spin.value()), float(self.p3_monitor_max_spin.value())),
            high_get=lambda: max(float(self.p3_monitor_min_spin.value()), float(self.p3_monitor_max_spin.value())),
            low_set=lambda v: self.p3_monitor_min_spin.setValue(int(v)),
            high_set=lambda v: self.p3_monitor_max_spin.setValue(int(v)),
            parent_get=lambda: self.p3_monitor_min_spin.parent(),
            blockers=(self.p3_monitor_min_spin, self.p3_monitor_max_spin),
        )

        self.tcode_freq_range_slider = _RangeProxy(
            low_get=lambda: _unit_to_tcode(
                min(float(self.p0_sent_min_spin.value()), float(self.p0_sent_max_spin.value())),
                *_effective_limits(self.config.device_limits.p0_freq_min, self.config.device_limits.p0_freq_max, 1.0, 100.0),
            ),
            high_get=lambda: _unit_to_tcode(
                max(float(self.p0_sent_min_spin.value()), float(self.p0_sent_max_spin.value())),
                *_effective_limits(self.config.device_limits.p0_freq_min, self.config.device_limits.p0_freq_max, 1.0, 100.0),
            ),
            low_set=lambda v: self.p0_sent_min_spin.setValue(_tcode_to_unit(
                int(v), *_effective_limits(self.config.device_limits.p0_freq_min, self.config.device_limits.p0_freq_max, 1.0, 100.0)
            )),
            high_set=lambda v: self.p0_sent_max_spin.setValue(_tcode_to_unit(
                int(v), *_effective_limits(self.config.device_limits.p0_freq_min, self.config.device_limits.p0_freq_max, 1.0, 100.0)
            )),
            parent_get=lambda: self.p0_sent_min_spin.parent(),
            blockers=(self.p0_sent_min_spin, self.p0_sent_max_spin),
        )
        self.f0_tcode_range_slider = _RangeProxy(
            low_get=lambda: _unit_to_tcode(
                min(float(self.f0_sent_min_spin.value()), float(self.f0_sent_max_spin.value())),
                *_effective_limits(self.config.device_limits.c0_freq_min, self.config.device_limits.c0_freq_max, 500.0, 1500.0),
            ),
            high_get=lambda: _unit_to_tcode(
                max(float(self.f0_sent_min_spin.value()), float(self.f0_sent_max_spin.value())),
                *_effective_limits(self.config.device_limits.c0_freq_min, self.config.device_limits.c0_freq_max, 500.0, 1500.0),
            ),
            low_set=lambda v: self.f0_sent_min_spin.setValue(_tcode_to_unit(
                int(v), *_effective_limits(self.config.device_limits.c0_freq_min, self.config.device_limits.c0_freq_max, 500.0, 1500.0)
            )),
            high_set=lambda v: self.f0_sent_max_spin.setValue(_tcode_to_unit(
                int(v), *_effective_limits(self.config.device_limits.c0_freq_min, self.config.device_limits.c0_freq_max, 500.0, 1500.0)
            )),
            parent_get=lambda: self.f0_sent_min_spin.parent(),
            blockers=(self.f0_sent_min_spin, self.f0_sent_max_spin),
        )
        self.p1_tcode_range_slider = _RangeProxy(
            low_get=lambda: _unit_to_tcode(
                min(float(self.p1_sent_min_spin.value()), float(self.p1_sent_max_spin.value())),
                *_effective_limits(self.config.device_limits.p1_cycles_min, self.config.device_limits.p1_cycles_max, 0.0, 20.0),
            ),
            high_get=lambda: _unit_to_tcode(
                max(float(self.p1_sent_min_spin.value()), float(self.p1_sent_max_spin.value())),
                *_effective_limits(self.config.device_limits.p1_cycles_min, self.config.device_limits.p1_cycles_max, 0.0, 20.0),
            ),
            low_set=lambda v: self.p1_sent_min_spin.setValue(_tcode_to_unit(
                int(v), *_effective_limits(self.config.device_limits.p1_cycles_min, self.config.device_limits.p1_cycles_max, 0.0, 20.0)
            )),
            high_set=lambda v: self.p1_sent_max_spin.setValue(_tcode_to_unit(
                int(v), *_effective_limits(self.config.device_limits.p1_cycles_min, self.config.device_limits.p1_cycles_max, 0.0, 20.0)
            )),
            parent_get=lambda: self.p1_sent_min_spin.parent(),
            blockers=(self.p1_sent_min_spin, self.p1_sent_max_spin),
        )
        self.p3_tcode_range_slider = _RangeProxy(
            low_get=lambda: _unit_to_tcode(
                min(float(self.p3_sent_min_spin.value()), float(self.p3_sent_max_spin.value())),
                *_effective_limits(self.config.device_limits.p3_cycles_min, self.config.device_limits.p3_cycles_max, 0.0, 20.0),
            ),
            high_get=lambda: _unit_to_tcode(
                max(float(self.p3_sent_min_spin.value()), float(self.p3_sent_max_spin.value())),
                *_effective_limits(self.config.device_limits.p3_cycles_min, self.config.device_limits.p3_cycles_max, 0.0, 20.0),
            ),
            low_set=lambda v: self.p3_sent_min_spin.setValue(_tcode_to_unit(
                int(v), *_effective_limits(self.config.device_limits.p3_cycles_min, self.config.device_limits.p3_cycles_max, 0.0, 20.0)
            )),
            high_set=lambda v: self.p3_sent_max_spin.setValue(_tcode_to_unit(
                int(v), *_effective_limits(self.config.device_limits.p3_cycles_min, self.config.device_limits.p3_cycles_max, 0.0, 20.0)
            )),
            parent_get=lambda: self.p3_sent_min_spin.parent(),
            blockers=(self.p3_sent_min_spin, self.p3_sent_max_spin),
        )

        # Keep callbacks/overlays in sync when spinboxes change.
        self.p0_monitor_min_spin.valueChanged.connect(lambda *_: self._on_p0_band_change())
        self.p0_monitor_max_spin.valueChanged.connect(lambda *_: self._on_p0_band_change())
        self.f0_monitor_min_spin.valueChanged.connect(lambda *_: self._on_f0_band_change())
        self.f0_monitor_max_spin.valueChanged.connect(lambda *_: self._on_f0_band_change())
        self.p1_monitor_min_spin.valueChanged.connect(lambda *_: self._on_p1_band_change())
        self.p1_monitor_max_spin.valueChanged.connect(lambda *_: self._on_p1_band_change())
        self.p3_monitor_min_spin.valueChanged.connect(lambda *_: self._on_p3_band_change())
        self.p3_monitor_max_spin.valueChanged.connect(lambda *_: self._on_p3_band_change())

        layout.addStretch()
        scroll_area.setWidget(widget)
        return scroll_area

    def _on_butterworth_toggle(self, state: int):
        """Toggle Butterworth filter (requires restart)"""
        enabled = state == 2
        self.config.audio.use_butterworth = enabled
        print(f"[Config] Butterworth filter {'enabled' if enabled else 'disabled'} (restart required)")

    def _on_zscore_threshold_change(self, value: float):
        """Update z-score threshold on all multi-band detectors at runtime."""
        if hasattr(self, 'audio_engine') and self.audio_engine is not None:
            self.audio_engine.set_zscore_threshold(value)
        print(f"[Config] Z-score threshold set to {value:.2f}")
    
    def _on_fft_size_change(self, index: int):
        """Update FFT size setting (requires restart to take effect)"""
        fft_sizes = [512, 1024, 2048, 4096, 8192]
        self.config.audio.fft_size = fft_sizes[index]
        print(f"[Config] FFT size changed to {fft_sizes[index]} (restart required)")
    
    def _on_spectrum_skip_change(self, index: int):
        """Update spectrum skip frames (takes effect immediately if engine running)"""
        skip_values = [1, 2, 4]
        self.config.audio.spectrum_skip_frames = skip_values[index]
        if self.audio_engine:
            self.audio_engine._spectrum_skip_frames = skip_values[index]
        print(f"[Config] Spectrum skip frames changed to {skip_values[index]}")
    
    def _on_metrics_global_toggle(self, state):
        """Master toggle for all metric auto-adjust checkboxes"""
        enabled = state == 2
        self.config.auto_adjust.metrics_global_enabled = enabled
        # Enable/disable all individual metric checkboxes
        for cb in (self.metric_peak_floor_cb, self.metric_audio_amp_cb,
                    self.metric_flux_balance_cb, self.metric_target_bps_cb):
            cb.setChecked(enabled)
            cb.setEnabled(enabled)
        print(f"[Metric] Global auto-adjust {'enabled' if enabled else 'disabled'}")
    
    def _on_metric_toggle(self, metric: str, enabled: bool):
        """Toggle a real-time metric-based auto-ranging metric"""
        if not hasattr(self, 'audio_engine') or self.audio_engine is None:
            print(f"[Metric] Audio engine not available yet")
            return
        
        self.audio_engine.enable_metric_autoranging(metric, enabled)
        status = "enabled" if enabled else "disabled"
        print(f"[Metric] {metric} {status}")
        
        # Update status label
        active_metrics = []
        if getattr(self, 'metric_peak_floor_cb', None) and self.metric_peak_floor_cb.isChecked():
            active_metrics.append("FloorMargin")

        if getattr(self, 'metric_audio_amp_cb', None) and self.metric_audio_amp_cb.isChecked():
            active_metrics.append("AudioAmp")
        if getattr(self, 'metric_flux_balance_cb', None) and self.metric_flux_balance_cb.isChecked():
            active_metrics.append("FluxBal")
        if getattr(self, 'metric_target_bps_cb', None) and self.metric_target_bps_cb.isChecked():
            active_metrics.append("TargetBPS")
        
        status_text = f"Metrics: [{', '.join(active_metrics) if active_metrics else 'idle'}]"
        if hasattr(self, 'metric_status_label'):
            self.metric_status_label.setText(status_text)
    
    def _on_metric_feedback(self, feedback_data: dict):
        """Handle feedback from a metric controller (update slider)"""
        metric = feedback_data.get('metric', '')
        adjustment = feedback_data.get('adjustment', 0.0)
        direction = feedback_data.get('direction', 'hold')
        
        if metric == 'peak_floor' and adjustment != 0:
            current = self.peak_floor_slider.value()
            new_val = current + adjustment
            pf_min, pf_max = BEAT_RANGE_LIMITS['peak_floor']
            new_val = max(pf_min, min(pf_max, new_val))
            if abs(new_val - current) > 0.001:
                self.peak_floor_slider.setValue(new_val)
                valley = feedback_data.get('valley', 0)
                margin = feedback_data.get('margin', 0)
                print(f"[Metric] peak_floor: valley={valley:.4f} ({direction}) -> {new_val:.4f}")
        
        elif metric == 'target_bps' and adjustment != 0:
            # Adjust peak_floor to hit target BPS
            # BUT: suppress lowering if valley-tracking wants to RAISE it (prevents oscillation)
            if feedback_data.get('direction', '') == 'lower':
                # Check if valley tracking is active and wants to raise
                if (hasattr(self, 'audio_engine') and self.audio_engine is not None
                    and self.audio_engine._metric_peak_floor_enabled
                    and len(self.audio_engine._valley_history) >= 3):
                    avg_valley = float(np.mean(self.audio_engine._valley_history))
                    current_pf = self.config.beat.peak_floor
                    if current_pf < avg_valley * 0.8:
                        # Valley tracking would raise peak_floor, so suppress BPS lowering
                        print(f"[Metric] target_bps: suppressed (valley={avg_valley:.4f} > pf={current_pf:.4f})")
                        return
            current = self.peak_floor_slider.value()
            new_val = current + adjustment
            pf_min, pf_max = BEAT_RANGE_LIMITS['peak_floor']
            new_val = max(pf_min, min(pf_max, new_val))
            if abs(new_val - current) > 0.001:
                self.peak_floor_slider.setValue(new_val)
                actual_bps = feedback_data.get('actual_bps', 0)
                target_bps = feedback_data.get('target_bps', 0)
                print(f"[Metric] target_bps: actual={actual_bps:.2f} target={target_bps:.2f} ({direction}) -> pf={new_val:.4f}")
                # Update the BPM display if we have one
                # bpm_actual_label now shows metronome BPM (updated in _on_beat)
        
        elif metric == 'audio_amp' and adjustment != 0:
            # Adjust audio amplification based on beat presence
            current = self.audio_gain_slider.value()
            new_val = current + adjustment
            aa_min, aa_max = BEAT_RANGE_LIMITS['audio_amp']
            new_val = max(aa_min, min(aa_max, new_val))
            if abs(new_val - current) > 0.001:
                self.audio_gain_slider.setValue(new_val)
                reason = feedback_data.get('reason', '')
                actual_bps = feedback_data.get('actual_bps', 0)
                print(f"[Metric] audio_amp: {reason} ({direction}) -> {new_val:.4f}")
        
        elif metric == 'flux_balance' and adjustment != 0:
            # Adjust flux_mult to balance flux ≈ energy bar heights
            current = self.flux_mult_slider.value()
            new_val = current + adjustment
            fm_min, fm_max = BEAT_RANGE_LIMITS['flux_mult']
            # Amplitude proportionality: flux_mult must always be >= 15% of audio_amp
            amp_floor = self.config.audio.gain * 0.15
            new_val = max(max(fm_min, amp_floor), min(fm_max, new_val))
            if abs(new_val - current) > 0.005:
                self.flux_mult_slider.setValue(new_val)
                ratio = feedback_data.get('ratio', 0)
                reason = feedback_data.get('reason', '')
                print(f"[Metric] flux_balance: {reason} ({direction}) -> fm={new_val:.2f}")
    
    def _on_target_bpm_change(self, value: float):
        """Handle target BPM spinbox change - converts to BPS for engine"""
        bps = value / 60.0
        if hasattr(self, 'audio_engine') and self.audio_engine is not None:
            self.audio_engine.set_target_bps(bps)
            print(f"[Config] Target BPM set to {value:.0f} ({bps:.2f} BPS)")
    
    def _on_bpm_tolerance_change(self, value: float):
        """Handle BPM tolerance spinbox change - converts to BPS for engine"""
        bps_tol = value / 60.0
        if hasattr(self, 'audio_engine') and self.audio_engine is not None:
            self.audio_engine.set_bps_tolerance(bps_tol)
            print(f"[Config] BPM tolerance set to ±{value:.0f} (±{bps_tol:.2f} BPS)")

    def _on_metric_response_speed_change(self, value: float):
        """Handle metric auto-adjust response speed change."""
        self.config.auto_adjust.metric_response_speed = value
        if hasattr(self, 'audio_engine') and self.audio_engine is not None:
            self.audio_engine.set_metric_response_speed(value)
        print(f"[Metric] Auto-adjust speed set to {value:.2f}x")
    
    # _on_bps_speed_change removed — speed hardcoded to max in audio_engine

    def _on_auto_align_toggle(self, enabled: bool):
        """Handle auto-align target BPM checkbox toggle"""
        self._auto_align_target_enabled = enabled
        self._auto_align_is_stable = False
        self._auto_align_stable_since = 0.0
        self._auto_align_last_adjust_time = 0.0
        if enabled:
            print("[Config] Auto-align target BPM enabled - will align to sensed BPM when stable")
        else:
            print("[Config] Auto-align target BPM disabled")

    def _on_auto_align_seconds_change(self, value: float):
        """Handle auto-align seconds spinbox change"""
        self._auto_align_required_seconds = value
        print(f"[Config] Auto-align requires {value:.1f}s of stable tempo before aligning")

    def _create_beat_detection_tab(self) -> QWidget:
        """Beat detection settings with vertical scroll"""
        # Outer scroll area (no wheel to prevent interference with parameter sliders)
        scroll_area = NoWheelScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(self._get_thin_scrollbar_style())
        
        # Content widget inside scroll area
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Detection type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Detection Type:"))
        self.detection_type_combo = QComboBox()
        self.detection_type_combo.addItems(["Peak Energy", "Spectral Flux", "Combined"])
        self.detection_type_combo.setCurrentIndex(2)  # Combined
        self.detection_type_combo.currentIndexChanged.connect(self._on_detection_type_change)
        type_layout.addWidget(self.detection_type_combo)
        type_layout.addStretch()
        # Wrap detection type in a groupbox
        detect_group = QGroupBox("Detection")
        detect_layout = QVBoxLayout(detect_group)
        detect_layout.addLayout(type_layout)

        # Global auto-adjust toggle moved to top Detection group
        self.metrics_global_cb = QCheckBox("Enable Auto-Adjust")
        self.metrics_global_cb.setChecked(self.config.auto_adjust.metrics_global_enabled)
        self.metrics_global_cb.setToolTip("Master toggle for all auto-adjust controls")
        self.metrics_global_cb.setStyleSheet("font-weight: bold; font-size: 10px;")
        self.metrics_global_cb.stateChanged.connect(self._on_metrics_global_toggle)
        detect_layout.addWidget(self.metrics_global_cb)

        layout.addWidget(detect_group)

        # Everything below detection type lives inside one windowshade section
        auto_levels_group = CollapsibleGroupBox("Auto-Levels", collapsed=True)
        auto_levels_layout = QVBoxLayout(auto_levels_group)

        # ===== AUTO-ADJUST (METRIC-BASED AUTO-RANGING) =====
        metric_layout = auto_levels_layout
        
        # Butterworth filter (mandatory for metrics)
        self.butterworth_checkbox = QCheckBox("Butterworth bandpass filter")
        self.butterworth_checkbox.setChecked(getattr(self.config.audio, 'use_butterworth', True))
        self.butterworth_checkbox.stateChanged.connect(self._on_butterworth_toggle)
        metric_layout.addWidget(self.butterworth_checkbox)
        
        # Metric controls row
        metric_ctrl_layout = QHBoxLayout()
        
        self.metric_peak_floor_cb = QCheckBox("Depth Margin")
        self.metric_peak_floor_cb.setToolTip("Auto-adjust depth threshold to track energy valley level (scales with amplification)")
        self.metric_peak_floor_cb.stateChanged.connect(lambda state: self._on_metric_toggle('peak_floor', state == 2))
        metric_ctrl_layout.addWidget(self.metric_peak_floor_cb)
        
        self.metric_audio_amp_cb = QCheckBox("Audio Amp (Beat)")
        self.metric_audio_amp_cb.setToolTip("No beats → raise audio_amp 2%/1.1s | Excess beats → lower audio_amp")
        self.metric_audio_amp_cb.stateChanged.connect(lambda state: self._on_metric_toggle('audio_amp', state == 2))
        metric_ctrl_layout.addWidget(self.metric_audio_amp_cb)
        
        self.metric_flux_balance_cb = QCheckBox("Flux Balance")
        self.metric_flux_balance_cb.setToolTip("Auto-adjust flux_mult to keep flux ≈ energy bar heights (0.01 steps/500ms)")
        self.metric_flux_balance_cb.stateChanged.connect(lambda state: self._on_metric_toggle('flux_balance', state == 2))
        metric_ctrl_layout.addWidget(self.metric_flux_balance_cb)
        
        metric_ctrl_layout.addStretch()
        metric_layout.addLayout(metric_ctrl_layout)

        self.metric_speed_slider = SliderWithLabel(
            "Auto-Adjust Speed",
            0.5,
            3.0,
            getattr(self.config.auto_adjust, 'metric_response_speed', 1.0),
            2
        )
        self.metric_speed_slider.valueChanged.connect(self._on_metric_response_speed_change)
        metric_layout.addWidget(self.metric_speed_slider)
        
        # ===== TARGET BPS CONTROLS =====
        bps_layout = QHBoxLayout()
        
        self.metric_target_bps_cb = QCheckBox("Target BPM")
        self.metric_target_bps_cb.setToolTip("Adjust depth threshold to achieve target beats per minute")
        self.metric_target_bps_cb.stateChanged.connect(lambda state: self._on_metric_toggle('target_bps', state == 2))
        bps_layout.addWidget(self.metric_target_bps_cb)
        
        bps_layout.addWidget(QLabel("Target:"))
        self.target_bpm_spin = QDoubleSpinBox()
        self.target_bpm_spin.setRange(30, 240)
        self.target_bpm_spin.setSingleStep(1)
        self.target_bpm_spin.setValue(110)
        self.target_bpm_spin.setDecimals(0)
        self.target_bpm_spin.setFixedWidth(65)
        self.target_bpm_spin.setSuffix(" BPM")
        self.target_bpm_spin.setToolTip("Target beats per minute (e.g., 110 BPM = 1.83 BPS)")
        self.target_bpm_spin.valueChanged.connect(self._on_target_bpm_change)
        bps_layout.addWidget(self.target_bpm_spin)
        
        bps_layout.addWidget(QLabel("±"))
        self.bpm_tolerance_spin = QDoubleSpinBox()
        self.bpm_tolerance_spin.setRange(3, 60)
        self.bpm_tolerance_spin.setSingleStep(1)
        self.bpm_tolerance_spin.setValue(30)
        self.bpm_tolerance_spin.setDecimals(0)
        self.bpm_tolerance_spin.setFixedWidth(60)
        self.bpm_tolerance_spin.setToolTip("Tolerance: system accepts ±this range around target BPM")
        self.bpm_tolerance_spin.valueChanged.connect(self._on_bpm_tolerance_change)
        bps_layout.addWidget(self.bpm_tolerance_spin)
        
        # Speed slider removed — hardcoded to max in audio_engine
        
        self.bpm_actual_label = QLabel("Metro: -- BPM")
        self.bpm_actual_label.setStyleSheet("color: #AAA; font-size: 9px;")
        bps_layout.addWidget(self.bpm_actual_label)
        
        self.auto_align_target_cb = QCheckBox("Auto-align")
        self.auto_align_target_cb.setToolTip("Automatically align target BPM to match sensed BPM when tempo is stable")
        self.auto_align_target_cb.setChecked(True)
        self.auto_align_target_cb.stateChanged.connect(lambda state: self._on_auto_align_toggle(state == 2))
        bps_layout.addWidget(self.auto_align_target_cb)
        
        self.auto_align_seconds_spin = QDoubleSpinBox()
        self.auto_align_seconds_spin.setRange(0.1, 8.0)
        self.auto_align_seconds_spin.setValue(0.2)
        self.auto_align_seconds_spin.setSingleStep(0.1)
        self.auto_align_seconds_spin.setDecimals(2)
        self.auto_align_seconds_spin.setSuffix("s")
        self.auto_align_seconds_spin.setFixedWidth(60)
        self.auto_align_seconds_spin.setToolTip("Seconds of stable tempo required before auto-aligning target BPM")
        self.auto_align_seconds_spin.valueChanged.connect(self._on_auto_align_seconds_change)
        bps_layout.addWidget(self.auto_align_seconds_spin)
        
        bps_layout.addStretch()
        metric_layout.addLayout(bps_layout)
        
        # Metric status label only (traffic light moved to control panel)
        status_row = QHBoxLayout()
        self.metric_status_label = QLabel("Metrics: [idle]")
        self.metric_status_label.setStyleSheet("color: #AAA; font-size: 9px;")
        status_row.addWidget(self.metric_status_label)
        status_row.addStretch()
        metric_layout.addLayout(status_row)
        
        # Enable metrics based on config (first load = True, then saved)
        global_on = self.config.auto_adjust.metrics_global_enabled
        self.metric_peak_floor_cb.setChecked(global_on)
        self.metric_audio_amp_cb.setChecked(global_on)
        self.metric_flux_balance_cb.setChecked(global_on)
        self.metric_target_bps_cb.setChecked(global_on)
        if global_on:
            print("[Config] Auto-enabled 4 core metrics from config")

        # ===== LEVELS: Audio Amplification, Sensitivity, Flux Multiplier =====
        levels_layout = auto_levels_layout
        
        # Frequency band selection with visibility toggle (red beat detection band)
        beat_slider_row = QHBoxLayout()
        self.freq_range_slider = RangeSliderWithLabel("Freq Range (Hz)", 30, 22050, 30, 4000, 0, log_scale=True)
        self.freq_range_slider.rangeChanged.connect(self._on_freq_band_change)
        beat_slider_row.addWidget(self.freq_range_slider)
        levels_layout.addLayout(beat_slider_row)
        
        # Audio amplification/gain: boost weak signals (0.15=quiet, 5.0=loud)
        aa_min, aa_max = BEAT_RANGE_LIMITS['audio_amp']
        self.audio_gain_slider = SliderWithLabel("Audio Amplification", aa_min, aa_max, self.config.audio.gain, 2)
        self.audio_gain_slider.valueChanged.connect(lambda v: setattr(self.config.audio, 'gain', v))
        levels_layout.addWidget(self.audio_gain_slider)
        
        # Sensitivity: higher = more beats detected (0.0=strict, 1.0=very sensitive)
        sens_min, sens_max = BEAT_RANGE_LIMITS['sensitivity']
        self.sensitivity_slider = SliderWithLabel("Sensitivity", sens_min, sens_max, self.config.beat.sensitivity)
        self.sensitivity_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'sensitivity', v))
        levels_layout.addWidget(self.sensitivity_slider)
        
        # Z-Score Threshold: lower = more z-score beats, higher = fewer (1.0-5.0)
        self.zscore_threshold_slider = SliderWithLabel("Z-Score Sens", 1.0, 5.0, 2.5)
        self.zscore_threshold_slider.valueChanged.connect(self._on_zscore_threshold_change)
        levels_layout.addWidget(self.zscore_threshold_slider)
        
        # Flux Multiplier
        fm_min, fm_max = BEAT_RANGE_LIMITS['flux_mult']
        self.flux_mult_slider = SliderWithLabel("Flux Multiplier", fm_min, fm_max, self.config.beat.flux_multiplier, 2)
        self.flux_mult_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'flux_multiplier', v))
        levels_layout.addWidget(self.flux_mult_slider)
        

        # ===== DEPTH/PEAKS: Depth, Peak Decay, Rise Sensitivity =====
        peaks_layout = auto_levels_layout
        
        # Peak floor: minimum energy to consider (0 = disabled)
        # Range 0.01-0.15: typical band_energy is 0.08-0.15 with default gain
        pf_min, pf_max = BEAT_RANGE_LIMITS['peak_floor']
        self.peak_floor_slider = SliderWithLabel("Depth", pf_min, pf_max, self.config.beat.peak_floor, 3)
        self.peak_floor_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'peak_floor', v))
        peaks_layout.addWidget(self.peak_floor_slider)
        
        # Peak decay
        pd_min, pd_max = BEAT_RANGE_LIMITS['peak_decay']
        self.peak_decay_slider = SliderWithLabel("Peak Decay", pd_min, pd_max, self.config.beat.peak_decay, 3)
        self.peak_decay_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'peak_decay', v))
        peaks_layout.addWidget(self.peak_decay_slider)
        
        # Rise sensitivity: 0 = disabled, higher = require more rise
        rs_min, rs_max = BEAT_RANGE_LIMITS['rise_sens']
        self.rise_sens_slider = SliderWithLabel("Rise Sensitivity", rs_min, rs_max, self.config.beat.rise_sensitivity)
        self.rise_sens_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'rise_sensitivity', v))
        peaks_layout.addWidget(self.rise_sens_slider)
        
        layout.addWidget(auto_levels_group)
        
        layout.addStretch()
        scroll_area.setWidget(widget)
        return scroll_area
    
    def _on_motion_freq_cutoff_change(self, value: int):
        """Handle motion frequency cutoff spinbox change"""
        self.config.beat.motion_freq_cutoff = float(value)
        print(f"[Config] Allow motion only below: {value} Hz (bands with lower edge >= {value} are filtered)")

    def _effective_fill_requirement(self, phase: str) -> float:
        cfg = self.config.stroke
        scale = float(np.clip(getattr(cfg, 'overall_amp_fill_required_scale', 1.0) or 1.0, 0.05, 20.0))

        if phase == 'syncopation':
            base_required = float(getattr(cfg, 'syncopation_overall_amp_fill_required', 0.12) or 0.12)
            if base_required >= 0.70:
                base_required = 0.12
        elif phase == 'downbeat':
            base_required = float(getattr(cfg, 'downbeat_overall_amp_fill_required', 0.08) or 0.08)
            if base_required >= 0.60:
                base_required = 0.08
        else:
            base_required = float(getattr(cfg, 'beat_overall_amp_fill_required', 0.10) or 0.10)
            if base_required >= 0.70:
                base_required = 0.10

        return float(np.clip(base_required * scale, 0.0, 1.0))

    def _preview_fill_requirement_ghosts(self) -> None:
        """Preview fill-gate requirements on dB/Hz + FFT visualizers (5s ghosts)."""
        down_val = self._effective_fill_requirement('downbeat')
        beat_val = self._effective_fill_requirement('beat')
        sync_val = self._effective_fill_requirement('syncopation')

        if hasattr(self, 'freqdb_canvas') and hasattr(self.freqdb_canvas, 'show_flux_ghost'):
            self.freqdb_canvas.show_flux_ghost('fill_req_downbeat_ratio', down_val, '% fill for (downbeat)', color='#66E0FF', duration_s=5.0, dashed=True, band='full', range_box=True, mode='occupancy')
            self.freqdb_canvas.show_flux_ghost('fill_req_beat_ratio', beat_val, '% fill for (beat)', color='#55CCFF', duration_s=5.0, dashed=True, band='full', range_box=True, mode='occupancy')
            self.freqdb_canvas.show_flux_ghost('fill_req_sync_ratio', sync_val, '% fill for (synco)', color='#44B8FF', duration_s=5.0, dashed=True, band='full', range_box=True, mode='occupancy')

        if hasattr(self, 'fft_bin_canvas') and hasattr(self.fft_bin_canvas, 'show_fill_ratio_ghost'):
            self.fft_bin_canvas.show_fill_ratio_ghost('fill_req_downbeat_ratio', down_val, '% fill for (downbeat)', color='#66E0FF', duration_s=5.0, dashed=True)
            self.fft_bin_canvas.show_fill_ratio_ghost('fill_req_beat_ratio', beat_val, '% fill for (beat)', color='#55CCFF', duration_s=5.0, dashed=True)
            self.fft_bin_canvas.show_fill_ratio_ghost('fill_req_sync_ratio', sync_val, '% fill for (synco)', color='#44B8FF', duration_s=5.0, dashed=True)
    
    def _on_freq_band_change(self, low=None, high=None):
        """Update frequency band in config and spectrum overlay"""
        # Handle both range slider (low, high params) and direct calls
        user_changed_slider = low is not None and high is not None
        if low is None:
            low = self.freq_range_slider.low() or 0.0
            high = self.freq_range_slider.high() or 22050.0
        low = float(low)  # type: ignore
        high = float(high)  # type: ignore
        
        self.config.beat.freq_low = low
        self.config.beat.freq_high = high
        
        # Re-initialize Butterworth filter with new band so beat detection actually uses it
        if hasattr(self, 'audio_engine') and self.audio_engine is not None:
            self.audio_engine._init_butterworth_filter()
        
        # Update spectrum overlay
        sr = self.config.audio.sample_rate
        max_freq = sr / 2
        if hasattr(self, 'freqdb_canvas') and hasattr(self.freqdb_canvas, 'set_frequency_band'):
            self.freqdb_canvas.set_frequency_band(low / max_freq, high / max_freq)

        # On user slider change, show 5s ghost range on dB/Hz and FFT-bin visualizers.
        if user_changed_slider:
            self._show_pulse_frequency_ghosts('beat_detect', low, high, 'Beat detect', '#FF6666')
    
    def _on_depth_band_change(self, low=None, high=None):
        """Update stroke depth frequency band in config and spectrum overlay"""
        if low is None:
            low = float(getattr(self.config.stroke, 'depth_freq_low', 0.0) or 0.0)
            high = float(getattr(self.config.stroke, 'depth_freq_high', self.config.audio.sample_rate / 2) or (self.config.audio.sample_rate / 2))
        low = float(low)  # type: ignore
        high = float(high)  # type: ignore
        
        self.config.stroke.depth_freq_low = low
        self.config.stroke.depth_freq_high = high
        
        # Update spectrum overlay (green band)
        if hasattr(self, 'freqdb_canvas') and hasattr(self.freqdb_canvas, 'set_depth_band'):
            self.freqdb_canvas.set_depth_band(low, high)
    
    def _on_p0_band_change(self, low=None, high=None):
        """Update P0 TCode frequency band in config and show 5s range ghosts."""
        if low is None:
            low = self.pulse_freq_range_slider.low() or 0.0
            high = self.pulse_freq_range_slider.high() or 22050.0
        low = float(low)  # type: ignore
        high = float(high)  # type: ignore
        
        self.config.pulse_freq.monitor_freq_min = low
        self.config.pulse_freq.monitor_freq_max = high
        self._show_pulse_frequency_ghosts('p0', low, high, 'Pulse monitor', '#5599FF')
    
    def _on_f0_band_change(self, low=None, high=None):
        """Update F0 TCode frequency band in config and show 5s range ghosts."""
        if low is None:
            low = self.f0_freq_range_slider.low() or 0.0
            high = self.f0_freq_range_slider.high() or 22050.0
        low = float(low)  # type: ignore
        high = float(high)  # type: ignore
        
        self.config.carrier_freq.monitor_freq_min = low
        self.config.carrier_freq.monitor_freq_max = high

        self._show_pulse_frequency_ghosts('f0', low, high, 'Carrier monitor', '#55DDFF')

    def _on_p1_band_change(self, low=None, high=None):
        """Update P1 monitor frequency band in config and show 5s range ghosts."""
        if low is None:
            low = self.p1_monitor_range_slider.low() or 0.0
            high = self.p1_monitor_range_slider.high() or 22050.0
        low = float(low)  # type: ignore
        high = float(high)  # type: ignore

        self.config.pulse_width.monitor_freq_min = low
        self.config.pulse_width.monitor_freq_max = high
        self._show_pulse_frequency_ghosts('p1', low, high, 'Pulse width monitor', '#FFB347')

    def _on_p3_band_change(self, low=None, high=None):
        """Update P3 monitor frequency band in config and show 5s range ghosts."""
        if low is None:
            low = self.p3_monitor_range_slider.low() or 0.0
            high = self.p3_monitor_range_slider.high() or 22050.0
        low = float(low)  # type: ignore
        high = float(high)  # type: ignore

        self.config.rise_time.monitor_freq_min = low
        self.config.rise_time.monitor_freq_max = high
        self._show_pulse_frequency_ghosts('p3', low, high, 'Rise time monitor', '#9DFF8A')

    def _show_pulse_frequency_ghosts(self, key: str, low_hz: float, high_hz: float, label: str, color: str) -> None:
        """Show temporary 5s frequency-range ghosts on Freq dB and FFT-bin visualizers."""
        low = float(max(0.0, min(low_hz, high_hz)))
        high = float(max(0.0, max(low_hz, high_hz)))

        if hasattr(self, 'freqdb_canvas') and hasattr(self.freqdb_canvas, 'show_flux_ghost'):
            self.freqdb_canvas.show_flux_ghost(
                f'{key}_monitor_range',
                low,
                f'{label} ({int(low)}-{int(high)} Hz)',
                color=color,
                duration_s=5.0,
                dashed=False,
                mode='hz_line',
                range_box=True,
                hz_max=high,
            )

        sample_rate = float(getattr(self.config.audio, 'sample_rate', 44100) or 44100)
        fft_size = int(getattr(self.config.audio, 'fft_size', 1024) or 1024)
        max_bin = max(1, fft_size // 2)
        low_bin = int(np.clip(round((low * fft_size) / max(1.0, sample_rate)), 0, max_bin))
        high_bin = int(np.clip(round((high * fft_size) / max(1.0, sample_rate)), 0, max_bin))

        if hasattr(self, 'fft_bin_canvas') and hasattr(self.fft_bin_canvas, 'show_bin_range_ghost'):
            self.fft_bin_canvas.show_bin_range_ghost(
                f'{key}_monitor_range',
                low_bin,
                high_bin,
                f'{label} bins',
                color=color,
                duration_s=5.0,
                dashed=False,
            )
    
    def _on_tempo_tracking_toggle(self, state):
        """Enable/disable tempo tracking"""
        enabled = state == 2  # Qt.CheckState.Checked
        self.config.beat.tempo_tracking_enabled = enabled
        if self.audio_engine:
            self.audio_engine.tempo_tracking_enabled = enabled
            if not enabled:
                # Reset tempo state when disabled
                self.audio_engine.smoothed_tempo = 0.0
                self.audio_engine.stable_tempo = 0.0
                self.audio_engine.beat_intervals.clear()
                self.audio_engine.beat_times.clear()
        self._apply_tempo_settings_enabled_state(enabled)
        print(f"[Config] Tempo tracking {'enabled' if enabled else 'disabled'}")

    def _on_tempo_settings_lock_toggle(self, state: int):
        """Lock/unlock tempo tuning controls in Tempo Settings."""
        self._apply_tempo_settings_enabled_state(self.tempo_tracking_checkbox.isChecked())

    def _apply_tempo_settings_enabled_state(self, tempo_enabled: bool):
        """Apply enabled state to tempo settings controls, honoring lock toggle."""
        lock_cb = getattr(self, 'tempo_settings_lock_cb', None)
        locked = bool(lock_cb is not None and lock_cb.isChecked())
        allow_edit = bool(tempo_enabled and not locked)
        for widget in getattr(self, '_tempo_settings_lock_targets', []):
            widget.setEnabled(allow_edit)
    
    def _on_time_sig_change(self, index: int):
        """Update time signature (beats per measure)"""
        beats_map = {0: 4, 1: 3, 2: 6}  # 4/4, 3/4, 6/8
        self.config.beat.beats_per_measure = beats_map.get(index, 4)
        # Update audio engine if running
        if self.audio_engine:
            self.audio_engine.beats_per_measure = self.config.beat.beats_per_measure
            # Reset measure tracking arrays to new size
            self.audio_engine.measure_energy_accum = [0.0] * self.config.beat.beats_per_measure
            self.audio_engine.measure_beat_counts = [0] * self.config.beat.beats_per_measure
            self.audio_engine.beat_position_in_measure = 0
        print(f"[Config] Time signature changed to {self.config.beat.beats_per_measure} beats/measure")
    
    def _on_stability_threshold_change(self, value: float):
        """Update stability threshold in config and audio engine"""
        self.config.beat.stability_threshold = value
        if self.audio_engine:
            self.audio_engine.stability_threshold = value
    
    def _on_tempo_timeout_change(self, value: float):
        """Update tempo timeout in config and audio engine"""
        self.config.beat.tempo_timeout_ms = int(value)
        if self.audio_engine:
            self.audio_engine.tempo_timeout_ms = value
    
    def _on_phase_snap_change(self, value: float):
        """Update phase snap weight in config and audio engine"""
        self.config.beat.phase_snap_weight = value
        if self.audio_engine:
            self.audio_engine.phase_snap_weight = value

    def _on_acf_interval_change(self, value: int):
        """Update ACF cadence in config and audio engine."""
        self.config.beat.acf_interval_ms = float(value)
        if self.audio_engine:
            self.audio_engine._acf_interval_ms = float(value)

    def _on_metronome_bpm_alpha_slow_change(self, value: float):
        """Update slow BPM smoothing alpha in config and audio engine."""
        self.config.beat.metronome_bpm_alpha_slow = value
        if self.audio_engine:
            self.audio_engine._metronome_bpm_alpha_slow = value

    def _on_metronome_bpm_alpha_fast_change(self, value: float):
        """Update fast BPM smoothing alpha in config and audio engine."""
        self.config.beat.metronome_bpm_alpha_fast = value
        if self.audio_engine:
            self.audio_engine._metronome_bpm_alpha_fast = value

    def _on_metronome_pll_window_change(self, value: float):
        """Update PLL correction window in config and audio engine."""
        self.config.beat.metronome_pll_window = value
        if self.audio_engine:
            self.audio_engine._metronome_pll_window = value

    def _on_metronome_pll_base_gain_change(self, value: float):
        """Update base PLL gain in config and audio engine."""
        self.config.beat.metronome_pll_base_gain = value
        if self.audio_engine:
            self.audio_engine._metronome_pll_base_gain = value

    def _on_metronome_pll_conf_gain_change(self, value: float):
        """Update confidence PLL gain in config and audio engine."""
        self.config.beat.metronome_pll_conf_gain = value
        if self.audio_engine:
            self.audio_engine._metronome_pll_conf_gain = value

    def _on_tempo_fusion_min_acf_weight_change(self, value: float):
        """Update min ACF fusion weight in config and audio engine."""
        self.config.beat.tempo_fusion_min_acf_weight = value
        if self.audio_engine:
            self.audio_engine._tempo_fusion_min_acf_weight = value

    def _on_tempo_fusion_max_acf_weight_change(self, value: float):
        """Update max ACF fusion weight in config and audio engine."""
        self.config.beat.tempo_fusion_max_acf_weight = value
        if self.audio_engine:
            self.audio_engine._tempo_fusion_max_acf_weight = value

    def _on_beat_dedup_fraction_change(self, value: float):
        """Update raw-onset de-dup fraction in config and audio engine."""
        self.config.beat.beat_dedup_fraction = value
        if self.audio_engine:
            self.audio_engine._beat_dedup_fraction = value

    def _on_phase_accept_window_ms_change(self, value: float):
        """Update phase acceptance window in ms in config and audio engine."""
        self.config.beat.phase_accept_window_ms = float(value)
        if self.audio_engine:
            self.audio_engine._phase_accept_window_ms = float(value)

    def _on_phase_accept_low_conf_mult_change(self, value: float):
        """Update low-confidence expansion multiplier for phase acceptance window."""
        self.config.beat.phase_accept_low_conf_mult = value
        if self.audio_engine:
            self.audio_engine._phase_accept_low_conf_mult = value

    def _on_octave_target_bias_confidence_max_change(self, value: float):
        """Update max confidence where target-BPM hint can guide octave disambiguation."""
        self.config.beat.octave_target_bias_confidence_max = value
        if self.audio_engine:
            self.audio_engine._octave_target_bias_confidence_max = value

    def _on_target_bps_lock_gate_toggle(self, enabled: bool):
        """Enable/disable lock-aware gating for target-BPS metric adjustments."""
        self.config.beat.target_bps_lock_gate_enabled = enabled
        if self.audio_engine:
            self.audio_engine._target_bps_lock_gate_enabled = enabled

    def _on_target_bps_lock_gate_acf_conf_change(self, value: float):
        """Update confidence threshold for lock-aware target-BPS gating."""
        self.config.beat.target_bps_lock_gate_acf_conf = value
        if self.audio_engine:
            self.audio_engine._target_bps_lock_gate_acf_conf = value

    def _on_target_bps_lock_gate_downbeats_change(self, value: int):
        """Update minimum downbeat matches for lock-aware target-BPS gating."""
        self.config.beat.target_bps_lock_gate_downbeats = int(value)
        if self.audio_engine:
            self.audio_engine._target_bps_lock_gate_downbeats = int(value)

    def _on_aggressive_tempo_snap_toggle(self, enabled: bool):
        """Toggle confidence-gated aggressive metronome BPM snapping."""
        self.config.beat.aggressive_tempo_snap_enabled = enabled
        if self.audio_engine:
            self.audio_engine._aggressive_tempo_snap_enabled = enabled

    def _on_aggressive_snap_confidence_change(self, value: float):
        """Update minimum ACF confidence for aggressive snap."""
        self.config.beat.aggressive_snap_confidence = value
        if self.audio_engine:
            self.audio_engine._aggressive_snap_confidence = value

    def _on_aggressive_snap_phase_error_ms_change(self, value: float):
        """Update max phase error allowed for aggressive snap."""
        self.config.beat.aggressive_snap_phase_error_ms = float(value)
        if self.audio_engine:
            self.audio_engine._aggressive_snap_phase_error_ms = float(value)

    def _on_aggressive_snap_min_matches_change(self, value: int):
        """Update minimum downbeat match count required for aggressive snap."""
        self.config.beat.aggressive_snap_min_matches = int(value)
        if self.audio_engine:
            self.audio_engine._aggressive_snap_min_matches = int(value)

    def _on_aggressive_snap_max_jump_change(self, value: float):
        """Update max relative BPM jump allowed for aggressive snap."""
        self.config.beat.aggressive_snap_max_bpm_jump_ratio = value
        if self.audio_engine:
            self.audio_engine._aggressive_snap_max_bpm_jump_ratio = value
    
    def _save_freq_preset(self, idx: int):
        """Preset slots are currently fixed and not user-configurable."""
        print(f"[Presets] Slot {idx+1} is reserved (empty)")
    
    def _load_freq_preset(self, idx: int):
        """Preset slots are currently fixed and not user-configurable."""
        print(f"[Presets] Slot {idx+1} is reserved (empty)")
    
    def _save_beat_preset(self, idx: int):
        """Preset slots are currently fixed and not user-configurable."""
        self._save_freq_preset(idx)
    
    def _load_beat_preset(self, idx: int):
        """Preset slots are currently fixed and not user-configurable."""
        self._load_freq_preset(idx)

    def _capture_learned_slot_payload(self) -> dict:
        """Capture current learning + key calibration state for hidden learned-slot storage."""
        return {
            'learning': {
                'teaching_learning_enabled': bool(getattr(self.config.beat, 'teaching_learning_enabled', True)),
                'teaching_use_fitted_rules': bool(getattr(self.config.beat, 'teaching_use_fitted_rules', True)),
                'teaching_apply_in_circle_mode': bool(getattr(self.config.beat, 'teaching_apply_in_circle_mode', False)),
                'teaching_isolation_mode': bool(getattr(self.config.beat, 'teaching_isolation_mode', True)),
                'teaching_learning_strength': float(getattr(self.config.beat, 'teaching_learning_strength', 0.55) or 0.55),
                'teaching_min_confidence': float(getattr(self.config.beat, 'teaching_min_confidence', 0.12) or 0.12),
                'teaching_no_motion_bias': float(getattr(self.config.beat, 'teaching_no_motion_bias', 1.0) or 1.0),
                'teaching_rule_fit_path': str(getattr(self.config.beat, 'teaching_rule_fit_path', '') or ''),
                'teaching_profile_path': str(getattr(self.config.beat, 'teaching_profile_path', '') or ''),
            },
            'calibration': {
                'freq_low': float(self.freq_range_slider.low() or 0.0),
                'freq_high': float(self.freq_range_slider.high() or 22050.0),
                'audio_gain': float(self.audio_gain_slider.value()),
                'sensitivity': float(self.sensitivity_slider.value()),
                'zscore_threshold': float(self.zscore_threshold_slider.value()),
                'flux_multiplier': float(self.flux_mult_slider.value()),
                'peak_floor': float(self.peak_floor_slider.value()),
                'peak_decay': float(self.peak_decay_slider.value()),
                'rise_sensitivity': float(self.rise_sens_slider.value()),
            },
            'saved_at': datetime.now().isoformat(timespec='seconds'),
        }

    def _apply_learned_slot_payload(self, payload: dict) -> None:
        """Apply a learned-slot payload to runtime config and sliders."""
        if not isinstance(payload, dict):
            return

        learning = payload.get('learning', {})
        if isinstance(learning, dict):
            bool_keys = (
                'teaching_learning_enabled',
                'teaching_use_fitted_rules',
                'teaching_apply_in_circle_mode',
                'teaching_isolation_mode',
            )
            float_keys = (
                'teaching_learning_strength',
                'teaching_min_confidence',
                'teaching_no_motion_bias',
            )
            str_keys = (
                'teaching_rule_fit_path',
                'teaching_profile_path',
            )
            for key in bool_keys:
                if key in learning:
                    setattr(self.config.beat, key, bool(learning.get(key)))
            for key in float_keys:
                if key in learning:
                    try:
                        raw_value = learning.get(key)
                        if isinstance(raw_value, (int, float, str)):
                            setattr(self.config.beat, key, float(raw_value))
                    except Exception:
                        pass
            for key in str_keys:
                if key in learning:
                    setattr(self.config.beat, key, str(learning.get(key) or '').strip())

        calibration = payload.get('calibration', {})
        if isinstance(calibration, dict):
            try:
                low = float(calibration.get('freq_low', self.freq_range_slider.low() or 0.0))
                high = float(calibration.get('freq_high', self.freq_range_slider.high() or 22050.0))
                self.freq_range_slider.setLow(low)
                self.freq_range_slider.setHigh(high)
            except Exception:
                pass
            try:
                if 'audio_gain' in calibration:
                    raw_audio_gain = calibration.get('audio_gain')
                    if isinstance(raw_audio_gain, (int, float, str)):
                        self.audio_gain_slider.setValue(float(raw_audio_gain))
                if 'sensitivity' in calibration:
                    raw_sensitivity = calibration.get('sensitivity')
                    if isinstance(raw_sensitivity, (int, float, str)):
                        self.sensitivity_slider.setValue(float(raw_sensitivity))
                if 'zscore_threshold' in calibration:
                    raw_zscore_threshold = calibration.get('zscore_threshold')
                    if isinstance(raw_zscore_threshold, (int, float, str)):
                        self.zscore_threshold_slider.setValue(float(raw_zscore_threshold))
                if 'flux_multiplier' in calibration:
                    raw_flux_multiplier = calibration.get('flux_multiplier')
                    if isinstance(raw_flux_multiplier, (int, float, str)):
                        self.flux_mult_slider.setValue(float(raw_flux_multiplier))
                if 'peak_floor' in calibration:
                    raw_peak_floor = calibration.get('peak_floor')
                    if isinstance(raw_peak_floor, (int, float, str)):
                        self.peak_floor_slider.setValue(float(raw_peak_floor))
                if 'peak_decay' in calibration:
                    raw_peak_decay = calibration.get('peak_decay')
                    if isinstance(raw_peak_decay, (int, float, str)):
                        self.peak_decay_slider.setValue(float(raw_peak_decay))
                if 'rise_sensitivity' in calibration:
                    raw_rise_sensitivity = calibration.get('rise_sensitivity')
                    if isinstance(raw_rise_sensitivity, (int, float, str)):
                        self.rise_sens_slider.setValue(float(raw_rise_sensitivity))
            except Exception:
                pass

        self._apply_learning_config_to_mapper()

    def _set_learned_profile_slot(self, idx: int) -> None:
        """Programmatically save current learning/calibration state into a hidden slot."""
        key = str(int(max(0, min(4, idx))))
        self.learned_profile_slots[key] = self._capture_learned_slot_payload()

    def _apply_learned_profile_slot(self, idx: int) -> None:
        """Programmatically apply a hidden learned slot to runtime state."""
        key = str(int(max(0, min(4, idx))))
        payload = self.learned_profile_slots.get(key)
        if isinstance(payload, dict) and payload:
            self._apply_learned_slot_payload(payload)
    
    def _save_presets_to_disk(self):
        """Preset persistence disabled: keep learned slots in-memory only."""
        return
    
    def _load_presets_from_disk(self):
        """Load hidden learned-slot payloads and initialize reserved empty slot buttons."""
        self.custom_beat_presets = {}
        self.learned_profile_slots = {}
        try:
            path = get_config_dir() / 'learned_profile_slots.json'
            if path.exists():
                payload = json.loads(path.read_text(encoding='utf-8'))
                slots = payload.get('slots', {}) if isinstance(payload, dict) else {}
                if isinstance(slots, dict):
                    for key, value in slots.items():
                        if str(key).isdigit() and isinstance(value, dict):
                            idx = int(str(key))
                            if 0 <= idx < 5:
                                self.learned_profile_slots[str(idx)] = value
        except Exception as e:
            print(f"[Presets] Error loading learned slots: {e}")

        for btn in getattr(self, 'preset_buttons', []):
            btn.setText("empty")
            btn.set_has_preset(False)
            btn.set_active(False)
            btn.setEnabled(False)
        revert_btn = getattr(self, 'revert_btn', None)
        if revert_btn is not None:
            revert_btn.setEnabled(False)
    
    def _create_tempo_response_group(self, lock_default: bool = True) -> QGroupBox:
        """Advanced tempo-response controls group used in Tempo Tracking popout."""
        tempo_resp_group = QGroupBox("Advanced Tempo Controls")
        tempo_resp_layout = QVBoxLayout(tempo_resp_group)

        lock_cb = QCheckBox("Lock advanced tempo controls")
        lock_cb.setChecked(bool(lock_default))
        lock_cb.setToolTip("Lock/unlock advanced tempo tracking tuning controls in this group")
        tempo_resp_layout.addWidget(lock_cb)

        tempo_resp_info = QLabel("Tune lock/relock speed and phase correction behavior.\nLower smoothing = faster response, higher can be steadier.")
        tempo_resp_info.setStyleSheet("color: #aaa; font-size: 11px;")
        tempo_resp_layout.addWidget(tempo_resp_info)

        def _set_slider_row_tooltip(widget: SliderWithLabel, text: str):
            widget.setToolTip(text)
            widget.label.setToolTip(text)
            widget.slider.setToolTip(text)
            widget.value_label.setToolTip(text)

        acf_row = QHBoxLayout()
        acf_label = QLabel("ACF interval (ms):")
        acf_label.setStyleSheet("color: #ccc;")
        acf_row.addWidget(acf_label)
        acf_spin = QSpinBox()
        acf_spin.setMinimum(150)
        acf_spin.setMaximum(800)
        acf_spin.setSingleStep(10)
        acf_spin.setValue(int(getattr(self.config.beat, 'acf_interval_ms', 250)))
        acf_spin.setToolTip("How often to run ACF tempo estimation")
        acf_label.setToolTip("How often to run ACF tempo estimation. Lower = faster tempo updates, higher = steadier updates.")
        acf_spin.valueChanged.connect(self._on_acf_interval_change)
        acf_row.addWidget(acf_spin)
        tempo_resp_layout.addLayout(acf_row)

        alpha_slow_slider = SliderWithLabel("BPM Adaptation (Uncertain)", 0.01, 0.20, getattr(self.config.beat, 'metronome_bpm_alpha_slow', 0.03), 3)
        alpha_slow_slider.valueChanged.connect(self._on_metronome_bpm_alpha_slow_change)
        _set_slider_row_tooltip(alpha_slow_slider, "How fast BPM locks when beat signal is unclear. Lower = safer, slower lock; Higher = aggressive adaptation")
        tempo_resp_layout.addWidget(alpha_slow_slider)

        alpha_fast_slider = SliderWithLabel("BPM Adaptation (Confident)", 0.05, 0.40, getattr(self.config.beat, 'metronome_bpm_alpha_fast', 0.22), 3)
        alpha_fast_slider.valueChanged.connect(self._on_metronome_bpm_alpha_fast_change)
        _set_slider_row_tooltip(alpha_fast_slider, "How fast BPM locks when beat is strong and clear. Lower = stable, conservative; Higher = snappy response")
        tempo_resp_layout.addWidget(alpha_fast_slider)

        pll_window_slider = SliderWithLabel("Timing Flex Window", 0.10, 0.50, getattr(self.config.beat, 'metronome_pll_window', 0.35), 2)
        pll_window_slider.valueChanged.connect(self._on_metronome_pll_window_change)
        _set_slider_row_tooltip(pll_window_slider, "How much timing drift (as beat fraction) allowed before auto-correction kicks in. Higher = more tolerant; Lower = stricter")
        tempo_resp_layout.addWidget(pll_window_slider)

        pll_base_slider = SliderWithLabel("Timing Correction Strength", 0.01, 0.20, getattr(self.config.beat, 'metronome_pll_base_gain', 0.09), 3)
        pll_base_slider.valueChanged.connect(self._on_metronome_pll_base_gain_change)
        _set_slider_row_tooltip(pll_base_slider, "How aggressively to adjust timing when out of sync. Lower = laid-back/behind-the-beat feel; Higher = tight/ahead tracking")
        tempo_resp_layout.addWidget(pll_base_slider)

        pll_conf_slider = SliderWithLabel("Confidence Boost", 0.00, 0.20, getattr(self.config.beat, 'metronome_pll_conf_gain', 0.08), 3)
        pll_conf_slider.valueChanged.connect(self._on_metronome_pll_conf_gain_change)
        _set_slider_row_tooltip(pll_conf_slider, "Extra timing correction when beat lock is solid. Amplifies the correction intensity. Lower = stable when locked; Higher = responsive")
        tempo_resp_layout.addWidget(pll_conf_slider)

        fusion_min_slider = SliderWithLabel("Fusion min ACF wt", 0.00, 0.80, getattr(self.config.beat, 'tempo_fusion_min_acf_weight', 0.20), 2)
        fusion_min_slider.valueChanged.connect(self._on_tempo_fusion_min_acf_weight_change)
        _set_slider_row_tooltip(fusion_min_slider, "Minimum ACF contribution when blending ACF BPM with onset BPM.")
        tempo_resp_layout.addWidget(fusion_min_slider)

        fusion_max_slider = SliderWithLabel("Fusion max ACF wt", 0.20, 1.00, getattr(self.config.beat, 'tempo_fusion_max_acf_weight', 0.95), 2)
        fusion_max_slider.valueChanged.connect(self._on_tempo_fusion_max_acf_weight_change)
        _set_slider_row_tooltip(fusion_max_slider, "Maximum ACF contribution when confidence is high.")
        tempo_resp_layout.addWidget(fusion_max_slider)

        dedup_slider = SliderWithLabel("Beat de-dup frac", 0.10, 0.35, getattr(self.config.beat, 'beat_dedup_fraction', 0.22), 2)
        dedup_slider.valueChanged.connect(self._on_beat_dedup_fraction_change)
        _set_slider_row_tooltip(dedup_slider, "Reject a second raw onset if it arrives within this fraction of the current beat period. Reduces double-beat chatter.")
        tempo_resp_layout.addWidget(dedup_slider)

        phase_accept_slider = SliderWithLabel("Phase accept win ms", 20.0, 220.0, getattr(self.config.beat, 'phase_accept_window_ms', 85.0), 0)
        phase_accept_slider.valueChanged.connect(
            lambda v: self._on_phase_accept_window_ms_change(v)
        )
        _set_slider_row_tooltip(phase_accept_slider, "Accept raw onsets only when they are this close (ms) to expected beat phase.")
        tempo_resp_layout.addWidget(phase_accept_slider)

        low_conf_mult_slider = SliderWithLabel("Low-conf win x", 1.00, 3.50, getattr(self.config.beat, 'phase_accept_low_conf_mult', 2.0), 2)
        low_conf_mult_slider.valueChanged.connect(self._on_phase_accept_low_conf_mult_change)
        _set_slider_row_tooltip(low_conf_mult_slider, "Multiplies phase-accept window when confidence is low, so relock stays flexible.")
        tempo_resp_layout.addWidget(low_conf_mult_slider)

        octave_target_bias_slider = SliderWithLabel(
            "Target-hint max conf",
            0.05,
            0.80,
            getattr(self.config.beat, 'octave_target_bias_confidence_max', 0.35),
            2,
        )
        octave_target_bias_slider.valueChanged.connect(self._on_octave_target_bias_confidence_max_change)
        _set_slider_row_tooltip(
            octave_target_bias_slider,
            "Only use target BPM to guide octave disambiguation below this ACF confidence."
        )
        tempo_resp_layout.addWidget(octave_target_bias_slider)

        target_bps_lock_gate_cb = QCheckBox("Gate Target BPM metric when metronome lock is confident")
        target_bps_lock_gate_cb.setChecked(getattr(self.config.beat, 'target_bps_lock_gate_enabled', True))
        target_bps_lock_gate_cb.setToolTip("When enabled, target-BPM metric stops nudging peak_floor while metronome lock is strong.")
        target_bps_lock_gate_cb.stateChanged.connect(lambda state: self._on_target_bps_lock_gate_toggle(state == 2))
        tempo_resp_layout.addWidget(target_bps_lock_gate_cb)

        target_bps_lock_conf_slider = SliderWithLabel(
            "Target-BPS lock conf",
            0.10,
            0.90,
            getattr(self.config.beat, 'target_bps_lock_gate_acf_conf', 0.40),
            2,
        )
        target_bps_lock_conf_slider.valueChanged.connect(self._on_target_bps_lock_gate_acf_conf_change)
        _set_slider_row_tooltip(
            target_bps_lock_conf_slider,
            "Minimum ACF confidence required before Target-BPS metric gating is applied."
        )
        tempo_resp_layout.addWidget(target_bps_lock_conf_slider)

        target_bps_lock_match_row = QHBoxLayout()
        target_bps_lock_match_label = QLabel("Target-BPS lock min downbeat matches:")
        target_bps_lock_match_label.setStyleSheet("color: #ccc;")
        target_bps_lock_match_row.addWidget(target_bps_lock_match_label)
        target_bps_lock_match_spin = QSpinBox()
        target_bps_lock_match_spin.setMinimum(0)
        target_bps_lock_match_spin.setMaximum(4)
        target_bps_lock_match_spin.setValue(int(getattr(self.config.beat, 'target_bps_lock_gate_downbeats', 1)))
        target_bps_lock_match_label.setToolTip("Minimum consecutive matching downbeats required before Target-BPS metric gating applies.")
        target_bps_lock_match_spin.setToolTip("Minimum consecutive matching downbeats required before Target-BPS metric gating applies.")
        target_bps_lock_match_spin.valueChanged.connect(self._on_target_bps_lock_gate_downbeats_change)
        target_bps_lock_match_row.addWidget(target_bps_lock_match_spin)
        tempo_resp_layout.addLayout(target_bps_lock_match_row)

        aggressive_snap_cb = QCheckBox("Aggressive tempo snap when lock is confident")
        aggressive_snap_cb.setChecked(getattr(self.config.beat, 'aggressive_tempo_snap_enabled', False))
        aggressive_snap_cb.setToolTip("When enabled, metronome BPM can hard-snap to target under strict confidence/phase safeguards.")
        aggressive_snap_cb.stateChanged.connect(lambda state: self._on_aggressive_tempo_snap_toggle(state == 2))
        tempo_resp_layout.addWidget(aggressive_snap_cb)

        snap_conf_slider = SliderWithLabel("Snap min confidence", 0.20, 0.90, getattr(self.config.beat, 'aggressive_snap_confidence', 0.55), 2)
        snap_conf_slider.valueChanged.connect(self._on_aggressive_snap_confidence_change)
        _set_slider_row_tooltip(snap_conf_slider, "Minimum ACF confidence required before aggressive snap is allowed.")
        tempo_resp_layout.addWidget(snap_conf_slider)

        snap_phase_slider = SliderWithLabel("Snap max phase err ms", 10.0, 120.0, getattr(self.config.beat, 'aggressive_snap_phase_error_ms', 35.0), 0)
        snap_phase_slider.valueChanged.connect(
            lambda v: self._on_aggressive_snap_phase_error_ms_change(v)
        )
        _set_slider_row_tooltip(snap_phase_slider, "Only snap if beat phase error is below this many milliseconds.")
        tempo_resp_layout.addWidget(snap_phase_slider)

        snap_jump_slider = SliderWithLabel("Snap max BPM jump", 0.03, 0.30, getattr(self.config.beat, 'aggressive_snap_max_bpm_jump_ratio', 0.12), 2)
        snap_jump_slider.valueChanged.connect(self._on_aggressive_snap_max_jump_change)
        _set_slider_row_tooltip(snap_jump_slider, "Maximum one-step relative BPM change allowed during aggressive snap.")
        tempo_resp_layout.addWidget(snap_jump_slider)

        snap_match_row = QHBoxLayout()
        snap_match_label = QLabel("Snap min downbeat matches:")
        snap_match_label.setStyleSheet("color: #ccc;")
        snap_match_row.addWidget(snap_match_label)
        snap_match_spin = QSpinBox()
        snap_match_spin.setMinimum(0)
        snap_match_spin.setMaximum(4)
        snap_match_spin.setValue(int(getattr(self.config.beat, 'aggressive_snap_min_matches', 1)))
        snap_match_label.setToolTip("Require this many matching downbeats before aggressive tempo snap can trigger.")
        snap_match_spin.setToolTip("Require this many matching downbeats before aggressive tempo snap can trigger.")
        snap_match_spin.valueChanged.connect(self._on_aggressive_snap_min_matches_change)
        snap_match_row.addWidget(snap_match_spin)
        tempo_resp_layout.addLayout(snap_match_row)

        reset_tempo_btn = QPushButton("Reset Tempo Response Defaults")
        reset_tempo_btn.setToolTip("Restore default values for all tempo-response tuning controls")

        def _reset_tempo_response_defaults():
            acf_spin.setValue(250)
            alpha_slow_slider.setValue(0.03)
            alpha_fast_slider.setValue(0.22)
            pll_window_slider.setValue(0.35)
            pll_base_slider.setValue(0.09)
            pll_conf_slider.setValue(0.08)
            fusion_min_slider.setValue(0.20)
            fusion_max_slider.setValue(0.95)
            dedup_slider.setValue(0.22)
            phase_accept_slider.setValue(85.0)
            low_conf_mult_slider.setValue(2.0)
            octave_target_bias_slider.setValue(0.35)
            target_bps_lock_gate_cb.setChecked(True)
            target_bps_lock_conf_slider.setValue(0.40)
            target_bps_lock_match_spin.setValue(1)
            aggressive_snap_cb.setChecked(False)
            snap_conf_slider.setValue(0.55)
            snap_phase_slider.setValue(35.0)
            snap_jump_slider.setValue(0.12)
            snap_match_spin.setValue(1)

        reset_tempo_btn.clicked.connect(_reset_tempo_response_defaults)
        tempo_resp_layout.addWidget(reset_tempo_btn)

        advanced_targets = [
            acf_label,
            acf_spin,
            alpha_slow_slider,
            alpha_fast_slider,
            pll_window_slider,
            pll_base_slider,
            pll_conf_slider,
            fusion_min_slider,
            fusion_max_slider,
            dedup_slider,
            phase_accept_slider,
            low_conf_mult_slider,
            octave_target_bias_slider,
            target_bps_lock_gate_cb,
            target_bps_lock_conf_slider,
            target_bps_lock_match_label,
            target_bps_lock_match_spin,
            aggressive_snap_cb,
            snap_conf_slider,
            snap_phase_slider,
            snap_jump_slider,
            snap_match_label,
            snap_match_spin,
            reset_tempo_btn,
        ]

        def _apply_advanced_lock_state() -> None:
            locked = bool(lock_cb.isChecked())
            for widget in advanced_targets:
                widget.setEnabled(not locked)

        lock_cb.stateChanged.connect(lambda _: _apply_advanced_lock_state())
        _apply_advanced_lock_state()
        return tempo_resp_group

    def _create_tempo_tracking_tab(self, include_advanced_controls: bool = False, advanced_locked: bool = True) -> QWidget:
        """Tempo tracking and rhythm settings"""
        scroll_area = NoWheelScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(self._get_thin_scrollbar_style())

        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # ===== TEMPO SETTINGS =====
        tempo_group = CollapsibleGroupBox("Tempo Settings", collapsed=False)
        tempo_layout = QVBoxLayout(tempo_group)
        
        # Enable/disable checkbox
        self.tempo_tracking_checkbox = QCheckBox("Enable Tempo Tracking")
        self.tempo_tracking_checkbox.setChecked(True)
        self.tempo_tracking_checkbox.stateChanged.connect(self._on_tempo_tracking_toggle)
        tempo_layout.addWidget(self.tempo_tracking_checkbox)

        self.tempo_settings_lock_cb = QCheckBox("Lock tempo settings")
        self.tempo_settings_lock_cb.setChecked(False)
        self.tempo_settings_lock_cb.setToolTip("Lock/unlock tempo tuning controls in this group")
        self.tempo_settings_lock_cb.stateChanged.connect(self._on_tempo_settings_lock_toggle)
        tempo_layout.addWidget(self.tempo_settings_lock_cb)
        
        # Time signature dropdown
        sig_layout = QHBoxLayout()
        sig_layout.addWidget(QLabel("Time Signature:"))
        self.time_sig_combo = QComboBox()
        self.time_sig_combo.addItems(["4/4 (4 beats)", "3/4 (3 beats)", "6/8 (6 beats)"])
        self.time_sig_combo.currentIndexChanged.connect(self._on_time_sig_change)
        sig_layout.addWidget(self.time_sig_combo)
        sig_layout.addStretch()
        tempo_layout.addLayout(sig_layout)
        
        # Stability threshold: lower = stricter (requires more consistent intervals before locking BPM)
        self.stability_threshold_slider = SliderWithLabel("Stability Threshold", 0.05, 0.4, 0.15, 2)
        self.stability_threshold_slider.valueChanged.connect(self._on_stability_threshold_change)
        tempo_layout.addWidget(self.stability_threshold_slider)
        
        # Tempo timeout: how long no beats before resetting tempo tracking
        self.tempo_timeout_slider = SliderWithLabel("Tempo Timeout (ms)", 500, 5000, 2000, 0)
        self.tempo_timeout_slider.valueChanged.connect(self._on_tempo_timeout_change)
        tempo_layout.addWidget(self.tempo_timeout_slider)
        
        # Phase snap: how much to nudge detected beats toward predicted time
        self.phase_snap_slider = SliderWithLabel("Phase Snap", 0.0, 0.8, 0.3, 2)
        self.phase_snap_slider.valueChanged.connect(self._on_phase_snap_change)
        tempo_layout.addWidget(self.phase_snap_slider)
        
        # Silence reset threshold (moved from Beat Detection)
        self.silence_reset_slider = SliderWithLabel("Silence Reset (ms)", 100, 3000, 400, 0)
        self.silence_reset_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'silence_reset_ms', int(v)))
        tempo_layout.addWidget(self.silence_reset_slider)

        self._tempo_settings_lock_targets = [
            self.time_sig_combo,
            self.stability_threshold_slider,
            self.tempo_timeout_slider,
            self.phase_snap_slider,
            self.silence_reset_slider,
        ]

        self._apply_tempo_settings_enabled_state(self.tempo_tracking_checkbox.isChecked())
        
        layout.addWidget(tempo_group)

        if include_advanced_controls:
            layout.addWidget(self._create_tempo_response_group(lock_default=advanced_locked))
        
        layout.addStretch()
        scroll_area.setWidget(widget)
        return scroll_area

    # Event handlers
    def _auto_connect_tcp(self):
        """Auto-connect TCP on program startup"""
        self.config.connection.host = self.host_edit.text()
        self.config.connection.port = self.port_spin.value()
        self.network_engine = ensure_network_engine(
            self.network_engine,
            self.config,
            self._network_status_callback,
            force_new=True,
        )
        print("[Main] Auto-connecting TCP on startup")

    def _on_connect(self):
        """Handle connect/disconnect button"""
        if self.network_engine is None:
            self.config.connection.host = self.host_edit.text()
            self.config.connection.port = self.port_spin.value()
            self.network_engine = ensure_network_engine(
                self.network_engine,
                self.config,
                self._network_status_callback,
            )
        else:
            toggle_user_connection(self.network_engine)

    def _on_connection_refresh(self):
        """Refresh TCP connection from the main status button."""
        self.config.connection.host = self.host_edit.text()
        self.config.connection.port = self.port_spin.value()
        if self.network_engine is None:
            self.network_engine = ensure_network_engine(
                self.network_engine,
                self.config,
                self._network_status_callback,
            )
            return

        try:
            self.network_engine.disconnect()
        except Exception:
            pass
        self.network_engine.user_connect()
    
    def _on_test(self):
        """Send test pattern"""
        trigger_network_test(self.network_engine)

    def _apply_pending_start(self) -> None:
        """Apply a queued start request captured during stop transition."""
        if self._transport_transition:
            return
        if self.is_running:
            self._transport_pending_start = False
            return
        self._transport_pending_start = False
        self._on_start_stop(True)

    def _apply_pending_stop(self) -> None:
        """Apply a queued stop request captured during start transition."""
        if self._transport_transition:
            return
        if not self.is_running:
            self._transport_pending_stop = False
            return
        self._transport_pending_stop = False
        self._on_start_stop(False)

    def _apply_pending_play(self) -> None:
        """Apply queued play/pause request captured during transport transition."""
        if self._transport_transition or not self.is_running:
            return
        desired = bool(self._transport_pending_play)
        self._on_play_pause(desired)

    def _sync_transport_buttons(self) -> None:
        """Synchronize Start/Play button state, text, and enabled flags."""
        if not hasattr(self, 'start_btn') or not hasattr(self, 'play_btn'):
            return

        running = bool(self.is_running)
        sending = bool(self.is_sending and self.is_running)
        ready = bool(self._transport_ready)
        transitioning = bool(self._transport_transition)

        ui_state = start_stop_ui_state(running)
        start_enabled = ready and (not transitioning)
        play_enabled = ready and running and (not transitioning)

        with self._signals_blocked(self.start_btn, self.play_btn):
            self.start_btn.setChecked(running)
            self.start_btn.setEnabled(start_enabled)
            self.start_btn.setText(ui_state.start_text)
            self.start_btn.setStyleSheet(f"color: {'#fff' if running else '#0af'};")

            self.play_btn.setChecked(sending)
            self.play_btn.setEnabled(play_enabled)
            self.play_btn.setText(play_button_text(sending))
            self.play_btn.setStyleSheet(f"color: {'#fff' if sending else '#0af'};")

        if transitioning:
            self.start_btn.setText("Working...")
    
    def _on_start_stop(self, checked: bool | None = None):
        """Start/stop audio capture and TCode pipeline.
        Start enables TCode sending (V0=0 until Play). Stop kills everything."""
        if checked is None:
            checked = not self.is_running

        if not self._transport_ready:
            if checked and not self.is_running:
                self._transport_pending_start = True
            self._sync_transport_buttons()
            return

        if self._transport_transition:
            if checked and not self.is_running:
                self._transport_pending_start = True
            elif not checked:
                # Stop should never be dropped; prioritize it over pending start.
                self._transport_pending_stop = True
                self._transport_pending_start = False
                self._transport_pending_play = None
            self._sync_transport_buttons()
            return

        self._transport_transition = True
        try:
            if checked:
                try:
                    # Reflect start intent immediately so a quick follow-up click
                    # during startup is interpreted as Stop, not another Start.
                    self.is_running = True
                    self._sync_transport_buttons()

                    self._start_engines()
                    ui_state = start_stop_ui_state(True)
                    self.start_btn.setText(ui_state.start_text)
                    self.play_btn.setEnabled(ui_state.play_enabled)
                    # Enable TCode sending immediately on Start (V0=0 until Play is pressed)
                    set_transport_sending(self.network_engine, True)
                    send_zero_volume_immediate(self.network_engine, duration_ms=160)
                except Exception as e:
                    print(f"[Main] Start failed: {e}")
                    self._stop_engines()
                    self.is_sending = False
                    set_transport_sending(self.network_engine, False)
            else:
                # Stop should immediately clear play/sending state before shutdown work.
                self._volume_ramp_active = False
                self._play_warmup_active = False
                self._play_warmup_seen_beat = False
                self.is_sending = False
                self._transport_pending_play = None

                # Make stop visually immediate and prevent second-click feel.
                self.is_running = False
                self._sync_transport_buttons()

                # Send zero-volume command before stopping (always, not just when is_sending)
                send_zero_volume_immediate(self.network_engine, duration_ms=160)
                set_transport_sending(self.network_engine, False)
                self._stop_engines()
                # Note: Auto-range state is preserved across stop/start - no reset here
        finally:
            self._transport_transition = False
            self._sync_transport_buttons()

            # Clear stale pending flags that no longer match runtime state.
            if self._transport_pending_start and self.is_running:
                self._transport_pending_start = False
            if self._transport_pending_stop and not self.is_running:
                self._transport_pending_stop = False

            # Stop wins over start when both were requested during transition.
            if self._transport_pending_stop and self.is_running:
                self._transport_pending_stop = False
                QTimer.singleShot(0, self._apply_pending_stop)
                return

            if self._transport_pending_start and not self.is_running:
                self._transport_pending_start = False
                QTimer.singleShot(0, self._apply_pending_start)
                return

            if self._transport_pending_play is not None and self.is_running:
                QTimer.singleShot(0, self._apply_pending_play)
    
    def _on_play_pause(self, checked: bool | None = None):
        """Play/pause motion generation. Pause sends V0=0 but keeps TCode pipeline active."""
        if checked is None:
            checked = not self.is_sending

        if self._transport_transition:
            self._transport_pending_play = bool(checked)
            self._sync_transport_buttons()
            return

        if not self.is_running:
            self.is_sending = False
            self._transport_pending_play = None
            self._sync_transport_buttons()
            return
        self.is_sending = checked
        if checked:
            # Re-instantiate StrokeMapper with current config (for live mode switching)
            self.stroke_mapper = StrokeMapper(self.config, self._send_command_direct, get_volume=lambda: self.volume_slider.value() / 100.0, audio_engine=self.audio_engine)
            self._apply_geometry_rest_to_mapper()
            self._apply_learning_config_to_mapper()
            # Warmup gate: allow audio analysis to settle and beat pickup before motion
            self._play_warmup_active = True
            self._play_warmup_started_at = time.time()
            self._play_warmup_seen_beat = False
            send_zero_volume_immediate(self.network_engine, duration_ms=1750)
            # Start volume ramp from 0 to set value over 1.3s
            ramp_state = begin_volume_ramp(time.time())
            self._volume_ramp_active = ramp_state.active
            self._volume_ramp_start_time = ramp_state.start_time
            self._volume_ramp_from = ramp_state.from_volume
            self._volume_ramp_to = ramp_state.to_volume
            # sending_enabled already True from Start — no need to set again
        else:
            # Send V0=0 immediately with fade, but keep TCode pipeline active
            self._play_warmup_active = False
            self._play_warmup_seen_beat = False
            self._volume_ramp_active = False
            send_zero_volume_immediate(self.network_engine, duration_ms=500)
            # DON'T disable sending_enabled — connection stays active until Stop
        self._transport_pending_play = None
        self._sync_transport_buttons()
    
    def _on_detection_type_change(self, index: int):
        """Change beat detection type"""
        self.config.beat.detection_type = BeatDetectionType(index + 1)
    
    def _on_mode_change(self, index: int):
        """Mode is temporarily pinned to Circle and selector is hidden."""
        self.config.stroke.mode = StrokeMode.SIMPLE_CIRCLE
        if hasattr(self, 'mode_combo') and self.mode_combo.currentIndex() != 0:
            with self._signals_blocked(self.mode_combo):
                self.mode_combo.setCurrentIndex(0)
        self._enforce_fixed_effect_axis_values()

    def _start_engines(self):
        """Initialize and start all engines"""
        # Set selected audio device and loopback mode
        combo_idx = self.device_combo.currentIndex()
        if combo_idx >= 0 and combo_idx in self.audio_device_map:
            self.config.audio.device_index = self.audio_device_map[combo_idx]
            self.config.audio.is_loopback = self.audio_device_is_loopback.get(combo_idx, False)
            is_loopback = "loopback" if self.config.audio.is_loopback else "input"
            print(f"[Main] Using audio device index: {self.config.audio.device_index} ({is_loopback})")

        self.audio_engine = AudioEngine(self.config, self._audio_callback)
        self.audio_engine.start()
        self.audio_engine.set_metric_response_speed(getattr(self.config.auto_adjust, 'metric_response_speed', 1.0))

        # Sync metric checkbox states to the new audio engine
        # (checkboxes may already be checked from previous start)
        self._sync_metric_checkboxes_to_engine()

        self.stroke_mapper = StrokeMapper(self.config, self._send_command_direct, get_volume=lambda: self.volume_slider.value() / 100.0, audio_engine=self.audio_engine)
        self._apply_geometry_rest_to_mapper()
        self._apply_learning_config_to_mapper()

        # Network engine is already started on program launch via _auto_connect_tcp
        # Only create if somehow missing
        self.network_engine = ensure_network_engine(
            self.network_engine,
            self.config,
            self._network_status_callback,
            dry_run_enabled=self._dry_run_enabled,
        )

        self.is_running = True
    
    def _sync_metric_checkboxes_to_engine(self):
        """Sync checked metric checkboxes to the audio engine after it's created.
        Fixes bug where auto-range doesn't activate until user toggles checkbox."""
        if not self.audio_engine:
            return
        metric_map = {
            'metric_peak_floor_cb': 'peak_floor',

            'metric_audio_amp_cb': 'audio_amp',
            'metric_flux_balance_cb': 'flux_balance',
            'metric_target_bps_cb': 'target_bps',
        }
        synced = []
        for attr, metric in metric_map.items():
            cb = getattr(self, attr, None)
            if cb is not None and cb.isChecked():
                self.audio_engine.enable_metric_autoranging(metric, True)
                synced.append(metric)
        if synced:
            print(f"[Metric] Synced {len(synced)} metrics to engine: {', '.join(synced)}")
    
    def _send_command_direct(self, cmd: TCodeCommand):
        """Send a command directly (used by StrokeMapper for arc strokes). Thread-safe."""
        if self.network_engine and self.is_sending:
            attach_cached_tcode_values(
                cmd,
                p0c0_enabled=self.config.device_limits.p0_c0_sending_enabled,
                cached_p0_enabled=self._cached_p0_enabled,
                cached_p0_val=self._cached_p0_val,
                cached_f0_enabled=self._cached_f0_enabled,
                cached_f0_val=self._cached_f0_val,
                cached_p1_enabled=self._cached_p1_enabled,
                cached_p1_val=self._cached_p1_val,
                cached_p3_enabled=self._cached_p3_enabled,
                cached_p3_val=self._cached_p3_val,
                freq_window_ms=int(self._freq_window_ms),
            )
            apply_volume_ramp(
                cmd,
                volume_ramp_active=self._volume_ramp_active,
                volume_ramp_start_time=self._volume_ramp_start_time,
                volume_ramp_duration=self._volume_ramp_duration,
                volume_ramp_from=self._volume_ramp_from,
                volume_ramp_to=self._volume_ramp_to,
            )
            # Cache actual tcode volume for display (includes silence fade + post-silence ramp)
            self._last_sent_volume_pct = float(cmd.volume) * 100.0
            self.network_engine.send_command(cmd)
    
    def _stop_engines(self):
        """Stop all engines and background threads"""
        self.is_running = False
        self.stroke_mapper = None

        if self.audio_engine:
            self.audio_engine.stop()
            self.audio_engine = None
    
    def _audio_callback(self, event: BeatEvent):
        """Called from audio thread on each frame - NO direct Qt widget access for thread safety"""
        # Emit signal for thread-safe GUI update
        self.signals.beat_detected.emit(event)

        # Get spectrum for visualization
        spectrum = None
        if self.audio_engine:
            spectrum = self.audio_engine.get_spectrum()
            if spectrum is not None:
                waveform = self.audio_engine.get_waveform()
                sample_rate = int(getattr(self.config.audio, 'sample_rate', 44100))
                spectrum_with_stats = {
                    'spectrum': spectrum,
                    'peak_energy': event.peak_energy,
                    'spectral_flux': event.spectral_flux,
                    'waveform': waveform,
                    'sample_rate': sample_rate,
                }
                self.signals.spectrum_ready.emit(spectrum_with_stats)

        # Process through stroke mapper
        if self.stroke_mapper and self.is_sending:
            if self._play_warmup_active:
                if event.is_beat:
                    self._play_warmup_seen_beat = True

                now = event.timestamp if event and event.timestamp > 0 else time.time()
                elapsed = max(0.0, now - self._play_warmup_started_at)
                warmup_ready = elapsed >= self._play_warmup_min_seconds and self._play_warmup_seen_beat
                warmup_timeout = elapsed >= self._play_warmup_max_seconds

                if not warmup_ready and not warmup_timeout:
                    return

                self._play_warmup_active = False

            cmd = self.stroke_mapper.process_beat(event)
            if cmd and self.network_engine:
                # Compute P0/F0 and attach to command (thread-safe, no widget access)
                self._compute_and_attach_tcode(cmd, event, spectrum)
                apply_volume_ramp(
                    cmd,
                    volume_ramp_active=self._volume_ramp_active,
                    volume_ramp_start_time=self._volume_ramp_start_time,
                    volume_ramp_duration=self._volume_ramp_duration,
                    volume_ramp_from=self._volume_ramp_from,
                    volume_ramp_to=self._volume_ramp_to,
                )
                self._last_sent_volume_pct = float(cmd.volume) * 100.0
                self.network_engine.send_command(cmd)
        elif event.is_beat and not self.is_sending:
            print("[Main] Beat detected but Play not enabled")
    
    def _extract_dominant_freq(self, spectrum: np.ndarray, sample_rate: int,
                               freq_low: float, freq_high: float) -> float:
        """Extract dominant frequency from a specific Hz range of the spectrum. Thread-safe."""
        return extract_dominant_freq(spectrum, sample_rate, freq_low, freq_high)
    
    def _compute_and_attach_tcode(self, cmd: TCodeCommand, event: BeatEvent, spectrum: Optional[np.ndarray] = None):
        """Compute P0/F0 TCode values and attach to command. Thread-safe (no widget access)."""
        now = time.time()

        def _effective_output_limits(raw_min: float, raw_max: float, default_min: float, default_max: float) -> tuple[float, float]:
            lo = float(raw_min)
            hi = float(raw_max)
            if hi <= lo or lo <= 0.0 or hi <= 0.0:
                lo = float(default_min)
                hi = float(default_max)
            return lo, hi
        
        # Extract dominant frequencies independently for P0 and F0 monitor ranges
        dom_freq = event.frequency if hasattr(event, 'frequency') else 0.0
        p0_dom_freq = dom_freq  # fallback
        f0_dom_freq = dom_freq  # fallback
        if spectrum is not None:
            sr = self.config.audio.sample_rate
            p0_dom_freq = self._extract_dominant_freq(spectrum, sr,
                self.config.pulse_freq.monitor_freq_min,
                self.config.pulse_freq.monitor_freq_max)
            f0_dom_freq = self._extract_dominant_freq(spectrum, sr,
                self.config.carrier_freq.monitor_freq_min,
                self.config.carrier_freq.monitor_freq_max)
        
        # Calculate dot speed for Speed mode
        dt = max(0.001, now - self._last_dot_time)
        delta_alpha = cmd.alpha - self._last_dot_alpha
        delta_beta = cmd.beta - self._last_dot_beta
        dot_speed = np.sqrt(delta_alpha**2 + delta_beta**2) / dt
        self._last_dot_alpha = cmd.alpha
        self._last_dot_beta = cmd.beta
        self._last_dot_time = now
        
        # --- P0 (Pulse Frequency) with short sliding window averaging ---
        p0_enabled = self._cached_p0_enabled
        if p0_enabled:
            pulse_mode = self._cached_pulse_mode
            pulse_invert = self._cached_pulse_invert
            freq_weight = self.config.pulse_freq.freq_weight
            
            if pulse_mode == 0:  # Hz mode
                in_low = self.config.pulse_freq.monitor_freq_min
                in_high = self.config.pulse_freq.monitor_freq_max
                norm = (p0_dom_freq - in_low) / max(1.0, in_high - in_low)
            elif pulse_mode == 2:  # Band (sub_bass) mode
                # Use sub_bass band energy directly — long booming bass = "feeling" the pulse
                sub_bass_energy = 0.0
                if self.audio_engine and hasattr(self.audio_engine, '_band_energies'):
                    sub_bass_energy = self.audio_engine._band_energies.get('sub_bass', 0.0)
                # Normalize: typical sub_bass energy 0-0.3 after gain
                norm = min(1.0, sub_bass_energy * 4.0)
            else:  # Speed mode
                norm = min(1.0, dot_speed / 10.0)
            
            norm = max(0.0, min(1.0, norm))
            norm_weighted = 0.5 + (norm - 0.5) * freq_weight
            norm_weighted = max(0.0, min(1.0, norm_weighted))
            
            if pulse_invert:
                norm_weighted = 1.0 - norm_weighted
            
            # Add sample to sliding window
            self._p0_freq_window.append((now, norm_weighted))
            
            # Remove samples older than window size
            window_cutoff = now - (self._freq_window_ms / 1000.0)
            while self._p0_freq_window and self._p0_freq_window[0][0] < window_cutoff:
                self._p0_freq_window.popleft()
            
            # Calculate average over window
            if self._p0_freq_window:
                avg_norm = sum(s[1] for s in self._p0_freq_window) / len(self._p0_freq_window)
            else:
                avg_norm = norm_weighted
            
            # Map averaged frequency to TCode output range (direct TCode, 0-9999)
            tcode_min_val = self._cached_tcode_freq_min
            tcode_max_val = self._cached_tcode_freq_max
            tcode_min_val = max(0, min(9999, tcode_min_val))
            tcode_max_val = max(0, min(9999, tcode_max_val))
            p0_val = int(tcode_min_val + avg_norm * (tcode_max_val - tcode_min_val))
            p0_val = max(0, min(9999, p0_val))
            
            # Send P0 using current low-latency window duration
            cmd.pulse_freq = p0_val
            cmd.pulse_freq_duration = int(self._freq_window_ms)
            self._cached_p0_val = p0_val
            # Display converted real output (with safe fallback defaults when limits are unset).
            dl = self.config.device_limits
            p0_lo, p0_hi = _effective_output_limits(dl.p0_freq_min, dl.p0_freq_max, 1.0, 100.0)
            hz = p0_lo + (p0_val / 9999.0) * (p0_hi - p0_lo)
            self._cached_pulse_display = f"Pulse Freq: {hz:.0f}Hz"
        else:
            cmd.pulse_freq = None
            self._cached_p0_val = None
            self._cached_pulse_display = "Pulse Freq: off"
            self._p0_freq_window.clear()  # Clear window when disabled
        
        # --- F0 (Carrier Frequency) with short sliding window averaging ---
        f0_enabled = self._cached_f0_enabled
        if f0_enabled:
            f0_mode = self._cached_f0_mode
            f0_invert = self._cached_f0_invert
            f0_weight = self.config.carrier_freq.freq_weight
            
            if f0_mode == 0:  # Hz mode
                f0_in_low = self.config.carrier_freq.monitor_freq_min
                f0_in_high = self.config.carrier_freq.monitor_freq_max
                f0_norm = (f0_dom_freq - f0_in_low) / max(1.0, f0_in_high - f0_in_low)
            elif f0_mode == 2:  # Band (mid) mode — voice, brass, dominant strings (500-2000 Hz)
                # Use mid band energy directly — strict rate limit below
                mid_energy = 0.0
                if self.audio_engine and hasattr(self.audio_engine, '_band_energies'):
                    mid_energy = self.audio_engine._band_energies.get('mid', 0.0)
                # Normalize: typical mid energy 0-0.2 after gain
                f0_norm = min(1.0, mid_energy * 5.0)
            else:  # Speed mode
                f0_norm = min(1.0, dot_speed / 10.0)
            
            f0_norm = max(0.0, min(1.0, f0_norm))
            f0_norm_weighted = 0.5 + (f0_norm - 0.5) * f0_weight
            f0_norm_weighted = max(0.0, min(1.0, f0_norm_weighted))
            
            if f0_invert:
                f0_norm_weighted = 1.0 - f0_norm_weighted
            
            # Add sample to sliding window
            self._f0_freq_window.append((now, f0_norm_weighted))
            
            # Remove samples older than window size
            f0_window_cutoff = now - (self._freq_window_ms / 1000.0)
            while self._f0_freq_window and self._f0_freq_window[0][0] < f0_window_cutoff:
                self._f0_freq_window.popleft()
            
            # Calculate average over window
            if self._f0_freq_window:
                f0_avg_norm = sum(s[1] for s in self._f0_freq_window) / len(self._f0_freq_window)
            else:
                f0_avg_norm = f0_norm_weighted
            
            # Map averaged frequency to TCode output range (direct TCode, 0-9999)
            f0_tcode_min = self._cached_f0_tcode_min
            f0_tcode_max = self._cached_f0_tcode_max
            f0_tcode_min = max(0, min(9999, f0_tcode_min))
            f0_tcode_max = max(0, min(9999, f0_tcode_max))
            f0_val_raw = int(f0_tcode_min + f0_avg_norm * (f0_tcode_max - f0_tcode_min))
            f0_val_raw = max(0, min(9999, f0_val_raw))
            
            # Smooth F0: limit change rate for smoother transitions
            if f0_mode == 2:
                # Band (mid) mode: strict rate limiter — ±500 tcode per 2 seconds
                # Must finish traveling to current target before accepting new one
                if self._c0_band_current is None:
                    self._c0_band_current = f0_val_raw
                    self._c0_band_target = f0_val_raw

                # Check if we've arrived at current target
                at_target = (self._c0_band_target is not None
                             and abs(self._c0_band_current - self._c0_band_target) < 5)

                if at_target:
                    # Accept new target only if different enough (>50 tcode)
                    current_target = self._c0_band_target
                    if current_target is not None and abs(f0_val_raw - current_target) > 50:
                        # Clamp new target to bounded jump from current position
                        delta_from_current = f0_val_raw - self._c0_band_current
                        delta_from_current = max(-self._c0_band_max_target_delta, min(self._c0_band_max_target_delta, delta_from_current))
                        self._c0_band_target = self._c0_band_current + delta_from_current
                        self._c0_band_target = max(0, min(9999, self._c0_band_target))
                        self._c0_band_last_target_time = now

                # Travel toward target at _c0_band_travel_rate tcode/sec (=250/s → 500 per 2s)
                if self._c0_band_target is not None and self._c0_band_current != self._c0_band_target:
                    max_step = max(1, int(self._c0_band_travel_rate * dt))
                    diff = self._c0_band_target - self._c0_band_current
                    step = max(-max_step, min(max_step, diff))
                    self._c0_band_current += step
                    self._c0_band_current = max(0, min(9999, self._c0_band_current))

                f0_val = int(self._c0_band_current)
            elif self._f0_last_sent_tcode is not None:
                delta = f0_val_raw - self._f0_last_sent_tcode
                if abs(delta) > self._f0_max_change_per_send:
                    if delta > 0:
                        f0_val = self._f0_last_sent_tcode + self._f0_max_change_per_send
                    else:
                        f0_val = self._f0_last_sent_tcode - self._f0_max_change_per_send
                else:
                    f0_val = f0_val_raw
            else:
                f0_val = f0_val_raw
            f0_val = max(0, min(9999, f0_val))
            self._f0_last_sent_tcode = f0_val
            
            # Generate short random duration for live response
            f0_duration = int(self._f0_duration_base_ms + random.uniform(-self._f0_duration_variance_ms, self._f0_duration_variance_ms))
            f0_duration = max(100, f0_duration)  # Minimum 100ms
            
            if cmd.tcode_tags is None:
                cmd.tcode_tags = {}
            cmd.tcode_tags['C0'] = f0_val  # restim uses C0 for carrier frequency, not F0
            cmd.tcode_tags['C0_duration'] = f0_duration
            self._cached_f0_val = f0_val
            # Display converted real output (with safe fallback defaults when limits are unset).
            dl = self.config.device_limits
            c0_lo, c0_hi = _effective_output_limits(dl.c0_freq_min, dl.c0_freq_max, 500.0, 1500.0)
            hz = c0_lo + (f0_val / 9999.0) * (c0_hi - c0_lo)
            self._cached_carrier_display = f"Carrier Freq: {hz:.0f}Hz"
        else:
            self._cached_f0_val = None
            self._cached_carrier_display = "Carrier Freq: off"
            self._f0_freq_window.clear()  # Clear window when disabled
            self._f0_last_sent_tcode = None  # Reset smoothing state when disabled
        
        # --- P1 (Pulse Width) with short sliding window averaging ---
        p1_enabled = self._cached_p1_enabled
        if p1_enabled:
            p1_mode = self._cached_p1_mode
            p1_invert = self._cached_p1_invert
            p1_weight = self.config.pulse_width.weight
            
            if p1_mode == 0:  # Volume (RMS energy) mode
                # Use spectrum RMS as volume proxy (0-1 normalized)
                if spectrum is not None and len(spectrum) > 0:
                    spec_rms = float(np.sqrt(np.mean(spectrum ** 2)))
                    # Normalize: typical spec_rms range ~0.0001-0.05, map with log scale
                    p1_norm = max(0.0, min(1.0, (np.log10(max(spec_rms, 1e-8)) + 4) / 3.0))
                else:
                    p1_norm = 0.5
            elif p1_mode == 1:  # Hz (dominant freq) mode
                p1_dom_freq = self._extract_dominant_freq(spectrum, self.config.audio.sample_rate,
                    self.config.pulse_width.monitor_freq_min, self.config.pulse_width.monitor_freq_max) if spectrum is not None else 0.0
                p1_in_low = self.config.pulse_width.monitor_freq_min
                p1_in_high = self.config.pulse_width.monitor_freq_max
                p1_norm = (p1_dom_freq - p1_in_low) / max(1.0, p1_in_high - p1_in_low)
            else:  # Speed (dot movement) mode
                p1_norm = min(1.0, dot_speed / 10.0)
            
            p1_norm = max(0.0, min(1.0, p1_norm))
            p1_norm_weighted = 0.5 + (p1_norm - 0.5) * p1_weight
            p1_norm_weighted = max(0.0, min(1.0, p1_norm_weighted))
            
            if p1_invert:
                p1_norm_weighted = 1.0 - p1_norm_weighted
            
            # Sliding window average
            self._p1_window.append((now, p1_norm_weighted))
            p1_window_cutoff = now - (self._freq_window_ms / 1000.0)
            while self._p1_window and self._p1_window[0][0] < p1_window_cutoff:
                self._p1_window.popleft()
            p1_avg = sum(s[1] for s in self._p1_window) / len(self._p1_window) if self._p1_window else p1_norm_weighted
            
            # Map to TCode range
            p1_tcode_min = self._cached_p1_tcode_min
            p1_tcode_max = self._cached_p1_tcode_max
            p1_val = int(p1_tcode_min + p1_avg * (p1_tcode_max - p1_tcode_min))
            p1_val = max(0, min(9999, p1_val))
            
            if cmd.tcode_tags is None:
                cmd.tcode_tags = {}
            cmd.tcode_tags['P1'] = p1_val
            cmd.tcode_tags['P1_duration'] = int(self._freq_window_ms)
            self._cached_p1_val = p1_val
            # Display converted real output (with safe fallback defaults when limits are unset).
            dl = self.config.device_limits
            p1_lo, p1_hi = _effective_output_limits(dl.p1_cycles_min, dl.p1_cycles_max, 0.0, 20.0)
            p1_cyc = p1_lo + (p1_val / 9999.0) * (p1_hi - p1_lo)
            self._cached_p1_display = f"Pulse Width: {p1_cyc:.1f}cyc"
        else:
            self._cached_p1_val = None
            self._cached_p1_display = "Pulse Width: off"
            self._p1_window.clear()
        
        # --- P3 (Rise Time) with short sliding window averaging ---
        p3_enabled = self._cached_p3_enabled
        if p3_enabled:
            p3_mode = self._cached_p3_mode
            p3_invert = self._cached_p3_invert
            p3_weight = self.config.rise_time.weight
            
            if p3_mode == 0:  # Brightness (spectral centroid) mode
                if spectrum is not None and len(spectrum) > 0:
                    sr = self.config.audio.sample_rate
                    freqs = np.linspace(0, sr / 2, len(spectrum))
                    total_energy = float(np.sum(spectrum))
                    if total_energy > 1e-10:
                        centroid = float(np.sum(freqs * spectrum) / total_energy)
                    else:
                        centroid = sr / 4  # midpoint fallback
                    # Normalize centroid: typical range 200-8000 Hz
                    p3_norm = max(0.0, min(1.0, (centroid - 200) / 7800))
                    # INVERT inherently: bright audio → LOW rise time (exciting)
                    # So high centroid → low p3_norm (before user invert)
                    p3_norm = 1.0 - p3_norm
                else:
                    p3_norm = 0.5
            elif p3_mode == 1:  # Hz (dominant freq) mode
                p3_dom_freq = self._extract_dominant_freq(spectrum, self.config.audio.sample_rate,
                    self.config.rise_time.monitor_freq_min, self.config.rise_time.monitor_freq_max) if spectrum is not None else 0.0
                p3_in_low = self.config.rise_time.monitor_freq_min
                p3_in_high = self.config.rise_time.monitor_freq_max
                p3_norm = (p3_dom_freq - p3_in_low) / max(1.0, p3_in_high - p3_in_low)
            else:  # Speed (dot movement) mode
                p3_norm = min(1.0, dot_speed / 10.0)
            
            p3_norm = max(0.0, min(1.0, p3_norm))
            p3_norm_weighted = 0.5 + (p3_norm - 0.5) * p3_weight
            p3_norm_weighted = max(0.0, min(1.0, p3_norm_weighted))
            
            if p3_invert:
                p3_norm_weighted = 1.0 - p3_norm_weighted
            
            # Sliding window average
            self._p3_window.append((now, p3_norm_weighted))
            p3_window_cutoff = now - (self._freq_window_ms / 1000.0)
            while self._p3_window and self._p3_window[0][0] < p3_window_cutoff:
                self._p3_window.popleft()
            p3_avg = sum(s[1] for s in self._p3_window) / len(self._p3_window) if self._p3_window else p3_norm_weighted
            
            # Map to TCode range
            p3_tcode_min = self._cached_p3_tcode_min
            p3_tcode_max = self._cached_p3_tcode_max
            p3_val = int(p3_tcode_min + p3_avg * (p3_tcode_max - p3_tcode_min))
            p3_val = max(0, min(9999, p3_val))
            
            if cmd.tcode_tags is None:
                cmd.tcode_tags = {}
            cmd.tcode_tags['P3'] = p3_val
            cmd.tcode_tags['P3_duration'] = int(self._freq_window_ms)
            self._cached_p3_val = p3_val
            # Display converted real output (with safe fallback defaults when limits are unset).
            dl = self.config.device_limits
            p3_lo, p3_hi = _effective_output_limits(dl.p3_cycles_min, dl.p3_cycles_max, 0.0, 20.0)
            p3_cyc = p3_lo + (p3_val / 9999.0) * (p3_hi - p3_lo)
            self._cached_p3_display = f"Rise Time: {p3_cyc:.1f}cyc"
        else:
            self._cached_p3_val = None
            self._cached_p3_display = "Rise Time: off"
            self._p3_window.clear()
        
        # Log
        p0_str = f"P0={cmd.pulse_freq:04d}" if cmd.pulse_freq is not None else "P0=off"
        c0_tag = cmd.tcode_tags.get('C0', None) if cmd.tcode_tags else None
        c0_str = f"C0={c0_tag:04d}" if c0_tag is not None else "C0=off"
        p1_tag = cmd.tcode_tags.get('P1', None) if cmd.tcode_tags else None
        p1_str = f"P1={p1_tag:04d}" if p1_tag is not None else "P1=off"
        p3_tag = cmd.tcode_tags.get('P3', None) if cmd.tcode_tags else None
        p3_str = f"P3={p3_tag:04d}" if p3_tag is not None else "P3=off"
        gate_str = ""
        mapper = getattr(self, 'stroke_mapper', None)
        gf = getattr(mapper, '_last_gate_fail', None) if mapper is not None else None
        if gf:
            gate_str = f" GATE_FAIL={gf}"
        print(f"[Main] Cmd: a={cmd.alpha:.2f} b={cmd.beta:.2f} v={cmd.volume:.2f} {p0_str} {c0_str} {p1_str} {p3_str}{gate_str}")
    
    def _network_status_callback(self, message: str, connected: bool):
        """Called from network thread on status change"""
        self.signals.status_changed.emit(message, connected)
    
    def _on_beat(self, event: BeatEvent):
        """Handle beat event in GUI thread"""
        # ===== METRONOME SYNC INDICATOR (updates every frame, not just on beat) =====
        acf_conf_raw = getattr(event, 'acf_confidence', 0.0)
        metro_bpm_raw = getattr(event, 'metronome_bpm', 0.0)
        try:
            acf_conf = float(acf_conf_raw)
        except (TypeError, ValueError):
            acf_conf = 0.0
        try:
            metro_bpm = float(metro_bpm_raw)
        except (TypeError, ValueError):
            metro_bpm = 0.0

        if not np.isfinite(acf_conf):
            acf_conf = 0.0
        if not np.isfinite(metro_bpm):
            metro_bpm = 0.0

        if hasattr(self, 'metronome_sync_indicator') and self.metronome_sync_indicator is not None:
            try:
                if metro_bpm <= 0 or acf_conf < 0.05:
                    self.metronome_sync_indicator.setStyleSheet("color: #333; font-size: 20px;")  # Off
                elif acf_conf < 0.25:
                    self.metronome_sync_indicator.setStyleSheet("color: #cc0; font-size: 20px;")  # Yellow: locking
                else:
                    self.metronome_sync_indicator.setStyleSheet("color: #0f0; font-size: 20px;")  # Green: locked
            except RuntimeError:
                pass

        # Update metronome BPM display (small label next to target BPM controls)
        if hasattr(self, 'bpm_actual_label'):
            try:
                if metro_bpm > 0:
                    self.bpm_actual_label.setText(f"Metro: {metro_bpm:.0f} BPM")
                else:
                    self.bpm_actual_label.setText("Metro: -- BPM")
            except RuntimeError:
                pass

        if event.is_beat:
            # Track beat time for auto-adjustment feature
            self._last_beat_time_for_auto = time.time()
            
            # ===== REAL-TIME METRIC FEEDBACK =====
            # Compute energy margin and apply metric-based adjustments
            if hasattr(self, 'audio_engine') and self.audio_engine is not None:
                # Get energy margin metric and apply feedback if enabled
                margin, should_adjust, direction = self.audio_engine.compute_energy_margin_feedback(
                    event.peak_energy, 
                    callback=self._on_metric_feedback
                )
                
                # Get BPS (beats per second) metric and adjust peak_floor to hit target
                actual_bps, bps_should_adjust, bps_direction = self.audio_engine.compute_bps_feedback(
                    event.timestamp,
                    callback=self._on_metric_feedback
                )
            
            # Light up the beat indicator (green for any beat)
            if hasattr(self, 'beat_indicator') and self.beat_indicator is not None:
                self.beat_indicator.setStyleSheet("color: #0f0; font-size: 24px;")
            # Reset timer to keep it lit for minimum duration
            if hasattr(self, 'beat_timer') and self.beat_timer is not None:
                self.beat_timer.stop()
                self.beat_timer.start(self.beat_indicator_min_duration)
            # Get tempo from audio engine (now includes smoothing, beat prediction, downbeat detection)
            if hasattr(self, 'audio_engine') and self.audio_engine is not None:
                tempo_info = self.audio_engine.get_tempo_info()
                if tempo_info['bpm'] > 0:
                    # Use event.is_downbeat (frozen at construction time) instead of
                    # polling get_tempo_info() which races with audio thread clearing the flag
                    is_downbeat = getattr(event, 'is_downbeat', False)
                    
                    # Light up downbeat indicator (cyan/blue for downbeat)
                    if is_downbeat:
                        if hasattr(self, 'downbeat_indicator') and self.downbeat_indicator is not None:
                            self.downbeat_indicator.setStyleSheet("color: #0ff; font-size: 24px;")
                        if hasattr(self, 'downbeat_timer') and self.downbeat_timer is not None:
                            self.downbeat_timer.stop()
                            self.downbeat_timer.start(self.beat_indicator_min_duration)
                        # Record downbeat for sensitivity metric
                        if hasattr(self, 'audio_engine') and self.audio_engine is not None:
                            pass  # downbeat recording removed
        # Show reset in GUI and console if tempo was reset
        if hasattr(event, 'tempo_reset') and event.tempo_reset:
            print("[GUI] Beat counter/tempo reset due to silence.")
    
    def _turn_off_beat_indicator(self):
        """Turn off beat indicator after minimum duration"""
        self.beat_indicator.setStyleSheet("color: #333; font-size: 24px;")
    
    def _turn_off_downbeat_indicator(self):
        """Turn off downbeat indicator after minimum duration"""
        self.downbeat_indicator.setStyleSheet("color: #333; font-size: 24px;")
    
    def _on_spectrum(self, spectrum: np.ndarray):
        """Queue spectrum for throttled update"""
        self._pending_spectrum = spectrum

    def _compute_visual_metrics(self, spectrum: np.ndarray) -> tuple[float, float]:
        """Compute visual peak/flux from spectrum only (independent of detection thresholds)."""
        arr = np.asarray(spectrum, dtype=np.float32)
        if arr.size == 0:
            return (0.0, 0.0)

        peak_raw = float(np.max(arr))
        if self._viz_prev_spectrum is None or len(self._viz_prev_spectrum) != len(arr):
            flux_raw = 0.0
        else:
            diff = arr - self._viz_prev_spectrum
            flux_raw = float(np.mean(np.maximum(0.0, diff)))
        self._viz_prev_spectrum = arr.copy()

        # Auto-reference with slow decay / fast capture for stable visual scaling
        self._viz_peak_ref = max(1e-4, self._viz_peak_ref * 0.995)
        self._viz_flux_ref = max(1e-5, self._viz_flux_ref * 0.995)
        if peak_raw > self._viz_peak_ref:
            self._viz_peak_ref = peak_raw
        if flux_raw > self._viz_flux_ref:
            self._viz_flux_ref = flux_raw

        peak_norm = float(np.clip(peak_raw / self._viz_peak_ref, 0.0, 1.0))
        flux_norm = float(np.clip(flux_raw / self._viz_flux_ref, 0.0, 1.0))
        return (peak_norm, flux_norm)
    
    def _do_spectrum_update(self):
        """Actually update spectrum at throttled rate - only update visible visualizer"""
        if self._pending_spectrum is not None:
            # Handle both old format (numpy array) and new format (dict with stats)
            if isinstance(self._pending_spectrum, dict):
                spectrum = self._pending_spectrum['spectrum']
                peak, flux = self._compute_visual_metrics(spectrum)
                waveform = self._pending_spectrum.get('waveform')
                sample_rate = int(self._pending_spectrum.get('sample_rate', getattr(self.config.audio, 'sample_rate', 44100)))
                # Only update the currently visible in-window visualizer for performance
                if hasattr(self, 'waveform_canvas') and self.waveform_canvas is not None and self.waveform_canvas.isVisible():
                    self.waveform_canvas.update_from_audio(waveform, sample_rate)
                elif hasattr(self, 'freqdb_canvas') and self.freqdb_canvas is not None and self.freqdb_canvas.isVisible():
                    self.freqdb_canvas.update_from_spectrum(spectrum, sample_rate)
                elif hasattr(self, 'fft_bin_canvas') and self.fft_bin_canvas is not None and self.fft_bin_canvas.isVisible():
                    self.fft_bin_canvas.update_from_spectrum(spectrum, sample_rate)
            else:
                # Legacy format - only update visible visualizer
                peak, flux = self._compute_visual_metrics(self._pending_spectrum)
                if hasattr(self, 'waveform_canvas') and self.waveform_canvas is not None and self.waveform_canvas.isVisible() and self.audio_engine is not None:
                    self.waveform_canvas.update_from_audio(self.audio_engine.get_waveform(), int(getattr(self.config.audio, 'sample_rate', 44100)))
                elif hasattr(self, 'freqdb_canvas') and self.freqdb_canvas is not None and self.freqdb_canvas.isVisible():
                    self.freqdb_canvas.update_from_spectrum(self._pending_spectrum, int(getattr(self.config.audio, 'sample_rate', 44100)))
                elif hasattr(self, 'fft_bin_canvas') and self.fft_bin_canvas is not None and self.fft_bin_canvas.isVisible():
                    self.fft_bin_canvas.update_from_spectrum(self._pending_spectrum, int(getattr(self.config.audio, 'sample_rate', 44100)))
            self._pending_spectrum = None
    
    def _on_status_change(self, message: str, connected: bool):
        """Update connection status"""
        self.status_label.setText("Connected" if connected else "Connect")
        self.status_label.setStyleSheet(f"color: {'#0af' if connected else '#fff'};")
        connection_toggle_action = getattr(self, 'connection_toggle_action', None)
        if connection_toggle_action is not None:
            connection_toggle_action.setText("Disconnect" if connected else "Connect")
        connection_test_action = getattr(self, 'connection_test_action', None)
        if connection_test_action is not None:
            connection_test_action.setEnabled(connected)

    def _get_effective_output_volume_percent(self) -> float:
        """Return the actual tcode volume percent last sent (includes silence fade, ramps)."""
        if not self.is_sending:
            return 0.0
        return float(self._last_sent_volume_pct)
    
    def _update_display(self):
        """Periodic display update + sync cached widget states for thread-safe audio access"""
        def _is_live_widget_attr(name: str) -> bool:
            widget = getattr(self, name, None)
            if widget is None:
                return False
            try:
                widget.parent()
            except RuntimeError:
                return False
            return True

        if self.stroke_mapper:
            alpha, beta = self.stroke_mapper.get_current_position()
            self.position_canvas.update_position(alpha, beta)
            self.alpha_label.setText(f"α: {alpha:.2f}")
            self.beta_label.setText(f"β: {beta:.2f}")

        # Sync widget states to cached values for thread-safe reading by audio thread.
        # Some controls may not exist yet (e.g., optional dialogs/tabs not instantiated),
        # so fall back to cached state instead of raising per-frame AttributeError.
        control_toggle_names = (
            'pulse_enabled_checkbox',
            'f0_enabled_checkbox',
            'p1_enabled_checkbox',
            'p3_enabled_checkbox',
        )
        controls_toggle_ready = all(_is_live_widget_attr(name) for name in control_toggle_names)
        if controls_toggle_ready:
            # P0/F0/P1/P3 enable state MUST be synced immediately (every frame) for instant response
            new_p0_enabled = self.pulse_enabled_checkbox.isChecked()
            new_f0_enabled = self.f0_enabled_checkbox.isChecked()
            new_p1_enabled = self.p1_enabled_checkbox.isChecked()
            new_p3_enabled = self.p3_enabled_checkbox.isChecked()
        else:
            new_p0_enabled = bool(getattr(self, '_cached_p0_enabled', False))
            new_f0_enabled = bool(getattr(self, '_cached_f0_enabled', False))
            new_p1_enabled = bool(getattr(self, '_cached_p1_enabled', False))
            new_p3_enabled = bool(getattr(self, '_cached_p3_enabled', False))
        
        # Handle P0/C0 checkboxes being unchecked (enabled→disabled transition)
        # Simply stop sending the axis — do NOT send 0 value, which still affects device
        if self._prev_p0_enabled and not new_p0_enabled:
            self._cached_p0_val = None
            self._cached_pulse_display = "Pulse Freq: off"
            self._p0_freq_window.clear()
            print("[Main] P0 disabled — stopped sending")
        if self._prev_f0_enabled and not new_f0_enabled:
            self._cached_f0_val = None
            self._cached_carrier_display = "Carrier Freq: off"
            self._f0_freq_window.clear()
            self._f0_last_sent_tcode = None
            print("[Main] C0 disabled — stopped sending")
        if self._prev_p1_enabled and not new_p1_enabled:
            self._cached_p1_val = None
            self._cached_p1_display = "Pulse Width: off"
            self._p1_window.clear()
            print("[Main] P1 disabled — stopped sending")
        if self._prev_p3_enabled and not new_p3_enabled:
            self._cached_p3_val = None
            self._cached_p3_display = "Rise Time: off"
            self._p3_window.clear()
            print("[Main] P3 disabled — stopped sending")
        
        self._prev_p0_enabled = new_p0_enabled
        self._prev_f0_enabled = new_f0_enabled
        self._prev_p1_enabled = new_p1_enabled
        self._prev_p3_enabled = new_p3_enabled
        self._cached_p0_enabled = new_p0_enabled
        self._cached_f0_enabled = new_f0_enabled
        self._cached_p1_enabled = new_p1_enabled
        self._cached_p3_enabled = new_p3_enabled
        
        # Update freq display labels — throttled to 100ms
        now = time.time()
        if now - self._last_freq_display_time > 0.1:
            self._last_freq_display_time = now
            # Update freq display labels from cached strings (written by audio thread)
            self.pulse_freq_label.setText(self._cached_pulse_display)
            self.carrier_freq_label.setText(self._cached_carrier_display)
            self.p1_display_label.setText(self._cached_p1_display)
            self.p3_display_label.setText(self._cached_p3_display)
            self.pulse_freq_label.setStyleSheet(f"color: {'#fff' if new_p0_enabled else '#0af'}; font-size: 10px;")
            self.carrier_freq_label.setStyleSheet(f"color: {'#fff' if new_f0_enabled else '#0af'}; font-size: 10px;")
            self.p1_display_label.setStyleSheet(f"color: {'#fff' if new_p1_enabled else '#0af'}; font-size: 10px;")
            self.p3_display_label.setStyleSheet(f"color: {'#fff' if new_p3_enabled else '#0af'}; font-size: 10px;")
            # Show target volume when stopped, actual sent tcode volume when running
            if self.is_sending:
                display_pct = self._last_sent_volume_pct
                self.volume_slider.value_label.setStyleSheet("color: #fff;")
                self.volume_slider.label.setStyleSheet("color: #fff;")
            else:
                display_pct = float(self.volume_slider.value())
                self.volume_slider.value_label.setStyleSheet("color: #0af;")
                self.volume_slider.label.setStyleSheet("color: #0af;")
            self.volume_slider.value_label.setText(f"{display_pct:.0f}")
            control_sync_names = (
                'pulse_mode_combo', 'pulse_invert_checkbox',
                'f0_mode_combo', 'f0_invert_checkbox',
                'tcode_freq_range_slider', 'f0_tcode_range_slider',
                'p1_mode_combo', 'p1_invert_checkbox',
                'p1_tcode_range_slider', 'p1_monitor_range_slider', 'p1_weight_slider',
                'p3_mode_combo', 'p3_invert_checkbox',
                'p3_tcode_range_slider', 'p3_monitor_range_slider', 'p3_weight_slider',
            )
            if all(_is_live_widget_attr(name) for name in control_sync_names):
                # Sync other combo/checkbox states for audio thread (throttled is fine)
                self._cached_pulse_mode = self.pulse_mode_combo.currentIndex()
                self._cached_pulse_invert = self.pulse_invert_checkbox.isChecked()
                self._cached_f0_mode = self.f0_mode_combo.currentIndex()
                self._cached_f0_invert = self.f0_invert_checkbox.isChecked()
                # Sync TCode Sent slider values for thread-safe access
                self._cached_tcode_freq_min = int(self.tcode_freq_range_slider.low())
                self._cached_tcode_freq_max = int(self.tcode_freq_range_slider.high())
                self._cached_f0_tcode_min = int(self.f0_tcode_range_slider.low())
                self._cached_f0_tcode_max = int(self.f0_tcode_range_slider.high())
                # Sync P1 (Pulse Width) widget states
                self._cached_p1_mode = self.p1_mode_combo.currentIndex()
                self._cached_p1_invert = self.p1_invert_checkbox.isChecked()
                self._cached_p1_tcode_min = int(self.p1_tcode_range_slider.low())
                self._cached_p1_tcode_max = int(self.p1_tcode_range_slider.high())
                self.config.pulse_width.monitor_freq_min = self.p1_monitor_range_slider.low()
                self.config.pulse_width.monitor_freq_max = self.p1_monitor_range_slider.high()
                self.config.pulse_width.weight = self.p1_weight_slider.value()
                # Sync P3 (Rise Time) widget states
                self._cached_p3_mode = self.p3_mode_combo.currentIndex()
                self._cached_p3_invert = self.p3_invert_checkbox.isChecked()
                self._cached_p3_tcode_min = int(self.p3_tcode_range_slider.low())
                self._cached_p3_tcode_max = int(self.p3_tcode_range_slider.high())
                self.config.rise_time.monitor_freq_min = self.p3_monitor_range_slider.low()
                self.config.rise_time.monitor_freq_max = self.p3_monitor_range_slider.high()
                self.config.rise_time.weight = self.p3_weight_slider.value()
            
            # Update peak floor bars on all visualizers
            peak_floor = self.config.beat.peak_floor
            for canvas_name in ['waveform_canvas', 'freqdb_canvas', 'fft_bin_canvas']:
                if hasattr(self, canvas_name):
                    canvas = getattr(self, canvas_name)
                    if hasattr(canvas, 'set_peak_floor'):
                        canvas.set_peak_floor(peak_floor)

        # Handle volume ramp completion
        if self._volume_ramp_active:
            elapsed = time.time() - self._volume_ramp_start_time
            if elapsed >= self._volume_ramp_duration:
                self._volume_ramp_active = False
    
        # ===== TIMER-DRIVEN METRIC FEEDBACK: Audio Amp =====
        # These fire from the display timer (not from _on_beat) so they can
        # detect the ABSENCE of beats and escalate accordingly.
        if hasattr(self, 'audio_engine') and self.audio_engine is not None:
            now = time.perf_counter()
            self.audio_engine.compute_audio_amp_feedback(now, callback=self._on_metric_feedback)
            self.audio_engine.compute_flux_balance_feedback(now, callback=self._on_metric_feedback)
            
            # ===== AUTO-ALIGN TARGET BPM (time-based) =====
            if self._auto_align_target_enabled:
                tempo_info = self.audio_engine.get_tempo_info()
                sensed_bpm = tempo_info.get('stable_bpm', 0.0)
                if sensed_bpm <= 0:
                    sensed_bpm = tempo_info.get('bpm', 0.0)
                    
                if sensed_bpm > 30 and sensed_bpm < 240:  # Valid BPM range
                    # Check if tempo is stable (stability > 0.5 or locked via downbeat matching)
                    stability = tempo_info.get('stability', 0.0)
                    consecutive_downbeats = tempo_info.get('consecutive_matching_downbeats', 0)
                    locked = consecutive_downbeats >= 3
                    
                    if stability > 0.5 or locked:
                        # Tempo is stable — start or continue timing
                        if not self._auto_align_is_stable:
                            self._auto_align_is_stable = True
                            self._auto_align_stable_since = now
                        self._last_sensed_bpm = sensed_bpm
                    else:
                        # Tempo unstable — reset timer immediately
                        self._auto_align_is_stable = False
                        self._auto_align_stable_since = 0.0
                    
                    acf_conf = float(np.clip(tempo_info.get('acf_confidence', 0.0), 0.0, 1.0))
                    conf_boost = float(np.clip((acf_conf - 0.25) / 0.45, 0.0, 1.0))
                    adaptive_required_seconds = self._auto_align_required_seconds * (1.0 - 0.50 * conf_boost)
                    adaptive_cooldown = self._auto_align_cooldown * (1.0 - 0.67 * conf_boost)

                    # Check if stable long enough to start aligning
                    if (self._auto_align_is_stable and 
                            (now - self._auto_align_stable_since) >= adaptive_required_seconds):
                        target_bpm_spin = getattr(self, 'target_bpm_spin', None)
                        if target_bpm_spin is not None:
                            current_target = target_bpm_spin.value()
                            diff = sensed_bpm - current_target
                            
                            # Only align if difference >= 1 BPM AND cooldown elapsed
                            if abs(diff) >= 1.0 and (now - self._auto_align_last_adjust_time) >= adaptive_cooldown:
                                step_bpm = 1
                                if acf_conf >= 0.55 and abs(diff) >= 4.0:
                                    step_bpm = 2
                                if acf_conf >= 0.70 and abs(diff) >= 8.0:
                                    step_bpm = 3
                                if diff > 0:
                                    new_target = min(int(current_target) + step_bpm, int(sensed_bpm))
                                else:
                                    new_target = max(int(current_target) - step_bpm, int(np.ceil(sensed_bpm)))
                                
                                if new_target != int(current_target):
                                    target_bpm_spin.setValue(new_target)
                                    self._auto_align_last_adjust_time = now
                                    stable_elapsed = now - self._auto_align_stable_since
                                    print(
                                        f"[Auto-align] Target BPM: {int(current_target)} -> {new_target} "
                                        f"(sensed: {sensed_bpm:.1f}, conf: {acf_conf:.2f}, stable for {stable_elapsed:.1f}s, step: {step_bpm})"
                                    )
                else:
                    # Invalid BPM, reset stability
                    self._auto_align_is_stable = False
                    self._auto_align_stable_since = 0.0
            
            # Keep metric state polling active even though traffic-light UI is removed.
            self.audio_engine.get_metric_states()

    def closeEvent(self, event):
        """Cleanup on close - ensure all threads are stopped before UI is destroyed"""
        shutdown_runtime(self._stop_engines, self.network_engine)

        persist_runtime_ui_to_config(self, self.config)
        
        # Save config before closing
        save_config(self.config)

        event.accept()


def main():
    """Main entry point - backup if not launched via run.py"""
    log_event("INFO", "Startup", "Application launch", frozen=bool(getattr(sys, "frozen", False)), debug_stdio=_DEBUG_STDIO_ENABLED)
    instance_lock = _SingleInstanceLock()
    if not instance_lock.acquire():
        log_event("WARN", "Startup", "Another instance detected; exiting")
        print("[Startup] Another bREadbeats instance is already running. Exiting.", flush=True)
        return

    atexit.register(instance_lock.release)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Show splash screen while loading (fallback for direct main.py execution)
    if getattr(sys, 'frozen', False):
        resource_dir = Path(getattr(sys, '_MEIPASS', Path(__file__).parent))
    else:
        resource_dir = Path(__file__).parent
    
    splash_path = resource_dir / 'splash_screen.png'
    if splash_path.exists():
        pixmap = QPixmap(str(splash_path))
        splash = QSplashScreen(pixmap)
        splash.show()
        app.processEvents()
    else:
        splash = None
    
    # Create main window
    window = BREadbeatsWindow()
    
    print("\nInitialization complete. Starting GUI...\n")
    if sys.stdout:
        sys.stdout.flush()
    
    # Close splash and show main window
    if splash:
        splash.finish(window)
    
    window.show()
    
    try:
        sys.exit(app.exec())
    finally:
        log_event("INFO", "Shutdown", "Application exit")
        instance_lock.release()


if __name__ == "__main__":
    main()
