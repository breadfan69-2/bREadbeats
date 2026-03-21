"""
bREadbeats - Custom Widget Classes
Extracted from main.py for modularization.
"""

import time
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout,
    QLabel, QScrollArea, QSizePolicy, QSlider,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRectF, QEvent
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen
import pyqtgraph as pg
from typing import Optional


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


    def set_peak_indicators_visible(self, visible: bool):
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

        # Keyboard-teaching preview marker (ghost intent dot)
        self.teacher_preview_scatter = pg.ScatterPlotItem(
            size=10,
            brush=pg.mkBrush('#ffd54f'),
            pen=pg.mkPen('#ffb300', width=1),
        )
        self.addItem(self.teacher_preview_scatter)

        self._last_x_display = 0.0
        self._last_y_display = 0.0
        self._ghost_theta: Optional[float] = None
        self._ghost_radius: float = 0.70
        self._ghost_last_real_theta: Optional[float] = None
        self._ghost_last_update_time: float = time.perf_counter()
        self._ghost_last_speed_scale: float = 1.0
        self._ghost_is_parked: bool = False
        
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
        self._last_x_display = float(x_display)
        self._last_y_display = float(y_display)

    # Fixed orbit radius for the ghost — decoupled from device position so it's
    # always visible regardless of whether the device is parked/silenced.
    _GHOST_ORBIT_RADIUS: float = 0.65

    def update_teacher_preview(self, is_parked: bool, speed_scale: float, bpm: float):
        """Show a ghost dot that orbits at a fixed radius to preview teaching speed.

        - Orbits CCW at base_omega = 2π * bpm / 240  rad/s  (1 rotation per measure).
        - speed_scale multiplies that angular velocity.
        - is_parked: ghost freezes at its current angle (shows a stationary dot).
        - Ghost radius is fixed at _GHOST_ORBIT_RADIUS; decoupled from device position.
        """
        import math
        now = time.perf_counter()
        dt = float(np.clip(now - self._ghost_last_update_time, 1e-3, 0.20))
        self._ghost_last_update_time = now

        if self._ghost_theta is None:
            self._ghost_theta = 0.0   # start at right edge (3 o'clock)

        self._ghost_is_parked = bool(is_parked)
        self._ghost_last_speed_scale = float(speed_scale)

        if not is_parked:
            safe_bpm = max(float(bpm), 20.0)
            base_omega = 2.0 * math.pi * safe_bpm / 240.0   # rad/s
            self._ghost_theta = float(
                self._ghost_theta + base_omega * float(speed_scale) * dt
            )

        preview_x = self._GHOST_ORBIT_RADIUS * float(np.cos(self._ghost_theta))
        preview_y = self._GHOST_ORBIT_RADIUS * float(np.sin(self._ghost_theta))
        self.teacher_preview_scatter.setData([preview_x], [preview_y])

    def clear_teacher_preview(self):
        self.teacher_preview_scatter.setData([], [])


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


class _NoWheelSlider(QSlider):
    """QSlider that ignores mouse-wheel so parent scroll areas can scroll."""
    def wheelEvent(self, event):
        event.ignore()


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
        self._last_value = float(default)
        
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
        
        self.slider = _NoWheelSlider(Qt.Orientation.Horizontal)
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
        self._last_value = real_value
        self.value_label.setText(f"{real_value:.{self.decimals}f}")
        self.valueChanged.emit(real_value)
        
    def value(self) -> float:
        try:
            self._last_value = self.slider.value() / self.multiplier
        except RuntimeError:
            pass
        return float(self._last_value)
    
    def setValue(self, value: float):
        self._last_value = float(value)
        try:
            self.slider.setValue(int(value * self.multiplier))
        except RuntimeError:
            return


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


class NoWheelScrollArea(QScrollArea):
    """
    Custom QScrollArea whose sliders ignore wheel events, but the
    scroll area itself scrolls normally on mouse wheel.
    Sliders block their own wheel via _NoWheelSlider.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def wheelEvent(self, event):
        super().wheelEvent(event)


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


