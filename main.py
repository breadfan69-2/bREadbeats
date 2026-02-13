"""
bREadbeats - Main Application
Qt GUI with beat detection, stroke mapping, and spectrum visualization.
"""

# Heavy imports - these are the slow ones, but splash is already showing by this point
import sys
from contextlib import contextmanager
import time

_import_t0 = time.perf_counter()
print("\n[Startup] main.py loading heavy modules...", flush=True)

import numpy as np
import queue
import threading
import os
import random
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSlider, QComboBox, QPushButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QTabWidget, QFrame,
    QGridLayout, QMenuBar, QMenu, QMessageBox, QFileDialog,
    QSplashScreen, QScrollArea, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QPixmap
from typing import Optional

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
from logging_utils import get_log_level, set_log_level
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
from presets_wiring import get_presets_file_path, load_presets_data, resolve_p0_tcode_bounds, save_presets_data
from slider_tuning_tracker import SliderTuningTracker
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


_active_slider_tracker: Optional[SliderTuningTracker] = None


def _set_active_slider_tracker(tracker: Optional[SliderTuningTracker]) -> None:
    global _active_slider_tracker
    _active_slider_tracker = tracker


def _track_slider_value(name: str, value: float) -> None:
    if _active_slider_tracker is None:
        return
    try:
        _active_slider_tracker.record_value(name, value)
    except Exception:
        pass


_apply_dict_to_dataclass = apply_dict_to_dataclass
_migrate_config = migrate_config


class SignalBridge(QObject):
    """Bridge for thread-safe signal emission"""
    beat_detected = pyqtSignal(object)
    spectrum_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str, bool)


class SpectrumCanvas(pg.PlotWidget):
    """Waterfall spectrogram visualizer using PyQtGraph ImageItem - scrolling history"""
    
    def __init__(self, parent=None, width=8, height=3):
        super().__init__(parent)
        
        # Dark theme matching restim-coyote3
        self.setBackground('#232323')
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        # No gridlines
        self.showGrid(x=False, y=False, alpha=0)
        self.hideAxis('left')
        self.hideAxis('bottom')
        
        # Waterfall dimensions
        # For waterfall: X = frequency (horizontal), Y = time (vertical, scrolls up)
        self.num_bins = 256  # Frequency bins (horizontal)
        self.history_len = 100  # Time history (vertical)
        
        # Waterfall data buffer (time x freq) - new data goes at bottom, scrolls up
        self.waterfall_data = np.zeros((self.history_len, self.num_bins), dtype=np.float32)
        
        # Create ImageItem for waterfall display
        self.img_item = pg.ImageItem()
        self.addItem(self.img_item)
        
        # Colormap: black -> blue -> cyan -> green -> yellow -> orange -> dark purple (no white)
        colors = [
            (0, 0, 0),        # Black (silence)
            (0, 0, 80),       # Dark blue
            (0, 50, 150),     # Blue
            (0, 150, 150),    # Cyan
            (0, 200, 50),     # Green
            (200, 200, 0),    # Yellow
            (255, 100, 0),    # Orange
            (40, 0, 40),      # Dark purple (highest amplitude)
        ]
        positions = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
        self.colormap = pg.ColorMap(positions, colors)
        lut = self.colormap.getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(lut)  # type: ignore
        
        # Set view range: X = freq bins, Y = time samples
        # Left margin includes peak indicator bars at negative X
        self.setXRange(-14, self.num_bins)
        self.setYRange(0, self.history_len)
        
        # 4 Frequency band indicators (vertical regions with different heights)
        # Heights staggered for visibility - labels placed at top of each band to stack
        # Band heights: beat=100%, stroke=92%, pulse=84%, carrier=76%
        
        # Band 1: Beat Detection (red) - full height (tallest)
        self.beat_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                              brush=pg.mkBrush(255, 50, 50, 60),
                                              pen=pg.mkPen('#888888', width=2),
                                              movable=True, span=(0, 1.0))
        self.beat_band.setBounds([0, self.num_bins])
        self.beat_band.sigRegionChanged.connect(self._on_beat_band_changed)
        self.addItem(self.beat_band)
        
        # Band 2: Stroke Depth (green) - 92% height
        self.depth_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                               brush=pg.mkBrush(50, 255, 50, 50),
                                               pen=pg.mkPen('#888888', width=2),
                                               movable=True, span=(0, 0.92))
        self.depth_band.setBounds([0, self.num_bins])
        self.depth_band.sigRegionChanged.connect(self._on_depth_band_changed)
        self.addItem(self.depth_band)
        
        # Band 3: Pulse/P0 TCode (blue) - 84% height
        self.p0_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                            brush=pg.mkBrush(50, 100, 255, 50),
                                            pen=pg.mkPen('#888888', width=2),
                                            movable=True, span=(0, 0.84))
        self.p0_band.setBounds([0, self.num_bins])
        self.p0_band.sigRegionChanged.connect(self._on_p0_band_changed)
        self.addItem(self.p0_band)
        
        # Band 4: Carrier/F0 TCode (cyan) - 76% height (shortest)
        self.f0_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                            brush=pg.mkBrush(0, 200, 255, 50),
                                            pen=pg.mkPen('#888888', width=2),
                                            movable=True, span=(0, 0.76))
        self.f0_band.setBounds([0, self.num_bins])
        self.f0_band.sigRegionChanged.connect(self._on_f0_band_changed)
        self.addItem(self.f0_band)
        
        # Labels placed at top of each respective band
        # Each label sits just inside the upper limit of its own box
        self.carrier_label = pg.TextItem("carrier", color='#00C8FF', anchor=(0.5, 1))
        self.carrier_label.setPos(5, self.history_len * 0.73)
        self.addItem(self.carrier_label)
        
        self.pulse_label = pg.TextItem("pulse", color='#3264FF', anchor=(0.5, 1))
        self.pulse_label.setPos(5, self.history_len * 0.81)
        self.addItem(self.pulse_label)
        
        self.depth_label = pg.TextItem("stroke", color='#32FF32', anchor=(0.5, 1))
        self.depth_label.setPos(5, self.history_len * 0.89)
        self.addItem(self.depth_label)
        
        self.beat_label = pg.TextItem("beat", color='#FF3232', anchor=(0.5, 1))
        self.beat_label.setPos(5, self.history_len * 0.97)
        self.addItem(self.beat_label)
        
        # Reference to parent window for slider updates
        self.parent_window = parent
        self.sample_rate = 44100  # Will be updated
        self._fft_bins = 513  # Default for 1024 FFT; updated on first spectrum
        self._updating = False  # Prevent recursion when setting bands
        
        # Peak indicator vertical bars on left side (3 thin bars: actual peak, peak floor, peak decay)
        # Positioned at negative X to be off to the left of main display
        bar_width = 4.6  # Width of each bar (about 8% thinner)
        bar_spacing = 0.8  # Gap between bars
        bar_x_start = -18.0  # Start position (leftmost bar)
        
        # Bar 1: Actual Peak (green)
        self.peak_actual_bar = pg.BarGraphItem(
            x=[bar_x_start],
            height=[0],
            width=bar_width,
            brush='#00FF00'
        )
        self.addItem(self.peak_actual_bar)
        
        # Bar 2: Peak Floor (yellow)
        self.peak_floor_bar = pg.BarGraphItem(
            x=[bar_x_start + bar_width + bar_spacing],
            height=[0],
            width=bar_width,
            brush='#FFD700'
        )
        self.addItem(self.peak_floor_bar)
        
        # Bar 3: Peak Decay (orange)
        self.peak_decay_bar = pg.BarGraphItem(
            x=[bar_x_start + 2 * (bar_width + bar_spacing)],
            height=[0],
            width=bar_width,
            brush='#FF8C00'
        )
        self.addItem(self.peak_decay_bar)
        
        # Store bar positions
        self._bar_width = bar_width
        self._bar_scale = self.history_len * 0.82  # Visual calibration to reduce top saturation
        
    def _hz_to_bin(self, hz: float) -> float:
        """Convert Hz to log-spaced bin index (0 to num_bins)"""
        nyquist = self.sample_rate / 2
        n = self._fft_bins if hasattr(self, '_fft_bins') and self._fft_bins > 1 else self.num_bins
        # Hz -> linear FFT index
        linear_idx = max(1.0, (hz / nyquist) * (n - 1))
        # Linear index -> log-spaced bin: inverse of logspace(0, log10(n-1), num_bins)
        log_max = np.log10(n - 1)
        return (np.log10(linear_idx) / log_max) * self.num_bins if log_max > 0 else 0.0
    
    def _bin_to_hz(self, bin_idx: float) -> float:
        """Convert log-spaced bin index to Hz"""
        nyquist = self.sample_rate / 2
        n = self._fft_bins if hasattr(self, '_fft_bins') and self._fft_bins > 1 else self.num_bins
        # Log-spaced bin -> linear FFT index: apply logspace mapping
        log_max = np.log10(n - 1)
        linear_idx = 10 ** ((bin_idx / self.num_bins) * log_max) if log_max > 0 else bin_idx
        # Linear index -> Hz
        return (linear_idx / (n - 1)) * nyquist
        
    def _on_beat_band_changed(self):
        """Handle beat detection band dragging"""
        if self._updating:
            return
        region = self.beat_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'freq_range_slider'):
            self._updating = True
            self.parent_window.freq_range_slider.setLow(int(low_hz))
            self.parent_window.freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        # Update label position to center of band (stacked)
        center_bin = (region[0] + region[1]) / 2  # type: ignore
        self.beat_label.setPos(center_bin, self.history_len * 0.97)
    
    def _on_depth_band_changed(self):
        """Handle stroke depth band dragging"""
        if self._updating:
            return
        region = self.depth_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'depth_freq_range_slider'):
            self._updating = True
            self.parent_window.depth_freq_range_slider.setLow(int(low_hz))
            self.parent_window.depth_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        # Update label position to center of band
        center_bin = (region[0] + region[1]) / 2  # type: ignore
        self.depth_label.setPos(center_bin, self.history_len * 0.89)
    
    def _on_p0_band_changed(self):
        """Handle P0 TCode band dragging"""
        if self._updating:
            return
        region = self.p0_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'pulse_freq_range_slider'):
            self._updating = True
            self.parent_window.pulse_freq_range_slider.setLow(int(low_hz))
            self.parent_window.pulse_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        # Update label position to center of band (above the band)
        center_bin = (region[0] + region[1]) / 2  # type: ignore
        self.pulse_label.setPos(center_bin, self.history_len * 0.81)
    
    def _on_f0_band_changed(self):
        """Handle F0 (carrier) band dragging"""
        if self._updating:
            return
        region = self.f0_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'f0_freq_range_slider'):
            self._updating = True
            self.parent_window.f0_freq_range_slider.setLow(int(low_hz))
            self.parent_window.f0_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (region[0] + region[1]) / 2  # type: ignore
        self.carrier_label.setPos(center_bin, self.history_len * 0.73)
    
    def set_sample_rate(self, sr: int):
        """Update sample rate for frequency calculations"""
        self.sample_rate = sr
    
    def set_frequency_band(self, low_norm: float, high_norm: float):
        """Update beat detection band (normalized 0-1)"""
        self._updating = True
        nyquist = self.sample_rate / 2
        low_bin = self._hz_to_bin(low_norm * nyquist)
        high_bin = self._hz_to_bin(high_norm * nyquist)
        self.beat_band.setRegion((low_bin, high_bin))
        # Update label position to center of band
        center_bin = (low_bin + high_bin) / 2
        self.beat_label.setPos(center_bin, self.history_len * 0.97)
        self._updating = False
    
    def set_depth_band(self, low_hz: float, high_hz: float):
        """Update stroke depth band (in Hz)"""
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.depth_band.setRegion((low_bin, high_bin))
        # Update label position to center of band
        center_bin = (low_bin + high_bin) / 2
        self.depth_label.setPos(center_bin, self.history_len * 0.89)
        self._updating = False
    
    def set_p0_band(self, low_hz: float, high_hz: float):
        """Update P0 TCode band (in Hz)"""
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.p0_band.setRegion((low_bin, high_bin))
        # Update label position to center of band
        center_bin = (low_bin + high_bin) / 2
        self.pulse_label.setPos(center_bin, self.history_len * 0.81)
        self._updating = False
    
    def set_f0_band(self, low_hz: float, high_hz: float):
        """Update F0 (carrier) TCode band (in Hz)"""
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.f0_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.carrier_label.setPos(center_bin, self.history_len * 0.73)
        self._updating = False
    
    def set_peak_and_flux(self, peak_value: float, flux_value: float):
        """Update peak indicator bars - actual peak and peak decay"""
        if hasattr(self, 'peak_actual_bar'):
            # Scale to waterfall Y range (0 to history_len)
            scale = getattr(self, '_bar_scale', self.history_len)
            self.peak_actual_bar.setOpts(height=[min(1.0, peak_value) * scale])
        if hasattr(self, 'peak_decay_bar'):
            scale = getattr(self, '_bar_scale', self.history_len)
            self.peak_decay_bar.setOpts(height=[min(1.0, flux_value) * scale])
    
    def set_peak_floor(self, peak_floor: float):
        """Update peak floor bar height"""
        if hasattr(self, 'peak_floor_bar'):
            scale = getattr(self, '_bar_scale', self.history_len)
            self.peak_floor_bar.setOpts(height=[peak_floor * scale])
    
    def set_peak_indicators_visible(self, visible: bool):
        """Show or hide peak indicator bars (peak_actual, peak_floor, peak_decay)"""
        if hasattr(self, 'peak_actual_bar'):
            self.peak_actual_bar.setVisible(visible)
        if hasattr(self, 'peak_floor_bar'):
            self.peak_floor_bar.setVisible(visible)
        if hasattr(self, 'peak_decay_bar'):
            self.peak_decay_bar.setVisible(visible)

    def set_range_indicators_visible(self, visible: bool):
        """Show or hide frequency range band indicators and labels"""
        self.beat_band.setVisible(visible)
        self.depth_band.setVisible(visible)
        self.p0_band.setVisible(visible)
        self.f0_band.setVisible(visible)
        self.beat_label.setVisible(visible)
        self.depth_label.setVisible(visible)
        self.pulse_label.setVisible(visible)
        self.carrier_label.setVisible(visible)
        
    def update_spectrum(self, spectrum: np.ndarray, peak_energy: Optional[float] = None, spectral_flux: Optional[float] = None):
        """Update waterfall with new spectrum data - scrolls upward with rainbow frequency colors"""
        if spectrum is None or len(spectrum) == 0:
            return
        
        # Use full Nyquist range (no cutoff)
        # Track source FFT size for hz/bin conversions
        self._fft_bins = len(spectrum)
        # Resample to num_bins using log-frequency spacing
        n = len(spectrum)
        if n > 1:
            x_old = np.arange(n)
            # Log-spaced indices: more resolution at low frequencies
            x_new = np.logspace(0, np.log10(n - 1), self.num_bins)
            x_new = np.clip(x_new, 0, n - 1)
            spectrum = np.interp(x_new, x_old, spectrum)
        elif len(spectrum) != self.num_bins:
            x_old = np.linspace(0, 1, n)
            x_new = np.linspace(0, 1, self.num_bins)
            spectrum = np.interp(x_new, x_old, spectrum)
        
        # Apply log scaling for better dynamic range visualization
        spectrum = np.log10(spectrum + 1e-6)
        # Normalize: full range -6 to 0 for normalized input, map to 0-1
        spectrum = np.clip((spectrum + 6) / 6, 0, 1)
        
        # Scroll waterfall up: shift all rows up by 1, put new data at bottom (row 0)
        self.waterfall_data[1:, :] = self.waterfall_data[:-1, :]
        self.waterfall_data[0, :] = spectrum
        
        # Create RGB image with frequency-based hue and amplitude-based brightness
        # Pre-compute base colors for each frequency bin if not done
        if not hasattr(self, '_freq_colors'):
            self._freq_colors = np.zeros((self.num_bins, 3), dtype=np.float32)
            for i in range(self.num_bins):
                hue = (i / self.num_bins) * 270  # 0-270 degrees (red->blue)
                h = hue / 60.0
                x_val = 1.0 - abs(h % 2 - 1)
                if h < 1:
                    r, g, b = 1.0, x_val, 0.0
                elif h < 2:
                    r, g, b = x_val, 1.0, 0.0
                elif h < 3:
                    r, g, b = 0.0, 1.0, x_val
                elif h < 4:
                    r, g, b = 0.0, x_val, 1.0
                else:
                    r, g, b = x_val, 0.0, 1.0
                self._freq_colors[i] = [r, g, b]
        
        # Vectorized: brightness = waterfall_data ^ 0.7 for gamma correction
        brightness = self.waterfall_data ** 0.7
        
        # Shape: (history, bins, 3) - multiply brightness by pre-computed colors
        # brightness is (history, bins), colors is (bins, 3)
        # Result: (history, bins, 3)
        rgb_data = (brightness[:, :, np.newaxis] * self._freq_colors[np.newaxis, :, :] * 255).astype(np.uint8)
        
        # Update image - transpose so X=freq, Y=time (new at bottom)
        # For RGB: shape should be (width, height, 3) so transpose first two axes
        self.img_item.setImage(rgb_data.transpose(1, 0, 2), autoLevels=False)
        # Position image to fill the view exactly (must be after setImage)
        self.img_item.setRect(0, 0, self.num_bins, self.history_len)
        
        # Update peak indicator bars with current peak energy
        if peak_energy is not None:
            self.set_peak_and_flux(peak_energy, spectral_flux or 0.0)


class MountainRangeCanvas(pg.PlotWidget):
    """Mountain range spectrum visualizer - glowing blue filled peaks"""
    
    def __init__(self, parent=None, width=8, height=3):
        super().__init__(parent)
        
        # Dark theme
        self.setBackground('#0a0a12')  # Very dark blue-black
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.showGrid(x=False, y=False, alpha=0)
        self.showAxis('left')
        self.getAxis('left').setTextPen(pg.mkPen('#888888'))
        self.getAxis('left').setTickPen(pg.mkPen('#666666'))
        self.hideAxis('bottom')
        
        # Spectrum dimensions
        self.num_bins = 256
        
        # Frequency values for x-axis (will be updated with sample rate)
        self.freq_values = np.linspace(0, self.num_bins, self.num_bins)
        
        # Set view range (left margin includes peak indicator bars at negative X)
        self.setXRange(-16, self.num_bins)
        self.setYRange(-120, 0)

        self._carrier_label_y = -33
        self._pulse_label_y = -24
        self._depth_label_y = -15
        self._beat_label_y = -6
        
        # Main spectrum curve (mountain peaks) - cyan fill with bright outline
        self.spectrum_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QColor(100, 200, 255, 255), width=2),  # Bright cyan outline
            fillLevel=-120,
            brush=pg.mkBrush(QColor(0, 120, 200, 120))  # Semi-transparent cyan fill
        )
        self.addItem(self.spectrum_curve)
        
        # Glow effect - slightly larger, more transparent version behind
        self.glow_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QColor(0, 150, 255, 80), width=6),
            fillLevel=-120,
            brush=pg.mkBrush(QColor(0, 80, 180, 40))
        )
        self.addItem(self.glow_curve)
        self.glow_curve.setZValue(-1)  # Behind the main curve
        
        # Peak markers for beat detection visualization
        self.peak_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 255, 255, 200),
            size=6
        )
        self.addItem(self.peak_scatter)
        
        # 4 Frequency band indicators (vertical regions with different heights)
        # Band heights: beat=100%, stroke=92%, pulse=84%, carrier=76%
        # Band 1: Beat Detection (red) - full height (tallest)
        self.beat_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                              brush=pg.mkBrush(255, 50, 50, 40),
                                              pen=pg.mkPen('#FF3232', width=1),
                                              movable=True, span=(0, 1.0))
        self.beat_band.setBounds([0, self.num_bins])
        self.beat_band.sigRegionChanged.connect(self._on_beat_band_changed)
        self.addItem(self.beat_band)
        
        # Band 2: Stroke Depth (green) - 92% height
        self.depth_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                               brush=pg.mkBrush(50, 255, 50, 35),
                                               pen=pg.mkPen('#32FF32', width=1),
                                               movable=True, span=(0, 0.92))
        self.depth_band.setBounds([0, self.num_bins])
        self.depth_band.sigRegionChanged.connect(self._on_depth_band_changed)
        self.addItem(self.depth_band)
        
        # Band 3: Pulse/P0 TCode (blue) - 84% height
        self.p0_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                            brush=pg.mkBrush(50, 100, 255, 35),
                                            pen=pg.mkPen('#3264FF', width=1),
                                            movable=True, span=(0, 0.84))
        self.p0_band.setBounds([0, self.num_bins])
        self.p0_band.sigRegionChanged.connect(self._on_p0_band_changed)
        self.addItem(self.p0_band)
        
        # Band 4: Carrier/F0 TCode (cyan) - 76% height (shortest)
        self.f0_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                            brush=pg.mkBrush(0, 200, 255, 35),
                                            pen=pg.mkPen('#00C8FF', width=1),
                                            movable=True, span=(0, 0.76))
        self.f0_band.setBounds([0, self.num_bins])
        self.f0_band.sigRegionChanged.connect(self._on_f0_band_changed)
        self.addItem(self.f0_band)
        
        # Labels placed at top of each respective band
        # Each label sits just inside the upper limit of its own box
        self.carrier_label = pg.TextItem("carrier", color='#00C8FF', anchor=(0.5, 1))
        self.carrier_label.setPos(5, self._carrier_label_y)
        self.addItem(self.carrier_label)
        
        self.pulse_label = pg.TextItem("pulse", color='#3264FF', anchor=(0.5, 1))
        self.pulse_label.setPos(5, self._pulse_label_y)
        self.addItem(self.pulse_label)
        
        self.depth_label = pg.TextItem("stroke", color='#32FF32', anchor=(0.5, 1))
        self.depth_label.setPos(5, self._depth_label_y)
        self.addItem(self.depth_label)
        
        self.beat_label = pg.TextItem("beat", color='#FF3232', anchor=(0.5, 1))
        self.beat_label.setPos(5, self._beat_label_y)
        self.addItem(self.beat_label)
        
        # Reference to parent window
        self.parent_window = parent
        self.sample_rate = 44100
        self._fft_bins = 513  # Default for 1024 FFT; updated on first spectrum
        self._updating = False
        
        # Peak indicator vertical bars on left side (3 thin bars: actual peak, peak floor, peak decay)
        # These are positioned at negative X to be off to the left of the main spectrum display
        bar_width = 4.4  # Width of each bar (10% wider for parity)
        bar_spacing = 0.8  # Gap between bars
        bar_x_start = -15.0  # Start position (leftmost bar)
        
        # Bar 1: Actual Peak (green) - current band energy level
        self.peak_actual_bar = pg.BarGraphItem(
            x=[bar_x_start],
            y0=[-120],
            height=[0],
            width=bar_width,
            brush='#00FF00'  # Green
        )
        self.addItem(self.peak_actual_bar)
        
        # Bar 2: Peak Floor (yellow) - threshold setting
        self.peak_floor_bar = pg.BarGraphItem(
            x=[bar_x_start + bar_width + bar_spacing],
            y0=[-120],
            height=[0],
            width=bar_width,
            brush='#FFD700'  # Gold/Yellow
        )
        self.addItem(self.peak_floor_bar)
        
        # Bar 3: Peak Decay (orange) - decayed peak tracker
        self.peak_decay_bar = pg.BarGraphItem(
            x=[bar_x_start + 2 * (bar_width + bar_spacing)],
            y0=[-120],
            height=[0],
            width=bar_width,
            brush='#FF8C00'  # Dark orange
        )
        self.addItem(self.peak_decay_bar)
        
        # Store bar x positions for updates
        self._bar_x_actual = bar_x_start
        self._bar_x_floor = bar_x_start + bar_width + bar_spacing
        self._bar_x_decay = bar_x_start + 2 * (bar_width + bar_spacing)
        self._bar_width = bar_width
        
        # Smoothing buffer for smoother animation
        self._smooth_spectrum = np.full(self.num_bins, -120.0)
        self._smoothing = 0.3  # 0 = no smoothing, 1 = max smoothing
        
    def _hz_to_bin(self, hz: float) -> float:
        """Convert Hz to log-spaced bin index"""
        nyquist = self.sample_rate / 2
        n = self._fft_bins if hasattr(self, '_fft_bins') and self._fft_bins > 1 else self.num_bins
        linear_idx = max(1.0, (hz / nyquist) * (n - 1))
        log_max = np.log10(n - 1)
        return (np.log10(linear_idx) / log_max) * self.num_bins if log_max > 0 else 0.0
    
    def _bin_to_hz(self, bin_idx: float) -> float:
        """Convert log-spaced bin index to Hz"""
        nyquist = self.sample_rate / 2
        n = self._fft_bins if hasattr(self, '_fft_bins') and self._fft_bins > 1 else self.num_bins
        log_max = np.log10(n - 1)
        linear_idx = 10 ** ((bin_idx / self.num_bins) * log_max) if log_max > 0 else bin_idx
        return (linear_idx / (n - 1)) * nyquist
        
    def _on_beat_band_changed(self):
        if self._updating:
            return
        region = self.beat_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'freq_range_slider'):
            self._updating = True
            self.parent_window.freq_range_slider.setLow(int(low_hz))
            self.parent_window.freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (region[0] + region[1]) / 2  # type: ignore
        self.beat_label.setPos(center_bin, self._beat_label_y)
    
    def _on_depth_band_changed(self):
        if self._updating:
            return
        region = self.depth_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'depth_freq_range_slider'):
            self._updating = True
            self.parent_window.depth_freq_range_slider.setLow(int(low_hz))
            self.parent_window.depth_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (region[0] + region[1]) / 2  # type: ignore
        self.depth_label.setPos(center_bin, self._depth_label_y)
    
    def _on_p0_band_changed(self):
        if self._updating:
            return
        region = self.p0_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'pulse_freq_range_slider'):
            self._updating = True
            self.parent_window.pulse_freq_range_slider.setLow(int(low_hz))
            self.parent_window.pulse_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (region[0] + region[1]) / 2  # type: ignore
        self.pulse_label.setPos(center_bin, self._pulse_label_y)
    
    def _on_f0_band_changed(self):
        if self._updating:
            return
        region = self.f0_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'f0_freq_range_slider'):
            self._updating = True
            self.parent_window.f0_freq_range_slider.setLow(int(low_hz))
            self.parent_window.f0_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (region[0] + region[1]) / 2  # type: ignore
        self.carrier_label.setPos(center_bin, self._carrier_label_y)
    
    def set_sample_rate(self, sr: int):
        self.sample_rate = sr
    
    def set_frequency_band(self, low_norm: float, high_norm: float):
        self._updating = True
        nyquist = self.sample_rate / 2
        low_bin = self._hz_to_bin(low_norm * nyquist)
        high_bin = self._hz_to_bin(high_norm * nyquist)
        self.beat_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.beat_label.setPos(center_bin, self._beat_label_y)
        self._updating = False
    
    def set_depth_band(self, low_hz: float, high_hz: float):
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.depth_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.depth_label.setPos(center_bin, self._depth_label_y)
        self._updating = False
    
    def set_p0_band(self, low_hz: float, high_hz: float):
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.p0_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.pulse_label.setPos(center_bin, self._pulse_label_y)
        self._updating = False
    
    def set_f0_band(self, low_hz: float, high_hz: float):
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.f0_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.carrier_label.setPos(center_bin, self._carrier_label_y)
        self._updating = False
    
    def set_peak_and_flux(self, peak_value: float, flux_value: float):
        """Update peak indicator bars - actual peak and peak decay (tracked peak)"""
        peak_db = float(np.clip(20 * np.log10(max(peak_value, 1e-6)), -120, 0))
        flux_db = float(np.clip(20 * np.log10(max(flux_value, 1e-6)), -120, 0))
        peak_h = (peak_db + 120.0) * 0.90
        flux_h = (flux_db + 120.0) * 0.90
        if hasattr(self, 'peak_actual_bar'):
            # Update actual peak bar (green) - current band energy level
            self.peak_actual_bar.setOpts(y0=[-120], height=[peak_h])
        if hasattr(self, 'peak_decay_bar'):
            # Update peak decay bar (orange) - this shows the decayed/tracked peak
            # flux_value here represents the tracked/decayed peak from audio engine
            self.peak_decay_bar.setOpts(y0=[-120], height=[flux_h])
    
    def set_peak_floor(self, peak_floor: float):
        """Update peak floor bar height"""
        if hasattr(self, 'peak_floor_bar'):
            peak_floor_db = float(np.clip(20 * np.log10(max(peak_floor, 1e-6)), -120, 0))
            self.peak_floor_bar.setOpts(y0=[-120], height=[(peak_floor_db + 120.0) * 0.90])
    
    def set_peak_indicators_visible(self, visible: bool):
        """Show or hide peak indicator bars (peak_actual, peak_floor, peak_decay)"""
        if hasattr(self, 'peak_actual_bar'):
            self.peak_actual_bar.setVisible(visible)
        if hasattr(self, 'peak_floor_bar'):
            self.peak_floor_bar.setVisible(visible)
        if hasattr(self, 'peak_decay_bar'):
            self.peak_decay_bar.setVisible(visible)

    def set_range_indicators_visible(self, visible: bool):
        """Show or hide frequency range band indicators and labels"""
        self.beat_band.setVisible(visible)
        self.depth_band.setVisible(visible)
        self.p0_band.setVisible(visible)
        self.f0_band.setVisible(visible)
        self.beat_label.setVisible(visible)
        self.depth_label.setVisible(visible)
        self.pulse_label.setVisible(visible)
        self.carrier_label.setVisible(visible)
        
    def update_spectrum(self, spectrum: np.ndarray, peak_energy: Optional[float] = None, spectral_flux: Optional[float] = None):
        """Update mountain range with new spectrum data"""
        if spectrum is None or len(spectrum) == 0:
            return
        
        # Cut off at Nyquist (22050 Hz at 44.1kHz sample rate)
        cutoff_hz = 22050
        nyquist = self.sample_rate / 2
        cutoff_idx = int((cutoff_hz / nyquist) * len(spectrum))
        spectrum = spectrum[:cutoff_idx]
            
        # Track source FFT size for hz/bin conversions
        self._fft_bins = len(spectrum)
        # Resample to num_bins using log-frequency spacing
        n = len(spectrum)
        if n > 1:
            x_old = np.arange(n)
            x_new = np.logspace(0, np.log10(n - 1), self.num_bins)
            x_new = np.clip(x_new, 0, n - 1)
            spectrum = np.interp(x_new, x_old, spectrum)
        elif len(spectrum) != self.num_bins:
            x_old = np.linspace(0, 1, n)
            x_new = np.linspace(0, 1, self.num_bins)
            spectrum = np.interp(x_new, x_old, spectrum)
        
        # Convert magnitude to dB for display
        spectrum = 20 * np.log10(spectrum + 1e-6)
        
        # Smooth the spectrum for less jittery animation
        self._smooth_spectrum = self._smoothing * self._smooth_spectrum + (1 - self._smoothing) * spectrum
        
        # Update curves
        x = np.arange(self.num_bins)
        self.spectrum_curve.setData(x, self._smooth_spectrum)
        self.glow_curve.setData(x, np.clip(self._smooth_spectrum + 1.5, -120, 0))
        
        # Find and mark peaks (local maxima above threshold)
        if peak_energy and peak_energy > 0.3:
            peaks = []
            peak_vals = []
            for i in range(2, self.num_bins - 2):
                if (self._smooth_spectrum[i] > self._smooth_spectrum[i-1] and 
                    self._smooth_spectrum[i] > self._smooth_spectrum[i+1] and
                    self._smooth_spectrum[i] > -40):
                    peaks.append(i)
                    peak_vals.append(self._smooth_spectrum[i])
            if peaks:
                self.peak_scatter.setData(peaks, peak_vals)
            else:
                self.peak_scatter.setData([], [])
        else:
            self.peak_scatter.setData([], [])
        
        # Update peak tracker line with current peak energy
        if peak_energy is not None:
            self.set_peak_and_flux(peak_energy, spectral_flux or 0.0)


class BarGraphCanvas(pg.PlotWidget):
    """Bar graph spectrum visualizer with frequency-based colors"""
    
    def __init__(self, parent=None, width=8, height=3):
        super().__init__(parent)
        
        # Dark theme
        self.setBackground('#0a0a12')
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.showGrid(x=False, y=False, alpha=0)
        self.hideAxis('left')
        self.hideAxis('bottom')
        
        # Spectrum dimensions - use fewer bars for cleaner look
        self.num_bars = 64
        
        # Set view range (left margin includes peak indicator bars at negative X)
        self.setXRange(-9, self.num_bars - 0.5)
        self.setYRange(0, 1.2)
        
        # Create bar items - each bar is a separate BarGraphItem for individual colors
        self.bars = []
        self.bar_colors = []
        
        # Generate frequency-based colors (rainbow: red->yellow->green->cyan->blue->magenta)
        for i in range(self.num_bars):
            # Hue goes from 0 (red/bass) to 270 (blue/high) degrees
            hue = int((i / self.num_bars) * 270)
            color = QColor.fromHsv(hue, 255, 255, 200)
            self.bar_colors.append(color)
        
        # Single BarGraphItem for all bars
        self.bar_item = pg.BarGraphItem(x=np.arange(self.num_bars), height=np.zeros(self.num_bars),
                                         width=0.8, brushes=self.bar_colors)
        self.addItem(self.bar_item)
        
        # 4 Frequency band indicators (vertical regions with different heights)
        # Band heights: beat=100%, stroke=92%, pulse=84%, carrier=76%
        # Band 1: Beat Detection (red) - full height (tallest)
        self.beat_band = pg.LinearRegionItem(values=(0, 32), orientation='vertical',
                                              brush=pg.mkBrush(255, 50, 50, 40),
                                              pen=pg.mkPen('#FF3232', width=1),
                                              movable=True, span=(0, 1.0))
        self.beat_band.setBounds([0, self.num_bars])
        self.beat_band.sigRegionChanged.connect(self._on_beat_band_changed)
        self.addItem(self.beat_band)
        
        # Band 2: Stroke Depth (green) - 92% height
        self.depth_band = pg.LinearRegionItem(values=(0, 32), orientation='vertical',
                                               brush=pg.mkBrush(50, 255, 50, 35),
                                               pen=pg.mkPen('#32FF32', width=1),
                                               movable=True, span=(0, 0.92))
        self.depth_band.setBounds([0, self.num_bars])
        self.depth_band.sigRegionChanged.connect(self._on_depth_band_changed)
        self.addItem(self.depth_band)
        
        # Band 3: Pulse/P0 TCode (blue) - 84% height
        self.p0_band = pg.LinearRegionItem(values=(0, 32), orientation='vertical',
                                            brush=pg.mkBrush(50, 100, 255, 35),
                                            pen=pg.mkPen('#3264FF', width=1),
                                            movable=True, span=(0, 0.84))
        self.p0_band.setBounds([0, self.num_bars])
        self.p0_band.sigRegionChanged.connect(self._on_p0_band_changed)
        self.addItem(self.p0_band)
        
        # Band 4: Carrier/F0 TCode (cyan) - 76% height (shortest)
        self.f0_band = pg.LinearRegionItem(values=(0, 32), orientation='vertical',
                                            brush=pg.mkBrush(0, 200, 255, 35),
                                            pen=pg.mkPen('#00C8FF', width=1),
                                            movable=True, span=(0, 0.76))
        self.f0_band.setBounds([0, self.num_bars])
        self.f0_band.sigRegionChanged.connect(self._on_f0_band_changed)
        self.addItem(self.f0_band)
        
        # Labels placed at top of each respective band
        # Each label sits just inside the upper limit of its own box
        self.carrier_label = pg.TextItem("carrier", color='#00C8FF', anchor=(0.5, 1))
        self.carrier_label.setPos(5, 0.73)
        self.addItem(self.carrier_label)
        
        self.pulse_label = pg.TextItem("pulse", color='#3264FF', anchor=(0.5, 1))
        self.pulse_label.setPos(5, 0.81)
        self.addItem(self.pulse_label)
        
        self.depth_label = pg.TextItem("stroke", color='#32FF32', anchor=(0.5, 1))
        self.depth_label.setPos(5, 0.89)
        self.addItem(self.depth_label)
        
        self.beat_label = pg.TextItem("beat", color='#FF3232', anchor=(0.5, 1))
        self.beat_label.setPos(5, 0.97)
        self.addItem(self.beat_label)
        
        # Reference to parent window
        self.parent_window = parent
        self.sample_rate = 44100
        self._fft_bins = 513  # Default for 1024 FFT; updated on first spectrum
        self._updating = False
        
        # Smoothing buffer
        self._smooth_heights = np.zeros(self.num_bars)
        self._smoothing = 0.4
        
        # Peak indicator vertical bars on left side (3 thin bars: actual peak, peak floor, peak decay)
        bar_width = 1.42  # Width of each bar (~7% thinner than prior tweak)
        bar_spacing = 0.4  # Gap between bars
        bar_x_start = -7.5  # Start position (leftmost bar)
        
        # Bar 1: Actual Peak (green)
        self.peak_actual_bar = pg.BarGraphItem(
            x=[bar_x_start],
            height=[0],
            width=bar_width,
            brush='#00FF00'
        )
        self.addItem(self.peak_actual_bar)
        
        # Bar 2: Peak Floor (yellow)
        self.peak_floor_bar = pg.BarGraphItem(
            x=[bar_x_start + bar_width + bar_spacing],
            height=[0],
            width=bar_width,
            brush='#FFD700'
        )
        self.addItem(self.peak_floor_bar)
        
        # Bar 3: Peak Decay (orange)
        self.peak_decay_bar = pg.BarGraphItem(
            x=[bar_x_start + 2 * (bar_width + bar_spacing)],
            height=[0],
            width=bar_width,
            brush='#FF8C00'
        )
        self.addItem(self.peak_decay_bar)
        
    def _hz_to_bin(self, hz: float) -> float:
        """Convert Hz to log-spaced bar index"""
        nyquist = self.sample_rate / 2
        n = self._fft_bins if hasattr(self, '_fft_bins') and self._fft_bins > 1 else self.num_bars
        linear_idx = max(1.0, (hz / nyquist) * (n - 1))
        log_max = np.log10(n - 1)
        return (np.log10(linear_idx) / log_max) * self.num_bars if log_max > 0 else 0.0
    
    def _bin_to_hz(self, bin_idx: float) -> float:
        """Convert log-spaced bar index to Hz"""
        nyquist = self.sample_rate / 2
        n = self._fft_bins if hasattr(self, '_fft_bins') and self._fft_bins > 1 else self.num_bars
        log_max = np.log10(n - 1)
        linear_idx = 10 ** ((bin_idx / self.num_bars) * log_max) if log_max > 0 else bin_idx
        return (linear_idx / (n - 1)) * nyquist
        
    def _on_beat_band_changed(self):
        if self._updating:
            return
        region = self.beat_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'freq_range_slider'):
            self._updating = True
            self.parent_window.freq_range_slider.setLow(int(low_hz))
            self.parent_window.freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (float(region[0]) + float(region[1])) / 2  # type: ignore
        self.beat_label.setPos(center_bin, 0.97)
    
    def _on_depth_band_changed(self):
        if self._updating:
            return
        region = self.depth_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'depth_freq_range_slider'):
            self._updating = True
            self.parent_window.depth_freq_range_slider.setLow(int(low_hz))
            self.parent_window.depth_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (float(region[0]) + float(region[1])) / 2  # type: ignore
        self.depth_label.setPos(center_bin, 0.89)
    
    def _on_p0_band_changed(self):
        if self._updating:
            return
        region = self.p0_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'pulse_freq_range_slider'):
            self._updating = True
            self.parent_window.pulse_freq_range_slider.setLow(int(low_hz))
            self.parent_window.pulse_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (float(region[0]) + float(region[1])) / 2  # type: ignore
        self.pulse_label.setPos(center_bin, 0.81)
    
    def _on_f0_band_changed(self):
        if self._updating:
            return
        region = self.f0_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'f0_freq_range_slider'):
            self._updating = True
            self.parent_window.f0_freq_range_slider.setLow(int(low_hz))
            self.parent_window.f0_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (float(region[0]) + float(region[1])) / 2  # type: ignore
        self.carrier_label.setPos(center_bin, 0.73)
    
    def set_sample_rate(self, sr: int):
        self.sample_rate = sr
    
    def set_frequency_band(self, low_norm: float, high_norm: float):
        self._updating = True
        nyquist = self.sample_rate / 2
        low_bin = self._hz_to_bin(low_norm * nyquist)
        high_bin = self._hz_to_bin(high_norm * nyquist)
        self.beat_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.beat_label.setPos(center_bin, 0.97)
        self._updating = False
    
    def set_depth_band(self, low_hz: float, high_hz: float):
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.depth_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.depth_label.setPos(center_bin, 0.89)
        self._updating = False
    
    def set_p0_band(self, low_hz: float, high_hz: float):
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.p0_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.pulse_label.setPos(center_bin, 0.81)
        self._updating = False
    
    def set_f0_band(self, low_hz: float, high_hz: float):
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.f0_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.carrier_label.setPos(center_bin, 0.73)
        self._updating = False
    
    def set_peak_and_flux(self, peak_value: float, flux_value: float):
        """Update peak indicator bars - actual peak and peak decay"""
        if hasattr(self, 'peak_actual_bar'):
            self.peak_actual_bar.setOpts(height=[min(1.2, peak_value)])
        if hasattr(self, 'peak_decay_bar'):
            self.peak_decay_bar.setOpts(height=[min(1.2, flux_value)])
    
    def set_peak_floor(self, peak_floor: float):
        """Update peak floor bar height"""
        if hasattr(self, 'peak_floor_bar'):
            self.peak_floor_bar.setOpts(height=[peak_floor])
    
    def set_peak_indicators_visible(self, visible: bool):
        """Show or hide peak indicator bars (peak_actual, peak_floor, peak_decay)"""
        if hasattr(self, 'peak_actual_bar'):
            self.peak_actual_bar.setVisible(visible)
        if hasattr(self, 'peak_floor_bar'):
            self.peak_floor_bar.setVisible(visible)
        if hasattr(self, 'peak_decay_bar'):
            self.peak_decay_bar.setVisible(visible)

    def set_range_indicators_visible(self, visible: bool):
        """Show or hide frequency range band indicators and labels"""
        self.beat_band.setVisible(visible)
        self.depth_band.setVisible(visible)
        self.p0_band.setVisible(visible)
        self.f0_band.setVisible(visible)
        self.beat_label.setVisible(visible)
        self.depth_label.setVisible(visible)
        self.pulse_label.setVisible(visible)
        self.carrier_label.setVisible(visible)
        
    def update_spectrum(self, spectrum: np.ndarray, peak_energy: Optional[float] = None, spectral_flux: Optional[float] = None):
        """Update bars with new spectrum data"""
        if spectrum is None or len(spectrum) == 0:
            return
        
        # Cut off at Nyquist (22050 Hz at 44.1kHz sample rate)
        cutoff_hz = 22050
        nyquist = self.sample_rate / 2
        cutoff_idx = int((cutoff_hz / nyquist) * len(spectrum))
        spectrum = spectrum[:cutoff_idx]
            
        # Track source FFT size for hz/bin conversions
        self._fft_bins = len(spectrum)
        # Resample to num_bars using log-frequency spacing
        n = len(spectrum)
        if n > 1:
            x_old = np.arange(n)
            x_new = np.logspace(0, np.log10(n - 1), self.num_bars)
            x_new = np.clip(x_new, 0, n - 1)
            spectrum = np.interp(x_new, x_old, spectrum)
        elif len(spectrum) != self.num_bars:
            x_old = np.linspace(0, 1, n)
            x_new = np.linspace(0, 1, self.num_bars)
            spectrum = np.interp(x_new, x_old, spectrum)
        
        # Apply log scaling
        spectrum = np.log10(spectrum + 1e-6)
        spectrum = np.clip((spectrum + 6) / 6, 0, 1)
        
        # Smooth for less jittery animation
        self._smooth_heights = self._smoothing * self._smooth_heights + (1 - self._smoothing) * spectrum
        
        # Update bar heights
        self.bar_item.setOpts(height=self._smooth_heights)
        
        # Update peak indicator bars with current peak energy
        if peak_energy is not None:
            self.set_peak_and_flux(peak_energy, spectral_flux or 0.0)


class PhosphorCanvas(pg.PlotWidget):
    """Digital Phosphor FFT visualizer - persistence display with decay"""
    
    def __init__(self, parent=None, width=8, height=3):
        super().__init__(parent)
        
        # Dark theme
        self.setBackground('#0a0a12')
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.showGrid(x=False, y=False, alpha=0)
        self.hideAxis('left')
        self.hideAxis('bottom')
        
        # Phosphor dimensions
        self.num_bins = 256  # Frequency bins (X axis)
        self.num_mag_levels = 100  # Magnitude levels (Y axis)
        
        # Hitmap for phosphor persistence (mag_levels x freq_bins)
        self.hitmap = np.zeros((self.num_mag_levels, self.num_bins), dtype=np.float32)
        
        # Create ImageItem for phosphor display
        self.img_item = pg.ImageItem()
        self.addItem(self.img_item)
        
        # Colormap: plasma-like (black -> purple -> orange -> yellow)
        colors = [
            (0, 0, 0),        # Black (no hits)
            (20, 0, 40),      # Dark purple
            (80, 0, 100),     # Purple
            (150, 30, 80),    # Magenta
            (200, 80, 30),    # Orange
            (255, 180, 0),    # Yellow-orange
            (255, 255, 100),  # Bright yellow (hottest)
        ]
        positions = [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0]
        self.colormap = pg.ColorMap(positions, colors)
        lut = self.colormap.getLookupTable(0.0, 1.0, 256)
        self.img_item.setLookupTable(lut)  # type: ignore
        
        # Set view range (left margin includes peak indicator bars at negative X)
        self.setXRange(-19, self.num_bins)
        self.setYRange(0, self.num_mag_levels)
        
        # Decay factor: lower = longer persistence
        self.decay = 0.92
        
        # 4 Frequency band indicators (vertical regions with different heights)
        # Band heights: beat=100%, stroke=92%, pulse=84%, carrier=76%
        # Band 1: Beat Detection (red) - full height (tallest)
        self.beat_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                              brush=pg.mkBrush(255, 50, 50, 40),
                                              pen=pg.mkPen('#FF3232', width=1),
                                              movable=True, span=(0, 1.0))
        self.beat_band.setBounds([0, self.num_bins])
        self.beat_band.sigRegionChanged.connect(self._on_beat_band_changed)
        self.addItem(self.beat_band)
        
        # Band 2: Stroke Depth (green) - 92% height
        self.depth_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                               brush=pg.mkBrush(50, 255, 50, 35),
                                               pen=pg.mkPen('#32FF32', width=1),
                                               movable=True, span=(0, 0.92))
        self.depth_band.setBounds([0, self.num_bins])
        self.depth_band.sigRegionChanged.connect(self._on_depth_band_changed)
        self.addItem(self.depth_band)
        
        # Band 3: Pulse/P0 TCode (blue) - 84% height
        self.p0_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                            brush=pg.mkBrush(50, 100, 255, 35),
                                            pen=pg.mkPen('#3264FF', width=1),
                                            movable=True, span=(0, 0.84))
        self.p0_band.setBounds([0, self.num_bins])
        self.p0_band.sigRegionChanged.connect(self._on_p0_band_changed)
        self.addItem(self.p0_band)
        
        # Band 4: Carrier/F0 TCode (cyan) - 76% height (shortest)
        self.f0_band = pg.LinearRegionItem(values=(0, 128), orientation='vertical',
                                            brush=pg.mkBrush(0, 200, 255, 35),
                                            pen=pg.mkPen('#00C8FF', width=1),
                                            movable=True, span=(0, 0.76))
        self.f0_band.setBounds([0, self.num_bins])
        self.f0_band.sigRegionChanged.connect(self._on_f0_band_changed)
        self.addItem(self.f0_band)
        
        # Labels placed at top of each respective band
        # Each label sits just inside the upper limit of its own box
        self.carrier_label = pg.TextItem("carrier", color='#00C8FF', anchor=(0.5, 1))
        self.carrier_label.setPos(5, self.num_mag_levels * 0.73)
        self.addItem(self.carrier_label)
        
        self.pulse_label = pg.TextItem("pulse", color='#3264FF', anchor=(0.5, 1))
        self.pulse_label.setPos(5, self.num_mag_levels * 0.81)
        self.addItem(self.pulse_label)
        
        self.depth_label = pg.TextItem("stroke", color='#32FF32', anchor=(0.5, 1))
        self.depth_label.setPos(5, self.num_mag_levels * 0.89)
        self.addItem(self.depth_label)
        
        self.beat_label = pg.TextItem("beat", color='#FF3232', anchor=(0.5, 1))
        self.beat_label.setPos(5, self.num_mag_levels * 0.97)
        self.addItem(self.beat_label)
        
        self.parent_window = parent
        self.sample_rate = 44100
        self._fft_bins = 513  # Default for 1024 FFT; updated on first spectrum
        self._updating = False
        
        # Peak indicator vertical bars on left side (3 thin bars: actual peak, peak floor, peak decay)
        bar_width = 4.6  # Width of each bar (about 8% thinner)
        bar_spacing = 0.8  # Gap between bars
        bar_x_start = -18.0  # Start position (leftmost bar)
        
        # Bar 1: Actual Peak (green)
        self.peak_actual_bar = pg.BarGraphItem(
            x=[bar_x_start],
            height=[0],
            width=bar_width,
            brush='#00FF00'
        )
        self.addItem(self.peak_actual_bar)
        
        # Bar 2: Peak Floor (yellow)
        self.peak_floor_bar = pg.BarGraphItem(
            x=[bar_x_start + bar_width + bar_spacing],
            height=[0],
            width=bar_width,
            brush='#FFD700'
        )
        self.addItem(self.peak_floor_bar)
        
        # Bar 3: Peak Decay (orange)
        self.peak_decay_bar = pg.BarGraphItem(
            x=[bar_x_start + 2 * (bar_width + bar_spacing)],
            height=[0],
            width=bar_width,
            brush='#FF8C00'
        )
        self.addItem(self.peak_decay_bar)
        
        # Store scale for Y axis
        self._bar_scale = self.num_mag_levels * 0.82
        
    def _hz_to_bin(self, hz: float) -> float:
        nyquist = self.sample_rate / 2
        n = self._fft_bins if hasattr(self, '_fft_bins') and self._fft_bins > 1 else self.num_bins
        linear_idx = max(1.0, (hz / nyquist) * (n - 1))
        log_max = np.log10(n - 1)
        return (np.log10(linear_idx) / log_max) * self.num_bins if log_max > 0 else 0.0
    
    def _bin_to_hz(self, bin_idx: float) -> float:
        nyquist = self.sample_rate / 2
        n = self._fft_bins if hasattr(self, '_fft_bins') and self._fft_bins > 1 else self.num_bins
        log_max = np.log10(n - 1)
        linear_idx = 10 ** ((bin_idx / self.num_bins) * log_max) if log_max > 0 else bin_idx
        return (linear_idx / (n - 1)) * nyquist
        
    def _on_beat_band_changed(self):
        if self._updating:
            return
        region = self.beat_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'freq_range_slider'):
            self._updating = True
            self.parent_window.freq_range_slider.setLow(int(low_hz))
            self.parent_window.freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (float(region[0]) + float(region[1])) / 2  # type: ignore
        self.beat_label.setPos(center_bin, self.num_mag_levels * 0.97)
    
    def _on_depth_band_changed(self):
        if self._updating:
            return
        region = self.depth_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'depth_freq_range_slider'):
            self._updating = True
            self.parent_window.depth_freq_range_slider.setLow(int(low_hz))
            self.parent_window.depth_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (float(region[0]) + float(region[1])) / 2  # type: ignore
        self.depth_label.setPos(center_bin, self.num_mag_levels * 0.89)
    
    def _on_p0_band_changed(self):
        if self._updating:
            return
        region = self.p0_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'pulse_freq_range_slider'):
            self._updating = True
            self.parent_window.pulse_freq_range_slider.setLow(int(low_hz))
            self.parent_window.pulse_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (float(region[0]) + float(region[1])) / 2  # type: ignore
        self.pulse_label.setPos(center_bin, self.num_mag_levels * 0.81)
    
    def _on_f0_band_changed(self):
        if self._updating:
            return
        region = self.f0_band.getRegion()
        low_hz = self._bin_to_hz(float(region[0]))  # type: ignore
        high_hz = self._bin_to_hz(float(region[1]))  # type: ignore
        if self.parent_window and hasattr(self.parent_window, 'f0_freq_range_slider'):
            self._updating = True
            self.parent_window.f0_freq_range_slider.setLow(int(low_hz))
            self.parent_window.f0_freq_range_slider.setHigh(int(high_hz))
            self._updating = False
        center_bin = (float(region[0]) + float(region[1])) / 2  # type: ignore
        self.carrier_label.setPos(center_bin, self.num_mag_levels * 0.73)
    
    def set_sample_rate(self, sr: int):
        self.sample_rate = sr
    
    def set_frequency_band(self, low_norm: float, high_norm: float):
        self._updating = True
        nyquist = self.sample_rate / 2
        low_bin = self._hz_to_bin(low_norm * nyquist)
        high_bin = self._hz_to_bin(high_norm * nyquist)
        self.beat_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.beat_label.setPos(center_bin, self.num_mag_levels * 0.97)
        self._updating = False
    
    def set_depth_band(self, low_hz: float, high_hz: float):
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.depth_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.depth_label.setPos(center_bin, self.num_mag_levels * 0.89)
        self._updating = False
    
    def set_p0_band(self, low_hz: float, high_hz: float):
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.p0_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.pulse_label.setPos(center_bin, self.num_mag_levels * 0.81)
        self._updating = False
    
    def set_f0_band(self, low_hz: float, high_hz: float):
        self._updating = True
        low_bin = self._hz_to_bin(low_hz)
        high_bin = self._hz_to_bin(high_hz)
        self.f0_band.setRegion((low_bin, high_bin))
        center_bin = (low_bin + high_bin) / 2
        self.carrier_label.setPos(center_bin, self.num_mag_levels * 0.73)
        self._updating = False
    
    def set_peak_and_flux(self, peak_value: float, flux_value: float):
        """Update peak indicator bars - actual peak and peak decay"""
        if hasattr(self, 'peak_actual_bar'):
            scale = getattr(self, '_bar_scale', self.num_mag_levels)
            self.peak_actual_bar.setOpts(height=[min(1.0, peak_value) * scale])
        if hasattr(self, 'peak_decay_bar'):
            scale = getattr(self, '_bar_scale', self.num_mag_levels)
            self.peak_decay_bar.setOpts(height=[min(1.0, flux_value) * scale])
    
    def set_peak_floor(self, peak_floor: float):
        """Update peak floor bar height"""
        if hasattr(self, 'peak_floor_bar'):
            scale = getattr(self, '_bar_scale', self.num_mag_levels)
            self.peak_floor_bar.setOpts(height=[peak_floor * scale])
    
    def set_peak_indicators_visible(self, visible: bool):
        """Show or hide peak indicator bars (peak_actual, peak_floor, peak_decay)"""
        if hasattr(self, 'peak_actual_bar'):
            self.peak_actual_bar.setVisible(visible)
        if hasattr(self, 'peak_floor_bar'):
            self.peak_floor_bar.setVisible(visible)
        if hasattr(self, 'peak_decay_bar'):
            self.peak_decay_bar.setVisible(visible)

    def set_range_indicators_visible(self, visible: bool):
        """Show or hide frequency range band indicators and labels"""
        self.beat_band.setVisible(visible)
        self.depth_band.setVisible(visible)
        self.p0_band.setVisible(visible)
        self.f0_band.setVisible(visible)
        self.beat_label.setVisible(visible)
        self.depth_label.setVisible(visible)
        self.pulse_label.setVisible(visible)
        self.carrier_label.setVisible(visible)
        
    def update_spectrum(self, spectrum: np.ndarray, peak_energy: Optional[float] = None, spectral_flux: Optional[float] = None):
        """Update phosphor display with new spectrum - accumulate hits with decay"""
        if spectrum is None or len(spectrum) == 0:
            return
        
        # Cut off at Nyquist (22050 Hz at 44.1kHz sample rate)
        cutoff_hz = 22050
        nyquist = self.sample_rate / 2
        cutoff_idx = int((cutoff_hz / nyquist) * len(spectrum))
        spectrum = spectrum[:cutoff_idx]
            
        # Track source FFT size for hz/bin conversions
        self._fft_bins = len(spectrum)
        # Resample to num_bins using log-frequency spacing
        n = len(spectrum)
        if n > 1:
            x_old = np.arange(n)
            x_new = np.logspace(0, np.log10(n - 1), self.num_bins)
            x_new = np.clip(x_new, 0, n - 1)
            spectrum = np.interp(x_new, x_old, spectrum)
        elif len(spectrum) != self.num_bins:
            x_old = np.linspace(0, 1, n)
            x_new = np.linspace(0, 1, self.num_bins)
            spectrum = np.interp(x_new, x_old, spectrum)
        
        # Apply log scaling and normalize to 0-1
        spectrum = np.log10(spectrum + 1e-6)
        spectrum = np.clip((spectrum + 6) / 6, 0, 1)
        
        # Apply decay to existing hitmap
        self.hitmap *= self.decay
        
        # Add current spectrum to hitmap
        # Each frequency bin adds a "hit" at its magnitude level
        for freq_bin in range(self.num_bins):
            mag_level = int(spectrum[freq_bin] * (self.num_mag_levels - 1))
            mag_level = min(max(0, mag_level), self.num_mag_levels - 1)
            self.hitmap[mag_level, freq_bin] += 0.5
        
        # Normalize for display
        display_data = np.clip(self.hitmap, 0, 1)
        
        # Update image (transpose for correct orientation)
        self.img_item.setImage(display_data.T, autoLevels=False, levels=(0, 1))
        self.img_item.setRect(0, 0, self.num_bins, self.num_mag_levels)
        
        # Update peak indicator bars with current peak energy
        if peak_energy is not None:
            self.set_peak_and_flux(peak_energy, spectral_flux or 0.0)


def launch_projectm():
    """Attempt to launch projectM standalone application"""
    import subprocess
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
                    subprocess.Popen([path], shell=False)
                    print(f"[Visualizer] Launched projectM from PATH")
                    return True
                except:
                    continue
        else:
            if os.path.exists(path):
                try:
                    subprocess.Popen([path], shell=False)
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


class PresetButton(QPushButton):
    """Custom button that emits different signals for left-click (load) vs right-click (save)"""
    
    left_clicked = pyqtSignal()
    right_clicked = pyqtSignal()
    
    def __init__(self, label: str):
        super().__init__(label)
        self.setFixedWidth(104)  # Fixed width for consistent layout even with custom names (1.6x wider for full text display)
        self.has_preset = False
        self.is_active = False
        self._update_style()
    
    def set_has_preset(self, has: bool):
        """Mark this button as having a saved preset"""
        self.has_preset = has
        self._update_style()
    
    def set_active(self, active: bool):
        """Mark this button as the currently loaded preset"""
        self.is_active = active
        self._update_style()
    
    def _update_style(self):
        """Update button appearance based on state"""
        if self.is_active:
            # Currently loaded preset - bright green border
            self.setStyleSheet("background-color: #4a6b4a; border: 2px solid #00ff00; font-weight: bold;")
        elif self.has_preset:
            # Has saved preset - subtle blue
            self.setStyleSheet("background-color: #4a5a6a; font-weight: bold;")
        else:
            # Empty slot - dark default
            self.setStyleSheet("background-color: #424242;")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.left_clicked.emit()
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit()
        super().mousePressEvent(event)


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
        
        # Draw selected range - purple-gray to match button/slider color
        low_pos = self._val_to_pos(self._low)
        high_pos = self._val_to_pos(self._high)
        painter.setBrush(QBrush(QColor(0x56, 0x5d, 0x7f)))  # #565d7f
        painter.drawRoundedRect(low_pos, track_y, high_pos - low_pos, track_h, 4, 4)
        
        # Draw handles - matches QSlider handle
        handle_w = 18
        handle_h = 18
        handle_y = h // 2 - handle_h // 2
        
        # Low handle
        painter.setBrush(QBrush(QColor(0x56, 0x5d, 0x7f)))  # #565d7f
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
                 default: float, decimals: int = 2, parent=None):
        super().__init__(parent)
        
        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
        self.multiplier = 10 ** decimals
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(name)
        self.label.setFixedWidth(120)
        self.label.setStyleSheet("color: #aaa;")
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * self.multiplier))
        self.slider.setMaximum(int(max_val * self.multiplier))
        self.slider.setValue(int(default * self.multiplier))
        self.slider.valueChanged.connect(self._on_change)
        
        self.value_label = QLabel(f"{default:.{decimals}f}")
        self.value_label.setFixedWidth(50)
        self.value_label.setStyleSheet("color: #0af;")
        
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)
        
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


class BREadbeatsWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("bREadbeats")
        self.setMinimumSize(400, 300)
        self.resize(1100, 950)
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
        self._slider_tracker = SliderTuningTracker(get_config_dir())
        _set_active_slider_tracker(self._slider_tracker)
        # Apply persisted log level early so downstream modules inherit
        set_log_level(getattr(self.config, 'log_level', 'INFO'))
        self.signals = SignalBridge()
        
        # Command queue
        self.cmd_queue = queue.Queue()
        
        # Initialize engines to None early (before UI setup needs to check them)
        self.audio_engine = None
        self.network_engine = None
        self.stroke_mapper = None
        self._dry_run_enabled = bool(getattr(self.config.device_limits, 'dry_run', False))
        
        # Setup UI
        self._setup_ui()
        
        # Load config values into UI sliders
        self._apply_config_to_ui()
        
        # Initialize indicator visibility: peak visible, range bands hidden (controlled by per-slider toggles)
        self._on_show_peak_indicators_toggle(True)
        self._on_toggle_beat_band(False)
        self._on_toggle_depth_band(False)
        self._on_toggle_p0_band(False)
        self._on_toggle_f0_band(False)
        
        # Load presets from disk
        self._load_presets_from_disk()
        
        # First-run device limits prompt (delayed so window is visible first)
        # Skip if user already entered values, opted out, or was already prompted
        dl = self.config.device_limits
        has_values = (dl.p0_freq_max > 0 or dl.c0_freq_max > 0)
        if not dl.prompted and not dl.dont_show_on_startup and not has_values:
            QTimer.singleShot(500, lambda: self._on_device_limits(first_run=True))
        
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
        
        # P0/F0 sliding window averaging (250ms window for smoother, more readable signal)
        from collections import deque
        import random
        self._p0_freq_window: deque = deque()  # (timestamp, norm_weighted) tuples
        self._f0_freq_window: deque = deque()  # (timestamp, norm_weighted) tuples
        self._p1_window: deque = deque()       # (timestamp, norm_weighted) tuples for Pulse Width
        self._p3_window: deque = deque()       # (timestamp, norm_weighted) tuples for Rise Time
        self._freq_window_ms: float = 250.0  # Window size in milliseconds
        self._p0_last_send_time: float = 0.0  # For throttling P0 sends
        self._f0_last_send_time: float = 0.0  # For throttling F0 sends
        self._f0_last_sent_tcode: Optional[int] = None  # Last F0 tcode value sent (for smoothing)
        self._f0_duration_base_ms: float = 900.0  # Base F0 duration (ms)
        # C0 Band mode rate limiter: +-500 tcode per 2 seconds, finish travel before new target
        self._c0_band_target: Optional[int] = None   # Current target tcode for band mode
        self._c0_band_current: Optional[int] = None   # Current sent tcode value (traveling)
        self._c0_band_last_target_time: float = 0.0   # When last target was set
        self._c0_band_travel_rate: float = 250.0       # Max tcode change per second (500/2s)
        self._f0_duration_variance_ms: float = 200.0  # ±variance for random duration
        self._f0_max_change_per_send: int = 300  # Max ±300 tcode change per send
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
        
        # Advanced controls dialog singleton reference
        self._advanced_controls_dialog = None
        
        # Auto-align target BPM tracking (wall-clock time-based)
        self._auto_align_target_enabled: bool = True  # Auto-align target BPM to metronome when stable
        self._auto_align_stable_since: float = 0.0      # time.time() when stability started
        self._auto_align_is_stable: bool = False         # currently in stable state
        self._auto_align_required_seconds: float = 1.2   # seconds of stability before first alignment
        self._auto_align_last_adjust_time: float = 0.0   # time.time() of last ±1 BPM adjustment
        self._auto_align_cooldown: float = 0.3            # seconds between each ±1 BPM step
        self._last_sensed_bpm: float = 0.0
        
        # State
        self.is_running = False
        self.is_sending = False
        
        # Auto-connect TCP on startup
        self._auto_connect_tcp()
        
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
                background-color: #565d7f;
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
                background-color: #565d7f;
                border: 1px solid #565d7f;
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
                background-color: #565d7f;
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
                background-color: #565d7f;
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
        top_layout = QHBoxLayout()
        top_layout.addWidget(self._create_connection_panel())
        top_layout.addWidget(self._create_control_panel())
        main_layout.addLayout(top_layout)
        
        # Middle: Visualizers + Splitter between viz and tabs+bottom
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        # Top half of splitter: Visualizers
        viz_widget = QWidget()
        viz_layout = QHBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.addWidget(self._create_spectrum_panel(), stretch=3)
        viz_layout.addWidget(self._create_position_panel(), stretch=1)
        viz_widget.setMinimumHeight(220)
        splitter.addWidget(viz_widget)

        # Bottom half of splitter: tabs only (absorbs compression)
        tabs_widget = QWidget()
        tabs_widget.setMinimumHeight(60)
        tabs_layout = QVBoxLayout(tabs_widget)
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        tabs_layout.setSpacing(0)
        tabs_layout.addWidget(self._create_settings_tabs())
        splitter.addWidget(tabs_widget)

        # Set initial splitter proportions (~72% viz, ~28% tabs)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter, stretch=1)

        # Bottom row: Presets + projectM launcher (LOCKED, never squishes)
        bottom_widget = QWidget()
        bottom_widget.setMinimumHeight(48)
        bottom_widget.setMaximumHeight(64)
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(self._create_presets_panel())
        
        bottom_layout.addStretch()  # Gap before Whip the Llama button
        
        # projectM launcher button (Winamp throwback)
        self.projectm_btn = QPushButton("Whip the Llama")
        self.projectm_btn.setToolTip("Launch projectM music visualizer (if installed)")
        self.projectm_btn.clicked.connect(self._on_launch_projectm)
        self.projectm_btn.setMaximumWidth(120)
        bottom_layout.addWidget(self.projectm_btn)

        main_layout.addWidget(bottom_widget)
    
    def _create_menu_bar(self):
        """Create menu bar with Menu, Options, and Help"""
        menubar = self.menuBar()
        assert menubar is not None
        
        # Menu (main menu with Performance submenu and About)
        main_menu = menubar.addMenu("Menu")
        assert main_menu is not None
        
        # Load Presets action
        load_presets_action = main_menu.addAction("Load Presets...")
        assert load_presets_action is not None
        load_presets_action.triggered.connect(self._on_load_presets)
        
        # Separator
        main_menu.addSeparator()
        
        # Performance submenu
        perf_menu = main_menu.addMenu("Performance")
        assert perf_menu is not None
        
        # FFT Size submenu
        fft_menu = perf_menu.addMenu("FFT Size (requires restart)")
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
        spec_menu = perf_menu.addMenu("Spectrum Updates")
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
        
        # Separator
        main_menu.addSeparator()
        
        # About action
        about_action = main_menu.addAction("About")
        assert about_action is not None
        about_action.triggered.connect(self._on_about)
        
        # Options menu (separate top-level menu)
        options_menu = menubar.addMenu("Options")
        assert options_menu is not None
        
        # Audio Device option
        audio_device_action = options_menu.addAction("Audio Device...")
        assert audio_device_action is not None
        audio_device_action.triggered.connect(self._on_options_audio_device)
        
        # Connection option
        connection_action = options_menu.addAction("Connection...")
        assert connection_action is not None
        connection_action.triggered.connect(self._on_options_connection)

        # Device Limits option
        device_limits_action = options_menu.addAction("Device Limits...")
        assert device_limits_action is not None
        device_limits_action.triggered.connect(self._on_device_limits)

        # Spectrum visualizer type submenu
        viz_menu = options_menu.addMenu("Spectrum Type")
        assert viz_menu is not None
        self.visualizer_type_combo = QComboBox()  # Hidden combo for state tracking
        self.visualizer_type_combo.addItems(["Waterfall", "Mountain Range", "Bar Graph", "Phosphor"])
        self.visualizer_type_combo.setCurrentIndex(1)  # Default: Mountain Range
        self._viz_type_actions = []
        for i, name in enumerate(["Waterfall", "Mountain Range", "Bar Graph", "Phosphor"]):
            action = viz_menu.addAction(name)
            assert action is not None
            action.setCheckable(True)
            action.setChecked(i == 1)  # Mountain Range default
            action.triggered.connect(lambda checked, idx=i: self._on_viz_menu_change(idx))
            self._viz_type_actions.append(action)

        # Show/Hide peak indicators
        self.show_peak_indicators_action = options_menu.addAction("Show Peak Indicators")
        assert self.show_peak_indicators_action is not None
        self.show_peak_indicators_action.setCheckable(True)
        self.show_peak_indicators_action.setChecked(True)  # Peak visible by default
        self.show_peak_indicators_action.triggered.connect(self._on_show_peak_indicators_menu_toggle)

        options_menu.addSeparator()

        # Log level submenu
        log_menu = options_menu.addMenu("Log Level")
        assert log_menu is not None
        self._log_level_actions = []
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            action = log_menu.addAction(level)
            assert action is not None
            action.setCheckable(True)
            action.triggered.connect(lambda checked, lvl=level: self._on_log_level_change(lvl))
            self._log_level_actions.append(action)
        self._sync_log_level_menu(getattr(self.config, 'log_level', 'INFO'))
        
        options_menu.addSeparator()
        
        # Advanced Controls dialog
        advanced_action = options_menu.addAction("Advanced Controls...")
        assert advanced_action is not None
        advanced_action.triggered.connect(self._on_advanced_controls)
        
        # Help menu (separate top-level menu)
        help_menu = menubar.addMenu("Help")
        assert help_menu is not None
        
        help_action = help_menu.addAction("Troubleshooting...")
        assert help_action is not None
        help_action.triggered.connect(self._on_help)
    
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
            "Please enter the limits you have configured in Restim:\n"
            "When set, displays will also show the converted real-world values.\n"
            "Leave at 0 to disable conversion for that axis."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        dl = self.config.device_limits
        
        # --- Pulse Freq / Carrier Freq group (always visible) ---
        main_group = QGroupBox("Pulse Freq / Carrier Freq  —  Hz")
        grid = QGridLayout(main_group)
        grid.addWidget(QLabel("Pulse Freq Min Hz:"), 0, 0)
        p0_min = QDoubleSpinBox()
        p0_min.setRange(0, 99999)
        p0_min.setDecimals(1)
        p0_min.setValue(dl.p0_freq_min)
        p0_min.setSpecialValueText("not set")
        grid.addWidget(p0_min, 0, 1)
        
        grid.addWidget(QLabel("Pulse Freq Max Hz:"), 0, 2)
        p0_max = QDoubleSpinBox()
        p0_max.setRange(0, 99999)
        p0_max.setDecimals(1)
        p0_max.setValue(dl.p0_freq_max)
        p0_max.setSpecialValueText("not set")
        grid.addWidget(p0_max, 0, 3)
        
        grid.addWidget(QLabel("Carrier Freq Min Hz:"), 1, 0)
        c0_min = QDoubleSpinBox()
        c0_min.setRange(0, 99999)
        c0_min.setDecimals(1)
        c0_min.setValue(dl.c0_freq_min)
        c0_min.setSpecialValueText("not set")
        grid.addWidget(c0_min, 1, 1)
        
        grid.addWidget(QLabel("Carrier Freq Max Hz:"), 1, 2)
        c0_max = QDoubleSpinBox()
        c0_max.setRange(0, 99999)
        c0_max.setDecimals(1)
        c0_max.setValue(dl.c0_freq_max)
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

    def _on_advanced_controls(self):
        """Show Advanced Controls dialog with experimental/expert settings"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QScrollArea, QGroupBox, QSpinBox
        
        # Prevent multiple instances — reuse existing dialog if open
        if hasattr(self, '_advanced_controls_dialog') and self._advanced_controls_dialog is not None:
            self._advanced_controls_dialog.raise_()
            self._advanced_controls_dialog.activateWindow()
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Advanced Controls")
        dialog.setMinimumWidth(450)
        dialog.setMinimumHeight(400)
        dialog.setModal(False)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # Clear reference when dialog is closed
        dialog.destroyed.connect(lambda: setattr(self, '_advanced_controls_dialog', None))
        self._advanced_controls_dialog = dialog
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        
        # Warning message and toggle at top
        warning_box = QGroupBox()
        warning_box.setStyleSheet("QGroupBox { background-color: #442200; border: 2px solid #ff6600; border-radius: 4px; padding: 8px; }")
        warning_layout = QVBoxLayout(warning_box)
        
        warning_label = QLabel("⚠️ DON'T BORK YOUR BEATS ⚠️")
        warning_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffaa00;")
        warning_layout.addWidget(warning_label)
        
        warning_text = QLabel("These controls are for advanced users. Incorrect settings\nmay cause erratic behavior or break beat detection.")
        warning_text.setStyleSheet("color: #ccaa66;")
        warning_layout.addWidget(warning_text)
        
        self._advanced_unlock_cb = QCheckBox("I understand, unlock advanced controls")
        self._advanced_unlock_cb.setStyleSheet("color: #ffcc00;")
        warning_layout.addWidget(self._advanced_unlock_cb)
        
        layout.addWidget(warning_box)
        
        # Scroll area for future controls
        scroll = NoWheelScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)
        
        # ===== Syncopation Controls =====
        syncope_group = QGroupBox("Syncopation / Double-Stroke")
        syncope_layout = QVBoxLayout(syncope_group)

        # On/Off checkbox
        syncope_enabled_cb = QCheckBox("Enable syncopation detection")
        syncope_enabled_cb.setChecked(self.config.beat.syncopation_enabled)
        syncope_enabled_cb.stateChanged.connect(
            lambda state: setattr(self.config.beat, 'syncopation_enabled', state == 2)
        )
        syncope_layout.addWidget(syncope_enabled_cb)

        # Band selector
        from PyQt6.QtWidgets import QComboBox, QHBoxLayout as QHBox
        band_row = QHBox()
        band_label = QLabel("Detection band:")
        band_label.setStyleSheet("color: #ccc;")
        band_row.addWidget(band_label)
        band_combo = QComboBox()
        band_options = ['any', 'sub_bass', 'low_mid', 'mid', 'high']
        band_combo.addItems(band_options)
        current_band = self.config.beat.syncopation_band
        if current_band in band_options:
            band_combo.setCurrentIndex(band_options.index(current_band))
        band_combo.currentTextChanged.connect(
            lambda text: setattr(self.config.beat, 'syncopation_band', text)
        )
        band_row.addWidget(band_combo)
        syncope_layout.addLayout(band_row)

        # Syncopation window slider
        syncope_window_slider = SliderWithLabel("Off-beat window (± beat fraction)", 0.05, 0.30, self.config.beat.syncopation_window, 2)
        syncope_window_slider.valueChanged.connect(
            lambda v: setattr(self.config.beat, 'syncopation_window', v)
        )
        syncope_layout.addWidget(syncope_window_slider)

        # BPM limit slider
        syncope_bpm_slider = SliderWithLabel("BPM limit (disable above)", 80.0, 200.0, self.config.beat.syncopation_bpm_limit, 0)
        syncope_bpm_slider.valueChanged.connect(
            lambda v: setattr(self.config.beat, 'syncopation_bpm_limit', v)
        )
        syncope_layout.addWidget(syncope_bpm_slider)

        # Arc size: fraction of circle (0.25=90°, 0.5=180°, 1.0=360°)
        syncope_arc_slider = SliderWithLabel("Syncopation arc size (circle fraction)", 0.10, 1.0, self.config.beat.syncopation_arc_size, 2)
        syncope_arc_slider.valueChanged.connect(
            lambda v: setattr(self.config.beat, 'syncopation_arc_size', v)
        )
        syncope_layout.addWidget(syncope_arc_slider)

        # Speed: duration as fraction of beat interval (0.25=quarter beat, 0.5=half, 1.0=full)
        syncope_speed_slider = SliderWithLabel("Syncopation speed (beat fraction)", 0.10, 1.0, self.config.beat.syncopation_speed, 2)
        syncope_speed_slider.valueChanged.connect(
            lambda v: setattr(self.config.beat, 'syncopation_speed', v)
        )
        syncope_layout.addWidget(syncope_speed_slider)

        scroll_layout.addWidget(syncope_group)

        # ===== Amplitude Gate Controls =====
        gate_group = QGroupBox("Amplitude Gate (Stroke vs Creep)")
        gate_layout = QVBoxLayout(gate_group)

        gate_info = QLabel("Controls when full strokes activate vs quiet creep mode.\nLower = more sensitive (strokes on quieter audio).")
        gate_info.setStyleSheet("color: #aaa; font-size: 11px;")
        gate_layout.addWidget(gate_info)

        # Gate high slider (threshold to enter FULL_STROKE)
        gate_high_slider = SliderWithLabel("Full stroke threshold (enter)", 0.01, 0.20, self.config.stroke.amplitude_gate_high, 3)
        gate_high_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'amplitude_gate_high', v)
        )
        gate_layout.addWidget(gate_high_slider)

        # Gate low slider (threshold to drop to CREEP_MICRO)
        gate_low_slider = SliderWithLabel("Creep threshold (exit)", 0.005, 0.10, self.config.stroke.amplitude_gate_low, 3)
        gate_low_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'amplitude_gate_low', v)
        )
        gate_layout.addWidget(gate_low_slider)

        scroll_layout.addWidget(gate_group)

        # ===== Noise Burst Controls (Hybrid System) =====
        burst_group = QGroupBox("Noise Burst (Transient Reaction)")
        burst_layout = QVBoxLayout(burst_group)

        burst_info = QLabel("React immediately to sudden loud sounds between beats.\nCombines noise-driven speed with metronome-timed arcs.")
        burst_info.setStyleSheet("color: #aaa; font-size: 11px;")
        burst_layout.addWidget(burst_info)

        # On/Off checkbox
        burst_enabled_cb = QCheckBox("Enable noise burst arcs")
        burst_enabled_cb.setChecked(self.config.stroke.noise_burst_enabled)
        burst_enabled_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'noise_burst_enabled', state == 2)
        )
        burst_layout.addWidget(burst_enabled_cb)

        # Flux multiplier slider
        burst_flux_slider = SliderWithLabel("Burst sensitivity (flux multiplier)", 0.5, 10.0, self.config.stroke.noise_burst_flux_multiplier, 1)
        burst_flux_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'noise_burst_flux_multiplier', v)
        )
        burst_layout.addWidget(burst_flux_slider)

        # Magnitude slider — scale the size of noise burst patterns
        burst_mag_slider = SliderWithLabel("Burst magnitude (pattern size)", 0.5, 10.0, self.config.stroke.noise_burst_magnitude, 1)
        burst_mag_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'noise_burst_magnitude', v)
        )
        burst_layout.addWidget(burst_mag_slider)

        scroll_layout.addWidget(burst_group)

        # ===== Beats Between Strokes =====
        bbs_group = QGroupBox("Stroke Timing")
        bbs_layout = QVBoxLayout(bbs_group)

        bbs_info = QLabel("Only fire full arc strokes every Nth beat.\nHigher = slower motion. Downbeats always fire.")
        bbs_info.setStyleSheet("color: #aaa; font-size: 11px;")
        bbs_layout.addWidget(bbs_info)

        bbs_row = QHBoxLayout()
        bbs_label = QLabel("Beats between strokes:")
        bbs_label.setStyleSheet("color: #ccc;")
        bbs_row.addWidget(bbs_label)
        bbs_spin = QSpinBox()
        bbs_spin.setMinimum(1)
        bbs_spin.setMaximum(8)
        bbs_spin.setValue(self.config.stroke.beats_between_strokes)
        bbs_spin.setToolTip("1 = every beat, 2 = every 2nd, 4 = every 4th, 8 = every 8th")
        bbs_spin.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'beats_between_strokes', v)
        )
        bbs_row.addWidget(bbs_spin)
        bbs_layout.addLayout(bbs_row)

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
        bbs_layout.addLayout(lead_row)

        scroll_layout.addWidget(bbs_group)

        # ===== Noise-Primary Mode =====
        noise_mode_group = QGroupBox("Noise vs Metronome Priority")
        noise_mode_layout = QVBoxLayout(noise_mode_group)

        noise_mode_info = QLabel("DEFAULT: metronome fires strokes, noise adds bursts.\n"
                                 "REVERSED: noise fires strokes, metronome verifies timing.\n"
                                 "Reversed mode reacts faster to transients.")
        noise_mode_info.setStyleSheet("color: #aaa; font-size: 11px;")
        noise_mode_layout.addWidget(noise_mode_info)

        noise_primary_cb = QCheckBox("Noise-primary mode (reversed)")
        noise_primary_cb.setChecked(self.config.stroke.noise_primary_mode)
        noise_primary_cb.stateChanged.connect(
            lambda state: setattr(self.config.stroke, 'noise_primary_mode', state == 2)
        )
        noise_mode_layout.addWidget(noise_primary_cb)

        scroll_layout.addWidget(noise_mode_group)

        # ===== Post-Silence Volume Ramp =====
        silence_ramp_group = QGroupBox("Post-Silence Volume Ramp")
        silence_ramp_layout = QVBoxLayout(silence_ramp_group)

        silence_ramp_info = QLabel("After silence (track change), reduce volume and slowly\nraise it back over a configurable duration.")
        silence_ramp_info.setStyleSheet("color: #aaa; font-size: 11px;")
        silence_ramp_layout.addWidget(silence_ramp_info)

        # Volume reduction slider (0% - 50%)
        vol_reduction_slider = SliderWithLabel("Volume reduction (%)", 0.0, 0.50, self.config.stroke.post_silence_vol_reduction, 2)
        vol_reduction_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'post_silence_vol_reduction', v)
        )
        silence_ramp_layout.addWidget(vol_reduction_slider)

        # Ramp duration slider (1.0 - 8.0 seconds)
        ramp_dur_slider = SliderWithLabel("Ramp duration (seconds)", 1.0, 8.0, self.config.stroke.post_silence_ramp_seconds, 1)
        ramp_dur_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'post_silence_ramp_seconds', v)
        )
        silence_ramp_layout.addWidget(ramp_dur_slider)

        scroll_layout.addWidget(silence_ramp_group)

        # ===== Flux Controls =====
        flux_group = QGroupBox("Flux Sensitivity")
        flux_layout = QVBoxLayout(flux_group)

        flux_info = QLabel("Controls how spectral flux affects mode switching.\nFlux threshold determines low vs high energy boundary.\nFlux drop ratio controls how much flux must drop to trigger creep fallback.")
        flux_info.setStyleSheet("color: #aaa; font-size: 11px;")
        flux_layout.addWidget(flux_info)

        # Flux threshold slider
        flux_thresh_slider = SliderWithLabel("Flux threshold", 0.005, 0.20, self.config.stroke.flux_threshold, 3)
        flux_thresh_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'flux_threshold', v)
        )
        flux_layout.addWidget(flux_thresh_slider)

        # Flux drop ratio slider
        flux_drop_slider = SliderWithLabel("Flux drop ratio (creep fallback)", 0.05, 0.50, self.config.stroke.flux_drop_ratio, 2)
        flux_drop_slider.valueChanged.connect(
            lambda v: setattr(self.config.stroke, 'flux_drop_ratio', v)
        )
        flux_layout.addWidget(flux_drop_slider)

        scroll_layout.addWidget(flux_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        
        # Initially disable scroll content until unlocked
        scroll_content.setEnabled(False)
        self._advanced_unlock_cb.stateChanged.connect(
            lambda state: scroll_content.setEnabled(state == 2)
        )
        
        layout.addWidget(scroll)
        
        dialog.show()

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
        
        # Stroke min/max reset
        stroke_box = QGroupBox()
        stroke_box.setStyleSheet("QGroupBox { border: 1px solid #555; padding: 4px; margin-top: 2px; }")
        sb_layout = QVBoxLayout(stroke_box)
        sb_layout.setSpacing(2)
        sb_layout.addWidget(QLabel("[Stroke Settings] Check stroke min/max:"))
        stroke_reset_btn = QPushButton("Reset to 0-100%")
        stroke_reset_btn.clicked.connect(lambda: (
            self.stroke_range_slider.setLow(0.0),
            self.stroke_range_slider.setHigh(1.0),
            self._on_stroke_range_change(0.0, 1.0)
        ))
        sb_layout.addWidget(stroke_reset_btn)
        g2_layout.addWidget(stroke_box)
        
        # Fullness reset
        full_box = QGroupBox()
        full_box.setStyleSheet("QGroupBox { border: 1px solid #555; padding: 4px; margin-top: 2px; }")
        fb_layout = QVBoxLayout(full_box)
        fb_layout.setSpacing(2)
        fb_layout.addWidget(QLabel("[Stroke Settings] Check stroke fullness:"))
        fullness_reset_btn = QPushButton("Reset to 100%")
        fullness_reset_btn.clicked.connect(lambda: (
            self.fullness_slider.setValue(1.0),
            setattr(self.config.stroke, 'stroke_fullness', 1.0)
        ))
        fb_layout.addWidget(fullness_reset_btn)
        g2_layout.addWidget(full_box)
        
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
        axb_layout.addWidget(QLabel("[Effects/Axis] Check axis weights (0=no motion):"))
        axis_reset_btn = QPushButton("Reset to 1.0")
        axis_reset_btn.clicked.connect(lambda: (
            self.alpha_weight_slider.setValue(1.0),
            self.beta_weight_slider.setValue(1.0),
            setattr(self.config, 'alpha_weight', 1.0),
            setattr(self.config, 'beta_weight', 1.0)
        ))
        axb_layout.addWidget(axis_reset_btn)
        g2_layout.addWidget(axis_box)
        
        scroll_layout.addWidget(group2)
        
        # === Too much motion? ===
        group3 = QGroupBox("Too much motion?")
        g3_layout = QVBoxLayout(group3)
        g3_layout.setSpacing(4)
        g3_layout.addWidget(QLabel("• [Beat Detection] Lower audio amplification,\n  sensitivity, flux multiplier"))
        g3_layout.addWidget(QLabel("• [Effects/Axis] Lower axis weights"))
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
    
    def _on_load_presets(self):
        """Open file dialog to load a presets .json file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Presets",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    import json
                    data = json.load(f)
                # Apply loaded presets to config
                self._apply_preset_data(data)
                print(f"[Config] Loaded presets from {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load presets:\n{e}")
    
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
        about_text = """bREadbeats v1.0
Live Audio to Restim

Inspired by:
    digitalparkinglot's creations
    edger477 (ideas from funscriptgenerator)
    diglet48 (wouldn't be here without restim!)
    shadlock0133 (music-vibes)

Bug reports/share your presets:
bREadfan_69@hotmail.com"""
        QMessageBox.information(self, "About bREadbeats", about_text)
    
    def _apply_config_to_ui(self):
        """Apply loaded config values to UI sliders"""
        try:
            beats_to_index = {4: 0, 3: 1, 6: 2}
            with self._signals_blocked(
                self.detection_type_combo,
                self.sensitivity_slider,
                self.peak_floor_slider,
                self.peak_decay_slider,
                self.rise_sens_slider,
                self.flux_mult_slider,
                self.audio_gain_slider,
                self.silence_reset_slider,
                self.freq_range_slider,
                self.metrics_global_cb,
                self.tempo_tracking_checkbox,
                self.time_sig_combo,
                self.stability_threshold_slider,
                self.tempo_timeout_slider,
                self.phase_snap_slider,
                self.mode_combo,
                self.stroke_range_slider,
                self.min_interval_slider,
                self.fullness_slider,
                self.min_depth_slider,
                self.freq_depth_slider,
                self.depth_freq_range_slider,
                self.flux_threshold_slider,
                self.flux_scaling_slider,
                self.phase_advance_slider,
                self.jitter_enabled,
                self.jitter_amplitude_slider,
                self.jitter_intensity_slider,
                self.creep_enabled,
                self.creep_speed_slider,
                self.alpha_weight_slider,
                self.beta_weight_slider,
                self.host_edit,
                self.port_spin,
                self.pulse_freq_range_slider,
                self.tcode_freq_range_slider,
                self.freq_weight_slider,
                self.f0_freq_range_slider,
                self.f0_tcode_range_slider,
                self.f0_weight_slider,
                self.volume_slider,
            ):
                # Beat detection tab
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
                self.metrics_global_cb.setChecked(self.config.auto_adjust.metrics_global_enabled)

                # Tempo tracking settings
                self.tempo_tracking_checkbox.setChecked(self.config.beat.tempo_tracking_enabled)
                self.time_sig_combo.setCurrentIndex(beats_to_index.get(self.config.beat.beats_per_measure, 0))
                self.stability_threshold_slider.setValue(self.config.beat.stability_threshold)
                self.tempo_timeout_slider.setValue(self.config.beat.tempo_timeout_ms)
                self.phase_snap_slider.setValue(self.config.beat.phase_snap_weight)
                self.mode_combo.setCurrentIndex(self.config.stroke.mode - 1)
                self.stroke_range_slider.setLow(self.config.stroke.stroke_min)
                self.stroke_range_slider.setHigh(self.config.stroke.stroke_max)
                self.min_interval_slider.setValue(self.config.stroke.min_interval_ms)
                self.fullness_slider.setValue(self.config.stroke.stroke_fullness)
                self.min_depth_slider.setValue(self.config.stroke.minimum_depth)
                self.freq_depth_slider.setValue(self.config.stroke.freq_depth_factor)
                self.flux_depth_slider.setValue(self.config.stroke.flux_depth_factor)
                self.depth_freq_range_slider.setLow(int(self.config.stroke.depth_freq_low))
                self.depth_freq_range_slider.setHigh(int(self.config.stroke.depth_freq_high))
                self.flux_threshold_slider.setValue(self.config.stroke.flux_threshold)
                self.flux_scaling_slider.setValue(self.config.stroke.flux_scaling_weight)
                self.phase_advance_slider.setValue(self.config.stroke.phase_advance)

                # Jitter/Creep tab
                self.jitter_enabled.setChecked(self.config.jitter.enabled)
                self.jitter_amplitude_slider.setValue(self.config.jitter.amplitude)
                self.jitter_intensity_slider.setValue(self.config.jitter.intensity)
                self.creep_enabled.setChecked(self.config.creep.enabled)
                self.creep_speed_slider.setValue(self.config.creep.speed)

                # Axis weights tab
                self.alpha_weight_slider.setValue(self.config.alpha_weight)
                self.beta_weight_slider.setValue(self.config.beta_weight)

                # Effects tab
                self.vol_reduction_limit_slider.setValue(self.config.stroke.vol_reduction_limit)

                # Connection settings
                self.host_edit.setText(self.config.connection.host)
                self.port_spin.setValue(self.config.connection.port)

                # Other tab (pulse freq settings)
                self.pulse_freq_range_slider.setLow(self.config.pulse_freq.monitor_freq_min)
                self.pulse_freq_range_slider.setHigh(self.config.pulse_freq.monitor_freq_max)
                self.tcode_freq_range_slider.setLow(self.config.pulse_freq.tcode_min)
                self.tcode_freq_range_slider.setHigh(self.config.pulse_freq.tcode_max)
                self.freq_weight_slider.setValue(self.config.pulse_freq.freq_weight)

                # Carrier freq (F0) settings
                self.f0_freq_range_slider.setLow(self.config.carrier_freq.monitor_freq_min)
                self.f0_freq_range_slider.setHigh(self.config.carrier_freq.monitor_freq_max)
                self.f0_tcode_range_slider.setLow(self.config.carrier_freq.tcode_min)
                self.f0_tcode_range_slider.setHigh(self.config.carrier_freq.tcode_max)
                self.f0_weight_slider.setValue(self.config.carrier_freq.freq_weight)

                # Volume (config stores 0-1, slider shows 0-100)
                self.volume_slider.setValue(int(self.config.volume * 100))

            # Set spectrum canvas sample rate and update all 3 frequency bands
            self.spectrum_canvas.set_sample_rate(self.config.audio.sample_rate)
            if hasattr(self, 'mountain_canvas'):
                self.mountain_canvas.set_sample_rate(self.config.audio.sample_rate)
            if hasattr(self, 'bar_canvas'):
                self.bar_canvas.set_sample_rate(self.config.audio.sample_rate)
            if hasattr(self, 'phosphor_canvas'):
                self.phosphor_canvas.set_sample_rate(self.config.audio.sample_rate)
            self._on_freq_band_change()  # Update beat detection band (red)
            
            # Apply mode-dependent limits after sliders are set
            self._on_mode_change(self.config.stroke.mode - 1)  # Apply axis weight limits for this mode
            self._on_depth_band_change()  # Update stroke depth band (green)
            self._on_p0_band_change()  # Update P0 TCode band (blue)
            self._on_f0_band_change()  # Update F0 TCode band (cyan)

            # Apply tempo tracking side effects after values are in place
            self._on_tempo_tracking_toggle(2 if self.config.beat.tempo_tracking_enabled else 0)

            # Log level menu (persisted)
            self._sync_log_level_menu(getattr(self.config, 'log_level', get_log_level()))
            
            print("[UI] Loaded all settings from config")
        except AttributeError as e:
            print(f"[UI] Warning: Could not apply all config values: {e}")
        
    def _create_connection_panel(self) -> QWidget:
        """Connection settings panel - simplified, host/port in Options menu (no visible groupbox)"""
        group = QWidget()
        layout = QHBoxLayout(group)  # Horizontal for compact layout
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Hidden Host/Port widgets (needed for functionality but now in Options menu)
        self.host_edit = QLineEdit(self.config.connection.host)
        self.host_edit.setVisible(False)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(self.config.connection.port)
        self.port_spin.setVisible(False)
        
        # Status
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: #f55;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Reset/Test buttons
        self.connect_btn = QPushButton("Reset")
        self.connect_btn.clicked.connect(self._on_connect)
        layout.addWidget(self.connect_btn)
        
        self.test_btn = QPushButton("Test")
        self.test_btn.clicked.connect(self._on_test)
        self.test_btn.setEnabled(False)
        layout.addWidget(self.test_btn)
        
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
        
        # Hidden preset buttons (still functional via Options menu)
        self.preset_mic_btn = QPushButton("🎤 Mic (Reactive)")
        self.preset_mic_btn.setVisible(False)
        self.preset_mic_btn.clicked.connect(self._set_device_preset_mic)
        
        self.preset_loopback_btn = QPushButton("🔊 System Audio")
        self.preset_loopback_btn.setVisible(False)
        self.preset_loopback_btn.clicked.connect(self._set_device_preset_loopback)
        
        # Connect device changes to update button states
        self.device_combo.currentIndexChanged.connect(self._update_preset_button_states)
        
        # Controls row - all on one row now
        btn_layout = QGridLayout()
        btn_layout.setSpacing(8)
        
        # Start/Stop audio capture
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setCheckable(True)
        self.start_btn.clicked.connect(self._on_start_stop)
        self.start_btn.setFixedSize(100, 40)
        btn_layout.addWidget(self.start_btn, 0, 0)
        
        # Play/Pause sending
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self._on_play_pause)
        self.play_btn.setEnabled(False)
        self.play_btn.setFixedSize(100, 40)
        btn_layout.addWidget(self.play_btn, 0, 1)
        
        # Volume slider (0 - 100) - uses compact label for control panel
        self.volume_slider = SliderWithLabel("Vol", 0, 100, 100, decimals=0)
        self.volume_slider.label.setFixedWidth(30)  # Compact label for controls box
        self.volume_slider.setFixedWidth(180)
        self.volume_slider.setContentsMargins(0, 0, 0, 0)
        btn_layout.addWidget(self.volume_slider, 0, 2, 1, 2)

        # Frequency displays - stacked vertically
        freq_display_layout = QVBoxLayout()
        freq_display_layout.setSpacing(0)
        
        # Carrier Freq display
        self.carrier_freq_label = QLabel("Carrier Freq: off")
        self.carrier_freq_label.setStyleSheet("color: #0af; font-size: 11px; font-weight: bold;")
        self.carrier_freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_display_layout.addWidget(self.carrier_freq_label)
        
        # Pulse Freq display
        self.pulse_freq_label = QLabel("Pulse Freq: off")
        self.pulse_freq_label.setStyleSheet("color: #f80; font-size: 11px; font-weight: bold;")
        self.pulse_freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_display_layout.addWidget(self.pulse_freq_label)
        
        # Pulse Width display
        self.p1_display_label = QLabel("Pulse Width: off")
        self.p1_display_label.setStyleSheet("color: #fa0; font-size: 10px;")
        self.p1_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_display_layout.addWidget(self.p1_display_label)
        
        # Rise Time display
        self.p3_display_label = QLabel("Rise Time: off")
        self.p3_display_label.setStyleSheet("color: #af0; font-size: 10px;")
        self.p3_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_display_layout.addWidget(self.p3_display_label)
        
        freq_display_widget = QWidget()
        freq_display_widget.setLayout(freq_display_layout)
        freq_display_widget.setFixedWidth(110)
        btn_layout.addWidget(freq_display_widget, 0, 4)

        # Right-side stack: traffic light / BPM / beat indicators
        right_stack = QVBoxLayout()
        right_stack.setSpacing(2)

        # Traffic light indicator (top of right stack)
        self.metric_traffic_light = TrafficLightWidget()
        self.metric_traffic_light.setToolTip("Green=All Locked, Yellow=Some Settled, Red=Adjusting")
        right_stack.addWidget(self.metric_traffic_light, alignment=Qt.AlignmentFlag.AlignCenter)

        # BPM display (middle of right stack)
        self.bpm_label = QLabel("BPM: --")
        self.bpm_label.setStyleSheet("color: #0a0; font-size: 14px; font-weight: bold;")
        self.bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_stack.addWidget(self.bpm_label)

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
        right_stack_widget.setFixedWidth(100)
        btn_layout.addWidget(right_stack_widget, 0, 5)

        btn_layout.setColumnStretch(6, 1)  # Allow last column to stretch
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
    
    def _set_device_preset_mic(self):
        """Filter to show only microphone/input devices"""
        import sounddevice as sd
        devices = sd.query_devices()
        
        # Find first non-loopback input device (regular microphone)
        loopback_keywords = ['stereo mix', 'what u hear', 'loopback', 'wave out mix', 'system audio']
        
        for combo_idx, device_idx in self.audio_device_map.items():
            # Skip if this is marked as loopback
            if self.audio_device_is_loopback.get(combo_idx, False):
                continue
            
            # Check if this device has input channels and is not a loopback device
            if device_idx < len(devices):
                dev = devices[device_idx]
                if dev['max_input_channels'] > 0:
                    dev_name = dev['name'].lower()
                    # Skip if it matches loopback keywords
                    if not any(keyword in dev_name for keyword in loopback_keywords):
                        self.device_combo.setCurrentIndex(combo_idx)
                        self.device_combo.currentIndexChanged.emit(combo_idx)
                        print(f"[Main] Switched to Microphone mode (Device {device_idx}: {dev['name']})")
                        self._update_preset_button_states()
                        return
        
        print("[Main] No microphone device found")
    
    def _set_device_preset_loopback(self):
        """Filter to show only system audio/playback loopback devices"""
        # First, try to find a marked loopback device (WASAPI output)
        for combo_idx, device_idx in self.audio_device_map.items():
            if self.audio_device_is_loopback.get(combo_idx, False):
                self.device_combo.setCurrentIndex(combo_idx)
                self.device_combo.currentIndexChanged.emit(combo_idx)
                print(f"[Main] Switched to System Audio mode (WASAPI Loopback Device {device_idx})")
                self._update_preset_button_states()
                return
        
        # Fallback: look for devices with loopback keywords
        import sounddevice as sd
        devices = sd.query_devices()
        loopback_keywords = ['stereo mix', 'what u hear', 'loopback', 'wave out mix', 'system audio']
        
        for combo_idx, device_idx in self.audio_device_map.items():
            if device_idx < len(devices):
                dev = devices[device_idx]
                if dev['max_input_channels'] > 0:
                    if any(keyword in dev['name'].lower() for keyword in loopback_keywords):
                        self.device_combo.setCurrentIndex(combo_idx)
                        self.device_combo.currentIndexChanged.emit(combo_idx)
                        print(f"[Main] Switched to System Audio mode (Device {device_idx}: {dev['name']})")
                        self._update_preset_button_states()
                        return
        
        print("[Main] No system audio/loopback device found. Enable 'Stereo Mix' or 'What U Hear' in sound settings")
    
    def _update_preset_button_states(self):
        """Update button colors based on current device selection"""
        current_combo_idx = self.device_combo.currentIndex()
        current_device_idx = self.audio_device_map.get(current_combo_idx)
        
        # Check if current device is marked as loopback or has loopback keywords
        is_loopback = self.audio_device_is_loopback.get(current_combo_idx, False)
        
        if not is_loopback and current_device_idx is not None:
            import sounddevice as sd
            devices = sd.query_devices()
            if current_device_idx < len(devices):
                dev_name = devices[current_device_idx]['name'].lower()
                loopback_keywords = ['stereo mix', 'what u hear', 'loopback', 'wave out mix', 'system audio']
                is_loopback = any(keyword in dev_name for keyword in loopback_keywords)
        
        # Check if current device is a regular microphone (input, not loopback)
        is_mic = current_device_idx is not None and not is_loopback
        if is_mic and current_device_idx is not None:
            import sounddevice as sd
            devices = sd.query_devices()
            if current_device_idx < len(devices):
                dev = devices[current_device_idx]
                # Must be input device
                is_mic = dev['max_input_channels'] > 0
        
        # Update button colors: green = active, white = inactive
        self.preset_mic_btn.setStyleSheet("color: #0a0; font-weight: bold;" if is_mic else "color: #fff;")
        self.preset_loopback_btn.setStyleSheet("color: #0a0; font-weight: bold;" if is_loopback else "color: #fff;")
    
    def _create_spectrum_panel(self) -> QWidget:
        """Spectrum visualizer panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create all visualizers (only one visible at a time)
        self.spectrum_canvas = SpectrumCanvas(self, width=8, height=3)
        self.mountain_canvas = MountainRangeCanvas(self, width=8, height=3)
        self.bar_canvas = BarGraphCanvas(self, width=8, height=3)
        self.phosphor_canvas = PhosphorCanvas(self, width=8, height=3)
        self.spectrum_canvas.setVisible(False)  # Start with mountain range
        self.bar_canvas.setVisible(False)
        self.phosphor_canvas.setVisible(False)
        
        layout.addWidget(self.spectrum_canvas)
        layout.addWidget(self.mountain_canvas)
        layout.addWidget(self.bar_canvas)
        layout.addWidget(self.phosphor_canvas)
        
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
        """Switch between visualizer types: 0=Waterfall, 1=Mountain, 2=Bar, 3=Phosphor"""
        self.spectrum_canvas.setVisible(index == 0)
        self.mountain_canvas.setVisible(index == 1)
        self.bar_canvas.setVisible(index == 2)
        self.phosphor_canvas.setVisible(index == 3)
        
        # Sync the frequency bands to the newly visible visualizer
        if hasattr(self, 'freq_range_slider'):
            self._on_freq_band_change()
        if hasattr(self, 'depth_freq_range_slider'):
            self._on_depth_band_change()
        if hasattr(self, 'pulse_freq_range_slider'):
            self._on_p0_band_change()
    
    def _on_show_peak_indicators_toggle(self, checked: bool):
        """Toggle visibility of peak indicator bars on all visualizers"""
        for canvas in [self.spectrum_canvas, self.mountain_canvas, self.bar_canvas, self.phosphor_canvas]:
            if hasattr(canvas, 'set_peak_indicators_visible'):
                canvas.set_peak_indicators_visible(checked)

    def _on_show_range_indicators_toggle(self, checked: bool):
        """Toggle visibility of range indicator bands on all visualizers"""
        for canvas in [self.spectrum_canvas, self.mountain_canvas, self.bar_canvas, self.phosphor_canvas]:
            if hasattr(canvas, 'set_range_indicators_visible'):
                canvas.set_range_indicators_visible(checked)

    def _on_show_peak_indicators_menu_toggle(self, checked: bool):
        """Handle Show Peak Indicators toggle from Options menu"""
        self._on_show_peak_indicators_toggle(checked)

    def _on_show_range_indicators_menu_toggle(self, checked: bool):
        """Handle Show Range Indicators toggle from Options menu"""
        self._on_show_range_indicators_toggle(checked)

    def _on_toggle_beat_band(self, checked: bool):
        """Toggle visibility of beat detection band (red) on all visualizers"""
        for canvas in [self.spectrum_canvas, self.mountain_canvas, self.bar_canvas, self.phosphor_canvas]:
            if hasattr(canvas, 'beat_band'):
                canvas.beat_band.setVisible(checked)
            if hasattr(canvas, 'beat_label'):
                canvas.beat_label.setVisible(checked)

    def _on_toggle_depth_band(self, checked: bool):
        """Toggle visibility of stroke depth band (green) on all visualizers"""
        for canvas in [self.spectrum_canvas, self.mountain_canvas, self.bar_canvas, self.phosphor_canvas]:
            if hasattr(canvas, 'depth_band'):
                canvas.depth_band.setVisible(checked)
            if hasattr(canvas, 'depth_label'):
                canvas.depth_label.setVisible(checked)

    def _on_toggle_p0_band(self, checked: bool):
        """Toggle visibility of pulse frequency band (blue) on all visualizers"""
        for canvas in [self.spectrum_canvas, self.mountain_canvas, self.bar_canvas, self.phosphor_canvas]:
            if hasattr(canvas, 'p0_band'):
                canvas.p0_band.setVisible(checked)
            if hasattr(canvas, 'pulse_label'):
                canvas.pulse_label.setVisible(checked)

    def _on_toggle_f0_band(self, checked: bool):
        """Toggle visibility of carrier frequency band (cyan) on all visualizers"""
        for canvas in [self.spectrum_canvas, self.mountain_canvas, self.bar_canvas, self.phosphor_canvas]:
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
    
    def _create_settings_tabs(self) -> QTabWidget:
        """Settings tabs with all the sliders"""
        tabs = QTabWidget()
        tabs.addTab(self._create_beat_detection_tab(), "Beat Detection")
        tabs.addTab(self._create_stroke_settings_tab(), "Stroke Settings")
        tabs.addTab(self._create_jitter_creep_tab(), "Effects / Axis")
        tabs.addTab(self._create_tempo_tracking_tab(), "Tempo Tracking")
        tabs.addTab(self._create_tcode_freq_tab(), "Pulse")
        return tabs
    
    def _create_presets_panel(self) -> QGroupBox:
        """Presets panel - always displayed below all tabs"""
        group = QGroupBox("Presets")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.custom_beat_presets = {}
        self._revert_settings = None  # Stores settings before preset load for revert
        self.preset_buttons = []
        for i in range(5):
            btn = PresetButton(f"{i+1}")
            btn.left_clicked.connect(lambda idx=i: self._load_beat_preset(idx))
            btn.right_clicked.connect(lambda idx=i: self._save_beat_preset(idx))
            self.preset_buttons.append(btn)
            layout.addWidget(btn)
        
        # Revert button - restores settings from before last preset load
        self.revert_btn = QPushButton("↩")  # Circular go-back arrow
        self.revert_btn.setFixedWidth(52)  # Half width of preset buttons (104/2)
        self.revert_btn.setToolTip("Revert to settings before last preset load")
        self.revert_btn.clicked.connect(self._revert_preset)
        self.revert_btn.setEnabled(False)  # Disabled until a preset is loaded
        layout.addWidget(self.revert_btn)
        
        return group
    
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
            'motion_intensity': self.motion_intensity_slider.value() if hasattr(self, 'motion_intensity_slider') else 1.0,
            'amp_gate_high': self.amp_gate_high_spin.value() if hasattr(self, 'amp_gate_high_spin') else 0.08,
            'amp_gate_low': self.amp_gate_low_spin.value() if hasattr(self, 'amp_gate_low_spin') else 0.04,
            'silence_reset_ms': int(self.silence_reset_slider.value()),
            'detection_type': self.detection_type_combo.currentIndex(),
            
            # Tempo Tracking
            'tempo_tracking_enabled': self.tempo_tracking_checkbox.isChecked(),
            'time_sig_index': self.time_sig_combo.currentIndex(),
            'stability_threshold': self.stability_threshold_slider.value(),
            'tempo_timeout_ms': int(self.tempo_timeout_slider.value()),
            'phase_snap_weight': self.phase_snap_slider.value(),

            # Stroke Settings Tab
            'stroke_mode': self.mode_combo.currentIndex(),
            'stroke_min': self.stroke_range_slider.low(),
            'stroke_max': self.stroke_range_slider.high(),
            'min_interval_ms': int(self.min_interval_slider.value()),
            'stroke_fullness': self.fullness_slider.value(),
            'minimum_depth': self.min_depth_slider.value(),
            'freq_depth_factor': self.freq_depth_slider.value(),
            'flux_depth_factor': self.flux_depth_slider.value(),
            'depth_freq_low': self.depth_freq_range_slider.low(),
            'depth_freq_high': self.depth_freq_range_slider.high(),
            'flux_threshold': self.flux_threshold_slider.value(),
            'flux_scaling_weight': self.flux_scaling_slider.value(),
            'phase_advance': self.phase_advance_slider.value(),

            # Jitter / Creep Tab
            'jitter_enabled': self.jitter_enabled.isChecked(),
            'jitter_amplitude': self.jitter_amplitude_slider.value(),
            'jitter_intensity': self.jitter_intensity_slider.value(),
            'creep_enabled': self.creep_enabled.isChecked(),
            'creep_speed': self.creep_speed_slider.value(),
            'thump_enabled': self.config.stroke.thump_enabled,

            # Axis Weights Tab
            'alpha_weight': self.alpha_weight_slider.value(),
            'beta_weight': self.beta_weight_slider.value(),

            # Effects Tab
            'vol_reduction_limit': self.vol_reduction_limit_slider.value(),

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
        from config import StrokeMode
        
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
        if 'motion_intensity' in preset_data and hasattr(self, 'motion_intensity_slider'):
            self.motion_intensity_slider.setValue(preset_data['motion_intensity'])
        if 'amp_gate_high' in preset_data and hasattr(self, 'amp_gate_high_spin'):
            self.amp_gate_high_spin.setValue(preset_data['amp_gate_high'])
        if 'amp_gate_low' in preset_data and hasattr(self, 'amp_gate_low_spin'):
            self.amp_gate_low_spin.setValue(preset_data['amp_gate_low'])
        self.silence_reset_slider.setValue(preset_data['silence_reset_ms'])
        self.detection_type_combo.setCurrentIndex(preset_data['detection_type'])
        
        # Tempo Tracking
        self.tempo_tracking_checkbox.setChecked(preset_data['tempo_tracking_enabled'])
        self._on_tempo_tracking_toggle(2 if preset_data['tempo_tracking_enabled'] else 0)
        self.time_sig_combo.setCurrentIndex(preset_data['time_sig_index'])
        self._on_time_sig_change(preset_data['time_sig_index'])
        self.stability_threshold_slider.setValue(preset_data['stability_threshold'])
        self._on_stability_threshold_change(preset_data['stability_threshold'])
        self.tempo_timeout_slider.setValue(preset_data['tempo_timeout_ms'])
        self._on_tempo_timeout_change(preset_data['tempo_timeout_ms'])
        self.phase_snap_slider.setValue(preset_data['phase_snap_weight'])
        self._on_phase_snap_change(preset_data['phase_snap_weight'])
        
        # Stroke Settings Tab
        self.mode_combo.setCurrentIndex(preset_data['stroke_mode'])
        self._on_mode_change(preset_data['stroke_mode'])
        self.stroke_range_slider.setLow(preset_data['stroke_min'])
        self.stroke_range_slider.setHigh(preset_data['stroke_max'])
        self.min_interval_slider.setValue(preset_data['min_interval_ms'])
        self.fullness_slider.setValue(preset_data['stroke_fullness'])
        self.min_depth_slider.setValue(preset_data['minimum_depth'])
        self.freq_depth_slider.setValue(preset_data['freq_depth_factor'])
        if 'flux_depth_factor' in preset_data:
            self.flux_depth_slider.setValue(preset_data['flux_depth_factor'])
        self.depth_freq_range_slider.setLow(preset_data['depth_freq_low'])
        self.depth_freq_range_slider.setHigh(preset_data['depth_freq_high'])
        self.flux_threshold_slider.setValue(preset_data['flux_threshold'])
        self.flux_scaling_slider.setValue(preset_data['flux_scaling_weight'])
        self.phase_advance_slider.setValue(preset_data['phase_advance'])
        
        # Jitter / Creep Tab
        self.jitter_enabled.setChecked(preset_data['jitter_enabled'])
        self.jitter_amplitude_slider.setValue(preset_data['jitter_amplitude'])
        self.jitter_intensity_slider.setValue(preset_data['jitter_intensity'])
        self.creep_enabled.setChecked(preset_data['creep_enabled'])
        self.creep_speed_slider.setValue(preset_data['creep_speed'])
        if 'thump_enabled' in preset_data:
            self.config.stroke.thump_enabled = preset_data['thump_enabled']
        
        # Axis Weights Tab
        self.alpha_weight_slider.setValue(preset_data['alpha_weight'])
        self.beta_weight_slider.setValue(preset_data['beta_weight'])
        
        # Effects Tab
        if 'vol_reduction_limit' in preset_data:
            self.vol_reduction_limit_slider.setValue(preset_data['vol_reduction_limit'])
        
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
        self.config.stroke.mode = StrokeMode(self.mode_combo.currentIndex() + 1)
        
        # Deactivate all preset buttons
        for btn in self.preset_buttons:
            btn.set_active(False)
        
        # Disable revert button (already reverted)
        self.revert_btn.setEnabled(False)
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
        scroll_area = NoWheelScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(self._get_thin_scrollbar_style())

        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ===== PULSE FREQUENCY =====
        pulse_group = CollapsibleGroupBox("Pulse Frequency - blue overlay on spectrum", collapsed=True)
        pulse_layout = QVBoxLayout(pulse_group)

        # Pulse Freq monitor slider with visibility toggle
        p0_slider_row = QHBoxLayout()
        self.pulse_freq_range_slider = RangeSliderWithLabel("Monitor Freq (Hz)", 30, 22050, 30, 4000, 0, log_scale=True)
        self.pulse_freq_range_slider.rangeChanged.connect(self._on_p0_band_change)
        p0_slider_row.addWidget(self.pulse_freq_range_slider)
        self.p0_band_toggle = QCheckBox("Show")
        self.p0_band_toggle.setToolTip("Show/hide blue overlay on spectrum")
        self.p0_band_toggle.setChecked(False)
        self.p0_band_toggle.stateChanged.connect(lambda state: self._on_toggle_p0_band(state == 2))
        p0_slider_row.addWidget(self.p0_band_toggle)
        pulse_layout.addLayout(p0_slider_row)

        self.tcode_freq_range_slider = RangeSliderWithLabel("Sent Value", 0, 9999, 2010, 7035, 0)
        pulse_layout.addWidget(self.tcode_freq_range_slider)

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
        self.pulse_enabled_checkbox.setChecked(False)
        pulse_mode_layout.addWidget(self.pulse_enabled_checkbox)
        pulse_mode_layout.addStretch()
        pulse_layout.addLayout(pulse_mode_layout)

        layout.addWidget(pulse_group)

        # ===== CARRIER FREQUENCY =====
        carrier_group = CollapsibleGroupBox("Carrier Frequency", collapsed=True)
        carrier_layout = QVBoxLayout(carrier_group)

        # Carrier Freq monitor slider with visibility toggle
        f0_slider_row = QHBoxLayout()
        self.f0_freq_range_slider = RangeSliderWithLabel("Monitor Freq (Hz)", 30, 22050, 30, 4000, 0, log_scale=True)
        self.f0_freq_range_slider.rangeChanged.connect(self._on_f0_band_change)
        f0_slider_row.addWidget(self.f0_freq_range_slider)
        self.f0_band_toggle = QCheckBox("Show")
        self.f0_band_toggle.setToolTip("Show/hide cyan overlay on spectrum")
        self.f0_band_toggle.setChecked(False)
        self.f0_band_toggle.stateChanged.connect(lambda state: self._on_toggle_f0_band(state == 2))
        f0_slider_row.addWidget(self.f0_band_toggle)
        carrier_layout.addLayout(f0_slider_row)

        self.f0_tcode_range_slider = RangeSliderWithLabel("Sent Value", 0, 9999, 0, 5000, 0)
        carrier_layout.addWidget(self.f0_tcode_range_slider)

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
        self.f0_enabled_checkbox.setChecked(False)
        f0_mode_layout.addWidget(self.f0_enabled_checkbox)
        f0_mode_layout.addStretch()
        carrier_layout.addLayout(f0_mode_layout)

        layout.addWidget(carrier_group)

        # ===== PULSE WIDTH =====
        p1_group = CollapsibleGroupBox("Pulse Width — higher = stronger, smoother", collapsed=True)
        p1_layout = QVBoxLayout(p1_group)

        self.p1_monitor_range_slider = RangeSliderWithLabel("Monitor Freq (Hz)", 30, 22050, 30, 4000, 0, log_scale=True)
        p1_layout.addWidget(self.p1_monitor_range_slider)

        self.p1_tcode_range_slider = RangeSliderWithLabel("Sent Value", 0, 9999, 1000, 8000, 0)
        p1_layout.addWidget(self.p1_tcode_range_slider)

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
        self.p1_enabled_checkbox.setChecked(False)
        p1_mode_layout.addWidget(self.p1_enabled_checkbox)
        p1_mode_layout.addStretch()
        p1_layout.addLayout(p1_mode_layout)

        layout.addWidget(p1_group)

        # ===== RISE TIME =====
        p3_group = CollapsibleGroupBox("Rise Time — higher = smoother, gentler", collapsed=True)
        p3_layout = QVBoxLayout(p3_group)

        self.p3_monitor_range_slider = RangeSliderWithLabel("Monitor Freq (Hz)", 30, 22050, 30, 4000, 0, log_scale=True)
        p3_layout.addWidget(self.p3_monitor_range_slider)

        self.p3_tcode_range_slider = RangeSliderWithLabel("Sent Value", 0, 9999, 1000, 8000, 0)
        p3_layout.addWidget(self.p3_tcode_range_slider)

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
        self.p3_enabled_checkbox.setChecked(False)
        p3_mode_layout.addWidget(self.p3_enabled_checkbox)
        p3_mode_layout.addStretch()
        p3_layout.addLayout(p3_mode_layout)

        layout.addWidget(p3_group)

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
                print(f"[Metric] peak_floor: valley={valley:.4f} ({direction}) → {new_val:.4f}")
        
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
                print(f"[Metric] target_bps: actual={actual_bps:.2f} target={target_bps:.2f} ({direction}) → pf={new_val:.4f}")
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
                print(f"[Metric] audio_amp: {reason} ({direction}) → {new_val:.4f}")
        
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
                print(f"[Metric] flux_balance: {reason} ({direction}) → fm={new_val:.2f}")
    
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
        layout.addWidget(detect_group)
        
        # ===== AUTO-ADJUST (METRIC-BASED AUTO-RANGING) =====
        metric_group = CollapsibleGroupBox("Auto-Adjust", collapsed=True)
        metric_layout = QVBoxLayout(metric_group)
        
        # Global enable/disable checkbox for all metrics
        self.metrics_global_cb = QCheckBox("Enable Auto-Adjust")
        self.metrics_global_cb.setChecked(self.config.auto_adjust.metrics_global_enabled)
        self.metrics_global_cb.setToolTip("Master toggle for all auto-adjust controls")
        self.metrics_global_cb.setStyleSheet("font-weight: bold; font-size: 10px;")
        self.metrics_global_cb.stateChanged.connect(self._on_metrics_global_toggle)
        metric_layout.addWidget(self.metrics_global_cb)
        
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
        self.auto_align_seconds_spin.setValue(1.2)
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
        
        layout.addWidget(metric_group)
        
        # Enable metrics based on config (first load = True, then saved)
        global_on = self.config.auto_adjust.metrics_global_enabled
        self.metric_peak_floor_cb.setChecked(global_on)
        self.metric_audio_amp_cb.setChecked(global_on)
        self.metric_flux_balance_cb.setChecked(global_on)
        self.metric_target_bps_cb.setChecked(global_on)
        if global_on:
            print("[Config] Auto-enabled 4 core metrics from config")
        
        # ===== LEVELS GROUP: Audio Amplification, Sensitivity, Flux Multiplier =====
        levels_group = CollapsibleGroupBox("Levels", collapsed=True)
        levels_layout = QVBoxLayout(levels_group)
        
        # Frequency band selection with visibility toggle (red beat detection band)
        beat_slider_row = QHBoxLayout()
        self.freq_range_slider = RangeSliderWithLabel("Freq Range (Hz)", 30, 22050, 30, 4000, 0, log_scale=True)
        self.freq_range_slider.rangeChanged.connect(self._on_freq_band_change)
        beat_slider_row.addWidget(self.freq_range_slider)
        self.beat_band_toggle = QCheckBox("Show")
        self.beat_band_toggle.setToolTip("Show/hide red overlay on spectrum")
        self.beat_band_toggle.setChecked(False)
        self.beat_band_toggle.stateChanged.connect(lambda state: self._on_toggle_beat_band(state == 2))
        beat_slider_row.addWidget(self.beat_band_toggle)
        levels_layout.addLayout(beat_slider_row)
        
        # Motion frequency cutoff: only generate strokes from bands below this Hz
        motion_cutoff_row = QHBoxLayout()
        motion_cutoff_label = QLabel("Motion Band Cutoff (Hz):")
        motion_cutoff_label.setStyleSheet("color: #CCC; font-size: 9px;")
        motion_cutoff_label.setToolTip("Only generate motion from beat bands below this frequency.\nHigher-band beats (hi-hat, cymbals) still feed BPM tracking but won't produce strokes.")
        motion_cutoff_row.addWidget(motion_cutoff_label)
        self.motion_freq_cutoff_spin = QSpinBox()
        self.motion_freq_cutoff_spin.setRange(60, 2000)
        self.motion_freq_cutoff_spin.setSingleStep(20)
        self.motion_freq_cutoff_spin.setValue(int(self.config.beat.motion_freq_cutoff))
        self.motion_freq_cutoff_spin.setSuffix(" Hz")
        self.motion_freq_cutoff_spin.setFixedWidth(80)
        self.motion_freq_cutoff_spin.setToolTip("Bands with lower Hz >= this value are filtered from motion generation")
        self.motion_freq_cutoff_spin.valueChanged.connect(self._on_motion_freq_cutoff_change)
        motion_cutoff_row.addWidget(self.motion_freq_cutoff_spin)
        motion_cutoff_row.addStretch()
        levels_layout.addLayout(motion_cutoff_row)
        
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
        
        layout.addWidget(levels_group)
        
        # ===== DEPTH/PEAKS GROUP: Depth, Peak Decay, Rise Sensitivity =====
        peaks_group = CollapsibleGroupBox("Depth & Peaks", collapsed=True)
        peaks_layout = QVBoxLayout(peaks_group)
        
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
        
        layout.addWidget(peaks_group)
        
        layout.addStretch()
        scroll_area.setWidget(widget)
        return scroll_area
    
    def _on_motion_freq_cutoff_change(self, value: int):
        """Handle motion frequency cutoff spinbox change"""
        self.config.beat.motion_freq_cutoff = float(value)
        print(f"[Config] Motion band cutoff: {value} Hz (bands with lower Hz >= {value} filtered from motion)")
    
    def _on_freq_band_change(self, low=None, high=None):
        """Update frequency band in config and spectrum overlay"""
        # Handle both range slider (low, high params) and direct calls
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
        self.spectrum_canvas.set_frequency_band(low / max_freq, high / max_freq)
        if hasattr(self, 'mountain_canvas'):
            self.mountain_canvas.set_frequency_band(low / max_freq, high / max_freq)
        if hasattr(self, 'bar_canvas'):
            self.bar_canvas.set_frequency_band(low / max_freq, high / max_freq)
        if hasattr(self, 'phosphor_canvas'):
            self.phosphor_canvas.set_frequency_band(low / max_freq, high / max_freq)
    
    def _on_depth_band_change(self, low=None, high=None):
        """Update stroke depth frequency band in config and spectrum overlay"""
        if low is None:
            low = self.depth_freq_range_slider.low() or 0.0
            high = self.depth_freq_range_slider.high() or 22050.0
        low = float(low)  # type: ignore
        high = float(high)  # type: ignore
        
        self.config.stroke.depth_freq_low = low
        self.config.stroke.depth_freq_high = high
        
        # Update spectrum overlay (green band)
        self.spectrum_canvas.set_depth_band(low, high)
        if hasattr(self, 'mountain_canvas'):
            self.mountain_canvas.set_depth_band(low, high)
        if hasattr(self, 'bar_canvas'):
            self.bar_canvas.set_depth_band(low, high)
        if hasattr(self, 'phosphor_canvas'):
            self.phosphor_canvas.set_depth_band(low, high)
    
    def _on_p0_band_change(self, low=None, high=None):
        """Update P0 TCode frequency band in config and spectrum overlay"""
        if low is None:
            low = self.pulse_freq_range_slider.low() or 0.0
            high = self.pulse_freq_range_slider.high() or 22050.0
        low = float(low)  # type: ignore
        high = float(high)  # type: ignore
        
        self.config.pulse_freq.monitor_freq_min = low
        self.config.pulse_freq.monitor_freq_max = high
        
        # Update spectrum overlay (blue band)
        self.spectrum_canvas.set_p0_band(low, high)
        if hasattr(self, 'mountain_canvas'):
            self.mountain_canvas.set_p0_band(low, high)
        if hasattr(self, 'bar_canvas'):
            self.bar_canvas.set_p0_band(low, high)
        if hasattr(self, 'phosphor_canvas'):
            self.phosphor_canvas.set_p0_band(low, high)
    
    def _on_f0_band_change(self, low=None, high=None):
        """Update F0 TCode frequency band in config and spectrum overlay"""
        if low is None:
            low = self.f0_freq_range_slider.low() or 0.0
            high = self.f0_freq_range_slider.high() or 22050.0
        low = float(low)  # type: ignore
        high = float(high)  # type: ignore
        
        self.config.carrier_freq.monitor_freq_min = low
        self.config.carrier_freq.monitor_freq_max = high
        
        # Update spectrum overlay (cyan band for F0)
        if hasattr(self, 'spectrum_canvas'):
            self.spectrum_canvas.set_f0_band(low, high)
        if hasattr(self, 'mountain_canvas'):
            self.mountain_canvas.set_f0_band(low, high)
        if hasattr(self, 'bar_canvas'):
            self.bar_canvas.set_f0_band(low, high)
        if hasattr(self, 'phosphor_canvas'):
            self.phosphor_canvas.set_f0_band(low, high)
    
    def _on_stroke_range_change(self, low: float, high: float):
        """Update stroke min/max in config"""
        self.config.stroke.stroke_min = low
        self.config.stroke.stroke_max = high

    def _on_motion_intensity_change(self, value: float):
        """Update motion intensity on the stroke mapper at runtime."""
        if hasattr(self, 'stroke_mapper') and self.stroke_mapper is not None:
            self.stroke_mapper.motion_intensity = value
        print(f"[Config] Motion intensity set to {value:.2f}")

    def _on_micro_effects_toggle(self, state):
        """Toggle micro-effects (beat jerks) in the stroke mapper."""
        enabled = state == 2
        if hasattr(self, 'stroke_mapper') and self.stroke_mapper is not None:
            self.stroke_mapper._micro_effects_enabled = enabled
        print(f"[Config] Micro-effects {'enabled' if enabled else 'disabled'}")

    def _on_amp_gate_high_change(self, value: float):
        """Update amplitude gate high threshold (above this -> FULL_STROKE)."""
        self.config.stroke.amplitude_gate_high = value
        print(f"[Config] Amplitude gate high set to {value:.2f}")

    def _on_amp_gate_low_change(self, value: float):
        """Update amplitude gate low threshold (below this -> CREEP_MICRO)."""
        self.config.stroke.amplitude_gate_low = value
        print(f"[Config] Amplitude gate low set to {value:.3f}")

    def _set_motion_preset(self, preset: str):
        """Apply a quick motion preset: gentle / normal / intense.
        
        Adjusts motion intensity (stroke size), z-score threshold,
        PATH 1 sensitivity, and rise sensitivity together to create
        coherent feel profiles — from smooth to jerky micro-motions.
        """
        presets = {
            'gentle':  {'motion': 0.50, 'zscore': 3.5, 'sensitivity': 0.30, 'rise_sens': 0.70},
            'normal':  {'motion': 1.00, 'zscore': 2.5, 'sensitivity': 0.50, 'rise_sens': 0.40},
            'intense': {'motion': 1.50, 'zscore': 1.8, 'sensitivity': 0.80, 'rise_sens': 0.10},
        }
        p = presets.get(preset, presets['normal'])
        # Motion intensity slider is independent — user controls stroke size separately
        self.zscore_threshold_slider.setValue(p['zscore'])
        self.sensitivity_slider.setValue(p['sensitivity'])
        self.rise_sens_slider.setValue(p['rise_sens'])
        print(f"[Config] Motion preset '{preset}': "
              f"z-score={p['zscore']}, sensitivity={p['sensitivity']}, "
              f"rise_sens={p['rise_sens']}")

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
        # Enable/disable related controls
        self.time_sig_combo.setEnabled(enabled)
        self.stability_threshold_slider.setEnabled(enabled)
        self.tempo_timeout_slider.setEnabled(enabled)
        self.phase_snap_slider.setEnabled(enabled)
        print(f"[Config] Tempo tracking {'enabled' if enabled else 'disabled'}")
    
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
    
    def _save_freq_preset(self, idx: int):
        """Save ALL settings from all 4 tabs to custom preset, with overwrite confirmation and optional rename"""
        from PyQt6.QtWidgets import QMessageBox, QInputDialog
        
        # Check if this slot already has a preset
        key = str(idx)
        custom_name = None
        if key in self.custom_beat_presets:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("WARNING - OVERWRITE PRESET")
            msg_box.setText(f"Preset {idx+1} already exists.\nAre you sure you want to overwrite it?")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            ok_button = msg_box.button(QMessageBox.StandardButton.Ok)
            rename_button = msg_box.addButton("Rename", QMessageBox.ButtonRole.AcceptRole)
            if ok_button:
                ok_button.setText("Overwrite")
            msg_box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            result = msg_box.exec()
            
            clicked_button = msg_box.clickedButton()
            if clicked_button == rename_button:
                # User wants to rename - show input dialog
                existing_name = self.custom_beat_presets[key].get('preset_name', f'Preset {idx+1}')
                new_name, ok = QInputDialog.getText(
                    self, "Rename Preset", 
                    "Enter a new name for this preset:",
                    text=existing_name
                )
                if ok and new_name.strip():
                    custom_name = new_name.strip()[:12]  # Limit to 12 chars for button display
                else:
                    print(f"[Config] Preset {idx+1} rename cancelled")
                    return
            elif result != QMessageBox.StandardButton.Ok:
                print(f"[Config] Preset {idx+1} overwrite cancelled")
                return

        preset_data = {
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
            'motion_intensity': self.motion_intensity_slider.value() if hasattr(self, 'motion_intensity_slider') else 1.0,
            'amp_gate_high': self.amp_gate_high_spin.value() if hasattr(self, 'amp_gate_high_spin') else 0.08,
            'amp_gate_low': self.amp_gate_low_spin.value() if hasattr(self, 'amp_gate_low_spin') else 0.04,
            'silence_reset_ms': int(self.silence_reset_slider.value()),
            'detection_type': self.detection_type_combo.currentIndex(),
            
            # Tempo Tracking (Beat Detection Tab)
            'tempo_tracking_enabled': self.tempo_tracking_checkbox.isChecked(),
            'time_sig_index': self.time_sig_combo.currentIndex(),
            'stability_threshold': self.stability_threshold_slider.value(),
            'tempo_timeout_ms': int(self.tempo_timeout_slider.value()),
            'phase_snap_weight': self.phase_snap_slider.value(),

            # Stroke Settings Tab
            'stroke_mode': self.mode_combo.currentIndex(),
            'stroke_min': self.stroke_range_slider.low(),
            'stroke_max': self.stroke_range_slider.high(),
            'min_interval_ms': int(self.min_interval_slider.value()),
            'stroke_fullness': self.fullness_slider.value(),
            'minimum_depth': self.min_depth_slider.value(),
            'freq_depth_factor': self.freq_depth_slider.value(),
            'flux_depth_factor': self.flux_depth_slider.value(),
            'depth_freq_low': self.depth_freq_range_slider.low(),
            'depth_freq_high': self.depth_freq_range_slider.high(),
            'flux_threshold': self.flux_threshold_slider.value(),
            'flux_scaling_weight': self.flux_scaling_slider.value(),
            'phase_advance': self.phase_advance_slider.value(),

            # Jitter / Creep Tab
            'jitter_enabled': self.jitter_enabled.isChecked(),
            'jitter_amplitude': self.jitter_amplitude_slider.value(),
            'jitter_intensity': self.jitter_intensity_slider.value(),
            'creep_enabled': self.creep_enabled.isChecked(),
            'creep_speed': self.creep_speed_slider.value(),
            'thump_enabled': self.config.stroke.thump_enabled,

            # Axis Weights Tab
            'alpha_weight': self.alpha_weight_slider.value(),
            'beta_weight': self.beta_weight_slider.value(),

            # Effects Tab
            'vol_reduction_limit': self.vol_reduction_limit_slider.value(),

            # Other Tab
            'pulse_freq_low': self.pulse_freq_range_slider.low(),
            'pulse_freq_high': self.pulse_freq_range_slider.high(),
            'tcode_min': int(self.tcode_freq_range_slider.low()),
            'tcode_max': int(self.tcode_freq_range_slider.high()),
            'freq_weight': self.freq_weight_slider.value(),
        }
        
        # Add custom name if provided
        if custom_name:
            preset_data['preset_name'] = custom_name
        
        self.custom_beat_presets[str(idx)] = preset_data
        # Mark this button as having a preset and make it active
        self.preset_buttons[idx].set_has_preset(True)
        self.preset_buttons[idx].set_active(True)
        
        # Update button text if custom name is set
        if custom_name:
            self.preset_buttons[idx].setText(custom_name)
        else:
            self.preset_buttons[idx].setText(str(idx + 1))
        
        # Deactivate other preset buttons
        for i, btn in enumerate(self.preset_buttons):
            if i != idx:
                btn.set_active(False)
        self._save_presets_to_disk()
        print(f"[Config] Saved preset {idx+1}{' (' + custom_name + ')' if custom_name else ''} with all settings")
    
    def _load_freq_preset(self, idx: int):
        """Load ALL settings from all 4 tabs from custom preset"""
        from config import StrokeMode
        key = str(idx)
        if key in self.custom_beat_presets:
            # Capture current settings before loading for revert functionality
            self._revert_settings = self._capture_current_settings()
            self.revert_btn.setEnabled(True)
            
            preset_data = self.custom_beat_presets[key]
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
            if 'motion_intensity' in preset_data and hasattr(self, 'motion_intensity_slider'):
                self.motion_intensity_slider.setValue(preset_data['motion_intensity'])
            if 'amp_gate_high' in preset_data and hasattr(self, 'amp_gate_high_spin'):
                self.amp_gate_high_spin.setValue(preset_data['amp_gate_high'])
            if 'amp_gate_low' in preset_data and hasattr(self, 'amp_gate_low_spin'):
                self.amp_gate_low_spin.setValue(preset_data['amp_gate_low'])
            if 'silence_reset_ms' in preset_data:
                self.silence_reset_slider.setValue(preset_data['silence_reset_ms'])
            self.detection_type_combo.setCurrentIndex(preset_data['detection_type'])
            
            # Tempo Tracking settings
            if 'tempo_tracking_enabled' in preset_data:
                self.tempo_tracking_checkbox.setChecked(preset_data['tempo_tracking_enabled'])
                self._on_tempo_tracking_toggle(2 if preset_data['tempo_tracking_enabled'] else 0)
            if 'time_sig_index' in preset_data:
                self.time_sig_combo.setCurrentIndex(preset_data['time_sig_index'])
                self._on_time_sig_change(preset_data['time_sig_index'])
            if 'stability_threshold' in preset_data:
                self.stability_threshold_slider.setValue(preset_data['stability_threshold'])
                self._on_stability_threshold_change(preset_data['stability_threshold'])
            if 'tempo_timeout_ms' in preset_data:
                self.tempo_timeout_slider.setValue(preset_data['tempo_timeout_ms'])
                self._on_tempo_timeout_change(preset_data['tempo_timeout_ms'])
            if 'phase_snap_weight' in preset_data:
                self.phase_snap_slider.setValue(preset_data['phase_snap_weight'])
                self._on_phase_snap_change(preset_data['phase_snap_weight'])
            
            # Stroke Settings Tab
            self.mode_combo.setCurrentIndex(preset_data['stroke_mode'])
            self._on_mode_change(preset_data['stroke_mode'])  # Apply axis weight limits for this mode
            self.stroke_range_slider.setLow(preset_data['stroke_min'])
            self.stroke_range_slider.setHigh(preset_data['stroke_max'])
            self.min_interval_slider.setValue(preset_data['min_interval_ms'])
            self.fullness_slider.setValue(preset_data['stroke_fullness'])
            self.min_depth_slider.setValue(preset_data['minimum_depth'])
            self.freq_depth_slider.setValue(preset_data['freq_depth_factor'])
            if 'flux_depth_factor' in preset_data:
                self.flux_depth_slider.setValue(preset_data['flux_depth_factor'])
            if 'depth_freq_low' in preset_data:
                self.depth_freq_range_slider.setLow(preset_data['depth_freq_low'])
            if 'depth_freq_high' in preset_data:
                self.depth_freq_range_slider.setHigh(preset_data['depth_freq_high'])
            self.flux_threshold_slider.setValue(preset_data['flux_threshold'])
            if 'flux_scaling_weight' in preset_data:
                self.flux_scaling_slider.setValue(preset_data['flux_scaling_weight'])
            if 'phase_advance' in preset_data:
                self.phase_advance_slider.setValue(preset_data['phase_advance'])
            # Jitter / Creep Tab
            self.jitter_enabled.setChecked(preset_data['jitter_enabled'])
            self.jitter_amplitude_slider.setValue(preset_data['jitter_amplitude'])
            self.jitter_intensity_slider.setValue(preset_data['jitter_intensity'])
            self.creep_enabled.setChecked(preset_data['creep_enabled'])
            self.creep_speed_slider.setValue(preset_data['creep_speed'])
            if 'thump_enabled' in preset_data:
                self.config.stroke.thump_enabled = preset_data['thump_enabled']
            # Axis Weights Tab
            self.alpha_weight_slider.setValue(preset_data['alpha_weight'])
            self.beta_weight_slider.setValue(preset_data['beta_weight'])

            # Effects Tab
            if 'vol_reduction_limit' in preset_data:
                self.vol_reduction_limit_slider.setValue(preset_data['vol_reduction_limit'])

            # Other Tab
            if 'pulse_freq_low' in preset_data:
                self.pulse_freq_range_slider.setLow(preset_data['pulse_freq_low'])
            if 'pulse_freq_high' in preset_data:
                self.pulse_freq_range_slider.setHigh(preset_data['pulse_freq_high'])
            # Support both new/old preset keys and legacy Hz-scale values
            p0_tcode_min, p0_tcode_max = resolve_p0_tcode_bounds(preset_data)
            if p0_tcode_min is not None:
                self.tcode_freq_range_slider.setLow(p0_tcode_min)
            if p0_tcode_max is not None:
                self.tcode_freq_range_slider.setHigh(p0_tcode_max)
            if 'freq_weight' in preset_data:
                self.freq_weight_slider.setValue(preset_data['freq_weight'])

            # --- Sync config object with UI (especially enum) ---
            self.config.stroke.mode = StrokeMode(self.mode_combo.currentIndex() + 1)
            
            # Mark this preset as active, deactivate others
            for i, btn in enumerate(self.preset_buttons):
                btn.set_active(i == idx)
            
            print(f"[Config] Loaded preset {idx+1} with all settings")
        else:
            print(f"[Config] Preset {idx+1} not saved yet")
    
    def _save_beat_preset(self, idx: int):
        """Alias for _save_freq_preset (called by right-click)"""
        self._save_freq_preset(idx)
    
    def _load_beat_preset(self, idx: int):
        """Alias for _load_freq_preset (called by left-click)"""
        self._load_freq_preset(idx)
    
    def _get_presets_file_path(self) -> Path:
        """Get the path to the presets file - exe folder when packaged, workspace when developing"""
        return get_presets_file_path(
            frozen=getattr(sys, 'frozen', False),
            executable_path=str(sys.executable),
            source_file=__file__,
        )
    
    def _save_presets_to_disk(self):
        """Save all custom presets to disk"""
        try:
            presets_file = self._get_presets_file_path()
            save_presets_data(presets_file, self.custom_beat_presets)
            print(f"[Presets] Saved {len(self.custom_beat_presets)} presets to {presets_file}")
        except Exception as e:
            print(f"[Presets] Error saving presets: {e}")
    
    def _load_presets_from_disk(self):
        """Load custom presets from disk"""
        try:
            presets_file = self._get_presets_file_path()
            was_missing = not presets_file.exists()
            self.custom_beat_presets = load_presets_data(
                presets_file,
                frozen=getattr(sys, 'frozen', False),
                meipass=getattr(sys, '_MEIPASS', None),
            )

            if was_missing and presets_file.exists() and self.custom_beat_presets:
                print(f"[Presets] Copied factory presets to {presets_file}")

            if self.custom_beat_presets:
                # Mark buttons that have saved presets and apply custom names
                for idx, preset_data in self.custom_beat_presets.items():
                    idx_int = int(idx)
                    if idx_int < len(self.preset_buttons):
                        self.preset_buttons[idx_int].set_has_preset(True)
                        # Apply custom name if stored
                        if isinstance(preset_data, dict) and 'preset_name' in preset_data:
                            self.preset_buttons[idx_int].setText(preset_data['preset_name'])
                print(f"[Presets] Loaded {len(self.custom_beat_presets)} presets from {presets_file}")
            else:
                print(f"[Presets] No presets file found, starting with empty presets")
        except Exception as e:
            print(f"[Presets] Error loading presets: {e}")
            self.custom_beat_presets = {}
    
    def _create_stroke_settings_tab(self) -> QWidget:
        """Stroke generation settings"""
        scroll_area = NoWheelScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(self._get_thin_scrollbar_style())

        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Stroke Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["1: Circle", "2: Spiral", "3: Teardrop", "4: User (Flux/Peak)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        # ===== MOTION INTENSITY =====
        motion_group = QGroupBox("Motion Intensity")
        motion_layout = QVBoxLayout(motion_group)
        
        # Motion intensity slider (0.25 - 2.0, default 1.0) — scales all stroke output
        self.motion_intensity_slider = SliderWithLabel("Intensity", 0.25, 2.0, 1.0)
        self.motion_intensity_slider.valueChanged.connect(self._on_motion_intensity_change)
        motion_layout.addWidget(self.motion_intensity_slider)
        
        # Quick preset buttons: Gentle / Normal / Intense
        motion_btn_layout = QHBoxLayout()
        self.motion_gentle_btn = QPushButton("Gentle")
        self.motion_gentle_btn.setToolTip("Low motion: intensity 0.50, z-score threshold 3.5")
        self.motion_gentle_btn.clicked.connect(lambda: self._set_motion_preset('gentle'))
        self.motion_gentle_btn.setStyleSheet("QPushButton { background: #335; color: #8af; }")
        motion_btn_layout.addWidget(self.motion_gentle_btn)
        
        self.motion_normal_btn = QPushButton("Normal")
        self.motion_normal_btn.setToolTip("Default motion: intensity 1.0, z-score threshold 2.5")
        self.motion_normal_btn.clicked.connect(lambda: self._set_motion_preset('normal'))
        self.motion_normal_btn.setStyleSheet("QPushButton { background: #353; color: #8f8; }")
        motion_btn_layout.addWidget(self.motion_normal_btn)
        
        self.motion_intense_btn = QPushButton("Intense")
        self.motion_intense_btn.setToolTip("High motion: intensity 1.50, z-score threshold 1.8")
        self.motion_intense_btn.clicked.connect(lambda: self._set_motion_preset('intense'))
        self.motion_intense_btn.setStyleSheet("QPushButton { background: #533; color: #f88; }")
        motion_btn_layout.addWidget(self.motion_intense_btn)
        
        motion_layout.addLayout(motion_btn_layout)
        
        # Micro-effects toggle (jerks on beats during low-amplitude creep mode)
        self.micro_effects_checkbox = QCheckBox("Micro-effects (beat jerks in creep mode)")
        self.micro_effects_checkbox.setChecked(True)
        self.micro_effects_checkbox.setToolTip("When enabled, small impulse jerks fire on beats during low-amplitude passages")
        self.micro_effects_checkbox.stateChanged.connect(self._on_micro_effects_toggle)
        motion_layout.addWidget(self.micro_effects_checkbox)
        
        # Amplitude gate thresholds (FULL_STROKE vs CREEP_MICRO switching)
        gate_layout = QHBoxLayout()
        gate_layout.addWidget(QLabel("Amp Gate:"))
        
        gate_layout.addWidget(QLabel("High"))
        self.amp_gate_high_spin = QDoubleSpinBox()
        self.amp_gate_high_spin.setRange(0.01, 0.50)
        self.amp_gate_high_spin.setSingleStep(0.01)
        self.amp_gate_high_spin.setDecimals(2)
        self.amp_gate_high_spin.setValue(self.config.stroke.amplitude_gate_high)
        self.amp_gate_high_spin.setToolTip("RMS above this triggers full arc strokes (FULL_STROKE mode)")
        self.amp_gate_high_spin.setFixedWidth(70)
        self.amp_gate_high_spin.valueChanged.connect(self._on_amp_gate_high_change)
        gate_layout.addWidget(self.amp_gate_high_spin)
        
        gate_layout.addWidget(QLabel("Low"))
        self.amp_gate_low_spin = QDoubleSpinBox()
        self.amp_gate_low_spin.setRange(0.001, 0.40)
        self.amp_gate_low_spin.setSingleStep(0.01)
        self.amp_gate_low_spin.setDecimals(3)
        self.amp_gate_low_spin.setValue(self.config.stroke.amplitude_gate_low)
        self.amp_gate_low_spin.setToolTip("RMS below this drops to creep rotation (CREEP_MICRO mode)")
        self.amp_gate_low_spin.setFixedWidth(70)
        self.amp_gate_low_spin.valueChanged.connect(self._on_amp_gate_low_change)
        gate_layout.addWidget(self.amp_gate_low_spin)
        
        gate_layout.addStretch()
        motion_layout.addLayout(gate_layout)
        
        layout.addWidget(motion_group)
        
        # ===== STROKE PARAMETERS =====
        params_group = CollapsibleGroupBox("Stroke Parameters", collapsed=True)
        params_layout = QVBoxLayout(params_group)
        
        self.stroke_range_slider = RangeSliderWithLabel("Stroke Min/Max", 0.0, 1.0, 0.2, 1.0, 2)
        self.stroke_range_slider.rangeChanged.connect(self._on_stroke_range_change)
        params_layout.addWidget(self.stroke_range_slider)
        
        self.min_interval_slider = SliderWithLabel("Min Interval (ms)", 50, 5000, 100, 0)
        self.min_interval_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'min_interval_ms', int(v)))
        params_layout.addWidget(self.min_interval_slider)
        
        self.fullness_slider = SliderWithLabel("Stroke Fullness", 0.0, 1.0, 0.7)
        self.fullness_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'stroke_fullness', v))
        params_layout.addWidget(self.fullness_slider)
        
        self.min_depth_slider = SliderWithLabel("Minimum Depth", 0.0, 1.0, 0.0)
        self.min_depth_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'minimum_depth', v))
        params_layout.addWidget(self.min_depth_slider)
        
        self.freq_depth_slider = SliderWithLabel("Freq Depth Factor", 0.0, 2.0, 0.3)
        self.freq_depth_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'freq_depth_factor', v))
        params_layout.addWidget(self.freq_depth_slider)
        
        self.flux_depth_slider = SliderWithLabel("Flux Rise Depth Factor", 0.0, 5.0, 0.0)
        self.flux_depth_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'flux_depth_factor', v))
        params_layout.addWidget(self.flux_depth_slider)
        
        layout.addWidget(params_group)
        
        # Frequency range for stroke depth - shown as green overlay on spectrum
        depth_freq_group = CollapsibleGroupBox("Depth Frequency Range (Hz) - green overlay", collapsed=True)
        depth_freq_layout = QVBoxLayout(depth_freq_group)
        
        # Depth Freq slider with visibility toggle (green stroke depth band)
        depth_slider_row = QHBoxLayout()
        self.depth_freq_range_slider = RangeSliderWithLabel("Depth Freq (Hz)", 30, 22050, 30, 4000, 0, log_scale=True)
        self.depth_freq_range_slider.rangeChanged.connect(self._on_depth_band_change)
        depth_slider_row.addWidget(self.depth_freq_range_slider)
        self.depth_band_toggle = QCheckBox("Show")
        self.depth_band_toggle.setToolTip("Show/hide green overlay on spectrum")
        self.depth_band_toggle.setChecked(False)
        self.depth_band_toggle.stateChanged.connect(lambda state: self._on_toggle_depth_band(state == 2))
        depth_slider_row.addWidget(self.depth_band_toggle)
        depth_freq_layout.addLayout(depth_slider_row)
        
        layout.addWidget(depth_freq_group)

        layout.addStretch()
        scroll_area.setWidget(widget)
        return scroll_area

    def _create_jitter_creep_tab(self) -> QWidget:
        """Effects (jitter + creep) and axis weight settings"""
        scroll_area = NoWheelScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(self._get_thin_scrollbar_style())

        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Combined Effects section (windowshade)
        effects_group = CollapsibleGroupBox("Effects", collapsed=True)
        effects_layout = QVBoxLayout(effects_group)

        self.jitter_enabled = QCheckBox("Jitter")
        self.jitter_enabled.setChecked(True)
        self.jitter_enabled.stateChanged.connect(lambda s: setattr(self.config.jitter, 'enabled', s == 2))
        effects_layout.addWidget(self.jitter_enabled)

        self.jitter_amplitude_slider = SliderWithLabel("Circle Size", 0.01, 0.1, 0.1, 3)
        self.jitter_amplitude_slider.valueChanged.connect(lambda v: setattr(self.config.jitter, 'amplitude', v))
        effects_layout.addWidget(self.jitter_amplitude_slider)

        self.jitter_intensity_slider = SliderWithLabel("Circle Speed", 0.0, 10.0, 0.5)
        self.jitter_intensity_slider.valueChanged.connect(lambda v: setattr(self.config.jitter, 'intensity', v))
        effects_layout.addWidget(self.jitter_intensity_slider)

        self.creep_enabled = QCheckBox("Creep")
        self.creep_enabled.setChecked(True)
        self.creep_enabled.stateChanged.connect(lambda s: setattr(self.config.creep, 'enabled', s == 2))
        effects_layout.addWidget(self.creep_enabled)

        self.creep_speed_slider = SliderWithLabel("Creep Speed", 0.0, 2.0, 0.02, 3)
        self.creep_speed_slider.valueChanged.connect(lambda v: setattr(self.config.creep, 'speed', v))
        effects_layout.addWidget(self.creep_speed_slider)

        layout.addWidget(effects_group)

        # Axis Weights section (moved from Stroke Settings)
        axis_group = QGroupBox("Axis Weights")
        axis_layout = QVBoxLayout(axis_group)

        self.alpha_weight_slider = SliderWithLabel("Alpha Weight", 0.0, 2.0, 1.0)
        self.alpha_weight_slider.valueChanged.connect(lambda v: setattr(self.config, 'alpha_weight', v))
        axis_layout.addWidget(self.alpha_weight_slider)

        self.beta_weight_slider = SliderWithLabel("Beta Weight", 0.0, 2.0, 1.0)
        self.beta_weight_slider.valueChanged.connect(lambda v: setattr(self.config, 'beta_weight', v))
        axis_layout.addWidget(self.beta_weight_slider)

        # Set initial tooltips (updated dynamically by _on_mode_change)
        self._update_axis_weight_tooltips()

        layout.addWidget(axis_group)

        # Volume Reduction Limit
        vol_limit_group = QGroupBox("Volume Reduction Limit")
        vol_limit_layout = QVBoxLayout(vol_limit_group)
        vol_limit_layout.addWidget(QLabel("Max % volume can be reduced by band/fade/creep effects"))

        self.vol_reduction_limit_slider = SliderWithLabel("Max Reduction %", 0, 20, 10, 0)
        self.vol_reduction_limit_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'vol_reduction_limit', v))
        vol_limit_layout.addWidget(self.vol_reduction_limit_slider)

        layout.addWidget(vol_limit_group)

        layout.addStretch()
        scroll_area.setWidget(widget)
        return scroll_area

    def _update_axis_weight_tooltips(self):
        """Update axis weight slider tooltips based on current stroke mode"""
        mode = getattr(self.config.stroke, 'mode', None)
        if mode and hasattr(mode, 'value'):
            mode_val = mode.value
        else:
            mode_val = 1
        if mode_val <= 3:
            tip = "Modes 1-3: Scales axis amplitude (0=off, 1=normal, max 1.25)"
        else:
            tip = "Mode 4 (User): Controls flux/peak response (0=flux, 1=balanced, 2=peak)"
        self.alpha_weight_slider.setToolTip(tip)
        self.beta_weight_slider.setToolTip(tip)
    
    def _create_tempo_tracking_tab(self) -> QWidget:
        """Tempo tracking and rhythm settings"""
        scroll_area = NoWheelScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(self._get_thin_scrollbar_style())

        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # ===== TEMPO SETTINGS =====
        tempo_group = QGroupBox("Tempo Settings")
        tempo_layout = QVBoxLayout(tempo_group)
        
        # Enable/disable checkbox
        self.tempo_tracking_checkbox = QCheckBox("Enable Tempo Tracking")
        self.tempo_tracking_checkbox.setChecked(True)
        self.tempo_tracking_checkbox.stateChanged.connect(self._on_tempo_tracking_toggle)
        tempo_layout.addWidget(self.tempo_tracking_checkbox)
        
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
        
        layout.addWidget(tempo_group)
        
        # Spectral flux control group
        flux_group = CollapsibleGroupBox("Spectral Flux Control", collapsed=True)
        flux_layout = QVBoxLayout(flux_group)
        flux_layout.addWidget(QLabel("Low flux→downbeats only, High flux→every beat"))
        
        self.flux_threshold_slider = SliderWithLabel("Flux Threshold", 0.001, 0.2, 0.03, 4)
        self.flux_threshold_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'flux_threshold', v))
        flux_layout.addWidget(self.flux_threshold_slider)
        
        self.flux_scaling_slider = SliderWithLabel("Flux Scaling (size)", 0.0, 2.0, 1.0, 2)
        self.flux_scaling_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'flux_scaling_weight', v))
        flux_layout.addWidget(self.flux_scaling_slider)
        
        self.phase_advance_slider = SliderWithLabel("Phase Advance (0=downbeats, 1=all)", 0.0, 1.0, self.config.stroke.phase_advance, 2)
        self.phase_advance_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'phase_advance', v))
        flux_layout.addWidget(self.phase_advance_slider)
        
        layout.addWidget(flux_group)

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
    
    def _on_test(self):
        """Send test pattern"""
        _, should_restore = trigger_network_test(self.network_engine)
        # Restore after a delay (test takes ~2.5 seconds)
        if should_restore:
            QTimer.singleShot(3000, lambda: set_transport_sending(self.network_engine, False))
    
    def _on_start_stop(self, checked: bool):
        """Start/stop audio capture and TCode pipeline.
        Start enables TCode sending (V0=0 until Play). Stop kills everything."""
        if checked:
            self._start_engines()
            ui_state = start_stop_ui_state(True)
            self.start_btn.setText(ui_state.start_text)
            self.play_btn.setEnabled(ui_state.play_enabled)
            # Enable TCode sending immediately on Start (V0=0 until Play is pressed)
            set_transport_sending(self.network_engine, True)
            send_zero_volume_immediate(self.network_engine, duration_ms=100)
        else:
            # Send zero-volume command before stopping (always, not just when is_sending)
            self._volume_ramp_active = False
            send_zero_volume_immediate(self.network_engine, duration_ms=100)
            set_transport_sending(self.network_engine, False)
            self._stop_engines()
            ui_state = start_stop_ui_state(False)
            self.start_btn.setText(ui_state.start_text)
            self.play_btn.setEnabled(ui_state.play_enabled)
            if ui_state.play_reset_checked:
                self.play_btn.setChecked(False)
            if ui_state.play_text is not None:
                self.play_btn.setText(ui_state.play_text)
            if ui_state.is_sending is not None:
                self.is_sending = ui_state.is_sending
            # Note: Auto-range state is preserved across stop/start - no reset here
    
    def _on_play_pause(self, checked: bool):
        """Play/pause motion generation. Pause sends V0=0 but keeps TCode pipeline active."""
        self.is_sending = checked
        if checked:
            # Re-instantiate StrokeMapper with current config (for live mode switching)
            self.stroke_mapper = StrokeMapper(self.config, self._send_command_direct, get_volume=lambda: self.volume_slider.value() / 100.0, audio_engine=self.audio_engine)
            self.stroke_mapper.motion_intensity = self.motion_intensity_slider.value()
            if hasattr(self, 'micro_effects_checkbox'):
                self.stroke_mapper._micro_effects_enabled = self.micro_effects_checkbox.isChecked()
            # Start volume ramp from 0 to set value over 1.3s
            ramp_state = begin_volume_ramp(time.time())
            self._volume_ramp_active = ramp_state.active
            self._volume_ramp_start_time = ramp_state.start_time
            self._volume_ramp_from = ramp_state.from_volume
            self._volume_ramp_to = ramp_state.to_volume
            # sending_enabled already True from Start — no need to set again
        else:
            # Send V0=0 immediately with fade, but keep TCode pipeline active
            self._volume_ramp_active = False
            send_zero_volume_immediate(self.network_engine, duration_ms=500)
            # DON'T disable sending_enabled — connection stays active until Stop
        self.play_btn.setText(play_button_text(checked))
    
    def _on_detection_type_change(self, index: int):
        """Change beat detection type"""
        self.config.beat.detection_type = BeatDetectionType(index + 1)
    
    def _on_mode_change(self, index: int):
        """Change stroke mode and adjust axis weight slider limits"""
        self.config.stroke.mode = StrokeMode(index + 1)
        
        # Modes 1-3: limit axis weights to 1.25 max
        # Mode 4 (USER): keep full 0-2 range for peak/flux balance control
        if index < 3:  # Modes 1, 2, 3 (Circle, Spiral, Teardrop)
            new_max = 1.25
            # Clamp current values if they exceed new max
            if self.alpha_weight_slider.value() > new_max:
                self.alpha_weight_slider.setValue(new_max)
            if self.beta_weight_slider.value() > new_max:
                self.beta_weight_slider.setValue(new_max)
            # Update slider max (need to update the internal slider)
            self.alpha_weight_slider.slider.setMaximum(int(new_max * self.alpha_weight_slider.multiplier))
            self.beta_weight_slider.slider.setMaximum(int(new_max * self.beta_weight_slider.multiplier))
            self.alpha_weight_slider.max_val = new_max
            self.beta_weight_slider.max_val = new_max
        else:  # Mode 4 (USER)
            new_max = 2.0
            self.alpha_weight_slider.slider.setMaximum(int(new_max * self.alpha_weight_slider.multiplier))
            self.beta_weight_slider.slider.setMaximum(int(new_max * self.beta_weight_slider.multiplier))
            self.alpha_weight_slider.max_val = new_max
            self.beta_weight_slider.max_val = new_max

        # Update axis weight tooltips for the current mode
        self._update_axis_weight_tooltips()

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

        # Sync metric checkbox states to the new audio engine
        # (checkboxes may already be checked from previous start)
        self._sync_metric_checkboxes_to_engine()

        self.stroke_mapper = StrokeMapper(self.config, self._send_command_direct, get_volume=lambda: self.volume_slider.value() / 100.0, audio_engine=self.audio_engine)
        self.stroke_mapper.motion_intensity = self.motion_intensity_slider.value()
        if hasattr(self, 'micro_effects_checkbox'):
            self.stroke_mapper._micro_effects_enabled = self.micro_effects_checkbox.isChecked()

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
            self.network_engine.send_command(cmd)
    
    def _stop_engines(self):
        """Stop all engines and background threads"""
        self.is_running = False

        # Clear any active trajectory on the stroke mapper
        if self.stroke_mapper and hasattr(self.stroke_mapper, '_trajectory'):
            self.stroke_mapper._trajectory = None
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
                spectrum_with_stats = {
                    'spectrum': spectrum,
                    'peak_energy': event.peak_energy,
                    'spectral_flux': event.spectral_flux
                }
                self.signals.spectrum_ready.emit(spectrum_with_stats)

        # Process through stroke mapper
        if self.stroke_mapper and self.is_sending:
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
        
        # --- P0 (Pulse Frequency) with 250ms sliding window averaging ---
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
            
            # Send P0 with 250ms duration for smooth transitions
            cmd.pulse_freq = p0_val
            cmd.pulse_freq_duration = int(self._freq_window_ms)  # 250ms duration
            self._cached_p0_val = p0_val
            # Display raw TCode; append Hz if device limits configured
            dl = self.config.device_limits
            if dl.p0_freq_min > 0 and dl.p0_freq_max > 0:
                hz = dl.p0_freq_min + (p0_val / 9999.0) * (dl.p0_freq_max - dl.p0_freq_min)
                self._cached_pulse_display = f"Pulse Freq: {p0_val} ({hz:.0f}Hz)"
            else:
                self._cached_pulse_display = f"Pulse Freq: {p0_val}"
        else:
            cmd.pulse_freq = None
            self._cached_p0_val = None
            self._cached_pulse_display = "Pulse Freq: off"
            self._p0_freq_window.clear()  # Clear window when disabled
        
        # --- F0 (Carrier Frequency) with 250ms sliding window averaging ---
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
                        # Clamp new target: max ±500 from current position
                        delta_from_current = f0_val_raw - self._c0_band_current
                        delta_from_current = max(-500, min(500, delta_from_current))
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
            
            # Generate random duration: 900ms ±200ms
            f0_duration = int(self._f0_duration_base_ms + random.uniform(-self._f0_duration_variance_ms, self._f0_duration_variance_ms))
            f0_duration = max(100, f0_duration)  # Minimum 100ms
            
            if cmd.tcode_tags is None:
                cmd.tcode_tags = {}
            cmd.tcode_tags['C0'] = f0_val  # restim uses C0 for carrier frequency, not F0
            cmd.tcode_tags['C0_duration'] = f0_duration
            self._cached_f0_val = f0_val
            # Display raw TCode; append Hz if device limits configured
            dl = self.config.device_limits
            if dl.c0_freq_min > 0 and dl.c0_freq_max > 0:
                hz = dl.c0_freq_min + (f0_val / 9999.0) * (dl.c0_freq_max - dl.c0_freq_min)
                self._cached_carrier_display = f"Carrier Freq: {f0_val} ({hz:.0f}Hz)"
            else:
                self._cached_carrier_display = f"Carrier Freq: {f0_val}"
        else:
            self._cached_f0_val = None
            self._cached_carrier_display = "Carrier Freq: off"
            self._f0_freq_window.clear()  # Clear window when disabled
            self._f0_last_sent_tcode = None  # Reset smoothing state when disabled
        
        # --- P1 (Pulse Width) with 250ms sliding window averaging ---
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
            # Display raw TCode; append converted value if device limits configured
            dl = self.config.device_limits
            if dl.p1_cycles_min > 0 and dl.p1_cycles_max > 0:
                p1_cyc = dl.p1_cycles_min + (p1_val / 9999.0) * (dl.p1_cycles_max - dl.p1_cycles_min)
                self._cached_p1_display = f"Pulse Width: {p1_val} ({p1_cyc:.1f}cyc)"
            else:
                self._cached_p1_display = f"Pulse Width: {p1_val}"
        else:
            self._cached_p1_val = None
            self._cached_p1_display = "Pulse Width: off"
            self._p1_window.clear()
        
        # --- P3 (Rise Time) with 250ms sliding window averaging ---
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
            # Display raw TCode; append converted value if device limits configured
            dl = self.config.device_limits
            if dl.p3_cycles_min > 0 and dl.p3_cycles_max > 0:
                p3_cyc = dl.p3_cycles_min + (p3_val / 9999.0) * (dl.p3_cycles_max - dl.p3_cycles_min)
                self._cached_p3_display = f"Rise Time: {p3_val} ({p3_cyc:.1f}cyc)"
            else:
                self._cached_p3_display = f"Rise Time: {p3_val}"
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
        print(f"[Main] Cmd: a={cmd.alpha:.2f} b={cmd.beta:.2f} v={cmd.volume:.2f} {p0_str} {c0_str} {p1_str} {p3_str}")
    
    def _network_status_callback(self, message: str, connected: bool):
        """Called from network thread on status change"""
        self.signals.status_changed.emit(message, connected)
    
    def _on_beat(self, event: BeatEvent):
        """Handle beat event in GUI thread"""
        # ===== METRONOME SYNC INDICATOR (updates every frame, not just on beat) =====
        acf_conf = getattr(event, 'acf_confidence', 0.0)
        metro_bpm = getattr(event, 'metronome_bpm', 0.0)
        if hasattr(self, 'metronome_sync_indicator') and self.metronome_sync_indicator is not None:
            if metro_bpm <= 0 or acf_conf < 0.05:
                self.metronome_sync_indicator.setStyleSheet("color: #333; font-size: 20px;")  # Off
            elif acf_conf < 0.25:
                self.metronome_sync_indicator.setStyleSheet("color: #cc0; font-size: 20px;")  # Yellow: locking
            else:
                self.metronome_sync_indicator.setStyleSheet("color: #0f0; font-size: 20px;")  # Green: locked

        # Update metronome BPM display (small label next to target BPM controls)
        if hasattr(self, 'bpm_actual_label'):
            if metro_bpm > 0:
                self.bpm_actual_label.setText(f"Metro: {metro_bpm:.0f} BPM")
            else:
                self.bpm_actual_label.setText("Metro: -- BPM")

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
                    confidence = tempo_info['confidence']
                    # Use event.is_downbeat (frozen at construction time) instead of
                    # polling get_tempo_info() which races with audio thread clearing the flag
                    is_downbeat = getattr(event, 'is_downbeat', False)
                    stability = tempo_info.get('stability', 0.0)
                    
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
                    
                    # Format BPM display — show ACF info when metronome is active
                    acf_active = tempo_info.get('acf_active', False)
                    if acf_active:
                        bpm_display = f"BPM: {tempo_info['bpm']:.1f}"
                        acf_c = tempo_info.get('acf_confidence', 0.0)
                        if acf_c < 0.15:
                            bpm_display += " (~)"
                    else:
                        bpm_display = f"BPM: {tempo_info['bpm']:.1f}"
                        if confidence < 0.5:
                            bpm_display += " (~)"
                        elif stability < 0.5:
                            bpm_display += " (~)"
                    if hasattr(self, 'bpm_label') and self.bpm_label is not None:
                        self.bpm_label.setText(bpm_display)
        # Show reset in GUI and console if tempo was reset
        if hasattr(event, 'tempo_reset') and event.tempo_reset:
            if hasattr(self, 'bpm_label') and self.bpm_label is not None:
                self.bpm_label.setText("BPM: ---")
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
    
    def _do_spectrum_update(self):
        """Actually update spectrum at throttled rate - only update visible visualizer"""
        if self._pending_spectrum is not None and hasattr(self, 'spectrum_canvas') and self.spectrum_canvas is not None:
            # Handle both old format (numpy array) and new format (dict with stats)
            if isinstance(self._pending_spectrum, dict):
                spectrum = self._pending_spectrum['spectrum']
                peak = self._pending_spectrum.get('peak_energy', 0)
                flux = self._pending_spectrum.get('spectral_flux', 0)
                # Only update the currently visible visualizer for performance
                if self.spectrum_canvas.isVisible():
                    self.spectrum_canvas.update_spectrum(spectrum, peak, flux)
                elif hasattr(self, 'mountain_canvas') and self.mountain_canvas is not None and self.mountain_canvas.isVisible():
                    self.mountain_canvas.update_spectrum(spectrum, peak, flux)
                elif hasattr(self, 'bar_canvas') and self.bar_canvas is not None and self.bar_canvas.isVisible():
                    self.bar_canvas.update_spectrum(spectrum, peak, flux)
                elif hasattr(self, 'phosphor_canvas') and self.phosphor_canvas is not None and self.phosphor_canvas.isVisible():
                    self.phosphor_canvas.update_spectrum(spectrum, peak, flux)
            else:
                # Legacy format - only update visible visualizer
                if self.spectrum_canvas.isVisible():
                    self.spectrum_canvas.update_spectrum(self._pending_spectrum)
                elif hasattr(self, 'mountain_canvas') and self.mountain_canvas is not None and self.mountain_canvas.isVisible():
                    self.mountain_canvas.update_spectrum(self._pending_spectrum)
                elif hasattr(self, 'bar_canvas') and self.bar_canvas is not None and self.bar_canvas.isVisible():
                    self.bar_canvas.update_spectrum(self._pending_spectrum)
                elif hasattr(self, 'phosphor_canvas') and self.phosphor_canvas is not None and self.phosphor_canvas.isVisible():
                    self.phosphor_canvas.update_spectrum(self._pending_spectrum)
            self._pending_spectrum = None
    
    def _on_status_change(self, message: str, connected: bool):
        """Update connection status"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {'#0f0' if connected else '#f55'};")
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.test_btn.setEnabled(connected)
    
    def _update_display(self):
        """Periodic display update + sync cached widget states for thread-safe audio access"""
        if self.stroke_mapper:
            alpha, beta = self.stroke_mapper.get_current_position()
            self.position_canvas.update_position(alpha, beta)
            self.alpha_label.setText(f"α: {alpha:.2f}")
            self.beta_label.setText(f"β: {beta:.2f}")

        # Sync widget states to cached values for thread-safe reading by audio thread
        # P0/F0/P1/P3 enable state MUST be synced IMMEDIATELY (every frame) for instant response
        new_p0_enabled = self.pulse_enabled_checkbox.isChecked()
        new_f0_enabled = self.f0_enabled_checkbox.isChecked()
        new_p1_enabled = self.p1_enabled_checkbox.isChecked()
        new_p3_enabled = self.p3_enabled_checkbox.isChecked()
        
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
            for canvas_name in ['mountain_canvas', 'spectrum_canvas', 'bar_canvas', 'phosphor_canvas']:
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
            now = time.time()
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
                    
                    # Check if stable long enough to start aligning
                    if (self._auto_align_is_stable and 
                            (now - self._auto_align_stable_since) >= self._auto_align_required_seconds):
                        current_target = self.target_bpm_spin.value()
                        diff = sensed_bpm - current_target
                        
                        # Only align if difference >= 1 BPM AND cooldown elapsed
                        if abs(diff) >= 1.0 and (now - self._auto_align_last_adjust_time) >= self._auto_align_cooldown:
                            if diff > 0:
                                new_target = min(int(current_target) + 1, int(sensed_bpm))
                            else:
                                new_target = max(int(current_target) - 1, int(np.ceil(sensed_bpm)))
                            
                            if new_target != int(current_target):
                                self.target_bpm_spin.setValue(new_target)
                                self._auto_align_last_adjust_time = now
                                stable_elapsed = now - self._auto_align_stable_since
                                print(f"[Auto-align] Target BPM: {int(current_target)} → {new_target} (sensed: {sensed_bpm:.1f}, stable for {stable_elapsed:.1f}s)")
                else:
                    # Invalid BPM, reset stability
                    self._auto_align_is_stable = False
                    self._auto_align_stable_since = 0.0
            
            # ===== TRAFFIC LIGHT UPDATE =====
            if hasattr(self, 'metric_traffic_light'):
                states = self.audio_engine.get_metric_states()
                if not states:
                    # No metrics enabled
                    self.metric_traffic_light.all_off()
                else:
                    any_adjusting = any(s == 'ADJUSTING' for s in states.values())
                    any_settled = any(s == 'SETTLED' for s in states.values())
                    all_settled = all(s == 'SETTLED' for s in states.values())
                    # Red = actively adjusting, Yellow = some settled, Green = all settled/locked
                    self.metric_traffic_light.set_state(
                        green=all_settled,
                        yellow=any_settled and not all_settled,
                        red=any_adjusting
                    )

    def closeEvent(self, event):
        """Cleanup on close - ensure all threads are stopped before UI is destroyed"""
        shutdown_runtime(self._stop_engines, self.network_engine)

        persist_runtime_ui_to_config(self, self.config)
        
        # Save config before closing
        save_config(self.config)

        # Save slider tuning report
        try:
            self._slider_tracker.save_reports()
        except Exception as e:
            print(f"[Tracker] Failed to save slider tuning report: {e}")

        # Save presets to disk
        self._save_presets_to_disk()

        event.accept()


def main():
    """Main entry point - backup if not launched via run.py"""
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
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
