"""
bREadbeats - Main Application
Qt GUI with beat detection, stroke mapping, and spectrum visualization.
"""

import sys
import numpy as np
import queue
import threading
import time
import json
import os
from pathlib import Path
from dataclasses import asdict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSlider, QComboBox, QPushButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QTabWidget, QFrame,
    QGridLayout, QSizePolicy, QMenuBar, QMenu, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRect
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QBrush, QPen
from typing import Optional

# PyQtGraph for high-performance real-time plotting
import pyqtgraph as pg
pg.setConfigOptions(antialias=False, useOpenGL=False)  # Disable for compatibility

# Keep matplotlib imports for fallback (unused but avoids import errors)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from config import Config, StrokeMode, BeatDetectionType
from audio_engine import AudioEngine, BeatEvent
from network_engine import NetworkEngine, TCodeCommand
from stroke_mapper import StrokeMapper


# Config persistence - use exe folder when packaged, home dir when running from source
def get_config_dir() -> Path:
    """Get config directory - exe folder when packaged, home dir otherwise"""
    if getattr(sys, 'frozen', False):
        # Running as packaged exe - save in same folder as exe
        return Path(sys.executable).parent
    else:
        # Running from source - use home directory
        config_dir = Path.home() / '.breadbeats'
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

def get_config_file() -> Path:
    """Get config file path"""
    return get_config_dir() / 'config.json'

def save_config(config: Config) -> bool:
    """Save config to JSON file"""
    try:
        config_file = get_config_file()
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        print(f"[Config] Saved to {config_file}")
        return True
    except Exception as e:
        print(f"[Config] Failed to save: {e}")
        return False

def load_config() -> Config:
    """Load config from JSON file, returns default if not found"""
    try:
        config_file = get_config_file()
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct Config from dict (handles nested dataclasses)
            config = Config()
            
            # Apply loaded values
            if 'beat' in data:
                for key, value in data['beat'].items():
                    if hasattr(config.beat, key):
                        setattr(config.beat, key, value)
            
            if 'stroke' in data:
                for key, value in data['stroke'].items():
                    if hasattr(config.stroke, key):
                        setattr(config.stroke, key, value)
                # Ensure mode is always a StrokeMode enum
                if hasattr(config.stroke, 'mode') and not isinstance(config.stroke.mode, StrokeMode):
                    try:
                        config.stroke.mode = StrokeMode(config.stroke.mode)
                    except Exception as e:
                        print(f"[Config] Warning: Could not convert stroke.mode to StrokeMode enum: {e}")
            
            if 'jitter' in data:
                for key, value in data['jitter'].items():
                    if hasattr(config.jitter, key):
                        setattr(config.jitter, key, value)
            
            if 'creep' in data:
                for key, value in data['creep'].items():
                    if hasattr(config.creep, key):
                        setattr(config.creep, key, value)
            
            if 'connection' in data:
                for key, value in data['connection'].items():
                    if hasattr(config.connection, key):
                        setattr(config.connection, key, value)
            
            if 'audio' in data:
                for key, value in data['audio'].items():
                    if hasattr(config.audio, key):
                        setattr(config.audio, key, value)
            
            if 'pulse_freq' in data:
                for key, value in data['pulse_freq'].items():
                    if hasattr(config.pulse_freq, key):
                        setattr(config.pulse_freq, key, value)
            
            # Top-level values
            if 'alpha_weight' in data:
                config.alpha_weight = data['alpha_weight']
            if 'beta_weight' in data:
                config.beta_weight = data['beta_weight']
            if 'volume' in data:
                config.volume = data['volume']
            
            print(f"[Config] Loaded from {config_file}")
            return config
        else:
            print(f"[Config] No saved config found, using defaults")
            return Config()
    except Exception as e:
        print(f"[Config] Failed to load: {e}, using defaults")
        return Config()


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
        self.setXRange(0, self.num_bins)
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
        self._updating = False  # Prevent recursion when setting bands
        
    def _hz_to_bin(self, hz: float) -> float:
        """Convert Hz to bin index (0 to num_bins)"""
        nyquist = self.sample_rate / 2
        return (hz / nyquist) * self.num_bins
    
    def _bin_to_hz(self, bin_idx: float) -> float:
        """Convert bin index to Hz"""
        nyquist = self.sample_rate / 2
        return (bin_idx / self.num_bins) * nyquist
        
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
        low_bin = low_norm * self.num_bins
        high_bin = high_norm * self.num_bins
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
        """Compatibility stub - waterfall doesn't use these lines"""
        pass
    
    def set_indicators_visible(self, visible: bool):
        """Show or hide all frequency band indicators and labels"""
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
        # Resample to num_bins using interpolation
        if len(spectrum) != self.num_bins:
            x_old = np.linspace(0, 1, len(spectrum))
            x_new = np.linspace(0, 1, self.num_bins)
            spectrum = np.interp(x_new, x_old, spectrum)
        
        # Apply log scaling for better dynamic range visualization
        spectrum = np.log10(spectrum + 1e-6)
        # Normalize: typical range is -6 to 0 for normalized input, map to 0-1
        spectrum = np.clip((spectrum + 4) / 4, 0, 1)
        
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


class MountainRangeCanvas(pg.PlotWidget):
    """Mountain range spectrum visualizer - glowing blue filled peaks"""
    
    def __init__(self, parent=None, width=8, height=3):
        super().__init__(parent)
        
        # Dark theme
        self.setBackground('#0a0a12')  # Very dark blue-black
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.showGrid(x=False, y=False, alpha=0)
        self.hideAxis('left')
        self.hideAxis('bottom')
        
        # Spectrum dimensions
        self.num_bins = 256
        
        # Frequency values for x-axis (will be updated with sample rate)
        self.freq_values = np.linspace(0, self.num_bins, self.num_bins)
        
        # Set view range
        self.setXRange(0, self.num_bins)
        self.setYRange(0, 1.2)
        
        # Main spectrum curve (mountain peaks) - cyan fill with bright outline
        self.spectrum_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QColor(100, 200, 255, 255), width=2),  # Bright cyan outline
            fillLevel=0,
            brush=pg.mkBrush(QColor(0, 120, 200, 120))  # Semi-transparent cyan fill
        )
        self.addItem(self.spectrum_curve)
        
        # Glow effect - slightly larger, more transparent version behind
        self.glow_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QColor(0, 150, 255, 80), width=6),
            fillLevel=0,
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
        self._updating = False
        
        # Smoothing buffer for smoother animation
        self._smooth_spectrum = np.zeros(self.num_bins)
        self._smoothing = 0.3  # 0 = no smoothing, 1 = max smoothing
        
    def _hz_to_bin(self, hz: float) -> float:
        """Convert Hz to bin index"""
        nyquist = self.sample_rate / 2
        return (hz / nyquist) * self.num_bins
    
    def _bin_to_hz(self, bin_idx: float) -> float:
        """Convert bin index to Hz"""
        nyquist = self.sample_rate / 2
        return (bin_idx / self.num_bins) * nyquist
        
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
        center_bin = (region[0] + region[1]) / 2  # type: ignore
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
        center_bin = (region[0] + region[1]) / 2  # type: ignore
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
        center_bin = (region[0] + region[1]) / 2  # type: ignore
        self.carrier_label.setPos(center_bin, 0.73)
    
    def set_sample_rate(self, sr: int):
        self.sample_rate = sr
    
    def set_frequency_band(self, low_norm: float, high_norm: float):
        self._updating = True
        low_bin = low_norm * self.num_bins
        high_bin = high_norm * self.num_bins
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
        """Not used in mountain view"""
        pass
    
    def set_indicators_visible(self, visible: bool):
        """Show or hide all frequency band indicators and labels"""
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
            
        # Resample to num_bins
        if len(spectrum) != self.num_bins:
            x_old = np.linspace(0, 1, len(spectrum))
            x_new = np.linspace(0, 1, self.num_bins)
            spectrum = np.interp(x_new, x_old, spectrum)
        
        # Apply log scaling
        spectrum = np.log10(spectrum + 1e-6)
        spectrum = np.clip((spectrum + 4) / 4, 0, 1)
        
        # Sharpen peaks - enhance local maxima to make peaks pointy
        sharpened = spectrum.copy()
        for i in range(1, len(spectrum) - 1):
            # Calculate how much this point is above its neighbors
            left = spectrum[i-1]
            right = spectrum[i+1]
            center = spectrum[i]
            avg_neighbors = (left + right) / 2
            # If this is a local peak, boost it
            if center > left and center > right:
                peak_boost = (center - avg_neighbors) * 1.5  # Amplify peaks
                sharpened[i] = min(1.2, center + peak_boost)
            # If this is a valley, deepen it slightly
            elif center < left and center < right:
                sharpened[i] = max(0, center * 0.9)
        spectrum = sharpened
        
        # Smooth the spectrum for less jittery animation
        self._smooth_spectrum = self._smoothing * self._smooth_spectrum + (1 - self._smoothing) * spectrum
        
        # Update curves
        x = np.arange(self.num_bins)
        self.spectrum_curve.setData(x, self._smooth_spectrum)
        self.glow_curve.setData(x, self._smooth_spectrum * 1.02)  # Slightly larger for glow
        
        # Find and mark peaks (local maxima above threshold)
        if peak_energy and peak_energy > 0.3:
            peaks = []
            peak_vals = []
            for i in range(2, self.num_bins - 2):
                if (self._smooth_spectrum[i] > self._smooth_spectrum[i-1] and 
                    self._smooth_spectrum[i] > self._smooth_spectrum[i+1] and
                    self._smooth_spectrum[i] > 0.5):
                    peaks.append(i)
                    peak_vals.append(self._smooth_spectrum[i])
            if peaks:
                self.peak_scatter.setData(peaks, peak_vals)
            else:
                self.peak_scatter.setData([], [])
        else:
            self.peak_scatter.setData([], [])


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
        
        # Set view range
        self.setXRange(-0.5, self.num_bars - 0.5)
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
        self._updating = False
        
        # Smoothing buffer
        self._smooth_heights = np.zeros(self.num_bars)
        self._smoothing = 0.4
        
    def _hz_to_bin(self, hz: float) -> float:
        """Convert Hz to bar index"""
        nyquist = self.sample_rate / 2
        return (hz / nyquist) * self.num_bars
    
    def _bin_to_hz(self, bin_idx: float) -> float:
        """Convert bar index to Hz"""
        nyquist = self.sample_rate / 2
        return (bin_idx / self.num_bars) * nyquist
        
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
        low_bin = low_norm * self.num_bars
        high_bin = high_norm * self.num_bars
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
        """Not used in bar view"""
        pass
    
    def set_indicators_visible(self, visible: bool):
        """Show or hide all frequency band indicators and labels"""
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
            
        # Resample to num_bars
        if len(spectrum) != self.num_bars:
            x_old = np.linspace(0, 1, len(spectrum))
            x_new = np.linspace(0, 1, self.num_bars)
            spectrum = np.interp(x_new, x_old, spectrum)
        
        # Apply log scaling
        spectrum = np.log10(spectrum + 1e-6)
        spectrum = np.clip((spectrum + 4) / 4, 0, 1)
        
        # Smooth for less jittery animation
        self._smooth_heights = self._smoothing * self._smooth_heights + (1 - self._smoothing) * spectrum
        
        # Update bar heights
        self.bar_item.setOpts(height=self._smooth_heights)


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
        
        # Set view range
        self.setXRange(0, self.num_bins)
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
        self._updating = False
        
    def _hz_to_bin(self, hz: float) -> float:
        nyquist = self.sample_rate / 2
        return (hz / nyquist) * self.num_bins
    
    def _bin_to_hz(self, bin_idx: float) -> float:
        nyquist = self.sample_rate / 2
        return (bin_idx / self.num_bins) * nyquist
        
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
        low_bin = low_norm * self.num_bins
        high_bin = high_norm * self.num_bins
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
        pass
    
    def set_indicators_visible(self, visible: bool):
        """Show or hide all frequency band indicators and labels"""
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
            
        # Resample to num_bins
        if len(spectrum) != self.num_bins:
            x_old = np.linspace(0, 1, len(spectrum))
            x_new = np.linspace(0, 1, self.num_bins)
            spectrum = np.interp(x_new, x_old, spectrum)
        
        # Apply log scaling and normalize to 0-1
        spectrum = np.log10(spectrum + 1e-6)
        spectrum = np.clip((spectrum + 4) / 4, 0, 1)
        
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
        
        # Dark theme
        self.setBackground('#232323')
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
        self.addItem(pg.PlotCurveItem(circle_x, circle_y, pen=pg.mkPen('#666666', width=1)))
        
        # Draw crosshairs
        self.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#555555', width=0.5)))
        self.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#555555', width=0.5)))
        
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
        
        self.trail_x.append(x_rot)
        self.trail_y.append(y_rot)
        if len(self.trail_x) > self.max_trail:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
        
        # Update trail curve
        if len(self.trail_x) > 1:
            self.trail_curve.setData(self.trail_x, self.trail_y)
        
        # Update position marker
        self.position_scatter.setData([x_rot], [y_rot])


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
                 high_default: float, decimals: int = 0, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
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
        ratio = (value - self.min_val) / (self.max_val - self.min_val)
        return int(self._handle_width/2 + ratio * (self.width() - self._handle_width))
    
    def _pos_to_val(self, pos: float) -> float:
        """Convert pixel position to value"""
        ratio = (pos - self._handle_width/2) / (self.width() - self._handle_width)
        ratio = max(0, min(1, ratio))
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
                 low_default: float, high_default: float, decimals: int = 0, parent=None):
        super().__init__(parent)
        
        self.decimals = decimals
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(name)
        self.label.setFixedWidth(120)
        self.label.setStyleSheet("color: #aaa;")
        
        self.slider = RangeSlider(min_val, max_val, low_default, high_default, decimals)
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
        self.valueChanged.emit(real_value)
        
    def value(self) -> float:
        return self.slider.value() / self.multiplier
    
    def setValue(self, value: float):
        self.slider.setValue(int(value * self.multiplier))


class BREadbeatsWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("bREadbeats")
        self.setMinimumSize(1000, 850)
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
        self.signals = SignalBridge()
        
        # Command queue
        self.cmd_queue = queue.Queue()
        
        # Initialize engines to None early (before UI setup needs to check them)
        self.audio_engine = None
        self.network_engine = None
        self.stroke_mapper = None
        
        # Setup UI
        self._setup_ui()
        
        # Load config values into UI sliders
        self._apply_config_to_ui()
        
        # Load presets from disk
        self._load_presets_from_disk()
        
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
        self._cached_p0_enabled: bool = True
        self._cached_f0_enabled: bool = True
        self._cached_pulse_mode: int = 0  # 0=Hz, 1=Speed
        self._cached_pulse_invert: bool = False
        self._cached_f0_mode: int = 0
        self._cached_f0_invert: bool = False
        self._cached_pulse_display: str = "Pulse: --"
        self._cached_carrier_display: str = "Carrier: --"
        # Cached TCode Sent Freq slider values (Hz) for thread-safe P0/F0 computation
        self._cached_tcode_freq_min: float = 30.0
        self._cached_tcode_freq_max: float = 105.0
        self._cached_f0_tcode_min: float = 500.0
        self._cached_f0_tcode_max: float = 1000.0
        # Track previous enabled state for send-zero-once logic
        self._prev_p0_enabled: bool = True
        self._prev_f0_enabled: bool = True
        self._last_freq_display_time: float = 0.0  # Throttle freq display updates to 100ms
        self._last_dot_alpha: float = 0.0
        self._last_dot_beta: float = 0.0
        self._last_dot_time: float = 0.0
        
        # Volume ramping state for play/stop
        self._volume_ramp_active: bool = False
        self._volume_ramp_start_time: float = 0.0
        self._volume_ramp_from: float = 0.0
        self._volume_ramp_to: float = 1.0
        self._volume_ramp_duration: float = 0.8  # 800ms
        
        # Auto beat detection adjustment state
        self._auto_adjust_enabled: dict = {
            'audio_amp': False,
            'peak_floor': False,
            'peak_decay': False,
            'rise_sens': False,
            'sensitivity': False,
            'flux_mult': False
        }
        self._last_beat_time_for_auto: float = 0.0  # Track last beat for auto-adjust
        self._auto_adjust_timer: Optional[QTimer] = None
        self._auto_adjust_interval_ms: int = 100  # Timer tick rate (just checks if cooldown elapsed)
        self._auto_threshold_sec: float = 0.43  # Beat interval threshold in seconds (428ms = 140 BPM)
        self._auto_upper_threshold_bpm: float = 160.0  # Upper BPM threshold - reverse adjustment above this
        self._auto_cooldown_sec: float = 0.10  # Cooldown between parameter adjustments (seconds)
        self._auto_param_index: int = 0  # Current position in the parameter cycle
        self._auto_last_adjust_time: float = 0.0  # When we last adjusted a parameter
        self._auto_param_order: list = ['sensitivity', 'audio_amp', 'flux_mult', 'rise_sens', 'peak_floor', 'peak_decay']  # Ordered by impact
        # Per-parameter lock states: HUNTING, REVERSING, LOCKED
        self._auto_param_state: dict = {
            'audio_amp': 'HUNTING',
            'peak_floor': 'HUNTING',
            'peak_decay': 'HUNTING',
            'rise_sens': 'HUNTING',
            'sensitivity': 'HUNTING',
            'flux_mult': 'HUNTING',
        }
        self._auto_flux_lock_count: int = 0  # Permanent lock after 2nd downbeat detection
        self._auto_oscillation_phase: float = 0.0  # Phase for oscillation sine wave
        self._auto_last_downbeat_conf: float = 0.0  # Last seen downbeat confidence
        self._auto_downbeat_threshold: float = 0.3  # Low threshold - just needs some downbeats detected
        self._auto_no_beat_since: float = 0.0  # Timestamp when beats were last lost (for 1500ms timer)
        self._auto_param_lock_time: dict = {  # When each param was locked (timestamp)
            'audio_amp': 0.0, 'peak_floor': 0.0, 'peak_decay': 0.0,
            'rise_sens': 0.0, 'sensitivity': 0.0, 'flux_mult': 0.0,
        }
        # Consecutive-beat lock: stop ALL hunting after N seconds of continuous beats
        self._auto_consec_lock_sec: float = 5.0  # Seconds of continuous beats before full lock
        self._auto_consec_beat_start: float = 0.0  # When consecutive beats started
        self._auto_consec_locked: bool = False  # True = all hunting stopped
        # Impact-based step sizes: (hunt_step, lock_time_ms, max_limit_when_hunting)
        # Oscillation amplitude is always 3/4 of step_size (computed in _adjust_single_param)
        self._auto_param_config: dict = {
            'sensitivity': (0.008, 5000.0, 1.0),     # 75% impact - small steps
            'audio_amp': (0.04, 5000.0, 1.0),        # 75% impact - LIMIT TO 1.0 when hunting
            'flux_mult': (0.015, 5000.0, 5.0),       # 40% impact
            'rise_sens': (0.008, 5000.0, 1.0),       # 30% impact
            'peak_floor': (0.004, 5000.0, 0.0),      # 15% impact
            'peak_decay': (0.002, 5000.0, 0.5),      # 10% impact
        }
        
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
        
        # Middle: Visualizers
        viz_layout = QHBoxLayout()
        viz_layout.addWidget(self._create_spectrum_panel(), stretch=3)
        viz_layout.addWidget(self._create_position_panel(), stretch=1)
        main_layout.addLayout(viz_layout)
        
        # Bottom: Tabs with sliders
        main_layout.addWidget(self._create_settings_tabs())
        
        # Bottom row: Presets + Visualizer controls
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self._create_presets_panel())
        
        # Visualizer controls in a groupbox
        visualizer_group = QGroupBox("Spectrum")
        visualizer_layout = QHBoxLayout(visualizer_group)
        visualizer_layout.setContentsMargins(8, 4, 8, 4)
        
        visualizer_layout.addWidget(QLabel("Type:"))
        self.visualizer_type_combo = QComboBox()
        self.visualizer_type_combo.addItems(["Waterfall", "Mountain Range", "Bar Graph", "Phosphor"])
        self.visualizer_type_combo.currentIndexChanged.connect(self._on_visualizer_type_change)
        visualizer_layout.addWidget(self.visualizer_type_combo)
        
        # Hide range indicators checkbox
        self.hide_indicators_checkbox = QCheckBox("Hide Range Indicators")
        self.hide_indicators_checkbox.setChecked(False)
        self.hide_indicators_checkbox.stateChanged.connect(self._on_hide_indicators_toggle)
        visualizer_layout.addWidget(self.hide_indicators_checkbox)
        
        bottom_layout.addWidget(visualizer_group)
        
        bottom_layout.addStretch()  # Gap before Whip the Llama button
        
        # projectM launcher button (Winamp throwback)
        self.projectm_btn = QPushButton("Whip the Llama")
        self.projectm_btn.setToolTip("Launch projectM music visualizer (if installed)")
        self.projectm_btn.clicked.connect(self._on_launch_projectm)
        self.projectm_btn.setMaximumWidth(120)
        bottom_layout.addWidget(self.projectm_btn)
        
        main_layout.addLayout(bottom_layout)
    
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
        
        # Help menu item
        help_action = main_menu.addAction("Help")
        assert help_action is not None
        help_action.triggered.connect(self._on_help)
        
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
        
        # Beat detection with auto button
        beat_box = QGroupBox()
        beat_box.setStyleSheet("QGroupBox { border: 1px solid #555; padding: 4px; margin-top: 2px; }")
        bb_layout = QVBoxLayout(beat_box)
        bb_layout.setSpacing(2)
        bb_layout.addWidget(QLabel("[Beat Detection] Raise sensitivity/amplification\nuntil you see blinking, or toggle auto:"))
        auto_btn_row = QHBoxLayout()
        auto_all_btn = QPushButton("Enable All Auto")
        auto_all_btn.setToolTip("Enable auto-adjustment for all beat detection sliders")
        auto_all_btn.clicked.connect(lambda: self._enable_all_auto_beat_detection(True))
        auto_btn_row.addWidget(auto_all_btn)
        auto_off_btn = QPushButton("Disable All Auto")
        auto_off_btn.clicked.connect(lambda: self._enable_all_auto_beat_detection(False))
        auto_btn_row.addWidget(auto_off_btn)
        bb_layout.addLayout(auto_btn_row)
        g1_layout.addWidget(beat_box)
        
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
        flb_layout.addWidget(QLabel("[Beat Detection] Check peak floor:"))
        floor_reset_btn = QPushButton("Reset to 0")
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
    
    def _enable_all_auto_beat_detection(self, enable: bool):
        """Enable or disable all auto-adjust checkboxes for beat detection"""
        if hasattr(self, 'audio_gain_auto_cb'):
            self.audio_gain_auto_cb.setChecked(enable)
        if hasattr(self, 'peak_floor_auto_cb'):
            self.peak_floor_auto_cb.setChecked(enable)
        if hasattr(self, 'peak_decay_auto_cb'):
            self.peak_decay_auto_cb.setChecked(enable)
        if hasattr(self, 'rise_sens_auto_cb'):
            self.rise_sens_auto_cb.setChecked(enable)
        if hasattr(self, 'sensitivity_auto_cb'):
            self.sensitivity_auto_cb.setChecked(enable)
        if hasattr(self, 'flux_mult_auto_cb'):
            self.flux_mult_auto_cb.setChecked(enable)
        # Update global auto-range checkbox state
        if hasattr(self, 'global_auto_range_cb'):
            # Block signals to prevent recursion
            self.global_auto_range_cb.blockSignals(True)
            self.global_auto_range_cb.setChecked(enable)
            self.global_auto_range_cb.blockSignals(False)
    
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
    
    def _on_about(self):
        """Show About dialog"""
        about_text = """bREadbeats v1.0
Live Audio to Restim

Inspired by:
    digitalparkinglot's creations
    edger477 (ideas from funscriptgenerator)
    diglet48 (wouldn't be here without restim!)

Bug reports/share your presets:
bREadfan_69@hotmail.com"""
        QMessageBox.information(self, "About bREadbeats", about_text)
    
    def _apply_config_to_ui(self):
        """Apply loaded config values to UI sliders"""
        try:
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
            
            # Set spectrum canvas sample rate and update all 3 frequency bands
            self.spectrum_canvas.set_sample_rate(self.config.audio.sample_rate)
            if hasattr(self, 'mountain_canvas'):
                self.mountain_canvas.set_sample_rate(self.config.audio.sample_rate)
            self._on_freq_band_change()  # Update beat detection band (red)
            
            # Tempo tracking settings
            self.tempo_tracking_checkbox.setChecked(self.config.beat.tempo_tracking_enabled)
            self._on_tempo_tracking_toggle(2 if self.config.beat.tempo_tracking_enabled else 0)
            beats_to_index = {4: 0, 3: 1, 6: 2}
            self.time_sig_combo.setCurrentIndex(beats_to_index.get(self.config.beat.beats_per_measure, 0))
            self.stability_threshold_slider.setValue(self.config.beat.stability_threshold)
            self.tempo_timeout_slider.setValue(self.config.beat.tempo_timeout_ms)
            self.phase_snap_slider.setValue(self.config.beat.phase_snap_weight)
            
            # Stroke settings tab
            self.mode_combo.setCurrentIndex(self.config.stroke.mode - 1)
            self._on_mode_change(self.config.stroke.mode - 1)  # Apply axis weight limits for this mode
            self.stroke_range_slider.setLow(self.config.stroke.stroke_min)
            self.stroke_range_slider.setHigh(self.config.stroke.stroke_max)
            self.min_interval_slider.setValue(self.config.stroke.min_interval_ms)
            self.fullness_slider.setValue(self.config.stroke.stroke_fullness)
            self.min_depth_slider.setValue(self.config.stroke.minimum_depth)
            self.freq_depth_slider.setValue(self.config.stroke.freq_depth_factor)
            self.depth_freq_range_slider.setLow(int(self.config.stroke.depth_freq_low))
            self.depth_freq_range_slider.setHigh(int(self.config.stroke.depth_freq_high))
            self._on_depth_band_change()  # Update stroke depth band (green)
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
            
            # Connection settings
            self.host_edit.setText(self.config.connection.host)
            self.port_spin.setValue(self.config.connection.port)
            
            # Other tab (pulse freq settings)
            self.pulse_freq_range_slider.setLow(self.config.pulse_freq.monitor_freq_min)
            self.pulse_freq_range_slider.setHigh(self.config.pulse_freq.monitor_freq_max)
            self._on_p0_band_change()  # Update P0 TCode band (blue)
            self.tcode_freq_range_slider.setLow(self.config.pulse_freq.tcode_freq_min)
            self.tcode_freq_range_slider.setHigh(self.config.pulse_freq.tcode_freq_max)
            self.freq_weight_slider.setValue(self.config.pulse_freq.freq_weight)
            
            # Carrier freq (F0) settings
            self.f0_freq_range_slider.setLow(self.config.carrier_freq.monitor_freq_min)
            self.f0_freq_range_slider.setHigh(self.config.carrier_freq.monitor_freq_max)
            self._on_f0_band_change()  # Update F0 TCode band (cyan)
            self.f0_tcode_range_slider.setLow(self.config.carrier_freq.tcode_freq_min)
            self.f0_tcode_range_slider.setHigh(self.config.carrier_freq.tcode_freq_max)
            self.f0_weight_slider.setValue(self.config.carrier_freq.freq_weight)
            
            # Volume
            self.volume_slider.setValue(self.config.volume)
            
            print("[UI] Loaded all settings from config")
        except AttributeError as e:
            print(f"[UI] Warning: Could not apply all config values: {e}")
        
    def _create_connection_panel(self) -> QGroupBox:
        """Connection settings panel - simplified, host/port in Options menu"""
        group = QGroupBox("TCP Connection")
        layout = QHBoxLayout(group)  # Horizontal for compact layout
        
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
    
    def _create_control_panel(self) -> QGroupBox:
        """Main control buttons - audio device selection moved to Options menu"""
        group = QGroupBox("Controls")
        layout = QVBoxLayout(group)
        
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
        
        # Volume slider (0.0 - 1.0) - uses compact label for control panel
        self.volume_slider = SliderWithLabel("Vol", 0.0, 1.0, 1.0, decimals=2)
        self.volume_slider.label.setFixedWidth(30)  # Compact label for controls box
        self.volume_slider.setFixedWidth(180)
        self.volume_slider.setContentsMargins(0, 0, 0, 0)
        btn_layout.addWidget(self.volume_slider, 0, 2, 1, 2)

        # Frequency displays - stacked vertically
        freq_display_layout = QVBoxLayout()
        freq_display_layout.setSpacing(0)
        
        # Carrier Freq display (shows sent F0 TCode value)
        self.carrier_freq_label = QLabel("Carrier: --")
        self.carrier_freq_label.setStyleSheet("color: #0af; font-size: 11px; font-weight: bold;")
        self.carrier_freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_display_layout.addWidget(self.carrier_freq_label)
        
        # Pulse Freq display (shows sent P0 TCode value)
        self.pulse_freq_label = QLabel("Pulse: --")
        self.pulse_freq_label.setStyleSheet("color: #f80; font-size: 11px; font-weight: bold;")
        self.pulse_freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_display_layout.addWidget(self.pulse_freq_label)
        
        freq_display_widget = QWidget()
        freq_display_widget.setLayout(freq_display_layout)
        freq_display_widget.setFixedWidth(80)
        btn_layout.addWidget(freq_display_widget, 0, 4)

        # Beat indicator (lights up on any beat)
        self.beat_indicator = QLabel("●")
        self.beat_indicator.setStyleSheet("color: #333; font-size: 24px;")
        self.beat_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.beat_indicator.setFixedWidth(30)
        btn_layout.addWidget(self.beat_indicator, 0, 5)
        
        # Downbeat indicator (lights up on downbeat/beat 1)
        self.downbeat_indicator = QLabel("●")
        self.downbeat_indicator.setStyleSheet("color: #333; font-size: 24px;")
        self.downbeat_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.downbeat_indicator.setFixedWidth(30)
        btn_layout.addWidget(self.downbeat_indicator, 0, 6)
        
        # Beat indicator timer for visual feedback duration
        self.beat_timer = QTimer()
        self.beat_timer.setSingleShot(True)
        self.beat_timer.timeout.connect(self._turn_off_beat_indicator)
        self.beat_indicator_min_duration = 100  # ms
        
        # Downbeat indicator timer
        self.downbeat_timer = QTimer()
        self.downbeat_timer.setSingleShot(True)
        self.downbeat_timer.timeout.connect(self._turn_off_downbeat_indicator)

        # BPM display (right next to indicators)
        self.bpm_label = QLabel("BPM: --")
        self.bpm_label.setStyleSheet("color: #0a0; font-size: 14px; font-weight: bold;")
        self.bpm_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.bpm_label.setFixedWidth(120)  # Fixed width to prevent layout jumping when text changes
        btn_layout.addWidget(self.bpm_label, 0, 7)

        btn_layout.setColumnStretch(8, 1)  # Allow last column to stretch
        layout.addLayout(btn_layout)

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
    
    def _create_spectrum_panel(self) -> QGroupBox:
        """Spectrum visualizer panel"""
        group = QGroupBox("Frequency Selection")
        layout = QVBoxLayout(group)
        
        # Create all visualizers (only one visible at a time)
        self.spectrum_canvas = SpectrumCanvas(self, width=8, height=3)
        self.mountain_canvas = MountainRangeCanvas(self, width=8, height=3)
        self.bar_canvas = BarGraphCanvas(self, width=8, height=3)
        self.phosphor_canvas = PhosphorCanvas(self, width=8, height=3)
        self.mountain_canvas.setVisible(False)  # Start with waterfall
        self.bar_canvas.setVisible(False)
        self.phosphor_canvas.setVisible(False)
        
        layout.addWidget(self.spectrum_canvas)
        layout.addWidget(self.mountain_canvas)
        layout.addWidget(self.bar_canvas)
        layout.addWidget(self.phosphor_canvas)
        
        return group
    
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
    
    def _on_hide_indicators_toggle(self, state: int):
        """Toggle visibility of frequency range indicators on all visualizers"""
        hide = state == 2  # Qt.Checked = 2
        # Apply to all visualizer canvases
        for canvas in [self.spectrum_canvas, self.mountain_canvas, self.bar_canvas, self.phosphor_canvas]:
            if hasattr(canvas, 'set_indicators_visible'):
                canvas.set_indicators_visible(not hide)
    
    def _create_position_panel(self) -> QGroupBox:
        """Alpha/Beta position display"""
        group = QGroupBox("Position (α/β)")
        layout = QVBoxLayout(group)

        # Position canvas (no rotation - fixed at 0)
        self.position_canvas = PositionCanvas(self, size=2, get_rotation=lambda: 0)
        layout.addWidget(self.position_canvas)

        # Position labels (hidden but still tracked internally)
        self.alpha_label = QLabel("α: 0.00")
        self.alpha_label.setVisible(False)
        self.beta_label = QLabel("β: 0.00")
        self.beta_label.setVisible(False)

        return group
    
    def _create_settings_tabs(self) -> QTabWidget:
        """Settings tabs with all the sliders"""
        tabs = QTabWidget()
        tabs.addTab(self._create_beat_detection_tab(), "Beat Detection")
        tabs.addTab(self._create_stroke_settings_tab(), "Stroke Settings")
        tabs.addTab(self._create_jitter_creep_tab(), "Effects / Axis")
        tabs.addTab(self._create_tempo_tracking_tab(), "Tempo Tracking")
        tabs.addTab(self._create_pulse_freq_tab(), "Pulse Freq")
        tabs.addTab(self._create_carrier_freq_tab(), "Carrier Freq")
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

            # Axis Weights Tab
            'alpha_weight': self.alpha_weight_slider.value(),
            'beta_weight': self.beta_weight_slider.value(),

            # Pulse Freq Tab
            'pulse_freq_low': self.pulse_freq_range_slider.low(),
            'pulse_freq_high': self.pulse_freq_range_slider.high(),
            'tcode_freq_min': self.tcode_freq_range_slider.low(),
            'tcode_freq_max': self.tcode_freq_range_slider.high(),
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
        
        # Axis Weights Tab
        self.alpha_weight_slider.setValue(preset_data['alpha_weight'])
        self.beta_weight_slider.setValue(preset_data['beta_weight'])
        
        # Pulse Freq Tab
        self.pulse_freq_range_slider.setLow(preset_data['pulse_freq_low'])
        self.pulse_freq_range_slider.setHigh(preset_data['pulse_freq_high'])
        self.tcode_freq_range_slider.setLow(preset_data['tcode_freq_min'])
        self.tcode_freq_range_slider.setHigh(preset_data['tcode_freq_max'])
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

    def _create_pulse_freq_tab(self) -> QWidget:
        """Pulse Frequency (P0 TCode) controls"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        freq_group = QGroupBox("Pulse Frequency Controls - blue overlay on spectrum")
        freq_layout = QVBoxLayout(freq_group)

        # Monitor frequency range (single range slider with two handles)
        self.pulse_freq_range_slider = RangeSliderWithLabel("Monitor Freq (Hz)", 30, 22050, 30, 4000, 0)
        self.pulse_freq_range_slider.rangeChanged.connect(self._on_p0_band_change)
        freq_layout.addWidget(self.pulse_freq_range_slider)

        # TCode output range slider (in Hz, same scale as Pulse display: Hz = TCode/67)
        # TCode range 0-9999 maps to ~0-150Hz via /67
        self.tcode_freq_range_slider = RangeSliderWithLabel("Sent Freq (Hz)", 0, 150, 30, 105, 0)
        freq_layout.addWidget(self.tcode_freq_range_slider)

        # Frequency weight slider - 0=no freq influence, 1=full tracking, 2=exaggerated
        self.freq_weight_slider = SliderWithLabel("Frequency Weight", 0.0, 2.0, 1.0, 2)
        freq_layout.addWidget(self.freq_weight_slider)

        # Mode toggle (Hz vs Speed) and Invert checkbox for Pulse
        pulse_mode_layout = QHBoxLayout()
        pulse_mode_layout.addWidget(QLabel("Mode:"))
        self.pulse_mode_combo = QComboBox()
        self.pulse_mode_combo.addItems(["Hz (dominant freq)", "Speed (dot movement)"])
        self.pulse_mode_combo.setCurrentIndex(0)
        pulse_mode_layout.addWidget(self.pulse_mode_combo)
        self.pulse_invert_checkbox = QCheckBox("Invert")
        self.pulse_invert_checkbox.setChecked(False)
        pulse_mode_layout.addWidget(self.pulse_invert_checkbox)
        self.pulse_enabled_checkbox = QCheckBox("Enable P0")
        self.pulse_enabled_checkbox.setChecked(True)
        pulse_mode_layout.addWidget(self.pulse_enabled_checkbox)
        pulse_mode_layout.addStretch()
        freq_layout.addLayout(pulse_mode_layout)

        layout.addWidget(freq_group)
        layout.addStretch()
        return widget

    def _create_carrier_freq_tab(self) -> QWidget:
        """Carrier Frequency (F0 TCode) controls - frequency axis"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # F0 Frequency Controls group (same structure as Pulse)
        f0_group = QGroupBox("Frequency Controls (F0)")
        f0_layout = QVBoxLayout(f0_group)

        # Monitor frequency range for F0
        self.f0_freq_range_slider = RangeSliderWithLabel("Monitor Freq (Hz)", 30, 22050, 30, 4000, 0)
        self.f0_freq_range_slider.rangeChanged.connect(self._on_f0_band_change)
        f0_layout.addWidget(self.f0_freq_range_slider)

        # TCode output range slider for F0 (display 500-1500 maps to TCode 0-9999)
        self.f0_tcode_range_slider = RangeSliderWithLabel("Sent Freq", 500, 1500, 500, 1000, 0)
        f0_layout.addWidget(self.f0_tcode_range_slider)

        # Frequency weight slider for F0
        self.f0_weight_slider = SliderWithLabel("Frequency Weight", 0.0, 2.0, 1.0, 2)
        f0_layout.addWidget(self.f0_weight_slider)

        # Mode toggle (Hz vs Speed) and Invert checkbox for F0
        f0_mode_layout = QHBoxLayout()
        f0_mode_layout.addWidget(QLabel("Mode:"))
        self.f0_mode_combo = QComboBox()
        self.f0_mode_combo.addItems(["Hz (dominant freq)", "Speed (dot movement)"])
        self.f0_mode_combo.setCurrentIndex(0)
        f0_mode_layout.addWidget(self.f0_mode_combo)
        self.f0_invert_checkbox = QCheckBox("Invert")
        self.f0_invert_checkbox.setChecked(False)
        f0_mode_layout.addWidget(self.f0_invert_checkbox)
        self.f0_enabled_checkbox = QCheckBox("Enable F0")
        self.f0_enabled_checkbox.setChecked(True)
        f0_mode_layout.addWidget(self.f0_enabled_checkbox)
        f0_mode_layout.addStretch()
        f0_layout.addLayout(f0_mode_layout)

        layout.addWidget(f0_group)
        layout.addStretch()
        return widget
    
    def _on_butterworth_toggle(self, state: int):
        """Toggle Butterworth filter (requires restart)"""
        enabled = state == 2
        self.config.audio.use_butterworth = enabled
        print(f"[Config] Butterworth filter {'enabled' if enabled else 'disabled'} (restart required)")
    
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
    
    def _on_auto_toggle(self, param: str, state: int):
        """Toggle auto-adjustment for a beat detection parameter"""
        enabled = state == 2
        self._auto_adjust_enabled[param] = enabled
        # Reset this param's lock state when re-enabled so it starts hunting fresh
        if enabled:
            self._auto_param_state[param] = 'HUNTING'
            if param == 'flux_mult':
                self._auto_flux_lock_count = 0
        print(f"[Auto] {param} auto-adjust {'enabled' if enabled else 'disabled'}")
        
        # Start/stop auto-adjust timer based on whether any auto is enabled
        any_enabled = any(self._auto_adjust_enabled.values())
        if any_enabled and self._auto_adjust_timer is None:
            self._auto_adjust_timer = QTimer()
            self._auto_adjust_timer.timeout.connect(self._auto_adjust_beat_detection)
            self._auto_adjust_timer.start(self._auto_adjust_interval_ms)
            print("[Auto] Started auto-adjust timer")
        elif not any_enabled and self._auto_adjust_timer is not None:
            self._auto_adjust_timer.stop()
            self._auto_adjust_timer = None
            print("[Auto] Stopped auto-adjust timer")
    
    def _on_auto_threshold_change(self, value: float):
        """Update auto-range threshold (beat interval in seconds)"""
        self._auto_threshold_sec = value
        bpm = 60.0 / value if value > 0 else 0
        print(f"[Auto] Threshold changed to {value:.2f}s ({bpm:.0f} BPM)")
    
    def _on_auto_cooldown_change(self, value: float):
        """Update cooldown between auto-adjust steps (seconds)"""
        self._auto_cooldown_sec = value
        print(f"[Auto] Cooldown changed to {value:.2f}s")
    
    def _update_param_config(self, param: str, step: Optional[float] = None, lock_time: Optional[float] = None):
        """Update step size or lock time for an auto-adjust parameter from spinbox"""
        print(f"[Spinbox] _update_param_config called: param={param}, step={step}, lock_time={lock_time}")
        if param in self._auto_param_config:
            cur_step, cur_lock, cur_max = self._auto_param_config[param]
            new_step = step if step is not None else cur_step
            new_lock = lock_time if lock_time is not None else cur_lock
            self._auto_param_config[param] = (new_step, new_lock, cur_max)
            print(f"[Spinbox] ✓ {param} config: step={new_step:.3f}, lock={new_lock:.0f}ms")
        else:
            print(f"[Spinbox] ✗ {param} NOT in _auto_param_config!")
    
    def _auto_adjust_beat_detection(self):
        """Auto-adjust beat detection with per-parameter lock order.
        
        Lock Order:
        - audio_amp: locks at first beat detection
        - sensitivity: locks when tempo detected; reverses when downbeat found, re-locks when lost
        - peak_floor/peak_decay/rise_sens: hunt until downbeat, lock. Resume if downbeat lost.
        - flux_mult: hunt until downbeat, lock. Resume once if lost. Lock permanently on 2nd detection.
        
        Shorter steps with larger oscillations to sweep parameter space during hunting.
        """
        # Hunt as soon as audio stream is running (Start pressed), don't require Play
        if not self.is_running:
            return
        
        current_time = time.time()
        if current_time - self._auto_last_adjust_time < self._auto_cooldown_sec:
            return
        
        # Get beat/tempo info
        tempo_info = {'bpm': 0, 'confidence': 0}
        if hasattr(self, 'audio_engine') and self.audio_engine is not None:
            tempo_info = self.audio_engine.get_tempo_info()
        bpm = tempo_info.get('bpm', 0)
        
        # Downbeat confidence (lowered threshold)
        downbeat_conf = 0.0
        if self.audio_engine and hasattr(self.audio_engine, 'downbeat_confidence'):
            downbeat_conf = self.audio_engine.downbeat_confidence
        
        # Audio presence check
        has_audio = False
        if self.audio_engine and hasattr(self.audio_engine, 'peak_envelope'):
            has_audio = self.audio_engine.peak_envelope > 0.001
        
        # Reset all param states during silence
        if not has_audio:
            any_changed = False
            for p in self._auto_param_state:
                if self._auto_param_state[p] != 'HUNTING':
                    self._auto_param_state[p] = 'HUNTING'
                    any_changed = True
            if any_changed or self._auto_consec_locked:
                self._auto_flux_lock_count = 0
                self._auto_consec_locked = False
                self._auto_consec_beat_start = 0.0
                self._auto_no_beat_since = 0.0
                for p in self._auto_param_lock_time:
                    self._auto_param_lock_time[p] = 0.0
                print(f"[Auto] Silence - all params reset to HUNTING")
            return
        
        # Threshold BPM from spinbox
        lower_threshold_bpm = 60.0 / self._auto_threshold_sec if self._auto_threshold_sec > 0 else 140
        
        # Detection flags (lowered thresholds per instructions)
        has_beat = bpm > 0
        has_tempo = bpm >= lower_threshold_bpm
        has_downbeat = downbeat_conf >= self._auto_downbeat_threshold
        too_fast = bpm >= self._auto_upper_threshold_bpm
        
        # === CONSECUTIVE BEAT LOCK ===
        if has_beat and has_tempo:
            if self._auto_consec_beat_start == 0.0:
                self._auto_consec_beat_start = current_time
            elif not self._auto_consec_locked and (current_time - self._auto_consec_beat_start) >= self._auto_consec_lock_sec:
                self._auto_consec_locked = True
                for p in self._auto_param_state:
                    if self._auto_param_state[p] != 'LOCKED':
                        self._auto_param_state[p] = 'LOCKED'
                print(f"[Auto] CONSECUTIVE LOCK - all params LOCKED after {self._auto_consec_lock_sec:.1f}s of continuous beats")
        else:
            self._auto_consec_beat_start = 0.0  # Reset timer if beat lost
        
        # If consecutive-locked, skip all adjustments until silence resets
        if self._auto_consec_locked:
            return
        
        # === PER-PARAMETER STATE TRANSITIONS ===
        
        # Track no-beat timer for audio_amp unlock
        if has_beat:
            self._auto_no_beat_since = 0.0  # Reset - we have beats
        elif self._auto_no_beat_since == 0.0:
            self._auto_no_beat_since = current_time  # Start the no-beat timer
        
        no_beat_duration = (current_time - self._auto_no_beat_since) if self._auto_no_beat_since > 0 else 0.0
        
        # audio_amp: lock at first beat detection
        # Unlock ONLY if sensitivity is HUNTING AND no beat for 1500ms
        if self._auto_param_state['audio_amp'] == 'HUNTING' and has_beat:
            self._auto_param_state['audio_amp'] = 'LOCKED'
            self._auto_param_lock_time['audio_amp'] = current_time
            print(f"[Auto] audio_amp LOCKED (first beat at {bpm:.0f} BPM)")
        elif self._auto_param_state['audio_amp'] == 'LOCKED' and not has_beat:
            sens_is_hunting = self._auto_param_state['sensitivity'] == 'HUNTING'
            lock_cfg = self._auto_param_config.get('audio_amp', (0, 5000.0, 0))
            min_lock_time_ms = lock_cfg[1]
            time_locked_ms = (current_time - self._auto_param_lock_time.get('audio_amp', 0)) * 1000
            if sens_is_hunting and no_beat_duration >= 1.5 and time_locked_ms >= min_lock_time_ms:
                self._auto_param_state['audio_amp'] = 'HUNTING'
                print(f"[Auto] audio_amp resumed HUNTING (sensitivity hunting + no beat for {no_beat_duration:.1f}s)")
        
        # sensitivity: lock when tempo detected
        if self._auto_param_state['sensitivity'] == 'HUNTING' and has_tempo:
            self._auto_param_state['sensitivity'] = 'LOCKED'
            self._auto_param_lock_time['sensitivity'] = current_time
            print(f"[Auto] sensitivity LOCKED (tempo at {bpm:.0f} BPM)")
        # When downbeat found, reverse sensitivity to find minimum
        elif self._auto_param_state['sensitivity'] == 'LOCKED' and has_downbeat:
            lock_cfg = self._auto_param_config.get('sensitivity', (0, 5000.0, 0))
            time_locked_ms = (current_time - self._auto_param_lock_time.get('sensitivity', 0)) * 1000
            if time_locked_ms >= lock_cfg[1]:
                self._auto_param_state['sensitivity'] = 'REVERSING'
                print(f"[Auto] sensitivity REVERSING (downbeat conf={downbeat_conf:.2f})")
        # When downbeat lost during reversal, re-lock
        elif self._auto_param_state['sensitivity'] == 'REVERSING' and not has_downbeat:
            self._auto_param_state['sensitivity'] = 'LOCKED'
            self._auto_param_lock_time['sensitivity'] = current_time
            print(f"[Auto] sensitivity re-LOCKED (downbeat lost)")
        # If sensitivity is locked but no tempo anymore, unlock after lock_time
        elif self._auto_param_state['sensitivity'] == 'LOCKED' and not has_tempo:
            lock_cfg = self._auto_param_config.get('sensitivity', (0, 5000.0, 0))
            time_locked_ms = (current_time - self._auto_param_lock_time.get('sensitivity', 0)) * 1000
            if time_locked_ms >= lock_cfg[1]:
                self._auto_param_state['sensitivity'] = 'HUNTING'
                print(f"[Auto] sensitivity resumed HUNTING (tempo lost after {time_locked_ms:.0f}ms locked)")
        
        # peak_floor, peak_decay, rise_sens: hunt until downbeat, resume if lost (respecting lock time)
        for p in ['peak_floor', 'peak_decay', 'rise_sens']:
            if self._auto_param_state[p] == 'HUNTING' and has_downbeat:
                self._auto_param_state[p] = 'LOCKED'
                self._auto_param_lock_time[p] = current_time
                print(f"[Auto] {p} LOCKED (downbeat detected)")
            elif self._auto_param_state[p] == 'LOCKED' and not has_downbeat and has_tempo:
                lock_cfg = self._auto_param_config.get(p, (0, 5000.0, 0))
                time_locked_ms = (current_time - self._auto_param_lock_time.get(p, 0)) * 1000
                if time_locked_ms >= lock_cfg[1]:
                    self._auto_param_state[p] = 'HUNTING'
                    print(f"[Auto] {p} resumed HUNTING (downbeat lost after {time_locked_ms:.0f}ms locked)")
        
        # flux_mult: hunt until downbeat, lock permanently on 2nd detection
        if self._auto_param_state['flux_mult'] == 'HUNTING' and has_downbeat:
            self._auto_flux_lock_count += 1
            self._auto_param_state['flux_mult'] = 'LOCKED'
            self._auto_param_lock_time['flux_mult'] = current_time
            permanent = " (PERMANENT)" if self._auto_flux_lock_count >= 2 else ""
            print(f"[Auto] flux_mult LOCKED{permanent} (count={self._auto_flux_lock_count})")
        elif self._auto_param_state['flux_mult'] == 'LOCKED' and not has_downbeat and has_tempo:
            if self._auto_flux_lock_count < 2:
                lock_cfg = self._auto_param_config.get('flux_mult', (0, 5000.0, 0))
                time_locked_ms = (current_time - self._auto_param_lock_time.get('flux_mult', 0)) * 1000
                if time_locked_ms >= lock_cfg[1]:
                    self._auto_param_state['flux_mult'] = 'HUNTING'
                    print(f"[Auto] flux_mult resumed HUNTING (downbeat lost after {time_locked_ms:.0f}ms locked)")
            # If >= 2, stay LOCKED permanently
        
        # === HANDLE TOO-FAST: reverse all non-locked params ===
        if too_fast:
            enabled_params = [p for p in self._auto_param_order if self._auto_adjust_enabled.get(p, False)]
            huntable = [p for p in enabled_params if self._auto_param_state.get(p) in ('HUNTING', 'REVERSING')]
            if huntable:
                self._auto_param_index = self._auto_param_index % len(huntable)
                param = huntable[self._auto_param_index]
                adjusted = self._adjust_single_param(param, False, False)  # Lower sensitivity, no oscillation
                if adjusted:
                    self._auto_last_adjust_time = current_time
                    print(f"[Auto] ↓ {param} (too fast {bpm:.0f} BPM)")
                self._auto_param_index = (self._auto_param_index + 1) % len(huntable)
            return
        
        # === ADJUST NEXT ENABLED NON-LOCKED PARAMETER ===
        enabled_params = [p for p in self._auto_param_order if self._auto_adjust_enabled.get(p, False)]
        if not enabled_params:
            return
        
        adjustable = [p for p in enabled_params if self._auto_param_state.get(p) in ('HUNTING', 'REVERSING')]
        if not adjustable:
            return
        
        self._auto_param_index = self._auto_param_index % len(adjustable)
        param = adjustable[self._auto_param_index]
        
        state = self._auto_param_state[param]
        raise_sensitivity = (state == 'HUNTING')  # HUNTING=raise, REVERSING=lower
        oscillate = (state == 'HUNTING')  # Only oscillate when hunting
        
        adjusted = self._adjust_single_param(param, raise_sensitivity, oscillate)
        
        if adjusted:
            self._auto_last_adjust_time = current_time
            direction = "↑" if raise_sensitivity else "↓"
            print(f"[Auto] {direction} {param} ({state}, BPM={bpm:.0f}, db={downbeat_conf:.2f})")
        
        self._auto_param_index = (self._auto_param_index + 1) % len(adjustable)
        self._auto_oscillation_phase += 0.3
    
    def _adjust_single_param(self, param: str, raise_sensitivity: bool, oscillate: bool = False) -> bool:
        """Adjust a single beat detection parameter with impact-based step sizes.
        
        Args:
            param: Parameter name
            raise_sensitivity: True to increase sensitivity, False to decrease
            oscillate: Add small oscillation on top of direction (for hunting mode)
        
        Returns True if a change was made.
        """
        config = self._auto_param_config.get(param)
        if not config:
            return False
        
        step_size, lock_time_ms, hunt_max = config
        
        # Oscillation amplitude is always 3/4 of step size
        osc_amp = step_size * 0.75
        
        # Calculate oscillation component (small sine wave)
        osc_offset = 0.0
        if oscillate:
            osc_offset = np.sin(self._auto_oscillation_phase) * osc_amp
        
        if param == 'audio_amp':
            current = self.audio_gain_slider.value()
            # LIMIT TO 1.0 when hunting (auto-range shouldn't boost beyond unity)
            max_val = hunt_max if raise_sensitivity else 10.0
            if raise_sensitivity:
                new_val = min(max_val, current + step_size + osc_offset)
            else:
                new_val = max(0.1, current - step_size + osc_offset)
            new_val = max(0.1, min(10.0, new_val))
            if abs(new_val - current) > 0.001:
                self.audio_gain_slider.setValue(new_val)
                return True
        elif param == 'peak_floor':
            current = self.peak_floor_slider.value()
            # Lower floor = more sensitive (inverted)
            if raise_sensitivity:
                new_val = current - step_size - osc_offset  # Lower
            else:
                new_val = current + step_size - osc_offset  # Raise
            new_val = max(0.0, min(0.8, new_val))
            if abs(new_val - current) > 0.001:
                self.peak_floor_slider.setValue(new_val)
                return True
        elif param == 'peak_decay':
            current = self.peak_decay_slider.value()
            # Lower decay = more sensitive (inverted)
            if raise_sensitivity:
                new_val = current - step_size - osc_offset  # Lower
            else:
                new_val = current + step_size - osc_offset  # Raise
            new_val = max(0.5, min(0.999, new_val))
            if abs(new_val - current) > 0.001:
                self.peak_decay_slider.setValue(new_val)
                return True
        elif param == 'rise_sens':
            current = self.rise_sens_slider.value()
            # Rise sensitivity inverted: lower = more tolerant, higher = stricter
            if raise_sensitivity:
                new_val = current - step_size - osc_offset  # Lower (more tolerant)
            else:
                new_val = current + step_size - osc_offset  # Raise (stricter)
            new_val = max(0.0, min(1.0, new_val))
            if abs(new_val - current) > 0.001:
                self.rise_sens_slider.setValue(new_val)
                return True
        elif param == 'sensitivity':
            current = self.sensitivity_slider.value()
            if raise_sensitivity:
                new_val = current + step_size + osc_offset
            else:
                new_val = current - step_size + osc_offset
            new_val = max(0.0, min(1.0, new_val))
            if abs(new_val - current) > 0.001:
                self.sensitivity_slider.setValue(new_val)
                return True
        elif param == 'flux_mult':
            current = self.flux_mult_slider.value()
            if raise_sensitivity:
                new_val = current + step_size + osc_offset
            else:
                new_val = current - step_size + osc_offset
            new_val = max(0.1, min(5.0, new_val))
            if abs(new_val - current) > 0.001:
                self.flux_mult_slider.setValue(new_val)
                return True
        return False
    
    def _create_beat_detection_tab(self) -> QWidget:
        """Beat detection settings"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Detection type with global auto-range toggle and threshold
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Detection Type:"))
        self.detection_type_combo = QComboBox()
        self.detection_type_combo.addItems(["Peak Energy", "Spectral Flux", "Combined"])
        self.detection_type_combo.setCurrentIndex(2)  # Combined
        self.detection_type_combo.currentIndexChanged.connect(self._on_detection_type_change)
        type_layout.addWidget(self.detection_type_combo)
        type_layout.addStretch()
        self.global_auto_range_cb = QCheckBox("auto-range")
        self.global_auto_range_cb.setToolTip("Toggle all auto-adjustment functions for beat detection")
        self.global_auto_range_cb.stateChanged.connect(lambda state: self._enable_all_auto_beat_detection(state == 2))
        type_layout.addWidget(self.global_auto_range_cb)
        # Threshold spinbox for auto-range (in seconds, 100ms-2000ms)
        self.auto_threshold_spin = QDoubleSpinBox()
        self.auto_threshold_spin.setRange(0.10, 2.00)
        self.auto_threshold_spin.setSingleStep(0.01)
        self.auto_threshold_spin.setDecimals(2)
        self.auto_threshold_spin.setValue(0.43)  # Default 428ms = 140 BPM
        self.auto_threshold_spin.setFixedWidth(60)
        self.auto_threshold_spin.setToolTip("Beat interval threshold (sec) - auto lowers sensitivity above this rate")
        self.auto_threshold_spin.valueChanged.connect(self._on_auto_threshold_change)
        type_layout.addWidget(self.auto_threshold_spin)
        # Cooldown spinbox for auto-range (seconds between parameter adjustments)
        type_layout.addWidget(QLabel("cd:"))
        self.auto_cooldown_spin = QDoubleSpinBox()
        self.auto_cooldown_spin.setRange(0.01, 5.00)
        self.auto_cooldown_spin.setSingleStep(0.05)
        self.auto_cooldown_spin.setDecimals(2)
        self.auto_cooldown_spin.setValue(0.10)  # Default 100ms between adjustments
        self.auto_cooldown_spin.setFixedWidth(60)
        self.auto_cooldown_spin.setToolTip("Cooldown (sec) between auto-adjust steps - higher = more stable, lower = faster convergence")
        self.auto_cooldown_spin.valueChanged.connect(self._on_auto_cooldown_change)
        type_layout.addWidget(self.auto_cooldown_spin)
        # Consecutive-beat full lock timer
        type_layout.addWidget(QLabel("lock:"))
        self.auto_consec_lock_spin = QDoubleSpinBox()
        self.auto_consec_lock_spin.setRange(1.0, 30.0)
        self.auto_consec_lock_spin.setSingleStep(0.5)
        self.auto_consec_lock_spin.setDecimals(1)
        self.auto_consec_lock_spin.setValue(5.0)
        self.auto_consec_lock_spin.setFixedWidth(75)
        self.auto_consec_lock_spin.setToolTip("Seconds of consecutive beats before ALL hunting stops (until silence)")
        self.auto_consec_lock_spin.valueChanged.connect(lambda v: setattr(self, '_auto_consec_lock_sec', v))
        type_layout.addWidget(self.auto_consec_lock_spin)
        layout.addLayout(type_layout)
        
        # Frequency band selection
        freq_group = QGroupBox("Frequency Band (Hz) - red overlay on spectrum")
        freq_layout = QVBoxLayout(freq_group)
        
        # Full range up to ~20kHz (Nyquist for 44100 Hz)

        # Get sample rate (default to 44100 if not available yet)
        sr = getattr(self.config.audio, 'sample_rate', 44100)
        nyquist = sr // 2
        self.freq_range_slider = RangeSliderWithLabel("Freq Range (Hz)", 30, 22050, 30, 4000, 0)
        self.freq_range_slider.rangeChanged.connect(self._on_freq_band_change)
        freq_layout.addWidget(self.freq_range_slider)
        
        layout.addWidget(freq_group)
        
        # Sliders - with better defaults
        # Sensitivity: higher = more beats detected (0.0=strict, 1.0=very sensitive) - with auto toggle
        sens_row = QHBoxLayout()
        self.sensitivity_slider = SliderWithLabel("Sensitivity", 0.0, 1.0, 0.7)
        self.sensitivity_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'sensitivity', v))
        sens_row.addWidget(self.sensitivity_slider, 1)
        self.sensitivity_auto_cb = QCheckBox("auto")
        self.sensitivity_auto_cb.setToolTip("Auto-raise when no beats detected, lower when too many")
        self.sensitivity_auto_cb.stateChanged.connect(lambda state: self._on_auto_toggle('sensitivity', state))
        sens_row.addWidget(self.sensitivity_auto_cb)
        self.sensitivity_step_spin = QDoubleSpinBox()
        self.sensitivity_step_spin.setRange(0.001, 0.1)
        self.sensitivity_step_spin.setSingleStep(0.001)
        self.sensitivity_step_spin.setDecimals(3)
        self.sensitivity_step_spin.setValue(0.008)
        self.sensitivity_step_spin.setFixedWidth(75)
        self.sensitivity_step_spin.setToolTip("Step size")
        self.sensitivity_step_spin.valueChanged.connect(lambda v: self._update_param_config('sensitivity', step=v))
        sens_row.addWidget(self.sensitivity_step_spin)
        self.sensitivity_lock_spin = QDoubleSpinBox()
        self.sensitivity_lock_spin.setRange(250.0, 5000.0)
        self.sensitivity_lock_spin.setSingleStep(250.0)
        self.sensitivity_lock_spin.setDecimals(0)
        self.sensitivity_lock_spin.setValue(5000.0)
        self.sensitivity_lock_spin.setFixedWidth(75)
        self.sensitivity_lock_spin.setToolTip("Lock time (ms)")
        self.sensitivity_lock_spin.valueChanged.connect(lambda v: self._update_param_config('sensitivity', lock_time=v))
        sens_row.addWidget(self.sensitivity_lock_spin)
        layout.addLayout(sens_row)
        
        # Peak floor: minimum energy to consider (0 = disabled) - with auto toggle
        peak_floor_row = QHBoxLayout()
        self.peak_floor_slider = SliderWithLabel("Peak Floor", 0.0, 0.8, 0.0, 2)
        self.peak_floor_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'peak_floor', v))
        peak_floor_row.addWidget(self.peak_floor_slider, 1)
        self.peak_floor_auto_cb = QCheckBox("auto")
        self.peak_floor_auto_cb.setToolTip("Auto-lower when no beats detected, raise when too many")
        self.peak_floor_auto_cb.stateChanged.connect(lambda state: self._on_auto_toggle('peak_floor', state))
        peak_floor_row.addWidget(self.peak_floor_auto_cb)
        self.peak_floor_step_spin = QDoubleSpinBox()
        self.peak_floor_step_spin.setRange(0.001, 0.1)
        self.peak_floor_step_spin.setSingleStep(0.001)
        self.peak_floor_step_spin.setDecimals(3)
        self.peak_floor_step_spin.setValue(0.004)
        self.peak_floor_step_spin.setFixedWidth(75)
        self.peak_floor_step_spin.setToolTip("Step size")
        self.peak_floor_step_spin.valueChanged.connect(lambda v: self._update_param_config('peak_floor', step=v))
        peak_floor_row.addWidget(self.peak_floor_step_spin)
        self.peak_floor_lock_spin = QDoubleSpinBox()
        self.peak_floor_lock_spin.setRange(250.0, 5000.0)
        self.peak_floor_lock_spin.setSingleStep(250.0)
        self.peak_floor_lock_spin.setDecimals(0)
        self.peak_floor_lock_spin.setValue(5000.0)
        self.peak_floor_lock_spin.setFixedWidth(75)
        self.peak_floor_lock_spin.setToolTip("Lock time (ms)")
        self.peak_floor_lock_spin.valueChanged.connect(lambda v: self._update_param_config('peak_floor', lock_time=v))
        peak_floor_row.addWidget(self.peak_floor_lock_spin)
        layout.addLayout(peak_floor_row)
        
        # Peak decay - with auto toggle
        peak_decay_row = QHBoxLayout()
        self.peak_decay_slider = SliderWithLabel("Peak Decay", 0.5, 0.999, 0.9, 3)
        self.peak_decay_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'peak_decay', v))
        peak_decay_row.addWidget(self.peak_decay_slider, 1)
        self.peak_decay_auto_cb = QCheckBox("auto")
        self.peak_decay_auto_cb.setToolTip("Auto-lower when no beats detected, raise when too many")
        self.peak_decay_auto_cb.stateChanged.connect(lambda state: self._on_auto_toggle('peak_decay', state))
        peak_decay_row.addWidget(self.peak_decay_auto_cb)
        self.peak_decay_step_spin = QDoubleSpinBox()
        self.peak_decay_step_spin.setRange(0.001, 0.1)
        self.peak_decay_step_spin.setSingleStep(0.001)
        self.peak_decay_step_spin.setDecimals(3)
        self.peak_decay_step_spin.setValue(0.002)
        self.peak_decay_step_spin.setFixedWidth(75)
        self.peak_decay_step_spin.setToolTip("Step size")
        self.peak_decay_step_spin.valueChanged.connect(lambda v: self._update_param_config('peak_decay', step=v))
        peak_decay_row.addWidget(self.peak_decay_step_spin)
        self.peak_decay_lock_spin = QDoubleSpinBox()
        self.peak_decay_lock_spin.setRange(250.0, 5000.0)
        self.peak_decay_lock_spin.setSingleStep(250.0)
        self.peak_decay_lock_spin.setDecimals(0)
        self.peak_decay_lock_spin.setValue(5000.0)
        self.peak_decay_lock_spin.setFixedWidth(75)
        self.peak_decay_lock_spin.setToolTip("Lock time (ms)")
        self.peak_decay_lock_spin.valueChanged.connect(lambda v: self._update_param_config('peak_decay', lock_time=v))
        peak_decay_row.addWidget(self.peak_decay_lock_spin)
        layout.addLayout(peak_decay_row)
        
        # Rise sensitivity: 0 = disabled, higher = require more rise - with auto toggle
        rise_sens_row = QHBoxLayout()
        self.rise_sens_slider = SliderWithLabel("Rise Sensitivity", 0.0, 1.0, 0.0)
        self.rise_sens_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'rise_sensitivity', v))
        rise_sens_row.addWidget(self.rise_sens_slider, 1)
        self.rise_sens_auto_cb = QCheckBox("auto")
        self.rise_sens_auto_cb.setToolTip("Auto-lower when no beats detected (more sensitive), raise when too many")
        self.rise_sens_auto_cb.stateChanged.connect(lambda state: self._on_auto_toggle('rise_sens', state))
        rise_sens_row.addWidget(self.rise_sens_auto_cb)
        self.rise_sens_step_spin = QDoubleSpinBox()
        self.rise_sens_step_spin.setRange(0.001, 0.1)
        self.rise_sens_step_spin.setSingleStep(0.001)
        self.rise_sens_step_spin.setDecimals(3)
        self.rise_sens_step_spin.setValue(0.008)
        self.rise_sens_step_spin.setFixedWidth(75)
        self.rise_sens_step_spin.setToolTip("Step size")
        self.rise_sens_step_spin.valueChanged.connect(lambda v: self._update_param_config('rise_sens', step=v))
        rise_sens_row.addWidget(self.rise_sens_step_spin)
        self.rise_sens_lock_spin = QDoubleSpinBox()
        self.rise_sens_lock_spin.setRange(250.0, 5000.0)
        self.rise_sens_lock_spin.setSingleStep(250.0)
        self.rise_sens_lock_spin.setDecimals(0)
        self.rise_sens_lock_spin.setValue(5000.0)
        self.rise_sens_lock_spin.setFixedWidth(75)
        self.rise_sens_lock_spin.setToolTip("Lock time (ms)")
        self.rise_sens_lock_spin.valueChanged.connect(lambda v: self._update_param_config('rise_sens', lock_time=v))
        rise_sens_row.addWidget(self.rise_sens_lock_spin)
        layout.addLayout(rise_sens_row)
        
        # Flux Multiplier - with auto toggle
        flux_mult_row = QHBoxLayout()
        self.flux_mult_slider = SliderWithLabel("Flux Multiplier", 0.1, 5.0, 1.0, 1)
        self.flux_mult_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'flux_multiplier', v))
        flux_mult_row.addWidget(self.flux_mult_slider, 1)
        self.flux_mult_auto_cb = QCheckBox("auto")
        self.flux_mult_auto_cb.setToolTip("Auto-raise when no beats detected, lower when too many")
        self.flux_mult_auto_cb.stateChanged.connect(lambda state: self._on_auto_toggle('flux_mult', state))
        flux_mult_row.addWidget(self.flux_mult_auto_cb)
        self.flux_mult_step_spin = QDoubleSpinBox()
        self.flux_mult_step_spin.setRange(0.001, 0.5)
        self.flux_mult_step_spin.setSingleStep(0.005)
        self.flux_mult_step_spin.setDecimals(3)
        self.flux_mult_step_spin.setValue(0.015)
        self.flux_mult_step_spin.setFixedWidth(75)
        self.flux_mult_step_spin.setToolTip("Step size")
        self.flux_mult_step_spin.valueChanged.connect(lambda v: self._update_param_config('flux_mult', step=v))
        flux_mult_row.addWidget(self.flux_mult_step_spin)
        self.flux_mult_lock_spin = QDoubleSpinBox()
        self.flux_mult_lock_spin.setRange(250.0, 5000.0)
        self.flux_mult_lock_spin.setSingleStep(250.0)
        self.flux_mult_lock_spin.setDecimals(0)
        self.flux_mult_lock_spin.setValue(5000.0)
        self.flux_mult_lock_spin.setFixedWidth(75)
        self.flux_mult_lock_spin.setToolTip("Lock time (ms)")
        self.flux_mult_lock_spin.valueChanged.connect(lambda v: self._update_param_config('flux_mult', lock_time=v))
        flux_mult_row.addWidget(self.flux_mult_lock_spin)
        layout.addLayout(flux_mult_row)
        
        # Audio amplification/gain: boost weak signals (0.1=quiet, 10.0=loud) - with auto toggle
        audio_gain_row = QHBoxLayout()
        self.audio_gain_slider = SliderWithLabel("Audio Amplification", 0.1, 10.0, 5.0, 1)
        self.audio_gain_slider.valueChanged.connect(lambda v: setattr(self.config.audio, 'gain', v))
        audio_gain_row.addWidget(self.audio_gain_slider, 1)
        self.audio_gain_auto_cb = QCheckBox("auto")
        self.audio_gain_auto_cb.setToolTip("Auto-raise when no beats detected, lower when too many")
        self.audio_gain_auto_cb.stateChanged.connect(lambda state: self._on_auto_toggle('audio_amp', state))
        audio_gain_row.addWidget(self.audio_gain_auto_cb)
        self.audio_amp_step_spin = QDoubleSpinBox()
        self.audio_amp_step_spin.setRange(0.001, 0.5)
        self.audio_amp_step_spin.setSingleStep(0.01)
        self.audio_amp_step_spin.setDecimals(3)
        self.audio_amp_step_spin.setValue(0.040)
        self.audio_amp_step_spin.setFixedWidth(75)
        self.audio_amp_step_spin.setToolTip("Step size")
        self.audio_amp_step_spin.valueChanged.connect(lambda v: self._update_param_config('audio_amp', step=v))
        audio_gain_row.addWidget(self.audio_amp_step_spin)
        self.audio_amp_lock_spin = QDoubleSpinBox()
        self.audio_amp_lock_spin.setRange(250.0, 5000.0)
        self.audio_amp_lock_spin.setSingleStep(250.0)
        self.audio_amp_lock_spin.setDecimals(0)
        self.audio_amp_lock_spin.setValue(5000.0)
        self.audio_amp_lock_spin.setFixedWidth(75)
        self.audio_amp_lock_spin.setToolTip("Lock time (ms)")
        self.audio_amp_lock_spin.valueChanged.connect(lambda v: self._update_param_config('audio_amp', lock_time=v))
        audio_gain_row.addWidget(self.audio_amp_lock_spin)
        layout.addLayout(audio_gain_row)
        
        # Butterworth filter checkbox (requires restart)
        self.butterworth_checkbox = QCheckBox("Use Butterworth bandpass filter (better bass isolation)")
        self.butterworth_checkbox.setChecked(getattr(self.config.audio, 'use_butterworth', True))
        self.butterworth_checkbox.stateChanged.connect(self._on_butterworth_toggle)
        layout.addWidget(self.butterworth_checkbox)
        
        layout.addStretch()
        return widget
    
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

            # Axis Weights Tab
            'alpha_weight': self.alpha_weight_slider.value(),
            'beta_weight': self.beta_weight_slider.value(),

            # Other Tab
            'pulse_freq_low': self.pulse_freq_range_slider.low(),
            'pulse_freq_high': self.pulse_freq_range_slider.high(),
            'tcode_freq_min': self.tcode_freq_range_slider.low(),
            'tcode_freq_max': self.tcode_freq_range_slider.high(),
            'freq_weight': self.freq_weight_slider.value(),

            # Auto-adjust toggle states
            'auto_audio_amp': self.audio_gain_auto_cb.isChecked(),
            'auto_peak_floor': self.peak_floor_auto_cb.isChecked(),
            'auto_peak_decay': self.peak_decay_auto_cb.isChecked(),
            'auto_rise_sens': self.rise_sens_auto_cb.isChecked(),
            'auto_sensitivity': self.sensitivity_auto_cb.isChecked(),
            'auto_flux_mult': self.flux_mult_auto_cb.isChecked(),
            'auto_global': self.global_auto_range_cb.isChecked(),
            'auto_threshold_sec': self._auto_threshold_sec,
            'auto_cooldown_sec': self._auto_cooldown_sec,
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
            # Axis Weights Tab
            self.alpha_weight_slider.setValue(preset_data['alpha_weight'])
            self.beta_weight_slider.setValue(preset_data['beta_weight'])

            # Other Tab
            if 'pulse_freq_low' in preset_data:
                self.pulse_freq_range_slider.setLow(preset_data['pulse_freq_low'])
            if 'pulse_freq_high' in preset_data:
                self.pulse_freq_range_slider.setHigh(preset_data['pulse_freq_high'])
            if 'tcode_freq_min' in preset_data:
                self.tcode_freq_range_slider.setLow(preset_data['tcode_freq_min'])
            if 'tcode_freq_max' in preset_data:
                self.tcode_freq_range_slider.setHigh(preset_data['tcode_freq_max'])
            if 'freq_weight' in preset_data:
                self.freq_weight_slider.setValue(preset_data['freq_weight'])

            # Auto-adjust toggle states
            if 'auto_audio_amp' in preset_data:
                self.audio_gain_auto_cb.setChecked(preset_data['auto_audio_amp'])
            if 'auto_peak_floor' in preset_data:
                self.peak_floor_auto_cb.setChecked(preset_data['auto_peak_floor'])
            if 'auto_peak_decay' in preset_data:
                self.peak_decay_auto_cb.setChecked(preset_data['auto_peak_decay'])
            if 'auto_rise_sens' in preset_data:
                self.rise_sens_auto_cb.setChecked(preset_data['auto_rise_sens'])
            if 'auto_sensitivity' in preset_data:
                self.sensitivity_auto_cb.setChecked(preset_data['auto_sensitivity'])
            if 'auto_flux_mult' in preset_data:
                self.flux_mult_auto_cb.setChecked(preset_data['auto_flux_mult'])
            if 'auto_global' in preset_data:
                self.global_auto_range_cb.blockSignals(True)
                self.global_auto_range_cb.setChecked(preset_data['auto_global'])
                self.global_auto_range_cb.blockSignals(False)
            if 'auto_threshold_sec' in preset_data:
                self.auto_threshold_spin.setValue(preset_data['auto_threshold_sec'])
            if 'auto_cooldown_sec' in preset_data:
                self.auto_cooldown_spin.setValue(preset_data['auto_cooldown_sec'])

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
        if getattr(sys, 'frozen', False):
            # Running as packaged exe - save in same folder as exe
            return Path(sys.executable).parent / "presets.json"
        else:
            # Running from source - use workspace folder (for editing factory presets)
            return Path(__file__).parent / "presets.json"
    
    def _save_presets_to_disk(self):
        """Save all custom presets to disk"""
        try:
            presets_file = self._get_presets_file_path()
            with open(presets_file, 'w') as f:
                json.dump(self.custom_beat_presets, f, indent=2)
            print(f"[Presets] Saved {len(self.custom_beat_presets)} presets to {presets_file}")
        except Exception as e:
            print(f"[Presets] Error saving presets: {e}")
    
    def _load_presets_from_disk(self):
        """Load custom presets from disk"""
        try:
            presets_file = self._get_presets_file_path()
            
            # If no user presets file exists, try to copy factory presets from bundled location
            if not presets_file.exists() and getattr(sys, 'frozen', False):
                meipass = getattr(sys, '_MEIPASS', None)
                if meipass:
                    factory_presets = Path(meipass) / 'presets.json'
                    if factory_presets.exists():
                        import shutil
                        shutil.copy(factory_presets, presets_file)
                        print(f"[Presets] Copied factory presets to {presets_file}")
            
            if presets_file.exists():
                with open(presets_file, 'r') as f:
                    self.custom_beat_presets = json.load(f)
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
                self.custom_beat_presets = {}
                print(f"[Presets] No presets file found, starting with empty presets")
        except Exception as e:
            print(f"[Presets] Error loading presets: {e}")
            self.custom_beat_presets = {}
    
    def _create_stroke_settings_tab(self) -> QWidget:
        """Stroke generation settings"""
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
        
        # Sliders
        self.stroke_range_slider = RangeSliderWithLabel("Stroke Min/Max", 0.0, 1.0, 0.2, 1.0, 2)
        self.stroke_range_slider.rangeChanged.connect(self._on_stroke_range_change)
        layout.addWidget(self.stroke_range_slider)
        
        self.min_interval_slider = SliderWithLabel("Min Interval (ms)", 50, 500, 100, 0)
        self.min_interval_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'min_interval_ms', int(v)))
        layout.addWidget(self.min_interval_slider)
        
        self.fullness_slider = SliderWithLabel("Stroke Fullness", 0.0, 1.0, 0.7)
        self.fullness_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'stroke_fullness', v))
        layout.addWidget(self.fullness_slider)
        
        self.min_depth_slider = SliderWithLabel("Minimum Depth", 0.0, 1.0, 0.0)
        self.min_depth_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'minimum_depth', v))
        layout.addWidget(self.min_depth_slider)
        
        self.freq_depth_slider = SliderWithLabel("Freq Depth Factor", 0.0, 1.0, 0.3)
        self.freq_depth_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'freq_depth_factor', v))
        layout.addWidget(self.freq_depth_slider)
        
        # Frequency range for stroke depth - shown as green overlay on spectrum
        depth_freq_group = QGroupBox("Depth Frequency Range (Hz) - green overlay on spectrum")
        depth_freq_layout = QVBoxLayout(depth_freq_group)
        
        self.depth_freq_range_slider = RangeSliderWithLabel("Depth Freq (Hz)", 30, 22050, 30, 4000, 0)
        self.depth_freq_range_slider.rangeChanged.connect(self._on_depth_band_change)
        depth_freq_layout.addWidget(self.depth_freq_range_slider)
        
        layout.addWidget(depth_freq_group)
        
        layout.addStretch()
        return widget
    
    def _create_jitter_creep_tab(self) -> QWidget:
        """Jitter and creep settings"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Jitter section
        jitter_group = QGroupBox("Vibration (micro-circles when idle)")
        jitter_layout = QVBoxLayout(jitter_group)
        
        self.jitter_enabled = QCheckBox("Enable Jitter")
        self.jitter_enabled.setChecked(True)
        self.jitter_enabled.stateChanged.connect(lambda s: setattr(self.config.jitter, 'enabled', s == 2))
        jitter_layout.addWidget(self.jitter_enabled)
        
        self.jitter_amplitude_slider = SliderWithLabel("Circle Size", 0.01, 0.1, 0.1, 3)
        self.jitter_amplitude_slider.valueChanged.connect(lambda v: setattr(self.config.jitter, 'amplitude', v))
        jitter_layout.addWidget(self.jitter_amplitude_slider)
        
        self.jitter_intensity_slider = SliderWithLabel("Circle Speed", 0.0, 10.0, 0.5)
        self.jitter_intensity_slider.valueChanged.connect(lambda v: setattr(self.config.jitter, 'intensity', v))
        jitter_layout.addWidget(self.jitter_intensity_slider)
        
        layout.addWidget(jitter_group)
        
        # Creep section
        creep_group = QGroupBox("Creep (slow drift when idle)")
        creep_layout = QVBoxLayout(creep_group)
        
        self.creep_enabled = QCheckBox("Enable Creep")
        self.creep_enabled.setChecked(True)
        self.creep_enabled.stateChanged.connect(lambda s: setattr(self.config.creep, 'enabled', s == 2))
        creep_layout.addWidget(self.creep_enabled)
        
        self.creep_speed_slider = SliderWithLabel("Creep Speed", 0.0, 0.1, 0.02, 3)
        self.creep_speed_slider.valueChanged.connect(lambda v: setattr(self.config.creep, 'speed', v))
        creep_layout.addWidget(self.creep_speed_slider)
        
        layout.addWidget(creep_group)
        
        # Axis Weights section (moved from Stroke Settings)
        axis_group = QGroupBox("Axis Weights")
        axis_layout = QVBoxLayout(axis_group)
        axis_layout.addWidget(QLabel("Modes 1-3: Scales axis amplitude (0=off, 1=normal, 2=double)"))
        axis_layout.addWidget(QLabel("Mode 4 (User): Controls flux/peak response (0=flux, 1=balanced, 2=peak)"))
        
        self.alpha_weight_slider = SliderWithLabel("Alpha Weight", 0.0, 2.0, 1.0)
        self.alpha_weight_slider.valueChanged.connect(lambda v: setattr(self.config, 'alpha_weight', v))
        axis_layout.addWidget(self.alpha_weight_slider)
        
        self.beta_weight_slider = SliderWithLabel("Beta Weight", 0.0, 2.0, 1.0)
        self.beta_weight_slider.valueChanged.connect(lambda v: setattr(self.config, 'beta_weight', v))
        axis_layout.addWidget(self.beta_weight_slider)
        
        layout.addWidget(axis_group)
        
        layout.addStretch()
        return widget
    
    def _create_tempo_tracking_tab(self) -> QWidget:
        """Tempo tracking and rhythm settings"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Enable/disable checkbox
        self.tempo_tracking_checkbox = QCheckBox("Enable Tempo Tracking")
        self.tempo_tracking_checkbox.setChecked(True)
        self.tempo_tracking_checkbox.stateChanged.connect(self._on_tempo_tracking_toggle)
        layout.addWidget(self.tempo_tracking_checkbox)
        
        # Time signature dropdown
        sig_layout = QHBoxLayout()
        sig_layout.addWidget(QLabel("Time Signature:"))
        self.time_sig_combo = QComboBox()
        self.time_sig_combo.addItems(["4/4 (4 beats)", "3/4 (3 beats)", "6/8 (6 beats)"])
        self.time_sig_combo.currentIndexChanged.connect(self._on_time_sig_change)
        sig_layout.addWidget(self.time_sig_combo)
        sig_layout.addStretch()
        layout.addLayout(sig_layout)
        
        # Stability threshold: lower = stricter (requires more consistent intervals before locking BPM)
        self.stability_threshold_slider = SliderWithLabel("Stability Threshold", 0.05, 0.4, 0.15, 2)
        self.stability_threshold_slider.valueChanged.connect(self._on_stability_threshold_change)
        layout.addWidget(self.stability_threshold_slider)
        
        # Tempo timeout: how long no beats before resetting tempo tracking
        self.tempo_timeout_slider = SliderWithLabel("Tempo Timeout (ms)", 500, 5000, 2000, 0)
        self.tempo_timeout_slider.valueChanged.connect(self._on_tempo_timeout_change)
        layout.addWidget(self.tempo_timeout_slider)
        
        # Phase snap: how much to nudge detected beats toward predicted time
        self.phase_snap_slider = SliderWithLabel("Phase Snap", 0.0, 0.8, 0.3, 2)
        self.phase_snap_slider.valueChanged.connect(self._on_phase_snap_change)
        layout.addWidget(self.phase_snap_slider)
        
        # Silence reset threshold (moved from Beat Detection)
        self.silence_reset_slider = SliderWithLabel("Silence Reset (ms)", 100, 3000, 400, 0)
        self.silence_reset_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'silence_reset_ms', int(v)))
        layout.addWidget(self.silence_reset_slider)
        
        # Spectral flux control group (moved from Stroke Settings)
        flux_group = QGroupBox("Spectral Flux Control")
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
        return widget
    
    # Event handlers
    def _auto_connect_tcp(self):
        """Auto-connect TCP on program startup"""
        self.config.connection.host = self.host_edit.text()
        self.config.connection.port = self.port_spin.value()
        self.network_engine = NetworkEngine(self.config, self._network_status_callback)
        self.network_engine.start()
        print("[Main] Auto-connecting TCP on startup")

    def _on_connect(self):
        """Handle connect/disconnect button"""
        if self.network_engine is None:
            self.config.connection.host = self.host_edit.text()
            self.config.connection.port = self.port_spin.value()
            self.network_engine = NetworkEngine(self.config, self._network_status_callback)
            self.network_engine.start()
        else:
            if self.network_engine.connected:
                self.network_engine.user_disconnect()
            else:
                self.network_engine.user_connect()
    
    def _on_test(self):
        """Send test pattern"""
        if self.network_engine and self.network_engine.connected:
            # Temporarily enable sending for test
            was_sending = self.network_engine.sending_enabled
            self.network_engine.set_sending_enabled(True)
            self.network_engine.send_test()
            # Restore after a delay (test takes ~2.5 seconds)
            if not was_sending:
                QTimer.singleShot(3000, lambda: self.network_engine.set_sending_enabled(was_sending) if self.network_engine else None)
    
    def _on_start_stop(self, checked: bool):
        """Start/stop audio capture"""
        if checked:
            self._start_engines()
            self.start_btn.setText("■ Stop")
            self.play_btn.setEnabled(True)
        else:
            # Send zero-volume command before stopping engines
            self._volume_ramp_active = False
            if self.network_engine and self.is_sending:
                zero_cmd = TCodeCommand(alpha=0.5, beta=0.5, volume=0.0, duration_ms=100)
                self.network_engine.send_command(zero_cmd)
            self._stop_engines()
            self.start_btn.setText("▶ Start")
            self.play_btn.setEnabled(False)
            self.play_btn.setChecked(False)
            self.play_btn.setText("▶ Play")  # Reset play button text
            self.is_sending = False
    
    def _on_play_pause(self, checked: bool):
        """Play/pause sending commands"""
        self.is_sending = checked
        if checked:
            # Re-instantiate StrokeMapper with current config (for live mode switching)
            self.stroke_mapper = StrokeMapper(self.config, self._send_command_direct, get_volume=lambda: self.volume_slider.value(), audio_engine=self.audio_engine)
            # Start volume ramp from 0 to 1 over 800ms
            self._volume_ramp_active = True
            self._volume_ramp_start_time = time.time()
            self._volume_ramp_from = 0.0
            self._volume_ramp_to = 1.0
        else:
            # Immediately stop volume ramp and send zero-volume command
            self._volume_ramp_active = False
            if self.network_engine:
                zero_cmd = TCodeCommand(alpha=0.5, beta=0.5, volume=0.0, duration_ms=100)
                self.network_engine.send_command(zero_cmd)
        if self.network_engine:
            self.network_engine.set_sending_enabled(checked)
        self.play_btn.setText("⏸ Pause" if checked else "▶ Play")
    
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

        self.stroke_mapper = StrokeMapper(self.config, self._send_command_direct, get_volume=lambda: self.volume_slider.value(), audio_engine=self.audio_engine)

        # Network engine is already started on program launch via _auto_connect_tcp
        # Only create if somehow missing
        if self.network_engine is None:
            self.network_engine = NetworkEngine(self.config, self._network_status_callback)
            self.network_engine.start()

        self.is_running = True
    
    def _send_command_direct(self, cmd: TCodeCommand):
        """Send a command directly (used by StrokeMapper for arc strokes). Thread-safe."""
        if self.network_engine and self.is_sending:
            # Attach cached P0/F0 values (computed by audio callback)
            if self._cached_p0_enabled and self._cached_p0_val is not None:
                cmd.pulse_freq = self._cached_p0_val
            if self._cached_f0_enabled and self._cached_f0_val is not None:
                if cmd.tcode_tags is None:
                    cmd.tcode_tags = {}
                cmd.tcode_tags['F0'] = self._cached_f0_val
            # Apply volume ramp multiplier (don't override volume - stroke mapper computed it)
            if self._volume_ramp_active:
                elapsed = time.time() - self._volume_ramp_start_time
                progress = min(1.0, elapsed / self._volume_ramp_duration)
                ramp_mult = self._volume_ramp_from + (self._volume_ramp_to - self._volume_ramp_from) * progress
                cmd.volume = cmd.volume * ramp_mult
            self.network_engine.send_command(cmd)
    
    def _stop_engines(self):
        """Stop all engines and background threads"""
        self.is_running = False

        # Stop stroke mapper arc thread if running
        if self.stroke_mapper and hasattr(self.stroke_mapper, '_arc_thread'):
            arc_thread = getattr(self.stroke_mapper, '_arc_thread', None)
            if arc_thread and arc_thread.is_alive():
                self.stroke_mapper._stop_arc = True
                arc_thread.join(timeout=1.0)
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
                # Apply volume ramp multiplier
                if self._volume_ramp_active:
                    elapsed = time.time() - self._volume_ramp_start_time
                    progress = min(1.0, elapsed / self._volume_ramp_duration)
                    ramp_mult = self._volume_ramp_from + (self._volume_ramp_to - self._volume_ramp_from) * progress
                    cmd.volume = cmd.volume * ramp_mult
                self.network_engine.send_command(cmd)
        elif event.is_beat and not self.is_sending:
            print("[Main] Beat detected but Play not enabled")
    
    def _extract_dominant_freq(self, spectrum: np.ndarray, sample_rate: int,
                               freq_low: float, freq_high: float) -> float:
        """Extract dominant frequency from a specific Hz range of the spectrum. Thread-safe."""
        if spectrum is None or len(spectrum) == 0:
            return 0.0
        freq_per_bin = sample_rate / (2 * len(spectrum))
        low_bin = max(0, int(freq_low / freq_per_bin))
        high_bin = min(len(spectrum) - 1, int(freq_high / freq_per_bin))
        if low_bin >= high_bin:
            return 0.0
        band = spectrum[low_bin:high_bin + 1]
        peak_bin = low_bin + int(np.argmax(band))
        return peak_bin * freq_per_bin
    
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
        
        # --- P0 (Pulse Frequency) ---
        p0_enabled = self._cached_p0_enabled
        if p0_enabled:
            pulse_mode = self._cached_pulse_mode
            pulse_invert = self._cached_pulse_invert
            freq_weight = self.config.pulse_freq.freq_weight
            
            if pulse_mode == 0:  # Hz mode
                in_low = self.config.pulse_freq.monitor_freq_min
                in_high = self.config.pulse_freq.monitor_freq_max
                norm = (p0_dom_freq - in_low) / max(1.0, in_high - in_low)
            else:  # Speed mode
                norm = min(1.0, dot_speed / 10.0)
            
            norm = max(0.0, min(1.0, norm))
            norm_weighted = 0.5 + (norm - 0.5) * freq_weight
            norm_weighted = max(0.0, min(1.0, norm_weighted))
            
            if pulse_invert:
                norm_weighted = 1.0 - norm_weighted
            
            # Map dominant frequency to TCode output range (Hz-based, matching Pulse display)
            # Sent Freq sliders are in Hz (0-150), convert to TCode 0-9999 via *67
            tcode_min_val = int(self._cached_tcode_freq_min * 67)
            tcode_max_val = int(self._cached_tcode_freq_max * 67)
            tcode_min_val = max(0, min(9999, tcode_min_val))
            tcode_max_val = max(0, min(9999, tcode_max_val))
            p0_val = int(tcode_min_val + norm_weighted * (tcode_max_val - tcode_min_val))
            p0_val = max(0, min(9999, p0_val))
            cmd.pulse_freq = p0_val
            self._cached_p0_val = p0_val
            display_freq = p0_val / 67  # Convert TCode to approx Hz (1.5-150hz range)
            self._cached_pulse_display = f"Pulse: {display_freq:.0f}hz"
        else:
            cmd.pulse_freq = None
            self._cached_p0_val = None
            self._cached_pulse_display = "Pulse: off"
        
        # --- F0 (Carrier Frequency) ---
        f0_enabled = self._cached_f0_enabled
        if f0_enabled:
            f0_mode = self._cached_f0_mode
            f0_invert = self._cached_f0_invert
            f0_weight = self.config.carrier_freq.freq_weight
            
            if f0_mode == 0:  # Hz mode
                f0_in_low = self.config.carrier_freq.monitor_freq_min
                f0_in_high = self.config.carrier_freq.monitor_freq_max
                f0_norm = (f0_dom_freq - f0_in_low) / max(1.0, f0_in_high - f0_in_low)
            else:  # Speed mode
                f0_norm = min(1.0, dot_speed / 10.0)
            
            f0_norm = max(0.0, min(1.0, f0_norm))
            f0_norm_weighted = 0.5 + (f0_norm - 0.5) * f0_weight
            f0_norm_weighted = max(0.0, min(1.0, f0_norm_weighted))
            
            if f0_invert:
                f0_norm_weighted = 1.0 - f0_norm_weighted
            
            # Map dominant frequency to TCode output range
            # Sent Freq sliders are 500-1500 display units, convert to TCode 0-9999: tcode = (slider - 500) * 10
            f0_tcode_min = int((self._cached_f0_tcode_min - 500) * 10)
            f0_tcode_max = int((self._cached_f0_tcode_max - 500) * 10)
            f0_tcode_min = max(0, min(9999, f0_tcode_min))
            f0_tcode_max = max(0, min(9999, f0_tcode_max))
            f0_val = int(f0_tcode_min + f0_norm_weighted * (f0_tcode_max - f0_tcode_min))
            f0_val = max(0, min(9999, f0_val))
            
            if cmd.tcode_tags is None:
                cmd.tcode_tags = {}
            cmd.tcode_tags['F0'] = f0_val
            self._cached_f0_val = f0_val
            display_f0 = f0_val / 10 + 500  # Convert TCode to display units (500-1500)
            self._cached_carrier_display = f"Carrier: {display_f0:.0f}"
        else:
            self._cached_f0_val = None
            self._cached_carrier_display = "Carrier: off"
        
        # Log
        p0_str = f"P0={cmd.pulse_freq:04d}" if cmd.pulse_freq is not None else "P0=off"
        f0_tag = cmd.tcode_tags.get('F0', None) if cmd.tcode_tags else None
        f0_str = f"F0={f0_tag:04d}" if f0_tag is not None else "F0=off"
        print(f"[Main] Cmd: a={cmd.alpha:.2f} b={cmd.beta:.2f} v={cmd.volume:.2f} {p0_str} {f0_str}")
    
    def _network_status_callback(self, message: str, connected: bool):
        """Called from network thread on status change"""
        self.signals.status_changed.emit(message, connected)
    
    def _on_beat(self, event: BeatEvent):
        """Handle beat event in GUI thread"""
        if event.is_beat:
            # Track beat time for auto-adjustment feature
            self._last_beat_time_for_auto = time.time()
            
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
                    is_downbeat = tempo_info.get('is_downbeat', False)
                    stability = tempo_info.get('stability', 0.0)
                    
                    # Light up downbeat indicator (cyan/blue for downbeat)
                    if is_downbeat:
                        if hasattr(self, 'downbeat_indicator') and self.downbeat_indicator is not None:
                            self.downbeat_indicator.setStyleSheet("color: #0ff; font-size: 24px;")
                        if hasattr(self, 'downbeat_timer') and self.downbeat_timer is not None:
                            self.downbeat_timer.stop()
                            self.downbeat_timer.start(self.beat_indicator_min_duration)
                    
                    # Format BPM display - simple, no beat position counter
                    bpm_display = f"BPM: {tempo_info['bpm']:.1f}"
                    # Add confidence/stability indicator
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
        """Actually update spectrum at throttled rate"""
        if self._pending_spectrum is not None and hasattr(self, 'spectrum_canvas') and self.spectrum_canvas is not None:
            # Handle both old format (numpy array) and new format (dict with stats)
            if isinstance(self._pending_spectrum, dict):
                spectrum = self._pending_spectrum['spectrum']
                peak = self._pending_spectrum.get('peak_energy', 0)
                flux = self._pending_spectrum.get('spectral_flux', 0)
                # Update all visualizers (only visible one renders)
                self.spectrum_canvas.update_spectrum(spectrum, peak, flux)
                if hasattr(self, 'mountain_canvas') and self.mountain_canvas is not None:
                    self.mountain_canvas.update_spectrum(spectrum, peak, flux)
                if hasattr(self, 'bar_canvas') and self.bar_canvas is not None:
                    self.bar_canvas.update_spectrum(spectrum, peak, flux)
                if hasattr(self, 'phosphor_canvas') and self.phosphor_canvas is not None:
                    self.phosphor_canvas.update_spectrum(spectrum, peak, flux)
            else:
                self.spectrum_canvas.update_spectrum(self._pending_spectrum)
                if hasattr(self, 'mountain_canvas') and self.mountain_canvas is not None:
                    self.mountain_canvas.update_spectrum(self._pending_spectrum)
                if hasattr(self, 'bar_canvas') and self.bar_canvas is not None:
                    self.bar_canvas.update_spectrum(self._pending_spectrum)
                if hasattr(self, 'phosphor_canvas') and self.phosphor_canvas is not None:
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
        # and update freq display labels — throttled to 100ms
        now = time.time()
        if now - self._last_freq_display_time > 0.1:
            self._last_freq_display_time = now
            # Update freq display labels from cached strings (written by audio thread)
            self.pulse_freq_label.setText(self._cached_pulse_display)
            self.carrier_freq_label.setText(self._cached_carrier_display)
            # Sync checkbox/combo states to cached vars for audio thread
            new_p0_enabled = self.pulse_enabled_checkbox.isChecked()
            new_f0_enabled = self.f0_enabled_checkbox.isChecked()
            
            # Send 0000 once when P0/F0 checkboxes are unchecked (enabled→disabled transition)
            if self._prev_p0_enabled and not new_p0_enabled:
                # P0 just got disabled — send P00000 once
                if self.network_engine and self.is_sending:
                    zero_cmd = TCodeCommand(0.0, 0.0, 100, 0.0)
                    zero_cmd.pulse_freq = 0
                    self.network_engine.send_command(zero_cmd)
                    print("[Main] P0 disabled — sent P00000")
                self._cached_p0_val = None
                self._cached_pulse_display = "Pulse: off"
            if self._prev_f0_enabled and not new_f0_enabled:
                # F0 just got disabled — send F00000 once
                if self.network_engine and self.is_sending:
                    zero_cmd = TCodeCommand(0.0, 0.0, 100, 0.0)
                    zero_cmd.tcode_tags = {'F0': 0}
                    self.network_engine.send_command(zero_cmd)
                    print("[Main] F0 disabled — sent F00000")
                self._cached_f0_val = None
                self._cached_carrier_display = "Carrier: off"
            
            self._prev_p0_enabled = new_p0_enabled
            self._prev_f0_enabled = new_f0_enabled
            self._cached_p0_enabled = new_p0_enabled
            self._cached_f0_enabled = new_f0_enabled
            self._cached_pulse_mode = self.pulse_mode_combo.currentIndex()
            self._cached_pulse_invert = self.pulse_invert_checkbox.isChecked()
            self._cached_f0_mode = self.f0_mode_combo.currentIndex()
            self._cached_f0_invert = self.f0_invert_checkbox.isChecked()
            # Sync TCode Sent Freq slider values for thread-safe access
            self._cached_tcode_freq_min = self.tcode_freq_range_slider.low()
            self._cached_tcode_freq_max = self.tcode_freq_range_slider.high()
            self._cached_f0_tcode_min = self.f0_tcode_range_slider.low()
            self._cached_f0_tcode_max = self.f0_tcode_range_slider.high()

        # Handle volume ramp completion
        if self._volume_ramp_active:
            elapsed = time.time() - self._volume_ramp_start_time
            if elapsed >= self._volume_ramp_duration:
                self._volume_ramp_active = False
    
    def _log_experimental_spinbox_shutdown_values(self):
        """Log final experimental spinbox values at shutdown for documentation"""
        print("\n" + "="*70)
        print("EXPERIMENTAL SPINBOX SHUTDOWN VALUES")
        print("="*70)
        
        print("\nParameter        │ Step Size  │ Lock Time (ms)")
        print("-" * 55)
        params = [
            ("sensitivity", self.sensitivity_step_spin, self.sensitivity_lock_spin),
            ("peak_floor", self.peak_floor_step_spin, self.peak_floor_lock_spin),
            ("peak_decay", self.peak_decay_step_spin, self.peak_decay_lock_spin),
            ("rise_sens", self.rise_sens_step_spin, self.rise_sens_lock_spin),
            ("flux_mult", self.flux_mult_step_spin, self.flux_mult_lock_spin),
            ("audio_amp", self.audio_amp_step_spin, self.audio_amp_lock_spin),
        ]
        
        for param_name, step_spin, lock_spin in params:
            step_val = step_spin.value()
            lock_val = lock_spin.value()
            print(f"{param_name:16} │ {step_val:.3f}     │ {lock_val:.0f}")
        
        print("-" * 55)
        consec_lock_val = self.auto_consec_lock_spin.value()
        print(f"Consecutive-lock timer: {consec_lock_val:.1f} seconds")
        print(f"Oscillation rule: 3/4 of step size (automatic)")
        print("="*70 + "\n")

    def closeEvent(self, event):
        """Cleanup on close - ensure all threads are stopped before UI is destroyed"""
        self._stop_engines()
        if self.network_engine:
            self.network_engine.stop()

        # Log experimental spinbox shutdown values
        self._log_experimental_spinbox_shutdown_values()

        # Save all settings from sliders to config before closing
        self.config.stroke.phase_advance = self.phase_advance_slider.value()
        self.config.pulse_freq.monitor_freq_min = self.pulse_freq_range_slider.low()
        self.config.pulse_freq.monitor_freq_max = self.pulse_freq_range_slider.high()
        self.config.pulse_freq.tcode_freq_min = self.tcode_freq_range_slider.low()
        self.config.pulse_freq.tcode_freq_max = self.tcode_freq_range_slider.high()
        self.config.pulse_freq.freq_weight = self.freq_weight_slider.value()
        
        # Save carrier freq (F0) settings
        self.config.carrier_freq.monitor_freq_min = self.f0_freq_range_slider.low()
        self.config.carrier_freq.monitor_freq_max = self.f0_freq_range_slider.high()
        self.config.carrier_freq.tcode_freq_min = self.f0_tcode_range_slider.low()
        self.config.carrier_freq.tcode_freq_max = self.f0_tcode_range_slider.high()
        self.config.carrier_freq.freq_weight = self.f0_weight_slider.value()
        
        self.config.volume = self.volume_slider.value()
        
        # Save tempo tracking settings
        self.config.beat.tempo_tracking_enabled = self.tempo_tracking_checkbox.isChecked()
        beats_map = {0: 4, 1: 3, 2: 6}
        self.config.beat.beats_per_measure = beats_map.get(self.time_sig_combo.currentIndex(), 4)
        self.config.beat.stability_threshold = self.stability_threshold_slider.value()
        self.config.beat.tempo_timeout_ms = int(self.tempo_timeout_slider.value())
        self.config.beat.phase_snap_weight = self.phase_snap_slider.value()
        
        # Save config before closing
        save_config(self.config)

        # Save presets to disk
        self._save_presets_to_disk()

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = BREadbeatsWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
