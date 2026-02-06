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
    QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPalette, QColor
from typing import Optional

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


class SpectrumCanvas(FigureCanvas):
    """Spectrum visualizer with interactive frequency band and peak/flux indicators"""
    
    def __init__(self, parent=None, width=8, height=3):
        # Dark theme matching restim-coyote3
        plt.style.use('dark_background')
        
        self.fig = Figure(figsize=(width, height), facecolor='#2d2d2d')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#232323')
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Pre-create bar plot for efficiency
        self.num_bars = 64
        self.x = np.arange(self.num_bars)
        self.bars = None
        self.band_overlay = None  # Frequency band selection overlay
        self.band_low = 0.0
        self.band_high = 0.1  # Default bass range
        
        # Peak and flux indicator lines
        self.peak_line = None
        self.flux_line = None
        self.peak_value = 0.0
        self.flux_value = 0.0
        
        # Interactive band dragging state
        self.drag_mode = None  # 'left', 'right', or 'move'
        self.drag_start_x = None
        self.band_width = None
        
        # Reference to main window for slider updates
        self.parent_window = parent
        
        # Mouse event handlers
        self.mpl_connect('button_press_event', self._on_press)
        self.mpl_connect('button_release_event', self._on_release)
        self.mpl_connect('motion_notify_event', self._on_motion)
        
        self.setup_plot()
        
    def setup_plot(self):
        """Initialize the spectrum plot"""
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.num_bars - 0.5)
        self.ax.set_ylim(0, 1.1)
        self.ax.set_xlabel('')
        self.ax.set_ylabel('Amplitude', fontsize=8, color='#999')
        self.ax.tick_params(axis='x', colors='#aaa', labelsize=7, which='both', length=0)
        self.ax.tick_params(axis='y', colors='#aaa', labelsize=7)
        self.ax.set_xticklabels([])
        self.ax.grid(axis='y', color='#444', alpha=0.3, linewidth=0.5)
        self.ax.spines['bottom'].set_color('#555')
        self.ax.spines['left'].set_color('#555')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        # Create initial bars (blue for unselected range)
        self.bars = self.ax.bar(self.x, np.zeros(self.num_bars), 
                                 color='#6688ff', alpha=0.8, width=0.8)
        
        # Create frequency band overlay (grey thin bar at top, no border)
        from matplotlib.patches import Rectangle
        low_bar = int(self.band_low * self.num_bars)
        high_bar = int(self.band_high * self.num_bars)
        self.band_overlay = Rectangle((low_bar - 0.5, 0.9), high_bar - low_bar + 1, 0.2,
                                        facecolor='#555555', alpha=0.5, edgecolor='none')
        self.ax.add_patch(self.band_overlay)
        
        # Create peak and flux threshold indicator lines
        self.peak_line = self.ax.axhline(y=0.0, color='#ff6644', linestyle='--', linewidth=1.5, alpha=0.6, label='Peak')
        self.flux_line = self.ax.axhline(y=0.0, color='#44ff66', linestyle=':', linewidth=1.5, alpha=0.6, label='Flux')
        self.ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
        
        self.fig.tight_layout(pad=0.5)
    
    def set_frequency_band(self, low_norm: float, high_norm: float):
        """Update frequency band overlay position (normalized 0-1)"""
        self.band_low = low_norm
        self.band_high = high_norm
        
        if self.band_overlay:
            low_bar = int(low_norm * self.num_bars)
            high_bar = int(high_norm * self.num_bars)
            width = max(1, high_bar - low_bar + 1)
            self.band_overlay.set_x(low_bar - 0.5)
            self.band_overlay.set_width(width)
    
    def set_peak_and_flux(self, peak_value: float, flux_value: float):
        """Update peak and flux indicator lines"""
        self.peak_value = np.clip(peak_value, 0, 1.1)
        self.flux_value = np.clip(flux_value, 0, 1.1)
        
        if self.peak_line:
            self.peak_line.set_ydata([self.peak_value, self.peak_value])
        if self.flux_line:
            self.flux_line.set_ydata([self.flux_value, self.flux_value])
        
    def _on_press(self, event):
        """Handle mouse press for interactive band adjustment"""
        if event.inaxes != self.ax or event.xdata is None:
            return
        
        low_bar = int(self.band_low * self.num_bars)
        high_bar = int(self.band_high * self.num_bars)
        x = event.xdata
        
        # Check if clicking on left edge (within 0.5 bars)
        if abs(x - low_bar) < 0.5:
            self.drag_mode = 'left'
            self.drag_start_x = x
        # Check if clicking on right edge (within 0.5 bars)
        elif abs(x - high_bar) < 0.5:
            self.drag_mode = 'right'
            self.drag_start_x = x
        # Check if clicking inside band (move entire band)
        elif low_bar <= x <= high_bar:
            self.drag_mode = 'move'
            self.drag_start_x = x
            self.band_width = high_bar - low_bar
    
    def _on_release(self, event):
        """Handle mouse release"""
        self.drag_mode = None
        self.drag_start_x = None
        self.band_width = None
    
    def _on_motion(self, event):
        """Handle mouse motion for dragging band"""
        if self.drag_mode is None or event.xdata is None:
            return
        
        x = event.xdata
        dx = x - self.drag_start_x
        
        # Only process if there's meaningful movement (at least 0.1 bar width)
        if abs(dx) < 0.1:
            return
        
        low_bar = int(self.band_low * self.num_bars)
        high_bar = int(self.band_high * self.num_bars)
        
        if self.drag_mode == 'left':
            # Dragging left edge - don't cross right edge
            new_low = max(0, min(high_bar - 1, low_bar + dx))
            self.band_low = new_low / self.num_bars
            
        elif self.drag_mode == 'right':
            # Dragging right edge - don't cross left edge
            new_high = min(self.num_bars - 1, max(low_bar + 1, high_bar + dx))
            self.band_high = new_high / self.num_bars
            
        elif self.drag_mode == 'move':
            # Moving entire band - keep width constant
            if self.band_width is not None:
                new_low = max(0, min(self.num_bars - self.band_width - 1, low_bar + int(dx)))
                new_high = new_low + self.band_width
                self.band_low = new_low / self.num_bars
                self.band_high = min(1.0, new_high / self.num_bars)
        
        # Update overlay
        self.set_frequency_band(self.band_low, self.band_high)
        
        # Update sliders in parent window if available
        if self.parent_window and hasattr(self.parent_window, 'freq_low_slider') and hasattr(self.parent_window, 'audio_engine'):
            sr = self.parent_window.audio_engine.config.audio.sample_rate if self.parent_window.audio_engine else 44100
            low_hz = self.band_low * sr / 2
            high_hz = self.band_high * sr / 2
            self.parent_window.freq_low_slider.setValue(int(low_hz))
            self.parent_window.freq_high_slider.setValue(int(high_hz))
        
        # Update drag start for next incremental motion
        self.drag_start_x = x
        self.draw_idle()
        
    def update_spectrum(self, spectrum: np.ndarray, peak_energy: Optional[float] = None, spectral_flux: Optional[float] = None):
        """Update with new spectrum data - efficient bar update"""
        if spectrum is None or len(spectrum) == 0 or self.bars is None:
            return
            
        # Downsample to bar count
        if len(spectrum) > self.num_bars:
            factor = len(spectrum) // self.num_bars
            spectrum = spectrum[:factor * self.num_bars].reshape(-1, factor).mean(axis=1)
        elif len(spectrum) < self.num_bars:
            spectrum = np.pad(spectrum, (0, self.num_bars - len(spectrum)))
        
        # Normalize
        max_val = np.max(spectrum) if np.max(spectrum) > 0 else 1
        spectrum = np.clip(spectrum / max_val, 0, 1)
        
        # Update peak and flux indicators if provided
        if peak_energy is not None and spectral_flux is not None:
            # Normalize to 0-1 range for visualization
            peak_norm = np.clip(peak_energy / max(1.0, max_val), 0, 1)
            flux_norm = np.clip(spectral_flux / 10.0, 0, 1)  # Flux typically 0-10 range
            self.set_peak_and_flux(peak_norm, flux_norm)
        
        # Update bar heights and colors
        low_bar = int(self.band_low * self.num_bars)
        high_bar = int(self.band_high * self.num_bars)
        
        try:
            for i, (bar, h) in enumerate(zip(self.bars, spectrum)):
                bar.set_height(h)
                # Highlight bars in the selected frequency band
                if low_bar <= i <= high_bar:
                    bar.set_color('#00ffaa')  # Cyan-green for selected band
                else:
                    bar.set_color('#00aaff')  # Blue for unselected
            
            self.draw_idle()
        except Exception:
            # Silently ignore matplotlib rendering errors
            pass


class PositionCanvas(FigureCanvas):
    """Alpha/Beta position visualizer - circular display"""
    
    def __init__(self, parent=None, size=2, get_rotation=None):
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(size, size), facecolor='#2d2d2d')
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.ax.set_facecolor('#232323')
        super().__init__(self.fig)
        self.setParent(parent)
        self.trail_x = []
        self.trail_y = []
        self.max_trail = 50
        self.trail_lines = []
        self.position_scatter = None
        self.get_rotation = get_rotation  # function to get rotation angle in degrees
        self.setup_plot()

    def setup_plot(self):
        """Initialize the position plot (clean, no grid/labels)"""
        self.ax.clear()
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        theta = np.linspace(0, 2*np.pi, 100)
        self.ax.plot(np.cos(theta), np.sin(theta), color='#666666', alpha=0.5, linewidth=1)
        self.ax.axhline(y=0, color='#555555', linewidth=0.5)
        self.ax.axvline(x=0, color='#555555', linewidth=0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.fig.tight_layout(pad=0.3)
        self.position_scatter = self.ax.scatter([], [], c='#00ffff', s=80, zorder=5)

    def update_position(self, alpha: float, beta: float):
        # Alpha = vertical (y-axis): 1.0 = top, -1.0 = bottom
        # Beta = horizontal (x-axis): 1.0 = LEFT, -1.0 = right (matches restim orientation)
        # Apply rotation if available
        angle_deg = self.get_rotation() if self.get_rotation else 0.0
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        # Map: x = -beta (negated to match restim), y = alpha (vertical), then rotate
        x_base = -beta  # Negated to match restim (positive beta = left)
        y_base = alpha
        x_rot = x_base * cos_a - y_base * sin_a
        y_rot = x_base * sin_a + y_base * cos_a
        self.trail_x.append(x_rot)
        self.trail_y.append(y_rot)
        if len(self.trail_x) > self.max_trail:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
        try:
            while len(self.trail_lines) > self.max_trail - 1:
                line = self.trail_lines.pop(0)
                line.remove()
            if len(self.trail_x) > 1:
                i = len(self.trail_x)
                alpha_val = i / len(self.trail_x)
                line, = self.ax.plot([self.trail_x[-2], self.trail_x[-1]],
                                     [self.trail_y[-2], self.trail_y[-1]],
                                     color='#00aaff', alpha=alpha_val * 0.5, linewidth=1)
                self.trail_lines.append(line)
            if self.position_scatter:
                self.position_scatter.set_offsets([[x_rot, y_rot]])
            self.draw_idle()
        except Exception:
            pass


class PresetButton(QPushButton):
    """Custom button that emits different signals for left-click (load) vs right-click (save)"""
    
    left_clicked = pyqtSignal()
    right_clicked = pyqtSignal()
    
    def __init__(self, label: str):
        super().__init__(label)
        self.setMinimumWidth(40)
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
        
        # Presets panel - always visible below tabs
        main_layout.addWidget(self._create_presets_panel())
    
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
            self.freq_low_slider.setValue(self.config.beat.freq_low)
            self.freq_high_slider.setValue(self.config.beat.freq_high)
            self._on_freq_band_change()  # Update spectrum overlay
            
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
            self.stroke_min_slider.setValue(self.config.stroke.stroke_min)
            self.stroke_max_slider.setValue(self.config.stroke.stroke_max)
            self.min_interval_slider.setValue(self.config.stroke.min_interval_ms)
            self.fullness_slider.setValue(self.config.stroke.stroke_fullness)
            self.min_depth_slider.setValue(self.config.stroke.minimum_depth)
            self.freq_depth_slider.setValue(self.config.stroke.freq_depth_factor)
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
            self.pulse_freq_low_slider.setValue(self.config.pulse_freq.monitor_freq_min)
            self.pulse_freq_high_slider.setValue(self.config.pulse_freq.monitor_freq_max)
            self.tcode_freq_min_slider.setValue(self.config.pulse_freq.tcode_freq_min)
            self.tcode_freq_max_slider.setValue(self.config.pulse_freq.tcode_freq_max)
            self.freq_weight_slider.setValue(self.config.pulse_freq.freq_weight)
            
            # Volume
            self.volume_slider.setValue(self.config.volume)
            
            print("[UI] Loaded all settings from config")
        except AttributeError as e:
            print(f"[UI] Warning: Could not apply all config values: {e}")
        
    def _create_connection_panel(self) -> QGroupBox:
        """Connection settings panel"""
        group = QGroupBox("TCP Connection")
        layout = QGridLayout(group)
        
        # Host/Port
        layout.addWidget(QLabel("Host:"), 0, 0)
        self.host_edit = QLineEdit(self.config.connection.host)
        layout.addWidget(self.host_edit, 0, 1)
        
        layout.addWidget(QLabel("Port:"), 0, 2)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(self.config.connection.port)
        self.port_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)  # Remove scroll/arrows
        layout.addWidget(self.port_spin, 0, 3)
        
        # Status
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: #f55;")
        layout.addWidget(self.status_label, 1, 0, 1, 2)
        
        # Reset/Test buttons
        self.connect_btn = QPushButton("Reset")
        self.connect_btn.clicked.connect(self._on_connect)
        layout.addWidget(self.connect_btn, 1, 2)
        
        self.test_btn = QPushButton("Test")
        self.test_btn.clicked.connect(self._on_test)
        self.test_btn.setEnabled(False)
        layout.addWidget(self.test_btn, 1, 3)
        
        return group
    
    def _create_control_panel(self) -> QGroupBox:
        """Main control buttons"""
        group = QGroupBox("Controls")
        layout = QVBoxLayout(group)
        
        # Audio device selector (first row)
        device_layout = QGridLayout()
        device_layout.setSpacing(5)
        
        device_layout.addWidget(QLabel("Audio Device:"), 0, 0)
        self.device_combo = QComboBox()
        self._populate_audio_devices()
        self.device_combo.setMinimumWidth(300)
        self.device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        device_layout.addWidget(self.device_combo, 0, 1, 1, 3)
        
        # Quick presets for common devices (second row)
        self.preset_mic_btn = QPushButton("🎤 Mic (Reactive)")
        self.preset_mic_btn.setFixedWidth(130)
        self.preset_mic_btn.clicked.connect(self._set_device_preset_mic)
        device_layout.addWidget(self.preset_mic_btn, 1, 1)
        
        self.preset_loopback_btn = QPushButton("🔊 System Audio")
        self.preset_loopback_btn.setFixedWidth(130)
        self.preset_loopback_btn.clicked.connect(self._set_device_preset_loopback)
        device_layout.addWidget(self.preset_loopback_btn, 1, 2)
        
        # Connect device changes to update button states
        self.device_combo.currentIndexChanged.connect(self._update_preset_button_states)
        
        device_layout.setColumnStretch(3, 1)
        layout.addLayout(device_layout)
        
        # Controls row - split into two rows for better layout
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

        # Pulse Freq display (shows sent TCode value x2)
        self.pulse_freq_label = QLabel("Pulse: --")
        self.pulse_freq_label.setStyleSheet("color: #f80; font-size: 12px; font-weight: bold;")
        self.pulse_freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pulse_freq_label.setFixedWidth(80)
        btn_layout.addWidget(self.pulse_freq_label, 0, 4)

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
        
        # Find WASAPI host API index
        wasapi_idx = None
        for idx, api in enumerate(hostapis):
            if 'WASAPI' in api['name']:
                wasapi_idx = idx
                break
        
        self.device_combo.clear()
        self.audio_device_map = {}  # Map combo index to device index
        self.audio_device_is_loopback = {}  # Track which devices should use WASAPI loopback
        
        loopback_keywords = ['stereo mix', 'what u hear', 'loopback', 'wave out mix', 'system audio']
        loopback_idx = None
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
                    
                    name = f"{clean_name} [WASAPI Loopback]"
                    self.device_combo.addItem(name)
                    self.audio_device_map[combo_idx] = i
                    self.audio_device_is_loopback[combo_idx] = True
                    
                    # Prefer WASAPI loopback for default
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
        
        # Pre-select Stereo Mix/loopback if available, otherwise first device
        if loopback_idx is not None:
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
        group = QGroupBox("Spectrum Analyzer")
        layout = QVBoxLayout(group)
        
        self.spectrum_canvas = SpectrumCanvas(self, width=8, height=3)
        layout.addWidget(self.spectrum_canvas)
        
        return group
    
    def _create_position_panel(self) -> QGroupBox:
        """Alpha/Beta position display with rotation slider"""
        group = QGroupBox("Position (α/β)")
        layout = QVBoxLayout(group)

        # Rotation slider (-180 to 180, default 0 for correct axis mapping)
        rot_layout = QHBoxLayout()
        rot_label = QLabel("Rotation (°):")
        rot_label.setStyleSheet("color: #aaa;")
        self.position_rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_rotation_slider.setMinimum(-180)
        self.position_rotation_slider.setMaximum(180)
        self.position_rotation_slider.setValue(0)
        self.position_rotation_slider.setFixedWidth(180)
        self.position_rotation_slider.setSingleStep(1)
        self.position_rotation_slider.setPageStep(10)
        self.position_rotation_slider.setTickInterval(30)
        self.position_rotation_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.position_rotation_slider.valueChanged.connect(lambda _: self.position_canvas.setup_plot())
        self.position_rotation_value = QLabel("0°")
        self.position_rotation_value.setStyleSheet("color: #0af;")
        self.position_rotation_slider.valueChanged.connect(
            lambda v: self.position_rotation_value.setText(f"{v}°"))
        rot_layout.addWidget(rot_label)
        rot_layout.addWidget(self.position_rotation_slider)
        rot_layout.addWidget(self.position_rotation_value)
        layout.addLayout(rot_layout)

        # Provide a function to get the current rotation value
        self.position_canvas = PositionCanvas(self, size=2, get_rotation=lambda: self.position_rotation_slider.value())
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
        tabs.addTab(self._create_jitter_creep_tab(), "Jitter / Creep")
        tabs.addTab(self._create_tempo_tracking_tab(), "Tempo Tracking")
        tabs.addTab(self._create_other_tab(), "Other")
        return tabs
    
    def _create_presets_panel(self) -> QGroupBox:
        """Presets panel - always displayed below all tabs"""
        group = QGroupBox("Presets (L=Load, R=Save)")
        layout = QHBoxLayout(group)
        
        self.custom_beat_presets = {}
        self.preset_buttons = []
        for i in range(5):
            btn = PresetButton(f"{i+1}")
            btn.left_clicked.connect(lambda idx=i: self._load_beat_preset(idx))
            btn.right_clicked.connect(lambda idx=i: self._save_beat_preset(idx))
            self.preset_buttons.append(btn)
            layout.addWidget(btn)
        
        layout.addStretch()
        return group

    def _create_other_tab(self) -> QWidget:
        """Other tab for dominant frequency to TCode P0xxxx and performance settings"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        freq_group = QGroupBox("Pulse Frequency Controls")
        freq_layout = QVBoxLayout(freq_group)

        self.pulse_freq_low_slider = SliderWithLabel("Monitor Freq Min (Hz)", 20, 1000, 20, 0)
        self.pulse_freq_high_slider = SliderWithLabel("Monitor Freq Max (Hz)", 20, 1000, 200, 0)
        freq_layout.addWidget(self.pulse_freq_low_slider)
        freq_layout.addWidget(self.pulse_freq_high_slider)

        # TCode output range sliders (in Hz, same scale as Pulse display: Hz = TCode/67)
        # TCode range 0-9999 maps to ~0-150Hz via /67
        self.tcode_freq_min_slider = SliderWithLabel("Sent Freq Min (Hz)", 0, 150, 30, 0)
        self.tcode_freq_max_slider = SliderWithLabel("Sent Freq Max (Hz)", 0, 150, 105, 0)
        freq_layout.addWidget(self.tcode_freq_min_slider)
        freq_layout.addWidget(self.tcode_freq_max_slider)

        # Optional: Frequency weight slider (not yet wired)
        self.freq_weight_slider = SliderWithLabel("Frequency Weight", 0.0, 1.0, 1.0, 2)
        freq_layout.addWidget(self.freq_weight_slider)

        # Display for current dominant frequency
        freq_display_layout = QHBoxLayout()
        freq_display_layout.addWidget(QLabel("Dominant Frequency:"))
        self.dominant_freq_label = QLabel("-- Hz")
        self.dominant_freq_label.setStyleSheet("color: #0af; font-size: 16px; font-weight: bold;")
        freq_display_layout.addWidget(self.dominant_freq_label)
        freq_layout.addLayout(freq_display_layout)

        layout.addWidget(freq_group)
        
        # Performance settings group
        perf_group = QGroupBox("Performance (requires restart)")
        perf_layout = QVBoxLayout(perf_group)
        
        # FFT Size dropdown
        fft_layout = QHBoxLayout()
        fft_layout.addWidget(QLabel("FFT Size:"))
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(["512 (fast, low res)", "1024 (balanced)", "2048 (slow, high res)"])
        # Set current based on config
        fft_sizes = [512, 1024, 2048]
        current_fft = getattr(self.config.audio, 'fft_size', 1024)
        if current_fft in fft_sizes:
            self.fft_size_combo.setCurrentIndex(fft_sizes.index(current_fft))
        else:
            self.fft_size_combo.setCurrentIndex(1)  # Default to 1024
        self.fft_size_combo.currentIndexChanged.connect(self._on_fft_size_change)
        fft_layout.addWidget(self.fft_size_combo)
        fft_layout.addStretch()
        perf_layout.addLayout(fft_layout)
        
        # Spectrum update rate dropdown
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Spectrum Updates:"))
        self.spectrum_skip_combo = QComboBox()
        self.spectrum_skip_combo.addItems(["Every frame (smooth)", "Every 2 frames (fast)", "Every 4 frames (faster)"])
        current_skip = getattr(self.config.audio, 'spectrum_skip_frames', 2)
        skip_map = {1: 0, 2: 1, 4: 2}
        self.spectrum_skip_combo.setCurrentIndex(skip_map.get(current_skip, 1))
        self.spectrum_skip_combo.currentIndexChanged.connect(self._on_spectrum_skip_change)
        skip_layout.addWidget(self.spectrum_skip_combo)
        skip_layout.addStretch()
        perf_layout.addLayout(skip_layout)
        
        layout.addWidget(perf_group)
        layout.addStretch()
        return widget
    
    def _on_fft_size_change(self, index: int):
        """Update FFT size setting (requires restart to take effect)"""
        fft_sizes = [512, 1024, 2048]
        self.config.audio.fft_size = fft_sizes[index]
        print(f"[Config] FFT size changed to {fft_sizes[index]} (restart required)")
    
    def _on_spectrum_skip_change(self, index: int):
        """Update spectrum skip frames (takes effect immediately if engine running)"""
        skip_values = [1, 2, 4]
        self.config.audio.spectrum_skip_frames = skip_values[index]
        if self.audio_engine:
            self.audio_engine._spectrum_skip_frames = skip_values[index]
        print(f"[Config] Spectrum skip frames changed to {skip_values[index]}")
    
    def _create_beat_detection_tab(self) -> QWidget:
        """Beat detection settings"""
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
        layout.addLayout(type_layout)
        
        # Frequency band selection
        freq_group = QGroupBox("Frequency Band (Hz) - shown as overlay on spectrum")
        freq_layout = QVBoxLayout(freq_group)
        
        # Full range up to ~20kHz (Nyquist for 44100 Hz)

        # Get sample rate (default to 44100 if not available yet)
        sr = getattr(self.config.audio, 'sample_rate', 44100)
        nyquist = sr // 2
        self.freq_low_slider = SliderWithLabel("Low Freq (Hz)", 20, nyquist, 20, 0)
        self.freq_low_slider.valueChanged.connect(self._on_freq_band_change)
        freq_layout.addWidget(self.freq_low_slider)

        self.freq_high_slider = SliderWithLabel("High Freq (Hz)", 20, nyquist, min(200, nyquist), 0)
        self.freq_high_slider.valueChanged.connect(self._on_freq_band_change)
        freq_layout.addWidget(self.freq_high_slider)
        
        layout.addWidget(freq_group)
        
        # Sliders - with better defaults
        # Sensitivity: higher = more beats detected (0.0=strict, 1.0=very sensitive)
        self.sensitivity_slider = SliderWithLabel("Sensitivity", 0.0, 1.0, 0.7)
        self.sensitivity_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'sensitivity', v))
        layout.addWidget(self.sensitivity_slider)
        
        # Peak floor: minimum energy to consider (0 = disabled)
        self.peak_floor_slider = SliderWithLabel("Peak Floor", 0.0, 0.8, 0.0, 2)
        self.peak_floor_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'peak_floor', v))
        layout.addWidget(self.peak_floor_slider)
        
        self.peak_decay_slider = SliderWithLabel("Peak Decay", 0.5, 0.999, 0.9, 3)
        self.peak_decay_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'peak_decay', v))
        layout.addWidget(self.peak_decay_slider)
        
        # Rise sensitivity: 0 = disabled, higher = require more rise
        self.rise_sens_slider = SliderWithLabel("Rise Sensitivity", 0.0, 1.0, 0.0)
        self.rise_sens_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'rise_sensitivity', v))
        layout.addWidget(self.rise_sens_slider)
        
        self.flux_mult_slider = SliderWithLabel("Flux Multiplier", 0.1, 5.0, 1.0, 1)
        self.flux_mult_slider.valueChanged.connect(lambda v: setattr(self.config.beat, 'flux_multiplier', v))
        layout.addWidget(self.flux_mult_slider)
        
        # Audio amplification/gain: boost weak signals (0.1=quiet, 10.0=loud)
        self.audio_gain_slider = SliderWithLabel("Audio Amplification", 0.1, 10.0, 5.0, 1)
        self.audio_gain_slider.valueChanged.connect(lambda v: setattr(self.config.audio, 'gain', v))
        layout.addWidget(self.audio_gain_slider)
        
        layout.addStretch()
        return widget
    
    def _on_freq_band_change(self, _=None):
        """Update frequency band in config and spectrum overlay"""
        low = self.freq_low_slider.value()
        high = self.freq_high_slider.value()
        
        # Ensure low < high
        if low >= high:
            high = low + 20
            self.freq_high_slider.setValue(high)
        
        self.config.beat.freq_low = low
        self.config.beat.freq_high = high
        
        # Update spectrum overlay
        sr = self.config.audio.sample_rate
        max_freq = sr / 2
        self.spectrum_canvas.set_frequency_band(low / max_freq, high / max_freq)
    
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
        """Save ALL settings from all 4 tabs to custom preset, with overwrite confirmation"""
        from PyQt6.QtWidgets import QMessageBox
        
        # Check if this slot already has a preset
        key = str(idx)
        if key in self.custom_beat_presets:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("WARNING - OVERWRITE PRESET")
            msg_box.setText(f"Preset {idx+1} already exists.\nAre you sure you want to overwrite it?")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            ok_button = msg_box.button(QMessageBox.StandardButton.Ok)
            if ok_button:
                ok_button.setText("Confirm")
            msg_box.setDefaultButton(QMessageBox.StandardButton.Cancel)
            result = msg_box.exec()
            if result != QMessageBox.StandardButton.Ok:
                print(f"[Config] Preset {idx+1} overwrite cancelled")
                return

        preset_data = {
            # Beat Detection Tab
            'freq_low': self.freq_low_slider.value(),
            'freq_high': self.freq_high_slider.value(),
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
            'stroke_min': self.stroke_min_slider.value(),
            'stroke_max': self.stroke_max_slider.value(),
            'min_interval_ms': int(self.min_interval_slider.value()),
            'stroke_fullness': self.fullness_slider.value(),
            'minimum_depth': self.min_depth_slider.value(),
            'freq_depth_factor': self.freq_depth_slider.value(),
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
            'pulse_freq_low': self.pulse_freq_low_slider.value(),
            'pulse_freq_high': self.pulse_freq_high_slider.value(),
            'tcode_freq_min': self.tcode_freq_min_slider.value(),
            'tcode_freq_max': self.tcode_freq_max_slider.value(),
            'freq_weight': self.freq_weight_slider.value(),
        }
        self.custom_beat_presets[str(idx)] = preset_data
        # Mark this button as having a preset and make it active
        self.preset_buttons[idx].set_has_preset(True)
        self.preset_buttons[idx].set_active(True)
        # Deactivate other preset buttons
        for i, btn in enumerate(self.preset_buttons):
            if i != idx:
                btn.set_active(False)
        self._save_presets_to_disk()
        print(f"[Config] Saved preset {idx+1} with all settings")
    
    def _load_freq_preset(self, idx: int):
        """Load ALL settings from all 4 tabs from custom preset"""
        from config import StrokeMode
        key = str(idx)
        if key in self.custom_beat_presets:
            preset_data = self.custom_beat_presets[key]
            # Beat Detection Tab
            self.freq_low_slider.setValue(preset_data['freq_low'])
            self.freq_high_slider.setValue(preset_data['freq_high'])
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
            self.stroke_min_slider.setValue(preset_data['stroke_min'])
            self.stroke_max_slider.setValue(preset_data['stroke_max'])
            self.min_interval_slider.setValue(preset_data['min_interval_ms'])
            self.fullness_slider.setValue(preset_data['stroke_fullness'])
            self.min_depth_slider.setValue(preset_data['minimum_depth'])
            self.freq_depth_slider.setValue(preset_data['freq_depth_factor'])
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
                self.pulse_freq_low_slider.setValue(preset_data['pulse_freq_low'])
            if 'pulse_freq_high' in preset_data:
                self.pulse_freq_high_slider.setValue(preset_data['pulse_freq_high'])
            if 'tcode_freq_min' in preset_data:
                self.tcode_freq_min_slider.setValue(preset_data['tcode_freq_min'])
            if 'tcode_freq_max' in preset_data:
                self.tcode_freq_max_slider.setValue(preset_data['tcode_freq_max'])
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
                # Mark buttons that have saved presets
                for idx in self.custom_beat_presets.keys():
                    idx_int = int(idx)
                    if idx_int < len(self.preset_buttons):
                        self.preset_buttons[idx_int].set_has_preset(True)
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
        self.stroke_min_slider = SliderWithLabel("Stroke Min", 0.0, 1.0, 0.2)
        self.stroke_min_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'stroke_min', v))
        layout.addWidget(self.stroke_min_slider)
        
        self.stroke_max_slider = SliderWithLabel("Stroke Max", 0.0, 1.0, 1.0)
        self.stroke_max_slider.valueChanged.connect(lambda v: setattr(self.config.stroke, 'stroke_max', v))
        layout.addWidget(self.stroke_max_slider)
        
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
        
        # Axis Weights section
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
        
        self.jitter_amplitude_slider = SliderWithLabel("Circle Size", 0.01, 0.2, 0.1, 3)
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
        
        self.creep_speed_slider = SliderWithLabel("Creep Speed", 0.0, 1.0, 0.25, 2)
        self.creep_speed_slider.valueChanged.connect(lambda v: setattr(self.config.creep, 'speed', v))
        creep_layout.addWidget(self.creep_speed_slider)
        
        layout.addWidget(creep_group)
        
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
        """Send a command directly (used by StrokeMapper for return strokes)"""
        # Always update volume before sending
        cmd.volume = self.volume_slider.value()
        if self.network_engine and self.is_sending:
            print(f"[Main] Sending cmd (direct): a={cmd.alpha:.2f} b={cmd.beta:.2f} v={cmd.volume:.2f}")
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
        """Called from audio thread on each frame"""
        # Emit signal for thread-safe GUI update
        self.signals.beat_detected.emit(event)

        # Track and display dominant frequency in 'Other' tab
        dom_freq = event.frequency if hasattr(event, 'frequency') else 0.0
        low = self.pulse_freq_low_slider.value()
        high = self.pulse_freq_high_slider.value()
        # Only update if within selected range
        if low <= dom_freq <= high:
            self.dominant_freq_label.setText(f"{dom_freq:.1f} Hz")
        else:
            self.dominant_freq_label.setText("-- Hz")

        # Get spectrum for visualization
        if self.audio_engine:
            spectrum = self.audio_engine.get_spectrum()
            if spectrum is not None:
                # Also emit peak and flux for indicator lines
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
                # Apply volume slider value, then apply silence factor if present
                base_volume = self.volume_slider.value()
                silence_factor = getattr(self.stroke_mapper, '_tcode_silence_volume_factor', 1.0)
                cmd.volume = base_volume * silence_factor
                # Use the actual dominant frequency
                dom_freq = event.frequency if hasattr(event, 'frequency') else 0.0
                # Map dominant frequency to TCode output range (Hz-based, matching Pulse display)
                # Sent Freq sliders are in Hz (0-150), convert to TCode 0-9999 via *67
                tcode_min_val = int(self.tcode_freq_min_slider.value() * 67)
                tcode_max_val = int(self.tcode_freq_max_slider.value() * 67)
                tcode_min_val = max(0, min(9999, tcode_min_val))
                tcode_max_val = max(0, min(9999, tcode_max_val))
                # Normalize dom_freq within selected input range
                in_low = self.pulse_freq_low_slider.value()
                in_high = self.pulse_freq_high_slider.value()
                norm = (dom_freq - in_low) / max(1, in_high - in_low)
                norm = max(0.0, min(1.0, norm))
                # Bias by movement speed (stroke intensity)
                # freq_weight: 1.0 = pure frequency, 0.0 = max intensity boost
                freq_weight = self.freq_weight_slider.value()  # 0.0–1.0
                intensity = getattr(event, 'intensity', 1.0)
                # Blend: norm + (1-freq_weight) * intensity boost
                norm_biased = norm + (1.0 - freq_weight) * intensity * (1.0 - norm)
                norm_biased = max(0.0, min(1.0, norm_biased))
                # Map to TCode output range
                p0_val = int(tcode_min_val + norm_biased * (tcode_max_val - tcode_min_val))
                p0_val = max(0, min(9999, p0_val))
                cmd.pulse_freq = p0_val
                # Update Pulse Freq display
                if hasattr(self, 'pulse_freq_label'):
                    display_freq = p0_val / 67  # Convert TCode to approx Hz (1.5-150hz range)
                    self.pulse_freq_label.setText(f"Pulse: {display_freq:.0f}hz")
                print(f"[Main] Sending cmd: a={cmd.alpha:.2f} b={cmd.beta:.2f} v={cmd.volume:.2f} P0={p0_val:04d}")
                self.network_engine.send_command(cmd)
        elif event.is_beat and not self.is_sending:
            print("[Main] Beat detected but Play not enabled")
    
    def _network_status_callback(self, message: str, connected: bool):
        """Called from network thread on status change"""
        self.signals.status_changed.emit(message, connected)
    
    def _on_beat(self, event: BeatEvent):
        """Handle beat event in GUI thread"""
        if event.is_beat:
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
                self.spectrum_canvas.update_spectrum(spectrum, peak, flux)
            else:
                self.spectrum_canvas.update_spectrum(self._pending_spectrum)
            self._pending_spectrum = None
    
    def _on_status_change(self, message: str, connected: bool):
        """Update connection status"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {'#0f0' if connected else '#f55'};")
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.test_btn.setEnabled(connected)
    
    def _update_display(self):
        """Periodic display update"""
        if self.stroke_mapper:
            alpha, beta = self.stroke_mapper.get_current_position()
            self.position_canvas.update_position(alpha, beta)
            self.alpha_label.setText(f"α: {alpha:.2f}")
            self.beta_label.setText(f"β: {beta:.2f}")
    
    def closeEvent(self, event):
        """Cleanup on close - ensure all threads are stopped before UI is destroyed"""
        self._stop_engines()
        if self.network_engine:
            self.network_engine.stop()

        # Save all settings from sliders to config before closing
        self.config.stroke.phase_advance = self.phase_advance_slider.value()
        self.config.pulse_freq.monitor_freq_min = self.pulse_freq_low_slider.value()
        self.config.pulse_freq.monitor_freq_max = self.pulse_freq_high_slider.value()
        self.config.pulse_freq.tcode_freq_min = self.tcode_freq_min_slider.value()
        self.config.pulse_freq.tcode_freq_max = self.tcode_freq_max_slider.value()
        self.config.pulse_freq.freq_weight = self.freq_weight_slider.value()
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
