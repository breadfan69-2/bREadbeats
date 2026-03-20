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
import threading
import random
from collections import deque
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSlider, QComboBox, QPushButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QLineEdit,
    QGridLayout, QMenu, QMessageBox,
    QSplashScreen, QScrollArea, QInputDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRectF, QEvent
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QPixmap
from typing import Any, Optional

# PyQtGraph for high-performance real-time plotting
import pyqtgraph as pg
pg.setConfigOptions(antialias=False, useOpenGL=False)  # Disable for compatibility

from config import (
    BEAT_RANGE_LIMITS,
    BeatDetectionType,
    StrokeMode,
)
from logging_utils import get_log_level, log_event, set_log_level
from audio_engine import AudioEngine, BeatEvent
from network_engine import TCodeCommand
from network_lifecycle import ensure_network_engine, toggle_user_connection
from command_wiring import apply_volume_ramp
from close_persist_wiring import persist_runtime_ui_to_config
from config_facade import (
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
from keyboard_teacher import KeyboardTeacher
from version import __version__

print(f"[Startup] main.py imports ready (+{(time.perf_counter()-_import_t0)*1000:.0f} ms)", flush=True)


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

from widgets import (
    SignalBridge,
    FFTBinBarGraphCanvas, PositionCanvas,
    RangeSlider, RangeSliderWithLabel, SliderWithLabel,
    CollapsibleGroupBox, NoWheelScrollArea,
    WaveformLiveCanvas, FrequencyDbLiveCanvas,
)
from stylesheet import get_main_stylesheet, get_thin_scrollbar_style
import event_handlers  # noqa: extracted event handlers



class BREadbeatsWindow(QMainWindow):
    """Main application window"""
    FIXED_JITTER_AMPLITUDE = 0.04
    FIXED_JITTER_INTENSITY = 190.0
    FIXED_AXIS_WEIGHT = 1.0
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("bREadbeats")
        self.setMinimumSize(400, 300)
        self.resize(825, 475)
        self.setStyleSheet(get_main_stylesheet())
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
        self._enforce_fixed_effect_axis_values()
        
        # Initialize engines to None early (required before learning-config apply)
        self.audio_engine = None
        self.network_engine = None
        self.stroke_mapper = None
        self._is_shutting_down = False
        
        self.config.stroke.mode = StrokeMode.SIMPLE_CIRCLE
        self._apply_release_learning_defaults()
        # Apply persisted log level early so downstream modules inherit
        set_log_level(getattr(self.config, 'log_level', 'INFO'))
        self.signals = SignalBridge()
        
        # Initialize optional UI state
        self._dry_run_enabled = bool(getattr(self.config.device_limits, 'dry_run', False))
        self._advanced_flux_threshold_slider = None
        self._advanced_flux_scaling_slider = None
        self._advanced_controls_scroll = None
        self._advanced_flux_group = None
        self._beat_detection_dialog = None
        self._beat_detection_popout_content = None
        self._pulse_settings_dialog = None
        self._pulse_settings_popout_content = None
        self._tempo_tracking_popout_content = None
        self._auto_fill_controls_widgets = {}
        self._motion_settings_dialog = None
        self._developer_controls_dialog = None
        self._developer_controls_tab_widget = None
        self._developer_unlock_dialog = None
        self._developer_controls_unlocked = False
        self._trigger_settings_tab_content = None
        self._auto_fill_tab_content = None
        self.jitter_effect_action = None
        self.intelligence_enabled_action = None
        self._beats_per_rotation_menu = None
        self._beats_per_rotation_actions = []
        self.connection_toggle_action = None
        self.connection_test_action = None
        
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
        self._cached_p0_enabled: bool = bool(getattr(self.config.pulse_freq, 'enabled', False))
        self._cached_f0_enabled: bool = bool(getattr(self.config.carrier_freq, 'enabled', False))
        self._cached_pulse_mode: int = max(0, min(1, int(getattr(self.config.pulse_freq, 'mode', 0) or 0)))
        self._cached_pulse_invert: bool = bool(getattr(self.config.pulse_freq, 'invert', False))
        self._cached_f0_mode: int = max(0, min(1, int(getattr(self.config.carrier_freq, 'mode', 0) or 0)))
        self._cached_f0_invert: bool = bool(getattr(self.config.carrier_freq, 'invert', False))
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
        self._cached_p1_mode: int = 0  # 0=Volume(RMS), 1=Hz, 2=Speed
        self._cached_p1_invert: bool = False
        self._cached_p1_display: str = "Pulse Width: off"
        self._cached_p1_tcode_min: int = 1000
        self._cached_p1_tcode_max: int = 8000
        self._prev_p1_enabled: bool = False
        
        # P3 (Rise Time) cached state
        self._cached_p3_enabled: bool = False
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
        self._f0_last_sent_tcode: Optional[int] = None  # Last F0 tcode value sent (for smoothing)
        self._f0_duration_base_ms: float = 220.0  # Base F0 duration (ms)
        # C0 Band mode rate limiter: fast travel for low-latency response
        self._c0_band_target: Optional[int] = None   # Current target tcode for band mode
        self._c0_band_current: Optional[int] = None   # Current sent tcode value (traveling)
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
        self._advanced_flux_threshold_slider = None
        self._advanced_flux_scaling_slider = None
        self._auto_fill_controls_widgets = {}
        self._motion_settings_tab_widget = None
        self._developer_controls_dialog = None
        self._developer_controls_tab_widget = None
        self._developer_unlock_dialog = None
        self._developer_controls_unlocked = False
        self._trigger_settings_tab_content = None
        self._auto_fill_tab_content = None
        
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
        self._pause_park_active: bool = False
        
        # Auto-connect TCP on startup
        self._auto_connect_tcp()

        # ── Keyboard Teaching Mode (dev-only) ──────────────────────────
        self._keyboard_teacher = KeyboardTeacher(base_dir=Path(__file__).parent)
        self._keyboard_teacher_label: Optional[QLabel] = None
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        # Mark transport UI as ready on the first event-loop turn.
        # This guarantees an early single Start click is queued then applied,
        # instead of feeling like it was dropped during startup warm-up.
        QTimer.singleShot(0, self._mark_transport_ready)

    def _enforce_fixed_effect_axis_values(self):
        self.config.jitter.amplitude = float(self.FIXED_JITTER_AMPLITUDE)
        self.config.jitter.intensity = float(self.FIXED_JITTER_INTENSITY)
        self.config.alpha_weight = float(self.FIXED_AXIS_WEIGHT)
        self.config.beta_weight = float(self.FIXED_AXIS_WEIGHT)

    def _mark_transport_ready(self) -> None:
        """Enable transport input after startup and apply queued Start, if any."""
        self._transport_ready = True
        self._sync_transport_buttons()
        if self._transport_pending_start and not self.is_running:
            self._transport_pending_start = False
            QTimer.singleShot(0, self._apply_pending_start)
        
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
        from ui_builders import create_menu_bar
        return create_menu_bar(self)

    def _build_motion_options_tab(self) -> QWidget:
        from ui_builders import build_motion_options_tab
        return build_motion_options_tab(self)

    def _create_connection_panel(self) -> QWidget:
        from ui_builders import create_connection_panel
        return create_connection_panel(self)

    def _create_control_panel(self) -> QWidget:
        from ui_builders import create_control_panel
        return create_control_panel(self)

    def _create_spectrum_panel(self) -> QWidget:
        from ui_builders import create_spectrum_panel
        return create_spectrum_panel(self)

    def _create_position_panel(self) -> QWidget:
        from ui_builders import create_position_panel
        return create_position_panel(self)

    def _create_main_controls_panel(self) -> QWidget:
        from ui_builders import create_main_controls_panel
        return create_main_controls_panel(self)

    def _create_tcode_freq_tab(self) -> QWidget:
        from ui_builders import create_tcode_freq_tab
        return create_tcode_freq_tab(self)

    def _create_beat_detection_tab(self) -> QWidget:
        from ui_builders import create_beat_detection_tab
        return create_beat_detection_tab(self)

    def _create_tempo_response_group(self, lock_default: bool = True) -> QGroupBox:
        from ui_builders import create_tempo_response_group
        return create_tempo_response_group(self, lock_default)

    def _create_tempo_tracking_tab(self, include_advanced_controls: bool = False, advanced_locked: bool = True) -> QWidget:
        from ui_builders import create_tempo_tracking_tab
        return create_tempo_tracking_tab(self, include_advanced_controls, advanced_locked)

    def _on_options_audio_device(self):
        from dialog_builders import on_options_audio_device
        return on_options_audio_device(self)

    def _dialog_set_device_mic(self, combo: QComboBox):
        from dialog_builders import dialog_set_device_mic
        return dialog_set_device_mic(self, combo)

    def _dialog_set_device_loopback(self, combo: QComboBox):
        from dialog_builders import dialog_set_device_loopback
        return dialog_set_device_loopback(self, combo)

    def _on_options_connection(self):
        from dialog_builders import on_options_connection
        return on_options_connection(self)

    def _open_developer_controls_window(self, tab_index: int = 0, scroll_to_flux: bool = False):
        from dialog_builders import open_developer_controls_window
        return open_developer_controls_window(self, tab_index, scroll_to_flux)

    def _show_developer_controls_unlock_popup(self):
        from dialog_builders import show_developer_controls_unlock_popup
        return show_developer_controls_unlock_popup(self)

    def _on_options_beat_detection(self):
        from dialog_builders import on_options_beat_detection
        return on_options_beat_detection(self)

    def _on_options_auto_fill_adaptation(self, as_tab: bool = False):
        from dialog_builders import on_options_auto_fill_adaptation
        return on_options_auto_fill_adaptation(self, as_tab)

    def _on_options_motion_settings(self, tab_index: int = 0):
        from dialog_builders import on_options_motion_settings
        return on_options_motion_settings(self, tab_index)

    def _on_options_motion_readiness(self, as_tab: bool = False):
        from dialog_builders import on_options_motion_readiness
        return on_options_motion_readiness(self, as_tab)

    def _on_device_limits(self, first_run: bool = False):
        from dialog_builders import on_device_limits
        return on_device_limits(self, first_run)

    def _sync_pulse_sent_spin_limits_from_device_limits(self):
        from dialog_builders import sync_pulse_sent_spin_limits_from_device_limits
        return sync_pulse_sent_spin_limits_from_device_limits(self)

    def _scroll_advanced_controls_to_flux(self):
        from dialog_builders import scroll_advanced_controls_to_flux
        return scroll_advanced_controls_to_flux(self)

    def _on_advanced_controls(self, scroll_to_flux: bool = False, as_tab: bool = False):
        from dialog_builders import on_advanced_controls
        return on_advanced_controls(self, scroll_to_flux, as_tab)

    def _on_help(self):
        from dialog_builders import on_help
        return on_help(self)

    def _on_fft_bin_diagnostics(self):
        from dialog_builders import on_fft_bin_diagnostics
        return on_fft_bin_diagnostics(self)

    def _on_about(self):
        from dialog_builders import on_about
        return on_about(self)

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

        def _persist_pulse_dialog_settings(_result: int) -> None:
            persist_runtime_ui_to_config(self, self.config)
            save_config(self.config)

        dialog.finished.connect(_persist_pulse_dialog_settings)

        self._pulse_settings_dialog = dialog
        dialog.show()
        dialog.raise_()
        _focus_pulse_dialog(dialog)
        QTimer.singleShot(0, lambda d=dialog: _focus_pulse_dialog(d))

    def _apply_geometry_rest_to_mapper(self) -> None:
        if not self.stroke_mapper:
            return
        if hasattr(self.stroke_mapper, 'configure_geometry_rest_state'):
            self.stroke_mapper.configure_geometry_rest_state()

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
    
    def _schedule_startup_notices(self):
        # First-run device limits prompt eligibility
        dl = self.config.device_limits
        has_values = (dl.p0_freq_max > 0 or dl.c0_freq_max > 0)
        show_device_limits = (not dl.prompted and not dl.dont_show_on_startup and not has_values)

        if show_device_limits:
            QTimer.singleShot(500, lambda: self._on_device_limits(first_run=True))
    
    def _apply_config_to_ui(self):
        from event_handlers import apply_config_to_ui
        apply_config_to_ui(self)

    def _populate_audio_devices(self):
        from event_handlers import populate_audio_devices
        populate_audio_devices(self)

    def _apply_release_learning_defaults(self) -> None:
        from event_handlers import apply_release_learning_defaults
        apply_release_learning_defaults(self)

    def _on_start_stop(self, checked: bool | None = None):
        from event_handlers import on_start_stop
        on_start_stop(self, checked)

    def _on_play_pause(self, checked: bool | None = None):
        from event_handlers import on_play_pause
        on_play_pause(self, checked)

    def _audio_callback(self, event: BeatEvent):
        from event_handlers import audio_callback
        audio_callback(self, event)

    def _on_beat(self, event: BeatEvent):
        from event_handlers import on_beat
        on_beat(self, event)

    def _on_spectrum(self, spectrum: np.ndarray):
        from event_handlers import on_spectrum
        on_spectrum(self, spectrum)

    def _do_spectrum_update(self):
        from event_handlers import do_spectrum_update
        do_spectrum_update(self)

    def _update_display(self):
        from event_handlers import update_display
        update_display(self)

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

    def _apply_learning_config_to_mapper(self) -> None:
        mapper_live = self.stroke_mapper
        if mapper_live is None:
            return
        if hasattr(mapper_live, 'configure_learning'):
            mapper_live.configure_learning(
                enabled=bool(self.config.beat.teaching_learning_enabled),
                use_fitted_rules=bool(self.config.beat.teaching_use_fitted_rules),
                learning_strength=float(self.config.beat.teaching_learning_strength),
                min_confidence=float(self.config.beat.teaching_min_confidence),
                no_motion_bias=float(self.config.beat.teaching_no_motion_bias),
                rule_fit_path=str(self.config.beat.teaching_rule_fit_path),
            )

    def _on_effects_jitter_toggle(self, checked: bool):
        self.config.jitter.enabled = bool(checked)

    def _on_intelligence_toggle(self, checked: bool):
        """Toggle intelligence (checked=on=normal mode, unchecked=simple mode)."""
        self.config.stroke.intelligence_enabled = bool(checked)
        # Show/hide beats-per-rotation submenu
        bpr_menu = getattr(self, '_beats_per_rotation_menu', None)
        if bpr_menu is not None:
            bpr_menu.menuAction().setVisible(not checked)
        # Grey out intelligence-dependent controls when in simple mode
        self._sync_intelligence_dependent_controls()

    def _on_beats_per_rotation_change(self, value: int):
        """Set beats-per-rotation for simple mode (1, 2, or 4)."""
        self.config.stroke.simple_mode_beats_per_rotation = max(1, min(4, value))
        # Update radio checkmarks
        for action in getattr(self, '_beats_per_rotation_actions', []):
            action.setChecked(action.text() == str(value))

    def _sync_intelligence_dependent_controls(self):
        """Grey out controls that are irrelevant in simple mode."""
        is_simple = not bool(getattr(self.config.stroke, 'intelligence_enabled', True))
        # These menu actions are all bypassed in simple mode
        for attr_name in ('jitter_effect_action', 'metronome_lock_required_action'):
            action = getattr(self, attr_name, None)
            if action is not None:
                action.setEnabled(not is_simple)

    def _sync_effects_menu_actions(self):
        jitter_action = getattr(self, 'jitter_effect_action', None)
        if jitter_action is not None:
            with self._signals_blocked(jitter_action):
                jitter_action.setChecked(bool(getattr(self.config.jitter, 'enabled', True)))
        intel_action = getattr(self, 'intelligence_enabled_action', None)
        if intel_action is not None:
            with self._signals_blocked(intel_action):
                intel_action.setChecked(bool(getattr(self.config.stroke, 'intelligence_enabled', True)))
        self._sync_intelligence_dependent_controls()

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

    def _on_viz_menu_change(self, index: int):
        """Handle spectrum type change from Options menu"""
        # Update checkmarks
        for i, action in enumerate(self._viz_type_actions):
            action.setChecked(i == index)
        self.visualizer_type_combo.setCurrentIndex(index)
        self._on_visualizer_type_change(index)
    
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

    def _silence_threshold_to_db(self, threshold_value: float, default_linear: float = 0.01) -> float:
        try:
            value = float(threshold_value)
        except Exception:
            value = float(default_linear)
        if not np.isfinite(value):
            value = float(default_linear)
        if value <= 0.0:
            return float(np.clip(value, -120.0, 12.0))
        return float(np.clip(20.0 * np.log10(max(min(value, 1.0), 1e-12)), -120.0, 12.0))

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

    def _on_energy_response_change(self, value: float) -> None:
        """Slider callback: update energy_response_strength in config."""
        strength = float(np.clip(float(value), 0.0, 2.0))
        setattr(self.config.stroke, 'energy_response_strength', strength)
        self._set_energy_response_display(strength)

    def _set_energy_response_display(self, value: float) -> None:
        """Update the value label on the Energy Response slider."""
        slider_widget = getattr(self, 'main_silence_close_slider', None)
        if slider_widget is None or not hasattr(slider_widget, 'value_label'):
            return
        v = float(np.clip(float(value), 0.0, 2.0))
        slider_widget.value_label.setText(f"{v:.2f}")

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
            self.audio_engine.set_spectrum_skip_frames(skip_values[index])
        print(f"[Config] Spectrum skip frames changed to {skip_values[index]}")
    
    def _on_metrics_global_toggle(self, state):
        """Master toggle for all metric auto-adjust checkboxes"""
        enabled = state == 2
        self.config.auto_adjust.metrics_global_enabled = enabled
        # Only Audio Amp remains auto-adjustable; other metrics are manual-only.
        self.metric_audio_amp_cb.setEnabled(enabled)
        self.metric_audio_amp_cb.setChecked(enabled)
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
        if getattr(self, 'metric_audio_amp_cb', None) and self.metric_audio_amp_cb.isChecked():
            active_metrics.append("AudioAmp")
        
        status_text = f"Metrics: [{', '.join(active_metrics) if active_metrics else 'idle'}]"
        if hasattr(self, 'metric_status_label'):
            self.metric_status_label.setText(status_text)
    
    def _on_metric_feedback(self, feedback_data: dict):
        """Handle feedback from a metric controller (update slider)"""
        if getattr(self, '_is_shutting_down', False):
            return

        def _slider_live(name: str) -> bool:
            slider_wrap = getattr(self, name, None)
            if slider_wrap is None:
                return False
            try:
                slider_wrap.value()
            except RuntimeError:
                return False
            return True

        metric = feedback_data.get('metric', '')
        adjustment = feedback_data.get('adjustment', 0.0)
        direction = feedback_data.get('direction', 'hold')
        
        if metric == 'audio_amp' and adjustment != 0:
            # Adjust audio amplification based on beat presence
            if not _slider_live('audio_gain_slider'):
                return
            current = self.audio_gain_slider.value()
            new_val = current + adjustment
            aa_min, aa_max = BEAT_RANGE_LIMITS['audio_amp']
            new_val = max(aa_min, min(aa_max, new_val))
            if abs(new_val - current) > 0.001:
                self.audio_gain_slider.setValue(new_val)
                reason = feedback_data.get('reason', '')
                actual_bps = feedback_data.get('actual_bps', 0)
                print(f"[Metric] audio_amp: {reason} ({direction}) -> {new_val:.4f}")
        
        elif metric == 'peak_floor':
            return

    def _on_metric_response_speed_change(self, value: float):
        """Handle metric auto-adjust response speed change."""
        self.config.auto_adjust.metric_response_speed = value
        if hasattr(self, 'audio_engine') and self.audio_engine is not None:
            self.audio_engine.set_metric_response_speed(value)
        print(f"[Metric] Auto-adjust speed set to {value:.2f}x")

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
            self.audio_engine.reinitialize_butterworth_filter()
        
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
    
    def _on_tempo_param_change(self, config_attr: str, engine_attr: str, value, transform=None):
        """Generic handler for simple tempo parameter changes.
        Sets config.beat.<config_attr> = value and audio_engine.<engine_attr> = value.
        """
        if transform is not None:
            value = transform(value)
        assert hasattr(self.config.beat, config_attr), f"bad tempo attr: {config_attr}"
        setattr(self.config.beat, config_attr, value)
        if self.audio_engine:
            setattr(self.audio_engine, engine_attr, value)

    def _on_aggressive_tempo_snap_toggle(self, enabled: bool):
        """Toggle confidence-gated aggressive metronome BPM snapping."""
        self.config.beat.aggressive_tempo_snap_enabled = enabled
        if self.audio_engine:
            self.audio_engine.set_aggressive_tempo_snap_enabled(enabled)

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
        if getattr(self, '_is_shutting_down', False):
            return
        if self._transport_transition:
            return
        if self.is_running:
            self._transport_pending_start = False
            return
        self._transport_pending_start = False
        self._on_start_stop(True)

    def _apply_pending_stop(self) -> None:
        """Apply a queued stop request captured during start transition."""
        if getattr(self, '_is_shutting_down', False):
            return
        if self._transport_transition:
            return
        if not self.is_running:
            self._transport_pending_stop = False
            return
        self._transport_pending_stop = False
        self._on_start_stop(False)

    def _apply_pending_play(self) -> None:
        """Apply queued play/pause request captured during transport transition."""
        if getattr(self, '_is_shutting_down', False):
            return
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
    
    def _on_detection_type_change(self, index: int):
        """Change beat detection type"""
        self.config.beat.detection_type = BeatDetectionType(index + 1)
    
    def _on_mode_change(self, index: int):
        """Mode is temporarily pinned to Circle."""
        self.config.stroke.mode = StrokeMode.SIMPLE_CIRCLE
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

        self.stroke_mapper = StrokeMapper(self.config, get_volume=lambda: self.volume_slider.value() / 100.0, audio_engine=self.audio_engine)
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
            'metric_audio_amp_cb': 'audio_amp',
        }
        synced = []
        for attr, metric in metric_map.items():
            cb = getattr(self, attr, None)
            if cb is not None and cb.isChecked():
                self.audio_engine.enable_metric_autoranging(metric, True)
                synced.append(metric)
        if synced:
            print(f"[Metric] Synced {len(synced)} metrics to engine: {', '.join(synced)}")
    
    def _stop_engines(self):
        """Stop all engines and background threads"""
        self.is_running = False
        self.stroke_mapper = None

        if self.audio_engine:
            self.audio_engine.stop()
            self.audio_engine = None
    
    def _compute_and_attach_tcode(self, cmd, event, spectrum=None):
        from tcode_wiring import compute_and_attach_tcode
        compute_and_attach_tcode(self, cmd, event, spectrum)

    def _network_status_callback(self, message: str, connected: bool):
        """Called from network thread on status change"""
        if getattr(self, '_is_shutting_down', False):
            return
        self.signals.status_changed.emit(message, connected)
    
    def _turn_off_beat_indicator(self):
        """Turn off beat indicator after minimum duration"""
        beat_indicator = getattr(self, 'beat_indicator', None)
        if beat_indicator is None:
            return
        try:
            beat_indicator.setStyleSheet("color: #333; font-size: 24px;")
        except RuntimeError:
            return
    
    def _turn_off_downbeat_indicator(self):
        """Turn off downbeat indicator after minimum duration"""
        downbeat_indicator = getattr(self, 'downbeat_indicator', None)
        if downbeat_indicator is None:
            return
        try:
            downbeat_indicator.setStyleSheet("color: #333; font-size: 24px;")
        except RuntimeError:
            return
    
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
    
    def _on_status_change(self, message: str, connected: bool):
        """Update connection status"""
        if getattr(self, '_is_shutting_down', False):
            return
        try:
            self.status_label.setText("Connected" if connected else "Connect")
            self.status_label.setStyleSheet(f"color: {'#0af' if connected else '#fff'};")
        except RuntimeError:
            return
        connection_toggle_action = getattr(self, 'connection_toggle_action', None)
        if connection_toggle_action is not None:
            connection_toggle_action.setText("Disconnect" if connected else "Connect")
        connection_test_action = getattr(self, 'connection_test_action', None)
        if connection_test_action is not None:
            connection_test_action.setEnabled(connected)

    def _toggle_keyboard_teaching(self) -> None:
        """Start or stop a keyboard teaching session (dev-only)."""
        teacher = self._keyboard_teacher
        if teacher.active:
            saved = teacher.stop_session()
            print(f"[KeyboardTeacher] Session stopped → {saved}")
            self._update_keyboard_teacher_label()
        else:
            path = teacher.start_session()
            print(f"[KeyboardTeacher] Session started → {path}")
            self._update_keyboard_teacher_label()

    def _update_keyboard_teacher_label(self) -> None:
        label = getattr(self, '_keyboard_teacher_label', None)
        if label is None:
            return
        teacher = self._keyboard_teacher
        try:
            if teacher.active:
                gate = teacher.last_gate_fail or "open"
                is_parked = teacher.is_parked
                scale = teacher.speed_scale
                step = teacher.speed_step
                bpm = teacher._last_bpm
                if is_parked:
                    state_str = "PARK"
                    color = "#ff8800"
                elif scale >= 1.0:
                    state_str = f"{scale:.3g}x"
                    color = "#ff4444"
                else:
                    denom = round(1.0 / scale)
                    state_str = f"1/{denom}x"
                    color = "#ff4444"
                label.setText(
                    f"🎹 [{state_str}]  step:{step:+d}  {bpm:.0f}bpm  gate:{gate}"
                )
                label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
                label.setVisible(True)
            else:
                label.setText("🎹 OFF")
                label.setStyleSheet("color: #666; font-size: 11px;")
                label.setVisible(True)
        except RuntimeError:
            pass

    _ARROW_MAP = {
        Qt.Key.Key_Up: "up",
        Qt.Key.Key_Down: "down",
        Qt.Key.Key_Left: "left",
        Qt.Key.Key_Right: "right",
    }

    def _handle_keyboard_teacher_key_event(self, event, is_press: bool) -> bool:
        teacher = getattr(self, '_keyboard_teacher', None)
        if teacher is None:
            return False

        if is_press:
            if (event.key() == Qt.Key.Key_K
                    and event.modifiers() == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)):
                self._toggle_keyboard_teaching()
                self._update_keyboard_teacher_label()
                return True

            direction = self._ARROW_MAP.get(event.key())
            if direction and teacher.active and not event.isAutoRepeat():
                teacher.on_arrow_down(direction)
                self._update_keyboard_teacher_label()
                return True
        else:
            direction = self._ARROW_MAP.get(event.key())
            if direction and teacher.active and not event.isAutoRepeat():
                teacher.on_arrow_up(direction)
                self._update_keyboard_teacher_label()
                return True
        return False

    def eventFilter(self, obj, event):
        try:
            if event.type() == QEvent.Type.KeyPress:
                if self._handle_keyboard_teacher_key_event(event, is_press=True):
                    event.accept()
                    return True
            elif event.type() == QEvent.Type.KeyRelease:
                if self._handle_keyboard_teacher_key_event(event, is_press=False):
                    event.accept()
                    return True
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event) -> None:
        if self._handle_keyboard_teacher_key_event(event, is_press=True):
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if self._handle_keyboard_teacher_key_event(event, is_press=False):
            event.accept()
            return

        super().keyReleaseEvent(event)

    def closeEvent(self, event):
        """Cleanup on close - ensure all threads are stopped before UI is destroyed"""
        self._is_shutting_down = True

        app = QApplication.instance()
        if app is not None:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass

        # Stop keyboard teaching session (flush captured data)
        teacher = getattr(self, '_keyboard_teacher', None)
        if teacher is not None and teacher.active:
            saved = teacher.stop_session()
            if saved:
                print(f"[KeyboardTeacher] Session saved to {saved}")

        for timer_name in ('update_timer', '_spectrum_timer', 'beat_timer', 'downbeat_timer'):
            timer = getattr(self, timer_name, None)
            if timer is None:
                continue
            try:
                timer.stop()
            except RuntimeError:
                pass

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
