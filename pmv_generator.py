from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QEvent, QEventLoop, QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pmv_audio_analysis import analyze_full_file, load_audio, AnalysisConfig as _AnalysisConfig
from pmv_automap import automap_optimize
from pmv_axis_converter import MultiAxisResult, convert_to_2d
from pmv_beat_engine import BeatTimeline, detect_beats
from config_persistence import get_config_dir
from pmv_controls import PMVControlsPanel
from pmv_funscript_io import FunscriptAction, FunscriptMetadata, read_funscript, write_csv, write_funscript
from pmv_position_mapper import PositionTimeline, generate_positions
from pmv_visualizations import AuxAxisPanel, VideoPreviewWidget, VisualizationArea
from funscript_edit_state import FunscriptEditState, LockedRegion
from funscript_utils import AXIS_SUFFIXES, strip_axis_suffix as _strip_axis_suffix

if TYPE_CHECKING:
    from orbital_replay import OrbitalReplayResult


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".wma", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".wmv", ".mov", ".flv"}
FUNSCRIPT_EXTENSIONS = {".funscript"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
IMPORTABLE_EXTENSIONS = SUPPORTED_EXTENSIONS | FUNSCRIPT_EXTENSIONS


def _merge_dict(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class _PipelineWorker(QObject):
    """Runs a single callable off the main thread and reports progress."""

    progress = pyqtSignal(str, str, float)  # step_name, message, percent
    finished = pyqtSignal(object)  # result or None
    error = pyqtSignal(str)  # error message

    def __init__(self, func, step_name: str):
        super().__init__()
        self._func = func
        self._step_name = step_name

    def run(self):
        def progress_cb(msg, pct):
            self.progress.emit(self._step_name, str(msg), float(pct))
        try:
            result = self._func(progress_cb)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _PreviewWorker(QObject):
    """Lightweight background worker for live previews (no progress reporting)."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func):
        super().__init__()
        self._func = func

    def run(self):
        try:
            result = self._func()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _BusyOverlay(QWidget):
    """Translucent overlay with an animated progress bar shown during background work."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container = QWidget(self)
        container.setFixedWidth(300)
        container.setStyleSheet(
            "background-color: rgba(30, 30, 30, 220); border: 1px solid #5d5d5d; border-radius: 8px;"
        )
        inner = QVBoxLayout(container)
        inner.setContentsMargins(16, 12, 16, 12)
        inner.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel("Processing\u2026")
        self._label.setStyleSheet("color: #e0e0e0; font-size: 13px; font-weight: bold; background: transparent; border: none;")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setFixedWidth(260)
        self._bar.setFixedHeight(18)
        self._bar.setRange(0, 0)  # indeterminate / busy animation
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(
            "QProgressBar { background: #4d4d4d; border: 1px solid #5d5d5d; border-radius: 4px; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #008b8b, stop:1 #00bfbf); border-radius: 3px; }"
        )
        inner.addWidget(self._bar, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(container)

        self.hide()

    # -- public helpers --

    def show_busy(self, message: str = "Processing\u2026") -> None:
        self._label.setText(message)
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())
        self.raise_()
        self.show()

    def hide_busy(self) -> None:
        self.hide()

    # keep overlay sized to parent
    def resizeEvent(self, event) -> None:  # noqa: N802
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())
        super().resizeEvent(event)


def _resample_actions(actions: list[FunscriptAction], step_ms: float,
                      duration_ms: int) -> np.ndarray:
    """Resample sparse funscript actions to a uniform grid via linear interp."""
    import numpy as np
    if not actions:
        n = max(1, int(duration_ms / step_ms) + 1)
        return np.full(n, 50.0)
    times = np.array([float(a.at) for a in actions])
    values = np.array([float(a.pos) for a in actions])
    grid = np.arange(0.0, float(duration_ms) + step_ms * 0.5, step_ms)
    return np.interp(grid, times, values)


def _grid_to_actions(grid: np.ndarray, step_ms: float,
                     tolerance: float = 0.8) -> list[FunscriptAction]:
    """Convert uniform grid back to sparse funscript actions (simplified)."""
    if len(grid) == 0:
        return []
    actions = [FunscriptAction(int(round(i * step_ms)), int(round(float(v))))
               for i, v in enumerate(grid)]
    # Inline simplification — keep points where linear interp deviates
    if len(actions) <= 2:
        return actions
    keep = [True] * len(actions)
    i = 0
    while i < len(actions) - 2:
        j = i + 1
        while j < len(actions) - 1:
            a_i, a_next = actions[i], actions[j + 1]
            dt_total = a_next.at - a_i.at
            if dt_total <= 0:
                j += 1
                continue
            can_remove = True
            for k in range(i + 1, j + 1):
                a_k = actions[k]
                t = (a_k.at - a_i.at) / dt_total
                interp = a_i.pos + t * (a_next.pos - a_i.pos)
                if abs(a_k.pos - interp) > tolerance:
                    can_remove = False
                    break
            if can_remove:
                keep[j] = False
                j += 1
            else:
                break
        i = j
    return [a for a, k in zip(actions, keep) if k]


def _apply_orbital_overlay(
    multi_axis: MultiAxisResult,
    timeline,
    beats: BeatTimeline,
    analysis_cfg: _AnalysisConfig,
    blend: float = 1.0,
    progress_callback=None,
    cached_orbital=None,
) -> tuple[MultiAxisResult, 'OrbitalReplayResult']:
    """Blend restim alpha/beta with orbital replay output.

    blend=0.0 → pure restim, blend=1.0 → pure orbital.
    Partial blend uses additive modulation: restim + (orbital - center) * blend.
    Returns (multi_axis, orbital_result) so caller can cache the orbital run.
    """
    import numpy as np
    from config_facade import load_config
    from orbital_replay import replay_orbital

    if cached_orbital is not None:
        result = cached_orbital
    else:
        result = replay_orbital(
            timeline=timeline,
            beat_timeline=beats,
            config=load_config(),
            analysis_cfg=analysis_cfg,
            progress_callback=progress_callback,
        )

    blend = float(np.clip(blend, 0.0, 1.0))
    if blend >= 1.0:
        # Pure orbital — direct replacement (original behaviour)
        multi_axis.axes["alpha"] = result.alpha_actions
        multi_axis.axes["beta"] = result.beta_actions
        return multi_axis, result

    duration_ms = int(timeline.duration_ms)
    step = 10.0  # 10ms grid = 100 Hz
    # Orbital alpha/beta may be in a different axis convention from restim.
    # Swap orbital axes to align with restim before blending.
    orbital_for_blend = {
        "alpha": result.beta_actions,   # orbital Y → restim alpha (vertical)
        "beta": result.alpha_actions,   # orbital X → restim beta (horizontal)
    }
    for axis_name in ("alpha", "beta"):
        restim_grid = _resample_actions(multi_axis.axes.get(axis_name, []),
                                        step, duration_ms)
        orbital_grid = _resample_actions(orbital_for_blend[axis_name], step, duration_ms)
        # Additive modulation: use restim as base, layer orbital deviation on top.
        # orbital_grid is 0-100; deviation = orbital_grid - 50.0
        # This preserves restim's stroke shape while adding orbital texture.
        blended = restim_grid + (orbital_grid - 50.0) * blend
        blended = np.clip(blended, 0.0, 100.0)
        multi_axis.axes[axis_name] = _grid_to_actions(blended, step)

    return multi_axis, result


class PMVGeneratorWindow(QMainWindow):
    """Step-through PMV generator window (Load -> Analyze -> Beats -> Generate -> Export)."""

    def __init__(self, parent: QWidget | None = None,
                 network_engine=None, position_canvas=None):
        super().__init__(parent)
        self.setWindowTitle("PMV Funscript Generator")
        self.resize(1200, 760)
        self.setAcceptDrops(True)

        self._network_engine = network_engine
        self._position_canvas = position_canvas

        self._file_path: str | None = None
        self._media_path: str | None = None
        self._samples = None
        self._timeline = None
        self._beats: BeatTimeline | None = None
        self._positions: PositionTimeline | None = None
        self._multi_axis: MultiAxisResult | None = None
        self._cached_orbital_result = None  # cache expensive orbital replay
        self._edit_state = FunscriptEditState(self)
        self._aux_edit_states: dict[str, FunscriptEditState] = {}  # axis_name -> edit state
        self._current_edit_axis: str = "main"
        self._live_preview_busy = False
        self._last_live_preview_signature: str | None = None
        self._pipeline_busy = False
        self._busy_cursor_set = False
        self._last_progress_pump = 0.0
        self._last_playback_status_update = 0.0
        self._last_preview_send_update = 0.0
        self._worker_thread: QThread | None = None
        self._worker: _PipelineWorker | None = None

        self._beat_preview_thread: QThread | None = None
        self._beat_preview_worker: _PreviewWorker | None = None
        self._live_preview_thread: QThread | None = None
        self._live_preview_worker: _PreviewWorker | None = None

        self._live_preview_timer = QTimer(self)
        self._live_preview_timer.setSingleShot(True)
        self._live_preview_timer.setInterval(500)
        self._live_preview_timer.timeout.connect(self._run_live_preview)

        self._beat_preview_timer = QTimer(self)
        self._beat_preview_timer.setSingleShot(True)
        self._beat_preview_timer.setInterval(1200)
        self._beat_preview_timer.timeout.connect(self._run_beat_preview)
        self._beat_preview_busy = False
        self._last_beat_preview_signature: str | None = None

        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(6)
        preset_row.addWidget(QLabel("Preset"))

        self.preset_combo = QComboBox(self)
        preset_row.addWidget(self.preset_combo, 1)

        self.load_preset_btn = QPushButton("Load", self)
        self.save_preset_btn = QPushButton("Save As", self)
        self.refresh_preset_btn = QPushButton("Refresh", self)
        self.open_script_btn = QPushButton("Open Script", self)
        self.load_folder_btn = QPushButton("Load Folder", self)
        self.regen_axes_btn = QPushButton("Regen Axes", self)
        self.regen_axes_btn.setToolTip("Regenerate alpha/beta and auxiliary axes from the current edited main script")
        self.regen_axes_btn.setEnabled(False)
        self.fill_blanks_btn = QPushButton("Fill Blanks", self)
        self.fill_blanks_btn.setToolTip(
            "Detect blank/flat regions in the loaded funscript and fill them with beat-driven motion.\n"
            "Existing motion is preserved; only gaps are filled."
        )
        self.fill_blanks_btn.setEnabled(False)
        preset_row.addWidget(self.load_preset_btn)
        preset_row.addWidget(self.save_preset_btn)
        preset_row.addWidget(self.refresh_preset_btn)
        preset_row.addWidget(self.open_script_btn)
        preset_row.addWidget(self.load_folder_btn)
        preset_row.addWidget(self.regen_axes_btn)
        preset_row.addWidget(self.fill_blanks_btn)

        root_layout.addLayout(preset_row)

        splitter = QSplitter(self)
        root_layout.addWidget(splitter, 1)

        self.controls = PMVControlsPanel(splitter)
        splitter.addWidget(self.controls)

        center = QWidget(splitter)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(6, 6, 6, 6)
        center_layout.setSpacing(6)

        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: #b0bec5; font-size: 11px;")
        center_layout.addWidget(self.file_label)

        self.video_preview = VideoPreviewWidget()  # parentless — independent top-level window

        self.visualizations = VisualizationArea(center)
        center_layout.addWidget(self.visualizations, 1)

        self.aux_panel = AuxAxisPanel(center)
        center_layout.addWidget(self.aux_panel, 1)

        # Bind edit state to visualization
        self.visualizations.set_edit_state(self._edit_state)
        self.aux_panel.edit_axis_changed.connect(self._on_edit_axis_switched)

        splitter.addWidget(center)
        splitter.setSizes([380, 820])

        self.setCentralWidget(root)

        self._busy_overlay = _BusyOverlay(root)

        status_bar = QStatusBar(self)
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(200)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.hide()
        status_bar.addPermanentWidget(self._progress_bar)
        self.setStatusBar(status_bar)

        self.controls.config_changed.connect(self._on_controls_changed)
        self.controls.step_bar.step_requested.connect(self._on_step_requested)
        self.visualizations.position_changed.connect(self._on_visualization_position_changed)
        self.visualizations.playback_panel.transport_changed.connect(self.video_preview.on_transport)
        self._edit_state.changed.connect(self._on_edit_state_changed)
        self.aux_panel.link_x_axis(self.visualizations.overlay_plot)
        self.load_preset_btn.clicked.connect(self._on_load_preset_clicked)
        self.save_preset_btn.clicked.connect(self._on_save_preset_clicked)
        self.refresh_preset_btn.clicked.connect(self._reload_preset_catalog)
        self.open_script_btn.clicked.connect(self._on_open_script_clicked)
        self.load_folder_btn.clicked.connect(self._on_load_folder_clicked)
        self.regen_axes_btn.clicked.connect(self._on_regen_axes_clicked)
        self.fill_blanks_btn.clicked.connect(self._on_fill_blanks_clicked)

        self._default_preset_dir, self._user_preset_dir = self._resolve_preset_dirs()
        self._ensure_default_presets()
        self._reload_preset_catalog()
        self._load_last_used_settings()
        self._refresh_step_availability()
        self._on_controls_changed()

    def _preset_dirs(self) -> tuple[Path, Path]:
        return self._default_preset_dir, self._user_preset_dir

    def _resolve_preset_dirs(self) -> tuple[Path, Path]:
        module_root = Path(__file__).resolve().parent
        config_root = get_config_dir()
        user_dir = config_root / "user_pmv_presets"

        if getattr(sys, "frozen", False):
            preferred_defaults = Path(sys.executable).resolve().parent / "defaults" / "pmv_presets"
            defaults_dir = preferred_defaults if self._is_dir_writable(preferred_defaults) else (config_root / "defaults" / "pmv_presets")
        else:
            defaults_dir = module_root / "defaults" / "pmv_presets"

        return defaults_dir, user_dir

    def _is_dir_writable(self, folder: Path) -> bool:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".pmv_write_probe.tmp"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _legacy_user_preset_dirs(self) -> list[Path]:
        candidates: list[Path] = [Path(__file__).resolve().parent / "user_pmv_presets"]
        if getattr(sys, "frozen", False):
            candidates.append(Path(sys.executable).resolve().parent / "user_pmv_presets")

        temp_root = Path(tempfile.gettempdir())
        try:
            candidates.extend(sorted(temp_root.glob("_MEI*/user_pmv_presets")))
        except Exception:
            pass

        unique: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _migrate_legacy_user_presets(self, user_dir: Path) -> int:
        migrated = 0
        try:
            user_resolved = user_dir.resolve()
        except Exception:
            user_resolved = user_dir

        for legacy_dir in self._legacy_user_preset_dirs():
            try:
                legacy_resolved = legacy_dir.resolve()
            except Exception:
                legacy_resolved = legacy_dir

            if legacy_resolved == user_resolved or not legacy_dir.exists():
                continue

            for source in sorted(legacy_dir.glob("*.json")):
                target = user_dir / source.name
                if target.exists():
                    continue
                try:
                    shutil.copy2(source, target)
                    migrated += 1
                except Exception:
                    continue

        return migrated

    def _build_default_presets(self) -> dict[str, dict]:
        base = self.controls.to_preset()
        presets = {
            "balanced": _merge_dict(base, {"pmv_preset_version": 1, "name": "Balanced"}),
            "high_energy": _merge_dict(
                base,
                {
                    "pmv_preset_version": 1,
                    "name": "High Energy",
                    "beat_detection": {"sensitivity": 0.70},
                    "mapping": {"energy_multiplier": 20.0, "pitch_range": 150.0},
                    "ml": {"cadence_mode": "fixed_1"},
                },
            ),
            "chill": _merge_dict(
                base,
                {
                    "pmv_preset_version": 1,
                    "name": "Chill",
                    "beat_detection": {"sensitivity": 0.30},
                    "mapping": {"energy_multiplier": 5.0, "pitch_range": 50.0},
                    "ml": {"cadence_mode": "fixed_4"},
                },
            ),
            "beat_focused": _merge_dict(
                base,
                {
                    "pmv_preset_version": 1,
                    "name": "Beat Focused",
                    "beat_detection": {"sensitivity": 0.60},
                    "mapping": {"pitch_range": 0.0},
                    "ml": {"enabled": False},
                },
            ),
            "ml_driven": _merge_dict(
                base,
                {
                    "pmv_preset_version": 1,
                    "name": "ML Driven",
                    "beat_detection": {"sensitivity": 0.40},
                    "ml": {"enabled": True, "strength": 0.90, "cadence_mode": "auto"},
                },
            ),
        }
        return presets

    def _ensure_default_presets(self) -> None:
        defaults_dir, user_dir = self._preset_dirs()
        defaults_dir.mkdir(parents=True, exist_ok=True)
        user_dir.mkdir(parents=True, exist_ok=True)
        migrated = self._migrate_legacy_user_presets(user_dir)

        for key, payload in self._build_default_presets().items():
            target = defaults_dir / f"{key}.json"
            if target.exists():
                continue
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if migrated > 0:
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Migrated {migrated} legacy PMV preset(s)", 5000)

    def _reload_preset_catalog(self) -> None:
        defaults_dir, user_dir = self._preset_dirs()
        self.preset_combo.clear()

        for scope, folder in (("Default", defaults_dir), ("User", user_dir)):
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.json")):
                label = f"[{scope}] {path.stem}"
                self.preset_combo.addItem(label, str(path))

        if self.preset_combo.count() <= 0:
            self.preset_combo.addItem("(no presets)", "")

    def _load_preset_from_path(self, preset_path: str, show_errors: bool = True) -> bool:
        if not preset_path:
            return False
        try:
            payload = json.loads(Path(preset_path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Preset must be a JSON object")
            self.controls.set_from_preset(payload)
            self._on_controls_changed()
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Loaded preset {Path(preset_path).stem}", 2000)
            return True
        except Exception as exc:
            self._show_error("Preset load failed", f"Unable to load preset: {exc}", show_errors)
            return False

    def _save_preset_to_path(self, preset_path: str, preset_name: str | None = None, show_errors: bool = True) -> bool:
        if not preset_path:
            return False
        try:
            payload = self.controls.to_preset()
            payload["pmv_preset_version"] = 1
            payload["name"] = str(preset_name or Path(preset_path).stem)
            target = Path(preset_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._reload_preset_catalog()
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Saved preset {target.stem}", 2000)
            return True
        except Exception as exc:
            self._show_error("Preset save failed", f"Unable to save preset: {exc}", show_errors)
            return False

    def _last_used_path(self) -> Path:
        return self._user_preset_dir / "_last_used.json"

    def _save_last_used_settings(self) -> None:
        try:
            payload = self.controls.to_preset()
            payload["pmv_preset_version"] = 1
            payload["name"] = "_last_used"
            target = self._last_used_path()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_last_used_settings(self) -> None:
        p = self._last_used_path()
        if p.is_file():
            self._load_preset_from_path(str(p), show_errors=False)

    def _on_load_preset_clicked(self) -> None:
        idx = self.preset_combo.currentIndex()
        path = str(self.preset_combo.itemData(idx) or "")
        self._load_preset_from_path(path, show_errors=True)

    def _on_save_preset_clicked(self) -> None:
        _defaults, user_dir = self._preset_dirs()
        user_dir.mkdir(parents=True, exist_ok=True)
        initial = str(user_dir / "custom_preset.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save PMV Preset", initial, "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path = f"{path}.json"
        self._save_preset_to_path(path, show_errors=True)

    def _show_error(self, title: str, message: str, show_errors: bool = True) -> None:
        self.controls.step_bar.set_step_status(1, "ready")
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(message, 3000)
        if show_errors:
            QMessageBox.critical(self, title, message)

    def _load_video_preview(self) -> None:
        """Show/hide the video preview popout based on current _media_path."""
        media = self._media_path
        if media is not None and Path(media).suffix.lower() in VIDEO_EXTENSIONS:
            self.video_preview.load_media(media)
            self.video_preview.show()
            self.video_preview.raise_()
        else:
            self.video_preview.load_media(None)
            self.video_preview.hide()

    def _progress(self, step_name: str, message: str, percent: float) -> None:
        import sys
        print(f"[PMV] {step_name}: {message} ({percent:.0f}%)", file=sys.stderr, flush=True)
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(f"{step_name}: {message} ({percent:.0f}%)")
        self._progress_bar.setValue(int(max(0, min(100, percent))))
        if not self._progress_bar.isVisible():
            self._progress_bar.show()

        # Keep the window repainting during long synchronous work (Automap/Generate).
        if self._pipeline_busy:
            now = time.monotonic()
            if (now - self._last_progress_pump) >= 0.05:
                app = QApplication.instance()
                if app is not None:
                    app.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
                self._last_progress_pump = now

    def _set_pipeline_busy(self, busy: bool, message: str | None = None) -> None:
        target = bool(busy)
        if target == self._pipeline_busy:
            return

        self._pipeline_busy = target

        enabled = not target
        self.controls.setEnabled(enabled)
        self.preset_combo.setEnabled(enabled)
        self.load_preset_btn.setEnabled(enabled)
        self.save_preset_btn.setEnabled(enabled)
        self.refresh_preset_btn.setEnabled(enabled)

        bar = self.statusBar()
        if bar is not None and message:
            bar.showMessage(str(message), 0)

        if target and not self._busy_cursor_set:
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            self._busy_cursor_set = True
            self._last_progress_pump = 0.0
        elif (not target) and self._busy_cursor_set:
            QApplication.restoreOverrideCursor()
            self._busy_cursor_set = False

    def _run_step_async(
        self,
        step_num: int,
        step_name: str,
        work_fn,
        on_success,
        error_label: str,
    ) -> None:
        """Run *work_fn(progress_cb)* on a background thread with progress bar."""
        if self._worker_thread is not None:
            return  # already running

        self.controls.step_bar.set_step_status(step_num, "running")
        self._set_pipeline_busy(True, f"{step_name}: starting...")
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._busy_overlay.show_busy(f"{step_name}\u2026")

        worker = _PipelineWorker(work_fn, step_name)
        thread = QThread(self)
        worker.moveToThread(thread)

        def handle_progress(sn: str, msg: str, pct: float) -> None:
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"{sn}: {msg} ({pct:.0f}%)")
            self._progress_bar.setValue(int(max(0, min(100, pct))))
            self._busy_overlay.show_busy(f"{sn}: {msg} ({pct:.0f}%)")

        def handle_finished(result: object) -> None:
            thread.quit()
            thread.wait()
            self._worker_thread = None
            self._worker = None
            try:
                on_success(result)
                self.controls.step_bar.set_step_status(step_num, "done")
                self.controls.on_step_completed(step_num)
                self._refresh_step_availability()
            except Exception as exc:
                self.controls.step_bar.set_step_status(step_num, "error")
                self._show_error(f"{step_name} failed", str(exc))
            finally:
                self._set_pipeline_busy(False)
                self._progress_bar.hide()
                self._hide_busy_if_idle()
                self._on_controls_changed()

        def handle_error(msg: str) -> None:
            thread.quit()
            thread.wait()
            self._worker_thread = None
            self._worker = None
            self.controls.step_bar.set_step_status(step_num, "error")
            self._show_error(f"{step_name} failed", f"{error_label}: {msg}")
            self._set_pipeline_busy(False)
            self._progress_bar.hide()
            self._hide_busy_if_idle()

        worker.progress.connect(handle_progress)
        worker.finished.connect(handle_finished)
        worker.error.connect(handle_error)
        thread.started.connect(worker.run)

        self._worker_thread = thread
        self._worker = worker
        thread.start()

    def _start_preview_thread(
        self,
        tag: str,
        work_fn,
        apply_fn,
        fail_label: str,
    ) -> bool:
        """Launch *work_fn()* on a background thread; call *apply_fn(result)* on the main thread.

        *tag* is ``"beat_preview"`` or ``"live_preview"``; the corresponding
        ``_<tag>_thread``, ``_<tag>_worker`` and ``_<tag>_busy`` attributes are
        managed automatically.  Returns ``True`` if the thread was started.
        """
        thread_attr = f"_{tag}_thread"
        worker_attr = f"_{tag}_worker"
        busy_attr = f"_{tag}_busy"

        if getattr(self, thread_attr, None) is not None:
            return False
        if self._worker_thread is not None:
            return False

        worker = _PreviewWorker(work_fn)
        thread = QThread(self)
        worker.moveToThread(thread)

        def _on_finished(result: object) -> None:
            thread.quit()
            thread.wait()
            setattr(self, thread_attr, None)
            setattr(self, worker_attr, None)
            try:
                apply_fn(result)
            except Exception as exc:
                bar = self.statusBar()
                if bar is not None:
                    bar.showMessage(f"{fail_label}: {exc}", 2800)
            finally:
                setattr(self, busy_attr, False)
                self._hide_busy_if_idle()
                # Re-check for queued config changes that arrived while busy
                self._on_controls_changed()

        def _on_error(msg: str) -> None:
            thread.quit()
            thread.wait()
            setattr(self, thread_attr, None)
            setattr(self, worker_attr, None)
            setattr(self, busy_attr, False)
            self._hide_busy_if_idle()
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"{fail_label}: {msg}", 2800)
            # Re-check for queued config changes that arrived while busy
            self._on_controls_changed()

        worker.finished.connect(_on_finished)
        worker.error.connect(_on_error)
        thread.started.connect(worker.run)

        setattr(self, thread_attr, thread)
        setattr(self, worker_attr, worker)
        self._busy_overlay.show_busy(f"{fail_label.replace(' failed', '')}\u2026")
        thread.start()
        return True

    def _hide_busy_if_idle(self) -> None:
        """Hide the busy overlay when no background thread is active."""
        if (
            self._worker_thread is None
            and self._beat_preview_thread is None
            and self._live_preview_thread is None
        ):
            self._busy_overlay.hide_busy()

    def _select_input_file(self) -> str | None:
        filter_str = (
            "Media Files (*.wav *.mp3 *.flac *.ogg *.aac *.wma *.m4a *.mp4 *.mkv *.avi *.webm *.wmv *.mov *.flv);;"
            "All Files (*.*)"
        )
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio or Video", "", filter_str)
        return path or None

    def _select_funscript_file(self) -> str | None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Existing Funscript",
            "",
            "Funscript Files (*.funscript);;All Files (*.*)",
        )
        return path or None

    def _select_funscript_files(self) -> list[str]:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Existing Funscripts",
            "",
            "Funscript Files (*.funscript);;All Files (*.*)",
        )
        return [p for p in paths if p]

    def _select_funscript_folder(self) -> str | None:
        path = QFileDialog.getExistingDirectory(self, "Open Funscript Folder", "")
        return path or None

    def _discover_matching_media(self, script_path: Path, metadata: FunscriptMetadata) -> Path | None:
        candidates: list[Path] = []

        # Try the script's own stem first (works for main files like video.funscript)
        for ext in SUPPORTED_EXTENSIONS:
            candidates.append(script_path.with_suffix(ext))

        # If the stem contains an axis suffix (e.g. video.alpha.funscript),
        # also try the base stem (video.mp4, etc.)
        base_stem, _axis = _strip_axis_suffix(script_path.stem)
        if _axis is not None:
            for ext in SUPPORTED_EXTENSIONS:
                candidates.append(script_path.parent / f"{base_stem}{ext}")

        title = str(metadata.title).strip()
        if title:
            title_path = script_path.with_name(title)
            if title_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                candidates.append(title_path)
            else:
                candidates.extend(title_path.with_suffix(ext) for ext in SUPPORTED_EXTENSIONS)

        # Also search up to 3 parent folders for matching media
        stems_to_try = {script_path.stem}
        if _axis is not None:
            stems_to_try.add(base_stem)
        parent = script_path.parent.parent
        for _ in range(3):
            if parent == parent.parent:
                break  # hit filesystem root
            for stem in stems_to_try:
                for ext in SUPPORTED_EXTENSIONS:
                    candidates.append(parent / f"{stem}{ext}")
            parent = parent.parent

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate).lower()
            if key in seen:
                continue
            seen.add(key)
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _on_open_script_clicked(self) -> None:
        self.open_funscripts(blocking=False)

    def _on_load_folder_clicked(self) -> None:
        self.open_funscript_folder(blocking=False)

    def _on_regen_axes_clicked(self) -> None:
        """Regenerate all auxiliary axes from the current edited main actions."""
        main_actions = self._current_main_actions()
        if len(main_actions) < 2:
            return
        axis_cfg = self.controls.get_axis_config()
        duration_ms = int(main_actions[-1].at) if main_actions else 1
        self._multi_axis = convert_to_2d(main_actions, axis_cfg, max(duration_ms, 1), audio_timeline=self._timeline)
        self.visualizations.set_multi_axis(self._multi_axis)
        self.aux_panel.set_multi_axis(self._multi_axis)
        axis_count = sum(1 for k, v in self._multi_axis.axes.items() if k != "main" and v)
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(f"Regenerated {axis_count} axes from {len(main_actions)} main actions", 3000)
        self._refresh_edit_axis_combo()

    # ------------------------------------------------------------------
    # Fill Blanks
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_blank_regions(
        actions: list[FunscriptAction],
        duration_ms: int,
        window_ms: int = 500,
        motion_threshold: int = 5,
    ) -> list[tuple[int, int]]:
        """Return ``(start_ms, end_ms)`` ranges where the funscript has no meaningful motion.

        A time window is considered *blank* when the peak-to-peak position range
        within it is less than *motion_threshold* (0-100 scale).  Consecutive
        blank windows are merged, and large timestamp gaps between actions are
        also reported as blank.
        """
        if not actions or duration_ms <= 0:
            return [(0, max(duration_ms, 0))]

        # Build per-window position range --------------------------------
        n_windows = max(1, (duration_ms + window_ms - 1) // window_ms)
        win_min = [101] * n_windows
        win_max = [-1] * n_windows

        for a in actions:
            idx = min(int(a.at) // window_ms, n_windows - 1)
            if a.pos < win_min[idx]:
                win_min[idx] = a.pos
            if a.pos > win_max[idx]:
                win_max[idx] = a.pos

        # Mark blank windows (no actions → min stays 101, also blank) ----
        blank = [False] * n_windows
        for i in range(n_windows):
            if win_max[i] < 0:
                # No action fell into this window at all
                blank[i] = True
            elif (win_max[i] - win_min[i]) < motion_threshold:
                blank[i] = True

        # Merge consecutive blank windows into ranges --------------------
        regions: list[tuple[int, int]] = []
        i = 0
        while i < n_windows:
            if blank[i]:
                start = i * window_ms
                while i < n_windows and blank[i]:
                    i += 1
                end = min(i * window_ms, duration_ms)
                regions.append((start, end))
            else:
                i += 1
        return regions

    def _on_fill_blanks_clicked(self) -> None:
        actions = self._current_main_actions()
        if len(actions) < 2 or self._beats is None or self._timeline is None:
            return

        duration_ms = int(actions[-1].at) if actions else 0
        if duration_ms <= 0:
            return

        threshold = self.controls.blank_threshold_spin.value()
        blank_regions = self._detect_blank_regions(actions, duration_ms, motion_threshold=int(threshold))
        if not blank_regions:
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage("No blank regions detected — nothing to fill.", 3000)
            return

        # Invert: blank regions → motion regions (which we lock) ---------
        motion_regions: list[tuple[int, int]] = []
        cursor = 0
        for bstart, bend in sorted(blank_regions):
            if bstart > cursor:
                motion_regions.append((cursor, bstart))
            cursor = max(cursor, bend)
        if cursor < duration_ms:
            motion_regions.append((cursor, duration_ms))

        # Lock motion regions so generate() preserves them ---------------
        self._edit_state.clear_all_locks()
        for mstart, mend in motion_regions:
            self._edit_state.lock_region(int(mstart), int(mend))

        # Run the normal generation pipeline — merge logic fills the gaps
        total_blank_ms = sum(b - a for a, b in blank_regions)
        ok = self.step_4_generate(show_errors=True, blocking=True)

        bar = self.statusBar()
        if bar is not None:
            if ok:
                bar.showMessage(
                    f"Filled {len(blank_regions)} blank region(s) "
                    f"({total_blank_ms / 1000:.1f}s) with beat-driven motion",
                    5000,
                )
            else:
                bar.showMessage("Fill blanks failed — see error above.", 3000)

    def _refresh_edit_axis_combo(self) -> None:
        """Update the axis selector combo with axes that have data."""
        names = ["main"]
        if self._multi_axis is not None:
            for k, v in self._multi_axis.axes.items():
                if k != "main" and v:
                    names.append(k)
        self.aux_panel.update_edit_axis_list(names)

    def _on_edit_axis_switched(self, axis_name: str) -> None:
        """Switch the edit state to a different axis."""
        if axis_name == self._current_edit_axis:
            return
        self._current_edit_axis = axis_name
        if axis_name == "main":
            self.aux_panel.set_edit_state(None)
            self.visualizations.set_edit_state(self._edit_state)
        else:
            if axis_name not in self._aux_edit_states:
                state = FunscriptEditState(self)
                if self._multi_axis and axis_name in self._multi_axis.axes:
                    state.load_actions(list(self._multi_axis.axes[axis_name]))
                self._aux_edit_states[axis_name] = state
            self.aux_panel.set_edit_state(self._aux_edit_states[axis_name])

    @staticmethod
    def _discover_sibling_axes(script_path: Path) -> dict[str, list[FunscriptAction]]:
        """Find sibling axis funscript files and return {axis_name: actions}."""
        from funscript_utils import discover_sibling_axes
        return discover_sibling_axes(script_path)

    @staticmethod
    def _axis_name_from_file(path: Path) -> str | None:
        """Infer axis name from filename; returns None for plain non-axis stems."""
        from funscript_utils import axis_name_from_file
        return axis_name_from_file(path)

    @staticmethod
    def _pick_primary_loaded_entry(
        loaded_entries: list[tuple[Path, list[FunscriptAction], FunscriptMetadata, str | None]],
    ) -> tuple[Path, list[FunscriptAction], FunscriptMetadata, str | None]:
        main_candidates = [entry for entry in loaded_entries if entry[3] in (None, "main")]
        if main_candidates:
            return max(
                main_candidates,
                key=lambda entry: (len(entry[1]), -len(entry[0].name), entry[0].name.lower()),
            )

        electrode_order = {"e1": 0, "e2": 1, "e3": 2, "e4": 3}
        electrode_candidates = [entry for entry in loaded_entries if entry[3] in electrode_order]
        if electrode_candidates:
            return min(
                electrode_candidates,
                key=lambda entry: (
                    electrode_order.get(entry[3] or "", 99),
                    -len(entry[1]),
                    entry[0].name.lower(),
                ),
            )

        return max(
            loaded_entries,
            key=lambda entry: (len(entry[1]), -len(entry[0].name), entry[0].name.lower()),
        )

    def _discover_parent_main_entry(
        self,
        script_paths: list[Path],
        loaded_entries: list[tuple[Path, list[FunscriptAction], FunscriptMetadata, str | None]],
    ) -> tuple[Path, list[FunscriptAction], FunscriptMetadata, str | None] | None:
        if any(entry[3] in (None, "main") for entry in loaded_entries):
            return None

        selected_keys = {str(path).lower() for path in script_paths}
        seen_candidates: set[str] = set()
        candidates: list[tuple[Path, list[FunscriptAction], FunscriptMetadata, str | None]] = []

        for script_path in script_paths:
            base_stem, axis_name = _strip_axis_suffix(script_path.stem)
            if axis_name in (None, "main"):
                continue

            parent_folder = script_path.parent.parent
            if not parent_folder.exists() or not parent_folder.is_dir():
                continue

            candidate = parent_folder / f"{base_stem}.funscript"
            candidate_key = str(candidate).lower()
            if candidate_key in selected_keys or candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)

            if not candidate.exists() or not candidate.is_file():
                continue

            try:
                actions, metadata = read_funscript(candidate)
            except Exception:
                continue

            candidates.append((candidate, actions, metadata, self._axis_name_from_file(candidate)))

        if not candidates:
            return None

        return self._pick_primary_loaded_entry(candidates)

    def _apply_opened_funscript(
        self,
        script_path: Path,
        metadata: FunscriptMetadata,
        actions: list[FunscriptAction],
        matching_media: Path | None,
        samples,
        analysis_sample_rate: int,
    ) -> None:
        # Determine which axis was selected and resolve main actions
        base_stem, selected_axis = _strip_axis_suffix(script_path.stem)
        sibling_axes = self._discover_sibling_axes(script_path)

        # If user opened an axis file (not main), try to use the main file's
        # actions as the primary position timeline; fall back to the selected
        # axis file's actions if the main file isn't available.
        if selected_axis is not None and selected_axis != "main":
            sibling_axes[selected_axis] = [FunscriptAction(a.at, a.pos) for a in actions]
            main_actions_from_file = sibling_axes.pop("main", None)
            if main_actions_from_file:
                actions = main_actions_from_file
            # Else: opened an axis file with no main sibling → use its actions as primary

        copied_actions = [FunscriptAction(a.at, a.pos) for a in actions]
        duration_ms = int(max(metadata.duration, copied_actions[-1].at if copied_actions else 0))

        # Preserve existing analysis / beat state so the user doesn't have to
        # re-run those steps after opening a script on top of already-analysed
        # audio.
        prev_timeline = self._timeline
        prev_beats = self._beats
        prev_samples = self._samples if samples is None else samples

        self._file_path = str(script_path)
        self._media_path = str(matching_media) if matching_media is not None else None
        self._load_video_preview()
        self._samples = prev_samples
        self._cached_orbital_result = None

        # Only clear positions/multi-axis; keep timeline & beats intact.
        self._live_preview_timer.stop()
        self._last_live_preview_signature = None
        self._positions = None
        self._multi_axis = None
        self._edit_state.load_actions([])
        self.visualizations.set_positions(
            PositionTimeline(actions=[], beat_actions=[], speed_profile=np.array([], dtype=np.float64), ml_results=None)
        )
        self.visualizations.set_multi_axis(MultiAxisResult(axes={"main": []}))
        self.aux_panel.set_multi_axis(MultiAxisResult(axes={"main": []}))

        # Restore preserved state
        self._timeline = prev_timeline
        self._beats = prev_beats

        if samples is not None:
            self.visualizations.set_audio_data(samples, analysis_sample_rate)
            self.controls.step_bar.set_step_status(1, "done")
        else:
            self.visualizations.set_audio_data(np.array([], dtype=np.float32), 0)
            self.visualizations.set_duration_hint(duration_ms)
            self.controls.step_bar.set_step_status(1, "ready")

        speed_profile = np.zeros(len(copied_actions), dtype=np.float64)
        self._positions = PositionTimeline(
            actions=[FunscriptAction(a.at, a.pos) for a in copied_actions],
            beat_actions=[FunscriptAction(a.at, a.pos) for a in copied_actions],
            speed_profile=speed_profile,
            ml_results=None,
        )

        # Build multi-axis: start with generated axes, then overlay any
        # sibling axis files that were found on disk.
        axis_duration_ms = max(duration_ms, 1)
        self._multi_axis = convert_to_2d(self._positions.actions, self.controls.get_axis_config(), axis_duration_ms, audio_timeline=self._timeline)
        for axis_name, axis_actions in sibling_axes.items():
            self._multi_axis.axes[axis_name] = [FunscriptAction(a.at, a.pos) for a in axis_actions]

        self._edit_state.load_actions(self._positions.actions)
        self._aux_edit_states.clear()
        self._current_edit_axis = "main"
        self.visualizations.set_positions(self._positions)
        self.visualizations.set_multi_axis(self._multi_axis)
        self.aux_panel.set_multi_axis(self._multi_axis)
        self._refresh_edit_axis_combo()
        self.visualizations.set_playback_position(0.0)
        if duration_ms > 0:
            self.visualizations.zoom_to_range(0.0, max(1000.0, float(duration_ms)))

        # Build file label showing loaded axes
        axis_count = sum(1 for k, v in self._multi_axis.axes.items() if k != "main" and v)
        axis_info = f" (+{axis_count} axes)" if axis_count > 0 else ""
        label = f"{script_path.name}{axis_info}"
        if matching_media is not None:
            label = f"{script_path.name}{axis_info} | {matching_media.name}"
        self.file_label.setText(label)
        tooltip = str(script_path)
        if matching_media is not None:
            tooltip = f"Script: {script_path}\nMedia: {matching_media}"
        self.file_label.setToolTip(tooltip)

        for step in (5,):
            self.controls.step_bar.set_step_status(step, "ready")
        self.controls.step_bar.set_step_status(4, "done")
        # Reflect preserved analysis/beat state in the step bar
        self.controls.step_bar.set_step_status(2, "done" if self._timeline is not None else "ready")
        self.controls.step_bar.set_step_status(3, "done" if self._beats is not None else "ready")
        self._last_live_preview_signature = self._live_preview_signature()
        self._refresh_step_availability()

        bar = self.statusBar()
        if bar is not None:
            parts = [f"Opened {script_path.name} ({len(copied_actions)} points)"]
            if axis_count > 0:
                axis_names = sorted(k for k, v in self._multi_axis.axes.items() if k != "main" and v)
                parts.append(f"+ {axis_count} axes: {', '.join(axis_names)}")
            if matching_media is not None:
                parts.append(f"media: {matching_media.name}")
            else:
                parts.append("(no matching media found)")
            bar.showMessage(" | ".join(parts), 6000)

    def load_converted_preview(
        self,
        axes: dict[str, list[FunscriptAction]],
        *,
        base_name: str = "converted",
        source_folder: Path | str | None = None,
    ) -> bool:
        preview_axes = {
            axis_name: [FunscriptAction(a.at, a.pos) for a in actions]
            for axis_name, actions in axes.items()
            if actions
        }
        if not preview_axes:
            return False

        primary_axis = next(
            (axis_name for axis_name in ("e1", "e2", "e3", "e4") if axis_name in preview_axes),
            None,
        )
        if primary_axis is None:
            primary_axis = max(
                preview_axes,
                key=lambda axis_name: (len(preview_axes[axis_name]), -len(axis_name), axis_name),
            )

        primary_actions = [FunscriptAction(a.at, a.pos) for a in preview_axes[primary_axis]]
        duration_ms = max(
            (int(actions[-1].at) for actions in preview_axes.values() if actions),
            default=(primary_actions[-1].at if primary_actions else 0),
        )

        analysis_cfg = self.controls.get_analysis_config()
        matching_media: Path | None = None
        samples = None

        if source_folder is not None:
            probe_folder = Path(source_folder)
            probe_script = probe_folder / f"{base_name}.funscript"
            probe_meta = FunscriptMetadata(title=base_name, duration=duration_ms)
            matching_media = self._discover_matching_media(probe_script, probe_meta)

            if matching_media is not None:
                if self._samples is not None and self._media_path is not None and Path(self._media_path) == matching_media:
                    samples = self._samples
                else:
                    try:
                        samples = load_audio(str(matching_media), analysis_cfg, lambda *_args: None)
                    except Exception:
                        samples = None
                        matching_media = None

        self._file_path = None
        self._media_path = str(matching_media) if matching_media is not None else None
        self._load_video_preview()
        self._samples = samples
        self._timeline = None
        self._beats = None
        self._cached_orbital_result = None

        self._live_preview_timer.stop()
        self._last_live_preview_signature = None
        self._positions = None
        self._multi_axis = None
        self._edit_state.load_actions([])
        if samples is not None:
            self.visualizations.set_audio_data(samples, int(analysis_cfg.sample_rate))
        else:
            self.visualizations.set_audio_data(np.array([], dtype=np.float32), 0)
            self.visualizations.set_duration_hint(duration_ms)

        speed_profile = np.zeros(len(primary_actions), dtype=np.float64)
        self._positions = PositionTimeline(
            actions=[FunscriptAction(a.at, a.pos) for a in primary_actions],
            beat_actions=[FunscriptAction(a.at, a.pos) for a in primary_actions],
            speed_profile=speed_profile,
            ml_results=None,
        )

        result_axes = {"main": [FunscriptAction(a.at, a.pos) for a in primary_actions]}
        for axis_name, axis_actions in preview_axes.items():
            result_axes[axis_name] = [FunscriptAction(a.at, a.pos) for a in axis_actions]

        self._multi_axis = MultiAxisResult(axes=result_axes)
        self._edit_state.load_actions(self._positions.actions)
        self._aux_edit_states.clear()
        self._current_edit_axis = "main"
        self.visualizations.set_positions(self._positions)
        self.visualizations.set_multi_axis(self._multi_axis)
        self.aux_panel.set_multi_axis(self._multi_axis)
        self._refresh_edit_axis_combo()

        default_edit_axis = next(
            (
                axis_name
                for axis_name in (
                    "e1",
                    "e2",
                    "e3",
                    "e4",
                    "pulse_frequency",
                    "carrier_frequency",
                    "frequency",
                )
                if axis_name in preview_axes
            ),
            None,
        )
        if default_edit_axis is not None:
            self.aux_panel.select_edit_axis(default_edit_axis)

        self.visualizations.set_playback_position(0.0)
        if duration_ms > 0:
            self.visualizations.zoom_to_range(0.0, max(1000.0, float(duration_ms)))

        axis_names = sorted(preview_axes.keys())
        label = f"Converted preview: {base_name} ({len(axis_names)} axes)"
        if matching_media is not None:
            label = f"{label} | {matching_media.name}"
        self.file_label.setText(label)

        tooltip = f"Converted preview: {base_name}\nPrimary: {primary_axis}\nAxes: {', '.join(axis_names)}"
        if matching_media is not None:
            tooltip += f"\nMedia: {matching_media}"
        self.file_label.setToolTip(tooltip)

        self.controls.step_bar.set_step_status(1, "done" if matching_media is not None else "ready")
        self.controls.step_bar.set_step_status(2, "ready")
        self.controls.step_bar.set_step_status(3, "ready")
        self.controls.step_bar.set_step_status(4, "done")
        self.controls.step_bar.set_step_status(5, "ready")
        self._refresh_step_availability()

        bar = self.statusBar()
        if bar is not None:
            parts = [
                f"Loaded converted preview {base_name}",
                f"primary: {primary_axis}",
                f"axes: {', '.join(axis_names)}",
            ]
            if matching_media is not None:
                parts.append(f"media: {matching_media.name}")
            bar.showMessage(" | ".join(parts), 7000)
        return True

    def open_funscript(
        self,
        file_path: str | None = None,
        show_errors: bool = True,
        *,
        blocking: bool = True,
    ) -> bool:
        path = file_path or self._select_funscript_file()
        if not path:
            return False

        script_path = Path(path)
        if script_path.suffix.lower() not in FUNSCRIPT_EXTENSIONS:
            self._show_error("Unsupported file", "Selected file is not a funscript.", show_errors)
            return False

        try:
            actions, metadata = read_funscript(script_path)
        except Exception as exc:
            self._show_error("Open failed", f"Unable to read funscript: {exc}", show_errors)
            return False

        preset = metadata.parameters.get("preset") if isinstance(metadata.parameters, dict) else None
        if isinstance(preset, dict):
            self.controls.set_from_preset(preset)

        analysis_cfg = self.controls.get_analysis_config()
        matching_media = self._discover_matching_media(script_path, metadata)

        # Fall back to the currently-loaded media when the funscript lives in a
        # different folder (e.g. Downloads) and no sibling media was found.
        if matching_media is None and self._media_path is not None:
            existing = Path(self._media_path)
            if existing.exists() and existing.is_file():
                matching_media = existing

        def compute(progress_cb):
            if matching_media is None:
                return None
            # Re-use already-loaded samples when the media path hasn't changed.
            if self._samples is not None and self._media_path is not None and Path(self._media_path) == matching_media:
                return self._samples
            progress_cb("Loading matching media", 5.0)
            return load_audio(str(matching_media), analysis_cfg, progress_cb)

        def apply(samples):
            self._apply_opened_funscript(
                script_path,
                metadata,
                actions,
                matching_media,
                samples,
                int(analysis_cfg.sample_rate),
            )

        if not blocking:
            if matching_media is None:
                apply(None)
                return True
            self._run_step_async(1, "Open Script", compute, apply, "Unable to open script")
            return True

        self.controls.step_bar.set_step_status(1, "running")
        self._set_pipeline_busy(True, "Open Script: starting...")
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        try:
            samples = compute(lambda msg, pct: self._progress("Open Script", msg, pct))
            apply(samples)
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(1, "error")
            self._show_error("Open failed", f"Unable to open script: {exc}", show_errors)
            return False
        finally:
            self._set_pipeline_busy(False)
            self._progress_bar.hide()

    def open_funscripts(
        self,
        file_paths: list[str] | None = None,
        show_errors: bool = True,
        *,
        blocking: bool = True,
    ) -> bool:
        paths = [p for p in (file_paths or self._select_funscript_files()) if p]
        if not paths:
            return False

        if len(paths) == 1:
            return self.open_funscript(paths[0], show_errors=show_errors, blocking=blocking)

        script_paths = sorted(
            [Path(p) for p in paths if Path(p).suffix.lower() in FUNSCRIPT_EXTENSIONS],
            key=lambda p: p.name.lower(),
        )
        if not script_paths:
            self._show_error("Unsupported file", "No selected files are funscripts.", show_errors)
            return False

        loaded_entries: list[tuple[Path, list[FunscriptAction], FunscriptMetadata, str | None]] = []
        read_failures = 0
        for script in script_paths:
            try:
                actions, metadata = read_funscript(script)
            except Exception:
                read_failures += 1
                continue
            loaded_entries.append((script, actions, metadata, self._axis_name_from_file(script)))

        if not loaded_entries:
            self._show_error("Open failed", "Unable to read any selected funscript files.", show_errors)
            return False

        parent_main_entry = self._discover_parent_main_entry(script_paths, loaded_entries)
        if parent_main_entry is not None:
            loaded_entries.append(parent_main_entry)

        primary_script, primary_actions, primary_meta, primary_axis = self._pick_primary_loaded_entry(loaded_entries)

        if isinstance(primary_meta.parameters, dict):
            preset = primary_meta.parameters.get("preset")
            if isinstance(preset, dict):
                self.controls.set_from_preset(preset)

        overlays: dict[str, list[FunscriptAction]] = {}
        for script, actions, _meta, axis_name in loaded_entries:
            if not actions:
                continue
            if script == primary_script:
                continue
            if axis_name in (None, "main"):
                continue
            copied = [FunscriptAction(a.at, a.pos) for a in actions]
            existing = overlays.get(axis_name)
            if existing is None or len(copied) > len(existing):
                overlays[axis_name] = copied

        copied_primary = [FunscriptAction(a.at, a.pos) for a in primary_actions]
        if not copied_primary:
            if overlays:
                fallback_axis = max(overlays, key=lambda k: len(overlays[k]))
                copied_primary = [FunscriptAction(a.at, a.pos) for a in overlays.pop(fallback_axis)]
                primary_axis = fallback_axis
            else:
                self._show_error("Open failed", "No action data found in selected scripts.", show_errors)
                return False

        analysis_cfg = self.controls.get_analysis_config()
        matching_media = self._discover_matching_media(primary_script, primary_meta)

        if matching_media is None and self._media_path is not None:
            existing = Path(self._media_path)
            if existing.exists() and existing.is_file():
                matching_media = existing

        def compute(progress_cb):
            if matching_media is None:
                return None
            if self._samples is not None and self._media_path is not None and Path(self._media_path) == matching_media:
                return self._samples
            progress_cb("Loading matching media", 5.0)
            return load_audio(str(matching_media), analysis_cfg, progress_cb)

        def apply(samples):
            self._apply_opened_funscript(
                primary_script,
                primary_meta,
                copied_primary,
                matching_media,
                samples,
                int(analysis_cfg.sample_rate),
            )
            multi_axis = self._multi_axis
            if multi_axis is None:
                return
            for axis_name, axis_actions in overlays.items():
                multi_axis.axes[axis_name] = [FunscriptAction(a.at, a.pos) for a in axis_actions]
            self.visualizations.set_multi_axis(multi_axis)
            self.aux_panel.set_multi_axis(multi_axis)
            self._refresh_edit_axis_combo()

            axis_count = sum(1 for k, v in multi_axis.axes.items() if k != "main" and v)
            axis_info = f" (+{axis_count} axes)" if axis_count > 0 else ""
            source_label = f"selected: {len(script_paths)} files"
            if primary_axis not in (None, "main"):
                source_label += f" | main from {primary_axis}"
            label = f"{source_label}{axis_info}"
            if matching_media is not None:
                label = f"{label} | {matching_media.name}"
            self.file_label.setText(label)
            tooltip = f"Selection ({len(script_paths)} files)\nPrimary: {primary_script}"
            if matching_media is not None:
                tooltip += f"\nMedia: {matching_media}"
            self.file_label.setToolTip(tooltip)

            bar = self.statusBar()
            if bar is not None:
                parts = [
                    f"Loaded selected scripts: {len(script_paths)}",
                    f"readable: {len(loaded_entries)}",
                    f"main points: {len(copied_primary)}",
                ]
                if axis_count > 0:
                    axis_names = sorted(k for k, v in multi_axis.axes.items() if k != "main" and v)
                    parts.append(f"axes: {', '.join(axis_names)}")
                if read_failures > 0:
                    parts.append(f"skipped unreadable: {read_failures}")
                bar.showMessage(" | ".join(parts), 7000)

        if not blocking:
            if matching_media is None:
                apply(None)
                return True
            self._run_step_async(1, "Open Script Selection", compute, apply, "Unable to open selected scripts")
            return True

        self.controls.step_bar.set_step_status(1, "running")
        self._set_pipeline_busy(True, "Open Script Selection: starting...")
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        try:
            samples = compute(lambda msg, pct: self._progress("Open Script Selection", msg, pct))
            apply(samples)
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(1, "error")
            self._show_error("Open failed", f"Unable to open selected scripts: {exc}", show_errors)
            return False
        finally:
            self._set_pipeline_busy(False)
            self._progress_bar.hide()

    def open_funscript_folder(
        self,
        folder_path: str | None = None,
        show_errors: bool = True,
        *,
        blocking: bool = True,
    ) -> bool:
        path = folder_path or self._select_funscript_folder()
        if not path:
            return False

        folder = Path(path)
        if not folder.exists() or not folder.is_dir():
            self._show_error("Invalid folder", "Selected path is not a folder.", show_errors)
            return False

        script_paths = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in FUNSCRIPT_EXTENSIONS],
            key=lambda p: p.name.lower(),
        )
        if not script_paths:
            self._show_error("No scripts found", "Folder contains no .funscript files.", show_errors)
            return False

        loaded_entries: list[tuple[Path, list[FunscriptAction], FunscriptMetadata, str | None]] = []
        read_failures = 0
        for script in script_paths:
            try:
                actions, metadata = read_funscript(script)
            except Exception:
                read_failures += 1
                continue
            loaded_entries.append((script, actions, metadata, self._axis_name_from_file(script)))

        if not loaded_entries:
            self._show_error("Open failed", "Unable to read any funscript files in folder.", show_errors)
            return False

        parent_main_entry = self._discover_parent_main_entry(script_paths, loaded_entries)
        if parent_main_entry is not None:
            loaded_entries.append(parent_main_entry)

        primary_script, primary_actions, primary_meta, primary_axis = self._pick_primary_loaded_entry(loaded_entries)

        if isinstance(primary_meta.parameters, dict):
            preset = primary_meta.parameters.get("preset")
            if isinstance(preset, dict):
                self.controls.set_from_preset(preset)

        overlays: dict[str, list[FunscriptAction]] = {}
        for script, actions, _meta, axis_name in loaded_entries:
            if not actions:
                continue
            if script == primary_script:
                continue
            if axis_name in (None, "main"):
                continue
            copied = [FunscriptAction(a.at, a.pos) for a in actions]
            existing = overlays.get(axis_name)
            if existing is None or len(copied) > len(existing):
                overlays[axis_name] = copied

        copied_primary = [FunscriptAction(a.at, a.pos) for a in primary_actions]
        if not copied_primary:
            if overlays:
                fallback_axis = max(overlays, key=lambda k: len(overlays[k]))
                copied_primary = [FunscriptAction(a.at, a.pos) for a in overlays.pop(fallback_axis)]
                primary_axis = fallback_axis
            else:
                self._show_error("Open failed", "No action data found in folder scripts.", show_errors)
                return False

        analysis_cfg = self.controls.get_analysis_config()
        matching_media = self._discover_matching_media(primary_script, primary_meta)

        if matching_media is None and self._media_path is not None:
            existing = Path(self._media_path)
            if existing.exists() and existing.is_file():
                matching_media = existing

        def compute(progress_cb):
            if matching_media is None:
                return None
            if self._samples is not None and self._media_path is not None and Path(self._media_path) == matching_media:
                return self._samples
            progress_cb("Loading matching media", 5.0)
            return load_audio(str(matching_media), analysis_cfg, progress_cb)

        def apply(samples):
            self._apply_opened_funscript(
                primary_script,
                primary_meta,
                copied_primary,
                matching_media,
                samples,
                int(analysis_cfg.sample_rate),
            )
            multi_axis = self._multi_axis
            if multi_axis is None:
                return
            for axis_name, axis_actions in overlays.items():
                multi_axis.axes[axis_name] = [FunscriptAction(a.at, a.pos) for a in axis_actions]
            self.visualizations.set_multi_axis(multi_axis)
            self.aux_panel.set_multi_axis(multi_axis)
            self._refresh_edit_axis_combo()

            axis_count = sum(1 for k, v in multi_axis.axes.items() if k != "main" and v)
            axis_info = f" (+{axis_count} axes)" if axis_count > 0 else ""
            source_label = f"folder: {folder.name}"
            if primary_axis not in (None, "main"):
                source_label += f" | main from {primary_axis}"
            label = f"{source_label}{axis_info}"
            if matching_media is not None:
                label = f"{label} | {matching_media.name}"
            self.file_label.setText(label)
            tooltip = f"Folder: {folder}\nPrimary: {primary_script}"
            if matching_media is not None:
                tooltip += f"\nMedia: {matching_media}"
            self.file_label.setToolTip(tooltip)

            bar = self.statusBar()
            if bar is not None:
                parts = [
                    f"Loaded folder {folder.name}",
                    f"scripts: {len(loaded_entries)}",
                    f"main points: {len(copied_primary)}",
                ]
                if axis_count > 0:
                    axis_names = sorted(k for k, v in multi_axis.axes.items() if k != "main" and v)
                    parts.append(f"axes: {', '.join(axis_names)}")
                if read_failures > 0:
                    parts.append(f"skipped unreadable: {read_failures}")
                bar.showMessage(" | ".join(parts), 7000)

        if not blocking:
            if matching_media is None:
                apply(None)
                return True
            self._run_step_async(1, "Open Script Folder", compute, apply, "Unable to open script folder")
            return True

        self.controls.step_bar.set_step_status(1, "running")
        self._set_pipeline_busy(True, "Open Script Folder: starting...")
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        try:
            samples = compute(lambda msg, pct: self._progress("Open Script Folder", msg, pct))
            apply(samples)
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(1, "error")
            self._show_error("Open failed", f"Unable to open script folder: {exc}", show_errors)
            return False
        finally:
            self._set_pipeline_busy(False)
            self._progress_bar.hide()

    def _refresh_step_availability(self) -> None:
        self.controls.step_bar.set_step_enabled(1, True)
        self.controls.step_bar.set_step_enabled(2, self._samples is not None)
        self.controls.step_bar.set_step_enabled(3, self._timeline is not None)
        self.controls.step_bar.set_step_enabled(4, self._beats is not None and len(self._beats.beats) > 0)
        current_actions = self._current_main_actions()
        has_exportable = (
            len(current_actions) > 0
            or (self._positions is not None and len(self._positions.beat_actions) > 0)
        )
        self.controls.step_bar.set_step_enabled(5, has_exportable)
        self.regen_axes_btn.setEnabled(len(current_actions) >= 2)
        self.fill_blanks_btn.setEnabled(
            len(current_actions) >= 2
            and self._beats is not None
            and len(self._beats.beats) > 0
        )

    def _on_edit_state_changed(self) -> None:
        self._refresh_step_availability()

    def _current_main_actions(self) -> list[FunscriptAction]:
        if self._edit_state.version > 0:
            return [FunscriptAction(a.at, a.pos) for a in self._edit_state.actions]
        if self._positions is None:
            return []
        return [FunscriptAction(a.at, a.pos) for a in self._positions.actions]

    @staticmethod
    def _merge_generated_actions_with_locked_regions(
        generated_actions: list[FunscriptAction],
        locked_actions: list[FunscriptAction],
        locked_regions: list[LockedRegion],
    ) -> list[FunscriptAction]:
        if not locked_regions:
            return [FunscriptAction(a.at, a.pos) for a in generated_actions]

        def is_locked(time_ms: int) -> bool:
            return any(region.start_ms <= time_ms <= region.end_ms for region in locked_regions)

        merged = [
            FunscriptAction(a.at, a.pos) for a in generated_actions if not is_locked(int(a.at))
        ]
        merged.extend(FunscriptAction(a.at, a.pos) for a in locked_actions if is_locked(int(a.at)))
        merged.sort(key=lambda action: action.at)
        return merged

    def _live_preview_signature(self) -> str:
        payload = self.controls.to_preset()
        subset = {
            "mapping": payload.get("mapping", {}),
            "ml": payload.get("ml", {}),
            "axis": payload.get("axis", {}),
            "automap": payload.get("automap", {}),
        }
        return json.dumps(subset, sort_keys=True)

    def _schedule_live_preview(self) -> None:
        if self._positions is None:
            return
        if self._timeline is None or self._beats is None or len(self._beats.beats) <= 0:
            return
        sig = self._live_preview_signature()
        if sig == self._last_live_preview_signature:
            return
        # Invalidate any in-flight preview so its stale result is discarded
        self._last_live_preview_signature = None
        self._live_preview_timer.start()

    def _run_live_preview(self, *, blocking: bool = False) -> None:
        if self._live_preview_busy:
            return
        if self._positions is None:
            return
        if self._timeline is None or self._beats is None or len(self._beats.beats) <= 0:
            return

        sig = self._live_preview_signature()
        if sig == self._last_live_preview_signature:
            return

        mapping_cfg = self.controls.get_mapping_config()
        axis_cfg = self.controls.get_axis_config()
        automap_cfg = self.controls.get_automap_config()

        if mapping_cfg.overflow_mode == "bounce" and mapping_cfg.pos_min == mapping_cfg.pos_max:
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage("Live preview blocked: Bounce overflow requires Position Min != Position Max", 2800)
            return

        # Keep slider interaction responsive; live preview skips expensive automap optimization.
        if automap_cfg.enabled:
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage("Live preview paused while Automap is enabled (click Generate to apply)", 2400)
            return

        self._live_preview_busy = True
        timeline = self._timeline
        beats = self._beats
        preview_sig = sig  # capture for staleness check

        def work():
            positions = generate_positions(timeline, beats, mapping_cfg)
            axis_source_actions = positions.beat_actions if len(positions.beat_actions) >= 2 else positions.actions
            multi_axis = convert_to_2d(
                axis_source_actions,
                axis_cfg,
                duration_ms=int(timeline.duration_ms),
                audio_timeline=timeline,
            )
            orbital_result = None
            blend = 1.0 if axis_cfg.alpha_beta_mode == "orbital" else axis_cfg.orbital_blend
            if blend > 0.0 and timeline is not None and beats is not None:
                analysis_cfg = self.controls.get_analysis_config()
                multi_axis, orbital_result = _apply_orbital_overlay(
                    multi_axis, timeline, beats, analysis_cfg, blend=blend,
                    cached_orbital=self._cached_orbital_result,
                )
            return positions, multi_axis, orbital_result

        def apply(result):
            # Discard if config changed while we were computing
            if self._live_preview_signature() != preview_sig:
                return
            # Guard: never overwrite dirty edits with a preview result
            if self._edit_state.dirty:
                return
            positions, multi_axis, orbital_result = result
            if orbital_result is not None:
                self._cached_orbital_result = orbital_result
            self._positions = positions
            self._multi_axis = multi_axis
            self.visualizations.set_positions(positions)
            self.visualizations.set_multi_axis(multi_axis)
            self.aux_panel.set_multi_axis(multi_axis)
            self._refresh_edit_axis_combo()
            self._last_live_preview_signature = preview_sig
            self._refresh_step_availability()
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage("Live preview updated", 1200)

        if blocking:
            try:
                apply(work())
            except Exception as exc:
                bar = self.statusBar()
                if bar is not None:
                    bar.showMessage(f"Live preview failed: {exc}", 2800)
            finally:
                self._live_preview_busy = False
            return

        if not self._start_preview_thread(
            "live_preview",
            work,
            apply,
            "Live preview failed",
        ):
            self._live_preview_busy = False

    def _beat_preview_signature(self) -> str:
        payload = self.controls.to_preset()
        subset = {"beat_detection": payload.get("beat_detection", {})}
        return json.dumps(subset, sort_keys=True)

    def _schedule_beat_preview(self) -> None:
        if self._timeline is None:
            return
        sig = self._beat_preview_signature()
        if sig == self._last_beat_preview_signature:
            return
        # Invalidate any in-flight preview so its stale result is discarded
        self._last_beat_preview_signature = None
        self._beat_preview_timer.start()

    def _run_beat_preview(self) -> None:
        if self._beat_preview_busy:
            return
        if self._timeline is None:
            return
        if self._edit_state.dirty:
            return

        sig = self._beat_preview_signature()
        if sig == self._last_beat_preview_signature:
            return

        self._beat_preview_busy = True
        beat_cfg = self.controls.get_beat_config()
        timeline = self._timeline
        preview_sig = sig  # capture for staleness check

        def apply(beats):
            # Discard if config changed while we were computing
            if self._beat_preview_signature() != preview_sig:
                return
            if self._edit_state.dirty:
                return
            self._beats = beats
            self._reset_from_step(4)
            self.visualizations.set_beats(beats)
            self._last_beat_preview_signature = preview_sig
            self._refresh_step_availability()
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(
                    f"Beat preview: {len(beats.beats)} beats @ {beats.tempo_bpm:.1f} BPM",
                    1500,
                )

        if not self._start_preview_thread(
            "beat_preview",
            lambda: detect_beats(timeline, beat_cfg),
            apply,
            "Beat preview failed",
        ):
            self._beat_preview_busy = False

    def _reset_from_step(self, step: int) -> None:
        if step <= 3:
            self._beat_preview_timer.stop()
            self._last_beat_preview_signature = None
        if step <= 4:
            self._live_preview_timer.stop()
            self._last_live_preview_signature = None

        if step <= 2:
            self._timeline = None
        if step <= 3:
            self._beats = None
            self.visualizations.set_beats(BeatTimeline(beats=[], tempo_bpm=0.0, tempo_confidence=0.0, beat_period_ms=0.0))
        if step <= 4:
            self._positions = None
            self._multi_axis = None
            self._edit_state.load_actions([])
            self.visualizations.set_positions(
                PositionTimeline(
                    actions=[],
                    beat_actions=[],
                    speed_profile=np.array([], dtype=np.float64),
                    ml_results=None,
                )
            )
            self.visualizations.set_multi_axis(MultiAxisResult(axes={"main": []}))
            self.aux_panel.set_multi_axis(MultiAxisResult(axes={"main": []}))

        for idx in range(max(2, step), 6):
            self.controls.step_bar.set_step_status(idx, "ready")
        self._refresh_step_availability()

    def step_1_load_audio(self, file_path: str | None = None, show_errors: bool = True, *, blocking: bool = True) -> bool:
        path = file_path or self._select_input_file()
        if not path:
            return False
        if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
            self._show_error("Unsupported file", "Selected file type is not supported.", show_errors)
            return False

        analysis_cfg = self.controls.get_analysis_config()

        def compute(progress_cb):
            return load_audio(path, analysis_cfg, progress_cb)

        def apply(samples):
            self._file_path = str(path)
            self._media_path = str(path)
            self._load_video_preview()
            self._samples = samples
            self._reset_from_step(2)
            self.file_label.setText(Path(path).name)
            self.file_label.setToolTip(str(path))
            self.visualizations.set_audio_data(samples, int(analysis_cfg.sample_rate))
            self.visualizations.set_positions(
                PositionTimeline(
                    actions=[],
                    beat_actions=[],
                    speed_profile=np.array([], dtype=np.float64),
                    ml_results=None,
                )
            )
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Loaded {Path(path).name}", 2500)

        if not blocking:
            self._run_step_async(1, "Load", compute, apply, "Unable to load media")
            return True

        self.controls.step_bar.set_step_status(1, "running")
        self._set_pipeline_busy(True, "Load: starting...")
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        try:
            result = compute(lambda msg, pct: self._progress("Load", msg, pct))
            apply(result)
            self.controls.step_bar.set_step_status(1, "done")
            self.controls.on_step_completed(1)
            self._refresh_step_availability()
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(1, "error")
            self._show_error("Load failed", f"Unable to load media: {exc}", show_errors)
            return False
        finally:
            self._set_pipeline_busy(False)
            self._progress_bar.hide()

    def step_2_analyze(self, show_errors: bool = True, *, blocking: bool = True) -> bool:
        if self._samples is None:
            self._show_error("Analyze blocked", "Load audio before analysis.", show_errors)
            return False

        analysis_cfg = self.controls.get_analysis_config()
        samples = self._samples

        def compute(progress_cb):
            return analyze_full_file(samples, analysis_cfg, progress_cb)

        def apply(timeline):
            self._timeline = timeline
            self._reset_from_step(3)
            self.visualizations.set_features(timeline)

        if not blocking:
            self._run_step_async(2, "Analyze", compute, apply, "Audio analysis failed")
            return True

        self.controls.step_bar.set_step_status(2, "running")
        self._set_pipeline_busy(True, "Analyze: starting...")
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        try:
            result = compute(lambda msg, pct: self._progress("Analyze", msg, pct))
            apply(result)
            self.controls.step_bar.set_step_status(2, "done")
            self.controls.on_step_completed(2)
            self._refresh_step_availability()
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(2, "error")
            self._show_error("Analyze failed", f"Audio analysis failed: {exc}", show_errors)
            return False
        finally:
            self._set_pipeline_busy(False)
            self._progress_bar.hide()

    def step_3_detect_beats(self, show_errors: bool = True, *, blocking: bool = True) -> bool:
        if self._timeline is None:
            self._show_error("Beat detection blocked", "Run analysis before beat detection.", show_errors)
            return False

        beat_cfg = self.controls.get_beat_config()
        timeline = self._timeline

        def compute(progress_cb):
            return detect_beats(timeline, beat_cfg, progress_cb)

        def apply(beats):
            self._beats = beats
            # Only reset the live-preview timer; preserve any loaded positions
            # so that opening a script + detecting beats can happen in any order.
            self._live_preview_timer.stop()
            self._last_live_preview_signature = None
            if self._positions is None:
                self._positions = None
                self._multi_axis = None
                self._edit_state.load_actions([])
                self.visualizations.set_positions(
                    PositionTimeline(actions=[], beat_actions=[], speed_profile=np.array([], dtype=np.float64), ml_results=None)
                )
                self.visualizations.set_multi_axis(MultiAxisResult(axes={"main": []}))
                self.aux_panel.set_multi_axis(MultiAxisResult(axes={"main": []}))
                for idx in range(4, 6):
                    self.controls.step_bar.set_step_status(idx, "ready")
            self._last_beat_preview_signature = self._beat_preview_signature()
            self.visualizations.set_beats(beats)
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Detected {len(beats.beats)} beats @ {beats.tempo_bpm:.1f} BPM", 3000)

        if not blocking:
            self._run_step_async(3, "Beats", compute, apply, "Beat detection failed")
            return True

        self.controls.step_bar.set_step_status(3, "running")
        self._set_pipeline_busy(True, "Beats: starting...")
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        try:
            result = compute(lambda msg, pct: self._progress("Beats", msg, pct))
            apply(result)
            self.controls.step_bar.set_step_status(3, "done")
            self.controls.on_step_completed(3)
            self._refresh_step_availability()
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(3, "error")
            self._show_error("Beat detection failed", f"Beat detection failed: {exc}", show_errors)
            return False
        finally:
            self._set_pipeline_busy(False)
            self._progress_bar.hide()

    def step_4_generate(self, show_errors: bool = True, *, blocking: bool = True) -> bool:
        if self._timeline is None or self._beats is None:
            self._show_error("Generate blocked", "Analyze audio and detect beats before generation.", show_errors)
            return False

        mapping_cfg = self.controls.get_mapping_config()
        axis_cfg = self.controls.get_axis_config()
        automap_cfg = self.controls.get_automap_config()

        if mapping_cfg.overflow_mode == "bounce" and mapping_cfg.pos_min == mapping_cfg.pos_max:
            self._show_error("Generate blocked", "Bounce overflow requires Position Min and Position Max to be different.", show_errors)
            return False

        timeline = self._timeline
        beats = self._beats
        locked_regions = [LockedRegion(r.start_ms, r.end_ms) for r in self._edit_state.locked_regions]
        locked_actions = [FunscriptAction(a.at, a.pos) for a in self._edit_state.get_locked_actions()]
        use_undoable_accept = self._edit_state.version > 0

        # Snapshot locked multi-axis actions so they survive regeneration
        locked_multi_axis: dict[str, list[FunscriptAction]] = {}
        if locked_regions and self._multi_axis is not None:
            for ax_name, ax_actions in self._multi_axis.axes.items():
                la = [FunscriptAction(a.at, a.pos) for a in ax_actions
                      if any(r.start_ms <= int(a.at) <= r.end_ms for r in locked_regions)]
                if la:
                    locked_multi_axis[ax_name] = la

        def compute(progress_cb):
            nonlocal mapping_cfg
            if automap_cfg.enabled:
                mapping_cfg = automap_optimize(
                    timeline, beats, mapping_cfg, automap_cfg,
                    lambda msg, pct: progress_cb(f"Automap: {msg}", pct * 0.4),
                )
            positions = generate_positions(
                timeline, beats, mapping_cfg,
                lambda msg, pct: progress_cb(f"Generate: {msg}", 40 + pct * 0.4),
            )
            final_actions = self._merge_generated_actions_with_locked_regions(
                positions.actions,
                locked_actions,
                locked_regions,
            )
            # Use sparse beat-level actions for 2D conversion (restim
            # semicircle arcs need large deltas between consecutive points).
            axis_source_actions = positions.beat_actions if len(positions.beat_actions) >= 2 else final_actions
            multi_axis = convert_to_2d(
                axis_source_actions, axis_cfg,
                duration_ms=int(timeline.duration_ms),
                progress_callback=lambda msg, pct: progress_cb(f"Axis: {msg}", 80 + pct * 0.15),
                audio_timeline=timeline,
            )
            blend = 1.0 if axis_cfg.alpha_beta_mode == "orbital" else axis_cfg.orbital_blend
            if blend > 0.0 and timeline is not None and beats is not None:
                analysis_cfg = self.controls.get_analysis_config()
                multi_axis, orbital_result = _apply_orbital_overlay(
                    multi_axis, timeline, beats, analysis_cfg, blend=blend,
                    progress_callback=lambda msg, pct: progress_cb(f"Orbital: {msg}", 95 + pct * 0.05),
                )
                self._cached_orbital_result = orbital_result
            # Merge locked multi-axis actions back into freshly generated axes
            if locked_multi_axis:
                for ax_name, prev_locked in locked_multi_axis.items():
                    new_actions = multi_axis.axes.get(ax_name, [])
                    multi_axis.axes[ax_name] = self._merge_generated_actions_with_locked_regions(
                        new_actions, prev_locked, locked_regions,
                    )
            if final_actions != positions.actions:
                positions = PositionTimeline(
                    actions=final_actions,
                    beat_actions=[FunscriptAction(a.at, a.pos) for a in positions.beat_actions],
                    speed_profile=positions.speed_profile,
                    ml_results=positions.ml_results,
                )
            return positions, multi_axis

        def apply(result):
            positions, multi_axis = result
            self._positions = positions
            self._multi_axis = multi_axis
            if use_undoable_accept:
                self._edit_state.accept_generation(positions.actions)
            else:
                self._edit_state.load_actions(positions.actions)
            self.visualizations.set_positions(positions)
            self.visualizations.set_multi_axis(multi_axis)
            self.aux_panel.set_multi_axis(multi_axis)
            self._refresh_edit_axis_combo()
            self._last_live_preview_signature = self._live_preview_signature()
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Generated {len(positions.actions)} actions", 3000)

        if not blocking:
            self._run_step_async(4, "Generate", compute, apply, "Script generation failed")
            return True

        self.controls.step_bar.set_step_status(4, "running")
        self._set_pipeline_busy(True, "Generate: running...")
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        try:
            result = compute(lambda msg, pct: self._progress("Generate", msg, pct))
            apply(result)
            self.controls.step_bar.set_step_status(4, "done")
            self.controls.on_step_completed(4)
            self._refresh_step_availability()
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(4, "error")
            self._show_error("Generate failed", f"Script generation failed: {exc}", show_errors)
            return False
        finally:
            self._set_pipeline_busy(False)
            self._progress_bar.hide()

    def step_5_export(
        self,
        output_dir: str | None = None,
        show_errors: bool = True,
        show_success: bool = True,
    ) -> bool:
        if self._positions is None or (not self._positions.actions and not self._positions.beat_actions):
            self._show_error("Export blocked", "Generate a script before export.", show_errors)
            return False

        folder = output_dir or QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return False

        self.controls.step_bar.set_step_status(5, "running")
        try:
            out_dir = Path(folder)
            out_dir.mkdir(parents=True, exist_ok=True)

            axis_cfg = self.controls.get_axis_config()
            enabled_axes = sorted(axis_cfg.enabled_axes or {"main"})
            output_format = str(self.controls.output_format_combo.currentText()).strip().lower()

            stem = Path(self._file_path).stem if self._file_path else "pmv_output"
            duration_ms = int(self._timeline.duration_ms if self._timeline is not None else 0)
            metadata = FunscriptMetadata(
                title=stem,
                duration=duration_ms,
                parameters={
                    "preset": self.controls.to_preset(),
                },
            )

            main_actions = self._current_main_actions()
            if not main_actions and self._positions is not None:
                main_actions = [FunscriptAction(a.at, a.pos) for a in self._positions.beat_actions]
            if not main_actions:
                raise RuntimeError("No main-axis actions are available for export.")

            # Use sparse beat-level actions for 2D conversion (restim
            # semicircle arcs need large deltas between consecutive points).
            beat_level = (
                [FunscriptAction(a.at, a.pos) for a in self._positions.beat_actions]
                if self._positions is not None and len(self._positions.beat_actions) >= 2
                else main_actions
            )
            export_multi_axis = convert_to_2d(beat_level, axis_cfg, duration_ms, audio_timeline=self._timeline)
            blend = 1.0 if axis_cfg.alpha_beta_mode == "orbital" else axis_cfg.orbital_blend
            if blend > 0.0 and self._timeline is not None and self._beats is not None:
                analysis_cfg = self.controls.get_analysis_config()
                export_multi_axis, _ = _apply_orbital_overlay(
                    export_multi_axis, self._timeline, self._beats, analysis_cfg, blend=blend,
                    cached_orbital=self._cached_orbital_result,
                )

            # Merge locked multi-axis actions into export
            locked_regions = [LockedRegion(r.start_ms, r.end_ms) for r in self._edit_state.locked_regions]
            if locked_regions and self._multi_axis is not None:
                for ax_name, ax_actions in self._multi_axis.axes.items():
                    prev_locked = [FunscriptAction(a.at, a.pos) for a in ax_actions
                                   if any(r.start_ms <= int(a.at) <= r.end_ms for r in locked_regions)]
                    if prev_locked:
                        new_actions = export_multi_axis.axes.get(ax_name, [])
                        export_multi_axis.axes[ax_name] = self._merge_generated_actions_with_locked_regions(
                            new_actions, prev_locked, locked_regions,
                        )

            exported = 0
            exported_paths: list[Path] = []
            for axis_name in enabled_axes:
                if axis_name == "main":
                    axis_actions = main_actions
                elif axis_name in self._aux_edit_states and self._aux_edit_states[axis_name].actions:
                    axis_actions = list(self._aux_edit_states[axis_name].actions)
                elif self._multi_axis is not None and axis_name in self._multi_axis.axes and self._multi_axis.axes[axis_name]:
                    axis_actions = list(self._multi_axis.axes[axis_name])
                else:
                    axis_actions = export_multi_axis.axes.get(axis_name, [])

                if not axis_actions:
                    continue

                suffix = "" if axis_name == "main" else f".{axis_name}"
                if output_format == "csv":
                    target = out_dir / f"{stem}{suffix}.csv"
                    write_csv(target, axis_actions)
                else:
                    target = out_dir / f"{stem}{suffix}.funscript"
                    write_funscript(target, axis_actions, metadata)

                if not target.exists():
                    raise RuntimeError(f"Export failed: file was not created at {target}")

                exported += 1
                exported_paths.append(target)

            if exported <= 0:
                raise RuntimeError("No axis output was produced for export.")

            self.controls.step_bar.set_step_status(5, "done")
            self._refresh_step_availability()

            ext = ".csv" if output_format == "csv" else ".funscript"
            preview_names = ", ".join(p.name for p in exported_paths[:3])
            more = "" if len(exported_paths) <= 3 else f", +{len(exported_paths) - 3} more"
            msg = f"Exported {exported} {ext} file(s) to {out_dir}: {preview_names}{more}"
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(msg, 4000)
            if show_success:
                QMessageBox.information(self, "Export Complete", msg)
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(5, "error")
            self._show_error("Export failed", f"Failed to export output files: {exc}", show_errors)
            return False

    def _on_controls_changed(self) -> None:
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage("PMV controls updated", 1800)

        # Don't auto-preview while user has unsaved edits
        if self._edit_state.dirty:
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(
                    "Manual edits active \u2014 click Generate to update unlocked regions", 3000
                )
            return

        self._schedule_beat_preview()
        self._schedule_live_preview()

    def _on_step_requested(self, step: int) -> None:
        if self._worker_thread is not None:
            return
        if step == 1:
            self.step_1_load_audio(blocking=False)
        elif step == 2:
            self.step_2_analyze(blocking=False)
        elif step == 3:
            self.step_3_detect_beats(blocking=False)
        elif step == 4:
            self.step_4_generate(blocking=False)
        elif step == 5:
            self.step_5_export()

    def _on_visualization_position_changed(self, time_ms: float) -> None:
        now = time.perf_counter()

        # Throttle transient status updates so playback doesn't flood the UI thread.
        if (now - self._last_playback_status_update) >= 0.25:
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Playback position {time_ms / 1000.0:.2f}s", 1200)
            self._last_playback_status_update = now

        self.aux_panel.set_playhead(time_ms)

        # Commands use ~33ms durations, so >30 Hz sends are redundant work.
        if (now - self._last_preview_send_update) >= (1.0 / 30.0):
            self._send_position_at_time(time_ms)
            self._last_preview_send_update = now

    def _interpolate_axis_at(self, axis_name: str, time_ms: float) -> float | None:
        """Interpolate a funscript axis value (0-100) at the given time."""
        if self._multi_axis is None:
            return None
        actions = self._multi_axis.axes.get(axis_name, [])
        if not actions:
            return None
        if len(actions) == 1:
            return float(actions[0].pos)

        # Binary search for surrounding actions
        if time_ms <= actions[0].at:
            return float(actions[0].pos)
        if time_ms >= actions[-1].at:
            return float(actions[-1].pos)

        lo, hi = 0, len(actions) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if actions[mid].at <= time_ms:
                lo = mid
            else:
                hi = mid

        a0, a1 = actions[lo], actions[hi]
        span = float(a1.at - a0.at)
        if span < 1.0:
            return float(a0.pos)
        t = (time_ms - a0.at) / span
        return float(a0.pos) + t * float(a1.pos - a0.pos)

    def _send_position_at_time(self, time_ms: float) -> None:
        """Send interpolated alpha/beta position to device or main window canvas."""
        # Try sending via shared network engine (ReStim connected)
        ne = self._network_engine
        if ne is not None and getattr(ne, 'connected', False):
            axis_cfg = self.controls.get_axis_config()
            transport_mode = str(getattr(axis_cfg, "preview_tcode_mode", "threephase") or "threephase").strip().lower()
            include_linear_axes = transport_mode != "fourphase"
            alpha_tc = 0.0
            beta_tc = 0.0

            if include_linear_axes:
                alpha_pos = self._interpolate_axis_at("alpha", time_ms)
                beta_pos = self._interpolate_axis_at("beta", time_ms)
                if alpha_pos is None or beta_pos is None:
                    return
                # Convert funscript 0-100 to tcode -1.0..1.0
                alpha_tc = (alpha_pos / 50.0) - 1.0
                beta_tc = (beta_pos / 50.0) - 1.0

            from network_engine import TCodeCommand
            vol_pos = self._interpolate_axis_at("volume", time_ms)
            vol = (vol_pos / 100.0) if vol_pos is not None else 1.0
            # TCode: swap alpha/beta to match restim's L0/L1 convention
            # (only the bREadbeats canvas display needs the un-swapped order)
            cmd = TCodeCommand(
                alpha=beta_tc,
                beta=alpha_tc,
                duration_ms=33,
                volume=vol,
                include_linear_axes=include_linear_axes,
            )

            # Aux axes enabled by send-toggle checkboxes
            send_axes = self.aux_panel.get_send_axes()
            _AUX_TCODE_MAP = {
                "pulse_frequency": "P0",
                "carrier_frequency": "C0",
                "pulse_width": "P1",
                "pulse_rise": "P3",
            }
            _ELECTRODE_TCODE_MAP = {
                "e1": "E1",
                "e2": "E2",
                "e3": "E3",
                "e4": "E4",
            }
            for axis_name, tag in _AUX_TCODE_MAP.items():
                if axis_name not in send_axes:
                    continue
                val = self._interpolate_axis_at(axis_name, time_ms)
                if val is None:
                    continue
                tcode_val = int(val / 100.0 * 9999)
                tcode_val = max(0, min(9999, tcode_val))
                if tag == "P0":
                    cmd.pulse_freq = tcode_val
                    cmd.pulse_freq_duration = 33
                else:
                    cmd.tcode_tags[tag] = tcode_val
                    cmd.tcode_tags[f"{tag}_duration"] = 33

            if transport_mode == "fourphase":
                for axis_name, tag in _ELECTRODE_TCODE_MAP.items():
                    if axis_name not in send_axes:
                        continue
                    val = self._interpolate_axis_at(axis_name, time_ms)
                    if val is None:
                        continue
                    tcode_val = int(val / 100.0 * 9999)
                    tcode_val = max(0, min(9999, tcode_val))
                    cmd.tcode_tags[tag] = tcode_val
                    cmd.tcode_tags[f"{tag}_duration"] = 33

            ne.send_immediate(cmd)
        elif self._position_canvas is not None:
            alpha_pos = self._interpolate_axis_at("alpha", time_ms)
            beta_pos = self._interpolate_axis_at("beta", time_ms)
            if alpha_pos is None or beta_pos is None:
                return

            # Convert funscript 0-100 to tcode -1.0..1.0
            alpha_tc = (alpha_pos / 50.0) - 1.0
            beta_tc = (beta_pos / 50.0) - 1.0

            # bREadbeats PositionCanvas: arg1=horizontal, arg2=vertical.
            # Restim-style alpha=vertical, beta=horizontal → swap for display.
            # Orbital replay already outputs device-convention → no swap.
            axis_cfg = self.controls.get_axis_config()
            if axis_cfg.alpha_beta_mode == "orbital":
                self._position_canvas.update_position(alpha_tc, beta_tc)
            else:
                self._position_canvas.update_position(beta_tc, alpha_tc)

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            if self.video_preview.isVisible():
                self.video_preview.raise_()
        super().changeEvent(event)

    def closeEvent(self, event) -> None:
        self._save_last_used_settings()
        self.video_preview.close()
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(5000)
            self._worker_thread = None
            self._worker = None
        super().closeEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            for url in mime.urls():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in IMPORTABLE_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            event.ignore()
            return

        for url in mime.urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in FUNSCRIPT_EXTENSIONS:
                self.open_funscript(str(path), blocking=False)
                event.acceptProposedAction()
                return
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.step_1_load_audio(str(path), blocking=False)
                event.acceptProposedAction()
                return
        event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        from pathlib import Path as _Path
        from PyQt6.QtGui import QIcon as _QIcon
        _icon = _Path(__file__).parent / "bREadbeats.ico"
        if _icon.exists():
            app.setWindowIcon(_QIcon(str(_icon)))
    except Exception:
        pass
    try:
        from stylesheet import get_main_stylesheet

        app.setStyleSheet(get_main_stylesheet())
    except Exception:
        pass

    window = PMVGeneratorWindow()
    window.show()
    sys.exit(app.exec())
