from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEventLoop, QTimer, Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from pmv_audio_analysis import analyze_full_file, load_audio
from pmv_automap import automap_optimize
from pmv_axis_converter import MultiAxisResult, convert_to_2d
from pmv_beat_engine import BeatTimeline, detect_beats
from pmv_controls import PMVControlsPanel
from pmv_funscript_io import FunscriptMetadata, write_csv, write_funscript
from pmv_position_mapper import PositionTimeline, generate_positions
from pmv_visualizations import VisualizationArea


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".wma", ".m4a"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".wmv", ".mov", ".flv"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def _merge_dict(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


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
        self._samples = None
        self._timeline = None
        self._beats: BeatTimeline | None = None
        self._positions: PositionTimeline | None = None
        self._multi_axis: MultiAxisResult | None = None
        self._live_preview_busy = False
        self._last_live_preview_signature: str | None = None
        self._pipeline_busy = False
        self._busy_cursor_set = False
        self._last_progress_pump = 0.0

        self._live_preview_timer = QTimer(self)
        self._live_preview_timer.setSingleShot(True)
        self._live_preview_timer.setInterval(140)
        self._live_preview_timer.timeout.connect(self._run_live_preview)

        self._beat_preview_timer = QTimer(self)
        self._beat_preview_timer.setSingleShot(True)
        self._beat_preview_timer.setInterval(300)
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
        preset_row.addWidget(self.load_preset_btn)
        preset_row.addWidget(self.save_preset_btn)
        preset_row.addWidget(self.refresh_preset_btn)

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

        self.visualizations = VisualizationArea(center)
        center_layout.addWidget(self.visualizations, 1)

        center_layout.addWidget(QLabel("Pipeline debug state"))
        self.debug_panel = QPlainTextEdit()
        self.debug_panel.setReadOnly(True)
        center_layout.addWidget(self.debug_panel, 1)

        splitter.addWidget(center)
        splitter.setSizes([380, 820])

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar(self))

        self.controls.config_changed.connect(self._on_controls_changed)
        self.controls.step_bar.step_requested.connect(self._on_step_requested)
        self.visualizations.position_changed.connect(self._on_visualization_position_changed)
        self.load_preset_btn.clicked.connect(self._on_load_preset_clicked)
        self.save_preset_btn.clicked.connect(self._on_save_preset_clicked)
        self.refresh_preset_btn.clicked.connect(self._reload_preset_catalog)

        self._ensure_default_presets()
        self._reload_preset_catalog()
        self._refresh_step_availability()
        self._on_controls_changed()

    def _preset_dirs(self) -> tuple[Path, Path]:
        root = Path(__file__).resolve().parent
        return root / "defaults" / "pmv_presets", root / "user_pmv_presets"

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

        for key, payload in self._build_default_presets().items():
            target = defaults_dir / f"{key}.json"
            if target.exists():
                continue
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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

    def _show_error(self, title: str, message: str, show_errors: bool) -> None:
        self.controls.step_bar.set_step_status(1, "ready")
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(message, 3000)
        if show_errors:
            QMessageBox.critical(self, title, message)

    def _progress(self, step_name: str, message: str, percent: float) -> None:
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(f"{step_name}: {message} ({percent:.0f}%)")

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

    def _select_input_file(self) -> str | None:
        filter_str = (
            "Media Files (*.wav *.mp3 *.flac *.ogg *.aac *.wma *.m4a *.mp4 *.mkv *.avi *.webm *.wmv *.mov *.flv);;"
            "All Files (*.*)"
        )
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio or Video", "", filter_str)
        return path or None

    def _refresh_step_availability(self) -> None:
        self.controls.step_bar.set_step_enabled(1, True)
        self.controls.step_bar.set_step_enabled(2, self._samples is not None)
        self.controls.step_bar.set_step_enabled(3, self._timeline is not None)
        self.controls.step_bar.set_step_enabled(4, self._beats is not None and len(self._beats.beats) > 0)
        has_exportable = (
            self._positions is not None
            and (len(self._positions.actions) > 0 or len(self._positions.beat_actions) > 0)
        )
        self.controls.step_bar.set_step_enabled(5, has_exportable)

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
        self._live_preview_timer.start()

    def _run_live_preview(self) -> None:
        if self._live_preview_busy:
            return
        if self._positions is None:
            return
        if self._timeline is None or self._beats is None or len(self._beats.beats) <= 0:
            return

        sig = self._live_preview_signature()
        if sig == self._last_live_preview_signature:
            return

        self._live_preview_busy = True
        try:
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

            positions = generate_positions(self._timeline, self._beats, mapping_cfg)
            axis_source_actions = positions.beat_actions if len(positions.beat_actions) >= 2 else positions.actions
            multi_axis = convert_to_2d(
                axis_source_actions,
                axis_cfg,
                duration_ms=int(self._timeline.duration_ms),
            )

            self._positions = positions
            self._multi_axis = multi_axis
            self.visualizations.set_positions(positions)
            self.visualizations.set_multi_axis(multi_axis)
            self._last_live_preview_signature = sig
            self._refresh_step_availability()

            bar = self.statusBar()
            if bar is not None:
                bar.showMessage("Live preview updated", 1200)
        except Exception as exc:
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Live preview failed: {exc}", 2800)
        finally:
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
        self._beat_preview_timer.start()

    def _run_beat_preview(self) -> None:
        if self._beat_preview_busy:
            return
        if self._timeline is None:
            return

        sig = self._beat_preview_signature()
        if sig == self._last_beat_preview_signature:
            return

        self._beat_preview_busy = True
        try:
            beat_cfg = self.controls.get_beat_config()
            beats = detect_beats(self._timeline, beat_cfg)
            self._beats = beats
            self._reset_from_step(4)
            self.visualizations.set_beats(beats)
            self._last_beat_preview_signature = sig
            self._refresh_step_availability()

            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(
                    f"Beat preview: {len(beats.beats)} beats @ {beats.tempo_bpm:.1f} BPM",
                    1500,
                )
        except Exception as exc:
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Beat preview failed: {exc}", 2800)
        finally:
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
            self.visualizations.set_positions(
                PositionTimeline(
                    actions=[],
                    beat_actions=[],
                    speed_profile=np.array([], dtype=np.float64),
                    ml_results=None,
                )
            )
            self.visualizations.set_multi_axis(MultiAxisResult(axes={"main": []}))

        for idx in range(max(2, step), 6):
            self.controls.step_bar.set_step_status(idx, "ready")
        self._refresh_step_availability()

    def step_1_load_audio(self, file_path: str | None = None, show_errors: bool = True) -> bool:
        path = file_path or self._select_input_file()
        if not path:
            return False
        if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
            self._show_error("Unsupported file", "Selected file type is not supported.", show_errors)
            return False

        self.controls.step_bar.set_step_status(1, "running")
        try:
            analysis_cfg = self.controls.get_analysis_config()
            samples = load_audio(path, analysis_cfg, lambda msg, pct: self._progress("Load", msg, pct))
            self._file_path = str(path)
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
            self.controls.step_bar.set_step_status(1, "done")
            self.controls.on_step_completed(1)
            self._refresh_step_availability()
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Loaded {Path(path).name}", 2500)
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(1, "error")
            self._show_error("Load failed", f"Unable to load media: {exc}", show_errors)
            return False

    def step_2_analyze(self, show_errors: bool = True) -> bool:
        if self._samples is None:
            self._show_error("Analyze blocked", "Load audio before analysis.", show_errors)
            return False

        self.controls.step_bar.set_step_status(2, "running")
        try:
            analysis_cfg = self.controls.get_analysis_config()
            self._timeline = analyze_full_file(self._samples, analysis_cfg, lambda msg, pct: self._progress("Analyze", msg, pct))
            self._reset_from_step(3)
            self.visualizations.set_features(self._timeline)
            self.visualizations.zoom_to_range(0.0, float(min(self._timeline.duration_ms, 30000)))
            self.controls.step_bar.set_step_status(2, "done")
            self.controls.on_step_completed(2)
            self._refresh_step_availability()
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(2, "error")
            self._show_error("Analyze failed", f"Audio analysis failed: {exc}", show_errors)
            return False

    def step_3_detect_beats(self, show_errors: bool = True) -> bool:
        if self._timeline is None:
            self._show_error("Beat detection blocked", "Run analysis before beat detection.", show_errors)
            return False

        self.controls.step_bar.set_step_status(3, "running")
        try:
            beat_cfg = self.controls.get_beat_config()
            self._beats = detect_beats(self._timeline, beat_cfg, lambda msg, pct: self._progress("Beats", msg, pct))
            self._reset_from_step(4)
            self._last_beat_preview_signature = self._beat_preview_signature()
            self.visualizations.set_beats(self._beats)
            self.controls.step_bar.set_step_status(3, "done")
            self.controls.on_step_completed(3)
            self._refresh_step_availability()
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Detected {len(self._beats.beats)} beats @ {self._beats.tempo_bpm:.1f} BPM", 3000)
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(3, "error")
            self._show_error("Beat detection failed", f"Beat detection failed: {exc}", show_errors)
            return False

    def step_4_generate(self, show_errors: bool = True) -> bool:
        if self._timeline is None or self._beats is None:
            self._show_error("Generate blocked", "Analyze audio and detect beats before generation.", show_errors)
            return False

        self.controls.step_bar.set_step_status(4, "running")
        self._set_pipeline_busy(True, "Generate: running...")
        try:
            mapping_cfg = self.controls.get_mapping_config()
            axis_cfg = self.controls.get_axis_config()
            automap_cfg = self.controls.get_automap_config()

            if mapping_cfg.overflow_mode == "bounce" and mapping_cfg.pos_min == mapping_cfg.pos_max:
                raise ValueError("Bounce overflow requires Position Min and Position Max to be different.")

            if automap_cfg.enabled:
                mapping_cfg = automap_optimize(
                    self._timeline,
                    self._beats,
                    mapping_cfg,
                    automap_cfg,
                    lambda msg, pct: self._progress("Automap", msg, pct),
                )

            self._positions = generate_positions(
                self._timeline,
                self._beats,
                mapping_cfg,
                lambda msg, pct: self._progress("Generate", msg, pct),
            )
            axis_source_actions = self._positions.beat_actions if len(self._positions.beat_actions) >= 2 else self._positions.actions
            self._multi_axis = convert_to_2d(
                axis_source_actions,
                axis_cfg,
                duration_ms=int(self._timeline.duration_ms),
                progress_callback=lambda msg, pct: self._progress("Axis", msg, pct),
            )
            self.visualizations.set_positions(self._positions)
            self.visualizations.set_multi_axis(self._multi_axis)
            self._last_live_preview_signature = self._live_preview_signature()
            self.controls.step_bar.set_step_status(4, "done")
            self.controls.on_step_completed(4)
            self._refresh_step_availability()
            bar = self.statusBar()
            if bar is not None:
                bar.showMessage(f"Generated {len(self._positions.actions)} actions", 3000)
            return True
        except Exception as exc:
            self.controls.step_bar.set_step_status(4, "error")
            self._show_error("Generate failed", f"Script generation failed: {exc}", show_errors)
            return False
        finally:
            self._set_pipeline_busy(False)

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

            exported = 0
            exported_paths: list[Path] = []
            for axis_name in enabled_axes:
                if axis_name == "main":
                    axis_actions = self._positions.actions
                    if not axis_actions:
                        # Fallback to beat-level actions when dense interpolation is unavailable.
                        axis_actions = self._positions.beat_actions
                elif self._multi_axis is not None:
                    axis_actions = self._multi_axis.axes.get(axis_name, [])
                else:
                    axis_actions = []

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
        analysis_cfg = self.controls.get_analysis_config()
        beat_cfg = self.controls.get_beat_config()
        mapping_cfg = self.controls.get_mapping_config()
        axis_cfg = self.controls.get_axis_config()
        automap_cfg = self.controls.get_automap_config()

        state = [
            f"Loaded file: {Path(self._file_path).name if self._file_path else 'None'}",
            f"Samples loaded: {self._samples is not None}",
            f"Analyzed: {self._timeline is not None}",
            f"Beats ready: {self._beats is not None}",
            f"Positions ready: {self._positions is not None}",
        ]
        self.debug_panel.setPlainText(
            "PMV pipeline state and controls.\n\n"
            "State:\n"
            + "\n".join(f"  {line}" for line in state)
            + "\n\n"
            "Analysis:\n"
            f"  sr={analysis_cfg.sample_rate} fft={analysis_cfg.fft_size} hop={analysis_cfg.hop_size} win={analysis_cfg.window_size}\n"
            f"  lowpass={analysis_cfg.lowpass_enabled}({analysis_cfg.lowpass_hz:.1f}Hz) highpass={analysis_cfg.highpass_enabled}({analysis_cfg.highpass_hz:.1f}Hz)\n"
            f"  freq_range={analysis_cfg.freq_min_hz:.1f}-{analysis_cfg.freq_max_hz:.1f} gain={analysis_cfg.gain:.2f}\n\n"
            "Beat Detection:\n"
            f"Sensitivity: {beat_cfg.sensitivity:.2f}\n"
            f"Refractory: {beat_cfg.refractory_ms:.1f} ms\n"
            f"Use librosa: {beat_cfg.use_librosa}\n"
            f"Use multibus: {beat_cfg.use_multibus}\n"
            f"Use FFT peaks: {beat_cfg.use_fft_peaks}\n"
            f"PLP enabled: {beat_cfg.plp_enabled}\n"
            f"Peak/seek ratio: {beat_cfg.peak_seek_ratio:.2f}\n"
            f"Peak beat threshold: {beat_cfg.peak_beat_threshold:.2f}\n\n"
            "Multibus weights:\n"
            f"  flux={beat_cfg.multibus_config.w_flux:.2f}\n"
            f"  band={beat_cfg.multibus_config.w_band:.2f}\n"
            f"  delta={beat_cfg.multibus_config.w_delta:.2f}\n"
            f"  phase={beat_cfg.multibus_config.w_phase:.2f}\n"
            f"  arm={beat_cfg.multibus_config.bus_arm_threshold:.2f}\n"
            f"  release={beat_cfg.multibus_config.bus_release_threshold:.2f}\n"
            f"  bus refractory={beat_cfg.multibus_config.bus_refractory_ms:.1f} ms\n\n"
            "Mapping and ML:\n"
            f"  pitch_range={mapping_cfg.pitch_range:.1f} center={mapping_cfg.center_offset:.1f} energy_mult={mapping_cfg.energy_multiplier:.1f}\n"
            f"  overflow={mapping_cfg.overflow_mode} min_delay={mapping_cfg.min_command_delay_ms:.1f}ms pps={mapping_cfg.points_per_second}\n"
            f"  ml_enabled={mapping_cfg.ml_config.enabled} strength={mapping_cfg.ml_config.strength:.2f} cadence={mapping_cfg.ml_config.cadence_mode}\n\n"
            "Axis and Automap:\n"
            f"  axis min_distance={axis_cfg.min_distance:.2f} speed_threshold={axis_cfg.speed_threshold_pct:.1f}%\n"
            f"  enabled_axes={sorted(axis_cfg.enabled_axes)}\n"
            f"  automap={automap_cfg.enabled} mode={automap_cfg.optimization_mode} target_y={automap_cfg.target_y_position:.1f}\n"
        )
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage("PMV controls updated", 1800)

        self._schedule_beat_preview()
        self._schedule_live_preview()

    def _on_step_requested(self, step: int) -> None:
        ok = False
        if step == 1:
            ok = self.step_1_load_audio()
        elif step == 2:
            ok = self.step_2_analyze()
        elif step == 3:
            ok = self.step_3_detect_beats()
        elif step == 4:
            ok = self.step_4_generate()
        elif step == 5:
            ok = self.step_5_export()

        if ok:
            self._on_controls_changed()

    def _on_visualization_position_changed(self, time_ms: float) -> None:
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(f"Playback position {time_ms / 1000.0:.2f}s", 1200)

        self._send_position_at_time(time_ms)

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
        alpha_pos = self._interpolate_axis_at("alpha", time_ms)
        beta_pos = self._interpolate_axis_at("beta", time_ms)
        if alpha_pos is None or beta_pos is None:
            return

        # Convert funscript 0-100 to tcode -1.0..1.0
        alpha_tc = (alpha_pos / 50.0) - 1.0
        beta_tc = (beta_pos / 50.0) - 1.0

        # Try sending via shared network engine (ReStim connected)
        ne = self._network_engine
        if ne is not None and getattr(ne, 'connected', False):
            from network_engine import TCodeCommand
            vol_pos = self._interpolate_axis_at("volume", time_ms)
            vol = (vol_pos / 100.0) if vol_pos is not None else 1.0
            cmd = TCodeCommand(alpha=alpha_tc, beta=beta_tc, duration_ms=33, volume=vol)
            ne.send_command(cmd)
        elif self._position_canvas is not None:
            # Fallback: update main window PositionCanvas
            self._position_canvas.update_position(alpha_tc, beta_tc)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            for url in mime.urls():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
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
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.step_1_load_audio(str(path))
                event.acceptProposedAction()
                return
        event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        from stylesheet import get_main_stylesheet

        app.setStyleSheet(get_main_stylesheet())
    except Exception:
        pass

    window = PMVGeneratorWindow()
    window.show()
    sys.exit(app.exec())
