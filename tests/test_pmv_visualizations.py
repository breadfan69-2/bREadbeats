from __future__ import annotations

import unittest

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from funscript_edit_state import FunscriptEditState
from pmv_audio_analysis import AudioTimeline
from pmv_axis_converter import MultiAxisResult
from pmv_beat_engine import BeatCandidate, BeatTimeline
from pmv_colors import FOURPHASE_AXIS_COLORS, FOURPHASE_AXIS_ORDER
from pmv_funscript_io import FunscriptAction
from pmv_position_mapper import PositionTimeline
from pmv_visualizations import VideoPreviewWidget, VisualizationArea


class TestPmvVisualizations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def _build_audio_timeline(self) -> AudioTimeline:
        sr = 48_000
        duration_ms = 4_000
        frame_times = np.arange(0.0, float(duration_ms), 40.0, dtype=np.float64)
        n = len(frame_times)

        return AudioTimeline(
            samples=np.sin(2.0 * np.pi * 1.5 * np.linspace(0.0, 4.0, sr * 4, endpoint=False, dtype=np.float64)).astype(np.float32),
            sample_rate=sr,
            duration_ms=duration_ms,
            frame_times_ms=frame_times,
            feature_frames=[],
            rms_per_frame=np.full(n, 0.2, dtype=np.float64),
            spectral_flux_per_frame=np.clip(np.sin(frame_times / 700.0) * 0.5 + 0.5, 0.0, 1.0),
            spectral_centroid_per_frame=np.full(n, 250.0, dtype=np.float64),
            spectral_flatness_per_frame=np.full(n, 0.1, dtype=np.float64),
            band_energies_per_frame={
                "sub_bass": np.full(n, 0.55, dtype=np.float64),
                "low_mid": np.full(n, 0.45, dtype=np.float64),
                "mid": np.full(n, 0.35, dtype=np.float64),
                "high": np.full(n, 0.25, dtype=np.float64),
            },
            rms_mean_10s=np.full(n, 0.2, dtype=np.float64),
            rms_std_10s=np.full(n, 0.03, dtype=np.float64),
            flux_mean_10s=np.full(n, 0.35, dtype=np.float64),
            bass_mean_10s=np.full(n, 0.5, dtype=np.float64),
            energy_trend_10s=np.zeros(n, dtype=np.float64),
            pitch_per_frame=np.full(n, 120.0, dtype=np.float64),
            pitch_confidence=np.full(n, 0.9, dtype=np.float64),
            p95_flux=0.95,
            p95_band_energies={"sub_bass": 0.55, "low_mid": 0.45, "mid": 0.35, "high": 0.25},
        )

    def test_bind_audio_flux_and_cursor(self):
        area = VisualizationArea()
        timeline = self._build_audio_timeline()

        area.set_audio_data(timeline.samples, timeline.sample_rate)
        area.set_features(timeline)
        area.set_playback_position(1250.0)

        self.assertAlmostEqual(float(area.playhead_line.pos().x()), 1250.0, places=3)
        x_wave, y_wave = area.wave_curve.getData()
        x_flux, y_flux = area.flux_curve.getData()
        self.assertGreater(len(x_wave), 0)
        self.assertGreater(len(y_wave), 0)
        self.assertGreater(len(x_flux), 0)
        self.assertGreater(len(y_flux), 0)
        self.assertGreater(area.playback_panel._duration_ms, 0.0)

    def test_bind_beats_positions_and_heatmap(self):
        area = VisualizationArea()

        beat_times = [500, 1000, 1500, 2000]
        beats = BeatTimeline(
            beats=[
                BeatCandidate(time_ms=float(t), confidence=0.8, source="test", beat_type="downbeat" if i == 0 else "beat")
                for i, t in enumerate(beat_times)
            ],
            tempo_bpm=120.0,
            tempo_confidence=0.9,
            beat_period_ms=500.0,
        )
        actions = [FunscriptAction(at=t, pos=20 + (i * 20)) for i, t in enumerate(beat_times)]
        positions = PositionTimeline(
            actions=actions,
            beat_actions=actions,
            speed_profile=np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64),
            ml_results=None,
        )

        area.set_beats(beats)
        area.set_positions(positions)

        # LOD system requires a view range to populate curves
        area.zoom_to_range(0.0, 2500.0)

        x_main, y_main = area.position_curve.getData()
        x_speed, y_speed = area.speed_curve.getData()
        self.assertEqual(len(x_main), 4)
        self.assertEqual(len(y_main), 4)
        self.assertEqual(len(x_speed), 4)
        self.assertEqual(len(y_speed), 4)
        # Beats are stored in _beat_data; beat_scatter is deprecated
        total_beats = sum(len(v) for v in area._beat_data.values())
        self.assertEqual(total_beats, 4)

    def test_electrode_aux_colors_match_shared_fourphase_palette(self):
        area = VisualizationArea()
        area.set_multi_axis(
            MultiAxisResult(
                axes={
                    axis_name: [FunscriptAction(at=0, pos=(index + 1) * 10)]
                    for index, axis_name in enumerate(FOURPHASE_AXIS_ORDER)
                }
            )
        )

        colors = {
            axis_name: area.extra_curves[axis_name].opts["pen"].color().name()
            for axis_name in FOURPHASE_AXIS_ORDER
        }

        self.assertEqual(colors, FOURPHASE_AXIS_COLORS)

    def test_zoom_and_toggle(self):
        area = VisualizationArea()

        area.zoom_to_range(100.0, 900.0)
        x_range = area.overlay_plot.viewRange()[0]
        self.assertAlmostEqual(x_range[0], 100.0, places=2)
        self.assertAlmostEqual(x_range[1], 900.0, places=2)

        area._set_trace_visible("Flux", False)
        self.assertFalse(area.flux_curve.isVisible())
        area._set_trace_visible("Flux", True)
        self.assertTrue(area.flux_curve.isVisible())

    def test_scroll_slider_reaches_beyond_30s(self):
        area = VisualizationArea()
        sr = 48_000
        samples = np.zeros(sr * 45, dtype=np.float32)
        area.set_audio_data(samples, sr)

        # Default view is full duration; zoom to a 10s window first so scrolling is meaningful
        area.zoom_to_range(0.0, 10000.0)
        area.nav_slider.setValue(1000)
        x_range = area.overlay_plot.viewRange()[0]
        self.assertGreater(x_range[0], 10000.0)
        self.assertGreater(x_range[1], 40000.0)

    def test_playback_auto_scrolls_when_playhead_exits_window(self):
        area = VisualizationArea()
        sr = 48_000
        samples = np.zeros(sr * 60, dtype=np.float32)
        area.set_audio_data(samples, sr)
        area.zoom_to_range(0.0, 30000.0)

        area._on_playback_position(35000.0)
        x_range = area.overlay_plot.viewRange()[0]
        self.assertGreater(x_range[0], 0.0)
        self.assertGreaterEqual(x_range[1], 35000.0)

    def test_playback_seek_emits_position(self):
        area = VisualizationArea()
        captured: list[float] = []
        area.position_changed.connect(captured.append)

        area.playback_panel.set_duration_ms(2000.0)
        area.playback_panel.seek(750.0)

        self.assertGreaterEqual(len(captured), 1)
        self.assertAlmostEqual(float(captured[-1]), 750.0, places=2)

    def test_playback_seek_emits_transport_seek(self):
        area = VisualizationArea()
        captured: list[tuple[str, float]] = []
        area.playback_panel.transport_changed.connect(lambda action, position: captured.append((str(action), float(position))))

        area.playback_panel.set_duration_ms(2000.0)
        area.playback_panel.seek(750.0)

        self.assertGreaterEqual(len(captured), 1)
        self.assertEqual(captured[-1][0], "seek")
        self.assertAlmostEqual(captured[-1][1], 750.0, places=2)

    def test_playback_panel_preview_volume_controls_track_state(self):
        area = VisualizationArea()
        panel = area.playback_panel

        panel.set_preview_volume(0.35)

        self.assertAlmostEqual(panel.preview_volume(), 0.35, places=2)
        self.assertEqual(panel.volume_slider.value(), 35)
        self.assertEqual(panel.volume_label.text(), "35%")

        panel.set_preview_muted(True)

        self.assertTrue(panel.preview_muted())
        self.assertTrue(panel.mute_btn.isChecked())

    def test_video_preview_widget_controls_sync_through_playback_panel(self):
        area = VisualizationArea()
        panel = area.playback_panel
        preview = VideoPreviewWidget()

        panel.duration_changed.connect(preview.set_duration_ms)
        panel.position_changed.connect(preview.set_playback_position)
        panel.preview_volume_changed.connect(preview.set_volume)
        panel.preview_muted_changed.connect(preview.set_muted)
        preview.seek_requested.connect(panel.seek)
        preview.volume_changed.connect(panel.set_preview_volume)
        preview.muted_changed.connect(panel.set_preview_muted)
        preview.play_requested.connect(panel.play)
        preview.pause_requested.connect(panel.pause)

        panel.set_duration_ms(1500.0)
        panel.set_external_media_active(True)
        preview.load_media("clip.mp4")

        panel.set_preview_volume(0.42)
        panel.set_preview_muted(True)

        self.assertEqual(preview._volume_slider.value(), 42)
        self.assertTrue(preview._mute_btn.isChecked())

        preview._volume_slider.setValue(65)
        preview._mute_btn.setChecked(False)

        self.assertAlmostEqual(panel.preview_volume(), 0.65, places=2)
        self.assertFalse(panel.preview_muted())

        preview._play_btn.click()
        self.assertTrue(panel._playing)

        preview._pause_btn.click()
        self.assertFalse(panel._playing)

        panel.seek(750.0)
        self.assertEqual(preview._seek_slider.value(), 500)

        preview._seek_slider.setValue(250)
        self.assertAlmostEqual(panel._position_ms, 375.0, places=2)

    def test_edit_mode_viewport_click_selects_point(self):
        area = VisualizationArea()
        state = FunscriptEditState()
        actions = [
            FunscriptAction(at=500, pos=20),
            FunscriptAction(at=1000, pos=55),
            FunscriptAction(at=1500, pos=80),
        ]
        positions = PositionTimeline(
            actions=actions,
            beat_actions=actions,
            speed_profile=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            ml_results=None,
        )

        state.load_actions(actions)
        area.set_edit_state(state)
        area.set_positions(positions)
        area.zoom_to_range(0.0, 2000.0)
        area.resize(900, 500)
        area.show()
        QApplication.processEvents()

        area._edit_mode_btn.setChecked(True)
        QApplication.processEvents()

        view_box = area.overlay_plot.getViewBox()
        scene_point = view_box.mapViewToScene(pg.Point(1000.0, 55.0))
        widget_point = area.overlay_plot.mapFromScene(scene_point)
        click_event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(widget_point),
            QPointF(widget_point),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        handled = area.eventFilter(area.overlay_plot.viewport(), click_event)

        self.assertTrue(handled)
        self.assertEqual(state.selection_indices, {1})


if __name__ == "__main__":
    unittest.main()
