from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from funscript_edit_state import LockedRegion
from pmv_axis_converter import MultiAxisResult
from pmv_generator import PMVGeneratorWindow
from pmv_funscript_io import FunscriptAction, FunscriptMetadata, read_funscript, write_funscript
from pmv_position_mapper import PositionTimeline


class TestPmvGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @staticmethod
    def _make_click_track(sr: int, duration_s: float, bpm: float) -> np.ndarray:
        total = int(sr * duration_s)
        samples = np.zeros(total, dtype=np.float32)

        beat_period = max(1, int(round(sr * 60.0 / bpm)))
        pulse_len = max(1, int(0.012 * sr))
        pulse = np.hanning(pulse_len).astype(np.float32)

        for start in range(0, total, beat_period):
            end = min(total, start + pulse_len)
            samples[start:end] += 0.9 * pulse[: end - start]

        tone = 0.22 * np.sin(2.0 * math.pi * 110.0 * np.arange(total, dtype=np.float32) / float(sr))
        return np.clip(samples + tone, -1.0, 1.0)

    @staticmethod
    def _write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

    def test_step_prerequisites(self):
        win = PMVGeneratorWindow()
        self.assertFalse(win.step_2_analyze(show_errors=False))
        self.assertFalse(win.step_3_detect_beats(show_errors=False))
        self.assertFalse(win.step_4_generate(show_errors=False))
        self.assertFalse(win.step_5_export(output_dir=None, show_errors=False, show_success=False))

    def test_end_to_end_pipeline_and_export(self):
        sr = 48_000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_wav = root / "clip.wav"
            export_dir = root / "out"

            samples = self._make_click_track(sr, duration_s=12.0, bpm=120.0)
            self._write_wav(input_wav, samples, sr)

            win = PMVGeneratorWindow()
            win.controls.use_librosa_chk.setChecked(False)

            self.assertTrue(win.step_1_load_audio(str(input_wav), show_errors=False))
            self.assertGreater(win.visualizations.playback_panel._duration_ms, 0.0)
            wave_x, wave_y = win.visualizations.wave_curve.getData()
            self.assertGreater(len(wave_x), 0)
            self.assertGreater(len(wave_y), 0)
            self.assertTrue(win.step_2_analyze(show_errors=False))
            self.assertTrue(win.step_3_detect_beats(show_errors=False))
            beats = win._beats
            self.assertIsNotNone(beats)
            if beats is None:
                self.fail("Beat timeline should not be None after step 3")
            self.assertGreater(len(beats.beats), 0)

            self.assertTrue(win.step_4_generate(show_errors=False))
            positions = win._positions
            self.assertIsNotNone(positions)
            if positions is None:
                self.fail("Position timeline should not be None after step 4")
            self.assertGreater(len(positions.actions), 0)

            self.assertTrue(win.step_5_export(str(export_dir), show_errors=False, show_success=False))
            exported = list(export_dir.glob("*.funscript"))
            self.assertGreaterEqual(len(exported), 1)

    def test_generate_rejects_invalid_bounce_range_without_freezing(self):
        sr = 48_000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_wav = root / "clip.wav"

            samples = self._make_click_track(sr, duration_s=8.0, bpm=120.0)
            self._write_wav(input_wav, samples, sr)

            win = PMVGeneratorWindow()
            win.controls.use_librosa_chk.setChecked(False)

            self.assertTrue(win.step_1_load_audio(str(input_wav), show_errors=False))
            self.assertTrue(win.step_2_analyze(show_errors=False))
            self.assertTrue(win.step_3_detect_beats(show_errors=False))

            win.controls.overflow_mode_combo.setCurrentText("bounce")
            win.controls.pos_min_spin.setValue(50)
            win.controls.pos_max_spin.setValue(50)

            self.assertFalse(win.step_4_generate(show_errors=False))

    def test_live_preview_updates_when_mapping_changes(self):
        sr = 48_000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_wav = root / "clip.wav"

            samples = self._make_click_track(sr, duration_s=10.0, bpm=120.0)
            self._write_wav(input_wav, samples, sr)

            win = PMVGeneratorWindow()
            win.controls.use_librosa_chk.setChecked(False)

            self.assertTrue(win.step_1_load_audio(str(input_wav), show_errors=False))
            self.assertTrue(win.step_2_analyze(show_errors=False))
            self.assertTrue(win.step_3_detect_beats(show_errors=False))
            self.assertTrue(win.step_4_generate(show_errors=False))

            self.assertIsNotNone(win._positions)
            if win._positions is None:
                self.fail("Positions should exist after generation")
            before = float(np.mean([a.pos for a in win._positions.actions]))
            win.controls.center_offset_slider.setValue(120.0)

            # Force preview for deterministic test behavior without event-loop timing assumptions.
            win._run_live_preview(blocking=True)

            self.assertIsNotNone(win._positions)
            if win._positions is None:
                self.fail("Positions should exist after live preview")
            after = float(np.mean([a.pos for a in win._positions.actions]))
            self.assertGreater(abs(after - before), 1.0)

    def test_export_falls_back_to_beat_actions_when_dense_actions_empty(self):
        sr = 48_000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_wav = root / "clip.wav"
            export_dir = root / "out"

            samples = self._make_click_track(sr, duration_s=8.0, bpm=120.0)
            self._write_wav(input_wav, samples, sr)

            win = PMVGeneratorWindow()
            win.controls.use_librosa_chk.setChecked(False)

            self.assertTrue(win.step_1_load_audio(str(input_wav), show_errors=False))
            self.assertTrue(win.step_2_analyze(show_errors=False))
            self.assertTrue(win.step_3_detect_beats(show_errors=False))
            self.assertTrue(win.step_4_generate(show_errors=False))

            self.assertIsNotNone(win._positions)
            if win._positions is None:
                self.fail("Positions should exist after generation")

            beat_actions = list(win._positions.beat_actions)
            self.assertGreater(len(beat_actions), 0)

            win._positions = PositionTimeline(
                actions=[],
                beat_actions=beat_actions,
                speed_profile=np.array([], dtype=np.float32),
                ml_results=win._positions.ml_results,
            )

            self.assertTrue(win.step_5_export(str(export_dir), show_errors=False, show_success=False))
            exported = list(export_dir.glob("*.funscript"))
            self.assertGreaterEqual(len(exported), 1)

    def test_export_uses_authoritative_edit_state_actions(self):
        sr = 48_000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_wav = root / "clip.wav"
            export_dir = root / "out"

            samples = self._make_click_track(sr, duration_s=8.0, bpm=120.0)
            self._write_wav(input_wav, samples, sr)

            win = PMVGeneratorWindow()
            win.controls.use_librosa_chk.setChecked(False)

            self.assertTrue(win.step_1_load_audio(str(input_wav), show_errors=False))
            self.assertTrue(win.step_2_analyze(show_errors=False))
            self.assertTrue(win.step_3_detect_beats(show_errors=False))
            self.assertTrue(win.step_4_generate(show_errors=False))

            edited_actions = [
                FunscriptAction(0, 7),
                FunscriptAction(500, 91),
                FunscriptAction(1000, 13),
            ]
            win._edit_state.load_actions(edited_actions)

            self.assertTrue(win.step_5_export(str(export_dir), show_errors=False, show_success=False))

            script_path = export_dir / f"{input_wav.stem}.funscript"
            exported_actions, _ = read_funscript(script_path)
            self.assertEqual(
                [(action.at, action.pos) for action in exported_actions],
                [(action.at, action.pos) for action in edited_actions],
            )

    def test_merge_generated_actions_preserves_locked_regions(self):
        generated = [
            FunscriptAction(0, 10),
            FunscriptAction(500, 20),
            FunscriptAction(1000, 30),
        ]
        locked = [
            FunscriptAction(450, 88),
            FunscriptAction(550, 92),
        ]
        merged = PMVGeneratorWindow._merge_generated_actions_with_locked_regions(
            generated,
            locked,
            [LockedRegion(400, 600)],
        )

        self.assertEqual(
            [(action.at, action.pos) for action in merged],
            [(0, 10), (450, 88), (550, 92), (1000, 30)],
        )

    def test_open_existing_funscript_without_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script_path = root / "existing.funscript"
            export_dir = root / "out"

            actions = [
                FunscriptAction(0, 25),
                FunscriptAction(600, 80),
                FunscriptAction(1200, 35),
            ]
            metadata = FunscriptMetadata(
                title="existing",
                duration=1500,
                parameters={
                    "preset": {
                        "mapping": {"center_offset": 12.0},
                        "axis": {"enabled_axes": ["main"]},
                    }
                },
            )
            write_funscript(script_path, actions, metadata)

            win = PMVGeneratorWindow()
            self.assertTrue(win.open_funscript(str(script_path), show_errors=False))

            self.assertIsNone(win._media_path)
            self.assertEqual(win._file_path, str(script_path))
            self.assertEqual([(a.at, a.pos) for a in win._edit_state.actions], [(0, 25), (600, 80), (1200, 35)])
            self.assertAlmostEqual(win.visualizations.playback_panel._duration_ms, 1500.0, places=2)
            self.assertAlmostEqual(win.controls.center_offset_slider.value(), 12.0, places=2)

            self.assertTrue(win.step_5_export(str(export_dir), show_errors=False, show_success=False))
            exported_actions, _ = read_funscript(export_dir / "existing.funscript")
            self.assertEqual([(a.at, a.pos) for a in exported_actions], [(0, 25), (600, 80), (1200, 35)])

    def test_open_selected_funscripts_merges_aux_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main_path = root / "main.funscript"
            alpha_path = root / "alpha.funscript"

            main_actions = [
                FunscriptAction(0, 20),
                FunscriptAction(500, 70),
                FunscriptAction(1000, 30),
            ]
            alpha_actions = [
                FunscriptAction(0, 45),
                FunscriptAction(500, 60),
                FunscriptAction(1000, 40),
            ]
            write_funscript(main_path, main_actions, FunscriptMetadata(title="main", duration=1000))
            write_funscript(alpha_path, alpha_actions, FunscriptMetadata(title="alpha", duration=1000))

            win = PMVGeneratorWindow()
            self.assertTrue(win.open_funscripts([str(main_path), str(alpha_path)], show_errors=False))

            self.assertEqual([(a.at, a.pos) for a in win._edit_state.actions], [(0, 20), (500, 70), (1000, 30)])
            self.assertIsNotNone(win._multi_axis)
            if win._multi_axis is None:
                self.fail("Multi-axis data should be loaded")
            self.assertIn("alpha", win._multi_axis.axes)
            self.assertEqual(
                [(a.at, a.pos) for a in win._multi_axis.axes["alpha"]],
                [(0, 45), (500, 60), (1000, 40)],
            )

    def test_open_selected_converted_funscripts_prefers_e1_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e1_path = root / "clip.e1.funscript"
            e2_path = root / "clip.e2.funscript"
            freq_path = root / "clip.frequency.funscript"

            e1_actions = [
                FunscriptAction(0, 10),
                FunscriptAction(500, 40),
                FunscriptAction(1000, 20),
            ]
            e2_actions = [
                FunscriptAction(0, 80),
                FunscriptAction(500, 50),
                FunscriptAction(1000, 70),
            ]
            freq_actions = [
                FunscriptAction(0, 60),
                FunscriptAction(500, 50),
                FunscriptAction(1000, 40),
            ]
            write_funscript(e1_path, e1_actions, FunscriptMetadata(title="clip.e1", duration=1000))
            write_funscript(e2_path, e2_actions, FunscriptMetadata(title="clip.e2", duration=1000))
            write_funscript(freq_path, freq_actions, FunscriptMetadata(title="clip.frequency", duration=1000))

            win = PMVGeneratorWindow()
            self.assertTrue(
                win.open_funscripts(
                    [str(freq_path), str(e2_path), str(e1_path)],
                    show_errors=False,
                )
            )

            self.assertEqual(
                [(a.at, a.pos) for a in win._positions.actions],
                [(0, 10), (500, 40), (1000, 20)],
            )
            self.assertEqual(
                [(a.at, a.pos) for a in win._multi_axis.axes["e1"]],
                [(0, 10), (500, 40), (1000, 20)],
            )

    def test_open_selected_converted_funscripts_uses_parent_main_if_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            converted = root / "converted"
            converted.mkdir()

            main_path = root / "clip.funscript"
            e1_path = converted / "clip.e1.funscript"
            e2_path = converted / "clip.e2.funscript"
            freq_path = converted / "clip.frequency.funscript"

            main_actions = [
                FunscriptAction(0, 25),
                FunscriptAction(500, 75),
                FunscriptAction(1000, 35),
            ]
            e1_actions = [
                FunscriptAction(0, 10),
                FunscriptAction(500, 40),
                FunscriptAction(1000, 20),
            ]
            e2_actions = [
                FunscriptAction(0, 80),
                FunscriptAction(500, 50),
                FunscriptAction(1000, 70),
            ]
            freq_actions = [
                FunscriptAction(0, 60),
                FunscriptAction(500, 50),
                FunscriptAction(1000, 40),
            ]

            write_funscript(main_path, main_actions, FunscriptMetadata(title="clip", duration=1000))
            write_funscript(e1_path, e1_actions, FunscriptMetadata(title="clip.e1", duration=1000))
            write_funscript(e2_path, e2_actions, FunscriptMetadata(title="clip.e2", duration=1000))
            write_funscript(freq_path, freq_actions, FunscriptMetadata(title="clip.frequency", duration=1000))

            win = PMVGeneratorWindow()
            self.assertTrue(
                win.open_funscripts(
                    [str(freq_path), str(e2_path), str(e1_path)],
                    show_errors=False,
                )
            )

            self.assertEqual(
                [(a.at, a.pos) for a in win._positions.actions],
                [(0, 25), (500, 75), (1000, 35)],
            )
            self.assertEqual(
                [(a.at, a.pos) for a in win._multi_axis.axes["e1"]],
                [(0, 10), (500, 40), (1000, 20)],
            )
            self.assertEqual(
                [(a.at, a.pos) for a in win._multi_axis.axes["frequency"]],
                [(0, 60), (500, 50), (1000, 40)],
            )

    def test_load_converted_preview_populates_axes(self):
        win = PMVGeneratorWindow()
        preview_axes = {
            "e1": [FunscriptAction(0, 10), FunscriptAction(1000, 20)],
            "e2": [FunscriptAction(0, 30), FunscriptAction(1000, 40)],
            "e3": [FunscriptAction(0, 50), FunscriptAction(1000, 60)],
            "e4": [FunscriptAction(0, 70), FunscriptAction(1000, 80)],
            "pulse_frequency": [FunscriptAction(0, 90), FunscriptAction(1000, 0)],
            "carrier_frequency": [FunscriptAction(0, 100), FunscriptAction(1000, 0)],
            "frequency": [FunscriptAction(0, 100), FunscriptAction(1000, 0)],
        }

        self.assertTrue(win.load_converted_preview(preview_axes, base_name="clip"))

        self.assertIsNotNone(win._multi_axis)
        self.assertEqual(
            [(a.at, a.pos) for a in win._positions.actions],
            [(0, 10), (1000, 20)],
        )
        self.assertEqual(
            [(a.at, a.pos) for a in win._multi_axis.axes["carrier_frequency"]],
            [(0, 100), (1000, 0)],
        )
        self.assertEqual(win._current_edit_axis, "e1")
        self.assertIn("Converted preview: clip", win.file_label.text())

    def test_load_converted_preview_auto_loads_matching_media(self):
        sr = 48_000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_path = root / "clip.wav"
            samples = self._make_click_track(sr, duration_s=4.0, bpm=120.0)
            self._write_wav(media_path, samples, sr)

            win = PMVGeneratorWindow()
            preview_axes = {
                "e1": [FunscriptAction(0, 10), FunscriptAction(1000, 20)],
                "e2": [FunscriptAction(0, 30), FunscriptAction(1000, 40)],
            }

            self.assertTrue(
                win.load_converted_preview(
                    preview_axes,
                    base_name="clip",
                    source_folder=root,
                )
            )

            self.assertEqual(win._media_path, str(media_path))
            self.assertIsNotNone(win._samples)
            self.assertIn(media_path.name, win.file_label.text())

    def test_send_position_at_time_emits_e1_to_e4_tags(self):
        class DummyEngine:
            def __init__(self):
                self.connected = True
                self.sent = []

            def send_immediate(self, cmd):
                self.sent.append(cmd)

        win = PMVGeneratorWindow()
        win._multi_axis = MultiAxisResult(
            axes={
                "alpha": [FunscriptAction(0, 50), FunscriptAction(1000, 50)],
                "beta": [FunscriptAction(0, 50), FunscriptAction(1000, 50)],
                "e1": [FunscriptAction(0, 10), FunscriptAction(1000, 10)],
                "e2": [FunscriptAction(0, 20), FunscriptAction(1000, 20)],
                "e3": [FunscriptAction(0, 30), FunscriptAction(1000, 30)],
                "e4": [FunscriptAction(0, 40), FunscriptAction(1000, 40)],
            }
        )
        win._network_engine = DummyEngine()
        combo = win.controls._preview_tcode_mode_combo
        combo.setCurrentIndex(combo.findData("fourphase"))
        win.aux_panel.get_send_axes = lambda: {"e1", "e2", "e3", "e4"}

        win._send_position_at_time(500.0)

        self.assertEqual(len(win._network_engine.sent), 1)
        cmd = win._network_engine.sent[0]
        self.assertFalse(cmd.include_linear_axes)
        self.assertNotIn("E0", cmd.tcode_tags)
        self.assertEqual(cmd.tcode_tags["E1"], 999)
        self.assertEqual(cmd.tcode_tags["E1_duration"], 33)
        self.assertEqual(cmd.tcode_tags["E2"], 1999)
        self.assertEqual(cmd.tcode_tags["E2_duration"], 33)
        self.assertEqual(cmd.tcode_tags["E3"], 2999)
        self.assertEqual(cmd.tcode_tags["E3_duration"], 33)
        self.assertEqual(cmd.tcode_tags["E4"], 3999)
        self.assertEqual(cmd.tcode_tags["E4_duration"], 33)
        wire = cmd.to_tcode()
        self.assertNotIn("L0", wire)
        self.assertNotIn("L1", wire)
        self.assertIn("E10999I33", wire)
        self.assertIn("E43999I33", wire)

    def test_send_position_at_time_threephase_omits_electrode_tags(self):
        class DummyEngine:
            def __init__(self):
                self.connected = True
                self.sent = []

            def send_immediate(self, cmd):
                self.sent.append(cmd)

        win = PMVGeneratorWindow()
        win._multi_axis = MultiAxisResult(
            axes={
                "alpha": [FunscriptAction(0, 25), FunscriptAction(1000, 25)],
                "beta": [FunscriptAction(0, 75), FunscriptAction(1000, 75)],
                "e1": [FunscriptAction(0, 10), FunscriptAction(1000, 10)],
                "e2": [FunscriptAction(0, 20), FunscriptAction(1000, 20)],
            }
        )
        win._network_engine = DummyEngine()
        combo = win.controls._preview_tcode_mode_combo
        combo.setCurrentIndex(combo.findData("threephase"))
        win.aux_panel.get_send_axes = lambda: {"alpha", "beta", "e1", "e2", "e3", "e4"}

        win._send_position_at_time(500.0)

        self.assertEqual(len(win._network_engine.sent), 1)
        cmd = win._network_engine.sent[0]
        self.assertTrue(cmd.include_linear_axes)
        self.assertNotIn("E1", cmd.tcode_tags)
        self.assertNotIn("E2", cmd.tcode_tags)
        wire = cmd.to_tcode()
        self.assertIn("L0", wire)
        self.assertIn("L1", wire)
        self.assertNotIn("E1", wire)
        self.assertNotIn("E2", wire)


if __name__ == "__main__":
    unittest.main()
