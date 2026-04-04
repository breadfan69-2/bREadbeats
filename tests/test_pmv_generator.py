from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication

from pmv_generator import PMVGeneratorWindow
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
            win._run_live_preview()

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


if __name__ == "__main__":
    unittest.main()
