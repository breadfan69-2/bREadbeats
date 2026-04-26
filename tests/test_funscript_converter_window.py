from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication

from funscript_converter_window import FunscriptConverterWindow
from pmv_funscript_io import FunscriptAction, FunscriptMetadata, read_funscript, write_funscript


def _make_actions(pairs: list[tuple[int, int]]) -> list[FunscriptAction]:
    return [FunscriptAction(at=t, pos=p) for t, p in pairs]


class TestFunscriptConverterWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_preview_in_generator_uses_callback(self):
        received = []

        def _preview_callback(
            base_name: str,
            axes: dict[str, list[FunscriptAction]],
            source_folder: Path | None,
        ) -> None:
            received.append((base_name, axes, source_folder))

        win = FunscriptConverterWindow(preview_callback=_preview_callback)
        win._loaded_axes = {
            "main": _make_actions([(0, 20), (500, 70), (1000, 30)]),
            "surge": _make_actions([(0, 60), (500, 40), (1000, 80)]),
        }
        win._source_folder = Path(tempfile.gettempdir())
        win._base_stem = "clip"
        win._freq_enabled.setChecked(True)
        win._run_conversion()

        win._on_preview_in_generator()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], "clip")
        self.assertIn("e1", received[0][1])
        self.assertIn("pulse_frequency", received[0][1])
        self.assertIn("carrier_frequency", received[0][1])
        self.assertIn("frequency", received[0][1])
        self.assertEqual(received[0][2], Path(tempfile.gettempdir()))
        win.close()

    def test_load_file_accepts_multi_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main_path = root / "clip.funscript"
            surge_path = root / "clip.surge.funscript"

            write_funscript(
                main_path,
                _make_actions([(0, 20), (500, 70), (1000, 30)]),
                FunscriptMetadata(title="clip", duration=1000),
            )
            write_funscript(
                surge_path,
                _make_actions([(0, 60), (500, 40), (1000, 80)]),
                FunscriptMetadata(title="clip.surge", duration=1000),
            )

            win = FunscriptConverterWindow()

            with patch(
                "funscript_converter_window.QFileDialog.getOpenFileNames",
                return_value=([str(main_path), str(surge_path)], "Funscript Files (*.funscript)"),
            ):
                win._on_load_file()

            self.assertEqual(win._base_stem, "clip")
            self.assertEqual(win._source_folder, root)
            self.assertIn("main", win._loaded_axes)
            self.assertIn("surge", win._loaded_axes)
            self.assertEqual(
                [(a.at, a.pos) for a in win._loaded_axes["main"]],
                [(0, 20), (500, 70), (1000, 30)],
            )
            self.assertEqual(
                [(a.at, a.pos) for a in win._loaded_axes["surge"]],
                [(0, 60), (500, 40), (1000, 80)],
            )

            win.close()

    def test_export_flushes_pending_frequency_reconvert(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_dir = root / "out"
            export_dir.mkdir()

            win = FunscriptConverterWindow()
            win._loaded_axes = {
                "main": _make_actions([(0, 50), (500, 50), (1000, 50)]),
                "surge": _make_actions([(0, 50), (500, 50), (1000, 50)]),
            }
            win._source_folder = root
            win._base_stem = "sample"

            win._run_conversion()
            self.assertNotIn("pulse_frequency", win._result)

            win._freq_enabled.setChecked(True)
            self.assertTrue(win._reconvert_timer.isActive())

            with patch(
                "funscript_converter_window.QFileDialog.getExistingDirectory",
                return_value=str(export_dir),
            ), patch("funscript_converter_window.QMessageBox.information"):
                win._on_export()

            self.assertFalse(win._reconvert_timer.isActive())

            pulse_actions, _ = read_funscript(export_dir / "sample.pulse_frequency.funscript")
            carrier_actions, _ = read_funscript(export_dir / "sample.carrier_frequency.funscript")
            frequency_actions, _ = read_funscript(export_dir / "sample.frequency.funscript")

            self.assertEqual([a.pos for a in pulse_actions], [45, 45, 45])
            self.assertEqual([a.pos for a in carrier_actions], [50, 50, 50])
            self.assertEqual(
                [a.pos for a in frequency_actions],
                [a.pos for a in carrier_actions],
            )

            win.close()