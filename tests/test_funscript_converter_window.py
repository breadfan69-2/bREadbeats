from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication

from funscript_converter_window import FunscriptConverterWindow
from pmv_funscript_io import FunscriptAction, read_funscript


def _make_actions(pairs: list[tuple[int, int]]) -> list[FunscriptAction]:
    return [FunscriptAction(at=t, pos=p) for t, p in pairs]


class TestFunscriptConverterWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

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