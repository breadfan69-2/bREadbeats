from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication

from config import Config
from funscript_converter_window import FunscriptConverterWindow
from pmv_funscript_io import FunscriptAction, FunscriptMetadata, read_funscript, write_funscript


def _make_actions(pairs: list[tuple[int, int]]) -> list[FunscriptAction]:
    return [FunscriptAction(at=t, pos=p) for t, p in pairs]


class TestFunscriptConverterWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_frequency_preview_defaults_on_and_uses_horizontal_zoom(self):
        win = FunscriptConverterWindow()

        self.assertTrue(win._freq_enabled.isChecked())
        self.assertEqual(len(win._plots), 6)
        for plot in win._plots:
            self.assertEqual(plot.getViewBox().state["mouseEnabled"], [True, False])

        self.assertAlmostEqual(win._pulse_surge_influence.value(), 1.0)
        self.assertAlmostEqual(win._pulse_speed_influence.value(), 0.15)
        self.assertAlmostEqual(win._carrier_surge_influence.value(), 1.0)
        self.assertAlmostEqual(win._carrier_speed_influence.value(), 0.10)

        win.close()

    def test_electrode_preview_colors_match_focstim_outputs(self):
        win = FunscriptConverterWindow()

        colors = [win._curves[index].opts["pen"].color().name() for index in range(4)]

        self.assertEqual(colors, ["#ff6b6b", "#4d96ff", "#ffd93d", "#6bcb77"])

        win.close()

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

    def test_preview_plots_include_frequency_axes(self):
        win = FunscriptConverterWindow()
        win._loaded_axes = {
            "main": _make_actions([(0, 0), (500, 50), (1000, 100)]),
            "surge": _make_actions([(0, 0), (500, 50), (1000, 100)]),
        }

        win._run_conversion()

        pulse_t, pulse_y = win._curves[4].getData()
        carrier_t, carrier_y = win._curves[5].getData()

        assert pulse_t is not None
        assert pulse_y is not None
        assert carrier_t is not None
        assert carrier_y is not None
        self.assertEqual(list(pulse_t), [0.0, 0.5, 1.0])
        self.assertEqual(list(pulse_y), [80, 59, 25])
        self.assertEqual(list(carrier_y), [60, 51, 43])
        self.assertEqual(list(carrier_t), [0.0, 0.5, 1.0])

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
            win._freq_enabled.setChecked(False)
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

            self.assertEqual([a.pos for a in pulse_actions], [55, 55, 55])
            self.assertEqual([a.pos for a in carrier_actions], [50, 50, 50])
            self.assertEqual(
                [a.pos for a in frequency_actions],
                [a.pos for a in carrier_actions],
            )

            win.close()

    def test_freq_config_uses_explicit_center_and_bounds(self):
        win = FunscriptConverterWindow()
        win._freq_enabled.setChecked(True)
        win._freq_scale.setValue(1.5)
        win._pulse_surge_influence.setValue(1.75)
        win._pulse_speed_influence.setValue(0.2)
        win._pulse_center.setValue(58.0)
        win._pulse_min.setValue(22.0)
        win._pulse_max.setValue(76.0)
        win._carrier_scale.setValue(1.2)
        win._carrier_surge_influence.setValue(1.5)
        win._carrier_speed_influence.setValue(0.12)
        win._carrier_center.setValue(51.0)
        win._carrier_min.setValue(41.0)
        win._carrier_max.setValue(59.0)

        cfg = win._get_freq_config()

        self.assertTrue(cfg.enabled)
        self.assertAlmostEqual(cfg.freq_scale, 1.5)
        self.assertAlmostEqual(cfg.pulse_surge_influence, 1.75)
        self.assertAlmostEqual(cfg.pulse_speed_influence, 0.2)
        self.assertAlmostEqual(cfg.pulse_center, 58.0)
        self.assertAlmostEqual(cfg.pulse_min, 22.0)
        self.assertAlmostEqual(cfg.pulse_max, 76.0)
        self.assertAlmostEqual(cfg.carrier_scale, 1.2)
        self.assertAlmostEqual(cfg.carrier_surge_influence, 1.5)
        self.assertAlmostEqual(cfg.carrier_speed_influence, 0.12)
        self.assertAlmostEqual(cfg.carrier_center, 51.0)
        self.assertAlmostEqual(cfg.carrier_min, 41.0)
        self.assertAlmostEqual(cfg.carrier_max, 59.0)

        win.close()

    def test_layout_and_wiring_restore_and_persist(self):
        saved_states: list[tuple[str, list[int]]] = []
        config = Config()
        config.funscript_converter.layout_model = "Pair At Bottom / Rear"
        config.funscript_converter.wiring_map = [2, 1, 0, 3]

        def _save_settings(updated_config: Config) -> bool:
            saved_states.append(
                (
                    updated_config.funscript_converter.layout_model,
                    list(updated_config.funscript_converter.wiring_map),
                )
            )
            return True

        win = FunscriptConverterWindow(config=config, save_settings=_save_settings)

        self.assertEqual(win._get_current_layout_model(), "Pair At Bottom / Rear")
        self.assertEqual(win._get_current_wiring_map(), (2, 1, 0, 3))

        win._layout_combo.setCurrentIndex(win._layout_combo.findData("Pair At Top"))
        win._swap_electrodes(0, 3)

        self.assertEqual(config.funscript_converter.layout_model, "Pair At Top")
        self.assertEqual(config.funscript_converter.wiring_map, [3, 1, 0, 2])
        self.assertGreaterEqual(len(saved_states), 2)
        self.assertEqual(saved_states[-1], ("Pair At Top", [3, 1, 0, 2]))

        win.close()