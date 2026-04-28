from __future__ import annotations

import types
import unittest

from PyQt6.QtWidgets import QApplication, QMainWindow

from ui_builders import create_menu_bar


class _DummyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            audio=types.SimpleNamespace(fft_size=1024, spectrum_skip_frames=2),
            stroke=types.SimpleNamespace(intelligence_enabled=True, simple_mode_beats_per_rotation=1),
            jitter=types.SimpleNamespace(enabled=True),
            beat=types.SimpleNamespace(tempo_lock_required=False),
            log_level="INFO",
        )
        self._pmv_launch_count = 0

    def _on_options_connection(self):
        return None

    def _on_connect(self):
        return None

    def _on_test(self):
        return None

    def _on_options_audio_device(self):
        return None

    def _on_device_limits(self):
        return None

    def _on_menu_fft_change(self, _index: int):
        return None

    def _on_menu_spectrum_change(self, _index: int):
        return None

    def _on_fft_bin_diagnostics(self):
        return None

    def _on_options_beat_detection(self):
        return None

    def _on_options_motion_settings(self):
        return None

    def _on_viz_menu_change(self, _index: int):
        return None

    def _on_intelligence_toggle(self, _checked: bool):
        return None

    def _on_beats_per_rotation_change(self, _value: int):
        return None

    def _on_effects_jitter_toggle(self):
        return None

    def _on_tempo_lock_required_toggle(self, _checked: bool):
        return None

    def _open_developer_controls_window(self):
        return None

    def _on_log_level_change(self, _level: str):
        return None

    def _sync_log_level_menu(self, _active_level: str):
        return None

    def _on_help(self):
        return None

    def _on_about(self):
        return None

    def _launch_pmv_generator(self):
        self._pmv_launch_count += 1

    def _launch_funscript_converter(self):
        return None


class TestPmvMenuIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_menu_contains_pmv_launcher_action(self):
        win = _DummyWindow()
        create_menu_bar(win)

        menubar = win.menuBar()
        self.assertIsNotNone(menubar)

        main_menu = None
        for action in menubar.actions():
            menu = action.menu()
            if menu is not None and action.text() == "Menu":
                main_menu = menu
                break

        self.assertIsNotNone(main_menu)
        assert main_menu is not None

        pmv_action = None
        for action in main_menu.actions():
            if action.text() == "PMV Funscript Generator...":
                pmv_action = action
                break

        self.assertIsNotNone(pmv_action)
        assert pmv_action is not None

        pmv_action.trigger()
        self.assertEqual(win._pmv_launch_count, 1)


if __name__ == "__main__":
    unittest.main()
