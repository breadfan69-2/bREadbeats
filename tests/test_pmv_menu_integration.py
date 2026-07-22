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
            live_tcode_mode="fourphase",
            live_fourphase_model="classic",
            live_fourphase_layout_model="Straight Line",
            live_fourphase_beat_radius_contrast_strength=0.0,
            live_fourphase_beat_speed_spread_strength=0.0,
            live_fourphase_beat_response_curves=["linear", "linear", "linear", "linear"],
            live_fourphase_band_mapping=[["mid", "high"], ["low_mid", "mid"], ["sub_bass", "low_mid"], ["sub_bass"]],
            live_fourphase_bandrouter_fill_mix=0.12,
            live_fourphase_bandrouter_idle_floor=0.10,
            live_fourphase_vertical_lift_mix=0.9,
            live_fourphase_vertical_lift_curve=1.0,
            live_fourphase_center_drift_mix=0.35,
            live_fourphase_trigger_bias_mix=1.0,
            live_fourphase_vertical_lift_band="sub_bass",
        )
        self._pmv_launch_count = 0
        self._converter_launch_count = 0
        self._live_fourphase_popup_count = 0
        self._live_mode_changes: list[str] = []
        self._live_fourphase_model_changes: list[str] = []
        self._live_fourphase_layout_changes: list[str] = []
        self._live_fourphase_band_mapping_changes: list[list[list[str]]] = []
        self._live_fourphase_bandrouter_fill_mix_changes: list[float] = []
        self._live_fourphase_bandrouter_idle_floor_changes: list[float] = []
        self._live_fourphase_vertical_lift_changes: list[float] = []
        self._live_fourphase_vertical_lift_curve_changes: list[float] = []
        self._live_fourphase_center_drift_changes: list[float] = []
        self._live_fourphase_trigger_bias_changes: list[float] = []
        self._live_fourphase_vertical_lift_band_changes: list[str] = []

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

    def _on_live_tcode_mode_change(self, mode: str):
        self._live_mode_changes.append(mode)
        self.config.live_tcode_mode = mode

    def _on_live_fourphase_model_change(self, model: str):
        self._live_fourphase_model_changes.append(model)
        self.config.live_fourphase_model = model

    def _on_live_fourphase_options_popup(self):
        self._live_fourphase_popup_count += 1

    @staticmethod
    def _normalize_live_fourphase_layout_model(layout_model: str | None) -> str:
        normalized = str(layout_model or "Straight Line").strip().lower()
        if normalized in {"straight", "straight line"}:
            return "Straight Line"
        if normalized == "pair at top":
            return "Pair At Top"
        if normalized == "pair at bottom / rear":
            return "Pair At Bottom / Rear"
        return "Pair At Middle" if normalized == "pair at middle" else "Straight Line"

    def _on_live_fourphase_layout_model_change(self, layout_model: str):
        normalized = self._normalize_live_fourphase_layout_model(layout_model)
        self._live_fourphase_layout_changes.append(normalized)
        self.config.live_fourphase_layout_model = normalized

    @staticmethod
    def _normalize_live_fourphase_band_mapping(mapping) -> list[list[str]]:
        defaults = [["mid", "high"], ["low_mid", "mid"], ["sub_bass", "low_mid"], ["sub_bass"]]
        if not isinstance(mapping, (list, tuple)) or len(mapping) != 4:
            return defaults
        normalized = []
        for index, entry in enumerate(mapping):
            if not isinstance(entry, (list, tuple)):
                normalized.append(list(defaults[index]))
                continue
            selected = []
            for raw_band in entry:
                band = str(raw_band or "sub_bass").strip().lower()
                if band not in {"sub_bass", "low_mid", "mid", "high"}:
                    continue
                if band not in selected:
                    selected.append(band)
                if len(selected) >= 3:
                    break
            normalized.append(selected if selected else list(defaults[index]))
        return normalized

    def _on_live_fourphase_bandrouter_mapping_change(self, electrode_index: int, band: str, checked: bool):
        mapping = self._normalize_live_fourphase_band_mapping(self.config.live_fourphase_band_mapping)
        selected = list(mapping[electrode_index])
        normalized_band = str(band or "sub_bass").strip().lower()
        if checked:
            if normalized_band not in selected and len(selected) < 3:
                selected.append(normalized_band)
        else:
            if normalized_band in selected and len(selected) > 1:
                selected = [entry for entry in selected if entry != normalized_band]
        mapping[electrode_index] = selected
        self._live_fourphase_band_mapping_changes.append([list(group) for group in mapping])
        self.config.live_fourphase_band_mapping = mapping

    def _prompt_live_fourphase_vertical_lift_mix(self):
        value, accepted = (1.25, True)
        if accepted:
            self._on_live_fourphase_vertical_lift_mix_change(value)

    def _prompt_live_fourphase_bandrouter_fill_mix(self):
        value, accepted = (0.28, True)
        if accepted:
            self._on_live_fourphase_bandrouter_fill_mix_change(value)

    def _on_live_fourphase_bandrouter_fill_mix_change(self, mix: float):
        self._live_fourphase_bandrouter_fill_mix_changes.append(mix)
        self.config.live_fourphase_bandrouter_fill_mix = mix

    def _prompt_live_fourphase_bandrouter_idle_floor(self):
        value, accepted = (0.18, True)
        if accepted:
            self._on_live_fourphase_bandrouter_idle_floor_change(value)

    def _on_live_fourphase_bandrouter_idle_floor_change(self, idle_floor: float):
        self._live_fourphase_bandrouter_idle_floor_changes.append(idle_floor)
        self.config.live_fourphase_bandrouter_idle_floor = idle_floor

    def _on_live_fourphase_vertical_lift_mix_change(self, mix: float):
        self._live_fourphase_vertical_lift_changes.append(mix)
        self.config.live_fourphase_vertical_lift_mix = mix

    def _prompt_live_fourphase_vertical_lift_curve(self):
        value, accepted = (1.5, True)
        if accepted:
            self._on_live_fourphase_vertical_lift_curve_change(value)

    def _on_live_fourphase_vertical_lift_curve_change(self, curve: float):
        self._live_fourphase_vertical_lift_curve_changes.append(curve)
        self.config.live_fourphase_vertical_lift_curve = curve

    def _prompt_live_fourphase_center_drift_mix(self):
        value, accepted = (0.55, True)
        if accepted:
            self._on_live_fourphase_center_drift_mix_change(value)

    def _on_live_fourphase_center_drift_mix_change(self, mix: float):
        self._live_fourphase_center_drift_changes.append(mix)
        self.config.live_fourphase_center_drift_mix = mix

    def _prompt_live_fourphase_trigger_bias_mix(self):
        value, accepted = (1.4, True)
        if accepted:
            self._on_live_fourphase_trigger_bias_mix_change(value)

    def _on_live_fourphase_trigger_bias_mix_change(self, mix: float):
        self._live_fourphase_trigger_bias_changes.append(mix)
        self.config.live_fourphase_trigger_bias_mix = mix

    def _on_live_fourphase_vertical_lift_band_change(self, band: str):
        self._live_fourphase_vertical_lift_band_changes.append(band)
        self.config.live_fourphase_vertical_lift_band = band

    def _sync_live_fourphase_vertical_lift_action(self):
        action = getattr(self, '_live_fourphase_vertical_lift_action', None)
        if action is not None:
            action.setText(f"Vertical Lift Mix... ({self.config.live_fourphase_vertical_lift_mix:.2f}x)")

    def _sync_live_fourphase_layout_menu(self, _active_layout: str | None = None):
        return None

    def _sync_live_fourphase_bandrouter_mapping_menu(self):
        return None

    def _sync_live_fourphase_vertical_lift_curve_action(self):
        action = getattr(self, '_live_fourphase_vertical_lift_curve_action', None)
        if action is not None:
            action.setText(f"Vertical Lift Curve... ({self.config.live_fourphase_vertical_lift_curve:.2f})")

    def _sync_live_fourphase_center_drift_action(self):
        action = getattr(self, '_live_fourphase_center_drift_action', None)
        if action is not None:
            action.setText(f"Center Drift Mix... ({self.config.live_fourphase_center_drift_mix:.2f}x)")

    def _sync_live_fourphase_bandrouter_fill_mix_action(self):
        action = getattr(self, '_live_fourphase_bandrouter_fill_mix_action', None)
        if action is not None:
            action.setText(f"Band-Routed Fill Proximity... ({self.config.live_fourphase_bandrouter_fill_mix:.2f}x)")

    def _sync_live_fourphase_bandrouter_idle_floor_action(self):
        action = getattr(self, '_live_fourphase_bandrouter_idle_floor_action', None)
        if action is not None:
            action.setText(f"Band-Routed Idle Floor... ({self.config.live_fourphase_bandrouter_idle_floor:.2f}x)")

    def _sync_live_fourphase_trigger_bias_action(self):
        action = getattr(self, '_live_fourphase_trigger_bias_action', None)
        if action is not None:
            action.setText(f"Trigger Bias Mix... ({self.config.live_fourphase_trigger_bias_mix:.2f}x)")

    def _sync_live_fourphase_vertical_lift_band_menu(self, _active_band: str | None = None):
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
        self._converter_launch_count += 1


class TestPmvMenuIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_tools_menu_contains_pmv_and_converter_actions(self):
        win = _DummyWindow()
        create_menu_bar(win)

        menubar = win.menuBar()
        self.assertIsNotNone(menubar)
        assert menubar is not None

        main_menu = None
        tools_menu = None
        for action in menubar.actions():
            menu = action.menu()
            if menu is not None and action.text() == "Menu":
                main_menu = menu
            if menu is not None and action.text() == "Tools":
                tools_menu = menu

        self.assertIsNotNone(main_menu)
        assert main_menu is not None
        self.assertIsNotNone(tools_menu)
        assert tools_menu is not None

        pmv_action = None
        converter_action = None
        for action in tools_menu.actions():
            if action.text() == "PMV Funscript Generator...":
                pmv_action = action
            if action.text() == "FunScript Converter (6→4 Phase)...":
                converter_action = action

        self.assertIsNotNone(pmv_action)
        assert pmv_action is not None
        self.assertIsNotNone(converter_action)
        assert converter_action is not None

        self.assertNotIn(
            "PMV Funscript Generator...",
            [action.text() for action in main_menu.actions()],
        )
        self.assertNotIn(
            "FunScript Converter (6→4 Phase)...",
            [action.text() for action in main_menu.actions()],
        )

        pmv_action.trigger()
        converter_action.trigger()
        self.assertEqual(win._pmv_launch_count, 1)
        self.assertEqual(win._converter_launch_count, 1)

    def test_options_menu_contains_phase_mode_submenu(self):
        win = _DummyWindow()
        create_menu_bar(win)

        menubar = win.menuBar()
        self.assertIsNotNone(menubar)
        assert menubar is not None

        options_menu = None
        for action in menubar.actions():
            menu = action.menu()
            if menu is not None and action.text() == "Options":
                options_menu = menu
                break

        self.assertIsNotNone(options_menu)
        assert options_menu is not None

        live_mode_menu = None
        for action in options_menu.actions():
            menu = action.menu()
            if menu is not None and action.text() == "Phase Mode":
                live_mode_menu = menu
                self.assertIn("direct FOC V4", action.toolTip())
                break

        self.assertIsNotNone(live_mode_menu)
        assert live_mode_menu is not None

        mode_actions = {action.text(): action for action in live_mode_menu.actions()}
        self.assertTrue(mode_actions["4phase (E1-E4)"].isChecked())
        self.assertFalse(mode_actions["3phase (L0/L1)"].isChecked())

        mode_actions["3phase (L0/L1)"].trigger()
        self.assertEqual(win.config.live_tcode_mode, "threephase")
        self.assertEqual(win._live_mode_changes, ["threephase"])

    def test_options_menu_contains_fourphase_options_popup_action(self):
        win = _DummyWindow()
        create_menu_bar(win)

        menubar = win.menuBar()
        self.assertIsNotNone(menubar)
        assert menubar is not None

        options_menu = None
        for action in menubar.actions():
            menu = action.menu()
            if menu is not None and action.text() == "Options":
                options_menu = menu
                break

        self.assertIsNotNone(options_menu)
        assert options_menu is not None

        fourphase_options_action = None
        for action in options_menu.actions():
            if action.text() == "4-Phase Options...":
                fourphase_options_action = action
                break

        self.assertIsNotNone(fourphase_options_action)
        assert fourphase_options_action is not None
        self.assertIsNone(fourphase_options_action.menu())

        fourphase_options_action.trigger()
        self.assertEqual(win._live_fourphase_popup_count, 1)


if __name__ == "__main__":
    unittest.main()
