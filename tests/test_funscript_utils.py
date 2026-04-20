"""Tests for funscript_utils — shared axis helpers."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from funscript_utils import (
    AXIS_SUFFIXES,
    axis_name_from_file,
    discover_sibling_axes,
    load_folder,
    strip_axis_suffix,
)


class TestStripAxisSuffix:
    def test_known_suffix(self):
        assert strip_axis_suffix("video.alpha") == ("video", "alpha")

    def test_no_suffix(self):
        assert strip_axis_suffix("video") == ("video", None)

    def test_main_suffix(self):
        assert strip_axis_suffix("video.main") == ("video", "main")

    def test_new_suffixes(self):
        for s in ("surge", "sway", "twist", "roll", "pitch", "carrier_frequency"):
            base, suffix = strip_axis_suffix(f"myvideo.{s}")
            assert suffix == s, f"Failed for {s}"
            assert base == "myvideo"

    def test_longest_match_wins(self):
        # pulse_frequency should not match just "frequency"
        base, suffix = strip_axis_suffix("test.pulse_frequency")
        assert suffix == "pulse_frequency"
        assert base == "test"

    def test_case_insensitive(self):
        base, suffix = strip_axis_suffix("Video.Alpha")
        assert suffix == "alpha"

    def test_all_known_suffixes_present(self):
        expected = {
            "main", "alpha", "beta", "alpha_prostate", "beta_prostate",
            "e1", "e2", "e3", "e4",
            "frequency", "pulse_frequency", "volume", "pulse_rise", "pulse_width",
            "surge", "sway", "twist", "roll", "pitch",
            "carrier_frequency",
        }
        assert AXIS_SUFFIXES == expected


class TestAxisNameFromFile:
    def test_axis_file(self):
        assert axis_name_from_file(Path("video.roll.funscript")) == "roll"

    def test_plain_file(self):
        assert axis_name_from_file(Path("video.funscript")) is None

    def test_main_stem(self):
        assert axis_name_from_file(Path("main.funscript")) == "main"


def _write_funscript(path: Path, actions: list[dict]) -> None:
    payload = {"version": "1.0", "actions": actions}
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestDiscoverSiblingAxes:
    def test_finds_siblings(self, tmp_path):
        _write_funscript(tmp_path / "vid.funscript", [{"at": 0, "pos": 50}])
        _write_funscript(tmp_path / "vid.surge.funscript", [{"at": 0, "pos": 30}])
        _write_funscript(tmp_path / "vid.roll.funscript", [{"at": 0, "pos": 70}])

        primary = tmp_path / "vid.funscript"
        siblings = discover_sibling_axes(primary, {"main", "surge", "roll"})
        assert "surge" in siblings
        assert "roll" in siblings
        assert "main" not in siblings  # primary excluded

    def test_empty_folder(self, tmp_path):
        _write_funscript(tmp_path / "vid.funscript", [{"at": 0, "pos": 50}])
        siblings = discover_sibling_axes(tmp_path / "vid.funscript", {"surge"})
        assert siblings == {}


class TestLoadFolder:
    def test_loads_all_axes(self, tmp_path):
        _write_funscript(tmp_path / "vid.funscript", [{"at": 0, "pos": 50}])
        _write_funscript(tmp_path / "vid.sway.funscript", [{"at": 0, "pos": 20}])

        axes = load_folder(tmp_path, {"main", "sway"})
        assert "main" in axes
        assert "sway" in axes

    def test_empty_folder(self, tmp_path):
        axes = load_folder(tmp_path)
        assert axes == {}
