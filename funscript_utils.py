from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmv_funscript_io import FunscriptAction

AXIS_SUFFIXES = {
    "main", "alpha", "beta", "alpha_prostate", "beta_prostate",
    "e1", "e2", "e3", "e4",
    "frequency", "pulse_frequency", "volume", "pulse_rise", "pulse_width",
    "surge", "sway", "twist", "roll", "pitch",
    "carrier_frequency",
}

_SORTED_SUFFIXES: list[str] | None = None


def _get_sorted_suffixes() -> list[str]:
    global _SORTED_SUFFIXES
    if _SORTED_SUFFIXES is None:
        _SORTED_SUFFIXES = sorted(AXIS_SUFFIXES, key=len, reverse=True)
    return _SORTED_SUFFIXES


def strip_axis_suffix(name: str) -> tuple[str, str | None]:
    """Return (base_stem, axis_name) by stripping a known axis suffix.

    ``'video.alpha'`` → ``('video', 'alpha')``
    ``'video'``       → ``('video', None)``
    """
    for suffix in _get_sorted_suffixes():
        tail = f".{suffix}"
        if name.lower().endswith(tail):
            return name[: len(name) - len(tail)], suffix
    return name, None


def axis_name_from_file(path: Path) -> str | None:
    """Infer axis name from filename; returns None for plain non-axis stems."""
    stem = path.stem
    lowered = stem.lower()
    if lowered == "main":
        return "main"
    _base_stem, suffix = strip_axis_suffix(stem)
    if suffix is not None:
        return suffix
    if lowered in AXIS_SUFFIXES:
        return lowered
    return None


def discover_sibling_axes(
    script_path: Path,
    suffixes: set[str] | None = None,
) -> dict[str, list[FunscriptAction]]:
    """Find sibling axis funscript files and return {axis_name: actions}."""
    from pmv_funscript_io import read_funscript

    if suffixes is None:
        suffixes = AXIS_SUFFIXES

    base_stem, _selected_axis = strip_axis_suffix(script_path.stem)
    folder = script_path.parent
    axes: dict[str, list[FunscriptAction]] = {}
    for suffix in suffixes:
        if suffix == "main":
            candidate = folder / f"{base_stem}.funscript"
        else:
            candidate = folder / f"{base_stem}.{suffix}.funscript"
        if not candidate.exists() or not candidate.is_file():
            continue
        if candidate == script_path:
            continue
        try:
            sibling_actions, _ = read_funscript(candidate)
            if sibling_actions:
                axes[suffix] = sibling_actions
        except Exception:
            pass
    return axes


def load_folder(
    folder: Path,
    suffixes: set[str] | None = None,
) -> dict[str, list[FunscriptAction]]:
    """Load all funscript axes from a folder, grouped by axis name.

    Returns {axis_name: actions} for the first base name found.
    Files with no recognized suffix are treated as 'main'.
    """
    from pmv_funscript_io import read_funscript

    if suffixes is None:
        suffixes = AXIS_SUFFIXES

    candidates = sorted(folder.glob("*.funscript"))
    if not candidates:
        return {}

    # Group by base name — use the first base name found
    target_base: str | None = None
    axes: dict[str, list[FunscriptAction]] = {}

    for path in candidates:
        base_stem, suffix = strip_axis_suffix(path.stem)
        if target_base is None:
            target_base = base_stem.lower()
        if base_stem.lower() != target_base:
            continue

        axis_name = suffix if suffix is not None else "main"
        try:
            actions, _ = read_funscript(path)
            if actions:
                axes[axis_name] = actions
        except Exception:
            pass

    return axes
