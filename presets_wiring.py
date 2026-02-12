import json
import shutil
from pathlib import Path


def get_presets_file_path(*, frozen: bool, executable_path: str, source_file: str) -> Path:
    """Resolve presets file path for packaged or source execution."""
    if frozen:
        return Path(executable_path).parent / "presets.json"
    return Path(source_file).parent / "presets.json"


def save_presets_data(presets_file: Path, custom_beat_presets: dict) -> None:
    """Persist custom presets mapping to disk."""
    with open(presets_file, 'w', encoding='utf-8') as f:
        json.dump(custom_beat_presets, f, indent=2)


def load_presets_data(
    presets_file: Path,
    *,
    frozen: bool,
    meipass: str | None,
) -> dict:
    """Load presets mapping from disk, copying bundled factory presets first when needed."""
    if not presets_file.exists() and frozen and meipass:
        factory_presets = Path(meipass) / 'presets.json'
        if factory_presets.exists():
            shutil.copy(factory_presets, presets_file)

    if presets_file.exists():
        with open(presets_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}

    return {}


def resolve_p0_tcode_bounds(preset_data: dict) -> tuple[int | None, int | None]:
    """Resolve P0 tcode min/max supporting old preset keys and legacy Hz-scale values."""
    p0_tcode_min = preset_data.get('tcode_min', preset_data.get('tcode_freq_min'))
    p0_tcode_max = preset_data.get('tcode_max', preset_data.get('tcode_freq_max'))

    if p0_tcode_min is not None and p0_tcode_min < 200:
        p0_tcode_min = int(p0_tcode_min * 67)
    if p0_tcode_max is not None and p0_tcode_max < 200:
        p0_tcode_max = int(p0_tcode_max * 67)

    return p0_tcode_min, p0_tcode_max
