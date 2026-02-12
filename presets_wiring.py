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
