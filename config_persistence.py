import json
import sys
from dataclasses import asdict
from pathlib import Path

from config import (
    Config,
    apply_dict_to_dataclass,
    migrate_config,
)


def get_config_dir() -> Path:
    """Get config directory - exe folder when packaged, home dir otherwise."""
    if getattr(sys, 'frozen', False):
        exe_dir = Path(sys.executable).parent
        probe = exe_dir / '.breadbeats_write_test.tmp'
        try:
            with open(probe, 'w', encoding='utf-8') as f:
                f.write('ok')
            probe.unlink(missing_ok=True)
            return exe_dir
        except Exception:
            config_dir = Path.home() / '.breadbeats'
            config_dir.mkdir(parents=True, exist_ok=True)
            return config_dir

    config_dir = Path.home() / '.breadbeats'
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_file() -> Path:
    """Get config file path."""
    return get_config_dir() / 'config.json'


def save_config(config: Config) -> bool:
    """Save config to JSON file."""
    try:
        config_file = get_config_file()
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2)
        print(f"[Config] Saved to {config_file}")
        return True
    except Exception as e:
        print(f"[Config] Failed to save: {e}")
        return False


def load_config() -> Config:
    """Load config from JSON file, returns default if not found."""
    try:
        config_file = get_config_file()
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            config = Config()
            apply_dict_to_dataclass(config, data)
            loaded_version = data.get('version')
            migrate_config(config, loaded_version)

            version = getattr(config, 'version', 'unknown')
            print(f"[Config] Loaded from {config_file} (version={version})")

            if loaded_version != version:
                try:
                    save_config(config)
                except Exception as e:
                    print(f"[Config] Warning: could not auto-save migrated config: {e}")
            return config

        print("[Config] No saved config found, using defaults")
        return Config()
    except Exception as e:
        print(f"[Config] Failed to load: {e}, using defaults")
        return Config()
