from pathlib import Path

import config_persistence
from config import Config


def get_config_dir() -> Path:
    return config_persistence.get_config_dir()


def get_config_file() -> Path:
    return config_persistence.get_config_file()


def save_config(config: Config) -> bool:
    return config_persistence.save_config(config)


def load_config() -> Config:
    return config_persistence.load_config()
