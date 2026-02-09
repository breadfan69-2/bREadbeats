"""Lightweight logging helper for console-tagged messages.

Provides level+tag output while keeping the existing print-style simplicity.
"""
from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger("breadbeats")
if not _logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s][%(tag)s] %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)


class _TagAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: dict[str, Any]):
        tag = kwargs.pop("tag", "App")
        kwargs.setdefault("extra", {})["tag"] = tag
        return msg, kwargs


_logger_adapter = _TagAdapter(_logger, {})


def log_event(level: str, tag: str, message: str, **fields: Any) -> None:
    """Log a message with level+tag, appending key=value fields when provided."""
    if fields:
        extras = " ".join(f"{k}={v}" for k, v in fields.items())
        message = f"{message} | {extras}"
    level_name = level.upper()
    level_val = getattr(logging, level_name, logging.INFO)
    _logger_adapter.log(level_val, message, tag=tag)


def set_log_level(level: str) -> None:
    """Set global log level (DEBUG/INFO/WARNING/ERROR)."""
    level_name = (level or "INFO").upper()
    level_val = getattr(logging, level_name, logging.INFO)
    _logger.setLevel(level_val)


def get_log_level() -> str:
    """Return current global log level name."""
    return logging.getLevelName(_logger.level)
