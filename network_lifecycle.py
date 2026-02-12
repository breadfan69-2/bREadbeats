from typing import Callable, Optional

from config import Config
from network_engine import NetworkEngine


def ensure_network_engine(
    existing_engine: Optional[NetworkEngine],
    config: Config,
    status_callback,
    *,
    dry_run_enabled: Optional[bool] = None,
    force_new: bool = False,
    engine_factory: Callable[[Config, object], NetworkEngine] = NetworkEngine,
) -> NetworkEngine:
    """Create/start a network engine if needed and apply dry-run if provided."""
    engine = None if force_new else existing_engine

    if engine is None:
        engine = engine_factory(config, status_callback)
        if dry_run_enabled is not None:
            engine.set_dry_run(dry_run_enabled)
        engine.start()
        return engine

    if dry_run_enabled is not None:
        engine.set_dry_run(dry_run_enabled)
    return engine


def toggle_user_connection(engine: NetworkEngine) -> None:
    """Toggle user connection state for an already created engine."""
    if engine.connected:
        engine.user_disconnect()
    else:
        engine.user_connect()
