import unittest
from typing import Any, cast

from config import Config
from network_lifecycle import ensure_network_engine, toggle_user_connection


class DummyEngine:
    def __init__(self, config, status_callback):
        self.config = config
        self.status_callback = status_callback
        self.started = False
        self.connected = False
        self.dry_run = None
        self.connect_calls = 0
        self.disconnect_calls = 0

    def start(self):
        self.started = True

    def set_dry_run(self, enabled: bool):
        self.dry_run = enabled

    def user_connect(self):
        self.connect_calls += 1
        self.connected = True

    def user_disconnect(self):
        self.disconnect_calls += 1
        self.connected = False


class TestNetworkLifecycle(unittest.TestCase):
    def test_ensure_network_engine_creates_and_starts(self):
        cfg = Config()
        engine = ensure_network_engine(
            None,
            cfg,
            status_callback=None,
            dry_run_enabled=True,
            engine_factory=cast(Any, DummyEngine),
        )
        dummy = cast(DummyEngine, engine)
        self.assertTrue(dummy.started)
        self.assertTrue(dummy.dry_run)

    def test_ensure_network_engine_reuses_existing(self):
        cfg = Config()
        existing = DummyEngine(cfg, None)
        existing.started = True

        reused = ensure_network_engine(
            cast(Any, existing),
            cfg,
            status_callback=None,
            dry_run_enabled=False,
            engine_factory=cast(Any, DummyEngine),
        )

        self.assertIs(reused, existing)
        self.assertEqual(cast(DummyEngine, reused).dry_run, False)

    def test_toggle_user_connection(self):
        cfg = Config()
        engine = DummyEngine(cfg, None)

        toggle_user_connection(cast(Any, engine))
        self.assertEqual(engine.connect_calls, 1)
        self.assertTrue(engine.connected)

        toggle_user_connection(cast(Any, engine))
        self.assertEqual(engine.disconnect_calls, 1)
        self.assertFalse(engine.connected)


if __name__ == "__main__":
    unittest.main()
