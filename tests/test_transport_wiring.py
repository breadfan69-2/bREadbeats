import unittest

from transport_wiring import (
    begin_volume_ramp,
    play_button_text,
    send_zero_volume_immediate,
    set_transport_sending,
    start_stop_ui_state,
    trigger_network_test,
)


class DummyEngine:
    def __init__(self):
        self.sent = []
        self.sending_enabled = None
        self.connected = False
        self.test_called = 0

    def send_immediate(self, cmd):
        self.sent.append(cmd)

    def set_sending_enabled(self, enabled: bool):
        self.sending_enabled = enabled

    def send_test(self):
        self.test_called += 1


class TestTransportWiring(unittest.TestCase):
    def test_send_zero_volume_immediate(self):
        engine = DummyEngine()
        send_zero_volume_immediate(engine, 123)

        self.assertEqual(len(engine.sent), 1)
        cmd = engine.sent[0]
        self.assertEqual(cmd.alpha, 0.5)
        self.assertEqual(cmd.beta, 0.5)
        self.assertEqual(cmd.volume, 0.0)
        self.assertEqual(cmd.duration_ms, 123)

    def test_set_transport_sending(self):
        engine = DummyEngine()
        set_transport_sending(engine, True)
        self.assertTrue(engine.sending_enabled)

    def test_begin_volume_ramp(self):
        state = begin_volume_ramp(42.0)
        self.assertTrue(state['active'])
        self.assertEqual(state['start_time'], 42.0)
        self.assertEqual(state['from'], 0.0)
        self.assertEqual(state['to'], 1.0)

    def test_trigger_network_test_connected_restore_needed(self):
        engine = DummyEngine()
        engine.connected = True
        engine.sending_enabled = False

        did_trigger, should_restore = trigger_network_test(engine)

        self.assertTrue(did_trigger)
        self.assertTrue(should_restore)
        self.assertEqual(engine.test_called, 1)
        self.assertTrue(engine.sending_enabled)

    def test_trigger_network_test_connected_no_restore(self):
        engine = DummyEngine()
        engine.connected = True
        engine.sending_enabled = True

        did_trigger, should_restore = trigger_network_test(engine)

        self.assertTrue(did_trigger)
        self.assertFalse(should_restore)
        self.assertEqual(engine.test_called, 1)

    def test_trigger_network_test_disconnected(self):
        engine = DummyEngine()
        engine.connected = False

        did_trigger, should_restore = trigger_network_test(engine)

        self.assertFalse(did_trigger)
        self.assertFalse(should_restore)
        self.assertEqual(engine.test_called, 0)

    def test_start_stop_ui_state_running(self):
        state = start_stop_ui_state(True)
        self.assertEqual(state['start_text'], "■ Stop")
        self.assertTrue(state['play_enabled'])
        self.assertFalse(state['play_reset_checked'])
        self.assertIsNone(state['play_text'])
        self.assertIsNone(state['is_sending'])

    def test_start_stop_ui_state_stopped(self):
        state = start_stop_ui_state(False)
        self.assertEqual(state['start_text'], "▶ Start")
        self.assertFalse(state['play_enabled'])
        self.assertTrue(state['play_reset_checked'])
        self.assertEqual(state['play_text'], "▶ Play")
        self.assertFalse(state['is_sending'])

    def test_play_button_text(self):
        self.assertEqual(play_button_text(True), "⏸ Pause")
        self.assertEqual(play_button_text(False), "▶ Play")


if __name__ == "__main__":
    unittest.main()
