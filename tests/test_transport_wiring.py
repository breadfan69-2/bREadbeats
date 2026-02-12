import unittest

from transport_wiring import begin_volume_ramp, send_zero_volume_immediate, set_transport_sending


class DummyEngine:
    def __init__(self):
        self.sent = []
        self.sending_enabled = None

    def send_immediate(self, cmd):
        self.sent.append(cmd)

    def set_sending_enabled(self, enabled: bool):
        self.sending_enabled = enabled


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


if __name__ == "__main__":
    unittest.main()
