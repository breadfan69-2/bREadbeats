import unittest

from audio_engine import AudioEngine


class TestTempoLockHysteresis(unittest.TestCase):
    def _engine(self) -> AudioEngine:
        engine = AudioEngine.__new__(AudioEngine)
        engine._tempo_lock_hysteresis_locked = False
        engine._tempo_lock_enter_conf_base = 0.20
        engine._tempo_lock_enter_conf_strict = 0.35
        engine._tempo_lock_exit_conf_base = 0.15
        engine._tempo_lock_exit_hold_s = 0.90
        engine._tempo_lock_drop_started_at = 0.0
        return engine

    def test_enters_lock_on_confidence_with_downbeat_match(self):
        engine = self._engine()
        locked = engine._compute_tempo_lock_state(acf_confidence=0.22, downbeat_matches=1, now=10.0)
        self.assertTrue(locked)

    def test_does_not_enter_lock_on_low_confidence_without_match(self):
        engine = self._engine()
        locked = engine._compute_tempo_lock_state(acf_confidence=0.22, downbeat_matches=0, now=10.0)
        self.assertFalse(locked)

    def test_stays_locked_during_brief_confidence_dip(self):
        engine = self._engine()
        engine._compute_tempo_lock_state(acf_confidence=0.30, downbeat_matches=1, now=10.0)
        locked = engine._compute_tempo_lock_state(acf_confidence=0.10, downbeat_matches=0, now=10.4)
        self.assertTrue(locked)

    def test_unlocks_after_sustained_low_confidence(self):
        engine = self._engine()
        engine._compute_tempo_lock_state(acf_confidence=0.30, downbeat_matches=1, now=10.0)
        engine._compute_tempo_lock_state(acf_confidence=0.10, downbeat_matches=0, now=10.1)
        locked = engine._compute_tempo_lock_state(acf_confidence=0.10, downbeat_matches=0, now=11.2)
        self.assertFalse(locked)


if __name__ == "__main__":
    unittest.main()
