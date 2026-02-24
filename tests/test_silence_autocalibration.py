import time
import unittest

from audio_engine import BeatEvent
from beat_intelligence import BeatIntelligence
from config import Config


class TestSilenceAutoCalibration(unittest.TestCase):
    def _event(self, *, mono: float, raw_rms_db: float) -> BeatEvent:
        return BeatEvent(
            timestamp=time.time(),
            intensity=0.0,
            frequency=0.0,
            is_beat=False,
            spectral_flux=0.0,
            peak_energy=0.0,
            is_downbeat=False,
            bpm=120.0,
            tempo_locked=False,
            metronome_bpm=120.0,
            is_syncopated=False,
            monotonic_timestamp=mono,
            raw_rms=0.0,
            raw_rms_db=raw_rms_db,
        )

    def test_startup_autocal_sets_runtime_thresholds_for_default_profile(self):
        cfg = Config()
        intelligence = BeatIntelligence(cfg)

        t0 = time.perf_counter()
        for i in range(180):
            now = t0 + (i * (1.0 / 60.0))
            event = self._event(mono=now, raw_rms_db=-72.0)
            intelligence.build_decision(event=event, dt=1.0 / 60.0)

        self.assertIsNotNone(intelligence._silence_runtime_open_threshold_db)
        self.assertIsNotNone(intelligence._silence_runtime_close_threshold_db)
        assert intelligence._silence_runtime_open_threshold_db is not None
        assert intelligence._silence_runtime_close_threshold_db is not None
        self.assertGreater(intelligence._silence_runtime_open_threshold_db, -72.0)
        self.assertGreater(
            intelligence._silence_runtime_close_threshold_db,
            intelligence._silence_runtime_open_threshold_db,
        )


if __name__ == "__main__":
    unittest.main()
