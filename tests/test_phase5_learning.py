"""
Phase 5 checkpoint tests — Cue-based Learning Adapter:
  Model loader with schema validation,
  Feature extraction (lookback aggregates),
  Linear inference (z-score normalized),
  Cadence rule derivation,
  Runtime blend with EMA smoothing.
"""

import json
import math
import os
import tempfile
import time
import unittest
from typing import Any
from pathlib import Path

import numpy as np

from audio_engine import BeatEvent
from beat_intelligence import BeatIntelligence, LearningOutputs
from config import Config


# ── Minimal valid rule_fit model for testing ────────────────────────

MINIMAL_RULE_FIT = {
    "status": "ok",
    "feature_columns": ["rms", "spectral_flux", "sub_bass_energy",
                         "rms_mean_10s", "flux_mean_10s"],
    "target_columns": ["speed_mult"],
    "normalization": {
        "mean": {"rms": -34.0, "spectral_flux": 0.3, "sub_bass_energy": 0.4,
                 "rms_mean_10s": -34.0, "flux_mean_10s": 0.3},
        "std": {"rms": 8.0, "spectral_flux": 0.2, "sub_bass_energy": 0.25,
                "rms_mean_10s": 8.0, "flux_mean_10s": 0.2},
    },
    "models": {
        "speed_mult": {
            "intercept": 0.4,
            "coefficients": {"rms": 0.08, "spectral_flux": 0.12,
                             "sub_bass_energy": 0.06,
                             "rms_mean_10s": 0.05, "flux_mean_10s": 0.04},
        },
    },
    "cadence_rule": {
        "quiet_threshold": 0.15,
        "mid_threshold": 0.45,
        "mapping": {"quiet": 4, "mid": 2, "loud": 1},
    },
}


class Phase5Mixin:
    """Shared helpers for Phase 5 tests."""

    def _event(self, **overrides) -> BeatEvent:
        payload: dict[str, Any] = dict(
            timestamp=time.time(),
            intensity=0.5,
            frequency=60.0,
            is_beat=False,
            spectral_flux=0.1,
            peak_energy=0.3,
            is_downbeat=False,
            bpm=120.0,
            tempo_locked=True,
            metronome_bpm=120.0,
            is_syncopated=False,
            monotonic_timestamp=time.perf_counter(),
            raw_rms=0.08,
            beat_band="sub_bass",
            fired_bands=["sub_bass"],
            acf_confidence=0.5,
        )
        payload.update(overrides)
        return BeatEvent(**payload)

    def _bi(self, **cfg_overrides) -> BeatIntelligence:
        cfg = Config()
        cfg.beat.tempo_lock_required = False
        for key, val in cfg_overrides.items():
            if hasattr(cfg.stroke, key):
                setattr(cfg.stroke, key, val)
            elif hasattr(cfg.beat, key):
                setattr(cfg.beat, key, val)
        return BeatIntelligence(cfg)

    def _write_model(self, model_dict: dict) -> str:
        """Write a model dict to a temp JSON file, return path."""
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(model_dict, f)
        return path


class TestModelLoader(Phase5Mixin, unittest.TestCase):
    """#12: _try_load_learning_model with schema validation."""

    def test_load_valid_model(self):
        bi = self._bi()
        path = self._write_model(MINIMAL_RULE_FIT)
        try:
            bi._try_load_learning_model(path)
            self.assertTrue(bi._learning_model_loaded)
            self.assertEqual(bi._learning_feature_columns,
                             ["rms", "spectral_flux", "sub_bass_energy",
                              "rms_mean_10s", "flux_mean_10s"])
            self.assertIn("quiet_threshold", bi._learning_cadence_rule)
        finally:
            os.unlink(path)

    def test_load_missing_status(self):
        bad_model = {**MINIMAL_RULE_FIT, "status": "error"}
        bi = self._bi()
        path = self._write_model(bad_model)
        try:
            bi._try_load_learning_model(path)
            self.assertFalse(bi._learning_model_loaded)
        finally:
            os.unlink(path)

    def test_load_missing_features(self):
        bad_model = {**MINIMAL_RULE_FIT, "feature_columns": []}
        bi = self._bi()
        path = self._write_model(bad_model)
        try:
            bi._try_load_learning_model(path)
            self.assertFalse(bi._learning_model_loaded)
        finally:
            os.unlink(path)

    def test_load_nonexistent_path(self):
        bi = self._bi()
        bi._try_load_learning_model("/nonexistent/path.json")
        self.assertFalse(bi._learning_model_loaded)

    def test_load_empty_path(self):
        bi = self._bi()
        bi._try_load_learning_model("")
        self.assertFalse(bi._learning_model_loaded)

    def test_load_corrupt_json(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write("{corrupt json")
        bi = self._bi()
        try:
            bi._try_load_learning_model(path)
            self.assertFalse(bi._learning_model_loaded)
        finally:
            os.unlink(path)


class TestFeatureExtraction(Phase5Mixin, unittest.TestCase):
    """Feature extraction with lookback aggregates."""

    def test_feature_keys(self):
        bi = self._bi()
        ev = self._event(raw_rms=0.05, spectral_flux=0.1, peak_energy=0.3)
        features = bi._build_runtime_feature_values(ev)
        expected_keys = {
            "rms", "spectral_flux",
            "sub_bass_energy", "low_mid_energy", "mid_energy", "high_energy",
            "low_high_ratio", "spectral_centroid_hz", "spectral_flatness",
            "rms_mean_10s", "rms_std_10s", "flux_mean_10s",
            "bass_mean_10s", "energy_trend_10s",
        }
        self.assertEqual(set(features.keys()), expected_keys)

    def test_rms_is_dbfs(self):
        bi = self._bi()
        ev = self._event(raw_rms=0.05, peak_energy=0.9)
        features = bi._build_runtime_feature_values(ev)
        # rms should be in dBFS now (negative value)
        self.assertLess(features["rms"], 0.0)

    def test_all_features_finite(self):
        bi = self._bi()
        ev = self._event(raw_rms=0.0, peak_energy=0.0)
        features = bi._build_runtime_feature_values(ev)
        for key, val in features.items():
            self.assertTrue(math.isfinite(val), f"{key} is not finite: {val}")

    def test_band_energies_propagate(self):
        bi = self._bi()
        bi.energies.sub_bass = 0.5
        bi.energies.high = 0.1
        ev = self._event()
        features = bi._build_runtime_feature_values(ev)
        self.assertAlmostEqual(features["sub_bass_energy"], 0.5)
        self.assertAlmostEqual(features["high_energy"], 0.1)


class TestInference(Phase5Mixin, unittest.TestCase):
    """_predict_learning_targets with speed_mult model."""

    def _loaded_bi(self) -> tuple:
        bi = self._bi()
        path = self._write_model(MINIMAL_RULE_FIT)
        bi._try_load_learning_model(path)
        return bi, path

    def test_predict_returns_speed_mult(self):
        bi, path = self._loaded_bi()
        try:
            features = {"rms": -30.0, "spectral_flux": 0.4, "sub_bass_energy": 0.5,
                        "rms_mean_10s": -32.0, "flux_mean_10s": 0.35}
            preds = bi._predict_learning_targets(features)
            self.assertIn("speed_mult", preds)
        finally:
            os.unlink(path)

    def test_predict_returns_floats(self):
        bi, path = self._loaded_bi()
        try:
            features = {"rms": -40.0, "spectral_flux": 0.1, "sub_bass_energy": 0.2,
                        "rms_mean_10s": -40.0, "flux_mean_10s": 0.1}
            preds = bi._predict_learning_targets(features)
            for v in preds.values():
                self.assertIsInstance(v, float)
        finally:
            os.unlink(path)

    def test_predict_without_model_returns_empty(self):
        bi = self._bi()
        preds = bi._predict_learning_targets({"rms": -30.0})
        self.assertEqual(preds, {})

    def test_predict_uses_zscore(self):
        """Verify z-score normalization: feature at mean → zero contribution."""
        bi, path = self._loaded_bi()
        try:
            # Features exactly at mean → all z-scores are 0 → prediction = intercept only
            mean = MINIMAL_RULE_FIT["normalization"]["mean"]
            preds = bi._predict_learning_targets(dict(mean))
            self.assertAlmostEqual(preds["speed_mult"], 0.4, places=4)
        finally:
            os.unlink(path)


class TestCadenceRule(Phase5Mixin, unittest.TestCase):
    """Cadence rule derivation from predicted speed_mult."""

    def _loaded_bi(self) -> tuple:
        bi = self._bi()
        path = self._write_model(MINIMAL_RULE_FIT)
        bi._try_load_learning_model(path)
        return bi, path

    def test_quiet_cadence(self):
        """Very low speed → quiet → 4 beats between strokes."""
        bi, path = self._loaded_bi()
        try:
            beats = bi._derive_cadence_beats(0.05)  # below quiet_threshold 0.15
            self.assertEqual(beats, 4)
        finally:
            os.unlink(path)

    def test_loud_cadence(self):
        """High speed → loud → 1 beat between strokes."""
        bi, path = self._loaded_bi()
        try:
            beats = bi._derive_cadence_beats(0.80)  # above mid_threshold 0.45
            self.assertEqual(beats, 1)
        finally:
            os.unlink(path)

    def test_mid_cadence(self):
        """Medium speed → mid → 2 beats between strokes."""
        bi, path = self._loaded_bi()
        try:
            beats = bi._derive_cadence_beats(0.30)  # between 0.15 and 0.45
            self.assertEqual(beats, 2)
        finally:
            os.unlink(path)

    def test_no_cadence_rule_returns_1(self):
        bi = self._bi()
        beats = bi._derive_cadence_beats(0.50)
        self.assertEqual(beats, 1)


class TestLearningAdapter(Phase5Mixin, unittest.TestCase):
    """#18: _update_learning_adapter runtime blend."""

    def _loaded_bi(self) -> tuple:
        bi = self._bi()
        path = self._write_model(MINIMAL_RULE_FIT)
        bi._learning_enabled = True
        bi._learning_use_fitted_rules = True
        bi._learning_strength = 0.55
        bi._learning_min_confidence = 0.10
        bi._learning_no_motion_bias = 1.0
        bi._try_load_learning_model(path)
        return bi, path

    def test_adapter_produces_active_output_on_beat(self):
        bi, path = self._loaded_bi()
        try:
            ev = self._event(is_beat=True, acf_confidence=0.5)
            outputs = bi._update_learning_adapter(ev)
            self.assertTrue(outputs.active)
        finally:
            os.unlink(path)

    def test_adapter_inactive_without_model(self):
        bi = self._bi()
        bi._learning_enabled = True
        ev = self._event(is_beat=True, acf_confidence=0.5)
        outputs = bi._update_learning_adapter(ev)
        self.assertFalse(outputs.active)

    def test_adapter_skips_non_beat_events(self):
        bi, path = self._loaded_bi()
        try:
            ev = self._event(is_beat=False, is_downbeat=False, is_syncopated=False)
            outputs = bi._update_learning_adapter(ev)
            self.assertFalse(outputs.active)  # no beats → returns default
        finally:
            os.unlink(path)

    def test_adapter_skips_low_confidence(self):
        bi, path = self._loaded_bi()
        try:
            bi._learning_min_confidence = 0.9
            ev = self._event(is_beat=True, acf_confidence=0.3)
            outputs = bi._update_learning_adapter(ev)
            self.assertFalse(outputs.active)
        finally:
            os.unlink(path)

    def test_outputs_bounded(self):
        bi, path = self._loaded_bi()
        try:
            ev = self._event(is_beat=True, acf_confidence=0.5, raw_rms=0.15)
            outputs = bi._update_learning_adapter(ev)
            self.assertGreaterEqual(outputs.speed_mult, 0.0)
            self.assertLessEqual(outputs.speed_mult, 1.0)
            self.assertIn(outputs.cadence_hint, (1, 2, 4))
        finally:
            os.unlink(path)

    def test_strength_zero_means_neutral(self):
        """With strength=0, speed_mult should be ~0.5 (neutral, no influence)."""
        bi, path = self._loaded_bi()
        try:
            bi._learning_strength = 0.0
            ev = self._event(is_beat=True, acf_confidence=0.5)
            outputs = bi._update_learning_adapter(ev)
            self.assertAlmostEqual(outputs.speed_mult, 0.5, delta=0.05)
        finally:
            os.unlink(path)


class TestLearningInBuildDecision(Phase5Mixin, unittest.TestCase):
    """Verify learning outputs wire through build_decision."""

    def test_decision_has_learning_field(self):
        bi = self._bi()
        ev = self._event()
        decision = bi.build_decision(ev, dt=1 / 60)
        self.assertIsInstance(decision.learning, LearningOutputs)

    def test_learning_inactive_by_default(self):
        bi = self._bi()
        ev = self._event()
        decision = bi.build_decision(ev, dt=1 / 60)
        self.assertFalse(decision.learning.active)

    def test_learning_active_with_model(self):
        bi = self._bi()
        path = self._write_model(MINIMAL_RULE_FIT)
        try:
            bi._learning_enabled = True
            bi._learning_use_fitted_rules = True
            bi._learning_strength = 0.55
            bi._learning_min_confidence = 0.10
            bi._try_load_learning_model(path)

            # Need a beat event for adapter to fire
            ev = self._event(is_downbeat=True, acf_confidence=0.5)
            decision = bi.build_decision(ev, dt=1 / 60)
            self.assertTrue(decision.learning.active)
        finally:
            os.unlink(path)


class TestConfigureLearning(Phase5Mixin, unittest.TestCase):
    """Test configure_learning API on BeatIntelligence."""

    def test_configure_learning_loads_model(self):
        bi = self._bi()
        path = self._write_model(MINIMAL_RULE_FIT)
        try:
            bi.configure_learning(
                enabled=True,
                use_fitted_rules=True,
                strength=0.5,
                min_confidence=0.1,
                no_motion_bias=1.0,
                rule_fit_path=path,
            )
            self.assertTrue(bi._learning_model_loaded)
            self.assertTrue(bi._learning_enabled)
        finally:
            os.unlink(path)

    def test_configure_learning_disabled_unloads_model(self):
        bi = self._bi()
        path = self._write_model(MINIMAL_RULE_FIT)
        try:
            bi.configure_learning(
                enabled=True, use_fitted_rules=True,
                strength=0.5, min_confidence=0.1, no_motion_bias=1.0,
                rule_fit_path=path,
            )
            self.assertTrue(bi._learning_model_loaded)

            bi.configure_learning(
                enabled=False, use_fitted_rules=False,
                strength=0.5, min_confidence=0.1, no_motion_bias=1.0,
                rule_fit_path=path,
            )
            self.assertFalse(bi._learning_model_loaded)
        finally:
            os.unlink(path)

    def test_configure_learning_clamps_strength(self):
        bi = self._bi()
        bi.configure_learning(
            enabled=False, use_fitted_rules=False,
            strength=5.0, min_confidence=-1.0, no_motion_bias=10.0,
        )
        self.assertLessEqual(bi._learning_strength, 1.0)
        self.assertGreaterEqual(bi._learning_min_confidence, 0.0)
        self.assertLessEqual(bi._learning_no_motion_bias, 3.0)


if __name__ == "__main__":
    unittest.main()
