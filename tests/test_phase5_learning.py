"""
Phase 5 checkpoint tests — Learning Adapter Boundary (#10-12, #18):
  Model loader with schema validation,
  Feature extraction,
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
    "feature_columns": ["rms", "log_energy", "sub_bass_energy"],
    "target_columns": ["arc_size", "creep_mix", "gate_strictness"],
    "normalization": {
        "mean": {"rms": 0.02, "log_energy": -4.0, "sub_bass_energy": 2.0},
        "std": {"rms": 0.025, "log_energy": 3.7, "sub_bass_energy": 3.2},
    },
    "models": {
        "arc_size": {
            "intercept": 0.5,
            "coefficients": {"rms": 0.1, "log_energy": -0.05, "sub_bass_energy": 0.08},
        },
        "creep_mix": {
            "intercept": 0.6,
            "coefficients": {"rms": -0.15, "log_energy": 0.08, "sub_bass_energy": -0.1},
        },
        "gate_strictness": {
            "intercept": 0.5,
            "coefficients": {"rms": 0.05, "log_energy": -0.02, "sub_bass_energy": 0.03},
        },
    },
    "cadence_rule": {
        "quiet_threshold": -0.4,
        "mid_threshold": 0.08,
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
            self.assertEqual(bi._learning_feature_columns, ["rms", "log_energy", "sub_bass_energy"])
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
    """#10: _build_runtime_feature_values."""

    def test_feature_keys(self):
        bi = self._bi()
        ev = self._event(raw_rms=0.05, spectral_flux=0.1, peak_energy=0.3)
        features = bi._build_runtime_feature_values(ev)
        expected_keys = {
            "rms", "log_energy", "spectral_flux", "flux_delta",
            "sub_bass_energy", "low_mid_energy", "mid_energy", "high_energy",
            "low_high_ratio", "spectral_centroid_hz", "spectral_bandwidth_hz",
            "spectral_rolloff_hz", "spectral_flatness",
        }
        self.assertEqual(set(features.keys()), expected_keys)

    def test_rms_from_raw_rms(self):
        bi = self._bi()
        ev = self._event(raw_rms=0.05, peak_energy=0.9)
        features = bi._build_runtime_feature_values(ev)
        self.assertAlmostEqual(features["rms"], 20.0 * math.log10(0.05), places=6)

    def test_log_energy_finite(self):
        bi = self._bi()
        ev = self._event(raw_rms=0.0, peak_energy=0.0)
        features = bi._build_runtime_feature_values(ev)
        self.assertTrue(math.isfinite(features["log_energy"]))

    def test_band_energies_propagate(self):
        bi = self._bi()
        bi.energies.sub_bass = 0.5
        bi.energies.high = 0.1
        ev = self._event()
        features = bi._build_runtime_feature_values(ev)
        self.assertAlmostEqual(features["sub_bass_energy"], 0.5)
        self.assertAlmostEqual(features["high_energy"], 0.1)


class TestInference(Phase5Mixin, unittest.TestCase):
    """#11: _predict_learning_targets."""

    def _loaded_bi(self) -> tuple:
        bi = self._bi()
        path = self._write_model(MINIMAL_RULE_FIT)
        bi._try_load_learning_model(path)
        return bi, path

    def test_predict_returns_targets(self):
        bi, path = self._loaded_bi()
        try:
            features = {"rms": 0.03, "log_energy": -3.0, "sub_bass_energy": 2.5}
            preds = bi._predict_learning_targets(features)
            self.assertIn("arc_size", preds)
            self.assertIn("creep_mix", preds)
            self.assertIn("gate_strictness", preds)
        finally:
            os.unlink(path)

    def test_predict_returns_floats(self):
        bi, path = self._loaded_bi()
        try:
            features = {"rms": 0.01, "log_energy": -5.0, "sub_bass_energy": 1.0}
            preds = bi._predict_learning_targets(features)
            for v in preds.values():
                self.assertIsInstance(v, float)
        finally:
            os.unlink(path)

    def test_predict_without_model_returns_empty(self):
        bi = self._bi()
        preds = bi._predict_learning_targets({"rms": 0.03})
        self.assertEqual(preds, {})

    def test_predict_uses_zscore(self):
        """Verify z-score normalization: feature at mean → zero contribution."""
        bi, path = self._loaded_bi()
        try:
            # Features exactly at mean → all z-scores are 0 → prediction = intercept only
            mean = MINIMAL_RULE_FIT["normalization"]["mean"]
            preds = bi._predict_learning_targets(dict(mean))
            self.assertAlmostEqual(preds["arc_size"], 0.5, places=4)
            self.assertAlmostEqual(preds["creep_mix"], 0.6, places=4)
        finally:
            os.unlink(path)


class TestCadenceRule(Phase5Mixin, unittest.TestCase):
    """Cadence rule derivation from z-scored RMS."""

    def _loaded_bi(self) -> tuple:
        bi = self._bi()
        path = self._write_model(MINIMAL_RULE_FIT)
        bi._try_load_learning_model(path)
        return bi, path

    def test_quiet_cadence(self):
        """Very low RMS → quiet → 4 beats between strokes."""
        bi, path = self._loaded_bi()
        try:
            # RMS well below mean (negative z-score beyond quiet_threshold)
            beats = bi._derive_cadence_beats({"rms": 0.0})
            self.assertEqual(beats, 4)
        finally:
            os.unlink(path)

    def test_loud_cadence(self):
        """High RMS → loud → 1 beat between strokes."""
        bi, path = self._loaded_bi()
        try:
            beats = bi._derive_cadence_beats({"rms": 0.10})
            self.assertEqual(beats, 1)
        finally:
            os.unlink(path)

    def test_no_cadence_rule_returns_1(self):
        bi = self._bi()
        beats = bi._derive_cadence_beats({"rms": 0.05})
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
            self.assertGreaterEqual(outputs.radius_mult, 0.3)
            self.assertLessEqual(outputs.radius_mult, 2.5)
            self.assertGreaterEqual(outputs.sync_size_mult, 0.5)
            self.assertLessEqual(outputs.sync_size_mult, 2.0)
            self.assertGreaterEqual(outputs.gate_bias, -1.0)
            self.assertLessEqual(outputs.gate_bias, 1.0)
            self.assertGreaterEqual(outputs.lead_ms, 0.0)
            self.assertLessEqual(outputs.lead_ms, 100.0)
        finally:
            os.unlink(path)

    def test_strength_zero_means_neutral(self):
        """With strength=0, all multipliers should be ~1.0 (no influence)."""
        bi, path = self._loaded_bi()
        try:
            bi._learning_strength = 0.0
            ev = self._event(is_beat=True, acf_confidence=0.5)
            outputs = bi._update_learning_adapter(ev)
            self.assertAlmostEqual(outputs.radius_mult, 1.0, delta=0.05)
            self.assertAlmostEqual(outputs.sync_size_mult, 1.0, delta=0.05)
            self.assertAlmostEqual(outputs.gate_bias, 0.0, delta=0.05)
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
