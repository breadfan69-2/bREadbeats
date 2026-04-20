"""Tests for funscript_converter — 6-axis to 4-phase math."""
from __future__ import annotations

import numpy as np
import pytest

from funscript_converter import (
    PRESETS,
    TETRA_VERTICES,
    FreqConfig,
    MixWeights,
    apply_permutation,
    convert,
    generate_freq_axes,
    mix_to_3d,
    tetrahedral_project,
    unify_timeline,
)
from pmv_funscript_io import FunscriptAction


# ---------------------------------------------------------------------------
# Tetrahedral vertex sanity
# ---------------------------------------------------------------------------

class TestTetraVertices:
    def test_centroid_is_origin(self):
        centroid = TETRA_VERTICES.sum(axis=0)
        np.testing.assert_allclose(centroid, [0, 0, 0], atol=1e-10)

    def test_four_vertices(self):
        assert TETRA_VERTICES.shape == (4, 3)


# ---------------------------------------------------------------------------
# Timeline unification
# ---------------------------------------------------------------------------

def _make_actions(pairs: list[tuple[int, int]]) -> list[FunscriptAction]:
    return [FunscriptAction(at=t, pos=p) for t, p in pairs]


class TestUnifyTimeline:
    def test_stroke_is_authoritative(self):
        axes = {
            "main": _make_actions([(0, 0), (1000, 100)]),
            "surge": _make_actions([(0, 0), (500, 100), (1000, 0)]),
        }
        ts, unified = unify_timeline(axes)
        assert len(ts) == 2  # stroke has 2 keyframes
        assert "main" in unified
        assert "surge" in unified

    def test_missing_axes_neutral(self):
        axes = {"main": _make_actions([(0, 50), (1000, 50)])}
        ts, unified = unify_timeline(axes)
        np.testing.assert_allclose(unified["sway"], [50.0, 50.0])
        np.testing.assert_allclose(unified["twist"], [50.0, 50.0])

    def test_empty_input(self):
        ts, unified = unify_timeline({})
        assert len(ts) == 0

    def test_no_stroke_uses_most_keyframes(self):
        axes = {
            "surge": _make_actions([(0, 0), (500, 50), (1000, 100)]),
            "sway": _make_actions([(0, 0), (1000, 100)]),
        }
        ts, unified = unify_timeline(axes)
        assert len(ts) == 3  # surge has more keyframes

    def test_interpolation(self):
        axes = {
            "main": _make_actions([(0, 0), (1000, 100)]),
            "surge": _make_actions([(0, 0), (1000, 100)]),
        }
        ts, unified = unify_timeline(axes)
        np.testing.assert_allclose(unified["surge"], [0.0, 100.0])


# ---------------------------------------------------------------------------
# 6→3 mixing
# ---------------------------------------------------------------------------

class TestMixTo3d:
    def test_stroke_only(self):
        unified = {
            "main": np.array([0.0, 50.0, 100.0]),
            "sway": np.array([50.0, 50.0, 50.0]),
            "surge": np.array([50.0, 50.0, 50.0]),
            "roll": np.array([50.0, 50.0, 50.0]),
            "pitch": np.array([50.0, 50.0, 50.0]),
            "twist": np.array([50.0, 50.0, 50.0]),
        }
        alpha, beta, gamma = mix_to_3d(unified)
        np.testing.assert_allclose(alpha, [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(beta, [0.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(gamma, [0.0, 0.0, 0.0], atol=1e-10)

    def test_clamping(self):
        # Extreme values that would exceed ±1 after mixing
        w = MixWeights(w_primary=1.0, w_secondary=1.0, w_twist=1.0, twist_phase=0.0)
        unified = {
            "main": np.array([100.0]),
            "sway": np.array([100.0]),
            "surge": np.array([100.0]),
            "roll": np.array([100.0]),
            "pitch": np.array([100.0]),
            "twist": np.array([100.0]),
        }
        alpha, beta, gamma = mix_to_3d(unified, w)
        assert np.all(alpha <= 1.0)
        assert np.all(beta <= 1.0)
        assert np.all(gamma <= 1.0)

    def test_twist_phase_zero_only_beta(self):
        w = MixWeights(w_primary=0.0, w_secondary=0.0, w_twist=1.0, twist_phase=0.0)
        unified = {
            "main": np.array([50.0]),
            "sway": np.array([50.0]),
            "surge": np.array([50.0]),
            "roll": np.array([50.0]),
            "pitch": np.array([50.0]),
            "twist": np.array([100.0]),  # full positive twist
        }
        alpha, beta, gamma = mix_to_3d(unified, w)
        assert abs(beta[0]) > 0.1  # twist affects beta
        assert abs(gamma[0]) < 1e-10  # no gamma at phase=0

    def test_twist_phase_90_only_gamma(self):
        import math
        w = MixWeights(w_primary=0.0, w_secondary=0.0, w_twist=1.0, twist_phase=math.pi / 2)
        unified = {
            "main": np.array([50.0]),
            "sway": np.array([50.0]),
            "surge": np.array([50.0]),
            "roll": np.array([50.0]),
            "pitch": np.array([50.0]),
            "twist": np.array([100.0]),
        }
        alpha, beta, gamma = mix_to_3d(unified, w)
        assert abs(beta[0]) < 1e-10  # no beta at phase=π/2
        assert abs(gamma[0]) > 0.1  # twist affects gamma


# ---------------------------------------------------------------------------
# Tetrahedral projection
# ---------------------------------------------------------------------------

class TestTetrahedralProject:
    def test_origin_all_equal(self):
        result = tetrahedral_project(
            np.array([0.0]), np.array([0.0]), np.array([0.0])
        )
        # At origin all dot products are 0 → all channels equal (0.0)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_output_range(self):
        rng = np.random.default_rng(42)
        alpha = rng.uniform(-1, 1, 100)
        beta = rng.uniform(-1, 1, 100)
        gamma = rng.uniform(-1, 1, 100)
        result = tetrahedral_project(alpha, beta, gamma)
        assert np.all(result >= -1e-10)
        assert np.all(result <= 1.0 + 1e-10)

    def test_at_least_one_zero_one_max(self):
        rng = np.random.default_rng(42)
        alpha = rng.uniform(-1, 1, 50)
        beta = rng.uniform(-1, 1, 50)
        gamma = rng.uniform(-1, 1, 50)
        result = tetrahedral_project(alpha, beta, gamma)
        for row in result:
            assert min(row) < 1e-10, "Minimum should be ~0"
            assert max(row) > 1.0 - 1e-10, "Maximum should be ~1"

    def test_vertex_dominance(self):
        # At vertex v0 direction, channel 0 should dominate
        result = tetrahedral_project(
            np.array([1.0]), np.array([0.0]), np.array([0.0])
        )
        assert result[0, 0] == pytest.approx(1.0, abs=1e-10)
        assert all(result[0, i] < 0.5 for i in [1, 2, 3])

    def test_stroke_sweep_opposite_channels(self):
        # Sweeping stroke from -1 to +1 — two channels should move oppositely
        alpha = np.linspace(-1, 1, 11)
        beta = np.zeros(11)
        gamma = np.zeros(11)
        result = tetrahedral_project(alpha, beta, gamma)
        # Channel 0 (aligned with alpha) should increase
        assert result[-1, 0] > result[0, 0]


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------

class TestPermutation:
    def test_identity(self):
        data = np.array([[0.1, 0.2, 0.3, 0.4]])
        result = apply_permutation(data, (0, 1, 2, 3))
        np.testing.assert_array_equal(result, data)

    def test_reorder(self):
        data = np.array([[0.1, 0.2, 0.3, 0.4]])
        result = apply_permutation(data, (3, 2, 1, 0))
        np.testing.assert_array_equal(result[0], [0.4, 0.3, 0.2, 0.1])

    def test_different_placements_differ(self):
        data = np.array([[0.1, 0.5, 0.9, 0.3]])
        r1 = apply_permutation(data, (0, 1, 2, 3))
        r2 = apply_permutation(data, (1, 2, 3, 0))
        assert not np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Frequency generation
# ---------------------------------------------------------------------------

class TestFreqAxes:
    def test_disabled(self):
        unified = {"surge": np.array([0.0, 50.0, 100.0])}
        result = generate_freq_axes(unified, FreqConfig(enabled=False))
        assert result == {}

    def test_enabled(self):
        unified = {"surge": np.array([0.0, 50.0, 100.0])}
        result = generate_freq_axes(unified, FreqConfig(enabled=True))
        assert "pulse_frequency" in result
        assert "carrier_frequency" in result
        # At surge=50 (neutral), output should be near 0
        assert result["pulse_frequency"][1] < 1.0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestConvert:
    def test_stroke_only(self):
        axes = {"main": _make_actions([(0, 0), (500, 50), (1000, 100)])}
        result = convert(axes)
        assert "e1" in result
        assert "e2" in result
        assert "e3" in result
        assert "e4" in result
        assert len(result["e1"]) == 3

    def test_all_outputs_in_range(self):
        axes = {"main": _make_actions([(0, 0), (500, 50), (1000, 100)])}
        result = convert(axes)
        for name in ["e1", "e2", "e3", "e4"]:
            for a in result[name]:
                assert 0 <= a.pos <= 100

    def test_with_freq(self):
        axes = {
            "main": _make_actions([(0, 0), (1000, 100)]),
            "surge": _make_actions([(0, 0), (1000, 100)]),
        }
        result = convert(axes, freq_config=FreqConfig(enabled=True))
        assert "pulse_frequency" in result
        assert "carrier_frequency" in result

    def test_empty_input(self):
        result = convert({})
        assert result == {}

    def test_permutation_changes_output(self):
        axes = {"main": _make_actions([(0, 0), (1000, 100)])}
        r1 = convert(axes, placement=(0, 1, 2, 3))
        r2 = convert(axes, placement=(1, 0, 3, 2))
        # E1 from r1 should differ from E1 of r2 when placement differs
        p1 = [a.pos for a in r1["e1"]]
        p2 = [a.pos for a in r2["e1"]]
        assert p1 != p2

    def test_round_trip(self, tmp_path):
        """Generate → export → re-read should produce valid data."""
        from pmv_funscript_io import read_funscript, write_funscript, FunscriptMetadata

        axes = {"main": _make_actions([(0, 0), (500, 50), (1000, 100)])}
        result = convert(axes)

        for name, actions in result.items():
            path = tmp_path / f"test.{name}.funscript"
            write_funscript(path, actions)
            loaded, _ = read_funscript(path)
            assert len(loaded) == len(actions)
            for orig, reloaded in zip(actions, loaded):
                assert orig.at == reloaded.at
                assert orig.pos == reloaded.pos
