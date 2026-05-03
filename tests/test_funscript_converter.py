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
    constrain_fourphase_coordinates,
    convert,
    decode_layout_controls,
    generate_freq_axes,
    mix_to_3d,
    mix_to_layout_controls,
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
        # Extreme values that would exceed unit radius after mixing
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
        np.testing.assert_allclose(np.sqrt(alpha**2 + beta**2 + gamma**2), [1.0])

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


class TestLayoutControlMix:
    def test_pair_position_rotates_axial_vs_rotation_mix(self):
        unified = {
            "main": np.array([100.0]),
            "sway": np.array([50.0]),
            "surge": np.array([100.0]),
            "roll": np.array([50.0]),
            "pitch": np.array([50.0]),
            "twist": np.array([50.0]),
        }

        top_axial, top_side, top_rotation = mix_to_layout_controls(unified, layout_model="Pair At Top")
        middle_axial, middle_side, middle_rotation = mix_to_layout_controls(unified, layout_model="Pair At Middle")
        bottom_axial, bottom_side, bottom_rotation = mix_to_layout_controls(unified, layout_model="Pair At Bottom / Rear")

        np.testing.assert_allclose(top_side, middle_side, atol=1e-10)
        np.testing.assert_allclose(bottom_side, middle_side, atol=1e-10)
        assert top_axial[0] > middle_axial[0] > bottom_axial[0]
        assert top_rotation[0] < middle_rotation[0] < bottom_rotation[0]


# ---------------------------------------------------------------------------
# Tetrahedral projection
# ---------------------------------------------------------------------------

class TestTetrahedralProject:
    def test_origin_all_equal(self):
        result = tetrahedral_project(
            np.array([0.0]), np.array([0.0]), np.array([0.0])
        )
        np.testing.assert_allclose(result[0], [1.0, 1.0, 1.0, 1.0], atol=1e-10)

    def test_output_range(self):
        rng = np.random.default_rng(42)
        alpha = rng.uniform(-1, 1, 100)
        beta = rng.uniform(-1, 1, 100)
        gamma = rng.uniform(-1, 1, 100)
        result = tetrahedral_project(alpha, beta, gamma)
        assert np.all(result >= -1e-10)
        assert np.all(result <= 1.0 + 1e-10)

    def test_projection_satisfies_fourphase_constraints(self):
        rng = np.random.default_rng(7)
        points = rng.normal(size=(1000, 3))
        points /= np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1.0)
        result = tetrahedral_project(points[:, 0], points[:, 1], points[:, 2])

        np.testing.assert_allclose(result.max(axis=1), 1.0, atol=1e-10)
        assert np.all(result[:, 0] <= result[:, 1] + result[:, 2] + result[:, 3] + 1e-10)
        assert np.all(result[:, 1] <= result[:, 0] + result[:, 2] + result[:, 3] + 1e-10)
        assert np.all(result[:, 2] <= result[:, 0] + result[:, 1] + result[:, 3] + 1e-10)
        assert np.all(result[:, 3] <= result[:, 0] + result[:, 1] + result[:, 2] + 1e-10)

    def test_vertex_matches_focstim_a_point(self):
        result = tetrahedral_project(
            np.array([1.0]), np.array([0.0]), np.array([0.0])
        )
        np.testing.assert_allclose(
            result[0],
            [1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            atol=1e-10,
        )

    def test_edge_direction_maps_to_ab_point(self):
        point = (TETRA_VERTICES[0] + TETRA_VERTICES[1])
        point = point / np.linalg.norm(point)
        result = tetrahedral_project(
            np.array([point[0]]), np.array([point[1]]), np.array([point[2]])
        )
        np.testing.assert_allclose(result[0], [1.0, 1.0, 0.0, 0.0], atol=1e-10)

    def test_face_direction_maps_to_abc_point(self):
        point = (TETRA_VERTICES[0] + TETRA_VERTICES[1] + TETRA_VERTICES[2])
        point = point / np.linalg.norm(point)
        result = tetrahedral_project(
            np.array([point[0]]), np.array([point[1]]), np.array([point[2]])
        )
        np.testing.assert_allclose(result[0], [1.0, 1.0, 1.0, 0.0], atol=1e-10)

    def test_half_radius_moves_halfway_to_edge_landmark(self):
        point = (TETRA_VERTICES[0] + TETRA_VERTICES[1])
        point = 0.5 * point / np.linalg.norm(point)
        result = tetrahedral_project(
            np.array([point[0]]), np.array([point[1]]), np.array([point[2]])
        )
        np.testing.assert_allclose(result[0], [1.0, 1.0, 0.5, 0.5], atol=1e-10)

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
        assert result[-1, 1] < result[0, 1]


class TestConstrainFourphaseCoordinates:
    def test_center_is_all_ones(self):
        projected = np.array([[0.0, 0.0, 0.0, 0.0]])
        result = constrain_fourphase_coordinates(projected)
        np.testing.assert_allclose(result[0], [1.0, 1.0, 1.0, 1.0], atol=1e-10)

    def test_vertex_is_repaired_to_focstim_a(self):
        projected = np.array([[1.0, 0.0, 0.0, 0.0]])
        result = constrain_fourphase_coordinates(projected)
        np.testing.assert_allclose(
            result[0],
            [1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            atol=1e-10,
        )


class TestLayoutDecoder:
    def test_pair_at_top_axial_anchors(self):
        result = decode_layout_controls(
            np.array([1.0, -0.5, -1.0]),
            np.zeros(3),
            np.zeros(3),
            "Pair At Top",
        )
        np.testing.assert_allclose(result[0], [1.0, 1.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(result[1], [1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0 / 3.0], atol=1e-10)
        np.testing.assert_allclose(result[2], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0], atol=1e-10)

    def test_pair_at_middle_axial_anchors(self):
        result = decode_layout_controls(
            np.array([1.0, -0.5, -1.0]),
            np.zeros(3),
            np.zeros(3),
            "Pair At Middle",
        )
        np.testing.assert_allclose(result[0], [1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-10)
        np.testing.assert_allclose(result[1], [0.0, 1.0, 1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(result[2], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0], atol=1e-10)

    def test_pair_at_bottom_rear_axial_anchors(self):
        result = decode_layout_controls(
            np.array([1.0, -0.5, -1.0]),
            np.zeros(3),
            np.zeros(3),
            "Pair At Bottom / Rear",
        )
        np.testing.assert_allclose(result[0], [1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-10)
        np.testing.assert_allclose(result[1], [1.0 / 3.0, 1.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-10)
        np.testing.assert_allclose(result[2], [0.0, 0.0, 1.0, 1.0], atol=1e-10)

    def test_legacy_layout_alias_maps_to_pair_at_middle(self):
        direct = decode_layout_controls(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            "Pair At Middle",
        )
        legacy = decode_layout_controls(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0]),
            "Triangle + Behind",
        )
        np.testing.assert_allclose(direct, legacy, atol=1e-10)


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
        assert "frequency" in result
        # At surge=50 (neutral), outputs should be at their center values.
        assert abs(result["pulse_frequency"][1] - 55.0) < 0.1
        assert abs(result["carrier_frequency"][1] - 50.0) < 0.1
        # Negative surge pushes above center; positive surge pulls below center.
        np.testing.assert_allclose(result["pulse_frequency"], [80.0, 55.0, 20.0])
        np.testing.assert_allclose(result["carrier_frequency"], [60.0, 50.0, 40.0])
        # frequency is identical to carrier_frequency
        np.testing.assert_array_equal(result["frequency"], result["carrier_frequency"])

    def test_enabled_custom_bounds(self):
        unified = {"surge": np.array([0.0, 50.0, 100.0])}
        result = generate_freq_axes(
            unified,
            FreqConfig(
                enabled=True,
                pulse_center=60.0,
                pulse_min=30.0,
                pulse_max=70.0,
                carrier_center=52.0,
                carrier_min=45.0,
                carrier_max=58.0,
            ),
        )

        np.testing.assert_allclose(result["pulse_frequency"], [70.0, 60.0, 30.0])
        np.testing.assert_allclose(result["carrier_frequency"], [58.0, 52.0, 45.0])


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
        assert "frequency" in result

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

    def test_layout_model_changes_output(self):
        axes = {"main": _make_actions([(0, 0), (1000, 100)])}
        top_pair = convert(axes, layout_model="Pair At Top")
        middle_pair = convert(axes, layout_model="Pair At Middle")
        assert [a.pos for a in top_pair["e2"]] != [a.pos for a in middle_pair["e2"]]

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
