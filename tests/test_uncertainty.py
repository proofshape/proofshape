"""Unit tests for R-10's per-vertex uncertainty. All synthetic data -- no GPU, no golden capture."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recon import mesh_cleanup, provenance, uncertainty

# --- compute_vertex_sigma ----------------------------------------------------------------------


def test_compute_vertex_sigma_matches_hand_computed_std():
    observations = {
        "vertex_index": np.array([0, 0, 0]),
        "frame_index": np.array([0, 1, 2]),
        "z_cam_mm": np.array([50.0, 50.0, 50.0]),
        "measured_depth_mm": np.array([51.0, 52.0, 53.0]),  # residuals [1, 2, 3]
    }

    result = uncertainty.compute_vertex_sigma(1, observations)

    assert result["sigma_mm"][0] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=0))
    assert result["residual_count"][0] == 3


def test_compute_vertex_sigma_is_nan_for_fewer_than_two_observations():
    observations = {
        "vertex_index": np.array([0, 1, 1]),
        "frame_index": np.array([0, 0, 1]),
        "z_cam_mm": np.array([50.0, 50.0, 50.0]),
        "measured_depth_mm": np.array([51.0, 51.0, 52.0]),
    }

    result = uncertainty.compute_vertex_sigma(2, observations)

    assert np.isnan(result["sigma_mm"][0])  # vertex 0: single observation
    assert result["residual_count"][0] == 1
    assert not np.isnan(result["sigma_mm"][1])  # vertex 1: two observations
    assert result["residual_count"][1] == 2


def test_compute_vertex_sigma_is_nan_for_unobserved_vertex():
    observations = {
        "vertex_index": np.empty(0, dtype=np.int64),
        "frame_index": np.empty(0, dtype=np.int64),
        "z_cam_mm": np.empty(0),
        "measured_depth_mm": np.empty(0),
    }

    result = uncertainty.compute_vertex_sigma(1, observations)

    assert np.isnan(result["sigma_mm"][0])
    assert result["residual_count"][0] == 0


# --- downgrade_high_sigma_vertices -----------------------------------------------------------


def test_downgrade_high_sigma_vertices():
    labels = np.array(
        [
            provenance.LABEL_OBSERVED,  # high sigma -> downgraded
            provenance.LABEL_OBSERVED,  # low sigma -> stays Observed
            provenance.LABEL_UNOBSERVED,  # already Unobserved, high sigma -> untouched
            provenance.LABEL_OBSERVED,  # NaN sigma (unmeasurable) -> untouched
        ]
    )
    sigma_mm = np.array([10.0, 1.0, 10.0, np.nan])

    new_labels, downgraded_mask = uncertainty.downgrade_high_sigma_vertices(
        labels, sigma_mm, threshold_mm=3.0
    )

    assert list(new_labels) == [
        provenance.LABEL_UNOBSERVED,
        provenance.LABEL_OBSERVED,
        provenance.LABEL_UNOBSERVED,
        provenance.LABEL_OBSERVED,
    ]
    np.testing.assert_array_equal(downgraded_mask, [True, False, False, False])
    # Original array is untouched.
    assert labels[0] == provenance.LABEL_OBSERVED


def test_downgrade_high_sigma_vertices_rejects_non_positive_threshold():
    with pytest.raises(ValueError, match="threshold_mm"):
        uncertainty.downgrade_high_sigma_vertices(
            np.array([provenance.LABEL_OBSERVED]), np.array([1.0]), threshold_mm=0.0
        )


# --- compute_sigma_distribution ----------------------------------------------------------------


def test_compute_sigma_distribution_matches_hand_computed_stats():
    sigma_mm = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    labels = np.array(
        [
            provenance.LABEL_OBSERVED,
            provenance.LABEL_OBSERVED,
            provenance.LABEL_OBSERVED,
            provenance.LABEL_UNOBSERVED,  # excluded: not Observed
            provenance.LABEL_OBSERVED,  # excluded: NaN sigma
        ]
    )

    distribution = uncertainty.compute_sigma_distribution(sigma_mm, labels)

    assert distribution["count"] == 3
    assert distribution["min_mm"] == pytest.approx(1.0)
    assert distribution["max_mm"] == pytest.approx(3.0)
    assert distribution["mean_mm"] == pytest.approx(2.0)
    assert distribution["median_mm"] == pytest.approx(2.0)
    assert distribution["p90_mm"] == pytest.approx(np.percentile([1.0, 2.0, 3.0], 90))


def test_compute_sigma_distribution_handles_no_observed_vertices():
    distribution = uncertainty.compute_sigma_distribution(
        np.array([np.nan]), np.array([provenance.LABEL_UNOBSERVED])
    )

    assert distribution["count"] == 0
    assert distribution["mean_mm"] is None


# --- sigma_mm.npz round trip --------------------------------------------------------------------


def test_sigma_npz_round_trips(tmp_path):
    sigma_mm = np.array([1.5, np.nan, 3.5])
    residual_count = np.array([3, 0, 2])
    path = tmp_path / "sigma_mm.npz"

    uncertainty.write_sigma_npz(path, sigma_mm, residual_count)
    read_back = uncertainty.read_sigma_npz(path)

    np.testing.assert_allclose(read_back["sigma_mm"], sigma_mm, equal_nan=True)
    np.testing.assert_array_equal(read_back["residual_count"], residual_count)


# --- uncertainty_run: end to end on synthetic R-09 output -----------------------------------------


def _write_synthetic_provenance_output(
    provenance_dir, vertices, faces, labels, observations
):
    provenance_dir.mkdir()
    mesh_cleanup.write_glb(
        provenance_dir / "mesh_provenance.glb",
        vertices,
        faces,
        provenance.labels_to_colors(labels),
    )
    provenance.write_observations_npz(provenance_dir / "observations.npz", observations)


def test_uncertainty_run_downgrades_and_writes_outputs(tmp_path):
    provenance_dir = tmp_path / "provenance"
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    labels = np.array(
        [
            provenance.LABEL_OBSERVED,
            provenance.LABEL_OBSERVED,
            provenance.LABEL_UNOBSERVED,
        ]
    )
    # Vertex 0: two observations with a wide residual spread (sigma > 3mm threshold) -> downgraded.
    # Vertex 1: two observations with a tight spread -> stays Observed.
    # Vertex 2: already Unobserved, no observations.
    observations = {
        "vertex_index": np.array([0, 0, 1, 1]),
        "frame_index": np.array([0, 1, 0, 1]),
        "z_cam_mm": np.array([50.0, 50.0, 50.0, 50.0]),
        "measured_depth_mm": np.array([40.0, 60.0, 50.1, 49.9]),
    }
    _write_synthetic_provenance_output(
        provenance_dir, vertices, faces, labels, observations
    )
    out_dir = tmp_path / "uncertainty_out"

    report = uncertainty.uncertainty_run(
        provenance_dir, out_dir, sigma_downgrade_threshold_mm=3.0
    )

    assert report["vertex_count"] == 3
    assert report["downgraded_vertex_count"] == 1
    assert (
        report["sigma_distribution_mm"]["count"] == 1
    )  # only vertex 1 is Observed w/ sigma

    read_vertices, read_faces, read_colors = mesh_cleanup.read_glb(
        out_dir / "mesh_uncertainty.glb"
    )
    np.testing.assert_allclose(read_vertices, vertices)
    np.testing.assert_array_equal(read_faces, faces)
    np.testing.assert_array_equal(
        read_colors,
        [
            provenance.UNOBSERVED_COLOR,
            provenance.OBSERVED_COLOR,
            provenance.UNOBSERVED_COLOR,
        ],
    )

    sigma_data = uncertainty.read_sigma_npz(out_dir / "sigma_mm.npz")
    assert sigma_data["sigma_mm"][0] == pytest.approx(10.0)  # std([-10, 10]) = 10
    assert sigma_data["sigma_mm"][1] == pytest.approx(0.1)  # std([0.1, -0.1]) = 0.1

    on_disk_report = json.loads((out_dir / "uncertainty.json").read_text())
    assert on_disk_report == report


def test_uncertainty_run_refuses_to_overwrite_existing_output(tmp_path):
    provenance_dir = tmp_path / "provenance"
    vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 0, 0]], dtype=np.int32)
    labels = np.array([provenance.LABEL_UNOBSERVED])
    observations = {
        "vertex_index": np.empty(0, dtype=np.int64),
        "frame_index": np.empty(0, dtype=np.int64),
        "z_cam_mm": np.empty(0),
        "measured_depth_mm": np.empty(0),
    }
    _write_synthetic_provenance_output(
        provenance_dir, vertices, faces, labels, observations
    )
    out_dir = tmp_path / "uncertainty_out"

    uncertainty.uncertainty_run(provenance_dir, out_dir)

    with pytest.raises(FileExistsError, match="already has files"):
        uncertainty.uncertainty_run(provenance_dir, out_dir)


def test_uncertainty_run_rejects_mesh_without_colors(tmp_path):
    provenance_dir = tmp_path / "provenance"
    provenance_dir.mkdir()
    mesh_cleanup.write_glb(
        provenance_dir / "mesh_provenance.glb",
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0, 0, 0]], dtype=np.int32),
    )  # no colors
    provenance.write_observations_npz(
        provenance_dir / "observations.npz",
        {
            "vertex_index": np.empty(0, dtype=np.int64),
            "frame_index": np.empty(0, dtype=np.int64),
            "z_cam_mm": np.empty(0),
            "measured_depth_mm": np.empty(0),
        },
    )

    with pytest.raises(ValueError, match="no vertex colors"):
        uncertainty.uncertainty_run(provenance_dir, tmp_path / "out")
