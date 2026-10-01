"""Unit tests for R-09's per-vertex provenance. All synthetic data -- no GPU, no golden capture."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recon import mesh_cleanup, provenance, tsdf_fusion

# --- compute_observations ---------------------------------------------------------------------
# A single-pixel-ray setup, same style as R-07's tsdf_fusion tests: intrinsic with cx=cy=1 so a
# vertex at x=y=0 (any z) always projects to pixel (1,1), identity extrinsic.

_RAY_INTRINSIC = np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]])[None]
_RAY_EXTRINSIC = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)[None]


def test_compute_observations_keeps_consistent_excludes_occluded_and_out_of_bounds():
    vertices = np.array(
        [
            [
                0.0,
                0.0,
                50.0,
            ],  # depth-consistent with the measured 50mm at pixel (1,1) -> kept
            [
                0.0,
                0.0,
                100.0,
            ],  # same pixel, but 50mm away from measured depth -> occluded
            [1000.0, 0.0, 50.0],  # projects far outside the 3x3 image -> out of bounds
        ]
    )
    depth_mm = np.zeros((1, 3, 3))
    depth_mm[0, 1, 1] = 50.0

    observations = provenance.compute_observations(
        vertices, _RAY_EXTRINSIC, _RAY_INTRINSIC, depth_mm, tolerance_mm=5.0
    )

    np.testing.assert_array_equal(observations["vertex_index"], [0])
    np.testing.assert_array_equal(observations["frame_index"], [0])
    np.testing.assert_allclose(observations["z_cam_mm"], [50.0])
    np.testing.assert_allclose(observations["measured_depth_mm"], [50.0])


def test_compute_observations_returns_empty_table_when_nothing_observed():
    vertices = np.array([[1000.0, 1000.0, 1000.0]])
    depth_mm = np.zeros((1, 3, 3))

    observations = provenance.compute_observations(
        vertices, _RAY_EXTRINSIC, _RAY_INTRINSIC, depth_mm, tolerance_mm=5.0
    )

    assert len(observations["vertex_index"]) == 0


def test_compute_observations_rejects_bad_shapes():
    with pytest.raises(ValueError, match="depth_mm"):
        provenance.compute_observations(
            np.zeros((1, 3)), _RAY_EXTRINSIC, _RAY_INTRINSIC, np.zeros((3, 3)), 5.0
        )
    with pytest.raises(ValueError, match="tolerance_mm"):
        provenance.compute_observations(
            np.zeros((1, 3)), _RAY_EXTRINSIC, _RAY_INTRINSIC, np.zeros((1, 3, 3)), 0.0
        )


# --- triangulation_angle_deg ------------------------------------------------------------------


def test_triangulation_angle_deg_right_angle():
    angle = provenance.triangulation_angle_deg(
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
        np.array([0.0, 10.0, 0.0]),
    )
    assert angle == pytest.approx(90.0)


def test_triangulation_angle_deg_near_zero_for_nearly_coincident_cameras():
    angle = provenance.triangulation_angle_deg(
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 0.0, 0.0]),
        np.array([100.001, 0.0, 0.0]),
    )
    assert angle == pytest.approx(0.0, abs=0.01)


def test_triangulation_angle_deg_rejects_camera_at_vertex():
    with pytest.raises(ValueError, match="coincides"):
        provenance.triangulation_angle_deg(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 0.0]),
        )


# --- label_vertices ----------------------------------------------------------------------------


def test_label_vertices_single_view_is_unobserved_wide_baseline_is_observed_narrow_is_not():
    vertices_mm = np.zeros((3, 3))
    camera_centers_mm = np.array(
        [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [100.001, 0.0, 0.0]]
    )  # frame 0, frame 1 (90deg from frame 0), frame 2 (~coincident with frame 0)
    observations = {
        "vertex_index": np.array([0, 1, 1, 2, 2]),
        "frame_index": np.array([0, 0, 1, 0, 2]),
        "z_cam_mm": np.zeros(5),
        "measured_depth_mm": np.zeros(5),
    }

    result = provenance.label_vertices(3, observations, camera_centers_mm, vertices_mm)

    assert list(result["label"]) == [
        provenance.LABEL_UNOBSERVED,  # vertex 0: seen by only frame 0
        provenance.LABEL_OBSERVED,  # vertex 1: frames 0 & 1, wide baseline
        provenance.LABEL_UNOBSERVED,  # vertex 2: frames 0 & 2, nearly coincident cameras
    ]
    np.testing.assert_array_equal(result["view_count"], [1, 2, 2])


def test_label_vertices_rejects_non_positive_angle_threshold():
    with pytest.raises(ValueError, match="min_triangulation_angle_deg"):
        provenance.label_vertices(
            1,
            {
                "vertex_index": np.empty(0, dtype=np.int64),
                "frame_index": np.empty(0, dtype=np.int64),
            },
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            min_triangulation_angle_deg=0.0,
        )


# --- label_faces / labels_to_colors -------------------------------------------------------------


def test_label_faces_requires_all_three_vertices_observed():
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    vertex_labels = np.array(
        [
            provenance.LABEL_OBSERVED,
            provenance.LABEL_OBSERVED,
            provenance.LABEL_OBSERVED,
            provenance.LABEL_UNOBSERVED,
        ]
    )

    face_labels = provenance.label_faces(faces, vertex_labels)

    assert list(face_labels) == [provenance.LABEL_OBSERVED, provenance.LABEL_UNOBSERVED]


def test_labels_to_colors_maps_per_convention():
    labels = np.array([provenance.LABEL_OBSERVED, provenance.LABEL_UNOBSERVED])

    colors = provenance.labels_to_colors(labels)

    np.testing.assert_array_equal(
        colors, [provenance.OBSERVED_COLOR, provenance.UNOBSERVED_COLOR]
    )


# --- observations.npz round trip ----------------------------------------------------------------


def test_observations_npz_round_trips(tmp_path):
    observations = {
        "vertex_index": np.array([0, 1]),
        "frame_index": np.array([2, 3]),
        "z_cam_mm": np.array([10.5, 20.5]),
        "measured_depth_mm": np.array([10.6, 20.4]),
    }
    path = tmp_path / "observations.npz"

    provenance.write_observations_npz(path, observations)
    read_back = provenance.read_observations_npz(path)

    for key, value in observations.items():
        np.testing.assert_allclose(read_back[key], value)


# --- provenance_run: end to end on synthetic files -----------------------------------------------


def _write_synthetic_run(run_dir, depth, extrinsic, intrinsic) -> None:
    run_dir.mkdir()
    frame_count = depth.shape[0]
    np.savez_compressed(
        run_dir / "poses.npz",
        frame_names=np.array([f"frame_{i}.jpg" for i in range(frame_count)]),
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
    )
    np.savez_compressed(
        run_dir / "depth.npz",
        depth=depth.astype(np.float32),
        depth_conf=np.ones_like(depth, dtype=np.float32),
    )


def _write_identity_alignment(path) -> None:
    path.write_text(
        json.dumps(
            {
                "scale": 1.0,
                "rotation": np.eye(3).tolist(),
                "translation_mm": [0.0, 0.0, 0.0],
            }
        )
    )


def _build_two_frame_scene():
    """Frame 0 at the board origin, frame 1 shifted 20mm in x (identity rotation on both).

    Vertex 0 (0,0,50) is seen by both frames with a wide (~22deg) triangulation angle -> Observed.
    Vertex 1 (5,5,30) is seen only by frame 0 -> Unobserved. Vertex 2 is seen by neither.
    """
    extrinsic = np.stack(
        [
            np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1),
            np.concatenate([np.eye(3), np.array([[-20.0], [0.0], [0.0]])], axis=1),
        ]
    )
    intrinsic = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])[
        None
    ].repeat(2, axis=0)
    depth = np.zeros((2, 21, 21))
    depth[0, 10, 10] = 50.0  # frame 0 sees vertex 0 at pixel (10,10)
    depth[1, 10, 6] = 50.0  # frame 1 sees vertex 0 at pixel (6,10)
    depth[0, 12, 12] = 30.0  # frame 0 sees vertex 1 at pixel (12,12)
    vertices = np.array(
        [[0.0, 0.0, 50.0], [5.0, 5.0, 30.0], [1000.0, 1000.0, 1000.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return extrinsic, intrinsic, depth, vertices, faces


def test_provenance_run_labels_vertices_and_writes_outputs(tmp_path):
    run_dir = tmp_path / "run"
    extrinsic, intrinsic, depth, vertices, faces = _build_two_frame_scene()
    _write_synthetic_run(run_dir, depth, extrinsic, intrinsic)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    mesh_ply_path = run_dir / "mesh.ply"
    tsdf_fusion.write_mesh_ply(mesh_ply_path, vertices, faces)
    out_dir = tmp_path / "provenance_out"

    report = provenance.provenance_run(mesh_ply_path, run_dir, alignment_json, out_dir)

    assert report["vertex_count"] == 3
    assert report["observed_vertex_count"] == 1  # only vertex 0
    assert report["unobserved_vertex_count"] == 2
    assert report["observed_face_count"] == 0  # the one face has unobserved vertices
    assert (out_dir / "mesh_provenance.glb").exists()
    read_vertices, read_faces, read_colors = mesh_cleanup.read_glb(
        out_dir / "mesh_provenance.glb"
    )
    np.testing.assert_allclose(read_vertices, vertices)
    np.testing.assert_array_equal(read_faces, faces)
    np.testing.assert_array_equal(
        read_colors,
        [
            provenance.OBSERVED_COLOR,
            provenance.UNOBSERVED_COLOR,
            provenance.UNOBSERVED_COLOR,
        ],
    )
    observations = provenance.read_observations_npz(out_dir / "observations.npz")
    assert len(observations["vertex_index"]) == 3  # 2 obs for vertex 0, 1 for vertex 1
    on_disk_report = json.loads((out_dir / "provenance.json").read_text())
    assert on_disk_report == report


def test_provenance_run_accepts_glb_input_too(tmp_path):
    run_dir = tmp_path / "run"
    extrinsic, intrinsic, depth, vertices, faces = _build_two_frame_scene()
    _write_synthetic_run(run_dir, depth, extrinsic, intrinsic)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    mesh_glb_path = run_dir / "mesh_clean.glb"
    mesh_cleanup.write_glb(mesh_glb_path, vertices, faces)
    out_dir = tmp_path / "provenance_out"

    report = provenance.provenance_run(mesh_glb_path, run_dir, alignment_json, out_dir)

    assert report["observed_vertex_count"] == 1


def test_provenance_run_rejects_unrecognised_mesh_format(tmp_path):
    with pytest.raises(ValueError, match="Unrecognised mesh format"):
        provenance._read_mesh(tmp_path / "mesh.obj")


def test_provenance_run_refuses_to_overwrite_existing_output(tmp_path):
    run_dir = tmp_path / "run"
    extrinsic, intrinsic, depth, vertices, faces = _build_two_frame_scene()
    _write_synthetic_run(run_dir, depth, extrinsic, intrinsic)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    mesh_ply_path = run_dir / "mesh.ply"
    tsdf_fusion.write_mesh_ply(mesh_ply_path, vertices, faces)
    out_dir = tmp_path / "provenance_out"

    provenance.provenance_run(mesh_ply_path, run_dir, alignment_json, out_dir)

    with pytest.raises(FileExistsError, match="already has files"):
        provenance.provenance_run(mesh_ply_path, run_dir, alignment_json, out_dir)
