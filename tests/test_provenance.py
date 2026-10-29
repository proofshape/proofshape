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


def _all_trusted(depth_mm: np.ndarray) -> np.ndarray:
    """R-24: compute_observations requires a trusted-pixel mask. Tests about geometry (occlusion,
    bounds) rather than confidence pass this, so trust is never what decides their outcome."""
    return np.ones(np.shape(depth_mm), dtype=bool)


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
        vertices,
        _RAY_EXTRINSIC,
        _RAY_INTRINSIC,
        depth_mm,
        tolerance_mm=5.0,
        trusted_pixel_mask=_all_trusted(depth_mm),
    )

    np.testing.assert_array_equal(observations["vertex_index"], [0])
    np.testing.assert_array_equal(observations["frame_index"], [0])
    np.testing.assert_allclose(observations["z_cam_mm"], [50.0])
    np.testing.assert_allclose(observations["measured_depth_mm"], [50.0])


def test_compute_observations_returns_empty_table_when_nothing_observed():
    vertices = np.array([[1000.0, 1000.0, 1000.0]])
    depth_mm = np.zeros((1, 3, 3))

    observations = provenance.compute_observations(
        vertices,
        _RAY_EXTRINSIC,
        _RAY_INTRINSIC,
        depth_mm,
        tolerance_mm=5.0,
        trusted_pixel_mask=_all_trusted(depth_mm),
    )

    assert len(observations["vertex_index"]) == 0


def test_compute_observations_rejects_bad_shapes():
    with pytest.raises(ValueError, match="depth_mm"):
        provenance.compute_observations(
            np.zeros((1, 3)),
            _RAY_EXTRINSIC,
            _RAY_INTRINSIC,
            np.zeros((3, 3)),
            5.0,
            trusted_pixel_mask=_all_trusted((3, 3)),
        )
    with pytest.raises(ValueError, match="tolerance_mm"):
        provenance.compute_observations(
            np.zeros((1, 3)),
            _RAY_EXTRINSIC,
            _RAY_INTRINSIC,
            np.zeros((1, 3, 3)),
            0.0,
            trusted_pixel_mask=_all_trusted((1, 3, 3)),
        )


# --- R-24: compute_observations only counts trusted pixels ---------------------------------------


def test_compute_observations_untrusted_pixel_cannot_vouch_for_a_vertex():
    # The vertex is depth-consistent with the measured 50mm at pixel (1,1) -- geometrically it
    # is exactly what R-09 counts as an observation. Only trust differs between the two calls.
    vertices = np.array([[0.0, 0.0, 50.0]])
    depth_mm = np.zeros((1, 3, 3))
    depth_mm[0, 1, 1] = 50.0
    untrusted_at_landed_pixel = _all_trusted(depth_mm)
    untrusted_at_landed_pixel[0, 1, 1] = False

    filtered = provenance.compute_observations(
        vertices,
        _RAY_EXTRINSIC,
        _RAY_INTRINSIC,
        depth_mm,
        tolerance_mm=5.0,
        trusted_pixel_mask=untrusted_at_landed_pixel,
    )
    # Control, same scene: confidence ignored (every pixel trusted) -- the observation is back,
    # proving it was the mask, not geometry, that removed it above.
    unfiltered = provenance.compute_observations(
        vertices,
        _RAY_EXTRINSIC,
        _RAY_INTRINSIC,
        depth_mm,
        tolerance_mm=5.0,
        trusted_pixel_mask=_all_trusted(depth_mm),
    )

    assert len(filtered["vertex_index"]) == 0
    np.testing.assert_array_equal(unfiltered["vertex_index"], [0])


def test_compute_observations_rejects_a_non_boolean_mask():
    # The mistake this guards against: passing depth_conf itself. Coercing it to bool would trust
    # every nonzero confidence, including VGGT's untrusted floor of 1.0.
    depth_mm = np.zeros((1, 3, 3))
    raw_confidence = np.ones((1, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="boolean"):
        provenance.compute_observations(
            np.zeros((1, 3)),
            _RAY_EXTRINSIC,
            _RAY_INTRINSIC,
            depth_mm,
            trusted_pixel_mask=raw_confidence,
        )


def test_compute_observations_rejects_a_mask_shaped_unlike_depth():
    depth_mm = np.zeros((1, 3, 3))

    with pytest.raises(ValueError, match="trusted_pixel_mask"):
        provenance.compute_observations(
            np.zeros((1, 3)),
            _RAY_EXTRINSIC,
            _RAY_INTRINSIC,
            depth_mm,
            trusted_pixel_mask=np.ones((1, 4, 4), dtype=bool),
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


def test_colors_to_labels_round_trips_labels_to_colors():
    labels = np.array(
        [
            provenance.LABEL_OBSERVED,
            provenance.LABEL_UNOBSERVED,
            provenance.LABEL_OBSERVED,
        ]
    )

    round_tripped = provenance.colors_to_labels(provenance.labels_to_colors(labels))

    np.testing.assert_array_equal(round_tripped, labels)


def test_colors_to_labels_rejects_unrecognised_colors():
    with pytest.raises(ValueError, match="match neither"):
        provenance.colors_to_labels(np.array([[1, 2, 3]], dtype=np.uint8))


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


def _write_synthetic_run(
    run_dir, depth, extrinsic, intrinsic, depth_conf=None, min_confidence=0.0
) -> None:
    run_dir.mkdir()
    frame_count = depth.shape[0]
    np.savez_compressed(
        run_dir / "poses.npz",
        frame_names=np.array([f"frame_{i}.jpg" for i in range(frame_count)]),
        extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32),
    )
    if depth_conf is None:
        depth_conf = np.ones_like(depth, dtype=np.float32)
    np.savez_compressed(
        run_dir / "depth.npz",
        depth=depth.astype(np.float32),
        depth_conf=depth_conf.astype(np.float32),
    )
    # R-24: provenance_run reads its confidence threshold from the source run's own run.json by
    # default, exactly like fuse_run -- writing it here keeps every existing synthetic test's
    # "trust every pixel" behaviour unchanged (min_confidence=0.0) unless a test overrides it.
    (run_dir / "run.json").write_text(
        json.dumps({"min_confidence": min_confidence}), encoding="utf-8"
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
        [[0.0, 0.0, 50.0], [5.0, 5.0, 30.0], [1000.0, 0.0, 50.0]], dtype=np.float32
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


# --- R-24: provenance_run only lets trusted pixels vouch for a vertex ----------------------------
#
# All on _build_two_frame_scene, where vertex 0 is Observed by geometry alone (two frames, ~22deg
# apart, both depth-consistent). Every pixel gets a comfortably trusted confidence of 5.0 except
# the ones a test deliberately sets to 1.0 -- VGGT's real confidence floor, the exact value D-045's
# threshold of 1.0001 exists to exclude -- so confidence is the only thing that differs between a
# test and its control, never the geometry.

_FRAME_0_VERTEX_0_PIXEL = (0, 10, 10)
_FRAME_1_VERTEX_0_PIXEL = (1, 10, 6)
_VGGT_CONFIDENCE_FLOOR = 1.0
_D045_THRESHOLD = 1.0001


def _write_two_frame_scene_run(
    tmp_path, untrusted_pixels=(), min_confidence=_D045_THRESHOLD
):
    """Write the two-frame scene with chosen pixels at VGGT's floor; return what provenance_run
    needs. `min_confidence` is what run.json records, i.e. what the default path will use."""
    run_dir = tmp_path / "run"
    extrinsic, intrinsic, depth, vertices, faces = _build_two_frame_scene()
    depth_conf = np.full(depth.shape, 5.0)
    for pixel in untrusted_pixels:
        depth_conf[pixel] = _VGGT_CONFIDENCE_FLOOR
    _write_synthetic_run(
        run_dir,
        depth,
        extrinsic,
        intrinsic,
        depth_conf=depth_conf,
        min_confidence=min_confidence,
    )
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    mesh_ply_path = run_dir / "mesh.ply"
    tsdf_fusion.write_mesh_ply(mesh_ply_path, vertices, faces)
    return mesh_ply_path, run_dir, alignment_json


def _vertex_colors(out_dir):
    _vertices, _faces, colors = mesh_cleanup.read_glb(out_dir / "mesh_provenance.glb")
    return colors


def test_provenance_run_vertex_seen_only_through_untrusted_pixels_is_unobserved(
    tmp_path,
):
    # The story's required test. Both of vertex 0's observing pixels sit at VGGT's floor; the
    # run's own run.json says 1.0001, so the default path must refuse both and leave vertex 0
    # Unobserved -- the hard rule's mechanism no longer accepts evidence fusion already rejected.
    mesh_ply_path, run_dir, alignment_json = _write_two_frame_scene_run(
        tmp_path, untrusted_pixels=[_FRAME_0_VERTEX_0_PIXEL, _FRAME_1_VERTEX_0_PIXEL]
    )

    report = provenance.provenance_run(
        mesh_ply_path, run_dir, alignment_json, tmp_path / "filtered"
    )
    # Control, same scene: confidence ignored (override 0.0, i.e. pre-R-24 behaviour) -- vertex 0
    # is Observed again, proving the test catches the regression rather than passing by accident.
    report_unfiltered = provenance.provenance_run(
        mesh_ply_path,
        run_dir,
        alignment_json,
        tmp_path / "unfiltered",
        min_confidence=0.0,
    )

    assert report["observed_vertex_count"] == 0
    np.testing.assert_array_equal(
        _vertex_colors(tmp_path / "filtered")[0], provenance.UNOBSERVED_COLOR
    )
    filtered_observations = provenance.read_observations_npz(
        tmp_path / "filtered" / "observations.npz"
    )
    assert 0 not in filtered_observations["vertex_index"].tolist()

    assert report_unfiltered["observed_vertex_count"] == 1
    np.testing.assert_array_equal(
        _vertex_colors(tmp_path / "unfiltered")[0], provenance.OBSERVED_COLOR
    )


def test_provenance_run_trust_is_judged_per_frame_not_per_vertex(tmp_path):
    # Only frame 1's pixel is untrusted. Frame 0 still observes vertex 0, but one view is not
    # two, so the vertex must drop to Unobserved -- one trusted frame can't carry an untrusted one.
    mesh_ply_path, run_dir, alignment_json = _write_two_frame_scene_run(
        tmp_path, untrusted_pixels=[_FRAME_1_VERTEX_0_PIXEL]
    )

    report = provenance.provenance_run(
        mesh_ply_path, run_dir, alignment_json, tmp_path / "out"
    )

    assert report["observed_vertex_count"] == 0
    observations = provenance.read_observations_npz(
        tmp_path / "out" / "observations.npz"
    )
    frames_seeing_vertex_0 = observations["frame_index"][
        observations["vertex_index"] == 0
    ]
    np.testing.assert_array_equal(frames_seeing_vertex_0, [0])
    # Unfiltered, the scene has 3 observations (2 for vertex 0, 1 for vertex 1); exactly the one
    # through frame 1's untrusted pixel is gone.
    assert report["observation_count"] == 2


def test_provenance_run_excludes_nan_confidence_even_at_threshold_zero(tmp_path):
    # PR #128's lesson for fuse_run, applied here: 0.0 is not "skip the mask" --
    # confidence_keep_mask's isfinite check still refuses a NaN-confidence pixel at any threshold.
    run_dir = tmp_path / "run"
    extrinsic, intrinsic, depth, vertices, faces = _build_two_frame_scene()
    depth_conf = np.ones_like(depth)
    depth_conf[_FRAME_1_VERTEX_0_PIXEL] = np.nan
    _write_synthetic_run(
        run_dir, depth, extrinsic, intrinsic, depth_conf=depth_conf, min_confidence=0.0
    )
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    mesh_ply_path = run_dir / "mesh.ply"
    tsdf_fusion.write_mesh_ply(mesh_ply_path, vertices, faces)

    report = provenance.provenance_run(
        mesh_ply_path, run_dir, alignment_json, tmp_path / "out"
    )

    assert report["min_confidence"] == 0.0
    assert report["confidence_dropped_pixel_count"] == 1
    assert report["observed_vertex_count"] == 0


def test_provenance_run_reads_threshold_from_run_json_and_records_its_source(tmp_path):
    mesh_ply_path, run_dir, alignment_json = _write_two_frame_scene_run(
        tmp_path, untrusted_pixels=[_FRAME_0_VERTEX_0_PIXEL, _FRAME_1_VERTEX_0_PIXEL]
    )

    report = provenance.provenance_run(
        mesh_ply_path, run_dir, alignment_json, tmp_path / "default"
    )
    report_override = provenance.provenance_run(
        mesh_ply_path,
        run_dir,
        alignment_json,
        tmp_path / "override",
        min_confidence=0.0,
    )

    assert report["min_confidence"] == _D045_THRESHOLD
    assert report["min_confidence_source"] == "run.json"
    # Exactly the two floor pixels fall below 1.0001; every other pixel is at 5.0.
    assert report["confidence_dropped_pixel_count"] == 2
    assert report_override["min_confidence"] == 0.0
    assert report_override["min_confidence_source"] == "override"
    assert report_override["confidence_dropped_pixel_count"] == 0


def test_provenance_run_refuses_a_run_without_run_json_before_writing_anything(
    tmp_path,
):
    mesh_ply_path, run_dir, alignment_json = _write_two_frame_scene_run(tmp_path)
    (run_dir / "run.json").unlink()
    out_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="run.json"):
        provenance.provenance_run(mesh_ply_path, run_dir, alignment_json, out_dir)
    assert not out_dir.exists()


def test_provenance_run_refuses_a_run_json_that_predates_min_confidence(tmp_path):
    mesh_ply_path, run_dir, alignment_json = _write_two_frame_scene_run(tmp_path)
    (run_dir / "run.json").write_text(json.dumps({"story": "R-01"}), encoding="utf-8")

    with pytest.raises(ValueError, match="min_confidence"):
        provenance.provenance_run(
            mesh_ply_path, run_dir, alignment_json, tmp_path / "out"
        )


# --- CLI ------------------------------------------------------------------------------------------


def test_main_passes_min_confidence_through_and_reports_its_source(tmp_path, capsys):
    mesh_ply_path, run_dir, alignment_json = _write_two_frame_scene_run(
        tmp_path, untrusted_pixels=[_FRAME_0_VERTEX_0_PIXEL, _FRAME_1_VERTEX_0_PIXEL]
    )
    out_dir = tmp_path / "out"

    exit_code = provenance.main(
        [
            str(mesh_ply_path),
            str(run_dir),
            str(alignment_json),
            "--out-dir",
            str(out_dir),
            "--min-confidence",
            "0",
        ]
    )

    assert exit_code == 0
    assert "min_confidence=0.0 (override)" in capsys.readouterr().out
    report = json.loads((out_dir / "provenance.json").read_text())
    assert (
        report["observed_vertex_count"] == 1
    )  # the override really reached provenance_run


def test_main_reports_a_named_failure_when_run_json_predates_the_field(
    tmp_path, capsys
):
    # The alignment path is left out on purpose: the CLI's own default
    # (<run_dir>/metric_alignment.json, where the helper wrote it) is the path exercised here.
    mesh_ply_path, run_dir, _alignment_json = _write_two_frame_scene_run(tmp_path)
    (run_dir / "run.json").write_text(json.dumps({"story": "R-01"}), encoding="utf-8")

    exit_code = provenance.main(
        [str(mesh_ply_path), str(run_dir), "--out-dir", str(tmp_path / "out")]
    )

    assert exit_code == 1
    error_output = capsys.readouterr().err
    assert "R-09 PROVENANCE: FAIL" in error_output
    assert "min_confidence" in error_output

# --- R-25: signed camera-facing gate and explicit depth-gate evidence -----------------------------

def test_compute_observations_distinguishes_front_and_back_facing_vertices():
    vertices = np.array([[0.0, 0.0, 50.0], [1.0, 0.0, 50.0]])
    extrinsic = np.array(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
    )
    intrinsic = np.array([[[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]])
    depth = np.full((1, 11, 11), 50.0)
    trusted = np.ones_like(depth, dtype=bool)
    # Camera is at the origin looking +z. The -z normal faces it; +z points away.
    normals = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])

    observations = provenance.compute_observations(
        vertices,
        extrinsic,
        intrinsic,
        depth,
        trusted_pixel_mask=trusted,
        vertex_normals_mm=normals,
    )

    np.testing.assert_array_equal(observations["vertex_index"], [0])
    np.testing.assert_array_equal(observations["depth_gate_rejected_count"], [0, 0])


def test_compute_observations_counts_depth_gate_rejection_per_vertex():
    vertices = np.array([[0.0, 0.0, 53.0]])
    extrinsic = np.array(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
    )
    intrinsic = np.array([[[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]])
    depth = np.full((1, 11, 11), 50.0)
    trusted = np.ones_like(depth, dtype=bool)
    normals = np.array([[0.0, 0.0, -1.0]])

    observations = provenance.compute_observations(
        vertices,
        extrinsic,
        intrinsic,
        depth,
        tolerance_mm=2.0,
        trusted_pixel_mask=trusted,
        vertex_normals_mm=normals,
    )

    assert observations["vertex_index"].size == 0
    np.testing.assert_array_equal(observations["depth_gate_rejected_count"], [1])


def test_observed_surface_area_fraction_is_area_weighted():
    # Separate triangles with areas 1 and 9 square mm.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [16.0, 0.0, 0.0],
            [10.0, 3.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    labels = np.array(
        [provenance.LABEL_OBSERVED] * 3 + [provenance.LABEL_UNOBSERVED] * 3
    )

    fraction = provenance.observed_surface_area_fraction(vertices, faces, labels)

    assert fraction == pytest.approx(0.1)
