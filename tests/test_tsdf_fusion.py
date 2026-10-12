"""Unit tests for R-07's TSDF fusion. All synthetic data -- no GPU, no golden capture."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recon import tsdf_fusion
from recon.vggt_runner import confidence_keep_mask

# --- build_voxel_grid ------------------------------------------------------------------------


def test_build_voxel_grid_shape_and_centers():
    centers, shape = tsdf_fusion.build_voxel_grid(
        np.array([0.0, 0.0, 0.0]), np.array([20.0, 10.0, 10.0]), voxel_size_mm=10.0
    )

    assert shape == (2, 1, 1)
    assert centers.shape == (2, 3)
    np.testing.assert_allclose(sorted(centers[:, 0].tolist()), [5.0, 15.0])
    np.testing.assert_allclose(centers[:, 1], [5.0, 5.0])
    np.testing.assert_allclose(centers[:, 2], [5.0, 5.0])


def test_build_voxel_grid_rejects_bad_bounds():
    with pytest.raises(ValueError, match="strictly greater"):
        tsdf_fusion.build_voxel_grid(
            np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0]), 1.0
        )


def test_build_voxel_grid_rejects_non_positive_voxel_size():
    with pytest.raises(ValueError, match="voxel_size_mm"):
        tsdf_fusion.build_voxel_grid(np.zeros(3), np.ones(3), voxel_size_mm=0.0)


# --- fuse_tsdf ---------------------------------------------------------------------------------
# A single-pixel-ray setup: a 3x3 image, intrinsic with cx=cy=1 (so a board-frame point at
# x=y=0, any z, projects exactly to pixel (1,1) regardless of z), identity extrinsic. Three
# voxels placed along that ray at z=30/50/70mm, with a measured depth of 50mm at pixel (1,1),
# pin the TSDF sign convention explicitly: in front of the surface is positive/free, behind is
# negative/occupied, matching the docstring's stated convention.

_RAY_INTRINSIC = np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]])[None]
_RAY_EXTRINSIC = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)[None]


def _ray_depth(value_at_center: float) -> np.ndarray:
    depth = np.zeros((1, 3, 3))
    depth[0, 1, 1] = value_at_center
    return depth


def test_fuse_tsdf_sign_convention_in_front_vs_behind_surface():
    voxel_centers = np.array([[0.0, 0.0, 30.0], [0.0, 0.0, 50.0], [0.0, 0.0, 70.0]])
    depth_mm = _ray_depth(50.0)

    tsdf, weight = tsdf_fusion.fuse_tsdf(
        depth_mm, _RAY_EXTRINSIC, _RAY_INTRINSIC, voxel_centers, truncation_mm=25.0
    )

    np.testing.assert_allclose(tsdf, [20.0, 0.0, -20.0])
    np.testing.assert_allclose(weight, [1.0, 1.0, 1.0])


def test_fuse_tsdf_truncates_and_excludes_far_behind_surface():
    # 25mm in front (clipped to the 10mm truncation band) and 25mm behind (excluded entirely --
    # "far behind" means occluded/unknown, not "very occupied").
    voxel_centers = np.array([[0.0, 0.0, 25.0], [0.0, 0.0, 75.0]])
    depth_mm = _ray_depth(50.0)

    tsdf, weight = tsdf_fusion.fuse_tsdf(
        depth_mm, _RAY_EXTRINSIC, _RAY_INTRINSIC, voxel_centers, truncation_mm=10.0
    )

    np.testing.assert_allclose(tsdf, [10.0, 0.0])
    np.testing.assert_allclose(weight, [1.0, 0.0])


def test_fuse_tsdf_averages_across_frames():
    voxel_centers = np.array([[0.0, 0.0, 50.0]])
    depth_mm = np.concatenate([_ray_depth(50.0), _ray_depth(54.0)], axis=0)
    extrinsic = np.concatenate([_RAY_EXTRINSIC, _RAY_EXTRINSIC], axis=0)
    intrinsic = np.concatenate([_RAY_INTRINSIC, _RAY_INTRINSIC], axis=0)

    tsdf, weight = tsdf_fusion.fuse_tsdf(
        depth_mm, extrinsic, intrinsic, voxel_centers, truncation_mm=25.0
    )

    # frame 0: sdf = 50-50 = 0; frame 1: sdf = 54-50 = 4. Running average: (0+4)/2 = 2.
    np.testing.assert_allclose(tsdf, [2.0])
    np.testing.assert_allclose(weight, [2.0])


def test_fuse_tsdf_keep_mask_excludes_masked_pixels():
    voxel_centers = np.array([[0.0, 0.0, 30.0], [0.0, 0.0, 50.0], [0.0, 0.0, 70.0]])
    depth_mm = _ray_depth(50.0)
    keep_mask = np.zeros((1, 3, 3), dtype=bool)  # pixel (1,1) masked out everywhere

    _tsdf, weight = tsdf_fusion.fuse_tsdf(
        depth_mm,
        _RAY_EXTRINSIC,
        _RAY_INTRINSIC,
        voxel_centers,
        truncation_mm=25.0,
        keep_mask=keep_mask,
    )

    np.testing.assert_allclose(weight, [0.0, 0.0, 0.0])


def test_fuse_tsdf_confidence_derived_mask_excludes_a_spurious_reading():
    # R-20's own acceptance criterion: "a low-confidence spurious region must not appear in the
    # fused TSDF." Two frames look at the same pixel (the x=y=0 trick above): frame 0 measures
    # the true, trusted surface at z=50; frame 1 measures a different, untrusted depth (z=100)
    # that would otherwise plant its own zero-crossing elsewhere in the column. keep_mask comes
    # from recon.vggt_runner.confidence_keep_mask on each frame's own confidence -- the same
    # function R-01's point cloud uses -- not a hand-written boolean array, so this actually
    # exercises the confidence predicate, not just fuse_tsdf's pre-existing generic masking.
    voxel_centers = np.array(
        [
            [0.0, 0.0, 30.0],
            [0.0, 0.0, 50.0],
            [0.0, 0.0, 70.0],
            [0.0, 0.0, 90.0],
            [0.0, 0.0, 110.0],
        ]
    )
    extrinsic = np.concatenate([_RAY_EXTRINSIC, _RAY_EXTRINSIC], axis=0)
    intrinsic = np.concatenate([_RAY_INTRINSIC, _RAY_INTRINSIC], axis=0)
    depth_mm = np.concatenate([_ray_depth(50.0), _ray_depth(100.0)], axis=0)
    confidence = np.zeros((2, 3, 3))
    confidence[0, 1, 1] = 5.0  # frame 0: trusted
    confidence[1, 1, 1] = 0.5  # frame 1: untrusted

    keep_mask = confidence_keep_mask(confidence, min_confidence=1.0)
    _tsdf, weight = tsdf_fusion.fuse_tsdf(
        depth_mm,
        extrinsic,
        intrinsic,
        voxel_centers,
        truncation_mm=25.0,
        keep_mask=keep_mask,
    )

    # Frame 1's reading at z=100 (truncation band [75,125]) never contributes: voxels at z=90
    # and z=110 stay entirely unobserved, even though they sit inside that band.
    np.testing.assert_allclose(weight, [1.0, 1.0, 1.0, 0.0, 0.0])

    # Control, in the same test: with confidence ignored (no mask at all), frame 1's spurious
    # reading DOES contribute -- proving this test would actually catch the problem, not just
    # pass regardless of whether fusion reads confidence at all.
    _tsdf_unfiltered, weight_unfiltered = tsdf_fusion.fuse_tsdf(
        depth_mm,
        extrinsic,
        intrinsic,
        voxel_centers,
        truncation_mm=25.0,
        keep_mask=None,
    )
    assert weight_unfiltered[3] > 0.0
    assert weight_unfiltered[4] > 0.0


def test_fuse_tsdf_rejects_bad_shapes():
    with pytest.raises(ValueError, match="depth_mm"):
        tsdf_fusion.fuse_tsdf(
            np.zeros((3, 3)), _RAY_EXTRINSIC, _RAY_INTRINSIC, np.zeros((1, 3)), 1.0
        )
    with pytest.raises(ValueError, match="truncation_mm"):
        tsdf_fusion.fuse_tsdf(
            _ray_depth(50.0), _RAY_EXTRINSIC, _RAY_INTRINSIC, np.zeros((1, 3)), 0.0
        )


# --- extract_mesh_marching_cubes ---------------------------------------------------------------


def test_extract_mesh_marching_cubes_finds_surface_at_expected_mm_location():
    shape = (2, 2, 2)
    tsdf = np.empty(shape)
    tsdf[:, :, 0] = 1.0  # z-index 0: in front / free
    tsdf[:, :, 1] = -1.0  # z-index 1: behind / occupied
    weight = np.ones(shape)
    origin_mm = np.array([100.0, 200.0, 300.0])

    vertices_mm, faces = tsdf_fusion.extract_mesh_marching_cubes(
        tsdf.reshape(-1),
        weight.reshape(-1),
        shape,
        voxel_size_mm=10.0,
        origin_mm=origin_mm,
    )

    assert len(vertices_mm) > 0
    assert len(faces) > 0
    # Zero-crossing is exactly halfway between the two z layers (linear interpolation, +1 to -1).
    np.testing.assert_allclose(vertices_mm[:, 2], 305.0, atol=1e-4)


def test_extract_mesh_marching_cubes_returns_empty_when_nothing_observed():
    shape = (2, 2, 2)
    tsdf = np.zeros(shape)
    weight = np.zeros(shape)

    vertices_mm, faces = tsdf_fusion.extract_mesh_marching_cubes(
        tsdf.reshape(-1),
        weight.reshape(-1),
        shape,
        voxel_size_mm=10.0,
        origin_mm=np.zeros(3),
    )

    assert len(vertices_mm) == 0
    assert len(faces) == 0


# --- write_mesh_ply / read_mesh_ply --------------------------------------------------------------


def test_mesh_ply_round_trips(tmp_path):
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    path = tmp_path / "mesh.ply"

    tsdf_fusion.write_mesh_ply(path, vertices, faces)
    read_vertices, read_faces = tsdf_fusion.read_mesh_ply(path)

    np.testing.assert_allclose(read_vertices, vertices)
    np.testing.assert_array_equal(read_faces, faces)


def test_write_mesh_ply_rejects_non_triangle_faces():
    with pytest.raises(ValueError, match="faces"):
        tsdf_fusion.write_mesh_ply("unused.ply", np.zeros((3, 3)), np.zeros((1, 4)))


# --- extract_mesh_poisson (Open3D not in requirements.txt -- see module docstring) --------------


def test_extract_mesh_poisson_without_open3d_raises_helpful_error():
    try:
        import open3d  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip(
            "open3d is installed in this environment; can't test the not-installed path"
        )

    with pytest.raises(ImportError, match="Open3D"):
        tsdf_fusion.extract_mesh_poisson(np.zeros((1, 3)))


def test_extract_mesh_poisson_on_a_synthetic_point_cloud():
    pytest.importorskip("open3d")
    rng = np.random.default_rng(0)
    # Points roughly on a sphere -- enough for Poisson to produce a real surface.
    directions = rng.normal(size=(500, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    points_mm = directions * 50.0

    vertices_mm, faces = tsdf_fusion.extract_mesh_poisson(points_mm, poisson_depth=6)

    assert vertices_mm.shape[1] == 3
    assert faces.shape[1] == 3
    assert len(vertices_mm) > 0
    assert len(faces) > 0


# --- fuse_run: end-to-end wiring on synthetic files ----------------------------------------------


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
    # R-20: fuse_run reads its confidence threshold from the source run's own run.json by
    # default -- writing it here keeps every existing synthetic test's "keep everything"
    # behaviour unchanged (min_confidence=0.0) unless a test deliberately overrides it.
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


def test_fuse_run_writes_mesh_and_report(tmp_path):
    run_dir = tmp_path / "run"
    # A point 50mm "above the board" (board-frame z=-50) sitting inside the default footprint,
    # seen by one frame whose rotation flips z so that board-frame "above" (negative z) is
    # camera-frame "in front" (positive z) -- see recon/part_segmentation.py's docstring for why
    # negative board-frame z means above the board.
    rotation = np.diag([1.0, -1.0, -1.0])
    extrinsic = np.concatenate([rotation, np.zeros((3, 1))], axis=1)[None]
    intrinsic = np.array([[10.0, 0.0, 50.0], [0.0, 10.0, 50.0], [0.0, 0.0, 1.0]])[None]
    depth = np.zeros((1, 100, 150))
    # Board-frame point (80, 60, -50): camera-frame (80, -60, 50) -> u=10*80/50+50=66,
    # v=10*(-60)/50+50=38.
    depth[0, 38, 66] = 50.0
    _write_synthetic_run(run_dir, depth, extrinsic, intrinsic)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    out_dir = tmp_path / "mesh_out"

    report = tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        out_dir,
        voxel_size_mm=10.0,
        footprint_mm=(160.0, 120.0),
        max_part_height_mm=100.0,
    )

    assert report["frame_count"] == 1
    assert report["method"] == "marching_cubes"
    assert report["observed_voxel_count"] >= 1
    assert report["total_voxel_count"] == 16 * 12 * 10  # (160/10) * (120/10) * (100/10)
    assert (out_dir / "mesh.ply").exists()
    vertices, faces = tsdf_fusion.read_mesh_ply(out_dir / "mesh.ply")
    assert len(vertices) == report["vertex_count"]
    assert len(faces) == report["face_count"]
    on_disk_report = json.loads((out_dir / "tsdf_fusion.json").read_text())
    assert on_disk_report == report


def test_fuse_run_refuses_to_overwrite_existing_output(tmp_path):
    run_dir = tmp_path / "run"
    rotation = np.diag([1.0, -1.0, -1.0])
    extrinsic = np.concatenate([rotation, np.zeros((3, 1))], axis=1)[None]
    intrinsic = np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])[None]
    depth = np.zeros((1, 10, 10))
    depth[0, 5, 5] = 30.0
    _write_synthetic_run(run_dir, depth, extrinsic, intrinsic)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    out_dir = tmp_path / "mesh_out"

    tsdf_fusion.fuse_run(run_dir, alignment_json, out_dir, voxel_size_mm=10.0)

    with pytest.raises(FileExistsError, match="already has files"):
        tsdf_fusion.fuse_run(run_dir, alignment_json, out_dir, voxel_size_mm=10.0)


def test_fuse_run_rejects_unknown_method(tmp_path):
    run_dir = tmp_path / "run"
    extrinsic = np.eye(3, 4)[None]
    intrinsic = np.eye(3)[None]
    depth = np.zeros((1, 4, 4))
    _write_synthetic_run(run_dir, depth, extrinsic, intrinsic)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    with pytest.raises(ValueError, match="method"):
        tsdf_fusion.fuse_run(
            run_dir, alignment_json, tmp_path / "out", method="nonsense"
        )


# --- R-20: fuse_run's own confidence-threshold wiring -------------------------------------------
#
# The fuse_tsdf-level test above already proves a confidence-derived mask keeps a spurious
# reading out of the fused TSDF. These tests cover what's actually new in R-20: fuse_run
# resolving the right threshold (from the run's own run.json by default, or an explicit
# override) and computing the right mask from it. `confidence_dropped_pixel_count` is used as
# the check because it is exact and deterministic -- unlike mesh vertex positions, it does not
# depend on marching cubes' neighbourhood requirements or this camera's perspective geometry.

_CONFIDENCE_TEST_FUSE_KWARGS = {
    "voxel_size_mm": 10.0,
    "footprint_mm": (160.0, 120.0),
    "max_part_height_mm": 100.0,
}


def _write_confidence_test_run(
    run_dir, true_confidence, spurious_confidence, min_confidence_in_run_json=0.0
) -> None:
    """One frame, two measurements, both of them real part geometry that `segment_part` would
    keep on its own (inside the default footprint and well above the height threshold) -- so
    confidence is the ONLY thing that can tell them apart. Without that, a test could pass for
    the wrong reason: a "spurious" point segmentation would have excluded anyway proves nothing
    about whether confidence filtering actually runs.

    `true_confidence` is at board-frame (80, 60, -50) (the same point
    `test_fuse_run_writes_mesh_and_report` uses). `spurious_confidence` is at board-frame
    (21, 21, -30) -- a different, unrelated location, also inside the footprint.
    """
    rotation = np.diag([1.0, -1.0, -1.0])
    extrinsic = np.concatenate([rotation, np.zeros((3, 1))], axis=1)[None]
    intrinsic = np.array([[10.0, 0.0, 50.0], [0.0, 10.0, 50.0], [0.0, 0.0, 1.0]])[None]
    depth = np.zeros((1, 100, 150))
    depth_conf = np.zeros((1, 100, 150))
    depth[0, 38, 66] = 50.0  # board-frame (80, 60, -50)
    depth_conf[0, 38, 66] = true_confidence
    depth[0, 43, 57] = 30.0  # board-frame (21, 21, -30)
    depth_conf[0, 43, 57] = spurious_confidence
    _write_synthetic_run(
        run_dir,
        depth,
        extrinsic,
        intrinsic,
        depth_conf=depth_conf,
        min_confidence=min_confidence_in_run_json,
    )


def test_fuse_run_confidence_threshold_excludes_the_spurious_pixel(tmp_path):
    run_dir = tmp_path / "run"
    _write_confidence_test_run(run_dir, true_confidence=5.0, spurious_confidence=0.5)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    report = tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        tmp_path / "out",
        min_confidence=1.0,
        **_CONFIDENCE_TEST_FUSE_KWARGS,
    )

    # Of the 1*100*150 = 15000 pixels, only the one true pixel (confidence 5.0) clears 1.0 --
    # the spurious pixel (0.5) and every untouched background pixel (0.0, never set) don't.
    assert report["confidence_dropped_pixel_count"] == 100 * 150 - 1

    # Control, in the same test: with confidence ignored (threshold 0), nothing is dropped on
    # confidence grounds -- proving this field actually reflects the threshold, not some fixed
    # number regardless of it.
    report_unfiltered = tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        tmp_path / "out_unfiltered",
        min_confidence=0.0,
        **_CONFIDENCE_TEST_FUSE_KWARGS,
    )
    assert report_unfiltered["confidence_dropped_pixel_count"] == 0

    # The field alone could be computed correctly while the mask it describes never reaches
    # fusion (confidence_dropped_pixel_count comes straight from confidence_mask, independent of
    # whether that mask is actually ANDed into keep_mask) -- observed_voxel_count depends on
    # keep_mask itself, so this is the assertion that would actually fail if that AND were
    # dropped: the spurious point's own voxel column only contributes when confidence lets it
    # through, so filtering it out must observe strictly fewer voxels than keeping it does.
    assert report["observed_voxel_count"] < report_unfiltered["observed_voxel_count"]


def test_fuse_run_reads_min_confidence_from_run_json_by_default(tmp_path):
    run_dir = tmp_path / "run"
    _write_confidence_test_run(
        run_dir,
        true_confidence=5.0,
        spurious_confidence=0.5,
        min_confidence_in_run_json=1.0,
    )
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    report = tsdf_fusion.fuse_run(
        run_dir, alignment_json, tmp_path / "out", **_CONFIDENCE_TEST_FUSE_KWARGS
    )

    assert report["min_confidence"] == 1.0
    assert report["min_confidence_source"] == "run.json"
    assert report["confidence_dropped_pixel_count"] == 100 * 150 - 1


def test_fuse_run_explicit_min_confidence_overrides_run_json(tmp_path):
    run_dir = tmp_path / "run"
    _write_confidence_test_run(
        run_dir,
        true_confidence=5.0,
        spurious_confidence=0.5,
        min_confidence_in_run_json=1.0,
    )
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    report = tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        tmp_path / "out",
        min_confidence=0.0,
        **_CONFIDENCE_TEST_FUSE_KWARGS,
    )

    assert report["min_confidence"] == 0.0
    assert report["min_confidence_source"] == "override"
    assert report["confidence_dropped_pixel_count"] == 0


def test_fuse_run_raises_if_run_json_missing_and_no_override_given(tmp_path):
    run_dir = tmp_path / "run"
    _write_confidence_test_run(run_dir, true_confidence=5.0, spurious_confidence=0.5)
    (run_dir / "run.json").unlink()
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    with pytest.raises(FileNotFoundError, match="run.json"):
        tsdf_fusion.fuse_run(
            run_dir, alignment_json, tmp_path / "out", **_CONFIDENCE_TEST_FUSE_KWARGS
        )


def test_fuse_run_raises_if_run_json_predates_min_confidence(tmp_path):
    run_dir = tmp_path / "run"
    _write_confidence_test_run(run_dir, true_confidence=5.0, spurious_confidence=0.5)
    # An R-01 run from before R-20 -- run.json exists but has no min_confidence field.
    (run_dir / "run.json").write_text(json.dumps({"story": "R-01"}), encoding="utf-8")
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    with pytest.raises(ValueError, match="min_confidence"):
        tsdf_fusion.fuse_run(
            run_dir, alignment_json, tmp_path / "out", **_CONFIDENCE_TEST_FUSE_KWARGS
        )


def test_fuse_run_poisson_path_also_respects_confidence(tmp_path, monkeypatch):
    captured = {}

    def fake_extract_mesh_poisson(
        points_mm, poisson_depth=tsdf_fusion.DEFAULT_POISSON_DEPTH
    ):
        captured["points_mm"] = np.asarray(points_mm)
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    monkeypatch.setattr(tsdf_fusion, "extract_mesh_poisson", fake_extract_mesh_poisson)

    run_dir = tmp_path / "run"
    _write_confidence_test_run(run_dir, true_confidence=5.0, spurious_confidence=0.5)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        tmp_path / "out",
        method="poisson",
        min_confidence=1.0,
        **_CONFIDENCE_TEST_FUSE_KWARGS,
    )

    # Only the true, trusted point (board-frame (80, 60, -50)) reaches Poisson -- the spurious
    # pixel is excluded by confidence before extract_mesh_poisson ever sees it, same as the
    # marching-cubes path.
    points_mm = captured["points_mm"]
    assert len(points_mm) == 1
    np.testing.assert_allclose(points_mm[0], [80.0, 60.0, -50.0])
