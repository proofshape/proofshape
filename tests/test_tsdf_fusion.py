"""Unit tests for R-07's TSDF fusion. All synthetic data -- no GPU, no golden capture."""

from __future__ import annotations

import json

import numpy as np
import pytest

from recon import tsdf_fusion

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
