"""Unit tests for R-07's TSDF fusion. All synthetic data -- no GPU, no golden capture."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from recon import tsdf_fusion
from recon.metric_alignment import project_points_to_camera
from recon.part_segmentation import (
    DEFAULT_FOOTPRINT_MM,
    BoardSurface,
    fit_board_plane_robust,
)
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


# --- measure_voxel_vote_consensus (R-33 criterion 2) -----------------------------------------
# Same single-pixel-ray fixtures as fuse_tsdf's own tests above.


def test_measure_voxel_vote_consensus_classifies_near_in_front_and_behind():
    # z=30: 20mm in front (in_front); z=50: exactly on the measured surface (near_surface);
    # z=70: 20mm behind, within the 25mm truncation band (behind).
    voxel_centers = np.array([[0.0, 0.0, 30.0], [0.0, 0.0, 50.0], [0.0, 0.0, 70.0]])
    depth_mm = _ray_depth(50.0)

    reports = tsdf_fusion.measure_voxel_vote_consensus(
        depth_mm,
        _RAY_EXTRINSIC,
        _RAY_INTRINSIC,
        voxel_centers,
        truncation_mm=25.0,
        near_surface_mm=5.0,
    )

    assert [r.contributing_frame_count for r in reports] == [1, 1, 1]
    assert [r.near_surface_frame_count for r in reports] == [0, 1, 0]
    assert [r.in_front_frame_count for r in reports] == [1, 0, 0]
    assert [r.behind_frame_count for r in reports] == [0, 0, 1]


def test_measure_voxel_vote_consensus_far_behind_contributes_nothing():
    voxel_centers = np.array(
        [[0.0, 0.0, 75.0]]
    )  # 25mm behind, outside a 10mm truncation band
    depth_mm = _ray_depth(50.0)

    (report,) = tsdf_fusion.measure_voxel_vote_consensus(
        depth_mm, _RAY_EXTRINSIC, _RAY_INTRINSIC, voxel_centers, truncation_mm=10.0
    )

    assert report.as_report() == {
        "contributing_frame_count": 0,
        "near_surface_frame_count": 0,
        "in_front_frame_count": 0,
        "behind_frame_count": 0,
    }


def test_measure_voxel_vote_consensus_tallies_one_vote_per_contributing_frame():
    # Same voxel, two frames: frame 0 reads it as on the surface, frame 1 reads it 4mm in front
    # -- with near_surface_mm=2, frame 1's vote is just outside the near band.
    voxel_centers = np.array([[0.0, 0.0, 50.0]])
    depth_mm = np.concatenate([_ray_depth(50.0), _ray_depth(54.0)], axis=0)
    extrinsic = np.concatenate([_RAY_EXTRINSIC, _RAY_EXTRINSIC], axis=0)
    intrinsic = np.concatenate([_RAY_INTRINSIC, _RAY_INTRINSIC], axis=0)

    (report,) = tsdf_fusion.measure_voxel_vote_consensus(
        depth_mm,
        extrinsic,
        intrinsic,
        voxel_centers,
        truncation_mm=25.0,
        near_surface_mm=2.0,
    )

    assert report.contributing_frame_count == 2
    assert report.near_surface_frame_count == 1
    assert report.in_front_frame_count == 1
    assert report.behind_frame_count == 0


def test_measure_voxel_vote_consensus_keep_mask_excludes_masked_pixels():
    voxel_centers = np.array([[0.0, 0.0, 50.0]])
    depth_mm = _ray_depth(50.0)
    keep_mask = np.zeros((1, 3, 3), dtype=bool)

    (report,) = tsdf_fusion.measure_voxel_vote_consensus(
        depth_mm,
        _RAY_EXTRINSIC,
        _RAY_INTRINSIC,
        voxel_centers,
        truncation_mm=25.0,
        keep_mask=keep_mask,
    )

    assert report.contributing_frame_count == 0


def test_measure_voxel_vote_consensus_rejects_bad_shape():
    with pytest.raises(ValueError, match="depth_mm"):
        tsdf_fusion.measure_voxel_vote_consensus(
            np.zeros((3, 3)), _RAY_EXTRINSIC, _RAY_INTRINSIC, np.zeros((1, 3)), 1.0
        )


def test_measure_voxel_vote_consensus_rejects_near_surface_mm_past_truncation():
    with pytest.raises(ValueError, match="near_surface_mm"):
        tsdf_fusion.measure_voxel_vote_consensus(
            _ray_depth(50.0),
            _RAY_EXTRINSIC,
            _RAY_INTRINSIC,
            np.zeros((1, 3)),
            truncation_mm=5.0,
            near_surface_mm=10.0,
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


def test_extract_mesh_marching_cubes_filters_a_thin_free_space_sliver_at_the_edge():
    # R-33 (review on #231): a thin sliver of observed free space (tsdf=+8) reaching right up to
    # never-observed voxels (tsdf stays at fuse_tsdf's own 0.0 default). A plain weight>0 mask
    # alone still lets skimage build a face here, using the never-observed voxels' raw 0.0 value
    # even though they're individually masked False -- this reproduces TabeenRaoof's own finding
    # that an observed-free-space/never-observed boundary fabricates walls. Confirmed empirically
    # (not every observed/unobserved boundary triggers skimage's mask quirk -- a full-width flat
    # slab doesn't, matching skimage's own corner-count-dependent activation rule; a thin sliver,
    # closer to the ragged edge of real multi-view camera coverage, does). The fix's per-cube
    # filter (review on #287) catches every face skimage builds here: only two voxels in the
    # whole volume are observed, so no cube can ever have all 8 corners observed.
    shape = (4, 3, 3)
    tsdf = np.zeros(shape)
    weight = np.zeros(shape)
    tsdf[0:2, 1:2, 1:2] = 8.0
    weight[0:2, 1:2, 1:2] = 1.0

    _vertices_mm, faces = tsdf_fusion.extract_mesh_marching_cubes(
        tsdf.reshape(-1),
        weight.reshape(-1),
        shape,
        voxel_size_mm=10.0,
        origin_mm=np.zeros(3),
    )

    assert len(faces) == 0

    # Not a vacuous test: skimage itself, given the raw unfiltered mask, does fabricate faces at
    # this exact boundary -- confirming this is a real fix, not a no-op on an already-safe case.
    from skimage.measure import marching_cubes

    old_mask = weight > 0.0
    _old_vertices, old_faces, _normals, _values = marching_cubes(
        tsdf, level=0.0, spacing=(10.0, 10.0, 10.0), mask=old_mask
    )
    assert len(old_faces) > 0


def test_extract_mesh_marching_cubes_returns_empty_when_nothing_crosses_zero():
    # Review on #231: skimage's own level-range check runs over the entire raw volume, ignoring
    # mask, and raises ValueError when nothing in the array ever crosses zero -- reachable even
    # before this fix, just rare in practice (a real grid almost always has an untouched 0.0
    # voxel somewhere). A fully-observed, all-positive volume hits it directly.
    shape = (2, 2, 2)
    tsdf = np.full(shape, 5.0)
    weight = np.ones(shape)

    vertices_mm, faces = tsdf_fusion.extract_mesh_marching_cubes(
        tsdf.reshape(-1),
        weight.reshape(-1),
        shape,
        voxel_size_mm=10.0,
        origin_mm=np.zeros(3),
    )

    assert len(vertices_mm) == 0
    assert len(faces) == 0


def test_extract_mesh_marching_cubes_keeps_the_real_plane_and_drops_the_sliver():
    # Review on #287: the sliver test above never reaches `marching_cubes` with a mask that
    # would have found anything to interpolate, so nothing pins which mask actually reaches
    # skimage, or that a real surface right next to a sliver survives the filter intact. This
    # scene does reach skimage for real: a fully observed block holding a genuine surface
    # (the x=2.5 voxel plane), plus a thin observed free-space sliver beside it that the raw,
    # unfiltered mask fabricates extra faces from -- reproduced directly below.
    shape = (12, 8, 8)
    tsdf = np.zeros(shape)
    weight = np.zeros(shape)
    x = np.arange(shape[0])[:, None, None] * np.ones(shape)
    tsdf[0:6] = x[0:6] - 2.5  # real surface at x = 2.5 voxels
    weight[0:6] = 1.0
    tsdf[6:9, 3:4, 3:4] = 8.0  # thin observed free-space sliver
    weight[6:9, 3:4, 3:4] = 1.0

    vertices_mm, faces = tsdf_fusion.extract_mesh_marching_cubes(
        tsdf.reshape(-1),
        weight.reshape(-1),
        shape,
        voxel_size_mm=1.0,
        origin_mm=np.zeros(3),
    )

    assert len(faces) > 0
    centroid_x = vertices_mm[faces].mean(axis=1)[:, 0]
    assert np.all(np.abs(centroid_x - 2.5) < 0.6)

    # Not a vacuous test: skimage itself, given the raw unfiltered mask, builds extra faces that
    # reach out to the sliver, well past the real plane -- confirming the filter is doing real
    # work on this scene, not passing it through untouched.
    from skimage.measure import marching_cubes

    old_mask = weight > 0.0
    _old_vertices, old_faces, _normals, _values = marching_cubes(
        tsdf, level=0.0, spacing=(1.0, 1.0, 1.0), mask=old_mask
    )
    assert len(old_faces) > len(faces)


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


# --- extract_mesh_poisson (Open3D lazily imported -- see its docstring) ---------------------------


def test_extract_mesh_poisson_without_open3d_raises_helpful_error(monkeypatch):
    # Open3D is installed everywhere now (C-08, D-047), so simulate its absence: a None entry in
    # sys.modules makes `import open3d` raise ImportError. Skipping whenever it's installed, as
    # this test used to, would mean it never runs in CI at all.
    monkeypatch.setitem(sys.modules, "open3d", None)

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

    reconstructed_radius = np.linalg.norm(vertices_mm, axis=1)
    absolute_error = np.abs(reconstructed_radius - 50.0)
    assert np.median(absolute_error) < 3.0
    assert np.quantile(absolute_error, 0.90) < 8.0

    centre = vertices_mm.mean(axis=0)
    centred = vertices_mm - centre
    a = centred[faces[:, 0]]
    b = centred[faces[:, 1]]
    c = centred[faces[:, 2]]
    signed_volume = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)
    assert signed_volume > 0.0


def test_poisson_normal_orientation_pins_open_part_toward_board_up():
    """R-27 final-review regression: a real open-bottomed part must not get the opposite
    global normal sign from another object after tangent-plane consistency."""
    o3d = pytest.importorskip("open3d")
    axis = np.linspace(-20.0, 20.0, 13)
    height = np.linspace(-20.0, 0.0, 9)
    xx, yy = np.meshgrid(axis, axis)
    top = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, -20.0)])
    side_points = []
    for z in height:
        side_points.extend(
            [
                np.column_stack(
                    [axis, np.full_like(axis, -20.0), np.full_like(axis, z)]
                ),
                np.column_stack(
                    [axis, np.full_like(axis, 20.0), np.full_like(axis, z)]
                ),
                np.column_stack(
                    [np.full_like(axis, -20.0), axis, np.full_like(axis, z)]
                ),
                np.column_stack(
                    [np.full_like(axis, 20.0), axis, np.full_like(axis, z)]
                ),
            ]
        )
    points = np.vstack([top, *side_points])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals()

    mode = tsdf_fusion._orient_poisson_normals(cloud, points)

    assert mode == "tangent_plane"
    normals = np.asarray(cloud.normals)
    strong_up = np.count_nonzero(normals[:, 2] < -0.2)
    strong_down = np.count_nonzero(normals[:, 2] > 0.2)
    assert strong_up > 1.25 * strong_down


def test_extract_mesh_poisson_handles_an_exactly_planar_grid():
    pytest.importorskip("open3d")
    axis = np.linspace(-20.0, 20.0, 15)
    xx, yy = np.meshgrid(axis, axis)
    points_mm = np.column_stack(
        [xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float64)]
    )

    vertices_mm, faces = tsdf_fusion.extract_mesh_poisson(points_mm, poisson_depth=5)

    assert len(vertices_mm) > 0
    assert len(faces) > 0
    assert np.isfinite(vertices_mm).all()


def test_extract_mesh_poisson_rejects_line_like_input_clearly():
    pytest.importorskip("open3d")
    points_mm = np.column_stack(
        [
            np.linspace(0.0, 20.0, 20),
            np.zeros(20),
            np.zeros(20),
        ]
    )

    with pytest.raises(ValueError, match="rank-deficient"):
        tsdf_fusion.extract_mesh_poisson(points_mm, poisson_depth=5)


# --- reanchor_frames_to_board (R-30) -----------------------------------------------------------


def _flat_board_surface(
    slope_x: float = 0.0, slope_y: float = 0.0, z_at_origin_mm: float = -2.0
) -> BoardSurface:
    """A BoardSurface with a known plane, for tests that need a fixed correction target --
    nothing in production code constructs one this way; measure_board_surface always fits one."""
    return BoardSurface(
        slope_x=slope_x,
        slope_y=slope_y,
        z_at_origin_mm=z_at_origin_mm,
        offset_mm=z_at_origin_mm,
        tilt_deg=0.0,
        residual_sigma_mm=0.0,
        inlier_count=1,
        candidate_count=1,
        board_layer_top_mm=None,
        height_threshold_mm=2.0,
    )


def _board_grid_points(
    slope_x: float, slope_y: float, z_at_origin_mm: float, count: int = 400
) -> np.ndarray:
    """A dense grid of board points (board-frame mm) lying exactly on the given plane, spanning
    the default footprint -- enough to satisfy DEFAULT_MIN_FRAME_BOARD_POINTS on its own."""
    side = int(np.ceil(np.sqrt(count)))
    footprint_width_mm, footprint_height_mm = DEFAULT_FOOTPRINT_MM
    x = np.linspace(10.0, footprint_width_mm - 10.0, side)
    y = np.linspace(10.0, footprint_height_mm - 10.0, side)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
    grid_z = slope_x * grid_x + slope_y * grid_y + z_at_origin_mm
    return np.column_stack([grid_x, grid_y, grid_z])


def test_reanchor_frames_to_board_corrects_a_drifted_frame_and_its_part():
    # Frame 0 already agrees with the consensus plane; frame 1's own board is tilted and offset
    # by a few mm/degrees (D-060's actual finding), *and* carries a thin flat "part" cluster 6mm
    # above its own (drifted) board -- the ghost-layer setup: before correction, frame 1's part
    # sits at a different true height above the real board than frame 0's does, which is exactly
    # what lets a pooled histogram mistake the drift for a second layer.
    target = _flat_board_surface(slope_x=0.0, slope_y=0.0, z_at_origin_mm=-2.0)
    frame0_board = _board_grid_points(0.0, 0.0, -2.0)
    frame0_part = np.array(
        [[80.0, 60.0, -8.0]]
    )  # 6mm above frame 0's own board (-2 - 6 = -8)

    drift_slope_x, drift_slope_y, drift_offset = 0.02, -0.015, -5.3
    frame1_board = _board_grid_points(drift_slope_x, drift_slope_y, drift_offset)
    frame1_part_z = drift_slope_x * 80.0 + drift_slope_y * 60.0 + drift_offset - 6.0
    frame1_part = np.array(
        [[80.0, 60.0, frame1_part_z]]
    )  # 6mm above frame 1's *own* board

    point_count = len(frame0_board) + 1
    board_frame_points = np.stack(
        [
            np.vstack([frame0_board, frame0_part]),
            np.vstack([frame1_board, frame1_part]),
        ]
    ).reshape(2, 1, point_count, 3)
    valid_depth = np.ones((2, 1, point_count), dtype=bool)
    board_extrinsic = np.tile(np.eye(3, 4), (2, 1, 1))

    corrected_points, _corrected_extrinsic, reports = (
        tsdf_fusion.reanchor_frames_to_board(
            board_frame_points, board_extrinsic, valid_depth, target
        )
    )

    # Candidate selection is the same coarse height band measure_board_surface itself uses --
    # wide enough to admit the lone part point too (it's within +-15mm), same as production; the
    # robust fit below is what actually has to reject that single outlier.
    assert [r.candidate_count for r in reports] == [
        len(frame0_board) + 1,
        len(frame1_board) + 1,
    ]

    # Frame 1's own board, re-fit after correction, now agrees with the target plane.
    corrected_frame1_board = corrected_points[1, 0, : len(frame1_board)]
    slope_x, slope_y, z_at_origin_mm, _sigma, _n = fit_board_plane_robust(
        corrected_frame1_board[:, 0],
        corrected_frame1_board[:, 1],
        corrected_frame1_board[:, 2],
        min_inlier_points=3,
    )
    assert slope_x == pytest.approx(0.0, abs=1e-6)
    assert slope_y == pytest.approx(0.0, abs=1e-6)
    assert z_at_origin_mm == pytest.approx(-2.0, abs=1e-6)

    # Both frames' parts now read the same true height above the target board -- no ghost layer.
    corrected_frame0_part = corrected_points[0, 0, -1]
    corrected_frame1_part = corrected_points[1, 0, -1]
    height0 = target.height_above_board_mm(corrected_frame0_part[None, :])[0]
    height1 = target.height_above_board_mm(corrected_frame1_part[None, :])[0]
    # Loose enough to allow the single part point's own (tiny) share of influence on the robust
    # fit it was swept into as a candidate -- see the comment above -- while still being two
    # orders of magnitude tighter than the multi-mm ghost-layer drift this corrects.
    assert height0 == pytest.approx(6.0, abs=0.01)
    assert height1 == pytest.approx(6.0, abs=0.01)


def test_reanchor_frames_to_board_raises_when_a_frame_has_too_few_candidates():
    target = _flat_board_surface()
    board_frame_points = np.zeros((1, 1, 5, 3))
    board_frame_points[0, 0, :, 2] = -2.0  # on-plane, but only 5 points
    valid_depth = np.ones((1, 1, 5), dtype=bool)
    board_extrinsic = np.tile(np.eye(3, 4), (1, 1, 1))

    with pytest.raises(ValueError, match="frame 0.*need at least"):
        tsdf_fusion.reanchor_frames_to_board(
            board_frame_points,
            board_extrinsic,
            valid_depth,
            target,
            min_frame_board_points=100,
        )


def test_reanchor_frames_to_board_extrinsic_round_trips_a_real_point():
    # The correction's whole point: the *same* physical depth measurement must reproject
    # identically whether expressed as (original point, original extrinsic) or (corrected point,
    # corrected extrinsic) -- this is the algebra `reanchor_frames_to_board`'s docstring derives,
    # checked directly rather than only by "the mesh looks right afterward".
    target = _flat_board_surface(slope_x=0.0, slope_y=0.0, z_at_origin_mm=-2.0)
    drifted_board = _board_grid_points(0.03, -0.02, -6.0)
    test_point = np.array([[80.0, 60.0, -40.0]])  # some other point this frame also saw
    all_points = np.vstack([drifted_board, test_point])
    point_count = len(all_points)

    board_frame_points = all_points.reshape(1, 1, point_count, 3)
    valid_depth = np.ones((1, 1, point_count), dtype=bool)
    rotation = np.array(
        [[0.9950, -0.0998, 0.0], [0.0998, 0.9950, 0.0], [0.0, 0.0, 1.0]]
    )  # an arbitrary ~5.7 deg rotation about z, not identity -- exercises R_board actually mattering
    translation = np.array([5.0, -3.0, 120.0])
    intrinsic = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    board_extrinsic = np.concatenate([rotation, translation[:, None]], axis=1)[
        None, :, :
    ]

    corrected_points, corrected_extrinsic, _reports = (
        tsdf_fusion.reanchor_frames_to_board(
            board_frame_points, board_extrinsic, valid_depth, target
        )
    )

    original_u, original_v, original_z = project_points_to_camera(
        test_point, rotation, translation, intrinsic
    )
    corrected_test_point = corrected_points[0, 0, -1][None, :]
    corrected_u, corrected_v, corrected_z = project_points_to_camera(
        corrected_test_point,
        corrected_extrinsic[0, :, :3],
        corrected_extrinsic[0, :, 3],
        intrinsic,
    )
    assert corrected_u == pytest.approx(original_u, abs=1e-6)
    assert corrected_v == pytest.approx(original_v, abs=1e-6)
    assert corrected_z == pytest.approx(original_z, abs=1e-6)


# --- fuse_run: end-to-end wiring on synthetic files ----------------------------------------------
#
# R-21: fuse_run now always fits the real board surface (measure_board_surface), which needs at
# least DEFAULT_MIN_BOARD_POINTS candidate board points spread across the footprint -- a single
# pixel, as R-07's original tests used, is far too few. `_board_and_part_depth` below builds a
# synthetic scene with a real (flat) board layer plus one part pixel, shared by every test in
# this section unless noted otherwise.
#
# Rotation flips z so that board-frame "above" (negative z) is camera-frame "in front" (positive
# z) -- see recon/part_segmentation.py's docstring for why negative board-frame z means above
# the board. With this rotation and zero translation, a pixel at depth D unprojects to
# board-frame z=-D, and to (x,y) = ((u-cx)/fx*D, (v-cy)/fy*D) -- so many pixels sharing the same
# depth D, at different (u,v), land on a flat board-frame plane at that one height (R-21's
# `measure_board_surface` needs enough of those to fit a plane).
_SCENE_ROTATION = np.diag([1.0, -1.0, -1.0])
_SCENE_EXTRINSIC = np.concatenate([_SCENE_ROTATION, np.zeros((3, 1))], axis=1)[None]
# cx=0, cy=120 (not symmetric): with this rotation, board-frame x=(u-cx)/fx*depth but
# y=-(v-cy)/fy*depth, so positive x needs u>cx while positive y needs v<cy -- cx/cy are placed
# so that non-negative pixel indices still land on the footprint's full x in [0,160], y in
# [0,120].
_SCENE_INTRINSIC = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 120.0], [0.0, 0.0, 1.0]])[
    None
]
_SCENE_BOARD_DEPTH_MM = (
    10.0  # board-frame z=-10 for every board pixel below (within the
)
# default 15mm search band `measure_board_surface` looks in around the assumed z=0)
_SCENE_PART_DEPTH_MM = 40.0  # board-frame z=-40, i.e. 30mm above the flat board plane
# NOTE: numpy fancy-indexes depth[0, _SCENE_PART_PIXEL] as *two full rows* (v=105 and v=20
# across every u), not the single pixel the name suggests -- a pre-existing quirk of this
# fixture (see _board_and_part_depth), not something this story's tests rely on or fix.
_SCENE_PART_PIXEL = (
    105,
    20,
)  # (v, u) -> board-frame (x=80, y=60), inside the default footprint


def _board_and_part_depth(
    include_board: bool = True, include_part: bool = True
) -> np.ndarray:
    """A (1, 130, 170) depth image: a grid of same-depth pixels spanning the default 160x120mm
    footprint (a flat board), plus one pixel 30mm above it (the "part"). `_SCENE_EXTRINSIC` /
    `_SCENE_INTRINSIC` above are the matching camera for every test that uses this."""
    depth = np.zeros((1, 130, 170))
    if include_board:
        u_grid = np.arange(0, 161)  # x = u in [0, 160]
        v_grid = np.arange(0, 121)  # y = 120 - v in [0, 120]
        depth[0, v_grid[:, None], u_grid[None, :]] = _SCENE_BOARD_DEPTH_MM
    if include_part:
        depth[0, _SCENE_PART_PIXEL] = _SCENE_PART_DEPTH_MM
    return depth


def _write_synthetic_run(
    run_dir,
    depth,
    extrinsic=_SCENE_EXTRINSIC,
    intrinsic=_SCENE_INTRINSIC,
    depth_conf=None,
    min_confidence=0.0,
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
    _write_synthetic_run(run_dir, _board_and_part_depth())
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
        min_kept_points=0,
    )

    assert report["frame_count"] == 1
    assert report["method"] == "marching_cubes"
    assert report["observed_voxel_count"] >= 1
    assert report["total_voxel_count"] == 16 * 12 * 10  # (160/10) * (120/10) * (100/10)
    assert report["board_surface"]["offset_mm"] == pytest.approx(-10.0, abs=0.5)
    assert report["min_confidence"] == 0.0
    assert report["min_confidence_source"] == "run.json"
    assert (out_dir / "mesh.ply").exists()
    vertices, faces = tsdf_fusion.read_mesh_ply(out_dir / "mesh.ply")
    assert len(vertices) == report["vertex_count"]
    assert len(faces) == report["face_count"]
    on_disk_report = json.loads((out_dir / "tsdf_fusion.json").read_text())
    assert on_disk_report == report


def test_fuse_run_grid_top_follows_the_fitted_plane(tmp_path):
    # The fitted plane sits at z=-10 (board-frame), not z=0 -- the voxel grid's top must follow
    # it, or the grid itself would clip the part's base before fusion even runs.
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, _board_and_part_depth())
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    report = tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        tmp_path / "mesh_out",
        voxel_size_mm=10.0,
        footprint_mm=(160.0, 120.0),
        max_part_height_mm=100.0,
        min_kept_points=0,
    )

    assert report["bounds_max_mm"][2] == pytest.approx(-10.0, abs=0.5)
    assert report["bounds_min_mm"][2] == pytest.approx(-110.0, abs=0.5)


def test_fuse_run_height_threshold_override_skips_the_layer_search(tmp_path):
    # R-26: an explicit height_threshold_mm must reach measure_board_surface too, not just
    # segment_part -- the whole point of the override is to work around a layer-search failure.
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, _board_and_part_depth())
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    report = tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        tmp_path / "mesh_out",
        voxel_size_mm=10.0,
        footprint_mm=(160.0, 120.0),
        max_part_height_mm=100.0,
        height_threshold_mm=5.0,
        min_kept_points=0,
    )

    assert report["board_surface"]["board_layer_top_mm"] is None


def test_fuse_run_excludes_invalid_depth_pixels_from_the_board_fit(tmp_path):
    # A non-finite depth pixel unprojects to a non-finite board-frame point; it must never reach
    # measure_board_surface (which would otherwise either crash on it or let it skew the fit).
    run_dir = tmp_path / "run"
    depth = _board_and_part_depth()
    depth[0, 0, 0] = np.nan
    _write_synthetic_run(run_dir, depth)
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
        min_kept_points=0,
    )

    assert report["board_surface"]["offset_mm"] == pytest.approx(-10.0, abs=0.5)


def test_fuse_run_raises_when_no_board_is_visible(tmp_path):
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, _board_and_part_depth(include_board=False))
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    with pytest.raises(ValueError, match="board points"):
        tsdf_fusion.fuse_run(
            run_dir, alignment_json, tmp_path / "mesh_out", voxel_size_mm=10.0
        )


def test_fuse_run_at_min_confidence_zero_still_excludes_non_finite_confidence(tmp_path):
    # Code review on PR #128 caught a real gap: an earlier "efficiency" version of this
    # function skipped loading depth_conf entirely whenever the resolved threshold was 0.0 (the
    # project-wide "no filtering" sentinel), on the assumption that 0.0 makes the confidence
    # mask a no-op. It doesn't: confidence_keep_mask's own isfinite check still excludes a
    # NaN/inf-confidence pixel at *any* threshold, including 0.0, and the cloud
    # (select_point_cloud) would exclude it too -- skipping the array let such a pixel into the
    # fused mesh when it shouldn't have been. Reverted; this proves the predicate still applies
    # at threshold 0.0, not just at every threshold above it.
    run_dir = tmp_path / "run"
    depth = _board_and_part_depth(include_part=False)
    depth_conf = np.ones_like(depth)
    depth[0, 105, 20] = _SCENE_PART_DEPTH_MM  # board-frame (80, 60, -40)
    depth_conf[0, 105, 20] = np.nan
    _write_synthetic_run(run_dir, depth, depth_conf=depth_conf, min_confidence=0.0)
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    report = tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        tmp_path / "out",
        voxel_size_mm=10.0,
        footprint_mm=(160.0, 120.0),
        max_part_height_mm=100.0,
        min_kept_points=0,
    )

    assert report["min_confidence"] == 0.0
    # Only the one NaN-confidence pixel is dropped on confidence grounds -- every board pixel
    # has a valid confidence of 1.0.
    assert report["confidence_dropped_pixel_count"] == 1


def test_fuse_run_refuses_to_overwrite_existing_output(tmp_path):
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, _board_and_part_depth())
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    out_dir = tmp_path / "mesh_out"

    tsdf_fusion.fuse_run(
        run_dir, alignment_json, out_dir, voxel_size_mm=10.0, min_kept_points=0
    )

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


def test_fuse_run_writes_vote_consensus_when_sample_points_given(tmp_path):
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, _board_and_part_depth())
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)
    out_dir = tmp_path / "mesh_out"
    sample_points_mm = np.array([[80.0, 60.0, -40.0], [10.0, 10.0, -10.0]])

    tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        out_dir,
        voxel_size_mm=10.0,
        min_kept_points=0,
        vote_sample_points_mm=sample_points_mm,
    )

    vote_report = json.loads((out_dir / "vote_consensus.json").read_text())
    assert len(vote_report["votes"]) == 2
    assert vote_report["sample_points_mm"] == sample_points_mm.tolist()
    # the part pixel's own unprojected point: this frame votes it on the surface.
    assert vote_report["votes"][0]["near_surface_frame_count"] == 1


def test_fuse_run_rejects_vote_sample_points_with_poisson(tmp_path):
    run_dir = tmp_path / "run"
    _write_synthetic_run(run_dir, _board_and_part_depth())
    alignment_json = run_dir / "metric_alignment.json"
    _write_identity_alignment(alignment_json)

    with pytest.raises(ValueError, match="marching_cubes"):
        tsdf_fusion.fuse_run(
            run_dir,
            alignment_json,
            tmp_path / "out",
            method="poisson",
            min_kept_points=0,
            vote_sample_points_mm=np.array([[80.0, 60.0, -40.0]]),
        )


# --- R-20: fuse_run's own confidence-threshold wiring -------------------------------------------
#
# The fuse_tsdf-level test above already proves a confidence-derived mask keeps a spurious
# reading out of the fused TSDF. These tests cover what's actually new in R-20: fuse_run
# resolving the right threshold (from the run's own run.json by default, or an explicit
# override) and computing the right mask from it. `confidence_dropped_pixel_count` is used as
# the check because it is exact and deterministic -- unlike mesh vertex positions, it does not
# depend on marching cubes' neighbourhood requirements or this scene's perspective geometry.

_CONFIDENCE_TEST_FUSE_KWARGS = {
    "voxel_size_mm": 10.0,
    "footprint_mm": (160.0, 120.0),
    "max_part_height_mm": 100.0,
    # R-26 (#180): these scenes use a couple of hand-placed pixels, not a realistic point count --
    # min_kept_points is about a real run's segmentation, not these confidence-wiring tests.
    "min_kept_points": 0,
}
_CONFIDENCE_TEST_PIXEL_COUNT = 1 * 130 * 170  # _board_and_part_depth's image size


def _write_confidence_test_run(
    run_dir, true_confidence, spurious_confidence, min_confidence_in_run_json=0.0
) -> None:
    """R-21's board scene (needed so `measure_board_surface` has enough points to fit), plus two
    of this story's own part pixels -- both real part geometry `segment_part` would keep on its
    own (inside the footprint, well above the board), so confidence is the ONLY thing that can
    tell them apart. Without that, a test could pass for the wrong reason: a "spurious" point
    segmentation would have excluded anyway proves nothing about whether confidence filtering
    actually runs. `include_part=False` avoids `_SCENE_PART_PIXEL`'s own fancy-indexing quirk
    (see its comment above) -- these two pixels are indexed directly and unambiguously instead.

    `true_confidence` is at board-frame (80, 60, -40) (pixel (105, 20), 30mm above the board).
    `spurious_confidence` is at board-frame (40, 60, -40) (pixel (105, 10)), a different,
    unrelated location at the same height, also inside the footprint.
    """
    depth = _board_and_part_depth(include_part=False)
    depth_conf = np.zeros_like(depth)
    depth[0, 105, 20] = _SCENE_PART_DEPTH_MM  # board-frame (80, 60, -40)
    depth_conf[0, 105, 20] = true_confidence
    depth[0, 105, 10] = _SCENE_PART_DEPTH_MM  # board-frame (40, 60, -40)
    depth_conf[0, 105, 10] = spurious_confidence
    _write_synthetic_run(
        run_dir,
        depth,
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

    # Of the scene's pixels, only the one true pixel (confidence 5.0) clears 1.0 -- the
    # spurious pixel (0.5) and every untouched board/background pixel (0.0, never set) don't.
    assert report["confidence_dropped_pixel_count"] == _CONFIDENCE_TEST_PIXEL_COUNT - 1

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
    assert report["confidence_dropped_pixel_count"] == _CONFIDENCE_TEST_PIXEL_COUNT - 1


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

    # Only the true, trusted point (board-frame (80, 60, -40)) reaches Poisson -- the spurious
    # pixel is excluded by confidence before extract_mesh_poisson ever sees it, same as the
    # marching-cubes path.
    points_mm = captured["points_mm"]
    assert len(points_mm) == 1
    np.testing.assert_allclose(points_mm[0], [80.0, 60.0, -40.0])
