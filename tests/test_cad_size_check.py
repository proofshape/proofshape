"""Unit tests for R-22's reconstructed-vs-CAD and reconstructed-vs-board size checks.

All synthetic data -- no GPU, no golden capture (same convention as test_part_segmentation.py):
``fixtures/data/`` is entirely gitignored (see .gitignore), so no test here may depend on it.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from recon import cad_size_check
from recon.board_pose import make_board
from recon.cad_size_check import (
    PrismCAD,
    _RunBundle,
    build_argument_parser,
    build_report,
    compute_height_above_board,
    fit_board_plane,
    main,
    measure_board_scale,
    measure_legacy_camera_centre_board_scale,
    measure_prism,
    prism_from_cad,
    read_ascii_stl,
    reconstruct_confidence_filtered_points,
    write_report,
)
from recon.metric_alignment import camera_centres
from recon.vggt_runner import write_ply

# --- shared synthetic-geometry helpers ---------------------------------------------------------


def _box_prism(side_mm: float = 30.0, height_mm: float = 60.0) -> PrismCAD:
    footprint = np.array(
        [[0.0, 0.0], [side_mm, 0.0], [side_mm, side_mm], [0.0, side_mm]]
    )
    edges = [(footprint[i], footprint[(i + 1) % 4]) for i in range(4)]
    return PrismCAD(footprint_xy=footprint, height_mm=height_mm, edges=edges)


def _triangle_prism(legs=(60.0, 26.0), height_mm: float = 30.0) -> PrismCAD:
    a, b = legs
    footprint = np.array([[0.0, 0.0], [a, 0.0], [0.0, b]])
    edges = [(footprint[i], footprint[(i + 1) % 3]) for i in range(3)]
    return PrismCAD(footprint_xy=footprint, height_mm=height_mm, edges=edges)


def _sample_inside_polygon(
    polygon_xy: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    polygon_f32 = polygon_xy.astype(np.float32)
    low, high = polygon_xy.min(axis=0), polygon_xy.max(axis=0)
    accepted: list[np.ndarray] = []
    while len(accepted) < count:
        candidates = rng.uniform(low, high, size=(count, 2))
        for point in candidates:
            if (
                cv2.pointPolygonTest(
                    polygon_f32, (float(point[0]), float(point[1])), False
                )
                >= 0
            ):
                accepted.append(point)
            if len(accepted) >= count:
                break
    return np.array(accepted[:count])


def _synthesize_prism_points(
    prism: PrismCAD,
    rotation_deg: float = 0.0,
    offset_xy: tuple[float, float] = (80.0, 60.0),
    true_scale: float = 1.0,
    board_plane: tuple[float, float, float] = (0.0, 0.0, 0.0),
    noise_mm: float = 0.0,
    points_per_edge: int = 600,
    top_points: int = 1200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A synthetic board-frame-mm point cloud: wall points on every footprint edge plus a top
    face, for a prism placed, rotated, scaled and sampled as given. ``true_scale`` is the
    physical part's real size relative to ``prism`` -- the knob the "must report smaller when
    the part is smaller" tests turn.
    """
    rng = rng or np.random.default_rng(0)
    theta = math.radians(rotation_deg)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    offset = np.array(offset_xy)
    true_height = prism.height_mm * true_scale
    true_edges = [
        (rotation @ (p0 * true_scale) + offset, rotation @ (p1 * true_scale) + offset)
        for p0, p1 in prism.edges
    ]

    xy_chunks: list[np.ndarray] = []
    height_chunks: list[np.ndarray] = []
    for p0, p1 in true_edges:
        t = rng.uniform(0.0, 1.0, points_per_edge)
        heights = rng.uniform(0.5, max(true_height - 0.5, 0.6), points_per_edge)
        xy_chunks.append(p0 + t[:, None] * (p1 - p0))
        height_chunks.append(heights)

    scaled_footprint = np.array(
        [rotation @ (vertex * true_scale) + offset for vertex in prism.footprint_xy]
    )
    top_xy = _sample_inside_polygon(scaled_footprint, top_points, rng)
    xy_chunks.append(top_xy)
    height_chunks.append(np.full(len(top_xy), true_height))

    xy = np.concatenate(xy_chunks)
    heights = np.concatenate(height_chunks)
    if noise_mm > 0:
        xy = xy + rng.normal(0.0, noise_mm, xy.shape)
        heights = heights + rng.normal(0.0, noise_mm, heights.shape)

    a, b, c = board_plane
    z = a * xy[:, 0] + b * xy[:, 1] + c - heights
    return np.column_stack([xy, z])


def _box_ascii_stl_text(
    side_mm: float = 30.0, height_mm: float = 60.0, name: str = "box"
) -> str:
    """A minimal but valid ASCII STL for an axis-aligned box -- facet normals are not checked
    by ``read_ascii_stl``/``prism_from_cad``, so they are left zeroed."""
    s, h = side_mm, height_mm
    base = [(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0)]
    top = [(0, 0, h), (s, 0, h), (s, s, h), (0, s, h)]
    triangles = [
        (base[0], base[1], base[2]),
        (base[0], base[2], base[3]),
        (top[0], top[2], top[1]),
        (top[0], top[3], top[2]),
    ]
    lines = [f"solid {name}"]
    for v0, v1, v2 in triangles:
        lines.append("  facet normal 0.0 0.0 0.0")
        lines.append("    outer loop")
        for vertex in (v0, v1, v2):
            lines.append(
                f"      vertex {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}"
            )
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    return "\n".join(lines) + "\n"


# --- read_ascii_stl -------------------------------------------------------------------------


def test_read_ascii_stl_parses_vertices(tmp_path: Path) -> None:
    stl_path = tmp_path / "box.stl"
    stl_path.write_text(_box_ascii_stl_text(), encoding="ascii")

    vertices = read_ascii_stl(stl_path)

    assert vertices.shape[1] == 3
    assert vertices.shape[0] == 12  # 4 triangles x 3 vertices
    assert np.isclose(vertices[:, 2].max(), 60.0)


def test_read_ascii_stl_rejects_binary(tmp_path: Path) -> None:
    stl_path = tmp_path / "binary.stl"
    stl_path.write_bytes(b"\x00" * 84 + b"\xff\xfe\xfd" * 4)

    with pytest.raises(ValueError, match="binary STL"):
        read_ascii_stl(stl_path)


def test_read_ascii_stl_rejects_missing_vertex_lines(tmp_path: Path) -> None:
    stl_path = tmp_path / "empty.stl"
    stl_path.write_text("solid empty\nendsolid empty\n", encoding="ascii")

    with pytest.raises(ValueError, match="no 'vertex' lines"):
        read_ascii_stl(stl_path)


def test_read_ascii_stl_rejects_non_solid_header(tmp_path: Path) -> None:
    stl_path = tmp_path / "not_stl.stl"
    stl_path.write_text("vertex 0 0 0\n", encoding="ascii")

    with pytest.raises(ValueError, match="does not start with 'solid'"):
        read_ascii_stl(stl_path)


# --- prism_from_cad --------------------------------------------------------------------------


def test_prism_from_cad_on_committed_s04_stl() -> None:
    # Ties the committed golden-set CAD file directly to this story's tests, the way the
    # acceptance criteria ask for a repeatable measurement that "can be rerun after any fix."
    stl_path = Path("fixtures/reference_cad/s04_triangular_prism_60-26-65x30mm.stl")
    prism = prism_from_cad(read_ascii_stl(stl_path))

    assert prism.height_mm == pytest.approx(30.0, abs=1e-3)
    lengths = sorted(np.linalg.norm(p1 - p0) for p0, p1 in prism.edges)
    assert lengths == pytest.approx([26.00047, 60.0, 65.00019], abs=1e-3)


def test_prism_from_cad_box(tmp_path: Path) -> None:
    stl_path = tmp_path / "box.stl"
    stl_path.write_text(
        _box_ascii_stl_text(side_mm=30.0, height_mm=60.0), encoding="ascii"
    )

    prism = prism_from_cad(read_ascii_stl(stl_path))

    assert prism.height_mm == pytest.approx(60.0)
    lengths = sorted(np.linalg.norm(p1 - p0) for p0, p1 in prism.edges)
    assert lengths == pytest.approx([30.0, 30.0, 30.0, 30.0])


def test_prism_from_cad_rejects_non_prism_z_levels() -> None:
    # Three distinct z levels -- not a simple base/top extrusion.
    vertices = np.array(
        [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 15], [0, 0, 30]], dtype=np.float64
    )
    with pytest.raises(ValueError, match="2 distinct z levels"):
        prism_from_cad(vertices)


def test_prism_from_cad_rejects_mismatched_footprints() -> None:
    vertices = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 20],
            [
                11,
                0,
                20,
            ],  # top footprint shifted from the base -- not a simple extrusion
            [0, 11, 20],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="Base and top footprints differ"):
        prism_from_cad(vertices)


def test_prism_from_cad_rejects_non_convex_footprint() -> None:
    # A square with one extra vertex pushed into the interior -- not on the convex hull.
    base = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 5, 0]]
    top = [[0, 0, 20], [10, 0, 20], [10, 10, 20], [0, 10, 20], [5, 5, 20]]
    vertices = np.array(base + top, dtype=np.float64)
    with pytest.raises(ValueError, match="convex hull"):
        prism_from_cad(vertices)


# --- fit_board_plane / compute_height_above_board ----------------------------------------------


def _board_region_points(
    board_plane: tuple[float, float, float],
    n: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(1)
    a, b, c = board_plane
    xy = rng.uniform([0.0, 0.0], [160.0, 120.0], size=(n, 2))
    z = a * xy[:, 0] + b * xy[:, 1] + c
    return np.column_stack([xy, z])


def test_fit_board_plane_recovers_offset_and_tilt() -> None:
    true_plane = (
        0.01,
        -0.02,
        3.3,
    )  # ~0.6-1.1 deg tilt, 3.3 mm offset -- within R-17's range
    points = _board_region_points(true_plane)

    fitted = fit_board_plane(points)

    assert fitted == pytest.approx(true_plane, abs=5e-3)


def test_fit_board_plane_robust_to_part_contamination() -> None:
    true_plane = (0.0, 0.0, 2.0)
    board_points = _board_region_points(true_plane, n=3000)
    # A part sitting on the board: a cluster of points several mm above the true plane, inside
    # the board rectangle, contaminating the candidate region fit_board_plane sees.
    rng = np.random.default_rng(2)
    part_points = _synthesize_prism_points(
        _box_prism(30.0, 60.0), offset_xy=(80.0, 60.0), board_plane=true_plane, rng=rng
    )
    # Only the near-board slice of the part (fit_board_plane's own |z|<15 window) matters here.
    near_board = np.abs(part_points[:, 2]) < 15
    points = np.concatenate([board_points, part_points[near_board]])

    fitted = fit_board_plane(points)

    assert fitted == pytest.approx(true_plane, abs=0.2)


def test_fit_board_plane_rejects_too_few_points() -> None:
    points = _board_region_points((0.0, 0.0, 0.0), n=10)
    with pytest.raises(ValueError, match="at least"):
        fit_board_plane(points, min_points=200)


def test_compute_height_above_board_matches_plane() -> None:
    plane = (0.01, -0.02, 3.0)
    a, b, c = plane
    points = np.array(
        [
            [10.0, 10.0, a * 10.0 + b * 10.0 + c],  # on the board -- height 0
            [10.0, 10.0, a * 10.0 + b * 10.0 + c - 5.0],  # 5 mm above the board
        ]
    )

    height = compute_height_above_board(points, plane)

    assert height == pytest.approx([0.0, 5.0], abs=1e-9)


# --- measure_prism -----------------------------------------------------------------------------


def test_measure_prism_recovers_box_dimensions() -> None:
    prism = _box_prism(side_mm=30.0, height_mm=60.0)
    points = _synthesize_prism_points(
        prism, noise_mm=0.3, rng=np.random.default_rng(10)
    )

    result = measure_prism(points, prism, board_plane=(0.0, 0.0, 0.0))

    assert result.height_mm == pytest.approx(60.0, abs=1.5)
    assert len(result.edges) == 4
    for edge in result.edges:
        assert edge.measured_length_mm == pytest.approx(30.0, abs=2.0)
        assert edge.ratio == pytest.approx(1.0, abs=0.07)


def test_measure_prism_recovers_triangular_prism_dimensions() -> None:
    prism = _triangle_prism(legs=(60.0, 26.0), height_mm=30.0)
    points = _synthesize_prism_points(
        prism, noise_mm=0.3, rng=np.random.default_rng(11)
    )

    result = measure_prism(points, prism, board_plane=(0.0, 0.0, 0.0))

    assert result.height_mm == pytest.approx(30.0, abs=1.5)
    cad_lengths = sorted(edge.cad_length_mm for edge in result.edges)
    assert cad_lengths == pytest.approx([26.0, 60.0, 65.391], abs=0.1)
    for edge in result.edges:
        assert edge.ratio == pytest.approx(1.0, abs=0.1)


def test_measure_prism_handles_rotation_offset_and_board_tilt() -> None:
    prism = _box_prism(side_mm=30.0, height_mm=60.0)
    board_plane = (0.01, -0.015, 2.5)
    points = _synthesize_prism_points(
        prism,
        rotation_deg=37.0,
        offset_xy=(70.0, 55.0),
        board_plane=board_plane,
        noise_mm=0.3,
        rng=np.random.default_rng(12),
    )

    result = measure_prism(points, prism, board_plane=board_plane)

    assert result.height_mm == pytest.approx(60.0, abs=1.5)
    for edge in result.edges:
        assert edge.ratio == pytest.approx(1.0, abs=0.08)


def test_measure_prism_reports_shrunk_dimensions_when_part_is_smaller() -> None:
    # The acceptance criteria's "must fail when the synthetic part is scaled" case: a part that
    # is genuinely 90% size must measure as smaller, not be silently reported at full size.
    prism = _box_prism(side_mm=30.0, height_mm=60.0)
    points = _synthesize_prism_points(
        prism, true_scale=0.9, noise_mm=0.3, rng=np.random.default_rng(13)
    )

    result = measure_prism(points, prism, board_plane=(0.0, 0.0, 0.0))

    assert result.height_ratio == pytest.approx(0.9, abs=0.07)
    for edge in result.edges:
        assert edge.ratio == pytest.approx(0.9, abs=0.08)
        assert edge.ratio < 0.98  # must not be reported as full-size


def test_measure_prism_ignores_stray_point_cluster() -> None:
    prism = _box_prism(side_mm=30.0, height_mm=60.0)
    rng = np.random.default_rng(14)
    points = _synthesize_prism_points(prism, noise_mm=0.3, rng=rng)
    # A spatially separate stray cluster -- not connected to the part's own blob.
    stray = rng.uniform([0.0, 0.0, -30.0], [160.0, 20.0, -2.0], size=(500, 3))
    points_with_stray = np.concatenate([points, stray])

    result = measure_prism(points_with_stray, prism, board_plane=(0.0, 0.0, 0.0))

    assert result.height_mm == pytest.approx(60.0, abs=1.5)
    for edge in result.edges:
        assert edge.ratio == pytest.approx(1.0, abs=0.08)


def test_measure_prism_rejects_too_few_candidate_points() -> None:
    prism = _box_prism(30.0, 60.0)
    points = np.array([[80.0, 60.0, -30.0], [80.0, 60.0, -31.0]])
    with pytest.raises(ValueError, match="Too few candidate points"):
        measure_prism(points, prism, board_plane=(0.0, 0.0, 0.0))


# --- measure_board_scale -----------------------------------------------------------------------


def _make_synthetic_board_bundle(
    alignment_scale: float = 1.0,
    focal_ratio: float = 1.0,
    frame_count: int = 3,
    corner_ids: range = range(12),
) -> _RunBundle:
    """Synthetic board run with exact geometry, trusted poses and explicit pixel mapping."""
    nominal = make_board().getChessboardCorners()
    corner_ids = list(corner_ids)
    focal, principal_point, image_size, distance = 1000.0, 150.0, 400, 1000.0
    vggt_intrinsic = np.array(
        [[focal, 0.0, principal_point], [0.0, focal, principal_point], [0.0, 0.0, 1.0]]
    )
    board_focal = focal * focal_ratio
    board_intrinsic_single = np.array(
        [
            [board_focal, 0.0, principal_point],
            [0.0, board_focal, principal_point],
            [0.0, 0.0, 1.0],
        ]
    )

    frame_names = [f"frame_{index}" for index in range(frame_count)]
    offsets = [(0.0, 0.0), (35.0, 10.0), (10.0, 45.0)]
    if frame_count > len(offsets):
        offsets.extend(
            [(20.0 * index, 15.0 * index) for index in range(len(offsets), frame_count)]
        )
    offsets = offsets[:frame_count]

    poses_extrinsic = np.zeros((frame_count, 3, 4), dtype=np.float64)
    poses_extrinsic[:, :, :3] = np.eye(3)
    for frame_index, (tx, ty) in enumerate(offsets):
        poses_extrinsic[frame_index, :, 3] = [tx, ty, distance]
    poses_intrinsic = np.tile(vggt_intrinsic[None, :, :], (frame_count, 1, 1))
    depth = np.full((frame_count, image_size, image_size), distance)
    depth_conf = np.ones((frame_count, image_size, image_size))
    board_image_size_hw = np.tile(
        np.array([[image_size, image_size]]), (frame_count, 1)
    )
    board_intrinsic = np.tile(board_intrinsic_single[None, :, :], (frame_count, 1, 1))
    board_extrinsic = poses_extrinsic.copy()
    board_pose_found = np.ones(frame_count, dtype=bool)
    preprocess_geometry = [
        {
            "frame_name": name,
            "original_size_hw": [image_size, image_size],
            "resized_size_hw": [image_size, image_size],
            "crop_top_px": 0,
            "crop_left_px": 0,
            "processed_size_hw": [image_size, image_size],
            "scale_x": 1.0,
            "scale_y": 1.0,
            "supported_for_metric_alignment": True,
        }
        for name in frame_names
    ]

    corner_frame_index: list[int] = []
    corner_id: list[int] = []
    corner_px: list[list[float]] = []
    for frame_index, (tx, ty) in enumerate(offsets):
        for cid in corner_ids:
            x, y, _z = nominal[cid]
            u = focal * (x + tx) / distance + principal_point
            v = focal * (y + ty) / distance + principal_point
            corner_frame_index.append(frame_index)
            corner_id.append(cid)
            corner_px.append([u, v])

    return _RunBundle(
        frame_names=frame_names,
        poses_extrinsic=poses_extrinsic,
        poses_intrinsic=poses_intrinsic,
        depth=depth,
        depth_conf=depth_conf,
        processed_size_hw=(image_size, image_size),
        preprocess_geometry=preprocess_geometry,
        alignment_scale=alignment_scale,
        alignment_rotation=np.eye(3),
        alignment_translation_mm=np.zeros(3),
        board_frame_names=frame_names,
        board_image_size_hw=board_image_size_hw,
        board_intrinsic=board_intrinsic,
        board_extrinsic=board_extrinsic,
        board_pose_found=board_pose_found,
        corner_frame_index=np.array(corner_frame_index),
        corner_id=np.array(corner_id),
        corner_px=np.array(corner_px),
    )


def test_measure_board_scale_recovers_unit_scale_with_synthetic_cameras() -> None:
    bundle = _make_synthetic_board_bundle(alignment_scale=1.0)

    result = measure_board_scale(bundle)

    assert result.scale == pytest.approx(1.0, abs=1e-3)
    assert result.observation_count == 36
    assert result.frame_count == 3
    assert result.rms_residual_mm == pytest.approx(0.0, abs=1e-6)


def test_measure_board_scale_detects_an_induced_scale_error() -> None:
    # R-04's alignment is wrong by a known factor; measure_board_scale must report that factor,
    # not silently agree with it -- the same "must fail when scaled" spirit as measure_prism's.
    bundle = _make_synthetic_board_bundle(alignment_scale=0.9)

    result = measure_board_scale(bundle)

    assert result.scale == pytest.approx(0.9, abs=1e-3)
    assert result.per_frame_scale_median == pytest.approx(0.9, abs=1e-3)


def test_legacy_camera_centre_diagnostic_remains_independent() -> None:
    bundle = _make_synthetic_board_bundle()
    board_extrinsic = bundle.board_extrinsic.copy()
    # Scale board-pose camera centres to 90% of reconstruction camera centres, reproducing the
    # old R-04 mechanism while leaving R-23's stored board-anchor transform untouched.
    centres = camera_centres(bundle.poses_extrinsic)
    wrong_centres = 0.9 * centres
    board_extrinsic[:, :, :3] = np.eye(3)
    board_extrinsic[:, :, 3] = -wrong_centres
    bundle = replace(bundle, board_extrinsic=board_extrinsic)

    current = measure_board_scale(bundle)
    legacy = measure_legacy_camera_centre_board_scale(bundle)

    assert current.scale == pytest.approx(1.0, abs=1e-3)
    assert legacy.scale == pytest.approx(0.9, abs=1e-3)


def test_measure_board_scale_reports_focal_ratio() -> None:
    bundle = _make_synthetic_board_bundle(focal_ratio=0.9)

    result = measure_board_scale(bundle)

    assert result.focal_ratio_board_over_vggt == pytest.approx(0.9, rel=1e-6)


def test_measure_board_scale_passes_each_frames_own_geometry_to_the_vggt_rescale(
    monkeypatch,
) -> None:
    # R-28 (#182): before this story, the focal-ratio diagnostic rescaled every frame's VGGT K
    # with the session-level processed_size_hw, ignoring per-frame crop. This confirms the real
    # wiring: resolve_vggt_frame_scale is called once per matched frame, with that frame's own
    # name and the bundle's recorded preprocess_geometry -- not bypassed.
    bundle = _make_synthetic_board_bundle(focal_ratio=0.9)
    calls: list[tuple] = []
    real_resolve = cad_size_check.resolve_vggt_frame_scale

    def recording_resolve(
        frame_name, preprocess_geometry, processed_size_hw, full_size_hw
    ):
        calls.append((frame_name, preprocess_geometry, processed_size_hw, full_size_hw))
        return real_resolve(
            frame_name, preprocess_geometry, processed_size_hw, full_size_hw
        )

    monkeypatch.setattr(cad_size_check, "resolve_vggt_frame_scale", recording_resolve)

    result = measure_board_scale(bundle)

    assert len(calls) == len(bundle.frame_names)
    assert {call[0] for call in calls} == set(bundle.frame_names)
    for call in calls:
        assert call[1] is bundle.preprocess_geometry
    assert result.focal_ratio_board_over_vggt == pytest.approx(0.9, rel=1e-6)


def test_measure_board_scale_rejects_too_few_observations() -> None:
    bundle = _make_synthetic_board_bundle(frame_count=1, corner_ids=range(2))
    with pytest.raises(ValueError, match="usable board corner observations"):
        measure_board_scale(bundle)


# --- reconstruct_confidence_filtered_points -----------------------------------------------------


def test_reconstruct_confidence_filtered_points_survives_dropped_pixels() -> None:
    # Regression test (review finding on PR #126): a real capture commonly has a handful of
    # non-finite depth pixels near an edge or occlusion. R-01's own select_point_cloud drops
    # those before points.ply is ever written, so points.ply's row count -- and therefore its
    # row order relative to depth_conf's full (S,H,W) grid -- cannot be assumed to match depth
    # confidence by raw position. This exercises exactly that: depth with some NaNs, confirming
    # the count comes out right (every finite pixel kept, every non-finite pixel dropped) without
    # going anywhere near points.ply at all.
    bundle = _make_synthetic_board_bundle(frame_count=1, corner_ids=range(12))
    depth = bundle.depth.copy()
    total_pixels = depth.size
    rng = np.random.default_rng(0)
    bad_rows = rng.integers(0, depth.shape[1], size=37)
    bad_cols = rng.integers(0, depth.shape[2], size=37)
    depth[0, bad_rows, bad_cols] = np.nan
    dropped_pixel_count = len(set(zip(bad_rows.tolist(), bad_cols.tolist())))
    bundle = replace(bundle, depth=depth)

    points = reconstruct_confidence_filtered_points(bundle, min_confidence=0.5)

    assert len(points) == total_pixels - dropped_pixel_count
    assert np.isfinite(points).all()


def test_reconstruct_confidence_filtered_points_applies_the_threshold() -> None:
    bundle = _make_synthetic_board_bundle(frame_count=1, corner_ids=range(12))
    confidence = bundle.depth_conf.copy()
    confidence[0, : confidence.shape[1] // 2, :] = (
        0.2  # half the image below the threshold
    )
    bundle = replace(bundle, depth_conf=confidence)

    low = reconstruct_confidence_filtered_points(bundle, min_confidence=0.0)
    high = reconstruct_confidence_filtered_points(bundle, min_confidence=0.5)

    assert len(high) < len(low)
    assert len(high) == pytest.approx(len(low) / 2, rel=0.05)


# --- write_report / build_report / CLI, end to end on tmp_path fixtures ------------------------


def _write_synthetic_run(run_dir: Path, board_dir: Path, prism: PrismCAD) -> None:
    """Write a minimal but complete synthetic R-01/R-02/R-04 output set under tmp_path, so
    load_bundle/build_report/main can be exercised without touching the gitignored, local-only
    fixtures/data/ directory.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    board_dir.mkdir(parents=True, exist_ok=True)

    bundle = _make_synthetic_board_bundle(alignment_scale=1.0)
    frame_count = len(bundle.frame_names)
    height, width = bundle.processed_size_hw

    np.savez(
        run_dir / "poses.npz",
        frame_names=np.array(bundle.frame_names),
        extrinsic=bundle.poses_extrinsic.astype(np.float32),
        intrinsic=bundle.poses_intrinsic.astype(np.float32),
    )
    np.savez(
        run_dir / "depth.npz",
        depth=bundle.depth.astype(np.float32),
        depth_conf=bundle.depth_conf.astype(np.float32),
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "processed_image_size_hw": [int(height), int(width)],
                "preprocess_geometry": bundle.preprocess_geometry,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metric_alignment.json").write_text(
        json.dumps(
            {
                "scale": bundle.alignment_scale,
                "rotation": bundle.alignment_rotation.tolist(),
                "translation_mm": bundle.alignment_translation_mm.tolist(),
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        board_dir / "board_poses.npz",
        frame_names=np.array(bundle.board_frame_names),
        image_size_hw=bundle.board_image_size_hw,
        intrinsic=bundle.board_intrinsic.astype(np.float64),
        extrinsic=bundle.board_extrinsic.astype(np.float64),
        pose_found=bundle.board_pose_found,
        corner_frame_index=bundle.corner_frame_index,
        corner_id=bundle.corner_id,
        corner_px=bundle.corner_px,
    )

    # Already in board-frame mm, since alignment above is the identity transform (scale=1,
    # rotation=I, translation=0) -- points.ply's "R-01 arbitrary frame" coordinates are then
    # exactly the board-frame-mm coordinates measure_prism/fit_board_plane expect. A flat board
    # region is included too (at true z=0), since fit_board_plane -- called by build_report/main
    # on this file's points -- needs real board-surface signal, not just the part's own points.
    rng = np.random.default_rng(20)
    part_points = _synthesize_prism_points(
        prism, offset_xy=(80.0, 60.0), noise_mm=0.3, rng=rng
    )
    board_points = _board_region_points((0.0, 0.0, 0.0), n=4000, rng=rng)
    points = np.concatenate([part_points, board_points]).astype(np.float32)
    colors = np.full((len(points), 3), 128, dtype=np.uint8)
    write_ply(run_dir / "points.ply", points, colors)
    assert frame_count == 3  # sanity: matches _make_synthetic_board_bundle's default


def test_build_report_with_and_without_cad(tmp_path: Path) -> None:
    prism = _box_prism(30.0, 60.0)
    run_dir, board_dir = tmp_path / "run", tmp_path / "board"
    _write_synthetic_run(run_dir, board_dir, prism)
    stl_path = tmp_path / "box.stl"
    stl_path.write_text(_box_ascii_stl_text(30.0, 60.0), encoding="ascii")

    board_only = build_report(run_dir, board_dir)
    assert "prism" not in board_only
    assert board_only["board_scale"]["scale"] == pytest.approx(1.0, abs=1e-3)

    with_cad = build_report(run_dir, board_dir, cad_path=stl_path)
    assert "prism" in with_cad
    assert with_cad["prism"]["height_ratio"] == pytest.approx(1.0, abs=0.08)
    for edge in with_cad["prism"]["edges"]:
        assert edge["ratio"] == pytest.approx(1.0, abs=0.08)


def test_write_report_refuses_overwrite(tmp_path: Path) -> None:
    out_path = tmp_path / "report.json"
    write_report(out_path, {"a": 1})

    with pytest.raises(FileExistsError, match="already exists"):
        write_report(out_path, {"a": 2})


def test_cli_writes_report_and_then_refuses_overwrite(tmp_path: Path, capsys) -> None:
    prism = _box_prism(30.0, 60.0)
    run_dir, board_dir = tmp_path / "run", tmp_path / "board"
    _write_synthetic_run(run_dir, board_dir, prism)
    stl_path = tmp_path / "box.stl"
    stl_path.write_text(_box_ascii_stl_text(30.0, 60.0), encoding="ascii")
    out_path = tmp_path / "out" / "cad_size_check.json"

    exit_code = main(
        [str(run_dir), str(board_dir), "--cad", str(stl_path), "--out", str(out_path)]
    )
    assert exit_code == 0
    assert "PASS" in capsys.readouterr().out
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["prism"]["height_ratio"] == pytest.approx(1.0, abs=0.08)

    exit_code_again = main(
        [str(run_dir), str(board_dir), "--cad", str(stl_path), "--out", str(out_path)]
    )
    assert exit_code_again == 1
    assert "FAIL" in capsys.readouterr().err


def test_cli_fails_loudly_on_missing_run_dir(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            str(tmp_path / "missing_run"),
            str(tmp_path / "missing_board"),
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().err


def test_cli_fails_loudly_on_degenerate_vggt_intrinsic(tmp_path: Path, capsys) -> None:
    # Caught in review (dalwalyk): measure_board_scale calls rescale_vggt_intrinsic
    # unconditionally for every matched frame, and rescale_vggt_intrinsic can raise
    # CalibrationFailure (a RuntimeError subclass, not a ValueError) on a non-positive focal
    # length -- main()'s except clause didn't list it, so a degenerate VGGT intrinsic on even
    # one frame crashed with a raw traceback instead of the intended FAIL message. A negative
    # (not zero) focal length keeps the intrinsic invertible, so this isolates
    # rescale_vggt_intrinsic's own check rather than failing earlier for an unrelated reason.
    prism = _box_prism(30.0, 60.0)
    run_dir, board_dir = tmp_path / "run", tmp_path / "board"
    _write_synthetic_run(run_dir, board_dir, prism)

    poses_path = run_dir / "poses.npz"
    with np.load(poses_path) as poses:
        frame_names = poses["frame_names"]
        extrinsic = poses["extrinsic"]
        intrinsic = poses["intrinsic"].copy()
    intrinsic[0, 0, 0] *= -1.0
    np.savez(
        poses_path, frame_names=frame_names, extrinsic=extrinsic, intrinsic=intrinsic
    )

    exit_code = main(
        [str(run_dir), str(board_dir), "--out", str(tmp_path / "out.json")]
    )

    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().err


def test_build_argument_parser_defaults() -> None:
    args = build_argument_parser().parse_args(["run_dir", "board_dir"])
    assert args.cad is None
    assert args.min_confidence == 0.0
    assert args.out is None


def test_cad_size_check_module_is_importable() -> None:
    # Pins that the module stays importable without any GPU/PyTorch dependency, the same
    # guarantee vggt_runner's own split gives the rest of this codebase's CPU-only tests.
    assert hasattr(cad_size_check, "build_report")
