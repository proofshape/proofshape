"""R-07: fuse R-01's per-frame depth into a metric TSDF volume, then extract a mesh.

TSDF (truncated signed distance function) fusion is voxel-projection based: for every voxel in a
board-frame grid, project it into every frame's camera and compare its camera-space depth against
that frame's *measured* depth at the pixel it lands on. Positive (surface is farther than the
voxel) means free space; negative (surface is nearer) means occupied/behind the surface; the
zero-crossing, extracted by marching cubes, is the mesh.

That needs each frame's camera pose *in the board frame* -- R-04 only gives points in the board
frame, not camera poses. `recon.metric_alignment.apply_similarity_transform_to_extrinsic` derives
it; see that function's docstring for the derivation. Depth used alongside its output extrinsic
must be pre-scaled by R-04's `scale` (this module's `fuse_run` does that).

Masking: per-frame, per-pixel board-frame points are classified with
`recon.part_segmentation.segment_part` (R-06) so board/table/hand pixels never contribute to the
volume -- the same height/footprint predicate R-06 uses to filter a point cloud, applied here to
filter depth samples before fusion instead.

R-21: the height check is relative to the real board surface, fit by
`recon.part_segmentation.measure_board_surface` from this run's own valid-depth pixels (depth
that is finite and positive -- an invalid pixel unprojects to the camera centre, not a board
point, and would corrupt the fit). The voxel grid's top is set from the fitted plane too, so the
grid itself doesn't repeat R-06's old z=0 assumption and clip the part's base before fusion even
starts.

R-20/D-045: a pixel VGGT's own confidence is too low to trust is also excluded here, with
`recon.vggt_runner.confidence_keep_mask` -- the same predicate R-01 applies when building the
point cloud in `points.ply`, so a pixel left out of the cloud is left out of the fused surface
too, rather than the two silently disagreeing on which pixels to trust. `fuse_run` reads the
threshold from the run's own `run.json` by default, so the fused surface always matches the
cloud that was actually written for that run; passing `min_confidence` explicitly overrides it.

R-30/D-060: some frames' own board sits a few mm/degrees off the consensus plane -- VGGT pose
error R-23's one global alignment can't correct. `reanchor_frames_to_board` fits and corrects
each frame's own board-plane drift, before any of it reaches the height check or the volume
above, removing the artifact at its source rather than trying to classify around it downstream
the way three rounds of R-26 tried and D-060 found couldn't work.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recon.metric_alignment import (
    apply_similarity_transform,
    apply_similarity_transform_to_extrinsic,
    project_points_to_camera,
)
from recon.part_segmentation import (
    DEFAULT_BOARD_FIT_MAX_ITERATIONS,
    DEFAULT_BOARD_FIT_SIGMA_CLIP,
    DEFAULT_BOARD_SEARCH_BAND_MM,
    DEFAULT_FOOTPRINT_MM,
    DEFAULT_MAX_BOARD_TILT_DEG,
    DEFAULT_MIN_KEPT_POINTS,
    BoardSurface,
    fit_board_plane_robust,
    measure_board_surface,
    segment_part,
)
from recon.vggt_runner import (
    confidence_keep_mask,
    resolve_min_confidence,
    unproject_depth_to_world,
)

DEFAULT_VOXEL_SIZE_MM = 2.0

# TSDF truncation as a multiple of voxel size, a standard fusion heuristic (KinectFusion-style),
# not a measured figure -- overridable via fuse_run's truncation_mm.
DEFAULT_TRUNCATION_VOXEL_MULTIPLE = 4.0

# Default voxel-grid z extent above the board, covering "hand-sized parts" (AGENTS.md's stated
# scope). An assumption for the default grid's coverage, not a measured constant -- configurable.
DEFAULT_MAX_PART_HEIGHT_MM = 150.0

DEFAULT_POISSON_DEPTH = 9

# R-30: a single frame's own board-candidate count needed to trust a per-frame re-anchoring fit.
# Deliberately smaller than DEFAULT_MIN_BOARD_POINTS (which pools every frame) -- a bare
# assumption (D-033), not yet measured against the golden set's real per-frame counts.
DEFAULT_MIN_FRAME_BOARD_POINTS = 100


def build_voxel_grid(
    bounds_min_mm: np.ndarray, bounds_max_mm: np.ndarray, voxel_size_mm: float
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Build an axis-aligned grid of voxel centres (board-frame mm) between two corners.

    Returns (`centers` (N,3), `shape` (nx,ny,nz)) with `centers` flattened in the same 'ij' order
    a caller must use to reshape a flat per-voxel array back to `shape` before marching cubes.
    """
    bounds_min_mm = np.asarray(bounds_min_mm, dtype=np.float64)
    bounds_max_mm = np.asarray(bounds_max_mm, dtype=np.float64)
    if bounds_min_mm.shape != (3,) or bounds_max_mm.shape != (3,):
        raise ValueError(
            f"bounds_min_mm/bounds_max_mm must have shape (3,), got "
            f"{bounds_min_mm.shape} and {bounds_max_mm.shape}"
        )
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be a positive, finite number.")
    if not np.all(bounds_max_mm > bounds_min_mm):
        raise ValueError(
            "bounds_max_mm must be strictly greater than bounds_min_mm on every axis."
        )

    counts = np.maximum(
        1, np.round((bounds_max_mm - bounds_min_mm) / voxel_size_mm).astype(int)
    )
    axis_coords = [
        bounds_min_mm[axis] + (np.arange(counts[axis]) + 0.5) * voxel_size_mm
        for axis in range(3)
    ]
    grid_x, grid_y, grid_z = np.meshgrid(*axis_coords, indexing="ij")
    centers = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    shape = (len(axis_coords[0]), len(axis_coords[1]), len(axis_coords[2]))
    return centers, shape


@dataclass(frozen=True)
class FrameReanchorReport:
    """What R-30's re-anchoring measured and did for one frame, for the golden rerun's figures."""

    frame_index: int
    candidate_count: int
    tilt_before_deg: float
    offset_shift_mm: float


def _rotation_aligning_unit_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The minimal rotation matrix taking unit vector `a` onto unit vector `b`.

    Rodrigues' rotation formula from the rotation axis `a x b` and angle between them. R-30's
    board normals are always near-vertical (tilt bounded by `DEFAULT_MAX_BOARD_TILT_DEG`
    upstream), so `a` and `b` are never antiparallel in practice; that degenerate case (a 180
    degree rotation, whose axis this formula can't recover from a near-zero cross product) is
    rejected rather than guessed at.
    """
    cross = np.cross(a, b)
    sin_angle = float(np.linalg.norm(cross))
    cos_angle = float(np.dot(a, b))
    if sin_angle < 1e-12:
        if cos_angle > 0:
            return np.eye(3)
        raise ValueError(
            "frame and global board normals are antiparallel -- no minimal rotation exists; "
            "this frame's fitted plane is not a small drift from the consensus."
        )
    axis = cross / sin_angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + skew * sin_angle + skew @ skew * (1.0 - cos_angle)


def reanchor_frames_to_board(
    board_frame_points: np.ndarray,
    board_extrinsic: np.ndarray,
    valid_depth: np.ndarray,
    global_board_surface: BoardSurface,
    footprint_mm: tuple[float, float] = DEFAULT_FOOTPRINT_MM,
    search_band_mm: float = DEFAULT_BOARD_SEARCH_BAND_MM,
    sigma_clip: float = DEFAULT_BOARD_FIT_SIGMA_CLIP,
    max_fit_iterations: int = DEFAULT_BOARD_FIT_MAX_ITERATIONS,
    min_frame_board_points: int = DEFAULT_MIN_FRAME_BOARD_POINTS,
) -> tuple[np.ndarray, np.ndarray, list[FrameReanchorReport]]:
    """R-30: correct each frame's own board-plane drift before its points enter fusion.

    D-060 traced R-17's raised-board-layer artifact to per-frame VGGT pose error: some frames'
    own board sits a few mm/degrees off the consensus plane R-23's one global alignment can't
    fix. For each frame, this fits that frame's *own* board plane (the same robust fit
    `measure_board_surface` uses, via `fit_board_plane_robust`, restricted to this frame's own
    candidate points) and applies the rigid (rotation + translation) correction that brings it
    onto `global_board_surface` -- not a per-point vertical shift, which can't correct tilt.

    The correction is applied to both representations `fuse_run` needs downstream: the frame's
    own `board_frame_points` (so the global board-plane fit and `segment_part`'s masking see the
    corrected geometry), and its `board_extrinsic` (so `fuse_tsdf`'s re-projection of voxels into
    this frame's real, unchanged depth measurements lands in the corrected frame). Derivation:
    with `x' = R_s @ x + t_s` the corrected point for original point `x`, and
    `x_cam = R_board @ x + t_board` (the frame's existing camera-from-board pose, unchanged, since
    the physical depth measurement itself doesn't change), substituting `x = R_s.T @ (x' - t_s)`
    gives `x_cam = (R_board @ R_s.T) @ x' + (t_board - R_board @ R_s.T @ t_s)` -- the corrected
    extrinsic below.

    Returns `(corrected_board_frame_points, corrected_board_extrinsic, reports)` -- one report
    per frame, for the golden rerun to record (candidate count, tilt before correction, and the
    z-shift applied at that frame's own candidate centroid).

    Raises:
        ValueError: a frame has fewer than `min_frame_board_points` of its own valid, in-footprint,
            near-the-assumed-board-height candidate points -- too little of this frame's own board
            is visible to trust a per-frame fit, rather than silently skipping correction for it
            or guessing.
    """
    board_frame_points = np.asarray(board_frame_points, dtype=np.float64)
    board_extrinsic = np.asarray(board_extrinsic, dtype=np.float64)
    valid_depth = np.asarray(valid_depth, dtype=bool)
    if board_frame_points.ndim != 4 or board_frame_points.shape[3] != 3:
        raise ValueError(
            f"board_frame_points must have shape (S,H,W,3), got {board_frame_points.shape}"
        )
    frame_count, height, width, _ = board_frame_points.shape
    if board_extrinsic.shape != (frame_count, 3, 4):
        raise ValueError(
            f"board_extrinsic must be ({frame_count},3,4), got {board_extrinsic.shape}"
        )
    if valid_depth.shape != (frame_count, height, width):
        raise ValueError(
            f"valid_depth must be ({frame_count},{height},{width}), got {valid_depth.shape}"
        )
    footprint_width_mm, footprint_height_mm = footprint_mm

    normal_global = np.array(
        [-global_board_surface.slope_x, -global_board_surface.slope_y, 1.0]
    )
    normal_global = normal_global / np.linalg.norm(normal_global)

    corrected_points = board_frame_points.copy()
    corrected_extrinsic = board_extrinsic.copy()
    reports: list[FrameReanchorReport] = []

    for frame_index in range(frame_count):
        points_flat = board_frame_points[frame_index].reshape(-1, 3)
        x_mm, y_mm, z_mm = points_flat[:, 0], points_flat[:, 1], points_flat[:, 2]
        in_footprint = (
            (x_mm >= 0.0)
            & (x_mm <= footprint_width_mm)
            & (y_mm >= 0.0)
            & (y_mm <= footprint_height_mm)
        )
        candidates = (
            valid_depth[frame_index].reshape(-1)
            & in_footprint
            & (np.abs(z_mm) <= search_band_mm)
        )
        candidate_count = int(np.count_nonzero(candidates))
        if candidate_count < min_frame_board_points:
            raise ValueError(
                f"frame {frame_index}: only {candidate_count} of its own board candidate points "
                f"within {search_band_mm} mm of z=0 inside the footprint; need at least "
                f"{min_frame_board_points} to trust a per-frame re-anchoring fit for this frame, "
                "rather than skip or guess its correction."
            )

        cx_mm, cy_mm, cz_mm = x_mm[candidates], y_mm[candidates], z_mm[candidates]
        slope_x, slope_y, z_at_origin_mm, _residual_sigma_mm, _inlier_count = (
            fit_board_plane_robust(
                cx_mm,
                cy_mm,
                cz_mm,
                sigma_clip,
                max_fit_iterations,
                min_frame_board_points,
            )
        )
        tilt_before_deg = float(np.degrees(np.arctan(np.hypot(slope_x, slope_y))))

        normal_local = np.array([-slope_x, -slope_y, 1.0])
        normal_local = normal_local / np.linalg.norm(normal_local)
        rotation_correction = _rotation_aligning_unit_vectors(
            normal_local, normal_global
        )

        centroid_x, centroid_y = float(np.mean(cx_mm)), float(np.mean(cy_mm))
        centroid_z_local = slope_x * centroid_x + slope_y * centroid_y + z_at_origin_mm
        centroid_rotated = rotation_correction @ np.array(
            [centroid_x, centroid_y, centroid_z_local]
        )
        centroid_target_z = global_board_surface.plane_z_mm(
            np.array([centroid_rotated[0]]), np.array([centroid_rotated[1]])
        )[0]
        translation_correction = np.array(
            [0.0, 0.0, float(centroid_target_z - centroid_rotated[2])]
        )

        corrected_points[frame_index] = (
            points_flat @ rotation_correction.T + translation_correction
        ).reshape(height, width, 3)

        rotation_board = board_extrinsic[frame_index, :, :3]
        translation_board = board_extrinsic[frame_index, :, 3]
        rotation_board_corrected = rotation_board @ rotation_correction.T
        translation_board_corrected = (
            translation_board - rotation_board_corrected @ translation_correction
        )
        corrected_extrinsic[frame_index, :, :3] = rotation_board_corrected
        corrected_extrinsic[frame_index, :, 3] = translation_board_corrected

        reports.append(
            FrameReanchorReport(
                frame_index=frame_index,
                candidate_count=candidate_count,
                tilt_before_deg=tilt_before_deg,
                offset_shift_mm=float(translation_correction[2]),
            )
        )

    return corrected_points, corrected_extrinsic, reports


def fuse_tsdf(
    depth_mm: np.ndarray,
    extrinsic_board: np.ndarray,
    intrinsic: np.ndarray,
    voxel_centers: np.ndarray,
    truncation_mm: float,
    keep_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse per-frame board-frame-mm depth into a TSDF over `voxel_centers`.

    Args:
        depth_mm: (S,H,W) depth already scaled to board-frame millimetres (R-04's `scale`
            applied). Non-positive or non-finite values mean "no measurement" at that pixel.
        extrinsic_board: (S,3,4) camera-from-board-mm extrinsics
            (`apply_similarity_transform_to_extrinsic`'s output).
        intrinsic: (S,3,3), matching `depth_mm`'s resolution (R-01's own `poses.npz` intrinsic).
        voxel_centers: (N,3) board-frame-mm points to fuse into (`build_voxel_grid`'s output).
        truncation_mm: signed-distance truncation band. A voxel farther behind a measured surface
            than this, from a given frame, gets no update from that frame at all (occlusion is
            unknown, not "very occupied"); a voxel farther in front is clipped to +truncation_mm
            (still a solid "this is empty space" signal).
        keep_mask: optional (S,H,W) boolean, True where that pixel is part geometry (R-06's
            `segment_part`, applied per-frame). False pixels never contribute -- this is what
            keeps the board/table/hands out of the fused volume.

    Returns:
        (`tsdf` (N,), `weight` (N,)) -- `weight == 0` means never observed by any frame.
    """
    depth_mm = np.asarray(depth_mm, dtype=np.float64)
    if depth_mm.ndim != 3:
        raise ValueError(f"depth_mm must have shape (S,H,W), got {depth_mm.shape}")
    frame_count, height, width = depth_mm.shape
    extrinsic_board = np.asarray(extrinsic_board, dtype=np.float64)
    if extrinsic_board.shape != (frame_count, 3, 4):
        raise ValueError(
            f"extrinsic_board must be ({frame_count},3,4), got {extrinsic_board.shape}"
        )
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (frame_count, 3, 3):
        raise ValueError(
            f"intrinsic must be ({frame_count},3,3), got {intrinsic.shape}"
        )
    voxel_centers = np.asarray(voxel_centers, dtype=np.float64)
    if voxel_centers.ndim != 2 or voxel_centers.shape[1] != 3:
        raise ValueError(
            f"voxel_centers must have shape (N,3), got {voxel_centers.shape}"
        )
    if not np.isfinite(truncation_mm) or truncation_mm <= 0:
        raise ValueError("truncation_mm must be a positive, finite number.")
    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.shape != depth_mm.shape:
            raise ValueError(
                f"keep_mask must match depth_mm's shape {depth_mm.shape}, got {keep_mask.shape}"
            )

    voxel_count = voxel_centers.shape[0]
    tsdf = np.zeros(voxel_count, dtype=np.float64)
    weight = np.zeros(voxel_count, dtype=np.float64)

    for frame_index in range(frame_count):
        rotation = extrinsic_board[frame_index, :, :3]
        translation = extrinsic_board[frame_index, :, 3]
        u, v, z_cam = project_points_to_camera(
            voxel_centers, rotation, translation, intrinsic[frame_index]
        )

        in_front = z_cam > 1e-9
        u_index = np.round(u).astype(np.int64)
        v_index = np.round(v).astype(np.int64)

        in_bounds = (
            in_front
            & (u_index >= 0)
            & (u_index < width)
            & (v_index >= 0)
            & (v_index < height)
        )
        clamped_u = np.where(in_bounds, u_index, 0)
        clamped_v = np.where(in_bounds, v_index, 0)

        depth_sample = depth_mm[frame_index, clamped_v, clamped_u]
        has_depth = in_bounds & np.isfinite(depth_sample) & (depth_sample > 0.0)
        if keep_mask is not None:
            sample_kept = keep_mask[frame_index, clamped_v, clamped_u]
            has_depth = has_depth & sample_kept

        signed_distance = depth_sample - z_cam
        # Far behind the surface: this frame saw something else first (occlusion), so it says
        # nothing about this voxel -- excluded entirely, not just clipped.
        contributes = has_depth & (signed_distance > -truncation_mm)
        truncated_distance = np.clip(signed_distance, -truncation_mm, truncation_mm)

        new_weight = weight[contributes] + 1.0
        tsdf[contributes] = (
            tsdf[contributes] * weight[contributes] + truncated_distance[contributes]
        ) / new_weight
        weight[contributes] = new_weight

    return tsdf, weight


def extract_mesh_marching_cubes(
    tsdf: np.ndarray,
    weight: np.ndarray,
    shape: tuple[int, int, int],
    voxel_size_mm: float,
    origin_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the zero-crossing surface as a mesh, ignoring never-observed voxels.

    Voxels with `weight == 0` are masked out of the marching-cubes input entirely (skimage's
    `mask` parameter), so unobserved space can never fabricate a surface -- the same "unobserved
    geometry never asserts anything" principle AGENTS.md's hard rule states for the verdict logic,
    applied here at the meshing level.
    """
    from skimage.measure import marching_cubes

    tsdf_volume = np.asarray(tsdf, dtype=np.float64).reshape(shape)
    weight_volume = np.asarray(weight, dtype=np.float64).reshape(shape)
    observed_mask = weight_volume > 0.0
    if not observed_mask.any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    try:
        vertices, faces, _normals, _values = marching_cubes(
            tsdf_volume,
            level=0.0,
            spacing=(voxel_size_mm, voxel_size_mm, voxel_size_mm),
            mask=observed_mask,
        )
    except RuntimeError as exc:
        # skimage raises rather than returning empty arrays when no observed cell straddles the
        # zero level -- a legitimate outcome (too few/too sparse observations to close a
        # surface), not a bug, so it's treated the same as the "nothing observed" case above.
        if "No surface found" not in str(exc):
            raise
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    vertices_mm = (vertices + np.asarray(origin_mm, dtype=np.float64)).astype(
        np.float32
    )
    return vertices_mm, faces.astype(np.int32)


def _orient_poisson_normals(point_cloud, points_mm: np.ndarray) -> str:
    """Orient Poisson normals coherently and pin a decisive overall board-frame sign.

    Full-rank clouds use Open3D tangent-plane consistency. Exactly planar clouds use a PCA
    normal directly because Open3D 0.18 sends coplanar data through Qhull/Delaunay. The
    consistency algorithms still leave one global +/- choice. In ProofShape board coordinates
    cameras are above the board in -z, so when substantially more normals have a strong +z
    component than a strong -z component, flip the whole field. The 25% vote margin avoids
    arbitrarily flipping closed/symmetric shapes whose up/down evidence is balanced.
    """
    points_mm = np.asarray(points_mm, dtype=np.float64)
    centred = points_mm - points_mm.mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] <= 0.0:
        raise ValueError("Poisson input does not span a surface.")

    tolerance = max(1e-9, float(singular_values[0]) * 1e-10)
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < 2:
        raise ValueError(
            "Poisson input is rank-deficient (point/line-like); at least a surface is required."
        )

    if rank == 2:
        _u, _s, vh = np.linalg.svd(centred, full_matrices=False)
        plane_normal = np.asarray(vh[-1], dtype=np.float64)
        normal_length = float(np.linalg.norm(plane_normal))
        if not np.isfinite(normal_length) or normal_length <= 0.0:
            raise ValueError("Could not determine a finite planar Poisson normal.")
        plane_normal /= normal_length
        point_cloud.normals = type(point_cloud.normals)(
            np.tile(plane_normal, (len(points_mm), 1))
        )
        mode = "planar_pca"
    else:
        tangent_k = min(30, len(points_mm) - 1)
        point_cloud.orient_normals_consistent_tangent_plane(tangent_k)
        mode = "tangent_plane"

    normals = np.asarray(point_cloud.normals, dtype=np.float64)
    strong_down = int(np.count_nonzero(normals[:, 2] > 0.2))
    strong_up = int(np.count_nonzero(normals[:, 2] < -0.2))
    if strong_down > max(1, math.ceil(1.25 * strong_up)):
        point_cloud.normals = type(point_cloud.normals)(-normals)

    return mode


def extract_mesh_poisson(
    points_mm: np.ndarray, poisson_depth: int = DEFAULT_POISSON_DEPTH
) -> tuple[np.ndarray, np.ndarray]:
    """Poisson surface reconstruction from a segmented board-frame-mm point cloud (the fallback).

    Lazily imports Open3D. R-07 kept it out of requirements.txt -- a heavy, GUI-adjacent
    dependency not needed for the primary marching-cubes path. C-08 has since pinned it there for
    its raycasting (D-047), but the import stays lazy so an environment without Open3D still gets
    this loud, specific error here rather than a failure at module import. Raises loudly, not
    silently, if it's unavailable.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "extract_mesh_poisson needs Open3D installed. It is pinned in requirements.txt "
            "(D-047), so this environment is missing it: pip install -r requirements.txt"
        ) from exc

    points_mm = np.asarray(points_mm, dtype=np.float64)
    if points_mm.ndim != 2 or points_mm.shape[1] != 3:
        raise ValueError(f"points_mm must have shape (N,3), got {points_mm.shape}")
    if len(points_mm) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_mm)
    point_cloud.estimate_normals()

    # R-27: Poisson requires a coherent normal field. Full-rank clouds use Open3D's
    # tangent-plane orientation; planar clouds use a deterministic PCA normal instead because
    # Open3D 0.18's tangent-plane path invokes Qhull/Delaunay and rejects coplanar inputs.
    _orient_poisson_normals(point_cloud, points_mm)

    mesh, _densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=poisson_depth
    )
    vertices_mm = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    return vertices_mm, faces


def write_mesh_ply(path: Path, vertices_mm: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary little-endian PLY with float32 xyz vertices and triangle faces."""
    vertices_mm = np.asarray(vertices_mm, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices_mm.ndim != 2 or vertices_mm.shape[1] != 3:
        raise ValueError(f"vertices_mm must have shape (N,3), got {vertices_mm.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3) (triangles), got {faces.shape}")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices_mm.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {faces.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    face_records = np.empty(faces.shape[0], dtype=face_dtype)
    face_records["count"] = 3
    face_records["indices"] = faces

    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices_mm.tobytes())
        handle.write(face_records.tobytes())


def read_mesh_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a mesh written by write_mesh_ply. Only understands that one fixed, triangles-only
    layout -- not a general PLY parser -- and raises rather than guessing on a mismatch.
    """
    path = Path(path)
    raw = path.read_bytes()
    header, separator, payload = raw.partition(b"end_header\n")
    if not separator:
        raise ValueError(
            f"{path} has no 'end_header' line; not a PLY this reader understands"
        )
    header_text = header.decode("ascii", errors="replace")

    vertex_count = None
    face_count = None
    for line in header_text.splitlines():
        if line.startswith("element vertex "):
            vertex_count = int(line.removeprefix("element vertex "))
        elif line.startswith("element face "):
            face_count = int(line.removeprefix("element face "))
    if vertex_count is None or face_count is None:
        raise ValueError(f"{path} header is missing 'element vertex' or 'element face'")

    vertex_bytes = vertex_count * 3 * 4
    vertices = np.frombuffer(payload[:vertex_bytes], dtype="<f4").reshape(
        vertex_count, 3
    )

    face_dtype = np.dtype([("count", "u1"), ("indices", "<i4", (3,))])
    face_payload = payload[
        vertex_bytes : vertex_bytes + face_count * face_dtype.itemsize
    ]
    if len(face_payload) != face_count * face_dtype.itemsize:
        raise ValueError(
            f"{path} face payload is truncated relative to its declared face count"
        )
    face_records = np.frombuffer(face_payload, dtype=face_dtype)
    if face_count and not np.all(face_records["count"] == 3):
        raise ValueError(
            f"{path} has a non-triangle face; this reader only understands triangles"
        )
    faces = face_records["indices"].astype(np.int32)

    return vertices.astype(np.float32), faces


def write_fusion_report(
    path: Path,
    reconstruction_run_dir: Path,
    alignment_json_path: Path,
    method: str,
    voxel_size_mm: float,
    truncation_mm: float,
    bounds_min_mm: np.ndarray,
    bounds_max_mm: np.ndarray,
    board_surface: BoardSurface,
    frame_count: int,
    observed_voxel_count: int,
    total_voxel_count: int,
    vertex_count: int,
    face_count: int,
    min_confidence: float,
    min_confidence_source: str,
    confidence_dropped_pixel_count: int,
    reanchor_reports: list[FrameReanchorReport],
    board_surface_before_reanchoring: BoardSurface,
) -> dict[str, Any]:
    report = {
        "story": ["R-07", "R-20", "R-21", "R-30"],
        "source_run_dir": str(reconstruction_run_dir),
        "source_alignment_json": str(alignment_json_path),
        "units": "mm",
        "frame": "R-02 ChArUco board frame",
        "method": method,
        "voxel_size_mm": voxel_size_mm,
        "truncation_mm": truncation_mm,
        "bounds_min_mm": list(np.asarray(bounds_min_mm, dtype=float)),
        "bounds_max_mm": list(np.asarray(bounds_max_mm, dtype=float)),
        "board_surface": board_surface.as_report(),
        "frame_count": frame_count,
        "observed_voxel_count": observed_voxel_count,
        "total_voxel_count": total_voxel_count,
        "vertex_count": vertex_count,
        "face_count": face_count,
        # R-20/D-045: which pixels were trusted, and where that threshold came from -- "run.json"
        # means it was read from the source run's own recorded min_confidence (the default, so
        # the fused surface matches points.ply by construction); "override" means fuse_run's
        # caller passed one explicitly.
        "min_confidence": min_confidence,
        "min_confidence_source": min_confidence_source,
        "confidence_dropped_pixel_count": confidence_dropped_pixel_count,
        # R-30: the board surface measured before any per-frame correction (also the target
        # every frame was corrected onto), the one measured after (same plane by construction,
        # but its tighter residual_sigma_mm is the actual before/after diagnostic), and each
        # frame's own pre-correction tilt/candidate count/applied shift -- what the golden
        # rerun's acceptance criteria ask to be recorded, not just asserted to "look better".
        "board_surface_before_reanchoring": board_surface_before_reanchoring.as_report(),
        "reanchor_reports": [
            {
                "frame_index": r.frame_index,
                "candidate_count": r.candidate_count,
                "tilt_before_deg": r.tilt_before_deg,
                "offset_shift_mm": r.offset_shift_mm,
            }
            for r in reanchor_reports
        ],
    }
    path = Path(path)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def fuse_run(
    reconstruction_run_dir: Path,
    alignment_json_path: Path,
    out_dir: Path,
    voxel_size_mm: float = DEFAULT_VOXEL_SIZE_MM,
    truncation_mm: float | None = None,
    method: str = "marching_cubes",
    height_threshold_mm: float | None = None,
    footprint_mm: tuple[float, float] = DEFAULT_FOOTPRINT_MM,
    max_part_height_mm: float = DEFAULT_MAX_PART_HEIGHT_MM,
    max_board_tilt_deg: float = DEFAULT_MAX_BOARD_TILT_DEG,
    min_confidence: float | None = None,
    min_kept_points: int = DEFAULT_MIN_KEPT_POINTS,
) -> dict[str, Any]:
    """Load R-01 depth/poses and an R-04 alignment, fuse, extract a mesh, and write the outputs.

    `method="marching_cubes"` (the primary path) fuses a TSDF volume and runs marching cubes.
    `method="poisson"` (the fallback) skips the volume entirely and runs Poisson reconstruction
    directly on the segmented, board-frame-mm point cloud -- Open3D is required for that path
    only (see `extract_mesh_poisson`).

    R-21: the board surface is fit from this run's own valid-depth pixels before segmenting, so
    the height cut (and the voxel grid's z top, below) follow the real board, not z=0.
    `height_threshold_mm` still overrides the measured value when given, same as `segment_run`.

    `min_confidence=None` (the default) reads the threshold the source run itself used
    (`run.json`'s own `min_confidence`), so the fused surface agrees with `points.ply` by
    construction (R-20). A run whose `run.json` is missing, or predates this field, is a
    genuine ambiguity -- falling back to 0.0 would silently fuse pixels the cloud itself
    excluded -- so it is an explicit error rather than a silent default. Pass `min_confidence`
    explicitly to override it (e.g. to reproduce a pre-R-20 fusion for comparison).
    """
    reconstruction_run_dir = Path(reconstruction_run_dir)
    alignment_json_path = Path(alignment_json_path)
    out_dir = Path(out_dir)
    if method not in ("marching_cubes", "poisson"):
        raise ValueError(
            f"method must be 'marching_cubes' or 'poisson', got {method!r}"
        )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already has files in it. Pick a new --out-dir or delete it deliberately; "
            "overwriting silently would let outputs from two runs get mixed."
        )

    # R-24 moved this rule into a shared helper so R-09's provenance resolves its threshold the
    # identical way -- see resolve_min_confidence's docstring.
    resolved_min_confidence, min_confidence_source = resolve_min_confidence(
        reconstruction_run_dir, min_confidence
    )

    with np.load(reconstruction_run_dir / "poses.npz") as poses:
        extrinsic = poses["extrinsic"].astype(np.float64)
        intrinsic = poses["intrinsic"].astype(np.float64)
    with np.load(reconstruction_run_dir / "depth.npz") as depth_data:
        depth = depth_data["depth"].astype(np.float64)
        depth_conf = depth_data["depth_conf"].astype(np.float64)

    alignment_json = json.loads(alignment_json_path.read_text(encoding="utf-8"))
    scale = float(alignment_json["scale"])
    rotation = np.asarray(alignment_json["rotation"], dtype=np.float64)
    translation = np.asarray(alignment_json["translation_mm"], dtype=np.float64)

    board_extrinsic = apply_similarity_transform_to_extrinsic(
        extrinsic, scale, rotation, translation
    )
    depth_mm = scale * depth

    frame_count, height, width = depth.shape
    original_world_points = unproject_depth_to_world(depth, extrinsic, intrinsic)
    board_frame_points = apply_similarity_transform(
        original_world_points.reshape(-1, 3), scale, rotation, translation
    ).reshape(frame_count, height, width, 3)

    # Only pixels with a real depth measurement feed the board-plane fit -- an invalid pixel
    # (depth <= 0 or non-finite) unprojects to the camera centre, not a board point.
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    flat_valid_depth = valid_depth.reshape(-1)
    board_surface_before_reanchoring = measure_board_surface(
        board_frame_points.reshape(-1, 3)[flat_valid_depth],
        footprint_mm,
        max_tilt_deg=max_board_tilt_deg,
        height_threshold_mm=height_threshold_mm,
    )

    # R-30: correct each frame's own board-plane drift (D-060) before any of it feeds
    # segmentation or fusion -- see reanchor_frames_to_board's docstring for the derivation.
    board_frame_points, board_extrinsic, reanchor_reports = reanchor_frames_to_board(
        board_frame_points,
        board_extrinsic,
        valid_depth,
        board_surface_before_reanchoring,
        footprint_mm,
    )

    # Re-fit after correction: the target plane is the same by construction, but this is the
    # fit that now actually describes the (much tighter) corrected cloud, and is what
    # segmentation and the voxel grid's bounds use from here on.
    flat_points = board_frame_points.reshape(-1, 3)
    board_surface = measure_board_surface(
        flat_points[flat_valid_depth],
        footprint_mm,
        max_tilt_deg=max_board_tilt_deg,
        height_threshold_mm=height_threshold_mm,
    )
    # segment_part itself requires every point finite; an invalid pixel's unprojected point can
    # still be non-finite (e.g. NaN depth), so it's excluded here too rather than passed through
    # and never classified as part geometry -- fuse_tsdf already drops invalid-depth pixels on
    # its own, independently of this mask, so their keep value doesn't otherwise matter.
    segmented_flat = np.zeros(flat_points.shape[0], dtype=bool)
    segmented_flat[flat_valid_depth] = segment_part(
        flat_points[flat_valid_depth],
        board_surface,
        height_threshold_mm,
        footprint_mm,
        min_kept_points=min_kept_points,
    )
    segmented_mask = segmented_flat.reshape(frame_count, height, width)

    # R-20/D-045: a pixel whose own confidence is too low to trust is excluded here too, with
    # the same predicate R-01's point cloud uses (confidence_keep_mask's own isfinite check
    # also drops a NaN/inf confidence pixel regardless of threshold), so a pixel left out of
    # points.ply is left out of the fused surface too -- computed unconditionally, even at
    # threshold 0.0, rather than skipping it as a "no-op": a once-attempted shortcut there
    # silently let a NaN/inf-confidence pixel through, which this predicate would not have.
    confidence_mask = confidence_keep_mask(depth_conf, resolved_min_confidence)
    keep_mask = segmented_mask & confidence_mask
    confidence_dropped_pixel_count = int(np.count_nonzero(~confidence_mask))

    # The grid follows the fitted (possibly tilted) plane, not z=0 -- otherwise the grid itself
    # would cut off the part's base before fusion even runs. Its top is the plane's highest
    # footprint corner (the deepest-into-the-table point of the real board); its bottom gives
    # every corner, even the shallowest, the full max_part_height_mm of clearance above it.
    footprint_corners_x = np.array([0.0, footprint_mm[0], 0.0, footprint_mm[0]])
    footprint_corners_y = np.array([0.0, 0.0, footprint_mm[1], footprint_mm[1]])
    corner_z_mm = board_surface.plane_z_mm(footprint_corners_x, footprint_corners_y)
    grid_top_mm = float(np.max(corner_z_mm))
    grid_bottom_mm = float(np.min(corner_z_mm)) - max_part_height_mm
    bounds_min_mm = np.array([0.0, 0.0, grid_bottom_mm])
    bounds_max_mm = np.array([footprint_mm[0], footprint_mm[1], grid_top_mm])
    resolved_truncation_mm = (
        truncation_mm or DEFAULT_TRUNCATION_VOXEL_MULTIPLE * voxel_size_mm
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    if method == "marching_cubes":
        voxel_centers, shape = build_voxel_grid(
            bounds_min_mm, bounds_max_mm, voxel_size_mm
        )
        tsdf, weight = fuse_tsdf(
            depth_mm,
            board_extrinsic,
            intrinsic,
            voxel_centers,
            resolved_truncation_mm,
            keep_mask,
        )
        origin_mm = bounds_min_mm + 0.5 * voxel_size_mm
        vertices_mm, faces = extract_mesh_marching_cubes(
            tsdf, weight, shape, voxel_size_mm, origin_mm
        )
        observed_voxel_count = int(np.count_nonzero(weight > 0.0))
        total_voxel_count = int(weight.shape[0])
    else:
        vertices_mm, faces = extract_mesh_poisson(board_frame_points[keep_mask])
        observed_voxel_count = int(np.count_nonzero(keep_mask))
        total_voxel_count = observed_voxel_count

    write_mesh_ply(out_dir / "mesh.ply", vertices_mm, faces)
    return write_fusion_report(
        out_dir / "tsdf_fusion.json",
        reconstruction_run_dir,
        alignment_json_path,
        method,
        voxel_size_mm,
        resolved_truncation_mm,
        bounds_min_mm,
        bounds_max_mm,
        board_surface,
        frame_count=frame_count,
        observed_voxel_count=observed_voxel_count,
        total_voxel_count=total_voxel_count,
        vertex_count=len(vertices_mm),
        face_count=len(faces),
        min_confidence=resolved_min_confidence,
        min_confidence_source=min_confidence_source,
        confidence_dropped_pixel_count=confidence_dropped_pixel_count,
        reanchor_reports=reanchor_reports,
        board_surface_before_reanchoring=board_surface_before_reanchoring,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse R-01 depth into a TSDF volume and extract a mesh (R-07/R-20/R-21)."
    )
    parser.add_argument("reconstruction_run_dir", type=Path)
    parser.add_argument(
        "alignment_json",
        type=Path,
        nargs="?",
        default=None,
        help="defaults to <reconstruction_run_dir>/metric_alignment.json (R-04's default output)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--voxel-size-mm", type=float, default=DEFAULT_VOXEL_SIZE_MM)
    parser.add_argument(
        "--method", choices=["marching_cubes", "poisson"], default="marching_cubes"
    )
    parser.add_argument(
        "--height-threshold-mm",
        type=float,
        default=None,
        help="override the measured board-layer threshold (mm above the fitted board surface)",
    )
    parser.add_argument(
        "--max-board-tilt-deg", type=float, default=DEFAULT_MAX_BOARD_TILT_DEG
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help=(
            "override the confidence threshold read from the run's own run.json "
            "(default: use whatever points.ply was already built with, D-045)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    alignment_json = args.alignment_json or (
        args.reconstruction_run_dir / "metric_alignment.json"
    )
    out_dir = args.out_dir or args.reconstruction_run_dir / "mesh"
    try:
        report = fuse_run(
            args.reconstruction_run_dir,
            alignment_json,
            out_dir,
            voxel_size_mm=args.voxel_size_mm,
            method=args.method,
            height_threshold_mm=args.height_threshold_mm,
            max_board_tilt_deg=args.max_board_tilt_deg,
            min_confidence=args.min_confidence,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"R-07 TSDF FUSION: FAIL - {exc}", file=sys.stderr)
        return 1

    print(
        f"R-07 TSDF FUSION: PASS - {report['method']}, {report['vertex_count']} vertices, "
        f"{report['face_count']} faces, min_confidence={report['min_confidence']} "
        f"({report['min_confidence_source']})"
    )
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
