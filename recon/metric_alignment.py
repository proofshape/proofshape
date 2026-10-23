"""R-04: align an arbitrary-scale reconstruction to the ChArUco board frame.

R-04 originally aligned R-01 and R-02 camera centres. R-23 replaces that scale-sensitive
anchor with the printed ChArUco board itself: R-02's detected corners are lifted through R-01
depth/cameras, then fit directly to the board's exact millimetre geometry. The written
scale/rotation/translation convention is unchanged for downstream consumers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from recon.board_pose import make_board

MIN_ALIGNMENT_FRAMES = 3
# Inherited from R-22's board-ruler check: enough observations to avoid fitting scale from a
# handful of corners while still allowing partial board visibility across trusted frames.
MIN_BOARD_CORNER_OBSERVATIONS = 30


def validate_known_size(
    measured_length_mm: float, expected_length_mm: float, tolerance_mm: float
) -> dict[str, float | bool]:
    """Compare an independently measured model length with a known reference."""
    values = (measured_length_mm, expected_length_mm, tolerance_mm)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Known-size values must be finite.")
    if expected_length_mm <= 0 or tolerance_mm < 0:
        raise ValueError(
            "Expected length must be positive and tolerance cannot be negative."
        )
    absolute_error_mm = abs(measured_length_mm - expected_length_mm)
    return {
        "measured_length_mm": measured_length_mm,
        "expected_length_mm": expected_length_mm,
        "absolute_error_mm": absolute_error_mm,
        "tolerance_mm": tolerance_mm,
        "within_tolerance": absolute_error_mm <= tolerance_mm,
    }


@dataclass(frozen=True)
class AlignmentResult:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    frame_names: list[str]
    reconstruction_centres: np.ndarray
    board_centres: np.ndarray
    residuals_mm: np.ndarray
    rms_residual_mm: float

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Map (N, 3) reconstruction-frame points into board-frame millimetres."""
        return apply_similarity_transform(
            points, self.scale, self.rotation, self.translation
        )


@dataclass(frozen=True)
class BoardAlignmentResult:
    """R-23 similarity solved from reconstructed board corners to their known mm geometry."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    observation_count: int
    frame_count: int
    residuals_mm: np.ndarray
    rms_residual_mm: float

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Map (N, 3) reconstruction-frame points into board-frame millimetres."""
        return apply_similarity_transform(
            points, self.scale, self.rotation, self.translation
        )


def apply_similarity_transform(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Map (N, 3) points through a similarity transform: scale * (points @ rotation.T) + translation.

    Pulled out of ``AlignmentResult.transform_points`` so a caller that only has the written
    ``metric_alignment.json`` (scale/rotation/translation_mm) -- not a rebuilt ``AlignmentResult``
    with its per-frame residuals -- can apply the same transform without duplicating the formula.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {points.shape}")
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    return scale * (points @ rotation.T) + translation


def apply_similarity_transform_to_extrinsic(
    extrinsic: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Map (S,3,4) camera-from-world extrinsics to camera-from-board-mm extrinsics.

    R-07 (TSDF fusion) needs each frame's camera pose *in the board frame* to project voxels
    into that frame's depth image -- R-04 only gives points in the board frame, not camera
    poses. Derivation: R-01's extrinsic is ``x_cam = R_o @ x_world + t_o`` (arbitrary units);
    R-04's transform is ``x_board_mm = scale * (rotation @ x_world) + translation``. Substituting
    one into the other and multiplying through by ``scale`` -- so camera-space coordinates come
    out in the same millimetres as ``x_board_mm``, not R-01's arbitrary units -- gives
    ``R_board = R_o @ rotation.T`` (a proper rotation: two orthonormal matrices compose cleanly)
    and ``t_board = scale * t_o - R_board @ translation``.

    Depth values used together with the returned extrinsic must be pre-scaled by ``scale``:
    multiplying camera-space coordinates by ``scale`` is exactly what turns R-01's arbitrary-unit
    depth into board-frame millimetres, since a pixel-to-ray direction (the intrinsics) is itself
    scale-invariant.
    """
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        raise ValueError(f"extrinsic must have shape (S,3,4), got {extrinsic.shape}")
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)

    original_rotation = extrinsic[:, :, :3]
    original_translation = extrinsic[:, :, 3]

    board_rotation = np.einsum("sij,kj->sik", original_rotation, rotation)
    board_translation = scale * original_translation - np.einsum(
        "sij,j->si", board_rotation, translation
    )
    return np.concatenate([board_rotation, board_translation[:, :, None]], axis=2)


def project_points_to_camera(
    points_board_frame_mm: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project (N,3) board-frame-mm points through one frame's camera: (u, v, z_cam).

    `rotation`/`translation` are one frame's (3,3)/(3,) slice of a camera-from-board-mm
    extrinsic (`apply_similarity_transform_to_extrinsic`'s output); `intrinsic` is that frame's
    (3,3) K. `z_cam` is camera-space depth along the optical axis -- positive means in front of
    the camera; a non-positive `z_cam` is a behind-the-camera point and `u`/`v` for it should be
    treated as meaningless by the caller, not divided-by-zero trusted.

    Shared by R-07's TSDF fusion (`tsdf_fusion.fuse_tsdf`) and R-09's per-vertex provenance
    (`provenance.compute_observations`), which both need the identical perspective-projection
    math applied to different point sets -- pulled into one place rather than duplicated a
    second time.
    """
    points_board_frame_mm = np.asarray(points_board_frame_mm, dtype=np.float64)
    if points_board_frame_mm.ndim != 2 or points_board_frame_mm.shape[1] != 3:
        raise ValueError(
            f"points_board_frame_mm must have shape (N,3), got {points_board_frame_mm.shape}"
        )
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)

    camera_points = points_board_frame_mm @ rotation.T + translation
    z_cam = camera_points[:, 2]

    in_front = z_cam > 1e-9
    pixel_homogeneous = camera_points @ intrinsic.T
    # Avoid a divide-by-zero warning for behind-camera points; their result is discarded by
    # `in_front` for callers that check it, regardless of what this division produces.
    safe_z = np.where(in_front, pixel_homogeneous[:, 2], 1.0)
    u = pixel_homogeneous[:, 0] / safe_z
    v = pixel_homogeneous[:, 1] / safe_z
    return u, v, z_cam


def camera_centres(extrinsic: np.ndarray) -> np.ndarray:
    """Convert camera-from-world [R|t] matrices to world-frame camera centres."""
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        raise ValueError(f"extrinsic must have shape (S,3,4), got {extrinsic.shape}")
    rotations = extrinsic[:, :, :3]
    translations = extrinsic[:, :, 3]
    return np.einsum("sij,sj->si", np.transpose(rotations, (0, 2, 1)), -translations)


def _validate_centres(source: np.ndarray, target: np.ndarray) -> None:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source and target centres must both have shape (N,3), got "
            f"{source.shape} and {target.shape}"
        )
    if len(source) < MIN_ALIGNMENT_FRAMES:
        raise ValueError(
            f"Need at least {MIN_ALIGNMENT_FRAMES} paired camera poses; found {len(source)}."
        )
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Camera centres must be finite.")
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        raise ValueError(
            "Reconstruction camera centres are collinear; similarity is ambiguous."
        )
    if np.linalg.matrix_rank(target - target.mean(axis=0)) < 2:
        raise ValueError("Board camera centres are collinear; similarity is ambiguous.")


def solve_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve target ~= scale * R @ source + translation with Umeyama's method."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    _validate_centres(source, target)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    covariance = target_zero.T @ source_zero / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_transpose) < 0:
        correction[-1, -1] = -1
    rotation = left @ correction @ right_transpose
    variance = np.mean(np.sum(source_zero * source_zero, axis=1))
    if variance <= 0:
        raise ValueError("Reconstruction camera centres have zero variance.")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Solved similarity scale is not positive and finite.")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def lift_board_corner_observations(
    frame_names: list[str],
    reconstruction_extrinsic: np.ndarray,
    reconstruction_intrinsic: np.ndarray,
    depth: np.ndarray,
    processed_size_hw: tuple[int, int],
    board_frame_names: list[str],
    board_image_size_hw: np.ndarray,
    corner_frame_index: np.ndarray,
    corner_id: np.ndarray,
    corner_px: np.ndarray,
    preprocess_geometry: list[dict[str, Any]] | None = None,
    board_pose_found: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lift R-02 ChArUco detections through R-01 depth/cameras into reconstruction space.

    The returned source points are in R-01's arbitrary reconstruction frame; targets are the
    exact D-034 board-corner coordinates in millimetres.  This is the shared primitive used by
    R-23 alignment and R-22's board-ruler diagnostic, so those two paths cannot drift apart.
    """
    names = [str(name) for name in frame_names]
    if len(set(names)) != len(names):
        raise ValueError("Frame names must be unique in the reconstruction output.")
    board_names = [str(name) for name in board_frame_names]
    if len(set(board_names)) != len(board_names):
        raise ValueError("Frame names must be unique in the board pose output.")

    reconstruction_extrinsic = np.asarray(reconstruction_extrinsic, dtype=np.float64)
    reconstruction_intrinsic = np.asarray(reconstruction_intrinsic, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    board_image_size_hw = np.asarray(board_image_size_hw)
    corner_frame_index = np.asarray(corner_frame_index)
    corner_id = np.asarray(corner_id)
    corner_px = np.asarray(corner_px, dtype=np.float64)
    found = (
        np.ones(len(board_names), dtype=bool)
        if board_pose_found is None
        else np.asarray(board_pose_found, dtype=bool)
    )
    if found.shape != (len(board_names),):
        raise ValueError("board_pose_found must have one value per board frame.")

    geometry_by_name: dict[str, dict[str, Any]] = {}
    if preprocess_geometry is not None:
        for description in preprocess_geometry:
            frame_name = str(description.get("frame_name", ""))
            if not frame_name:
                raise ValueError("Every preprocess_geometry entry needs a frame_name.")
            if frame_name in geometry_by_name:
                raise ValueError("preprocess_geometry frame names must be unique.")
            geometry_by_name[frame_name] = description

    frame_count = len(names)
    if reconstruction_extrinsic.shape != (frame_count, 3, 4):
        raise ValueError("reconstruction_extrinsic must have shape (S,3,4).")
    if reconstruction_intrinsic.shape != (frame_count, 3, 3):
        raise ValueError("reconstruction_intrinsic must have shape (S,3,3).")
    if depth.ndim != 3 or depth.shape[0] != frame_count:
        raise ValueError(
            "depth must have shape (S,H,W) with one image per reconstruction frame."
        )
    if board_image_size_hw.shape != (len(board_names), 2):
        raise ValueError("board_image_size_hw must have shape (S2,2).")
    if corner_px.ndim != 2 or corner_px.shape[1] != 2:
        raise ValueError("corner_px must have shape (N,2).")
    if not (len(corner_frame_index) == len(corner_id) == len(corner_px)):
        raise ValueError(
            "corner_frame_index, corner_id and corner_px must have equal length."
        )

    processed_height, processed_width = processed_size_hw
    if (processed_height, processed_width) != tuple(depth.shape[1:]):
        raise ValueError(
            "processed_size_hw must match depth image dimensions; got "
            f"{processed_size_hw} versus {tuple(depth.shape[1:])}."
        )

    nominal_corners = np.asarray(make_board().getChessboardCorners(), dtype=np.float64)
    frame_index_by_name = {name: index for index, name in enumerate(names)}
    observations_by_frame: dict[int, list[int]] = {}
    for row in range(len(corner_id)):
        board_frame_index = int(corner_frame_index[row])
        if board_frame_index < 0 or board_frame_index >= len(board_names):
            raise ValueError(
                "corner_frame_index contains an out-of-range board frame index."
            )
        cid = int(corner_id[row])
        if cid < 0 or cid >= len(nominal_corners):
            raise ValueError("corner_id contains an out-of-range ChArUco corner id.")
        if not found[board_frame_index]:
            continue
        reconstruction_index = frame_index_by_name.get(board_names[board_frame_index])
        if reconstruction_index is not None:
            observations_by_frame.setdefault(reconstruction_index, []).append(row)

    source_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []

    for frame_index, rows in observations_by_frame.items():
        row_indices = np.asarray(rows, dtype=np.int64)
        board_frame_index = int(corner_frame_index[row_indices[0]])
        full_height, full_width = board_image_size_hw[board_frame_index]
        if full_height <= 0 or full_width <= 0:
            raise ValueError("board image dimensions must be positive.")

        pixels_full_res = corner_px[row_indices]
        frame_name = names[frame_index]
        description = geometry_by_name.get(frame_name)
        if description is not None:
            if description.get("supported_for_metric_alignment") is False:
                raise ValueError(
                    f"{frame_name}: recorded reconstruction preprocessing is not supported for "
                    "metric alignment: "
                    f"{description.get('unsupported_reason', 'unknown reason')}"
                )
            recorded_original = tuple(
                int(value) for value in description["original_size_hw"]
            )
            if recorded_original != (int(full_height), int(full_width)):
                raise ValueError(
                    f"{frame_name}: board image size {int(full_height)}x{int(full_width)} does "
                    f"not match reconstruction preprocessing metadata {recorded_original}."
                )
            recorded_processed = tuple(
                int(value) for value in description["processed_size_hw"]
            )
            if recorded_processed != (processed_height, processed_width):
                raise ValueError(
                    f"{frame_name}: preprocessing metadata says processed size "
                    f"{recorded_processed}, but depth uses {(processed_height, processed_width)}."
                )
            scale_x = float(description["scale_x"])
            scale_y = float(description["scale_y"])
            crop_left = float(description.get("crop_left_px", 0))
            crop_top = float(description.get("crop_top_px", 0))
            u = (pixels_full_res[:, 0] + 0.5) * scale_x - 0.5 - crop_left
            v = (pixels_full_res[:, 1] + 0.5) * scale_y - 0.5 - crop_top
        else:
            # Backward compatibility for pre-R-23 golden runs. Their stored frames are landscape
            # and VGGT did not crop them vertically. A portrait/ambiguous legacy run must be
            # regenerated so the exact resize/crop map is recorded rather than guessed.
            if full_height >= full_width or processed_height >= processed_width:
                raise ValueError(
                    f"{frame_name}: reconstruction predates preprocess_geometry and is not a "
                    "clearly uncropped landscape frame; rerun R-01 to record resize/crop metadata."
                )
            u = (pixels_full_res[:, 0] + 0.5) * processed_width / full_width - 0.5
            v = (pixels_full_res[:, 1] + 0.5) * processed_height / full_height - 0.5
        in_bounds = (
            (u >= 0)
            & (u <= processed_width - 1)
            & (v >= 0)
            & (v <= processed_height - 1)
        )
        if not in_bounds.any():
            continue

        selected_rows = row_indices[in_bounds]
        selected_u = u[in_bounds]
        selected_v = v[in_bounds]
        coordinates = np.stack([selected_v, selected_u])
        depth_values = map_coordinates(
            depth[frame_index], coordinates, order=1, mode="nearest"
        )
        usable = np.isfinite(depth_values) & (depth_values > 0)
        if not usable.any():
            continue

        selected_rows = selected_rows[usable]
        selected_u = selected_u[usable]
        selected_v = selected_v[usable]
        depth_values = depth_values[usable]

        intrinsic = reconstruction_intrinsic[frame_index]
        extrinsic = reconstruction_extrinsic[frame_index]
        if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
            continue
        inverse_intrinsic = np.linalg.inv(intrinsic)
        pixels_homogeneous = np.stack(
            [selected_u, selected_v, np.ones(len(selected_u))], axis=1
        )
        rays = pixels_homogeneous @ inverse_intrinsic.T
        camera_points = rays * depth_values[:, None]
        world_points = (camera_points - extrinsic[:, 3]) @ extrinsic[:, :3]

        source_chunks.append(world_points)
        target_chunks.append(nominal_corners[corner_id[selected_rows].astype(int)])
        frame_chunks.append(np.full(len(world_points), frame_index, dtype=np.int64))

    if not source_chunks:
        raise ValueError(
            "No usable board corner observations could be lifted through reconstruction depth."
        )

    source = np.concatenate(source_chunks)
    target = np.concatenate(target_chunks)
    frame_of_point = np.concatenate(frame_chunks)
    if len(source) < MIN_BOARD_CORNER_OBSERVATIONS:
        raise ValueError(
            f"Only {len(source)} usable board corner observations; need at least "
            f"{MIN_BOARD_CORNER_OBSERVATIONS}."
        )
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Lifted and nominal board corners must be finite.")
    return source, target, frame_of_point


def align_board_geometry(
    frame_names: list[str],
    reconstruction_extrinsic: np.ndarray,
    reconstruction_intrinsic: np.ndarray,
    depth: np.ndarray,
    processed_size_hw: tuple[int, int],
    board_frame_names: list[str],
    board_image_size_hw: np.ndarray,
    corner_frame_index: np.ndarray,
    corner_id: np.ndarray,
    corner_px: np.ndarray,
    preprocess_geometry: list[dict[str, Any]] | None = None,
    board_pose_found: np.ndarray | None = None,
) -> BoardAlignmentResult:
    """Solve reconstruction -> board-mm similarity directly from the in-scene printed board."""
    source, target, frame_of_point = lift_board_corner_observations(
        frame_names,
        reconstruction_extrinsic,
        reconstruction_intrinsic,
        depth,
        processed_size_hw,
        board_frame_names,
        board_image_size_hw,
        corner_frame_index,
        corner_id,
        corner_px,
        preprocess_geometry,
        board_pose_found,
    )
    scale, rotation, translation = solve_similarity(source, target)
    predicted = scale * (source @ rotation.T) + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    return BoardAlignmentResult(
        scale=scale,
        rotation=rotation,
        translation=translation,
        observation_count=len(source),
        frame_count=len(np.unique(frame_of_point)),
        residuals_mm=residuals,
        rms_residual_mm=float(np.sqrt(np.mean(residuals**2))),
    )


def align_camera_centres(
    frame_names: list[str],
    reconstruction_extrinsic: np.ndarray,
    board_frame_names: list[str],
    board_extrinsic: np.ndarray,
    board_pose_found: np.ndarray | None = None,
) -> AlignmentResult:
    """Pair poses by frame name and solve their board-frame similarity transform."""
    # Both sides must have unique frame names, checked here rather than left to be caught (or
    # not) downstream: a name->index dict silently keeps whichever occurrence it saw last, so a
    # duplicate on either side would otherwise mean pairing against an arbitrary one of the
    # duplicates with no indication anything was ambiguous (issue #44 — this used to be checked
    # only on the reconstruction side, via a side effect of how paired names happened to be
    # counted downstream, which could not see a duplicate that never occurred more than once in
    # frame_names).
    reconstruction_names_str = [str(name) for name in frame_names]
    if len(set(reconstruction_names_str)) != len(reconstruction_names_str):
        raise ValueError("Frame names must be unique in the reconstruction output.")
    board_names_str = [str(name) for name in board_frame_names]
    if len(set(board_names_str)) != len(board_names_str):
        raise ValueError("Frame names must be unique in the board pose output.")

    reconstruction_by_name = {
        name: index for index, name in enumerate(reconstruction_names_str)
    }
    board_by_name = {name: index for index, name in enumerate(board_names_str)}
    found = (
        np.ones(len(board_frame_names), dtype=bool)
        if board_pose_found is None
        else np.asarray(board_pose_found, dtype=bool)
    )
    if found.shape != (len(board_frame_names),):
        raise ValueError("board_pose_found must have one value per board frame.")

    paired_names: list[str] = []
    reconstruction_indices: list[int] = []
    board_indices: list[int] = []
    for name in reconstruction_names_str:
        board_index = board_by_name.get(name)
        if board_index is not None and found[board_index]:
            paired_names.append(name)
            reconstruction_indices.append(reconstruction_by_name[name])
            board_indices.append(board_index)
    reconstruction_centres = camera_centres(reconstruction_extrinsic)[
        reconstruction_indices
    ]
    board_centres = camera_centres(board_extrinsic)[board_indices]
    scale, rotation, translation = solve_similarity(
        reconstruction_centres, board_centres
    )
    predicted = scale * (reconstruction_centres @ rotation.T) + translation
    residuals = np.linalg.norm(predicted - board_centres, axis=1)
    return AlignmentResult(
        scale=scale,
        rotation=rotation,
        translation=translation,
        frame_names=paired_names,
        reconstruction_centres=reconstruction_centres,
        board_centres=board_centres,
        residuals_mm=residuals,
        rms_residual_mm=float(np.sqrt(np.mean(residuals**2))),
    )


def load_and_align(
    reconstruction_run_dir: Path, board_pose_dir: Path
) -> BoardAlignmentResult:
    """Load R-01/R-02 outputs and anchor the metric transform on the board itself."""
    reconstruction_run_dir = Path(reconstruction_run_dir)
    board_pose_dir = Path(board_pose_dir)

    run_path = reconstruction_run_dir / "run.json"
    poses_path = reconstruction_run_dir / "poses.npz"
    depth_path = reconstruction_run_dir / "depth.npz"
    board_path = board_pose_dir / "board_poses.npz"
    for path in (run_path, poses_path, depth_path, board_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is required for board-anchored metric alignment."
            )

    run = json.loads(run_path.read_text(encoding="utf-8"))
    processed_size_hw = tuple(int(value) for value in run["processed_image_size_hw"])
    preprocess_geometry = run.get("preprocess_geometry")
    with np.load(poses_path) as reconstruction:
        reconstruction_names = [str(name) for name in reconstruction["frame_names"]]
        reconstruction_extrinsic = reconstruction["extrinsic"]
        reconstruction_intrinsic = reconstruction["intrinsic"]
    with np.load(depth_path) as depth_data:
        depth = depth_data["depth"]
    with np.load(board_path) as board:
        required = (
            "frame_names",
            "image_size_hw",
            "corner_frame_index",
            "corner_id",
            "corner_px",
            "pose_found",
        )
        for key in required:
            if key not in board:
                raise ValueError(f"{board_path} is missing {key}")
        board_names = [str(name) for name in board["frame_names"]]
        board_image_size_hw = board["image_size_hw"]
        corner_frame_index = board["corner_frame_index"]
        corner_id = board["corner_id"]
        corner_px = board["corner_px"]
        board_pose_found = board["pose_found"]

    return align_board_geometry(
        reconstruction_names,
        reconstruction_extrinsic,
        reconstruction_intrinsic,
        depth,
        processed_size_hw,
        board_names,
        board_image_size_hw,
        corner_frame_index,
        corner_id,
        corner_px,
        preprocess_geometry,
        board_pose_found,
    )


def write_alignment(
    path: Path, result: AlignmentResult | BoardAlignmentResult
) -> dict[str, Any]:
    """Write the similarity transform without changing downstream scale/rotation/translation keys."""
    report: dict[str, Any] = {
        "story": "R-04",
        "source_frame": "R-01 reconstruction frame (arbitrary scale)",
        "target_frame": "R-02 ChArUco board frame (millimetres)",
        "scale": result.scale,
        "rotation": result.rotation.tolist(),
        "translation_mm": result.translation.tolist(),
        "rms_residual_mm": result.rms_residual_mm,
    }
    if isinstance(result, BoardAlignmentResult):
        report.update(
            {
                "alignment_anchor": "in-scene ChArUco board geometry (R-23)",
                "paired_frame_count": result.frame_count,
                "board_corner_observation_count": result.observation_count,
                "note": (
                    "R-23 anchors the similarity directly on lifted ChArUco corners and their "
                    "known millimetre geometry, independent of board-pose camera-centre scale."
                ),
            }
        )
    else:
        report.update(
            {
                "alignment_anchor": "camera centres (legacy R-04)",
                "paired_frame_count": len(result.frame_names),
                "per_frame_residual_mm": dict(
                    zip(result.frame_names, result.residuals_mm.tolist(), strict=True)
                ),
                "note": (
                    "Residuals measure camera-centre alignment self-consistency, not object "
                    "accuracy."
                ),
            }
        )
    path = Path(path)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align R-01 reconstruction output to R-02 board poses."
    )
    parser.add_argument("reconstruction_run_dir", type=Path)
    parser.add_argument("board_pose_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = load_and_align(args.reconstruction_run_dir, args.board_pose_dir)
    out = args.out or args.reconstruction_run_dir / "metric_alignment.json"
    write_alignment(out, result)
    print(
        f"R-04 ALIGNMENT: PASS - {result.frame_count} board frames, "
        f"{result.observation_count} corner observations, "
        f"RMS residual={result.rms_residual_mm:.3f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
