"""R-44: move each photo rigidly onto its own ChArUco corners before fusion.

R-23 solves one similarity for the whole capture from every photo's lifted board corners. After it,
each photo still sits a median 2.23 mm from where its own corners say it should be (P-41's review,
281 photos on ten parts). R-30 (`recon.tsdf_fusion.reanchor_frames_to_board`) then corrects each
photo's board-plane tilt and height -- three of a pose's six degrees of freedom -- and leaves the
other three: where the photo sits in the board plane (x, y) and its rotation about the board's
normal. A sideways shift doesn't move a flat top face, but it moves every wall and edge, which is
where P-41 found the error.

This module fits, for each photo on its own, the rigid transform (rotation and translation, no
scale) that takes that photo's lifted corners onto their printed positions, and applies it to the
photo's points and camera pose. `fuse_run` runs it before R-30's plane step, so R-30 then works on
photos that are already on their own corners.

Why rigid and not a similarity: P-41 tried both. Rescaling each photo's depth bought the same wall
gain, added stray board surface on `g05_00`, and would need a per-photo depth factor carried into
provenance. A rigid correction changes camera poses only, which `corrected_extrinsic.npz` already
carries to provenance and uncertainty.

Why the fit drops far-off corners (`fit_rigid_trimmed`): a single corner can lift badly wrong
inside an otherwise good photo, most likely one next to the part's silhouette, where depth is
interpolated between the board and the part (this module's box test found up to ~11 mm that way).
R-38 judges a photo by its median corner, so it doesn't catch one. A plain fit lets that corner
drag the whole photo: on `g02_06`, frames 7 and 8 have a corner 41.7 and 27.5 mm off after a plain
fit, which left their corners 11.8 and 8.5 mm RMS off and moved the correction at the board centre
by up to 2.2 mm -- the size of the correction itself (@TabeenRaoof's review on #442, reproduced on
the saved run). P-41's screening fusions (`experiment.py`) also trimmed, by the rule this follows;
an earlier version of this module said otherwise, wrongly, from P-41's measurement script.
Every photo still reports its worst corner, dropped or not, so a bad corner stays visible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recon.metric_alignment import lift_board_corner_observations

# A photo needs this many of its own lifted corners before a rigid fit to them is trusted. Three
# non-collinear points fix a rigid pose exactly, leaving nothing to average the corner noise over.
# Eight is P-41's figure, kept so R-44's results stay comparable with the screening run. It is a
# choice, not a measured threshold. On the saved s-04, s-05 and pentagon runs, photos carry 13 to
# 25 corners, and every minimum from 4 to 8 leaves the same single photo (one with no corners at
# all) uncorrected.
DEFAULT_MIN_CORNERS_PER_FRAME = 8

# The trimming rule, as @TabeenRaoof described P-41's `experiment.py` on #442: after a fit, drop
# every corner more than max(3 x the median residual, 0.5 mm) from its printed position, refit, and
# repeat at most three times. The 3x and 0.5 mm are P-41's choices, kept for comparability, not
# measured thresholds. On `g02_06` this keeps 12 of frame 7's 15 corners (0.39 mm RMS after) and 12
# of frame 8's (0.24 mm); P-41's saved runs kept 13 (0.42 mm) and 12 (0.26 mm). The small gap is in
# details of P-41's loop not visible here, so the measurement checks the two against each other.
TRIM_MEDIAN_MULTIPLE = 3.0
TRIM_FLOOR_MM = 0.5
TRIM_MAX_REFITS = 3

STATUS_CORRECTED = "corrected"
STATUS_SKIPPED = "skipped"

SKIP_REASON_REJECTED = "rejected by R-38's alignment (its own corners were outliers)"
SKIP_REASON_NO_CORNERS = "no corners lifted through this photo's depth"


def fit_rigid(
    source_mm: np.ndarray, target_mm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation and translation with `target ~= rotation @ source + translation`.

    Kabsch's method, with the usual determinant guard so a reflection is never returned. No
    scale: R-44 corrects where a photo sits, not how big its depth is (see the module docstring).

    Coplanar points are fine -- ChArUco corners always are -- as long as they are not all on one
    line, which is refused rather than answered with an arbitrary rotation about that line.
    """
    source = np.asarray(source_mm, dtype=np.float64)
    target = np.asarray(target_mm, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or source.shape != target.shape:
        raise ValueError(
            f"source and target must both be (N,3); got {source.shape} and {target.shape}"
        )
    if len(source) < 3:
        raise ValueError(f"a rigid fit needs at least 3 point pairs, got {len(source)}")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    # Rank below 2 means every point is on one line (or one spot): the rotation about that
    # line is then unconstrained, and returning one would be a guess.
    singular_values = np.linalg.svd(source_centered, compute_uv=False)
    if singular_values[1] <= 1e-9 * max(singular_values[0], 1.0):
        raise ValueError("the points are collinear; a rigid fit to them is not unique.")

    left, _values, right_transpose = np.linalg.svd(target_centered.T @ source_centered)
    correction = np.eye(3)
    if np.linalg.det(left @ right_transpose) < 0:
        correction[2, 2] = -1.0
    rotation = left @ correction @ right_transpose
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def correct_extrinsic_for_moved_points(
    extrinsic: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """The camera-from-board pose that sees moved points where the old pose saw the originals.

    The photo's depth measurement doesn't change, only where its points sit on the board. With
    `x' = R x + t` a moved point and `x_cam = R_c x + t_c` the original pose, substituting
    `x = R^T (x' - t)` gives `x_cam = (R_c R^T) x' + (t_c - R_c R^T t)`. This is the same
    derivation as R-30's in `reanchor_frames_to_board`.
    """
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.shape != (3, 4):
        raise ValueError(f"extrinsic must be (3,4), got {extrinsic.shape}")
    camera_rotation = extrinsic[:, :3]
    camera_translation = extrinsic[:, 3]
    corrected_rotation = camera_rotation @ rotation.T
    corrected_translation = camera_translation - corrected_rotation @ translation
    corrected = np.empty((3, 4))
    corrected[:, :3] = corrected_rotation
    corrected[:, 3] = corrected_translation
    return corrected


def fit_rigid_trimmed(
    source_mm: np.ndarray,
    target_mm: np.ndarray,
    min_corners: int,
    median_multiple: float = TRIM_MEDIAN_MULTIPLE,
    floor_mm: float = TRIM_FLOOR_MM,
    max_refits: int = TRIM_MAX_REFITS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`fit_rigid`, then drop far-off corners and refit, at most `max_refits` times.

    After each fit, a corner is kept only if it is still kept and lies within
    max(`median_multiple` x the median residual of the kept corners, `floor_mm`) of its target.
    A dropped corner is never re-admitted, and the loop stops early once nothing more is dropped.
    The floor stops a photo whose corners all agree to a fraction of a millimetre from being
    trimmed down to its tightest few.

    Returns `(rotation, translation, kept)`, `kept` a boolean mask over the input corners.

    Raises:
        ValueError: trimming would leave fewer than `min_corners`, or the corners are collinear
            (`fit_rigid`). The caller then leaves the photo uncorrected and reports why, rather
            than trusting a fit to the few corners that survived.
    """
    source = np.asarray(source_mm, dtype=np.float64)
    target = np.asarray(target_mm, dtype=np.float64)
    kept = np.ones(len(source), dtype=bool)
    rotation, translation = fit_rigid(source, target)
    for _refit in range(max_refits):
        residuals = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        cutoff_mm = max(median_multiple * float(np.median(residuals[kept])), floor_mm)
        still_kept = kept & (residuals <= cutoff_mm)
        if np.array_equal(still_kept, kept):
            break
        kept_count = int(np.count_nonzero(still_kept))
        if kept_count < min_corners:
            raise ValueError(
                f"trimming far-off corners leaves {kept_count} of {len(source)}, fewer than "
                f"{min_corners}"
            )
        kept = still_kept
        rotation, translation = fit_rigid(source[kept], target[kept])
    return rotation, translation, kept


@dataclass(frozen=True)
class FrameCornerReport:
    """What R-44 measured and did for one photo, for `tsdf_fusion.json`.

    Every photo of the capture gets one: `status` is "corrected" or "skipped", and a skipped photo
    says why in `skip_reason` and keeps today's behaviour. Distances are in board-frame mm.
    `rms_before_mm` is the root-mean-square 3-D distance of the photo's corners from their printed
    positions after R-23's global alignment. `corners_used` is how many survived
    `fit_rigid_trimmed`, and `rms_after_mm` is their RMS distance after the fit.
    `worst_corner_after_mm` is the largest distance after the fit over *every* lifted corner,
    dropped ones included, so a corner the fit set aside is still visible. `shift_at_reference_mm` is how far the correction moves
    the reference point (the board footprint's centre, where a part sits) within the board plane,
    `height_shift_at_reference_mm` how far it moves it along the board's normal, and
    `rotation_deg` the total angle of the correction's rotation. The fields that describe a fit
    are `None` for a skipped photo, and `rms_before_mm` is `None` when it had no corners at all.
    """

    frame_index: int
    status: str
    corner_count: int
    corners_used: int | None
    skip_reason: str | None
    rms_before_mm: float | None
    rms_after_mm: float | None
    worst_corner_after_mm: float | None
    shift_at_reference_mm: float | None
    height_shift_at_reference_mm: float | None
    rotation_deg: float | None
    rotation: list[list[float]] | None
    translation_mm: list[float] | None

    def as_report(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "status": self.status,
            "corner_count": self.corner_count,
            "corners_used": self.corners_used,
            "skip_reason": self.skip_reason,
            "rms_before_mm": self.rms_before_mm,
            "rms_after_mm": self.rms_after_mm,
            "worst_corner_after_mm": self.worst_corner_after_mm,
            "shift_at_reference_mm": self.shift_at_reference_mm,
            "height_shift_at_reference_mm": self.height_shift_at_reference_mm,
            "rotation_deg": self.rotation_deg,
            "rotation": self.rotation,
            "translation_mm": self.translation_mm,
        }


def lift_run_corners(
    reconstruction_run_dir: Path,
    board_pose_dir: Path,
    frame_names: list[str],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lift a run's ChArUco corners through its own depth, in R-01's reconstruction frame.

    Uses `lift_board_corner_observations`, the same primitive R-23's alignment is solved from, with
    the same inputs `metric_alignment.load_and_align` gives it, so the corners R-44 fits each
    photo to are exactly the ones the global alignment used. `board_pose_dir` is the folder whose
    `board_poses.npz` the alignment read: `board_pose_calibrated/` when the service calibrated
    the camera from the board, `board_pose/` otherwise.

    Returns `(corner_points, targets_mm, frame_of_corner)`. The points are not yet moved through
    the global alignment; the caller does that with the run's own similarity.
    """
    reconstruction_run_dir = Path(reconstruction_run_dir)
    board_poses_path = Path(board_pose_dir) / "board_poses.npz"
    run_json_path = reconstruction_run_dir / "run.json"
    for required in (board_poses_path, run_json_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} is needed to lift this run's board corners for R-44's "
                "per-photo correction."
            )
    run = json.loads(run_json_path.read_text(encoding="utf-8"))
    if "processed_image_size_hw" not in run:
        raise ValueError(
            f"{run_json_path} has no processed_image_size_hw; this run predates R-23's corner "
            "lifting and can't be corrected from its own corners. Re-run R-01."
        )
    with np.load(board_poses_path) as board:
        return lift_board_corner_observations(
            frame_names,
            extrinsic,
            intrinsic,
            depth,
            tuple(int(value) for value in run["processed_image_size_hw"]),
            [str(name) for name in board["frame_names"]],
            board["image_size_hw"],
            board["corner_frame_index"],
            board["corner_id"],
            board["corner_px"],
            run.get("preprocess_geometry"),
            board["pose_found"],
        )


def _rms_distance_mm(points: np.ndarray, targets: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((points - targets) ** 2, axis=1))))


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    # trace(R) = 1 + 2 cos(theta); clipped because rounding can push it a hair past +-1.
    cos_angle = (float(np.trace(rotation)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def reanchor_frames_to_corners(
    board_frame_points: np.ndarray,
    board_extrinsic: np.ndarray,
    corner_points_mm: np.ndarray,
    corner_targets_mm: np.ndarray,
    frame_of_corner: np.ndarray,
    reference_point_mm: np.ndarray,
    min_corners_per_frame: int = DEFAULT_MIN_CORNERS_PER_FRAME,
    skip_frame_indices: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray, list[FrameCornerReport]]:
    """R-44: move each photo rigidly onto its own lifted ChArUco corners.

    `board_frame_points` (S,H,W,3) and `board_extrinsic` (S,3,4) are every photo's points and
    camera-from-board pose after R-23's global alignment, as `fuse_run` holds them.
    `corner_points_mm` (N,3) are the lifted corners in that same frame, `corner_targets_mm` (N,3)
    where the printed board says each one is, and `frame_of_corner` (N,) which photo each came
    from -- `recon.metric_alignment.lift_board_corner_observations`' output, moved through the
    same global alignment.

    For each photo with at least `min_corners_per_frame` corners, fits the rigid transform taking
    its corners onto their targets, dropping far-off corners (`fit_rigid_trimmed`), and applies
    it to that photo's points and pose (`correct_extrinsic_for_moved_points`). Nothing is raised
    for a photo that can't be corrected: it keeps its global-alignment points and pose exactly,
    and its report says why. That covers a photo in `skip_frame_indices` (R-38 rejected it), one
    with no corners, one with too few, one whose trimming leaves too few, and one whose corners
    are all on one line, which `fit_rigid` refuses.

    Returns `(corrected_points, corrected_extrinsic, reports)`, with one report per photo in
    index order.
    """
    board_frame_points = np.asarray(board_frame_points, dtype=np.float64)
    board_extrinsic = np.asarray(board_extrinsic, dtype=np.float64)
    corner_points = np.asarray(corner_points_mm, dtype=np.float64)
    corner_targets = np.asarray(corner_targets_mm, dtype=np.float64)
    frame_of_corner = np.asarray(frame_of_corner)
    reference = np.asarray(reference_point_mm, dtype=np.float64)

    if board_frame_points.ndim != 4 or board_frame_points.shape[3] != 3:
        raise ValueError(
            f"board_frame_points must have shape (S,H,W,3), got {board_frame_points.shape}"
        )
    frame_count, height, width, _ = board_frame_points.shape
    if board_extrinsic.shape != (frame_count, 3, 4):
        raise ValueError(
            f"board_extrinsic must be ({frame_count},3,4), got {board_extrinsic.shape}"
        )
    if (
        corner_points.ndim != 2
        or corner_points.shape[1] != 3
        or corner_points.shape != corner_targets.shape
    ):
        raise ValueError(
            "corner points and targets must both be (N,3); got "
            f"{corner_points.shape} and {corner_targets.shape}"
        )
    if frame_of_corner.shape != (len(corner_points),):
        raise ValueError(
            "frame_of_corner needs one entry per corner; got "
            f"{frame_of_corner.shape} for {len(corner_points)} corners"
        )
    if len(frame_of_corner) and (
        frame_of_corner.min() < 0 or frame_of_corner.max() >= frame_count
    ):
        raise ValueError(
            f"frame_of_corner names a photo outside 0..{frame_count - 1}; a corner from a photo "
            "this run doesn't have can't be matched to any points."
        )
    if not (np.isfinite(corner_points).all() and np.isfinite(corner_targets).all()):
        raise ValueError("corner points and targets must be finite.")
    if reference.shape != (3,):
        raise ValueError(f"reference_point_mm must be (3,), got {reference.shape}")
    if min_corners_per_frame < 3:
        raise ValueError("min_corners_per_frame must be at least 3 for a rigid fit.")

    corrected_points = board_frame_points.copy()
    corrected_extrinsic = board_extrinsic.copy()
    reports: list[FrameCornerReport] = []

    for frame_index in range(frame_count):
        rows = frame_of_corner == frame_index
        corner_count = int(np.count_nonzero(rows))
        frame_corners = corner_points[rows]
        frame_targets = corner_targets[rows]
        rms_before = (
            _rms_distance_mm(frame_corners, frame_targets) if corner_count else None
        )

        # Checked in this order so a photo gets the most specific reason: R-38's rejection says
        # the photo's own data is wrong, which outranks "too few corners".
        skip_reason: str | None = None
        if frame_index in skip_frame_indices:
            skip_reason = SKIP_REASON_REJECTED
        elif corner_count == 0:
            skip_reason = SKIP_REASON_NO_CORNERS
        elif corner_count < min_corners_per_frame:
            skip_reason = f"fewer than {min_corners_per_frame} corners ({corner_count})"

        rotation = translation = kept = None
        if skip_reason is None:
            try:
                rotation, translation, kept = fit_rigid_trimmed(
                    frame_corners, frame_targets, min_corners_per_frame
                )
            except ValueError as exc:
                # Counts and shapes are checked above, so only two things reach here: trimming
                # left too few corners to trust, or the corners are collinear (a photo that sees
                # one row of the board has no rotation to fit about that row).
                skip_reason = f"no usable rigid fit: {exc}"

        if skip_reason is not None:
            reports.append(
                FrameCornerReport(
                    frame_index=frame_index,
                    status=STATUS_SKIPPED,
                    corner_count=corner_count,
                    corners_used=None,
                    skip_reason=skip_reason,
                    rms_before_mm=rms_before,
                    rms_after_mm=None,
                    worst_corner_after_mm=None,
                    shift_at_reference_mm=None,
                    height_shift_at_reference_mm=None,
                    rotation_deg=None,
                    rotation=None,
                    translation_mm=None,
                )
            )
            continue

        moved_corners = frame_corners @ rotation.T + translation
        corner_distances_after = np.linalg.norm(moved_corners - frame_targets, axis=1)
        reference_shift = rotation @ reference + translation - reference

        flat_points = board_frame_points[frame_index].reshape(-1, 3)
        corrected_points[frame_index] = (
            flat_points @ rotation.T + translation
        ).reshape(height, width, 3)
        corrected_extrinsic[frame_index] = correct_extrinsic_for_moved_points(
            board_extrinsic[frame_index], rotation, translation
        )

        reports.append(
            FrameCornerReport(
                frame_index=frame_index,
                status=STATUS_CORRECTED,
                corner_count=corner_count,
                corners_used=int(np.count_nonzero(kept)),
                skip_reason=None,
                rms_before_mm=rms_before,
                rms_after_mm=float(np.sqrt(np.mean(corner_distances_after[kept] ** 2))),
                worst_corner_after_mm=float(np.max(corner_distances_after)),
                shift_at_reference_mm=float(np.linalg.norm(reference_shift[:2])),
                height_shift_at_reference_mm=float(reference_shift[2]),
                rotation_deg=_rotation_angle_deg(rotation),
                rotation=rotation.tolist(),
                translation_mm=translation.tolist(),
            )
        )

    return corrected_points, corrected_extrinsic, reports
