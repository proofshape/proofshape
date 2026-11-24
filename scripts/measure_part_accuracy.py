"""P-41: measure what a printed part reads against its own CAD, the way the verdict sees it, and
how far each photo sits from where the board says it is.

P-41's accuracy review (docs/accuracy-review-2026-11-20.md) needed two measurements nothing in
the repo produced, and both were first made in a scratch session. This script is those two
measurements as committed, tested code, so a second member can re-run the review's tables instead
of taking them on trust (the same reason R-33's hand classification became
`measure_wall_share.py`).

Three subcommands:

    python scripts/measure_part_accuracy.py run CAPTURE_DIR --cad CAD.stl --out OUT_DIR
    python scripts/measure_part_accuracy.py report OUT_DIR [OUT_DIR ...]
    python scripts/measure_part_accuracy.py corners OUT_DIR [OUT_DIR ...]

`run` puts one capture through the service's own `run_reconstruction_pipeline`, then through
`global_register`, `check_ambiguity` and `compute_deviation` against the given CAD -- the same
calls, in the same order, as `recon.service`'s `run_post_reconstruction_measurement`. It writes
`OUT_DIR/artifacts/` (the pipeline's own files) plus `measure.npz` and `measure.json`. It needs
the model weights and a GPU (`PROOFSHAPE_RECON_DEVICE`), exactly as the service does. Everything
after it is plain numpy on the saved files.

`report` reads `measure.npz` and prints, per part: how much of the Observed surface is measured at
all, how far top-face and wall vertices sit from the CAD, the single worst reading, how much of
the Observed surface is nowhere near the part, and the smallest tolerance at which v1's verdict
rule would Pass the part.

`corners` reads `artifacts/` and prints, per capture, how far each photo's own ChArUco corners
land from their known board positions: after R-23's one global alignment, after R-30's per-photo
board-plane correction, and what a rigid fit of that one photo to its own corners would leave.

What this script deliberately does not do:

- It chooses no threshold. The surface classes and the "stray" distance below are reporting
  bins for this review, named as such, and no verdict or pipeline code reads them.
- It never opens a photograph. Captures are read by the pipeline only.
- It does not re-fuse anything. The review's "what if each photo were re-anchored" figures came
  from a scratch harness, and reproducing them in repo code is R-44's job, not this script's.

A part is always compared with its *own* CAD here, so nothing it reads is a planted defect. A
deviation is still three things together, and this script does not separate them: how far the
print itself departs from nominal CAD, reconstruction error, and registration error (the pose is
fitted to the very surface being judged). Calipers give the first of those for a few dimensions
(`fixtures/caliper_*.json`); nothing here subtracts it. So these are CAD-relative readings on
good parts, which is what the verdict thresholds -- not a measurement of reconstruction accuracy
on its own. Comparing a part with a different CAD (the planted-defect case) works the same way,
but then the numbers mean something else, and the review says so wherever it does that.

Two labels to read carefully in `corners`. The per-photo figure is the root-mean-square of that
photo's corner distances. R-38's `per_frame_residual_mm`, already saved in every run's
`metric_alignment.json`, is the per-photo *median* of the same distances after the global
alignment: close, but a different statistic. And the height column is the magnitude of each
photo's *mean* signed z residual at its corners, so it shows whether a photo sits high or low as
a whole, not how flat its board is.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# R-31: the macOS thread guard has to run before Open3D is imported, exactly as recon.service
# does it. Run directly, this script would otherwise load Open3D first with whatever
# OMP_NUM_THREADS the shell happened to have, and `run` then deadlocks at torch's first threaded
# operation (#261). Importing the module applies the guard; on other platforms it does nothing.
from recon import runtime_env as _runtime_env  # noqa: F401  # isort: skip

import open3d as o3d  # isort: skip

from recon.cad_size_check import compute_height_above_board
from recon.corner_reanchoring import DEFAULT_MIN_CORNERS_PER_FRAME, fit_rigid
from recon.metric_alignment import (
    apply_similarity_transform,
    lift_board_corner_observations,
)
from recon.part_segmentation import DEFAULT_FOOTPRINT_MM
from recon.provenance import LABEL_OBSERVED, LABEL_UNOBSERVED

# --- reporting bins: this review's own, read by nothing else ----------------------------------

# A model vertex is binned by the CAD face nearest to it, using that face's normal in the CAD's
# own frame (CAD +z is up, the convention every committed reference CAD follows). "top" is a
# face pointing up, "wall" a vertical face. The verdict does NOT bin this way -- it splits by
# height percentile (inspection.verdict.TOP_REGION_PERCENTILE) -- and the review reports both,
# because the difference between the two is one of its findings.
TOP_FACE_MIN_NORMAL_Z = 0.9
WALL_FACE_MAX_ABS_NORMAL_Z = 0.1
SURFACE_TOP = "top"
SURFACE_WALL = "wall"
SURFACE_OTHER = "other"

# An Observed vertex farther than this from the registered CAD is counted as "stray": surface
# that is nowhere near the part (board skin, mostly). 5 mm is a round number comfortably past
# every part's own wall scatter in the review's runs, not a measured threshold.
STRAY_DISTANCE_MM = 5.0

# The tolerances the verdict rule is tried at. 3.7 is in the list because the current gate sits
# at 0.70 * 5.278 = 3.6946 mm (D-077, D-078), so the first round value above it is where a part
# with small readings first Passes.
DEFAULT_TOLERANCES_MM = (
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    3.7,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    9.0,
    10.0,
    12.0,
)

# A photo needs this many of its own corners before a rigid fit to them means anything. This was
# the review's own choice of eight; R-44 adopted the same figure for the pipeline, and this reads
# it from there so the review's numbers and the pipeline's correction can't quietly diverge.
DEFAULT_MIN_CORNERS_PER_PHOTO = DEFAULT_MIN_CORNERS_PER_FRAME

MEASURE_NPZ = "measure.npz"
MEASURE_JSON = "measure.json"


# --- per-photo corner residuals -----------------------------------------------------------------


def photo_corner_residuals(
    corner_points_mm: np.ndarray,
    corner_targets_mm: np.ndarray,
    photo_of_corner: np.ndarray,
    *,
    min_corners: int = DEFAULT_MIN_CORNERS_PER_PHOTO,
    reference_point_mm: np.ndarray | None = None,
    photo_indices: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """How far each photo's own corners sit from their known board positions.

    `corner_points_mm` are a capture's lifted ChArUco corners, already in the board frame;
    `corner_targets_mm` are where the printed board says each one is; `photo_of_corner` says
    which photo each row came from. Returns `(measured, skipped)`:

    - `measured`: one dict per photo with at least `min_corners` corners -- `rms_mm` (3-D),
      `in_plane_rms_mm` (x and y only), `height_mean_mm` (signed z), `rms_after_rigid_fit_mm`
      (what is left if that one photo is moved rigidly onto its own corners), and
      `shift_at_reference_mm` (how far that rigid move carries `reference_point_mm`, in the board
      plane -- "how far off is the part in this photo").
    - `skipped`: one dict per photo with fewer corners, with the count and the reason, so a
      photo that could not be measured is visible rather than silently missing from a median.

    `photo_indices` lists every photo that has to be accounted for. A photo in it with no corner
    at all -- its board was not found, or none of its corners landed on valid depth -- never
    appears in `photo_of_corner`, so without this list it would vanish from both results (found
    on review, @dalwalyk, PR #427: a real s-04 run reported 30 photos, 29 measured and nothing
    skipped). With it, such a photo is listed in `skipped` with a count of zero. Left as `None`,
    only the photos that have corners are considered, which is all a caller with no frame list
    can ask for.

    `reference_point_mm` defaults to the centre of the board's footprint on the board plane,
    which is where a part is placed.
    """
    points = np.asarray(corner_points_mm, dtype=np.float64)
    targets = np.asarray(corner_targets_mm, dtype=np.float64)
    photo_index = np.asarray(photo_of_corner)
    if points.shape != targets.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"corner points and targets must both be (N,3); got {points.shape} and "
            f"{targets.shape}"
        )
    if len(photo_index) != len(points):
        raise ValueError(
            "photo_of_corner needs one entry per corner; got "
            f"{len(photo_index)} for {len(points)} corners"
        )
    if min_corners < 3:
        raise ValueError("min_corners must be at least 3 for a rigid fit.")
    if reference_point_mm is None:
        reference = np.array(
            [DEFAULT_FOOTPRINT_MM[0] / 2.0, DEFAULT_FOOTPRINT_MM[1] / 2.0, 0.0]
        )
    else:
        reference = np.asarray(reference_point_mm, dtype=np.float64)

    photos_with_corners = sorted({int(photo) for photo in np.unique(photo_index)})
    if photo_indices is None:
        photos = photos_with_corners
    else:
        photos = sorted({int(photo) for photo in photo_indices})
        unlisted = sorted(set(photos_with_corners) - set(photos))
        if unlisted:
            raise ValueError(
                f"photo_of_corner names photo(s) {unlisted} that photo_indices does not list; "
                "a corner from a photo nobody asked about would be dropped without a trace."
            )

    measured: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for photo in photos:
        rows = photo_index == photo
        corner_count = int(np.count_nonzero(rows))
        if corner_count == 0:
            skipped.append(
                {
                    "photo_index": int(photo),
                    "corner_count": 0,
                    "reason": "no corners lifted through this photo's depth",
                }
            )
            continue
        if corner_count < min_corners:
            skipped.append(
                {
                    "photo_index": int(photo),
                    "corner_count": corner_count,
                    "reason": f"fewer than {min_corners} corners",
                }
            )
            continue

        photo_points = points[rows]
        photo_targets = targets[rows]
        residual = photo_points - photo_targets
        rotation, translation = fit_rigid(photo_points, photo_targets)
        residual_after = photo_points @ rotation.T + translation - photo_targets
        moved_reference = rotation @ reference + translation
        measured.append(
            {
                "photo_index": int(photo),
                "corner_count": corner_count,
                "rms_mm": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                "in_plane_rms_mm": float(
                    np.sqrt(np.mean(np.sum(residual[:, :2] ** 2, axis=1)))
                ),
                "height_mean_mm": float(np.mean(residual[:, 2])),
                "rms_after_rigid_fit_mm": float(
                    np.sqrt(np.mean(np.sum(residual_after**2, axis=1)))
                ),
                "shift_at_reference_mm": float(
                    np.linalg.norm((moved_reference - reference)[:2])
                ),
            }
        )
    return measured, skipped


@dataclass(frozen=True)
class LiftedCorners:
    """One capture's lifted corners, in the board frame, under two stages of the pipeline."""

    frame_names: list[str]
    photo_of_corner: np.ndarray
    targets_mm: np.ndarray
    # After R-23's single global similarity only.
    after_global_alignment_mm: np.ndarray
    # After R-30's per-photo board-plane correction too; None when the run has no
    # fusion/corrected_extrinsic.npz (a run from before R-30, or one that never reached fusion).
    after_plane_reanchoring_mm: np.ndarray | None
    rejected_photo_indices: frozenset[int]


def load_lifted_corners(artifact_dir: Path) -> LiftedCorners:
    """Lift a finished run's ChArUco corners into the board frame, from its own saved files.

    Uses `recon.metric_alignment.lift_board_corner_observations`, the same primitive R-23's
    alignment is solved from, so this cannot drift from what the pipeline itself aligned on.
    The board-pose folder is `board_pose_calibrated/` when the service had to calibrate the
    camera from the board (frames with no EXIF, which is every phone-app capture) and
    `board_pose/` otherwise -- the same choice `run_reconstruction_pipeline` makes.
    """
    artifact_dir = Path(artifact_dir)
    reconstruction_dir = artifact_dir / "reconstruction"
    board_dir = artifact_dir / "board_pose_calibrated"
    if not board_dir.is_dir():
        board_dir = artifact_dir / "board_pose"
    for required in (
        reconstruction_dir / "run.json",
        reconstruction_dir / "poses.npz",
        reconstruction_dir / "depth.npz",
        reconstruction_dir / "metric_alignment.json",
        board_dir / "board_poses.npz",
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} is needed to lift this run's board corners."
            )

    run = json.loads((reconstruction_dir / "run.json").read_text(encoding="utf-8"))
    alignment = json.loads(
        (reconstruction_dir / "metric_alignment.json").read_text(encoding="utf-8")
    )
    with np.load(reconstruction_dir / "poses.npz") as poses:
        frame_names = [str(name) for name in poses["frame_names"]]
        extrinsic = poses["extrinsic"].astype(np.float64)
        intrinsic = poses["intrinsic"].astype(np.float64)
    with np.load(reconstruction_dir / "depth.npz") as depth_data:
        depth = depth_data["depth"].astype(np.float64)
    with np.load(board_dir / "board_poses.npz") as board:
        source, targets, photo_of_corner = lift_board_corner_observations(
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

    scale = float(alignment["scale"])
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["translation_mm"], dtype=np.float64)
    after_global = apply_similarity_transform(source, scale, rotation, translation)

    after_plane = None
    corrected_path = artifact_dir / "fusion" / "corrected_extrinsic.npz"
    if corrected_path.is_file():
        with np.load(corrected_path) as corrected:
            corrected_extrinsic = corrected["extrinsic"].astype(np.float64)
        if corrected_extrinsic.shape != extrinsic.shape:
            raise ValueError(
                f"{corrected_path} has shape {corrected_extrinsic.shape}, expected "
                f"{extrinsic.shape} to match poses.npz."
            )
        # R-30 changes each photo's camera-from-board pose, not its depth. A corner's position
        # in that photo's own camera, in millimetres, is therefore the same before and after;
        # only the pose it is mapped back through differs. Camera-space position first (R-01's
        # own extrinsic, scaled into mm by R-04's scale), then back out through the corrected
        # pose: x_board = R_c^T (x_cam - t_c).
        after_plane = np.empty_like(after_global)
        for photo in np.unique(photo_of_corner):
            rows = photo_of_corner == photo
            camera_mm = scale * (
                source[rows] @ extrinsic[photo, :, :3].T + extrinsic[photo, :, 3]
            )
            corrected_rotation = corrected_extrinsic[photo, :, :3]
            corrected_translation = corrected_extrinsic[photo, :, 3]
            after_plane[rows] = (camera_mm - corrected_translation) @ corrected_rotation

    rejected_names = set(alignment.get("rejected_frame_names", []))
    rejected = frozenset(
        index for index, name in enumerate(frame_names) if name in rejected_names
    )
    return LiftedCorners(
        frame_names=frame_names,
        photo_of_corner=np.asarray(photo_of_corner),
        targets_mm=targets,
        after_global_alignment_mm=after_global,
        after_plane_reanchoring_mm=after_plane,
        rejected_photo_indices=rejected,
    )


def summarise_corner_residuals(
    corners: LiftedCorners, *, min_corners: int = DEFAULT_MIN_CORNERS_PER_PHOTO
) -> dict[str, Any]:
    """One capture's corner residuals, as medians over its photos.

    A photo R-38 rejected is left out of every figure here and counted separately: its depth is
    already known to be wrong and it never reaches fusion, so including it would report an
    error the pipeline does not actually fuse.

    Every photo of the capture is in exactly one of three counts -- measured, skipped (with its
    reason) or rejected -- and the three add up to `photo_count`. That is checked here rather
    than assumed, because a photo that drops out of all three is how a median quietly comes to
    describe fewer photos than the table says.
    """
    keep = np.array(
        [
            int(photo) not in corners.rejected_photo_indices
            for photo in corners.photo_of_corner
        ],
        dtype=bool,
    )
    targets = corners.targets_mm[keep]
    photo_of_corner = corners.photo_of_corner[keep]
    photo_count = len(corners.frame_names)
    not_rejected = [
        photo
        for photo in range(photo_count)
        if photo not in corners.rejected_photo_indices
    ]

    measured_global, skipped = photo_corner_residuals(
        corners.after_global_alignment_mm[keep],
        targets,
        photo_of_corner,
        min_corners=min_corners,
        photo_indices=not_rejected,
    )
    rejected_count = photo_count - len(not_rejected)
    if len(measured_global) + len(skipped) + rejected_count != photo_count:
        raise ValueError(
            f"{len(measured_global)} measured + {len(skipped)} skipped + {rejected_count} "
            f"rejected photos do not add up to the capture's {photo_count}."
        )
    summary: dict[str, Any] = {
        "photo_count": photo_count,
        "rejected_photo_count": rejected_count,
        "measured_photo_count": len(measured_global),
        "skipped_photo_count": len(skipped),
        "skipped_photos": skipped,
        "after_global_alignment": _median_block(measured_global),
        "after_plane_reanchoring": None,
        "per_photo_after_global_alignment": measured_global,
    }
    if corners.after_plane_reanchoring_mm is not None:
        measured_plane, _skipped_again = photo_corner_residuals(
            corners.after_plane_reanchoring_mm[keep],
            targets,
            photo_of_corner,
            min_corners=min_corners,
            photo_indices=not_rejected,
        )
        summary["after_plane_reanchoring"] = _median_block(measured_plane)
        summary["per_photo_after_plane_reanchoring"] = measured_plane
    return summary


def _median_block(per_photo: list[dict[str, Any]]) -> dict[str, float] | None:
    if not per_photo:
        return None
    rms = np.array([photo["rms_mm"] for photo in per_photo])
    in_plane = np.array([photo["in_plane_rms_mm"] for photo in per_photo])
    height = np.array([photo["height_mean_mm"] for photo in per_photo])
    after = np.array([photo["rms_after_rigid_fit_mm"] for photo in per_photo])
    shift = np.array([photo["shift_at_reference_mm"] for photo in per_photo])
    return {
        "rms_median_mm": float(np.median(rms)),
        "rms_p95_mm": float(np.percentile(rms, 95)),
        "rms_max_mm": float(rms.max()),
        "in_plane_rms_median_mm": float(np.median(in_plane)),
        # Each photo's signed mean z residual says whether that photo sits high or low as a
        # whole. Its magnitude is summarised two ways, because the median alone hides the few
        # photos that are still well off after R-30.
        "height_mean_abs_median_mm": float(np.median(np.abs(height))),
        "height_mean_abs_max_mm": float(np.abs(height).max()),
        "rms_after_rigid_fit_median_mm": float(np.median(after)),
        "shift_at_reference_median_mm": float(np.median(shift)),
        "shift_at_reference_max_mm": float(shift.max()),
    }


# --- per-part deviation summary -----------------------------------------------------------------


def _cad_scene(
    cad_vertices_mm: np.ndarray, cad_faces: np.ndarray
) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(cad_vertices_mm, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(cad_faces, dtype=np.uint32)),
    )
    return scene


def _to_cad_frame(vertices_mm: np.ndarray, cad_to_model: np.ndarray) -> np.ndarray:
    model_to_cad = np.linalg.inv(np.asarray(cad_to_model, dtype=np.float64))
    vertices = np.asarray(vertices_mm, dtype=np.float64)
    return vertices @ model_to_cad[:3, :3].T + model_to_cad[:3, 3]


def classify_by_cad_face(
    vertices_mm: np.ndarray,
    cad_vertices_mm: np.ndarray,
    cad_faces: np.ndarray,
    cad_to_model: np.ndarray,
) -> np.ndarray:
    """Bin each model vertex by the CAD face nearest to it: top, wall, or other.

    Done in the CAD's own frame (the model vertices are taken back through `cad_to_model`), so
    neither the part's yaw on the board nor the board frame's own z-points-into-the-table
    convention changes which face counts as "up".
    """
    cad_vertices = np.asarray(cad_vertices_mm, dtype=np.float64)
    faces = np.asarray(cad_faces, dtype=np.int64)
    vertices_cad = _to_cad_frame(vertices_mm, cad_to_model)
    if len(vertices_cad) == 0:
        return np.empty(0, dtype=object)

    scene = _cad_scene(cad_vertices, faces)
    nearest = scene.compute_closest_points(
        o3d.core.Tensor(vertices_cad.astype(np.float32))
    )
    face_index = nearest["primitive_ids"].numpy().astype(np.int64)

    triangles = cad_vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    # A zero-area CAD triangle has no normal; it is binned "other" rather than divided by zero.
    normal_z = np.where(
        lengths[:, 0] > 0.0, normals[:, 2] / np.maximum(lengths[:, 0], 1e-300), 0.0
    )

    surface = np.full(len(vertices_cad), SURFACE_OTHER, dtype=object)
    vertex_normal_z = normal_z[face_index]
    surface[vertex_normal_z > TOP_FACE_MIN_NORMAL_Z] = SURFACE_TOP
    surface[np.abs(vertex_normal_z) < WALL_FACE_MAX_ABS_NORMAL_Z] = SURFACE_WALL
    return surface


def distance_outside_cad_mm(
    vertices_mm: np.ndarray,
    cad_vertices_mm: np.ndarray,
    cad_faces: np.ndarray,
    cad_to_model: np.ndarray,
) -> np.ndarray:
    """Signed distance from each model vertex to the registered CAD: positive outside it."""
    vertices_cad = _to_cad_frame(vertices_mm, cad_to_model)
    if len(vertices_cad) == 0:
        return np.empty(0, dtype=np.float64)
    scene = _cad_scene(cad_vertices_mm, cad_faces)
    return (
        scene.compute_signed_distance(o3d.core.Tensor(vertices_cad.astype(np.float32)))
        .numpy()
        .astype(np.float64)
    )


def _spread(values_mm: np.ndarray) -> dict[str, float | int] | None:
    values = np.asarray(values_mm, dtype=np.float64)
    if len(values) == 0:
        return None
    median = float(np.median(values))
    return {
        "count": len(values),
        "median_mm": median,
        "p05_mm": float(np.percentile(values, 5)),
        "p95_mm": float(np.percentile(values, 95)),
        "mad_mm": float(np.median(np.abs(values - median))),
        "abs_p95_mm": float(np.percentile(np.abs(values), 95)),
        "abs_max_mm": float(np.abs(values).max()),
    }


def pose_yaw_deg(cad_to_model: np.ndarray) -> float:
    """The CAD pose's rotation about the board normal, for comparing two runs' poses."""
    transform = np.asarray(cad_to_model, dtype=np.float64)
    return float(math.degrees(math.atan2(transform[1, 0], transform[0, 0])))


def summarise_part(measure: dict[str, np.ndarray]) -> dict[str, Any]:
    """One part's deviation figures, from a `measure.npz` loaded as a dict.

    "Measured" follows I-06 exactly: `deviation_measured > 0.5`. "Unmeasured" is Observed surface
    I-06 left out -- under D-086 that is surface outside the CAD's outline.
    """
    observed = np.asarray(measure["observed"]).astype(bool)
    measured = np.asarray(measure["deviation_measured"]) > 0.5
    deviation = np.asarray(measure["deviation_mm"], dtype=np.float64)
    vertices = np.asarray(measure["vertices"], dtype=np.float64)
    if not (len(observed) == len(measured) == len(deviation) == len(vertices)):
        raise ValueError("measure arrays must all have one entry per vertex.")

    surface = classify_by_cad_face(
        vertices, measure["cad_vertices"], measure["cad_faces"], measure["cad_to_model"]
    )
    distance = distance_outside_cad_mm(
        vertices, measure["cad_vertices"], measure["cad_faces"], measure["cad_to_model"]
    )
    observed_count = int(np.count_nonzero(observed))
    measured_count = int(np.count_nonzero(measured))
    unmeasured_observed = int(np.count_nonzero(observed & ~measured))
    stray = int(np.count_nonzero(observed & (distance > STRAY_DISTANCE_MM)))
    return {
        "vertex_count": len(vertices),
        "observed_count": observed_count,
        "measured_count": measured_count,
        "unmeasured_observed_share": (
            unmeasured_observed / observed_count if observed_count else None
        ),
        "stray_observed_share": stray / observed_count if observed_count else None,
        "top": _spread(deviation[measured & (surface == SURFACE_TOP)]),
        "wall": _spread(deviation[measured & (surface == SURFACE_WALL)]),
        "worst_reading_mm": (
            float(np.abs(deviation[measured]).max()) if measured_count else None
        ),
        "pose_yaw_deg": pose_yaw_deg(measure["cad_to_model"]),
    }


def tolerance_sweep(
    measure: dict[str, np.ndarray],
    tolerances_mm: tuple[float, ...] = DEFAULT_TOLERANCES_MM,
) -> dict[str, Any]:
    """Try v1's verdict rule on this part at each tolerance, for one non-critical `face` feature.

    Orientation ambiguity is deliberately left out (`orientation_ambiguous=False`): this asks
    what the *measurement* supports, and the ambiguity flag is reported beside it, not folded
    in. So this is not a service verdict, and the review says so wherever it quotes it. A
    non-critical feature is used so the sweep is not cut short by I-10's intake gate.
    """
    from inspection.verdict import compute_verdict

    observed = np.asarray(measure["observed"]).astype(bool)
    labels = np.where(observed, LABEL_OBSERVED, LABEL_UNOBSERVED).astype(object)
    board_plane = tuple(float(value) for value in measure["board_plane"])
    verdicts: dict[float, str] = {}
    for tolerance in tolerances_mm:
        spec = {
            "features": [
                {
                    "name": "part",
                    "kind": "face",
                    "tolerance_mm": float(tolerance),
                    "critical": False,
                }
            ]
        }
        result = compute_verdict(
            measure["deviation_mm"],
            measure["deviation_measured"],
            labels,
            measure["sigma_mm"],
            measure["vertices"],
            board_plane,
            spec,
        )
        verdicts[float(tolerance)] = result["verdict"]

    passing = [
        tolerance for tolerance, verdict in verdicts.items() if verdict == "Pass"
    ]
    failing = [
        tolerance for tolerance, verdict in verdicts.items() if verdict == "Fail"
    ]
    return {
        "verdicts": verdicts,
        "first_pass_mm": min(passing) if passing else None,
        "last_fail_mm": max(failing) if failing else None,
    }


def verdict_pool_sizes(measure: dict[str, np.ndarray]) -> dict[str, int]:
    """How the verdict's own height split lines up with the CAD faces, for this part.

    The verdict puts the top `100 - TOP_REGION_PERCENTILE` percent of measured vertices, by
    height above the board, in its out-of-plane pool, and everything else in its in-plane pool.
    This counts how many vertices sitting on a CAD *top* face end up in the in-plane pool, which
    is the mismatch the review reports.
    """
    from inspection.verdict import TOP_REGION_PERCENTILE

    observed = np.asarray(measure["observed"]).astype(bool)
    measured = np.asarray(measure["deviation_measured"]) > 0.5
    judged = observed & measured
    if not judged.any():
        return {"out_of_plane": 0, "in_plane": 0, "top_face_in_in_plane_pool": 0}
    vertices = np.asarray(measure["vertices"], dtype=np.float64)[judged]
    surface = classify_by_cad_face(
        vertices, measure["cad_vertices"], measure["cad_faces"], measure["cad_to_model"]
    )
    height = compute_height_above_board(
        vertices, tuple(float(value) for value in measure["board_plane"])
    )
    out_of_plane = height >= np.percentile(height, TOP_REGION_PERCENTILE)
    return {
        "out_of_plane": int(np.count_nonzero(out_of_plane)),
        "in_plane": int(np.count_nonzero(~out_of_plane)),
        "top_face_in_in_plane_pool": int(
            np.count_nonzero(~out_of_plane & (surface == SURFACE_TOP))
        ),
    }


# --- run: one capture through the real pipeline and the real measurement ------------------------


def _current_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def run_part(capture_dir: Path, cad_path: Path, out_dir: Path) -> dict[str, Any]:
    """Reconstruct one capture and measure it against `cad_path`, exactly as the service would.

    Imports are local on purpose: `recon.service` pulls in FastAPI and, through the pipeline,
    torch, and neither `report` nor `corners` (nor this module's tests) should need them.
    """
    capture_dir = Path(capture_dir)
    cad_path = Path(cad_path)
    out_dir = Path(out_dir)
    # Checked before anything heavy is imported or run: a mixed output folder is refused in
    # the first second, not after a minute of reconstruction.
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already has files in it. Pick a new --out or delete it deliberately; "
            "overwriting silently would let outputs from two runs get mixed."
        )

    from inspection import registration
    from inspection.cad import load_mesh
    from inspection.deviation import compute_deviation

    from recon.mesh_cleanup import read_glb
    from recon.part_segmentation import BoardSurface
    from recon.provenance import colors_to_labels
    from recon.service import ReconstructionFailure, run_reconstruction_pipeline
    from recon.uncertainty import read_sigma_npz

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = out_dir / "artifacts"

    started = time.perf_counter()
    try:
        pipeline = run_reconstruction_pipeline(capture_dir, artifact_dir)
    except ReconstructionFailure as exc:
        # The service turns this into a failed session; here it is the whole result, so it is
        # reported with its own code and message rather than as a bare traceback.
        raise ValueError(f"reconstruction failed: {exc.as_dict()}") from exc
    pipeline_s = time.perf_counter() - started

    cad_vertices, cad_faces, cad_units = load_mesh(cad_path)
    cad_mesh = (cad_vertices, cad_faces, cad_units)
    vertices, faces, colors = read_glb(Path(pipeline["glb_path"]))
    if colors is None:
        raise ValueError(
            f"{pipeline['glb_path']} has no provenance colours; nothing says which vertices "
            "are Observed."
        )
    labels = colors_to_labels(colors)
    model = (vertices, faces, labels)
    fusion_report = json.loads(
        (artifact_dir / "fusion" / "tsdf_fusion.json").read_text(encoding="utf-8")
    )
    board_surface = BoardSurface(**fusion_report["board_surface"])

    started = time.perf_counter()
    candidates = registration.global_register(model, cad_mesh)
    ambiguity = registration.check_ambiguity(candidates, model, cad_mesh)
    registration_s = time.perf_counter() - started
    best = ambiguity.best
    deviation_mm, deviation_measured = compute_deviation(
        model, cad_mesh, best.transformation, board_surface
    )
    sigma_mm = read_sigma_npz(artifact_dir / "uncertainty" / "sigma_mm.npz")["sigma_mm"]

    np.savez_compressed(
        out_dir / MEASURE_NPZ,
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        observed=(labels == LABEL_OBSERVED),
        deviation_mm=np.asarray(deviation_mm, dtype=np.float64),
        deviation_measured=np.asarray(deviation_measured, dtype=np.float64),
        sigma_mm=np.asarray(sigma_mm, dtype=np.float64),
        cad_to_model=np.asarray(best.transformation, dtype=np.float64),
        cad_vertices=np.asarray(cad_vertices, dtype=np.float64),
        cad_faces=np.asarray(cad_faces, dtype=np.int64),
        board_plane=np.array(
            [
                board_surface.slope_x,
                board_surface.slope_y,
                board_surface.z_at_origin_mm,
            ]
        ),
    )
    record = {
        "story": "P-41",
        "commit": _current_commit(),
        "capture_dir": str(capture_dir),
        "frame_count": json.loads(
            (artifact_dir / "reconstruction" / "run.json").read_text(encoding="utf-8")
        ).get("frame_count"),
        "cad": cad_path.name,
        "pipeline_s": pipeline_s,
        "registration_s": registration_s,
        "stage_timings_s": pipeline.get("stage_timings_s"),
        "observed_fraction": pipeline.get("observed_fraction"),
        "registration": {
            "orientation_ambiguous": bool(ambiguity.orientation_ambiguous),
            "coverage_ratio": ambiguity.coverage_ratio,
            "best_fitness": float(best.fitness),
            "best_inlier_rmse_mm": float(best.inlier_rmse),
            "pose_yaw_deg": pose_yaw_deg(best.transformation),
        },
    }
    (out_dir / MEASURE_JSON).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


# --- report and corners: printing ---------------------------------------------------------------


def load_measure(out_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    out_dir = Path(out_dir)
    npz_path = out_dir / MEASURE_NPZ
    json_path = out_dir / MEASURE_JSON
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(
            f"{out_dir} has no {MEASURE_NPZ}/{MEASURE_JSON}; run the `run` subcommand first."
        )
    with np.load(npz_path) as data:
        measure = {key: data[key] for key in data.files}
    record = json.loads(json_path.read_text(encoding="utf-8"))
    return measure, record


def _fmt(value: float | None, width: int = 6, digits: int = 2) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value:{width}.{digits}f}"


def report_parts(out_dirs: list[Path]) -> list[dict[str, Any]]:
    """Print the per-part table and return each part's figures."""
    rows: list[dict[str, Any]] = []
    header = (
        f"{'part':<14} {'observed':>8} {'unmeasured':>10} {'stray':>6} "
        f"{'top |95%|':>9} {'wall median':>11} {'wall |95%|':>10} {'worst':>6} "
        f"{'first Pass':>10} {'ambiguous':>9} {'fitness':>7} {'yaw':>7}"
    )
    print(header)
    for out_dir in out_dirs:
        measure, record = load_measure(out_dir)
        summary = summarise_part(measure)
        sweep = tolerance_sweep(measure)
        pools = verdict_pool_sizes(measure)
        registration_record = record.get("registration", {})
        top = summary["top"] or {}
        wall = summary["wall"] or {}
        unmeasured = summary["unmeasured_observed_share"]
        stray = summary["stray_observed_share"]
        print(
            f"{Path(out_dir).name:<14} {summary['observed_count']:>8d} "
            f"{'-' if unmeasured is None else f'{100 * unmeasured:.0f}%':>10} "
            f"{'-' if stray is None else f'{100 * stray:.0f}%':>6} "
            f"{_fmt(top.get('abs_p95_mm'), 9)} {_fmt(wall.get('median_mm'), 11)} "
            f"{_fmt(wall.get('abs_p95_mm'), 10)} {_fmt(summary['worst_reading_mm'])} "
            f"{_fmt(sweep['first_pass_mm'], 10, 1)} "
            f"{registration_record.get('orientation_ambiguous', '-')!s:>9} "
            f"{_fmt(registration_record.get('best_fitness'), 7)} "
            f"{_fmt(summary['pose_yaw_deg'], 7, 1)}"
        )
        rows.append(
            {
                "part": Path(out_dir).name,
                "summary": summary,
                "sweep": sweep,
                "pools": pools,
                "registration": registration_record,
                "commit": record.get("commit"),
            }
        )
    return rows


def report_corners(out_dirs: list[Path], *, min_corners: int) -> list[dict[str, Any]]:
    """Print the per-capture corner table and return each capture's figures."""
    rows: list[dict[str, Any]] = []
    print(
        f"{'capture':<14} {'photos':>6} {'measured':>8} {'skipped':>7} {'rejected':>8} | "
        f"{'global: rms median':>18} {'p95':>6} {'max':>6} | "
        f"{'after R-30: in-plane':>20} {'|mean z|':>8} {'max':>6} {'shift':>6} {'max':>6} | "
        f"{'rigid fit leaves':>16}"
    )
    for out_dir in out_dirs:
        corners = load_lifted_corners(Path(out_dir) / "artifacts")
        summary = summarise_corner_residuals(corners, min_corners=min_corners)
        after_global = summary["after_global_alignment"] or {}
        after_plane = summary["after_plane_reanchoring"] or {}
        # "what a rigid fit leaves" is the same whichever stage it starts from, to within the
        # corner noise; the later stage is reported when the run has one.
        leaves = (after_plane or after_global).get("rms_after_rigid_fit_median_mm")
        print(
            f"{Path(out_dir).name:<14} {summary['photo_count']:>6d} "
            f"{summary['measured_photo_count']:>8d} {summary['skipped_photo_count']:>7d} "
            f"{summary['rejected_photo_count']:>8d} | "
            f"{_fmt(after_global.get('rms_median_mm'), 18)} "
            f"{_fmt(after_global.get('rms_p95_mm'))} {_fmt(after_global.get('rms_max_mm'))} | "
            f"{_fmt(after_plane.get('in_plane_rms_median_mm'), 20)} "
            f"{_fmt(after_plane.get('height_mean_abs_median_mm'), 8)} "
            f"{_fmt(after_plane.get('height_mean_abs_max_mm'))} "
            f"{_fmt(after_plane.get('shift_at_reference_median_mm'))} "
            f"{_fmt(after_plane.get('shift_at_reference_max_mm'))} | "
            f"{_fmt(leaves, 16)}"
        )
        rows.append({"capture": Path(out_dir).name, **summary})
    return rows


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P-41: measure a part against its own CAD, and each photo against the board."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser(
        "run", help="reconstruct one capture and measure it against a CAD"
    )
    run.add_argument("capture_dir", type=Path)
    run.add_argument("--cad", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)

    report = subcommands.add_parser("report", help="print the per-part table")
    report.add_argument("out_dirs", type=Path, nargs="+")
    report.add_argument(
        "--json", type=Path, default=None, help="also write the figures here"
    )

    corners = subcommands.add_parser(
        "corners", help="print per-capture corner residuals"
    )
    corners.add_argument("out_dirs", type=Path, nargs="+")
    corners.add_argument(
        "--min-corners", type=int, default=DEFAULT_MIN_CORNERS_PER_PHOTO
    )
    corners.add_argument(
        "--json", type=Path, default=None, help="also write the figures here"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.command == "run":
            record = run_part(args.capture_dir, args.cad, args.out)
            print(
                f"MEASURE PART: done - {args.out} "
                f"(fitness {record['registration']['best_fitness']:.2f}, "
                f"ambiguous {record['registration']['orientation_ambiguous']})"
            )
            return 0
        if args.command == "report":
            rows = report_parts(args.out_dirs)
        else:
            rows = report_corners(args.out_dirs, min_corners=args.min_corners)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"MEASURE PART: FAIL - {exc}", file=sys.stderr)
        return 1
    if args.json is not None:
        args.json.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
