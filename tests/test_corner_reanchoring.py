"""R-44: each photo moved rigidly onto its own ChArUco corners (recon.corner_reanchoring).

All synthetic. The real-capture before/after record is a written check in stories/R/R-44.md.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from recon import corner_reanchoring as cr
from recon import tsdf_fusion
from recon.board_pose import make_board
from recon.metric_alignment import project_points_to_camera

# The D-034 board's 35 interior corners, in board-frame mm (z = 0 on the board surface).
BOARD_CORNERS_MM = np.asarray(make_board().getChessboardCorners(), dtype=np.float64)
BOARD_CENTRE_MM = np.array([80.0, 60.0, 0.0])


def _rotation_about_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _rotation_about_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ]
    )


def _drift(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Where a photo with this pose error puts true board-frame points."""
    return points @ rotation.T + translation


def _scene(frame_drifts, corner_ids_per_frame=None, extra_points=None):
    """Points, poses and lifted corners for photos that each carry their own planted drift.

    Each photo's "points" are the board corners themselves plus `extra_points` (a part above
    the board), all moved by that photo's drift -- the same error R-44 is meant to undo.
    Returns (board_frame_points (S,1,P,3), board_extrinsic (S,3,4), corner_points, targets,
    frame_of_corner, true_points (P,3)).
    """
    if extra_points is None:
        extra_points = np.array([[80.0, 60.0, -25.0]])
    true_points = np.vstack([BOARD_CORNERS_MM, extra_points])
    frame_points = []
    extrinsics = []
    corner_chunks = []
    target_chunks = []
    frame_chunks = []
    for frame_index, (rotation, translation) in enumerate(frame_drifts):
        frame_points.append(_drift(true_points, rotation, translation))
        extrinsics.append(np.eye(3, 4))
        ids = (
            np.arange(len(BOARD_CORNERS_MM))
            if corner_ids_per_frame is None
            else np.asarray(corner_ids_per_frame[frame_index], dtype=int)
        )
        corner_chunks.append(_drift(BOARD_CORNERS_MM[ids], rotation, translation))
        target_chunks.append(BOARD_CORNERS_MM[ids])
        frame_chunks.append(np.full(len(ids), frame_index))
    return (
        np.stack(frame_points)[:, None, :, :],
        np.stack(extrinsics),
        np.concatenate(corner_chunks),
        np.concatenate(target_chunks),
        np.concatenate(frame_chunks),
        true_points,
    )


def test_a_planted_sideways_shift_and_rotation_come_back_onto_the_board():
    # Photo 0 is already right. Photo 1 is shifted sideways and turned about the board normal
    # (the three degrees of freedom R-30 leaves alone). Photo 2 also carries tilt and height.
    drifts = [
        (np.eye(3), np.zeros(3)),
        (_rotation_about_z(1.5), np.array([2.1, -1.4, 0.0])),
        (_rotation_about_z(-0.8) @ _rotation_about_x(0.6), np.array([-1.2, 0.9, 0.7])),
    ]
    points, extrinsic, corners, targets, frame_of_corner, true_points = _scene(drifts)

    corrected, _extrinsic, reports = cr.reanchor_frames_to_corners(
        points, extrinsic, corners, targets, frame_of_corner, BOARD_CENTRE_MM
    )

    for frame_index, (rotation, translation) in enumerate(drifts):
        # Every point of the photo, the part above the board included, is back where it truly is.
        assert corrected[frame_index, 0] == pytest.approx(true_points, abs=1e-9)
        report = reports[frame_index]
        assert report.status == cr.STATUS_CORRECTED
        assert report.corner_count == len(BOARD_CORNERS_MM)
        assert report.rms_after_mm == pytest.approx(0.0, abs=1e-9)
        assert report.worst_corner_after_mm == pytest.approx(0.0, abs=1e-9)
        # The recovered transform is the planted drift's inverse.
        assert np.array(report.rotation) == pytest.approx(rotation.T, abs=1e-12)
        assert np.array(report.translation_mm) == pytest.approx(
            -rotation.T @ translation, abs=1e-9
        )

    assert reports[1].rotation_deg == pytest.approx(1.5, abs=1e-9)
    # Photo 1's correction carries the board centre back by exactly its planted in-plane error.
    planted_centre_error = (
        _drift(BOARD_CENTRE_MM[None], *drifts[1])[0] - BOARD_CENTRE_MM
    )
    assert reports[1].shift_at_reference_mm == pytest.approx(
        float(np.linalg.norm(planted_centre_error[:2])), abs=1e-9
    )
    assert reports[1].height_shift_at_reference_mm == pytest.approx(0.0, abs=1e-9)
    assert reports[2].height_shift_at_reference_mm != pytest.approx(0.0, abs=1e-3)


def test_corners_that_already_agree_give_no_correction():
    points, extrinsic, corners, targets, frame_of_corner, _true = _scene(
        [(np.eye(3), np.zeros(3))]
    )

    corrected, corrected_extrinsic, reports = cr.reanchor_frames_to_corners(
        points, extrinsic, corners, targets, frame_of_corner, BOARD_CENTRE_MM
    )

    assert corrected == pytest.approx(points, abs=1e-9)
    assert corrected_extrinsic == pytest.approx(extrinsic, abs=1e-9)
    assert reports[0].rms_before_mm == pytest.approx(0.0, abs=1e-9)
    assert reports[0].shift_at_reference_mm == pytest.approx(0.0, abs=1e-9)
    assert reports[0].rotation_deg == pytest.approx(0.0, abs=1e-6)


def test_corrected_points_and_corrected_pose_stay_consistent_with_each_other():
    # The photo's depth measurement doesn't change, so a point it saw must project to the same
    # pixel at the same depth whether expressed as (original point, original pose) or
    # (corrected point, corrected pose). Same check as R-30's round-trip test.
    rotation_error = _rotation_about_z(2.0) @ _rotation_about_x(-0.7)
    translation_error = np.array([1.7, -2.2, 0.5])
    points, _identity, corners, targets, frame_of_corner, _true = _scene(
        [(rotation_error, translation_error)]
    )
    camera_rotation = _rotation_about_x(180.0 - 35.0) @ _rotation_about_z(20.0)
    camera_translation = np.array([-70.0, 40.0, 260.0])
    extrinsic = np.concatenate([camera_rotation, camera_translation[:, None]], axis=1)[
        None
    ]
    intrinsic = np.array([[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]])

    corrected, corrected_extrinsic, _reports = cr.reanchor_frames_to_corners(
        points, extrinsic, corners, targets, frame_of_corner, BOARD_CENTRE_MM
    )

    original = project_points_to_camera(
        points[0, 0], camera_rotation, camera_translation, intrinsic
    )
    after = project_points_to_camera(
        corrected[0, 0],
        corrected_extrinsic[0, :, :3],
        corrected_extrinsic[0, :, 3],
        intrinsic,
    )
    for before_values, after_values in zip(original, after):
        assert after_values == pytest.approx(before_values, abs=1e-6)


def test_a_photo_with_too_few_corners_is_left_alone_and_reported():
    drift = (_rotation_about_z(1.0), np.array([2.0, 1.0, 0.0]))
    points, extrinsic, corners, targets, frame_of_corner, _true = _scene(
        [drift, drift], corner_ids_per_frame=[np.arange(35), np.array([0, 1, 7, 8, 14])]
    )

    corrected, corrected_extrinsic, reports = cr.reanchor_frames_to_corners(
        points, extrinsic, corners, targets, frame_of_corner, BOARD_CENTRE_MM
    )

    assert reports[0].status == cr.STATUS_CORRECTED
    assert reports[1].status == cr.STATUS_SKIPPED
    assert reports[1].corner_count == 5
    assert reports[1].skip_reason == "fewer than 8 corners (5)"
    # The before-residual is still measured, so a skipped photo's error is visible.
    assert reports[1].rms_before_mm > 1.0
    assert reports[1].rms_after_mm is None
    assert corrected[1] == pytest.approx(points[1])
    assert corrected_extrinsic[1] == pytest.approx(extrinsic[1])


def test_the_corner_minimum_is_inclusive_and_configurable():
    drift = (_rotation_about_z(1.0), np.array([2.0, 1.0, 0.0]))
    eight_corners = np.array([0, 1, 2, 7, 8, 9, 14, 15])
    points, extrinsic, corners, targets, frame_of_corner, true_points = _scene(
        [drift], corner_ids_per_frame=[eight_corners]
    )

    corrected, _extrinsic, reports = cr.reanchor_frames_to_corners(
        points, extrinsic, corners, targets, frame_of_corner, BOARD_CENTRE_MM
    )
    assert reports[0].status == cr.STATUS_CORRECTED
    assert corrected[0, 0] == pytest.approx(true_points, abs=1e-9)

    _corrected, _extrinsic, stricter = cr.reanchor_frames_to_corners(
        points,
        extrinsic,
        corners,
        targets,
        frame_of_corner,
        BOARD_CENTRE_MM,
        min_corners_per_frame=9,
    )
    assert stricter[0].status == cr.STATUS_SKIPPED


def test_a_photo_r38_rejected_is_left_alone_even_with_plenty_of_corners():
    drift = (_rotation_about_z(1.0), np.array([2.0, 1.0, 0.0]))
    points, extrinsic, corners, targets, frame_of_corner, _true = _scene([drift, drift])

    corrected, corrected_extrinsic, reports = cr.reanchor_frames_to_corners(
        points,
        extrinsic,
        corners,
        targets,
        frame_of_corner,
        BOARD_CENTRE_MM,
        skip_frame_indices=frozenset({1}),
    )

    assert reports[0].status == cr.STATUS_CORRECTED
    assert reports[1].status == cr.STATUS_SKIPPED
    assert reports[1].skip_reason == cr.SKIP_REASON_REJECTED
    assert reports[1].corner_count == 35
    assert corrected[1] == pytest.approx(points[1])
    assert corrected_extrinsic[1] == pytest.approx(extrinsic[1])


def test_a_photo_with_no_corners_is_reported_not_dropped():
    drift = (_rotation_about_z(1.0), np.array([2.0, 1.0, 0.0]))
    points, extrinsic, corners, targets, frame_of_corner, _true = _scene(
        [drift, drift], corner_ids_per_frame=[np.arange(35), np.array([], dtype=int)]
    )

    _corrected, _extrinsic, reports = cr.reanchor_frames_to_corners(
        points, extrinsic, corners, targets, frame_of_corner, BOARD_CENTRE_MM
    )

    assert [report.frame_index for report in reports] == [0, 1]
    assert reports[1].status == cr.STATUS_SKIPPED
    assert reports[1].corner_count == 0
    assert reports[1].skip_reason == cr.SKIP_REASON_NO_CORNERS
    assert reports[1].rms_before_mm is None


def test_a_photo_whose_corners_are_all_on_one_row_is_skipped_not_guessed():
    drift = (_rotation_about_z(1.0), np.array([2.0, 1.0, 0.0]))
    one_row = np.arange(7)  # corner ids 0..6 are the board's first row: collinear
    points, extrinsic, corners, targets, frame_of_corner, _true = _scene(
        [drift], corner_ids_per_frame=[one_row]
    )

    corrected, _extrinsic, reports = cr.reanchor_frames_to_corners(
        points,
        extrinsic,
        corners,
        targets,
        frame_of_corner,
        BOARD_CENTRE_MM,
        min_corners_per_frame=3,
    )

    assert reports[0].status == cr.STATUS_SKIPPED
    assert reports[0].skip_reason.startswith("no usable rigid fit")
    assert corrected == pytest.approx(points)


def test_one_corner_40mm_off_does_not_drag_the_photo():
    # @TabeenRaoof's review on #442: on g02_06 a plain fit let one corner 41.7 mm off pull its
    # photo's correction up to 2.2 mm at the board centre. Trimmed, the photo must come back to
    # within 0.3 mm of its planted correction, and the bad corner must still be reported.
    drift = (_rotation_about_z(1.2), np.array([2.0, -1.5, 0.0]))
    points, extrinsic, corners, targets, frame_of_corner, true_points = _scene([drift])
    corners = corners.copy()
    corners[17] += np.array([40.0, 0.0, 0.0])

    corrected, _extrinsic, reports = cr.reanchor_frames_to_corners(
        points, extrinsic, corners, targets, frame_of_corner, BOARD_CENTRE_MM
    )

    report = reports[0]
    assert report.status == cr.STATUS_CORRECTED
    assert report.corner_count == 35
    assert report.corners_used == 34
    assert report.rms_after_mm == pytest.approx(0.0, abs=1e-9)
    assert report.worst_corner_after_mm == pytest.approx(40.0, abs=0.5)
    # Every point of the photo, the part above the board included, is back where it truly is.
    distances = np.linalg.norm(corrected[0, 0] - true_points, axis=1)
    assert distances.max() < 0.3


def test_a_plain_fit_would_have_been_dragged_by_that_corner():
    # The same scene through fit_rigid alone, to show the test above is testing something:
    # the bad corner moves an untrimmed fit well past 0.3 mm at the board centre.
    drift = (_rotation_about_z(1.2), np.array([2.0, -1.5, 0.0]))
    _points, _extrinsic, corners, targets, _frame_of_corner, _true = _scene([drift])
    corners = corners.copy()
    corners[17] += np.array([40.0, 0.0, 0.0])

    rotation, translation = cr.fit_rigid(corners, targets)
    planted_rotation, planted_translation = drift
    true_centre_seen = planted_rotation @ BOARD_CENTRE_MM + planted_translation
    centre_error = rotation @ true_centre_seen + translation - BOARD_CENTRE_MM
    assert np.linalg.norm(centre_error) > 0.3


def test_trimming_stops_at_the_floor_when_every_corner_agrees_closely():
    # Corners that all agree to well under the 0.5 mm floor are noise, not outliers: none dropped.
    rng = np.random.default_rng(7)
    points, extrinsic, corners, targets, frame_of_corner, _true = _scene(
        [(np.eye(3), np.zeros(3))]
    )
    noisy = corners + rng.normal(scale=0.05, size=corners.shape)

    _corrected, _extrinsic, reports = cr.reanchor_frames_to_corners(
        points, extrinsic, noisy, targets, frame_of_corner, BOARD_CENTRE_MM
    )

    assert reports[0].corners_used == reports[0].corner_count == 35


def test_a_photo_trimming_leaves_with_too_few_corners_is_left_alone():
    # Nine corners, two of them far off: trimming keeps seven, under the minimum of eight, so the
    # photo is not corrected from what's left.
    drift = (_rotation_about_z(1.0), np.array([2.0, 1.0, 0.0]))
    nine = np.array([0, 1, 2, 7, 8, 9, 14, 15, 16])
    points, extrinsic, corners, targets, frame_of_corner, _true = _scene(
        [drift], corner_ids_per_frame=[nine]
    )
    corners = corners.copy()
    corners[[0, 8]] += np.array([[25.0, 0.0, 0.0], [0.0, -30.0, 0.0]])

    corrected, corrected_extrinsic, reports = cr.reanchor_frames_to_corners(
        points, extrinsic, corners, targets, frame_of_corner, BOARD_CENTRE_MM
    )

    assert reports[0].status == cr.STATUS_SKIPPED
    assert "leaves 7 of 9, fewer than 8" in reports[0].skip_reason
    assert corrected == pytest.approx(points)
    assert corrected_extrinsic == pytest.approx(extrinsic)


def test_the_rule_cannot_trim_when_bad_corners_are_a_large_share_and_the_report_shows_it():
    # A known limit of P-41's median-based rule, recorded rather than hidden: with four of ten
    # corners far off, the first fit is dragged so far that 3x the median residual exceeds every
    # residual, nothing is dropped, and the photo is corrected badly. The report's rms_after_mm
    # is what makes such a photo visible in the measurement.
    drift = (_rotation_about_z(1.0), np.array([2.0, 1.0, 0.0]))
    ten = np.array([0, 1, 2, 3, 7, 8, 9, 10, 14, 15])
    points, extrinsic, corners, targets, frame_of_corner, _true = _scene(
        [drift], corner_ids_per_frame=[ten]
    )
    corners = corners.copy()
    corners[[0, 3, 6, 9]] += np.array(
        [[25.0, 0.0, 0.0], [0.0, -30.0, 0.0], [-20.0, 10.0, 0.0], [0.0, 0.0, 35.0]]
    )

    _corrected, _extrinsic, reports = cr.reanchor_frames_to_corners(
        points, extrinsic, corners, targets, frame_of_corner, BOARD_CENTRE_MM
    )

    assert reports[0].status == cr.STATUS_CORRECTED
    assert reports[0].corners_used == 10
    assert reports[0].rms_after_mm > 10.0


def test_reports_serialise_to_plain_json_types():
    points, extrinsic, corners, targets, frame_of_corner, _true = _scene(
        [(np.eye(3), np.zeros(3))]
    )
    _corrected, _extrinsic, reports = cr.reanchor_frames_to_corners(
        points, extrinsic, corners, targets, frame_of_corner, BOARD_CENTRE_MM
    )

    round_tripped = json.loads(json.dumps(reports[0].as_report()))
    assert round_tripped["status"] == "corrected"
    assert len(round_tripped["rotation"]) == 3


def test_inputs_that_cannot_be_matched_are_refused():
    points, extrinsic, corners, targets, frame_of_corner, _true = _scene(
        [(np.eye(3), np.zeros(3))]
    )
    with pytest.raises(ValueError, match="outside 0..0"):
        cr.reanchor_frames_to_corners(
            points, extrinsic, corners, targets, frame_of_corner + 1, BOARD_CENTRE_MM
        )
    with pytest.raises(ValueError, match="one entry per corner"):
        cr.reanchor_frames_to_corners(
            points, extrinsic, corners, targets, frame_of_corner[:-1], BOARD_CENTRE_MM
        )
    with pytest.raises(ValueError, match="at least 3"):
        cr.reanchor_frames_to_corners(
            points,
            extrinsic,
            corners,
            targets,
            frame_of_corner,
            BOARD_CENTRE_MM,
            min_corners_per_frame=2,
        )
    bad = corners.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cr.reanchor_frames_to_corners(
            points, extrinsic, bad, targets, frame_of_corner, BOARD_CENTRE_MM
        )


def test_fit_rigid_never_returns_a_reflection():
    source = BOARD_CORNERS_MM
    mirrored = source * np.array([1.0, -1.0, 1.0])
    rotation, _translation = cr.fit_rigid(source, mirrored)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


# --- fuse_run end to end: a box seen by six photos, each with its own sideways error -------------
#
# A 40 x 40 x 30 mm box on the board, ray-cast into six photos taken from all round at 45 degrees.
# Each photo's pose is then given its own planted error -- a sideways shift of 1-3 mm and a turn
# of under a degree about the board's normal, the three degrees of freedom R-30 leaves alone --
# while its depth stays the true depth. That is the error P-41 measured on real captures.
#
# Only corners whose whole 4x4 pixel neighbourhood sees the board are kept. A corner next to the
# box's silhouette gets its depth interpolated between the board and the box and lifts up to
# ~11 mm wrong (found building this test); that is real-capture behaviour, but it would make this
# test about corner noise rather than about the correction.

_IMAGE_HEIGHT, _IMAGE_WIDTH = 120, 160
_CAMERA_INTRINSIC = np.array([[150.0, 0.0, 80.0], [0.0, 150.0, 60.0], [0.0, 0.0, 1.0]])
_BOX_MIN_MM = np.array([60.0, 40.0, -30.0])  # board-frame "up" is -z
_BOX_MAX_MM = np.array([100.0, 80.0, 0.0])
_AZIMUTHS_DEG = (0, 60, 120, 180, 240, 300)
_PLANTED_DRIFTS = (
    (0.8, (2.5, -1.0)),
    (-0.6, (-2.0, 2.0)),
    (0.4, (1.5, 2.5)),
    (-0.9, (-2.5, -1.5)),
    (0.5, (0.5, -2.8)),
    (-0.3, (-1.0, 1.0)),
)


def _look_at_board(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Camera-from-board extrinsic (3,4) for a camera 260 mm from the box, looking at it."""
    target = np.array([80.0, 60.0, -15.0])
    azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
    centre = target + 260.0 * np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            -math.sin(elevation),
        ]
    )
    forward = (target - centre) / np.linalg.norm(target - centre)
    right = np.cross(forward, np.array([0.0, 0.0, -1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return np.concatenate([rotation, (-rotation @ centre)[:, None]], axis=1)


def _pixel_rays_in_board_frame(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera centre and per-pixel rays, scaled so the ray parameter equals camera depth."""
    rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
    v, u = np.mgrid[0:_IMAGE_HEIGHT, 0:_IMAGE_WIDTH]
    rays_camera = np.stack(
        [
            (u - _CAMERA_INTRINSIC[0, 2]) / _CAMERA_INTRINSIC[0, 0],
            (v - _CAMERA_INTRINSIC[1, 2]) / _CAMERA_INTRINSIC[1, 1],
            np.ones(u.shape),
        ],
        axis=-1,
    )
    return -rotation.T @ translation, rays_camera @ rotation


def _board_plane_depth(extrinsic: np.ndarray) -> np.ndarray:
    centre, rays = _pixel_rays_in_board_frame(extrinsic)
    return -centre[2] / rays[..., 2]


def _render_depth(extrinsic: np.ndarray) -> np.ndarray:
    """True depth of the board plane and the box, nearest hit per pixel."""
    centre, rays = _pixel_rays_in_board_frame(extrinsic)
    depth = _board_plane_depth(extrinsic)
    depth = np.where(depth > 0, depth, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        slab_near = (_BOX_MIN_MM - centre) / rays
        slab_far = (_BOX_MAX_MM - centre) / rays
    entry = np.max(np.minimum(slab_near, slab_far), axis=-1)
    exit_ = np.min(np.maximum(slab_near, slab_far), axis=-1)
    hits_box = (entry <= exit_) & (entry > 0)
    depth = np.where(hits_box & (entry < depth), entry, depth)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def _clean_corner_pixels(
    extrinsic: np.ndarray, depth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rotation, translation = extrinsic[:, :3], extrinsic[:, 3]
    camera = BOARD_CORNERS_MM @ rotation.T + translation
    u = _CAMERA_INTRINSIC[0, 0] * camera[:, 0] / camera[:, 2] + _CAMERA_INTRINSIC[0, 2]
    v = _CAMERA_INTRINSIC[1, 1] * camera[:, 1] / camera[:, 2] + _CAMERA_INTRINSIC[1, 2]
    board_depth = _board_plane_depth(extrinsic)
    corner_ids = []
    corner_px = []
    for corner_id in range(len(BOARD_CORNERS_MM)):
        u0, v0 = int(np.floor(u[corner_id])), int(np.floor(v[corner_id]))
        if not (1 <= u0 < _IMAGE_WIDTH - 3 and 1 <= v0 < _IMAGE_HEIGHT - 3):
            continue
        window = (slice(v0 - 1, v0 + 3), slice(u0 - 1, u0 + 3))
        if np.all(np.abs(depth[window] - board_depth[window]) < 1e-6):
            corner_ids.append(corner_id)
            corner_px.append([u[corner_id], v[corner_id]])
    return np.array(corner_ids), np.array(corner_px)


def _write_box_capture(
    root,
    rejected_frame_names=(),
    scale=1.0,
    rotation=None,
    translation_mm=None,
):
    """Write the box capture as R-01/R-02/R-04 files. Returns (run_dir, board_dir,
    alignment_json, true_extrinsic (S,3,4)).

    `scale`, `rotation` and `translation_mm` are R-04's similarity (reconstruction frame to board
    mm). With anything but the identity, `poses.npz` and `depth.npz` are written in a
    reconstruction frame of their own, as R-01's always are, so the corners R-44 fits to only
    land in the board frame if they go through that similarity first."""
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    translation_mm = (
        np.zeros(3)
        if translation_mm is None
        else np.asarray(translation_mm, dtype=np.float64)
    )
    run_dir = root / "run"
    board_dir = root / "board_pose"
    run_dir.mkdir()
    board_dir.mkdir()
    true_extrinsic = np.stack(
        [_look_at_board(azimuth, 45.0) for azimuth in _AZIMUTHS_DEG]
    )
    depth = np.stack([_render_depth(extrinsic) for extrinsic in true_extrinsic])
    # The pose each photo is given carries its planted error; its depth stays true.
    given_extrinsic = np.stack(
        [
            cr.correct_extrinsic_for_moved_points(
                extrinsic,
                _rotation_about_z(yaw_deg),
                np.array([shift[0], shift[1], 0.0]),
            )
            for extrinsic, (yaw_deg, shift) in zip(true_extrinsic, _PLANTED_DRIFTS)
        ]
    )
    frame_names = np.array([f"frame_{index}.jpg" for index in range(len(depth))])
    corner_frame_index, corner_id, corner_px = [], [], []
    for frame_index, extrinsic in enumerate(true_extrinsic):
        ids, pixels = _clean_corner_pixels(extrinsic, depth[frame_index])
        corner_frame_index.extend([frame_index] * len(ids))
        corner_id.extend(ids.tolist())
        corner_px.extend(pixels.tolist())

    # Invert apply_similarity_transform_to_extrinsic: R_o = R_board @ rotation and
    # t_o = (t_board + R_board @ translation) / scale, with depth in the same arbitrary units.
    reconstruction_extrinsic = np.empty_like(given_extrinsic)
    for frame_index, board_extrinsic in enumerate(given_extrinsic):
        board_rotation, board_translation = (
            board_extrinsic[:, :3],
            board_extrinsic[:, 3],
        )
        reconstruction_extrinsic[frame_index, :, :3] = board_rotation @ rotation
        reconstruction_extrinsic[frame_index, :, 3] = (
            board_translation + board_rotation @ translation_mm
        ) / scale
    np.savez_compressed(
        run_dir / "poses.npz",
        frame_names=frame_names,
        extrinsic=reconstruction_extrinsic,
        intrinsic=np.repeat(_CAMERA_INTRINSIC[None], len(depth), axis=0),
    )
    np.savez_compressed(
        run_dir / "depth.npz", depth=depth / scale, depth_conf=np.ones_like(depth)
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "min_confidence": 0.0,
                "processed_image_size_hw": [_IMAGE_HEIGHT, _IMAGE_WIDTH],
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        board_dir / "board_poses.npz",
        frame_names=frame_names,
        pose_found=np.ones(len(frame_names), dtype=bool),
        image_size_hw=np.tile([_IMAGE_HEIGHT, _IMAGE_WIDTH], (len(frame_names), 1)),
        corner_frame_index=np.array(corner_frame_index),
        corner_id=np.array(corner_id),
        corner_px=np.array(corner_px),
    )
    alignment_json = run_dir / "metric_alignment.json"
    alignment_json.write_text(
        json.dumps(
            {
                "scale": scale,
                "rotation": np.asarray(rotation).tolist(),
                "translation_mm": np.asarray(translation_mm).tolist(),
                "rejected_frame_names": list(rejected_frame_names),
            }
        ),
        encoding="utf-8",
    )
    return run_dir, board_dir, alignment_json, true_extrinsic


def _fuse_box(run_dir, alignment_json, out_dir, **kwargs):
    return tsdf_fusion.fuse_run(
        run_dir,
        alignment_json,
        out_dir,
        voxel_size_mm=2.0,
        max_part_height_mm=50.0,
        min_kept_points=0,
        **kwargs,
    )


def _mean_wall_distance_mm(vertices_mm: np.ndarray) -> float:
    """Mean distance of the mesh's wall vertices from the box's true side faces.

    "Wall" is every vertex between 5 and 25 mm up (clear of the board and the top face) within
    6 mm of the box's outline, inside or outside it.
    """
    x, y, z = vertices_mm.T
    near_box = (
        (z < -5.0)
        & (z > -25.0)
        & (x > _BOX_MIN_MM[0] - 6.0)
        & (x < _BOX_MAX_MM[0] + 6.0)
        & (y > _BOX_MIN_MM[1] - 6.0)
        & (y < _BOX_MAX_MM[1] + 6.0)
    )
    wall = vertices_mm[near_box]
    assert len(wall) > 100, "the fused box should have walls to measure"
    outside_x = np.maximum(
        np.maximum(_BOX_MIN_MM[0] - wall[:, 0], wall[:, 0] - _BOX_MAX_MM[0]), 0
    )
    outside_y = np.maximum(
        np.maximum(_BOX_MIN_MM[1] - wall[:, 1], wall[:, 1] - _BOX_MAX_MM[1]), 0
    )
    inside_distance = np.minimum.reduce(
        [
            wall[:, 0] - _BOX_MIN_MM[0],
            _BOX_MAX_MM[0] - wall[:, 0],
            wall[:, 1] - _BOX_MIN_MM[1],
            _BOX_MAX_MM[1] - wall[:, 1],
        ]
    )
    is_inside = (outside_x == 0) & (outside_y == 0)
    distance = np.where(is_inside, inside_distance, np.hypot(outside_x, outside_y))
    return float(np.mean(np.abs(distance)))


def test_fuse_run_walls_sit_closer_to_the_box_with_the_correction(tmp_path):
    run_dir, board_dir, alignment_json, true_extrinsic = _write_box_capture(tmp_path)

    off = _fuse_box(run_dir, alignment_json, tmp_path / "off")
    on = _fuse_box(
        run_dir,
        alignment_json,
        tmp_path / "on",
        corner_reanchoring=True,
        board_pose_dir=board_dir,
    )

    off_vertices, _faces = tsdf_fusion.read_mesh_ply(tmp_path / "off" / "mesh.ply")
    on_vertices, _faces = tsdf_fusion.read_mesh_ply(tmp_path / "on" / "mesh.ply")
    off_wall_mm = _mean_wall_distance_mm(off_vertices)
    on_wall_mm = _mean_wall_distance_mm(on_vertices)
    # Measured when written: 1.23 mm off, 0.60 mm on (the floor a 2 mm voxel allows).
    assert on_wall_mm < 0.7 * off_wall_mm
    assert on_wall_mm < 0.8

    assert off["corner_reanchoring"]["enabled"] is False
    assert off["corner_reanchoring"]["frames"] == []
    assert on["corner_reanchoring"]["enabled"] is True
    assert on["corner_reanchoring"]["corrected_frame_count"] == len(_AZIMUTHS_DEG)
    assert on["plane_reanchoring"] is True
    # R-30's plane step still ran on top of the corner fit.
    assert len(on["reanchor_reports"]) == len(_AZIMUTHS_DEG)

    # The poses fusion hands on -- the file provenance and uncertainty read -- are the true ones.
    with np.load(tmp_path / "on" / "corrected_extrinsic.npz") as corrected:
        assert corrected["extrinsic"] == pytest.approx(true_extrinsic, abs=0.01)


def test_fuse_run_moves_the_corners_through_the_runs_own_global_alignment(tmp_path):
    # R-01 works in an arbitrary frame and scale; the corners are lifted there and must go
    # through R-04's similarity before each photo is fitted to them, like every other point.
    run_dir, board_dir, alignment_json, true_extrinsic = _write_box_capture(
        tmp_path,
        scale=37.5,
        rotation=_rotation_about_z(40.0) @ _rotation_about_x(25.0),
        translation_mm=np.array([12.0, -30.0, 7.0]),
    )

    report = _fuse_box(
        run_dir,
        alignment_json,
        tmp_path / "on",
        corner_reanchoring=True,
        board_pose_dir=board_dir,
    )

    assert report["corner_reanchoring"]["corrected_frame_count"] == len(_AZIMUTHS_DEG)
    for frame in report["corner_reanchoring"]["frames"]:
        assert frame["rms_after_mm"] < 0.01
    with np.load(tmp_path / "on" / "corrected_extrinsic.npz") as corrected:
        assert corrected["extrinsic"] == pytest.approx(true_extrinsic, abs=0.01)
    vertices, _faces = tsdf_fusion.read_mesh_ply(tmp_path / "on" / "mesh.ply")
    assert _mean_wall_distance_mm(vertices) < 0.8


def test_fuse_run_reports_each_photos_corner_fit(tmp_path):
    run_dir, board_dir, alignment_json, _true = _write_box_capture(tmp_path)

    report = _fuse_box(
        run_dir,
        alignment_json,
        tmp_path / "on",
        corner_reanchoring=True,
        board_pose_dir=board_dir,
    )

    frames = report["corner_reanchoring"]["frames"]
    assert [frame["frame_index"] for frame in frames] == list(range(len(_AZIMUTHS_DEG)))
    for frame, (yaw_deg, _shift) in zip(frames, _PLANTED_DRIFTS):
        assert frame["status"] == "corrected"
        assert frame["corner_count"] >= cr.DEFAULT_MIN_CORNERS_PER_FRAME
        # Every photo starts visibly off (the smallest planted error leaves 0.96 mm RMS) and
        # ends on its corners.
        assert frame["rms_before_mm"] > 0.5
        assert frame["rms_after_mm"] < 0.01
        assert frame["rotation_deg"] == pytest.approx(abs(yaw_deg), abs=0.01)
    on_disk = json.loads((tmp_path / "on" / "tsdf_fusion.json").read_text())
    assert on_disk["corner_reanchoring"] == report["corner_reanchoring"]
    assert on_disk["corner_reanchoring"]["board_pose_dir"] == str(board_dir)
    assert on_disk["corner_reanchoring"]["min_corners_per_frame"] == 8


def test_fuse_run_leaves_a_photo_r38_rejected_uncorrected_and_says_so(tmp_path):
    run_dir, board_dir, alignment_json, _true = _write_box_capture(
        tmp_path, rejected_frame_names=["frame_5.jpg"]
    )

    report = _fuse_box(
        run_dir,
        alignment_json,
        tmp_path / "on",
        corner_reanchoring=True,
        board_pose_dir=board_dir,
    )

    frames = report["corner_reanchoring"]["frames"]
    assert frames[5]["status"] == "skipped"
    assert frames[5]["skip_reason"] == cr.SKIP_REASON_REJECTED
    assert report["corner_reanchoring"]["corrected_frame_count"] == 5
    assert report["corner_reanchoring"]["skipped_frame_count"] == 1
    with np.load(run_dir / "poses.npz") as poses:
        given = poses["extrinsic"]
    with np.load(tmp_path / "on" / "corrected_extrinsic.npz") as corrected:
        # The rejected photo keeps its global-alignment pose: neither R-44 nor R-30 touches it.
        assert corrected["extrinsic"][5] == pytest.approx(given[5])


def test_provenance_labels_the_corrected_mesh_with_the_corrected_poses(tmp_path):
    from recon.provenance import provenance_run

    run_dir, board_dir, alignment_json, _true = _write_box_capture(tmp_path)
    fusion_dir = tmp_path / "on"
    _fuse_box(
        run_dir,
        alignment_json,
        fusion_dir,
        corner_reanchoring=True,
        board_pose_dir=board_dir,
    )

    with_corrected = provenance_run(
        fusion_dir / "mesh.ply",
        run_dir,
        alignment_json,
        tmp_path / "provenance_corrected",
        fusion_dir=fusion_dir,
    )
    with_global = provenance_run(
        fusion_dir / "mesh.ply",
        run_dir,
        alignment_json,
        tmp_path / "provenance_global",
    )

    assert with_corrected["extrinsic_source"] == "reanchored"
    # Checked against the poses it was fused from, more of the mesh is confirmed by the photos
    # than against the drifted global poses (measured when written: 1417 against 1293 of 2032).
    assert (
        with_corrected["observed_vertex_count"] > with_global["observed_vertex_count"]
    )


def test_fuse_run_can_skip_r30s_plane_step_for_the_measurement(tmp_path):
    run_dir, board_dir, alignment_json, _true = _write_box_capture(tmp_path)

    report = _fuse_box(
        run_dir,
        alignment_json,
        tmp_path / "corners_only",
        corner_reanchoring=True,
        board_pose_dir=board_dir,
        plane_reanchoring=False,
    )

    assert report["plane_reanchoring"] is False
    assert report["reanchor_reports"] == []
    assert report["corner_reanchoring"]["corrected_frame_count"] == len(_AZIMUTHS_DEG)


def test_fuse_run_refuses_corner_reanchoring_without_the_board_files(tmp_path):
    run_dir, _board_dir, alignment_json, _true = _write_box_capture(tmp_path)

    with pytest.raises(ValueError, match="needs board_pose_dir"):
        _fuse_box(run_dir, alignment_json, tmp_path / "a", corner_reanchoring=True)
    with pytest.raises(FileNotFoundError, match="board_poses.npz"):
        _fuse_box(
            run_dir,
            alignment_json,
            tmp_path / "b",
            corner_reanchoring=True,
            board_pose_dir=tmp_path / "nowhere",
        )


def test_fuse_run_cli_passes_the_r44_flags_through(tmp_path):
    run_dir, board_dir, alignment_json, _true = _write_box_capture(tmp_path)
    out_dir = tmp_path / "cli"

    exit_code = tsdf_fusion.main(
        [
            str(run_dir),
            str(alignment_json),
            "--out-dir",
            str(out_dir),
            "--corner-reanchoring",
            "--board-pose-dir",
            str(board_dir),
            "--min-corners-per-frame",
            "10",
            "--no-plane-reanchoring",
        ]
    )

    assert exit_code == 0
    report = json.loads((out_dir / "tsdf_fusion.json").read_text())
    assert report["corner_reanchoring"]["enabled"] is True
    assert report["corner_reanchoring"]["min_corners_per_frame"] == 10
    assert report["plane_reanchoring"] is False
