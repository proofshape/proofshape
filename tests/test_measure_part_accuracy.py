"""Unit tests for P-40's measurement script. All synthetic: no GPU, no captures, no fixtures.

`run` itself (one capture through the real pipeline) is a written check, recorded in
docs/accuracy-review-2026-11-20.md: it needs the model weights and the photographs, the same
limit every other measurement script here has. Everything it feeds -- the rigid fit, the
per-photo corner residuals, the corner lifting, the surface bins, the per-part summary and the
tolerance sweep -- is tested below.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from recon.board_pose import make_board
from recon.metric_alignment import apply_similarity_transform_to_extrinsic
from scripts import measure_part_accuracy as m

# --- helpers ----------------------------------------------------------------------------------


def _rotation_about_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])


def _board_corner_grid() -> np.ndarray:
    """A 5 x 4 grid of points on the board plane, 20 mm apart, like ChArUco corners."""
    xs, ys = np.meshgrid(np.arange(5) * 20.0 + 20.0, np.arange(4) * 20.0 + 20.0)
    return np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)


def _box_cad(
    size_x: float, size_y: float, size_z: float
) -> tuple[np.ndarray, np.ndarray]:
    """A closed box with outward-facing triangles, base on z = 0, +z up (the CAD convention)."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [size_x, 0.0, 0.0],
            [size_x, size_y, 0.0],
            [0.0, size_y, 0.0],
            [0.0, 0.0, size_z],
            [size_x, 0.0, size_z],
            [size_x, size_y, size_z],
            [0.0, size_y, size_z],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # bottom, normal -z
            [4, 5, 6],
            [4, 6, 7],  # top, normal +z
            [0, 1, 5],
            [0, 5, 4],  # y = 0 wall
            [1, 2, 6],
            [1, 6, 5],  # x = size_x wall
            [2, 3, 7],
            [2, 7, 6],  # y = size_y wall
            [3, 0, 4],
            [3, 4, 7],  # x = 0 wall
        ]
    )
    return vertices, faces


# The board frame has z pointing into the table, so a part sitting on the board is the CAD
# flipped about x (the same flip inspection.registration.CAD_TO_BOARD_FLIP applies), then
# turned and moved. Every CAD-frame point below is placed through this one transform.
def _cad_to_board(yaw_deg: float, shift_xy: tuple[float, float]) -> np.ndarray:
    flip = np.diag([1.0, -1.0, -1.0])
    transform = np.eye(4)
    transform[:3, :3] = _rotation_about_z(yaw_deg) @ flip
    transform[:2, 3] = shift_xy
    return transform


def _place(points_cad: np.ndarray, cad_to_model: np.ndarray) -> np.ndarray:
    return points_cad @ cad_to_model[:3, :3].T + cad_to_model[:3, 3]


def _measure(
    vertices: np.ndarray,
    deviation: np.ndarray,
    measured: np.ndarray,
    observed: np.ndarray,
    cad: tuple[np.ndarray, np.ndarray],
    cad_to_model: np.ndarray,
    sigma: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    return {
        "vertices": vertices,
        "faces": np.empty((0, 3), dtype=np.int64),
        "observed": observed,
        "deviation_mm": deviation,
        "deviation_measured": measured.astype(np.float64),
        "sigma_mm": np.full(len(vertices), 0.5) if sigma is None else sigma,
        "cad_to_model": cad_to_model,
        "cad_vertices": cad[0],
        "cad_faces": cad[1],
        "board_plane": np.array([0.0, 0.0, 0.0]),
    }


# --- fit_rigid --------------------------------------------------------------------------------


def test_fit_rigid_recovers_a_planted_rotation_and_shift_from_coplanar_points():
    source = _board_corner_grid()
    planted_rotation = _rotation_about_z(1.7)
    planted_translation = np.array([2.5, -1.0, 0.4])
    target = source @ planted_rotation.T + planted_translation

    rotation, translation = m.fit_rigid(source, target)

    np.testing.assert_allclose(rotation, planted_rotation, atol=1e-9)
    np.testing.assert_allclose(translation, planted_translation, atol=1e-9)


def test_fit_rigid_never_returns_a_reflection():
    # A mirrored copy is fitted exactly by a reflection, which is what the unguarded
    # least-squares answer is. A camera pose can't be a mirror image, so the fit has to come
    # back as the nearest proper rotation (determinant +1), not as that reflection.
    source = np.random.default_rng(0).normal(scale=30.0, size=(12, 3))
    mirrored = source * np.array([-1.0, 1.0, 1.0])

    rotation, _translation = m.fit_rigid(source, mirrored)

    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)


def test_fit_rigid_refuses_collinear_points_rather_than_guess_a_rotation():
    line = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
    )
    with pytest.raises(ValueError, match="collinear"):
        m.fit_rigid(line, line + 1.0)


def test_fit_rigid_refuses_fewer_than_three_pairs_and_mismatched_shapes():
    with pytest.raises(ValueError, match="at least 3"):
        m.fit_rigid(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="must both be"):
        m.fit_rigid(np.zeros((4, 3)), np.zeros((5, 3)))


# --- photo_corner_residuals -------------------------------------------------------------------


def test_a_sideways_shifted_photo_reads_its_shift_and_a_rigid_fit_removes_it():
    targets = _board_corner_grid()
    # Photo 0 is exactly on the board. Photo 1 is turned 0.4 degrees and moved 1.5 mm sideways.
    turned = targets @ _rotation_about_z(0.4).T + np.array([1.5, -0.5, 0.0])
    points = np.concatenate([targets, turned])
    all_targets = np.concatenate([targets, targets])
    photo_of_corner = np.repeat([0, 1], len(targets))
    reference = np.array([80.0, 60.0, 0.0])

    measured, skipped = m.photo_corner_residuals(
        points, all_targets, photo_of_corner, reference_point_mm=reference
    )

    assert skipped == []
    exact, shifted = measured
    assert exact["rms_mm"] == pytest.approx(0.0, abs=1e-9)
    assert exact["shift_at_reference_mm"] == pytest.approx(0.0, abs=1e-9)

    assert shifted["in_plane_rms_mm"] == pytest.approx(
        shifted["rms_mm"]
    )  # no height error
    assert shifted["in_plane_rms_mm"] > 1.0
    assert shifted["height_mean_mm"] == pytest.approx(0.0, abs=1e-9)
    assert shifted["rms_after_rigid_fit_mm"] == pytest.approx(0.0, abs=1e-9)
    # The fit undoes the planted move, so it carries the reference point back by exactly as far
    # as the planted move carried it.
    planted_move = _rotation_about_z(0.4) @ reference + np.array([1.5, -0.5, 0.0])
    assert shifted["shift_at_reference_mm"] == pytest.approx(
        np.linalg.norm((planted_move - reference)[:2]), abs=1e-6
    )


def test_a_height_only_error_reads_as_height_not_as_sideways():
    targets = _board_corner_grid()
    measured, _skipped = m.photo_corner_residuals(
        targets + np.array([0.0, 0.0, 0.8]), targets, np.zeros(len(targets), dtype=int)
    )
    assert measured[0]["height_mean_mm"] == pytest.approx(0.8)
    assert measured[0]["in_plane_rms_mm"] == pytest.approx(0.0, abs=1e-9)
    assert measured[0]["shift_at_reference_mm"] == pytest.approx(0.0, abs=1e-9)


def test_a_photo_with_too_few_corners_is_listed_as_skipped_not_dropped_silently():
    targets = _board_corner_grid()
    few = targets[:5]
    points = np.concatenate([targets, few])
    photo_of_corner = np.concatenate([np.zeros(len(targets), dtype=int), np.full(5, 3)])

    measured, skipped = m.photo_corner_residuals(
        points, points.copy(), photo_of_corner, min_corners=8
    )

    assert [photo["photo_index"] for photo in measured] == [0]
    assert skipped == [
        {"photo_index": 3, "corner_count": 5, "reason": "fewer than 8 corners"}
    ]


def test_corner_residuals_refuse_inconsistent_inputs():
    targets = _board_corner_grid()
    with pytest.raises(ValueError, match="one entry per corner"):
        m.photo_corner_residuals(targets, targets, np.zeros(3, dtype=int))
    with pytest.raises(ValueError, match="at least 3"):
        m.photo_corner_residuals(
            targets, targets, np.zeros(len(targets), dtype=int), min_corners=2
        )


def test_the_default_reference_point_is_the_centre_of_the_board_footprint():
    targets = _board_corner_grid()
    turned = targets @ _rotation_about_z(1.0).T
    measured, _skipped = m.photo_corner_residuals(
        turned, targets, np.zeros(len(targets), dtype=int)
    )
    centre = np.array(
        [m.DEFAULT_FOOTPRINT_MM[0] / 2.0, m.DEFAULT_FOOTPRINT_MM[1] / 2.0, 0.0]
    )
    expected = np.linalg.norm((_rotation_about_z(-1.0) @ centre - centre)[:2])
    assert measured[0]["shift_at_reference_mm"] == pytest.approx(expected, abs=1e-6)


# --- summarise_corner_residuals ---------------------------------------------------------------


def _lifted(after_global: np.ndarray, after_plane: np.ndarray | None, rejected=()):
    targets = _board_corner_grid()
    photo_count = len(after_global) // len(targets)
    return m.LiftedCorners(
        frame_names=[f"frame_{index:04d}.jpg" for index in range(photo_count)],
        photo_of_corner=np.repeat(np.arange(photo_count), len(targets)),
        targets_mm=np.tile(targets, (photo_count, 1)),
        after_global_alignment_mm=after_global,
        after_plane_reanchoring_mm=after_plane,
        rejected_photo_indices=frozenset(rejected),
    )


def test_a_rejected_photo_is_counted_but_kept_out_of_every_median():
    targets = _board_corner_grid()
    good = targets + np.array([1.0, 0.0, 0.0])
    # R-38's kind of photo: its corners are far out, and it never reaches fusion.
    wild = targets + np.array([150.0, 0.0, 0.0])
    corners = _lifted(np.concatenate([good, wild]), None, rejected=(1,))

    summary = m.summarise_corner_residuals(corners)

    assert summary["photo_count"] == 2
    assert summary["rejected_photo_count"] == 1
    assert summary["measured_photo_count"] == 1
    assert summary["after_global_alignment"]["rms_max_mm"] == pytest.approx(1.0)
    assert summary["after_plane_reanchoring"] is None


def test_the_plane_stage_reports_what_r30_leaves_sideways():
    targets = _board_corner_grid()
    # Before R-30 a photo is 1.2 mm sideways and 0.9 mm high; R-30 removes the height only.
    before = targets + np.array([1.2, 0.0, 0.9])
    after = targets + np.array([1.2, 0.0, 0.0])
    summary = m.summarise_corner_residuals(_lifted(before, after))

    assert summary["after_global_alignment"]["rms_median_mm"] == pytest.approx(1.5)
    plane = summary["after_plane_reanchoring"]
    assert plane["in_plane_rms_median_mm"] == pytest.approx(1.2)
    assert plane["height_mean_abs_median_mm"] == pytest.approx(0.0, abs=1e-9)
    assert plane["rms_after_rigid_fit_median_mm"] == pytest.approx(0.0, abs=1e-9)
    assert plane["shift_at_reference_median_mm"] == pytest.approx(1.2)


# --- load_lifted_corners ----------------------------------------------------------------------

SCENE_SCALE = 250.0  # millimetres per reconstruction unit, as R-04 would solve it
CAMERA_HEIGHT_MM = 300.0
FOCAL_PX = 400.0
IMAGE_HW = (300, 400)  # landscape, as the pre-R-23 lifting path requires


def _write_run(tmp_path, *, corrected_shift_mm: np.ndarray | None, rejected=()):
    """Two cameras looking straight down at the board, with exact depth and exact corners.

    The reconstruction frame is the board frame divided by SCENE_SCALE, so R-04's alignment is a
    pure scale and every lifted corner lands exactly on its nominal board position.
    """
    nominal = np.asarray(make_board().getChessboardCorners(), dtype=np.float64)
    corner_ids = np.arange(20)
    centres_mm = [
        np.array([80.0, 60.0, -CAMERA_HEIGHT_MM]),
        np.array([70.0, 65.0, -CAMERA_HEIGHT_MM]),
    ]
    height, width = IMAGE_HW
    intrinsic = np.array(
        [[FOCAL_PX, 0.0, width / 2.0], [0.0, FOCAL_PX, height / 2.0], [0.0, 0.0, 1.0]]
    )

    extrinsics, frame_index, ids, pixels = [], [], [], []
    for photo, centre in enumerate(centres_mm):
        # Looking along +z (into the table) from above the board: x_cam = x_board - centre.
        extrinsics.append(
            np.concatenate([np.eye(3), (-centre / SCENE_SCALE)[:, None]], axis=1)
        )
        camera_mm = nominal[corner_ids] - centre
        u = FOCAL_PX * camera_mm[:, 0] / camera_mm[:, 2] + width / 2.0
        v = FOCAL_PX * camera_mm[:, 1] / camera_mm[:, 2] + height / 2.0
        frame_index.extend([photo] * len(corner_ids))
        ids.extend(corner_ids.tolist())
        pixels.extend(np.stack([u, v], axis=1).tolist())
    extrinsic = np.stack(extrinsics)
    names = np.array(["frame_0000.jpg", "frame_0001.jpg"])

    artifact_dir = tmp_path / "artifacts"
    reconstruction_dir = artifact_dir / "reconstruction"
    board_dir = artifact_dir / "board_pose"
    reconstruction_dir.mkdir(parents=True)
    board_dir.mkdir()
    (reconstruction_dir / "run.json").write_text(
        json.dumps({"processed_image_size_hw": list(IMAGE_HW)}), encoding="utf-8"
    )
    np.savez(
        reconstruction_dir / "poses.npz",
        frame_names=names,
        extrinsic=extrinsic,
        intrinsic=np.stack([intrinsic, intrinsic]),
    )
    # Every pixel sees the flat board, CAMERA_HEIGHT_MM away along the optical axis.
    np.savez(
        reconstruction_dir / "depth.npz",
        depth=np.full((2, height, width), CAMERA_HEIGHT_MM / SCENE_SCALE),
    )
    (reconstruction_dir / "metric_alignment.json").write_text(
        json.dumps(
            {
                "scale": SCENE_SCALE,
                "rotation": np.eye(3).tolist(),
                "translation_mm": [0.0, 0.0, 0.0],
                "rejected_frame_names": [str(names[index]) for index in rejected],
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        board_dir / "board_poses.npz",
        frame_names=names,
        image_size_hw=np.array([IMAGE_HW, IMAGE_HW]),
        corner_frame_index=np.array(frame_index),
        corner_id=np.array(ids),
        corner_px=np.array(pixels),
        pose_found=np.array([True, True]),
    )
    if corrected_shift_mm is not None:
        board_extrinsic = apply_similarity_transform_to_extrinsic(
            extrinsic, SCENE_SCALE, np.eye(3), np.zeros(3)
        )
        # Moving photo 1's board points by `shift` is the pose change t' = t - R @ shift
        # (R-30's own derivation, with no rotation).
        corrected = board_extrinsic.copy()
        corrected[1, :, 3] = (
            board_extrinsic[1, :, 3] - board_extrinsic[1, :, :3] @ corrected_shift_mm
        )
        (artifact_dir / "fusion").mkdir()
        np.savez(
            artifact_dir / "fusion" / "corrected_extrinsic.npz", extrinsic=corrected
        )
    return artifact_dir, nominal[corner_ids]


def test_lifted_corners_land_on_the_board_and_follow_the_corrected_poses(tmp_path):
    shift = np.array([0.0, 0.0, 2.0])
    artifact_dir, nominal = _write_run(tmp_path, corrected_shift_mm=shift)

    corners = m.load_lifted_corners(artifact_dir)

    assert corners.frame_names == ["frame_0000.jpg", "frame_0001.jpg"]
    np.testing.assert_allclose(corners.targets_mm, np.tile(nominal, (2, 1)), atol=1e-9)
    # An exact scene: every lifted corner is on its nominal position after the global alignment.
    np.testing.assert_allclose(
        corners.after_global_alignment_mm, corners.targets_mm, atol=1e-4
    )
    # Through the corrected poses, photo 0 is unchanged and photo 1 has moved by the shift.
    first = corners.photo_of_corner == 0
    np.testing.assert_allclose(
        corners.after_plane_reanchoring_mm[first], corners.targets_mm[first], atol=1e-4
    )
    np.testing.assert_allclose(
        corners.after_plane_reanchoring_mm[~first],
        corners.targets_mm[~first] + shift,
        atol=1e-4,
    )


def test_a_run_without_corrected_poses_reports_no_plane_stage(tmp_path):
    artifact_dir, _nominal = _write_run(
        tmp_path, corrected_shift_mm=None, rejected=(1,)
    )
    corners = m.load_lifted_corners(artifact_dir)
    assert corners.after_plane_reanchoring_mm is None
    assert corners.rejected_photo_indices == frozenset({1})


def test_the_calibrated_board_folder_is_preferred_when_it_exists(tmp_path):
    artifact_dir, _nominal = _write_run(tmp_path, corrected_shift_mm=None)
    # The service writes board_pose_calibrated/ when the frames carry no EXIF, and every later
    # stage reads that one. A calibrated folder with nothing in it must therefore be the one
    # that is looked in -- and reported missing -- not silently skipped for the other.
    (artifact_dir / "board_pose_calibrated").mkdir()
    with pytest.raises(FileNotFoundError, match="board_pose_calibrated"):
        m.load_lifted_corners(artifact_dir)


def test_a_missing_run_file_is_named(tmp_path):
    artifact_dir, _nominal = _write_run(tmp_path, corrected_shift_mm=None)
    (artifact_dir / "reconstruction" / "depth.npz").unlink()
    with pytest.raises(FileNotFoundError, match="depth.npz"):
        m.load_lifted_corners(artifact_dir)


def test_corrected_poses_of_the_wrong_shape_are_refused(tmp_path):
    artifact_dir, _nominal = _write_run(tmp_path, corrected_shift_mm=np.zeros(3))
    np.savez(
        artifact_dir / "fusion" / "corrected_extrinsic.npz",
        extrinsic=np.zeros((3, 3, 4)),
    )
    with pytest.raises(ValueError, match="expected"):
        m.load_lifted_corners(artifact_dir)


# --- surface bins and the per-part summary ----------------------------------------------------

BOX = _box_cad(40.0, 30.0, 20.0)
POSE = _cad_to_board(yaw_deg=37.0, shift_xy=(60.0, 45.0))


def test_vertices_are_binned_by_the_cad_face_they_sit_on_whatever_the_parts_pose():
    points_cad = np.array(
        [
            [20.0, 15.0, 20.3],  # just above the top face
            [40.4, 15.0, 10.0],  # just outside the x = 40 wall
            [20.0, -0.3, 10.0],  # just outside the y = 0 wall
            [20.0, 15.0, -0.5],  # under the base: a bottom face, neither top nor wall
        ]
    )
    surface = m.classify_by_cad_face(_place(points_cad, POSE), BOX[0], BOX[1], POSE)
    assert surface.tolist() == [
        m.SURFACE_TOP,
        m.SURFACE_WALL,
        m.SURFACE_WALL,
        m.SURFACE_OTHER,
    ]


def test_no_vertices_bin_to_nothing():
    assert len(m.classify_by_cad_face(np.empty((0, 3)), BOX[0], BOX[1], POSE)) == 0
    assert len(m.distance_outside_cad_mm(np.empty((0, 3)), BOX[0], BOX[1], POSE)) == 0


def test_distance_is_positive_outside_the_cad_and_negative_inside():
    points_cad = np.array([[43.0, 15.0, 10.0], [20.0, 15.0, 10.0], [20.0, 15.0, 27.5]])
    distance = m.distance_outside_cad_mm(_place(points_cad, POSE), BOX[0], BOX[1], POSE)
    np.testing.assert_allclose(distance, [3.0, -10.0, 7.5], atol=1e-4)


def test_pose_yaw_reads_the_rotation_about_the_board_normal():
    assert m.pose_yaw_deg(POSE) == pytest.approx(37.0)


def test_the_part_summary_counts_measured_unmeasured_and_stray_surface():
    top = np.array([[10.0 + step, 15.0, 20.0] for step in range(10)])
    wall = np.array([[40.0, 5.0 + step, 10.0] for step in range(10)])
    # Observed but outside the CAD's outline, so I-06 leaves it unmeasured: 2 mm out, on the wall.
    outside = np.array([[42.0, 5.0 + step, 10.0] for step in range(5)])
    # Observed surface nowhere near the part (board skin): far past the stray distance.
    stray = np.array([[80.0, 5.0 + step, 1.0] for step in range(5)])
    unobserved = np.array([[20.0, 15.0, 20.0]] * 4)
    vertices = _place(np.concatenate([top, wall, outside, stray, unobserved]), POSE)

    deviation = np.concatenate(
        [
            np.full(10, 0.2),
            np.linspace(-2.0, -1.0, 10),
            np.zeros(5),
            np.zeros(5),
            np.zeros(4),
        ]
    )
    measured = np.concatenate([np.ones(20, bool), np.zeros(14, bool)])
    observed = np.concatenate([np.ones(30, bool), np.zeros(4, bool)])
    summary = m.summarise_part(
        _measure(vertices, deviation, measured, observed, BOX, POSE)
    )

    assert summary["vertex_count"] == 34
    assert summary["observed_count"] == 30
    assert summary["measured_count"] == 20
    assert summary["unmeasured_observed_share"] == pytest.approx(10 / 30)
    assert summary["stray_observed_share"] == pytest.approx(5 / 30)
    assert summary["top"]["count"] == 10
    assert summary["top"]["median_mm"] == pytest.approx(0.2)
    assert summary["wall"]["count"] == 10
    assert summary["wall"]["median_mm"] == pytest.approx(-1.5)
    assert summary["wall"]["abs_max_mm"] == pytest.approx(2.0)
    assert summary["worst_reading_mm"] == pytest.approx(2.0)
    assert summary["pose_yaw_deg"] == pytest.approx(37.0)


def test_a_part_with_nothing_measured_reports_none_not_zero():
    vertices = _place(np.array([[20.0, 15.0, 20.0], [40.0, 15.0, 10.0]]), POSE)
    summary = m.summarise_part(
        _measure(vertices, np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), BOX, POSE)
    )
    assert summary["top"] is None
    assert summary["wall"] is None
    assert summary["worst_reading_mm"] is None
    assert summary["unmeasured_observed_share"] is None
    assert summary["stray_observed_share"] is None


def test_the_part_summary_refuses_arrays_of_different_lengths():
    vertices = _place(np.array([[20.0, 15.0, 20.0], [40.0, 15.0, 10.0]]), POSE)
    measure = _measure(
        vertices, np.zeros(2), np.ones(2, bool), np.ones(2, bool), BOX, POSE
    )
    measure["deviation_mm"] = np.zeros(3)
    with pytest.raises(ValueError, match="one entry per vertex"):
        m.summarise_part(measure)


# --- tolerance sweep and the verdict's own pools ----------------------------------------------


def _part_on_a_flat_board(top_count: int, wall_count: int):
    """A box's top-face and wall vertices, on a board at z = 0 with z into the table."""
    flat = _cad_to_board(yaw_deg=0.0, shift_xy=(60.0, 45.0))
    top = np.array(
        [[5.0 + 30.0 * step / top_count, 15.0, 20.0] for step in range(top_count)]
    )
    wall = np.array(
        [[40.0, 15.0, 1.0 + 15.0 * step / wall_count] for step in range(wall_count)]
    )
    return _place(np.concatenate([top, wall]), flat), flat


def test_a_part_with_small_readings_first_passes_at_the_existing_gate():
    vertices, flat = _part_on_a_flat_board(10, 90)
    measure = _measure(
        vertices, np.full(100, 0.3), np.ones(100, bool), np.ones(100, bool), BOX, flat
    )
    sweep = m.tolerance_sweep(measure)
    # Nothing is out by more than 0.3 mm, so nothing Fails. Pass still waits for the in-plane
    # floor: tolerance / 0.70 has to reach 5.278 mm, which 3.5 does not and 3.7 does.
    assert sweep["last_fail_mm"] is None
    assert sweep["first_pass_mm"] == pytest.approx(3.7)
    assert sweep["verdicts"][3.5] == "Rescan"


def test_one_bad_wall_vertex_sets_both_ends_of_the_sweep():
    vertices, flat = _part_on_a_flat_board(10, 90)
    deviation = np.full(100, 0.3)
    deviation[50] = -4.2  # a wall vertex, so it is judged against the in-plane floor
    measure = _measure(
        vertices, deviation, np.ones(100, bool), np.ones(100, bool), BOX, flat
    )

    sweep = m.tolerance_sweep(measure)

    # Fail needs 4.2 > tolerance + 0.70 * 5.278, so only below 0.51 mm.
    assert sweep["last_fail_mm"] == pytest.approx(0.5)
    # Pass needs every vertex inside the tolerance, so the first listed value at or above 4.2.
    assert sweep["first_pass_mm"] == pytest.approx(4.5)


def test_a_part_that_never_passes_reports_none():
    vertices, flat = _part_on_a_flat_board(10, 90)
    deviation = np.full(100, 0.3)
    deviation[50] = -30.0
    measure = _measure(
        vertices, deviation, np.ones(100, bool), np.ones(100, bool), BOX, flat
    )
    assert m.tolerance_sweep(measure)["first_pass_mm"] is None


def test_top_face_vertices_that_miss_the_top_tenth_are_counted_in_the_in_plane_pool():
    vertices, flat = _part_on_a_flat_board(30, 70)
    # Lift ten of the thirty top-face vertices slightly, as reconstruction noise would: only
    # those ten make the verdict's top-tenth-by-height pool.
    vertices = vertices.copy()
    vertices[:10, 2] -= 0.5  # z points into the table, so smaller z is higher
    measure = _measure(
        vertices, np.zeros(100), np.ones(100, bool), np.ones(100, bool), BOX, flat
    )
    pools = m.verdict_pool_sizes(measure)
    assert pools == {
        "out_of_plane": 10,
        "in_plane": 90,
        "top_face_in_in_plane_pool": 20,
    }


def test_pool_sizes_are_zero_when_nothing_is_judged():
    vertices, flat = _part_on_a_flat_board(3, 3)
    measure = _measure(
        vertices, np.zeros(6), np.zeros(6, bool), np.ones(6, bool), BOX, flat
    )
    assert m.verdict_pool_sizes(measure) == {
        "out_of_plane": 0,
        "in_plane": 0,
        "top_face_in_in_plane_pool": 0,
    }


# --- the command line -------------------------------------------------------------------------


def _write_measure(out_dir):
    vertices, flat = _part_on_a_flat_board(10, 90)
    measure = _measure(
        vertices, np.full(100, 0.3), np.ones(100, bool), np.ones(100, bool), BOX, flat
    )
    out_dir.mkdir(parents=True)
    np.savez(out_dir / m.MEASURE_NPZ, **measure)
    (out_dir / m.MEASURE_JSON).write_text(
        json.dumps(
            {
                "commit": "abc1234",
                "registration": {"orientation_ambiguous": False, "best_fitness": 0.97},
            }
        ),
        encoding="utf-8",
    )


def test_report_prints_one_row_per_part_and_writes_its_figures(tmp_path, capsys):
    _write_measure(tmp_path / "s-04")
    figures = tmp_path / "figures.json"

    assert m.main(["report", str(tmp_path / "s-04"), "--json", str(figures)]) == 0

    printed = capsys.readouterr().out
    assert "s-04" in printed and "first Pass" in printed
    rows = json.loads(figures.read_text(encoding="utf-8"))
    assert rows[0]["part"] == "s-04"
    assert rows[0]["sweep"]["first_pass_mm"] == pytest.approx(3.7)
    assert rows[0]["commit"] == "abc1234"


def test_report_fails_loudly_on_a_folder_that_was_never_run(tmp_path, capsys):
    assert m.main(["report", str(tmp_path)]) == 1
    assert "run the `run` subcommand first" in capsys.readouterr().err


def test_corners_prints_one_row_per_capture(tmp_path, capsys):
    _write_run(tmp_path / "s-04", corrected_shift_mm=np.array([1.0, 0.0, 0.0]))
    figures = tmp_path / "corners.json"

    assert m.main(["corners", str(tmp_path / "s-04"), "--json", str(figures)]) == 0

    assert "s-04" in capsys.readouterr().out
    row = json.loads(figures.read_text(encoding="utf-8"))[0]
    assert row["measured_photo_count"] == 2
    # Photo 1 was moved 1 mm sideways by the corrected poses; photo 0 was not.
    assert row["after_plane_reanchoring"]["shift_at_reference_max_mm"] == pytest.approx(
        1.0, abs=1e-3
    )


def test_run_refuses_an_output_folder_that_already_has_files(tmp_path, capsys):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "left-over.txt").write_text("from an earlier run", encoding="utf-8")

    code = m.main(
        ["run", str(tmp_path / "capture"), "--cad", "part.stl", "--out", str(out_dir)]
    )

    assert code == 1
    assert "already has files" in capsys.readouterr().err
