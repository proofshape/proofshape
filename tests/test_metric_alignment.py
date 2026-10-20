from __future__ import annotations

import numpy as np
import pytest

from recon.board_pose import make_board
from recon.metric_alignment import (
    align_board_geometry,
    align_camera_centres,
    apply_similarity_transform,
    apply_similarity_transform_to_extrinsic,
    camera_centres,
    project_points_to_camera,
    solve_similarity,
    validate_known_size,
)
from recon.vggt_runner import unproject_depth_to_world


def _extrinsics_from_centres(centres: np.ndarray) -> np.ndarray:
    extrinsics = np.zeros((len(centres), 3, 4), dtype=np.float64)
    extrinsics[:, :, :3] = np.eye(3)
    extrinsics[:, :, 3] = -centres
    return extrinsics


def test_similarity_recovers_scale_rotation_translation() -> None:
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected_scale = 4.0
    expected_translation = np.array([10.0, -3.0, 8.0])
    target = expected_scale * (source @ rotation.T) + expected_translation

    scale, recovered_rotation, recovered_translation = solve_similarity(source, target)

    assert scale == pytest.approx(expected_scale)
    assert recovered_rotation @ rotation.T == pytest.approx(np.eye(3))
    assert recovered_translation == pytest.approx(expected_translation)


def test_align_pairs_frames_and_reports_residuals() -> None:
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    target = 2.5 * source + np.array([5.0, 6.0, 7.0])
    result = align_camera_centres(
        names,
        _extrinsics_from_centres(source),
        ["c.jpg", "a.jpg", "d.jpg", "b.jpg"],
        _extrinsics_from_centres(target[[2, 0, 3, 1]]),
        np.array([True, True, True, False]),
    )

    assert result.frame_names == ["a.jpg", "c.jpg", "d.jpg"]
    assert result.rms_residual_mm == pytest.approx(0.0, abs=1e-10)
    assert result.transform_points(np.array([[1.0, 2.0, 3.0]])) == pytest.approx(
        np.array([[7.5, 11.0, 14.5]])
    )


def test_align_rejects_duplicate_reconstruction_frame_names() -> None:
    names = ["a.jpg", "a.jpg", "b.jpg"]
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    target = 2.5 * source + np.array([5.0, 6.0, 7.0])

    with pytest.raises(ValueError, match="unique in the reconstruction output"):
        align_camera_centres(
            names,
            _extrinsics_from_centres(source),
            ["a.jpg", "b.jpg", "c.jpg"],
            _extrinsics_from_centres(target),
        )


def test_align_rejects_duplicate_board_frame_names() -> None:
    """Regression for issue #44: a duplicate board name used to silently pair against
    whichever occurrence a name->index dict happened to keep, with no error and no
    indication the pairing was ambiguous. Reproduced here with the second "c.jpg" holding a
    deliberately wrong centre, far enough off that a silent mispairing would be obvious in the
    residual if this raised nothing.
    """
    names = ["a.jpg", "b.jpg", "c.jpg"]
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    good_target = 2.5 * source + np.array([5.0, 6.0, 7.0])
    # Board side lists "c.jpg" twice: once with the correct centre, once with a wrong one.
    board_names = ["a.jpg", "b.jpg", "c.jpg", "c.jpg"]
    board_target = np.vstack([good_target, [[999.0, 999.0, 999.0]]])

    with pytest.raises(ValueError, match="unique in the board pose output"):
        align_camera_centres(
            names,
            _extrinsics_from_centres(source),
            board_names,
            _extrinsics_from_centres(board_target),
        )


def _synthetic_board_alignment_inputs(
    corner_ids: range = range(12),
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Three straight-down cameras whose lifted ChArUco corners are exact board coordinates."""
    nominal = make_board().getChessboardCorners()
    corner_ids_list = list(corner_ids)
    frame_names = ["a.jpg", "b.jpg", "c.jpg"]
    height = width = 400
    focal = 1000.0
    principal = 150.0
    distance = 1000.0
    intrinsic_single = np.array(
        [[focal, 0.0, principal], [0.0, focal, principal], [0.0, 0.0, 1.0]]
    )
    reconstruction_intrinsic = np.tile(intrinsic_single[None], (3, 1, 1))

    # x_cam = x_world + t. Different x/y translations give non-collinear camera centres while
    # every board point remains on the same z=distance plane, so one constant depth map per frame
    # is sufficient for exact synthetic lifting.
    translations = np.array(
        [[0.0, 0.0, distance], [80.0, 20.0, distance], [20.0, 100.0, distance]]
    )
    reconstruction_extrinsic = np.zeros((3, 3, 4), dtype=np.float64)
    reconstruction_extrinsic[:, :, :3] = np.eye(3)
    reconstruction_extrinsic[:, :, 3] = translations
    depth = np.full((3, height, width), distance, dtype=np.float64)

    corner_frame_index: list[int] = []
    corner_id: list[int] = []
    corner_px: list[list[float]] = []
    for frame_index, translation in enumerate(translations):
        for cid in corner_ids_list:
            x, y, _z = nominal[cid]
            u = focal * (x + translation[0]) / distance + principal
            v = focal * (y + translation[1]) / distance + principal
            corner_frame_index.append(frame_index)
            corner_id.append(cid)
            corner_px.append([u, v])

    return (
        frame_names,
        reconstruction_extrinsic,
        reconstruction_intrinsic,
        depth,
        (height, width),
        frame_names.copy(),
        np.tile(np.array([[height, width]]), (3, 1)),
        np.asarray(corner_frame_index),
        np.asarray(corner_id),
        np.asarray(corner_px),
    )


def test_board_anchor_ignores_camera_centre_scale_error() -> None:
    (
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
    ) = _synthetic_board_alignment_inputs()

    reconstruction_centres = camera_centres(reconstruction_extrinsic)
    # Model the R-22 failure mechanism: board-pose camera centres use a focal assumption 10%
    # smaller than the reconstruction source, so fitting centres alone shrinks the whole scene.
    wrong_board_centres = 0.9 * reconstruction_centres
    legacy = align_camera_centres(
        frame_names,
        reconstruction_extrinsic,
        board_frame_names,
        _extrinsics_from_centres(wrong_board_centres),
    )
    assert legacy.scale == pytest.approx(0.9)

    anchored = align_board_geometry(
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
        preprocess_geometry=_identity_preprocess_geometry(frame_names),
    )

    assert anchored.scale == pytest.approx(1.0, abs=1e-6)
    assert anchored.observation_count == 36
    assert anchored.frame_count == 3
    assert anchored.rms_residual_mm == pytest.approx(0.0, abs=1e-6)


def test_board_anchor_rejects_too_few_usable_corners() -> None:
    (
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
    ) = _synthetic_board_alignment_inputs(corner_ids=range(2))

    with pytest.raises(ValueError, match="usable board corner observations"):
        align_board_geometry(
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
        )



def _identity_preprocess_geometry(
    frame_names: list[str], height: int = 400, width: int = 400
) -> list[dict[str, object]]:
    return [
        {
            "frame_name": name,
            "original_size_hw": [height, width],
            "resized_size_hw": [height, width],
            "crop_top_px": 0,
            "crop_left_px": 0,
            "processed_size_hw": [height, width],
            "scale_x": 1.0,
            "scale_y": 1.0,
            "supported_for_metric_alignment": True,
        }
        for name in frame_names
    ]


def _portrait_preprocess_geometry(
    frame_names: list[str],
    full_height: int,
    full_width: int,
) -> tuple[list[dict[str, object]], int, int]:
    resized_width = 518
    resized_height = round(full_height * (resized_width / full_width) / 14) * 14
    crop_top = max(0, (resized_height - 518) // 2)
    processed_height = min(resized_height, 518)
    geometry = [
        {
            "frame_name": name,
            "original_size_hw": [full_height, full_width],
            "resized_size_hw": [resized_height, resized_width],
            "crop_top_px": crop_top,
            "crop_left_px": 0,
            "processed_size_hw": [processed_height, resized_width],
            "scale_x": resized_width / full_width,
            "scale_y": resized_height / full_height,
            "supported_for_metric_alignment": True,
        }
        for name in frame_names
    ]
    return geometry, processed_height, resized_width


def test_board_anchor_handles_portrait_resize_and_crop() -> None:
    names = ["a.jpg", "b.jpg", "c.jpg"]
    full_height, full_width = 1920, 1080
    geometry, processed_height, processed_width = _portrait_preprocess_geometry(
        names, full_height, full_width
    )
    nominal = np.asarray(make_board().getChessboardCorners(), dtype=np.float64)
    focal_full = 0.75 * full_width
    cx_full, cy_full = full_width / 2, full_height / 2
    distance = 600.0

    sx = float(geometry[0]["scale_x"])
    sy = float(geometry[0]["scale_y"])
    crop_top = float(geometry[0]["crop_top_px"])
    intrinsic = np.array(
        [
            [focal_full * sx, 0.0, (cx_full + 0.5) * sx - 0.5],
            [0.0, focal_full * sy, (cy_full + 0.5) * sy - 0.5 - crop_top],
            [0.0, 0.0, 1.0],
        ]
    )

    offsets = [(-80.0, -40.0), (-60.0, -10.0), (-90.0, 20.0)]
    extrinsics = np.zeros((3, 3, 4), dtype=np.float64)
    extrinsics[:, :, :3] = np.eye(3)
    corner_frame_index: list[int] = []
    corner_id: list[int] = []
    corner_px: list[list[float]] = []
    for frame_index, (tx, ty) in enumerate(offsets):
        extrinsics[frame_index, :, 3] = [tx, ty, distance]
        for cid, (x, y, _z) in enumerate(nominal):
            u_full = focal_full * (x + tx) / distance + cx_full
            v_full = focal_full * (y + ty) / distance + cy_full
            v_processed = (v_full + 0.5) * sy - 0.5 - crop_top
            if 0 <= v_processed <= processed_height - 1:
                corner_frame_index.append(frame_index)
                corner_id.append(cid)
                corner_px.append([u_full, v_full])

    result = align_board_geometry(
        names,
        extrinsics,
        np.tile(intrinsic[None], (3, 1, 1)),
        np.full((3, processed_height, processed_width), distance, dtype=np.float64),
        (processed_height, processed_width),
        names,
        np.tile([[full_height, full_width]], (3, 1)),
        np.asarray(corner_frame_index),
        np.asarray(corner_id),
        np.asarray(corner_px, dtype=np.float64),
        preprocess_geometry=geometry,
        board_pose_found=np.array([True, True, True]),
    )

    assert result.scale == pytest.approx(1.0, abs=1e-6)
    assert result.rms_residual_mm == pytest.approx(0.0, abs=1e-5)


def test_board_anchor_recovers_nontrivial_similarity() -> None:
    (
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
    ) = _synthetic_board_alignment_inputs()

    # Start from an exact board-space synthetic reconstruction, then express all geometry in
    # another reconstruction frame related by x_board = s * (R @ x_recon) + t.
    expected_scale = 2.5
    expected_rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    expected_translation = np.array([40.0, -25.0, 12.0])

    # x_board = s * (R_similarity @ x_recon) + t_similarity. Substituting into
    # x_cam_board = R_board @ x_board + t_board and expressing camera coordinates in
    # reconstruction units gives R_recon = R_board @ R_similarity and
    # t_recon = (R_board @ t_similarity + t_board) / s.
    transformed_extrinsic = reconstruction_extrinsic.copy()
    for frame_index in range(len(frame_names)):
        board_rotation = reconstruction_extrinsic[frame_index, :, :3]
        board_translation = reconstruction_extrinsic[frame_index, :, 3]
        transformed_extrinsic[frame_index, :, :3] = (
            board_rotation @ expected_rotation
        )
        transformed_extrinsic[frame_index, :, 3] = (
            board_rotation @ expected_translation + board_translation
        ) / expected_scale
    transformed_depth = depth / expected_scale

    result = align_board_geometry(
        frame_names,
        transformed_extrinsic,
        reconstruction_intrinsic,
        transformed_depth,
        processed_size_hw,
        board_frame_names,
        board_image_size_hw,
        corner_frame_index,
        corner_id,
        corner_px,
        preprocess_geometry=_identity_preprocess_geometry(frame_names),
        board_pose_found=np.array([True, True, True]),
    )

    assert result.scale == pytest.approx(expected_scale, rel=1e-6)
    assert result.rotation == pytest.approx(expected_rotation, abs=1e-6)
    assert result.translation == pytest.approx(expected_translation, abs=1e-5)


def test_board_anchor_ignores_frames_without_trusted_board_pose() -> None:
    (
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
    ) = _synthetic_board_alignment_inputs()

    # Corrupt frame c completely. If pose_found is honoured, those 12 corners are excluded and
    # the remaining two frames still recover the exact identity alignment.
    depth[2] = 9999.0
    result = align_board_geometry(
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
        preprocess_geometry=_identity_preprocess_geometry(frame_names),
        board_pose_found=np.array([True, True, False]),
    )

    assert result.frame_count == 2
    assert result.observation_count == 24
    assert result.scale == pytest.approx(1.0, abs=1e-6)


def test_camera_centres_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        camera_centres(np.zeros((3, 4)))


def test_alignment_requires_non_collinear_poses() -> None:
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="collinear"):
        solve_similarity(centres, centres)


def test_apply_similarity_transform_matches_transform_points() -> None:
    """apply_similarity_transform is the formula AlignmentResult.transform_points delegates to
    (R-06 needs it directly, from a loaded metric_alignment.json, without a full AlignmentResult).
    """
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([5.0, 6.0, 7.0])
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])

    result = apply_similarity_transform(points, 2.5, rotation, translation)

    assert result == pytest.approx(2.5 * (points @ rotation.T) + translation)


def test_apply_similarity_transform_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        apply_similarity_transform(np.zeros((2, 2)), 1.0, np.eye(3), np.zeros(3))


def test_apply_similarity_transform_to_extrinsic_identity_is_a_no_op() -> None:
    extrinsic = np.array(
        [[[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 2.0]]]
    )

    board_extrinsic = apply_similarity_transform_to_extrinsic(
        extrinsic, scale=1.0, rotation=np.eye(3), translation=np.zeros(3)
    )

    assert board_extrinsic == pytest.approx(extrinsic)


def test_apply_similarity_transform_to_extrinsic_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        apply_similarity_transform_to_extrinsic(
            np.zeros((2, 3, 3)), 1.0, np.eye(3), np.zeros(3)
        )


def test_apply_similarity_transform_to_extrinsic_matches_unprojected_points() -> None:
    """R-07's load-bearing test: a point unprojected from a frame's depth using this function's
    board-frame extrinsic (with depth pre-scaled by `scale`) must land exactly where
    apply_similarity_transform would put the same point unprojected in the original frame. If the
    R_board/t_board derivation were wrong, this would silently warp or mis-scale every mesh R-07
    produces, with no error anywhere -- so this is checked by composition, not just inspection.
    """
    original_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    original_translation = np.array([1.0, -1.0, 2.0])
    extrinsic = np.concatenate(
        [original_rotation, original_translation[:, None]], axis=1
    )[None]
    intrinsic = np.array([[50.0, 0.0, 0.5], [0.0, 50.0, 0.5], [0.0, 0.0, 1.0]])[None]
    depth = np.array([[10.0, 12.0], [8.0, 15.0]])

    similarity_rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    similarity_translation = np.array([5.0, -2.0, 7.0])
    scale = 3.0

    original_points = unproject_depth_to_world(depth[None], extrinsic, intrinsic)
    expected_board_points = apply_similarity_transform(
        original_points.reshape(-1, 3),
        scale,
        similarity_rotation,
        similarity_translation,
    )

    board_extrinsic = apply_similarity_transform_to_extrinsic(
        extrinsic, scale, similarity_rotation, similarity_translation
    )
    board_points = unproject_depth_to_world(
        (scale * depth)[None], board_extrinsic, intrinsic
    )

    assert board_points.reshape(-1, 3) == pytest.approx(expected_board_points)


def test_project_points_to_camera_matches_hand_computed_pinhole_projection() -> None:
    """Shared by R-07's TSDF fusion and R-09's provenance (both need the identical perspective-
    projection math applied to different point sets) -- hand-verified independently rather than
    just trusted from its callers' own tests.
    """
    points = np.array(
        [[2.0, 3.0, 5.0], [0.0, 0.0, -1.0]]
    )  # second point is behind the camera
    rotation = np.eye(3)
    translation = np.zeros(3)
    intrinsic = np.array([[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]])

    u, v, z_cam = project_points_to_camera(points, rotation, translation, intrinsic)

    # u = fx*x/z + cx = 10*2/5 + 1 = 5; v = fy*y/z + cy = 10*3/5 + 1 = 7.
    assert u[0] == pytest.approx(5.0)
    assert v[0] == pytest.approx(7.0)
    assert z_cam[0] == pytest.approx(5.0)
    assert z_cam[1] == pytest.approx(
        -1.0
    )  # behind the camera -- callers must check this sign


def test_project_points_to_camera_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        project_points_to_camera(np.zeros((2, 2)), np.eye(3), np.zeros(3), np.eye(3))


def test_known_size_validation_uses_explicit_tolerance() -> None:
    result = validate_known_size(19.2, 20.0, 1.0)

    assert result["absolute_error_mm"] == pytest.approx(0.8)
    assert result["within_tolerance"] is True

    with pytest.raises(ValueError, match="positive"):
        validate_known_size(20.0, 0.0, 1.0)
