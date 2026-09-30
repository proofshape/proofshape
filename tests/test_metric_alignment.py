from __future__ import annotations

import numpy as np
import pytest

from recon.metric_alignment import (
    align_camera_centres,
    apply_similarity_transform,
    camera_centres,
    solve_similarity,
    validate_known_size,
)


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


def test_known_size_validation_uses_explicit_tolerance() -> None:
    result = validate_known_size(19.2, 20.0, 1.0)

    assert result["absolute_error_mm"] == pytest.approx(0.8)
    assert result["within_tolerance"] is True

    with pytest.raises(ValueError, match="positive"):
        validate_known_size(20.0, 0.0, 1.0)
