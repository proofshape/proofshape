"""Tests for R-02 board detection and per-frame pose (recon/board_pose.py).

The pose tests use synthetic frames: the D-034 board texture is warped into a camera image with a
known K and a known pose, so the recovered pose can be checked against ground truth. The golden
capture can't serve that role — it has no ground-truth poses, and it lives outside git, so CI
can't see it. The real-frame check is recorded in stories/R-02.md instead.
"""

from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from recon import board_pose
from recon.board_pose import (
    BOARD_INTERIOR_CORNERS,
    INTRINSICS_SOURCE_EXIF,
    INTRINSICS_SOURCE_SUPPLIED,
    MIN_CORNERS_FOR_POSE,
    REASON_IMAGE_UNREADABLE,
    REASON_INTRINSICS_UNAVAILABLE,
    REASON_NOT_DETECTED,
    REASON_OCCLUDED,
    REASON_POSE_FAILED,
    STATUS_OK,
    PoseFailure,
    corners_are_collinear,
    detect_charuco_corners,
    estimate_board_poses,
    estimate_frame_pose,
    intrinsics_from_focal_35mm,
    load_frame_gray,
    main,
    make_board,
    make_detector,
    read_exif_fields,
    solve_board_pose,
)

BOARD_PDF = Path(__file__).resolve().parents[1] / "fixtures" / "charuco_board.pdf"
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 1200
FOCAL_35MM = 26.0  # what the golden capture's iPhone 16 frames report
TEXTURE_PX_PER_MM = 8
TEXTURE_MARGIN_MM = 10
BOARD_CENTRE_MM = np.array([80.0, 60.0, 0.0])
# Rotation vector in radians: an oblique, slightly rolled view of the board.
TILTED_VIEW = (0.4, -0.2, 0.1)

# Test tolerances on a noise-free synthetic render, not claims about real-frame accuracy.
MAX_ROTATION_ERROR_DEG = 0.5
MAX_TRANSLATION_ERROR_MM = 1.0

# A "part" sitting in the middle of the board, in board millimetres (x0, y0, x1, y1).
PART_IN_MIDDLE_MM = (50, 35, 110, 85)
# Covers every row but the bottom one: the corners left visible are all on one line.
ALL_BUT_ONE_ROW_MM = (-10, -10, 170, 80)
# Leaves a strip of markers visible but fewer than MIN_CORNERS_FOR_POSE corners.
MOST_OF_BOARD_MM = (-10, -10, 120, 120)


def render_board(
    rotation_vector=TILTED_VIEW,
    distance_mm: float = 400.0,
    occlude_mm: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the board as a camera at a known pose would see it.

    Returns (grayscale image, true extrinsic (3,4), K). The camera looks at the board centre from
    ``distance_mm`` away. ``occlude_mm`` paints a flat grey rectangle on the board plane, standing
    in for a part resting on it.
    """
    board = make_board()
    intrinsic = intrinsics_from_focal_35mm(FOCAL_35MM, IMAGE_WIDTH, IMAGE_HEIGHT)
    margin_px = TEXTURE_MARGIN_MM * TEXTURE_PX_PER_MM
    texture = board.generateImage(
        (
            160 * TEXTURE_PX_PER_MM + 2 * margin_px,
            120 * TEXTURE_PX_PER_MM + 2 * margin_px,
        ),
        marginSize=margin_px,
        borderBits=1,
    )

    rotation, _ = cv2.Rodrigues(np.array(rotation_vector, dtype=np.float64))
    # Choose t so the board centre lands on the optical axis at the requested distance.
    translation = np.array([0.0, 0.0, distance_mm]) - rotation @ BOARD_CENTRE_MM

    # Board plane (z = 0) to image is the homography K [r1 r2 t]; the texture is in pixels, so
    # map texture pixels back to board millimetres first.
    board_mm_to_image = intrinsic @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation]
    )
    texture_px_from_mm = np.array(
        [
            [TEXTURE_PX_PER_MM, 0, margin_px],
            [0, TEXTURE_PX_PER_MM, margin_px],
            [0, 0, 1.0],
        ]
    )
    image = cv2.warpPerspective(
        texture,
        board_mm_to_image @ np.linalg.inv(texture_px_from_mm),
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        flags=cv2.INTER_AREA,
        borderValue=128,
    )

    if occlude_mm is not None:
        x0, y0, x1, y1 = occlude_mm
        rectangle_mm = np.array(
            [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float64
        )
        rectangle_px = cv2.perspectiveTransform(rectangle_mm, board_mm_to_image)[0]
        cv2.fillConvexPoly(image, np.round(rectangle_px).astype(np.int32), 60)

    true_extrinsic = np.hstack([rotation, translation.reshape(3, 1)])
    return image, true_extrinsic, intrinsic


def save_jpeg(
    path: Path,
    gray: np.ndarray,
    focal_35mm: float | None = FOCAL_35MM,
    orientation: int | None = None,
) -> Path:
    """Save a frame as a phone would: JPEG with the EXIF fields R-02 reads."""
    exif = Image.Exif()
    if orientation is not None:
        exif[board_pose.EXIF_ORIENTATION_TAG] = orientation
    if focal_35mm is not None:
        exif.get_ifd(0x8769)[board_pose.EXIF_FOCAL_35MM_TAG] = int(focal_35mm)
    Image.fromarray(gray).save(path, quality=95, exif=exif)
    return path


def rotation_error_deg(estimated: np.ndarray, truth: np.ndarray) -> float:
    relative = estimated[:, :3] @ truth[:, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def assert_pose_close(estimated: np.ndarray, truth: np.ndarray) -> None:
    assert rotation_error_deg(estimated, truth) < MAX_ROTATION_ERROR_DEG
    assert np.linalg.norm(estimated[:, 3] - truth[:, 3]) < MAX_TRANSLATION_ERROR_MM


# --- board definition and intrinsics -------------------------------------------------------


def test_board_matches_decision_d034() -> None:
    board = make_board()

    assert board.getChessboardSize() == (8, 6)
    assert board.getSquareLength() == 20.0
    assert board.getMarkerLength() == 15.0
    # OpenCV's legacy layout is what the calib.io print actually is (D-034 correction).
    assert board.getLegacyPattern() is True
    assert len(board.getChessboardCorners()) == BOARD_INTERIOR_CORNERS == 35
    # Interior corners sit on the 20 mm grid, starting one square in from the outer edge.
    assert np.allclose(board.getChessboardCorners()[0], [20.0, 20.0, 0.0])
    assert np.allclose(board.getChessboardCorners()[-1], [140.0, 100.0, 0.0])


def render_committed_board_pdf(points_to_px: float) -> np.ndarray:
    """Rasterise fixtures/charuco_board.pdf — the file the golden-capture board was printed from."""
    pdf = pdfium.PdfDocument(str(BOARD_PDF))
    try:
        return pdf[0].render(scale=points_to_px, grayscale=True).to_numpy()
    finally:
        pdf.close()


def test_code_board_detects_the_committed_print_file() -> None:
    """The board in code must match the board on paper, not just the board in D-034's prose.

    This is the test that would have caught the legacy-layout mismatch: synthetic renders use
    make_board() on both sides, so they agree with each other even when both are wrong.
    """
    points_to_px = 2.0
    image = render_committed_board_pdf(points_to_px)
    board = make_board()

    corner_px, corner_id, _ = detect_charuco_corners(image, make_detector(board))

    assert len(corner_id) == BOARD_INTERIOR_CORNERS

    # The PDF is drawn in points (1/72 inch), so its corner pitch must come out as 20 mm. This
    # is the board-pitch scale source of D-007, checked at its origin.
    px_per_mm = points_to_px * 72.0 / 25.4
    corner_by_id = dict(zip(corner_id.tolist(), corner_px))
    pitch_mm = np.linalg.norm(corner_by_id[1] - corner_by_id[0]) / px_per_mm
    assert pitch_mm == pytest.approx(20.0, rel=0.01)

    # And the other layout finds no corners at all, which is the failure mode this guards.
    board.setLegacyPattern(False)
    assert len(detect_charuco_corners(image, make_detector(board))[1]) == 0


def test_intrinsics_from_focal_35mm_uses_the_diagonal() -> None:
    intrinsic = intrinsics_from_focal_35mm(26.0, 5712, 4284)

    # 26 mm-equivalent on a 5712x4284 frame: f = 26 * 7140 px / 43.267 mm.
    assert intrinsic[0, 0] == pytest.approx(26.0 * 7140.0 / np.hypot(36.0, 24.0))
    assert intrinsic[1, 1] == intrinsic[0, 0]
    assert intrinsic[0, 2] == pytest.approx(2855.5)
    assert intrinsic[1, 2] == pytest.approx(2141.5)
    # The width-based shortcut (f = 26/36 * width) would be ~4% short; guard against it.
    assert intrinsic[0, 0] > 1.03 * (26.0 / 36.0 * 5712)


def test_intrinsics_from_focal_35mm_rejects_non_positive_focal() -> None:
    with pytest.raises(ValueError, match="positive"):
        intrinsics_from_focal_35mm(0.0, 100, 100)


# --- EXIF and pixel frame --------------------------------------------------------------------


def test_read_exif_fields_returns_focal_and_orientation(tmp_path: Path) -> None:
    path = save_jpeg(
        tmp_path / "f.jpg", np.zeros((20, 40), np.uint8), 26.0, orientation=6
    )

    assert read_exif_fields(path) == (26.0, 6)


def test_read_exif_fields_reports_missing_and_zero_focal_as_none(
    tmp_path: Path,
) -> None:
    no_exif = save_jpeg(tmp_path / "a.jpg", np.zeros((8, 8), np.uint8), None)
    zero_focal = tmp_path / "b.jpg"
    exif = Image.Exif()
    exif.get_ifd(0x8769)[board_pose.EXIF_FOCAL_35MM_TAG] = 0
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(zero_focal, exif=exif)

    assert read_exif_fields(no_exif) == (None, None)
    assert read_exif_fields(zero_focal)[0] is None


def test_read_exif_fields_tolerates_a_file_that_is_not_an_image(tmp_path: Path) -> None:
    garbage = tmp_path / "broken.jpg"
    garbage.write_bytes(b"not a jpeg")

    assert read_exif_fields(garbage) == (None, None)


def test_load_frame_gray_ignores_exif_orientation(tmp_path: Path) -> None:
    # Stored 40 wide x 20 tall, tagged "rotate 90 CW to display" — like the s-02 golden frames.
    path = save_jpeg(
        tmp_path / "rotated.jpg", np.zeros((20, 40), np.uint8), orientation=6
    )

    assert load_frame_gray(path).shape == (20, 40)
    # OpenCV's default *would* rotate it; this is why the flag matters (R-01's VGGT loader doesn't).
    assert cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).shape == (40, 20)


# --- pose solving ----------------------------------------------------------------------------


def test_corners_are_collinear() -> None:
    one_row = np.array([[20, 20, 0], [40, 20, 0], [60, 20, 0]], dtype=np.float64)
    one_diagonal = np.array([[20, 20, 0], [40, 40, 0], [60, 60, 0]], dtype=np.float64)
    triangle = np.array([[20, 20, 0], [40, 20, 0], [20, 40, 0]], dtype=np.float64)

    assert corners_are_collinear(one_row)
    assert corners_are_collinear(one_diagonal)
    assert not corners_are_collinear(triangle)


def test_pose_recovered_on_an_unoccluded_render() -> None:
    image, true_extrinsic, intrinsic = render_board()
    board = make_board()

    corner_px, corner_id, marker_count = detect_charuco_corners(
        image, make_detector(board)
    )
    solution = solve_board_pose(corner_px, corner_id, board, intrinsic)

    assert len(corner_id) == BOARD_INTERIOR_CORNERS
    assert marker_count > 0
    assert_pose_close(solution.extrinsic, true_extrinsic)
    assert solution.reprojection_rms_px < 1.0


def test_pose_recovered_when_the_part_hides_the_middle_of_the_board() -> None:
    # Acceptance criterion: a frame where the board is partly hidden by the part.
    image, true_extrinsic, intrinsic = render_board(occlude_mm=PART_IN_MIDDLE_MM)
    board = make_board()

    corner_px, corner_id, _ = detect_charuco_corners(image, make_detector(board))
    solution = solve_board_pose(corner_px, corner_id, board, intrinsic)

    assert MIN_CORNERS_FOR_POSE <= len(corner_id) < BOARD_INTERIOR_CORNERS
    assert_pose_close(solution.extrinsic, true_extrinsic)


def test_pose_rejected_when_visible_corners_are_all_on_one_row() -> None:
    image, _, intrinsic = render_board(occlude_mm=ALL_BUT_ONE_ROW_MM)
    board = make_board()
    corner_px, corner_id, _ = detect_charuco_corners(image, make_detector(board))
    assert (
        len(corner_id) >= MIN_CORNERS_FOR_POSE
    )  # enough corners — the geometry is the problem

    with pytest.raises(PoseFailure) as failure:
        solve_board_pose(corner_px, corner_id, board, intrinsic)

    assert failure.value.reason == REASON_OCCLUDED
    assert "one line" in failure.value.message


def test_pose_solved_from_real_l_shaped_corners() -> None:
    """Regression, from real data: s-03/IMG_3946, where only board column 0 plus corner 1 was
    visible. That's not collinear, so the pose is determined, but OpenCV's IPPE solver refused
    it (and 2 other s-03 frames). IPPE accepts a *noise-free* projection of the same pose, so
    only the real detected pixel positions reproduce the failure; they're copied here verbatim
    (12 numbers, not a photograph). Intrinsics are the EXIF-derived K for that 5712x4284 frame.
    """
    corner_id = np.array([0, 1, 7, 14, 21, 28], dtype=np.int32)
    corner_px = np.array(
        [
            [2000.93, 2669.41],
            [2031.55, 2356.03],
            [2333.57, 2667.86],
            [2665.14, 2667.01],
            [2999.10, 2667.17],
            [3334.92, 2669.14],
        ]
    )
    intrinsic = intrinsics_from_focal_35mm(FOCAL_35MM, 5712, 4284)

    solution = solve_board_pose(corner_px, corner_id, make_board(), intrinsic)

    # No ground truth for a real frame, so check it's a sound fit rather than a specific pose.
    assert solution.reprojection_rms_px < 2.0
    assert solution.extrinsic[2, 3] > 0


def test_pose_rejected_with_too_few_corners() -> None:
    board = make_board()
    intrinsic = intrinsics_from_focal_35mm(FOCAL_35MM, IMAGE_WIDTH, IMAGE_HEIGHT)
    too_few = MIN_CORNERS_FOR_POSE - 1

    with pytest.raises(PoseFailure) as failure:
        solve_board_pose(
            np.zeros((too_few, 2)), np.arange(too_few, dtype=np.int32), board, intrinsic
        )

    assert failure.value.reason == REASON_OCCLUDED
    assert f"only {too_few} of 35" in failure.value.message


def test_pose_reports_solver_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    image, _, intrinsic = render_board()
    board = make_board()
    corner_px, corner_id, _ = detect_charuco_corners(image, make_detector(board))
    monkeypatch.setattr(
        board_pose.cv2, "solvePnP", lambda *args, **kwargs: (False, None, None)
    )

    with pytest.raises(PoseFailure) as failure:
        solve_board_pose(corner_px, corner_id, board, intrinsic)

    assert failure.value.reason == REASON_POSE_FAILED


# --- one frame, end to end -------------------------------------------------------------------


def estimate(path: Path, intrinsic_override=None):
    board = make_board()
    return estimate_frame_pose(path, board, make_detector(board), intrinsic_override)


def test_frame_pose_from_exif_intrinsics(tmp_path: Path) -> None:
    image, true_extrinsic, _ = render_board(occlude_mm=PART_IN_MIDDLE_MM)
    result = estimate(save_jpeg(tmp_path / "f.jpg", image))

    assert result.status == STATUS_OK
    assert result.intrinsics_source == INTRINSICS_SOURCE_EXIF
    assert result.image_size_hw == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert_pose_close(result.extrinsic, true_extrinsic)


def test_blank_frame_reports_board_not_detected(tmp_path: Path) -> None:
    blank = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 128, np.uint8)
    result = estimate(save_jpeg(tmp_path / "blank.jpg", blank))

    assert result.status == REASON_NOT_DETECTED
    assert result.extrinsic is None


def test_mostly_hidden_board_reports_partially_occluded(tmp_path: Path) -> None:
    image, _, _ = render_board(occlude_mm=MOST_OF_BOARD_MM)
    result = estimate(save_jpeg(tmp_path / "hidden.jpg", image))

    assert result.status == REASON_OCCLUDED
    assert result.marker_count > 0  # the board is there, just mostly covered
    assert result.extrinsic is None


def test_missing_focal_reports_intrinsics_unavailable_but_keeps_corners(
    tmp_path: Path,
) -> None:
    image, _, _ = render_board()
    result = estimate(save_jpeg(tmp_path / "f.jpg", image, focal_35mm=None))

    assert result.status == REASON_INTRINSICS_UNAVAILABLE
    assert result.extrinsic is None
    # Corners are still worth keeping for R-03's calibration.
    assert len(result.corner_id) == BOARD_INTERIOR_CORNERS


def test_supplied_intrinsics_take_precedence_over_exif(tmp_path: Path) -> None:
    image, true_extrinsic, intrinsic = render_board()
    # Deliberately wrong EXIF focal: the supplied K must be the one used.
    result = estimate(save_jpeg(tmp_path / "f.jpg", image, focal_35mm=50), intrinsic)

    assert result.status == STATUS_OK
    assert result.intrinsics_source == INTRINSICS_SOURCE_SUPPLIED
    assert np.array_equal(result.intrinsic, intrinsic)
    assert_pose_close(result.extrinsic, true_extrinsic)


def test_unreadable_frame_is_reported_not_raised(tmp_path: Path) -> None:
    garbage = tmp_path / "broken.jpg"
    garbage.write_bytes(b"not a jpeg")

    result = estimate(garbage)

    assert result.status == REASON_IMAGE_UNREADABLE


# --- a whole capture -------------------------------------------------------------------------


@pytest.fixture
def capture_dir(tmp_path: Path) -> Path:
    """Three frames: a good one, one with the part in the middle, and one with no board."""
    folder = tmp_path / "s-test"
    folder.mkdir()
    save_jpeg(folder / "IMG_0001.jpg", render_board()[0])
    save_jpeg(folder / "IMG_0002.jpg", render_board(occlude_mm=PART_IN_MIDDLE_MM)[0])
    save_jpeg(
        folder / "IMG_0003.jpg", np.full((IMAGE_HEIGHT, IMAGE_WIDTH), 128, np.uint8)
    )
    (folder / "notes.txt").write_text("not a frame")
    return folder


def test_estimate_board_poses_writes_outputs(capture_dir: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    run_info = estimate_board_poses(capture_dir, out_dir)

    assert run_info["frame_count"] == 3
    assert run_info["posed_count"] == 2
    assert run_info["failure_counts"] == {REASON_NOT_DETECTED: 1}
    statuses = [frame["status"] for frame in run_info["frames"]]
    assert statuses == [STATUS_OK, STATUS_OK, REASON_NOT_DETECTED]
    assert run_info["frames"][2]["message"]  # every failure says why

    saved = np.load(out_dir / "board_poses.npz")
    assert list(saved["frame_names"]) == [
        "IMG_0001.jpg",
        "IMG_0002.jpg",
        "IMG_0003.jpg",
    ]
    assert saved["pose_found"].tolist() == [True, True, False]
    assert saved["extrinsic"].shape == (3, 3, 4)
    assert np.isfinite(saved["extrinsic"][:2]).all()
    # A frame without a pose must be NaN, never an identity/zero pose that looks measured.
    assert np.isnan(saved["extrinsic"][2]).all()
    assert np.isnan(saved["reprojection_rms_px"][2])
    # The corner table is consistent with the per-frame counts.
    assert len(saved["corner_id"]) == int(saved["corner_count"].sum())
    assert (
        len(saved["corner_px"])
        == len(saved["corner_frame_index"])
        == len(saved["corner_id"])
    )
    assert (out_dir / "board_poses.json").is_file()


def test_estimate_board_poses_refuses_to_overwrite(
    capture_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    estimate_board_poses(capture_dir, out_dir)

    with pytest.raises(FileExistsError, match="already has files"):
        estimate_board_poses(capture_dir, out_dir)


def test_missing_capture_folder_raises_with_the_download_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_golden_capture"):
        estimate_board_poses(tmp_path / "nope", tmp_path / "out")


def test_main_exit_codes(capture_dir: Path, tmp_path: Path, capsys) -> None:
    assert main([str(capture_dir), "--out-dir", str(tmp_path / "ok")]) == 0
    assert "2/3 frames posed" in capsys.readouterr().out

    assert main([str(tmp_path / "missing"), "--out-dir", str(tmp_path / "x")]) == 1

    no_board = tmp_path / "no-board"
    no_board.mkdir()
    save_jpeg(no_board / "a.jpg", np.full((64, 64), 128, np.uint8))
    assert main([str(no_board), "--out-dir", str(tmp_path / "nb")]) == 1
