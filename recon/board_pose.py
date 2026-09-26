"""R-02: detect the ChArUco board in every frame and solve a metric camera pose in the board frame.

This is the metric half of the reconstruction. R-01's VGGT poses live in an arbitrary, unit-less
frame; the poses written here are in millimetres, in the frame of the printed board, and R-04
aligns the two. Nothing here depends on PyTorch or a GPU, so the whole module runs on a laptop and
in CI.

Board frame (OpenCV's ChArUco convention, kept rather than re-oriented so R-03's calibration and
this module agree without a conversion step): origin at the outer corner of the square next to
the first interior corner (id 0, at (20, 20, 0) mm), x along the 8-square side, y along the
6-square side, z = 0 on the paper and pointing *into* the table. A camera above the board
therefore has a negative z camera centre.

Output layout (one directory per capture), matching R-01's convention of camera-from-world
extrinsics with x_cam = R @ x_world + t, so R-04 can pair the two by frame name:

    board_poses.npz   frame_names (S,), pose_found (S,) bool,
                      extrinsic (S,3,4) camera-from-board in mm (NaN where no pose),
                      intrinsic (S,3,3) (NaN where unavailable), image_size_hw (S,2),
                      marker_count (S,), corner_count (S,), reprojection_rms_px (S,) (NaN where no pose),
                      corner_frame_index (N,), corner_id (N,), corner_px (N,2)  — every detected
                      ChArUco corner, kept so R-03 can calibrate without re-detecting
    board_poses.json  board parameters, intrinsics source, per-frame status and failure reason,
                      summary counts and timings

Pixel frame: frames are read as stored on the sensor, *ignoring* the EXIF Orientation tag. VGGT's
loader (R-01) does the same, and the two stories must agree on the pixel frame or R-04 would be
aligning cameras that are rotated about their optical axis relative to each other. The tag is
recorded per frame so nobody has to rediscover this.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import ExifTags, Image, UnidentifiedImageError

FRAME_SUFFIXES = (".jpg", ".jpeg")

# The printed golden-capture board, fixed by D-034. If any of these change, the PDF in
# fixtures/charuco_board.pdf has to be regenerated to match, and vice versa.
BOARD_DICTIONARY_ID = cv2.aruco.DICT_5X5_250
BOARD_SQUARES_X = 8
BOARD_SQUARES_Y = 6
BOARD_SQUARE_MM = 20.0
BOARD_MARKER_MM = 15.0
BOARD_INTERIOR_CORNERS = (BOARD_SQUARES_X - 1) * (BOARD_SQUARES_Y - 1)

# A planar pose is solvable from four non-collinear points, but four leaves no redundancy to
# average out corner noise. Six is a chosen guard, not a measured threshold; revisit once R-03/R-04
# show how pose quality actually falls off with corner count.
MIN_CORNERS_FOR_POSE = 6

# "35 mm equivalent" focal length is defined by matching the diagonal field of view of a 36x24 mm
# frame (CIPA DC-008). Using the diagonal rather than the 36 mm width matters for a 4:3 phone
# sensor: the width-based shortcut gives a focal about 4% short.
FULL_FRAME_DIAGONAL_MM = math.hypot(36.0, 24.0)

EXIF_ORIENTATION_TAG = 0x0112
EXIF_FOCAL_35MM_TAG = 0xA405

STATUS_OK = "ok"
# Failure reasons. The first two are also contract ErrorDetail codes (contracts/openapi.yaml);
# the rest are R-02-internal and R-14 decides how to surface them.
REASON_NOT_DETECTED = "board_not_detected"
REASON_OCCLUDED = "board_partially_occluded"
REASON_IMAGE_UNREADABLE = "image_unreadable"
REASON_INTRINSICS_UNAVAILABLE = "intrinsics_unavailable"
REASON_POSE_FAILED = "pose_solve_failed"

INTRINSICS_SOURCE_EXIF = "exif_35mm_equivalent"
INTRINSICS_SOURCE_SUPPLIED = "supplied"


def make_board() -> cv2.aruco.CharucoBoard:
    """Build the D-034 board: DICT_5X5_250, 8x6 squares of 20 mm, 15 mm markers.

    The layout flag is OpenCV's *legacy* pattern — see the comment below and D-034's correction.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(BOARD_DICTIONARY_ID)
    board = cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y), BOARD_SQUARE_MM, BOARD_MARKER_MM, dictionary
    )
    # The print was generated with calib.io's "ChArUco Legacy" box unchecked, but that board is
    # the one OpenCV calls the *legacy* layout: the committed fixtures/charuco_board.pdf detects
    # 35/35 corners with setLegacyPattern(True) and 0/35 with False, and the real golden frames
    # behave the same way. With the wrong flag the markers are still found but no corners are
    # interpolated, so every frame fails. tests/test_board_pose.py checks this against the PDF.
    board.setLegacyPattern(True)
    return board


def make_detector(board: cv2.aruco.CharucoBoard) -> cv2.aruco.CharucoDetector:
    return cv2.aruco.CharucoDetector(board)


def find_capture_frames(capture_dir: Path) -> list[Path]:
    """Return the capture's JPEG frames sorted by name.

    Same rule as R-01's runner: R-04 pairs VGGT poses with board poses by frame name, and a stable
    order keeps the two output files row-aligned as well.
    """
    capture_dir = Path(capture_dir)
    if not capture_dir.is_dir():
        raise FileNotFoundError(
            f"Capture folder not found: {capture_dir}. "
            "Run `bash fixtures/download_golden_capture.sh` first."
        )

    frames: list[Path] = []
    for path in sorted(capture_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES:
            frames.append(path)

    if not frames:
        raise FileNotFoundError(f"No .jpg/.jpeg frames found in {capture_dir}.")
    return frames


def read_exif_fields(image_path: Path) -> tuple[float | None, int | None]:
    """Return (35 mm equivalent focal length, EXIF orientation), either None if absent.

    Pillow reads only the header here, not the pixels, so this is cheap on 24 MP frames. A file
    Pillow can't parse at all yields (None, None); whether the frame is usable is then decided by
    the pixel decode, which reports it as unreadable rather than this function raising.
    """
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            orientation = exif.get(EXIF_ORIENTATION_TAG)
            focal_35mm = exif.get_ifd(ExifTags.IFD.Exif).get(EXIF_FOCAL_35MM_TAG)
    except (UnidentifiedImageError, OSError):
        return None, None

    # A 0 focal is how some cameras spell "unknown"; treat it as missing rather than dividing by it.
    if focal_35mm is not None and float(focal_35mm) <= 0:
        focal_35mm = None
    return (
        float(focal_35mm) if focal_35mm is not None else None,
        int(orientation) if orientation is not None else None,
    )


def intrinsics_from_focal_35mm(
    focal_35mm: float, width: int, height: int
) -> np.ndarray:
    """Pinhole K from a 35 mm equivalent focal, principal point at the image centre.

    This is an approximation, and knowingly so: it ignores the principal-point offset and lens
    distortion, and phones round the EXIF value to a whole millimetre. It exists so R-02 can
    produce a pose before R-03 calibrates intrinsics from the board; R-03's result replaces it
    through ``intrinsic_override``. Poses built on it carry that source in board_poses.json.
    """
    if focal_35mm <= 0:
        raise ValueError(f"focal_35mm must be positive, got {focal_35mm}")
    focal_px = focal_35mm * math.hypot(width, height) / FULL_FRAME_DIAGONAL_MM
    # OpenCV puts pixel centres on integer coordinates, so the image centre is (w-1)/2, (h-1)/2.
    return np.array(
        [
            [focal_px, 0.0, (width - 1) / 2.0],
            [0.0, focal_px, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_frame_gray(image_path: Path) -> np.ndarray | None:
    """Read a frame as grayscale in its stored pixel frame (EXIF orientation deliberately ignored).

    See the module docstring: this must match the pixel frame R-01's VGGT loader sees.
    """
    return cv2.imread(
        str(image_path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION
    )


def detect_charuco_corners(
    gray: np.ndarray, detector: cv2.aruco.CharucoDetector
) -> tuple[np.ndarray, np.ndarray, int]:
    """Detect the board and return (corner_px (N,2), corner_id (N,), marker_count).

    Empty arrays, not None, when nothing is found, so callers can always ask for ``len``.
    """
    charuco_corners, charuco_ids, _marker_corners, marker_ids = detector.detectBoard(
        gray
    )
    marker_count = 0 if marker_ids is None else len(marker_ids)
    if charuco_ids is None or len(charuco_ids) == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.int32),
            marker_count,
        )
    corner_px = charuco_corners.reshape(-1, 2).astype(np.float64)
    corner_id = charuco_ids.reshape(-1).astype(np.int32)
    return corner_px, corner_id, marker_count


def corners_are_collinear(board_points_mm: np.ndarray) -> bool:
    """True if the board-frame corner positions all lie on one line.

    Collinear points leave the rotation about that line undetermined, so solvePnP would return a
    confident-looking but arbitrary pose. Board corners sit exactly on a 20 mm grid, so "collinear"
    here is exact (the second singular value is zero up to float error), not a tuned threshold.
    """
    centred = board_points_mm[:, :2] - board_points_mm[:, :2].mean(axis=0)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    return bool(singular_values[-1] < 1e-6 * BOARD_SQUARE_MM)


@dataclass
class PoseSolution:
    extrinsic: np.ndarray  # (3,4) camera-from-board, mm
    reprojection_rms_px: float


class PoseFailure(Exception):
    """A frame's pose could not be solved. ``reason`` is one of the REASON_* codes."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def solve_board_pose(
    corner_px: np.ndarray,
    corner_id: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    intrinsic: np.ndarray,
) -> PoseSolution:
    """Solve the camera-from-board pose from detected ChArUco corners.

    Raises PoseFailure with a specific reason instead of returning a doubtful pose: a quiet wrong
    pose here becomes a quiet wrong scale in R-04.
    """
    corner_count = len(corner_id)
    if corner_count < MIN_CORNERS_FOR_POSE:
        raise PoseFailure(
            REASON_OCCLUDED,
            f"only {corner_count} of {BOARD_INTERIOR_CORNERS} board corners visible; "
            f"need at least {MIN_CORNERS_FOR_POSE} for a pose",
        )

    all_board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    board_points_mm = all_board_corners[corner_id]
    if corners_are_collinear(board_points_mm):
        raise PoseFailure(
            REASON_OCCLUDED,
            f"the {corner_count} visible board corners lie on one line, which leaves the pose "
            "undetermined",
        )

    # Lens distortion is not modelled until R-03 calibrates it; zeros is the honest default for
    # an EXIF-only K, and it's what the reprojection error below is measured against.
    distortion = np.zeros(5, dtype=np.float64)

    # SQPnP finds the global minimum without a starting guess, so it can't get stuck the way the
    # default iterative solver can on a few corners. It replaced IPPE, OpenCV's planar-specific
    # solver, after IPPE refused 3 of 22 real s-03 frames whose visible corners formed an "L"
    # (one column plus one neighbour). SQPnP solved those, and on every frame both solved, the
    # two gave the same pose after refinement. LM refinement then minimises reprojection error
    # over all the corners.
    solved, rotation_vector, translation = cv2.solvePnP(
        board_points_mm, corner_px, intrinsic, distortion, flags=cv2.SOLVEPNP_SQPNP
    )
    if not solved:
        raise PoseFailure(REASON_POSE_FAILED, "solvePnP (SQPnP) did not converge")
    rotation_vector, translation = cv2.solvePnPRefineLM(
        board_points_mm, corner_px, intrinsic, distortion, rotation_vector, translation
    )

    if not (np.isfinite(rotation_vector).all() and np.isfinite(translation).all()):
        raise PoseFailure(REASON_POSE_FAILED, "solved pose contains non-finite values")
    # The board origin must be in front of the camera. A negative depth is the mirror-image
    # solution a planar solve can fall into, and it would be silently wrong downstream.
    if float(translation[2, 0]) <= 0:
        raise PoseFailure(
            REASON_POSE_FAILED,
            "solved pose puts the board behind the camera (mirror solution)",
        )

    projected, _ = cv2.projectPoints(
        board_points_mm, rotation_vector, translation, intrinsic, distortion
    )
    residuals = projected.reshape(-1, 2) - corner_px
    reprojection_rms_px = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))

    rotation, _ = cv2.Rodrigues(rotation_vector)
    extrinsic = np.hstack([rotation, translation.reshape(3, 1)])
    return PoseSolution(extrinsic=extrinsic, reprojection_rms_px=reprojection_rms_px)


@dataclass
class FrameResult:
    frame_name: str
    status: str  # STATUS_OK or one of the REASON_* codes
    message: str
    image_size_hw: tuple[int, int] | None
    exif_orientation: int | None
    intrinsic: np.ndarray | None
    intrinsics_source: str | None
    marker_count: int
    corner_px: np.ndarray
    corner_id: np.ndarray
    extrinsic: np.ndarray | None
    reprojection_rms_px: float | None


def estimate_frame_pose(
    image_path: Path,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
    intrinsic_override: np.ndarray | None = None,
) -> FrameResult:
    """Detect the board in one frame and solve its pose. Never raises for a bad frame.

    A failed frame is a normal outcome (the supplier's hand, the part itself, glare); it's returned
    with a reason so the caller can report every failure rather than stopping at the first.
    """
    image_path = Path(image_path)
    focal_35mm, exif_orientation = read_exif_fields(image_path)
    empty_px = np.empty((0, 2), dtype=np.float64)
    empty_id = np.empty(0, dtype=np.int32)

    gray = load_frame_gray(image_path)
    if gray is None:
        return FrameResult(
            frame_name=image_path.name,
            status=REASON_IMAGE_UNREADABLE,
            message="OpenCV could not decode the image",
            image_size_hw=None,
            exif_orientation=exif_orientation,
            intrinsic=None,
            intrinsics_source=None,
            marker_count=0,
            corner_px=empty_px,
            corner_id=empty_id,
            extrinsic=None,
            reprojection_rms_px=None,
        )
    height, width = gray.shape

    # Detection runs before intrinsics are resolved on purpose: the corners are worth keeping for
    # R-03's calibration even when this frame's EXIF can't give us a K.
    corner_px, corner_id, marker_count = detect_charuco_corners(gray, detector)

    def failed(
        reason: str,
        message: str,
        intrinsic: np.ndarray | None = None,
        intrinsics_source: str | None = None,
    ) -> FrameResult:
        return FrameResult(
            frame_name=image_path.name,
            status=reason,
            message=message,
            image_size_hw=(height, width),
            exif_orientation=exif_orientation,
            intrinsic=intrinsic,
            intrinsics_source=intrinsics_source,
            marker_count=marker_count,
            corner_px=corner_px,
            corner_id=corner_id,
            extrinsic=None,
            reprojection_rms_px=None,
        )

    if intrinsic_override is not None:
        intrinsic = np.asarray(intrinsic_override, dtype=np.float64)
        intrinsics_source = INTRINSICS_SOURCE_SUPPLIED
    elif focal_35mm is not None:
        intrinsic = intrinsics_from_focal_35mm(focal_35mm, width, height)
        intrinsics_source = INTRINSICS_SOURCE_EXIF
    else:
        # No silent default focal: a guessed K would give a pose whose scale error nobody could
        # see. R-03's calibrated intrinsics are the fix for frames like this.
        return failed(
            REASON_INTRINSICS_UNAVAILABLE,
            "no EXIF 35 mm equivalent focal length and no intrinsics supplied",
        )

    # Zero markers means the board isn't in the frame at all. Markers but too few corners means
    # the board is there and mostly hidden — solve_board_pose reports that as occluded, which is
    # the more useful thing to tell the person holding the phone.
    if marker_count == 0:
        return failed(
            REASON_NOT_DETECTED,
            "no board markers found in the frame",
            intrinsic,
            intrinsics_source,
        )

    try:
        solution = solve_board_pose(corner_px, corner_id, board, intrinsic)
    except PoseFailure as failure:
        return failed(failure.reason, failure.message, intrinsic, intrinsics_source)

    return FrameResult(
        frame_name=image_path.name,
        status=STATUS_OK,
        message=(
            f"{len(corner_id)} of {BOARD_INTERIOR_CORNERS} corners, "
            f"reprojection RMS {solution.reprojection_rms_px:.2f} px"
        ),
        image_size_hw=(height, width),
        exif_orientation=exif_orientation,
        intrinsic=intrinsic,
        intrinsics_source=intrinsics_source,
        marker_count=marker_count,
        corner_px=corner_px,
        corner_id=corner_id,
        extrinsic=solution.extrinsic,
        reprojection_rms_px=solution.reprojection_rms_px,
    )


def write_board_pose_outputs(
    out_dir: Path, results: list[FrameResult], run_info: dict[str, Any]
) -> None:
    """Write board_poses.npz and board_poses.json. Refuses to overwrite, like R-01's runner."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already has files in it. Pick a new --out-dir or delete it deliberately; "
            "overwriting silently would let outputs from two runs get mixed."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_count = len(results)
    # NaN, not zeros or identity, for frames without a pose: an identity extrinsic is a valid
    # camera and would pass straight through R-04 as if it were measured.
    extrinsic = np.full((frame_count, 3, 4), np.nan, dtype=np.float64)
    intrinsic = np.full((frame_count, 3, 3), np.nan, dtype=np.float64)
    image_size_hw = np.zeros((frame_count, 2), dtype=np.int32)
    marker_count = np.zeros(frame_count, dtype=np.int32)
    corner_count = np.zeros(frame_count, dtype=np.int32)
    reprojection_rms_px = np.full(frame_count, np.nan, dtype=np.float64)
    pose_found = np.zeros(frame_count, dtype=bool)
    corner_frame_index: list[np.ndarray] = []
    corner_id: list[np.ndarray] = []
    corner_px: list[np.ndarray] = []

    for index, result in enumerate(results):
        if result.extrinsic is not None:
            extrinsic[index] = result.extrinsic
            reprojection_rms_px[index] = result.reprojection_rms_px
            pose_found[index] = True
        if result.intrinsic is not None:
            intrinsic[index] = result.intrinsic
        if result.image_size_hw is not None:
            image_size_hw[index] = result.image_size_hw
        marker_count[index] = result.marker_count
        corner_count[index] = len(result.corner_id)
        corner_frame_index.append(np.full(len(result.corner_id), index, dtype=np.int32))
        corner_id.append(result.corner_id)
        corner_px.append(result.corner_px)

    np.savez_compressed(
        out_dir / "board_poses.npz",
        frame_names=np.array([result.frame_name for result in results]),
        pose_found=pose_found,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        image_size_hw=image_size_hw,
        marker_count=marker_count,
        corner_count=corner_count,
        reprojection_rms_px=reprojection_rms_px,
        corner_frame_index=np.concatenate(corner_frame_index),
        corner_id=np.concatenate(corner_id),
        corner_px=np.concatenate(corner_px),
    )
    with open(out_dir / "board_poses.json", "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2, sort_keys=True)
        handle.write("\n")


def estimate_board_poses(
    capture_dir: Path,
    out_dir: Path,
    intrinsic_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run R-02 on one capture folder, write the outputs, and return the board_poses.json contents."""
    total_started = time.perf_counter()
    frame_paths = find_capture_frames(capture_dir)
    board = make_board()
    detector = make_detector(board)

    results: list[FrameResult] = []
    frame_summaries: list[dict[str, Any]] = []
    for frame_path in frame_paths:
        started = time.perf_counter()
        result = estimate_frame_pose(frame_path, board, detector, intrinsic_override)
        elapsed_s = time.perf_counter() - started
        results.append(result)
        frame_summaries.append(
            {
                "frame_name": result.frame_name,
                "status": result.status,
                "message": result.message,
                "marker_count": result.marker_count,
                "corner_count": len(result.corner_id),
                "reprojection_rms_px": result.reprojection_rms_px,
                "exif_orientation": result.exif_orientation,
                "intrinsics_source": result.intrinsics_source,
                "elapsed_s": elapsed_s,
            }
        )

    posed_count = 0
    failure_counts: dict[str, int] = {}
    for result in results:
        if result.status == STATUS_OK:
            posed_count += 1
        else:
            failure_counts[result.status] = failure_counts.get(result.status, 0) + 1

    run_info: dict[str, Any] = {
        "story": "R-02",
        "capture_dir": str(capture_dir),
        "board": {
            "decision": "D-034",
            "dictionary": "DICT_5X5_250",
            "squares_xy": [BOARD_SQUARES_X, BOARD_SQUARES_Y],
            "square_mm": BOARD_SQUARE_MM,
            "marker_mm": BOARD_MARKER_MM,
            "legacy_pattern": False,
        },
        "units": "mm",
        "extrinsic_convention": "camera-from-board, x_cam = R @ x_board + t (OpenCV)",
        "pixel_frame": "stored sensor pixels; EXIF orientation ignored to match R-01",
        "min_corners_for_pose": MIN_CORNERS_FOR_POSE,
        "opencv_version": cv2.__version__,
        "frame_count": len(results),
        "posed_count": posed_count,
        "failure_counts": failure_counts,
        "frames": frame_summaries,
    }
    run_info["total_s"] = time.perf_counter() - total_started
    write_board_pose_outputs(out_dir, results, run_info)
    return run_info


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect the ChArUco board and solve per-frame camera poses (R-02)."
    )
    parser.add_argument(
        "capture_dir", type=Path, help="folder of JPEG frames, e.g. .../s-04"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output folder (default: fixtures/data/board_poses/<capture folder name>)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    out_dir = args.out_dir or Path("fixtures/data/board_poses") / args.capture_dir.name
    try:
        run_info = estimate_board_poses(args.capture_dir, out_dir)
    except (FileNotFoundError, FileExistsError) as exc:
        print(f"R-02 BOARD POSE: FAIL - {exc}", file=sys.stderr)
        return 1

    print(
        f"R-02 BOARD POSE: {run_info['posed_count']}/{run_info['frame_count']} frames posed "
        f"({run_info['total_s']:.1f} s)"
    )
    for frame in run_info["frames"]:
        if frame["status"] != STATUS_OK:
            print(f"  {frame['frame_name']}: {frame['status']} - {frame['message']}")
    print(f"Outputs written to {out_dir}")
    # A capture where no frame could be posed is useless to R-04, so that's a failing exit code;
    # individual failed frames are reported above but are expected in a real capture.
    return 0 if run_info["posed_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
