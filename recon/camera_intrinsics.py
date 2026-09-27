"""R-03: session camera intrinsics from R-02 ChArUco observations.

R-02 already stores every detected board corner, so R-03 reuses those observations instead of
detecting the board again. The primary result is one session-level K and distortion vector in
the original stored-sensor pixel frame. If calibration cannot be solved, the fallback is VGGT's
R-01 intrinsics, explicitly rescaled from VGGT's processed-image pixels to full-resolution
pixels. Reprojection RMS is a pixel self-consistency measure, not a metric-accuracy claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from recon.board_pose import MIN_CORNERS_FOR_POSE, make_board

MIN_CALIBRATION_VIEWS = 3
SOURCE_CHARUCO = "charuco_calibration"
SOURCE_VGGT_FALLBACK = "vggt_fallback"


class CalibrationFailure(RuntimeError):
    """Board observations could not support a valid session calibration."""


@dataclass(frozen=True)
class BoardView:
    frame_name: str
    object_points: np.ndarray
    image_points: np.ndarray


@dataclass(frozen=True)
class BoardSession:
    frame_names: list[str]
    image_size_hw: tuple[int, int]
    views: list[BoardView]
    initial_intrinsic: np.ndarray | None


@dataclass(frozen=True)
class IntrinsicsResult:
    intrinsic: np.ndarray
    distortion: np.ndarray
    source: str
    reprojection_rms_px: float | None
    per_view_frame_names: list[str]
    per_view_rms_px: np.ndarray
    fallback_reason: str | None = None


def _require_keys(data: Any, path: Path, keys: tuple[str, ...]) -> None:
    missing: list[str] = []
    for key in keys:
        if key not in data:
            missing.append(key)
    if missing:
        raise ValueError(
            f"{path} is missing required arrays: {', '.join(missing)}"
        )


def _median_initial_intrinsic(intrinsics: np.ndarray) -> np.ndarray | None:
    valid: list[np.ndarray] = []
    for intrinsic in intrinsics:
        if intrinsic.shape != (3, 3):
            continue
        if not np.all(np.isfinite(intrinsic)):
            continue
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            continue
        valid.append(intrinsic.astype(np.float64))
    if not valid:
        return None
    return np.median(np.stack(valid, axis=0), axis=0)


def load_board_session(board_pose_dir: Path) -> BoardSession:
    """Load R-02 board_poses.npz and group its saved corners by frame."""

    board_pose_dir = Path(board_pose_dir)
    path = board_pose_dir / "board_poses.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"R-02 output not found: {path}. Run python -m recon.board_pose first."
        )

    with np.load(path) as data:
        _require_keys(
            data,
            path,
            (
                "frame_names",
                "image_size_hw",
                "intrinsic",
                "corner_frame_index",
                "corner_id",
                "corner_px",
            ),
        )
        frame_names = [str(name) for name in data["frame_names"]]
        image_sizes = np.asarray(data["image_size_hw"], dtype=np.int32)
        r02_intrinsics = np.asarray(data["intrinsic"], dtype=np.float64)
        corner_frame_index = np.asarray(data["corner_frame_index"], dtype=np.int32)
        corner_ids = np.asarray(data["corner_id"], dtype=np.int32)
        corner_px = np.asarray(data["corner_px"], dtype=np.float64)

    frame_count = len(frame_names)
    if image_sizes.shape != (frame_count, 2):
        raise ValueError(
            f"{path}: image_size_hw must be ({frame_count},2), got {image_sizes.shape}"
        )
    if r02_intrinsics.shape != (frame_count, 3, 3):
        raise ValueError(
            f"{path}: intrinsic must be ({frame_count},3,3), got {r02_intrinsics.shape}"
        )
    if not (
        corner_frame_index.ndim == 1
        and corner_ids.ndim == 1
        and corner_px.ndim == 2
        and corner_px.shape[1] == 2
        and len(corner_frame_index) == len(corner_ids) == len(corner_px)
    ):
        raise ValueError(f"{path}: corner arrays have inconsistent shapes")

    sizes: list[tuple[int, int]] = []
    for height, width in image_sizes:
        if int(height) > 0 and int(width) > 0:
            sizes.append((int(height), int(width)))
    if not sizes:
        raise CalibrationFailure("R-02 did not record a usable image size.")
    unique_sizes = set(sizes)
    if len(unique_sizes) != 1:
        raise CalibrationFailure(
            "R-03 requires one stored-sensor image size per session; "
            f"R-02 recorded {sorted(unique_sizes)}."
        )

    board_corners = np.asarray(
        make_board().getChessboardCorners(), dtype=np.float64
    )
    views: list[BoardView] = []
    for frame_index, frame_name in enumerate(frame_names):
        selection = corner_frame_index == frame_index
        ids = corner_ids[selection]
        pixels = corner_px[selection]
        if len(ids) < MIN_CORNERS_FOR_POSE:
            continue
        if np.any(ids < 0) or np.any(ids >= len(board_corners)):
            raise ValueError(
                f"{path}: frame {frame_name} has an invalid ChArUco corner id"
            )
        object_points = board_corners[ids]
        centred = object_points[:, :2] - np.mean(
            object_points[:, :2], axis=0
        )
        if np.linalg.matrix_rank(centred) < 2:
            continue
        views.append(
            BoardView(
                frame_name=frame_name,
                object_points=object_points.astype(np.float32),
                image_points=pixels.astype(np.float32),
            )
        )

    return BoardSession(
        frame_names=frame_names,
        image_size_hw=sizes[0],
        views=views,
        initial_intrinsic=_median_initial_intrinsic(r02_intrinsics),
    )


def _valid_intrinsic(intrinsic: np.ndarray, context: str) -> np.ndarray:
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise CalibrationFailure(
            f"{context} returned K with shape {intrinsic.shape}, expected (3,3)."
        )
    if not np.all(np.isfinite(intrinsic)):
        raise CalibrationFailure(f"{context} returned non-finite intrinsics.")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise CalibrationFailure(f"{context} returned non-positive focal length.")
    return intrinsic


def _per_view_reprojection(
    views: list[BoardView],
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    rvecs: tuple[np.ndarray, ...] | list[np.ndarray],
    tvecs: tuple[np.ndarray, ...] | list[np.ndarray],
) -> np.ndarray:
    errors: list[float] = []
    for view, rvec, tvec in zip(views, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(
            view.object_points, rvec, tvec, intrinsic, distortion
        )
        residual = projected.reshape(-1, 2) - view.image_points
        squared = np.sum(residual * residual, axis=1)
        errors.append(float(np.sqrt(np.mean(squared))))
    return np.asarray(errors, dtype=np.float64)


def calibrate_from_board(session: BoardSession) -> IntrinsicsResult:
    """Calibrate one K/distortion model from all usable views in the session."""

    if len(session.views) < MIN_CALIBRATION_VIEWS:
        raise CalibrationFailure(
            f"Need at least {MIN_CALIBRATION_VIEWS} usable board views; "
            f"found {len(session.views)}."
        )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for view in session.views:
        object_points.append(view.object_points)
        image_points.append(view.image_points)

    height, width = session.image_size_hw
    flags = 0
    initial = None
    if session.initial_intrinsic is not None:
        initial = session.initial_intrinsic.copy()
        flags = cv2.CALIB_USE_INTRINSIC_GUESS

    try:
        rms, intrinsic, distortion, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            (width, height),
            initial,
            None,
            flags=flags,
        )
    except cv2.error as exc:
        raise CalibrationFailure(
            f"OpenCV board calibration failed: {exc}"
        ) from exc

    intrinsic = _valid_intrinsic(intrinsic, "OpenCV calibration")
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(distortion)):
        raise CalibrationFailure(
            "OpenCV calibration returned non-finite distortion."
        )
    if not math.isfinite(float(rms)):
        raise CalibrationFailure(
            "OpenCV calibration returned non-finite reprojection RMS."
        )

    return IntrinsicsResult(
        intrinsic=intrinsic,
        distortion=distortion,
        source=SOURCE_CHARUCO,
        reprojection_rms_px=float(rms),
        per_view_frame_names=[view.frame_name for view in session.views],
        per_view_rms_px=_per_view_reprojection(
            session.views, intrinsic, distortion, rvecs, tvecs
        ),
    )


def rescale_vggt_intrinsic(
    intrinsic: np.ndarray,
    processed_size_hw: tuple[int, int],
    full_size_hw: tuple[int, int],
) -> np.ndarray:
    """Convert VGGT K from resized-image pixels to full-resolution sensor pixels."""

    processed_height, processed_width = processed_size_hw
    full_height, full_width = full_size_hw
    if min(processed_height, processed_width, full_height, full_width) <= 0:
        raise ValueError("Image dimensions must be positive.")

    scaled = _valid_intrinsic(intrinsic, "VGGT").copy()
    scale_x = full_width / processed_width
    scale_y = full_height / processed_height
    scaled[0, 0] *= scale_x
    scaled[0, 1] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 0] *= scale_y
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def load_vggt_fallback(model_run_dir: Path, session: BoardSession) -> np.ndarray:
    """Load R-01 predicted K values, rescale them, and take a session median."""

    model_run_dir = Path(model_run_dir)
    poses_path = model_run_dir / "poses.npz"
    run_path = model_run_dir / "run.json"
    if not poses_path.is_file() or not run_path.is_file():
        raise FileNotFoundError(
            "VGGT fallback requires poses.npz and run.json from R-01 in "
            f"{model_run_dir}."
        )

    with open(run_path, "r", encoding="utf-8") as handle:
        run_info = json.load(handle)
    processed = run_info.get("processed_image_size_hw")
    if not isinstance(processed, list) or len(processed) != 2:
        raise ValueError(
            f"{run_path}: processed_image_size_hw is missing or invalid"
        )
    processed_size = (int(processed[0]), int(processed[1]))

    with np.load(poses_path) as data:
        _require_keys(data, poses_path, ("frame_names", "intrinsic"))
        model_names = [str(name) for name in data["frame_names"]]
        model_intrinsics = np.asarray(data["intrinsic"], dtype=np.float64)

    if model_intrinsics.shape != (len(model_names), 3, 3):
        raise ValueError(
            f"{poses_path}: intrinsic shape is {model_intrinsics.shape}"
        )

    wanted = set(session.frame_names)
    predictions: list[np.ndarray] = []
    for frame_name, intrinsic in zip(
        model_names, model_intrinsics, strict=True
    ):
        if frame_name in wanted:
            predictions.append(
                rescale_vggt_intrinsic(
                    intrinsic, processed_size, session.image_size_hw
                )
            )
    if not predictions:
        raise CalibrationFailure(
            "VGGT fallback has no frame names in common with R-02."
        )

    stacked = np.stack(predictions, axis=0)
    fallback = np.eye(3, dtype=np.float64)
    fallback[0, 0] = float(np.median(stacked[:, 0, 0]))
    fallback[1, 1] = float(np.median(stacked[:, 1, 1]))
    fallback[0, 2] = float(np.median(stacked[:, 0, 2]))
    fallback[1, 2] = float(np.median(stacked[:, 1, 2]))
    return _valid_intrinsic(fallback, "VGGT fallback")


def reprojection_with_fixed_intrinsic(
    session: BoardSession,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> tuple[float | None, list[str], np.ndarray]:
    """Report board reprojection error for fallback K after solving only pose."""

    total_squared = 0.0
    point_count = 0
    frame_names: list[str] = []
    errors: list[float] = []

    for view in session.views:
        try:
            solved, rvec, tvec = cv2.solvePnP(
                view.object_points,
                view.image_points,
                intrinsic,
                distortion,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            continue
        if not solved:
            continue
        projected, _ = cv2.projectPoints(
            view.object_points, rvec, tvec, intrinsic, distortion
        )
        residual = projected.reshape(-1, 2) - view.image_points
        squared = np.sum(residual * residual, axis=1)
        total_squared += float(np.sum(squared))
        point_count += len(squared)
        frame_names.append(view.frame_name)
        errors.append(float(np.sqrt(np.mean(squared))))

    if point_count == 0:
        return None, frame_names, np.asarray(errors, dtype=np.float64)
    return (
        float(math.sqrt(total_squared / point_count)),
        frame_names,
        np.asarray(errors, dtype=np.float64),
    )


def estimate_session_intrinsics(
    board_pose_dir: Path, model_run_dir: Path
) -> tuple[BoardSession, IntrinsicsResult]:
    session = load_board_session(board_pose_dir)
    try:
        return session, calibrate_from_board(session)
    except CalibrationFailure as failure:
        intrinsic = load_vggt_fallback(model_run_dir, session)
        distortion = np.zeros(5, dtype=np.float64)
        rms, frame_names, per_view = reprojection_with_fixed_intrinsic(
            session, intrinsic, distortion
        )
        return session, IntrinsicsResult(
            intrinsic=intrinsic,
            distortion=distortion,
            source=SOURCE_VGGT_FALLBACK,
            reprojection_rms_px=rms,
            per_view_frame_names=frame_names,
            per_view_rms_px=per_view,
            fallback_reason=str(failure),
        )


def write_outputs(
    out_dir: Path,
    session: BoardSession,
    result: IntrinsicsResult,
    board_pose_dir: Path,
    model_run_dir: Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already has files. Use a new output folder or delete it deliberately."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    rms_value = (
        np.nan
        if result.reprojection_rms_px is None
        else result.reprojection_rms_px
    )
    np.savez_compressed(
        out_dir / "intrinsics.npz",
        intrinsic=result.intrinsic,
        distortion=result.distortion,
        image_size_hw=np.asarray(session.image_size_hw, dtype=np.int32),
        reprojection_rms_px=np.asarray(rms_value, dtype=np.float64),
        per_view_frame_names=np.asarray(result.per_view_frame_names),
        per_view_rms_px=result.per_view_rms_px,
        source=np.asarray(result.source),
    )

    per_view: dict[str, float] = {}
    for name, error in zip(
        result.per_view_frame_names, result.per_view_rms_px, strict=True
    ):
        per_view[name] = float(error)

    run_info: dict[str, Any] = {
        "story": "R-03",
        "source": result.source,
        "board_pose_dir": str(board_pose_dir),
        "model_run_dir": str(model_run_dir),
        "pixel_frame": (
            "stored sensor pixels; EXIF orientation ignored to match R-01/R-02"
        ),
        "image_size_hw": list(session.image_size_hw),
        "intrinsic": result.intrinsic.tolist(),
        "distortion": result.distortion.tolist(),
        "usable_board_views": len(session.views),
        "reprojection_rms_px": result.reprojection_rms_px,
        "reprojection_note": (
            "pixel self-consistency only; not a metric-accuracy or tolerance claim"
        ),
        "per_view_reprojection_rms_px": per_view,
        "fallback_reason": result.fallback_reason,
        "board_geometry_source": "recon.board_pose.make_board (D-034)",
        "opencv_version": cv2.__version__,
    }
    with open(out_dir / "intrinsics.json", "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return run_info


def run_intrinsics(
    board_pose_dir: Path, model_run_dir: Path, out_dir: Path
) -> dict[str, Any]:
    session, result = estimate_session_intrinsics(
        board_pose_dir, model_run_dir
    )
    return write_outputs(
        out_dir, session, result, board_pose_dir, model_run_dir
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from R-02 board corners (R-03)."
    )
    parser.add_argument(
        "board_pose_dir",
        type=Path,
        help="R-02 output folder, for example fixtures/data/board_poses/s-04",
    )
    parser.add_argument(
        "--model-run-dir",
        type=Path,
        default=None,
        help="R-01 fallback folder; default fixtures/data/recon_runs/<sample>",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default fixtures/data/camera_intrinsics/<sample>",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    sample = args.board_pose_dir.name
    model_run_dir = (
        args.model_run_dir or Path("fixtures/data/recon_runs") / sample
    )
    out_dir = (
        args.out_dir or Path("fixtures/data/camera_intrinsics") / sample
    )
    try:
        info = run_intrinsics(
            args.board_pose_dir, model_run_dir, out_dir
        )
    except (
        CalibrationFailure,
        FileNotFoundError,
        FileExistsError,
        ValueError,
    ) as exc:
        print(f"R-03 INTRINSICS: FAIL - {exc}", file=sys.stderr)
        return 1

    rms = info["reprojection_rms_px"]
    rms_text = "unavailable" if rms is None else f"{rms:.3f} px"
    print(
        f"R-03 INTRINSICS: PASS - source={info['source']}, "
        f"reprojection RMS={rms_text}"
    )
    if info["fallback_reason"]:
        print(f"  fallback reason: {info['fallback_reason']}")
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
