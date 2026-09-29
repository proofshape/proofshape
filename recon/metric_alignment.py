"""R-04: align an arbitrary-scale reconstruction to the ChArUco board frame.

R-01 and R-02 both store camera-from-world extrinsics. R-04 converts those
extrinsics to camera centres, pairs them by frame name, and solves the
similarity transform from the R-01 frame to the metric board frame.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MIN_ALIGNMENT_FRAMES = 3


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
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N,3), got {points.shape}")
        return self.scale * (points @ self.rotation.T) + self.translation


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
) -> AlignmentResult:
    """Load R-01/R-02 outputs and align their paired camera centres."""
    reconstruction_run_dir = Path(reconstruction_run_dir)
    board_pose_dir = Path(board_pose_dir)
    with np.load(reconstruction_run_dir / "poses.npz") as reconstruction:
        reconstruction_names = [str(name) for name in reconstruction["frame_names"]]
        reconstruction_extrinsic = reconstruction["extrinsic"]
    with np.load(board_pose_dir / "board_poses.npz") as board:
        required = ("frame_names", "extrinsic")
        for key in required:
            if key not in board:
                raise ValueError(
                    f"{board_pose_dir / 'board_poses.npz'} is missing {key}"
                )
        board_names = [str(name) for name in board["frame_names"]]
        board_extrinsic = board["extrinsic"]
        found = board.get("pose_found", None)
    return align_camera_centres(
        reconstruction_names,
        reconstruction_extrinsic,
        board_names,
        board_extrinsic,
        found,
    )


def write_alignment(path: Path, result: AlignmentResult) -> dict[str, Any]:
    """Write a JSON report with the transform and every per-frame residual."""
    report = {
        "story": "R-04",
        "source_frame": "R-01 reconstruction frame (arbitrary scale)",
        "target_frame": "R-02 ChArUco board frame (millimetres)",
        "scale": result.scale,
        "rotation": result.rotation.tolist(),
        "translation_mm": result.translation.tolist(),
        "paired_frame_count": len(result.frame_names),
        "rms_residual_mm": result.rms_residual_mm,
        "per_frame_residual_mm": dict(
            zip(result.frame_names, result.residuals_mm.tolist(), strict=True)
        ),
        "note": "Residuals measure camera-centre alignment self-consistency, not object accuracy.",
    }
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
        f"R-04 ALIGNMENT: PASS - {len(result.frame_names)} frames, "
        f"RMS residual={result.rms_residual_mm:.3f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
