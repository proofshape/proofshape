"""R-06: keep only the part geometry from R-04's board-frame point cloud.

R-04 aligns R-01's arbitrary-scale reconstruction to the board frame, in millimetres, but its
point cloud still has the board pattern, the table around it, and anything else a frame happened
to see (a hand, background clutter). This module drops everything that isn't the part, using two
independent checks against the board frame:

  - Height: keep points that sit above the board plane by at least a configurable threshold.
  - Footprint: keep points inside the printed ChArUco pattern's rectangle in x/y.

Footprint is the *pattern* rectangle, not the physical A4/Letter sheet edge: `docs/glossary.md`
defines "reference sheet" as "the ChArUco board" for tier 1, and no code anywhere detects the
actual paper outline (docs/manual.md flags that as a manual, unbuilt check) -- the pattern is the
only footprint this project has an exact, already-defined geometry for.

Height uses `board_pose.py`'s z convention, not the more intuitive-looking `z >= threshold`:
origin at the pattern's outer corner, z = 0 on the paper, z increasing *into* the table, so "a
camera above the board therefore has a negative z camera centre" (board_pose.py's own docstring).
A point above the board -- like the top of a part sitting on it -- therefore has *negative* z, and
height above the board is `-z`. Getting this backwards would silently keep the table and drop the
part, so it is pinned by an explicit unit test, not left to a comment alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from recon.board_pose import BOARD_SQUARE_MM, BOARD_SQUARES_X, BOARD_SQUARES_Y
from recon.metric_alignment import apply_similarity_transform
from recon.vggt_runner import read_ply, write_ply

DEFAULT_HEIGHT_THRESHOLD_MM = 2.0

# The printed pattern's rectangle in the board frame: x in [0, BOARD_SQUARES_X*BOARD_SQUARE_MM],
# y in [0, BOARD_SQUARES_Y*BOARD_SQUARE_MM]. Derived from board_pose.py's constants, not
# re-declared, so the two modules can't drift apart (currently 160.0 x 120.0 mm, D-034's board).
DEFAULT_FOOTPRINT_MM = (
    BOARD_SQUARES_X * BOARD_SQUARE_MM,
    BOARD_SQUARES_Y * BOARD_SQUARE_MM,
)


def segment_part(
    points_board_frame_mm: np.ndarray,
    height_threshold_mm: float = DEFAULT_HEIGHT_THRESHOLD_MM,
    footprint_mm: tuple[float, float] = DEFAULT_FOOTPRINT_MM,
) -> np.ndarray:
    """Return a boolean keep-mask: True where a point is part geometry, not board/background.

    Args:
        points_board_frame_mm: (N, 3) points already in the board frame, in millimetres (R-04's
            `AlignmentResult.transform_points` / `apply_similarity_transform` output).
        height_threshold_mm: minimum height above the board plane to keep a point. Configurable
            per the acceptance criteria; default 2 mm.
        footprint_mm: (width_mm, height_mm) of the keep rectangle, anchored at the board origin
            (x in [0, width_mm], y in [0, height_mm]). Defaults to the printed pattern's own size.

    Raises:
        ValueError: on a malformed shape, non-finite input, or a non-positive threshold/footprint.
    """
    points_board_frame_mm = np.asarray(points_board_frame_mm, dtype=np.float64)
    if points_board_frame_mm.ndim != 2 or points_board_frame_mm.shape[1] != 3:
        raise ValueError(
            f"points_board_frame_mm must have shape (N,3), got {points_board_frame_mm.shape}"
        )
    if not np.isfinite(points_board_frame_mm).all():
        raise ValueError("points_board_frame_mm must be finite.")
    if not (np.isfinite(height_threshold_mm) and height_threshold_mm > 0):
        raise ValueError("height_threshold_mm must be a positive, finite number.")
    footprint_width_mm, footprint_height_mm = footprint_mm
    if not (
        np.isfinite(footprint_width_mm)
        and footprint_width_mm > 0
        and np.isfinite(footprint_height_mm)
        and footprint_height_mm > 0
    ):
        raise ValueError("footprint_mm must be a pair of positive, finite numbers.")

    x_mm = points_board_frame_mm[:, 0]
    y_mm = points_board_frame_mm[:, 1]
    z_mm = points_board_frame_mm[:, 2]

    height_above_board_mm = -z_mm
    keep_height = height_above_board_mm >= height_threshold_mm
    keep_footprint = (
        (x_mm >= 0.0)
        & (x_mm <= footprint_width_mm)
        & (y_mm >= 0.0)
        & (y_mm <= footprint_height_mm)
    )
    return keep_height & keep_footprint


def load_alignment(alignment_json_path: Path) -> dict[str, Any]:
    """Read R-04's metric_alignment.json into the fields apply_similarity_transform needs."""
    alignment_json_path = Path(alignment_json_path)
    data = json.loads(alignment_json_path.read_text(encoding="utf-8"))
    required = ("scale", "rotation", "translation_mm")
    for key in required:
        if key not in data:
            raise ValueError(f"{alignment_json_path} is missing '{key}'")
    return {
        "scale": float(data["scale"]),
        "rotation": np.asarray(data["rotation"], dtype=np.float64),
        "translation": np.asarray(data["translation_mm"], dtype=np.float64),
    }


def write_segmentation_report(
    path: Path,
    reconstruction_run_dir: Path,
    alignment_json_path: Path,
    height_threshold_mm: float,
    footprint_mm: tuple[float, float],
    total_points: int,
    kept_points: int,
) -> dict[str, Any]:
    """Write a JSON report describing what was kept and why, mirroring R-04/R-05's convention."""
    report = {
        "story": "R-06",
        "source_points_ply": str(Path(reconstruction_run_dir) / "points.ply"),
        "source_alignment_json": str(alignment_json_path),
        "units": "mm",
        "frame": "R-02 ChArUco board frame",
        "height_threshold_mm": height_threshold_mm,
        "footprint_mm": {"width": footprint_mm[0], "height": footprint_mm[1]},
        "total_point_count": total_points,
        "kept_point_count": kept_points,
        "removed_point_count": total_points - kept_points,
    }
    path = Path(path)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def segment_run(
    reconstruction_run_dir: Path,
    alignment_json_path: Path,
    out_dir: Path,
    height_threshold_mm: float = DEFAULT_HEIGHT_THRESHOLD_MM,
    footprint_mm: tuple[float, float] = DEFAULT_FOOTPRINT_MM,
) -> dict[str, Any]:
    """Load an R-01 point cloud and an R-04 alignment, segment, and write the outputs.

    Writes `out_dir/segmented_points.ply` (kept points, board-frame mm) and
    `out_dir/part_segmentation.json` (the report). Refuses to overwrite an existing non-empty
    `out_dir`, matching R-01/R-02's guard against mixing two runs' outputs.
    """
    reconstruction_run_dir = Path(reconstruction_run_dir)
    alignment_json_path = Path(alignment_json_path)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already has files in it. Pick a new --out-dir or delete it deliberately; "
            "overwriting silently would let outputs from two runs get mixed."
        )

    points, colors = read_ply(reconstruction_run_dir / "points.ply")
    alignment = load_alignment(alignment_json_path)
    points_board_frame_mm = apply_similarity_transform(
        points, alignment["scale"], alignment["rotation"], alignment["translation"]
    )
    keep = segment_part(points_board_frame_mm, height_threshold_mm, footprint_mm)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_ply(
        out_dir / "segmented_points.ply",
        points_board_frame_mm[keep].astype(np.float32),
        colors[keep],
    )
    return write_segmentation_report(
        out_dir / "part_segmentation.json",
        reconstruction_run_dir,
        alignment_json_path,
        height_threshold_mm,
        footprint_mm,
        total_points=len(points),
        kept_points=int(np.count_nonzero(keep)),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment the part from the board in an R-04-aligned point cloud (R-06)."
    )
    parser.add_argument("reconstruction_run_dir", type=Path)
    parser.add_argument(
        "alignment_json",
        type=Path,
        nargs="?",
        default=None,
        help="defaults to <reconstruction_run_dir>/metric_alignment.json (R-04's default output)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--height-threshold-mm", type=float, default=DEFAULT_HEIGHT_THRESHOLD_MM
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    alignment_json = args.alignment_json or (
        args.reconstruction_run_dir / "metric_alignment.json"
    )
    out_dir = args.out_dir or args.reconstruction_run_dir / "segmentation"
    try:
        report = segment_run(
            args.reconstruction_run_dir,
            alignment_json,
            out_dir,
            height_threshold_mm=args.height_threshold_mm,
        )
    except FileExistsError as exc:
        print(f"R-06 SEGMENTATION: FAIL - {exc}", file=sys.stderr)
        return 1

    total = report["total_point_count"]
    kept = report["kept_point_count"]
    fraction = (kept / total * 100.0) if total else 0.0
    print(f"R-06 SEGMENTATION: PASS - kept {kept}/{total} points ({fraction:.1f}%)")
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
