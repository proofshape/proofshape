"""R-20: measure, on one real R-01+R-04 run, how a confidence threshold trades off stray
points against part coverage -- the evidence D-045's chosen default is picked from.

R-17 found that 11.5-29.3% of segmented points on the golden set are "stray": segmented points
outside the largest connected cluster of occupied voxels, more than 2 mm above the fitted board.
R-17's own measurement was a one-off script. This module replaces it with committed, tested code
so the sweep behind D-045 (and any later re-measurement, e.g. after R-22 lands) can be rerun
rather than re-derived by hand.

The sweep, per threshold:
  1. Unproject R-01's saved depth/poses to world points, then into the board frame with R-04's
     alignment (the same two steps R-07's `fuse_run` and R-09's `provenance_run` both do).
  2. Drop pixels `recon.vggt_runner.confidence_keep_mask` would drop at that threshold -- the
     same predicate R-01's point cloud and R-07's fusion use, so this sweep measures exactly the
     choice those two stages make.
  3. Run R-06/R-21's `segment_part` (board/table exclusion, relative to the real fitted board
     surface, not z=0) on what survives.
  4. Call `stray_point_fraction` on the segmented points.

"Part coverage" compares each threshold's largest cluster against threshold 0's (no confidence
filtering) by voxel membership, not point count -- point count is the wrong measure here because
many overlapping frames over-sample the same surface, so losing half the points on a wall that is
still fully covered should not look like losing half the part.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from recon.metric_alignment import apply_similarity_transform
from recon.part_segmentation import (
    DEFAULT_FOOTPRINT_MM,
    DEFAULT_MAX_BOARD_TILT_DEG,
    measure_board_surface,
    segment_part,
)
from recon.vggt_runner import confidence_keep_mask, unproject_depth_to_world

DEFAULT_VOXEL_SIZE_MM = 2.0
# R-17's definition: a voxel needs at least 3 points to count as "occupied" (not just a single
# ray's worth of noise landing there by chance).
DEFAULT_MIN_POINTS_PER_VOXEL = 3
DEFAULT_THRESHOLDS = (0.0, 1.0001, 1.001, 1.01, 1.05, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0)

# A guard against a dense voxel grid blowing up memory if an extreme outlier widens the bounding
# box -- not expected on segmented, footprint-bounded points, but cheap to check.
_MAX_VOXEL_GRID_CELLS = 50_000_000


def stray_point_fraction(
    points_mm: np.ndarray,
    voxel_size_mm: float = DEFAULT_VOXEL_SIZE_MM,
    min_points_per_voxel: int = DEFAULT_MIN_POINTS_PER_VOXEL,
) -> dict[str, Any]:
    """R-17's stray-point measure on a segmented point cloud.

    Bins points into `voxel_size_mm` voxels; a voxel is "occupied" once it holds at least
    `min_points_per_voxel` points; the largest 26-connected cluster of occupied voxels is taken
    to be the part, and every point outside it is a stray point.

    Returns a dict with:
        stray_fraction: float in [0, 1] (nan if `points_mm` is empty -- nothing to measure).
        part_voxel_keys: frozenset of (ix, iy, iz) integer voxel indices in the largest cluster,
            so a caller can compare cluster membership across thresholds (see
            `part_voxel_coverage`) without re-deriving it.
        part_point_count, stray_point_count, total_point_count: int.
        x_span_mm, y_span_mm, top_height_mm: the part cluster's bounding extent (nan if the
            cluster is empty) -- `top_height_mm` is `-z` at the highest point, matching
            `recon.part_segmentation`'s "height above the board is -z" convention.
    """
    points_mm = np.asarray(points_mm, dtype=np.float64)
    if points_mm.ndim != 2 or points_mm.shape[1] != 3:
        raise ValueError(f"points_mm must have shape (N,3), got {points_mm.shape}")
    if not np.isfinite(voxel_size_mm) or voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be a positive, finite number.")
    if min_points_per_voxel < 1:
        raise ValueError("min_points_per_voxel must be at least 1.")

    empty_extents = {
        "x_span_mm": float("nan"),
        "y_span_mm": float("nan"),
        "top_height_mm": float("nan"),
    }

    if len(points_mm) == 0:
        return {
            "stray_fraction": float("nan"),
            "part_voxel_keys": frozenset(),
            "part_point_count": 0,
            "stray_point_count": 0,
            "total_point_count": 0,
            **empty_extents,
        }

    voxel_indices = np.floor(points_mm / voxel_size_mm).astype(np.int64)
    # Work in a dense array sized only to the voxels actually touched, not the whole scene.
    mins = voxel_indices.min(axis=0)
    shape = tuple((voxel_indices.max(axis=0) - mins) + 1)
    cell_count = int(np.prod(shape))
    if cell_count > _MAX_VOXEL_GRID_CELLS:
        raise ValueError(
            f"Voxel grid would need {cell_count} cells for this point spread; check points_mm "
            "for an extreme outlier."
        )
    local_indices = voxel_indices - mins
    counts = np.zeros(shape, dtype=np.int32)
    np.add.at(counts, tuple(local_indices.T), 1)
    occupied = counts >= min_points_per_voxel

    labeled, _ = ndimage.label(occupied, structure=np.ones((3, 3, 3)))
    if labeled.max() == 0:
        # No voxel reached min_points_per_voxel -- by this definition every point is stray.
        return {
            "stray_fraction": 1.0,
            "part_voxel_keys": frozenset(),
            "part_point_count": 0,
            "stray_point_count": len(points_mm),
            "total_point_count": len(points_mm),
            **empty_extents,
        }

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # background label, never the largest cluster
    largest_label = int(np.argmax(component_sizes))

    point_labels = labeled[tuple(local_indices.T)]
    in_part_cluster = point_labels == largest_label
    part_point_count = int(in_part_cluster.sum())
    stray_point_count = int(len(points_mm) - part_point_count)

    part_voxel_local = np.argwhere(labeled == largest_label)
    part_voxel_global = part_voxel_local + mins
    part_voxel_keys = frozenset(tuple(row) for row in part_voxel_global.tolist())

    part_points = points_mm[in_part_cluster]

    return {
        "stray_fraction": stray_point_count / len(points_mm),
        "part_voxel_keys": part_voxel_keys,
        "part_point_count": part_point_count,
        "stray_point_count": stray_point_count,
        "total_point_count": len(points_mm),
        "x_span_mm": float(part_points[:, 0].max() - part_points[:, 0].min()),
        "y_span_mm": float(part_points[:, 1].max() - part_points[:, 1].min()),
        "top_height_mm": float((-part_points[:, 2]).max()),
    }


def part_voxel_coverage(
    part_voxel_keys: frozenset, baseline_voxel_keys: frozenset
) -> float:
    """Fraction of the baseline (threshold 0) part cluster's voxels still present at this
    threshold. nan if the baseline cluster was itself empty -- there is nothing to measure
    coverage against.
    """
    if not baseline_voxel_keys:
        return float("nan")
    return len(part_voxel_keys & baseline_voxel_keys) / len(baseline_voxel_keys)


def sweep_run(
    reconstruction_run_dir: Path,
    alignment_json_path: Path,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    height_threshold_mm: float | None = None,
    footprint_mm: tuple[float, float] = DEFAULT_FOOTPRINT_MM,
    max_board_tilt_deg: float = DEFAULT_MAX_BOARD_TILT_DEG,
    voxel_size_mm: float = DEFAULT_VOXEL_SIZE_MM,
    min_points_per_voxel: int = DEFAULT_MIN_POINTS_PER_VOXEL,
) -> dict[str, Any]:
    """Load one real R-01 run and R-04 alignment, and measure stray fraction and part voxel
    coverage at each confidence threshold. 0.0 (no filtering) is always evaluated first, whether
    or not it is in `thresholds`, since every other threshold's coverage is measured against it.

    R-21: the board surface is fit once, from this run's own valid-depth pixels, before the
    threshold loop runs -- the same board shouldn't shift depending on which confidence
    threshold is being measured. `height_threshold_mm=None` (the default) uses that fitted
    surface's own measured threshold, same as `fuse_run`/`segment_run`; passing a value
    overrides it but the height is still measured above the fitted plane, never above z=0.
    """
    reconstruction_run_dir = Path(reconstruction_run_dir)
    alignment_json_path = Path(alignment_json_path)

    with np.load(reconstruction_run_dir / "poses.npz") as poses:
        extrinsic = poses["extrinsic"].astype(np.float64)
        intrinsic = poses["intrinsic"].astype(np.float64)
    with np.load(reconstruction_run_dir / "depth.npz") as depth_data:
        depth = depth_data["depth"].astype(np.float64)
        depth_conf = depth_data["depth_conf"].astype(np.float64)

    alignment_json = json.loads(alignment_json_path.read_text(encoding="utf-8"))
    scale = float(alignment_json["scale"])
    rotation = np.asarray(alignment_json["rotation"], dtype=np.float64)
    translation = np.asarray(alignment_json["translation_mm"], dtype=np.float64)

    world_points = unproject_depth_to_world(depth, extrinsic, intrinsic)
    board_frame_points = apply_similarity_transform(
        world_points.reshape(-1, 3), scale, rotation, translation
    )
    flat_confidence = depth_conf.reshape(-1)

    # Non-finite points (e.g. a zero/invalid depth sample) are dropped once, up front, exactly
    # as select_point_cloud drops them -- segment_part itself requires finite input and would
    # otherwise raise on the very first threshold.
    finite = np.isfinite(board_frame_points).all(axis=1)
    board_frame_points = board_frame_points[finite]
    flat_confidence = flat_confidence[finite]

    # measure_board_surface needs real depth measurements, not confidence -- the same
    # valid_depth definition fuse_run uses (finite, positive depth; a zero/invalid pixel
    # unprojects to the camera centre, not a board point, regardless of its confidence).
    valid_depth = (np.isfinite(depth) & (depth > 0.0)).reshape(-1)[finite]
    board_surface = measure_board_surface(
        board_frame_points[valid_depth], footprint_mm, max_tilt_deg=max_board_tilt_deg
    )
    resolved_height_threshold_mm = (
        board_surface.height_threshold_mm
        if height_threshold_mm is None
        else height_threshold_mm
    )

    # 0.0 first (coverage's reference point), then the requested thresholds ascending, with
    # duplicates dropped but order preserved (dict.fromkeys keeps first-seen order).
    ordered_thresholds = list(dict.fromkeys([0.0, *sorted(thresholds)]))

    rows: list[dict[str, Any]] = []
    baseline_voxel_keys: frozenset = frozenset()
    for threshold in ordered_thresholds:
        trusted = confidence_keep_mask(flat_confidence, threshold)
        trusted_points = board_frame_points[trusted]
        keep = segment_part(
            trusted_points, board_surface, height_threshold_mm, footprint_mm
        )
        segmented_points = trusted_points[keep]

        measurement = stray_point_fraction(
            segmented_points, voxel_size_mm, min_points_per_voxel
        )
        if threshold == 0.0:
            baseline_voxel_keys = measurement["part_voxel_keys"]

        rows.append(
            {
                "min_confidence": threshold,
                "segmented_point_count": measurement["total_point_count"],
                "stray_fraction": measurement["stray_fraction"],
                "part_point_count": measurement["part_point_count"],
                "part_voxel_coverage": part_voxel_coverage(
                    measurement["part_voxel_keys"], baseline_voxel_keys
                ),
                "x_span_mm": measurement["x_span_mm"],
                "y_span_mm": measurement["y_span_mm"],
                "top_height_mm": measurement["top_height_mm"],
            }
        )

    return {
        "story": ["R-20", "R-21"],
        "source_run_dir": str(reconstruction_run_dir),
        "source_alignment_json": str(alignment_json_path),
        "voxel_size_mm": voxel_size_mm,
        "min_points_per_voxel": min_points_per_voxel,
        "board_surface": board_surface.as_report(),
        "height_threshold_mm": resolved_height_threshold_mm,
        "footprint_mm": list(footprint_mm),
        "thresholds": rows,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "R-20: sweep VGGT confidence thresholds on one real reconstruction run and report "
            "stray-point fraction and part voxel coverage at each (evidence for D-045)."
        )
    )
    parser.add_argument("reconstruction_run_dir", type=Path)
    parser.add_argument(
        "alignment_json",
        type=Path,
        nargs="?",
        default=None,
        help="defaults to <reconstruction_run_dir>/metric_alignment.json",
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS)
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="defaults to <reconstruction_run_dir>/confidence_sweep.json",
    )
    parser.add_argument(
        "--height-threshold-mm",
        type=float,
        default=None,
        help="override the measured board-layer threshold (mm above the fitted board surface)",
    )
    parser.add_argument(
        "--max-board-tilt-deg", type=float, default=DEFAULT_MAX_BOARD_TILT_DEG
    )
    parser.add_argument("--voxel-size-mm", type=float, default=DEFAULT_VOXEL_SIZE_MM)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    alignment_json = args.alignment_json or (
        args.reconstruction_run_dir / "metric_alignment.json"
    )
    out_path = args.out or (args.reconstruction_run_dir / "confidence_sweep.json")
    if out_path.exists():
        print(
            f"R-20 CONFIDENCE SWEEP: FAIL - {out_path} already exists. Pick a new --out or "
            "delete it deliberately; overwriting silently would let two sweeps get mixed.",
            file=sys.stderr,
        )
        return 1

    try:
        report = sweep_run(
            args.reconstruction_run_dir,
            alignment_json,
            thresholds=args.thresholds,
            height_threshold_mm=args.height_threshold_mm,
            max_board_tilt_deg=args.max_board_tilt_deg,
            voxel_size_mm=args.voxel_size_mm,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"R-20 CONFIDENCE SWEEP: FAIL - {exc}", file=sys.stderr)
        return 1

    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"R-20 CONFIDENCE SWEEP: PASS - {len(report['thresholds'])} thresholds, "
        f"written to {out_path}"
    )
    for row in report["thresholds"]:
        print(
            f"  min_confidence={row['min_confidence']:<6} "
            f"stray={row['stray_fraction']:.3f} "
            f"coverage={row['part_voxel_coverage']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
