"""R-06/R-21: keep only the part geometry from R-04's board-frame point cloud.

R-04 aligns R-01's arbitrary-scale reconstruction to the board frame, in millimetres, but its
point cloud still has the board pattern, the table around it, and anything else a frame happened
to see (a hand, background clutter). This module drops everything that isn't the part, using two
independent checks against the board frame:

  - Height: keep points that sit above the *measured* board surface by at least a threshold.
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

R-21: R-04 aligns camera centres, not the board plane, so the real board does not land exactly on
z = 0 -- R-17 measured it 1.6-5.4 mm below z = 0 on the golden set, with tilt up to 0.9 degrees.
`measure_board_surface` fits a robust plane to the board points actually seen around the part, and
`segment_part`'s height check is relative to that plane, not to z = 0. The threshold itself is
measured from the fitted plane's own board layer (see `measure_board_surface`'s docstring), not a
fixed 2 mm, so a reconstruction with a thicker or noisier board layer (R-17 found a weak second
layer 4-6 mm up on two of the five golden objects) still gets a threshold that clears it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from recon.board_pose import BOARD_SQUARE_MM, BOARD_SQUARES_X, BOARD_SQUARES_Y
from recon.metric_alignment import apply_similarity_transform
from recon.vggt_runner import read_ply, write_ply

# The printed pattern's rectangle in the board frame: x in [0, BOARD_SQUARES_X*BOARD_SQUARE_MM],
# y in [0, BOARD_SQUARES_Y*BOARD_SQUARE_MM]. Derived from board_pose.py's constants, not
# re-declared, so the two modules can't drift apart (currently 160.0 x 120.0 mm, D-034's board).
DEFAULT_FOOTPRINT_MM = (
    BOARD_SQUARES_X * BOARD_SQUARE_MM,
    BOARD_SQUARES_Y * BOARD_SQUARE_MM,
)

# --- R-21: board-plane fit parameters. Chosen values, not measured constants -- all overridable.

# Candidate board points for the plane fit are points inside the footprint with |z| within this
# band of the assumed z=0. Wide enough to hold every offset R-17 measured (1.6-5.4 mm) plus
# headroom, narrow enough to exclude any hand-sized part, which sits well above it.
DEFAULT_BOARD_SEARCH_BAND_MM = 15.0

# Robust-fit outlier rejection: a candidate point more than this many robust-sigma from the
# current plane is dropped before the next refit. Standard choice for MAD-based sigma clipping.
DEFAULT_BOARD_FIT_SIGMA_CLIP = 3.0

# Refit/reject is repeated at most this many times; it stops early once the inlier set stops
# changing. A cap, not a tuned constant -- the fit has always converged well before this on the
# golden set.
DEFAULT_BOARD_FIT_MAX_ITERATIONS = 20

# After the robust fit, the board plane must still have at least this many inlier points, or the
# fit is too thin to trust -- raise rather than silently accept a near-empty plane.
DEFAULT_MIN_BOARD_POINTS = 1000

# A fitted board tilt beyond this is treated as implausible (board_pose's own detection already
# rejects a badly-posed board) rather than fed into the height cut. 2 degrees is 5.6 mm of rise
# across the 160 mm board; the golden set's measured maximum is 0.8 degrees. Overridable per the
# acceptance criteria.
DEFAULT_MAX_BOARD_TILT_DEG = 2.0

# The board-layer-top search (see measure_board_surface) histograms height above the fitted plane
# in bins this wide...
DEFAULT_BOARD_LAYER_BIN_MM = 0.5

# ...out to this far above the plane. A dense layer that reaches this limit means the search
# didn't find where the board layer ends -- that's an error, not a guess (see
# measure_board_surface).
DEFAULT_MAX_BOARD_LAYER_MM = 10.0

# A height bin counts as part of the board layer if it holds at least this fraction of the
# tallest bin's point count. Low enough to catch a raised second layer separated from the main
# board by a sparser gap (R-17's s-01/s-02), high enough not to chase a handful of noise points
# arbitrarily far up.
DEFAULT_BOARD_LAYER_DENSITY_FRACTION = 0.05

# The height threshold never goes below this, even if the fitted board layer is thinner --
# R-06's original fixed margin, kept as a floor once the board is this clean.
DEFAULT_MIN_BOARD_CLEARANCE_MM = 2.0


@dataclass(frozen=True)
class BoardSurface:
    """A robust planar fit to the real board surface, in the board frame (R-21).

    The plane is z = slope_x*x + slope_y*y + z_at_origin_mm. `height_above_board_mm` measures a
    point's height above *this* plane, not above z = 0, using the same -z-is-up convention as the
    rest of this module.
    """

    slope_x: float
    slope_y: float
    z_at_origin_mm: float
    offset_mm: float
    tilt_deg: float
    residual_sigma_mm: float
    inlier_count: int
    candidate_count: int
    board_layer_top_mm: float
    height_threshold_mm: float

    def plane_z_mm(self, x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
        return self.slope_x * x_mm + self.slope_y * y_mm + self.z_at_origin_mm

    def height_above_board_mm(self, points_board_frame_mm: np.ndarray) -> np.ndarray:
        """Height above *this* plane for each point, positive meaning above the board."""
        x_mm = points_board_frame_mm[:, 0]
        y_mm = points_board_frame_mm[:, 1]
        z_mm = points_board_frame_mm[:, 2]
        return self.plane_z_mm(x_mm, y_mm) - z_mm

    def as_report(self) -> dict[str, Any]:
        return {
            "slope_x": self.slope_x,
            "slope_y": self.slope_y,
            "z_at_origin_mm": self.z_at_origin_mm,
            "offset_mm": self.offset_mm,
            "tilt_deg": self.tilt_deg,
            "residual_sigma_mm": self.residual_sigma_mm,
            "inlier_count": self.inlier_count,
            "candidate_count": self.candidate_count,
            "board_layer_top_mm": self.board_layer_top_mm,
            "height_threshold_mm": self.height_threshold_mm,
        }


def measure_board_surface(
    points_board_frame_mm: np.ndarray,
    footprint_mm: tuple[float, float] = DEFAULT_FOOTPRINT_MM,
    search_band_mm: float = DEFAULT_BOARD_SEARCH_BAND_MM,
    sigma_clip: float = DEFAULT_BOARD_FIT_SIGMA_CLIP,
    max_fit_iterations: int = DEFAULT_BOARD_FIT_MAX_ITERATIONS,
    min_board_points: int = DEFAULT_MIN_BOARD_POINTS,
    max_tilt_deg: float = DEFAULT_MAX_BOARD_TILT_DEG,
    layer_bin_mm: float = DEFAULT_BOARD_LAYER_BIN_MM,
    max_board_layer_mm: float = DEFAULT_MAX_BOARD_LAYER_MM,
    layer_density_fraction: float = DEFAULT_BOARD_LAYER_DENSITY_FRACTION,
    min_clearance_mm: float = DEFAULT_MIN_BOARD_CLEARANCE_MM,
) -> BoardSurface:
    """Fit the real board surface from the board points around the part, and size the height cut.

    Two things happen here, both against the *same* candidate points (everything inside the
    footprint within `search_band_mm` of the assumed z=0 -- wide enough for the board's offset,
    narrow enough to exclude any hand-sized part):

    1. A robust plane fit (median start, then least-squares with iterative MAD-sigma clipping,
       deterministic -- no RANSAC, no random state) gives the board's offset and tilt.
    2. A histogram of height-above-that-plane finds where the board's own point layer actually
       ends -- not just the fitted plane's noise, but any secondary layer sitting above it (R-17
       found VGGT reconstructing a weak second board layer 4-6 mm up on two of the five golden
       objects). The height threshold is set to clear the *top* of the highest dense bin, so a
       raised layer separated from the main board by a sparser gap doesn't slip through the cut.

    Raises:
        ValueError: too few candidate or inlier board points, a degenerate (collinear) point
            spread, a tilt beyond `max_tilt_deg`, or a dense point layer that still reaches
            `max_board_layer_mm` without thinning out (the search didn't find where it ends --
            this also catches a part that is itself thin and flat across the whole footprint).
    """
    points_board_frame_mm = np.asarray(points_board_frame_mm, dtype=np.float64)
    if points_board_frame_mm.ndim != 2 or points_board_frame_mm.shape[1] != 3:
        raise ValueError(
            f"points_board_frame_mm must have shape (N,3), got {points_board_frame_mm.shape}"
        )
    if not np.isfinite(points_board_frame_mm).all():
        raise ValueError("points_board_frame_mm must be finite.")

    x_mm = points_board_frame_mm[:, 0]
    y_mm = points_board_frame_mm[:, 1]
    z_mm = points_board_frame_mm[:, 2]
    footprint_width_mm, footprint_height_mm = footprint_mm

    in_footprint = (
        (x_mm >= 0.0)
        & (x_mm <= footprint_width_mm)
        & (y_mm >= 0.0)
        & (y_mm <= footprint_height_mm)
    )
    candidates = in_footprint & (np.abs(z_mm) <= search_band_mm)
    candidate_count = int(np.count_nonzero(candidates))
    if candidate_count < min_board_points:
        raise ValueError(
            f"only {candidate_count} candidate board points within {search_band_mm} mm of z=0 "
            f"inside the footprint; need at least {min_board_points}. Too little of the board is "
            "visible around the part to fit a surface -- never falling back to z=0 silently."
        )

    cx_mm = x_mm[candidates]
    cy_mm = y_mm[candidates]
    cz_mm = z_mm[candidates]

    median_z_mm = float(np.median(cz_mm))
    start_sigma_mm = 1.4826 * float(np.median(np.abs(cz_mm - median_z_mm)))
    if start_sigma_mm > 0:
        inlier = np.abs(cz_mm - median_z_mm) <= sigma_clip * start_sigma_mm
    else:
        inlier = np.ones_like(cz_mm, dtype=bool)

    coefficients = np.array([0.0, 0.0, median_z_mm])
    residual_sigma_mm = start_sigma_mm
    for _ in range(max_fit_iterations):
        inlier_count = int(np.count_nonzero(inlier))
        if inlier_count < 3:
            raise ValueError(
                f"only {inlier_count} inlier board points survived robust fitting; too few to "
                "fit a plane."
            )
        design = np.column_stack([cx_mm[inlier], cy_mm[inlier], np.ones(inlier_count)])
        coefficients, _, rank, _ = np.linalg.lstsq(design, cz_mm[inlier], rcond=None)
        if rank < 3:
            raise ValueError(
                "candidate board points do not span a plane (rank-deficient fit) -- "
                "the footprint's x/y spread is degenerate."
            )
        residuals_mm = cz_mm - (
            coefficients[0] * cx_mm + coefficients[1] * cy_mm + coefficients[2]
        )
        residual_sigma_mm = 1.4826 * float(np.median(np.abs(residuals_mm[inlier])))
        if residual_sigma_mm <= 0:
            new_inlier = inlier
        else:
            new_inlier = np.abs(residuals_mm) <= sigma_clip * residual_sigma_mm
        if np.array_equal(new_inlier, inlier):
            break
        inlier = new_inlier

    inlier_count = int(np.count_nonzero(inlier))
    if inlier_count < min_board_points:
        raise ValueError(
            f"only {inlier_count} inlier board points after robust fitting; need at least "
            f"{min_board_points}. Not falling back to z=0 silently."
        )

    slope_x, slope_y, z_at_origin_mm = (float(c) for c in coefficients)
    tilt_deg = float(np.degrees(np.arctan(np.hypot(slope_x, slope_y))))
    if tilt_deg > max_tilt_deg:
        raise ValueError(
            f"fitted board tilt {tilt_deg:.2f} deg exceeds max_tilt_deg={max_tilt_deg}. "
            "If this board really is tilted this much, pass a larger max_tilt_deg explicitly "
            "rather than silently accepting it."
        )

    offset_mm = (
        slope_x * (footprint_width_mm / 2.0)
        + slope_y * (footprint_height_mm / 2.0)
        + z_at_origin_mm
    )

    # Board-layer top: histogram height-above-plane for *every* candidate (not just inliers --
    # a raised second layer is exactly what sigma-clipping just excluded as an outlier).
    plane_z_at_candidates_mm = slope_x * cx_mm + slope_y * cy_mm + z_at_origin_mm
    height_above_board_mm = plane_z_at_candidates_mm - cz_mm
    bin_edges_mm = np.arange(
        -layer_bin_mm, max_board_layer_mm + layer_bin_mm, layer_bin_mm
    )
    histogram, _ = np.histogram(height_above_board_mm, bin_edges_mm)
    peak_count = int(histogram.max())
    qualifying_bins = np.nonzero(histogram >= layer_density_fraction * peak_count)[0]
    top_bin_index = int(qualifying_bins.max())
    if top_bin_index == len(histogram) - 1:
        raise ValueError(
            f"the board's point layer is still dense at {max_board_layer_mm} mm above the fitted "
            "plane -- its top wasn't found within the search zone. Pass an explicit "
            "height_threshold_mm, or raise max_board_layer_mm if the board layer is genuinely "
            "this thick."
        )
    board_layer_top_mm = float(bin_edges_mm[top_bin_index + 1])
    height_threshold_mm = max(min_clearance_mm, board_layer_top_mm)

    return BoardSurface(
        slope_x=slope_x,
        slope_y=slope_y,
        z_at_origin_mm=z_at_origin_mm,
        offset_mm=offset_mm,
        tilt_deg=tilt_deg,
        residual_sigma_mm=residual_sigma_mm,
        inlier_count=inlier_count,
        candidate_count=candidate_count,
        board_layer_top_mm=board_layer_top_mm,
        height_threshold_mm=height_threshold_mm,
    )


def segment_part(
    points_board_frame_mm: np.ndarray,
    board_surface: BoardSurface,
    height_threshold_mm: float | None = None,
    footprint_mm: tuple[float, float] = DEFAULT_FOOTPRINT_MM,
) -> np.ndarray:
    """Return a boolean keep-mask: True where a point is part geometry, not board/background.

    Args:
        points_board_frame_mm: (N, 3) points already in the board frame, in millimetres (R-04's
            `AlignmentResult.transform_points` / `apply_similarity_transform` output).
        board_surface: the fitted board plane (`measure_board_surface`'s output). Height is
            measured above *this* surface, not above z=0 (R-21).
        height_threshold_mm: minimum height above `board_surface` to keep a point. Defaults to
            `board_surface.height_threshold_mm` (measured from the board's own point layer); an
            explicit value here still applies above the measured plane, not above z=0.
        footprint_mm: (width_mm, height_mm) of the keep rectangle, anchored at the board origin
            (x in [0, width_mm], y in [0, height_mm]). Defaults to the printed pattern's own size.

    Raises:
        ValueError: on a malformed shape, non-finite input, or a non-positive threshold/footprint.
        TypeError: if `board_surface` isn't a `BoardSurface`.
    """
    points_board_frame_mm = np.asarray(points_board_frame_mm, dtype=np.float64)
    if points_board_frame_mm.ndim != 2 or points_board_frame_mm.shape[1] != 3:
        raise ValueError(
            f"points_board_frame_mm must have shape (N,3), got {points_board_frame_mm.shape}"
        )
    if not np.isfinite(points_board_frame_mm).all():
        raise ValueError("points_board_frame_mm must be finite.")
    if not isinstance(board_surface, BoardSurface):
        raise TypeError(
            "board_surface must be a BoardSurface (see measure_board_surface) -- "
            "height is never measured from z=0 directly."
        )
    resolved_threshold_mm = (
        board_surface.height_threshold_mm
        if height_threshold_mm is None
        else height_threshold_mm
    )
    if not (np.isfinite(resolved_threshold_mm) and resolved_threshold_mm > 0):
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

    height_above_board_mm = board_surface.height_above_board_mm(points_board_frame_mm)
    keep_height = height_above_board_mm >= resolved_threshold_mm
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
    board_surface: BoardSurface,
    height_threshold_mm: float,
    height_threshold_source: str,
    footprint_mm: tuple[float, float],
    total_points: int,
    kept_points: int,
) -> dict[str, Any]:
    """Write a JSON report describing what was kept and why, mirroring R-04/R-05's convention."""
    report = {
        "story": ["R-06", "R-21"],
        "source_points_ply": str(Path(reconstruction_run_dir) / "points.ply"),
        "source_alignment_json": str(alignment_json_path),
        "units": "mm",
        "frame": "R-02 ChArUco board frame",
        "board_surface": board_surface.as_report(),
        "height_threshold_mm": height_threshold_mm,
        "height_threshold_source": height_threshold_source,
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
    height_threshold_mm: float | None = None,
    footprint_mm: tuple[float, float] = DEFAULT_FOOTPRINT_MM,
    max_tilt_deg: float = DEFAULT_MAX_BOARD_TILT_DEG,
) -> dict[str, Any]:
    """Load an R-01 point cloud and an R-04 alignment, segment, and write the outputs.

    Fits the board surface from the aligned points themselves (R-21), then segments relative to
    it. Writes `out_dir/segmented_points.ply` (kept points, board-frame mm) and
    `out_dir/part_segmentation.json` (the report, including the fitted board surface). Refuses to
    overwrite an existing non-empty `out_dir`, matching R-01/R-02's guard against mixing two runs'
    outputs.
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
    board_surface = measure_board_surface(
        points_board_frame_mm, footprint_mm, max_tilt_deg=max_tilt_deg
    )
    height_threshold_source = "measured"
    resolved_threshold_mm = board_surface.height_threshold_mm
    if height_threshold_mm is not None:
        height_threshold_source = "override"
        resolved_threshold_mm = height_threshold_mm
    keep = segment_part(
        points_board_frame_mm, board_surface, height_threshold_mm, footprint_mm
    )

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
        board_surface,
        resolved_threshold_mm,
        height_threshold_source,
        footprint_mm,
        total_points=len(points),
        kept_points=int(np.count_nonzero(keep)),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment the part from the board in an R-04-aligned point cloud (R-06/R-21)."
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
        "--height-threshold-mm",
        type=float,
        default=None,
        help="override the measured board-layer threshold (mm above the fitted board surface)",
    )
    parser.add_argument(
        "--max-board-tilt-deg", type=float, default=DEFAULT_MAX_BOARD_TILT_DEG
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
            max_tilt_deg=args.max_board_tilt_deg,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"R-06 SEGMENTATION: FAIL - {exc}", file=sys.stderr)
        return 1

    total = report["total_point_count"]
    kept = report["kept_point_count"]
    fraction = (kept / total * 100.0) if total else 0.0
    print(f"R-06 SEGMENTATION: PASS - kept {kept}/{total} points ({fraction:.1f}%)")
    print(
        f"Board surface: offset {report['board_surface']['offset_mm']:.2f} mm, "
        f"tilt {report['board_surface']['tilt_deg']:.2f} deg, "
        f"height threshold {report['height_threshold_mm']:.2f} mm "
        f"({report['height_threshold_source']})"
    )
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
