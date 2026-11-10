"""R-33: measure #231/#258's wall-surface face classification on a served, fused mesh.

#231 found two problems with a served `mesh_provenance.glb`: a board skin labelled Observed
(R-30's problem, now fixed), and wall surfaces that fill the whole fusion volume, labelled
Unobserved, with no story owning them. Both reviews classified the mesh's own faces by centroid
into four categories and reported each one's face share and Observed share -- done by hand each
time, never committed as code. This script is that classification, so R-33's first acceptance
criterion (did R-30 already shrink the wall share?) can be measured the same way again, on demand,
rather than redone by hand.

Categories, checked in this order -- each one claims whatever the previous ones didn't, so a face
with mixed properties (for example, a low board-skin face that also touches the volume's edge)
lands in the category checked last, not the one checked first:

1. `wall`: everything (the default, nothing else claims it).
2. `board_skin`: height above the board under `--board-skin-height-mm` (default 8).
3. `part`: centroid x/y inside `--part-x-mm`/`--part-y-mm` *and* height above the board under
   `--part-height-mm`. Neither is recorded anywhere this script can read it from, so both default
   to being derived from `recon.completion.isolate_part`'s core on this same mesh (its own x/y
   footprint and its overall kept height, each padded by `--footprint-pad-mm`, default 3) --
   `--part-x-mm`/`--part-y-mm`/`--part-height-mm` still override it when given.
4. `volume_boundary`: within `--volume-boundary-mm` (default 3) of the fusion volume's five
   *non-board* faces -- both x sides, both y sides, and the far z side, but not the near z side,
   which is the board itself (review on #278, @TabeenRaoof: `fuse_run`'s own volume bounds put
   the board at `bounds_max_mm[2]`, confirmed between -0.14 and 0.20 mm on ten real runs; counting
   that side as `volume_boundary` instead of `board_skin` hides part of the skin in exactly the
   before/after comparison this story exists to make). Checked last/highest priority: in #231's
   own table (the order this script follows), `volume_boundary` has a nonzero Observed share
   lower than `part`'s, only explicable if it's carved out *after* `part`.

Checked against two different by-hand classifications from #231 (2026-11-03) and #258
(2026-11-07) (review on #278, @TabeenRaoof, who had both in their own session logs): the two
disagreed with each other on both the boundary-priority order and which six-or-five faces count
as the volume's boundary. Neither is uniquely "the" classification; this script follows #231's
order and five-face boundary, as a documented choice, not a verified match to either.

"Height above the board" is `recon.part_segmentation.BoardSurface.height_above_board_mm`, the
fitted-plane-aware convention the rest of the pipeline already uses, not raw -z -- also a choice,
not a confirmed match to the by-hand versions, though @TabeenRaoof measured it changes little
(post-R-30 board tilt is under 0.17 deg, under 0.5 mm across the board). The board surface and
the fusion volume's bounds are both read from `fuse_run`'s own `tsdf_fusion.json`.

    python scripts/measure_wall_share.py FUSION_DIR PROVENANCE_GLB
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from recon import completion, mesh_cleanup, provenance
from recon.part_segmentation import BoardSurface

DEFAULT_VOLUME_BOUNDARY_MM = 3.0
DEFAULT_BOARD_SKIN_HEIGHT_MM = 8.0
DEFAULT_FOOTPRINT_PAD_MM = 3.0

CATEGORY_WALL = "wall"
CATEGORY_BOARD_SKIN = "board_skin"
CATEGORY_PART = "part"
CATEGORY_VOLUME_BOUNDARY = "volume_boundary"
CATEGORIES = (
    CATEGORY_PART,
    CATEGORY_BOARD_SKIN,
    CATEGORY_VOLUME_BOUNDARY,
    CATEGORY_WALL,
)


def classify_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    board_surface: BoardSurface,
    bounds_min_mm: np.ndarray,
    bounds_max_mm: np.ndarray,
    part_x_mm: tuple[float, float],
    part_y_mm: tuple[float, float],
    part_height_mm: float,
    volume_boundary_mm: float = DEFAULT_VOLUME_BOUNDARY_MM,
    board_skin_height_mm: float = DEFAULT_BOARD_SKIN_HEIGHT_MM,
) -> np.ndarray:
    """One of `CATEGORIES` per face, by its centroid. See the module docstring for the order."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    bounds_min_mm = np.asarray(bounds_min_mm, dtype=np.float64)
    bounds_max_mm = np.asarray(bounds_max_mm, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F,3), got {faces.shape}")

    centroids_mm = vertices[faces].mean(axis=1)
    height_above_board_mm = board_surface.height_above_board_mm(centroids_mm)

    # bounds_max_mm[2] is the board itself (fuse_run sets it to the fitted plane's own highest
    # footprint corner, review on #278) -- excluded here, or every low face everywhere on the
    # board, not just near its x/y edges, would count as volume_boundary instead of board_skin.
    touches_side_boundary = np.any(
        np.abs(centroids_mm[:, :2] - bounds_min_mm[:2]) <= volume_boundary_mm, axis=1
    ) | np.any(
        np.abs(centroids_mm[:, :2] - bounds_max_mm[:2]) <= volume_boundary_mm, axis=1
    )
    touches_far_z = np.abs(centroids_mm[:, 2] - bounds_min_mm[2]) <= volume_boundary_mm
    touches_boundary = touches_side_boundary | touches_far_z
    in_part_xy = (
        (centroids_mm[:, 0] >= part_x_mm[0])
        & (centroids_mm[:, 0] <= part_x_mm[1])
        & (centroids_mm[:, 1] >= part_y_mm[0])
        & (centroids_mm[:, 1] <= part_y_mm[1])
    )

    categories = np.full(len(faces), CATEGORY_WALL, dtype=object)
    categories[height_above_board_mm < board_skin_height_mm] = CATEGORY_BOARD_SKIN
    categories[in_part_xy & (height_above_board_mm < part_height_mm)] = CATEGORY_PART
    categories[touches_boundary] = CATEGORY_VOLUME_BOUNDARY
    return categories


def summarize(
    categories: np.ndarray, face_labels: np.ndarray
) -> dict[str, dict[str, float]]:
    """Per category: face share of the whole mesh, Observed share within it, and its own share
    of every Observed face in the mesh -- the same four columns #231's and #258's tables report."""
    total_faces = len(categories)
    total_observed = int(np.count_nonzero(face_labels == provenance.LABEL_OBSERVED))
    summary = {}
    for category in CATEGORIES:
        in_category = categories == category
        category_count = int(np.count_nonzero(in_category))
        observed_in_category = int(
            np.count_nonzero(in_category & (face_labels == provenance.LABEL_OBSERVED))
        )
        summary[category] = {
            "face_count": category_count,
            "face_share": category_count / total_faces if total_faces else 0.0,
            "observed_share": (
                observed_in_category / category_count if category_count else 0.0
            ),
            "share_of_all_observed": (
                observed_in_category / total_observed if total_observed else 0.0
            ),
        }
    return summary


def derive_part_footprint(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_labels: np.ndarray,
    board_surface: BoardSurface,
    pad_mm: float = DEFAULT_FOOTPRINT_PAD_MM,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """`(part_x_mm, part_y_mm, part_height_mm)` from `completion.isolate_part`'s own core and
    overall kept height on this mesh, each padded by `pad_mm` -- review on #278, @TabeenRaoof:
    the same source #231 used for s-04's footprint and #258 used for all five objects, so a rerun
    needs no number typed by hand. Raises whatever `isolate_part` raises if the mesh can't be
    isolated (for example `CompletionInputError` on too little Observed surface)."""
    isolation = completion.isolate_part(vertices, faces, vertex_labels, board_surface)
    core_min = isolation.report["core_xy_min_mm"]
    core_max = isolation.report["core_xy_max_mm"]
    top_mm = isolation.report["kept_height_range_mm"][1]
    return (
        (core_min[0] - pad_mm, core_max[0] + pad_mm),
        (core_min[1] - pad_mm, core_max[1] + pad_mm),
        top_mm + pad_mm,
    )


def measure(
    fusion_dir: Path,
    provenance_glb: Path,
    part_x_mm: tuple[float, float] | None = None,
    part_y_mm: tuple[float, float] | None = None,
    part_height_mm: float | None = None,
    volume_boundary_mm: float = DEFAULT_VOLUME_BOUNDARY_MM,
    board_skin_height_mm: float = DEFAULT_BOARD_SKIN_HEIGHT_MM,
    footprint_pad_mm: float = DEFAULT_FOOTPRINT_PAD_MM,
) -> dict[str, dict[str, float]]:
    """Load one real `fuse_run`/`provenance_run` pair and return `summarize`'s report.

    `part_x_mm`/`part_y_mm`/`part_height_mm` left `None` are derived with
    `derive_part_footprint`; any supplied explicitly overrides just that one.
    """
    fusion_report = json.loads((Path(fusion_dir) / "tsdf_fusion.json").read_text())
    board_surface = BoardSurface(**fusion_report["board_surface"])
    bounds_min_mm = np.asarray(fusion_report["bounds_min_mm"], dtype=np.float64)
    bounds_max_mm = np.asarray(fusion_report["bounds_max_mm"], dtype=np.float64)

    vertices, faces, colors = mesh_cleanup.read_glb(Path(provenance_glb))
    if colors is None:
        raise ValueError(f"{provenance_glb} has no vertex colors to read labels from")
    vertex_labels = provenance.colors_to_labels(colors)
    face_labels = provenance.label_faces(faces, vertex_labels)

    if part_x_mm is None or part_y_mm is None or part_height_mm is None:
        derived_x_mm, derived_y_mm, derived_height_mm = derive_part_footprint(
            vertices, faces, vertex_labels, board_surface, pad_mm=footprint_pad_mm
        )
        part_x_mm = derived_x_mm if part_x_mm is None else part_x_mm
        part_y_mm = derived_y_mm if part_y_mm is None else part_y_mm
        part_height_mm = derived_height_mm if part_height_mm is None else part_height_mm

    categories = classify_faces(
        vertices,
        faces,
        board_surface,
        bounds_min_mm,
        bounds_max_mm,
        part_x_mm,
        part_y_mm,
        part_height_mm,
        volume_boundary_mm=volume_boundary_mm,
        board_skin_height_mm=board_skin_height_mm,
    )
    return summarize(categories, face_labels)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "fusion_dir", type=Path, help="fuse_run's out_dir (holds tsdf_fusion.json)"
    )
    parser.add_argument(
        "provenance_glb", type=Path, help="provenance_run's mesh_provenance.glb"
    )
    parser.add_argument(
        "--part-x-mm",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="override the part's x footprint; default derives it with completion.isolate_part",
    )
    parser.add_argument(
        "--part-y-mm",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="override the part's y footprint; default derives it with completion.isolate_part",
    )
    parser.add_argument(
        "--part-height-mm",
        type=float,
        default=None,
        help="override the part's height cap; default derives it with completion.isolate_part",
    )
    parser.add_argument(
        "--volume-boundary-mm", type=float, default=DEFAULT_VOLUME_BOUNDARY_MM
    )
    parser.add_argument(
        "--board-skin-height-mm", type=float, default=DEFAULT_BOARD_SKIN_HEIGHT_MM
    )
    parser.add_argument(
        "--footprint-pad-mm", type=float, default=DEFAULT_FOOTPRINT_PAD_MM
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    summary = measure(
        args.fusion_dir,
        args.provenance_glb,
        part_x_mm=tuple(args.part_x_mm) if args.part_x_mm else None,
        part_y_mm=tuple(args.part_y_mm) if args.part_y_mm else None,
        part_height_mm=args.part_height_mm,
        volume_boundary_mm=args.volume_boundary_mm,
        board_skin_height_mm=args.board_skin_height_mm,
        footprint_pad_mm=args.footprint_pad_mm,
    )
    header = (
        f"{'category':<16}{'faces':>8}{'share':>9}{'observed':>11}{'of all obs.':>13}"
    )
    print(header)
    for category, row in summary.items():
        print(
            f"{category:<16}{row['face_count']:>8}{row['face_share'] * 100:>8.1f}%"
            f"{row['observed_share'] * 100:>10.1f}%{row['share_of_all_observed'] * 100:>12.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
