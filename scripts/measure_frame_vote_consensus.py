"""R-33 criterion 2: do wall-classified points get decided by fewer, less-agreeing frames than
real part points?

#231 found wall surfaces filling the whole fusion volume. `recon.tsdf_fusion.fuse_tsdf`'s
per-voxel update has no requirement that more than one frame agree a voxel is *on* the surface
before trusting a near-zero crossing -- `weight` just counts frames that said anything at all.
The suspected mechanism: with a limited number of mostly-downward viewpoints, a single frame's
single noisy depth sample can leave a stray near-zero crossing nothing else corroborates. This
script measures that directly rather than assuming it, by comparing a sample of wall-classified
points against a sample of part points from the *same* real mesh, using
`measure_wall_share`'s own classification to pick the two samples and
`recon.tsdf_fusion.measure_voxel_vote_consensus` to score each one.

It reruns `fuse_run` a second time on the same reconstruction, passing the sample points through
as `vote_sample_points_mm` -- this is wasteful (fusion runs twice) but keeps the vote logic
exactly in step with `fuse_tsdf`'s own per-frame rules rather than a second, possibly-drifted
copy of them, and avoids re-deriving depth/extrinsic/keep_mask outside `fuse_run` itself.
`voxel_size_mm`, `truncation_mm`, `height_threshold_mm` and `min_confidence` are read back from
the first (real) run's own `tsdf_fusion.json` and reproduced exactly; `measure` checks the rerun's
own face/vertex counts against that first run's recorded ones and refuses to report anything if
they differ. `min_kept_points`, `max_part_height_mm`, `max_board_tilt_deg` and `footprint_mm` are
not recorded there and are not overridable here either, so a mismatch most likely means the first
run didn't leave one of those at its default. `--part-x-mm`/`--part-y-mm`/`--part-height-mm`
override the sampling footprint `measure_wall_share.derive_part_footprint` would otherwise derive.

The "part" sample is Observed faces only (review on #281, @TabeenRaoof: on a real s-04 run, 35%
of part-footprint faces were Unobserved -- a base flange or similar low-agreement surface mixed
into "real, kept part geometry" would blur exactly the contrast this criterion tests). The "wall"
sample is every wall-classified face, Observed or not, matching criterion 1's own definition of
that category (almost all Unobserved, by #231's own numbers).

    python -m scripts.measure_frame_vote_consensus RECONSTRUCTION_RUN_DIR ALIGNMENT_JSON \\
        FUSION_DIR PROVENANCE_GLB OUT_DIR

Run with `-m` (not `python scripts/measure_frame_vote_consensus.py`): this script imports
`scripts.measure_wall_share`, and `scripts/` has no `__init__.py` to put the repository root on
`sys.path` the way a package install would -- `-m` does that itself. `measure_wall_share.py`
itself has no such import and runs fine either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from recon import mesh_cleanup, provenance
from recon.part_segmentation import BoardSurface
from recon.tsdf_fusion import DEFAULT_NEAR_SURFACE_MM, fuse_run
from scripts import measure_wall_share

DEFAULT_SAMPLE_COUNT = 50
DEFAULT_SEED = 0

VOTE_KEYS = (
    "contributing_frame_count",
    "near_surface_frame_count",
    "in_front_frame_count",
    "behind_frame_count",
)


def sample_points(
    fusion_dir: Path,
    provenance_glb: Path,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    part_x_mm: tuple[float, float] | None = None,
    part_y_mm: tuple[float, float] | None = None,
    part_height_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`(wall_points_mm, part_points_mm)`: up to `sample_count` face centroids of each category
    from `provenance_glb`, classified the same way `measure_wall_share` does. Returns fewer than
    asked for a category with less surface than that, rather than sampling with replacement."""
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
        derived_x, derived_y, derived_height = measure_wall_share.derive_part_footprint(
            vertices, faces, vertex_labels, board_surface
        )
        part_x_mm = derived_x if part_x_mm is None else part_x_mm
        part_y_mm = derived_y if part_y_mm is None else part_y_mm
        part_height_mm = derived_height if part_height_mm is None else part_height_mm

    categories = measure_wall_share.classify_faces(
        vertices,
        faces,
        board_surface,
        bounds_min_mm,
        bounds_max_mm,
        part_x_mm,
        part_y_mm,
        part_height_mm,
    )
    centroids_mm = vertices[faces].mean(axis=1)
    rng = np.random.default_rng(seed)

    def _sample(indices: np.ndarray) -> np.ndarray:
        indices = indices.copy()
        rng.shuffle(indices)
        return centroids_mm[indices[:sample_count]]

    wall_indices = np.flatnonzero(categories == measure_wall_share.CATEGORY_WALL)
    # Review on #281, @TabeenRaoof: an unfiltered "part" sample mixes in Unobserved part-footprint
    # surface (35% of it, on a real s-04 run) such as the base flange -- possibly the same
    # low-agreement kind of surface as the walls, which would blur the contrast this measures.
    part_indices = np.flatnonzero(
        (categories == measure_wall_share.CATEGORY_PART)
        & (face_labels == provenance.LABEL_OBSERVED)
    )
    return _sample(wall_indices), _sample(part_indices)


def summarize_votes(
    votes: list[dict[str, int]], wall_count: int
) -> dict[str, dict[str, float]]:
    """Median of each vote count, wall-group vs part-group -- a real surface point should show
    a high near_surface_frame_count; a spurious wall point, by the suspected mechanism, a low
    contributing_frame_count split across in_front/behind rather than clustered at near_surface.

    Also reports `near_surface_le_1_share`: the share of a group's samples decided "on the
    surface" by at most one frame -- the suspected mechanism's own "often one" prediction
    (review on #281, @TabeenRaoof), which a median alone doesn't test directly.
    """

    def _stats(group: list[dict[str, int]]) -> dict[str, float]:
        if not group:
            return {key: 0.0 for key in VOTE_KEYS} | {"near_surface_le_1_share": 0.0}
        stats = {
            key: float(np.median([item[key] for item in group])) for key in VOTE_KEYS
        }
        stats["near_surface_le_1_share"] = float(
            np.mean([item["near_surface_frame_count"] <= 1 for item in group])
        )
        return stats

    return {
        "wall": _stats(votes[:wall_count]),
        "part": _stats(votes[wall_count:]),
    }


def measure(
    reconstruction_run_dir: Path,
    alignment_json_path: Path,
    fusion_dir: Path,
    provenance_glb: Path,
    out_dir: Path,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    near_surface_mm: float = DEFAULT_NEAR_SURFACE_MM,
    part_x_mm: tuple[float, float] | None = None,
    part_y_mm: tuple[float, float] | None = None,
    part_height_mm: float | None = None,
) -> dict[str, dict[str, float]]:
    wall_points_mm, part_points_mm = sample_points(
        fusion_dir,
        provenance_glb,
        sample_count=sample_count,
        seed=seed,
        part_x_mm=part_x_mm,
        part_y_mm=part_y_mm,
        part_height_mm=part_height_mm,
    )
    if len(wall_points_mm) == 0 or len(part_points_mm) == 0:
        raise ValueError(
            f"found {len(wall_points_mm)} wall and {len(part_points_mm)} part sample points; "
            "need at least one of each to compare"
        )
    sample_points_mm = np.vstack([wall_points_mm, part_points_mm])

    fusion_report = json.loads((Path(fusion_dir) / "tsdf_fusion.json").read_text())
    rerun_report = fuse_run(
        reconstruction_run_dir,
        alignment_json_path,
        out_dir,
        voxel_size_mm=fusion_report["voxel_size_mm"],
        truncation_mm=fusion_report["truncation_mm"],
        height_threshold_mm=fusion_report["board_surface"]["height_threshold_mm"],
        min_confidence=fusion_report["min_confidence"],
        vote_sample_points_mm=sample_points_mm,
        vote_near_surface_mm=near_surface_mm,
    )
    # Non-blocking suggestion, review on #281, @TabeenRaoof: min_kept_points, max_part_height_mm,
    # max_board_tilt_deg and footprint_mm aren't recorded in tsdf_fusion.json and so can't be
    # reproduced here (see the module docstring) -- if the first run didn't leave one of them at
    # its default, this rerun silently fuses a different mesh and the vote comparison means
    # nothing. A face/vertex-count mismatch is the cheap, concrete sign that assumption broke.
    if (rerun_report["face_count"], rerun_report["vertex_count"]) != (
        fusion_report["face_count"],
        fusion_report["vertex_count"],
    ):
        raise ValueError(
            f"the rerun produced {rerun_report['face_count']} faces/{rerun_report['vertex_count']} "
            f"vertices, not the first run's {fusion_report['face_count']}/"
            f"{fusion_report['vertex_count']} -- it didn't reproduce the original fusion, so this "
            "comparison isn't against the mesh it claims to be. Check min_kept_points, "
            "max_part_height_mm, max_board_tilt_deg and footprint_mm on the first run."
        )

    vote_report = json.loads((Path(out_dir) / "vote_consensus.json").read_text())
    groups = ["wall"] * len(wall_points_mm) + ["part"] * len(part_points_mm)
    summary = summarize_votes(vote_report["votes"], len(wall_points_mm))
    # Review on #281, @TabeenRaoof: vote_consensus.json has the raw votes but not which samples
    # are wall vs part, so the golden results couldn't be re-analysed without re-sampling.
    (Path(out_dir) / "wall_vs_part_votes.json").write_text(
        json.dumps(
            {
                "story": "R-33",
                "samples": [
                    {"group": group, "point_mm": point, **vote}
                    for group, point, vote in zip(
                        groups, vote_report["sample_points_mm"], vote_report["votes"]
                    )
                ],
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reconstruction_run_dir", type=Path)
    parser.add_argument("alignment_json", type=Path)
    parser.add_argument(
        "fusion_dir",
        type=Path,
        help="an existing fuse_run out_dir for this reconstruction",
    )
    parser.add_argument(
        "provenance_glb", type=Path, help="that same run's mesh_provenance.glb"
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="fresh directory for this script's own fuse_run rerun",
    )
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--near-surface-mm", type=float, default=DEFAULT_NEAR_SURFACE_MM
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    summary = measure(
        args.reconstruction_run_dir,
        args.alignment_json,
        args.fusion_dir,
        args.provenance_glb,
        args.out_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        near_surface_mm=args.near_surface_mm,
        part_x_mm=tuple(args.part_x_mm) if args.part_x_mm else None,
        part_y_mm=tuple(args.part_y_mm) if args.part_y_mm else None,
        part_height_mm=args.part_height_mm,
    )
    print(f"{'':<12}" + "".join(f"{key:>24}" for key in VOTE_KEYS))
    for group in ("wall", "part"):
        print(
            f"{group:<12}"
            + "".join(f"{summary[group][key]:>24.1f}" for key in VOTE_KEYS)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
