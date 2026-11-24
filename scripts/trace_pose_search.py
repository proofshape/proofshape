"""I-15: trace where the pose search keeps or loses the right pose, stage by stage.

I-15's first criterion asks, for the G-05 control with R-44's correction on, "whether a candidate
near the reference leaves the global search from either seed path, whether it survives candidate
selection, and what refinement does with it." This script makes that record for any model and
CAD, against a placement read from the photographs.

It runs the same calls the service runs (`recon/service.py`): `global_register`, then
`check_ambiguity`. It adds two things:

* `global_register(..., stages=...)` hands back the lists it already builds (every seeded
  candidate, what survived deduplication, what was selected), so nothing here re-implements the
  search and the record can't drift from it;
* each selected candidate is also put through `refine` on its own, with the seed
  `check_ambiguity` uses, because `check_ambiguity` merges refined poses that agree and so doesn't
  say what became of each one.

Every candidate at every stage is judged with `inspection.pose_judgement.judge_pose`: the photo
reading for face, gross yaw and centre, and the model's own walls for fine yaw where the part has
straight walls.

The reading is required. I-15 says the reference is read before anything is compared, and this
script is a comparison.

Usage (from the repository root):

    PYTHONPATH=.:inspect OMP_NUM_THREADS=1 .venv/bin/python scripts/trace_pose_search.py \\
        --model <variant>/uncertainty/mesh_uncertainty.glb \\
        --cad fixtures/reference_cad/<part>.stl \\
        --faces caption --turned 5 --clockwise \\
        --squares-from-left 4 --squares-from-caption 3 --face matches \\
        --side-normal 0 -1 --symmetry 360 --straight-walls \\
        --out <new file>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# R-31: the macOS thread guard has to run before Open3D is imported (see measure_part_accuracy).
from recon import runtime_env as _runtime_env  # noqa: F401  # isort: skip

import open3d as o3d  # isort: skip

from inspection import pose_judgement, registration
from inspection.registration import CandidatePose

from recon.provenance import LABEL_OBSERVED

# Share of Observed vertices farther than this from the posed CAD surface is reported, as P-41
# reported it ("12 to 44% of the Observed vertices are more than 5 mm from the part").
STRAY_DISTANCE_MM = 5.0

# The model's walls are read at these fractions of the CAD's height above the board, and the
# median taken. All three sit above I-15's support cut (at most a quarter of the height).
WALL_SLICE_FRACTIONS = (0.4, 0.5, 0.6)

STAGES = ("seeds", "deduplication", "selection", "refinement", "final ranking")


class _Judge:
    """Judges poses for one part, caching the model's outline at each fixed slice height."""

    def __init__(
        self,
        model_vertices: np.ndarray,
        model_faces: np.ndarray,
        cad_vertices: np.ndarray,
        cad_faces: np.ndarray,
        reading: pose_judgement.PhotoReading,
        *,
        named_side_normal_cad_xy: tuple[float, float],
        symmetry_deg: float,
        straight_walls: bool,
    ) -> None:
        self.model_vertices = model_vertices
        self.model_faces = model_faces
        self.cad_vertices = cad_vertices
        self.cad_faces = cad_faces
        self.reading = reading
        self.named_side_normal_cad_xy = named_side_normal_cad_xy
        self.symmetry_deg = symmetry_deg
        self.straight_walls = straight_walls
        self.cad_height = float(np.ptp(cad_vertices[:, 2]))
        self._model_loops: dict[float, np.ndarray | None] = {}

    def _model_loop(self, fraction: float) -> np.ndarray | None:
        """The model's outline at a fixed fraction of the CAD's height above the board.

        Fixed for the part, not taken from each pose. A refined pose can sit a few millimetres
        higher or lower than another, and slicing the model at each pose's own mid-height then
        compared poses against different outlines of the same model: on `g02_01` (15 mm) the
        wall error of two poses with nearly the same yaw read 22.3 and 0.4 degrees. The model's
        walls belong to the model, so every pose is measured against the same slices.
        """
        if fraction not in self._model_loops:
            # Board z points into the table, so h mm above the board is z = -h.
            level = -fraction * self.cad_height
            self._model_loops[fraction] = pose_judgement.outline_at(
                self.model_vertices, self.model_faces, level
            )
        return self._model_loops[fraction]

    def wall(self, pose: np.ndarray) -> pose_judgement.WallOffset | None:
        """Median wall offset over three slices, or None if fewer than two give outlines.

        Three heights rather than one, because one slice can pass through a noisy band of the
        reconstruction; the median of three can't be set by one bad slice. The CAD is cut at the
        same fractions of its own posed height: its walls are vertical, so its outline is the
        same at any of them.
        """
        if not self.straight_walls:
            return None
        posed = pose_judgement._transform(self.cad_vertices, pose)
        base = float(posed[:, 2].max())
        top = float(posed[:, 2].min())
        offsets = []
        aligned = []
        for fraction in WALL_SLICE_FRACTIONS:
            model_loop = self._model_loop(fraction)
            cad_loop = pose_judgement.outline_at(
                posed, self.cad_faces, base + fraction * (top - base)
            )
            if model_loop is None or cad_loop is None:
                continue
            result = pose_judgement.wall_offset(model_loop, cad_loop)
            offsets.append(result.offset_deg)
            aligned.append(result.aligned_fraction)
        if len(offsets) < 2:
            return None
        return pose_judgement.WallOffset(
            offset_deg=float(np.median(offsets)),
            aligned_fraction=float(np.median(aligned)),
        )

    def record(self, candidate: CandidatePose) -> dict[str, Any]:
        wall = self.wall(candidate.transformation)
        judgement = pose_judgement.judge_pose(
            self.cad_vertices,
            candidate.transformation,
            self.reading,
            named_side_normal_cad_xy=self.named_side_normal_cad_xy,
            symmetry_deg=self.symmetry_deg,
            wall_yaw_offset_deg=None if wall is None else wall.offset_deg,
        )
        reasons = list(judgement.reasons)
        if self.straight_walls and wall is None:
            # Never silently judged on the photo alone when walls were asked for: say so.
            reasons.append(
                "no closed outline at mid-height, so the walls weren't checked"
            )
        centre = pose_judgement.posed_centre_xy(
            self.cad_vertices, candidate.transformation
        )
        return {
            "source": candidate.source,
            # Kept so two records' poses can be compared as I-15 defines "same" (posed-surface
            # gap), not only by yaw and centre, which differ by a symmetry turn on symmetric parts.
            "transformation": np.asarray(candidate.transformation).tolist(),
            "fitness": candidate.fitness,
            "inlier_rmse_mm": candidate.inlier_rmse,
            "named_side_faces_deg": pose_judgement.pose_rotation_of_xy_direction(
                candidate.transformation, self.named_side_normal_cad_xy
            ),
            "centre_xy_mm": [float(centre[0]), float(centre[1])],
            "gross_yaw_error_deg": judgement.gross_yaw_error_deg,
            "wall_yaw_error_deg": judgement.wall_yaw_error_deg,
            "wall_aligned_fraction": None if wall is None else wall.aligned_fraction,
            "centre_error_mm": judgement.centre_error_mm,
            "centre_margin_mm": judgement.centre_margin_mm,
            "right": judgement.right and not (self.straight_walls and wall is None),
            "reasons": reasons,
        }


def _contains(candidates: Sequence[CandidatePose], wanted: CandidatePose) -> bool:
    # The search's lists hold the same objects from stage to stage, so identity is exact.
    return any(candidate is wanted for candidate in candidates)


def observed_stray_share(
    model_vertices: np.ndarray,
    model_labels: np.ndarray,
    cad_vertices: np.ndarray,
    cad_faces: np.ndarray,
    pose: np.ndarray,
) -> float | None:
    """Share of Observed model vertices farther than STRAY_DISTANCE_MM from the posed CAD."""
    observed = np.asarray(model_labels, dtype=object) == LABEL_OBSERVED
    if not observed.any():
        return None
    posed = pose_judgement._triangle_mesh(
        pose_judgement._transform(cad_vertices, pose), cad_faces
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(posed))
    distances = scene.compute_distance(
        o3d.core.Tensor(model_vertices[observed].astype(np.float32))
    ).numpy()
    return float(np.mean(distances > STRAY_DISTANCE_MM))


def stage_lost_at(right_at: dict[str, bool]) -> str | None:
    """The stage at which the right pose was finally lost, or None if the final pick is right.

    Presence isn't monotone: a right seed can miss selection while another selected seed
    refines into the right pose. So the stage named is the one after the *last* stage that
    still held a right candidate, not the first stage that didn't.
    """
    if right_at[STAGES[-1]]:
        return None
    last_present = -1
    for index, stage in enumerate(STAGES):
        if right_at[stage]:
            last_present = index
    return STAGES[last_present + 1]


def trace(
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
    cad_mesh: tuple[np.ndarray, np.ndarray, Any],
    reading: pose_judgement.PhotoReading,
    *,
    named_side_normal_cad_xy: tuple[float, float],
    symmetry_deg: float = pose_judgement.NO_SYMMETRY_DEG,
    straight_walls: bool = True,
    seed: int = 0,
    register_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the search as the service does and record each stage against the reading.

    register_kwargs are passed to `global_register` only (tests use them to make it smaller).
    The service passes none.
    """
    model_vertices = np.asarray(model[0], dtype=np.float64)
    model_faces = np.asarray(model[1], dtype=np.int64)
    model_labels = np.asarray(model[2], dtype=object)
    cad_vertices = np.asarray(cad_mesh[0], dtype=np.float64)
    cad_faces = np.asarray(cad_mesh[1], dtype=np.int64)
    judge = _Judge(
        model_vertices,
        model_faces,
        cad_vertices,
        cad_faces,
        reading,
        named_side_normal_cad_xy=named_side_normal_cad_xy,
        symmetry_deg=symmetry_deg,
        straight_walls=straight_walls,
    )

    stages: dict[str, Any] = {}
    selected = registration.global_register(
        model, cad_mesh, seed=seed, stages=stages, **(register_kwargs or {})
    )
    seeded: list[CandidatePose] = stages["seeded"]
    deduplicated: list[CandidatePose] = stages["deduplicated"]
    # global_register attaches the free-scale residual to its first pick with dataclasses
    # .replace, which makes a new object, so membership is checked against "selected" as
    # recorded just before that.
    selected_recorded: list[CandidatePose] = stages["selected"]

    ranked_seeds = sorted(seeded, key=registration._candidate_sort_key)
    seed_records = []
    for rank, candidate in enumerate(ranked_seeds, start=1):
        record = judge.record(candidate)
        record["fitness_rank"] = rank
        record["path"] = "ransac" if candidate.source.startswith("ransac") else "yaw"
        record["survived_deduplication"] = _contains(deduplicated, candidate)
        record["selected"] = _contains(selected_recorded, candidate)
        seed_records.append(record)
    reseed_records = []
    for candidate in stages["reseeded"]:
        record = judge.record(candidate)
        record["selected"] = _contains(selected_recorded, candidate)
        reseed_records.append(record)

    refinement_records = []
    for candidate in selected:
        before = judge.record(candidate)
        try:
            refined = registration.refine(candidate, model, cad_mesh, seed=seed)
        except ValueError as exc:
            refinement_records.append(
                {"from": before, "refined": None, "failure": str(exc)}
            )
            continue
        refinement_records.append(
            {"from": before, "refined": judge.record(refined), "failure": None}
        )

    final: dict[str, Any]
    try:
        ambiguity = registration.check_ambiguity(selected, model, cad_mesh, seed=seed)
    except ValueError as exc:
        final = {"refused": str(exc)}
    else:
        best = judge.record(ambiguity.best)
        final = {
            "refused": None,
            "best": best,
            "runner_up": None
            if ambiguity.runner_up is None
            else judge.record(ambiguity.runner_up),
            "orientation_ambiguous": ambiguity.orientation_ambiguous,
            "coverage_ratio": ambiguity.coverage_ratio,
            "threshold": ambiguity.threshold,
            "observed_stray_share": observed_stray_share(
                model_vertices,
                model_labels,
                cad_vertices,
                cad_faces,
                ambiguity.best.transformation,
            ),
        }

    right_at = {
        "seeds": any(record["right"] for record in seed_records + reseed_records),
        "deduplication": any(
            record["right"] and record["survived_deduplication"]
            for record in seed_records
        )
        or any(record["right"] for record in reseed_records),
        "selection": any(record["from"]["right"] for record in refinement_records),
        "refinement": any(
            record["refined"] is not None and record["refined"]["right"]
            for record in refinement_records
        ),
        "final ranking": final.get("refused") is None and final["best"]["right"],
    }
    lost_at = stage_lost_at(right_at)

    # What the service would serve, in I-15's criterion 3 terms.
    if final.get("refused") is not None:
        outcome = "refused"
    elif final["best"]["right"]:
        outcome = "right"
    elif final["orientation_ambiguous"]:
        outcome = "flagged"
    else:
        outcome = "served wrong, unflagged"

    by_path: dict[str, dict[str, Any]] = {}
    for path in ("ransac", "yaw"):
        on_path = [record for record in seed_records if record["path"] == path]
        right_on_path = [record for record in on_path if record["right"]]
        by_path[path] = {
            "candidates": len(on_path),
            "right": len(right_on_path),
            "best_right_fitness_rank": None
            if not right_on_path
            else min(record["fitness_rank"] for record in right_on_path),
        }

    return {
        "reading": {
            "named_side_faces_deg": reading.named_side_faces_deg,
            "centre_xy_mm": list(reading.centre_xy_mm),
            "face_matches": reading.face_matches,
        },
        "named_side_normal_cad_xy": list(named_side_normal_cad_xy),
        "symmetry_deg": symmetry_deg,
        "straight_walls": straight_walls,
        "margins": {
            "gross_yaw_deg": pose_judgement.GROSS_YAW_MARGIN_DEG,
            "wall_yaw_deg": pose_judgement.WALL_YAW_MARGIN_DEG,
            "centre_cap_mm": pose_judgement.CENTRE_MARGIN_CAP_MM,
        },
        "search": {
            "extent_mm": stages["extent_mm"],
            "correspondence_distance_mm": stages["correspondence_distance_mm"],
            "seeded": len(seeded),
            "deduplicated": len(deduplicated),
            "selected": len(selected),
            "by_path": by_path,
        },
        "right_candidate_present_at": right_at,
        "lost_at": lost_at,
        "outcome": outcome,
        "seeds": seed_records,
        "reseeds": reseed_records,
        "refinement": refinement_records,
        "final": final,
    }


def _face_matches(value: str) -> bool | None:
    return {"matches": True, "differs": False, "not-judged": None}[value]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="served model GLB")
    parser.add_argument("--cad", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new JSON file")
    reading = parser.add_argument_group("photo reading (the placement sheet's words)")
    reading.add_argument(
        "--faces", required=True, choices=["caption", "far", "left", "right"]
    )
    reading.add_argument("--turned", type=float, required=True, help="degrees, >= 0")
    way = reading.add_mutually_exclusive_group(required=True)
    way.add_argument("--clockwise", action="store_true", help="seen from above")
    way.add_argument("--anticlockwise", action="store_true", help="seen from above")
    reading.add_argument("--squares-from-left", type=float, required=True)
    reading.add_argument("--squares-from-caption", type=float, required=True)
    reading.add_argument(
        "--face", required=True, choices=["matches", "differs", "not-judged"]
    )
    part = parser.add_argument_group("the part")
    part.add_argument(
        "--side-normal",
        type=float,
        nargs=2,
        required=True,
        metavar=("X", "Y"),
        help="outward normal of the sheet's named side, in the CAD's own xy",
    )
    part.add_argument("--symmetry", type=float, default=pose_judgement.NO_SYMMETRY_DEG)
    walls = part.add_mutually_exclusive_group(required=True)
    walls.add_argument("--straight-walls", action="store_true")
    walls.add_argument("--no-straight-walls", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.out.exists():
        raise FileExistsError(
            f"{args.out} exists. Pick a new --out; overwriting would mix two runs' records."
        )

    from inspection.cad import load_mesh

    from recon.mesh_cleanup import read_glb
    from recon.provenance import colors_to_labels

    vertices, faces, colors = read_glb(args.model)
    if colors is None:
        raise ValueError(
            f"{args.model} has no provenance colours; nothing says which vertices are "
            "Observed, and refinement refuses a model without them."
        )
    model = (vertices, faces, colors_to_labels(colors))
    cad_vertices, cad_faces, cad_units = load_mesh(args.cad)
    reading = pose_judgement.reading_from_sheet(
        faces=args.faces,
        turned_deg=args.turned,
        turned_clockwise=args.clockwise,
        squares_from_left=args.squares_from_left,
        squares_from_caption=args.squares_from_caption,
        face_matches=_face_matches(args.face),
    )

    record = trace(
        model,
        (cad_vertices, cad_faces, cad_units),
        reading,
        named_side_normal_cad_xy=(args.side_normal[0], args.side_normal[1]),
        symmetry_deg=args.symmetry,
        straight_walls=args.straight_walls,
        seed=args.seed,
    )
    record["inputs"] = {
        "model": str(args.model),
        "cad": str(args.cad),
        "argv": sys.argv[1:] if argv is None else list(argv),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1), encoding="utf-8")

    search = record["search"]
    print(f"seeded {search['seeded']}, kept {search['selected']}")
    for path, counts in search["by_path"].items():
        print(
            f"  {path}: {counts['right']} of {counts['candidates']} right, "
            f"best right at fitness rank {counts['best_right_fitness_rank']}"
        )
    print(f"right candidate present at: {record['right_candidate_present_at']}")
    print(f"lost at: {record['lost_at']}; outcome: {record['outcome']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
