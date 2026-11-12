"""A-05: the disagreement map -- where the K completion samples disagree, as next-view guidance.

A-04 leaves K completed shapes of the part (LaS-Comp, D-057), one per seed. Where the camera saw the
part they agree, because every seed is completing the same observed surface; where it didn't, each
seed guesses, and the guesses differ. The spread across samples is the disagreement map (proposal
§6.6). **It drives capture guidance and nothing else** (D-005, D-006): it tells the next-view
scorer (`recon.next_view`, C-08) where one more photograph would settle the most guessing. It never
enters a verdict and never changes served geometry. This module only reads `completion/` and writes
inside its own `completion/disagreement/` folder.

The method is A-12's validated one (stories/A/A-12.md), applied per point instead of per pair:

1. **Spread.** For every point of every accepted sample k (`sample_<k>.ply`), the unsigned
   point-to-triangle distance to each other sample's mesh (`runner/sample_<j>_mesh_zup.ply`, turned
   back to the board frame); the point's spread is the median over the other samples. The median,
   so one outlier seed can't move every other sample's values (at K = 3 it is the mean of two).
2. **Seen or unseen.** A point within 2 mm of the partial cloud (`input/partial_zup.ply`) is
   surface the camera saw; beyond that it is guessed. This is A-12's split, unchanged.
3. **The board band.** Points lower above the board than the cut the partial was isolated with
   (`board_surface.height_threshold_mm`) are never targets: that is the board-contact face, which no
   camera above the board can see, and the band every partial is cut from, which no new photo
   would put into it.
4. **The weight (D-066).** The session's own noise level is the median spread over the seen
   points: what the model does with surface it was shown. An unseen point counts by how far its
   spread exceeds that, times the surface area it stands for, so a view's total reads as "excess
   spread it would see", in mm·mm². No fixed threshold is invented.
5. **Targets.** Weighted points are pooled into 2 mm voxels (R-07's TSDF voxel), which keeps the
   scorer's ray count near the served mesh's vertex count.

Like A-04, it never raises: every outcome is a status in `disagreement.json`. `unavailable` is the
normal state on every host without a completion runner (D-058), so it adds no warning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from recon import completion_contract as contract
from recon.part_segmentation import BoardSurface

STORY = "A-05"
SCHEMA_VERSION = 1

DISAGREEMENT_DIR_NAME = "disagreement"
REPORT_FILE = "disagreement.json"
TARGETS_FILE = "targets.npz"
# A-04's names for what this module reads (recon/completion.py); restated rather than imported so
# loading the map doesn't import the whole completion stage.
COMPLETION_REPORT = "completion.json"
FUSION_REPORT = "tsdf_fusion.json"

# A-12's seen/unseen split: "a point more than 2 mm from the partial is surface the camera didn't
# see" (stories/A/A-12.md). Reused, not re-chosen.
UNSEEN_BEYOND_MM = 2.0
# R-07's TSDF voxel. Finer than this the served mesh can't tell two surfaces apart anyway.
TARGET_VOXEL_MM = 2.0

USABLE_COMPLETION_STATUSES = ("complete", "partial")
MIN_SAMPLES = 2
STATUSES = ("unavailable", "complete", "failed")


@dataclass(frozen=True)
class DisagreementTargets:
    """What the next-view scorer aims at: board-frame points, each with the excess-spread area it
    stands for (mm·mm²). `normals` are outward unit normals when `cull_back_faces` is true, and all
    zero otherwise."""

    points_mm: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    cull_back_faces: bool


@dataclass(frozen=True)
class DisagreementOutcome:
    status: str
    report: dict[str, Any] = field(default_factory=dict)


class DisagreementInputError(ValueError):
    """A completion folder this stage can't build a map from."""


# --- geometry -------------------------------------------------------------------------------------


def _scene(
    vertices_mm: np.ndarray, faces: np.ndarray
) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(vertices_mm, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)),
    )
    return scene


def mesh_area_mm2(vertices_mm: np.ndarray, faces: np.ndarray) -> float:
    corners = np.asarray(vertices_mm, dtype=np.float64)[
        np.asarray(faces, dtype=np.int64)
    ]
    return float(
        0.5
        * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        ).sum()
    )


def winding_check(vertices_mm: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Whether a mesh's face normals can be trusted to point outward.

    Consistent means every edge two faces share is walked once in each direction, so the winding
    never flips across an edge. Only then does the signed volume's sign say which way the whole
    surface faces; a mesh with flipped patches can still have a positive volume.
    """
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keys = directed[:, 0] * (len(vertices_mm) + 1) + directed[:, 1]
    _, counts = np.unique(keys, return_counts=True)
    repeated_directed_edges = int(np.count_nonzero(counts > 1))
    # The divergence theorem, about the mesh's own centroid so a mesh with a few boundary edges
    # doesn't pick up a volume from where the origin happens to be.
    centred = np.asarray(vertices_mm, dtype=np.float64)
    centred = centred - centred.mean(axis=0)
    corners = centred[faces]
    signed_volume_mm3 = float(
        np.einsum(
            "ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])
        ).sum()
        / 6.0
    )
    return {
        "repeated_directed_edges": repeated_directed_edges,
        "signed_volume_mm3": signed_volume_mm3,
        "consistent": repeated_directed_edges == 0 and signed_volume_mm3 != 0.0,
    }


def _heights_above_board(
    points_mm: np.ndarray,
    board_surface: BoardSurface | None,
    completion_report: dict[str, Any],
    partial_mm: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Each point's height above the board, the band cut, and how both were found.

    With the session's `BoardSurface` (R-21) this is exactly the plane A-04 isolated the part with.
    A completion folder copied on its own (A-12's) has no fusion record, so the fallback is a level
    plane through the partial's lowest point, which A-04 recorded as sitting
    `kept_height_range_mm[0]` above the board: right for a level board, off by the tilt otherwise.
    """
    isolation = (completion_report.get("input") or {}).get("isolation") or {}
    if board_surface is not None:
        heights = board_surface.height_above_board_mm(points_mm)
        clearance_mm = float(board_surface.height_threshold_mm)
        method = {
            "method": "board_surface",
            "clearance_mm": clearance_mm,
            "completion_clearance_mm": isolation.get("clearance_mm"),
        }
        return heights, clearance_mm, method
    kept_range = isolation.get("kept_height_range_mm")
    if "clearance_mm" not in isolation or not kept_range:
        raise DisagreementInputError(
            "no board surface was given and completion.json records no isolation clearance and "
            "kept height range to fall back on"
        )
    clearance_mm = float(isolation["clearance_mm"])
    # -z is up, so the lowest partial point has the largest z.
    plane_z_mm = float(partial_mm[:, 2].max()) + float(kept_range[0])
    method = {
        "method": "level_plane_from_partial",
        "clearance_mm": clearance_mm,
        "plane_z_mm": plane_z_mm,
    }
    return plane_z_mm - points_mm[:, 2], clearance_mm, method


def _median_or_none(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def _pool_into_voxels(
    points_mm: np.ndarray,
    normals: np.ndarray,
    weights: np.ndarray,
    cull_back_faces: bool,
) -> DisagreementTargets:
    """One target per occupied 2 mm voxel: summed weight, weight-averaged position and normal.

    With culling on, the key also carries the normal's dominant axis and sign, so the two sides of
    a wall thinner than a voxel stay two targets facing opposite ways.
    """
    if len(points_mm) == 0:
        return DisagreementTargets(
            np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), cull_back_faces
        )
    cells = np.floor(points_mm / TARGET_VOXEL_MM).astype(np.int64)
    if cull_back_faces:
        axis = np.argmax(np.abs(normals), axis=1)
        sign = normals[np.arange(len(normals)), axis] > 0
        keys = np.column_stack([cells, axis * 2 + sign])
    else:
        keys = cells
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    target_count = int(inverse.max()) + 1
    summed_weights = np.bincount(inverse, weights=weights, minlength=target_count)
    positions = (
        np.column_stack(
            [
                np.bincount(
                    inverse,
                    weights=weights * points_mm[:, axis],
                    minlength=target_count,
                )
                for axis in range(3)
            ]
        )
        / summed_weights[:, None]
    )
    if cull_back_faces:
        summed_normals = np.column_stack(
            [
                np.bincount(
                    inverse, weights=weights * normals[:, axis], minlength=target_count
                )
                for axis in range(3)
            ]
        )
        lengths = np.linalg.norm(summed_normals, axis=1)
        lengths[lengths == 0.0] = 1.0
        target_normals = summed_normals / lengths[:, None]
    else:
        target_normals = np.zeros((target_count, 3))
    return DisagreementTargets(
        positions, target_normals, summed_weights, cull_back_faces
    )


# --- the map --------------------------------------------------------------------------------------


def build_disagreement_map(
    completion_dir: Path,
    completion_report: dict[str, Any],
    board_surface: BoardSurface | None,
) -> tuple[DisagreementTargets, dict[str, Any]]:
    """Every step in the module docstring, for a completion folder with at least two samples.
    Returns the targets and the report's measured fields. Raises on bad input; the stage catches."""
    completion_dir = Path(completion_dir)
    sample_records = sorted(
        completion_report.get("samples") or [], key=lambda s: int(s["k"])
    )
    # The accepted samples are the report's list, never a glob of runner/: a rejected sample's mesh
    # stays there (A-04 records why it was rejected) and must not vote.
    ks = [int(record["k"]) for record in sample_records]
    count = len(ks)

    partial_mm = contract.z_up_to_board(
        contract.read_points_ply(completion_dir / contract.PARTIAL_POINTS_PATH)
    )
    points: list[np.ndarray] = []
    meshes: list[tuple[np.ndarray, np.ndarray]] = []
    for k in ks:
        # sample_<k>.ply is write_ply's layout, the same one the partial uses.
        points.append(contract.read_points_ply(completion_dir / f"sample_{k}.ply"))
        vertices_z_up, faces = contract.read_mesh_ply_strict(
            completion_dir / contract.RUNNER_DIR / contract.sample_mesh_name(k)
        )
        meshes.append((contract.z_up_to_board(vertices_z_up), faces))

    windings = [winding_check(vertices, faces) for vertices, faces in meshes]
    cull_back_faces = all(winding["consistent"] for winding in windings)

    # distances[k][:, j]: sample k's points to sample j's surface; NaN on the sample's own column.
    # One scene at a time, so only one ~370k-triangle BVH is in memory at once.
    distances = [np.full((len(p), count), np.nan) for p in points]
    normals = [np.zeros((len(p), 3)) for p in points]
    for j, (vertices, faces) in enumerate(meshes):
        scene = _scene(vertices, faces)
        for k in range(count):
            query = o3d.core.Tensor(points[k].astype(np.float32))
            if k == j:
                if cull_back_faces:
                    face_normals = scene.compute_closest_points(query)[
                        "primitive_normals"
                    ].numpy()
                    normals[k] = face_normals.astype(np.float64) * np.sign(
                        windings[j]["signed_volume_mm3"]
                    )
                continue
            distances[k][:, j] = scene.compute_distance(query).numpy()
        del scene

    tree = cKDTree(partial_mm)
    all_points = np.vstack(points)
    sample_of_point = np.concatenate([np.full(len(p), i) for i, p in enumerate(points)])
    spread_mm = np.concatenate([np.nanmedian(d, axis=1) for d in distances])
    nearest_partial_mm, _ = tree.query(all_points)
    seen = nearest_partial_mm <= UNSEEN_BEYOND_MM
    heights_mm, clearance_mm, board_method = _heights_above_board(
        all_points, board_surface, completion_report, partial_mm
    )
    in_band = heights_mm < clearance_mm

    if not np.any(seen):
        raise DisagreementInputError(
            "no sample point lies within the seen distance of the partial, so the session has no "
            "noise level to measure the spread against"
        )
    noise_floor_mm = float(np.median(spread_mm[seen]))

    areas = np.array([mesh_area_mm2(vertices, faces) for vertices, faces in meshes])
    point_area_mm2 = (
        areas[sample_of_point]
        / np.array([len(p) for p in points])[sample_of_point]
        / count
    )
    is_target = ~seen & ~in_band
    weights = (
        np.where(is_target, np.maximum(spread_mm - noise_floor_mm, 0.0), 0.0)
        * point_area_mm2
    )
    weighted = weights > 0.0
    targets = _pool_into_voxels(
        all_points[weighted],
        np.vstack(normals)[weighted],
        weights[weighted],
        cull_back_faces,
    )

    # A-12's pairwise U/O, on these points, so the two can be compared directly.
    pairs = []
    for a in range(count):
        for b in range(a + 1, count):
            pooled = np.concatenate([distances[a][:, b], distances[b][:, a]])
            pooled_seen = np.concatenate(
                [seen[sample_of_point == a], seen[sample_of_point == b]]
            )
            pairs.append(
                {
                    "k": [ks[a], ks[b]],
                    "unseen_median_mm": _median_or_none(pooled[~pooled_seen]),
                    "seen_median_mm": _median_or_none(pooled[pooled_seen]),
                }
            )

    total_weight = float(weights.sum())
    measured = {
        "k_used": ks,
        "seeds": [int(record["seed"]) for record in sample_records],
        "point_count": len(all_points),
        "seen_point_count": int(np.count_nonzero(seen)),
        "board_band_point_count": int(np.count_nonzero(in_band)),
        "unseen_point_count": int(np.count_nonzero(is_target)),
        "weighted_point_count": int(np.count_nonzero(weighted)),
        "board": board_method,
        "noise_floor_mm": noise_floor_mm,
        "unseen_median_spread_mm": _median_or_none(spread_mm[is_target]),
        "unseen_p90_spread_mm": float(np.percentile(spread_mm[is_target], 90))
        if np.any(is_target)
        else None,
        "total_weight_mm3": total_weight,
        "target_count": len(targets.weights),
        "cull_back_faces": cull_back_faces,
        "samples": [
            {
                "k": ks[i],
                "area_mm2": float(areas[i]),
                "weight_share": float(
                    weights[sample_of_point == i].sum() / total_weight
                )
                if total_weight > 0
                else 0.0,
                "winding": windings[i],
            }
            for i in range(count)
        ],
        "pairs": pairs,
    }
    return targets, measured


# --- the stage ------------------------------------------------------------------------------------


def _base_report() -> dict[str, Any]:
    return {
        "story": STORY,
        "schema_version": SCHEMA_VERSION,
        "status": None,
        "reason": None,
        "guidance_only": True,
        "units": "mm",
        "frame": "board frame (recon/board_pose.py): millimetres, +z into the table",
        "weight_units": "mm*mm^2 (excess spread times the surface area it stands for)",
        "parameters": {
            "unseen_beyond_mm": UNSEEN_BEYOND_MM,
            "target_voxel_mm": TARGET_VOXEL_MM,
            "min_samples": MIN_SAMPLES,
        },
        "timings_s": {},
    }


def _write_targets_atomic(path: Path, targets: DisagreementTargets) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz.tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(
                handle,
                points_mm=targets.points_mm.astype(np.float64),
                normals=targets.normals.astype(np.float64),
                weights=targets.weights.astype(np.float64),
                cull_back_faces=np.array(targets.cull_back_faces),
            )
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read_board_surface(fusion_dir: Path) -> BoardSurface:
    report = json.loads((Path(fusion_dir) / FUSION_REPORT).read_text(encoding="utf-8"))
    if "board_surface" not in report:
        raise DisagreementInputError(f"{FUSION_REPORT} has no board_surface record")
    return BoardSurface(**report["board_surface"])


def _finish(
    out_dir: Path, report: dict[str, Any], started: float
) -> DisagreementOutcome:
    report["timings_s"]["total"] = time.perf_counter() - started
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        contract.write_json_atomic(out_dir / REPORT_FILE, report)
    except OSError as exc:
        report["status"] = "failed"
        report["reason"] = (
            f"{report.get('reason') or ''} ({REPORT_FILE} not written: {exc})"
        )
    return DisagreementOutcome(status=report["status"], report=report)


def disagreement_stage(
    completion_dir: Path,
    *,
    fusion_dir: Path | None = None,
    out_dir: Path | None = None,
) -> DisagreementOutcome:
    """Build the map for one completion folder. Never raises: every outcome is a status.

    Writes `<completion_dir>/disagreement/` (or `out_dir`). Refuses a folder that already holds a
    result, except an `unavailable` record: a folder prepared without a runner and ingested later
    from another machine (A-12's route, `completion.ingest_existing`) can then be mapped.
    `fusion_dir` supplies the session's board plane; without it the band falls back to a level
    plane (`_heights_above_board`).
    """
    started = time.perf_counter()
    completion_dir = Path(completion_dir)
    out_dir = (
        Path(out_dir) if out_dir is not None else completion_dir / DISAGREEMENT_DIR_NAME
    )
    report = _base_report()
    try:
        if out_dir.exists() and any(out_dir.iterdir()):
            previous_status = None
            try:
                previous_status = json.loads(
                    (out_dir / REPORT_FILE).read_text(encoding="utf-8")
                ).get("status")
            except (OSError, ValueError):
                pass
            leftover = {path.name for path in out_dir.iterdir()} - {REPORT_FILE}
            if previous_status != "unavailable" or leftover:
                # Nothing is written, so the earlier record survives intact.
                report["status"] = "failed"
                report["reason"] = (
                    f"{out_dir} already has a result; refusing to overwrite it"
                )
                report["timings_s"]["total"] = time.perf_counter() - started
                return DisagreementOutcome(status="failed", report=report)

        completion_report = json.loads(
            (completion_dir / COMPLETION_REPORT).read_text(encoding="utf-8")
        )
        completion_status = completion_report.get("status")
        report["completion_status"] = completion_status
        if completion_status not in USABLE_COMPLETION_STATUSES:
            report["status"] = "unavailable"
            report["reason"] = (
                f"completion status is {completion_status!r}; no samples to compare"
            )
            return _finish(out_dir, report, started)
        sample_count = len(completion_report.get("samples") or [])
        if sample_count < MIN_SAMPLES:
            report["status"] = "unavailable"
            report["reason"] = (
                f"{sample_count} accepted sample(s); a spread needs at least {MIN_SAMPLES}"
            )
            return _finish(out_dir, report, started)

        board_surface = (
            _read_board_surface(fusion_dir) if fusion_dir is not None else None
        )
        targets, measured = build_disagreement_map(
            completion_dir, completion_report, board_surface
        )
        report.update(measured)
        report["timings_s"]["map"] = time.perf_counter() - started
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_targets_atomic(out_dir / TARGETS_FILE, targets)
        report["status"] = "complete"
        return _finish(out_dir, report, started)
    except Exception as exc:  # noqa: BLE001 -- like A-04 (D-058): guidance never fails a session.
        report["status"] = "failed"
        report["reason"] = f"{type(exc).__name__}: {exc}"
        if out_dir.exists() and any(
            path.name != REPORT_FILE for path in out_dir.iterdir()
        ):
            # A half-written result must not be mistaken for this run's.
            (out_dir / TARGETS_FILE).unlink(missing_ok=True)
        return _finish(out_dir, report, started)


def load_targets(disagreement_dir: Path) -> DisagreementTargets | None:
    """The scorer's input, or None when there is no complete map to use."""
    disagreement_dir = Path(disagreement_dir)
    try:
        report = json.loads(
            (disagreement_dir / REPORT_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    if report.get("status") != "complete":
        return None
    with np.load(disagreement_dir / TARGETS_FILE) as data:
        points_mm = np.asarray(data["points_mm"], dtype=np.float64)
        normals = np.asarray(data["normals"], dtype=np.float64)
        weights = np.asarray(data["weights"], dtype=np.float64)
        cull_back_faces = bool(data["cull_back_faces"])
    if (
        points_mm.ndim != 2
        or points_mm.shape[1] != 3
        or normals.shape != points_mm.shape
    ):
        raise ValueError(
            f"{disagreement_dir / TARGETS_FILE} has malformed target arrays"
        )
    if weights.shape != (len(points_mm),) or not np.all(np.isfinite(weights)):
        raise ValueError(f"{disagreement_dir / TARGETS_FILE} has malformed weights")
    return DisagreementTargets(points_mm, normals, weights, cull_back_faces)


# --- the CLI --------------------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the disagreement map (A-05) for a completion folder: where the K "
        "completion samples disagree, as capture guidance only."
    )
    parser.add_argument(
        "completion_dir",
        type=Path,
        help="A completion folder holding completion.json, sample_<k>.ply and runner/ meshes",
    )
    parser.add_argument(
        "--fusion-dir",
        type=Path,
        default=None,
        help="The session's fusion folder (tsdf_fusion.json), for its board plane. Without it the "
        "board band falls back to a level plane through the partial's lowest point.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    outcome = disagreement_stage(
        args.completion_dir, fusion_dir=args.fusion_dir, out_dir=args.out_dir
    )
    print(json.dumps(outcome.report, indent=2, sort_keys=True))
    return 0 if outcome.status == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
