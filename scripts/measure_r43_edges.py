"""Reproduce R-43's offline edge comparison from saved R-32 inputs and a fusion sweep.

Run from the checkout root with python -m scripts.measure_r43_edges.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from recon import cad_size_check as size
from recon import mesh_cleanup, provenance, uncertainty
from scripts.measure_g04_board_error import load_board_pitch_mm

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    "s-04": ("sess_e8d97b4c3a37", "s04_triangular_prism_60-26-65x30mm.stl"),
    "s-05": ("sess_4ab66282d7d7", "s05_rect_prism_30x30x60mm.stl"),
}
VOXELS = ("2.0", "1.5", "1.0")
Line = tuple[np.ndarray, np.ndarray]


@dataclass
class WallEdgeFit:
    lengths_mm: tuple[float | None, ...]
    angle_offsets_deg: tuple[float | None, ...]
    first_fit_angle_offsets_deg: tuple[float | None, ...]
    wall_point_counts: tuple[int, ...]
    lines: tuple[Line | None, ...]
    rounds: int
    unstable_point_count: int
    converged: bool = True


def _direction_degrees(direction: np.ndarray) -> float:
    return math.degrees(math.atan2(direction[1], direction[0]))


def _angle_offsets(
    fitted: Sequence[Line | None], seeds: Sequence[Line]
) -> tuple[float | None, ...]:
    # A line has no preferred orientation, so compare the unsigned acute angle.
    return tuple(
        None
        if line is None
        else math.degrees(math.acos(np.clip(abs(float(line[1] @ seed[1])), 0, 1)))
        for line, seed in zip(fitted, seeds, strict=True)
    )


def fit_wall_edges(
    band_xy: np.ndarray,
    posed_edges: Sequence[Line],
    gate_mm: float,
    *,
    max_rounds: int = 50,
) -> WallEdgeFit:
    """Refit neighbouring wall lines without permuting their original CAD edge indices."""
    points = np.asarray(band_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite wall samples with shape (N, 2).")
    if len(posed_edges) < 3 or not math.isfinite(gate_mm) or gate_mm <= 0:
        raise ValueError(
            "A wall fit needs at least three posed edges and a positive gate."
        )
    seeds = []
    for start, end in posed_edges:
        direction = end - start
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Posed wall edges must have finite, distinct endpoints.")
        seeds.append(((start + end) / 2, direction / norm))

    assignment_lines = seeds.copy()
    assignments: list[np.ndarray] = []
    seen_assignments: dict[bytes, int] = {}
    unstable = np.zeros(len(points), dtype=bool)
    first_angles = None
    for round_index in range(1, max_rounds + 1):
        origins = np.array([origin for origin, _ in assignment_lines])
        directions = np.array([direction for _, direction in assignment_lines])
        normals = np.column_stack((-directions[:, 1], directions[:, 0]))
        distances = np.abs(points @ normals.T - (origins * normals).sum(axis=1))
        nearest = distances.argmin(axis=1)
        assigned = np.where(
            distances[np.arange(len(points)), nearest] < gate_mm, nearest, -1
        )
        assigned[unstable] = -1
        if assignments and np.array_equal(assigned, assignments[-1]):
            break

        cycle_start = seen_assignments.get(assigned.tobytes())
        if cycle_start is not None:
            # A few corner or gate-boundary points can alternate between assignments.
            # Exclude only samples whose labels varied within the detected cycle, so the
            # final wall fits do not depend on which phase happened to end the replay.
            for earlier in assignments[cycle_start + 1 :]:
                unstable |= earlier != assigned
            assigned[unstable] = -1
            assignments.clear()
            seen_assignments.clear()
        assignment_key = assigned.tobytes()
        seen_assignments[assignment_key] = len(assignments)
        assignments.append(assigned)

        fitted: list[Line | None] = []
        counts = []
        for edge_index, (origin, direction) in enumerate(assignment_lines):
            selected = points[assigned == edge_index]
            counts.append(len(selected))
            if len(selected) < 5:
                # Keep this wall in the next assignment so its few points are not
                # reassigned to a neighbour. It has no fitted line or derived lengths.
                fitted.append(None)
                continue
            line = size._fit_wall_line(selected, (origin, origin + direction))
            fitted.append(line)
            assignment_lines[edge_index] = line

        if first_angles is None:
            first_angles = _angle_offsets(fitted, seeds)
    else:
        raise ValueError(f"Wall assignments did not stabilise in {max_rounds} rounds.")

    corners = []
    for edge_index, line in enumerate(fitted):
        prior_line = fitted[edge_index - 1]
        corners.append(
            None
            if line is None or prior_line is None
            else size._line_intersection(prior_line, line)
        )

    lengths = []
    for edge_index, start in enumerate(corners):
        end = corners[(edge_index + 1) % len(corners)]
        lengths.append(
            None if start is None or end is None else float(np.linalg.norm(end - start))
        )
    assert first_angles is not None
    return WallEdgeFit(
        tuple(lengths),
        _angle_offsets(fitted, seeds),
        first_angles,
        tuple(counts),
        tuple(fitted),
        round_index,
        int(unstable.sum()),
    )


def signed_edge_errors(
    lengths_mm: Sequence[float | None], caliper_mm: Sequence[float | None]
) -> tuple[float | None, ...]:
    if len(lengths_mm) != len(caliper_mm):
        raise ValueError(
            "Measured lengths and caliper readings must have the same edge indices."
        )
    return tuple(
        None if measured is None or truth is None else measured - truth
        for measured, truth in zip(lengths_mm, caliper_mm, strict=True)
    )


def measure_stage(
    points: np.ndarray,
    plane: tuple[float, float, float],
    seed: size.PrismMeasurement,
    calipers: Sequence[float | None],
    *,
    wall_profiles: bool = False,
) -> dict:
    heights = size.compute_height_above_board(points, plane)
    band = points[
        (heights > 0.3 * seed.height_mm) & (heights < 0.7 * seed.height_mm), :2
    ]
    fit = fit_wall_edges(
        band, seed.transformed_edges, max(5.0, 3.0 * seed.alignment_residual_mm)
    )
    signed = signed_edge_errors(fit.lengths_mm, calipers)
    measured = [abs(error) for error in signed if error is not None]
    result = asdict(fit)
    del result[
        "lines"
    ]  # Line vectors are internal; report the board-frame directions below.
    result["directions_deg"] = [
        None if line is None else _direction_degrees(line[1]) for line in fit.lines
    ]
    result["signed_errors_mm"] = signed
    result["measured_edges"] = len(measured)
    result["mean_abs_error_mm"] = statistics.mean(measured) if measured else None
    if wall_profiles:
        supported = [
            (edge_index, (line[0], line[0] + line[1]))
            for edge_index, line in enumerate(fit.lines)
            if line is not None
        ]
        if supported:
            profiles = size.measure_wall_profiles(
                points,
                [edge for _, edge in supported],
                plane,
                seed.height_mm,
                seed.alignment_residual_mm,
            )
            result["wall_spread_mm"] = {
                supported[profile.edge_index][0]: profile.thickness_median_mm
                for profile in profiles
            }
    return result


def observed_mesh(path: Path) -> tuple[np.ndarray, float, float]:
    vertices, faces, colors = mesh_cleanup.read_glb(path)
    if colors is None:
        raise ValueError(f"{path} is missing provenance colours.")
    labels = provenance.colors_to_labels(colors)
    observed = provenance.label_faces(faces, labels) == provenance.LABEL_OBSERVED
    points = vertices[np.unique(faces[observed])].astype(float)
    triangles = vertices[faces].astype(float)
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    return points, float(areas[observed].sum()), float(areas.sum())


def measure_capture(
    sample: str,
    input_root: Path,
    sweep_dir: Path,
    features: list[dict],
    *,
    filter_diagnostic: bool,
) -> dict:
    session, cad_name = SAMPLES[sample]
    artifacts = input_root / f"{sample}-all-vggt" / session / "artifacts"
    source = artifacts / "reconstruction"
    bundle = size.load_bundle(source, artifacts / "board_pose")
    cloud, _ = size.read_ply(source / "points.ply")
    pitch = load_board_pitch_mm()
    raw = size.apply_board_pitch_correction(
        size.apply_similarity_transform(
            cloud,
            bundle.alignment_scale,
            bundle.alignment_rotation,
            bundle.alignment_translation_mm,
        ),
        *pitch,
    )
    plane = size.fit_board_plane(raw)
    prism = size.prism_from_cad(
        size.read_ascii_stl(ROOT / "fixtures/reference_cad" / cad_name)
    )
    seed = size.measure_prism(raw, prism, plane)
    calipers = size.match_caliper_edge_truth(
        prism, [f for f in features if f["object"] == sample]
    )
    result = {
        "caliper_mm": calipers,
        "seed_directions_deg": [
            _direction_degrees(end - start) for start, end in seed.transformed_edges
        ],
        "raw": measure_stage(raw, plane, seed, calipers),
        "grids": {},
    }
    for voxel in VOXELS:
        folder = sweep_dir / sample / voxel
        fusion = folder / "fusion"
        extracted, extracted_faces = mesh_cleanup.read_mesh_ply(fusion / "mesh.ply")
        cleaned, _, _ = mesh_cleanup.read_glb(folder / "mesh_clean.glb")
        observed, observed_area, total_area = observed_mesh(
            folder / "uncertainty/mesh_uncertainty.glb"
        )
        stages = {
            "extracted": measure_stage(
                size.apply_board_pitch_correction(extracted, *pitch),
                plane,
                seed,
                calipers,
            ),
            "cleaned": measure_stage(
                size.apply_board_pitch_correction(cleaned, *pitch),
                plane,
                seed,
                calipers,
            ),
            "observed": measure_stage(
                size.apply_board_pitch_correction(observed, *pitch),
                plane,
                seed,
                calipers,
                wall_profiles=True,
            ),
            "observed_area_mm2": observed_area,
            "total_area_mm2": total_area,
        }
        result["grids"][voxel] = stages
        if not filter_diagnostic:
            continue
        # Use the exact first two cleanup operations to separate the component filter from
        # hole filling and export. The alternative meshes still go through normal provenance.
        manifold, faces, _ = mesh_cleanup.drop_non_manifold_faces(
            extracted, extracted_faces
        )
        filtered, _, report = mesh_cleanup.filter_connected_components(manifold, faces)
        stages["after_component_filter"] = measure_stage(
            size.apply_board_pitch_correction(filtered, *pitch), plane, seed, calipers
        )
        stages["kept_components"] = {"0.1": report["component_count_after"]}
        stages["filter_diagnostic"] = {}
        with TemporaryDirectory(prefix="r43-component-filter-") as directory:
            for threshold in (0.01, 0.0):
                variant = Path(directory) / str(threshold)
                clean = variant / "mesh_clean.glb"
                report = mesh_cleanup.cleanup_run(
                    fusion / "mesh.ply", clean, min_component_face_fraction=threshold
                )
                provenance.provenance_run(
                    clean,
                    source,
                    source / "metric_alignment.json",
                    variant / "provenance",
                    fusion_dir=fusion,
                )
                uncertainty.uncertainty_run(
                    variant / "provenance", variant / "uncertainty"
                )
                observed, _, _ = observed_mesh(
                    variant / "uncertainty/mesh_uncertainty.glb"
                )
                stages["kept_components"][str(threshold)] = report[
                    "component_count_after"
                ]
                stages["filter_diagnostic"][str(threshold)] = measure_stage(
                    size.apply_board_pitch_correction(observed, *pitch),
                    plane,
                    seed,
                    calipers,
                )
    return result


def print_fit(sample: str, stage: str, fit: dict) -> None:
    print(
        sample,
        stage,
        "signed error (mm)",
        fit["signed_errors_mm"],
        "first fit vs seed (deg)",
        fit["first_fit_angle_offsets_deg"],
        "refit vs seed (deg)",
        fit["angle_offsets_deg"],
        "unstable samples",
        fit["unstable_point_count"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, required=True, help="Shared runs/r32-runs folder"
    )
    parser.add_argument(
        "--sweep-dir", type=Path, required=True, help="Output from R-43's fusion sweep"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON report path"
    )
    parser.add_argument("--filter-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    features = json.loads((ROOT / "fixtures/caliper_g-06.json").read_text())["features"]
    report = {
        sample: measure_capture(
            sample,
            args.input_root,
            args.sweep_dir,
            features,
            filter_diagnostic=args.filter_diagnostic,
        )
        for sample in SAMPLES
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for sample, capture in report.items():
        print(sample, "seed directions (deg)", capture["seed_directions_deg"])
        print_fit(sample, "raw", capture["raw"])
        for voxel, stages in capture["grids"].items():
            for stage in ("extracted", "cleaned", "observed"):
                print_fit(sample, f"{voxel} {stage}", stages[stage])
    print(args.output)


if __name__ == "__main__":
    main()
