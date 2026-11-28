"""G-15: one row per part, a capture's photographs against the same part captured through the app.

    python scripts/compare_app_captures.py PHOTOS_ROOT APP_ROOT [--verdicts DIR] [--json OUT]

PHOTOS_ROOT and APP_ROOT each hold one folder per part, named the same in both, and each folder is
what `measure_part_accuracy.py run` wrote (`measure.npz`, `measure.json`). The figures come from
that script's own `report_parts`, so nothing here re-measures a part. `--verdicts` is a folder of
`<part>.json` files, each the body `GET /v1/sessions/{id}/verdict` returned for the app session.

The one figure this script adds is whether the two captures put the CAD on the part the same way.
I-15's `poses_are_same` compares two CAD poses in one model's frame. Two captures of the same part
sit at different places on the board, so their poses can't be compared directly: a part moved
40 mm gives poses 40 mm apart that are both right. So each model's Observed vertices are carried
into its own CAD's frame (the inverse of its `cad_to_model`). If both poses sit the CAD on the
part the same way, the two point sets then overlap. The app's points are ICP-aligned to the
photographs' points, started from every 10-degree turn about the vertical, and the best fit's
transform is how differently the two poses sit the CAD on the part. Its size is I-15's `surface_gap_mm` between the CAD and the CAD
moved by that transform, judged against I-15's own margin (`SAME_POSE_MARGIN_MM`).

What this does not tell you: which of two different poses is the right one. A gap says only that
they differ. If no start finds at least three point pairs within `ICP_MAX_DISTANCE_MM`, the two
captures don't overlap and there is nothing to measure a gap from, so the comparison is
indeterminate (no gap, no "same"), never a gap of zero.

Only points within `NEAR_CAD_MM` of each capture's own posed CAD are used, so that surface that
isn't the part (board skin, mostly) doesn't drag the alignment; that cut is made
against each capture's own pose, so it slightly favours the poses agreeing, never disagreeing.
`NEAR_CAD_MM` and `ICP_MAX_DISTANCE_MM` are reporting choices for this comparison, not measured
thresholds, and no verdict or pipeline code reads them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# R-31: the macOS thread guard has to run before Open3D is imported (see measure_part_accuracy).
from recon import runtime_env as _runtime_env  # noqa: F401  # isort: skip

import numpy as np  # isort: skip
import open3d as o3d  # isort: skip

from inspection.pose_judgement import SAME_POSE_MARGIN_MM, surface_gap_mm

from scripts.measure_part_accuracy import MEASURE_NPZ, report_parts

NEAR_CAD_MM = 5.0
ICP_MAX_DISTANCE_MM = 3.0
YAW_START_STEP_DEG = 10


# A rigid fit needs three point pairs; with fewer the alignment has found nothing.
MIN_CORRESPONDENCES = 3


@dataclass(frozen=True)
class PoseComparison:
    """`gap_mm` is None when the captures don't overlap (see the module docstring)."""

    gap_mm: float | None
    icp_fitness: float

    @property
    def same(self) -> bool | None:
        if self.gap_mm is None:
            return None
        return self.gap_mm <= SAME_POSE_MARGIN_MM


def observed_in_cad_frame(
    vertices: np.ndarray, observed: np.ndarray, cad_to_model: np.ndarray
) -> np.ndarray:
    """The Observed vertices, carried from the model's frame into the CAD's own frame."""
    model_to_cad = np.linalg.inv(np.asarray(cad_to_model, dtype=np.float64))
    points = np.asarray(vertices, dtype=np.float64)[np.asarray(observed, dtype=bool)]
    return points @ model_to_cad[:3, :3].T + model_to_cad[:3, 3]


def near_cad(
    points: np.ndarray, cad_vertices: np.ndarray, cad_faces: np.ndarray, max_mm: float
) -> np.ndarray:
    """The points within max_mm of the CAD surface (both in the CAD's frame)."""
    if len(points) == 0:
        return points
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(cad_vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(cad_faces, dtype=np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    distance = scene.compute_distance(
        o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    ).numpy()
    return points[distance <= max_mm]


def compare_poses(
    photos: dict[str, np.ndarray],
    app: dict[str, np.ndarray],
    *,
    near_mm: float = NEAR_CAD_MM,
    icp_max_mm: float = ICP_MAX_DISTANCE_MM,
) -> PoseComparison:
    """How differently two captures' poses sit the same CAD on the part (see module docstring).

    Each argument is a capture's `measure.npz` contents: vertices, observed, cad_to_model,
    cad_vertices, cad_faces. The CAD is taken from the photographs' file.
    """
    cad_vertices, cad_faces = photos["cad_vertices"], photos["cad_faces"]
    target = near_cad(
        observed_in_cad_frame(
            photos["vertices"], photos["observed"], photos["cad_to_model"]
        ),
        cad_vertices,
        cad_faces,
        near_mm,
    )
    source = near_cad(
        observed_in_cad_frame(app["vertices"], app["observed"], app["cad_to_model"]),
        cad_vertices,
        cad_faces,
        near_mm,
    )
    if len(target) < 3 or len(source) < 3:
        raise ValueError(
            "fewer than 3 Observed points near the CAD in one of the captures"
        )
    source_cloud, target_cloud = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source)
    target_cloud.points = o3d.utility.Vector3dVector(target)
    # Started from the identity alone, ICP can't turn far enough to catch a pose turned by more
    # than a few degrees: on G-15's g02_05 it stopped at a 1.6 mm gap while the maps showed the
    # photographs' pose turned about 13 degrees. So it starts from every YAW_START_STEP_DEG turn
    # about the CAD's centre, and the best fit wins (most overlap, then least residual).
    centre = 0.5 * (np.min(cad_vertices, axis=0) + np.max(cad_vertices, axis=0))
    best = None
    for yaw_deg in range(-180, 180, YAW_START_STEP_DEG):
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            icp_max_mm,
            _turn_about_z(yaw_deg, centre),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )
        key = (round(float(result.fitness), 3), -float(result.inlier_rmse))
        if best is None or key > best[0]:
            best = (key, result)
    assert best is not None
    result = best[1]
    if len(result.correspondence_set) < MIN_CORRESPONDENCES:
        # Every start found (almost) nothing: the first start would win by default, and its
        # transform is just the start, so a gap read from it would be made up.
        return PoseComparison(gap_mm=None, icp_fitness=float(result.fitness))
    gap = surface_gap_mm(
        cad_vertices, cad_faces, np.eye(4), np.asarray(result.transformation)
    )
    return PoseComparison(gap_mm=float(gap), icp_fitness=float(result.fitness))


def _turn_about_z(yaw_deg: float, centre: np.ndarray) -> np.ndarray:
    angle = np.radians(yaw_deg)
    turn = np.eye(4)
    turn[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    turn[:3, 3] = centre - turn[:3, :3] @ centre
    return turn


def service_verdict(verdicts_dir: Path | None, part: str) -> tuple[str, list[str]]:
    """The verdict and reason codes the service returned for the part's app session, if saved."""
    if verdicts_dir is None or not (verdicts_dir / f"{part}.json").exists():
        return "—", []
    body = json.loads((verdicts_dir / f"{part}.json").read_text())
    return str(body.get("verdict", "?")), [
        str(code) for code in body.get("reason_codes") or []
    ]


def compare_parts(
    photos_root: Path,
    app_root: Path,
    parts: Sequence[str],
    verdicts_dir: Path | None = None,
) -> list[dict[str, Any]]:
    photo_rows = {
        row["part"]: row for row in report_parts([photos_root / p for p in parts])
    }
    app_rows = {row["part"]: row for row in report_parts([app_root / p for p in parts])}
    rows = []
    for part in parts:
        with (
            np.load(photos_root / part / MEASURE_NPZ) as photos_npz,
            np.load(app_root / part / MEASURE_NPZ) as app_npz,
        ):
            pose = compare_poses(dict(photos_npz), dict(app_npz))
        verdict, codes = service_verdict(verdicts_dir, part)
        rows.append(
            {
                "part": part,
                "photos": photo_rows[part],
                "app": app_rows[part],
                "pose_gap_mm": pose.gap_mm,
                "pose_same": pose.same,
                "pose_icp_fitness": pose.icp_fitness,
                "service_verdict": verdict,
                "service_reason_codes": codes,
            }
        )
    return rows


def _cell(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pair(photos: float | None, app: float | None, digits: int = 2) -> str:
    """report_parts gives None for a figure with nothing measured (an unmeasured surface)."""
    return f"{_cell(photos, digits)} / {_cell(app, digits)}"


def _p95(spread: dict[str, Any] | None) -> float | None:
    return None if spread is None else spread["abs_p95_mm"]


def _pose_cell(gap_mm: float | None, same: bool | None) -> str:
    if gap_mm is None:
        return "indeterminate (no overlap)"
    return f"{gap_mm:.2f} ({'same' if same else 'different'})"


def _yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    """One row per part, every cell "photographs / app"."""
    lines = [
        (
            "| Part | Walls, 95% within (mm) | Top face, 95% within | Worst reading | Stray share "
            "| Unmeasured share | First Pass (mm) | Ambiguous | Pose gap (mm) | Service verdict (app) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        p, a = row["photos"], row["app"]
        ps, as_ = p["summary"], a["summary"]
        codes = ", ".join(f"`{code}`" for code in row["service_reason_codes"])
        lines.append(
            "| "
            + " | ".join(
                [
                    row["part"],
                    _pair(_p95(ps["wall"]), _p95(as_["wall"])),
                    _pair(_p95(ps["top"]), _p95(as_["top"])),
                    _pair(ps["worst_reading_mm"], as_["worst_reading_mm"]),
                    _pair(ps["stray_observed_share"], as_["stray_observed_share"]),
                    _pair(
                        ps["unmeasured_observed_share"],
                        as_["unmeasured_observed_share"],
                    ),
                    _pair(
                        p["sweep"]["first_pass_mm"],
                        a["sweep"]["first_pass_mm"],
                        digits=1,
                    ),
                    (
                        f"{_yes_no(p['registration']['orientation_ambiguous'])} / "
                        f"{_yes_no(a['registration']['orientation_ambiguous'])}"
                    ),
                    _pose_cell(row["pose_gap_mm"], row["pose_same"]),
                    f"{row['service_verdict']} {codes}".strip(),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("photos_root", type=Path)
    parser.add_argument("app_root", type=Path)
    parser.add_argument(
        "--parts", nargs="+", default=None, help="default: every folder in both"
    )
    parser.add_argument("--verdicts", type=Path, default=None)
    parser.add_argument(
        "--json", type=Path, default=None, help="also write the rows here"
    )
    args = parser.parse_args(argv)
    parts = args.parts or sorted(
        d.name
        for d in args.photos_root.iterdir()
        if (d / MEASURE_NPZ).exists()
        and (args.app_root / d.name / MEASURE_NPZ).exists()
    )
    rows = compare_parts(args.photos_root, args.app_root, parts, args.verdicts)
    print(markdown_table(rows))
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
