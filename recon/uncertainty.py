"""R-10: per-vertex uncertainty (sigma), the last piece of the provenance system.

Per `docs/glossary.md` and `docs/decisions.md`'s D-006: sigma per vertex is the spread of depth
residuals across the frames that observe it -- not the spread across generative completion
samples. "Spread" is standard deviation here, the conventional meaning of sigma; residual is
`measured_depth_mm - z_cam_mm` per observation, exactly what R-09's `observations.npz` persists
for this purpose. A vertex needs 2+ observations for sigma to be defined; fewer gets NaN, not a
fabricated zero.

This story depends only on R-09's *persisted output* (`mesh_provenance.glb` for vertices/faces/
current labels, `observations.npz` for the residual data) -- no camera poses, no depth maps, no
re-running any projection.

The downgrade threshold is NOT sigma_floor: sigma_floor is "measured per reference tier in the
October calibration study" (`docs/glossary.md`) -- a number that does not exist yet, the same
situation D-032 already settled for I-01's tolerance-table fields. This story's own acceptance
criteria say "a threshold," not "sigma_floor," so DEFAULT_SIGMA_DOWNGRADE_THRESHOLD_MM is its own
configurable, unmeasured assumption, same pattern as every prior story's defaults.

"Downgraded" means relabelled Unobserved -- the pipeline only has two provenance states
(`docs/glossary.md`'s Provenance entry). An Unobserved vertex can't be downgraded further, and a
vertex with undefined (NaN) sigma is left alone: absence of evidence of high spread isn't
evidence of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from recon.mesh_cleanup import read_glb, write_glb
from recon.provenance import (
    LABEL_OBSERVED,
    LABEL_UNOBSERVED,
    colors_to_labels,
    labels_to_colors,
    read_observations_npz,
)

DEFAULT_SIGMA_DOWNGRADE_THRESHOLD_MM = 3.0


def compute_vertex_sigma(
    vertex_count: int, observations: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Per-vertex sigma_mm (std of depth residuals, NaN where fewer than 2 observations) and
    residual_count (how many observations contributed)."""
    sigma_mm = np.full(vertex_count, np.nan, dtype=np.float64)
    residual_count = np.zeros(vertex_count, dtype=np.int64)

    vertex_index = observations["vertex_index"]
    residuals = observations["measured_depth_mm"] - observations["z_cam_mm"]

    residuals_by_vertex: dict[int, list[float]] = {}
    for index, residual in zip(vertex_index.tolist(), residuals.tolist(), strict=True):
        residuals_by_vertex.setdefault(index, []).append(residual)

    for index, values in residuals_by_vertex.items():
        residual_count[index] = len(values)
        if len(values) >= 2:
            sigma_mm[index] = float(np.std(np.asarray(values), ddof=0))

    return {"sigma_mm": sigma_mm, "residual_count": residual_count}


def downgrade_high_sigma_vertices(
    labels: np.ndarray,
    sigma_mm: np.ndarray,
    threshold_mm: float = DEFAULT_SIGMA_DOWNGRADE_THRESHOLD_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel Observed vertices whose defined sigma exceeds threshold_mm as Unobserved.

    Vertices already Unobserved, or whose sigma is undefined (NaN, fewer than 2 observations),
    are left untouched -- you can't downgrade what's already at the floor, and you can't penalise
    a spread you were never able to measure. Returns (new_labels, downgraded_mask); the input
    array is not mutated.
    """
    if not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise ValueError("threshold_mm must be a positive, finite number.")

    is_observed = labels == LABEL_OBSERVED
    has_defined_sigma = ~np.isnan(sigma_mm)
    exceeds_threshold = has_defined_sigma & (sigma_mm > threshold_mm)
    downgraded_mask = is_observed & exceeds_threshold

    new_labels = labels.copy()
    new_labels[downgraded_mask] = LABEL_UNOBSERVED
    return new_labels, downgraded_mask


def compute_sigma_distribution(
    sigma_mm: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Summary stats of sigma_mm over Observed vertices with a defined sigma.

    This is what R-19 will call against real golden-object data once it's available; here it's
    built and tested generically against synthetic sigma values.
    """
    selected = sigma_mm[(labels == LABEL_OBSERVED) & ~np.isnan(sigma_mm)]
    if len(selected) == 0:
        return {
            "count": 0,
            "min_mm": None,
            "max_mm": None,
            "mean_mm": None,
            "median_mm": None,
            "p90_mm": None,
        }
    return {
        "count": len(selected),
        "min_mm": float(np.min(selected)),
        "max_mm": float(np.max(selected)),
        "mean_mm": float(np.mean(selected)),
        "median_mm": float(np.median(selected)),
        "p90_mm": float(np.percentile(selected, 90)),
    }


def write_sigma_npz(
    path: Path, sigma_mm: np.ndarray, residual_count: np.ndarray
) -> None:
    np.savez_compressed(path, sigma_mm=sigma_mm, residual_count=residual_count)


def read_sigma_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {"sigma_mm": data["sigma_mm"], "residual_count": data["residual_count"]}


def write_uncertainty_report(
    path: Path,
    provenance_dir: Path,
    vertex_count: int,
    downgraded_vertex_count: int,
    threshold_mm: float,
    distribution: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "story": "R-10",
        "source_provenance_dir": str(provenance_dir),
        "vertex_count": vertex_count,
        "downgraded_vertex_count": downgraded_vertex_count,
        "sigma_downgrade_threshold_mm": threshold_mm,
        "sigma_distribution_mm": distribution,
    }
    path = Path(path)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def uncertainty_run(
    provenance_dir: Path,
    out_dir: Path,
    sigma_downgrade_threshold_mm: float = DEFAULT_SIGMA_DOWNGRADE_THRESHOLD_MM,
) -> dict[str, Any]:
    """Load R-09's provenance output, compute per-vertex sigma, downgrade, and write the result."""
    provenance_dir = Path(provenance_dir)
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already has files in it. Pick a new --out-dir or delete it deliberately; "
            "overwriting silently would let outputs from two runs get mixed."
        )

    vertices, faces, colors = read_glb(provenance_dir / "mesh_provenance.glb")
    if colors is None:
        raise ValueError(
            f"{provenance_dir / 'mesh_provenance.glb'} has no vertex colors to decode"
        )
    labels = colors_to_labels(colors)
    observations = read_observations_npz(provenance_dir / "observations.npz")

    sigma_result = compute_vertex_sigma(len(vertices), observations)
    new_labels, downgraded_mask = downgrade_high_sigma_vertices(
        labels, sigma_result["sigma_mm"], sigma_downgrade_threshold_mm
    )
    distribution = compute_sigma_distribution(sigma_result["sigma_mm"], new_labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_glb(
        out_dir / "mesh_uncertainty.glb", vertices, faces, labels_to_colors(new_labels)
    )
    write_sigma_npz(
        out_dir / "sigma_mm.npz",
        sigma_result["sigma_mm"],
        sigma_result["residual_count"],
    )
    return write_uncertainty_report(
        out_dir / "uncertainty.json",
        provenance_dir,
        vertex_count=len(vertices),
        downgraded_vertex_count=int(np.count_nonzero(downgraded_mask)),
        threshold_mm=sigma_downgrade_threshold_mm,
        distribution=distribution,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-vertex uncertainty and downgrade high-sigma vertices (R-10)."
    )
    parser.add_argument("provenance_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--sigma-downgrade-threshold-mm",
        type=float,
        default=DEFAULT_SIGMA_DOWNGRADE_THRESHOLD_MM,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    out_dir = args.out_dir or args.provenance_dir / "uncertainty"
    try:
        report = uncertainty_run(
            args.provenance_dir,
            out_dir,
            sigma_downgrade_threshold_mm=args.sigma_downgrade_threshold_mm,
        )
    except FileExistsError as exc:
        print(f"R-10 UNCERTAINTY: FAIL - {exc}", file=sys.stderr)
        return 1

    print(
        f"R-10 UNCERTAINTY: PASS - {report['downgraded_vertex_count']}/{report['vertex_count']} "
        f"vertices downgraded (threshold {report['sigma_downgrade_threshold_mm']} mm)"
    )
    print(f"Outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
