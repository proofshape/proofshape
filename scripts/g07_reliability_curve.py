"""G-07: does R-10's per-vertex sigma predict real, per-vertex measured error?

The proposal's verdict table scales the Pass/Fail tolerance by a factor `k` ("a planning prior
of 3; the M3 measurement sets it", D-032). Nothing has checked, until this story, whether
R-10's per-vertex sigma (`recon/uncertainty.py`) predicts real error at all -- `k` only means
anything if it does.

Runs the R-07 (TSDF fusion) -> R-08 (mesh cleanup) -> R-09 (provenance) -> R-10 (uncertainty)
chain for every board-reference specimen with independent caliper height truth (`s-04`, `s-05`
from G-06; `g02_01`..`g02_06` from G-03) -- idempotent, skipping stages whose output already
exists, and CPU-only (R-01's `points.ply`/`poses.npz`/`depth.npz` are already cached, so this
never re-runs VGGT). For each specimen, every near-top-face Observed vertex's own sigma
(`uncertainty/sigma_mm.npz`) is paired with that same vertex's height-above-board error against
the specimen's caliper height truth (`recon.cad_size_check.per_vertex_height_reliability`).

**A real bug found and fixed while building this (not from guessing, from the real numbers
coming out wrong):** the board plane must be the one R-07's own TSDF fusion already computed and
re-anchored (R-30's per-frame drift correction), written to `mesh/tsdf_fusion.json`'s
`board_surface` -- *not* re-fit from the served/provenance mesh's own vertices. The TSDF-fused
mesh only contains the segmented-out *part* (R-07 explicitly discards the board layer before
meshing), so it has few or no real board-level points to fit a plane from at all; a first version
of this script fit one anyway (`cad_size_check.fit_board_plane` on the mesh's own vertices) and
it silently produced a badly tilted/offset plane -- caught because the resulting per-vertex
height errors came out at 5-12 mm, roughly 10x G-04's own established ~0.86 mm floor for the same
feature, which is implausible as real noise and was traced to this. `tsdf_fusion.json`'s
`board_surface` fields (`slope_x`, `slope_y`, `z_at_origin_mm`) are the exact same
`z = a*x + b*y + c` convention `cad_size_check.compute_height_above_board` expects, read directly
rather than re-derived, guaranteeing consistency with how the mesh was actually built.

**A real, identified caveat, not fixed here:** the TSDF-fused mesh this story measures is placed
using the legacy, unweighted R-04 `align_camera_centres` path (`metric_alignment.json`, read by
`recon.tsdf_fusion.fuse_run`), not R-23's board-anchored `align_board_geometry` that G-04's own
`measure_g04_board_error.py` uses (and that R-38 just patched with per-frame outlier rejection).
These are two different alignment solves; R-23's own measured board_scale range (0.9971-0.9994
across the golden set) suggests a *global scale* mismatch between them is a sub-1% effect, but it
has not been measured for this mesh pipeline specifically (board_surface's own re-anchoring only
corrects tilt/offset, not scale). Per-specimen error is reported alongside the pooled bins
precisely so a systematic per-specimen offset (consistent with a residual scale effect) can be
told apart from vertex-level scatter (what sigma should actually predict) by eye, rather than
silently pooling the two together.

`k` is reported as a measured figure (the coverage-fraction ratio of error to sigma, plus the
pooled Pearson correlation between them) -- whether that figure means sigma reliably predicts
error, or doesn't, is this story's own judgement call once the real numbers are in, not a
threshold hardcoded here (same reason G-04's own aggregate is computed by script but interpreted
in its decision entry, not auto-verdicted).

    python scripts/g07_reliability_curve.py [--out PATH] [--bin-count N] [--coverage 0.95]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from recon.cad_size_check import per_vertex_height_reliability
from recon.mesh_cleanup import cleanup_run
from recon.provenance import provenance_run
from recon.tsdf_fusion import fuse_run
from recon.uncertainty import read_sigma_npz, uncertainty_run

SPECIMEN_IDS = (
    "s-04",
    "s-05",
    "g02_01",
    "g02_02",
    "g02_03",
    "g02_04",
    "g02_05",
    "g02_06",
)

RECON_RUN_ROOT = Path("fixtures/data/recon_runs")
CALIPER_G03_PATH = Path("fixtures/caliper_g-03.json")
CALIPER_G06_PATH = Path("fixtures/caliper_g-06.json")

DEFAULT_BIN_COUNT = 6
DEFAULT_COVERAGE = 0.95

# The real observed range for this run's 8 specimens: 2.08-3.91mm for 7 of them, 23.35mm for
# g02_02 -- the same IMG_4665 bad-frame issue D-073/R-38 found in R-23's board-anchored path,
# still present here since R-38 only patched align_board_geometry, not this legacy
# align_camera_centres path tsdf_fusion.fuse_run still reads from metric_alignment.json (see the
# module docstring). 10mm is a generous threshold well above the normal range, not reverse-fit
# tightly around g02_02's own value.
MAX_LEGACY_ALIGNMENT_RMS_RESIDUAL_MM = 10.0


def _caliper_height_truth_mm(specimen_id: str) -> float:
    """s-04/s-05's height truth is in G-06's caliper file; the g02 set's is in G-03's."""
    path = CALIPER_G06_PATH if specimen_id in ("s-04", "s-05") else CALIPER_G03_PATH
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    matches = [
        f
        for f in features
        if f.get("object") == specimen_id and f.get("kind") == "height"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one height caliper feature for {specimen_id} in {path}, "
            f"found {len(matches)}."
        )
    return matches[0]["mean_mm"]


def ensure_pipeline(specimen_id: str) -> Path:
    """Run R-07->R-10 for `specimen_id` if their outputs don't already exist, and return the
    mesh dir (`tsdf_fusion.json`, `provenance/`, `uncertainty/`). Idempotent and CPU-only: R-01's
    `points.ply`/`poses.npz`/`depth.npz` are already cached, so this never re-runs VGGT.
    """
    run_dir = RECON_RUN_ROOT / specimen_id
    alignment_json = run_dir / "metric_alignment.json"
    mesh_dir = run_dir / "mesh"
    mesh_ply = mesh_dir / "mesh.ply"
    mesh_clean_glb = mesh_dir / "mesh_clean.glb"
    provenance_dir = mesh_dir / "provenance"
    uncertainty_dir = mesh_dir / "uncertainty"

    if not mesh_ply.exists():
        fuse_run(run_dir, alignment_json, mesh_dir)
    if not mesh_clean_glb.exists():
        cleanup_run(mesh_ply, mesh_clean_glb)
    if not (provenance_dir / "mesh_provenance.glb").exists():
        provenance_run(mesh_clean_glb, run_dir, alignment_json, provenance_dir)
    if not (uncertainty_dir / "sigma_mm.npz").exists():
        uncertainty_run(provenance_dir, uncertainty_dir)
    return mesh_dir


def _read_board_plane(mesh_dir: Path) -> tuple[float, float, float]:
    """The board plane R-07's own TSDF fusion fit and re-anchored (R-30) while building this
    mesh -- see the module docstring for why this must be read, not re-fit from the mesh's own
    (board-free) vertices. `tsdf_fusion.json`'s `board_surface.slope_x/slope_y/z_at_origin_mm`
    are the exact `z = a*x + b*y + c` convention `compute_height_above_board` expects."""
    report = json.loads((mesh_dir / "tsdf_fusion.json").read_text(encoding="utf-8"))
    board_surface = report["board_surface"]
    return (
        board_surface["slope_x"],
        board_surface["slope_y"],
        board_surface["z_at_origin_mm"],
    )


def measure_specimen(specimen_id: str) -> dict[str, Any]:
    mesh_dir = ensure_pipeline(specimen_id)
    uncertainty_dir = mesh_dir / "uncertainty"
    served_glb = uncertainty_dir / "mesh_uncertainty.glb"
    sigma_mm = read_sigma_npz(uncertainty_dir / "sigma_mm.npz")["sigma_mm"]

    board_plane = _read_board_plane(mesh_dir)
    caliper_height_truth_mm = _caliper_height_truth_mm(specimen_id)

    alignment = json.loads(
        (RECON_RUN_ROOT / specimen_id / "metric_alignment.json").read_text(
            encoding="utf-8"
        )
    )
    legacy_alignment_rms_residual_mm = alignment["rms_residual_mm"]
    legacy_alignment_ok = (
        legacy_alignment_rms_residual_mm <= MAX_LEGACY_ALIGNMENT_RMS_RESIDUAL_MM
    )

    pairs = per_vertex_height_reliability(
        served_glb, sigma_mm, board_plane, caliper_height_truth_mm
    )
    errors_mm = [error for _sigma, error in pairs]
    return {
        "specimen_id": specimen_id,
        "caliper_height_truth_mm": caliper_height_truth_mm,
        "legacy_alignment_rms_residual_mm": legacy_alignment_rms_residual_mm,
        "legacy_alignment_ok": legacy_alignment_ok,
        "vertex_count": len(pairs),
        "median_abs_error_mm": statistics.median(errors_mm) if errors_mm else None,
        "mean_abs_error_mm": statistics.fmean(errors_mm) if errors_mm else None,
        "pairs": pairs,
    }


def bin_by_sigma(
    pairs: list[tuple[float, float]], bin_count: int = DEFAULT_BIN_COUNT
) -> list[dict[str, Any]]:
    """Quantile-bin (sigma_mm, abs_error_mm) pairs by sigma; report count, mean sigma, mean and
    p95 abs error per bin -- the "curve" this story's acceptance criteria ask for, as a table
    (see G-07's decision for why a table, not a plotted image)."""
    if not pairs:
        raise ValueError("No (sigma, error) pairs to bin.")
    sigma_mm = np.array([sigma for sigma, _error in pairs])
    error_mm = np.array([error for _sigma, error in pairs])
    order = np.argsort(sigma_mm)
    sigma_mm, error_mm = sigma_mm[order], error_mm[order]

    edges = np.quantile(sigma_mm, np.linspace(0.0, 1.0, bin_count + 1))
    bins: list[dict[str, Any]] = []
    for i in range(bin_count):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (sigma_mm >= lo) & (
            sigma_mm <= hi if i == bin_count - 1 else sigma_mm < hi
        )
        if in_bin.sum() == 0:
            continue
        bins.append(
            {
                "sigma_range_mm": [float(lo), float(hi)],
                "vertex_count": int(in_bin.sum()),
                "mean_sigma_mm": float(sigma_mm[in_bin].mean()),
                "mean_abs_error_mm": float(error_mm[in_bin].mean()),
                "p95_abs_error_mm": float(np.percentile(error_mm[in_bin], 95)),
            }
        )
    return bins


def measure_k(
    pairs: list[tuple[float, float]], coverage: float = DEFAULT_COVERAGE
) -> dict[str, Any]:
    """Report the measured figures a `k` choice would rest on: the coverage-fraction ratio of
    error to sigma, and the pooled Pearson correlation between them. Does not itself decide
    whether sigma reliably predicts error -- that judgement belongs in the decision entry, once
    the real numbers are in (same reason G-04's aggregate is computed here, interpreted there)."""
    sigma_mm = np.array([sigma for sigma, _error in pairs])
    error_mm = np.array([error for _sigma, error in pairs])
    positive_sigma = sigma_mm > 0
    if positive_sigma.sum() == 0:
        return {
            "k_at_coverage": None,
            "coverage": coverage,
            "pearson_correlation": None,
            "note": "No vertex has sigma_mm > 0; a k cannot be computed.",
        }
    ratios = error_mm[positive_sigma] / sigma_mm[positive_sigma]
    k_at_coverage = float(np.percentile(ratios, coverage * 100))
    correlation = (
        float(np.corrcoef(sigma_mm, error_mm)[0, 1]) if len(sigma_mm) > 1 else None
    )
    return {
        "k_at_coverage": k_at_coverage,
        "coverage": coverage,
        "pearson_correlation": correlation,
        "note": (
            f"Smallest k such that abs_error_mm <= k * sigma_mm holds for {coverage:.0%} of "
            "vertices with sigma_mm > 0, pooled across every specimen."
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fixtures/data/g07_reliability/g07-reliability-evidence.json"),
    )
    parser.add_argument("--bin-count", type=int, default=DEFAULT_BIN_COUNT)
    parser.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    return parser


def pool_pairs(
    results: list[dict[str, Any]], *, require_legacy_alignment_ok: bool = False
) -> list[tuple[float, float]]:
    """Pool every result's (sigma_mm, abs_error_mm) pairs. When `require_legacy_alignment_ok`,
    specimens whose legacy alignment's rms_residual_mm exceeded
    `MAX_LEGACY_ALIGNMENT_RMS_RESIDUAL_MM` are skipped entirely -- their part measurement is
    evidence the alignment failed for that one capture, not evidence about per-vertex
    reliability (same reasoning G-04's own `aggregate()` uses for its own specimen exclusion)."""
    return [
        pair
        for result in results
        if not require_legacy_alignment_ok or result["legacy_alignment_ok"]
        for pair in result["pairs"]
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)

    results = []
    for specimen_id in SPECIMEN_IDS:
        print(f"G-07: measuring {specimen_id}...", file=sys.stderr)
        results.append(measure_specimen(specimen_id))

    all_pairs = pool_pairs(results)
    bins = bin_by_sigma(all_pairs, bin_count=args.bin_count)
    k_estimate = measure_k(all_pairs, coverage=args.coverage)

    excluded_specimen_ids = [
        result["specimen_id"] for result in results if not result["legacy_alignment_ok"]
    ]
    clean_pairs = pool_pairs(results, require_legacy_alignment_ok=True)
    clean_bins = bin_by_sigma(clean_pairs, bin_count=args.bin_count)
    clean_k_estimate = measure_k(clean_pairs, coverage=args.coverage)

    evidence = {
        "story": "G-07",
        "specimens": [
            {k: v for k, v in result.items() if k != "pairs"}
            | {"pair_count": len(result["pairs"])}
            for result in results
        ],
        "pairs": all_pairs,
        "bins": bins,
        "k_estimate": k_estimate,
        "excluded_specimen_ids": excluded_specimen_ids,
        "clean_bins": clean_bins,
        "clean_k_estimate": clean_k_estimate,
        "clean_note": (
            f"bins/k_estimate recomputed excluding specimens whose legacy "
            f"metric_alignment.json rms_residual_mm exceeds "
            f"{MAX_LEGACY_ALIGNMENT_RMS_RESIDUAL_MM}mm ({excluded_specimen_ids}) -- see the "
            "module docstring for why this specific legacy-alignment path is still unpatched."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"G-07 RELIABILITY: {len(all_pairs)} vertex pairs across {len(results)} specimens"
    )
    for result in results:
        print(
            f"  {result['specimen_id']}: n={result['vertex_count']}, "
            f"median|error|={result['median_abs_error_mm']}, "
            f"mean|error|={result['mean_abs_error_mm']}"
        )
    print("  sigma bins:")
    for bin_ in bins:
        print(
            f"    sigma in {bin_['sigma_range_mm']}: n={bin_['vertex_count']}, "
            f"mean|error|={bin_['mean_abs_error_mm']:.3f} mm, "
            f"p95|error|={bin_['p95_abs_error_mm']:.3f} mm"
        )
    print(
        f"  k at {k_estimate['coverage']:.0%} coverage: {k_estimate['k_at_coverage']}, "
        f"pearson r: {k_estimate['pearson_correlation']}"
    )
    print(
        f"  excluding known-bad legacy alignment ({excluded_specimen_ids}): "
        f"{len(clean_pairs)} pairs, k at {clean_k_estimate['coverage']:.0%} coverage: "
        f"{clean_k_estimate['k_at_coverage']}, pearson r: "
        f"{clean_k_estimate['pearson_correlation']}"
    )
    for bin_ in clean_bins:
        print(
            f"    clean sigma in {bin_['sigma_range_mm']}: n={bin_['vertex_count']}, "
            f"mean|error|={bin_['mean_abs_error_mm']:.3f} mm, "
            f"p95|error|={bin_['p95_abs_error_mm']:.3f} mm"
        )
    print(f"Evidence written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
