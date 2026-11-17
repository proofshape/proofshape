"""G-07: does R-10's per-vertex sigma predict real, per-vertex measured error?

The proposal's verdict table scales the Pass/Fail tolerance by a factor `k` ("a planning prior
of 3; the M3 measurement sets it", D-032). Nothing has checked, until this story, whether
R-10's per-vertex sigma (`recon/uncertainty.py`) predicts real error at all -- `k` only means
anything if it does.

Runs the R-04 (metric alignment) -> R-07 (TSDF fusion) -> R-08 (mesh cleanup) -> R-09
(provenance) -> R-10 (uncertainty) chain fresh, every run, for every board-reference specimen
with independent caliper height truth (`s-04`, `s-05` from G-06; `g02_01`..`g02_06` from G-03).
For each specimen, every near-top-face Observed vertex's own sigma (`uncertainty/sigma_mm.npz`)
is paired with that same vertex's *signed* height-above-board error against the specimen's
caliper height truth (`recon.cad_size_check.per_vertex_height_reliability`).

**Two real bugs found and fixed while building this, from the numbers coming out wrong, not from
guessing:**

1. The board plane must be the one R-07's own TSDF fusion already computed and re-anchored
   (R-30's per-frame drift correction), written to `mesh/tsdf_fusion.json`'s `board_surface` --
   *not* re-fit from the served/provenance mesh's own vertices. The TSDF-fused mesh only contains
   the segmented-out *part* (R-07 discards the board layer before meshing), so it has few or no
   real board-level points to fit a plane from at all. `tsdf_fusion.json`'s `board_surface`
   fields (`slope_x`, `slope_y`, `z_at_origin_mm`) are the exact same `z = a*x + b*y + c`
   convention `cad_size_check.compute_height_above_board` expects, read directly (`_read_board_
   plane`) rather than re-derived.

2. `ensure_pipeline` used to skip regenerating `metric_alignment.json` and the mesh pipeline when
   their output already existed on disk -- but a *stale* cached `metric_alignment.json` (written
   before R-38's per-frame outlier rejection merged) is indistinguishable from a fresh one by
   that check alone. Caught on review (@TabeenRaoof, PR #367): `g02_02`'s cached file was R-23's
   board-anchored output from before R-38, carrying a 23.35 mm RMS residual from the uncorrected
   `IMG_4665` frame; the current code gives 1.863 mm with that frame rejected. This script now
   regenerates the alignment unconditionally every run, the same two calls
   `recon/service.py`'s live path makes (`metric_alignment.load_and_align` +
   `metric_alignment.write_alignment`), and rebuilds the entire mesh pipeline fresh from it
   (`ensure_pipeline` deletes `mesh/` before rebuilding) -- never R-01's own
   `points.ply`/`poses.npz`/`depth.npz`, the one genuinely expensive step (VGGT), which this
   script never touches.

**Offset vs. scatter, not one pooled number (also from review):** each specimen sits at a
near-constant *signed* offset from its own caliper height, with much smaller vertex-to-vertex
scatter around that offset. R-10's sigma is the spread of depth residuals *at a vertex* -- it
cannot see a whole-part systematic offset; that is `max(sigma, sigma_floor)`'s job (D-073),
specifically sigma_floor's. Pooling raw absolute error across specimens mixes the two: if the
specimens with the biggest offsets happen to have the lowest sigma (as they did before the
fixes above), the pooled correlation can come out negative even though sigma genuinely tracks
the scatter once the offset is removed. `specimen_offsets_mm`/`scatter_pairs` isolate that
scatter; `pool_pairs` keeps the raw, offset-contaminated pooling for comparison. Both are
reported; D-077 draws the conclusion from the scatter-only one, since that is what the story's
own acceptance criteria ("does sigma predict error") actually asks about.

`k` is reported as a measured figure (the coverage-fraction ratio of error to sigma, plus the
Pearson correlation), on both poolings -- whether either means sigma reliably predicts error is
this story's own judgement call once the real numbers are in, not a threshold hardcoded here
(same reason G-04's own aggregate is computed by script but interpreted in its decision entry,
not auto-verdicted).

    python scripts/g07_reliability_curve.py [--out PATH] [--bin-count N] [--coverage 0.95]
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from recon import metric_alignment
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
BOARD_POSE_ROOT = Path("fixtures/data/board_poses")
CALIPER_G03_PATH = Path("fixtures/caliper_g-03.json")
CALIPER_G06_PATH = Path("fixtures/caliper_g-06.json")

DEFAULT_BIN_COUNT = 6
DEFAULT_COVERAGE = 0.95


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
    """Rebuild R-04 (alignment) -> R-07 -> R-08 -> R-09 -> R-10 for `specimen_id` fresh, every
    run -- never trust a cached mesh or alignment that might predate the code that built them
    (see the module docstring: a stale `metric_alignment.json` silently carried a pre-R-38
    23.35 mm residual for `g02_02`). CPU-only: R-01's own `points.ply`/`poses.npz`/`depth.npz`
    (the one genuinely expensive step, VGGT) are never touched or regenerated here.
    """
    run_dir = RECON_RUN_ROOT / specimen_id
    board_pose_dir = BOARD_POSE_ROOT / specimen_id
    alignment_json = run_dir / "metric_alignment.json"
    mesh_dir = run_dir / "mesh"

    # The same two calls recon/service.py's live path makes (metric_alignment.load_and_align +
    # write_alignment) -- this script can never again drift from what production actually does.
    alignment = metric_alignment.load_and_align(run_dir, board_pose_dir)
    metric_alignment.write_alignment(alignment_json, alignment)

    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)

    mesh_ply = mesh_dir / "mesh.ply"
    mesh_clean_glb = mesh_dir / "mesh_clean.glb"
    provenance_dir = mesh_dir / "provenance"
    uncertainty_dir = mesh_dir / "uncertainty"

    fuse_run(run_dir, alignment_json, mesh_dir)
    cleanup_run(mesh_ply, mesh_clean_glb)
    provenance_run(mesh_clean_glb, run_dir, alignment_json, provenance_dir)
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

    pairs = per_vertex_height_reliability(
        served_glb, sigma_mm, board_plane, caliper_height_truth_mm
    )
    return {
        "specimen_id": specimen_id,
        "caliper_height_truth_mm": caliper_height_truth_mm,
        "pairs": pairs,
    }


def pool_pairs(results: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Pool every result's (sigma_mm, signed_error_mm) pairs into (sigma_mm, abs_error_mm) --
    mixes each specimen's own systematic offset with its vertex-level scatter. Kept for
    transparency/comparison with `scatter_pairs`, which removes the offset first; see the module
    docstring for why that mixing can flip the sign of a pooled correlation."""
    return [
        (sigma, abs(error)) for result in results for sigma, error in result["pairs"]
    ]


def specimen_offsets_mm(results: list[dict[str, Any]]) -> dict[str, float]:
    """Each specimen's own median *signed* error -- the systematic, whole-part offset R-10's
    sigma cannot see (sigma is the spread of depth residuals at a vertex; a whole-part offset is
    sigma_floor's job, D-073's `max(sigma, sigma_floor)`)."""
    return {
        result["specimen_id"]: statistics.median(
            error for _sigma, error in result["pairs"]
        )
        for result in results
        if result["pairs"]
    }


def specimen_summary(
    results: list[dict[str, Any]], offsets: dict[str, float]
) -> list[dict[str, Any]]:
    """Per-specimen offset/scatter/sigma table: median signed error (the offset), median
    absolute deviation from that offset (the scatter), median sigma."""
    summary: list[dict[str, Any]] = []
    for result in results:
        pairs = result["pairs"]
        offset = offsets.get(result["specimen_id"])
        sigmas = [sigma for sigma, _error in pairs]
        scatter = (
            [abs(error - offset) for _sigma, error in pairs]
            if offset is not None
            else []
        )
        summary.append(
            {
                "specimen_id": result["specimen_id"],
                "vertex_count": len(pairs),
                "median_signed_error_mm": offset,
                "mad_scatter_mm": statistics.median(scatter) if scatter else None,
                "median_sigma_mm": statistics.median(sigmas) if sigmas else None,
            }
        )
    return summary


def scatter_pairs(
    results: list[dict[str, Any]], offsets: dict[str, float]
) -> list[tuple[float, float]]:
    """(sigma_mm, abs(error_mm - that specimen's own offset)) -- isolates the per-vertex scatter
    sigma actually claims to predict, with each specimen's whole-part offset removed first."""
    return [
        (sigma, abs(error - offsets[result["specimen_id"]]))
        for result in results
        if result["specimen_id"] in offsets
        for sigma, error in result["pairs"]
    ]


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


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)

    results = []
    for specimen_id in SPECIMEN_IDS:
        print(f"G-07: measuring {specimen_id}...", file=sys.stderr)
        result = measure_specimen(specimen_id)
        results.append(result)
        print(f"  {specimen_id}: n={len(result['pairs'])}", file=sys.stderr)

    offsets = specimen_offsets_mm(results)
    summary = specimen_summary(results, offsets)

    pooled_pairs = pool_pairs(results)
    pooled_bins = bin_by_sigma(pooled_pairs, bin_count=args.bin_count)
    pooled_k = measure_k(pooled_pairs, coverage=args.coverage)

    scattered_pairs = scatter_pairs(results, offsets)
    scatter_bins = bin_by_sigma(scattered_pairs, bin_count=args.bin_count)
    scatter_k = measure_k(scattered_pairs, coverage=args.coverage)

    evidence = {
        "story": "G-07",
        "specimen_summary": summary,
        "pooled_pairs": pooled_pairs,
        "pooled_bins": pooled_bins,
        "pooled_k_estimate": pooled_k,
        "pooled_note": (
            "offset-contaminated: each specimen's own systematic offset is mixed in with its "
            "vertex-level scatter. Kept for comparison; see scatter_* for what sigma actually "
            "claims to predict."
        ),
        "scatter_pairs": scattered_pairs,
        "scatter_bins": scatter_bins,
        "scatter_k_estimate": scatter_k,
        "scatter_note": (
            "each specimen's own median signed error (its offset) is subtracted before "
            "pooling -- isolates the per-vertex scatter sigma actually claims to predict."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"G-07 RELIABILITY: {len(pooled_pairs)} vertex pairs across {len(results)} specimens"
    )
    print("  specimen | n | offset (signed) | scatter (MAD) | median sigma")
    for row in summary:
        print(
            f"    {row['specimen_id']}: n={row['vertex_count']}, "
            f"offset={row['median_signed_error_mm']}, "
            f"scatter={row['mad_scatter_mm']}, "
            f"median_sigma={row['median_sigma_mm']}"
        )
    print(
        f"  pooled (offset+scatter mixed): k at {pooled_k['coverage']:.0%} coverage: "
        f"{pooled_k['k_at_coverage']}, pearson r: {pooled_k['pearson_correlation']}"
    )
    for bin_ in pooled_bins:
        print(
            f"    pooled sigma in {bin_['sigma_range_mm']}: n={bin_['vertex_count']}, "
            f"mean|error|={bin_['mean_abs_error_mm']:.3f} mm, "
            f"p95|error|={bin_['p95_abs_error_mm']:.3f} mm"
        )
    print(
        f"  scatter-only (offset removed): k at {scatter_k['coverage']:.0%} coverage: "
        f"{scatter_k['k_at_coverage']}, pearson r: {scatter_k['pearson_correlation']}"
    )
    for bin_ in scatter_bins:
        print(
            f"    scatter sigma in {bin_['sigma_range_mm']}: n={bin_['vertex_count']}, "
            f"mean|error|={bin_['mean_abs_error_mm']:.3f} mm, "
            f"p95|error|={bin_['p95_abs_error_mm']:.3f} mm"
        )
    print(f"Evidence written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
