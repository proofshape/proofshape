"""G-04: measure per-feature error for the board reference against caliper truth, and derive
sigma_floor_mm from it.

Runs the full chain -- R-01 (reconstruction), R-02 (board pose), R-04/R-23 (board-anchored
metric alignment), then the extended R-22 `cad_size_check.build_report` -- on each of the six
G-02 specimens (`g02_01`..`g02_06`), against G-03's caliper truth for each specimen and G-06's
measured (anisotropic) board pitch. Idempotent: an existing `recon_run_dir`/`board_pose_dir` is
reused rather than re-run, since R-01 (VGGT) is the expensive step.

**Edges are excluded from the aggregate entirely (not just the symmetric/near-regular
specimens), per review on PR #348 (@TabeenRaoof):** `measure_prism`'s edge length is its 2nd-
to-98th-percentile point extent, which reads a perfect, noiseless, reconstruction-free prism
about 4% short of its true corner-to-corner length by construction -- a caliper reads corner to
corner, so this alone biases every edge negative before any real error. Worse, `_fit_rigid_2d`'s
exhaustive discrete pose search was shown, on the same photos and the same code run twice, to
land in two different poses for g02_03's triangle (board scale agreed to 4 decimals; edge
numbers did not) -- not a continuous noise effect, a qualitatively wrong fit, confirmed visually
against the point cloud for g02_01 and g02_03. Both are real, pre-existing bugs in code this
story only newly depends on; fixing them (corner-to-corner edge measurement via fitted-wall-line
intersection, and a pose search that cannot land wrong on these shapes) is real work, left open,
not attempted here. `match_caliper_edge_truth`'s own ambiguous-length-class exclusion
(``conflict_atol_mm``) still runs and is still correct as far as it goes, but it is not why these
specimens' edges are actually unusable -- the fit itself is why, for every specimen, not only the
symmetric ones.

Height has no such dependency: it comes from the top-face point-density peak, computed before
`_fit_rigid_2d` ever runs, and is the only caliper-matched feature this aggregate trusts.

A whole specimen is excluded if its own board-anchored alignment (R-23) didn't converge --
`board_scale` more than `MAX_BOARD_SCALE_DEVIATION` from 1.0, found live on g02_02 (board_scale
0.83, rms_residual_mm 21x every other specimen's). Review on PR #348 traced this to one frame
(`IMG_4665`, median per-corner residual 124.8 mm against every other frame's single digits):
R-23's unweighted least-squares has no per-frame outlier rejection, and `metric_alignment.main`
prints PASS with no residual threshold at all, so this failure would reach the live service
silently today. That specimen's part measurement is evidence the alignment failed for its
capture, not evidence about the part -- a robust (per-frame-rejecting) R-23 solve plus an
alignment-residual gate into Unverifiable is the real fix, left open, not attempted here.

    python scripts/measure_g04_board_error.py [--device mps] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

# Running this file directly sets sys.path[0] to scripts/, while importing it as a module sets
# sys.path[0] to the repository root. Put the checked-out repository first explicitly so both
# entry styles import the exact same recon package instead of a stale editable/site-packages copy.
REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_text = str(REPO_ROOT)
if not sys.path or sys.path[0] != repo_root_text:
    sys.path.insert(0, repo_root_text)

from recon import board_pose, metric_alignment, vggt_runner
from recon.cad_size_check import build_report
from recon.vggt_runner import DEFAULT_MIN_CONFIDENCE

SPECIMEN_IDS = ("g02_01", "g02_02", "g02_03", "g02_04", "g02_05", "g02_06")

# R-23's own real-data range (its story: board scale 0.9971-0.9994 across the s-01..s-05 golden
# set) is sub-1% deviation from 1.0. 5% is a generous multiple of that normal range -- a
# specimen whose board-anchored alignment misses by more than this did not converge, and its
# part measurement is not evidence about the part; it is evidence the alignment failed for that
# one capture (not reverse-fit to any specific specimen found here).
MAX_BOARD_SCALE_DEVIATION = 0.05


def _nearest_rank_index(sample_count: int, percentile: float) -> int:
    """Zero-based nearest-rank index for a non-empty sample.

    Nearest-rank percentile P is ceil(P * N) in one-based indexing, so subtract 1 for Python's
    zero-based indexing. Keeping this in one helper prevents the height and in-plane aggregates
    from silently using different percentile conventions.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return max(0, min(sample_count - 1, math.ceil(percentile * sample_count) - 1))


CAPTURE_ROOT = Path("fixtures/data/specimen_capture")
RECON_RUN_ROOT = Path("fixtures/data/recon_runs")
BOARD_POSE_ROOT = Path("fixtures/data/board_poses")
REFERENCE_CAD_DIR = Path("fixtures/reference_cad")
CALIPER_G03_PATH = Path("fixtures/caliper_g-03.json")
CALIPER_G06_PATH = Path("fixtures/caliper_g-06.json")


def load_board_pitch_mm(
    caliper_g06_path: Path = CALIPER_G06_PATH,
) -> tuple[float, float]:
    """G-06's measured board pitch (across, down), from its own `board_span` caliper features
    (`square_count=8`/`6`) -- read from the file, not hardcoded, so a re-measurement is picked
    up automatically."""
    features = json.loads(caliper_g06_path.read_text(encoding="utf-8"))["features"]
    spans = {
        f["square_count"]: f["mean_mm"]
        for f in features
        if f.get("object") == "board"
        and f.get("kind") == "board_span"
        and f.get("square_count") in (6, 8)
    }
    if set(spans) != {6, 8}:
        raise ValueError(
            f"Expected board_span features for square_count 6 and 8 in {caliper_g06_path}, "
            f"found {sorted(spans)}."
        )
    return (spans[8], spans[6])


def _reference_cad_path(specimen_id: str) -> Path:
    matches = sorted(REFERENCE_CAD_DIR.glob(f"{specimen_id}_*.stl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one reference CAD file for {specimen_id} under "
            f"{REFERENCE_CAD_DIR}, found {len(matches)}: {matches}"
        )
    return matches[0]


def ensure_reconstruction(specimen_id: str, device: str) -> tuple[Path, Path]:
    """Run R-01/R-02/R-04 for `specimen_id` if their outputs don't already exist, and return
    (recon_run_dir, board_pose_dir). Re-running VGGT is the expensive step this skips when
    possible."""
    capture_dir = CAPTURE_ROOT / specimen_id
    recon_run_dir = RECON_RUN_ROOT / specimen_id
    board_pose_dir = BOARD_POSE_ROOT / specimen_id

    if not (recon_run_dir / "points.ply").exists():
        exit_code = vggt_runner.main(
            [str(capture_dir), "--out-dir", str(recon_run_dir), "--device", device]
        )
        if exit_code != 0:
            raise RuntimeError(f"R-01 failed for {specimen_id} (exit {exit_code})")
    if not (board_pose_dir / "board_poses.npz").exists():
        exit_code = board_pose.main(
            [str(capture_dir), "--out-dir", str(board_pose_dir)]
        )
        if exit_code != 0:
            raise RuntimeError(f"R-02 failed for {specimen_id} (exit {exit_code})")
    if not (recon_run_dir / "metric_alignment.json").exists():
        exit_code = metric_alignment.main([str(recon_run_dir), str(board_pose_dir)])
        if exit_code != 0:
            raise RuntimeError(f"R-04 failed for {specimen_id} (exit {exit_code})")
    return recon_run_dir, board_pose_dir


def measure_specimen(
    specimen_id: str,
    device: str,
    caliper_features_by_specimen: dict[str, list[dict[str, Any]]],
    board_pitch_mm: tuple[float, float],
) -> dict[str, Any]:
    recon_run_dir, board_pose_dir = ensure_reconstruction(specimen_id, device)
    report = build_report(
        recon_run_dir,
        board_pose_dir,
        cad_path=_reference_cad_path(specimen_id),
        min_confidence=DEFAULT_MIN_CONFIDENCE,
        caliper_features=caliper_features_by_specimen[specimen_id],
        board_pitch_mm=board_pitch_mm,
    )
    board_scale = report["board_scale"]["scale"]
    board_alignment_ok = abs(board_scale - 1.0) <= MAX_BOARD_SCALE_DEVIATION
    exclusion_reason = (
        None
        if board_alignment_ok
        else (
            f"R-23's board-anchored alignment did not converge for this capture: "
            f"board_scale={board_scale:.4f} (every other specimen in this run is within "
            f"{MAX_BOARD_SCALE_DEVIATION * 100:.0f}% of 1.0, matching R-23's own documented "
            f"0.9971-0.9994 range); rms_residual_mm="
            f"{report['board_scale']['rms_residual_mm']:.3f} vs 2-4 mm for every other "
            "specimen. Excluded from the aggregate: this is evidence the alignment failed for "
            "this one capture, not evidence about the part."
        )
    )
    return {
        "specimen_id": specimen_id,
        "recon_run_dir": str(recon_run_dir),
        "board_pose_dir": str(board_pose_dir),
        "board_scale": board_scale,
        "board_scale_rms_residual_mm": report["board_scale"]["rms_residual_mm"],
        "board_alignment_ok": board_alignment_ok,
        "exclusion_reason": exclusion_reason,
        # R-37 real-data evidence: these let two runs prove the square/triangle pose and raw
        # corner-to-corner edge measurements are stable even when a caliper length class is
        # ambiguous and therefore absent from prism.caliper.edges.
        "prism_rotation_deg": report["prism"]["rotation_deg"],
        "prism_alignment_residual_mm": report["prism"]["alignment_residual_mm"],
        "prism_edges": report["prism"]["edges"],
        "caliper": report["prism"]["caliper"],
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool every included specimen's **height** caliper error (only) into one distribution, per
    Proposal §12 ("report mean and 95th-percentile error... sigma_floor per tier feeds the
    pre-check"). Edges are not pooled here -- see the module docstring: `measure_prism`'s edge
    length and `_fit_rigid_2d`'s pose search both have real, pre-existing problems that make an
    edge-level number untrustworthy (confirmed by review, not assumed), while height is computed
    independently of both and is the only feature this aggregate currently trusts. A specimen
    with `board_alignment_ok=False` contributes nothing: its board itself didn't reconstruct at
    the right scale, so nothing it reports is evidence about the part's own accuracy.

    sigma_floor_mm is the 95th percentile (nearest-rank) of absolute height error -- this story's
    own concrete choice. With n this small, nearest-rank p95 is close to (and, at n<=20, can
    equal) the maximum; `max_error_mm` is reported alongside so that isn't mistaken for a finer-
    grained percentile than the sample size supports.
    """
    errors_mm: list[float] = []
    excluded_specimen_ids: list[str] = []
    for result in results:
        if not result["board_alignment_ok"]:
            excluded_specimen_ids.append(result["specimen_id"])
            continue
        height = result["caliper"]["height"]
        if height is not None:
            errors_mm.append(height["caliper_error_mm"])

    if not errors_mm:
        raise ValueError(
            "No specimen contributed a height caliper feature to aggregate."
        )

    absolute_errors_mm = sorted(abs(e) for e in errors_mm)
    mean_mm = statistics.fmean(errors_mm)
    p95_index = _nearest_rank_index(len(absolute_errors_mm), 0.95)
    sigma_floor_mm = absolute_errors_mm[p95_index]
    return {
        "feature_count": len(errors_mm),
        "excluded_specimen_ids": excluded_specimen_ids,
        "mean_error_mm": mean_mm,
        "sigma_floor_mm": sigma_floor_mm,
        "max_error_mm": absolute_errors_mm[-1],
        "sigma_floor_definition": (
            "95th percentile (nearest-rank) of |caliper_error_mm|, pooled across every "
            "included specimen's height feature only -- edges are excluded (see module "
            "docstring: measure_prism/_fit_rigid_2d aren't trustworthy yet). mean_error_mm is "
            "the signed mean (systematic bias). With n this small, sigma_floor_mm and "
            "max_error_mm are close or equal; both are reported rather than implying more "
            "precision than the sample size supports."
        ),
    }


def aggregate_in_plane(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool the fixed corner-to-corner edge errors for R-37's in-plane floor check.

    This deliberately stays separate from G-04's historical height-only aggregate above, so the
    corrected D-073 result is not silently redefined. Only edge entries that caliper_edge_errors
    could match unambiguously are present in each specimen's report. A failed board alignment
    excludes the whole specimen for the same reason as the height aggregate.
    """
    errors_mm: list[float] = []
    excluded_specimen_ids: list[str] = []
    for result in results:
        if not result["board_alignment_ok"]:
            excluded_specimen_ids.append(result["specimen_id"])
            continue
        errors_mm.extend(
            float(edge["caliper_error_mm"]) for edge in result["caliper"]["edges"]
        )

    definition = (
        "95th percentile (nearest-rank) of |caliper_error_mm| across every unambiguous "
        "corner-to-corner edge comparison from included board-reference specimens. "
        "mean_error_mm is the signed mean (systematic bias). R-37 only: this is separate "
        "from D-073's historical height-only sigma_floor."
    )
    if not errors_mm:
        return {
            "feature_count": 0,
            "excluded_specimen_ids": excluded_specimen_ids,
            "mean_error_mm": None,
            "sigma_floor_mm": None,
            "max_error_mm": None,
            "sigma_floor_definition": definition,
            "measurable": False,
        }

    absolute_errors_mm = sorted(abs(error) for error in errors_mm)
    p95_index = _nearest_rank_index(len(absolute_errors_mm), 0.95)
    return {
        "feature_count": len(errors_mm),
        "excluded_specimen_ids": excluded_specimen_ids,
        "mean_error_mm": statistics.fmean(errors_mm),
        "sigma_floor_mm": absolute_errors_mm[p95_index],
        "max_error_mm": absolute_errors_mm[-1],
        "sigma_floor_definition": definition,
        "measurable": True,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cuda", "mps"), default="mps")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fixtures/data/g04_board_error/g04-board-error-evidence.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)

    caliper_features = json.loads(CALIPER_G03_PATH.read_text(encoding="utf-8"))[
        "features"
    ]
    caliper_features_by_specimen: dict[str, list[dict[str, Any]]] = {
        specimen_id: [f for f in caliper_features if f.get("object") == specimen_id]
        for specimen_id in SPECIMEN_IDS
    }
    board_pitch_mm = load_board_pitch_mm()

    results = []
    for specimen_id in SPECIMEN_IDS:
        print(f"G-04: measuring {specimen_id}...", file=sys.stderr)
        results.append(
            measure_specimen(
                specimen_id, args.device, caliper_features_by_specimen, board_pitch_mm
            )
        )

    summary = aggregate(results)
    in_plane_summary = aggregate_in_plane(results)
    evidence = {
        "story": "G-04",
        "board_pitch_mm": {"across": board_pitch_mm[0], "down": board_pitch_mm[1]},
        "min_confidence": DEFAULT_MIN_CONFIDENCE,
        "specimens": results,
        "summary": summary,
        "in_plane_summary_r37": in_plane_summary,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"G-04 BOARD ERROR (heights only): {summary['feature_count']} features, "
        f"mean={summary['mean_error_mm']:.3f} mm, "
        f"sigma_floor (p95)={summary['sigma_floor_mm']:.3f} mm, "
        f"max={summary['max_error_mm']:.3f} mm"
    )
    if in_plane_summary["measurable"]:
        print(
            f"R-37 IN-PLANE ERROR: {in_plane_summary['feature_count']} edge features, "
            f"mean={in_plane_summary['mean_error_mm']:.3f} mm, "
            f"sigma_floor (p95)={in_plane_summary['sigma_floor_mm']:.3f} mm, "
            f"max={in_plane_summary['max_error_mm']:.3f} mm"
        )
    else:
        print("R-37 IN-PLANE ERROR: not measurable; no unambiguous edge features.")
    print(f"Evidence written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
