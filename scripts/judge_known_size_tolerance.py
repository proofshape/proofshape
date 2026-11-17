"""R-16: judge R-04's recorded known-size comparison(s) against a real, measured tolerance from
the ground-truth calibration study (G-03 caliper measurement, G-04 noise floor) -- never a
tolerance this story derives or approximates itself (D-003).

The only measured tolerance that exists is D-073's sigma_floor_mm for **height (out-of-plane)
features on the board reference only**: the team-agreed figure recorded in `docs/decisions.md`,
**0.86 mm**. That figure is a named constant here, not re-read from a local evidence file by
default -- `fixtures/data/g04_board_error/g04-board-error-evidence.json` is gitignored
(machine-specific, absent on a clean checkout) and, per review on PR #363 (@TabeenRaoof),
different machines' own G-04 re-runs give slightly different figures (0.8565 vs. 0.85) purely
from floating-point/VGGT run-to-run noise -- the DoD's second-member run needs the one number the
team actually agreed on, not whichever figure happens to sit on this machine. Pass `--evidence
PATH` to instead recompute from a local evidence file, e.g. to cross-check a fresh G-04 run
against D-073's recorded value.

D-073 states explicitly that in-plane features -- edge lengths, wall positions -- have no
measured floor yet, pending R-37, and that R-16 (this story) should judge an in-plane comparison
as insufficient_evidence rather than against the height floor, which does not measure what an
edge-length difference needs. `judge_all` reads that distinction from each comparison's own
declared `feature_kind` rather than re-deriving or hardcoding it per comparison.

    python scripts/judge_known_size_tolerance.py [--evidence PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, TypedDict

FeatureKind = Literal["height", "in_plane"]
Verdict = Literal["within_tolerance", "out_of_tolerance", "insufficient_evidence"]

# D-073's team-agreed sigma_floor_mm for height (out-of-plane) features on the board reference.
# Not the raw evidence-file figure (0.8564702691269304, which varies slightly machine to machine
# -- review on PR #363): this is the number recorded in docs/decisions.md.
D073_HEIGHT_SIGMA_FLOOR_MM = 0.86


class KnownSizeComparison(TypedDict):
    specimen_id: str
    feature: str
    feature_kind: FeatureKind
    difference_mm: float
    source: str


# Every known reconstruction-vs-caliper comparison this story judges. Each is a recorded,
# already-computed figure (like R-04's original s-04 entry, hand-recorded from PR #39's prose)
# -- not something `main()` re-derives from live fixture data, which would reintroduce the same
# clean-checkout dependency fixed for sigma_floor_mm above.
KNOWN_SIZE_COMPARISONS: tuple[KnownSizeComparison, ...] = (
    # s-04's triangular face, B-C leg (apex to the base corner opposite the hypotenuse):
    # reconstruction 60.27 mm vs. caliper mean 60.011 mm (R-04 notes, PR #39). An edge length --
    # in-plane, not height.
    {
        "specimen_id": "s-04",
        "feature": "B-C leg (triangular face)",
        "feature_kind": "in_plane",
        "difference_mm": 60.27 - 60.011,
        "source": "R-04 notes, PR #39",
    },
    # s-04/s-05 height and edges, from G-06's caliper truth (fixtures/caliper_g-06.json) run
    # through the current board-anchored cad_size_check.build_report -- independent of G-04's
    # g02_* set, so not circular (review on PR #363, @TabeenRaoof; reproduced independently here
    # on 2026-11-17: s-04 board_scale=0.9987 rms=2.078mm, s-05 board_scale=0.9982 rms=2.349mm,
    # both clean, nothing rejected). Heights are the real height/out-of-plane comparisons this
    # story can judge against D-073's floor. Edges are in-plane, and -- per D-073 -- not just
    # untested against a tolerance but independently known-unreliable themselves
    # (measure_prism's percentile convention and _fit_rigid_2d's pose search, both pending R-37):
    # listed so they're shown considered, not missed, landing at the same insufficient_evidence
    # verdict either way.
    {
        "specimen_id": "s-04",
        "feature": "height",
        "feature_kind": "height",
        "difference_mm": -0.515342513152607,
        "source": "cad_size_check.build_report vs. caliper_g-06.json, 2026-11-17",
    },
    {
        "specimen_id": "s-05",
        "feature": "height",
        "feature_kind": "height",
        "difference_mm": -0.207044697203131,
        "source": "cad_size_check.build_report vs. caliper_g-06.json, 2026-11-17",
    },
    {
        "specimen_id": "s-04",
        "feature": "edge, nominal 26mm leg",
        "feature_kind": "in_plane",
        "difference_mm": -0.359544081188513,
        "source": "cad_size_check.build_report vs. caliper_g-06.json, 2026-11-17",
    },
    {
        "specimen_id": "s-04",
        "feature": "edge, nominal 65mm hypotenuse",
        "feature_kind": "in_plane",
        "difference_mm": -8.756555769159,
        "source": "cad_size_check.build_report vs. caliper_g-06.json, 2026-11-17",
    },
    {
        "specimen_id": "s-04",
        "feature": "edge, nominal 60mm leg",
        "feature_kind": "in_plane",
        "difference_mm": -4.219056180144278,
        "source": "cad_size_check.build_report vs. caliper_g-06.json, 2026-11-17",
    },
    {
        "specimen_id": "s-05",
        "feature": "wall edge 0 (nominal 30mm)",
        "feature_kind": "in_plane",
        "difference_mm": -4.601536209590435,
        "source": "cad_size_check.build_report vs. caliper_g-06.json, 2026-11-17",
    },
    {
        "specimen_id": "s-05",
        "feature": "wall edge 1 (nominal 30mm)",
        "feature_kind": "in_plane",
        "difference_mm": -3.578555375016048,
        "source": "cad_size_check.build_report vs. caliper_g-06.json, 2026-11-17",
    },
    {
        "specimen_id": "s-05",
        "feature": "wall edge 2 (nominal 30mm)",
        "feature_kind": "in_plane",
        "difference_mm": -1.224959513072633,
        "source": "cad_size_check.build_report vs. caliper_g-06.json, 2026-11-17",
    },
    {
        "specimen_id": "s-05",
        "feature": "wall edge 3 (nominal 30mm)",
        "feature_kind": "in_plane",
        "difference_mm": -5.098188441192392,
        "source": "cad_size_check.build_report vs. caliper_g-06.json, 2026-11-17",
    },
)


def load_height_sigma_floor_mm(evidence_path: Path | None = None) -> float:
    """D-073's team-agreed height sigma_floor_mm (0.86 mm) by default. Pass `evidence_path` to
    instead recompute it from a local G-04 evidence file (gitignored, machine-specific) --
    `main()` does not do this by default; see the module docstring."""
    if evidence_path is None:
        return D073_HEIGHT_SIGMA_FLOOR_MM
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    return evidence["summary"]["sigma_floor_mm"]


def judge_known_size_comparison(
    difference_mm: float, sigma_floor_mm: float | None
) -> Verdict:
    """Classify one reconstruction-vs-caliper comparison against a measured tolerance.

    `sigma_floor_mm=None` means no measured tolerance exists yet for this comparison's feature
    kind. Per D-003 (never state a tolerance the calibration study hasn't measured), the only
    honest verdict in that case is insufficient_evidence -- not a pass/fail dressed up from a
    number this function would otherwise have to invent."""
    if sigma_floor_mm is None:
        return "insufficient_evidence"
    return (
        "within_tolerance"
        if abs(difference_mm) <= sigma_floor_mm
        else "out_of_tolerance"
    )


def judge_all(
    comparisons: tuple[KnownSizeComparison, ...], height_sigma_floor_mm: float
) -> list[tuple[KnownSizeComparison, Verdict]]:
    """Judge every known comparison. Only `feature_kind == "height"` has a measured tolerance
    today (G-04/D-073); `"in_plane"` has none yet (pending R-37)."""
    sigma_floor_by_kind: dict[FeatureKind, float | None] = {
        "height": height_sigma_floor_mm,
        "in_plane": None,
    }
    return [
        (
            comparison,
            judge_known_size_comparison(
                comparison["difference_mm"],
                sigma_floor_by_kind[comparison["feature_kind"]],
            ),
        )
        for comparison in comparisons
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--evidence",
        type=Path,
        default=None,
        help=(
            "Recompute the height sigma_floor_mm from this local G-04 evidence file instead of "
            "using D-073's recorded 0.86 mm."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    height_sigma_floor_mm = load_height_sigma_floor_mm(args.evidence)
    for comparison, verdict in judge_all(KNOWN_SIZE_COMPARISONS, height_sigma_floor_mm):
        print(
            f"{comparison['specimen_id']} {comparison['feature']}: "
            f"{comparison['difference_mm']:+.3f} mm ({comparison['feature_kind']}, "
            f"{comparison['source']}) -> {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
