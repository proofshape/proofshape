"""R-16: judge R-04's recorded known-size comparison(s) against a real, measured tolerance from
the ground-truth calibration study (G-03 caliper measurement, G-04 noise floor) -- never a
tolerance this story derives or approximates itself (D-003).

G-04's own evidence file (`fixtures/data/g04_board_error/g04-board-error-evidence.json`, D-073)
is the only measured tolerance that exists: sigma_floor_mm for **height (out-of-plane) features
on the board reference only**. D-073 states explicitly that in-plane features -- edge lengths,
wall positions -- have no measured floor yet, pending R-37, and that R-16 (this story) should
judge an in-plane comparison as insufficient_evidence rather than against the height floor, which
does not measure what an edge-length difference needs. This script reads that distinction from
the comparison's own declared `feature_kind` rather than re-deriving or hardcoding it.

    python scripts/judge_known_size_tolerance.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypedDict

FeatureKind = Literal["height", "in_plane"]
Verdict = Literal["within_tolerance", "out_of_tolerance", "insufficient_evidence"]

G04_EVIDENCE_PATH = Path("fixtures/data/g04_board_error/g04-board-error-evidence.json")


class KnownSizeComparison(TypedDict):
    specimen_id: str
    feature: str
    feature_kind: FeatureKind
    difference_mm: float
    source: str


# R-04's one recorded comparison (PR #39, confirmed in `stories/R/R-04.md`'s notes): s-04's
# triangular face, B-C leg (apex to the base corner opposite the hypotenuse), reconstruction
# 60.27 mm vs. caliper mean 60.011 mm. This is an edge length -- an in-plane feature, not height.
KNOWN_SIZE_COMPARISONS: tuple[KnownSizeComparison, ...] = (
    {
        "specimen_id": "s-04",
        "feature": "B-C leg (triangular face)",
        "feature_kind": "in_plane",
        "difference_mm": 60.27 - 60.011,
        "source": "R-04 notes, PR #39",
    },
)


def load_height_sigma_floor_mm(evidence_path: Path = G04_EVIDENCE_PATH) -> float:
    """G-04's measured sigma_floor_mm for height (out-of-plane) features on the board reference
    (D-073) -- read from the evidence file, not hardcoded, so a re-measurement is picked up
    automatically."""
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    return evidence["summary"]["sigma_floor_mm"]


def judge_known_size_comparison(
    feature_kind: FeatureKind, difference_mm: float, sigma_floor_mm: float | None
) -> Verdict:
    """Classify one reconstruction-vs-caliper comparison against a measured tolerance.

    `sigma_floor_mm=None` means no measured tolerance exists yet for this feature_kind. Per
    D-003 (never state a tolerance the calibration study hasn't measured), the only honest
    verdict in that case is insufficient_evidence -- not a pass/fail dressed up from a number
    this function would otherwise have to invent."""
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
                comparison["feature_kind"],
                comparison["difference_mm"],
                sigma_floor_by_kind[comparison["feature_kind"]],
            ),
        )
        for comparison in comparisons
    ]


def main() -> int:
    height_sigma_floor_mm = load_height_sigma_floor_mm()
    for comparison, verdict in judge_all(KNOWN_SIZE_COMPARISONS, height_sigma_floor_mm):
        print(
            f"{comparison['specimen_id']} {comparison['feature']}: "
            f"{comparison['difference_mm']:+.3f} mm ({comparison['feature_kind']}, "
            f"{comparison['source']}) -> {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
