"""I-09: flag a spec feature Unverifiable at intake, before any session or photo exists.

Proposal §7.3/§7.4: each critical feature's tolerance is compared against `k * sigma_floor` for
the reference tier the purchase order requires, and anything below it is declared Unverifiable
before a single photo is taken -- so the buyer routes it to a CMM knowingly instead of
discovering endless Rescans in the field.

**Board tier only.** `sigma_floor` is measured per reference tier (G-04/R-37: board; G-10:
sheet and card, not yet measured). Nothing in this codebase derives which tier an order
*requires* from its CAD size or anything else -- `reference_tier_used` (`recon/service.py`) is an
*output* of capture, decided well after intake, and G-10's own story only measures the sheet/card
floors, never addresses tier selection. This module uses D-073's measured height floor for
`face`/`overall` and D-078's measured in-plane floor for `edge`/`hole`; the applicable floor is a
named scope choice, not a silent assumption that every order uses the same measurement.

**The residual gap is real and not fixed here.** D-077 found that even `D077_K` applied to
`D073_SIGMA_FLOOR_BOARD_MM` (0.70 * 0.86 = 0.60 mm) sits below the largest whole-part offset its
own measurement found (`s-04`, 0.73 mm) -- a tolerance between 0.60 mm and 0.73 mm on a part
shaped like `s-04` would still not be flagged, even though the method's own measured reliability
doesn't cover it. Raising `k` or `sigma_floor` to paper over that gap would be exactly the
"dressed-up tolerance" D-003 exists to prevent; D-077 left it as a tradeoff for a team decision,
not resolved by this story. Reported honestly: the numbers below are the real measured ones, not
adjusted to make the gap disappear.

A spec's `tolerance_mm` can be missing or null (A-03's own `InspectionSpec` would refuse that
without a justifying `assumptions` entry, but `recon/order_store.py` stores a buyer-supplied
spec as an opaque, unvalidated `dict` -- an order created any other way can still carry one). A
null tolerance is refused outright, never defaulted to "ok" or "unverifiable" against a number
that isn't there (the same "no invented number" discipline A-03 itself applies).

**Feature kind matters, found on review (@mbj1994, PR #372): the height floor does not apply to
every kind.** A-03's `SpecFeature.kind` is `Literal["hole", "face", "edge", "overall"]` --
`"hole"` and `"edge"` are in-plane measurements, not the vertical, out-of-plane dimension
D-073's 0.86 mm floor measures. D-078 now supplies the separate 5.278 mm in-plane floor from
R-37's 11 real edge comparisons. Use the height and in-plane floors for their respective kinds;
do not apply either floor to an unrecognised kind.

`"face"`/`"overall"` keep using the height floor, as a **named, stated assumption**, not a
default nobody chose: a face or an overall-size check is read here as checking against the
part's measured dimension in the same sense G-04's own height figure was measured (the
dimension perpendicular to the board), which is the closest of the two measured floors to what
those two kinds plausibly mean. This is an assumption, not a verified mapping -- revisit if a
real per-kind convention is ever written down. An unrecognised or missing `kind` is refused,
not assumed to be either tier.
"""

from __future__ import annotations

import math
from typing import Any

# G-04's measured height sigma_floor_mm (D-073), board reference only.
D073_SIGMA_FLOOR_BOARD_MM = 0.86
# R-37's measured in-plane edge sigma_floor_mm (D-078), board reference only.
D078_SIGMA_FLOOR_IN_PLANE_MM = 5.278

# G-07's consumable_k_estimate (measured 0.688, D-077) -- the k figure fit against max(sigma,
# sigma_floor) on total error, the exact quantity this story's Unverifiable row uses. Not the
# scatter-only k (~0.25, a different question D-077 also measured and explicitly warns against
# using here).
D077_K = 0.70

UNVERIFIABLE_THRESHOLD_HEIGHT_MM = D077_K * D073_SIGMA_FLOOR_BOARD_MM
UNVERIFIABLE_THRESHOLD_IN_PLANE_MM = D077_K * D078_SIGMA_FLOOR_IN_PLANE_MM

# Kinds with their measured board-reference floor. `face`/`overall` as height-like remains a
# named assumption; see the module docstring.
HEIGHT_LIKE_KINDS = frozenset({"face", "overall"})
IN_PLANE_KINDS = frozenset({"hole", "edge"})


def check_intake_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Check every feature in `spec["features"]` against the board tier's Unverifiable
    threshold. Returns one result dict per feature:
    `{"feature_name", "tolerance_mm", "status", "reason"}`, `status` one of
    `"ok"`/`"unverifiable"`/`"refused"`.

    A spec with no `features` key, or an empty list, returns `[]` -- nothing to check yet
    (D-004: no spec, no verdict machinery).
    """
    results: list[dict[str, Any]] = []
    for feature in spec.get("features", []):
        name = feature.get("name", "<unnamed feature>")
        tolerance_mm = feature.get("tolerance_mm")
        kind = feature.get("kind")

        if tolerance_mm is None:
            results.append(
                {
                    "feature_name": name,
                    "tolerance_mm": None,
                    "status": "refused",
                    "reason": "tolerance_mm is missing or null; refused, not defaulted.",
                }
            )
            continue

        if isinstance(tolerance_mm, bool) or not isinstance(tolerance_mm, (int, float)):
            results.append(
                {
                    "feature_name": name,
                    "tolerance_mm": None,
                    "status": "refused",
                    "reason": (
                        "tolerance_mm must be a finite positive number; refused, not defaulted."
                    ),
                }
            )
            continue
        try:
            numeric_tolerance_mm = float(tolerance_mm)
        except OverflowError:
            numeric_tolerance_mm = math.inf
        if not math.isfinite(numeric_tolerance_mm) or numeric_tolerance_mm <= 0:
            results.append(
                {
                    "feature_name": name,
                    "tolerance_mm": None,
                    "status": "refused",
                    "reason": (
                        "tolerance_mm must be a finite positive number; refused, not defaulted."
                    ),
                }
            )
            continue
        tolerance_mm = numeric_tolerance_mm

        if not isinstance(kind, str):
            results.append(
                {
                    "feature_name": name,
                    "tolerance_mm": tolerance_mm,
                    "status": "refused",
                    "reason": "kind is missing or not a string; refused, not assumed.",
                }
            )
            continue

        if kind in IN_PLANE_KINDS:
            sigma_floor_mm = D078_SIGMA_FLOOR_IN_PLANE_MM
            threshold_mm = UNVERIFIABLE_THRESHOLD_IN_PLANE_MM
        elif kind in HEIGHT_LIKE_KINDS:
            sigma_floor_mm = D073_SIGMA_FLOOR_BOARD_MM
            threshold_mm = UNVERIFIABLE_THRESHOLD_HEIGHT_MM
        else:
            results.append(
                {
                    "feature_name": name,
                    "tolerance_mm": tolerance_mm,
                    "status": "refused",
                    "reason": (
                        f"kind={kind!r} is unrecognised; refused rather than assumed to be "
                        "either the height or the in-plane tier."
                    ),
                }
            )
            continue

        if tolerance_mm < threshold_mm:
            results.append(
                {
                    "feature_name": name,
                    "tolerance_mm": tolerance_mm,
                    "status": "unverifiable",
                    "reason": (
                        f"tolerance_mm ({tolerance_mm} mm) is below k*sigma_floor "
                        f"({threshold_mm:.4f} mm; sigma_floor={sigma_floor_mm:.3f} mm, "
                        f"k={D077_K:.2f}) for the {kind} feature on the board reference."
                    ),
                }
            )
        else:
            results.append(
                {
                    "feature_name": name,
                    "tolerance_mm": tolerance_mm,
                    "status": "ok",
                    "reason": None,
                }
            )
    return results
