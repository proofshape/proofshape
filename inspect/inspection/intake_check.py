"""I-09: flag a spec feature Unverifiable at intake, before any session or photo exists.

Proposal §7.3/§7.4: each critical feature's tolerance is compared against `k * sigma_floor` for
the reference tier the purchase order requires, and anything below it is declared Unverifiable
before a single photo is taken -- so the buyer routes it to a CMM knowingly instead of
discovering endless Rescans in the field.

**Board tier only.** `sigma_floor` is measured per reference tier (G-04: board; G-10: sheet and
card, not yet measured). Nothing in this codebase derives which tier an order *requires* from
its CAD size or anything else -- `reference_tier_used` (`recon/service.py`) is an *output* of
capture, decided well after intake, and G-10's own story only measures the sheet/card floors,
never addresses tier selection. This module checks every feature against the board tier's own
measured floor, which is the only one that exists -- a named, stated scope limitation, not a
silent assumption that every order uses the board.

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
"""

from __future__ import annotations

from typing import Any

# G-04's measured height sigma_floor_mm (D-073), board reference only.
D073_SIGMA_FLOOR_BOARD_MM = 0.86

# G-07's consumable_k_estimate (measured 0.688, D-077) -- the k figure fit against max(sigma,
# sigma_floor) on total error, the exact quantity this story's Unverifiable row uses. Not the
# scatter-only k (~0.25, a different question D-077 also measured and explicitly warns against
# using here).
D077_K = 0.70

UNVERIFIABLE_THRESHOLD_MM = D077_K * D073_SIGMA_FLOOR_BOARD_MM


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
        if tolerance_mm < UNVERIFIABLE_THRESHOLD_MM:
            results.append(
                {
                    "feature_name": name,
                    "tolerance_mm": tolerance_mm,
                    "status": "unverifiable",
                    "reason": (
                        f"tolerance_mm ({tolerance_mm} mm) is below k*sigma_floor "
                        f"({UNVERIFIABLE_THRESHOLD_MM:.3f} mm) for the board reference."
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
