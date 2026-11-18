"""I-09: flag a spec feature Unverifiable at intake, before any session or photo exists.

Proposal §7.3/§7.4: each critical feature's tolerance is compared against `k * sigma_floor` for
the reference tier the purchase order requires, and anything below it is declared Unverifiable
before a single photo is taken -- so the buyer routes it to a CMM knowingly instead of
discovering endless Rescans in the field.

**Board tier only.** `sigma_floor` is measured per reference tier (G-04/R-37: board; G-10:
sheet and card, not yet measured). Nothing in this codebase derives which tier an order
*requires* from its CAD size or anything else -- `reference_tier_used` (`recon/service.py`) is an
*output* of capture, decided well after intake, and G-10's own story only measures the sheet/card
floors, never addresses tier selection. This module uses D-078's measured in-plane floor for
`edge`/`hole`, the only two kinds whose axis is unambiguous; the applicable floor is a named
scope choice, not a silent assumption that every order uses the same measurement.

**The residual gap is real and not fixed here.** D-077 found that even `D077_K` applied to
`D073_SIGMA_FLOOR_BOARD_MM` (0.70 * 0.86 = 0.60 mm) sits below the largest whole-part offset its
own measurement found (`s-04`, 0.73 mm) -- a tolerance between 0.60 mm and 0.73 mm on a part
shaped like `s-04` would still not be flagged, even though the method's own measured reliability
doesn't cover it. D-078's in-plane floor has the same kind of gap, and a larger one: 5.278 mm is
the nearest-rank p95 of only 11 real edge comparisons, so for that sample size it is also the
*maximum* measured edge error -- at least one real edge was measured exactly at the reported
floor, so `k * 5.278 = 3.6946 mm` sits below the worst edge error R-37 actually found. Raising
`k` or `sigma_floor` to paper over either gap would be exactly the "dressed-up tolerance" D-003
exists to prevent; both are left as a tradeoff for a team decision, not resolved by this story.
Reported honestly: the numbers below are the real measured ones, not adjusted to make the gap
disappear.

A spec's `tolerance_mm` can be missing or null (A-03's own `InspectionSpec` would refuse that
without a justifying `assumptions` entry, but `recon/order_store.py` stores a buyer-supplied
spec as an opaque, unvalidated `dict` -- an order created any other way can still carry one). A
null tolerance is refused outright, never defaulted to "ok" or "unverifiable" against a number
that isn't there (the same "no invented number" discipline A-03 itself applies).

**Feature kind matters, found on review (@mbj1994, PR #372): a height floor does not apply to
every kind.** A-03's `SpecFeature.kind` is `Literal["hole", "face", "edge", "overall"]` --
`"hole"` and `"edge"` are unambiguously in-plane measurements, checked against D-078's measured
in-plane floor. `"face"`/`"overall"` are refused instead (found on a second review, @TabeenRaoof,
PR #372): a face can be a top face (the out-of-plane dimension D-073 measures) or a side face
(in-plane), and "overall" doesn't say which of a part's dimensions it means either. Nothing in
this codebase derives which axis a `face`/`overall` feature checks, so picking either floor would
be guessing, and picking the *smaller* one (as an earlier version of this module did, using
D-073's height floor for both) understates how hard the feature actually is to verify -- the
opposite of what an Unverifiable-at-intake check exists to catch. `D073_SIGMA_FLOOR_BOARD_MM` and
`UNVERIFIABLE_THRESHOLD_HEIGHT_MM` stay in this module because they are real measured facts (and
feed the gap analysis above), not because any `kind` is checked against them today. An
unrecognised or missing `kind` is refused the same way, for the same reason: no measured floor is
assumed for a kind that doesn't state its axis.

**Malformed input is refused, not raised.** `recon/order_store.py` stores a spec as an opaque,
unvalidated `dict` (found on review, @TabeenRaoof, PR #372): a `features` key that is present but
not a list, or a list entry that is not an object, previously raised inside this function,
which `recon/service.py` does not catch -- and because an order is saved before its response is
built, a single malformed read corrupts every future read of that same order, not just the one
request that created it. Both cases are refused instead, the same way a malformed `tolerance_mm`
already was: a missing or null `features` key means nothing to check yet (`[]`); anything else
malformed becomes one or more `"refused"` results, never an exception.
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

# The only two kinds with an unambiguous axis, and so a measured board-reference floor.
IN_PLANE_KINDS = frozenset({"hole", "edge"})
# `face`/`overall` don't state which axis they check; see the module docstring. Refused the
# same way an unrecognised kind is, not defaulted to either floor.
AXIS_AMBIGUOUS_KINDS = frozenset({"face", "overall"})

_MALFORMED_SPEC_NAME = "<malformed spec>"
_MALFORMED_FEATURE_NAME = "<malformed feature>"
_UNNAMED_FEATURE_NAME = "<unnamed feature>"


def _refused(name: str, tolerance_mm: float | None, reason: str) -> dict[str, Any]:
    return {
        "feature_name": name,
        "tolerance_mm": tolerance_mm,
        "status": "refused",
        "reason": reason,
    }


def check_intake_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Check every feature in `spec["features"]` against the board tier's Unverifiable
    threshold. Returns one result dict per feature:
    `{"feature_name", "tolerance_mm", "status", "reason"}`, `status` one of
    `"ok"`/`"unverifiable"`/`"refused"`.

    A spec with no `features` key, or a null one, returns `[]` -- nothing to check yet
    (D-004: no spec, no verdict machinery). This never raises: `spec` is an opaque, unvalidated
    dict (see the module docstring), and a malformed `features` list, or a malformed entry
    inside it, is refused the same way a malformed `tolerance_mm` already was, not an exception
    `recon/service.py` would have to catch.
    """
    features = spec.get("features")
    if features is None:
        return []
    if not isinstance(features, list):
        return [
            _refused(
                _MALFORMED_SPEC_NAME,
                None,
                "spec['features'] must be a list; refused, not assumed.",
            )
        ]

    results: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict):
            results.append(
                _refused(
                    _MALFORMED_FEATURE_NAME,
                    None,
                    "feature entry is not an object; refused, not assumed.",
                )
            )
            continue

        name = feature.get("name")
        if not isinstance(name, str) or not name.strip():
            name = _UNNAMED_FEATURE_NAME
        tolerance_mm = feature.get("tolerance_mm")
        kind = feature.get("kind")

        if tolerance_mm is None:
            results.append(
                _refused(
                    name,
                    None,
                    "tolerance_mm is missing or null; refused, not defaulted.",
                )
            )
            continue

        if isinstance(tolerance_mm, bool) or not isinstance(tolerance_mm, (int, float)):
            results.append(
                _refused(
                    name,
                    None,
                    "tolerance_mm must be a finite positive number; refused, not defaulted.",
                )
            )
            continue
        try:
            numeric_tolerance_mm = float(tolerance_mm)
        except OverflowError:
            numeric_tolerance_mm = math.inf
        if not math.isfinite(numeric_tolerance_mm) or numeric_tolerance_mm <= 0:
            results.append(
                _refused(
                    name,
                    None,
                    "tolerance_mm must be a finite positive number; refused, not defaulted.",
                )
            )
            continue
        tolerance_mm = numeric_tolerance_mm

        if not isinstance(kind, str):
            results.append(
                _refused(
                    name,
                    tolerance_mm,
                    "kind is missing or not a string; refused, not assumed.",
                )
            )
            continue

        if kind in IN_PLANE_KINDS:
            sigma_floor_mm = D078_SIGMA_FLOOR_IN_PLANE_MM
            threshold_mm = UNVERIFIABLE_THRESHOLD_IN_PLANE_MM
        elif kind in AXIS_AMBIGUOUS_KINDS:
            results.append(
                _refused(
                    name,
                    tolerance_mm,
                    f"kind={kind!r} does not state which dimension is checked (height or "
                    "in-plane), and only axis-specific floors are measured (D-073 height, "
                    "D-078 in-plane); refused, not assumed.",
                )
            )
            continue
        else:
            results.append(
                _refused(
                    name,
                    tolerance_mm,
                    f"kind={kind!r} is unrecognised; refused rather than assumed to be "
                    "either the height or the in-plane tier.",
                )
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
