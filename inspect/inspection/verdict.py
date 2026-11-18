"""I-07: proposal 7.2's four-row verdict table, for `declared_at: post_reconstruction`
(Unverifiable's intake-time path is I-09's job, not this one's).

| Verdict | Condition |
|---|---|
| Pass | Every critical region Observed with sigma <= tolerance/k, and no Observed deviation above tolerance |
| Fail | An Observed deviation exceeds tolerance by more than k*sigma at that location |
| Rescan | A critical region is Unobserved or high-sigma; registration is ambiguous; or the only deviation lies in an Unobserved region |

The hard rule governs every row: geometry that was not observed never fails a part and never
contributes to a Pass -- it can only trigger a Rescan (proposal 7.2; AGENTS.md).

**What a "region" is in v1, resolved (caught on review, @TabeenRaoof, #302):** no code anywhere
binds a named spec feature to actual CAD geometry -- `inspect/inspection/cad.py` only loads
coordinates/units, and I-06's `compute_deviation` is one global per-vertex array over the whole
posed mesh. v1 therefore does what proposal 7.4 literally says ("stops at the structured spec
and applies tolerances globally, naming each feature in the report"): every named feature is
judged against the *same* global measurement, using only that feature's own `tolerance_mm` to
decide its own Pass/Fail/Rescan. Binding a feature to real geometry is the Tier 3 extension, not
this story's.

**`max(sigma, sigma_floor)`, never raw sigma, for out-of-plane comparisons (issue #157):**
`compute_vertex_sigma` can return sigma close to 0 from two near-identical residuals at n=2
observations -- the "zero spread means infinite precision" trap. Every out-of-plane comparison
floors sigma at `D073_SIGMA_FLOOR_BOARD_MM` first, via `np.fmax` (not `np.maximum` -- `fmax`
treats a NaN sigma, R-10's own "fewer than two observations" marker, as "use the floor", where
`maximum` would propagate the NaN into the comparison and silently break it).

**sigma_floor covers out-of-plane (height, D-073, 0.86 mm) and in-plane (wall, D-078, R-37,
5.278 mm) comparisons, each with its own measured figure, and every Observed vertex in a pool is
checked, not just the pool's single worst one (caught on review, @TabeenRaoof, PR #379):** the
first version of this module floored only out-of-plane sigma and left in-plane raw (D-073's
"pending R-37" framing, true when this story started but superseded once PR #364/D-078 landed a
real measured in-plane floor) and evaluated each pool against only its own single
largest-deviation vertex. The reviewer gave concrete counterexamples where a second vertex, not
the pool's worst one, was independently over tolerance or over the sigma-vs-tolerance margin --
proposal 7.2's own wording binds *every* Observed vertex ("no Observed deviation above
tolerance" for Pass; an exceeding deviation "at that location" for Fail means any location, not
whichever one the pool's own argmax happens to pick). Each Observed, measured vertex is
classified out-of-plane (top `100 - TOP_REGION_PERCENTILE` percent of this session's own
height-above-board distribution) or in-plane (everything else), using
`recon.cad_size_check.compute_height_above_board` and the same percentile convention
G-07/D-077's `per_vertex_height_reliability` already established for this exact board/wall
distinction -- reused here, not reinvented. Each pool's own sigma is floored with `np.fmax` at
its own figure (`D073_SIGMA_FLOOR_BOARD_MM` out-of-plane, `D078_SIGMA_FLOOR_IN_PLANE_MM`
in-plane -- `fmax` treats a NaN sigma, R-10's "fewer than two observations" marker, as "use the
floor" in both pools alike, so neither pool needs a separate NaN-is-insufficient-evidence path
any more). A pool then Fails if *any* of its vertices has `|deviation| > tolerance + k *
sigma_eff` at that vertex; otherwise it Passes only if *every* vertex has `|deviation| <=
tolerance` *and* `sigma_eff <= tolerance / k`; otherwise it Rescans as high-sigma. The two pools
are evaluated independently (D-073/D-078 give them different floors, so they cannot share one
sigma), and a feature's own rank is the worse of its two pools.

**`k`/`sigma_floor` are consumed as given, never re-measured or re-derived here** (G-07's D-077,
G-04's D-073, R-37's D-078) -- defined independently in this module rather than imported from
`inspection.intake_check` (I-09), matching this project's own existing precedent of each
consumer keeping its own named copy of a team-agreed figure (G-07's own script does the same
rather than importing G-04's).

**Scope difference from I-09, by design, stated rather than left implicit (caught on review,
@TabeenRaoof, PR #379):** I-09 refuses `face`/`overall` features outright at intake because
nothing says which axis they check, and judges only `hole`/`edge` against D-078's figure. This
module instead classifies by *vertex* height, not by feature *kind* -- consistent with the
module's own "what a region is in v1" resolution above (every named feature, of any kind, is
judged against the same global per-vertex measurement, since v1 binds no feature to actual CAD
geometry). A `face`/`overall` feature here is judged against whichever pool(s) its global
measurement's own worst vertices fall into, not refused for axis ambiguity the way I-09 refuses
it at intake. This is intentional for v1 (the two stories answer different questions -- I-09
flags Unverifiable before any photo exists; this module judges a real measurement after one
does) but is a real scope mismatch worth a team decision before Tier 3 binds features to regions.

**`reason_codes` vocabulary, defined here (caught on review, #302: none existed anywhere):**
`f"{feature_name}_not_seen"` (Unobserved), `f"{feature_name}_deviation_out_of_spec"` (Fail --
matches the contract's own committed example naming exactly, from either pool),
`f"{feature_name}_high_sigma"` (out-of-plane pool Rescan), `f"{feature_name}_in_plane_high_sigma"`
(in-plane pool Rescan -- kept distinct from the out-of-plane code so a report can say which
pool's floor was the problem), plus the already-defined `registration.REASON_ORIENTATION_AMBIGUOUS`
and `"registration_failed"` (this module's own, for when registration/ambiguity/deviation
couldn't be computed at all -- a best-effort post-reconstruction step, not a reconstruction
failure).

A *critical* feature (contract's `SpecFeature.critical`) with a `tolerance_mm` that isn't a
finite, positive, non-bool number -- missing, null, a string, negative, zero, NaN, infinite, or
a literal `True`/`False` -- is refused outright (raises `RefusedToleranceError`), never defaulted
or coerced into a comparison. `_valid_tolerance_mm` matches I-09's own `check_intake_spec` check
exactly (PR #379 review, @TabeenRaoof: the first version of this check only caught `None`, so a
string tolerance crashed `_evaluate_pool` outright, and a negative one produced a wrong Fail on
a clean part -- the one thing this story must never do). The contract's `VerdictResult` has no
"refused" verdict value to represent any of this, so the caller (`getVerdict`) turns it into an
error response, not a verdict. A missing `critical` key is treated as critical (refuse), the
same "refuse, don't assume" discipline as an invalid tolerance itself -- only an explicit
`critical: false` is read as non-critical. A non-critical feature with no valid tolerance is not
refused: it is listed as its own region, contributing nothing (no tolerance exists to judge it
against), rather than aborting the whole verdict for a feature the contract itself says doesn't
need one (caught on review, @TabeenRaoof, PR #379 -- the contract's own 422 description already
said "a critical spec feature," but the first version refused on *any* feature's null
tolerance). A malformed spec (`features` present but not a list, or an entry that is not an
object) is refused the same way, matching I-09's own "refuse, don't crash on an opaque,
unvalidated spec dict" discipline (`recon/order_store.py` stores a buyer-supplied spec
unvalidated), rather than raising an uncaught `AttributeError`/`TypeError` `getVerdict` would
have to let through as a 500.

**A Pass requires at least one region to actually have `contributes_to_verdict: True`**
(`NoJudgedFeaturesError`, caught on review, @TabeenRaoof and @mbj1994, PR #379): `verdict_rank`
starts at Pass and only a judged feature ever raises it, so a spec with nothing to judge -- no
`features` at all, or every feature non-critical with no valid tolerance -- fell through to a
bare Pass regardless of what the part actually looked like, including past a planted defect
many times any plausible tolerance. The caller (`getVerdict`) turns this into the existing
`404 verdict_not_available`, not a new contract error code -- there is as little a verdict to
serve as when post-reconstruction measurement hasn't run yet.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from recon.cad_size_check import compute_height_above_board

# G-04's measured out-of-plane (height) sigma_floor_mm (D-073), board reference only.
D073_SIGMA_FLOOR_BOARD_MM = 0.86

# R-37's measured in-plane (wall) sigma_floor_mm (D-078), board reference only.
D078_SIGMA_FLOOR_IN_PLANE_MM = 5.278

# G-07's consumable_k_estimate (measured 0.688, D-077) -- total error against
# max(sigma, sigma_floor), the exact quantity this module's Fail/Pass rows use.
D077_K = 0.70

# D-073's own out-of-plane/in-plane split, reusing G-07/D-077's percentile convention
# (recon.cad_size_check.per_vertex_height_reliability) rather than reinventing it: the top
# 100 - TOP_REGION_PERCENTILE percent of this session's own Observed-measured height
# distribution is out-of-plane (top face); everything else is in-plane (wall).
TOP_REGION_PERCENTILE = 90.0

LABEL_OBSERVED = "Observed"

REASON_REGISTRATION_FAILED = "registration_failed"


class RefusedToleranceError(ValueError):
    """A spec feature's tolerance_mm is missing or null -- refused, never defaulted."""


class NoJudgedFeaturesError(ValueError):
    """No feature in the spec was ever actually evaluated against a Pass/Fail/Rescan rule -- an
    empty or missing `features` list, or every feature non-critical with no valid tolerance. A
    Pass under these conditions would be the "dressed-up" claim D-003 rules out (D-004: no
    spec, no verdict machinery) -- caught on review, @TabeenRaoof and @mbj1994, PR #379:
    `verdict_rank` starts at Pass and nothing ever raised it, so `{}`, `{"features": []}`, and
    R-34's own `{"tolerance_mm": 4.0}` placeholder shape all Passed a part with a planted 12mm
    defect. The caller (`getVerdict`) turns this into `404 verdict_not_available`, the same code
    already used when post-reconstruction measurement hasn't run -- there is similarly nothing
    to serve a verdict from."""


def _valid_tolerance_mm(tolerance_mm: Any) -> float | None:
    """`tolerance_mm` as a finite positive float, or `None` if it isn't one -- `None` itself,
    a non-numeric value (including `bool`, which is a `numbers` subtype of `int` but is never a
    tolerance), NaN, infinity, or a non-positive number. Matches I-09's own
    `check_intake_spec` check exactly (PR #379 review, @TabeenRaoof): a malformed tolerance
    reaching this story's `post_reconstruction` rows must be refused the same way I-09 already
    refuses it at intake, not silently coerced into a comparison (a string tolerance raised
    inside `_evaluate_pool`; a negative one produced a wrong Fail on a clean part)."""
    if tolerance_mm is None or isinstance(tolerance_mm, bool):
        return None
    if not isinstance(tolerance_mm, (int, float)):
        return None
    try:
        numeric_tolerance_mm = float(tolerance_mm)
    except OverflowError:
        numeric_tolerance_mm = math.inf
    if not math.isfinite(numeric_tolerance_mm) or numeric_tolerance_mm <= 0:
        return None
    return numeric_tolerance_mm


def _evaluate_pool(
    deviation_mm: np.ndarray,
    sigma_mm: np.ndarray,
    floor_mm: float,
    tolerance_mm: float,
) -> tuple[str, float] | None:
    """Evaluate one pool (out-of-plane or in-plane) against proposal 7.2's Pass/Fail rows,
    checking *every* vertex in the pool, not just its single largest-deviation one (PR #379
    review): Fail if any vertex's deviation exceeds `tolerance + k * sigma_eff` at that vertex;
    Pass only if every vertex's deviation is within tolerance and every vertex's sigma_eff is
    within `tolerance / k`; otherwise Rescan as high-sigma. `sigma_eff` is `sigma_mm` floored at
    `floor_mm` via `np.fmax`, which also treats a NaN sigma (R-10's "fewer than two
    observations" marker) as "use the floor" rather than propagating it.

    Returns `(verdict, representative_deviation_mm)`, the verdict one of
    `"Pass"`/`"Fail"`/`"Rescan"` and the representative deviation the pool's own
    largest-magnitude vertex (for reporting only -- the rule above does not depend on it).
    `None` if the pool is empty.
    """
    if len(deviation_mm) == 0:
        return None
    sigma_eff_mm = np.fmax(sigma_mm, floor_mm)
    representative_deviation_mm = float(
        deviation_mm[int(np.argmax(np.abs(deviation_mm)))]
    )
    if np.any(np.abs(deviation_mm) > tolerance_mm + D077_K * sigma_eff_mm):
        return "Fail", representative_deviation_mm
    if np.all(np.abs(deviation_mm) <= tolerance_mm) and np.all(
        sigma_eff_mm <= tolerance_mm / D077_K
    ):
        return "Pass", representative_deviation_mm
    return "Rescan", representative_deviation_mm


def compute_verdict(
    deviation_mm: np.ndarray,
    deviation_measured: np.ndarray,
    labels: np.ndarray,
    sigma_mm: np.ndarray,
    vertex_positions_mm: np.ndarray,
    board_plane: tuple[float, float, float],
    spec: dict[str, Any],
    *,
    orientation_ambiguous: bool = False,
    ambiguity_reason_codes: Sequence[str] = (),
    registration_failed: bool = False,
) -> dict[str, Any]:
    """Compute a `declared_at: post_reconstruction` verdict. `deviation_mm`, `deviation_measured`,
    `labels`, `sigma_mm` and `vertex_positions_mm` are all per-vertex, same order, same length
    (I-06's `_DEVIATION_MM`/`_DEVIATION_MEASURED`, R-09's labels, R-10's sigma_mm, and the served
    mesh's own board-frame-mm vertex positions). `board_plane` is `(slope_x, slope_y,
    z_at_origin_mm)` -- `BoardSurface`'s own plane fields, the same tuple shape
    `scripts/g07_reliability_curve.py`'s `_read_board_plane` builds -- used only to classify each
    vertex out-of-plane vs in-plane (D-073); it is not re-fit here. `registration_failed=True`
    means I-05/I-06 could not be run at all for this session (e.g. a degenerate CAD or no
    Observed faces) -- distinct from `orientation_ambiguous`, which means they ran but the result
    can't be trusted.

    Returns a dict matching `VerdictResult` minus `declared_at` (always
    `"post_reconstruction"` here -- the caller sets it).
    """
    features = spec.get("features")
    if features is None:
        features = []
    elif not isinstance(features, list):
        raise RefusedToleranceError(
            "spec['features'] must be a list; refused, not assumed."
        )
    for feature in features:
        if not isinstance(feature, dict):
            raise RefusedToleranceError(
                "a feature entry must be an object; refused, not assumed."
            )
        if (
            feature.get("critical", True)
            and _valid_tolerance_mm(feature.get("tolerance_mm")) is None
        ):
            raise RefusedToleranceError(
                f"critical feature {feature.get('name', '<unnamed feature>')!r} has a "
                "missing, null, or non-finite-positive tolerance_mm; refused, not defaulted."
            )

    observed_measured = (np.asarray(labels) == LABEL_OBSERVED) & (
        np.asarray(deviation_measured) > 0.5
    )
    has_observed_data = bool(np.any(observed_measured))

    out_of_plane_deviation = out_of_plane_sigma_raw = np.empty(0)
    in_plane_deviation = in_plane_sigma_raw = np.empty(0)
    overall_worst_deviation_mm: float | None = None
    if has_observed_data:
        observed_indices = np.flatnonzero(observed_measured)
        observed_deviation = np.asarray(deviation_mm)[observed_indices]
        observed_sigma_raw = np.asarray(sigma_mm, dtype=np.float64)[observed_indices]

        # D-073/D-078: classify each Observed, measured vertex out-of-plane (top face) vs
        # in-plane (wall) before deciding anything -- each pool gets its own measured floor.
        height_above_board_mm = compute_height_above_board(
            np.asarray(vertex_positions_mm)[observed_indices], board_plane
        )
        top_region_threshold_mm = np.percentile(
            height_above_board_mm, TOP_REGION_PERCENTILE
        )
        out_of_plane_mask = height_above_board_mm >= top_region_threshold_mm
        in_plane_mask = ~out_of_plane_mask

        out_of_plane_deviation = observed_deviation[out_of_plane_mask]
        out_of_plane_sigma_raw = observed_sigma_raw[out_of_plane_mask]
        in_plane_deviation = observed_deviation[in_plane_mask]
        in_plane_sigma_raw = observed_sigma_raw[in_plane_mask]

        overall_worst_deviation_mm = float(
            observed_deviation[int(np.argmax(np.abs(observed_deviation)))]
        )

    rank = {"Pass": 0, "Rescan": 1, "Fail": 2}
    verdict_rank = 0
    regions: list[dict[str, Any]] = []
    reason_codes: list[str] = list(ambiguity_reason_codes)
    if registration_failed:
        reason_codes.append(REASON_REGISTRATION_FAILED)

    for feature in features:
        name = feature.get("name", "<unnamed feature>")
        tolerance_mm = _valid_tolerance_mm(feature.get("tolerance_mm"))

        if not has_observed_data:
            regions.append(
                {
                    "name": name,
                    "provenance": "unobserved",
                    "deviation_mm": None,
                    "contributes_to_verdict": False,
                }
            )
            reason_codes.append(f"{name}_not_seen")
            verdict_rank = max(verdict_rank, rank["Rescan"])
            continue

        if orientation_ambiguous or registration_failed:
            regions.append(
                {
                    "name": name,
                    "provenance": "observed",
                    "deviation_mm": overall_worst_deviation_mm,
                    "contributes_to_verdict": False,
                }
            )
            verdict_rank = max(verdict_rank, rank["Rescan"])
            continue

        if tolerance_mm is None:
            # Non-critical feature, no tolerance to judge it against (the contract's own
            # 422 description restricts refusal to critical features) -- contributes nothing,
            # not even a reason code, rather than guessing a number or aborting the verdict.
            regions.append(
                {
                    "name": name,
                    "provenance": "observed",
                    "deviation_mm": overall_worst_deviation_mm,
                    "contributes_to_verdict": False,
                }
            )
            continue

        feature_rank = rank["Pass"]
        feature_reason_codes: list[str] = []
        feature_deviation_mm: float | None = None
        feature_contributes = False

        out_of_plane_result = _evaluate_pool(
            out_of_plane_deviation,
            out_of_plane_sigma_raw,
            D073_SIGMA_FLOOR_BOARD_MM,
            tolerance_mm,
        )
        if out_of_plane_result is not None:
            oop_verdict, oop_deviation_mm = out_of_plane_result
            feature_rank = max(feature_rank, rank[oop_verdict])
            if oop_verdict == "Fail":
                feature_reason_codes.append(f"{name}_deviation_out_of_spec")
                feature_deviation_mm = oop_deviation_mm
                feature_contributes = True
            elif oop_verdict == "Pass":
                feature_deviation_mm = oop_deviation_mm
                feature_contributes = True
            else:
                feature_reason_codes.append(f"{name}_high_sigma")

        in_plane_result = _evaluate_pool(
            in_plane_deviation,
            in_plane_sigma_raw,
            D078_SIGMA_FLOOR_IN_PLANE_MM,
            tolerance_mm,
        )
        if in_plane_result is not None:
            ip_verdict, ip_deviation_mm = in_plane_result
            feature_rank = max(feature_rank, rank[ip_verdict])
            if ip_verdict == "Fail":
                feature_reason_codes.append(f"{name}_deviation_out_of_spec")
                if feature_deviation_mm is None or abs(ip_deviation_mm) > abs(
                    feature_deviation_mm
                ):
                    feature_deviation_mm = ip_deviation_mm
                feature_contributes = True
            elif ip_verdict == "Pass":
                if feature_deviation_mm is None:
                    feature_deviation_mm = ip_deviation_mm
                    feature_contributes = True
            else:
                feature_reason_codes.append(f"{name}_in_plane_high_sigma")

        if feature_deviation_mm is None:
            feature_deviation_mm = overall_worst_deviation_mm

        regions.append(
            {
                "name": name,
                "provenance": "observed",
                "deviation_mm": feature_deviation_mm,
                "contributes_to_verdict": feature_contributes,
            }
        )
        reason_codes.extend(feature_reason_codes)
        verdict_rank = max(verdict_rank, feature_rank)

    verdict = {0: "Pass", 1: "Rescan", 2: "Fail"}[verdict_rank]
    if verdict == "Pass" and not any(
        region["contributes_to_verdict"] for region in regions
    ):
        # Every other way to reach Pass sets contributes_to_verdict on at least one region
        # (a pool only returns "Pass" by setting it) -- the only way to reach it with none set
        # is that no feature was ever judged at all (no features, or every one non-critical
        # with no valid tolerance). That is "nothing to check," not "nothing wrong."
        raise NoJudgedFeaturesError(
            "no feature in this spec was judged against a Pass/Fail/Rescan rule; refused, "
            "not defaulted to Pass."
        )
    return {
        "verdict": verdict,
        "orientation_ambiguous": orientation_ambiguous,
        "regions": regions,
        "reason_codes": reason_codes,
    }
